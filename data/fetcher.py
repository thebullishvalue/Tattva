"""
Tattva — Unified data fetcher: yfinance macro / commodity / OHLCV universe.
तत्त्व (Tattva) — "Principle / Essence"

Each external call is wrapped with:
  1. **Two-tier cache** (memory + disk, TTL + versioned keys) — `data/cache.py`
  2. **Circuit breaker** (CLOSED → OPEN → HALF_OPEN per service) — `data/circuit_breaker.py`
  3. **Retry-with-backoff** (1s → 2s → 4s) for transient failures
  4. **Stale-fallback** — if a fetch fails AND the circuit is open, the last-good
     snapshot is returned so the UI keeps working through API outages.

Macro data is sourced from Sanket's Global Macro bond-ETF universe via yfinance
(replaces the broken Stooq direct-yield endpoints, which started returning HTML
error pages instead of CSV in late 2025).
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (
    GLOBAL_MACRO_MAP,
    MACRO_SYMBOLS_YF,
    COMMODITY_TARGETS,
    INDEX_TARGETS_MAP,
    ALL_TARGETS,
)
from data.cache import ohlcv_cache, macro_cache
from data.circuit_breaker import (
    yfinance_circuit,
    CircuitBreakerError,
    RetryWithBackoff,
)

log = logging.getLogger(__name__)


# ─── Constituent OHLCV (yfinance) ────────────────────────────────────────────

@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download(
    symbols_tuple: tuple[str, ...],
    start_yf: str,
    end_yf: str,
) -> pd.DataFrame:
    """Single raw yfinance batch call — wrapped with retry."""
    raw = yf.download(
        list(symbols_tuple),
        start=start_yf,
        end=end_yf,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance batch response")
    return raw


def fetch_constituent_ohlcv(
    symbols: list[str],
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV for an index/basket's constituents via yfinance."""
    start_yf = str(pd.Timestamp(start_date).date())
    end_yf = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))
    # Sorted tuple → deterministic cache key regardless of input order.
    sym_key = tuple(sorted(symbols))

    cached = ohlcv_cache.get(sym_key, start_yf, end_yf)
    if cached is not None:
        return cached

    try:
        raw = yfinance_circuit.call(
            _yfinance_batch_download, sym_key, start_yf, end_yf
        )
    except CircuitBreakerError as e:
        stale = ohlcv_cache.get_stale(sym_key, start_yf, end_yf)
        if stale is not None:
            log.warning("yfinance circuit open, serving stale OHLCV")
            return stale
        log.error("yfinance unavailable, no stale snapshot: %s", e)
        return {}
    except Exception as e:
        log.error("yfinance batch download failed: %s", e)
        stale = ohlcv_cache.get_stale(sym_key, start_yf, end_yf)
        return stale if stale is not None else {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                sub = raw.xs(sym, level=0, axis=1)
                close_col = sub.get(
                    "Close", sub.iloc[:, 0] if len(sub.columns) else pd.Series()
                )
                if not sub.empty and not close_col.isnull().all():
                    if isinstance(sub.columns, pd.MultiIndex):
                        sub.columns = [c[0] for c in sub.columns]
                    result[sym] = sub
            except KeyError:
                pass

    if result:
        ohlcv_cache.put(sym_key, start_yf, end_yf, value=result)
    return result


# ─── Macro data (yfinance Global Macro universe + commodities/FX) ───────────


def _load_macro_snapshots_newest_first() -> list[pd.DataFrame]:
    """Prior cached macro frames (ticker-columned), newest first, for column backfill."""
    snaps: list[pd.DataFrame] = []
    try:
        paths = sorted(
            macro_cache._disk_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:  # noqa: BLE001
        return snaps
    for p in paths:
        try:
            with open(p, "rb") as f:
                val, _ = pickle.load(f)
            if isinstance(val, pd.DataFrame) and not val.empty:
                snaps.append(val)
        except Exception:  # noqa: BLE001
            continue
    return snaps


# Module-level registry of the most recent backfill's per-column staleness —
# {ticker: last_native_date} for columns filled from a snapshot that was
# itself older than STALE_BACKFILL_DAYS behind the current frame. Read by
# app.py's freshness section (see B1 in the audit) to surface a warning
# instead of silently ffilling a possibly weeks-old value across the whole
# frame with no user-visible signal beyond a log line nobody sees on
# Streamlit. Cleared/repopulated on every call to _backfill_missing_columns.
STALE_BACKFILLS: dict[str, str] = {}

# How many trading days a backfilled column's true last-native observation is
# allowed to lag the frame's end before it's dropped instead of filled. A
# stale snapshot ffilled across N weeks of rows would flatten that
# predictor's momentum/PCA loading for the whole window — worse than not
# having the column at all past this point.
STALE_BACKFILL_DAYS = 10


def _backfill_missing_columns(combined: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Refill columns yfinance dropped/rate-limited (absent or all-NaN) from the most
    recent prior snapshot that has them.

    yfinance routinely rate-limits a handful of tickers per batch (e.g. GC=F)
    while the rest succeed. The partial frame is non-empty, so it bypasses the
    all-or-nothing stale fallback and gets cached — silently dropping a TARGET column
    (Gold = GC=F) and failing the walk-forward with "Need 1500+ data points". Backfilling
    the missing columns keeps the frame complete and re-heals the cache. Scans
    newest→oldest so it recovers even a previously-poisoned snapshot.

    A backfilled column's tail is usually only ~1 day stale (the snapshot
    that just missed this ticker), but "newest snapshot" can itself be old
    (e.g. after a multi-day gap in usage) — nothing previously enforced the
    "~1 day" claim. A column whose TRUE last-native observation (before the
    snapshot's own carry-forward) is more than STALE_BACKFILL_DAYS trading
    days behind the frame's end is dropped rather than filled — flat
    momentum ffilled across weeks would distort that predictor's PCA loading
    silently. Columns backfilled from a snapshot within the threshold are
    still recorded in STALE_BACKFILLS (with their true last-native date) so
    the UI can surface which predictors were carried from a snapshot at all.
    """
    if combined.empty:
        return combined
    missing = [t for t in tickers if t not in combined.columns or combined[t].isna().all()]
    STALE_BACKFILLS.clear()
    if not missing:
        return combined

    frame_end = combined.index.max()
    filled: list[str] = []
    dropped: list[str] = []
    for snap in _load_macro_snapshots_newest_first():
        if not missing:
            break
        snap_aligned = snap.reindex(combined.index).ffill()
        for t in list(missing):
            if t in snap_aligned.columns and not snap_aligned[t].isna().all():
                # True last-native date for this ticker WITHIN the snapshot,
                # before our own ffill/reindex — i.e. how old the underlying
                # observation actually is, not how far we carried it forward.
                native = snap[t].dropna()
                last_native = native.index.max() if len(native) else None
                if last_native is not None:
                    days_behind = int(np.busday_count(
                        pd.Timestamp(last_native).date(), pd.Timestamp(frame_end).date()
                    ))
                    if days_behind > STALE_BACKFILL_DAYS:
                        dropped.append(t)
                        missing.remove(t)
                        continue
                    STALE_BACKFILLS[t] = str(pd.Timestamp(last_native).date())
                combined[t] = snap_aligned[t]
                filled.append(t)
                missing.remove(t)

    if filled:
        log.warning("Macro backfill from snapshot for rate-limited/missing tickers: %s", filled)
    if dropped:
        log.warning(
            "Macro tickers dropped (snapshot backfill was > %d trading days stale): %s",
            STALE_BACKFILL_DAYS, dropped,
        )
        combined = combined.drop(columns=dropped, errors="ignore")
    if missing:
        log.warning("Macro tickers still unavailable after backfill (no prior snapshot): %s", missing)
    return combined


def _fetch_macro_live_uncached(start_str: str, end_str: str) -> pd.DataFrame:
    """Single yfinance batch for the full macro universe (Sanket-style).

    Combines Sanket's Global Macro bond ETFs (proxy for global yield dynamics)
    with the existing commodity + FX symbols. One batch call, one circuit hit.
    """
    tickers = tuple(sorted(
        set(GLOBAL_MACRO_MAP.values())
        | set(MACRO_SYMBOLS_YF.values())
        | set(INDEX_TARGETS_MAP.values())  # index price levels (Aarambh targets)
    ))
    if not tickers:
        return pd.DataFrame()

    try:
        yf_raw = yfinance_circuit.call(
            _yfinance_batch_download_macro, tickers, start_str, end_str
        )
    except CircuitBreakerError as e:
        log.warning("yfinance macro fetch blocked by circuit: %s", e)
        return pd.DataFrame()
    except Exception as e:
        log.warning("Yahoo Finance macro fetch failed: %s", e)
        return pd.DataFrame()

    if yf_raw is None or yf_raw.empty:
        return pd.DataFrame()

    if isinstance(yf_raw.columns, pd.MultiIndex):
        if "Close" in yf_raw.columns.get_level_values(0):
            combined = yf_raw["Close"]
        elif "Adj Close" in yf_raw.columns.get_level_values(0):
            combined = yf_raw["Adj Close"]
        else:
            combined = yf_raw
    else:
        combined = yf_raw

    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    combined = combined.sort_index()

    if not combined.empty:
        combined = combined.ffill()
        # yfinance rate-limits a few tickers per batch; refill those columns from the
        # last good snapshot so a partial response never drops a target (e.g. Gold=GC=F).
        combined = _backfill_missing_columns(combined, tickers)
    return combined


@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download_macro(
    tickers_tuple: tuple[str, ...], start: str, end: str
) -> pd.DataFrame:
    """Macro yfinance batch fetch — separated to allow distinct retry budget.

    Uses ``auto_adjust=True`` and ``threads=True`` (matching Sanket's
    ``fetch_batch_data``) so the macro universe is pulled in parallel by
    yfinance's internal pool.
    """
    raw = yf.download(
        list(tickers_tuple),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        raise ValueError("Empty yfinance macro response")
    return raw


def fetch_macro_live(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Fetch macro indicators with cache + rate-limit + circuit + stale fallback."""
    start_str = str(pd.Timestamp(start_date).date())
    end_str = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))

    cached = macro_cache.get(start_str, end_str)
    if cached is not None:
        return cached

    try:
        combined = _fetch_macro_live_uncached(start_str, end_str)
    except Exception as e:
        log.error("Macro fetch raised unexpectedly: %s", e)
        combined = pd.DataFrame()

    if not combined.empty:
        macro_cache.put(start_str, end_str, value=combined)
        return combined

    # Nothing came back this run — try a stale snapshot before returning empty.
    stale = macro_cache.get_stale(start_str, end_str)
    if stale is not None:
        log.warning("Macro fetch empty; serving last-good snapshot")
        return stale
    return combined


# ─── Commodity model dataset (Aarambh matrix from yfinance) ──────────────────


def fetch_commodity_dataset(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> tuple[pd.DataFrame | None, str | None]:
    """Build the Aarambh model matrix from the full yfinance macro universe.

    Wraps :func:`fetch_macro_live`, restricts the Close frame to the combined
    ``GLOBAL_MACRO_MAP`` (bond/rates/equity/risk/real-asset ETFs) +
    ``MACRO_SYMBOLS_YF`` (commodities + FX) universe, renames yfinance tickers
    to friendly names (Gold / Silver / Copper / US 10-Year Yield / VIX / …),
    and exposes the DatetimeIndex as a ``DATE`` column. The result is the model
    matrix the Streamlit app feeds the walk-forward engine: numeric predictor
    columns + a date column.

    Returns ``(df, None)`` on success or ``(None, error_message)`` on failure.
    """
    macro_df = fetch_macro_live(start_date, end_date)
    if macro_df is None or macro_df.empty:
        return None, "No macro/commodity data returned from yfinance."

    # ticker → friendly name (inverse of the combined friendly → ticker maps).
    name_to_ticker = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **INDEX_TARGETS_MAP}
    ticker_to_name = {ticker: name for name, ticker in name_to_ticker.items()}
    present = [t for t in ticker_to_name if t in macro_df.columns]
    if not present:
        return None, "Macro/commodity symbols missing from yfinance response."

    renamed = macro_df[present].rename(columns=ticker_to_name)

    # Drop columns yfinance returned empty / near-empty: failed or delisted
    # tickers (e.g. a renamed ETF) come back as all-NaN. Such a column survives the
    # app's causal ffill() still all-NaN and would wipe every row at the subsequent
    # dropna() step. Require ≥20% real coverage; always retain the target columns
    # regardless (so they stay selectable). NOTE: a retained-but-sparse TARGET (e.g.
    # ^CNXSC / Nifty Smallcap 100) still leaks into the predictor set when it is not
    # the active target — the app's per-feature history guard drops it there.
    coverage = renamed.notna().mean()
    keep = [c for c in renamed.columns if coverage[c] >= 0.20]
    for tgt in ALL_TARGETS:  # always retain every target column (commodity/FX/index)
        if tgt in renamed.columns and tgt not in keep:
            keep.append(tgt)
    if not keep:
        return None, "All macro/commodity columns were empty."

    df = renamed[keep].copy()

    # Drop weekend rows. A few weekend-trading tickers (FX crosses) create Sat/Sun
    # index dates; the upstream ffill then back-fills the other ~180 columns, yielding
    # a fully ff-filled, entirely-stale weekend row that *looks* complete (100%
    # coverage) but carries Friday's values. That bogus latest row produced illogical
    # signals and a card-vs-plot mismatch (the card read it; Nirnay-aligned plots
    # dropped it). Equities/commodities don't trade weekends, so these rows are pure
    # artifacts — remove them so the latest row is always a real trading day.
    df = df[df.index.dayofweek < 5]

    # Inject exogenous (non-yfinance) target columns — e.g. Jeera (NCDEX cumin)
    # from a published Google Sheet. Each is reindexed onto the macro (US-
    # calendar) index and forward-filled so the NCDEX series aligns to the model
    # matrix; leading dates with no prior price stay NaN and are dropped by the
    # app's per-target dropna. A fetch failure simply omits the column (the
    # target then can't be selected) rather than breaking the whole dataset.
    #
    # DELIBERATE: the macro index is the calendar SPINE. A sheet date the macro
    # lacks (an NSE-trading day that's a US holiday) is dropped by the reindex —
    # measured at ~4 days / 6y for the Nifty PE sheet (negligible). The alternative
    # (union the sheet's dates in) would re-create partial rows — India fresh, US
    # ff-filled — i.e. the very thing the partial-session gate flags. So the spine
    # stays; true per-market alignment would need exchange calendars (future work).
    for _name, _series in _fetch_exogenous_targets(df.index).items():
        df[_name] = _series

    df.insert(0, "DATE", pd.to_datetime(df.index))
    df = df.reset_index(drop=True)
    return df, None


def _fetch_exogenous_targets(index: pd.Index) -> dict[str, pd.Series]:
    """Fetch non-yfinance target series and align them to the macro ``index``.

    Returns a ``{column_name: aligned_series}`` map. Resilient: any source that
    fails (live + cache + committed snapshot all unavailable) is skipped.
    """
    out: dict[str, pd.Series] = {}
    from data.sheets import SHEET_SOURCES, fetch_sheet_series
    for _name in SHEET_SOURCES:
        try:
            s = fetch_sheet_series(_name, index.min(), index.max())
            if s is not None and not s.empty:
                out[_name] = s.reindex(index).ffill()
        except Exception as e:  # noqa: BLE001
            log.warning("Exogenous %s fetch skipped: %s", _name, e)
    return out
