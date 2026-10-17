"""
Tattva — Unified data fetcher: yfinance macro / commodity / OHLCV universe.
तत्त्व (Tattva) — "Principle / Essence"

Each external call is wrapped with:
  1. **Two-tier cache** (memory + disk, TTL + versioned keys) — `data/cache.py`
  2. **Circuit breaker** (CLOSED → OPEN → HALF_OPEN per service) — `data/circuit_breaker.py`
  3. **Retry-with-backoff** (1s → 2s → 4s) for transient (whole-batch) failures.
     An answer of "no data for that symbol" is a :class:`NoDataError` — not
     retried, not counted against the service (see ``data/circuit_breaker.py``).
  4. **Partial-success completion** — a batch is ONE yfinance call, but yfinance
     rate-limits a few tickers per batch, so it can come back incomplete. Rather
     than caching the incomplete result: OHLCV does one targeted LIVE re-fetch
     of just the missing symbols; the macro universe backfills the missing
     columns from the last-good snapshot (_backfill_missing_columns), and says
     which on the frame itself (``attrs["stale_backfills"]``).
  5. **Stale-fallback** — if a fetch fails AND the circuit is open, the last-good
     snapshot is returned so the UI keeps working through API outages, and the
     frame says so (``attrs["source"] == "stale-snapshot"``).

What the shared model matrix CONTAINS is decided by configuration, never by a
session: it is the macro universe plus ``core.config.CATALOGUE_TARGETS``, the
catalogue frozen at import. Free-form stock symbols a user types are fetched one
at a time (``fetch_stock_target_series``) and never enter this frame.

Macro data is sourced from Sanket's Global Macro bond-ETF universe via yfinance
(replaces the broken Stooq direct-yield endpoints, which started returning HTML
error pages instead of CSV in late 2025).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (
    GLOBAL_MACRO_MAP,
    MACRO_SYMBOLS_YF,
    INDEX_TARGETS_MAP,
    CATALOGUE_TARGETS,
)
from data.cache import ohlcv_cache, macro_cache
from data.circuit_breaker import (
    yfinance_circuit,
    CircuitBreakerError,
    NoDataError,
    RetryWithBackoff,
)

log = logging.getLogger(__name__)


#: Tickers naming a SOURCE rather than a yfinance symbol (Jeera's NCDEX sheet,
#: the Nifty PE sheet); each has its own fetch path, and sending one to
#: yf.download is a guaranteed 404 that used to count against the yfinance
#: circuit on every run of those targets.
_NON_YF_TICKER_SUFFIXES = (".NCDEX", ".SHEET")


def is_yfinance_ticker(ticker: str | None) -> bool:
    """Can this ticker be sent to yfinance at all? False for source sentinels."""
    return bool(ticker) and not str(ticker).upper().endswith(_NON_YF_TICKER_SUFFIXES)


# ─── Telling "no such symbol" from "service down" ────────────────────────────
# yf.download never raises for a bad ticker. It swallows the per-symbol error,
# logs it on the "yfinance" logger, and returns an empty frame — and it does
# exactly the same for a rate limit. The only place the difference survives is
# the log line, so the call is made with a capturing handler attached and an
# empty answer is classified by what yfinance said about it. Unexplained
# emptiness is treated as a service failure, which is the conservative reading:
# it was the only reading before this existed.

_NO_DATA_MARKERS = ("possibly delisted", "no price data found", "quote not found",
                    "no timezone found", "no data found", "http error 404")
_SERVICE_MARKERS = ("too many requests", "rate limit", "http error 429",
                    "timed out", "timeout", "connection", "ssl", "curl:",
                    "http error 5")


class _YFLogCapture(logging.Handler):
    """Collect yfinance's ERROR records raised by THIS thread's call.

    Records from other threads are kept too — yf.download with threads=True
    logs its summary from the calling thread but per-ticker errors from its
    pool — so a concurrent session's rate limit can leak into this capture.
    That errs toward "service failure", the same direction as the default.
    """

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []
        self._lock_ = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage().lower()
        except Exception:  # noqa: BLE001
            return
        with self._lock_:
            self.messages.append(msg)


def _download_classified(**kwargs) -> pd.DataFrame:
    """``yf.download(**kwargs)``; an empty result raises NoDataError or ValueError."""
    cap = _YFLogCapture()
    yf_log = logging.getLogger("yfinance")
    yf_log.addHandler(cap)
    try:
        raw = yf.download(**kwargs)
    finally:
        yf_log.removeHandler(cap)
    if raw is not None and not (hasattr(raw, "empty") and raw.empty):
        return raw
    said = " | ".join(cap.messages)
    if any(m in said for m in _SERVICE_MARKERS):
        raise ValueError(f"yfinance service failure: {said[:200]}")
    if any(m in said for m in _NO_DATA_MARKERS):
        raise NoDataError(f"no data: {said[:200]}")
    raise ValueError("Empty yfinance response")


# ─── OHLCV (yfinance) ────────────────────────────────────────────────────────

@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _yfinance_batch_download(
    symbols_tuple: tuple[str, ...],
    start_yf: str,
    end_yf: str,
) -> pd.DataFrame:
    """Single raw yfinance batch call — wrapped with retry.

    ``auto_adjust=False`` — not dividend-adjusted, deliberately. See the note
    on the macro batch below: dividend-adjusted history is rewritten backwards
    on every distribution, which is incompatible with this system's
    non-repainting guarantee.
    """
    return _download_classified(
        tickers=list(symbols_tuple),
        start=start_yf,
        end=end_yf,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )


def _ohlcv_key(symbols, start_date, end_date) -> tuple[tuple[str, ...], str, str]:
    """(sorted symbols, start, end+1d) — one key convention for every OHLCV read."""
    start_yf = str(pd.Timestamp(start_date).date())
    end_yf = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))
    return tuple(sorted(symbols)), start_yf, end_yf


def fetch_ohlcv(
    symbols: list[str],
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV for a set of yfinance symbols → ``{symbol: frame}``.

    Used for a target's own OHLCV (Swayam's input), the catalogue targets the
    macro batch does not carry, and single free-form stock prices. ``force``
    skips the fresh-cache read for this call only (a user Refresh); the result
    is still cached, and a failed live pull still falls back to the last-good
    snapshot. Source sentinels (sheet / NCDEX tickers) are dropped before any
    network call — they have no OHLCV on yfinance.
    """
    symbols = [s for s in symbols if is_yfinance_ticker(s)]
    if not symbols:
        return {}
    sym_key, start_yf, end_yf = _ohlcv_key(symbols, start_date, end_date)

    if not force:
        cached = ohlcv_cache.get(sym_key, start_yf, end_yf)
        if cached is not None:
            return cached

    try:
        raw = yfinance_circuit.call(
            _yfinance_batch_download, sym_key, start_yf, end_yf
        )
    except NoDataError as e:
        # An answer, not an outage: yfinance has nothing for these symbols.
        log.info("yfinance has no data for %s: %s", list(sym_key)[:5], e)
        stale = ohlcv_cache.get_stale(sym_key, start_yf, end_yf)
        return stale if stale is not None else {}
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

    result = _parse_ohlcv_batch(raw, symbols)

    # Partial success: yfinance rate-limits a handful of tickers per batch, so a
    # multi-symbol batch can come back short. Do ONE targeted re-fetch of JUST the
    # missing symbols (goes through the same circuit + backoff) to complete the set,
    # rather than caching an incomplete result that would be reused as-is. Only when
    # SOME tickers succeeded (a total-empty batch is an outage, not per-ticker limits
    # — the circuit/stale path handles that).
    missing = [s for s in symbols if s not in result]
    if result and missing:
        try:
            raw2 = yfinance_circuit.call(
                _yfinance_batch_download, tuple(sorted(missing)), start_yf, end_yf
            )
            recovered = _parse_ohlcv_batch(raw2, missing)
            if recovered:
                log.info("re-fetch recovered %d/%d rate-limited tickers",
                         len(recovered), len(missing))
                result.update(recovered)
        except Exception as e:   # circuit open / still limited — keep what we have
            log.warning("targeted re-fetch of %d missing tickers failed: %s",
                        len(missing), e)

    if result:
        ohlcv_cache.put(sym_key, start_yf, end_yf, value=result)
    return result


def probe_symbol(
    ticker: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> tuple[pd.Series | None, str]:
    """Does ``ticker`` exist on yfinance? → ``(close_series | None, outcome)``.

    ``outcome`` is ``"ok"`` (usable series, cached under the same key
    :func:`fetch_ohlcv` reads, so the analysis that follows is a cache hit),
    ``"not_found"`` (yfinance answered and has nothing usable), or
    ``"unavailable"`` (rate limit, timeout, circuit open — the question was
    never answered). Symbol resolution needs that third outcome: a symbol that
    could not be checked must not be reported, or remembered, as invalid.
    """
    if not is_yfinance_ticker(ticker):
        return None, "not_found"
    sym_key, start_yf, end_yf = _ohlcv_key([ticker], start_date, end_date)
    result = ohlcv_cache.get(sym_key, start_yf, end_yf)
    if result is None:
        try:
            raw = yfinance_circuit.call(
                _yfinance_batch_download, sym_key, start_yf, end_yf
            )
        except NoDataError:
            return None, "not_found"
        except Exception as e:  # noqa: BLE001 - circuit open / service failure
            log.warning("Symbol probe %s could not reach yfinance: %s", ticker, e)
            return None, "unavailable"
        result = _parse_ohlcv_batch(raw, [ticker])
        if result:
            ohlcv_cache.put(sym_key, start_yf, end_yf, value=result)
    s = _close_series(result.get(ticker))
    return (s, "ok") if s is not None else (None, "not_found")


def _close_series(odf: pd.DataFrame | None) -> pd.Series | None:
    """A usable Close series from one symbol's OHLCV frame, or None (<20 rows)."""
    if odf is None or odf.empty or "Close" not in odf.columns:
        return None
    s = pd.to_numeric(odf["Close"], errors="coerce").dropna()
    if len(s) < 20:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _parse_ohlcv_batch(raw, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Extract a {symbol: OHLCV frame} dict from a group_by='ticker' yfinance batch.
    Symbols yfinance dropped (rate-limited/delisted) simply don't appear."""
    out: dict[str, pd.DataFrame] = {}
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
                    out[sym] = sub
            except KeyError:
                pass
    return out


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
    recorded ON THE FRAME — ``combined.attrs["stale_backfills"] = {ticker:
    last_native_date}`` — so the UI can say which predictors were carried from
    a snapshot at all.

    On the frame, not in a registry beside it. The registry this replaced was
    keyed by Streamlit session and filled only on a LIVE fetch, while the frame
    itself is cached process-wide: a second session served the same rebuilt
    frame from cache got no warning, and a later live fetch overwrote the
    notice with a different pull than the one the engines had used. Metadata
    that travels with the data cannot describe the wrong data. (pandas keeps
    ``attrs`` through pickling, so the disk cache preserves it too.)
    """
    if combined.empty:
        return combined
    registry: dict[str, str] = {}
    combined.attrs["stale_backfills"] = registry
    missing = [t for t in tickers if t not in combined.columns or combined[t].isna().all()]
    if not missing:
        return combined

    frame_end = combined.index.max()
    filled: list[str] = []
    dropped: list[str] = []
    # SELECT BY COVERAGE, NOT BY RECENCY.
    #
    # This loop used to take the newest snapshot that merely CONTAINED the
    # ticker. The snapshot pool is not homogeneous — observed spans on one
    # machine were 2610, 2089, 783 and 2351 rows — so "newest" can be a frame
    # covering seven fewer years. Reindexed onto the current calendar the
    # missing head becomes NaN, ffill cannot fill backwards, and the rebuilt
    # column silently loses ~1800 rows of history. That instrument's admission
    # and print counts change, the factor basis changes with them, and every
    # published date moves.
    #
    # Measured 2026-08-17: two fetches minutes apart rebuilt different tickers
    # (transient timeouts differ run to run) and 1563 settled input cells
    # differed, moving 5743 published output cells from the first post-burn-in
    # row onward — on 1 of 4 fetch pairs. The snapshots themselves are NOT the
    # problem: on shared dates they match a live pull to ~1e-4, with zero dates
    # invented by ffill. The defect was choosing among them by mtime.
    #
    # So rank candidates per ticker by how much of the REQUIRED index each one
    # actually covers, and use recency only to break ties.
    _snapshots = _load_macro_snapshots_newest_first()

    def _ranked_snapshots(ticker):
        # Snapshots carrying `ticker`, best coverage of combined.index first.
        scored = []
        for order, snap in enumerate(_snapshots):
            if ticker not in snap.columns:
                continue
            col = snap[ticker]
            if col.dropna().empty:
                continue
            covered = int(col.reindex(combined.index).notna().sum())
            scored.append((-covered, order, snap))     # order = recency tiebreak
        scored.sort(key=lambda z: (z[0], z[1]))
        return [z[2] for z in scored]

    for t in list(missing):
        for snap in _ranked_snapshots(t):
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
                    # Too stale to carry — but try the remaining candidates
                    # before giving up. Previously the first stale hit dropped
                    # the ticker outright, discarding a usable older snapshot.
                    continue
                registry[t] = str(pd.Timestamp(last_native).date())
            combined[t] = snap.reindex(combined.index).ffill()[t]
            filled.append(t)
            missing.remove(t)
            break
        else:
            # No candidate was both present and fresh enough.
            registry.pop(t, None)
            dropped.append(t)
            if t in missing:
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


def _macro_tickers() -> tuple[str, ...]:
    """The macro batch's ticker set: predictors + index price levels."""
    return tuple(sorted(
        set(GLOBAL_MACRO_MAP.values())
        | set(MACRO_SYMBOLS_YF.values())
        | set(INDEX_TARGETS_MAP.values())  # index price levels (Mūla targets)
    ))


def _universe_tag(tickers: tuple[str, ...]) -> str:
    """Short digest of the ticker set, baked into the macro cache key.

    The key used to be ``(start, end)`` alone, so a deploy that added an
    instrument to the universe kept serving that day's cached frames — which
    silently lacked the new column — until the TTL ran out.
    """
    return hashlib.sha1("|".join(tickers).encode("utf-8")).hexdigest()[:8]


def _fetch_macro_live_uncached(start_str: str, end_str: str) -> pd.DataFrame:
    """Single yfinance batch for the full macro universe (Sanket-style).

    Combines Sanket's Global Macro bond ETFs (proxy for global yield dynamics)
    with the existing commodity + FX symbols. One batch call, one circuit hit.
    """
    tickers = _macro_tickers()
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

    Uses ``auto_adjust=False`` (no dividend rewriting — see below) and
    ``threads=True`` (matching Sanket's
    ``fetch_batch_data``) so the macro universe is pulled in parallel by
    yfinance's internal pool.
    """
    # ── auto_adjust=False — the non-repainting guarantee lives here ──────
    # Adjusted prices are not a fixed history. yfinance applies each dividend
    # and split BACKWARDS across the whole series, so a past close is rewritten
    # every time an instrument distributes. Measured on current panel members:
    # a close one year back had already been rescaled by 5.67% (HYG), 4.42%
    # (LQD), 4.38% (TLT), 3.73% (VNQ) — and 13 ex-dividend events landed across
    # just eight names in 90 days. The panel carries ~209 instruments, mostly
    # distributing ETFs, so on a typical day several have their history moved.
    #
    # A deterministic engine fed a rewritten history publishes different values
    # for dates it had already published. That is exactly the repaint two runs
    # 20 minutes apart showed: identical row counts and date range, fair value
    # 23,449 -> 23,832 (45x the spot move), instruments admitted 61 -> 62, two
    # walk-forward windows flipping sign. The engine was never at fault — the
    # reproducibility harness holds one dataset fixed and truncates it, which
    # proves the engine does not look ahead but cannot see the INPUT change
    # underneath it.
    #
    # With auto_adjust=False a printed close is not rewritten by distributions.
    # It is not quite immutable: Yahoo's `Close` is still SPLIT-adjusted, so a
    # split rescales an instrument's past — rare, and the right behaviour (a
    # split is not a price move). The cost is that a dividend shows as a real
    # price drop in the ETF predictors; for a valuation on the LEVEL of log
    # price that is defensible (the traded level is what it is), and the
    # index / futures / FX targets carry no adjustment at all.
    raw = _download_classified(
        tickers=list(tickers_tuple),
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=True,
    )
    return raw


def _with_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """A shallow copy of ``frame`` whose ``attrs["source"]`` says where it came
    from — without writing that into the cached object every caller shares."""
    out = frame.copy(deep=False)
    out.attrs = {**frame.attrs, "source": source}
    return out


def fetch_macro_live(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch macro indicators with cache + rate-limit + circuit + stale fallback.

    ``force`` skips the fresh-cache read for this call only (a user Refresh).
    The returned frame's ``attrs["source"]`` is ``"live"``, ``"cache"`` or
    ``"stale-snapshot"`` — the last meaning a live pull was wanted and failed,
    which a forced refresh must be able to say rather than quietly hand back
    yesterday's data.
    """
    start_str = str(pd.Timestamp(start_date).date())
    end_str = str(pd.Timestamp(end_date).date() + pd.Timedelta(days=1))
    tag = _universe_tag(_macro_tickers())

    if not force:
        cached = macro_cache.get(start_str, end_str, tag)
        if cached is not None:
            return _with_source(cached, "cache")

    try:
        combined = _fetch_macro_live_uncached(start_str, end_str)
    except Exception as e:
        log.error("Macro fetch raised unexpectedly: %s", e)
        combined = pd.DataFrame()

    if not combined.empty:
        macro_cache.put(start_str, end_str, tag, value=combined)
        return _with_source(combined, "live")

    # Nothing came back this run — try a stale snapshot before returning empty.
    stale = macro_cache.get_stale(start_str, end_str, tag)
    if stale is not None:
        log.warning("Macro fetch empty; serving last-good snapshot")
        return _with_source(stale, "stale-snapshot")
    return combined


# ─── Commodity model dataset (FVO matrix from yfinance) ──────────────────


def fetch_commodity_dataset(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    force: bool = False,
) -> tuple[pd.DataFrame | None, str | None]:
    """Build the shared model matrix from the full yfinance macro universe.

    Wraps :func:`fetch_macro_live`, restricts the Close frame to the combined
    ``GLOBAL_MACRO_MAP`` (bond/rates/equity/risk/real-asset ETFs) +
    ``MACRO_SYMBOLS_YF`` (commodities + FX) universe, renames yfinance tickers
    to friendly names (Gold / Silver / Copper / US 10-Year Yield / VIX / …),
    adds the catalogue targets the batch does not carry, and exposes the
    DatetimeIndex as a ``DATE`` column. Both engines read this one frame.

    ``force=True`` re-pulls live for this call (a user Refresh): the macro
    batch, the catalogue-target OHLCV and the sheet series each skip their
    fresh-cache read once. Nothing else is affected — the next ordinary call is
    a cache hit on what this one fetched.

    ``df.attrs`` carries what the frame cannot show in its cells:
      * ``source``           — "live" / "cache" / "stale-snapshot" (the macro batch)
      * ``stale_backfills``  — {friendly name: "YYYY-MM-DD"} predictors rebuilt
                               from a prior snapshot this fetch (a rebuilt column
                               is a DIFFERENT series, not just a staler one)
      * ``panel_columns`` / ``panel_fingerprint`` — the composition digest

    Returns ``(df, None)`` on success or ``(None, error_message)`` on failure.
    """
    macro_df = fetch_macro_live(start_date, end_date, force=force)
    if macro_df is None or macro_df.empty:
        return None, "No macro/commodity data returned from yfinance."
    _source = str(macro_df.attrs.get("source", "live"))
    _backfilled = dict(macro_df.attrs.get("stale_backfills") or {})

    # ticker → friendly name (inverse of the combined friendly → ticker maps).
    name_to_ticker = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **INDEX_TARGETS_MAP}
    ticker_to_name = {ticker: name for name, ticker in name_to_ticker.items()}
    present = [t for t in ticker_to_name if t in macro_df.columns]
    if not present:
        return None, "Macro/commodity symbols missing from yfinance response."

    renamed = macro_df[present].rename(columns=ticker_to_name)
    _backfilled = {ticker_to_name.get(t, t): d for t, d in _backfilled.items()}

    # Drop columns yfinance returned empty / near-empty: failed or delisted
    # tickers (e.g. a renamed ETF) come back as all-NaN. Such a column survives the
    # app's causal ffill() still all-NaN and would wipe every row at the subsequent
    # dropna() step. Require ≥20% real coverage; always retain the target columns
    # regardless (so they stay selectable). NOTE: a retained-but-sparse TARGET
    # still reaches the predictor set when it is not the active target — the app's
    # per-feature history guard drops it there.
    coverage = renamed.notna().mean()
    keep = [c for c in renamed.columns if coverage[c] >= 0.20]
    for tgt in CATALOGUE_TARGETS:  # always retain every catalogue target column
        if tgt in renamed.columns and tgt not in keep:
            keep.append(tgt)
    if not keep:
        return None, "All macro/commodity columns were empty."

    df = renamed[keep].copy()

    # Drop weekend rows. A few weekend-trading tickers (FX crosses) create Sat/Sun
    # index dates; the upstream ffill then back-fills the other ~180 columns, yielding
    # a fully ff-filled, entirely-stale weekend row that *looks* complete (100%
    # coverage) but carries Friday's values. That bogus latest row produced illogical
    # signals and a card-vs-plot mismatch (the card read it; Swayam-aligned plots
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
    for _name, _series in _fetch_exogenous_targets(df.index, force=force).items():
        df[_name] = _series

    # Catalogue targets that are yfinance symbols but NOT members of the macro
    # universe. A target arrives above only if it is also a PREDICTOR — true of
    # the commodities and FX that predate crypto and the base metals, false of
    # those. Injected here, once, because this frame is the shared model matrix:
    # the app reads it, the research preflight reads it, and each tuning study
    # loads it independently. Patching any one consumer leaves the others with a
    # different universe — which is how the suite reported "36/45 targets" and
    # the Swayam study died on `KeyError: ['Aluminium']` while the app itself
    # ran that target perfectly well through its own per-target injection.
    for _name, _series in _fetch_catalogue_targets(
            df.index, df.columns, start_date, end_date, force=force).items():
        df[_name] = _series

    # ── Panel fingerprint ────────────────────────────────────────────────
    # The published history is a function of the panel's COMPOSITION, not just
    # its values: the factor basis is derived from whoever is in the
    # cross-section, so losing an instrument to a rate limit rewrites every
    # date. Measured on a 24-name panel, dropping one predictor moved past
    # fair values by a median 0.31% and up to 8%; dropping three, up to 24%.
    #
    # Without durable storage there is nowhere to persist the expected panel,
    # so the next best thing is to make a change VISIBLE. Two runs reporting
    # the same fingerprint provably used the same cross-section; a changed
    # fingerprint says the difference came from the data, not the code, before
    # anyone spends an afternoon on the engine.
    _columns = sorted(df.columns)
    _fingerprint = _panel_fingerprint(df)

    df.insert(0, "DATE", pd.to_datetime(df.index))
    df = df.reset_index(drop=True)
    # Set last, explicitly, rather than trusting pandas to carry attrs through
    # the renames, joins and resets above.
    df.attrs = {
        "source": _source,
        "stale_backfills": {k: v for k, v in _backfilled.items() if k in df.columns},
        "panel_columns": _columns,
        "panel_fingerprint": _fingerprint,
    }
    return df, None


def _panel_fingerprint(df: "pd.DataFrame", *, content: bool = True) -> str:
    """A short digest of the panel — membership, span, and SETTLED CONTENT.

    The original version hashed shape and membership only, on the reasoning that
    prices legitimately move on the newest bar so a content digest would change
    every run and say nothing. That reasoning was half right and the omission was
    material: two fetches minutes apart produced IDENTICAL fingerprints while the
    engine's settled `ConvictionRaw` differed by 50% on a past date. The digest
    reported "nothing changed" during an actual repaint, which is the one job it
    had.

    The fix is to hash content while EXCLUDING the newest row. Intraday ticking
    lives entirely in that last bar, so dropping it keeps the digest stable
    through a session — but any change to a settled value, from a vendor
    revision, a snapshot/live mismatch, or a differing fill, moves it
    immediately. A digest that is stable when it should be and moves when
    history moves is the one worth printing to the console.

    `content=False` recovers the old membership-only behaviour, for callers that
    genuinely want to ask "is this the same cross-section" independently of the
    values in it.
    """
    import hashlib
    h = hashlib.sha256()
    cols = sorted(str(c) for c in df.columns)
    h.update(("\n".join(cols)).encode("utf-8"))
    try:
        h.update(f"||{df.index.min()}|{df.index.max()}|{len(df)}".encode("utf-8"))
    except Exception:            # noqa: BLE001 - a digest must never break a run
        h.update(b"||?")
    if content:
        try:
            settled = df.iloc[:-1] if len(df) > 1 else df.iloc[:0]
            # float64 bytes in a fixed column order: exact, and independent of
            # however pandas chooses to lay the frame out internally.
            for c in cols:
                if c in settled.columns:
                    v = pd.to_numeric(settled[c], errors="coerce").to_numpy(dtype="float64")
                    h.update(np.ascontiguousarray(v).tobytes())
        except Exception:        # noqa: BLE001 - never break a run over a digest
            h.update(b"||content-unavailable")
    return h.hexdigest()[:12]


def _fetch_catalogue_targets(
    index: pd.Index,
    have: pd.Index,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    force: bool = False,
) -> dict[str, pd.Series]:
    """Price columns for catalogue targets the macro universe does not carry.

    Aligned to the macro index and forward-filled, exactly as the exogenous
    (sheet) targets are — same spine, same treatment, so a target's column
    means the same thing regardless of which map it came from. A ticker that
    fails to fetch is simply omitted, leaving the existing "target column
    missing" guard to report it, rather than breaking the whole dataset.

    Iterates ``CATALOGUE_TARGETS`` — the catalogue frozen at import — never
    ``ALL_TARGETS``, which grows with every free-form symbol any session types.
    Iterating the live dict pulled one user's stock into every later user's
    panel, where it became a predictor of Gold and moved every published fair
    value (the panel fingerprint changed with it).

    Fetched over the caller's ``(start_date, end_date)`` rather than the
    frame's first and last row, so the key matches the app's own OHLCV fetch
    for the same target over the same window: one download, not two.
    """
    missing = [(n, t) for n, t in CATALOGUE_TARGETS.items()
               if n not in have and is_yfinance_ticker(t)]
    if not missing:
        return {}
    out: dict[str, pd.Series] = {}
    try:
        got = fetch_ohlcv([t for _, t in missing], start_date, end_date,
                          force=force) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("Catalogue target fetch skipped: %s", e)
        return {}
    for _name, _tkr in missing:
        o = got.get(_tkr)
        if o is None or getattr(o, "empty", True) or "Close" not in o.columns:
            continue
        ser = o["Close"]
        ser.index = pd.DatetimeIndex(ser.index).normalize()
        ser = ser[~ser.index.duplicated(keep="last")]
        out[_name] = ser.reindex(index).ffill()
    return out


def _fetch_exogenous_targets(index: pd.Index, force: bool = False) -> dict[str, pd.Series]:
    """Fetch non-yfinance target series and align them to the macro ``index``.

    Returns a ``{column_name: aligned_series}`` map. Resilient: any source that
    fails (live and cache both unavailable) is skipped.
    """
    out: dict[str, pd.Series] = {}
    from data.sheets import SHEET_SOURCES, fetch_sheet_series
    for _name in SHEET_SOURCES:
        try:
            s = fetch_sheet_series(_name, index.min(), index.max(), force=force)
            if s is not None and not s.empty:
                out[_name] = s.reindex(index).ffill()
        except Exception as e:  # noqa: BLE001
            log.warning("Exogenous %s fetch skipped: %s", _name, e)
    return out


def fetch_stock_target_series(
    ticker: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    force: bool = False,
) -> pd.Series | None:
    """Close series for a single free-form stock target.

    Individual-stock targets (config.is_stock_target) are deliberately
    NOT part of the shared model matrix: the matrix's composition is fixed by
    configuration (see the module docstring), and a stock column in it would be
    one more predictor for every other target. The caller joins this series to
    a private copy of the matrix for the run that needs it.

    Reuses fetch_ohlcv (two-tier cache + circuit breaker + stale fallback) — a
    later Swayam OHLCV fetch of the SAME ticker is then a cache hit, so this
    costs no extra network call overall.

    Returns ``None`` if nothing usable came back.
    """
    ohlcv_map = fetch_ohlcv([ticker], start_date, end_date, force=force) or {}
    return _close_series(ohlcv_map.get(ticker))
