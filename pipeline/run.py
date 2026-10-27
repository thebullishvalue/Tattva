"""
Tattva — the run, as step functions: fetch → prepare → Mūla → Swayam → the loop.
तत्त्व (Tattva) — "Principle / Essence"

app.py's phases call these same functions and wrap them only in progress
display, caching and reruns; :func:`run_target` chains them without
Streamlit, for the offline study and the tests. So the study runs the normal
pipeline by construction, and ``tests/test_pipeline_equivalence.py`` holds it
to that: on the offline fixture, ``run_target``'s call_ts equals the app's
byte for byte.

``fetch`` is the source of the four network fetches (``fetch_commodity_dataset``,
``fetch_ohlcv``, ``fetch_stock_target_series``, ``fetch_sheet_series``);
None is ``data.fetcher``, looked up when called, so a test's stub of that
module is what runs.

No Streamlit dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from core.config import (ALL_TARGETS, ANALYSIS_EPOCH, MIN_DATA_POINTS, STOCK_TARGET_MARKETS,
                         get_instrument_config, is_stock_target, swayam_macro_columns)
from core.instruments import SHEET_SERIES, instrument, predictor_universe
from core.knobs import knob
from core.trust import PriorTable, load_priors
from data.calendars import resolve_exchange, session_bars, session_today
from data.prep import PreparedPanel, prepare_panel
from engines.mula import MulaEngine
from engines.swayam import (aggregate_views, build_swayam_frames, default_swayam_members,
                            effective_member_count)
from pipeline.assemble import assemble, describe_priors

#: The Swayam page's drill-down reads only these per-view columns. The kernel
#: emits ~260 per view; at 12 views × ~2,300 rows that is ~60 MB per target,
#: held in the live session, the Swayam analysis cache and up to six result
#: bundles. Everything that needs the full width (aggregation, the effective
#: view count) has consumed it before the frames are kept.
SWAYAM_DRILLDOWN_COLS = ("Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition")


def _source(fetch):
    if fetch is not None:
        return fetch
    import data.fetcher as fetcher
    return fetcher


# ═══════════════════════════════════════════════════════════════════════
#  Fetch
# ═══════════════════════════════════════════════════════════════════════

def join_target(panel: pd.DataFrame, target: str, *, epoch, end, fetch=None) -> pd.DataFrame:
    """Join a target's price to a private copy of the panel when the batch has not.

    The macro batch fetches the registered universe, and the catalogue targets
    outside it are added upstream for every consumer. Two kinds of target are
    deliberately kept OUT of that shared frame and joined here, for the run
    that selects them only:

      * free-form STOCK symbols — a per-target ticker set would break the
        batch's (start, end) cache key, and a stock in the shared frame was
        one more predictor of every other target;
      * Google-Sheet series (Jeera, Nifty 50 - PE) — in the shared frame they
        were predictors of every target while the sheet answered and vanished
        when it did not, so a Sheets outage rewrote everyone's history.

    Fetched from ``epoch``, aligned to the panel's DATE spine and
    forward-filled; leading NaNs are left for the prep's target dropna. A
    missing catalogue column is a real fetch failure, left for the caller's
    guard to report — never papered over by a second fetch.
    """
    if target in panel.columns:
        return panel
    src = _source(fetch)
    if is_stock_target(target):
        ticker = ALL_TARGETS.get(target)
        s = src.fetch_stock_target_series(ticker, epoch, end) if ticker else None
    elif target in SHEET_SERIES:
        s = src.fetch_sheet_series(target, epoch, end)
    else:
        return panel
    if s is None or not len(s):
        return panel
    spine = pd.to_datetime(panel["DATE"], errors="coerce").dt.normalize()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    panel = panel.copy()
    panel[target] = s.reindex(spine).ffill().to_numpy()
    return panel


def fetch_panel(target: str, *, epoch=ANALYSIS_EPOCH, end=None, fetch=None) -> pd.DataFrame:
    """The shared macro panel from ``epoch`` to ``end``, with ``target``'s column joined."""
    end = pd.Timestamp.today() if end is None else pd.Timestamp(end)
    panel, error = _source(fetch).fetch_commodity_dataset(epoch, end)
    if error or panel is None:
        raise RuntimeError(f"panel fetch failed: {error}")
    return join_target(panel, target, epoch=epoch, end=end, fetch=fetch)


def target_bars(target: str, *, epoch=ANALYSIS_EPOCH, end=None, fetch=None,
                force: bool = False) -> tuple[pd.DataFrame | None, int]:
    """``(bars, dropped)``: the target's own OHLCV on its own weekday sessions.

    The pre-spine warm-up bars are kept (``data.calendars.session_bars``);
    ``dropped`` counts the non-session bars removed. None for a target with no
    bar feed (a sheet series) or when the fetch returned nothing.
    """
    inst = instrument(target)
    if not inst.has_ohlcv:
        return None, 0
    end = pd.Timestamp.today() if end is None else pd.Timestamp(end)
    bars = _source(fetch).fetch_ohlcv([inst.ticker], epoch, end, force=force).get(inst.ticker)
    if bars is None or bars.empty:
        return None, 0
    kept = session_bars(bars, inst.ticker)
    return kept, len(bars) - len(kept)


def swayam_drivers(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    """Swayam's MMR driver pool: the same panel Mūla values against, date-indexed.

    Every numeric column except other free-form stocks. Swayam's own leakage
    guard (``swayam_macro_columns``) drops the target and its near-replicas.
    """
    cols = [c for c in panel.select_dtypes(include=[np.number]).columns
            if c not in STOCK_TARGET_MARKETS or c == target]
    drivers = panel[["DATE"] + cols].copy()
    drivers.index = pd.to_datetime(drivers.pop("DATE"), errors="coerce").dt.normalize()
    drivers = drivers[~drivers.index.isna()]
    return drivers[~drivers.index.duplicated(keep="last")]


# ═══════════════════════════════════════════════════════════════════════
#  Prepare, Mūla, Swayam
# ═══════════════════════════════════════════════════════════════════════

def _studied_table(target: str) -> PriorTable | None:
    """The committed pooled-prior table for ``target``'s h and L; None when there
    is none or it was studied at another h or L (the run then refuses it)."""
    try:
        return load_priors(get_instrument_config(target).forecast_horizon, knob("EXECUTION_LAG"))
    except ValueError:
        return None


def prepare(panel: pd.DataFrame, target: str, *, epoch=ANALYSIS_EPOCH, bars=None,
            date_col: str = "DATE", priors: PriorTable | None | str = "default") -> PreparedPanel:
    """``data.prep.prepare_panel`` on the target's registered cross-section.

    ``priors`` is the pooled-prior table the run is compared with:
    ``'default'`` (the committed study file for the target's h and L, or
    none), None, or a ``PriorTable``. With one that studied ``target``, the
    history's fingerprint hashes the settled rows up to the session the study
    hashed it to (``PriorTable.fingerprint_upto``), so it equals the study's
    record until that history itself changes — hashed to the newest settled
    session instead, it would differ on every later day.
    """
    table = _studied_table(target) if isinstance(priors, str) and priors == "default" else priors
    numeric = panel.select_dtypes(include=[np.number]).columns.tolist()
    return prepare_panel(panel, target, predictor_universe(target, numeric), date_col,
                         epoch=epoch, min_target_rows=MIN_DATA_POINTS, bars=bars,
                         fingerprint_upto=table.fingerprint_upto(target) if table is not None else None)


def fit_mula(prep: PreparedPanel, cfg, progress: Callable[[float, str], None] | None = None) -> MulaEngine:
    """Fit Mūla on the prepared panel with the instrument's config (THE horizon h).

    ``progress(frac, message)`` is called as the valuation pass runs.
    """
    target = prep.report["target"]
    engine = MulaEngine()
    engine.fit(prep.data[target].astype(float), prep.data[prep.features].astype(float),
               feature_names=prep.features, config=cfg,
               exchange=resolve_exchange(instrument(target).ticker),
               epoch=prep.report["epoch"], fingerprint=prep.fingerprint,
               progress_callback=progress)
    return engine


def fvo_frame(engine: MulaEngine, prep: PreparedPanel) -> pd.DataFrame:
    """Mūla's published frame, indexed by session date (one engine row per spine date)."""
    fvo_ts = engine.ts_data.copy()
    fvo_ts.index = pd.DatetimeIndex(prep.data.index, name="Date")
    return fvo_ts


def _slim_view_dfs(dfs: dict) -> dict:
    """The per-view frames trimmed to what the drill-down displays."""
    out = {}
    for name, df in (dfs or {}).items():
        cols = [c for c in SWAYAM_DRILLDOWN_COLS if c in df.columns]
        out[name] = df[cols] if cols else df.iloc[:, :0]
    return out


def build_swayam(target: str, cfg, ohlcv: pd.DataFrame | None, prep: PreparedPanel, *, asof,
                 drivers: pd.DataFrame,
                 progress: Callable[[int, int, str], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """``(swayam_daily, meta)``: Swayam's breadth on the model spine, and its by-products.

    The view bank runs on the target's own bars (``ohlcv``) with the
    leakage-guarded ``drivers``; the views are aggregated with equal weights
    (``aggregate_views``: ``Oversold_Pct, Overbought_Pct, Total_Analyzed,
    N_Voting`` and the one published ``Lean``), then carried onto the
    prepared spine with ``_Native`` marking the sessions that are a genuine
    view-bank observation (a carried row is not a new reading). ``meta``:
    ``view_dfs`` (the drill-down columns), ``n_eff``, ``native_last`` (dated
    ≤ ``asof``, d*), ``n_members``, ``n_drivers`` and ``n_drivers_kept``.
    ``progress(done, total, view)`` is called per view.
    """
    meta = {"view_dfs": {}, "n_eff": None, "native_last": None,
            "n_members": 0, "n_drivers": len(drivers.columns), "n_drivers_kept": 0}
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(), meta
    cols = swayam_macro_columns(target, list(drivers.columns))
    members = default_swayam_members(cfg.swayam_lengths, cfg.swayam_roc_frac)
    meta.update(n_members=len(members), n_drivers_kept=len(cols))
    frames = build_swayam_frames(
        ohlcv, drivers, cols, members=members,
        regime_sensitivity=cfg.swayam_regime_sensitivity, base_weight=cfg.swayam_base_weight,
        num_vars=cfg.swayam_mmr_num_vars, oversold=cfg.swayam_oversold,
        overbought=cfg.swayam_overbought, progress_cb=progress,
    )
    if not frames:
        return pd.DataFrame(), meta
    meta["n_eff"] = effective_member_count(frames)
    daily = aggregate_views(frames)
    meta["view_dfs"] = _slim_view_dfs(frames)
    if daily.empty:
        return daily, meta

    # Carry the breadth onto the model spine. The bars can end a session
    # before the panel does (yfinance publishes the OHLCV and close feeds on
    # different schedules), so the spine's newest rows may be carried; the
    # true last native date at or before d* is recorded.
    nd = daily.copy()
    nd.index = pd.to_datetime(nd.index).normalize()
    nd = nd[~nd.index.duplicated(keep="last")].sort_index()
    native = nd.index if asof is None else nd.index[nd.index <= pd.Timestamp(asof)]
    meta["native_last"] = pd.Timestamp(native.max()) if len(native) else None
    swayam_daily = nd.reindex(prep.data.index.rename(None), method="ffill").dropna(how="all")
    swayam_daily["_Native"] = swayam_daily.index.isin(set(nd.index))
    return swayam_daily, meta


# ═══════════════════════════════════════════════════════════════════════
#  One target, end to end
# ═══════════════════════════════════════════════════════════════════════

def run_target(target: str, *, end=None, priors: PriorTable | None | str = "default",
               fetch=None) -> dict:
    """The whole run for ``target``, without Streamlit: the bundle app.py stores.

    ``end`` is the last date fetched (None: today, and the call's recency is
    measured to the target's current session; a date: a replay as of it).
    ``priors`` is ``'default'`` (the committed study file for this h and L,
    or none), None (per-instrument grades only) or a ``PriorTable``; the
    history's fingerprint is hashed to match the same table (``prepare``).

    Returns ``engine, fvo_ts, swayam_daily, swayam_view_dfs, swayam_n_eff,
    swayam_native_last, call_ts, grades_ts, edges, evidence, verdict, asof,
    horizon, exec_lag, epoch, fingerprint, priors_meta``.
    """
    cfg, inst = get_instrument_config(target), instrument(target)
    h, lag = int(cfg.forecast_horizon), int(knob("EXECUTION_LAG"))
    table = load_priors(h, lag) if isinstance(priors, str) and priors == "default" else priors
    panel = fetch_panel(target, epoch=ANALYSIS_EPOCH, end=end, fetch=fetch)
    bars, _dropped = target_bars(target, epoch=ANALYSIS_EPOCH, end=end, fetch=fetch)
    prep = prepare(panel, target, epoch=ANALYSIS_EPOCH, bars=bars, priors=table)
    engine = fit_mula(prep, cfg)
    fvo_ts = fvo_frame(engine, prep)
    swayam_daily, sw = build_swayam(target, cfg, bars, prep, asof=engine.asof,
                                    drivers=swayam_drivers(panel, target))
    today = (session_today(resolve_exchange(inst.ticker)) if end is None
             else pd.Timestamp(end).normalize())
    asm = assemble(fvo_ts, swayam_daily, instrument=inst, cfg=cfg, asof=engine.asof, today=today,
                   priors=table, epoch=engine.epoch, fingerprint=engine.fingerprint)
    return {
        "engine": engine, "fvo_ts": fvo_ts, "swayam_daily": swayam_daily,
        "swayam_view_dfs": sw["view_dfs"], "swayam_n_eff": sw["n_eff"],
        "swayam_native_last": sw["native_last"],
        "call_ts": asm.call_ts, "grades_ts": asm.grades_ts, "edges": asm.edges,
        "evidence": asm.evidence, "verdict": asm.verdict, "asof": engine.asof, "horizon": h,
        "exec_lag": lag, "epoch": engine.epoch, "fingerprint": engine.fingerprint,
        "priors_meta": describe_priors(table),
    }
