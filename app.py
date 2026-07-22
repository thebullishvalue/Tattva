"""
Tattva — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — Two systems. One conclusion. A top-down macro forecast and a bottom-up
basket regime read — across commodities, FX, and equity indices — unified by
adaptive convergence.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os

# ── BLAS thread pinning (MUST run before numpy/sklearn import) ────────────────
# The walk-forward fits hundreds of small models sequentially. On Streamlit
# Community Cloud the container is throttled to ~1 shared vCPU but the host
# reports many logical CPUs, so OpenBLAS/MKL spawn one thread per reported core
# and thrash — turning each tiny PCA/Ridge solve into a thread-contention storm
# (the #1 reason the walk-forward is far slower on cloud than locally). One
# thread per process is strictly faster for many-small-matrix workloads here.
# os.environ.setdefault → respects any explicit override from the environment.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ── Numba cache OUTSIDE the app tree (MUST run before numba is imported) ──────
# @njit(cache=True) kernels write .nbc/.nbi artifacts. If those land in the app
# directory (default: <module>/__pycache__), Streamlit's file watcher treats each
# write as a source change and reruns the script — restarting the whole pipeline
# mid-compile. Point Numba's cache at the home cache dir (writable, NOT watched).
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tattva", "numba"),
)

import json
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Warning suppression ──────────────────────────────────────────────────────
# A blanket `category=RuntimeWarning` filter used to sit here — it silenced
# every RuntimeWarning process-wide, including any GENUINE numeric issue
# (overflow, invalid divide, degenerate log/sqrt) anywhere in the math stack,
# not just the known-noisy sources it was meant to cover (audit finding C6).
# The one legitimate source found by auditing (nanmean's "Mean of empty
# slice" on the engine's own warm-up rows) is now scoped locally at its call
# site (engines/aarambh.py's _compute_breadth_metrics) instead. FutureWarning
# stays broadly suppressed — it's pandas/numpy API-deprecation noise, not a
# correctness signal, so it doesn't carry the same risk of masking a real bug.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*YF.download.*")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
pd.options.mode.chained_assignment = None

# ── Path setup ───────────────────────────────────────────────────────────────
# Force PROJECT_ROOT to the FRONT of sys.path (ahead of site-packages) so the
# project's own packages (analytics, core, data, …) always win over any
# same-named package that happens to be installed in the environment. The
# project dirs carry __init__.py so they resolve as regular packages.
PROJECT_ROOT = Path(__file__).resolve().parent
_pr = str(PROJECT_ROOT)
if _pr in sys.path:
    sys.path.remove(_pr)
sys.path.insert(0, _pr)

# ── UI ───────────────────────────────────────────────────────────────────────
from ui.theme import inject_css, VERSION, PRODUCT_NAME, COMPANY, progress_bar
from ui.tabs.tab_convergence import render_convergence_tab
from ui.components import (
    render_header,
    render_info_box,
    build_hero_verdict,
    render_hero_card,
    render_warning_box,
    render_control_hint,
    section_gap,
)
from ui.tabs.tab_aarambh import render_aarambh_tab
from ui.tabs.tab_nirnay import render_nirnay_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_constituent_ohlcv, fetch_macro_live, fetch_commodity_dataset, fetch_stock_target_series
from data.constituents import get_commodity_basket, get_nirnay_mode
from data.calendars import trading_days_behind, is_session, session_mask, resolve_exchange
from data.universe import resolve_stock_symbol

# ── Engines ──────────────────────────────────────────────────────────────────
from engines.aarambh import FairValueEngine
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries, apply_polarity

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.conviction_model import UnifiedConvictionModel
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import LOOKBACK_WINDOWS, MIN_DATA_POINTS, STALENESS_DAYS, SESSION_FRESH_FLOOR, TARGET_EXCLUDED_PREDICTORS, TARGET_POLARITY, ALL_TARGETS, TARGET_CATEGORIES, TARGET_ARCHETYPE, FORECAST_HORIZON, UI_AGREEMENT_STRONG, UI_AGREEMENT_MODERATE, INTEL_N_TRIALS, RAW_YIELD_PREDICTORS, DIV_LOOKBACK, TIMEFRAME_TRADING_DAYS, swayam_macro_columns, FREEFORM_STOCK_CATEGORIES, register_stock_target, get_instrument_config
from engines.nirnay_self import build_swayam_frames, effective_member_count, default_swayam_members
from core.config import GLOBAL_MACRO_MAP, MACRO_SYMBOLS_YF, INDEX_TARGETS_MAP

# Friendly column name → ticker, for resolving each predictor/target column to its
# exchange (holiday-aware data freshness + target-session spine filtering). Targets
# (incl. sheet/NCDEX sentinels) are merged last so they win any name collision.
_COLUMN_TICKERS = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **INDEX_TARGETS_MAP, **ALL_TARGETS}


# ─── Per-config result cache ─────────────────────────────────────────────────
# The full result of an analysis is the set of session-state keys below. We
# snapshot them per cache_key so revisiting a previously-computed config (e.g.
# the user switches Gold → Silver → Gold) restores instantly instead of
# recomputing the whole 5-phase pipeline. Bounded (LRU) to cap memory.
_BUNDLE_KEYS = (
    "engine", "aarambh_ts", "nirnay_daily", "nirnay_constituent_dfs",
    "convergence_df", "divergence_events", "nishkarsh_result", "last_agreement",
    "nishkarsh_conv_normalized", "wf_results",
    "intelligence_active_weights", "intelligence_active_thresholds",
    "intelligence_active_profile",
    # The consensus headline's full history + DDM-smoothed trend
    # (hero-history plot / TREND row; the headline scalar itself lives in
    # nishkarsh_conv_normalized above), and the calibrated variant
    # (CALIBRATED evidence row / amber overlay) — must all travel with the
    # bundle so a cached target switch-back doesn't leave the PREVIOUS
    # target's headline state in session state.
    "hero_series", "hero_smoothed",
    "nishkarsh_calibrated_score", "nishkarsh_calibrated_signal",
    "calibrated_conv_series",
    # Per-target UI metadata that must travel with the result bundle —
    # otherwise a cached target switch-back leaves the PREVIOUS target's
    # value in session state (e.g. the Nirnay tab showing a stale
    # "basket source: snapshot" hint for a target resolved live, or the
    # Convergence tab's "breadth carried forward" notice firing/missing
    # based on the WRONG target's basket-freshness timestamp).
    "nirnay_basket_source", "nirnay_native_last", "nirnay_mode", "nirnay_swayam_n_eff",
)
# Keep the last N configs. The comment here previously said "the 3
# commodities" — stale since the universe grew to 30+ targets (commodities,
# FX, India/US indices, sector ETFs; audit finding E5). Each entry is a full
# 5-phase pipeline result, so this stays modest rather than trying to cover
# the whole universe; 6 covers a session that browses a handful of targets
# (e.g. all commodities, or an index + its close comparators) without
# recomputing.
_RESULTS_CACHE_MAX = 6

# Baskets at/above this size get their per-constituent frames trimmed before
# entering the _RESULTS_CACHE_MAX-deep results_cache LRU (audit finding F19).
# nirnay_constituent_dfs carries ~200 columns per constituent (the full
# run_full_analysis output); only the ~9 the Nirnay tab's drill-down actually
# displays (_NIRNAY_DRILLDOWN_COLS) are needed once the result is just sitting
# in the switch-back cache. A small commodity basket (~15-20 names) is cheap
# either way and kept at full width so nothing else that might read the wider
# frame in-session breaks; an uncapped large index (S&P 500 ~500 names) is
# where the ~200-column full width, multiplied across up to 6 LRU entries,
# actually matters.
_CONSTITUENT_TRIM_THRESHOLD = 60
_NIRNAY_DRILLDOWN_COLS = (
    "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
    "Regime", "Vol_Regime", "Change_Point", "Confidence",
)


def _bundle_nirnay_constituent_dfs(dfs: dict) -> dict:
    """Trim nirnay_constituent_dfs to the Nirnay tab's drill-down columns
    before it enters the per-config results_cache LRU, for baskets at/above
    _CONSTITUENT_TRIM_THRESHOLD names. Only affects the SNAPSHOT stored in
    results_cache — the live session_state copy the active render reads
    (and engines.nirnay.aggregate_constituent_timeseries, which needs the
    full width and runs before this snapshot is taken) is never touched.
    """
    if not dfs or len(dfs) < _CONSTITUENT_TRIM_THRESHOLD:
        return dfs
    trimmed = {}
    for sym, df in dfs.items():
        cols = [c for c in _NIRNAY_DRILLDOWN_COLS if c in df.columns]
        trimmed[sym] = df[cols] if cols else df.iloc[:, :0]
    return trimmed


def _ensure_stock_target_column(df: pd.DataFrame, active_target: str) -> pd.DataFrame:
    """Inject a self-archetype target's Close into the model matrix.

    Individual-stock targets (TARGET_ARCHETYPE == 'self') are deliberately
    NOT part of the macro batch universe fetch_commodity_dataset pulls (cache
    coherence — see fetch_stock_target_series's docstring); their price
    column is injected per-target here, the same pattern
    data.fetcher._fetch_exogenous_targets uses for sheet targets: aligned to
    the matrix's DATE spine, ffilled, leading NaNs left for the per-target
    dropna downstream. No-op when the column already exists or the target
    isn't a stock. Mutates st.session_state['data'] too, so a target switch
    or a cached rerun sees the column without re-fetching.
    """
    if active_target in df.columns or TARGET_ARCHETYPE.get(active_target) != "self":
        return df
    ticker = ALL_TARGETS.get(active_target)
    if not ticker:
        return df
    end = pd.Timestamp.today()
    s = fetch_stock_target_series(ticker, end - pd.Timedelta(days=365 * 9), end)
    if s is None:
        return df                      # the guard right after this call fires cleanly
    spine = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    df = df.copy()
    df[active_target] = s.reindex(spine).ffill().to_numpy()
    st.session_state["data"] = df
    return df


# ─── UI Rendering helpers ────────────────────────────────────────────────────

def _render_header() -> None:
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Cross-Asset Fair-Value + Basket Regime Intelligence  |  Unified Convergence"
    )


def _render_landing_page() -> None:
    """Render the landing page with three system cards."""
    section_gap()
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown("""
        <div class='system-card aarambh'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                AARAMBH
            </h3>
            <p>Walk-forward ensemble regression on the selected target (commodities, FX, indices & ETFs) vs the macro/FX universe, with robust quantile z-scores and DDM filtering.</p>
            <div class='spec'>
                <span>Ensemble:</span> PCA-OLS + Huber<br>
                <span>Validation:</span> Walk-forward OOS<br>
                <span>Bounds:</span> Rolling robust quantiles
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='system-card nirnay'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                NIRNAY
            </h3>
            <p>Per-instrument MSF + MMR analysis across a basket of related ETFs & miners, with HMM/GARCH/CUSUM regime intelligence aggregation.</p>
            <div class='spec'>
                <span>Signal:</span> MSF + MMR oscillator<br>
                <span>Breadth:</span> Oversold / Overbought %<br>
                <span>Regime:</span> HMM · GARCH · CUSUM
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='system-card convergence'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
                CONVERGENCE
            </h3>
            <p>Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.</p>
            <div class='spec'>
                <span>Fusion:</span> Aarambh + Nirnay<br>
                <span>Smoothing:</span> Leaky DDM<br>
                <span>Range:</span> Soft \u00b1100 limit
            </div>
        </div>
        """, unsafe_allow_html=True)
    section_gap()
    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING DATA
        </h4>
        <p>Pick an <strong>Asset Class → Target</strong> (Commodities · FX · India &amp; US Indices · Sector ETFs) in the <strong>Sidebar</strong>,<br>
           then execute <strong>Run Analysis</strong> to fetch the live yfinance data and initialize both engines.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_primary_signal(nishkarsh_norm, agreement, aarambh_signal) -> None:
    """Render the hero Tattva convergence signal card.

    All interpretation logic lives in ``ui.components.build_hero_verdict`` (a
    pure, unit-testable function); this wrapper only gathers session-state
    inputs and hands the verdict to ``render_hero_card``. The headline chain
    is NORMALIZED CONSENSUS -> Aarambh-only, always paired with an honest
    trust read (non-overlapping Val IC + walk-forward durability, attributed
    to the calibrated composite variant), the CALIBRATED variant as a
    second-opinion evidence row, and a minimum-n-gated precedent read.

    The headline is ``nishkarsh_norm`` (passed in) — the normalized
    consensus, the SAME object as the Unified Signal plot's top row and the
    TATTVA CONVICTION card, so hero/card/plot reconcile identically by
    construction. The calibrated composite (Phase 4e) feeds the CALIBRATED
    evidence row; ``hero_smoothed`` (DDM of the consensus) feeds TREND.
    """
    profile     = st.session_state.get("intelligence_active_profile")  # dict | None
    wf          = st.session_state.get("wf_results")                   # list[dict] | None
    div_events  = st.session_state.get("divergence_events")            # DataFrame | None
    prec        = st.session_state.get("precedent_summary")            # dict | None
    hero_smoothed = st.session_state.get("hero_smoothed")               # pd.Series | None
    calib_score  = st.session_state.get("nishkarsh_calibrated_score")   # float | None
    calib_signal = st.session_state.get("nishkarsh_calibrated_signal")  # str | None

    # DEGENERATE-CONVERGENCE GATE: `nishkarsh_norm` is None exactly when the
    # Aarambh∩Nirnay alignment found no overlap (empty/unresolvable basket) —
    # the headline chain then falls through to the honest "Aarambh only (no
    # basket convergence)" source automatically. The calibrated composite is
    # gated too: with no basket it was computed against neutral PLACEHOLDER
    # nirnay stats (a half-weight Aarambh-only signal wearing a convergence
    # label), and divergence events are silenced for the same reason — the
    # detector compared Aarambh against a basket that doesn't exist.
    _has_genuine_convergence = nishkarsh_norm is not None
    if not _has_genuine_convergence:
        hero_smoothed = None
        calib_score, calib_signal = None, None

    # Active instrument's forecast horizon - for interpretation copy only.
    try:
        FWD_HORIZON = get_instrument_config(st.session_state.get("active_target", "")).forecast_horizon
    except KeyError:
        FWD_HORIZON = FORECAST_HORIZON

    val_ic = None
    if profile and profile.get("val_ic") is not None:
        try: val_ic = float(profile["val_ic"])
        except (TypeError, ValueError): val_ic = None
    wf_ics = [r["ic"] for r in wf if isinstance(r, dict) and r.get("ic") == r.get("ic")] if wf else []
    wf_pos = (sum(1 for v in wf_ics if v > 0) / len(wf_ics)) if wf_ics else None
    wf_n = len(wf_ics) if wf_ics else None
    # RECENT divergence count only (audit finding F7) — div_events spans the
    # WHOLE history (6+ years), so a bare len() reads in the hundreds and is a
    # permanent, meaningless alarm ("N divergence events flagged"). Count only
    # events within the last DIV_LOOKBACK trading days of the series (the same
    # window CrossSystemDivergenceDetector uses for its own persistence flag),
    # anchored on the LATEST event date in the table (a proxy for "today" —
    # div_events carries no direct handle on the engine's current as-of date).
    n_div = 0
    if (_has_genuine_convergence and div_events is not None
            and hasattr(div_events, "__len__") and len(div_events)):
        try:
            _div_dates = pd.to_datetime(div_events.index, errors="coerce")
            _valid_dates = _div_dates.dropna()
            if len(_valid_dates):
                _cutoff = _valid_dates.max() - pd.Timedelta(days=int(DIV_LOOKBACK * 1.5))
                n_div = int((_div_dates >= _cutoff).sum())
            else:
                n_div = int(len(div_events))
        except Exception:
            n_div = int(len(div_events))

    verdict = build_hero_verdict(
        calib_conviction=(float(calib_score) if calib_score is not None else None),
        calib_signal=(calib_signal if calib_signal is not None else None),
        has_profile=bool(profile),
        consensus=nishkarsh_norm,
        aarambh_signal=aarambh_signal,
        agreement=float(agreement or 0.0),
        val_ic=val_ic,
        wf_pos=wf_pos,
        precedent=prec,
        n_divergences=n_div,
        horizon_days=FWD_HORIZON,
        agreement_strong=UI_AGREEMENT_STRONG,
        agreement_moderate=UI_AGREEMENT_MODERATE,
        # DDM-smoothed value of the SAME consensus series ([-1,+1]) — lets
        # the card interpret today's print against its own trend (TREND row)
        # instead of leaving that to the hero-history plot.
        smoothed=(float(hero_smoothed.iloc[-1])
                  if hero_smoothed is not None and len(hero_smoothed) else None),
        wf_n=wf_n,
        div_window=DIV_LOOKBACK,
    )
    render_hero_card(verdict)
    section_gap()


def _render_model_passport_sidebar(current_universe: str, current_index: str | None = None) -> None:
    """Sidebar Passport — visible in every mode.

    Faithful port of Sanket's `_render_model_passport_sidebar`, adapted to
    Nishkarsh's (universe, index) keying. Surfaces:
      • Profile state (Default / Calibrated / Calibrated · ⚠ on mismatch)
      • Trained-on label · Train IC · Val IC · Updated timestamp
      • Universe-mismatch warning when the saved profile was fit on a
        different universe than the active sidebar selection
      • Import / Export / Reset controls

    Caller must be inside a ``with st.sidebar:`` context.
    """
    from convergence import intelligence as intel

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Model Passport</div>', unsafe_allow_html=True)

    # Intelligence-mode toggle. Default ON. When OFF, any saved profile is
    # ignored and CrossValidator falls back to its ±10% adaptive-shift
    # heuristic on top of the factory 0.30/0.25/0.25/0.20 base allocation —
    # NOT the bare fixed weights (audit finding F20: the help text previously
    # implied the base weights are applied verbatim when OFF; the ±10%
    # per-day clarity-based shift always runs on top of them in that path —
    # see convergence.cross_validator.CrossValidator.compute_convergence).
    # Thresholds ARE the bare factory defaults when OFF — the composite's own
    # p75/p90-anchored COMPOSITE_THRESHOLDS (no equivalent heuristic exists
    # for thresholds).
    intelligence_mode = st.toggle(
        "Intelligence Mode",
        value=bool(st.session_state.get("intelligence_mode", True)),
        help=(
            "When ON, Tattva uses the persisted calibrated profile for the "
            "selected universe (if one exists). When OFF, Tattva runs on the "
            "factory 0.30 / 0.25 / 0.25 / 0.20 dimension weights (adaptively "
            "shifted ±10% per day by signal clarity — not applied verbatim) "
            "and the composite's data-anchored factory thresholds "
            "(±0.11 moderate / ±0.18 strong)."
        ),
        key="passport_intel_toggle",
    )
    st.session_state["intelligence_mode"] = intelligence_mode

    # What profile (if any) is saved for THIS universe?
    saved_profile = intel.load_profile_for(current_universe, current_index)

    # Status card values
    if intelligence_mode and saved_profile is not None:
        cal_universe = saved_profile.universe
        cal_index    = saved_profile.selected_index
        cal_label    = cal_index or cal_universe or "—"
        cur_label    = current_index or current_universe or "—"
        universe_mismatch = cal_label != "—" and cur_label != "—" and cal_label != cur_label
        train_v = float(saved_profile.train_ic or 0.0)
        val_v   = float(saved_profile.val_ic or 0.0)
        train_str = f"{train_v:+.3f}"
        val_str   = f"{val_v:+.3f}"
        updated   = saved_profile.timestamp or "—"
        train_color = "var(--emerald)" if train_v > 0 else "var(--rose)"
        val_color   = "var(--emerald)" if val_v   > 0 else "var(--rose)"
        if universe_mismatch:
            profile_label = "Calibrated · ⚠"
            card_class = "warning"
        else:
            profile_label = "Calibrated"
            card_class = "success" if (val_v > 0 and train_v > 0) else "warning"
    elif not intelligence_mode:
        cal_label = "—"
        profile_label = "Default · Off"
        train_str = val_str = updated = "—"
        train_color = val_color = "var(--ink-secondary)"
        card_class = "neutral"
        universe_mismatch = False
    else:
        cal_label = "—"
        profile_label = "Default"
        train_str = val_str = updated = "—"
        train_color = val_color = "var(--ink-secondary)"
        card_class = "neutral"
        universe_mismatch = False

    def _trim(s: str, n: int = 22) -> str:
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    cal_label_disp = _trim(cal_label)

    st.markdown(f"""
    <div class="metric-card {card_class}" style="
            min-height:auto;
            padding:0.85rem 0.95rem;
            margin-bottom:0.7rem;
            animation:none;">
        <h4 style="margin:0 0 0.3rem 0;">Profile</h4>
        <h2 style="font-size:1.05rem; margin:0 0 0.7rem 0; letter-spacing:-0.01em;">{profile_label}</h2>
        <div style="display:flex; flex-direction:column; gap:0.32rem;
                    padding-top:0.55rem;
                    border-top:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.62rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Trained on</span>
                <span style="color:var(--ink-secondary); font-weight:500; max-width:62%; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{cal_label_disp}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Train IC</span>
                <span style="color:{train_color}; font-weight:600;">{train_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.65rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Val IC</span>
                <span style="color:{val_color}; font-weight:600;">{val_str}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; font-family:var(--data); font-size:0.6rem;">
                <span style="color:var(--ink-tertiary); text-transform:uppercase; letter-spacing:0.1em; font-size:0.58rem;">Updated</span>
                <span style="color:var(--ink-secondary);">{updated}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if universe_mismatch:
        st.markdown(f"""
        <div style="font-family:var(--data); font-size:0.62rem; color:var(--amber);
                    background:rgba(212,168,83,0.08);
                    border:1px solid rgba(212,168,83,0.22);
                    border-radius:6px; padding:0.55rem 0.65rem;
                    margin-bottom:0.7rem; line-height:1.45;">
            <span style="font-weight:700;">Profile mismatch — calibrated weights still active.</span><br>
            Profile fit on <b>{_trim(cal_label, 28)}</b><br>
            Active universe is <b>{_trim(current_index or current_universe, 28)}</b><br>
            <span style="color:var(--ink-tertiary);">Weights learned for one universe do not generalise.
            Reset to defaults or run a new calibration for the current selection.</span>
        </div>
        """, unsafe_allow_html=True)

    # Import / Export / Reset controls
    with st.expander("↑ Import Profile", expanded=False):
        uploaded = st.file_uploader(
            " ", type=["json"], label_visibility="collapsed", key="passport_uploader",
        )
        if uploaded is not None:
            try:
                payload = json.load(uploaded)
                if isinstance(payload, dict) and "weights" in payload:
                    imported = intel.IntelligenceProfile.from_dict(payload)
                    intel.save_profile(imported)
                    st.toast("Profile imported.", icon="✅")
                    st.success(f"Profile imported · {imported.universe}")
                    st.rerun()
                else:
                    st.error("Import failed: file is not a valid profile dict (missing 'weights').")
            except Exception as e:
                st.error(f"Import failed: {e}")

    if saved_profile is not None:
        export_payload = saved_profile.to_dict()
        ts_slug = (saved_profile.timestamp or "").split(" ")[0] or "snapshot"
        # Sanitize for a safe filename: spaces → "_", and "/" (e.g. "USD/INR")
        # → "-" so it doesn't read as a path separator in the download.
        _slug = (saved_profile.selected_index or saved_profile.universe or "profile")
        _slug = _slug.replace(" ", "_").replace("/", "-")
        fname = f"tattva_profile_{_trim(_slug, 30)}_{ts_slug}.json"
        st.download_button(
            "↓ Export Profile",
            data=json.dumps(export_payload, indent=2, default=str),
            file_name=fname,
            mime="application/json",
            use_container_width=True,
            key="passport_export",
        )
        if st.button("↺ Reset to Defaults", use_container_width=True, key="passport_reset"):
            intel.delete_profile(saved_profile.universe, saved_profile.selected_index)
            # Streamlit's toast `icon=` only accepts emojis from a curated
            # whitelist; "↺" (U+21BA, our "Reset" mark used on the button) is
            # rejected. Drop the icon — the "↺" stays on the button text where
            # the user actually sees it.
            st.toast("Profile reset.")
            st.rerun()


def _render_footer() -> None:
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="app-footer">'
        f'<div class="content">'
        f'\u00a9 {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;\u00b7&nbsp; {COMPANY} &nbsp;\u00b7&nbsp; v{VERSION} &nbsp;\u00b7&nbsp; {ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="TATTVA | Unified Convergence",
        page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
        layout="wide", initial_sidebar_state="collapsed",
    )
    inject_css()

    # Replay dynamic stock-target registration on every rerun. register_stock_target
    # mutates module-level core.config dicts (ALL_TARGETS etc.) which survive
    # Streamlit reruns WITHIN a process but are never persisted — only
    # st.session_state survives a rerun as the durable record, so a freeform
    # symbol resolved earlier this session (e.g. "RELIANCE (NSE)") must be
    # re-registered before anything below resolves active_target against
    # ALL_TARGETS/TARGET_ARCHETYPE. Idempotent — safe every rerun.
    for _dname, _dmeta in st.session_state.get("dynamic_stock_targets", {}).items():
        register_stock_target(_dname, _dmeta["ticker"], _dmeta["market"])

    # Single main-area progress slot, created up front (outside the sidebar) so the
    # SAME themed progress bar drives everything from the moment "Run Analysis" is
    # clicked — the fetch, the data-prep spine, the engines, convergence — instead of
    # a sidebar spinner that then hands off to a separate bar with a gap between them.
    # Empty (invisible) until the first progress_bar() call; cleared when a run ends.
    progress_container = st.empty()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:0.5rem 0 0.75rem 0;">
            <div style="font-family:var(--display);font-size:1.35rem;font-weight:700;color:var(--amber);letter-spacing:0.04em;">TATTVA</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.6rem;margin-top:0.1rem;letter-spacing:0.06em;text-transform:uppercase;">तत्त्व | Unified Convergence</div>
        </div>
        <hr style="margin: 0.5rem 0; opacity: 0.1;">
        """,
            unsafe_allow_html=True,
        )

        # Two-level selection: Asset Class → Target. Keeps the growing target
        # roster (commodities, FX, India & US indices, sector-ETF universe)
        # logically grouped instead of one long flat list.
        all_names = list(ALL_TARGETS.keys())
        prev_commodity = st.session_state.get("selected_commodity", all_names[0])
        if prev_commodity not in all_names:
            prev_commodity = all_names[0]

        _categories = list(TARGET_CATEGORIES.keys())
        # Freeform categories (India/US Stocks) stay EMPTY in TARGET_CATEGORIES
        # (a dynamic name renders a text input, not a list entry — see
        # core.config.register_stock_target), so plain membership can never
        # find a previously-resolved stock target there. Check
        # dynamic_stock_targets first so re-selecting a stock category across
        # reruns doesn't silently snap back to the first category.
        _dyn_meta = st.session_state.get("dynamic_stock_targets", {}).get(prev_commodity)
        if _dyn_meta is not None:
            prev_cat = next(
                (cat for cat, mkt in FREEFORM_STOCK_CATEGORIES.items() if mkt == _dyn_meta["market"]),
                _categories[0],
            )
        else:
            prev_cat = next(
                (c for c, names in TARGET_CATEGORIES.items() if prev_commodity in names),
                _categories[0],
            )
        # Seed widget state BEFORE instantiation so the (options-changing) target
        # selectbox never holds a value outside its current category — the classic
        # Streamlit "key + dynamic options" pitfall. We drive both via session_state
        # keys, not `index=`.
        st.session_state.setdefault("target_category", prev_cat)
        if st.session_state["target_category"] not in _categories:
            st.session_state["target_category"] = prev_cat

        st.markdown('<div class="sidebar-title">Asset Class</div>', unsafe_allow_html=True)
        sel_cat = st.selectbox(
            "Asset Class", _categories,
            label_visibility="collapsed", key="target_category",
            help="Choose an asset class, then a target within it.",
        )

        if sel_cat in FREEFORM_STOCK_CATEGORIES:
            # India Stocks / US Stocks: no constituent basket to browse — enter
            # a symbol directly. The asset class supplies the suffix policy
            # (data.universe.resolve_stock_symbol): India tries SYMBOL.NS
            # first, then SYMBOL.BO; US uses the bare symbol.
            _market = FREEFORM_STOCK_CATEGORIES[sel_cat]
            st.markdown('<div class="sidebar-title" style="margin-top:0.5rem;">Symbol</div>', unsafe_allow_html=True)
            _raw_symbol = st.text_input(
                "Symbol", key=f"stock_symbol_{_market}",
                label_visibility="collapsed",
                placeholder="e.g. RELIANCE, TATASTEEL" if _market == "india" else "e.g. AAPL, BRK.B",
                help="Nirnay runs in Swayam self-mode on this instrument's own OHLCV "
                     "(no constituent basket exists for a single stock).",
            )
            selected_commodity = None
            if _raw_symbol and _raw_symbol.strip():
                with st.spinner("Resolving symbol…"):
                    _ticker, _exch_or_err = resolve_stock_symbol(_raw_symbol, _market)
                if _ticker is None:
                    st.error(_exch_or_err)
                else:
                    _base = _ticker.rsplit(".", 1)[0] if _market == "india" else _raw_symbol.strip().upper()
                    selected_commodity = f"{_base} ({_exch_or_err})"
                    register_stock_target(selected_commodity, _ticker, _market)
                    _dyn = st.session_state.setdefault("dynamic_stock_targets", {})
                    _dyn[selected_commodity] = {"ticker": _ticker, "market": _market}
                    render_control_hint(f"{_raw_symbol.strip().upper()} → {_ticker} · {_exch_or_err}")
            else:
                render_control_hint(
                    "NSE (.NS) checked first, then BSE (.BO)" if _market == "india"
                    else "US listing · symbol as typed"
                )
        else:
            cat_targets = TARGET_CATEGORIES.get(sel_cat, all_names)

            # Keep the target selection valid for the chosen category.
            if st.session_state.get("target_select") not in cat_targets:
                st.session_state["target_select"] = (
                    prev_commodity if prev_commodity in cat_targets else cat_targets[0]
                )
            st.markdown('<div class="sidebar-title" style="margin-top:0.5rem;">Target</div>', unsafe_allow_html=True)
            selected_commodity = st.selectbox(
                "Target", cat_targets,
                label_visibility="collapsed", key="target_select",
                help="Aarambh forecasts this target's forward return; Nirnay reads "
                     "bottom-up breadth — across its constituent basket (index members, "
                     "producers, sector ETFs), or as a Swayam self-ensemble on the "
                     "instrument's own price (commodities & stocks).",
            )
        # Show the Nirnay mode as a subtle hint. For a FREEFORM stock the
        # resolution hint just above already states the ticker/exchange and the
        # Symbol help text explains Swayam, so a 'self' line here would repeat —
        # suppress it there only. For a dropdown 'self' target (the commodity
        # futures) show a Swayam label; for basket targets show the archetype.
        _arch = TARGET_ARCHETYPE.get(selected_commodity, "") if selected_commodity else ""
        _is_freeform = sel_cat in FREEFORM_STOCK_CATEGORIES
        if _arch and not (_arch == "self" and _is_freeform):
            if _arch == "self":
                render_control_hint("Nirnay · Swayam self-ensemble (own OHLCV)")
            else:
                _arch_label = {"producer": "producer cross-section",
                               "hybrid": "agribusiness + futures", "proxy": "cross-asset proxy",
                               "index": "index constituents"}.get(_arch, _arch)
                render_control_hint(f"Nirnay basket · {_arch_label}")

        df = None
        has_data = "data" in st.session_state and "run_analysis" in st.session_state

        if selected_commodity is None:
            # Freeform stock category with no symbol resolved yet (empty input
            # or a resolution error already shown above) — nothing to run/switch
            # to, so don't render either button.
            render_control_hint("Enter a symbol above to continue.")
            if has_data:
                df = st.session_state["data"]
        elif not has_data:
            # Initial load. The fetch pulls the entire macro universe once and
            # is target-agnostic — the chosen commodity only selects Aarambh's
            # target column and Nirnay's basket.
            if st.button("Run Analysis", type="primary"):
                # No spinner — drive the main-area progress bar from the very first
                # click. The fetch is one blocking call, so we show the stage before it
                # (3%) and after it (15%); the analysis picks the bar up from there on
                # the rerun, so the experience reads as one continuous progress bar.
                progress_bar(progress_container, 3, "Fetching Market Data",
                             "yfinance · global macro universe · ~9y daily history")
                _end = pd.Timestamp.today()
                # Walk-forward needs MIN_DATA_POINTS (1500) daily observations.
                # ~9 years of calendar history clears that with headroom.
                _start = _end - pd.Timedelta(days=365 * 9)
                df, error = fetch_commodity_dataset(_start, _end)
                if error or df is None:
                    progress_container.empty()
                    st.error(f"Failed: {error}")
                    return
                progress_bar(progress_container, 15, "Market Data Loaded",
                             f"{df.shape[1]} series × {df.shape[0]} rows · preparing analysis…")
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state["data"] = df
                st.session_state["selected_commodity"] = selected_commodity
                st.session_state["active_target"] = selected_commodity
                st.session_state["nishkarsh_index"] = selected_commodity
                st.session_state["run_analysis"] = True
                st.rerun()
        else:
            df = st.session_state["data"]
            # Post-load target switch — re-runs the engines on the already
            # fetched universe (no re-fetch; only the Nirnay basket re-pulls).
            if selected_commodity != st.session_state.get("active_target"):
                if st.button(f"Switch target → {selected_commodity}", type="primary"):
                    st.session_state["selected_commodity"] = selected_commodity
                    st.session_state["active_target"] = selected_commodity
                    st.session_state["nishkarsh_index"] = selected_commodity
                    st.session_state.pop("active_features", None)  # re-default predictors for new target
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── Landing page if no data loaded ──────────────────────────────────
    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    # A stock target's price column is injected here — BEFORE numeric_cols/
    # commodity_options are computed below — so target_col never falls back
    # to some other target for a stock on its first render (Model
    # Configuration's "Apply Configuration" button, further down, writes
    # that fallback straight into st.session_state["active_target"] if it's
    # ever wrong). Cheap no-op once the column already exists (cached fetch).
    df = _ensure_stock_target_column(df, st.session_state.get("active_target", ""))

    # ─── Sidebar: Model Configuration ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Model Configuration</div>', unsafe_allow_html=True)

        # Target is chosen once in the sidebar "Target Commodity" selector;
        # resolve it here for predictor configuration.
        commodity_options = [c for c in ALL_TARGETS if c in numeric_cols] or numeric_cols
        target_col = st.session_state.get("active_target", commodity_options[0])
        if target_col not in numeric_cols:
            target_col = commodity_options[0]

        # Date column is always the dataset's DATE column — auto-detected.
        date_candidates = [c for c in all_cols if "date" in c.lower()]
        date_col = date_candidates[0] if date_candidates else "None"

        # Read-only target chip (set via the Target Commodity selector above).
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:0.5rem;'
            'padding:0.35rem 0 0.55rem 0;font-family:var(--data);">'
            '<span style="color:var(--ink-tertiary);text-transform:uppercase;'
            'letter-spacing:0.1em;font-size:0.58rem;">Target</span>'
            f'<span style="color:var(--amber);font-weight:700;font-size:0.92rem;">{target_col}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Exclude self-replicating predictors (e.g. GLTR for a precious metal)
        # so they can't leak into Aarambh's fair-value residual.
        _excluded = set(TARGET_EXCLUDED_PREDICTORS.get(target_col, []))
        available = [c for c in numeric_cols if c != target_col and c not in _excluded]
        # Default to the entire macro universe as predictors (bonds, rates,
        # equity/risk, real-asset & commodity/FX). Users can deselect below.
        valid_defaults = list(available)

        if "active_features" not in st.session_state:
            st.session_state["active_features"] = tuple(valid_defaults or available[:3])

        with st.expander("Predictor Columns", expanded=False):
            render_control_hint("Select predictors · click Apply to recompute")
            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state["active_features"] if f in available],
                label_visibility="collapsed",
            )
            if not staging_features:
                st.warning("Select at least one predictor.")
                staging_features = [f for f in st.session_state["active_features"] if f in available] or available[:3]

            staging_set = set(staging_features)
            active_set = set(st.session_state["active_features"])
            has_changes = staging_set != active_set

            if has_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"−{len(removed)} removed")
                render_control_hint(f"Pending · {' · '.join(parts)}")

            if st.button("Apply Configuration" if has_changes else "No changes", disabled=not has_changes, type="primary" if has_changes else "secondary"):
                if has_changes:
                    st.session_state["active_target"] = target_col
                    st.session_state["active_features"] = tuple(staging_features)
                    st.session_state["active_date_col"] = date_col
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

            if len(st.session_state["active_features"]) != len(available):
                st.info(f"Active: {len(st.session_state['active_features'])}/{len(available)} predictors")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            if st.button("Reset Analysis", type="secondary", use_container_width=True):
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("aarambh_engine", None)
                st.session_state.pop("aarambh_fit_key", None)
                st.session_state.pop("wf_results", None)
                st.session_state.pop("results_cache", None)  # drop all cached configs
                st.session_state.pop("run_analysis", None)
                st.session_state.pop("nishkarsh_result", None)
                st.rerun()

            # Force a live re-pull of the whole universe, then recompute — for when
            # the data is stale/partial (the freshness notices point here). Reset =
            # re-run on cached data (fast); Refresh = re-fetch live + re-run (slower).
            # Snapshot-preserving: if the live pull fails (rate-limit / circuit open),
            # the cache's stale fallback keeps the app working on last-good data.
            if st.button("Refresh Data", type="secondary", use_container_width=True):
                from data.cache import begin_force_refresh
                begin_force_refresh()   # next fetches bypass TTL; disk snapshot kept
                # Same main-area progress bar as Run Analysis (no spinner) — the recompute
                # on rerun picks it up from ~15%, so refresh reads as one continuous bar.
                progress_bar(progress_container, 3, "Re-fetching Live Market Data",
                             "yfinance · full universe · bypassing cache · ~30–60s")
                _rend = pd.Timestamp.today()
                _rdf, _rerr = fetch_commodity_dataset(_rend - pd.Timedelta(days=365 * 9), _rend)
                if _rdf is not None:
                    progress_bar(progress_container, 15, "Live Data Refreshed",
                                 f"{_rdf.shape[1]} series × {_rdf.shape[0]} rows · recomputing…")
                    st.session_state["data"] = _rdf   # keep run_analysis → stay in results
                else:
                    progress_container.empty()
                for _k in ("engine", "engine_cache", "aarambh_engine", "aarambh_fit_key",
                           "wf_results", "results_cache", "nishkarsh_result",
                           "precedent_summary", "_prec_key", "_precedent_analogs_cache", "conv_norm_params",
                           # Horizon-independent Nirnay cache (audit finding F17) —
                           # must be dropped on a live re-fetch too, else Refresh
                           # Data re-pulls Aarambh's macro universe live but
                           # silently keeps serving the PRE-refresh Nirnay
                           # basket/constituent analysis.
                           "_nirnay_fetch_cache", "_nirnay_analysis_cache"):
                    st.session_state.pop(_k, None)
                # The convergence tab's actual per-config normalization cache key is
                # "conv_norm_causal::<engine_cache>" (ui/tabs/tab_convergence.py) — the
                # legacy "conv_norm_params" prefix below predates that rename and no
                # longer matches anything, so those z-score caches survived every
                # "Refresh Data" click unpruned (audit finding C1). Sweep both
                # prefixes so a future rename doesn't reintroduce the same gap.
                for _prefix in ("conv_norm_params", "conv_norm_causal::"):
                    for _k in [k for k in list(st.session_state) if str(k).startswith(_prefix)]:
                        st.session_state.pop(_k, None)
                st.rerun()
            render_control_hint("Force-fetch live data · recompute · slower than Reset")

        # ── Model Passport (Sanket-style) ──────────────────────────────
        # Surfaces the active calibrated profile (Intelligence Mode). Each target
        # keys its own profile (must match the key used at calibration time — see
        # _intel_index below).
        _current_universe = st.session_state.get("active_target") or st.session_state.get("selected_commodity", "Gold")
        _current_index = st.session_state.get("nishkarsh_index", _current_universe)
        _render_model_passport_sidebar(_current_universe, _current_index)

        st.markdown('<hr style="margin: 1rem 0 0.75rem 0; opacity: 0.05;">', unsafe_allow_html=True)
        st.markdown(
            '<div class="system-spec">'
            f'<div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>'
            '<div class="spec-row"><span class="spec-label">Engine</span><span class="spec-value">Convergence</span></div>'
            '<div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">yfinance</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ─── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    # Per-instrument config — every engine knob (Nirnay/Swayam, Aarambh forecast,
    # DDM, convergence weights, precedent) is read from THIS target's own config
    # (core.config.INSTRUMENT_CONFIGS), so an instrument can be retuned in
    # isolation. Falls back to the base defaults for any target that somehow
    # isn't registered (shouldn't happen — catalogue targets register at import,
    # stocks via register_stock_target before analysis).
    try:
        _icfg = get_instrument_config(active_target)
    except KeyError:
        from core.config import InstrumentConfig as _IC
        _icfg = _IC()
    active_features = list(st.session_state.get("active_features", [c for c in numeric_cols if c != active_target]))
    # Never let the target — or a self-replicating predictor (e.g. GLTR for a
    # precious metal) — leak into its own predictor set.
    _excluded_feats = {active_target, *TARGET_EXCLUDED_PREDICTORS.get(active_target, [])}
    active_features = [f for f in active_features if f not in _excluded_feats]
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Data freshness notice ──────────────────────────────────────────────
    # Measured in TRADING days behind (weekends ignored) so Friday data read on a
    # Sunday is "current", not stale. Tiered, design-consistent: a calm info note
    # when 1–2 trading days behind (today's bar often isn't published yet), and a
    # prominent warning once genuinely stale (source hasn't updated). The signal
    # always reflects the as-of date shown, never "today".
    # "Trading days behind" is counted on the TARGET's own exchange calendar via
    # data.calendars.trading_days_behind — holiday-aware when exchange_calendars is
    # installed (so Diwali/Thanksgiving no longer over-count by ~1), else it degrades
    # to the exact legacy Mon–Fri busday count. The partial-session check below remains
    # the calendar-agnostic primary freshness signal (native coverage).
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True).dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                # `latest_date` is a tz-naive EXCHANGE-LOCAL date, but "today" has no
                # single frame: a UTC-hosted deploy (Streamlit Cloud) rolls past
                # midnight ahead of an IST/EST exchange, over-counting a current bar
                # as "1 day behind". Anchor to the EARLIER of UTC and machine-local —
                # that brackets the realistic tz band, so a tz skew never *overstates*
                # staleness. Genuine staleness (≥ STALENESS_DAYS) and the exact
                # partial-session gate below still fire normally.
                today = min(datetime.now(timezone.utc).date(), datetime.now().date())
                # trading days strictly after the data date, up to & including today,
                # on the target exchange's calendar (holiday-aware when available).
                _tgt_ticker = ALL_TARGETS.get(active_target)
                behind = trading_days_behind(_tgt_ticker, latest_date.date(), today)
                ds = latest_date.strftime("%d %b %Y")
                if behind >= STALENESS_DAYS:
                    render_warning_box(
                        title="Latest data unavailable",
                        content=(f"Newest data is {ds} — {behind} trading days behind. The price source "
                                 f"(yfinance) hasn't published more recent data, so every signal below "
                                 f"reflects {ds}, not today. Use Refresh Data in the sidebar to pull the "
                                 f"latest once the source updates."),
                    )
                elif behind >= 1:
                    render_info_box(
                        "Data freshness",
                        f"Signals are as of {ds} ({behind} trading day"
                        f"{'s' if behind > 1 else ''} behind — today's bar may not be published yet).",
                    )

                # Predictors carried from a snapshot backfill (data.fetcher's
                # rate-limit recovery, audit finding B1): a rate-limited ticker
                # this run was refilled from the most recent prior snapshot
                # that HAD it, which may itself be stale. Surface which
                # columns and how old, rather than a silent log.warning no
                # one watching a Streamlit deploy will ever see.
                from data.fetcher import _current_stale_backfills
                _stale_backfills = _current_stale_backfills()
                if _stale_backfills:
                    _sb_items = sorted(_stale_backfills.items(), key=lambda kv: kv[1])
                    _sb_preview = ", ".join(f"{k} (as of {v})" for k, v in _sb_items[:5])
                    _sb_more = f" +{len(_sb_items) - 5} more" if len(_sb_items) > 5 else ""
                    render_info_box(
                        "Predictors carried from snapshot",
                        f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                        f"from a prior cached snapshot: {_sb_preview}{_sb_more}. Their momentum is "
                        f"flat until the next successful live fetch.",
                    )

                # Session completeness (Phase 2 — exchange-aware): of the inputs whose
                # market was OPEN on the latest date, how many actually posted a fresh
                # value vs are still forward-filled? Columns whose exchange was CLOSED
                # that day (e.g. US on Thanksgiving) are legitimately carried forward
                # and EXCLUDED — only genuinely-lagging open markets count, so a global
                # holiday no longer trips a false "partial session". Native freshness =
                # changed vs the prior row (continuous prices move every session). With
                # the calendar lib absent, is_session is "is a weekday" → every column
                # counts → identical to the prior calendar-agnostic gate.
                num = df.select_dtypes(include=[np.number])
                if len(num) >= 2:
                    cols = list(num.columns)
                    last_r = num.iloc[-1].to_numpy(dtype=np.float64)
                    prev_r = num.iloc[-2].to_numpy(dtype=np.float64)
                    finite = np.isfinite(last_r) & np.isfinite(prev_r)
                    should_post = np.array(
                        [is_session(_COLUMN_TICKERS.get(c), latest_date.date()) for c in cols]
                    )
                    judged = finite & should_post
                    denom = int(judged.sum())
                    if denom >= 3:   # need a few open markets before judging completeness
                        fresh_frac = float(((last_r != prev_r) & judged).sum() / denom)
                        # Skip the warning when latest_date is today: the session is
                        # still in progress, so most prices are forward-filled by
                        # design — that is expected, not a data problem.
                        _session_is_live = (latest_date.date() >= today)
                        if fresh_frac < SESSION_FRESH_FLOOR and not _session_is_live:
                            render_warning_box(
                                title="Partial latest session",
                                content=(f"Only {fresh_frac:.0%} of the markets open on {ds} have posted — the "
                                         f"rest are forward-filled from the prior session, so the macro predictors "
                                         f"and bottom-up breadth behind the latest signal are stale. Treat it as "
                                         f"provisional; use Refresh Data in the sidebar once those markets post."),
                            )

                # Per-source freshness for the ACTIVE target specifically — it can
                # lag the macro universe (sheet behind, or its market shut on a
                # holiday) with the gap forward-filled. Find its true last update:
                #   • sheet target  → exact, from the source series.
                #   • yfinance/other → detect the ff-filled tail (continuous prices
                #     don't repeat, so a run of identical closes = forward-filled days).
                try:
                    from data.sheets import SHEET_SOURCES, fetch_sheet_series
                    t_last = None
                    if active_target in SHEET_SOURCES:
                        s = fetch_sheet_series(active_target)
                        if s is not None and len(s):
                            t_last = pd.Timestamp(s.index.max()).to_pydatetime()
                    elif active_target in df.columns and active_date in df.columns:
                        tv = pd.to_numeric(df[active_target], errors="coerce").to_numpy()
                        tdates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True)
                        j = len(tv) - 1
                        while j > 0 and np.isfinite(tv[j]) and tv[j] == tv[j - 1]:
                            j -= 1
                        if j < len(tv) - 1 and pd.notna(tdates.iloc[j]):
                            t_last = pd.Timestamp(tdates.iloc[j]).to_pydatetime()
                    if t_last is not None:
                        t_behind = trading_days_behind(_tgt_ticker, t_last.date(), today)
                        if t_behind >= 1:
                            _today_is_session = is_session(_tgt_ticker, today)
                            if _today_is_session and t_behind == 1:
                                # Market is open but today's bar is forward-filled —
                                # most likely yfinance rate-limited this ticker during
                                # the last fetch and the backfill used a prior snapshot.
                                # Prompt a manual refresh rather than crying "stale".
                                render_info_box(
                                    f"{active_target} price not yet updated",
                                    (f"Today's bar is carried forward from {t_last.strftime('%d %b %Y')} — "
                                     f"the {active_target} market is open but yfinance may have rate-limited "
                                     f"this ticker during the last fetch. Use Refresh Data in the sidebar "
                                     f"to pull the latest price."),
                                )
                            else:
                                render_warning_box(
                                    title=f"{active_target} data is lagging",
                                    content=(f"This target last updated {t_last.strftime('%d %b %Y')} "
                                             f"({t_behind} trading day{'s' if t_behind > 1 else ''} behind the macro "
                                             f"universe) — more recent rows are forward-filled from that value, so "
                                             f"its latest signal may be stale."),
                                )
                except Exception:
                    pass
        except Exception:
            pass

    # ─── Clean & Fit Engine ────────────────────────────────────────────────
    # Guard: a selected target whose column failed to fetch (e.g. a sheet/source
    # outage on a later run, while it stays selected) is silently dropped by the
    # column filter below and would KeyError at the per-column coercion. Fail clean.
    # Stock targets (archetype 'self') are never IN the macro batch fetch to begin
    # with — inject their price column here (single-ticker fetch, cached) before
    # the guard checks for it.
    df = _ensure_stock_target_column(df, active_target)
    if active_target not in df.columns:
        _tgt_ticker_guard = ALL_TARGETS.get(active_target, "?")
        if TARGET_ARCHETYPE.get(active_target) == "self":
            console.failure("Stock target fetch failed", f"'{active_target}' (ticker {_tgt_ticker_guard}) — yfinance returned no usable data.")
            st.error(f"'{active_target}' price fetch failed (ticker {_tgt_ticker_guard}) — yfinance returned no data. "
                     f"Check the symbol, or try again once yfinance recovers.")
        else:
            console.failure("Target column missing", f"'{active_target}' not in fetched dataset — its source fetch failed.")
            st.error(f"'{active_target}' data is currently unavailable (its source fetch failed). "
                     f"Pick another target, or re-run once the source is back online.")
        return
    # Data-preparation diagnostics — collected at each stage so the terminal can
    # show exactly how the row count evolves (no "dark spots"), and so a failure
    # explains itself instead of a bare "Need 1500+". Emitted via _log_prep() below.
    _prep = {"target": active_target, "min_required": MIN_DATA_POINTS}
    _tgt_ticker_prep = ALL_TARGETS.get(active_target)
    _tgt_exch_prep = resolve_exchange(_tgt_ticker_prep) or "weekday"

    def _log_prep(stage: str = "complete") -> None:
        """Print the data-prep pipeline to the terminal (visible on success & failure)."""
        console.section("DATA PREPARATION")
        console.item("Target", f"{active_target}  (ticker={_tgt_ticker_prep or 'n/a'}, exch={_tgt_exch_prep})")
        console.item("Horizon", f"forecast {_prep.get('fwd_h','?')}d · momentum {_prep.get('fwd_k','?')}d")
        console.item("Rows · fetched", _prep.get("rows_initial", "?"))
        console.item("Rows · after session spine", f"{_prep.get('rows_session','?')}  ({_prep.get('sessions_dropped','?')} non-session rows removed)")
        console.item("Features · requested", _prep.get("feats_requested", "?"))
        console.item("Features · dropped (short history)", f"{len(_prep.get('feats_dropped', []))}"
                     + (f" → {', '.join(_prep['feats_dropped'][:6])}{'…' if len(_prep.get('feats_dropped', [])) > 6 else ''}" if _prep.get("feats_dropped") else ""))
        console.item("Features · kept", _prep.get("feats_kept", "?"))
        console.item("Rows · after dropna (final spine)", _prep.get("rows_final", "?"))
        if "valid_rows" in _prep:
            console.item("Rows · usable (momentum warmup trimmed)", _prep["valid_rows"])
            console.item("Labels · real (tail forecasts excluded)", _prep["label_valid"])
        if _prep.get("interior_gap_rows", 0):
            # Rows lost to a non-finite predictor value AFTER the leading warmup
            # has already ended (e.g. a yield/price print at/below zero, a
            # temporary data gap) — distinct from the expected warmup trim above.
            # Was previously invisible: the row-wise validity check silently
            # dropped these for every target (audit finding F4).
            console.item("Rows · lost to interior gap (non-finite predictor mid-series)",
                         _prep["interior_gap_rows"])
        if stage == "complete":
            console.checkpoint(f"Data spine ≥ {MIN_DATA_POINTS}", "OK")

    cols = [active_target] + active_features + ([active_date] if active_date != "None" and active_date in df.columns else [])
    data = df[[c for c in cols if c in df.columns]].copy()
    _prep["rows_initial"] = len(data)
    _prep["feats_requested"] = len(active_features)
    if active_date != "None" and active_date in data.columns:
        data[active_date] = pd.to_datetime(data[active_date], errors="coerce", dayfirst=True)
        data = data.dropna(subset=[active_date]).sort_values(active_date)
    for col in [active_target] + active_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    _rows_pre_session = len(data)
    # Phase 3 — target-exchange session spine, applied FIRST so every filter below
    # operates in the target's real trading-session space (not the US-weekday spine).
    # The fetched matrix is a Mon–Fri spine (FX trades every weekday), so a row on the
    # TARGET's own market holiday carries its last close forward: a fake no-change bar
    # with stale predictors. Restricting to genuine sessions up front matters because
    # the feature-history guard and dropna below count rows — measuring them on the US
    # spine while the walk-forward actually runs on (fewer) India/exchange sessions can
    # leave a target just under MIN_DATA_POINTS (India indices have more holidays, so
    # ~1582 US weekdays = ~1496 NSE sessions). No-op for 24×5 FX and under the weekday
    # fallback (lib absent); the `.any()` guard refuses to blank the frame on misfire.
    if active_date != "None" and active_date in data.columns and len(data):
        _smask = session_mask(ALL_TARGETS.get(active_target), data[active_date])
        if _smask.any():
            data = data[_smask].reset_index(drop=True)
    _prep["rows_session"] = len(data)
    _prep["sessions_dropped"] = max(0, _rows_pre_session - len(data))
    data[[active_target] + active_features] = data[[active_target] + active_features].ffill()
    # Drop features with insufficient real history. We ffill (causal: carry last known
    # value forward) but deliberately do NOT bfill — backfilling leading NaNs would inject
    # future values into the past (look-ahead bias). The consequence is that a young or
    # near-empty series (e.g. a just-listed ETF, or a ticker yfinance returned ~nothing for)
    # keeps its leading NaNs, and dropna(subset=all features) would then collapse the whole
    # window to the intersection — as little as 1 row. So drop any feature still carrying a
    # NaN within the most recent MIN_DATA_POINTS *target-session* rows: those can't support
    # the walk-forward window without backfilled fakery. Measuring the tail in session space
    # (after the restriction above) means a feature too young for THIS target's calendar
    # (e.g. SGOV, listed 2020, vs an NSE target) is dropped, extending the usable window back
    # rather than capping it. Survivors are non-null over the tail, so the dropna below
    # retains >= MIN_DATA_POINTS rows whenever the target itself has the history.
    _win = min(MIN_DATA_POINTS, len(data)) if len(data) else 0
    _feats_before_guard = list(active_features)
    active_features = [
        f for f in active_features
        if f in data.columns and _win and data[f].tail(_win).notna().all()
    ]
    _prep["feats_dropped"] = [f for f in _feats_before_guard if f not in active_features]
    _prep["feats_kept"] = len(active_features)
    data = data.dropna(subset=[active_target] + active_features).reset_index(drop=True)
    _prep["rows_final"] = len(data)
    if len(data) < MIN_DATA_POINTS:
        # Explain the shortfall on the terminal — which stage cost the rows.
        _log_prep(stage="fail")
        console.failure(
            "Insufficient data spine for walk-forward",
            f"{active_target}: {len(data)} usable {_tgt_exch_prep} sessions after cleaning, "
            f"need ≥{MIN_DATA_POINTS}. Fetched {_prep['rows_initial']} rows → "
            f"{_prep['rows_session']} after session spine → {len(data)} after dropna "
            f"({_prep['feats_kept']} features kept, {len(_prep['feats_dropped'])} dropped for short history).",
        )
        st.error(
            f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis — "
            f"'{active_target}' yielded only {len(data)} usable {_tgt_exch_prep} trading sessions "
            f"after cleaning. Try a longer history, a target with more data, or fewer young predictors."
        )
        return
    active_features = [f for f in active_features if f in data.columns]
    if not active_features:
        _log_prep(stage="fail")
        console.failure("No valid features", f"{active_target}: every predictor was dropped for short history.")
        st.error("No valid features found after data cleaning.")
        return
    # Returns-based forecasting takes log() of the target → it must be strictly
    # positive. Every shipped target is (prices/levels/ratios), but a future target
    # (a spread, a net position, a yield differential) could go ≤0; fail clean
    # rather than silently producing all-NaN forecasts.
    if (pd.to_numeric(data[active_target], errors="coerce") <= 0).any():
        _log_prep(stage="fail")
        console.failure("Non-positive target", f"{active_target}: contains values ≤ 0; log-return engine needs a strictly positive series.")
        st.error(f"'{active_target}' has non-positive values — the returns-based engine needs a "
                 f"strictly positive series (it forecasts log-returns).")
        return

    # ── Predictive representation: FORECAST the forward return from lagged
    # macro MOMENTUM (ex-ante), rather than explaining the same-day return.
    #   • Features X[t] = trailing FWD_MOM_K-day cumulative log-return of each
    #     predictor (a momentum/trend signal known at time t).
    #   • Target  y[t] = forward FWD_HORIZON-day log-return of the commodity
    #     (t → t+h). The last h rows have no realized future — they are the
    #     LIVE forecasts we trade, so we keep them (target filled 0 only so the
    #     regression doesn't choke; the signal there is the prediction itself).
    # The engine runs in forward_signal mode: conviction is driven by the
    # prediction (expected forward return), and R²/R²-vs-RW measure real
    # out-of-sample forecast skill.
    # Per-instrument forecast horizon + predictor-momentum window (this target's
    # own InstrumentConfig). Daily bars throughout.
    FWD_HORIZON = _icfg.forecast_horizon   # forecast horizon (trading days)
    FWD_MOM_K = _icfg.forecast_momentum    # trailing momentum window for predictors
    _prep["fwd_h"], _prep["fwd_k"] = FWD_HORIZON, FWD_MOM_K
    _lvl = data[[active_target] + active_features].astype(float)
    # Log-return for prices/levels (the target is always one of these — every
    # ALL_TARGETS entry is a price/level, never a raw yield). RAW_YIELD_PREDICTORS
    # (^IRX/^FVX/^TNX/^TYX) are percent-point RATE series, not prices: they can
    # print at/near/below zero (2020-21 zero-rate era), and log() of a
    # non-positive value is NaN. Previously that NaN poisoned _mom.notna().all()
    # for EVERY predictor on that row (the row-wise validity check requires ALL
    # features finite) — silently deleting rows for every target around the most
    # informative volatility regime (audit finding F4). Yield columns instead get
    # an arithmetic level-diff, which is well-defined at any sign and is the
    # economically correct "momentum" for a rate series (a move, not a return).
    # Built as separate blocks and joined via ONE pd.concat rather than
    # assigning each block into an initially-empty frame column-by-column —
    # the latter triggers pandas' "highly fragmented DataFrame"
    # PerformanceWarning on every rerun (same fragmentation hazard already
    # documented in engines/nirnay.py's block-build comments).
    _yield_feats = [f for f in active_features if f in RAW_YIELD_PREDICTORS]
    _price_cols = [c for c in _lvl.columns if c not in _yield_feats]
    _ret_parts = []
    if _price_cols:
        _ret_parts.append(np.log(_lvl[_price_cols].where(_lvl[_price_cols] > 0)).diff().replace([np.inf, -np.inf], np.nan))
    if _yield_feats:
        _ret_parts.append(_lvl[_yield_feats].diff())
    _ret = pd.concat(_ret_parts, axis=1)[_lvl.columns]
    _mom = _ret[active_features].rolling(FWD_MOM_K, min_periods=FWD_MOM_K).sum()
    _fwd = _ret[active_target].rolling(FWD_HORIZON, min_periods=FWD_HORIZON).sum().shift(-FWD_HORIZON)
    # Keep only rows with fully-formed momentum features (drop the warmup head);
    # the forward-target NaN tail is retained for live forecasting.
    _valid = _mom.notna().all(axis=1).to_numpy()
    _label_valid = _fwd.loc[_valid].notna().to_numpy()   # False for last FWD_HORIZON rows (no real label)
    _prep["valid_rows"] = int(_valid.sum())
    _prep["label_valid"] = int(_label_valid.sum())
    # Rows invalid AFTER the leading warmup has ended (first True in _valid) are
    # an INTERIOR gap — a predictor printed non-finite momentum mid-series (see
    # RAW_YIELD_PREDICTORS comment above) — as opposed to the expected warmup
    # trim before any window is fully formed. Surfaced so a future predictor
    # with the same non-positive-print risk doesn't silently delete rows again.
    if _valid.any():
        _first_valid = int(np.argmax(_valid))
        _prep["interior_gap_rows"] = int((~_valid[_first_valid:]).sum())
    else:
        _prep["interior_gap_rows"] = 0
    # Date-range fingerprint for the cache key. `data` carries a RangeIndex (reset at
    # load), so the real dates live in the active_date column, not the index — using
    # the index here would be integers (AttributeError on .date()). Fall back to a
    # valid-row-count surrogate when there's no date column.
    if active_date != "None" and active_date in data.columns:
        _vd = pd.to_datetime(data.loc[_valid, active_date], errors="coerce").dropna()
        _date_range = f"{_vd.iloc[0].date()}_{_vd.iloc[-1].date()}" if len(_vd) else f"n{int(_valid.sum())}"
    else:
        _date_range = f"n{int(_valid.sum())}"
    data = data.loc[_valid].reset_index(drop=True)
    X = _mom.loc[_valid].to_numpy()
    y = np.nan_to_num(_fwd.loc[_valid].to_numpy(), nan=0.0)
    cache_key = f"fwd{FWD_HORIZON}m{FWD_MOM_K}|{active_target}|{'|'.join(sorted(active_features))}|{_date_range}"
    if st.session_state.get("engine_cache") != cache_key:
        # ── Restore from the per-config result cache if this exact config was
        # already computed this session (e.g. the user switched commodities and
        # came back) — full reuse, no recompute. ─────────────────────────────
        _rcache = st.session_state.setdefault("results_cache", {})
        if cache_key in _rcache:
            for _bk, _bv in _rcache[cache_key].items():
                st.session_state[_bk] = _bv
            _rcache[cache_key] = _rcache.pop(cache_key)  # mark most-recently-used
            st.session_state["engine_cache"] = cache_key
            console.header("TATTVA — Cached Result Restored", f"v{VERSION}")
            console.success(f"Restored {active_target} from session cache — no recompute")
            st.rerun()
        if "engine" in st.session_state:
            del st.session_state["engine"]

        # ════ RUN HEADER ════
        console.header("TATTVA — Unified Convergence Analysis", f"v{VERSION}")
        console.main_header("ANALYSIS CONFIGURATION", {
            "Run ID": generate_run_id(),
            "Target": active_target,
            "Predictors": f"{len(active_features)} columns",
            "Date Range": f"{data.shape[0]} observations",
        })
        # Full data-preparation trace (row evolution, session spine, dropped features)
        # so the pipeline has no dark spots — printed once per new computation.
        _log_prep(stage="complete")

        # Reuse the hoisted main-area progress slot (created at the top of main())
        # so the bar continues from where the fetch left it (~15%) with no gap.

        # ── Phase 1: Data Loading ─────────────────────────────────────────
        # HORIZON-INDEPENDENT: basket resolution, macro fetch, and constituent
        # OHLCV depend only on active_target, never on the forecast horizon
        # (FWD_HORIZON/FWD_MOM_K). Cached separately (audit finding F17) so a
        # re-run for the same target reuses this expensive fetch — the
        # walk-forward engine cache_key includes the horizon, this one
        # deliberately doesn't. (The former user-switchable Signal-Horizon lens
        # was removed; the horizon is now a single fixed per-instrument value,
        # but the fetch/compute split it motivated is retained and still saves
        # the re-fetch on every rerun.)
        # Mode resolution — 'self' (Nirnay-Swayam) for individual-stock targets
        # (TARGET_ARCHETYPE == 'self'), 'basket' for everything else. Single
        # source of truth in data.constituents.get_nirnay_mode; see
        # NIRNAY_SWAYAM_PLAN.md §6.1.
        nirnay_mode = get_nirnay_mode(active_target)
        st.session_state["nirnay_mode"] = nirnay_mode

        _nirnay_fetch_key = f"nirnay_fetch::{active_target}"
        _nf_cache = st.session_state.get("_nirnay_fetch_cache")
        if _nf_cache is not None and _nf_cache.get("key") == _nirnay_fetch_key:
            console.start_phase("DATA ACQUISITION", 1, 5)
            constituents = _nf_cache["constituents"]
            src_msg = _nf_cache["src_msg"]
            constituent_ohlcv = _nf_cache["constituent_ohlcv"]
            nirnay_macro_df = _nf_cache["nirnay_macro_df"]
            macro_cols_list = _nf_cache["macro_cols_list"]
            st.session_state["nirnay_basket_source"] = src_msg
            console.item("Basket/Macro/OHLCV", "reused cached fetch (horizon-independent)")
            progress_bar(progress_container, 20, "Data Acquisition Reused", f"{len(constituent_ohlcv)} Constituents · {len(macro_cols_list)} Macros (cached)")
            console.end_phase("DATA ACQUISITION")
        else:
            console.start_phase("DATA ACQUISITION", 1, 5)
            _resolve_sub = (f"{active_target} · own OHLCV (Swayam self-ensemble)" if nirnay_mode == "self"
                            else f"{active_target} · related producers / constituents / sector ETFs")
            progress_bar(progress_container, 16, "Resolving Nirnay Source", _resolve_sub)

            console.section("Basket Resolution")
            if nirnay_mode == "self":
                # No constituent basket — Nirnay-Swayam formulates breadth on
                # the target's OWN OHLCV. "constituents" is the target's own
                # ticker (a 1-symbol fetch list); Phase 3 branches on
                # nirnay_mode to build the self-ensemble instead of iterating
                # constituents as separate instruments.
                constituents = [ALL_TARGETS[active_target]]
                src_msg = "swayam · self-referential ensemble"
            else:
                constituents, src_msg = get_commodity_basket(active_target)
            # Surfaced in the Nirnay tab (not just the console — audit finding B4):
            # a "snapshot (N)" source means live scrape + cache both failed, and
            # for an uncapped large index (S&P 500 / Nasdaq 100) N is a small
            # fraction of the true index, so breadth there is read from a partial
            # basket, not the full constituent set.
            st.session_state["nirnay_basket_source"] = src_msg
            console.item("Target", active_target)
            console.item("Source", src_msg)
            console.item("Count", len(constituents))
            if constituents:
                console.item("Symbols", f"{', '.join(constituents[:3])}...")
            console.success(f"Resolved {len(constituents)}-instrument {active_target} basket")
            progress_bar(progress_container, 17, "Fetching Nirnay Macro Data", "yfinance · Global Macro ETFs · FX · Commodities · ~9y")

            console.section("Macro Data")
            end_date = pd.Timestamp.today()
            # Match the Aarambh model-dataset window (~9y) so the Nirnay basket and
            # macro features overlap the FULL series — convergence + Intelligence
            # calibration then run on real data, not neutral placeholders.
            start_date = end_date - pd.Timedelta(days=365 * 9)
            macro_df = fetch_macro_live(start_date, end_date)
            console.item("Date Range", f"{start_date.date()} to {end_date.date()}")
            if not macro_df.empty:
                console.item("YF Columns", f"{len(macro_df.columns)} symbols")
                console.item("Rows", len(macro_df))
                console.success(f"Macro data: {len(macro_df.columns)} symbols × {len(macro_df)} rows")
            else:
                console.warning("No macro data available")
            progress_bar(progress_container, 18, "Fetching Constituent OHLCV", f"yfinance · {len(constituents)} basket constituents")

            console.section("Constituent OHLCV")
            constituent_ohlcv = {}
            if constituents:
                constituent_ohlcv = fetch_constituent_ohlcv(constituents, start_date, end_date)
                console.item("Requested", len(constituents))
                console.item("Downloaded", len(constituent_ohlcv))
                if constituent_ohlcv:
                    sample = list(constituent_ohlcv.items())[0]
                    console.item("Sample", f"{sample[0]}: {len(sample[1])} rows")
                console.success(f"OHLCV data for {len(constituent_ohlcv)} constituents")
            progress_bar(progress_container, 19, "Assembling Macro Indicators", f"{len(constituent_ohlcv)} constituents downloaded · aligning macro frame")

            console.section("Nirnay Macro Assembly")
            nirnay_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
            if not nirnay_macro_df.empty:
                console.item("YF Symbols", len(nirnay_macro_df.columns))
                console.success(f"Macro indicators: {len(nirnay_macro_df.columns)} × {len(nirnay_macro_df)} rows")
            macro_cols_list = list(nirnay_macro_df.columns) if not nirnay_macro_df.empty else []
            console.end_phase("DATA ACQUISITION")
            progress_bar(progress_container, 20, "Data Acquisition Complete", f"{len(constituent_ohlcv)} Constituents · {len(nirnay_macro_df.columns)} Macros")

            st.session_state["_nirnay_fetch_cache"] = {
                "key": _nirnay_fetch_key,
                "constituents": constituents, "src_msg": src_msg,
                "constituent_ohlcv": constituent_ohlcv,
                "nirnay_macro_df": nirnay_macro_df, "macro_cols_list": macro_cols_list,
            }

        # ── Phase 2: Aarambh FairValueEngine ─────────────────────────────
        console.start_phase("AARAMBH ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Aarambh Engine", f"Walk-Forward · {len(active_features)} Predictors · {len(data)} Rows")

        # PCA component count — this target's own config (default 2, per the
        # aarambh_full PCA lever, research/TUNING_COVERAGE.md). Single local so the
        # console line below and the engine.fit call can never disagree.
        _N_PCA = _icfg.pca_components

        console.section("Engine Configuration")
        console.item("Mode", f"Predictive · forecast {FWD_HORIZON}d forward return")
        console.item("Target", active_target)
        console.item("Features", f"{len(active_features)} macro momentum ({FWD_MOM_K}d) → PCA({_N_PCA}) causal")
        console.item("Observations", f"{len(data)} rows")
        console.item("Min Data Points", MIN_DATA_POINTS)
        console.item("Lookback Windows", f"{LOOKBACK_WINDOWS}")

        console.section("Walk-Forward Regression")
        # Reuse an already-fit Aarambh engine for this exact config if a prior
        # (possibly interrupted) execution in THIS session already produced one.
        # `engine_cache` is only set at the end of Phase 5, so a Streamlit rerun
        # mid-pipeline (yfinance retry, cloud reconnect, stray interaction) would
        # otherwise re-enter this block and re-run the expensive walk-forward.
        # Keyed by cache_key → identical inputs → identical fit, so reuse is safe.
        if (st.session_state.get("aarambh_fit_key") == cache_key
                and isinstance(st.session_state.get("aarambh_engine"), FairValueEngine)):
            engine = st.session_state["aarambh_engine"]
            console.item("Walk-Forward", "reused cached fit (resumed run)")
            progress_bar(progress_container, 40, "Aarambh Engine Reused", "Cached walk-forward fit")
        else:
            engine = FairValueEngine()
            # Pass the genuine price LEVEL into fit() so the forward-change
            # table and divergence detection use real (non-overlapping) price
            # differences instead of reconstructing a pseudo-price from the
            # h-day FORWARD-return target y (which would sum each daily return
            # up to h times — see FairValueEngine.fit's `price` docstring).
            _price_level = data[active_target].to_numpy(dtype=np.float64)
            # `config=_icfg` threads this instrument's per-instrument Aarambh
            # training knobs (refit / min-max train / ensemble / ridge / huber /
            # lookback) into the walk-forward, so Aarambh is tuned per instrument
            # / asset class exactly like Nirnay and Swayam.
            engine.fit(X, y, feature_names=active_features, forward_signal=True, n_pca_components=_N_PCA, purge=FWD_HORIZON, label_mask=_label_valid, price=_price_level, config=_icfg, progress_callback=lambda pct, msg: progress_bar(progress_container, int(20 + pct * 20), "Running Aarambh Engine", msg))
            # Carry the raw price LEVEL on the engine output too (returns-space
            # modeling otherwise leaves only return-scale columns). Used by the
            # Aarambh tab for price display and by the Intelligence tuner.
            engine.ts_data["Price"] = _price_level
            st.session_state["aarambh_engine"] = engine
            st.session_state["aarambh_fit_key"] = cache_key

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()
        console.section("Engine Results")
        console.item("Signal", f"{sig['signal']} ({sig['strength']})")
        console.item("Conviction", f"{sig['conviction_score']:+.0f}")
        console.item("OOS R²", f"{stats['r2_oos']:.3f} (forecast vs realized fwd return)")
        console.item("Model Spread", f"{sig['model_spread'] * 10000:.1f} bps (ensemble disagreement)")
        # NOTE: R²-vs-RW, OU half-life and Hurst are computed on the forecast
        # series in predictive mode and are NOT meaningful (the RW baseline is
        # overlap-inflated; OU/Hurst describe forecast persistence, not mean
        # reversion). Omitted here — see the Diagnostics tab / Val IC instead.
        console.success(f"Aarambh engine complete | {len(engine.ts_data)} output rows")
        console.end_phase("AARAMBH ENGINE")
        progress_bar(progress_container, 40, "Aarambh Engine Complete", f"Signal: {sig['signal']} ({sig['strength']}) · Conviction: {sig['conviction_score']:+.0f}")

        # ── Phase 3: Nirnay Constituent Analysis ──────────────────────────
        # HORIZON-INDEPENDENT (audit finding F17): per-constituent MSF/MMR/regime
        # analysis and aggregation depend only on active_target's basket +
        # macro window, never on the forecast horizon. This is the ~30s
        # (Nifty 50) to ~5min (uncapped S&P 500) cost a re-run would otherwise
        # re-pay for byte-identical output. Cached under the SAME
        # _nirnay_fetch_key as Phase 1; only the target-calendar reindex below
        # (cheap — no yfinance calls) re-runs, since the horizon's warm-up trim
        # can shift the target's date spine slightly. (Former Signal-Horizon
        # lens removed; horizon is now a single fixed per-instrument value.)
        console.start_phase("NIRNAY ENGINE", 3, 5)
        _nirnay_sub = ("MSF+MMR+Regime · Swayam self-ensemble (own OHLCV)" if nirnay_mode == "self"
                       else f"MSF+MMR+Regime · {len(constituent_ohlcv)} Constituents")
        progress_bar(progress_container, 42, "Running Nirnay Engine", _nirnay_sub)

        _na_cache = st.session_state.get("_nirnay_analysis_cache")
        if _na_cache is not None and _na_cache.get("key") == _nirnay_fetch_key:
            nirnay_constituent_dfs = _na_cache["nirnay_constituent_dfs"]
            nirnay_daily_pre_reindex = _na_cache["nirnay_daily_pre_reindex"]
            if nirnay_mode == "self" and "n_eff" in _na_cache:
                st.session_state["nirnay_swayam_n_eff"] = _na_cache["n_eff"]
            console.item("Per-Stock Analysis", "reused cached fit (horizon-independent)")
            progress_bar(progress_container, 74, "Nirnay Engine Reused", f"{len(nirnay_constituent_dfs)} Stocks (cached)")
        else:
            nirnay_daily_pre_reindex = pd.DataFrame()
            nirnay_constituent_dfs = {}

            if constituent_ohlcv and nirnay_mode == "self":
                console.section("Swayam Self-Ensemble")
                target_tkr = constituents[0]
                target_ohlcv = constituent_ohlcv.get(target_tkr)
                if target_ohlcv is not None and not target_ohlcv.empty:
                    # Leakage guard (NIRNAY_SWAYAM_PLAN.md §4.4): drop the
                    # target's own macro column + its excluded-predictor
                    # near-replicas from the MMR driver pool — a member's
                    # Close correlates ~1.0 with the target's own macro
                    # column, which would let MMR "explain" the target with
                    # itself and silently zero the deviation oscillator.
                    swayam_cols = swayam_macro_columns(active_target, macro_cols_list)
                    # This target's own Swayam grid (from its InstrumentConfig).
                    _swayam_members = default_swayam_members(_icfg.swayam_lengths, _icfg.swayam_roc_frac)
                    console.item("Views (grid)", f"{len(_swayam_members)} · timescale × information-set × mechanism")
                    console.item("MSF Length Grid", str(_icfg.swayam_lengths))
                    console.item("Regime Sensitivity", _icfg.nirnay_regime_sensitivity)
                    console.item("Base Weight", _icfg.nirnay_base_weight)
                    console.item("Macro Columns (post-leakage-guard)", len(swayam_cols))

                    def _swayam_progress(done, total, name):
                        pct_val = int(45 + done / max(total, 1) * 30)
                        progress_bar(progress_container, pct_val, f"View {name}", f"{done}/{total} views")

                    nirnay_constituent_dfs = build_swayam_frames(
                        target_ohlcv, nirnay_macro_df, swayam_cols,
                        members=_swayam_members,
                        regime_sensitivity=_icfg.nirnay_regime_sensitivity, base_weight=_icfg.nirnay_base_weight,
                        num_vars=_icfg.nirnay_mmr_num_vars,
                        oversold=_icfg.nirnay_oversold, overbought=_icfg.nirnay_overbought,
                        progress_cb=_swayam_progress,
                    )
                    n_eff = effective_member_count(nirnay_constituent_dfs)
                    st.session_state["nirnay_swayam_n_eff"] = n_eff
                    console.success(f"Swayam ensemble: {len(nirnay_constituent_dfs)} views · ~{n_eff:.1f} effective")

                if nirnay_constituent_dfs:
                    console.section("Aggregation")
                    nirnay_daily_pre_reindex = aggregate_constituent_timeseries(nirnay_constituent_dfs)
                    # Self mode is definitionally co-directional (the target
                    # vs itself) — apply_polarity is never invoked here
                    # (INV: self-mode polarity is always a no-op).
            elif constituent_ohlcv:
                total = len(constituent_ohlcv)
                console.section("Per-Stock Analysis")
                console.item("Constituents", total)
                console.item("MSF Length", _icfg.nirnay_msf_length)
                console.item("ROC Length", _icfg.nirnay_roc_len)
                console.item("Regime Sensitivity", _icfg.nirnay_regime_sensitivity)
                console.item("Base Weight", _icfg.nirnay_base_weight)
                console.item("MMR Top-N Drivers", _icfg.nirnay_mmr_num_vars)
                console.item("Oversold / Overbought", f"{_icfg.nirnay_oversold} / {_icfg.nirnay_overbought}")
                console.item("Macro Columns", len(macro_cols_list))

                for i, (sym, ohlcv_df) in enumerate(constituent_ohlcv.items()):
                    try:
                        merged = ohlcv_df.copy()
                        if nirnay_macro_df is not None and not nirnay_macro_df.empty:
                            merged = merged.join(nirnay_macro_df, how="left")
                            merged[macro_cols_list] = merged[macro_cols_list].ffill()

                        n_rows = len(merged)
                        has_macro = len([c for c in macro_cols_list if c in merged.columns])

                        result_df, _ = run_full_analysis(
                            merged, length=_icfg.nirnay_msf_length, roc_len=_icfg.nirnay_roc_len,
                            regime_sensitivity=_icfg.nirnay_regime_sensitivity, base_weight=_icfg.nirnay_base_weight,
                            num_vars=_icfg.nirnay_mmr_num_vars,
                            oversold=_icfg.nirnay_oversold, overbought=_icfg.nirnay_overbought,
                            macro_columns=macro_cols_list,
                        )
                        nirnay_constituent_dfs[sym] = result_df

                        last_row = result_df.iloc[-1]
                        osc = last_row.get('Unified_Osc', 0)
                        cond = last_row.get('Condition', 'N/A')
                        regime = last_row.get('Regime', 'N/A')
                        console.detail(f"[{i+1}/{total}] {sym}: osc={osc:+.1f} [{cond}] regime={regime} rows={n_rows} macros={has_macro}")

                        pct_val = int(45 + (i + 1) / total * 30)
                        progress_bar(progress_container, pct_val, f"Analyzing {sym}", f"Osc={osc:+.1f} [{cond}] Regime={regime}")

                    except Exception as e:
                        console.failure(f"{sym}", str(e))

                if nirnay_constituent_dfs:
                    console.section("Aggregation")
                    nirnay_daily_pre_reindex = aggregate_constituent_timeseries(nirnay_constituent_dfs)
                    # Re-orient breadth to the target's direction for inverse baskets
                    # (no-op for co-directional baskets, i.e. all current targets).
                    _polarity = TARGET_POLARITY.get(active_target, 1)
                    if _polarity < 0:
                        nirnay_daily_pre_reindex = apply_polarity(nirnay_daily_pre_reindex, _polarity)
                        console.item("Polarity", f"{_polarity} (basket inverted to target)")

            st.session_state["_nirnay_analysis_cache"] = {
                "key": _nirnay_fetch_key,
                "nirnay_constituent_dfs": nirnay_constituent_dfs,
                "nirnay_daily_pre_reindex": nirnay_daily_pre_reindex,
                "n_eff": st.session_state.get("nirnay_swayam_n_eff") if nirnay_mode == "self" else None,
            }

        # ── HORIZON-DEPENDENT tail: reindex onto the target's calendar ──────
        # Cheap (pure pandas, no yfinance) — re-runs since the horizon's
        # warm-up trim can shift the target's date spine.
        nirnay_daily = nirnay_daily_pre_reindex
        if not nirnay_daily.empty:
            # Carry the basket forward onto the TARGET's trading calendar. The
            # constituents (often global / US-listed) trade on a different calendar
            # than the target — on a Monday-morning IST run, or when the target's
            # market is open but the basket's is on holiday, the basket's last close
            # IS its current value. Reindexing it onto the target's dates (ff-fill)
            # lets the SIGNAL, cards and plots all reach the target's latest session
            # instead of truncating to the slowest constituent. We record the
            # basket's true last-native date so the UI can flag how much is carried
            # over (the partial-session notice covers the row-level staleness).
            st.session_state["nirnay_native_last"] = pd.Timestamp(nirnay_daily.index.max())
            if active_date in data.columns:
                _cal = pd.DatetimeIndex(sorted(pd.to_datetime(
                    data[active_date], errors="coerce").dropna().dt.normalize().unique()))
                _nd = nirnay_daily.copy()
                _nd.index = pd.to_datetime(_nd.index).normalize()
                _nd = _nd[~_nd.index.duplicated(keep="last")].sort_index()
                # _Native marks rows that are a genuine basket observation
                # (present in _nd BEFORE the reindex) vs carried forward by
                # the ffill below (the basket's market was closed/hadn't
                # posted that day). Carried through so the calibration
                # overlap gate can require NATIVE overlap, not ffilled
                # rows masquerading as fresh Nirnay signal (audit finding
                # F21) — the UI's own "breadth carried forward" notice
                # already discloses this to the user; the gate didn't.
                _native_dates = set(_nd.index)
                nirnay_daily = _nd.reindex(_cal, method="ffill").dropna(how="all")
                nirnay_daily["_Native"] = nirnay_daily.index.isin(_native_dates)
            console.item("Trading Days", len(nirnay_daily))
            if len(nirnay_daily) > 0:
                last = nirnay_daily.iloc[-1]
                console.item("Avg Signal", f"{last.get('Avg_Signal', 0):+.2f}")
                console.item("Oversold %", f"{last.get('Oversold_Pct', 0):.0f}%")
                console.item("Overbought %", f"{last.get('Overbought_Pct', 0):.0f}%")
                console.item("Buy Signals", int(last.get('Buy_Signals', 0)))
                console.item("Sell Signals", int(last.get('Sell_Signals', 0)))
            console.success(f"Nirnay aggregation: {len(nirnay_daily)} trading days")

        console.end_phase("NIRNAY ENGINE")
        progress_bar(progress_container, 75, "Nirnay Engine Complete", f"{len(nirnay_constituent_dfs)} Stocks · {len(nirnay_daily)} Trading Days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 78, "Computing Convergence", "Cross-Validation · DDM Filtering")

        console.section("Cross-Validation Setup")
        # ── Intelligence Mode — PRIOR profile resolution ─────────────────
        # The first convergence pass uses the PRIOR profile (saved from a
        # previous run, if any). After this pass we auto-calibrate on the
        # fresh data and apply the new weights in-place (see Phase 4.5).
        # When the Intelligence Mode toggle is OFF, both passes use the
        # factory defaults — no calibration runs.
        from convergence import intelligence as _intel_mod
        _intel_universe = active_target
        # One calibrated profile per target (the forecast-lens tag was dropped
        # from the key when the Signal-Horizon selector was removed — there is a
        # single horizon now, so no Tactical/Positional profiles to disambiguate).
        _intel_index = st.session_state.get("nishkarsh_index", _intel_universe)
        _intel_enabled = bool(st.session_state.get("intelligence_mode", True))
        if _intel_enabled:
            _prior_w, _prior_t, _prior_profile = _intel_mod.resolve_active(_intel_universe, _intel_index)
        else:
            _prior_w, _prior_t, _prior_profile = (
                _icfg.weights_seed(),          # this instrument's convergence-weight seed
                _icfg.composite_thresholds(),  # …and its classification threshold seed
                None,
            )
        if _prior_profile is not None:
            console.item(
                "Prior profile",
                f"✅ {_prior_profile.universe} · val IC {_prior_profile.val_ic:+.3f} · "
                f"trained {_prior_profile.timestamp}",
            )
        else:
            console.item("Prior profile", "None (first run / no profile)")
        console.item(
            "Intelligence Mode",
            "ON (auto-calibrate after convergence)" if _intel_enabled else "OFF (defaults locked)",
        )

        # First-pass validator uses the learned profile's weights when available,
        # else this instrument's own convergence-weight seed (from its config).
        _validator_weights = _prior_w if _prior_profile is not None else _icfg.weights_seed()
        # In self mode "constituents" is a 1-item list (the target's own
        # ticker) — the actual vote count is the Swayam ensemble's member
        # count (all views always report, so coverage reads 1.0), not 1.
        _expected_constituents = (
            (len(nirnay_constituent_dfs) or None) if nirnay_mode == "self"
            else (len(constituents) or None)
        )
        validator = CrossValidator(
            active_weights=_validator_weights,
            expected_constituents=_expected_constituents,
        )
        divergence_detector = CrossSystemDivergenceDetector()

        aarambh_ts = engine.ts_data.copy()  # carries "Price" (set after fit)
        if active_date != "None" and active_date in data.columns:
            aarambh_ts["Date"] = pd.to_datetime(data[active_date].values)
            aarambh_ts = aarambh_ts.set_index("Date")
        else:
            aarambh_ts["Date"] = np.arange(len(aarambh_ts))
        aarambh_ts = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
        console.item("Aarambh Dates", len(aarambh_ts))

        nirnay_by_date = {}
        if not nirnay_daily.empty:
            nirnay_unique = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")]
            for idx in nirnay_unique.index:
                key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
                nirnay_by_date[key] = nirnay_unique.loc[idx]
            console.item("Nirnay Dates", len(nirnay_by_date))

        console.section("Daily Convergence Scoring")
        overlap_count = 0
        native_overlap_count = 0
        skipped_warmup = 0
        total_dates = len(aarambh_ts.index)
        for i, ts_idx in enumerate(aarambh_ts.index):
            ts_date = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            date_str = str(ts_date)
            row_a = aarambh_ts.loc[ts_idx]
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[-1]
            # Skip the engine's own [0, MIN_TRAIN_SIZE) warm-up rows — the
            # `Valid` column (engines/aarambh.py) is False there because no
            # genuine walk-forward forecast covers them (see A3 in the audit).
            # Scoring them would feed the Intelligence calibration frame and
            # the walk-forward IC a fabricated "neutral" convergence reading
            # instead of genuinely excluding the unfit region.
            if not bool(row_a.get("Valid", True)):
                skipped_warmup += 1
                continue
            aarambh_sig = {
                "conviction_score": float(row_a.get("ConvictionBounded", 0)),
                "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
                "regime": str(row_a.get("Regime", "NEUTRAL")),
            }
            if date_str in nirnay_by_date:
                row_n = nirnay_by_date[date_str]
                nirnay_stats = {
                    "oversold_pct": float(row_n.get("Oversold_Pct", 50)),
                    "overbought_pct": float(row_n.get("Overbought_Pct", 50)),
                    "avg_unified_osc": float(row_n.get("Avg_Signal", 0)),
                    "regime_bull_pct": float(row_n.get("Regime_Bull_Pct", 33)),
                    "regime_bear_pct": float(row_n.get("Regime_Bear_Pct", 33)),
                    "regime_neutral": float(row_n.get("Regime_Neutral", 34)),
                    "num_constituents": int(row_n.get("Total_Analyzed", 0)),
                }
                overlap_count += 1
                if bool(row_n.get("_Native", True)):
                    native_overlap_count += 1
            else:
                nirnay_stats = {
                    "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                    "regime_bull_pct": 33, "regime_bear_pct": 33,
                    "regime_neutral": 34, "num_constituents": 0,
                }
            validator.compute_convergence(aarambh_sig, nirnay_stats, date_str)
            divergence_detector.detect(aarambh_sig, nirnay_stats, date_str)

            if (i + 1) % 10 == 0 or i == total_dates - 1:
                pct_val = int(78 + (i + 1) / total_dates * 7)
                progress_bar(progress_container, pct_val, "Computing Convergence", f"{i + 1}/{total_dates} Dates Scored")

        console.item("Total Aarambh Dates", len(aarambh_ts))
        console.item("Skipped (warm-up, no genuine forecast)", skipped_warmup)
        console.item("Overlap Dates", f"{overlap_count} ({native_overlap_count} native, "
                     f"{overlap_count - native_overlap_count} carried-forward)")
        console.success("Convergence scoring complete")

        # ── Intelligence Mode overlap gate ───────────────────────────────
        # Decided HERE (overlap_count is final) rather than just before the
        # calibration block itself, because the "First-Pass"/"(initial pass)"
        # labels below are chosen from _intel_enabled — deciding the gate
        # after those labels were already picked meant a skip run advertised
        # a "first pass" that would never have a second pass.
        #
        # Skip entirely when the Aarambh/Nirnay overlap is too thin (an
        # Aarambh-only target with an empty/unresolvable basket, or a target
        # whose basket barely overlaps the model's date range). With no
        # overlap every date gets the same neutral nirnay_stats default
        # (app.py's fallback above), so the continuous consensus direction
        # degenerates to aarambh_bull/2 and every nirnay-driven dim score is
        # constant — the tuner would then "calibrate" what is really a
        # half-weight Aarambh-only signal against forward returns and persist
        # it as a convergence profile (under the old hard direction gate the
        # score was a constant 0 and the objective all-NaN; the continuous
        # form makes the gate MORE necessary, since the degenerate signal now
        # looks alive). 60 overlap dates is a low bar (~3 trading months)
        # chosen only to exclude the genuinely-empty-basket case, not to
        # second-guess a real but short-history calibration.
        #
        # Gated on NATIVE overlap, not raw overlap (audit finding F21):
        # nirnay_daily is forward-filled onto the target's calendar before
        # this loop runs (the basket's own market may be closed on a day the
        # target trades), so `overlap_count` alone would happily count a long
        # run of carried-forward, non-fresh Nirnay rows as "signal to
        # calibrate against" — the SAME breadth reading repeated across many
        # dates, not genuinely new cross-sectional information each day.
        _MIN_OVERLAP_FOR_CALIBRATION = 60
        if _intel_enabled and native_overlap_count < _MIN_OVERLAP_FOR_CALIBRATION:
            console.warning(
                f"Intelligence calibration skipped: only {native_overlap_count} NATIVE "
                f"Aarambh/Nirnay overlap dates (< {_MIN_OVERLAP_FOR_CALIBRATION}; "
                f"{overlap_count} total overlap incl. carried-forward rows) — "
                f"convergence_score would be dominated by repeated, non-fresh breadth."
            )
            _intel_enabled = False

        # ── 4a. First-pass conviction model ─────────────────────────────
        # First-pass DDM filter on the convergence_score from the first
        # validator pass. Labeled "first-pass" only when Intelligence Mode
        # is ON (a second pass will follow); just "Conviction Model" otherwise.
        _first_pass_label = "First-Pass Conviction Model" if _intel_enabled else "Conviction Model"
        console.section("Conviction Model (initial pass)" if _intel_enabled else "Conviction Model")
        progress_bar(
            progress_container, 83, _first_pass_label,
            "DDM Filter · Prior Weights" if (_intel_enabled and _prior_profile is not None) else "DDM Filter · Default Weights",
        )
        convergence_df = validator.get_convergence_series()
        # DDM smoothing = the shared consensus-filter tuning (CONV_DDM_*).
        conviction_model = UnifiedConvictionModel(
            leak_rate=_icfg.ddm_leak,
            drift_scale=_icfg.ddm_drift,
            long_run_var=_icfg.ddm_lrv,
        )
        results = conviction_model.fit(
            convergence_df["convergence_score"].tolist(),
            convergence_df.index.tolist(),
        )
        if results:
            latest = results[-1]
            _pre_label = "DDM Conviction (pre-cal)" if _intel_enabled else "DDM Conviction"
            _sig_label = "DDM Signal (pre-cal)" if _intel_enabled else "DDM Signal"
            console.item(_pre_label, f"{latest.nishkarsh_conviction:+.0f}")
            console.item(_sig_label, latest.nishkarsh_signal)
        console.success(f"Initial conviction: {len(results)} scores computed")

        # ── 4b. AUTO-CALIBRATION (Intelligence Mode) ────────────────────
        # Runs Optuna TPE on the fresh convergence_df + aarambh_ts. The
        # search learns optimal (weights, thresholds) for this universe,
        # persists them to disk, and we immediately re-apply them below
        # so the user's signals reflect the calibrated state on THIS run.
        # (The overlap gate that can force _intel_enabled = False lives
        # earlier now, right after the convergence-scoring summary — see
        # "Intelligence Mode overlap gate" above — so it decides before the
        # first-pass labels are chosen, not after.)
        _final_profile: _intel_mod.IntelligenceProfile | None = None
        if _intel_enabled:
            console.section("Intelligence Calibration")
            _n_trials = int(st.session_state.get("intel_n_trials", INTEL_N_TRIALS))
            progress_bar(
                progress_container, 84,
                "Intelligence Mode · Setup",
                f"Building Tuner · Purged K-Fold CV + Holdout · {_n_trials} Trials",
            )
            try:
                tuner = _intel_mod.ConvergenceTuner(
                    convergence_df, aarambh_ts,
                    universe=_intel_universe, selected_index=_intel_index,
                    target_col="Price",
                    horizons=_icfg.hold_horizons,  # validate at this instrument's forecast horizons
                )
                console.item("TPE Trials", _n_trials)
                console.item("Objective", f"{tuner.n_cv_folds}-fold CV (purged) · L2 {tuner.l2_alpha}")
                console.item("Holdout", f"{int((1-tuner.train_frac)*100)}% purged chronological")
                console.item("Horizons", " · ".join(str(h) for h in tuner.horizons))
                console.item("Opt Rows", len(tuner.train_frame))
                console.item("Holdout Rows", len(tuner.val_frame))

                def _cal_cb(trial_num: int, total: int, best: float) -> None:
                    # Map trial progress to 84% → 90% on the global bar.
                    # Linear interpolation gives smooth motion through the loop.
                    pct = int(84 + (trial_num / max(1, total)) * 6)
                    progress_bar(
                        progress_container, pct,
                        "Intelligence Mode · Calibrating",
                        f"Optuna Trial {trial_num}/{total} · Best Score {best:+.4f}",
                    )

                _final_profile, _ = tuner.optimize(n_trials=_n_trials, progress_callback=_cal_cb)
                tuner.evaluate_validation()
                _final_profile = tuner._make_profile()
                _intel_mod.save_profile(_final_profile)

                progress_bar(
                    progress_container, 90,
                    "Intelligence Mode · Profile Saved",
                    f"Train IC {_final_profile.train_ic:+.3f} · Val IC {_final_profile.val_ic:+.3f}",
                )
                console.item("Train IC", f"{_final_profile.train_ic:+.4f}")
                console.item("Val IC",   f"{_final_profile.val_ic:+.4f}")
                # Top-3 most important parameters
                if _final_profile.sensitivity:
                    _top3 = sorted(_final_profile.sensitivity.items(), key=lambda kv: -kv[1])[:3]
                    console.item("Top drivers", " · ".join(f"{k} {v:.0f}%" for k, v in _top3))
                console.success(
                    f"Calibration complete · val IC {_final_profile.val_ic:+.3f} · "
                    f"persisted to disk ({_intel_universe} · {_intel_index})"
                )
            except Exception as _cal_e:
                console.warning(f"Calibration failed: {_cal_e} — falling back to prior profile / defaults")
                _final_profile = _prior_profile

        # ── 4c. APPLY calibrated weights + thresholds to current run ────
        # Either we just calibrated (use _final_profile) or Intelligence
        # Mode is OFF (use defaults). Vectorized recomputation of
        # convergence_score from existing dim_* columns — no need to
        # re-loop CrossValidator over every date.
        if _final_profile is not None:
            progress_bar(
                progress_container, 91,
                "Applying Calibrated Profile",
                "Re-Weighting Convergence · Vectorized Recompute",
            )
            convergence_df = _intel_mod.apply_calibrated_weights(
                convergence_df, _final_profile.weights,
            )
            progress_bar(
                progress_container, 92,
                "Re-Fitting Conviction Model",
                "Post-Calibration DDM Pass",
            )
            # Re-fit the conviction model with the new convergence_score
            # (same consensus-filter DDM as the initial pass).
            conviction_model = UnifiedConvictionModel(
                leak_rate=_icfg.ddm_leak,
                drift_scale=_icfg.ddm_drift,
                long_run_var=_icfg.ddm_lrv,
            )
            results = conviction_model.fit(
                convergence_df["convergence_score"].tolist(),
                convergence_df.index.tolist(),
            )
            console.section("Conviction Model (post-cal)")
            if results:
                latest = results[-1]
                console.item("DDM Conviction", f"{latest.nishkarsh_conviction:+.0f}")
                console.item("DDM Signal", latest.nishkarsh_signal)
                console.item("DDM Band", f"[{latest.confidence_lower:.0f}, {latest.confidence_upper:.0f}]")
            console.success("Re-fit complete with calibrated profile")
            _active_w = _final_profile.weights
            _active_t = _final_profile.thresholds
        elif _prior_profile is not None:
            # Intelligence Mode OFF, or the overlap gate skipped calibration
            # THIS run — but `validator` above was already constructed with
            # `_prior_w` (a saved profile's weights), so convergence_df's
            # dim_* composite reflects the PRIOR calibrated weights, not
            # factory defaults. Publish what actually ran, so the Passport /
            # Convergence cards don't report "Default" while the computed
            # series is calibrated (previously always fell to the defaults
            # branch below regardless of whether a prior profile seeded the
            # first pass).
            _active_w = _prior_w
            _active_t = _prior_t
        else:
            # Genuinely no profile involved — record this instrument's own
            # config seed as the active weights + thresholds (per-instrument).
            _active_w = _icfg.weights_seed()
            _active_t = _icfg.composite_thresholds()

        # Publish to session state so the Passport sidebar + Convergence cards
        # see the calibrated state immediately on the next rerun.
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_thresholds"] = _active_t
        st.session_state["intelligence_active_profile"] = (
            _final_profile.to_dict() if _final_profile is not None
            else (_prior_profile.to_dict() if _prior_profile is not None else None)
        )

        # ── 4d. NORMALIZED CONSENSUS — the HEADLINE object ───────────────────
        # Product decision (consensus-headline): the hero card headlines the
        # normalized consensus — the causal expanding-z average of Aarambh's
        # ConvictionRaw and Nirnay's Avg_Signal, classified with its OWN
        # factory p75/p90-anchored thresholds (±0.26/±0.39). This is the SAME object as the
        # Unified Signal plot's top row and the TATTVA CONVICTION card, so
        # hero, card and plot reconcile identically by construction (no
        # cross-object reconciliation needed), and it is the object
        # hero_study historically validated as the directional read. It is
        # never classified with calibrated thresholds (audit finding F1) —
        # the calibrated composite below is a separate evidence row.
        # `consensus_series` is the single source for the full history
        # (hero-history plot + DDM trend); the dict is its last point.
        from convergence.normalization import (
            compute_normalized_convergence, consensus_series, classify_convergence_score,
        )
        _consensus_full = consensus_series(aarambh_ts, nirnay_daily)
        _nishkarsh_norm = compute_normalized_convergence(aarambh_ts, nirnay_daily)
        if _nishkarsh_norm:
            console.section("Normalized Consensus (headline)")
            console.item("Conviction", f"{_nishkarsh_norm['value']:+.2f}")
            console.item("Signal", _nishkarsh_norm['signal'])
            console.item("  Aarambh contribution", f"{_nishkarsh_norm['aarambh_norm']:+.2f}")
            console.item("  Nirnay contribution",  f"{_nishkarsh_norm['nirnay_norm']:+.2f}")

        # ── 4e. CALIBRATED SIGNAL — the learned variant (evidence row) ──────
        # convergence_score (post apply_calibrated_weights, ±100 scale) IS the
        # exact quantity Intelligence Mode's Optuna search scores and bins
        # with _active_t — so this is the one place the learned thresholds are
        # semantically valid (audit findings F1/F2). Not the headline: it
        # feeds the hero's CALIBRATED evidence row (second opinion on the
        # consensus read) and carries the Val IC the trust chip reports. The
        # RAW factory-weight composite is no longer surfaced in the UI at all
        # — it remains the research baseline in
        # research/calibration_lift_study.py (raw-vs-calibrated ablation).
        _calibrated_score = float(convergence_df["convergence_score"].iloc[-1]) if not convergence_df.empty else 0.0
        _calibrated_signal = classify_convergence_score(_calibrated_score, _active_t)
        console.section("Calibrated Signal (evidence)")
        console.item("Score", f"{_calibrated_score:+.1f}")
        console.item("Signal", _calibrated_signal)

        console.section("Divergence Detection")
        progress_bar(progress_container, 93, "Detecting Divergences", "Cross-System Disagreement Analysis")
        events = divergence_detector.get_events()
        console.item("Total Events", len(events))
        if not events.empty:
            event_types = events['divergence_type'].value_counts()
            for etype, count in event_types.items():
                console.item(f"  {etype}", count)
        console.success("Divergence analysis complete")

        # ── Walk-Forward Validation (durability check, runs every analysis) ──
        # Re-calibrates on each expanding window and scores IC on the next
        # unseen block. Many genuine OOS grades → distinguishes a durable edge
        # from a lucky recent regime. Results power the Diagnostics tab.
        console.section("Walk-Forward Validation")
        progress_bar(progress_container, 94, "Walk-Forward Validation", "Rolling OOS IC · Re-Calibration")
        try:
            _hold_grid = _icfg.hold_horizons  # IC durability at this instrument's forecast horizons
            _wf_frame = _intel_mod._build_calibration_frame(
                convergence_df, aarambh_ts, target_col="Price", horizons=_hold_grid,
            )
            _wf_results = _intel_mod.walk_forward_ic(_wf_frame, horizons=_hold_grid)
            st.session_state["wf_results"] = _wf_results
            _wf_ics = [r["ic"] for r in _wf_results if r["ic"] == r["ic"]]  # drop NaN
            if _wf_ics:
                _wf_mean = sum(_wf_ics) / len(_wf_ics)
                _wf_pos = sum(1 for v in _wf_ics if v > 0)
                console.item("Windows", len(_wf_ics))
                console.item("Mean OOS IC", f"{_wf_mean:+.3f}")
                console.item("Positive", f"{_wf_pos}/{len(_wf_ics)}")
                console.success(f"Walk-forward: mean OOS IC {_wf_mean:+.3f} ({_wf_pos}/{len(_wf_ics)} +ve)")
            else:
                console.warning("Walk-forward produced no scorable windows")
        except Exception as _wf_e:
            st.session_state["wf_results"] = None
            console.warning(f"Walk-forward validation skipped: {_wf_e}")

        console.end_phase("CONVERGENCE")
        _conv_complete_sub = (
            f"{overlap_count} Overlap Dates · {len(events)} Divergence Events · "
            f"{'Calibrated Profile Applied' if (_intel_enabled and _final_profile is not None) else 'Factory Defaults'}"
        )
        progress_bar(progress_container, 95, "Convergence Phase Complete", _conv_complete_sub)

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 96, "Storing Results", "Session State · Cache")
        console.section("Session State")

        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
        st.session_state["aarambh_ts"] = aarambh_ts
        st.session_state["nirnay_daily"] = nirnay_daily
        st.session_state["nirnay_constituent_dfs"] = nirnay_constituent_dfs
        st.session_state["convergence_df"] = convergence_df
        st.session_state["divergence_events"] = events
        st.session_state["nishkarsh_result"] = results[-1] if results else None
        st.session_state["last_agreement"] = convergence_df["agreement_ratio"].iloc[-1] if not convergence_df.empty else 0
        # THE CALIBRATED variant (CALIBRATED evidence row + trust-chip Val IC
        # — Phase 4e). The headline itself is the normalized consensus
        # (nishkarsh_conv_normalized below + hero_series), so no separate
        # headline scalars are stored.
        st.session_state["nishkarsh_calibrated_score"] = _calibrated_score
        st.session_state["nishkarsh_calibrated_signal"] = _calibrated_signal

        # Series for the hero-history plot + TREND row.
        #   • hero_series   — the HEADLINE object's full history: the
        #     normalized consensus ([-1,+1], from consensus_series — the
        #     single source shared with the Unified Signal plot's top row)
        #   • hero_smoothed — DDM of the SAME consensus (config DDM params,
        #     tanh-bounded like every other DDM consumer) — the trend the
        #     hero's TREND row compares today's print against
        #   • calibrated_conv_series — DDM of the CALIBRATED composite (the
        #     Unified Signal plot's amber overlay); index converted to a
        #     genuine DatetimeIndex (convergence_df's index is DATE STRINGS —
        #     a string index would silently reindex to all-NaN, verification
        #     finding V1)
        if not _consensus_full.empty:
            st.session_state["hero_series"] = _consensus_full["Consensus"].rename("HeroConsensus")
            from analytics.ddm_filter import drift_diffusion_filter as _ddm
            from analytics.utils import _apply_conviction_bounds as _bound
            _cons_filt, _, _ = _ddm(
                _consensus_full["Consensus"].to_numpy() * 100.0,
                leak_rate=_icfg.ddm_leak,
                drift_scale=_icfg.ddm_drift,
                long_run_var=_icfg.ddm_lrv,
            )
            st.session_state["hero_smoothed"] = pd.Series(
                _bound(_cons_filt) / 100.0, index=_consensus_full.index,
                name="HeroConsensusSmoothed",
            )
        else:
            st.session_state["hero_series"] = None
            st.session_state["hero_smoothed"] = None
        if results:
            _ccs_index = pd.to_datetime(convergence_df.index, errors="coerce")
            st.session_state["calibrated_conv_series"] = pd.Series(
                [r.nishkarsh_conviction / 100.0 for r in results],
                index=_ccs_index, name="CalibratedConvergence",
            )
        else:
            st.session_state["calibrated_conv_series"] = None

        # THE HEADLINE: the normalized consensus dict (value + signal +
        # per-engine contributions) — single source of truth from
        # convergence/normalization.py, shared verbatim with the TATTVA
        # CONVICTION card and the Unified Signal plot's top row.
        st.session_state["nishkarsh_conv_normalized"] = _nishkarsh_norm

        # Display signal = what the UI cards show: the consensus headline,
        # then the DDM signal as a last resort.
        display_signal = (
            _nishkarsh_norm["signal"] if _nishkarsh_norm
            else (results[-1].nishkarsh_signal if results else "N/A")
        )

        console.item("Aarambh Engine", "✅ Cached")
        console.item("Nirnay Daily", f"✅ {len(nirnay_daily)} rows")
        console.item("Constituent Results", f"✅ {len(nirnay_constituent_dfs)} stocks")
        console.item("Convergence DF", f"✅ {len(convergence_df)} rows")
        console.item("Convergence Result", f"✅ {display_signal}")

        console.end_phase("FINAL ASSEMBLY")

        console.summary("RUN SUMMARY", {
            "Total Phases": "5/5 complete",
            "Aarambh Rows": len(engine.ts_data),
            "Nirnay Stocks": len(nirnay_constituent_dfs),
            "Nirnay Trading Days": len(nirnay_daily),
            "Convergence Scores": len(convergence_df),
            "Overlap Dates": overlap_count,
            "Divergence Events": len(events),
            "Status": "SUCCESS",
        })

        console.line('═', 70)
        console._write(f"  {Colors.BOLD}{Colors.GREEN}Analysis Complete{Colors.RESET}")
        console.line('═', 70)
        console._write()

        # Snapshot this config's full result into the bounded per-config cache
        # so revisiting it later (commodity switch-back, predictor toggle-back)
        # restores instantly. LRU-evict to keep memory bounded.
        _rcache = st.session_state.setdefault("results_cache", {})
        _rcache.pop(cache_key, None)
        _bundle_snapshot = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
        # Trim large baskets' per-constituent frames before they enter the LRU
        # (audit finding F19) — see _bundle_nirnay_constituent_dfs's docstring.
        _bundle_snapshot["nirnay_constituent_dfs"] = _bundle_nirnay_constituent_dfs(
            _bundle_snapshot.get("nirnay_constituent_dfs") or {}
        )
        _rcache[cache_key] = _bundle_snapshot
        while len(_rcache) > _RESULTS_CACHE_MAX:
            _rcache.pop(next(iter(_rcache)))

        progress_bar(progress_container, 100, "Analysis Complete", f"Convergence: {display_signal}")
        time.sleep(0.25)
        progress_container.empty()
        st.session_state["run_requested"] = True
        st.rerun()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()
    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))
    if "aarambh_ts" not in st.session_state:
        st.session_state["aarambh_ts"] = ts.copy()

    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    agreement = st.session_state.get("last_agreement", 0)

    # ─── Precedent base rate for the hero (co-equal second opinion) ────────
    # A 33-target non-overlapping study (hero_study.py) found the analog precedent
    # is a STRONGER directional read than the convergence signal, and adds genuine,
    # independent value — while the plot markers add nothing (they ARE the
    # convergence's own inputs). So the hero reads the precedent alongside its
    # signal: agreement raises confidence, disagreement is flagged as a divergence.
    # NOTE: the specific quoted numbers in the original study (IC +0.226 vs +0.158)
    # were measured under a since-changed analog config (the old .55/.35/.10
    # Mahalanobis/trajectory/recency blend, not the current pure-Mahalanobis 1/0/0 —
    # see analytics.analogs.ANALOG_W_*) and with a look-ahead full-sample
    # normalization the live tab avoids (causal expanding z-scores). Re-run
    # hero_study.py with the shipped config before quoting a specific number again;
    # the qualitative conclusion (precedent >= convergence, markers add nothing) is
    # the part that's load-bearing here, not the exact ICs.
    # Content-aware key (not just row count): include the latest Price so an intraday
    # refresh that updates the last bar without adding a row still recomputes.
    #
    # Computed ONCE here over the FIXED precedent term structure
    # (core.config.PRECEDENT_HORIZONS = 1/3/5/10/20/60d) and cached as the raw
    # analog list — the tab (ui/tabs/tab_precedent.py) previously called
    # find_similar_periods AGAIN with its own hold_horizons on every render,
    # re-running the expensive part (feature-frame build incl. rolling Hurst,
    # Mahalanobis distance, Theiler selection) a second time for the same
    # ts/target/mom_window (audit finding F18). summarize_forward is cheap
    # (pure aggregation), so both the hero's single-horizon read and the tab's
    # per-horizon cards derive from this one cached analog list.
    #
    # The analog STATE features use this instrument's forecast-momentum window;
    # the term structure is this instrument's precedent_horizons span.
    _plast = float(ts["Price"].iloc[-1]) if "Price" in ts.columns and len(ts) else 0.0
    _pkey = f"{active_target}|{len(ts)}|{_plast:.6g}"
    if st.session_state.get("_prec_key") != _pkey:   # recompute only when inputs change
        _prec_summary = None
        _cached_analogs: list = []
        _cached_display_hold: tuple = ()
        try:
            from analytics.analogs import find_similar_periods as _fsp, summarize_forward as _sf
            _display_hold = _icfg.precedent_horizons
            _analogs = _fsp(ts, active_target, hold_horizons=_display_hold, mom_window=_icfg.forecast_momentum,
                            maha_weight=_icfg.analog_w_maha, trajectory_weight=_icfg.analog_w_traj,
                            recency_weight=_icfg.analog_w_recv)
            _cached_analogs = _analogs
            _cached_display_hold = _display_hold
            _ps = _sf(_analogs, _display_hold) if _analogs else {}
            # Hero precedent second-opinion reads at this instrument's forecast
            # horizon (a member of its precedent_horizons); fall back to the
            # nearest available shorter horizon if it is ever absent.
            _hp = int(_icfg.forecast_horizon)
            if _hp not in _ps:
                _cands = [h for h in _display_hold if h <= _hp]
                _hp = max(_cands) if _cands else min(_display_hold)
            _row = _ps.get(int(_hp))
            if _row:
                _med = _row["median"]
                _prec_summary = {
                    "horizon": int(_hp), "median": float(_med),
                    "positive_pct": float(_row["positive_pct"]), "n": int(_row["n"]),
                    "dir": 1 if _med > 0 else -1 if _med < 0 else 0,
                }
        except Exception:
            _prec_summary = None
        st.session_state["precedent_summary"] = _prec_summary
        st.session_state["_precedent_analogs_cache"] = {
            "pkey": _pkey, "periods": _cached_analogs, "display_hold": _cached_display_hold,
        }
        st.session_state["_prec_key"] = _pkey

    # ─── Primary Signal (Above Tabs, Always Visible) ───────────────────────
    _render_primary_signal(nishkarsh_norm, agreement, signal)

    # ─── Sidebar Discovery Hint (passive — the sidebar collapse control lives
    # in Streamlit's own chrome; this is a directional pointer, not a button) ──
    st.markdown(
        """
        <div class="sidebar-hint">
            <svg class="sidebar-hint-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
            <span class="sidebar-hint-label">CONFIGURE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Timeframe Filter — with robust persistence ───────────────────────
    if 'tf_selected' not in st.session_state:
        st.session_state.tf_selected = '6M'
    # Derived from TIMEFRAME_TRADING_DAYS (core/config.py) rather than a
    # second hard-coded {3M:63, 6M:126, ...} literal — the two used to drift
    # independently with no shared source (audit finding F15).
    TIMEFRAMES = {**TIMEFRAME_TRADING_DAYS, 'ALL': None}

    tf_cols = st.columns(len(TIMEFRAMES), gap="small")
    for i, tf in enumerate(TIMEFRAMES.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", type=btn_type, width='stretch'):
                st.session_state.tf_selected = tf
                st.rerun()
    selected_tf = st.session_state.tf_selected

    # Ensure timeframe survives config changes by always applying it
    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            from pandas import DateOffset
            max_date = ts["Date"].max()
            offsets = {"3M": DateOffset(months=3), "6M": DateOffset(months=6), "1Y": DateOffset(years=1), "2Y": DateOffset(years=2)}
            cutoff = max_date - offsets.get(selected_tf, DateOffset(years=1))
            ts_filtered = ts[ts["Date"] >= cutoff]
        else:
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days):]
    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ─── Tabs with Error Boundaries ─────────────────────────────────────────
    # Streamlit renders every tab's content on each script run (there is no
    # built-in lazy-loading of inactive tabs) — the CSS just hides the
    # inactive panels. A `rendered_tabs` session-state set was previously
    # written here on every render but never read anywhere, under a comment
    # claiming lazy loading that isn't actually happening (audit finding C5).
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "CONVERGENCE", "AARAMBH", "NIRNAY", "PRECEDENT", "DIAGNOSTICS", "DATA",
    ])

    # Error boundary wrapper
    def _safe_render(name, render_fn):
        """Render a tab with graceful error handling."""
        try:
            render_fn()
        except Exception as e:
            st.markdown(
                f'<div style="background:rgba(251,113,133,0.05);border:1px solid rgba(251,113,133,0.2);'
                f'border-radius:var(--r-md);padding:var(--sp-6);margin:var(--sp-4) 0;">'
                f'<div style="font-family:var(--display);font-size:0.72rem;font-weight:700;color:var(--rose);'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:var(--sp-3);">'
                f'Error in {name}</div>'
                f'<div style="font-family:var(--data);font-size:0.8rem;color:var(--ink-secondary);line-height:1.6;">'
                f'{str(e)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab1:
        _safe_render("Convergence", lambda: render_convergence_tab(ts_filtered))
    with tab2:
        _safe_render("Aarambh", lambda: render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target))
    with tab3:
        _safe_render("Nirnay", lambda: render_nirnay_tab(selected_tf=selected_tf))
    # Reuse the analog list already computed above (Precedent base-rate for the
    # hero) instead of having the tab call find_similar_periods a second time
    # for the same (ts, target, mom_window) — audit finding F18. Guarded on
    # the pkey matching THIS render's ts/target/horizon; a mismatch (shouldn't
    # happen since the precompute above always runs first) falls back to None,
    # and the tab recomputes itself exactly as before.
    _prec_cache = st.session_state.get("_precedent_analogs_cache")
    _cached_periods = (
        _prec_cache["periods"] if _prec_cache and _prec_cache.get("pkey") == _pkey else None
    )
    with tab4:
        # Precedent term structure + momentum/horizon come from this instrument's
        # own config (precedent_horizons / forecast_momentum / forecast_horizon).
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.forecast_momentum, _icfg.forecast_horizon,
            precomputed_periods=_cached_periods))
    with tab5:
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats))
    with tab6:
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    _render_footer()


if __name__ == "__main__":
    main()
