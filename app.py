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

import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import html
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
# site (engines/mula/base.py._compute_breadth_metrics) instead. FutureWarning
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
    build_hero_verdict,
    render_hero_card,
    render_control_hint,
    render_ticker,
    render_section_header,
    render_kpi_strip,
    panel,
    render_nav_brand,
    render_warning_box,
    render_top_bar,
    render_notice_rail,
    render_rail_readout,
)
from ui.tabs.tab_mula import render_mula_tab
from ui.tabs.tab_swayam import render_swayam_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_constituent_ohlcv, fetch_macro_live, fetch_commodity_dataset, fetch_stock_target_series
from data.calendars import trading_days_behind, is_session, session_mask, resolve_exchange
from data.universe import resolve_stock_symbol

# ── Engines ──────────────────────────────────────────────────────────────────
from engines.mula import FairValueEngine
from engines.mula.blocks import block_membership
from engines.swayam import aggregate_views
from engines.swayam.kernel import view_skill_weights

# ── Valuation engine (MŪLA) ─────────────────────────────────────────────────
# MulaEngine SUBCLASSES FairValueEngine, so every isinstance check and cache
# path downstream works unchanged. Engine 2 is Swayam, unchanged. There is no
# engine selector: this import guard exists only so that a broken or missing
# engines.mula package degrades silently to the pipeline core (no ECM layer)
# instead of taking the app down.
try:
    from engines.mula import MulaEngine
    _HAS_V2_ENGINES = True
except Exception:  # pragma: no cover — degraded environments only
    MulaEngine = None
    _HAS_V2_ENGINES = False

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.conviction_model import UnifiedConvictionModel
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import LOOKBACK_WINDOWS, MIN_DATA_POINTS, STALENESS_DAYS, SESSION_FRESH_FLOOR, TARGET_EXCLUDED_PREDICTORS, ALL_TARGETS, TARGET_CATEGORIES, is_stock_target, FORECAST_HORIZON, RAW_YIELD_PREDICTORS, DIV_LOOKBACK, TIMEFRAME_TRADING_DAYS, swayam_macro_columns, FREEFORM_STOCK_CATEGORIES, register_stock_target, get_instrument_config
from engines.swayam import build_swayam_frames, effective_member_count, default_swayam_members
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
    "engine", "fvo_ts", "swayam_daily", "swayam_view_dfs",
    "convergence_df", "divergence_events", "nishkarsh_result", "last_agreement",
    "nishkarsh_conv_normalized", "wf_results",
    "intelligence_active_weights", "intelligence_active_thresholds",
    "intelligence_active_profile",
    # The consensus's full history (Convergence-tab marker tiers) and the
    # weighted composite (hero WEIGHTED row / amber overlay) — must travel
    # with the bundle so a cached target switch-back doesn't leave the
    # PREVIOUS target's state in session.
    "hero_series",
    "nishkarsh_calibrated_score", "nishkarsh_calibrated_signal",
    "calibrated_conv_series",
    # Per-target UI metadata that must travel with the result bundle —
    # otherwise a cached target switch-back leaves the PREVIOUS target's
    # value in session state (e.g. the Swayam tab showing a stale
    # "basket source: snapshot" hint for a target resolved live, or the
    # Convergence tab's "breadth carried forward" notice firing/missing
    # based on the WRONG target's basket-freshness timestamp).
    "swayam_native_last", "swayam_n_eff",
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
# swayam_view_dfs carries ~200 columns per constituent (the full
# kernel output); only the ~9 the Swayam tab's drill-down actually
# displays (_SWAYAM_DRILLDOWN_COLS) are needed once the result is just sitting
# in the switch-back cache. A small commodity basket (~15-20 names) is cheap
# either way and kept at full width so nothing else that might read the wider
# frame in-session breaks; an uncapped large index (S&P 500 ~500 names) is
# where the ~200-column full width, multiplied across up to 6 LRU entries,
# actually matters.
_CONSTITUENT_TRIM_THRESHOLD = 60
_SWAYAM_DRILLDOWN_COLS = (
    "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
    "Regime", "Vol_Regime", "Change_Point", "Confidence",
)


def _bundle_swayam_view_dfs(dfs: dict) -> dict:
    """Trim swayam_view_dfs to the Swayam tab's drill-down columns
    before it enters the per-config results_cache LRU, for baskets at/above
    _CONSTITUENT_TRIM_THRESHOLD names. Only affects the SNAPSHOT stored in
    results_cache — the live session_state copy the active render reads
    (and engines.swayam.aggregate_views, which needs the
    full width and runs before this snapshot is taken) is never touched.
    """
    if not dfs or len(dfs) < _CONSTITUENT_TRIM_THRESHOLD:
        return dfs
    trimmed = {}
    for sym, df in dfs.items():
        cols = [c for c in _SWAYAM_DRILLDOWN_COLS if c in df.columns]
        trimmed[sym] = df[cols] if cols else df.iloc[:, :0]
    return trimmed


def _ensure_stock_target_column(df: pd.DataFrame, active_target: str) -> pd.DataFrame:
    """Inject a target's Close into the model matrix when the batch has not.

    The macro batch fetches exactly the columns in GLOBAL_MACRO_MAP +
    MACRO_SYMBOLS_YF (+ the index levels merged into it). A target whose
    ticker is in one of those maps therefore arrives for free — which is true
    of every commodity and FX target that predates this, because each is also
    a PREDICTOR under the same name (Gold, Copper, Dollar Index …). A target
    that is not in those maps has no column at all, and the guard downstream
    reports it as a failed source fetch.

    Catalogue targets outside the macro maps — Aluminium, Zinc, the crypto
    bank — are handled UPSTREAM, by ``data.fetcher._fetch_catalogue_targets``,
    so their column is already present by the time this runs. That is
    deliberate: the shared matrix is read by the app, by the research
    preflight and by each tuning study independently, and injecting per
    consumer is how the suite ended up seeing "36/45 targets" while the app
    saw all 45. One injection, upstream, for everyone.

    This helper therefore stays narrow, covering only the free-form STOCK
    targets that are deliberately kept OUT of the batch for cache coherence
    (a per-target ticker set would break the batch's (start, end) key). If a
    catalogue target's column is missing here, that is a real fetch failure
    and the guard downstream should say so — not be papered over by a second
    silent fetch.

    Aligned to the matrix's DATE spine, ffilled, leading NaNs left for the
    per-target dropna downstream. Mutates st.session_state['data'] too, so a
    target switch or a cached rerun sees the column without re-fetching.
    """
    if active_target in df.columns or not is_stock_target(active_target):
        return df
    ticker = ALL_TARGETS.get(active_target)
    if not ticker:
        return df
    end = pd.Timestamp.today()
    s = fetch_stock_target_series(ticker, end - pd.DateOffset(days=365 * 9), end)
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

def _render_header(frame=None) -> None:
    """Cold-start masthead (and, if a panel is passed, the tape under it).

    Only the landing page uses this. Once a session is loaded the masthead's
    job — say what this is — is done, and the command bar takes over: it
    carries the same mark plus the thing the masthead cannot, namely which
    instrument you are looking at and what it is worth. Two persistent
    headers stacked on every page was one more than the screen could justify.
    """
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Cross-Asset Fair Value · Self-Referential Breadth · Unified Convergence",
    )
    if frame is not None:
        render_ticker(frame)


#: The three systems, as the cold-start screen describes them. Data, not
#: markup — the landing page renders them through one template, so the three
#: panels cannot drift apart in structure the way three hand-written HTML
#: blocks did (they already had: two said "Ensemble/Signal", one said
#: "Fusion", and the label column was a <span> in one and a <div> in another).
_SYSTEM_PANELS = (
    ("mula", "System 01", "Mūla", "Top-down valuation",
     "Prices the target against the whole traded macro cross-section with a dynamic "
     "cointegrating regression on log price. It answers one question: is this instrument "
     "cheap or dear relative to everything it moves with?",
     (("Estimator", "Recursive discounted DLM"),
      ("Equilibrium", "MP factors + asset blocks"),
      ("Reversion", "ECM κ̂ · predictive likelihood"))),
    ("swayam", "System 02", "SWAYAM", "Bottom-up breadth",
     "Reads the instrument's own internals through a self-referential bank of views "
     "spanning timescale, information set and mechanism, then aggregates them by "
     "realised skill rather than by a fixed grid.",
     (("Signal", "MSF + MMR oscillator"),
      ("Breadth", "Oversold / overbought share"),
      ("Regime", "HMM \u00b7 GARCH \u00b7 CUSUM"))),
    ("convergence", "System 03", "CONVERGENCE", "Adaptive fusion",
     "Scores the two systems against each other across four dimensions \u2014 direction, "
     "breadth, magnitude, regime \u2014 with weights learned forward from resolved "
     "outcomes, then filters the composite through a leaky DDM.",
     (("Fusion", "Mūla \u00d7 Swayam"),
      ("Weights", "Learned online, causal"),
      ("Filter", "Leaky drift-diffusion"))),
)


def _render_landing_page() -> None:
    """Cold start — a description of the product, built from the product's own parts.

    Every block here now uses the same components the analysis pages use: a
    section header for each division, `render_kpi_strip` for the coverage
    numbers, and `panel()` for each system. The previous version was built
    from four compositions that existed nowhere else in the app — a bespoke
    `.fact-row` of oversized numerals where every other count in the product
    is a metric card, `.system-card` with a 2px coloured top bar where every
    other container is a 1px hairline panel, a floating `.outcome-grid` with
    no container at all, and no section headers, which made the landing page
    the only page not on the section-rhythm contract. It read as a different
    product's marketing page bolted to the front of this one.

    The claim still leads, because a reader who has not run anything needs to
    know what the thing IS before they are shown what it covers.
    """
    from core.config import TARGET_CATEGORIES, ALL_TARGETS

    _n_cat, _n_tgt = len(TARGET_CATEGORIES), len(ALL_TARGETS)

    # ── The proposition ───────────────────────────────────────────────────
    st.markdown(
        """<div class="lede">
  <div class="lede-claim">Two independent systems price the same instrument,
    and a third measures how much their agreement has actually been worth.</div>
  <div class="lede-cta">Pick an asset class and a target in the rail, then
    <strong>Run Analysis</strong>.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Coverage — the app's own KPI grammar, not a bespoke number row ─────
    render_section_header("Coverage", icon="layers")
    render_kpi_strip(
        [
            {"label": "Asset Classes", "value": str(_n_cat),
             "subtext": "Commodities, FX, India and US indices, sector ETFs, "
                        "and any listed stock by symbol"},
            {"label": "Catalogue Targets", "value": str(_n_tgt),
             "subtext": "Each with its own engine configuration — horizon, filter, "
                        "breadth tier, precedent term structure"},
            {"label": "Daily History Per Run", "value": "~9y",
             "subtext": "Walk-forward throughout; every score is out-of-sample with "
                        "respect to everything after it"},
        ],
        max_cols=3,
        key="landing-coverage",
    )

    # ── The three systems, as panels ──────────────────────────────────────
    render_section_header("Systems", icon="cpu")
    cols = st.columns(3, gap="small")
    for col, (cls, eyebrow, name, kicker, body, specs) in zip(cols, _SYSTEM_PANELS):
        with col:
            with panel(f"landing-{cls}", name, context=kicker):
                st.markdown(
                    f'<div class="panel-copy">{body}</div>'
                    '<div class="panel-specs">'
                    + "".join(
                        f'<div class="lookback-row"><span class="lbl">{k}</span>'
                        f'<span class="val">{v}</span></div>'
                        for k, v in specs
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ── What a run returns ────────────────────────────────────────────────
    render_section_header("What a run returns", icon="target")
    _out = (
        ("A directional claim", "One verdict, with the six gates that condition it and "
                                "the single binding constraint named."),
        ("A measured edge", "Walk-forward IC across expanding windows — the honest "
                            "answer to whether this has paid before."),
        ("An independent check", "A non-parametric base rate from the most similar "
                                 "historical states, which does not depend on the models."),
        ("The evidence", "Every series, weight and diagnostic behind the verdict, "
                         "exportable."),
    )
    # ONE markdown block, not four panels. These four cards are static text, so
    # they gain nothing from a Streamlit container and lose something real to
    # it: on 1.52 the anonymous row Streamlit wraps markdown in sizes to 31px
    # around 47px of copy and will not grow, so each panel came out ~15px
    # short and clipped its own last line at `overflow: hidden` — and clipped
    # by a different amount per card, which is why the four were uneven. A
    # single grid of plain divs has no wrapper to collapse, and CSS grid gives
    # the uniform two-up layout the panels were only approximating.
    st.markdown(
        '<div class="outcome-grid">'
        + "".join(
            f'<div class="outcome"><div class="o-t">{html.escape(t)}</div>'
            f'<div class="o-d">{html.escape(d)}</div></div>'
            for t, d in _out
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _compute_hero_verdict(nishkarsh_norm, agreement, fvo_signal) -> dict:
    """Gather session-state inputs and build the hero conviction-chain verdict.

    All interpretation lives in ``ui.components.build_hero_verdict`` (a pure,
    unit-testable function — see research/test_hero_verdict.py); this wrapper
    only gathers session-state inputs and returns the verdict. Rendering
    (``render_hero_card``) is the caller's job — the Overview page renders
    the full card; every other page can read the same verdict object for its
    top-bar status chip / KPI strip without disagreeing with what Overview
    shows, since both read the identical dict computed once per rerun.

    The card is a CONVICTION CHAIN: direction from Mūla, then gates whose
    product is the conviction, with the smallest gate named as the binding
    constraint. Every engine contributes exactly one gate — Mūla (mispricing
    and reversion), Swayam (breadth), Convergence (agreement + normalized
    consensus), the walk-forward read (edge), and Precedent (base rate).
    """
    wf          = st.session_state.get("wf_results")                   # list[dict] | None
    div_events  = st.session_state.get("divergence_events")            # DataFrame | None
    prec        = st.session_state.get("precedent_summary")            # dict | None
    # MŪLA read (None on the legacy path — the ECM gate it feeds is
    # conditional and the chain collapses to its incumbent shape).
    mula_read   = st.session_state.get("mula_summary") or None

    # DEGENERATE-CONVERGENCE GATE: `nishkarsh_norm` is None exactly when the
    # Mūla/Swayam alignment found no overlap. The verdict handles that case
    # directly now — `swayam_breadth=None` leaves the corroboration gate at a
    # neutral 0.5 and says "no breadth read" on the card, rather than the
    # signal quietly becoming half-weight Mūla wearing a convergence label.
    # Divergences are silenced for the same reason: the detector would be
    # comparing Mūla against breadth that does not exist.
    if nishkarsh_norm is None:
        div_events = None

    # Active instrument's forecast horizon - for interpretation copy only.
    try:
        FWD_HORIZON = get_instrument_config(st.session_state.get("active_target", "")).forecast_horizon
    except KeyError:
        FWD_HORIZON = FORECAST_HORIZON

    wf_ics = [r["ic"] for r in wf if isinstance(r, dict) and r.get("ic") == r.get("ic")] if wf else []
    # The edge number IS the walk-forward mean now. There is no calibration to
    # hold out from: weights are learned forward, so every window's IC is
    # already out-of-sample with respect to everything after it, and a separate
    # "Val IC" would just be one more window of the same thing.
    oos_ic = float(np.mean(wf_ics)) if wf_ics else None
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
    if (div_events is not None
            and hasattr(div_events, "__len__") and len(div_events)):
        try:
            _idx = div_events.index
            if isinstance(_idx, pd.DatetimeIndex):
                _div_dates = _idx
            elif _idx.inferred_type in ("string", "date", "datetime", "datetime64"):
                _div_dates = pd.DatetimeIndex(pd.to_datetime(_idx, errors="coerce"))
            else:
                # A POSITIONAL index must not be parsed as dates. `pd.to_datetime`
                # accepts bare integers and reads them as epoch NANOSECONDS, so a
                # RangeIndex silently becomes 1970-01-01+0ns … +2ns: the cutoff
                # lands in 1970, every row clears it, and n_div reports the whole
                # history — precisely the permanent meaningless alarm the window
                # above exists to prevent. It also trips NumPy 2.x's
                # generic-unit timedelta deprecation ("implicit conversion of
                # bare integers"), which is scheduled to become an error.
                _div_dates = pd.DatetimeIndex([])
            _valid_dates = _div_dates.dropna()
            if len(_valid_dates):
                _cutoff = pd.Timestamp(_valid_dates.max()) - pd.DateOffset(days=int(DIV_LOOKBACK * 1.5))
                n_div = int((_div_dates >= _cutoff).sum())
            else:
                # No usable dates — fall back to the raw count rather than a
                # window computed from nothing.
                n_div = int(len(div_events))
        except Exception:
            n_div = int(len(div_events))

    # Swayam's current breadth — the corroboration gate. Read from the daily
    # frame's last row rather than a session scalar so it is the same object
    # the Swayam tab shows.
    _sw = st.session_state.get("swayam_daily")
    swayam_breadth = None
    if _sw is not None and not _sw.empty:
        _last = _sw.iloc[-1]
        swayam_breadth = {
            "oversold_pct": float(_last.get("Oversold_Pct", 50.0)),
            "overbought_pct": float(_last.get("Overbought_Pct", 50.0)),
        }

    # The convergence layer's own output: how far the two engines concur
    # (dimension-weighted by learned skill) and where the normalized consensus
    # — the series the Convergence tab plots — currently points.
    _cdf = st.session_state.get("convergence_df")
    convergence_read = None
    if _cdf is not None and not _cdf.empty:
        convergence_read = {
            "agreement_ratio": float(_cdf["agreement_ratio"].iloc[-1])
            if "agreement_ratio" in _cdf.columns else 0.5,
            "consensus": (float(nishkarsh_norm["value"])
                          if nishkarsh_norm and nishkarsh_norm.get("value") is not None
                          else None),
        }

    verdict = build_hero_verdict(
        fvo_signal=fvo_signal,
        swayam_breadth=swayam_breadth,
        convergence=convergence_read,
        wf_ic=oos_ic,
        wf_pos=wf_pos,
        wf_n=wf_n,
        precedent=prec,
        n_divergences=n_div,
        horizon_days=FWD_HORIZON,
        div_window=DIV_LOOKBACK,
        mula=mula_read,
    )
    return verdict


def _render_model_passport_sidebar(current_universe: str, current_index: str | None = None) -> None:
    """Sidebar Passport — what the model is doing right now.

    This used to be a PROFILE manager: it showed which calibrated profile was
    loaded, its train/val IC and timestamp, warned when the profile had been fit
    on a different universe, and offered Import / Export / Reset. All of that
    existed because calibration produced a persisted artefact that could be
    stale, mismatched, or shared — and because the output depended on which
    artefact happened to be loaded.

    Nothing is persisted now. Weights are learned forward from the data in
    front of the model, every run, so there is no profile to import, no
    mismatch to warn about, and nothing to reset to. What remains worth showing
    is the state the run actually reached.
    """
    st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)

    w = st.session_state.get("intelligence_active_weights") or {}
    wf = st.session_state.get("wf_results") or []

    if not w:
        render_control_hint("Run an analysis to populate.")
        return

    # Dimension weights, heaviest first, then the walk-forward read — one
    # key/value block rather than two differently-styled fragments, so the
    # rail's status area has a single visual grammar.
    rows = [(k, f"{v:.3f}", "accent") for k, v in sorted(w.items(), key=lambda kv: -kv[1])]
    if wf:
        _ics = [r["ic"] for r in wf if np.isfinite(r.get("ic", float("nan")))]
        if _ics:
            _mean = float(np.mean(_ics))
            _pos = sum(1 for v in _ics if v > 0)
            rows.append(("WF IC", f"{_mean:+.3f}", "long" if _mean > 0 else "short"))
            rows.append(("Windows +", f"{_pos}/{len(_ics)}", ""))
    render_rail_readout(rows)


#: The DURABLE record of the appearance choice — a plain session key, never a
#: widget key.
#:
#: This distinction is the whole fix for the theme flipping back on its own.
#: Streamlit garbage-collects the state of any widget that was NOT instantiated
#: during a run. The appearance control lives at the bottom of the rail, so
#: every run that returns or reruns before reaching it — clicking Run Analysis
#: (which calls st.rerun() from inside the button handler), switching target,
#: Reset, Refresh — discarded `theme_mode` entirely. The next run then found no
#: value and fell back to Slate. That is why Paper survived idle reruns but
#: died on exactly the actions that matter, and why it looked "entirely buggy"
#: rather than simply broken.
#:
#: A plain key is never collected, so it survives every one of those paths.
_THEME_CHOICE = "theme_choice"


#: The two appearances. Both are reading surfaces — Paper is the light one you
#: read a result on and print from, Slate the dark one you work on.
#:
#: PAPER LEADS, and the order is the default: `theme_choice()` falls back to
#: APPEARANCES[0] for any unset or unrecognised value, so first-in-tuple IS
#: first-run. Kept as one fact rather than a separate DEFAULT_ constant, so the
#: toggle's left-to-right order and the default can never disagree.
APPEARANCES = ("Paper", "Slate")


def theme_choice() -> str:
    """The appearance the user last chose, always one of ``APPEARANCES``.

    A value that is not in the list is treated as unset. That matters across a
    rename: a session opened before this list changed still holds the old
    string in the durable key, and handing an unknown option to the segmented
    control as its default is an error rather than a fallback.
    """
    choice = st.session_state.get(_THEME_CHOICE)
    return choice if choice in APPEARANCES else APPEARANCES[0]


def _render_appearance_control() -> None:
    """The theme switch — LAST control in the rail, deliberately.

    It was previously the first control under the brand mark, which gave the
    least consequential switch in the application the most valuable position
    in it. Slate (dark) is the working theme; Paper (light) is for reading
    a result and for print.

    Called from exactly one of the two rail passes per rerun — the cold-start
    branch returns before the second pass exists, so the key is instantiated
    once either way.
    """
    _box = st.container(key="appearance")
    with _box:
        _render_appearance_body()


def _render_appearance_body() -> None:
    """The switch itself. Split out so the keyed container above can be
    pinned to the foot of the rail by CSS."""
    st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
    _mode = st.segmented_control(
        "Appearance", list(APPEARANCES), key="theme_mode",
        default=theme_choice(), label_visibility="collapsed",
        help="Slate — dark, for working. Paper — light, for reading and print.",
    )
    # Mirror the widget into the DURABLE key, and rerun so the stylesheet at
    # the top of main() is re-injected with the new value. Without the rerun
    # the change would land half-way down the page and the run would render
    # as a mix of both themes.
    if _mode is not None and _mode != theme_choice():
        st.session_state[_THEME_CHOICE] = _mode
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
        page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzNENkZFOCIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzNENkZFOCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
        layout="wide", initial_sidebar_state="expanded",
    )
    # ─── Resolve the theme BEFORE anything is styled ──────────────────────
    # This must read the appearance control's WIDGET key, not the derived
    # `theme` key, and it must run here — first.
    #
    # The bug it fixes: `theme` is written by _render_appearance_control(),
    # which runs deep in the sidebar, i.e. AFTER this line. On the rerun that
    # followed a click, inject_css() therefore still saw the PREVIOUS theme
    # while every chart — which resolves its palette at render time, further
    # down the script — already saw the new one. The result was a page whose
    # chrome and whose plots disagreed about which theme was active: exactly
    # "some elements show up, some do not". The theme was always one rerun
    # behind, and within that rerun it was applied inconsistently.
    #
    # Read the DURABLE choice, not the widget key: the widget's state is
    # discarded by Streamlit on any run that does not reach it (see
    # _THEME_CHOICE). Deriving `theme` here, first, makes the whole script —
    # CSS, charts, tables, iframes — agree on one value for the whole run.
    st.session_state["theme"] = "light" if theme_choice() == "Paper" else "dark"
    inject_css(theme=st.session_state["theme"])

    # Replay dynamic stock-target registration on every rerun. register_stock_target
    # mutates module-level core.config dicts (ALL_TARGETS etc.) which survive
    # Streamlit reruns WITHIN a process but are never persisted — only
    # st.session_state survives a rerun as the durable record, so a freeform
    # symbol resolved earlier this session (e.g. "RELIANCE (NSE)") must be
    # re-registered before anything below resolves active_target against
    # ALL_TARGETS/STOCK_TARGET_MARKETS. Idempotent — safe every rerun.
    for _dname, _dmeta in st.session_state.get("dynamic_stock_targets", {}).items():
        register_stock_target(_dname, _dmeta["ticker"], _dmeta["market"])

    # Single main-area progress slot, created up front (outside the sidebar) so the
    # SAME themed progress bar drives everything from the moment "Run Analysis" is
    # clicked — the fetch, the data-prep spine, the engines, convergence — instead of
    # a sidebar spinner that then hands off to a separate bar with a gap between them.
    # Empty (invisible) until the first progress_bar() call; cleared when a run ends.
    progress_container = st.empty()

    # ─── The control rail ──────────────────────────────────────────────────
    # Everything GLOBAL lives here — which instrument, what to do with the
    # session, how the model is behaving, how the app looks. Everything LOCAL
    # to a page (the chart window) lives in that page's toolbar strip. A
    # control's position is the only reliable statement of its scope, so the
    # two are never mixed.
    #
    # Rail order is by frequency of use: Instrument (every visit) → Session
    # (occasionally) → Model (read-only status) → Appearance (almost never).
    # The theme switch used to be first, directly under the brand.
    #
    # st.navigation (called once the pipeline below has run) pins its page-nav
    # rail to the TOP of the sidebar by design — that's Streamlit's own
    # behavior, not call order — so this content renders below it.
    # ──────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # The mark, always at the very top of the rail. Streamlit pins its
        # page-nav to the top of the sidebar and nothing rendered from Python
        # can precede it in the DOM — so the brand is emitted here (in the
        # pass that always runs, cold start included) and lifted above the nav
        # by CSS: `.nav-brand` is absolutely positioned against the sidebar
        # content box, which reserves room for it with a padding-top. That is
        # why it no longer sits wedged between the nav and the controls.
        render_nav_brand()
        st.markdown('<div class="sidebar-title">Instrument</div>', unsafe_allow_html=True)

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

        # Widget labels are the real <label> elements now, not markdown
        # pretending to be one: the rail's group headers (INSTRUMENT /
        # SESSION / MODEL) name the SECTION, and each control names itself.
        # Screen readers get a proper accessible name out of it for free —
        # a collapsed label plus a floating div above it gives them nothing.
        sel_cat = st.selectbox("Asset Class", _categories, key="target_category")

        if sel_cat in FREEFORM_STOCK_CATEGORIES:
            # India Stocks / US Stocks: no constituent basket to browse — enter
            # a symbol directly. The asset class supplies the suffix policy
            # (data.universe.resolve_stock_symbol): India tries SYMBOL.NS
            # first, then SYMBOL.BO; US uses the bare symbol.
            _market = FREEFORM_STOCK_CATEGORIES[sel_cat]
            _raw_symbol = st.text_input(
                "Symbol", key=f"stock_symbol_{_market}",
                placeholder="RELIANCE" if _market == "india" else "AAPL",
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
                render_control_hint(".NS then .BO" if _market == "india" else "US listing")
        else:
            cat_targets = TARGET_CATEGORIES.get(sel_cat, all_names)

            # Keep the target selection valid for the chosen category.
            if st.session_state.get("target_select") not in cat_targets:
                st.session_state["target_select"] = (
                    prev_commodity if prev_commodity in cat_targets else cat_targets[0]
                )
            selected_commodity = st.selectbox("Target", cat_targets, key="target_select")
        # (Two wrapped prose hints used to sit here — "Swayam · self-referential
        # view bank (own OHLCV)" and, further down, "231 macro instruments ·
        # full cross-section (1 excluded as self-replicating)". Between them
        # they put five lines of grey sentence fragments directly under the
        # control they described, which is what made the rail read as texty.
        # The same facts are now key/value rows in the Source readout below —
        # scannable, aligned, and a third of the height.)

        # (No engine selector: Mūla IS the valuation engine and Swayam is the
        # breadth engine — there is nothing to choose between. The import
        # guard above is the only fallback, and it is silent.)

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
            # is target-agnostic — the chosen commodity only selects Mūla's
            # target column and Swayam's basket.
            if st.button("Run Analysis", type="primary", width="stretch"):
                # No spinner — drive the main-area progress bar from the very first
                # click. The fetch is one blocking call, so we show the stage before it
                # (3%) and after it (15%); the analysis picks the bar up from there on
                # the rerun, so the experience reads as one continuous progress bar.
                progress_bar(progress_container, 3, "Fetching Market Data",
                             "yfinance · global macro universe · ~9y daily history")
                _end = pd.Timestamp.today()
                # Walk-forward needs MIN_DATA_POINTS (1500) daily observations.
                # ~9 years of calendar history clears that with headroom.
                _start = _end - pd.DateOffset(days=365 * 9)
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
            # fetched universe (no re-fetch; only the Swayam basket re-pulls).
            if selected_commodity != st.session_state.get("active_target"):
                if st.button(f"Switch → {selected_commodity}", type="primary",
                             width="stretch"):
                    st.session_state["selected_commodity"] = selected_commodity
                    st.session_state["active_target"] = selected_commodity
                    st.session_state["nishkarsh_index"] = selected_commodity
                    st.session_state.pop("active_features", None)  # re-default predictors for new target
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

    # ─── Landing page if no data loaded ──────────────────────────────────
    if df is None:
        with st.sidebar:
            _render_appearance_control()
        _render_header()
        # A session that HAD a run and no longer has its data is a different
        # state from a cold start, and it must not be shown as one. It happens
        # when the server drops session_state under memory pressure — routine
        # on Streamlit Community Cloud, where the results cache holds up to six
        # full pipeline results. The symptom was an app that "went back to the
        # landing page with the rail still looking loaded, and clicking did
        # nothing": the rail read `run_analysis` and drew itself as live, while
        # the page had no frame to render and every control pointed at state
        # that was gone. Say so, and offer the one action that fixes it.
        if st.session_state.get("run_analysis") and "data" not in st.session_state:
            render_warning_box(
                title="Session data expired",
                content=("This session had a completed run, but its fetched data is no "
                         "longer in memory — the server dropped it, which happens on "
                         "hosted deployments when memory is reclaimed. Nothing is lost "
                         "except the cached frames. Run the analysis again to rebuild "
                         "them."),
            )
            # Drop the stale flag so the rail stops advertising a live session.
            for _k in ("run_analysis", "engine_cache", "results_cache"):
                st.session_state.pop(_k, None)
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
        # Target is chosen once in the rail's Instrument group above; resolve
        # it here for predictor configuration.
        commodity_options = [c for c in ALL_TARGETS if c in numeric_cols] or numeric_cols
        target_col = st.session_state.get("active_target", commodity_options[0])
        if target_col not in numeric_cols:
            target_col = commodity_options[0]

        # Date column is always the dataset's DATE column — auto-detected.
        date_candidates = [c for c in all_cols if "date" in c.lower()]
        date_col = date_candidates[0] if date_candidates else "None"

        # (A read-only "Target: <name>" chip used to render here. It restated
        # the selector directly above it and the command bar's own instrument
        # block — three copies of one string, none of which could disagree.)

        # The valuation panel is the WHOLE macro cross-section, minus this
        # target's self-replicating near-duplicates (e.g. GLTR for a precious
        # metal, which would let the regression explain gold with gold). There
        # is no predictor picker any more: the Mūla engine prices the target
        # against the traded opportunity set, and hand-deselecting instruments
        # from that set does not make it a better opportunity set — it makes it
        # a smaller one with an undocumented reason. The engine already handles
        # a wide panel on its own terms (Marchenko-Pastur decides how many
        # factors are real; an instrument that never prints is never admitted).
        _excluded = set(TARGET_EXCLUDED_PREDICTORS.get(target_col, []))
        available = [c for c in numeric_cols if c != target_col and c not in _excluded]
        st.session_state["active_features"] = tuple(available)
        st.session_state["active_date_col"] = date_col


        # ── Model Passport ─────────────────────────────────────────────
        # Surfaces the learned dimension weights + walk-forward read. (Each
        # target used to key its own persisted profile here — see
        # _intel_index below).
        _current_universe = st.session_state.get("active_target") or st.session_state.get("selected_commodity", "Gold")
        _current_index = st.session_state.get("nishkarsh_index", _current_universe)
        _render_model_passport_sidebar(_current_universe, _current_index)

        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
            # One per row, each stretched to the rail width — the same shape as
            # the selectboxes above them. Side by side they were half-width
            # against full-width controls, and "Refresh" broke mid-word.
            _do_reset = st.button("Reset", type="secondary", width="stretch",
                                  help="Re-run both engines on the data already in "
                                       "session — no network fetch. Fast.")
            _do_refresh = st.button("Refresh", type="secondary", width="stretch",
                                    help="Force-fetch the live universe, then recompute. "
                                         "Slower; use when the data is stale or partial.")
            if _do_reset:
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("fvo_engine", None)
                st.session_state.pop("fvo_fit_key", None)
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
            if _do_refresh:
                from data.cache import begin_force_refresh
                begin_force_refresh()   # next fetches bypass TTL; disk snapshot kept
                # Same main-area progress bar as Run Analysis (no spinner) — the recompute
                # on rerun picks it up from ~15%, so refresh reads as one continuous bar.
                progress_bar(progress_container, 3, "Re-fetching Live Market Data",
                             "yfinance · full universe · bypassing cache · ~30–60s")
                _rend = pd.Timestamp.today()
                _rdf, _rerr = fetch_commodity_dataset(_rend - pd.DateOffset(days=365 * 9), _rend)
                if _rdf is not None:
                    progress_bar(progress_container, 15, "Live Data Refreshed",
                                 f"{_rdf.shape[1]} series × {_rdf.shape[0]} rows · recomputing…")
                    st.session_state["data"] = _rdf   # keep run_analysis → stay in results
                else:
                    progress_container.empty()
                for _k in ("engine", "engine_cache", "fvo_engine", "fvo_fit_key",
                           "wf_results", "results_cache", "nishkarsh_result",
                           "precedent_summary", "_prec_key", "_precedent_analogs_cache", "conv_norm_params",
                           # Horizon-independent Swayam cache (audit finding F17) —
                           # must be dropped on a live re-fetch too, else Refresh
                           # Data re-pulls the Mūla macro universe live but
                           # silently keeps serving the PRE-refresh Swayam
                           # basket/constituent analysis.
                           "_swayam_fetch_cache", "_swayam_analysis_cache"):
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

        _render_appearance_control()

    # ─── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    # Per-instrument config — every engine knob (Swayam, Mūla forecast,
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

    # ─── Data freshness notices ─────────────────────────────────────────────
    # QUEUED, not rendered. These used to paint straight onto the top of the
    # page, above the command bar — so on a lagging source the first thing on
    # screen was three stacked apology boxes and the instrument was pushed
    # below the fold, on precisely the days the numbers most needed scrutiny.
    # They are collected here and rendered by the page shell (render_notice_
    # rail) directly BENEATH the chrome they qualify. Same notices, same
    # wording, same triggers — different place, a third of the height.
    _notices: list[dict] = []

    def _notice(kind: str, title: str, body: str) -> None:
        _notices.append({"kind": kind, "title": title, "body": body})

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
                    _notice(
                        "warning", "Latest data unavailable",
                        (f"Newest data is {ds} — {behind} trading days behind. The price source "
                         f"(yfinance) hasn't published more recent data, so every signal below "
                         f"reflects {ds}, not today. Use Refresh in the rail to pull the "
                         f"latest once the source updates."),
                    )
                elif behind >= 1:
                    _notice(
                        "info", "Data freshness",
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
                    _notice(
                        # WARNING, not info. The old copy said only that momentum
                        # goes flat, which understates it: a snapshot-reconstructed
                        # column is reindexed onto this run's calendar and ffilled
                        # (data.fetcher._backfill_missing_columns), so it is a
                        # DIFFERENT SERIES from a live pull, not a staler one.
                        # Measured 2026-08-17: two fetches minutes apart that hit
                        # different timeouts backfilled different tickers, and 1563
                        # settled input cells then differed — moving 5743 published
                        # output cells, from the first post-burn-in row onward. That
                        # happened on 1 of 4 fetch pairs.
                        "warning", "Predictors carried from snapshot — history differs",
                        f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                        f"from a prior cached snapshot: {_sb_preview}{_sb_more}. A rebuilt column "
                        f"is not merely staler than a live one — it is a different series, so "
                        f"THIS RUN'S PUBLISHED HISTORY MAY DIFFER from a run that fetched cleanly. "
                        f"Compare the panel fingerprint before treating this run's past values as "
                        f"final; re-run once the source is healthy for an authoritative record.",
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
                            _notice(
                                "warning", "Partial latest session",
                                (f"Only {fresh_frac:.0%} of the markets open on {ds} have posted — the "
                                 f"rest are forward-filled from the prior session, so the macro predictors "
                                 f"and bottom-up breadth behind the latest signal are stale. Treat it as "
                                 f"provisional; use Refresh in the rail once those markets post."),
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
                                _notice(
                                    "info", f"{active_target} price not yet updated",
                                    (f"Today's bar is carried forward from {t_last.strftime('%d %b %Y')} — "
                                     f"the {active_target} market is open but yfinance may have rate-limited "
                                     f"this ticker during the last fetch. Use Refresh in the rail "
                                     f"to pull the latest price."),
                                )
                            else:
                                _notice(
                                    "warning", f"{active_target} data is lagging",
                                    (f"This target last updated {t_last.strftime('%d %b %Y')} "
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
        if is_stock_target(active_target):
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
        console.item("Scoring Horizon", f"{_prep.get('fwd_h','?')}d "
                     "(convergence / analogs / display — the engine is fit to no label)")
        console.item("Rows · fetched", _prep.get("rows_initial", "?"))
        console.item("Rows · after session spine", f"{_prep.get('rows_session','?')}  ({_prep.get('sessions_dropped','?')} non-session rows removed)")
        console.item("Instruments · requested", _prep.get("feats_requested", "?"))
        console.item("Instruments · dropped (short history)", f"{len(_prep.get('feats_dropped', []))}"
                     + (f" → {', '.join(_prep['feats_dropped'][:6])}{'…' if len(_prep.get('feats_dropped', [])) > 6 else ''}" if _prep.get("feats_dropped") else ""))
        if _prep.get("yield_cols_dropped"):
            # Raw yield LEVELS are rate series, not prices: they print at/near/
            # below zero and the engine's log transform is undefined there. The
            # tradeable expression of the curve is already in the panel as the
            # Treasury ETF complex, so they are excluded rather than transformed.
            console.item("Instruments · dropped (raw yield levels, not prices)",
                         _prep["yield_cols_dropped"])
        console.item("Instruments · in valuation panel", _prep.get("feats_kept", "?"))
        # The fingerprint makes a composition change visible. Same digest across
        # two runs = same cross-section, so any difference in output came from
        # the data's VALUES or the code. A changed digest says the panel itself
        # moved, which rewrites every published date — see the note in
        # data/fetcher.py::_panel_fingerprint.
        _fp = (df.attrs or {}).get("panel_fingerprint") if hasattr(df, "attrs") else None
        if _fp:
            console.item("Panel fingerprint", _fp)
        console.item("Rows · after dropna (final spine)", _prep.get("rows_final", "?"))
        if "burn_in" in _prep and isinstance(_prep.get("rows_final"), int):
            # Unlike a rolling momentum window there is no warm-up head to trim
            # here — every row carries a finite level for the target and every
            # surviving instrument. The engine applies its own burn-in
            # internally and flags those rows `Valid = False`.
            _pub = max(0, _prep["rows_final"] - _prep["burn_in"])
            console.item("Rows · valued (after engine burn-in)",
                         f"{_pub}  ({_prep['burn_in']} burn-in rows published as Valid=False)")
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
    # NOTE on the Mūla print mask. The engine admits an instrument to the
    # cross-section only on days it genuinely traded — a carried-forward quote
    # otherwise enters as a fabricated zero return and drags whatever factor it
    # loads on toward zero on every foreign holiday. The exact answer is the
    # vendor's own NaN mask, but it no longer exists by the time this runs:
    # data/fetcher.py forward-fills the combined frame at source (see its
    # `combined.ffill()`), so a mask taken here would be all-True and would
    # silently assert that a Nikkei quote on a Tokyo holiday is a real print.
    # Rather than pass a mask that looks exact and is not, the panel is handed
    # over without one and the engine infers prints from where values actually
    # change (FairValueEngine._infer_printed) — conservative in the same
    # direction as the gate itself. Threading the true mask would mean
    # returning it from the fetcher alongside the prices, through the cache
    # layer; the `printed=` parameter on fit() is there for when that happens.
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
    # The valuation regression models LOG price → the target must be strictly
    # positive. Every shipped target is a price/level/ratio, but a future target
    # (a spread, a net position, a yield differential) could go ≤0; fail clean
    # rather than silently producing all-NaN valuations.
    if (pd.to_numeric(data[active_target], errors="coerce") <= 0).any():
        _log_prep(stage="fail")
        console.failure("Non-positive target", f"{active_target}: contains values ≤ 0; the log-level valuation engine needs a strictly positive series.")
        st.error(f"'{active_target}' has non-positive values — the valuation engine models "
                 f"log price and needs a strictly positive series.")
        return

    # ── Valuation representation: price the target against the traded
    # opportunity set, rather than forecasting its next move.
    #   • Panel   = the LEVEL of every macro predictor. The Mūla engine takes
    #     its own logs and differences internally, integrates the resulting
    #     factors back into levels, and regresses log price on them with
    #     time-varying coefficients. It therefore wants prices, not the
    #     pre-engineered momentum features the previous engine consumed.
    #   • Target  = the target's own price level. There is no forward label
    #     and so no forward-label overlap, no purge gap, and no zero-filled
    #     tail: the newest row is valued the same way every other row is.
    # FWD_HORIZON survives as a SCORING horizon only — the convergence layer,
    # the precedent analogs and the Intelligence calibrator all score against
    # the h-day forward return, and the UI projects the current mispricing
    # over it. Daily bars throughout.
    FWD_HORIZON = _icfg.forecast_horizon   # scoring / display horizon (trading days)
    _prep["fwd_h"] = FWD_HORIZON
    # RAW_YIELD_PREDICTORS (^IRX/^FVX/^TNX/^TYX) are percent-point RATE series,
    # not prices: they print at/near/below zero (2020-21 zero-rate era), and
    # the engine's log transform is undefined there. A yield LEVEL is also not
    # an instrument the target can be valued against — the tradeable expression
    # of the curve is already in the panel as the Treasury ETF complex
    # (SHY/IEF/TLT/…), which the block map classifies as "Rates". So they are
    # dropped from the valuation panel rather than transformed.
    _yield_feats = [f for f in active_features if f in RAW_YIELD_PREDICTORS]
    if _yield_feats:
        active_features = [f for f in active_features if f not in RAW_YIELD_PREDICTORS]
        _prep["yield_cols_dropped"] = len(_yield_feats)
        # Re-stamp the kept count: it was recorded after the short-history guard
        # but before this exclusion, so the prep log would otherwise report a
        # panel four instruments wider than the one the engine is handed.
        _prep["feats_kept"] = len(active_features)
    if not active_features:
        _log_prep(stage="fail")
        console.failure("Empty valuation panel", f"{active_target}: no price-level predictors survived.")
        st.error("No price-level predictors remain — the valuation engine needs a cross-section of prices.")
        return
    # No warm-up head to trim, and no row-validity mask: every surviving row
    # already carries a finite level for the target and every instrument (the
    # dropna above). The predecessor needed both — a rolling momentum window
    # left a NaN head, and its forward labels left a zero-filled tail — so the
    # prep log reported "usable rows" and "real labels" as separate counts.
    # Here the only exclusion is the engine's own burn-in, applied internally
    # and reported as `Valid`.
    _prep["burn_in"] = int(_icfg.fvo_burn_in)
    # Date-range fingerprint for the cache key. `data` carries a RangeIndex (reset at
    # load), so the real dates live in the active_date column, not the index — using
    # the index here would be integers (AttributeError on .date()). Fall back to a
    # row-count surrogate when there's no date column.
    if active_date != "None" and active_date in data.columns:
        _vd = pd.to_datetime(data[active_date], errors="coerce").dropna()
        _date_range = f"{_vd.iloc[0].date()}_{_vd.iloc[-1].date()}" if len(_vd) else f"n{len(data)}"
    else:
        _date_range = f"n{len(data)}"
    cache_key = f"fvo{FWD_HORIZON}|{active_target}|{'|'.join(sorted(active_features))}|{_date_range}"
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
        # HORIZON-INDEPENDENT: the macro fetch and the target's own OHLCV depend
        # only on active_target, never on the scoring horizon, so they are cached
        # separately (audit finding F17) and survive a re-run.
        #
        # There is no mode resolution left to do. Every target reads breadth off
        # its own price through Swayam; the branch that used to choose between
        # that and a constituent basket — and the basket resolution, the
        # snapshot fallbacks, and the up-to-503-symbol OHLCV fetch behind it —
        # went with the basket engine. On a large index that fetch was ~13 of
        # the run's ~14 minutes.
        _swayam_fetch_key = f"swayam_fetch::{active_target}"
        _nf_cache = st.session_state.get("_swayam_fetch_cache")
        if _nf_cache is not None and _nf_cache.get("key") == _swayam_fetch_key:
            console.start_phase("DATA ACQUISITION", 1, 5)
            target_ohlcv = _nf_cache["target_ohlcv"]
            swayam_macro_df = _nf_cache["swayam_macro_df"]
            macro_cols_list = _nf_cache["macro_cols_list"]
            console.item("Macro/OHLCV", "reused cached fetch (horizon-independent)")
            progress_bar(progress_container, 19, "Data Acquisition Reused", f"{len(macro_cols_list)} Macros (cached)")
            console.end_phase("DATA ACQUISITION")
        else:
            console.start_phase("DATA ACQUISITION", 1, 5)
            progress_bar(progress_container, 16, "Resolving Swayam Source",
                         f"{active_target} · own OHLCV (self-referential ensemble)")

            console.section("Macro Data")
            end_date = pd.Timestamp.today()
            # Match the Mūla model-dataset window (~9y) so the Swayam views and
            # macro drivers overlap the FULL series — convergence then runs on
            # real data, not neutral placeholders.
            start_date = end_date - pd.DateOffset(days=365 * 9)
            macro_df = fetch_macro_live(start_date, end_date)
            console.item("Date Range", f"{start_date.date()} to {end_date.date()}")
            if not macro_df.empty:
                console.item("YF Columns", f"{len(macro_df.columns)} symbols")
                console.item("Rows", len(macro_df))
                console.success(f"Macro data: {len(macro_df.columns)} symbols × {len(macro_df)} rows")
            else:
                console.warning("No macro data available")

            console.section("Target OHLCV")
            # Swayam needs the target's own OHLC (and volume where it exists —
            # the volume-dependent views abstain when it does not; see
            # ensemble._is_volume_dependent).
            _tgt_ticker = ALL_TARGETS[active_target]
            progress_bar(progress_container, 18, "Fetching Target OHLCV", f"yfinance · {_tgt_ticker}")
            _ohlcv = fetch_constituent_ohlcv([_tgt_ticker], start_date, end_date)
            target_ohlcv = _ohlcv.get(_tgt_ticker)
            if target_ohlcv is not None and not target_ohlcv.empty:
                console.item("Symbol", _tgt_ticker)
                console.item("Rows", len(target_ohlcv))
                console.item("Has Volume", bool("Volume" in target_ohlcv.columns
                                                and target_ohlcv["Volume"].fillna(0).abs().sum() > 0))
                console.success(f"Target OHLCV: {len(target_ohlcv)} rows")
            else:
                console.warning(f"No OHLCV for {_tgt_ticker} — Swayam breadth will be unavailable")

            console.section("Swayam Macro Assembly")
            swayam_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
            if not swayam_macro_df.empty:
                console.item("YF Symbols", len(swayam_macro_df.columns))
                console.success(f"Macro indicators: {len(swayam_macro_df.columns)} × {len(swayam_macro_df)} rows")
            macro_cols_list = list(swayam_macro_df.columns) if not swayam_macro_df.empty else []
            console.end_phase("DATA ACQUISITION")
            progress_bar(progress_container, 19, "Data Acquisition Complete", f"{len(swayam_macro_df.columns)} Macros")

            st.session_state["_swayam_fetch_cache"] = {
                "key": _swayam_fetch_key,
                "target_ohlcv": target_ohlcv,
                "swayam_macro_df": swayam_macro_df,
                "macro_cols_list": macro_cols_list,
            }

        # ── Phase 2: Mūla (FairValueEngine) ─────────────────────────────────
        console.start_phase("MŪLA ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Mūla Engine", f"Valuation · {len(active_features)} Instruments · {len(data)} Rows")

        _price_level = data[active_target].to_numpy(dtype=np.float64)
        _cal = (pd.to_datetime(data[active_date].values)
                if active_date != "None" and active_date in data.columns
                else pd.RangeIndex(len(data)))
        _tgt_px = pd.Series(_price_level, index=_cal, name=active_target)
        _expl_px = data[active_features].astype(float)
        _expl_px.index = _cal

        _blk_names, _blk_map = block_membership(active_features)

        console.section("Engine Configuration")
        console.item("Mode", "Valuation · dynamic cointegrating regression on log price")
        console.item("Target", active_target)
        console.item("Cross-section", f"{len(active_features)} macro instruments → {len(_blk_names)} asset-class blocks")
        console.item("Blocks", ", ".join(f"{b}({sum(1 for v in _blk_map.values() if v == b)})" for b in _blk_names))
        console.item("Observations", f"{len(data)} rows")
        console.item("Burn-in", f"{_icfg.fvo_burn_in} rows before first publication")
        console.item("Print Floor", f"{_icfg.fvo_min_prints} prints before an instrument may join")
        console.item("Coefficient Memory", f"deltas {_icfg.fvo_valuation_deltas}")
        console.item("Min Data Points", MIN_DATA_POINTS)
        console.item("Lookback Windows", f"{LOOKBACK_WINDOWS}")

        console.section("Recursive Valuation")
        # Reuse an already-fit Mūla engine for this exact config if a prior
        # (possibly interrupted) execution in THIS session already produced one.
        # `engine_cache` is only set at the end of Phase 5, so a Streamlit rerun
        # mid-pipeline (yfinance retry, cloud reconnect, stray interaction) would
        # otherwise re-enter this block and re-run the expensive valuation pass.
        # Keyed by cache_key → identical inputs → identical fit, so reuse is safe.
        if (st.session_state.get("fvo_fit_key") == cache_key
                and isinstance(st.session_state.get("fvo_engine"), FairValueEngine)):
            engine = st.session_state["fvo_engine"]
            console.item("Valuation", "reused cached fit (resumed run)")
            progress_bar(progress_container, 40, "Mūla Engine Reused", "Cached valuation pass")
        else:
            if _HAS_V2_ENGINES:
                engine = MulaEngine()
                console.item("Valuation Core", "Mūla — cointegrating core + ECM / expert pooling")
            else:
                engine = FairValueEngine()
            # `config=_icfg` threads this instrument's per-instrument Mūla knobs
            # (burn-in, print floor, discount grid, lookback windows) into the
            # valuation, so Mūla is tuned per instrument / asset class exactly
            # like Swayam and Swayam.
            engine.fit(
                _tgt_px, _expl_px,
                feature_names=active_features, config=_icfg,
                progress_callback=lambda pct, msg: progress_bar(
                    progress_container, int(20 + pct * 20), "Running Mūla Engine", msg),
            )
            # Carry the raw price LEVEL on the engine output too. `Actual` is
            # already the price here, but the analog matcher and Intelligence
            # tuner both key off a column literally named "Price".
            engine.ts_data["Price"] = _price_level
            st.session_state["fvo_engine"] = engine
            st.session_state["fvo_fit_key"] = cache_key
        # MŪLA summary travels through session state for the hero card's ECM
        # gate and the Precedent state vector (empty dict on the legacy path).
        st.session_state["mula_summary"] = getattr(engine, "mula_summary", {})

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()
        console.section("Engine Results")
        console.item("Signal", f"{sig['signal']} ({sig['strength']})")
        console.item("Conviction", f"{sig['conviction_score']:+.0f}")
        console.item("Mūla Osc", f"{sig['fvo']:+.2f}σ ({sig['pct_mispricing'] * 100:+.2f}% vs fair value)")
        console.item("Fair Value", f"{sig['fair_value']:,.2f} vs price {sig['actual']:,.2f}")
        console.item("OOS R²", f"{stats['r2_oos']:.3f} (log price vs fitted level)")
        console.item("R² vs Trailing Mean", f"{stats['r2_vs_anchor']:+.3f} (edge over a 252d anchor)")
        console.item("Model Spread", f"{sig['model_spread'] * 10000:.1f} bps (predictive SD of fair value)")
        console.item("Valuation Confidence", f"{sig['valuation_confidence']:.2f} "
                     f"(mean-reversion {sig['mr_prob']:.2f} × cross-sectional agreement {sig['xs_consistency']:.2f})")
        # DFA Hurst is deliberately not printed: its single log-log slope is
        # biased sharply upward for a short-memory series and would read
        # "trending" for a gap ADF calls strongly stationary. See
        # FairValueEngine._compute_hurst.
        console.item("Gap Half-Life", f"{sig['gap_half_life']:.0f}d (online AR1) · OU {sig['ou_half_life']:.0f}d")
        console.item("Gap Stationarity", f"ADF p={sig['adf_pvalue']:.4f} "
                     f"({'stationary — reversion licensed' if sig['adf_pvalue'] < 0.05 else 'unit root not rejected'})")
        # `n_available` is the count that PRINTED today, not the count ADMITTED
        # (that is NAdmitted, the instruments past the print floor). Labelling
        # it "admitted" made a normal early-session reading look like a
        # collapsed cross-section: a 05:49 UTC run shows ~87 of 234 printed
        # because Asia is open and the US is not, which is not the same claim as
        # "only 87 instruments qualify".
        # Name any gap the reader can see on the chart. A break is never "the
        # value was zero" — it is the system declining to publish — and the four
        # causes want different responses. Only "incomplete fetch" is worth
        # re-running for, so it is the only one reported as a warning.
        try:
            _wr = fvo_ts.get("WithheldReason") if fvo_ts is not None else None
            if _wr is not None:
                _c = _wr[_wr.astype(str) != ""].astype(str).value_counts()
                if len(_c):
                    console.item("Gaps in the trace",
                                 " · ".join(f"{k}: {int(v)}" for k, v in _c.items()))
                _bad = int(_c.get("incomplete fetch", 0))
                if _bad:
                    _dates = _wr.index[_wr.astype(str) == "incomplete fetch"]
                    _shown = ", ".join(str(d)[:10] for d in list(_dates)[-3:])
                    console.warning(
                        f"{_bad} session(s) withheld as INCOMPLETE FETCH ({_shown}). "
                        f"The calendar says those exchanges were open, so this is a "
                        f"data gap in this run rather than a market closure — re-run "
                        f"to fill it."
                    )
        except Exception:                       # noqa: BLE001 - never break a run to report
            pass

        console.item("Factors", f"k={sig['k_factors']} above the MP edge · "
                                f"{sig['n_available']} instruments printed so far today")
        console.item("Market Regime", f"{sig['market_regime']} (stress pct {sig['stress']:.2f})")
        console.success(f"Mūla engine complete | {len(engine.ts_data)} output rows "
                        f"({int(engine.ts_data['Valid'].sum())} valued, {engine.min_train_size} burn-in)")
        console.end_phase("MŪLA ENGINE")
        progress_bar(progress_container, 40, "Mūla Engine Complete", f"Signal: {sig['signal']} ({sig['strength']}) · Conviction: {sig['conviction_score']:+.0f}")

        # ── Phase 3: Swayam Breadth ───────────────────────────────────────
        # HORIZON-INDEPENDENT (audit finding F17): the view bank and its
        # aggregation depend only on the target's own OHLCV + the macro driver
        # window, never on the scoring horizon. Cached under the SAME
        # _swayam_fetch_key as Phase 1; only the target-calendar reindex below
        # (cheap — no yfinance calls) re-runs.
        console.start_phase("SWAYAM ENGINE", 3, 5)
        progress_bar(progress_container, 42, "Running Swayam Engine",
                     "MSF+MMR+Regime · self-referential view bank")

        _na_cache = st.session_state.get("_swayam_analysis_cache")
        if _na_cache is not None and _na_cache.get("key") == _swayam_fetch_key:
            swayam_view_dfs = _na_cache["swayam_view_dfs"]
            swayam_daily_pre_reindex = _na_cache["swayam_daily_pre_reindex"]
            if "n_eff" in _na_cache:
                st.session_state["swayam_n_eff"] = _na_cache["n_eff"]
            console.item("View Bank", "reused cached fit (horizon-independent)")
            progress_bar(progress_container, 74, "Swayam Engine Reused", f"{len(swayam_view_dfs)} Views (cached)")
        else:
            swayam_daily_pre_reindex = pd.DataFrame()
            swayam_view_dfs = {}

            if target_ohlcv is not None and not target_ohlcv.empty:
                console.section("Self-Referential View Bank")
                # Leakage guard: drop the target's own macro column + its
                # excluded-predictor near-replicas from the MMR driver pool — a
                # view's Close correlates ~1.0 with the target's own macro
                # column, which would let MMR "explain" the target with itself
                # and silently zero the deviation oscillator.
                swayam_cols = swayam_macro_columns(active_target, macro_cols_list)
                _swayam_members = default_swayam_members(_icfg.swayam_lengths, _icfg.swayam_roc_frac)
                console.item("Views (bank)", f"{len(_swayam_members)} · timescale × information-set × mechanism")
                console.item("Timescale Span", str(_icfg.swayam_lengths))
                console.item("Regime Sensitivity", _icfg.swayam_regime_sensitivity)
                console.item("Base Weight", _icfg.swayam_base_weight)
                console.item("Macro Columns (post-leakage-guard)", len(swayam_cols))

                def _swayam_progress(done, total, name):
                    pct_val = int(45 + done / max(total, 1) * 30)
                    progress_bar(progress_container, pct_val, f"View {name}", f"{done}/{total} views")

                swayam_view_dfs = build_swayam_frames(
                    target_ohlcv, swayam_macro_df, swayam_cols,
                    members=_swayam_members,
                    regime_sensitivity=_icfg.swayam_regime_sensitivity,
                    base_weight=_icfg.swayam_base_weight,
                    num_vars=_icfg.swayam_mmr_num_vars,
                    oversold=_icfg.swayam_oversold, overbought=_icfg.swayam_overbought,
                    progress_cb=_swayam_progress,
                )
                n_eff = effective_member_count(swayam_view_dfs)
                st.session_state["swayam_n_eff"] = n_eff
                console.success(f"View bank: {len(swayam_view_dfs)} views · ~{n_eff:.1f} effective")

            if swayam_view_dfs:
                console.section("Aggregation")
                # Views are weighted by their own realised skill, estimated
                # causally (analytics.adaptive) — a timescale that has stopped
                # predicting this instrument contributes less to breadth, and
                # no grid had to be chosen for that to happen.
                _view_w = view_skill_weights(swayam_view_dfs, horizon=FWD_HORIZON)
                swayam_daily_pre_reindex = aggregate_views(swayam_view_dfs, weights=_view_w)
                if not _view_w.empty:
                    _last = _view_w.iloc[-1].sort_values(ascending=False)
                    console.item("View Weighting", f"skill-weighted · {len(_last)} views")
                    console.item("Top Views", " · ".join(f"{k} {v:.2f}" for k, v in _last.head(4).items()))
                    console.item("Weakest View", f"{_last.index[-1]} {_last.iloc[-1]:.2f}")

            st.session_state["_swayam_analysis_cache"] = {
                "key": _swayam_fetch_key,
                "swayam_view_dfs": swayam_view_dfs,
                "swayam_daily_pre_reindex": swayam_daily_pre_reindex,
                "n_eff": st.session_state.get("swayam_n_eff"),
            }

        # ── HORIZON-DEPENDENT tail: reindex onto the target's calendar ──────
        # Cheap (pure pandas, no yfinance) — re-runs since the horizon's
        # warm-up trim can shift the target's date spine.
        swayam_daily = swayam_daily_pre_reindex
        if not swayam_daily.empty:
            # Carry the basket forward onto the TARGET's trading calendar. The
            # views share the target's own calendar by construction, but the macro
            # driver pool behind MMR does not
            # than the target — on a Monday-morning IST run, or when the target's
            # market is open but the basket's is on holiday, the basket's last close
            # IS its current value. Reindexing it onto the target's dates (ff-fill)
            # lets the SIGNAL, cards and plots all reach the target's latest session
            # instead of truncating to the slowest constituent. We record the
            # basket's true last-native date so the UI can flag how much is carried
            # over (the partial-session notice covers the row-level staleness).
            st.session_state["swayam_native_last"] = pd.Timestamp(swayam_daily.index.max())
            if active_date in data.columns:
                _cal = pd.DatetimeIndex(sorted(pd.to_datetime(
                    data[active_date], errors="coerce").dropna().dt.normalize().unique()))
                _nd = swayam_daily.copy()
                _nd.index = pd.to_datetime(_nd.index).normalize()
                _nd = _nd[~_nd.index.duplicated(keep="last")].sort_index()
                # _Native marks rows that are a genuine basket observation
                # (present in _nd BEFORE the reindex) vs carried forward by
                # the ffill below (the basket's market was closed/hadn't
                # posted that day). Carried through so the calibration
                # overlap gate can require NATIVE overlap, not ffilled
                # rows masquerading as fresh Swayam signal (audit finding
                # F21) — the UI's own "breadth carried forward" notice
                # already discloses this to the user; the gate didn't.
                _native_dates = set(_nd.index)
                swayam_daily = _nd.reindex(_cal, method="ffill").dropna(how="all")
                swayam_daily["_Native"] = swayam_daily.index.isin(_native_dates)
            console.item("Trading Days", len(swayam_daily))
            if len(swayam_daily) > 0:
                last = swayam_daily.iloc[-1]
                console.item("Avg Signal", f"{last.get('Avg_Signal', 0):+.2f}")
                console.item("Oversold %", f"{last.get('Oversold_Pct', 0):.0f}%")
                console.item("Overbought %", f"{last.get('Overbought_Pct', 0):.0f}%")
                console.item("Buy Signals", int(last.get('Buy_Signals', 0)))
                console.item("Sell Signals", int(last.get('Sell_Signals', 0)))
            console.success(f"Swayam aggregation: {len(swayam_daily)} trading days")

        console.end_phase("SWAYAM ENGINE")
        progress_bar(progress_container, 75, "Swayam Engine Complete", f"{len(swayam_view_dfs)} Views · {len(swayam_daily)} Trading Days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 76, "Computing Convergence", "Cross-Validation · DDM Filtering")

        console.section("Cross-Validation Setup")
        # ── Convergence weights: prior, then learned forward ─────────────
        # The first pass builds the dim_* matrix with the PRIOR weights. The
        # composite is then recomputed (Phase 4b) with weights learned online
        # from resolved outcomes only, so no score depends on data that had
        # not happened when it was published.
        #
        # There is no profile to resolve and no mode to be in. The Optuna
        # search that used to run here fit the whole history at once and wrote
        # its winner to disk, which meant the output depended both on future
        # data and on when you last calibrated; both are gone.
        from convergence import intelligence as _intel_mod
        _prior_w = _icfg.weights_seed()
        console.item("Dimension weights", "prior → learned online (causal, no calibration step)")

        # First pass builds the dim_* sub-scores; the composite it also writes
        # is superseded below by the online-weighted recomputation.
        _validator_weights = _prior_w
        # The vote count is the Swayam bank's member count. Every view always
        # reports, so coverage reads 1.0 — unlike a basket, where a constituent
        # could simply be missing that day and breadth was read off a partial
        # cross-section without saying so.
        _expected_constituents = len(swayam_view_dfs) or None
        validator = CrossValidator(
            active_weights=_validator_weights,
            expected_constituents=_expected_constituents,
        )
        divergence_detector = CrossSystemDivergenceDetector()

        fvo_ts = engine.ts_data.copy()  # carries "Price" (set after fit)
        if active_date != "None" and active_date in data.columns:
            fvo_ts["Date"] = pd.to_datetime(data[active_date].values)
            fvo_ts = fvo_ts.set_index("Date")
        else:
            fvo_ts["Date"] = np.arange(len(fvo_ts))
        fvo_ts = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
        console.item("Mūla Dates", len(fvo_ts))

        swayam_by_date = {}
        if not swayam_daily.empty:
            swayam_unique = swayam_daily[~swayam_daily.index.duplicated(keep="last")]
            for idx in swayam_unique.index:
                key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
                swayam_by_date[key] = swayam_unique.loc[idx]
            console.item("Swayam Dates", len(swayam_by_date))

        console.section("Daily Convergence Scoring")
        overlap_count = 0
        native_overlap_count = 0
        skipped_warmup = 0
        total_dates = len(fvo_ts.index)
        for i, ts_idx in enumerate(fvo_ts.index):
            ts_date = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            date_str = str(ts_date)
            row_a = fvo_ts.loc[ts_idx]
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[-1]
            # Skip the engine's own [0, MIN_TRAIN_SIZE) warm-up rows — the
            # `Valid` column (engines/mula/base.py) is False there because no
            # genuine walk-forward forecast covers them (see A3 in the audit).
            # Scoring them would feed the Intelligence calibration frame and
            # the walk-forward IC a fabricated "neutral" convergence reading
            # instead of genuinely excluding the unfit region.
            if not bool(row_a.get("Valid", True)):
                skipped_warmup += 1
                continue
            fvo_sig = {
                "conviction_score": float(row_a.get("ConvictionBounded", 0)),
                "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
                "regime": str(row_a.get("Regime", "NEUTRAL")),
            }
            if date_str in swayam_by_date:
                row_n = swayam_by_date[date_str]
                swayam_stats = {
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
                swayam_stats = {
                    "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                    "regime_bull_pct": 33, "regime_bear_pct": 33,
                    "regime_neutral": 34, "num_constituents": 0,
                }
            validator.compute_convergence(fvo_sig, swayam_stats, date_str)
            divergence_detector.detect(fvo_sig, swayam_stats, date_str)

            if (i + 1) % 10 == 0 or i == total_dates - 1:
                # 76-82: the step that follows this loop reports 83, so the
                # loop must finish below it. It used to run to 85 and the bar
                # visibly went backwards at the hand-off.
                pct_val = int(76 + (i + 1) / total_dates * 6)
                progress_bar(progress_container, pct_val, "Computing Convergence", f"{i + 1}/{total_dates} Dates Scored")

        console.item("Total Mūla Dates", len(fvo_ts))
        console.item("Skipped (warm-up, no genuine forecast)", skipped_warmup)
        console.item("Overlap Dates", f"{overlap_count} ({native_overlap_count} native, "
                     f"{overlap_count - native_overlap_count} carried-forward)")
        console.success("Convergence scoring complete")

        # ── Online-weighting overlap gate ────────────────────────────────
        # Skip the online re-weighting when the Mūla/Swayam overlap is too thin.
        # With no genuine overlap every date takes the same neutral swayam_stats
        # default, so the continuous consensus direction degenerates to
        # fvo_bull/2 and every Swayam-driven dim score is constant — the learner
        # would then be scoring what is really a half-weight Mūla-only signal and
        # attributing its skill to dimensions that never varied.
        #
        # Gated on NATIVE overlap, not raw overlap (audit finding F21):
        # swayam_daily is forward-filled onto the target's calendar before the
        # scoring loop, so `overlap_count` alone would count a long run of
        # carried-forward, non-fresh breadth rows as new information each day.
        # 60 dates (~3 trading months) excludes the genuinely-degenerate case
        # without second-guessing a real but short history.
        _MIN_OVERLAP_FOR_LEARNING = 60
        _learn_weights = native_overlap_count >= _MIN_OVERLAP_FOR_LEARNING
        if not _learn_weights:
            console.warning(
                f"Online dimension weighting skipped: only {native_overlap_count} NATIVE "
                f"Mūla/Swayam overlap dates (< {_MIN_OVERLAP_FOR_LEARNING}; "
                f"{overlap_count} total incl. carried-forward) — the dimensions would be "
                f"constant and their measured skill meaningless. Prior weights stand."
            )

        # ── 4a. First-pass conviction model ─────────────────────────────
        # First-pass DDM filter on the convergence_score from the first
        # validator pass. Labeled "first-pass" only when Intelligence Mode
        # is ON (a second pass will follow); just "Conviction Model" otherwise.
        _first_pass_label = "First-Pass Conviction Model" if _learn_weights else "Conviction Model"
        console.section("Conviction Model (initial pass)" if _learn_weights else "Conviction Model")
        progress_bar(progress_container, 83, _first_pass_label, "DDM Filter · Prior Weights")
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
            _pre_label = "DDM Conviction (pre-weighting)" if _learn_weights else "DDM Conviction"
            _sig_label = "DDM Signal (pre-weighting)" if _learn_weights else "DDM Signal"
            console.item(_pre_label, f"{latest.nishkarsh_conviction:+.0f}")
            console.item(_sig_label, latest.nishkarsh_signal)
        console.success(f"Initial conviction: {len(results)} scores computed")

        # ── 4b. Learn the dimension weights, forward only ───────────────
        # Replaces the Optuna calibration that used to sit here. That search
        # fit the whole history and applied its winner back across the same
        # history, so every published score changed whenever a session was
        # added; it also persisted the winner to disk, making the output depend
        # on when it was last run. This learns the same thing — which
        # dimensions actually predict this instrument — from resolved outcomes
        # only, at a fraction of the cost and with no state carried between
        # runs.
        _weight_hist = pd.DataFrame()
        if _learn_weights:
            console.section("Online Dimension Weighting")
            progress_bar(progress_container, 86, "Learning Dimension Weights",
                         "Causal · discounted directional skill per dimension")
            try:
                convergence_df, _weight_hist = _intel_mod.apply_online_weights(
                    convergence_df, fvo_ts, horizon=FWD_HORIZON, target_col="Price",
                )
                if not _weight_hist.empty:
                    _wnow = _weight_hist.iloc[-1]
                    _wprior = _intel_mod.PRIOR_WEIGHTS
                    console.item("Horizon", f"{FWD_HORIZON}d (outcome resolves h days after the call)")
                    console.item("Learned Weights",
                                 " · ".join(f"{k} {v:.3f}" for k, v in _wnow.sort_values(ascending=False).items()))
                    console.item("Prior Weights",
                                 " · ".join(f"{k} {_wprior[k]:.2f}" for k in _wnow.index))
                    _moved = max(abs(_wnow[k] - _wprior[k]) for k in _wnow.index)
                    console.item("Max Shift From Prior", f"{_moved:+.3f}")
                    console.success(f"Dimension weights learned over {len(_weight_hist)} dates (no repaint)")
                else:
                    console.warning("Online weighting produced no coverage — prior weights stand")
            except Exception as _we:
                console.warning(f"Online weighting failed: {_we} — prior weights stand")

            # Re-fit the conviction model on the re-weighted composite.
            progress_bar(progress_container, 92, "Re-Fitting Conviction Model",
                         "DDM pass on the online-weighted composite")
            conviction_model = UnifiedConvictionModel(
                leak_rate=_icfg.ddm_leak, drift_scale=_icfg.ddm_drift, long_run_var=_icfg.ddm_lrv,
            )
            results = conviction_model.fit(
                convergence_df["convergence_score"].tolist(),
                convergence_df.index.tolist(),
            )
            console.section("Conviction Model (re-weighted)")
            if results:
                latest = results[-1]
                console.item("DDM Conviction", f"{latest.nishkarsh_conviction:+.0f}")
                console.item("DDM Signal", latest.nishkarsh_signal)
                console.item("DDM Band", f"[{latest.confidence_lower:.0f}, {latest.confidence_upper:.0f}]")
            console.success("Re-fit complete on online-weighted convergence")

        _active_w = (dict(_weight_hist.iloc[-1]) if not _weight_hist.empty
                     else _icfg.weights_seed())
        _active_t = _icfg.composite_thresholds()

        # Publish to session state so the Passport sidebar + Convergence cards
        # see the calibrated state immediately on the next rerun.
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_thresholds"] = _active_t
        st.session_state["intelligence_active_profile"] = (
            {"weights": _active_w, "learned": bool(_learn_weights)}
        )

        # ── 4d. NORMALIZED CONSENSUS ─────────────────────────────────────────
        # The causal expanding-z average of Mūla's ConvictionRaw and Swayam's
        # Avg_Signal. It headlined the hero card until the card became a
        # conviction chain; it is now an INPUT to that chain's convergence
        # gate (a consensus pointing against the Mūla call shuts the gate), and
        # the top row of the Unified Signal plot. `consensus_series` is the
        # single source for the full history; the dict is its last point.
        from convergence.normalization import (
            compute_normalized_convergence, consensus_series, classify_convergence_score,
        )
        _consensus_full = consensus_series(fvo_ts, swayam_daily)
        _nishkarsh_norm = compute_normalized_convergence(fvo_ts, swayam_daily)
        if _nishkarsh_norm:
            console.section("Normalized Consensus (headline)")
            console.item("Conviction", f"{_nishkarsh_norm['value']:+.2f}")
            console.item("Signal", _nishkarsh_norm['signal'])
            console.item("  Mūla contribution", f"{_nishkarsh_norm['fvo_norm']:+.2f}")
            console.item("  Swayam contribution",  f"{_nishkarsh_norm['swayam_norm']:+.2f}")

        # ── 4e. WEIGHTED COMPOSITE — the second construction ────────────────
        # convergence_score (post online re-weighting, ±100 scale) IS the exact
        # quantity the learner scores, so this is the one place the learned
        # weights are semantically valid (audit findings F1/F2). It feeds the
        # hero's WEIGHTED evidence row — a genuine second opinion, being a
        # dimension-weighted construction of the same two engines rather than
        # their plain 50/50 mean. The
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
                convergence_df, fvo_ts, target_col="Price", horizons=_hold_grid,
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
            f"{'Online-Weighted' if _learn_weights else 'Prior Weights'}"
        )
        progress_bar(progress_container, 95, "Convergence Phase Complete", _conv_complete_sub)

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 96, "Storing Results", "Session State · Cache")
        console.section("Session State")

        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
        st.session_state["fvo_ts"] = fvo_ts
        st.session_state["swayam_daily"] = swayam_daily
        st.session_state["swayam_view_dfs"] = swayam_view_dfs
        st.session_state["convergence_df"] = convergence_df
        st.session_state["divergence_events"] = events
        st.session_state["nishkarsh_result"] = results[-1] if results else None
        st.session_state["last_agreement"] = convergence_df["agreement_ratio"].iloc[-1] if not convergence_df.empty else 0
        # The weighted composite — the hero's WEIGHTED evidence row and the
        # Unified Signal plot's amber overlay.
        st.session_state["nishkarsh_calibrated_score"] = _calibrated_score
        st.session_state["nishkarsh_calibrated_signal"] = _calibrated_signal

        # `hero_series` — the normalized consensus's full history (from
        # consensus_series, the single source shared with the Unified Signal
        # plot's top row). It is no longer a "hero" series in the sense of
        # being what the card plots: the card has no plot. It survives because
        # the Convergence tab's marker tiers are causal quantiles of THIS
        # distribution (see _series_tier there), which needs the history.
        #
        # `hero_smoothed` (a DDM of the same consensus) went with the TREND
        # evidence row it existed to feed.
        st.session_state["hero_series"] = (
            _consensus_full["Consensus"].rename("HeroConsensus")
            if not _consensus_full.empty else None)

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

        console.item("Mūla Engine", "✅ Cached")
        console.item("Swayam Daily", f"✅ {len(swayam_daily)} rows")
        console.item("View Bank", f"✅ {len(swayam_view_dfs)} views")
        console.item("Convergence DF", f"✅ {len(convergence_df)} rows")
        console.item("Convergence Result", f"✅ {display_signal}")

        console.end_phase("FINAL ASSEMBLY")

        console.summary("RUN SUMMARY", {
            "Total Phases": "5/5 complete",
            "Mūla Rows": len(engine.ts_data),
            "Swayam Views": len(swayam_view_dfs),
            "Swayam Trading Days": len(swayam_daily),
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
        # (audit finding F19) — see _bundle_swayam_view_dfs's docstring.
        _bundle_snapshot["swayam_view_dfs"] = _bundle_swayam_view_dfs(
            _bundle_snapshot.get("swayam_view_dfs") or {}
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
    if "fvo_ts" not in st.session_state:
        st.session_state["fvo_ts"] = ts.copy()

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
            _analogs = _fsp(ts, active_target, hold_horizons=_display_hold,
                            mom_window=_icfg.analog_mom_window)
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
                    "positive_pct": float(_row["positive_pct"]),
                    # n_eff, not n, is what the hero's precedent gate reads:
                    # ten analogs carried by one episode is one observation.
                    "n": int(round(_row.get("n_eff", _row["n"]))),
                    "n_raw": int(_row["n"]),
                    "n_eff": float(_row.get("n_eff", _row["n"])),
                    "p25": float(_row.get("p25", float("nan"))),
                    "p75": float(_row.get("p75", float("nan"))),
                    "usable": bool(_row.get("usable", True)),
                    "note": str(_row.get("note", "")),
                    "dir": 1 if _med > 0 else -1 if _med < 0 else 0,
                }
        except Exception:
            _prec_summary = None
        st.session_state["precedent_summary"] = _prec_summary
        st.session_state["_precedent_analogs_cache"] = {
            "pkey": _pkey, "periods": _cached_analogs, "display_hold": _cached_display_hold,
        }
        st.session_state["_prec_key"] = _pkey

    # ─── Hero verdict — computed once. Rendered in full only on Overview;
    # every page's top-bar/KPIs can read the SAME object, so nothing can ever
    # disagree with what Overview shows (see _compute_hero_verdict's docstring).
    verdict = _compute_hero_verdict(nishkarsh_norm, agreement, signal)

    # (A decorative "◄ CONFIGURE" arrow pointing at the sidebar rendered here.
    # It was the first thing on every page, it was amber — the caution colour,
    # spent on a signpost — and it pointed at a rail that is open by default
    # and self-evidently a control rail. Removed rather than restyled.)

    # ─── Timeframe — read here, RENDERED in a chart panel header ───────────
    # Read here (the filtered series below needs it) but RENDERED inside the
    # panel header of each page's primary chart. Reading state up here and
    # drawing the widget further down is safe and is the standard Streamlit
    # pattern: a widget interaction reruns the script, so session_state
    # already holds the new value by the time this line executes.
    #
    # Derived from TIMEFRAME_TRADING_DAYS (core/config.py) rather than a
    # second hard-coded {3M:63, 6M:126, ...} literal — the two used to drift
    # independently with no shared source (audit finding F15).
    TIMEFRAMES = {**TIMEFRAME_TRADING_DAYS, 'ALL': None}
    if st.session_state.get("tf_selected") not in TIMEFRAMES:
        st.session_state["tf_selected"] = "6M"
    selected_tf = st.session_state["tf_selected"]

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

    # ─── Command-bar status (price / change / freshness chip) — shared by
    # every page's shell. Deliberately independent of the queued freshness
    # notices (which keep their own richer explanation, rendered in the rail
    # below the bar) — this is only the one-glance summary. ────────────────
    _cb_price = _cb_chg = None
    if "Price" in ts.columns and len(ts) >= 2:
        _cb_tail = pd.to_numeric(ts["Price"], errors="coerce").tail(2)
        if pd.notna(_cb_tail.iloc[-1]):
            _cb_price = float(_cb_tail.iloc[-1])
            if pd.notna(_cb_tail.iloc[-2]) and _cb_tail.iloc[-2] != 0:
                # PERCENT POINTS, not a fraction. This was previously passed
                # as a fraction (0.0042) into a "%.2f%%" format, so a +0.42%
                # session printed as "0.00%" in the command bar — the one
                # number on screen a desk reads before any other, wrong by
                # two orders of magnitude on every render.
                _cb_chg = float(_cb_tail.iloc[-1] / _cb_tail.iloc[-2] - 1.0) * 100.0
    _cb_status_label, _cb_status_tone, _cb_asof = "", "neutral", ""
    try:
        if active_date != "None" and active_date in df.columns:
            _cb_last_dt = pd.to_datetime(ts["Date"], errors="coerce").max()
            if pd.notna(_cb_last_dt):
                _cb_today = min(datetime.now(timezone.utc).date(), datetime.now().date())
                _cb_behind = trading_days_behind(ALL_TARGETS.get(active_target), _cb_last_dt.date(), _cb_today)
                if _cb_behind >= STALENESS_DAYS:
                    _cb_status_label, _cb_status_tone = "STALE", "danger"
                elif _cb_behind >= 1:
                    _cb_status_label, _cb_status_tone = f"{_cb_behind}D BEHIND", "warning"
                else:
                    _cb_status_label, _cb_status_tone = "LIVE", "success"
                _cb_asof = _cb_last_dt.strftime("%d %b %Y")
    except Exception:
        pass

    def _top_bar(*, toolbar: bool = False) -> None:
        """The page shell, identical on every page.

        Order is fixed and means something: the TAPE (the world) sits above
        the COMMAND BAR (this instrument), which sits above the NOTICE RAIL
        (the caveats on it). Page content follows.

        The chart-window control used to dock here as a toolbar strip. It has
        moved into the panel header of each page's primary chart — a control
        that reframes a chart belongs on that chart, not in page chrome three
        elements above it.
        """
        render_ticker(data)
        render_top_bar(
            target=active_target, price=_cb_price, change_pct=_cb_chg,
            status_label=_cb_status_label, status_tone=_cb_status_tone,
            meta_items=[
                ("Window", selected_tf),
                ("Horizon", f"{FWD_HORIZON}D"),
                ("As of", _cb_asof),
            ],
            open_strip=toolbar,
        )
        render_notice_rail(_notices)

    # Error boundary wrapper — unchanged from the previous per-tab dispatch,
    # just reused per-page now instead of per-tab.
    def _safe_render(name, render_fn):
        """Render a page's content with graceful error handling."""
        try:
            render_fn()
        except Exception as e:
            st.markdown(
                f'<div class="warning-box">'
                f'<div class="interp-title">Error in {html.escape(name)}</div>'
                f'<div class="interp-body">{html.escape(str(e))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ─── App shell — one real page per analytical surface (st.navigation),
    # restyled into an institutional nav rail via theme.css. Every page below
    # is a THIN wrapper: none of them recompute the pipeline above, they only
    # call the exact same tab-render functions app.py has always called, with
    # the exact same arguments — this is a presentation-layer restructure,
    # not a change to what gets computed or when.
    # ─────────────────────────────────────────────────────────────────────
    def _page_overview() -> None:
        """Overview — scan first, then read.

        The KPI strip now leads and the conviction chain follows it. It was
        the other way round: a ~400px verdict card, then a section header,
        then the six numbers that summarise it — so the one row a returning
        user actually needs sat below the fold, under prose they had already
        read. Six numbers across the top answers "what changed since I last
        looked" in one saccade; the chain below answers "why", for the reader
        who wants it. Nothing was removed, and the numbers are the same
        objects the card is built from, so the two cannot disagree.
        """
        _top_bar()
        _swd = st.session_state.get("swayam_daily")
        _sw_os = (float(_swd["Oversold_Pct"].iloc[-1])
                  if _swd is not None and not _swd.empty and "Oversold_Pct" in _swd.columns else None)
        _cdf = st.session_state.get("convergence_df")
        _agree = (float(_cdf["agreement_ratio"].iloc[-1])
                  if _cdf is not None and not _cdf.empty and "agreement_ratio" in _cdf.columns else None)
        _fvo_val = float(signal.get("fvo", 0.0) or 0.0)
        render_kpi_strip([
            {"label": "Signal", "value": verdict["signal"], "subtext": verdict["action"]["prose"],
             "color_class": ("success" if verdict["signal_class"] == "buy"
                             else "danger" if verdict["signal_class"] == "sell" else "neutral")},
            {"label": "Conviction", "value": f"{verdict['conviction']:.2f}",
             "subtext": verdict["action"]["label"], "color_class": "accent"},
            {"label": "Walk-Forward Edge",
             "value": (f"{verdict['trust']['oos_ic']:+.3f}" if verdict["trust"].get("oos_ic") is not None else "—"),
             "subtext": verdict["trust"]["chip"], "color_class": "info"},
            {"label": "Mispricing", "value": f"{_fvo_val:+.2f}σ",
             "subtext": f"{abs(float(signal.get('pct_mispricing', 0.0) or 0.0)) * 100:.1f}% vs fair value",
             "color_class": "success" if _fvo_val < 0 else "danger" if _fvo_val > 0 else "neutral"},
            {"label": "Swayam Breadth", "value": (f"{_sw_os:.0f}%" if _sw_os is not None else "—"),
             "subtext": "oversold share", "color_class": "neutral"},
            {"label": "Engine Agreement", "value": (f"{_agree:.0%}" if _agree is not None else "—"),
             "subtext": "Mūla vs Swayam", "color_class": "neutral"},
        ], max_cols=6)
        render_section_header(
            "Conviction Chain",
            "One directional claim from Mūla, then every condition that can invalidate it. "
            "Conviction is their product, so the smallest gate is the binding constraint.",
            icon="target", accent="accent",
        )
        render_hero_card(verdict)

    def _page_fvo() -> None:
        _top_bar()
        _safe_render("Mūla", lambda: render_mula_tab(
            engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target))

    def _page_swayam() -> None:
        _top_bar()
        _safe_render("Swayam", lambda: render_swayam_tab(selected_tf=selected_tf))

    def _page_convergence() -> None:
        _top_bar()
        _safe_render("Convergence", lambda: render_convergence_tab(ts_filtered))

    def _page_precedent() -> None:
        _top_bar()
        # Reuse the analog list already computed above (Precedent base-rate for
        # the hero) instead of having the tab call find_similar_periods a second
        # time for the same (ts, target, mom_window) — audit finding F18. Guarded
        # on the pkey matching THIS render's ts/target/horizon; a mismatch
        # (shouldn't happen since the precompute above always runs first) falls
        # back to None, and the tab recomputes itself exactly as before.
        _prec_cache = st.session_state.get("_precedent_analogs_cache")
        _cached_periods = (
            _prec_cache["periods"] if _prec_cache and _prec_cache.get("pkey") == _pkey else None
        )
        # Precedent term structure + momentum/horizon come from this
        # instrument's own config (precedent_horizons / analog_mom_window /
        # forecast_horizon).
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.analog_mom_window, _icfg.forecast_horizon,
            precomputed_periods=_cached_periods))

    def _page_diagnostics() -> None:
        _top_bar()
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(
            engine, ts_filtered, x_axis, x_title, signal, model_stats))

    def _page_data() -> None:
        _top_bar()
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    pages = {
        "": [st.Page(_page_overview, title="Overview", icon=":material/dashboard:", default=True)],
        # Convergence leads: it is the read that combines the other two, so it
        # is the one a returning user opens first. Mūla and Swayam follow as
        # its inputs, Precedent as the independent check on all three.
        "Engines": [
            st.Page(_page_convergence, title="Convergence", icon=":material/merge_type:"),
            st.Page(_page_fvo, title="Mūla", icon=":material/monitoring:"),
            st.Page(_page_swayam, title="Swayam", icon=":material/hub:"),
            st.Page(_page_precedent, title="Precedent", icon=":material/history:"),
        ],
        "System": [
            st.Page(_page_diagnostics, title="Diagnostics", icon=":material/monitor_heart:"),
            st.Page(_page_data, title="Data", icon=":material/table_chart:"),
        ],
    }
    st.navigation(pages, position="sidebar").run()

    _render_footer()


if __name__ == "__main__":
    main()
