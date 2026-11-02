"""
Tattva — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — one claim, graded. Mūla values the instrument against the whole macro
cross-section; Swayam reads its own internals; a conviction chain of independent
gates says how far the claim can be trusted — across commodities, FX, crypto,
equity indices and listed stocks.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os

# ── BLAS thread pinning (MUST run before numpy import) ────────────────────────
# The recursive engines solve thousands of small linear systems in sequence
# (one per session per discount-grid member). On Streamlit Community Cloud the
# container is throttled to ~1 shared vCPU but the host reports many logical
# CPUs, so OpenBLAS/MKL spawn one thread per reported core and thrash on every
# tiny solve. One thread per process is strictly faster for many-small-matrix
# workloads like this.
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
    format_conviction,
    window_cutoff,
    edge_periods,
    GATE_TITLES,
)
from ui.tabs.tab_mula import render_mula_tab
from ui.tabs.tab_swayam import render_swayam_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_ohlcv, fetch_commodity_dataset, fetch_stock_target_series, is_yfinance_ticker
from data.calendars import trading_days_behind, is_session, session_mask, resolve_exchange
from data.universe import resolve_stock_symbol

# ── Engines ──────────────────────────────────────────────────────────────────
# Mūla (valuation, with its error-correction layer) and Swayam (breadth). There
# is no engine selector and no fallback: the import guard that used to sit
# here could not do what it promised — FairValueEngine lives in the same
# package, so a broken engines.mula failed on the line above the guard.
from engines.mula import MulaEngine, FairValueEngine
from engines.mula.blocks import block_membership
from engines.swayam import aggregate_views
from engines.swayam.kernel import view_skill_weights

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import MIN_DATA_POINTS, STALENESS_DAYS, SESSION_FRESH_FLOOR, ALL_TARGETS, TARGET_CATEGORIES, is_stock_target, STOCK_TARGET_MARKETS, FORECAST_HORIZON, RAW_YIELD_PREDICTORS, DIV_LOOKBACK, TIMEFRAME_TRADING_DAYS, swayam_macro_columns, FREEFORM_STOCK_CATEGORIES, register_stock_target, get_instrument_config
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
#
# THE RULE: every per-target value a page reads is either one of these keys or
# derived from one. `mula_summary` used to sit outside the bundle as a parallel
# copy, so a cached switch-back showed the PREVIOUS target's error-correction
# read on the hero — the ECM gate, and so the conviction, belonged to the
# wrong instrument. It is now read off the bundled engine.
_BUNDLE_KEYS = (
    "engine", "fvo_ts", "swayam_daily", "swayam_view_dfs", "swayam_view_weights",
    "swayam_n_eff", "swayam_native_last",
    "convergence_df", "divergence_events", "hero_series",
    "intelligence_active_weights", "intelligence_active_profile",
    "edges",
)
# Keep the last N configs. Each entry is a full pipeline result; 6 covers a
# session that browses a handful of targets without recomputing.
_RESULTS_CACHE_MAX = 6

#: Every session key a run produces or caches — the ONE list Start over and
#: Refresh clear. They used to keep two hand-maintained lists that had already
#: drifted (Refresh cleared the Swayam caches, Reset did not, so a Reset an
#: hour later paired fresh Mūla data with stale Swayam frames).
_SESSION_RESULT_KEYS = _BUNDLE_KEYS + (
    "engine_cache", "fvo_engine", "fvo_fit_key", "results_cache",
    "precedent_summary", "_prec_key", "_precedent_analogs_cache", "_analog_skill_cache",
    "_swayam_fetch_cache", "_swayam_analysis_cache",
)

#: The Swayam tab's drill-down reads only these per-view columns. The kernel
#: emits ~270 per view; at 15 views × ~2,300 rows that is ~76 MB per target,
#: held in the live session, the Swayam analysis cache and up to six result
#: bundles — the likely cause of hosted sessions being dropped under memory
#: pressure. Everything that needs the full width (skill weights, aggregation,
#: the effective view count) has consumed it before the frames are stored.
_SWAYAM_DRILLDOWN_COLS = (
    "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
    "Regime", "Vol_Regime", "Change_Point", "Confidence",
)


def _slim_view_dfs(dfs: dict) -> dict:
    """The per-view frames trimmed to what the drill-down displays."""
    out = {}
    for name, df in (dfs or {}).items():
        cols = [c for c in _SWAYAM_DRILLDOWN_COLS if c in df.columns]
        out[name] = df[cols] if cols else df.iloc[:, :0]
    return out


def _clear_results() -> None:
    """Drop every run product from the session (the data frame is kept)."""
    for _k in _SESSION_RESULT_KEYS:
        st.session_state.pop(_k, None)
    for _k in [k for k in list(st.session_state) if str(k).startswith("conv_norm_causal::")]:
        st.session_state.pop(_k, None)


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
    per-target dropna downstream. Joined to a LOCAL copy only: it used to be
    written back into st.session_state['data'], after which every later
    target in the session — Gold included — carried the stock as a predictor
    and every one of its published fair values moved. The series fetch is
    cached, so re-joining on each rerun costs nothing.
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
        tagline="Cross-Asset Fair Value · Self-Referential Breadth · One Graded Call",
    )
    if frame is not None:
        render_ticker(frame)


#: The three systems, as the cold-start screen describes them. Data, not
#: markup — the landing page renders them through one template, so the three
#: panels cannot drift apart in structure the way three hand-written HTML
#: blocks did (they already had: two said "Ensemble/Signal", one said
#: "Fusion", and the label column was a <span> in one and a <div> in another).
_SYSTEM_PANELS = (
    ("mula", "System 01", "Mūla", "The world's valuation",
     "Prices the target against the whole traded macro cross-section with a dynamic "
     "cointegrating regression on log price, then asks the question a trade depends on: "
     "does price, not just fair value, close the gap? It makes the one call Tattva issues.",
     (("Estimator", "Recursive discounted DLM"),
      ("Equilibrium", "MP factors + asset blocks"),
      ("Reversion", "Price-side error correction"))),
    ("swayam", "System 02", "Swayam", "The instrument's own read",
     "Reads the instrument's own price through a bank of views spanning timescale, "
     "information set and mechanism, and weights each view by its own realised skill. "
     "It confirms or contradicts the call from the inside.",
     (("Signal", "MSF + MMR oscillator"),
      ("Breadth", "Net oversold share of views"),
      ("Regime", "HMM \u00b7 GARCH \u00b7 CUSUM"))),
    ("convergence", "System 03", "Convergence", "Agreement, measured",
     "Scores the two systems against each other across direction, breadth, magnitude and "
     "regime, with weights learned forward from each dimension's signed skill — and grades "
     "the result the way every engine is graded.",
     (("Fusion", "Mūla \u00d7 Swayam"),
      ("Weights", "Learned online, causal"),
      ("Graded", "Walk-forward IC"))),
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
    from core.config import TARGET_CATEGORIES, CATALOGUE_TARGETS

    _n_cat, _n_tgt = len(TARGET_CATEGORIES), len(CATALOGUE_TARGETS)

    # ── The proposition ───────────────────────────────────────────────────
    st.markdown(
        """<div class="lede">
  <div class="lede-claim">One engine values the instrument against the world, a second
    reads its own price from fifteen angles, and Tattva makes one call — graded by six
    independent conditions, with the one that limits it named.</div>
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
             "subtext": "Commodities, FX, crypto, India and US indices, and any listed "
                        "stock by symbol"},
            {"label": "Catalogue Targets", "value": str(_n_tgt),
             "subtext": "Every one runnable on free data; each carries its own "
                        "configuration, defaulting to its asset class's"},
            {"label": "Daily History Per Run", "value": "~9y",
             "subtext": "Every published value uses only data available on its own "
                        "date, so re-running extends the record instead of rewriting it"},
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
        ("One graded call", "Cheap, rich or fair against the cross-section — and six "
                            "independent conditions, the weakest named as the limit."),
        ("A measured edge", "Every engine's lean scored on non-overlapping windows at the "
                            "call's horizon: the honest answer to whether it has paid."),
        ("A base rate", "What followed the most similar past states of the system — no "
                        "model fitted to returns, matched on the engines' own reading."),
        ("The record", "Every series behind the call, joined by date and exportable."),
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


def _recent_divergences(window: int) -> int:
    """Divergence events within the last ``window`` SCORED sessions.

    Anchored on the newest scored session, not on the newest event: the count
    used to run 1.5×window calendar days back from the latest EVENT, so a
    burst two years ago was reported as "N events in the last 20 sessions".
    """
    events = st.session_state.get("divergence_events")
    cdf = st.session_state.get("convergence_df")
    if events is None or not len(events) or cdf is None or cdf.empty:
        return 0
    sessions = pd.DatetimeIndex(pd.to_datetime(cdf.index, errors="coerce")).dropna().sort_values()
    if not len(sessions):
        return 0
    start = sessions[max(0, len(sessions) - int(window))]
    if "date" in getattr(events, "columns", ()):
        when = pd.to_datetime(events["date"], errors="coerce")
    elif isinstance(events.index, pd.DatetimeIndex) or events.index.inferred_type in (
            "string", "date", "datetime", "datetime64"):
        # A positional index must never be parsed as dates — pd.to_datetime
        # reads bare integers as epoch nanoseconds and every row "clears" 1970.
        when = pd.to_datetime(pd.Series(events.index), errors="coerce")
    else:
        return 0
    return int((when >= start).sum())


def _compute_hero_verdict(signal: dict) -> dict:
    """Gather the run's readings and build the conviction chain. Pure assembly.

    All interpretation lives in ``ui.components.build_hero_verdict`` (pure,
    unit-tested in tests/test_hero_verdict.py); this only collects inputs.
    Computed once per rerun, so every page's command bar and KPI strip read
    the identical object Overview renders and cannot disagree with it.
    """
    from analytics.analogs import MIN_EFFECTIVE_N
    try:
        horizon = get_instrument_config(st.session_state.get("active_target", "")).forecast_horizon
    except KeyError:
        horizon = FORECAST_HORIZON

    # Swayam's current breadth — read from the same frame the Swayam page
    # shows. A tail that is carried forward (no native Swayam row) still
    # counts: the Convergence page discloses the carry.
    _sw = st.session_state.get("swayam_daily")
    swayam_breadth = None
    if _sw is not None and not _sw.empty:
        _last = _sw.iloc[-1]
        swayam_breadth = {"oversold_pct": float(_last.get("Oversold_Pct", 0.0)),
                          "overbought_pct": float(_last.get("Overbought_Pct", 0.0))}
    _cdf = st.session_state.get("convergence_df")
    agreement = (float(_cdf["agreement_ratio"].dropna().iloc[-1])
                 if _cdf is not None and not _cdf.empty and "agreement_ratio" in _cdf.columns
                 and _cdf["agreement_ratio"].notna().any() else None)

    return build_hero_verdict(
        mula=signal,
        swayam_breadth=swayam_breadth,
        agreement=agreement,
        edge=(st.session_state.get("edges") or {}).get("mula"),
        precedent=st.session_state.get("precedent_summary"),
        n_divergences=_recent_divergences(DIV_LOOKBACK),
        horizon_days=int(horizon),
        div_window=DIV_LOOKBACK,
        min_precedent_n=MIN_EFFECTIVE_N,
    )


def _render_model_passport_sidebar() -> None:
    """Rail readout — what the run reached, not a control.

    Two facts worth a glance from any page: the dimension weights the
    convergence learner arrived at, and what Mūla's call has been worth out of
    sample. Nothing is persisted between runs, so there is no profile to load,
    compare or reset — only the state this run reached.
    """
    st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
    w = st.session_state.get("intelligence_active_weights") or {}
    edge = (st.session_state.get("edges") or {}).get("mula") or {}
    if not w and not edge:
        render_control_hint("Run an analysis to populate.")
        return
    rows = [(f"{k.capitalize()} weight", f"{v:.2f}", "accent")
            for k, v in sorted(w.items(), key=lambda kv: -kv[1])]
    ic = edge.get("ic")
    if ic is not None and np.isfinite(ic):
        rows.append(("Mūla edge (IC)", f"{ic:+.3f}", "long" if ic > 0 else "short"))
        _pos, nb = edge_periods(edge)
        if nb:
            rows.append(("Periods +", f"{_pos}/{nb}", ""))
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
        prev_commodity = st.session_state.get("active_target", all_names[0])
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
                _clear_results()
                st.session_state["data"] = df
                st.session_state["active_target"] = selected_commodity
                st.session_state["run_analysis"] = True
                st.rerun()
        else:
            df = st.session_state["data"]
            # Post-load target switch — re-runs the engines on the already
            # fetched universe. No macro re-fetch: both engines read the one
            # panel in session; only the new target's own OHLCV is fetched.
            if selected_commodity != st.session_state.get("active_target"):
                if st.button(f"Switch → {selected_commodity}", type="primary",
                             width="stretch"):
                    st.session_state["active_target"] = selected_commodity
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

    # A stock target's price column is joined here — BEFORE numeric_cols are
    # computed below — so the target resolves to itself on its first render
    # instead of falling back to some other column. Local to this run.
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

        st.session_state["active_date_col"] = date_col

        _render_model_passport_sidebar()

        if st.session_state.get("run_analysis"):
            st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
            # One per row, each stretched to the rail width — the same shape as
            # the selectboxes above them.
            _do_refresh = st.button(
                "Refresh data", type="secondary", width="stretch",
                help="Fetch the live universe now (bypassing the one-hour cache), then "
                     "recompute. Use when the data is stale or partial.")
            _do_restart = st.button(
                "Start over", type="secondary", width="stretch",
                help="Clear this session's results and return to the start screen. "
                     "Data fetched within the last hour is reused on the next run.")
            # (The second button was labelled "Reset" and promised "re-run both
            # engines on the data already in session — no network fetch". It
            # dropped the data and returned to the landing page. Re-running
            # deterministic engines on identical data would reproduce the same
            # numbers anyway, so the honest control is the one it actually was.)
            if _do_restart:
                _clear_results()
                for _k in ("data", "run_analysis"):
                    st.session_state.pop(_k, None)
                st.rerun()

            # One forced live pull of the whole universe; the target's own OHLCV
            # is forced once on the recompute that follows. A failed or
            # stale-snapshot pull is said out loud on the next render rather than
            # leaving the user believing they refreshed.
            if _do_refresh:
                progress_bar(progress_container, 3, "Re-fetching Live Market Data",
                             "yfinance · full universe · bypassing cache · ~30–60s")
                _rend = pd.Timestamp.today()
                _rdf, _rerr = fetch_commodity_dataset(
                    _rend - pd.DateOffset(days=365 * 9), _rend, force=True)
                _src = ((_rdf.attrs or {}).get("source") if _rdf is not None else None)
                if _rdf is not None and _src != "stale-snapshot":
                    progress_bar(progress_container, 15, "Live Data Refreshed",
                                 f"{_rdf.shape[1]} series × {_rdf.shape[0]} rows · recomputing…")
                    st.session_state["data"] = _rdf
                    st.session_state["_force_ohlcv"] = True
                else:
                    progress_container.empty()
                    _last = pd.to_datetime(df["DATE"], errors="coerce").max() if "DATE" in df.columns else None
                    st.session_state["_pending_notices"] = [{
                        "kind": "warning", "title": "Refresh failed",
                        "body": (f"The live pull did not complete ({_rerr or 'the source served a stale snapshot'}). "
                                 f"Showing last-good data"
                                 + (f" through {_last:%d %b %Y}" if _last is not None and pd.notna(_last) else "")
                                 + ". Try again once yfinance recovers."),
                    }]
                    if _rdf is not None:
                        st.session_state["data"] = _rdf
                # Recompute only when the data actually changed. A pull that
                # returned nothing leaves the session's frame as it was, and
                # re-running deterministic engines on it would reproduce the
                # same numbers after ~30 s of work.
                if _rdf is not None:
                    _clear_results()
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
    # The valuation panel is the WHOLE macro cross-section, minus the target,
    # its self-replicating near-duplicates (e.g. GLTR for a precious metal,
    # the US equity complex for a US index) and any free-form stock. There is
    # no predictor picker: the Mūla engine prices the target against the
    # traded opportunity set, and hand-deselecting instruments from that set
    # does not make it a better opportunity set — it makes it a smaller one
    # with an undocumented reason. The exclusions are read from ONE place, the
    # instrument's own config (`excluded_predictors`); the parallel global map
    # it is built from is no longer consulted here.
    _excluded_feats = {active_target, *_icfg.excluded_predictors}
    active_features = [c for c in numeric_cols
                       if c not in _excluded_feats and c not in STOCK_TARGET_MARKETS]
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

    # Notices queued by an action on a PREVIOUS run (a failed Refresh). Read,
    # not consumed, here: a run that recomputes ends in st.rerun() and never
    # renders, so the notice is dropped only once a page has actually shown it
    # (just before the page renders, below).
    for _pn in st.session_state.get("_pending_notices") or []:
        _notice(_pn["kind"], _pn["title"], _pn["body"])

    # ONE freshness reading, shared by these notices and the command bar's
    # status chip. They used to be computed separately — the notice from the
    # panel's newest date (any market), the chip from the model spine (the
    # target's sessions only) — so on a target-market holiday the notice said
    # "as of 16 Oct" while the chip said 15 Oct. The signal is as of the
    # target's last session, so that is the date both read.
    #
    # Measured in TRADING days behind on the target's own exchange calendar
    # (holiday-aware when exchange_calendars is installed, Mon–Fri otherwise),
    # so Friday data read on a Sunday is current, not stale. "Today" is the
    # EARLIER of UTC and machine-local: a UTC-hosted deploy rolls past midnight
    # ahead of an IST/EST exchange, and taking the earlier date means a
    # time-zone skew can never overstate staleness.
    _tgt_ticker = ALL_TARGETS.get(active_target)
    today = min(datetime.now(timezone.utc).date(), datetime.now().date())
    _fresh = {"as_of": None, "behind": None, "label": "", "tone": "neutral"}
    if active_date != "None" and active_date in df.columns:
        try:
            _all_dates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True)
            _mask = _all_dates.notna()
            if active_target in df.columns:
                _mask &= pd.to_numeric(df[active_target], errors="coerce").notna()
            _cand = _all_dates[_mask]
            _sm = session_mask(_tgt_ticker, _cand) if len(_cand) else np.zeros(0, dtype=bool)
            _dates = _cand[_sm] if _sm.any() else _cand
            if len(_dates):
                latest_date = _dates.max().to_pydatetime().replace(tzinfo=None)
                behind = trading_days_behind(_tgt_ticker, latest_date.date(), today)
                _fresh.update(as_of=latest_date, behind=behind)
                if behind >= STALENESS_DAYS:
                    _fresh.update(label="STALE", tone="danger")
                elif behind >= 1:
                    _fresh.update(label=f"{behind}D BEHIND", tone="warning")
                else:
                    # "Current", not "live": this is end-of-day data that can be
                    # hours old; it is the newest session the source has.
                    _fresh.update(label="CURRENT", tone="success")
                ds = latest_date.strftime("%d %b %Y")
                if behind >= STALENESS_DAYS:
                    _notice(
                        "warning", "Latest data unavailable",
                        (f"Newest data is {ds} — {behind} trading days behind. The price source "
                         f"(yfinance) hasn't published more recent data, so every signal below "
                         f"reflects {ds}, not today. Use Refresh data in the rail once the "
                         f"source updates."),
                    )
                elif behind >= 1:
                    _notice(
                        "info", "Data freshness",
                        f"Signals are as of {ds} ({behind} trading day"
                        f"{'s' if behind > 1 else ''} behind — today's bar may not be published yet).",
                    )

                # Predictors refilled from a cached snapshot (data.fetcher's
                # rate-limit recovery). Read off the frame itself, so a frame
                # served from cache carries its own warning — it used to live in
                # a per-session registry that a cached frame arrived without.
                _stale_backfills = (df.attrs or {}).get("stale_backfills") or {}
                if _stale_backfills:
                    _sb_items = sorted(_stale_backfills.items(), key=lambda kv: kv[1])
                    _sb_preview = ", ".join(f"{k} (as of {v})" for k, v in _sb_items[:5])
                    _sb_more = f" +{len(_sb_items) - 5} more" if len(_sb_items) > 5 else ""
                    _notice(
                        # WARNING, not info: a snapshot-reconstructed column is
                        # reindexed onto this run's calendar and ffilled, so it is
                        # a DIFFERENT SERIES from a live pull, not a staler one —
                        # and every published date can move with it.
                        "warning", "Predictors carried from snapshot — history differs",
                        f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                        f"from a prior cached snapshot: {_sb_preview}{_sb_more}. A rebuilt column "
                        f"is not merely staler than a live one — it is a different series, so "
                        f"this run's published history may differ from a run that fetched "
                        f"cleanly. Compare the panel fingerprint before treating past values as "
                        f"final; refresh once the source is healthy for an authoritative record.",
                    )

                # Session completeness (exchange-aware): of the inputs whose market
                # was OPEN on the panel's newest date, how many actually posted a
                # fresh value rather than a carried-forward one? Markets that were
                # closed that day are legitimately carried and excluded, so a
                # foreign holiday does not trip a false "partial session".
                num = df.select_dtypes(include=[np.number])
                _panel_last = _all_dates.max()
                if len(num) >= 2 and pd.notna(_panel_last):
                    cols = list(num.columns)
                    last_r = num.iloc[-1].to_numpy(dtype=np.float64)
                    prev_r = num.iloc[-2].to_numpy(dtype=np.float64)
                    finite = np.isfinite(last_r) & np.isfinite(prev_r)
                    should_post = np.array(
                        [is_session(_COLUMN_TICKERS.get(c), _panel_last.date()) for c in cols])
                    judged = finite & should_post
                    denom = int(judged.sum())
                    if denom >= 3:
                        fresh_frac = float(((last_r != prev_r) & judged).sum() / denom)
                        # A session still in progress is expected to be mostly
                        # carried forward — that is not a data problem.
                        if fresh_frac < SESSION_FRESH_FLOOR and _panel_last.date() < today:
                            _notice(
                                "warning", "Partial latest session",
                                (f"Only {fresh_frac:.0%} of the markets open on "
                                 f"{_panel_last:%d %b %Y} have posted — the rest are forward-filled "
                                 f"from the prior session, so the inputs behind the latest reading "
                                 f"are stale. Treat it as provisional; refresh once those markets post."),
                            )

                # The target's own last genuine print. Its column is forward-filled
                # across days it did not post, and continuous prices do not repeat
                # exactly, so a run of identical closes at the tail is a carried
                # value. One rule for every source — the sheet targets used to be
                # re-fetched here, on every rerun, to answer the same question.
                if active_target in df.columns:
                    tv = pd.to_numeric(df[active_target], errors="coerce").to_numpy()
                    j = len(tv) - 1
                    while j > 0 and np.isfinite(tv[j]) and tv[j] == tv[j - 1]:
                        j -= 1
                    t_last = (pd.Timestamp(_all_dates.iloc[j]).to_pydatetime()
                              if j < len(tv) - 1 and pd.notna(_all_dates.iloc[j]) else None)
                    if t_last is not None:
                        t_behind = trading_days_behind(_tgt_ticker, t_last.date(), today)
                        if t_behind >= 1:
                            if is_session(_tgt_ticker, today) and t_behind == 1:
                                _notice(
                                    "info", f"{active_target} price not yet updated",
                                    (f"Today's bar is carried forward from {t_last:%d %b %Y} — the "
                                     f"market is open but the last fetch did not return a new price "
                                     f"for it. Refresh data in the rail to pull the latest."),
                                )
                            else:
                                _notice(
                                    "warning", f"{active_target} data is lagging",
                                    (f"This target last updated {t_last:%d %b %Y} ({t_behind} trading "
                                     f"day{'s' if t_behind > 1 else ''} behind the macro universe) — "
                                     f"later rows carry that value forward, so its latest reading may "
                                     f"be stale."),
                                )
        except Exception as _fe:                      # noqa: BLE001 - never break a run to report
            console.warning(f"Freshness check skipped: {type(_fe).__name__}: {_fe}")

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

        # ── Phase 1: Data Acquisition ─────────────────────────────────────
        # HORIZON-INDEPENDENT: Swayam's inputs depend on the target and on the
        # panel in session, never on the scoring horizon, so they are cached
        # separately and survive a horizon change. The key carries the data's
        # identity (date range + panel fingerprint): it used to carry the target
        # alone, so Swayam frames from an earlier pull were reused against a
        # newer Mūla panel after any re-run that did not go through Refresh.
        #
        # ONE panel feeds both engines. Swayam's MMR driver pool used to be a
        # second fetch of the macro universe, keyed by TICKER — so the leakage
        # guard below, which drops the target and its near-replicas by NAME,
        # matched nothing (0 drivers dropped for every target; GLTR at r = 0.985
        # stayed in Gold's pool), and after the cache's hour the two engines
        # read two different pulls. The drivers are now the same friendly-named
        # frame Mūla values against.
        _fp = (df.attrs or {}).get("panel_fingerprint", "")
        _swayam_fetch_key = f"swayam_fetch::{active_target}|{_date_range}|{_fp}"
        _nf_cache = st.session_state.get("_swayam_fetch_cache")
        console.start_phase("DATA ACQUISITION", 1, 5)
        if _nf_cache is not None and _nf_cache.get("key") == _swayam_fetch_key:
            target_ohlcv = _nf_cache["target_ohlcv"]
            swayam_macro_df = _nf_cache["swayam_macro_df"]
            macro_cols_list = _nf_cache["macro_cols_list"]
            console.item("Target OHLCV / drivers", "reused (same target, same panel)")
            progress_bar(progress_container, 19, "Data Acquisition Reused", f"{len(macro_cols_list)} drivers (cached)")
        else:
            progress_bar(progress_container, 16, "Resolving Swayam Source",
                         f"{active_target} · own OHLCV (self-referential view bank)")
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=365 * 9)

            console.section("Target OHLCV")
            # Swayam reads the target's own OHLC (and volume where it exists —
            # the volume-dependent views abstain when it does not). A
            # sheet-sourced target has no OHLCV feed at all; it is said once,
            # here, instead of being sent to yfinance as a guaranteed failure.
            _tgt_ticker = ALL_TARGETS[active_target]
            target_ohlcv = None
            if is_yfinance_ticker(_tgt_ticker):
                progress_bar(progress_container, 18, "Fetching Target OHLCV", f"yfinance · {_tgt_ticker}")
                _ohlcv = fetch_ohlcv([_tgt_ticker], start_date, end_date,
                                     force=bool(st.session_state.pop("_force_ohlcv", False)))
                target_ohlcv = _ohlcv.get(_tgt_ticker)
            if target_ohlcv is not None and not target_ohlcv.empty:
                console.item("Symbol", _tgt_ticker)
                console.item("Rows", len(target_ohlcv))
                console.item("Has Volume", bool("Volume" in target_ohlcv.columns
                                                and target_ohlcv["Volume"].fillna(0).abs().sum() > 0))
                console.success(f"Target OHLCV: {len(target_ohlcv)} rows")
            elif not is_yfinance_ticker(_tgt_ticker):
                console.warning(f"{active_target} has no OHLCV source ({_tgt_ticker}) — Swayam unavailable")
            else:
                console.warning(f"No OHLCV for {_tgt_ticker} — Swayam breadth will be unavailable")

            console.section("Swayam Drivers")
            _dcols = [c for c in df.select_dtypes(include=[np.number]).columns
                      if c not in STOCK_TARGET_MARKETS or c == active_target]
            if "DATE" in df.columns:
                swayam_macro_df = df[["DATE"] + _dcols].copy()
                swayam_macro_df.index = pd.to_datetime(swayam_macro_df.pop("DATE"), errors="coerce").dt.normalize()
                swayam_macro_df = swayam_macro_df[~swayam_macro_df.index.isna()]
                swayam_macro_df = swayam_macro_df[~swayam_macro_df.index.duplicated(keep="last")]
            else:
                swayam_macro_df = pd.DataFrame()
            macro_cols_list = list(swayam_macro_df.columns)
            console.item("Driver pool", f"{len(macro_cols_list)} panel columns (the panel Mūla reads)")
            progress_bar(progress_container, 19, "Data Acquisition Complete", f"{len(macro_cols_list)} drivers")
            st.session_state["_swayam_fetch_cache"] = {
                "key": _swayam_fetch_key,
                "target_ohlcv": target_ohlcv,
                "swayam_macro_df": swayam_macro_df,
                "macro_cols_list": macro_cols_list,
            }
        console.end_phase("DATA ACQUISITION")

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
        console.item("Lookback Windows", f"{tuple(_icfg.fvo_lookback_windows)}")

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
            engine = MulaEngine()
            console.item("Valuation Core", "Mūla — cointegrating core + error-correction layer")
            # `config=_icfg` threads this instrument's own Mūla knobs (burn-in,
            # print floor, discount grid, lookback windows, horizon).
            engine.fit(
                _tgt_px, _expl_px,
                feature_names=active_features, config=_icfg,
                progress_callback=lambda pct, msg: progress_bar(
                    progress_container, int(20 + pct * 20), "Running Mūla Engine", msg),
            )
            # Carry the raw price LEVEL on the engine output too. `Actual` is
            # already the price here, but the analog matcher and the edge
            # scorer both key off a column literally named "Price".
            engine.ts_data["Price"] = _price_level
            st.session_state["fvo_engine"] = engine
            st.session_state["fvo_fit_key"] = cache_key

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()

        def _fmt(v, spec, none="n/a"):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return none
            return format(v, spec) if np.isfinite(v) else none

        console.section("Engine Results")
        console.item("Valuation", f"{sig.get('valuation_state', '?')} · oscillator "
                     f"{_fmt(sig.get('fvo'), '+.2f')}σ ({_fmt((sig.get('pct_mispricing') or 0) * 100, '+.2f')}% vs fair value)")
        console.item("Fair Value", f"{_fmt(sig.get('fair_value'), ',.2f')} vs price {_fmt(sig.get('actual'), ',.2f')}")
        if sig.get("asof_row_withheld"):
            console.warning(f"Latest session not valued ({sig.get('withheld_reason')}) — reading is the last valued session")
        console.item("OOS R²", f"{_fmt(stats.get('r2_oos'), '.3f')} (log price vs fitted level)")
        console.item("R² vs Trailing Mean", f"{_fmt(stats.get('r2_vs_anchor'), '+.3f')} (edge over a 252d anchor)")
        console.item("Model Spread", f"{_fmt((sig.get('model_spread') or 0) * 10000, '.1f')} bps (SD of the fair-value level)")
        console.item("Cross-Section", f"{_fmt(sig.get('xs_consistency'), '.2f')} (leave-one-block-out sign agreement)")
        console.item("Price Reverts", f"P={_fmt(sig.get('mr_prob'), '.2f')} · κ̂ {_fmt(sig.get('mula_kappa'), '+.4f')}/d · "
                     f"expected {FWD_HORIZON}d move {_fmt(sig.get('expected_drift_pct'), '+.2f')}% "
                     f"± {_fmt(sig.get('drift_sd_pct'), '.2f')}%")
        console.item("Gap Half-Life", f"{_fmt(sig.get('gap_half_life'), '.0f')}d (online AR1) · "
                     f"OU {_fmt(sig.get('ou_half_life'), '.0f')}d (full history)")
        _adf = sig.get("adf_pvalue")
        console.item("Gap Stationarity", f"ADF p={_fmt(_adf, '.4f')}"
                     + (" (stationary)" if _adf is not None and np.isfinite(_adf) and _adf < 0.05
                        else " (unit root not rejected)"))
        # Name any gap the reader can see on the chart. A break is never "the
        # value was zero" — it is the system declining to publish — and the
        # causes want different responses. Only "incomplete fetch" is worth
        # re-running for, so it is the only one reported as a warning. (This
        # report read `fvo_ts` a hundred lines before it was assigned, so it
        # raised UnboundLocalError into a bare `except: pass` on every run and
        # the warning it exists for never once printed.)
        try:
            _wr = engine.ts_data.get("WithheldReason")
            if _wr is not None:
                _wr = _wr.astype(str)
                _c = _wr[_wr != ""].value_counts()
                if len(_c):
                    console.item("Gaps in the trace", " · ".join(f"{k}: {int(v)}" for k, v in _c.items()))
                _bad = int(_c.get("incomplete fetch", 0))
                if _bad:
                    _dates = pd.DatetimeIndex(_cal)[np.asarray(_wr == "incomplete fetch")]
                    _shown = ", ".join(f"{d:%Y-%m-%d}" for d in list(_dates)[-3:])
                    console.warning(
                        f"{_bad} session(s) withheld as INCOMPLETE FETCH ({_shown}). The calendar "
                        f"says those exchanges were open, so this is a data gap in this run rather "
                        f"than a market closure — refresh to fill it."
                    )
        except Exception as _ge:                # noqa: BLE001 - never break a run to report
            console.warning(f"Gap report skipped: {type(_ge).__name__}: {_ge}")

        # `n_available` is the count that PRINTED today, not the count admitted
        # past the print floor — early in a session that is a fraction of it.
        console.item("Factors", f"k={sig.get('k_factors')} above the MP edge · "
                                f"{sig.get('n_available')} instruments printed so far today")
        console.item("Market Regime", f"{sig.get('market_regime')} (stress pct {_fmt(sig.get('stress'), '.2f')})")
        console.success(f"Mūla engine complete | {len(engine.ts_data)} output rows "
                        f"({int(engine.ts_data['Valid'].sum())} valued, {engine.min_train_size} burn-in)")
        console.end_phase("MŪLA ENGINE")
        progress_bar(progress_container, 40, "Mūla Engine Complete",
                     f"{sig.get('valuation_state', '')} · {_fmt(sig.get('fvo'), '+.2f')}σ")

        # ── Phase 3: Swayam Breadth ───────────────────────────────────────
        # HORIZON-INDEPENDENT (audit finding F17): the view bank and its
        # aggregation depend only on the target's own OHLCV + the macro driver
        # window, never on the scoring horizon. Cached under the SAME
        # _swayam_fetch_key as Phase 1; only the target-calendar reindex below
        # (cheap — no yfinance calls) re-runs.
        console.start_phase("SWAYAM ENGINE", 3, 5)
        # Per-target: reset before this target fills it, so a target with no
        # Swayam read never bundles the previous target's date.
        st.session_state["swayam_native_last"] = None
        progress_bar(progress_container, 42, "Running Swayam Engine",
                     "MSF+MMR+Regime · self-referential view bank")

        _na_cache = st.session_state.get("_swayam_analysis_cache")
        if _na_cache is not None and _na_cache.get("key") == f"{_swayam_fetch_key}|h{FWD_HORIZON}":
            swayam_view_dfs = _na_cache["swayam_view_dfs"]
            swayam_daily_pre_reindex = _na_cache["swayam_daily_pre_reindex"]
            st.session_state["swayam_n_eff"] = _na_cache.get("n_eff")
            st.session_state["swayam_view_weights"] = _na_cache.get("view_weights")
            console.item("View Bank", "reused cached fit (same target, panel and horizon)")
            progress_bar(progress_container, 74, "Swayam Engine Reused", f"{len(swayam_view_dfs)} Views (cached)")
        else:
            swayam_daily_pre_reindex = pd.DataFrame()
            swayam_view_dfs = {}
            st.session_state["swayam_n_eff"] = None
            st.session_state["swayam_view_weights"] = None

            if target_ohlcv is not None and not target_ohlcv.empty:
                console.section("Self-Referential View Bank")
                # Leakage guard: drop the target's own macro column + its
                # excluded-predictor near-replicas from the MMR driver pool — a
                # view's Close correlates ~1.0 with the target's own macro
                # column, which would let MMR "explain" the target with itself
                # and silently zero the deviation oscillator.
                swayam_cols = swayam_macro_columns(active_target, macro_cols_list)
                console.item("Leakage guard", f"{len(macro_cols_list) - len(swayam_cols)} drivers dropped "
                             f"(the target and its near-replicas)")
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
                    st.session_state["swayam_view_weights"] = {k: float(v) for k, v in _view_w.iloc[-1].items()}
                    console.item("View Weighting", f"skill-weighted · {len(_last)} views")
                    console.item("Top Views", " · ".join(f"{k} {v:.2f}" for k, v in _last.head(4).items()))
                    console.item("Weakest View", f"{_last.index[-1]} {_last.iloc[-1]:.2f}")
                # Everything that needs the full ~270-column frames has run;
                # from here on only the drill-down's columns are kept.
                swayam_view_dfs = _slim_view_dfs(swayam_view_dfs)

            # The skill weights are scored at the horizon, so the cache is too.
            st.session_state["_swayam_analysis_cache"] = {
                "key": f"{_swayam_fetch_key}|h{FWD_HORIZON}",
                "swayam_view_dfs": swayam_view_dfs,
                "swayam_daily_pre_reindex": swayam_daily_pre_reindex,
                "n_eff": st.session_state.get("swayam_n_eff"),
                "view_weights": st.session_state.get("swayam_view_weights"),
            }

        # ── HORIZON-DEPENDENT tail: reindex onto the target's calendar ──────
        # Cheap (pure pandas, no yfinance) — re-runs since the horizon's
        # warm-up trim can shift the target's date spine.
        swayam_daily = swayam_daily_pre_reindex
        if not swayam_daily.empty:
            # Carry the breadth forward onto the model spine's calendar. The
            # views are built on the target's own OHLCV, which can end a session
            # before the panel does (yfinance publishes the OHLCV feed and the
            # close feed on different schedules). Reindexing onto the spine
            # (ff-fill) lets every page reach the latest valued session; the
            # true last-native date is recorded so the UI can say how much of
            # the tail is carried rather than observed.
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
                console.item("Net Breadth", f"{last.get('Oversold_Pct', 0) - last.get('Overbought_Pct', 0):+.0f} pts")
            console.success(f"Swayam aggregation: {len(swayam_daily)} trading days")

        console.end_phase("SWAYAM ENGINE")
        progress_bar(progress_container, 75, "Swayam Engine Complete", f"{len(swayam_view_dfs)} Views · {len(swayam_daily)} Trading Days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 76, "Computing Convergence", "Four dimensions · per session")
        from convergence import intelligence as _intel_mod
        from convergence.conviction_model import CompositeDDM
        from convergence.normalization import (consensus_series, compute_normalized_convergence,
                                               classify_convergence_score)
        from core.semantics import valuation_lean, breadth_lean

        # The first pass scores each session with the instrument's PRIOR
        # dimension weights; the learner then re-weights forward from resolved
        # outcomes only (4b), so no score depends on data that had not happened
        # when it was published.
        validator = CrossValidator(active_weights=_icfg.weights_seed())
        divergence_detector = CrossSystemDivergenceDetector()

        fvo_ts = engine.ts_data.copy()  # carries "Price" (set after fit)
        if active_date != "None" and active_date in data.columns:
            fvo_ts["Date"] = pd.to_datetime(data[active_date].values)
            fvo_ts = fvo_ts.set_index("Date")
        else:
            fvo_ts["Date"] = np.arange(len(fvo_ts))
        fvo_ts = fvo_ts[~fvo_ts.index.duplicated(keep="last")]

        swayam_by_date = {}
        if not swayam_daily.empty:
            for idx, row in swayam_daily[~swayam_daily.index.duplicated(keep="last")].iterrows():
                swayam_by_date[str(pd.Timestamp(idx).date())] = row

        # Every Mūla session with a published valuation is offered; a date with
        # no Swayam row is SKIPPED, not scored against a placeholder. The old
        # stand-in (50% oversold AND 50% overbought, regime 33/33/34) was an
        # impossible state that the learner, the walk-forward frame and the
        # divergence detector all scored as if it were a reading.
        overlap_count = native_overlap_count = 0
        total_dates = len(fvo_ts.index)
        for i, ts_idx in enumerate(fvo_ts.index):
            row_a = fvo_ts.loc[ts_idx]
            date_str = str(pd.Timestamp(ts_idx).date())
            mula_in = {
                "fvo": row_a.get("FVO", np.nan),
                "oversold_breadth": row_a.get("OversoldBreadth", 0.0),
                "overbought_breadth": row_a.get("OverboughtBreadth", 0.0),
                "regime": str(row_a.get("Regime", "")),
            }
            sw_in = None
            row_n = swayam_by_date.get(date_str)
            if row_n is not None:
                sw_in = {
                    "oversold_pct": float(row_n.get("Oversold_Pct", 0.0)),
                    "overbought_pct": float(row_n.get("Overbought_Pct", 0.0)),
                    "avg_unified_osc": float(row_n.get("Avg_Signal", 0.0)),
                    "regime_bull_pct": float(row_n.get("Regime_Bull_Pct", 0.0)),
                    "regime_bear_pct": float(row_n.get("Regime_Bear_Pct", 0.0)),
                    "native": bool(row_n.get("_Native", True)),
                }
            if validator.compute_convergence(mula_in, sw_in, date_str) is not None:
                overlap_count += 1
                native_overlap_count += int(sw_in["native"])
            divergence_detector.detect(mula_in, sw_in, date_str)
            if (i + 1) % 25 == 0 or i == total_dates - 1:
                progress_bar(progress_container, int(76 + (i + 1) / total_dates * 6),
                             "Computing Convergence", f"{i + 1}/{total_dates} sessions")
        convergence_df = validator.get_convergence_series()
        console.item("Scored sessions", f"{overlap_count} ({native_overlap_count} with a native "
                     f"Swayam row, {overlap_count - native_overlap_count} carried forward)")

        # ── 4b. Learn the dimension weights, forward only ───────────────
        # Gated on NATIVE overlap at the learner's own warm-up: below it the
        # learner can only return its prior, so "learned" would be a label on
        # the prior.
        _learn_weights = native_overlap_count >= _intel_mod.LEARNER_WARMUP
        _weight_hist = pd.DataFrame()
        if _learn_weights and not convergence_df.empty:
            console.section("Online Dimension Weighting")
            progress_bar(progress_container, 84, "Learning Dimension Weights",
                         "Causal · each dimension's signed skill")
            try:
                convergence_df, _weight_hist = _intel_mod.apply_online_weights(
                    convergence_df, fvo_ts, horizon=FWD_HORIZON, target_col="Price",
                    prior=_icfg.weights_seed(),
                )
            except Exception as _we:                # noqa: BLE001 - prior weights stand
                console.warning(f"Online weighting failed: {type(_we).__name__}: {_we} — prior weights stand")
        elif not convergence_df.empty:
            console.warning(
                f"Online weighting skipped: {native_overlap_count} native overlap sessions "
                f"(< {_intel_mod.LEARNER_WARMUP}, the learner's warm-up) — prior weights stand.")
        _prior_w = {k.removeprefix("w_"): float(v) for k, v in _icfg.weights_seed().items()}
        _tot = sum(_prior_w.values()) or 1.0
        _prior_w = {k: v / _tot for k, v in _prior_w.items()}
        _active_w = ({k: float(v) for k, v in _weight_hist.iloc[-1].items()}
                     if not _weight_hist.empty else _prior_w)
        if not _weight_hist.empty:
            console.item("Learned weights", " · ".join(f"{k} {v:.3f}" for k, v in
                                                       sorted(_active_w.items(), key=lambda kv: -kv[1])))
            console.item("Prior weights", " · ".join(f"{k} {_prior_w[k]:.2f}" for k in _active_w))
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_profile"] = {"weights": _active_w,
                                                           "learned": not _weight_hist.empty}

        # ── 4c. The composite's drift-diffusion filter — one pass ─────────
        # On the learned composite, which is the only construction the
        # learned weights mean anything for. (Two DDM passes used to run here,
        # one on the prior-weighted composite, and neither was shown anywhere.)
        if not convergence_df.empty:
            progress_bar(progress_container, 90, "Filtering the Composite", "Leaky drift-diffusion")
            _ddm = CompositeDDM(leak_rate=_icfg.ddm_leak, drift_scale=_icfg.ddm_drift,
                                long_run_var=_icfg.ddm_lrv).fit_series(convergence_df["convergence_score"])
            for _c in ("smoothed", "upper", "lower", "label"):
                convergence_df[f"ddm_{_c}"] = _ddm[_c].to_numpy()
            _last_c = convergence_df.iloc[-1]
            console.item("Weighted composite", f"{_last_c['convergence_score']:+.1f} "
                         f"({classify_convergence_score(_last_c['convergence_score'])}) · filtered "
                         f"{_last_c['ddm_smoothed']:+.1f} ({_last_c['ddm_label']})")

        # ── 4d. The consensus — evidence for the Convergence page ─────────
        _consensus_full = consensus_series(fvo_ts, swayam_daily)
        _cons_now = compute_normalized_convergence(fvo_ts, swayam_daily)
        if _cons_now:
            console.item("Consensus", f"{_cons_now['value']:+.2f} ({_cons_now['label']}) · "
                         f"Mūla {_cons_now['fvo_norm']:+.2f}, Swayam {_cons_now['swayam_norm']:+.2f}")

        console.section("Divergence Detection")
        events = divergence_detector.get_events()
        console.item("Events", len(events))
        if not events.empty:
            for etype, count in events["divergence_type"].value_counts().items():
                console.item(f"  {etype}", count)

        # ── 4e. Every engine graded the same way ─────────────────────────
        # The lean each engine publishes, against the h-day forward return,
        # on non-overlapping windows. The hero's edge gate reads Mūla's — the
        # call it gates. It used to read a walk-forward of the unsigned
        # AGREEMENT composite at 5 days on overlapping rows: it graded neither
        # the call nor the composite, and returned −0.01 on a synthetic signal
        # with a planted IC of +0.31.
        console.section("Walk-Forward Edge")
        progress_bar(progress_container, 93, "Grading Every Engine", f"Non-overlapping {FWD_HORIZON}d IC")
        _price = pd.to_numeric(fvo_ts["Price"], errors="coerce")
        _edges: dict = {}
        try:
            _fvo = pd.to_numeric(fvo_ts["FVO"], errors="coerce")
            _edges["mula"] = _intel_mod.series_edge(
                _fvo.map(valuation_lean).where(_fvo.notna()), _price, FWD_HORIZON)
            if not swayam_daily.empty:
                # Native rows only: a carried-forward breadth row is not a new call.
                _sw = (swayam_daily[swayam_daily["_Native"].astype(bool)]
                       if "_Native" in swayam_daily.columns else swayam_daily)
                _edges["swayam"] = _intel_mod.series_edge(
                    pd.Series([breadth_lean(a, b) for a, b in zip(_sw["Oversold_Pct"], _sw["Overbought_Pct"])],
                              index=_sw.index), _price, FWD_HORIZON)
            if not convergence_df.empty:
                _cs_idx = pd.to_datetime(convergence_df.index, errors="coerce")
                _edges["composite"] = _intel_mod.series_edge(
                    pd.Series(-convergence_df["convergence_score"].to_numpy(dtype=float) / 100.0,
                              index=_cs_idx), _price, FWD_HORIZON)
            for _k, _e in _edges.items():
                console.item(f"{_k} edge", (f"IC {_e['ic']:+.3f} ± {_e['se']:.3f} · {_e['n']} calls · "
                                            f"{_e['positive_blocks']}/{_e['scored_blocks']} periods +")
                             if _e and _e.get("ic") is not None and np.isfinite(_e["ic"]) else "not graded")
        except Exception as _ee:                    # noqa: BLE001 - a diagnostic, not the run
            console.warning(f"Edge grading skipped: {type(_ee).__name__}: {_ee}")
        console.end_phase("CONVERGENCE")
        progress_bar(progress_container, 95, "Convergence Complete",
                     f"{overlap_count} sessions · {len(events)} divergences · "
                     f"{'learned' if not _weight_hist.empty else 'prior'} weights")

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 96, "Storing Results", "Session state · cache")
        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
        st.session_state["fvo_ts"] = fvo_ts
        st.session_state["swayam_daily"] = swayam_daily
        st.session_state["swayam_view_dfs"] = swayam_view_dfs
        st.session_state["convergence_df"] = convergence_df
        st.session_state["divergence_events"] = events
        st.session_state["edges"] = _edges
        # The consensus's full history — the Convergence page's chart and the
        # Data page's export read it.
        st.session_state["hero_series"] = (
            _consensus_full["Consensus"].rename("Consensus") if not _consensus_full.empty else None)

        console.end_phase("FINAL ASSEMBLY")
        console.summary("RUN SUMMARY", {
            "Total Phases": "5/5 complete",
            "Mūla Rows": len(engine.ts_data),
            "Swayam Views": len(swayam_view_dfs),
            "Swayam Trading Days": len(swayam_daily),
            "Scored Sessions": overlap_count,
            "Divergence Events": len(events),
            "Status": "SUCCESS",
        })
        console.line('═', 70)
        console._write(f"  {Colors.BOLD}{Colors.GREEN}Analysis Complete{Colors.RESET}")
        console.line('═', 70)
        console._write()

        # Snapshot this config's full result into the bounded per-config cache
        # so revisiting it later restores instantly. LRU-evict to bound memory.
        _rcache = st.session_state.setdefault("results_cache", {})
        _rcache.pop(cache_key, None)
        _rcache[cache_key] = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
        while len(_rcache) > _RESULTS_CACHE_MAX:
            _rcache.pop(next(iter(_rcache)))

        progress_bar(progress_container, 100, "Analysis Complete", f"{active_target}")
        time.sleep(0.25)
        progress_container.empty()
        st.rerun()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    ts = engine.ts_data.copy()
    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))

    # ─── Precedent base rate (the hero's sixth gate) ────────────────────────
    # Computed ONCE over the instrument's fixed precedent term structure and
    # cached as the raw analog list, so the Precedent page summarises the same
    # list instead of re-running the matcher. The key carries the engine
    # config as well as the last price: a different panel or horizon is a
    # different state vector, not the same one with a new close.
    _plast = float(ts["Price"].iloc[-1]) if "Price" in ts.columns and len(ts) else 0.0
    _pkey = f"{st.session_state.get('engine_cache', active_target)}|{len(ts)}|{_plast:.6g}"
    if st.session_state.get("_prec_key") != _pkey:
        _prec_summary, _cached_analogs = None, []
        try:
            from analytics.analogs import find_similar_periods as _fsp, summarize_forward as _sf
            _display_hold = tuple(_icfg.precedent_horizons)
            _cached_analogs = _fsp(ts, active_target, hold_horizons=_display_hold,
                                   mom_window=_icfg.analog_mom_window) or []
            _ps = _sf(_cached_analogs, _display_hold) if _cached_analogs else {}
            # The gate reads at this instrument's forecast horizon (a member of
            # its precedent term structure); the nearest shorter one otherwise.
            _hp = int(_icfg.forecast_horizon)
            if _hp not in _ps:
                _cands = [h for h in _display_hold if h <= _hp]
                _hp = max(_cands) if _cands else min(_display_hold)
            _row = _ps.get(int(_hp))
            if _row:
                _prec_summary = {
                    "horizon": int(_hp), "median": float(_row["median"]),
                    "positive_pct": float(_row["positive_pct"]),
                    # n_eff, not n: ten analogs carried by one episode are one
                    # observation.
                    "n": int(_row["n"]), "n_eff": float(_row.get("n_eff", _row["n"])),
                    "p25": float(_row.get("p25", float("nan"))),
                    "p75": float(_row.get("p75", float("nan"))),
                    "usable": bool(_row.get("usable", True)),
                    "note": str(_row.get("note", "") or ""),
                }
        except Exception as _pe:                       # noqa: BLE001 - a gate, not the run
            console.warning(f"Precedent read skipped: {type(_pe).__name__}: {_pe}")
        st.session_state["precedent_summary"] = _prec_summary
        st.session_state["_precedent_analogs_cache"] = {"pkey": _pkey, "periods": _cached_analogs}
        st.session_state["_prec_key"] = _pkey

    # ─── Hero verdict — computed once. Rendered in full only on Overview;
    # every page's command bar reads the SAME object, so nothing can disagree
    # with what Overview shows.
    verdict = _compute_hero_verdict(signal)

    # ─── Chart window — read here, RENDERED in a chart panel header ────────
    # Read here (the filtered series below needs it) but drawn inside the panel
    # header of each page's primary chart; a widget interaction reruns the
    # script, so session_state already holds the new value by this line. The
    # cut comes from the one helper every page uses (ui.components.window_cutoff).
    if st.session_state.get("tf_selected") not in (*TIMEFRAME_TRADING_DAYS, "ALL"):
        st.session_state["tf_selected"] = "6M"
    selected_tf = st.session_state["tf_selected"]
    ts_filtered = ts
    if selected_tf != "ALL":
        if pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            _cut = window_cutoff(ts["Date"].max(), selected_tf)
            ts_filtered = ts[ts["Date"] >= _cut] if _cut is not None else ts
        else:
            ts_filtered = ts.iloc[max(0, len(ts) - TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)):]
    x_axis = ts_filtered["Date"]

    # ─── Command-bar status (price / change / freshness chip) ──────────────
    _cb_price = _cb_chg = None
    if "Price" in ts.columns and len(ts) >= 2:
        _cb_tail = pd.to_numeric(ts["Price"], errors="coerce").tail(2)
        if pd.notna(_cb_tail.iloc[-1]):
            _cb_price = float(_cb_tail.iloc[-1])
            if pd.notna(_cb_tail.iloc[-2]) and _cb_tail.iloc[-2] != 0:
                # PERCENT POINTS, not a fraction.
                _cb_chg = float(_cb_tail.iloc[-1] / _cb_tail.iloc[-2] - 1.0) * 100.0
    _cb_status_label, _cb_status_tone = _fresh["label"], _fresh["tone"]
    _cb_asof = _fresh["as_of"].strftime("%d %b %Y") if _fresh["as_of"] is not None else ""
    _edges = st.session_state.get("edges") or {}

    def _top_bar(*, windowed: bool = False) -> None:
        """The page shell, identical on every page.

        Order is fixed and means something: the TAPE (the world) sits above
        the COMMAND BAR (this instrument), which sits above the NOTICE RAIL
        (the caveats on it). The bar names the chart window only on pages
        whose charts it actually frames — on Overview, Precedent and System it
        changed nothing and still claimed "Window 6M".
        """
        render_ticker(data)
        _meta = [("Call", f"{verdict['signal']} · {format_conviction(verdict)}"),
                 ("Horizon", f"{FWD_HORIZON}D"), ("As of", _cb_asof)]
        if windowed:
            _meta.insert(1, ("Window", selected_tf))
        render_top_bar(
            target=active_target, price=_cb_price, change_pct=_cb_chg,
            status_label=_cb_status_label, status_tone=_cb_status_tone,
            meta_items=_meta,
        )
        render_notice_rail(_notices)

    def _safe_render(name, render_fn):
        """Render a page's content; a failure says which page and why."""
        try:
            render_fn()
        except Exception as e:                       # noqa: BLE001 - one page, not the app
            console.warning(f"{name} page failed: {type(e).__name__}: {e}")
            render_warning_box(title=f"{name} could not be drawn", content=str(e))

    # ─── App shell — one real page per analytical surface (st.navigation).
    # Every page below is a THIN wrapper over the run above: none recompute
    # the pipeline, so every page reads the same objects.
    def _page_overview() -> None:
        """Overview — the call, then why.

        Six numbers across the top answer "what changed since I last looked";
        the conviction chain under them answers "why", for the reader who
        wants it. They are built from the same objects, so they cannot disagree.
        """
        _top_bar()
        _swd = st.session_state.get("swayam_daily")
        _os = _ob = None
        if _swd is not None and not _swd.empty and {"Oversold_Pct", "Overbought_Pct"} <= set(_swd.columns):
            _os, _ob = float(_swd["Oversold_Pct"].iloc[-1]), float(_swd["Overbought_Pct"].iloc[-1])
        _cdf = st.session_state.get("convergence_df")
        _agree = (float(_cdf["agreement_ratio"].dropna().iloc[-1])
                  if _cdf is not None and not _cdf.empty and "agreement_ratio" in _cdf.columns
                  and _cdf["agreement_ratio"].notna().any() else None)
        _val = verdict["valuation"]
        _fvo_v = _val.get("fvo")
        _trust = verdict["trust"]
        _bind = verdict.get("binding")
        render_kpi_strip([
            {"label": "Call", "value": verdict["signal"],
             "subtext": verdict["action"]["label"].title(),
             "color_class": ("success" if verdict["signal_class"] == "buy"
                             else "danger" if verdict["signal_class"] == "sell" else "neutral")},
            {"label": "Conviction", "value": format_conviction(verdict),
             "subtext": (("capped by " + GATE_TITLES[_bind].lower()) if _bind
                         else "every gate holds" if verdict.get("conviction") is not None
                         else "no claim to grade"),
             "color_class": "accent"},
            {"label": "Valuation", "value": (_val.get("state") or "—").title() if _fvo_v is not None else "—",
             "subtext": (f"{_fvo_v:+.2f}σ · {abs(_val['pct']):.1f}% {_val['side']} fair value"
                         if _fvo_v is not None else "no published valuation"),
             "color_class": ("success" if _fvo_v is not None and _fvo_v <= -0.5
                             else "danger" if _fvo_v is not None and _fvo_v >= 0.5 else "neutral")},
            {"label": "Walk-Forward Edge",
             "value": f"{_trust['oos_ic']:+.3f}" if _trust.get("oos_ic") is not None else "—",
             "subtext": (f"{_trust['chip'].title()} · ± {_trust['se']:.3f}"
                         if _trust.get("se") is not None else _trust["chip"].title()),
             "color_class": "info"},
            {"label": "Swayam Net Breadth",
             "value": f"{_os - _ob:+.0f} pts" if _os is not None else "—",
             "subtext": f"{_os:.0f}% oversold · {_ob:.0f}% overbought" if _os is not None else "no Swayam read",
             "color_class": ("success" if _os is not None and _os - _ob > 10
                             else "danger" if _os is not None and _os - _ob < -10 else "neutral")},
            {"label": "Engine Agreement", "value": f"{_agree:.0%}" if _agree is not None else "—",
             "subtext": "Mūla × Swayam, four dimensions", "color_class": "neutral"},
        ], max_cols=6)
        render_section_header(
            "Conviction Chain",
            "One directional claim from Mūla, then six independent conditions that can each "
            "invalidate it. Conviction is their product, so the weakest gate is the binding "
            "constraint.",
            icon="target", accent="accent",
        )
        render_hero_card(verdict)

    def _page_mula() -> None:
        _top_bar(windowed=True)
        _safe_render("Mūla", lambda: render_mula_tab(
            engine, ts_filtered, x_axis, signal, model_stats, ts, active_target,
            edge=_edges.get("mula"), horizon=FWD_HORIZON))

    def _page_swayam() -> None:
        _top_bar(windowed=True)
        _safe_render("Swayam", lambda: render_swayam_tab(
            selected_tf=selected_tf, edge=_edges.get("swayam"), horizon=FWD_HORIZON))

    def _page_convergence() -> None:
        _top_bar(windowed=True)
        _safe_render("Convergence", lambda: render_convergence_tab(
            ts_filtered, edge=_edges.get("composite"), horizon=FWD_HORIZON))

    def _page_precedent() -> None:
        _top_bar()
        _prec_cache = st.session_state.get("_precedent_analogs_cache")
        _cached_periods = (_prec_cache["periods"]
                           if _prec_cache and _prec_cache.get("pkey") == _pkey else None)
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.analog_mom_window, _icfg.forecast_horizon,
            precomputed_periods=_cached_periods))

    def _page_system() -> None:
        _top_bar()
        _safe_render("System", lambda: render_diagnostics_tab(engine, signal, horizon=FWD_HORIZON))

    def _page_data() -> None:
        _top_bar(windowed=True)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    pages = {
        "": [st.Page(_page_overview, title="Overview", icon=":material/dashboard:", default=True)],
        # In the order the conviction chain reads them: the claim (Mūla), the
        # instrument's own confirmation (Swayam), their agreement
        # (Convergence), and the base rate (Precedent).
        "Engines": [
            st.Page(_page_mula, title="Mūla", icon=":material/monitoring:"),
            st.Page(_page_swayam, title="Swayam", icon=":material/hub:"),
            st.Page(_page_convergence, title="Convergence", icon=":material/merge_type:"),
            st.Page(_page_precedent, title="Precedent", icon=":material/history:"),
        ],
        "System": [
            st.Page(_page_system, title="System", icon=":material/monitor_heart:"),
            st.Page(_page_data, title="Data", icon=":material/table_chart:"),
        ],
    }
    st.session_state.pop("_pending_notices", None)     # shown on this render
    st.navigation(pages, position="sidebar").run()

    _render_footer()


if __name__ == "__main__":
    main()
