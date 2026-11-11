"""
Tattva — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — one valuation, and a call only where its record supports one. Mūla
values the instrument against the whole macro cross-section; Swayam reads its
own internals; the call is licensed by its own pooled, as-issued record —
across commodities, FX, crypto, equity indices and listed stocks.

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
    window_cutoff,
    edge_periods,
)
from ui.tabs.tab_mula import render_mula_tab
from ui.tabs.tab_swayam import render_swayam_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab
from ui.tabs.tab_universe import render_universe_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_ohlcv, fetch_commodity_dataset, fetch_stock_target_series, is_yfinance_ticker
from data.universe import resolve_stock_symbol

# ── Pipeline ─────────────────────────────────────────────────────────────────
from engines.mula import FairValueEngine
from core.config import EPOCH
from pipeline.prep import PrepError, config_for, fetch_window, log_report, prepare
from pipeline.freshness import freshness
from pipeline.run import run_prepared
from core.trust import class_record

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id
from core.config import (ALL_TARGETS, TARGET_CATEGORIES, is_stock_target, DIV_LOOKBACK,
                         TIMEFRAME_TRADING_DAYS, FREEFORM_STOCK_CATEGORIES,
                         register_stock_target)


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
    "edges", "precedent_summary", "analogs", "precedent_skill",
)
# Keep the last N configs. Each entry is a full pipeline result; 6 covers a
# session that browses a handful of targets without recomputing.
_RESULTS_CACHE_MAX = 6

#: Every session key a run produces or caches — the ONE list Start over and
#: Refresh clear. They used to keep two hand-maintained lists that had already
#: drifted (Refresh cleared the Swayam caches, Reset did not, so a Reset an
#: hour later paired fresh Mūla data with stale Swayam frames).
_SESSION_RESULT_KEYS = _BUNDLE_KEYS + (
    "engine_cache", "results_cache", "_analog_skill_cache",
)

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

    Catalogue targets outside the macro maps — Aluminium, the crypto
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
    s = fetch_stock_target_series(ticker, *fetch_window())
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
        tagline="Cross-Asset Fair Value · Self-Referential Breadth · A Graded Record",
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
     "does price, not just fair value, close the gap? Its valuation is the one claim Tattva makes.",
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
    reads its own price from a bank of views, and Tattva says where it stands — cheap, fair
    or rich — and issues a call only where that call's own record supports one.</div>
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
            {"label": "Daily History", "value": f"Since {EPOCH[:4]}",
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
        ("A valuation state", "Cheap, rich or fair against the cross-section, with six "
                              "readings behind it, each shown with its number."),
        ("The call's record", "Every claim graded as issued — entered at the next close, "
                              "against what the same direction earned on any day — pooled "
                              "by asset class. A call is printed only where that record "
                              "supports it."),
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


def _compute_hero_verdict(signal: dict, target: str, horizon: int) -> dict:
    """Gather the run's readings and build the verdict. Pure assembly.

    All interpretation lives in ``ui.components.build_hero_verdict``. Computed
    once per rerun, so every page's command bar reads the object Overview renders.
    """
    from analytics.analogs import MIN_EFFECTIVE_N
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
        call=(st.session_state.get("edges") or {}).get("call"),
        class_record=class_record(target),
        precedent=st.session_state.get("precedent_summary"),
        precedent_skill=st.session_state.get("precedent_skill"),
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
                             "yfinance · global macro universe · daily history")
                df, error = fetch_commodity_dataset(*fetch_window())
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

    active_target = st.session_state.get("active_target", "")
    # A stock target's price column is joined to a local copy of the frame.
    df = _ensure_stock_target_column(df, active_target)

    with st.sidebar:
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
                _rdf, _rerr = fetch_commodity_dataset(*fetch_window(), force=True)
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

    _icfg = config_for(active_target)
    FWD_HORIZON = _icfg.forecast_horizon

    # ─── Data freshness: status chip + queued notices ──────────────────────
    _fresh, _notices = freshness(df, active_target)
    # Notices queued by an action on a previous run (a failed Refresh) are
    # dropped only once a page has rendered them (below).
    _notices = list(st.session_state.get("_pending_notices") or []) + _notices

    # ─── Prepare this target's panel ───────────────────────────────────────
    try:
        prep = prepare(df, active_target)
    except PrepError as e:
        log_report(e.report, stage="fail")
        console.failure(e.title, e.detail)
        st.error(e.user_message)
        return
    data = df  # the tape reads the fetched panel

    cache_key = f"{prep.cache_key}|{(df.attrs or {}).get('panel_fingerprint', '')}"
    if st.session_state.get("engine_cache") != cache_key:
        # Restore a result this session already computed for this exact config.
        _rcache = st.session_state.setdefault("results_cache", {})
        if cache_key in _rcache:
            for _bk, _bv in _rcache[cache_key].items():
                st.session_state[_bk] = _bv
            _rcache[cache_key] = _rcache.pop(cache_key)  # most-recently-used
            st.session_state["engine_cache"] = cache_key
            console.success(f"Restored {active_target} from session cache — no recompute")
            st.rerun()
        st.session_state.pop("engine", None)

        console.header("TATTVA — Unified Convergence Analysis", f"v{VERSION}")
        console.main_header("ANALYSIS CONFIGURATION", {
            "Run ID": generate_run_id(), "Target": active_target,
            "Predictors": f"{len(prep.features)} columns",
            "Date Range": f"{len(prep.price)} observations",
        })
        log_report(prep.report)

        # The target's own OHLCV — Swayam's input. A sheet-sourced target has
        # none; Swayam is then skipped and reads as absent.
        _tkr = ALL_TARGETS.get(active_target)
        target_ohlcv = None
        if is_yfinance_ticker(_tkr):
            progress_bar(progress_container, 18, "Fetching Target OHLCV", f"yfinance · {_tkr}")
            _start, _end = fetch_window()
            target_ohlcv = fetch_ohlcv(
                [_tkr], _start, _end,
                force=bool(st.session_state.pop("_force_ohlcv", False))).get(_tkr)
        if target_ohlcv is None or target_ohlcv.empty:
            console.warning(f"No OHLCV for {active_target} ({_tkr}) — Swayam unavailable")

        result = run_prepared(
            prep, df, ohlcv=target_ohlcv,
            progress=lambda pct, title, detail="": progress_bar(progress_container, pct, title, detail))

        for _bk in _BUNDLE_KEYS:
            st.session_state[_bk] = getattr(result, _bk)
        st.session_state["engine_cache"] = cache_key
        console.summary("RUN SUMMARY", {
            "Mūla Rows": len(result.engine.ts_data),
            "Swayam Views": len(result.swayam_view_dfs),
            "Scored Sessions": len(result.convergence_df),
            "Divergence Events": len(result.divergence_events),
            "Status": "SUCCESS",
        })
        _rcache = st.session_state.setdefault("results_cache", {})
        _rcache.pop(cache_key, None)
        _rcache[cache_key] = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
        while len(_rcache) > _RESULTS_CACHE_MAX:
            _rcache.pop(next(iter(_rcache)))
        time.sleep(0.25)
        progress_container.empty()
        st.rerun()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    ts = engine.ts_data.copy()
    ts["Date"] = prep.calendar
    _cached_periods = st.session_state.get("analogs")

    # ─── Hero verdict — computed once. Rendered in full only on Overview;
    # every page's command bar reads the SAME object, so nothing can disagree
    # with what Overview shows.
    verdict = _compute_hero_verdict(signal, active_target, FWD_HORIZON)

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
        _meta = [("Read", f"{verdict['signal']} · {verdict['action']['label']}"),
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
        """Overview — the read, its record, then the evidence."""
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
        _own = (_edges.get("call") or {})
        _has_rec = _trust.get("hit") is not None and _trust.get("base") is not None
        _has_own = bool(_own.get("n_claims")) and np.isfinite(_own.get("hit", np.nan))
        render_kpi_strip([
            {"label": "Read", "value": verdict["signal"],
             "subtext": verdict["action"]["label"].title(),
             "color_class": ("success" if verdict["signal_class"] == "buy"
                             else "danger" if verdict["signal_class"] == "sell" else "neutral")},
            {"label": "Valuation", "value": (_val.get("state") or "—").title() if _fvo_v is not None else "—",
             "subtext": (f"{_fvo_v:+.2f} · {abs(_val['pct']):.1f}% {_val['side']} fair value"
                         if _fvo_v is not None else "no published valuation"),
             "color_class": ("success" if _fvo_v is not None and _fvo_v <= -0.5
                             else "danger" if _fvo_v is not None and _fvo_v >= 0.5 else "neutral")},
            {"label": "Class Record",
             "value": f"{_trust['hit']:.1f}%" if _has_rec else "—",
             "subtext": (f"vs {_trust['base']:.1f}% any day · t {_trust['t']:+.2f}"
                         if _has_rec and _trust.get("t") is not None else "no pooled record"),
             "color_class": "accent"},
            {"label": "This Target",
             "value": f"{_own['hit']:.1f}%" if _has_own else "—",
             "subtext": (f"vs {_own['base']:.1f}% any day · {_own['n_claims']:,} claims"
                         if _has_own else "no graded claims yet"),
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
            "The Read",
            "Where the target sits against the cross-section, whether the record licenses "
            "calling it, and the six readings behind it.",
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
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.analog_mom_window, _icfg.forecast_horizon,
            precomputed_periods=_cached_periods))

    def _page_universe() -> None:
        _top_bar()
        _safe_render("Universe", lambda: render_universe_tab(df, active_target))

    def _page_system() -> None:
        _top_bar()
        _safe_render("System", lambda: render_diagnostics_tab(engine, signal, horizon=FWD_HORIZON))

    def _page_data() -> None:
        _top_bar(windowed=True)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    pages = {
        "": [st.Page(_page_overview, title="Overview", icon=":material/dashboard:", default=True)],
        # In the order the evidence reads them: the claim (Mūla), the
        # instrument's own confirmation (Swayam), their agreement
        # (Convergence), and the base rate (Precedent).
        "Engines": [
            st.Page(_page_universe, title="Universe", icon=":material/grid_view:"),
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
