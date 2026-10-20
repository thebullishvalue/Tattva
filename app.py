"""
Tattva — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — one claim, sized only by measured skill, graded as issued. Mūla values
the instrument against the whole macro cross-section and makes the one claim;
its trust and every sizer's weight come from their own price-side grades
(core.call), and the call is composed and graded at every session — across
commodities, FX, crypto, equity indices and listed stocks.

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
    render_hero_card,
    render_control_hint,
    render_ticker,
    render_section_header,
    render_kpi_strip,
    render_note,
    panel,
    render_nav_brand,
    render_warning_box,
    render_top_bar,
    render_notice_rail,
    render_rail_readout,
    window_cutoff,
)
from ui.tabs.tab_mula import render_mula_tab
from ui.tabs.tab_swayam import render_swayam_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_commodity_dataset
from data.calendars import trading_days_behind, resolve_exchange, session_today
from data.prep import PrepError
from data.universe import resolve_stock_symbol

# ── Engines ──────────────────────────────────────────────────────────────────
# Mūla (valuation, with its error-correction layer) and Swayam (breadth),
# fitted through the pipeline's step functions (pipeline.run) — the same
# functions the offline study runs. There is no engine selector and no
# fallback.
from engines.mula import FairValueEngine
from engines.mula.blocks import block_membership

# ── The pipeline: the step functions and the closed loop ─────────────────────
from pipeline.run import (join_target, target_bars, swayam_drivers, prepare, fit_mula,
                          fvo_frame, build_swayam)
from pipeline.assemble import assemble, describe_priors

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import (MIN_DATA_POINTS, STALENESS_DAYS, ANALYSIS_EPOCH, ALL_TARGETS,
                         TARGET_CATEGORIES, is_stock_target, RAW_YIELD_PREDICTORS,
                         TIMEFRAME_TRADING_DAYS, FREEFORM_STOCK_CATEGORIES,
                         register_stock_target, get_instrument_config)
from core.instruments import instrument
from core.call import format_size
from core.knobs import knob
from core.trust import load_priors


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
#
#: ``call_ts`` (the call at every session), ``grades_ts`` (the as-of grades it
#: read), ``evidence`` and ``verdict`` (call_ts at d*) are the closed loop
#: (pipeline.assemble). ``asof`` (d*, Mūla's as-of session), ``horizon`` (h,
#: resolved once from the instrument's config), ``exec_lag`` (L), ``epoch``,
#: ``fingerprint`` (which history was valued) and ``priors_meta`` (which
#: pooled priors, or none) travel with the result they describe.
_BUNDLE_KEYS = (
    "engine", "fvo_ts", "swayam_daily", "swayam_view_dfs", "swayam_view_weights",
    "swayam_n_eff", "swayam_native_last",
    "convergence_df", "divergence_events", "hero_series",
    "intelligence_active_weights", "intelligence_active_profile",
    "call_ts", "grades_ts", "edges", "evidence", "verdict",
    "asof", "horizon", "exec_lag", "epoch", "fingerprint", "priors_meta",
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
    "_prec_key", "_precedent_analogs_cache", "_analog_skill_cache",
    "_swayam_fetch_cache", "_swayam_analysis_cache",
)

def _clear_results() -> None:
    """Drop every run product from the session (the data frame is kept)."""
    for _k in _SESSION_RESULT_KEYS:
        st.session_state.pop(_k, None)
    for _k in [k for k in list(st.session_state) if str(k).startswith("conv_norm_causal::")]:
        st.session_state.pop(_k, None)


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
  <div class="lede-claim">One engine values the instrument against the world and makes
    one claim; Tattva sizes it only by what its price-side grade has measured, and grades
    the call on the sessions it was issued.</div>
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
            {"label": "Daily History", "value": f"Since {ANALYSIS_EPOCH:%Y}",
             "subtext": "One fixed start for every run, and every published value uses "
                        "only data available on its own date, so re-running extends the "
                        "record instead of rewriting it"},
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
        ("One call, graded as issued", "Cheap, rich or fair against the cross-section, "
                                       "sized by the claim's measured grade, with what limits "
                                       "it named."),
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


def _render_claim_evidence(evidence: dict, edges: dict, horizon: int) -> None:
    """Overview's EVIDENCE→claim block: who closes the gap, the claim by side, its unit.

    Read off the pipeline's evidence and the per-side claim grades; nothing is
    estimated here, and none of it moves the call. The claim is about PRICE,
    so whether price does the closing is measured and shown rather than
    assumed.
    """
    gs = evidence.get("gap_split") or {}
    cu = evidence.get("claim_unit") or {}
    buy, sell = edges.get("claim:buy") or {}, edges.get("claim:sell") or {}

    def _f(d, k):
        try:
            v = float(d.get(k))
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    share, se_share, b_fv = _f(gs, "price_share"), _f(gs, "se_share"), _f(gs, "b_fv")
    render_section_header(
        "Who Closes the Gap?",
        "EVIDENCE → claim · does not move the call. When Mūla's gap was open, how much of its "
        f"closing over the next {horizon} sessions did price do, and how much did fair value do "
        "by moving toward price?",
        icon="layers", accent="violet",
    )

    def _side(e):
        hit, base = _f(e, "hit"), _f(e, "base")
        return f"{hit:.0f}% vs {base:.0f}%" if hit is not None and base is not None else "—"

    render_kpi_strip([
        {"label": "Price's share", "value": f"{share * 100:+.0f}%" if share is not None else "—",
         "subtext": (f"± {se_share * 100:.0f}% · of the gap closed in {horizon}D · "
                     f"{gs.get('n_win', 0)} windows" if se_share is not None else "not measured"),
         "color_class": "neutral"},
        {"label": "Fair value toward price", "value": f"{b_fv:+.2f}" if b_fv is not None else "—",
         "subtext": f"of each unit of gap, over {horizon}D", "color_class": "neutral"},
        {"label": "BUY claims hit", "value": _side(buy),
         "subtext": "vs the side-matched no-skill rate", "color_class": "neutral"},
        {"label": "SELL claims hit", "value": _side(sell),
         "subtext": "vs the side-matched no-skill rate", "color_class": "neutral"},
        {"label": "The claim's unit",
         "value": f"mean {cu['mean']:+.2f}σ" if _f(cu, "mean") is not None else "—",
         "subtext": (f"SD {cu['sd']:.2f} · RICH:CHEAP {cu['rich_cheap']:.1f} · SELL share "
                     f"{cu['sell_share']:.0%} · C12 contract "
                     f"{'holds' if cu.get('passes') else 'fails'}"
                     if _f(cu, "sd") is not None and _f(cu, "rich_cheap") is not None else "not measured"),
         "color_class": "neutral"},
    ], max_cols=5, key="overview-evidence")
    if share is not None and se_share is not None:
        render_note(
            f"While the gap was open, price closed {share:.0%} (± {se_share:.0%}) of it within "
            f"{horizon} sessions; fair value moving toward price did the rest. Tattva therefore "
            f"does not assume price does the closing: it acts only when the claim's price-side "
            f"grade clears the bar.")


def _asof_status(signal: dict, ticker: str | None, today) -> dict:
    """The command bar's as-of reading: d*, how far behind today it is, its chip.

    ``as_of`` is d* — the session the call is about — never the panel's newest
    date. ``behind`` counts the target's sessions after d* up to ``today``
    (``data.calendars.session_today``). The chip is the data-recency read
    (CURRENT / kD BEHIND / STALE), or PROVISIONAL when d* is a session still
    forming — which is today's by construction, so it is current as well.
    ``valued_as_of`` is set when the newest row was withheld and d* is older.
    """
    asof = signal.get("asof")
    out = {"as_of": asof, "behind": None, "label": "", "tone": "neutral",
           "provisional": bool(signal.get("provisional")), "valued_as_of": None}
    if asof is None:
        return out
    behind = trading_days_behind(ticker, pd.Timestamp(asof).date(), pd.Timestamp(today).date())
    out["behind"] = behind
    if out["provisional"]:
        out.update(label="PROVISIONAL", tone="warning")
    elif behind >= STALENESS_DAYS:
        out.update(label="STALE", tone="danger")
    elif behind >= 1:
        out.update(label=f"{behind}D BEHIND", tone="warning")
    else:
        # "Current", not "live": this is end-of-day data that can be hours
        # old; it is the newest session the source has.
        out.update(label="CURRENT", tone="success")
    if signal.get("asof_row_withheld"):
        out["valued_as_of"] = asof
    return out


def _render_call_readout(verdict: dict) -> None:
    """Rail readout — the call, read from the one verdict, not a control.

    Rendered AFTER the pipeline, into a slot the rail reserved, so it always
    shows the result the pages show: during a recompute the slot is empty
    rather than the previous target's. Action, size, the claim's grade, the
    call's own grade and the session it is about. No learned weights: nothing
    the rail used to list moved the call.
    """
    st.markdown('<div class="sidebar-title">Call</div>', unsafe_allow_html=True)
    if not verdict:
        render_control_hint("Run an analysis to populate.")
        return
    render_rail_readout(_call_readout_rows(verdict))


def _call_readout_rows(verdict: dict) -> list[tuple[str, str, str]]:
    """The rail readout's ``(label, value, tone)`` rows, built from the verdict.

    Pure, so a test can build them: this text is drawn on every page, outside
    any page's error boundary. The call's IC is formatted only when it is
    defined; a warmed-up call whose issued calls are all the same reads
    'ungraded'.
    """
    act = verdict["action"]
    t, cg = verdict.get("claim_t"), verdict["call_grade"]
    warm = knob("GRADE_WARMUP_WIN")
    if cg["graded"] and cg["ic"] is not None and cg["se"] is not None:
        call_grade = f"IC {cg['ic']:+.3f} ± {cg['se']:.3f}"
    else:
        call_grade = f"{cg['n_win']} of {warm} windows" if cg["n_win"] < warm else "ungraded"
    return [
        ("Call", act, "long" if act == "BUY" else "short" if act == "SELL" else ""),
        ("Size", format_size(verdict["size"]), "accent" if verdict["size"] else ""),
        ("Claim t_post", f"{t:.2f}" if t is not None else "ungraded",
         "caution" if not verdict["trust"] else "long"),
        ("Call grade", call_grade, ""),
        ("As of", f"{verdict['asof']:%d %b %Y}" if verdict.get("asof") is not None else "—", ""),
    ]


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
                             f"yfinance · global macro universe · daily since "
                             f"{ANALYSIS_EPOCH:%b %Y}")
                # From the fixed epoch, never "today minus N years": a rolling
                # window moved every published value by one session a day.
                df, error = fetch_commodity_dataset(ANALYSIS_EPOCH, pd.Timestamp.today())
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

    # A stock or sheet target's price column is joined here — BEFORE
    # numeric_cols are computed below — so the target resolves to itself on its
    # first render instead of falling back to some other column. Local to this run.
    df = join_target(df, st.session_state.get("active_target", ""),
                     epoch=ANALYSIS_EPOCH, end=pd.Timestamp.today())

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

        # The call readout's place in the rail. Filled after the pipeline,
        # from the verdict, so it can never show a previous run's call.
        _call_slot = st.container()

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
                _rdf, _rerr = fetch_commodity_dataset(
                    ANALYSIS_EPOCH, pd.Timestamp.today(), force=True)
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

    # ─── Resolve the instrument — once ─────────────────────────────────────
    # How it is tuned (core.config.INSTRUMENT_CONFIGS), what it is
    # (core.instruments) and THE horizon h are resolved here and passed
    # explicitly to everything below; nothing downstream re-reads them. An
    # unregistered target is a registration bug, so it fails loudly here —
    # there is no default config to fall back to (a default carried no leakage
    # exclusions and would have valued Gold against GLTR).
    active_target = st.session_state.get("active_target", target_col)
    try:
        _icfg = get_instrument_config(active_target)
        _inst = instrument(active_target)
    except KeyError as _ke:
        console.failure("Unregistered target", str(_ke.args[0]))
        st.error(str(_ke.args[0]))
        return
    FWD_HORIZON = int(_icfg.forecast_horizon)   # THE horizon h (trading days)
    _tgt_ticker = _inst.ticker
    _tgt_exch = resolve_exchange(_tgt_ticker)
    # The valuation panel is the REGISTERED cross-section
    # (core.instruments.predictor_universe, applied by pipeline.run.prepare):
    # the macro universe and the registered catalogue predictors, minus the
    # raw yield levels, the sheet series, the target, its self-replicating
    # near-duplicates (GLTR for a precious metal, the US equity complex for a
    # US index) and its declared same-exposure twins (INDA and IIND.L for
    # USD/INR). There is no predictor picker: Mūla prices the target against
    # the traded opportunity set, and hand-deselecting instruments from it
    # makes a smaller set with an undocumented reason, not a better one.
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Notices ────────────────────────────────────────────────────────────
    # QUEUED, not rendered. These used to paint straight onto the top of the
    # page, above the command bar — so on a lagging source the first thing on
    # screen was three stacked apology boxes and the instrument was pushed
    # below the fold, on precisely the days the numbers most needed scrutiny.
    # They are collected and rendered by the page shell (render_notice_rail)
    # directly BENEATH the chrome they qualify.
    _notices: list[dict] = []

    def _notice(kind: str, title: str, body: str) -> None:
        _notices.append({"kind": kind, "title": title, "body": body})

    # Notices queued by an action on a PREVIOUS run (a failed Refresh). Read,
    # not consumed, here: a run that recomputes ends in st.rerun() and never
    # renders, so the notice is dropped only once a page has actually shown it
    # (just before the page renders, below).
    for _pn in st.session_state.get("_pending_notices") or []:
        _notice(_pn["kind"], _pn["title"], _pn["body"])

    # ─── Clean & Fit Engine ────────────────────────────────────────────────
    # Guard: a selected target whose column failed to fetch (e.g. a sheet/source
    # outage on a later run, while it stays selected) has no column to value.
    # Fail clean. Stock and sheet targets were joined above if their fetch
    # answered at all.
    if active_target not in df.columns:
        if is_stock_target(active_target):
            console.failure("Stock target fetch failed", f"'{active_target}' (ticker {_tgt_ticker}) — yfinance returned no usable data.")
            st.error(f"'{active_target}' price fetch failed (ticker {_tgt_ticker}) — yfinance returned no data. "
                     f"Check the symbol, or try again once yfinance recovers.")
        else:
            console.failure("Target column missing", f"'{active_target}' not in fetched dataset — its source fetch failed.")
            st.error(f"'{active_target}' data is currently unavailable (its source fetch failed). "
                     f"Pick another target, or re-run once the source is back online.")
        return
    if active_date == "None" or active_date not in df.columns:
        st.error("The panel has no date column; every published value is dated, so it cannot be valued.")
        return

    # ─── The target's own OHLCV and Swayam's driver pool ───────────────────
    # Read BEFORE prep: the spine keeps the sessions the TARGET printed, and
    # the target's own bars are the record of that (data.calendars.
    # target_printed). Swayam reads the same bars. Cached per target and
    # panel fingerprint, so a rerun or a horizon change reuses them.
    #
    # ONE panel feeds both engines. Swayam's MMR driver pool used to be a
    # second fetch of the macro universe, keyed by TICKER — so its leakage
    # guard, which drops the target and its near-replicas by NAME, matched
    # nothing, and after the cache's hour the two engines read two different
    # pulls. The drivers are the same friendly-named frame Mūla values against
    # (Mūla admits a subset of it: no yields, no near-replicas or twins).
    _fp = (df.attrs or {}).get("panel_fingerprint", "")
    _swayam_fetch_key = f"swayam_fetch::{active_target}|{_fp}"
    _nf_cache = st.session_state.get("_swayam_fetch_cache")
    if _nf_cache is not None and _nf_cache.get("key") == _swayam_fetch_key:
        target_ohlcv = _nf_cache["target_ohlcv"]
        swayam_macro_df = _nf_cache["swayam_macro_df"]
        _ohlcv_rows_dropped = _nf_cache.get("rows_dropped", 0)
    else:
        if _inst.has_ohlcv:
            progress_bar(progress_container, 16, "Fetching Target OHLCV", f"yfinance · {_tgt_ticker}")
        # Only the target's own weekday sessions, keeping the pre-spine
        # warm-up bars (pipeline.run.target_bars).
        target_ohlcv, _ohlcv_rows_dropped = target_bars(
            active_target, epoch=ANALYSIS_EPOCH, end=pd.Timestamp.today(),
            force=bool(st.session_state.pop("_force_ohlcv", False)))
        swayam_macro_df = swayam_drivers(df, active_target)
        st.session_state["_swayam_fetch_cache"] = {
            "key": _swayam_fetch_key,
            "target_ohlcv": target_ohlcv,
            "swayam_macro_df": swayam_macro_df,
            "rows_dropped": _ohlcv_rows_dropped,
        }

    # ─── Prepare the panel (data.prep — pure, as-of invariant) ─────────────
    # From the fixed epoch, on the sessions the target printed; features
    # forward-filled, never back-filled; only the TARGET's missing rows
    # dropped. A young feature enters the cross-section causally through the
    # engine's print floor (FVO_MIN_PRINTS) and can no longer move the start.
    try:
        prep = prepare(df, active_target, epoch=ANALYSIS_EPOCH, bars=target_ohlcv,
                       date_col=active_date)
    except PrepError as _pe:
        progress_container.empty()
        console.failure("Insufficient target history", str(_pe))
        st.error(f"{_pe} Pick another target, or re-run once its source has more history.")
        return
    data = prep.data
    active_features = prep.features
    if not active_features:
        progress_container.empty()
        console.failure("Empty valuation panel", f"{active_target}: no registered predictor is in the panel.")
        st.error("No price-level predictors remain — the valuation engine needs a cross-section of prices.")
        return
    # The valuation regression models LOG price → the target must be strictly
    # positive. Every shipped target is a price/level/ratio, but a future target
    # (a spread, a net position, a yield differential) could go ≤0; fail clean
    # rather than silently producing all-NaN valuations.
    if (data[active_target] <= 0).any():
        progress_container.empty()
        console.failure("Non-positive target", f"{active_target}: contains values ≤ 0; the log-level valuation engine needs a strictly positive series.")
        st.error(f"'{active_target}' has non-positive values — the valuation engine models "
                 f"log price and needs a strictly positive series.")
        return

    def _log_prep() -> None:
        """Print the data-prep trace to the terminal (row evolution, start, fingerprints)."""
        r = prep.report
        console.section("DATA PREPARATION")
        console.item("Target", f"{active_target}  (ticker={_tgt_ticker or 'n/a'}, exch={_tgt_exch or 'weekday'})")
        console.item("Horizon h", f"{FWD_HORIZON}d (the instrument config's; the engine is fit to no label)")
        console.item("Analysis epoch", f"{ANALYSIS_EPOCH:%Y-%m-%d} (structural — the history starts "
                     f"at max(epoch, the target's first print))")
        console.item("Rows · fetched", r["rows_initial"])
        console.item("Rows · from the epoch", r["rows_epoch"])
        console.item("Rows · target-printed sessions",
                     f"{r['rows_session']}  ({r['sessions_dropped']} non-session rows, "
                     f"{r['target_not_printed']} sessions the target did not print · {r['printed_source']})")
        _yields = [c for c in numeric_cols if c in RAW_YIELD_PREDICTORS]
        if _yields:
            # Raw yield LEVELS are rate series, not prices: they print at/near/
            # below zero and the engine's log transform is undefined there. The
            # tradeable expression of the curve is already in the panel as the
            # Treasury ETF complex, so they are excluded rather than transformed.
            console.item("Instruments · excluded (raw yield levels, not prices)", len(_yields))
        if _inst.twins:
            console.item("Instruments · excluded (same-exposure twins)", ", ".join(_inst.twins))
        if r["feats_absent"]:
            console.item("Instruments · registered, absent from this panel", len(r["feats_absent"]))
        console.item("Instruments · in valuation panel", r["feats_kept"])
        # Same panel digest across two runs = same cross-section, so any
        # difference in output came from the VALUES or the code (see
        # data/fetcher.py::_panel_fingerprint). The history fingerprint hashes
        # the settled prepared rows the engine values.
        if _fp:
            console.item("Panel fingerprint", _fp)
        console.item("History fingerprint", prep.fingerprint)
        console.item("History start", f"{prep.start:%Y-%m-%d}")
        console.item("Rows · final spine", r["rows_final"])
        # The engine applies its own burn-in internally and flags those rows
        # `Valid = False`; there is no warm-up head to trim here.
        _pub = max(0, r["rows_final"] - int(_icfg.fvo_burn_in))
        console.item("Rows · valued (after engine burn-in)",
                     f"{_pub}  ({_icfg.fvo_burn_in} burn-in rows published as Valid=False)")
        console.checkpoint(f"Target history ≥ {MIN_DATA_POINTS}", "OK")

    # The run's identity: target, h, the prepared history's fingerprint (which
    # covers the cross-section's membership and every settled value) and its
    # span, whose end carries the newest, still-unsettled row.
    _date_range = f"{prep.start.date()}_{data.index[-1].date()}"
    cache_key = f"fvo{FWD_HORIZON}|{active_target}|{prep.fingerprint}|{_date_range}"
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
        # Full data-preparation trace (row evolution, spine, start, fingerprints)
        # so the pipeline has no dark spots — printed once per new computation.
        _log_prep()

        # Reuse the hoisted main-area progress slot (created at the top of main())
        # so the bar continues from where the fetch left it (~15%) with no gap.

        # ── Phase 1: Data Acquisition ─────────────────────────────────────
        # HORIZON-INDEPENDENT: the target's OHLCV and Swayam's driver pool were
        # read above, before prep, keyed by target and panel fingerprint.
        console.start_phase("DATA ACQUISITION", 1, 5)
        console.section("Target OHLCV")
        if target_ohlcv is not None and not target_ohlcv.empty:
            console.item("Symbol", _tgt_ticker)
            console.item("Rows", f"{len(target_ohlcv)} target sessions "
                                 f"({_ohlcv_rows_dropped} non-session bars dropped)")
            console.item("Has Volume", bool("Volume" in target_ohlcv.columns
                                            and target_ohlcv["Volume"].fillna(0).abs().sum() > 0))
            console.success(f"Target OHLCV: {len(target_ohlcv)} rows")
        elif not _inst.has_ohlcv:
            console.warning(f"{active_target} has no OHLCV source ({_tgt_ticker}) — Swayam unavailable")
        else:
            console.warning(f"No OHLCV for {_tgt_ticker} — Swayam breadth will be unavailable")
        console.section("Swayam Drivers")
        console.item("Driver pool", f"{swayam_macro_df.shape[1]} panel columns (the panel Mūla reads)")
        progress_bar(progress_container, 19, "Data Acquisition Complete",
                     f"{swayam_macro_df.shape[1]} drivers")
        console.end_phase("DATA ACQUISITION")

        # ── Phase 2: Mūla (FairValueEngine) ─────────────────────────────────
        console.start_phase("MŪLA ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Mūla Engine", f"Valuation · {len(active_features)} Instruments · {len(data)} Rows")

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
            console.item("Valuation Core", "Mūla — cointegrating core + error-correction layer")
            # `_icfg` threads this instrument's own Mūla knobs (burn-in, print
            # floor, discount grid, lookback windows) and THE horizon h. The
            # engine publishes `Price` itself; its output is not edited here.
            engine = fit_mula(prep, _icfg, progress=lambda pct, msg: progress_bar(
                progress_container, int(20 + pct * 20), "Running Mūla Engine", msg))
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
                    _dates = data.index[np.asarray(_wr == "incomplete fetch")]
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
        # pipeline.run.build_swayam: the view bank on the target's own bars,
        # skill-weighted at h, carried onto the model spine with _Native
        # marking genuine view-bank sessions. Its inputs are the target, the
        # panel (fingerprinted in _swayam_fetch_key), h and d* (both fixed by
        # those), so the result is cached under that key and h.
        console.start_phase("SWAYAM ENGINE", 3, 5)
        # d*, Mūla's as-of session. Swayam's stored readings (the view weights
        # and the last native date) are dated at or before it, never after.
        _d_star = engine.asof
        progress_bar(progress_container, 42, "Running Swayam Engine",
                     "MSF+MMR+Regime · self-referential view bank")
        _sw_key = f"{_swayam_fetch_key}|h{FWD_HORIZON}"
        _na_cache = st.session_state.get("_swayam_analysis_cache")
        if _na_cache is not None and _na_cache.get("key") == _sw_key:
            swayam_daily, _sw = _na_cache["swayam_daily"], _na_cache["meta"]
            console.item("View Bank", "reused cached fit (same target, panel and horizon)")
            progress_bar(progress_container, 74, "Swayam Engine Reused",
                         f"{len(_sw['view_dfs'])} Views (cached)")
        else:
            if target_ohlcv is not None and not target_ohlcv.empty:
                console.section("Self-Referential View Bank")
                console.item("Timescale Span", str(_icfg.swayam_lengths))
                console.item("Regime Sensitivity", _icfg.swayam_regime_sensitivity)
                console.item("Base Weight", _icfg.swayam_base_weight)

            def _swayam_progress(done, total, name):
                pct_val = int(45 + done / max(total, 1) * 30)
                progress_bar(progress_container, pct_val, f"View {name}", f"{done}/{total} views")

            swayam_daily, _sw = build_swayam(active_target, _icfg, target_ohlcv, prep,
                                             asof=_d_star, drivers=swayam_macro_df,
                                             progress=_swayam_progress)
            if _sw["view_dfs"]:
                # Leakage guard: the target's own column and its near-replicas
                # are dropped from the MMR driver pool, or a view's MMR would
                # "explain" the target with itself.
                console.item("Leakage guard", f"{_sw['n_drivers'] - _sw['n_drivers_kept']} drivers "
                             f"dropped (the target and its near-replicas)")
                console.item("Views (bank)", f"{_sw['n_members']} · timescale × information-set × mechanism")
                console.success(f"View bank: {len(_sw['view_dfs'])} views · ~{_sw['n_eff']:.1f} effective")
            if _sw["view_weights"]:
                _last = pd.Series(_sw["view_weights"]).sort_values(ascending=False)
                console.item("View Weighting", f"skill-weighted · {len(_last)} views")
                console.item("Top Views", " · ".join(f"{k} {v:.2f}" for k, v in _last.head(4).items()))
                console.item("Weakest View", f"{_last.index[-1]} {_last.iloc[-1]:.2f}")
            st.session_state["_swayam_analysis_cache"] = {"key": _sw_key, "swayam_daily": swayam_daily,
                                                          "meta": _sw}
        swayam_view_dfs = _sw["view_dfs"]
        st.session_state["swayam_n_eff"] = _sw["n_eff"]
        st.session_state["swayam_view_weights"] = _sw["view_weights"]
        st.session_state["swayam_native_last"] = _sw["native_last"]
        if not swayam_daily.empty:
            console.item("Trading Days", len(swayam_daily))
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
        from core.semantics import breadth_lean

        # The first pass scores each session with the instrument's PRIOR
        # dimension weights; the learner then re-weights forward from resolved
        # outcomes only (4b), so no score depends on data that had not happened
        # when it was published.
        validator = CrossValidator(active_weights=_icfg.weights_seed())
        divergence_detector = CrossSystemDivergenceDetector()

        # Mūla's published frame, date-indexed. The prepared spine has one row
        # per session, so the engine's row positions and these dates agree.
        fvo_ts = fvo_frame(engine, prep)

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

        # ── 4e. The Swayam and Convergence pages' own leans ──────────────
        # Graded for their pages only (the standalone lean and the composite);
        # neither moves the call. The claim, the sizers and the call itself
        # are graded in 4f, on the decision sample the call uses.
        console.section("Page Leans")
        progress_bar(progress_container, 93, "Grading the Page Leans", f"Non-overlapping {FWD_HORIZON}d IC")
        _price = pd.to_numeric(fvo_ts["Price"], errors="coerce")
        _edges: dict = {}
        try:
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

        # ── 4f. The call: composed at every session, graded as issued ─────
        # pipeline.assemble closes the loop: the inputs read by date, the
        # as-of grades (shrunk to the class prior when the study's file
        # exists), the call at every session (core.call), the call graded on
        # the sessions it was issued, and the verdict at d*. Every page reads
        # that verdict; nothing downstream re-derives a lean or a threshold.
        console.section("The Call")
        progress_bar(progress_container, 94, "Composing the Call",
                     f"Every session · {FWD_HORIZON}D from the next close")
        _lag = int(knob("EXECUTION_LAG"))
        try:
            _priors = load_priors(FWD_HORIZON, _lag)
        except ValueError as _pe:             # a file studied at another h or L is refused
            console.warning(f"Pooled priors refused ({_pe}) — per-instrument grades only")
            _priors = None
        _asm = assemble(fvo_ts, swayam_daily, instrument=_inst, cfg=_icfg, asof=engine.asof,
                        today=session_today(_tgt_exch), priors=_priors,
                        epoch=engine.epoch, fingerprint=engine.fingerprint)
        _edges.update(_asm.edges)
        # ONE horizon, from the instrument's config, in every object (C1).
        _hs = {int(engine.horizon), FWD_HORIZON, int(_asm.verdict["horizon_days"]),
               int(_asm.call_ts.attrs["h"]), *(int(e["horizon"]) for e in _edges.values())}
        assert len(_hs) == 1, f"one run carries more than one horizon: {sorted(_hs)}"
        _v = _asm.verdict
        console.item("Call", f"{_v['action']} · size {format_size(_v['size'])}")
        console.item("Reason", _asm.call_ts["reason"].get(engine.asof, "no valuation published")
                     if engine.asof is not None else "no valuation published")
        console.item("Call grade", _v["call_grade"]["status"])
        console.item("Priors", describe_priors(_priors)["label"])
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
        st.session_state["call_ts"] = _asm.call_ts
        st.session_state["grades_ts"] = _asm.grades_ts
        st.session_state["evidence"] = _asm.evidence
        st.session_state["verdict"] = _asm.verdict
        # Which session, which h and L, which history and which priors this
        # result is about.
        st.session_state["asof"] = engine.asof
        st.session_state["horizon"] = FWD_HORIZON
        st.session_state["exec_lag"] = _lag
        st.session_state["epoch"] = ANALYSIS_EPOCH
        st.session_state["fingerprint"] = prep.fingerprint
        st.session_state["priors_meta"] = describe_priors(_priors)
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
    ts["Date"] = data.index.to_numpy()

    # ─── Data freshness — measured from d* ──────────────────────────────────
    # ONE reading, shared by these notices and the command bar's status chip:
    # the target's own sessions from d* (Mūla's as-of session) to today on the
    # target's exchange (data.calendars.session_today), holiday-aware, so
    # Friday's valuation read on a Sunday is current, not stale. A withheld
    # newest row is data, not a valuation — it cannot make the call fresher.
    _status = _asof_status(signal, _tgt_ticker, session_today(_tgt_exch))
    _d_star = _status["as_of"]
    _ds = f"{_d_star:%d %b %Y}" if _d_star is not None else ""
    try:
        if _d_star is not None and not _status["provisional"]:
            _behind = _status["behind"]
            if _behind >= STALENESS_DAYS:
                _notice(
                    "warning", "Latest data unavailable",
                    (f"Mūla's latest valuation is {_ds} — {_behind} trading days behind. The price "
                     f"source (yfinance) hasn't published more recent data, so every signal below "
                     f"reflects {_ds}, not today. Use Refresh data in the rail once the source updates."),
                )
            elif _behind >= 1:
                _notice(
                    "info", "Data freshness",
                    f"Signals are as of {_ds} ({_behind} trading day"
                    f"{'s' if _behind > 1 else ''} behind — today's bar may not be published yet).",
                )

        # Predictors refilled from a cached snapshot (data.fetcher's rate-limit
        # recovery). Read off the frame itself, so a frame served from cache
        # carries its own warning.
        _stale_backfills = (df.attrs or {}).get("stale_backfills") or {}
        if _stale_backfills:
            _sb_items = sorted(_stale_backfills.items(), key=lambda kv: kv[1])
            _sb_preview = ", ".join(f"{k} (as of {v})" for k, v in _sb_items[:5])
            _sb_more = f" +{len(_sb_items) - 5} more" if len(_sb_items) > 5 else ""
            _notice(
                # WARNING, not info: a snapshot-reconstructed column is reindexed
                # onto this run's calendar and ffilled, so it is a DIFFERENT SERIES
                # from a live pull, not a staler one — and every published date can
                # move with it.
                "warning", "Predictors carried from snapshot — history differs",
                f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                f"from a prior cached snapshot: {_sb_preview}{_sb_more}. A rebuilt column "
                f"is not merely staler than a live one — it is a different series, so "
                f"this run's published history may differ from a run that fetched "
                f"cleanly. Compare the history fingerprint before treating past values as "
                f"final; refresh once the source is healthy for an authoritative record.",
            )

        # Session completeness. The engine's gate is the ONE definition of a
        # complete session (it counts the instruments whose markets were open);
        # these notices only read what it decided about the newest row. (The app
        # used to re-derive completeness with a threshold of its own over every
        # column, which agreed with the gate only by coincidence.)
        if signal.get("provisional"):
            _pr = signal.get("published_row")
            _nsch = (engine.ts_data["NScheduled"].iloc[_pr]
                     if _pr is not None and "NScheduled" in engine.ts_data.columns else np.nan)
            _nsch_txt = f" of {int(_nsch)}" if np.isfinite(_nsch) else ""
            _notice(
                "info", "Session still forming",
                (f"{_ds} is today's session and has not closed: {signal.get('n_available', 0)}"
                 f"{_nsch_txt} instruments whose markets are open have printed. The reading is "
                 f"PROVISIONAL — published, never learned from, and settled when the session closes."),
            )
        elif signal.get("asof_row_withheld") and signal.get("withheld_reason") == "incomplete fetch":
            _notice(
                "warning", "Partial latest session",
                (f"The newest session ({data.index[-1]:%d %b %Y}) failed the engine's completeness "
                 f"gate: its markets were open, but too few of them posted in this fetch, so it is "
                 f"not valued. The call is as of {_ds}; refresh once those markets post."),
            )

        # The target lagging the panel: the spine keeps only sessions the TARGET
        # printed, so panel rows after its last print are not valued until it
        # does. Only the target's FINISHED sessions count — the panel can carry
        # a row dated by a market whose day has already rolled (Asia's
        # tomorrow) while the target's own session of that date has not begun.
        _panel_last = pd.to_datetime(df[active_date], errors="coerce").max()
        _tgt_last = data.index[-1]
        _upto = (min(_panel_last.normalize(), session_today(_tgt_exch) - pd.Timedelta(days=1))
                 if pd.notna(_panel_last) else _tgt_last)
        if _upto > _tgt_last:
            _t_behind = trading_days_behind(_tgt_ticker, _tgt_last.date(), _upto.date())
            if _t_behind >= 1:
                _notice(
                    "warning", f"{active_target} data is lagging",
                    (f"This target last printed {_tgt_last:%d %b %Y}, {_t_behind} of its sessions "
                     f"behind the macro universe (which runs to {_panel_last:%d %b %Y}). Sessions "
                     f"it has not printed are not valued until it does."),
                )
    except Exception as _fe:                          # noqa: BLE001 - never break a run to report
        console.warning(f"Freshness notices skipped: {type(_fe).__name__}: {_fe}")

    # ─── The verdict — call_ts at d*, composed in the pipeline (Phase 4f).
    # Rendered in full on Overview; the command bar, the rail and every page
    # read this SAME object, so nothing can disagree with what Overview shows.
    verdict = st.session_state["verdict"]
    with _call_slot:
        _render_call_readout(verdict)

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

    # ─── Command-bar status (price / change at d*, recency chip) ───────────
    # The price and its change are d*'s, the session the call is about — not
    # the newest row's, which may be a session the engine withheld.
    _cb_price = _cb_chg = None
    if _d_star is not None:
        _cb_tail = pd.Series(pd.to_numeric(engine.ts_data["Price"], errors="coerce").to_numpy(),
                             index=data.index).loc[:_d_star].tail(2)
        if len(_cb_tail) and pd.notna(_cb_tail.iloc[-1]):
            _cb_price = float(_cb_tail.iloc[-1])
            if len(_cb_tail) == 2 and pd.notna(_cb_tail.iloc[-2]) and _cb_tail.iloc[-2] != 0:
                # PERCENT POINTS, not a fraction.
                _cb_chg = float(_cb_tail.iloc[-1] / _cb_tail.iloc[-2] - 1.0) * 100.0
    _cb_status_label, _cb_status_tone = _status["label"], _status["tone"]
    # 'As of d*'. When the newest row was withheld it says so: the call is
    # VALUED as of d*, and the newer session is named with the engine's reason.
    if _status["valued_as_of"] is not None:
        _cb_asof_items = [("Valued as of", _ds),
                          ("Newest", f"{data.index[-1]:%d %b} · {signal.get('withheld_reason') or 'withheld'}")]
    else:
        _cb_asof_items = [("As of", _ds)]
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
        # 'k of h sessions elapsed': how much of the call's horizon has run
        # since d*, the session it was made on (the verdict's own count).
        _elapsed = verdict["elapsed"]
        _meta = [("Call", verdict["action"] + (f" · size {format_size(verdict['size'])}"
                                               if verdict["size"] else "")),
                 ("Horizon", f"{FWD_HORIZON}D from the next close"
                  + (f" · {_elapsed} of {FWD_HORIZON} elapsed" if _elapsed else "")),
                 *_cb_asof_items]
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
        """Overview — the call, its grade and the chain behind it, in decision order.

        The hero is the call and the one sentence that explains it; under it,
        how the call has done on the sessions it was issued (GRADE→call); then
        the chain — the claim, what can void it, what it has earned, what
        scales it — and the size's arithmetic. All of it is the one verdict.
        Below, labelled evidence about the claim, which moves nothing.
        """
        _top_bar()
        render_hero_card(verdict)
        _render_claim_evidence(st.session_state.get("evidence") or {}, _edges, FWD_HORIZON)

    def _page_mula() -> None:
        _top_bar(windowed=True)
        _safe_render("Mūla", lambda: render_mula_tab(
            engine, ts_filtered, x_axis, signal, model_stats, ts, active_target,
            edge=_edges.get("claim"), horizon=FWD_HORIZON))

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
        # The analog list is matched once over the instrument's fixed
        # precedent term structure and cached, so the page summarises the same
        # list on every rerun instead of re-running the matcher. The key
        # carries the engine config as well as the last price: a different
        # panel or horizon is a different state vector. Nothing on the call
        # path reads it.
        _plast = float(ts["Price"].iloc[-1]) if "Price" in ts.columns and len(ts) else 0.0
        _pkey = f"{st.session_state.get('engine_cache', active_target)}|{len(ts)}|{_plast:.6g}"
        if st.session_state.get("_prec_key") != _pkey:
            _analogs = []
            try:
                from analytics.analogs import find_similar_periods as _fsp
                _analogs = _fsp(ts, active_target, hold_horizons=tuple(_icfg.precedent_horizons),
                                mom_window=_icfg.analog_mom_window) or []
            except Exception as _pe:                   # noqa: BLE001 - one page, not the run
                console.warning(f"Precedent match skipped: {type(_pe).__name__}: {_pe}")
            st.session_state["_precedent_analogs_cache"] = {"pkey": _pkey, "periods": _analogs}
            st.session_state["_prec_key"] = _pkey
        _cached_periods = st.session_state["_precedent_analogs_cache"]["periods"]
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.analog_mom_window, FWD_HORIZON,
            precomputed_periods=_cached_periods))

    def _page_system() -> None:
        _top_bar()
        _safe_render("System", lambda: render_diagnostics_tab(engine, signal, horizon=FWD_HORIZON))

    def _page_data() -> None:
        _top_bar(windowed=True)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    pages = {
        "": [st.Page(_page_overview, title="Overview", icon=":material/dashboard:", default=True)],
        # The claim's engine (Mūla) and the Swayam sizer's engine, then two
        # parallel readings (Convergence, Precedent) that do not move the call.
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
