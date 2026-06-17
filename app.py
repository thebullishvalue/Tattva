"""
Tattva v2.0.0 — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — Two systems. One conclusion. Commodity fair-value + basket regime intelligence unified by adaptive convergence.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Warning suppression ──────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*YF.download.*")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
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
    render_nishkarsh_signal_card,
    render_warning_box,
    render_metric_card,
    render_chart_skeleton,
    render_collapsible_section,
    render_collapsible_section_close,
    section_gap,
)
from ui.tabs.tab_aarambh import render_aarambh_tab
from ui.tabs.tab_nirnay import render_nirnay_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_constituent_ohlcv, fetch_macro_live, fetch_commodity_dataset
from data.constituents import get_commodity_basket

# ── Engines ──────────────────────────────────────────────────────────────────
from engines.aarambh import FairValueEngine
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries, apply_polarity

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.conviction_model import UnifiedConvictionModel
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import LOOKBACK_WINDOWS, MIN_DATA_POINTS, STALENESS_DAYS, COLOR_RED, COMMODITY_TARGETS, TARGET_EXCLUDED_PREDICTORS, TARGET_POLARITY


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
)
_RESULTS_CACHE_MAX = 3  # keep the last N configs (≈ the 3 commodities)


# ─── UI Rendering helpers ────────────────────────────────────────────────────

def _render_header() -> None:
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Commodity Fair-Value + Basket Regime Intelligence  |  Unified Convergence"
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
            <p>Walk-forward ensemble regression on the selected target (commodities & FX) vs the macro/FX universe, with conformal z-scores and DDM filtering.</p>
            <div class='spec'>
                <span>Ensemble:</span> Ridge + Huber + ENet + WLS<br>
                <span>Validation:</span> Walk-forward OOS<br>
                <span>Bounds:</span> Conformal prediction
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
                <span>Regime:</span> HMM Probabilities<br>
                <span>Projection:</span> 90D Path + Bands<br>
                <span>Trend:</span> DFA Hurst exponent
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
                <span>Scope:</span> Multi-temporal<br>
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
        <p>Pick a <strong>Target</strong> (Gold · Silver · Copper · Cotton · USD/INR) in the <strong>Sidebar</strong>,<br>
           then execute <strong>Run Analysis</strong> to fetch the live yfinance data and initialize both engines.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_primary_signal(nishkarsh_norm, agreement, aarambh_signal) -> None:
    """Render the hero Nishkarsh convergence signal card.

    The conviction value now reflects the **normalized convergence** shown in
    the Unified Signal plot \u2014 i.e. the average of the two systems' z-scored
    (and clipped) signals, in ``[-1, +1]``. The signal classification and the
    interpretation paragraph follow the same scale.
    """
    if nishkarsh_norm:
        conv = nishkarsh_norm["value"]
        sig = nishkarsh_norm["signal"]
        a_norm = nishkarsh_norm["aarambh_norm"]
        n_norm = nishkarsh_norm["nirnay_norm"]
        agreement_text = "Strong" if agreement > 0.7 else "Moderate" if agreement > 0.5 else "Weak"
        explanation = (
            f"Normalized convergence {conv:+.2f} ({sig}). "
            f"Aarambh contribution: {a_norm:+.2f}; Nirnay contribution: {n_norm:+.2f}. "
            f"Agreement: {agreement:.0%} ({agreement_text}). "
            f"{'Both systems aligned.' if agreement > 0.6 else 'Mixed signals \u2014 wait for clearer convergence.'}"
        )
    else:
        conv = float(aarambh_signal.get("conviction_score", 0))
        sig = aarambh_signal.get("signal", "HOLD")
        explanation = f"Conviction: {conv:+.2f} ({sig})."

    render_nishkarsh_signal_card(
        signal=sig,
        conviction=conv,
        agreement=agreement,
        explanation=explanation,
    )
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

    # Intelligence-mode toggle. Default ON. When OFF, factory weights are
    # used regardless of any saved profile.
    intelligence_mode = st.toggle(
        "Intelligence Mode",
        value=bool(st.session_state.get("intelligence_mode", True)),
        help=(
            "When ON, Tattva uses the persisted calibrated profile for the "
            "selected universe (if one exists). When OFF, Tattva runs on "
            "the factory 0.30 / 0.25 / 0.25 / 0.20 dimension weights and "
            "symmetric ±0.3 / ±0.5 thresholds."
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
        page_title=f"TATTVA | Unified Convergence",
        page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
        layout="wide", initial_sidebar_state="collapsed",
    )
    inject_css()

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

        st.markdown('<div class="sidebar-title">Target</div>', unsafe_allow_html=True)
        commodity_names = list(COMMODITY_TARGETS.keys())
        prev_commodity = st.session_state.get("selected_commodity", commodity_names[0])
        if prev_commodity not in commodity_names:
            prev_commodity = commodity_names[0]
        selected_commodity = st.selectbox(
            "Target", commodity_names,
            index=commodity_names.index(prev_commodity),
            label_visibility="collapsed",
            help="Aarambh fair-value target. Predictors are the commodity/FX universe; "
                 "Nirnay runs on a basket of related ETFs & miners.",
        )

        df = None
        has_data = "data" in st.session_state and "run_analysis" in st.session_state

        if not has_data:
            # Initial load. The fetch pulls the entire macro universe once and
            # is target-agnostic — the chosen commodity only selects Aarambh's
            # target column and Nirnay's basket.
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Fetching commodity data and running analysis..."):
                    _end = pd.Timestamp.today()
                    # Walk-forward needs MIN_DATA_POINTS (1500) daily observations.
                    # ~9 years of calendar history clears that with headroom.
                    _start = _end - pd.Timedelta(days=365 * 9)
                    df, error = fetch_commodity_dataset(_start, _end)
                    if error or df is None:
                        st.error(f"Failed: {error}")
                        return
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
        commodity_options = [c for c in COMMODITY_TARGETS if c in numeric_cols] or numeric_cols
        target_col = st.session_state.get("active_target", commodity_options[0])
        if target_col not in numeric_cols:
            target_col = commodity_options[0]
        active_target_state = target_col

        # Date column is always the dataset's DATE column — auto-detected.
        date_candidates = [c for c in all_cols if "date" in c.lower()]
        date_col = date_candidates[0] if date_candidates else "None"
        active_date_state = date_col

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
            st.caption("Select predictors, then click Apply to recompute.")
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
                st.caption(f"Pending: {', '.join(parts)}")

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

        # ── Model Passport (Sanket-style) ──────────────────────────────
        # Surfaces the active calibrated profile (Intelligence Mode). Each
        # commodity target keys its own per-universe profile.
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
    active_features = list(st.session_state.get("active_features", [c for c in numeric_cols if c != active_target]))
    # Never let the target — or a self-replicating predictor (e.g. GLTR for a
    # precious metal) — leak into its own predictor set.
    _excluded_feats = {active_target, *TARGET_EXCLUDED_PREDICTORS.get(active_target, [])}
    active_features = [f for f in active_features if f not in _excluded_feats]
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Data staleness warning ────────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True).dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                data_age = (now_utc - latest_date).days
                if data_age > STALENESS_DAYS:
                    render_warning_box(
                        title="Stale Data",
                        content=f"Last data point is from {latest_date.strftime('%d %b %Y')} ({data_age} days ago). Analysis may be outdated."
                    )
        except Exception:
            pass

    # ─── Clean & Fit Engine ────────────────────────────────────────────────
    cols = [active_target] + active_features + ([active_date] if active_date != "None" and active_date in df.columns else [])
    data = df[[c for c in cols if c in df.columns]].copy()
    if active_date != "None" and active_date in data.columns:
        data[active_date] = pd.to_datetime(data[active_date], errors="coerce", dayfirst=True)
        data = data.dropna(subset=[active_date]).sort_values(active_date)
    for col in [active_target] + active_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[[active_target] + active_features] = data[[active_target] + active_features].ffill().bfill()
    data = data.dropna(subset=[active_target] + active_features).reset_index(drop=True)
    if len(data) < MIN_DATA_POINTS:
        st.error(f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis.")
        return
    active_features = [f for f in active_features if f in data.columns]
    if not active_features:
        st.error("No valid features found after data cleaning.")
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
    FWD_HORIZON = 10   # forecast horizon (trading days)
    FWD_MOM_K = 20     # trailing momentum window for predictors
    _lvl = data[[active_target] + active_features].astype(float)
    _ret = np.log(_lvl.where(_lvl > 0)).diff().replace([np.inf, -np.inf], np.nan)
    _mom = _ret[active_features].rolling(FWD_MOM_K, min_periods=FWD_MOM_K).sum()
    _fwd = _ret[active_target].rolling(FWD_HORIZON, min_periods=FWD_HORIZON).sum().shift(-FWD_HORIZON)
    # Keep only rows with fully-formed momentum features (drop the warmup head);
    # the forward-target NaN tail is retained for live forecasting.
    _valid = _mom.notna().all(axis=1).to_numpy()
    data = data.loc[_valid].reset_index(drop=True)
    X = _mom.loc[_valid].to_numpy()
    y = np.nan_to_num(_fwd.loc[_valid].to_numpy(), nan=0.0)
    cache_key = f"fwd{FWD_HORIZON}m{FWD_MOM_K}|{active_target}|{'|'.join(sorted(active_features))}|{len(data)}"
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

        # Custom styled progress container
        progress_container = st.empty()

        # ── Phase 1: Data Loading ─────────────────────────────────────────
        console.start_phase("DATA ACQUISITION", 1, 5)
        progress_bar(progress_container, 2, "Resolving Basket", f"{active_target} · related ETFs & miners")

        console.section("Basket Resolution")
        constituents, src_msg = get_commodity_basket(active_target)
        console.item("Target", active_target)
        console.item("Source", src_msg)
        console.item("Count", len(constituents))
        if constituents:
            console.item("Symbols", f"{', '.join(constituents[:3])}...")
        console.success(f"Resolved {len(constituents)}-instrument {active_target} basket")
        progress_bar(progress_container, 5, "Fetching Macro Data", "yfinance · Global Macro ETFs · FX · Commodities")

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
        progress_bar(progress_container, 10, "Fetching OHLCV Data", f"yfinance · {len(constituents)} constituents")

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
        progress_bar(progress_container, 15, "Assembling Macro Indicators", "yfinance · Global Macro ETFs · FX · Commodities")

        console.section("Nirnay Macro Assembly")
        nirnay_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
        if not nirnay_macro_df.empty:
            console.item("YF Symbols", len(nirnay_macro_df.columns))
            console.success(f"Macro indicators: {len(nirnay_macro_df.columns)} × {len(nirnay_macro_df)} rows")
        macro_cols_list = list(nirnay_macro_df.columns) if not nirnay_macro_df.empty else []
        console.end_phase("DATA ACQUISITION")
        progress_bar(progress_container, 20, "Data Acquisition Complete", f"{len(constituent_ohlcv)} Constituents · {len(nirnay_macro_df.columns)} Macros")

        # ── Phase 2: Aarambh FairValueEngine ─────────────────────────────
        console.start_phase("AARAMBH ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running Aarambh Engine", f"Walk-Forward · {len(active_features)} Predictors · {len(data)} Rows")

        console.section("Engine Configuration")
        console.item("Mode", f"Predictive · forecast {FWD_HORIZON}d forward return")
        console.item("Target", active_target)
        console.item("Features", f"{len(active_features)} macro momentum ({FWD_MOM_K}d) → PCA(20) causal")
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
            engine.fit(X, y, feature_names=active_features, forward_signal=True, n_pca_components=20, progress_callback=lambda pct, msg: progress_bar(progress_container, int(20 + pct * 20), "Running Aarambh Engine", msg))
            # Carry the raw price LEVEL on the engine output (returns-space
            # modeling otherwise leaves only return-scale columns). Used by the
            # Aarambh tab for price display and by the Intelligence tuner.
            engine.ts_data["Price"] = data[active_target].values
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
        console.start_phase("NIRNAY ENGINE", 3, 5)
        progress_bar(progress_container, 42, "Running Nirnay Engine", f"MSF+MMR+Regime · {len(constituent_ohlcv)} Constituents")

        nirnay_daily = pd.DataFrame()
        nirnay_constituent_dfs = {}

        if constituent_ohlcv:
            total = len(constituent_ohlcv)
            console.section("Per-Stock Analysis")
            console.item("Constituents", total)
            console.item("MSF Length", 20)
            console.item("ROC Length", 14)
            console.item("Regime Sensitivity", 1.5)
            console.item("Base Weight", 0.6)
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
                        merged, length=20, roc_len=14,
                        regime_sensitivity=1.5, base_weight=0.6,
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
                nirnay_daily = aggregate_constituent_timeseries(nirnay_constituent_dfs)
                # Re-orient breadth to the target's direction for inverse baskets
                # (no-op for co-directional baskets, i.e. all current targets).
                _polarity = TARGET_POLARITY.get(active_target, 1)
                if _polarity < 0:
                    nirnay_daily = apply_polarity(nirnay_daily, _polarity)
                    console.item("Polarity", f"{_polarity} (basket inverted to target)")
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
        _intel_index = st.session_state.get("nishkarsh_index", _intel_universe)
        _intel_enabled = bool(st.session_state.get("intelligence_mode", True))
        if _intel_enabled:
            _prior_w, _prior_t, _prior_profile = _intel_mod.resolve_active(_intel_universe, _intel_index)
        else:
            _prior_w, _prior_t, _prior_profile = (
                _intel_mod.DEFAULT_WEIGHTS.copy(),
                _intel_mod.DEFAULT_THRESHOLDS.copy(),
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

        # First-pass validator uses prior weights when available.
        _validator_weights = _prior_w if _prior_profile is not None else None
        validator = CrossValidator(
            active_weights=_validator_weights,
            expected_constituents=len(constituents) or None,
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
        total_dates = len(aarambh_ts.index)
        for i, ts_idx in enumerate(aarambh_ts.index):
            ts_date = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            date_str = str(ts_date)
            row_a = aarambh_ts.loc[ts_idx]
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[-1]
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
                    "regime_bull": float(row_n.get("Regime_Bull_Pct", 33)),
                    "regime_weak_bull": 0,
                    "regime_bear": float(row_n.get("Regime_Bear_Pct", 33)),
                    "regime_weak_bear": 0,
                    "regime_neutral": float(row_n.get("Regime_Neutral", 34)),
                    "num_constituents": int(row_n.get("Total_Analyzed", 0)),
                }
                overlap_count += 1
            else:
                nirnay_stats = {
                    "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                    "regime_bull": 33, "regime_weak_bull": 0, "regime_bear": 33,
                    "regime_weak_bear": 0, "regime_neutral": 34, "num_constituents": 0,
                }
            validator.compute_convergence(aarambh_sig, nirnay_stats, date_str)
            divergence_detector.detect(aarambh_sig, nirnay_stats, date_str)

            if (i + 1) % 10 == 0 or i == total_dates - 1:
                pct_val = int(78 + (i + 1) / total_dates * 7)
                progress_bar(progress_container, pct_val, "Computing Convergence", f"{i + 1}/{total_dates} Dates Scored")

        console.item("Total Aarambh Dates", len(aarambh_ts))
        console.item("Overlap Dates", overlap_count)
        console.success(f"Convergence scoring complete")

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
        conviction_model = UnifiedConvictionModel()
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
        _final_profile: _intel_mod.IntelligenceProfile | None = None
        if _intel_enabled:
            console.section("Intelligence Calibration")
            _n_trials = int(st.session_state.get("intel_n_trials", 50))
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
            conviction_model = UnifiedConvictionModel()
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
            console.success(f"Re-fit complete with calibrated profile")
            _active_w = _final_profile.weights
            _active_t = _final_profile.thresholds
        else:
            # Intelligence Mode OFF — record the default state.
            _active_w = _intel_mod.DEFAULT_WEIGHTS.copy()
            _active_t = _intel_mod.DEFAULT_THRESHOLDS.copy()

        # Publish to session state so the Passport sidebar + Convergence cards
        # see the calibrated state immediately on the next rerun.
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_thresholds"] = _active_t
        st.session_state["intelligence_active_profile"] = (
            _final_profile.to_dict() if _final_profile is not None else None
        )

        # ── 4d. Normalized convergence with calibrated thresholds ───────
        from convergence.normalization import compute_normalized_convergence
        _nishkarsh_norm = compute_normalized_convergence(
            aarambh_ts, nirnay_daily, thresholds=_active_t,
        )
        if _nishkarsh_norm:
            console.section("Normalized Convergence (UI display)")
            console.item("Conviction", f"{_nishkarsh_norm['value']:+.2f}")
            console.item("Signal", _nishkarsh_norm['signal'])
            console.item("  Aarambh contribution", f"{_nishkarsh_norm['aarambh_norm']:+.2f}")
            console.item("  Nirnay contribution",  f"{_nishkarsh_norm['nirnay_norm']:+.2f}")

        console.section("Divergence Detection")
        progress_bar(progress_container, 93, "Detecting Divergences", "Cross-System Disagreement Analysis")
        events = divergence_detector.get_events()
        console.item("Total Events", len(events))
        if not events.empty:
            event_types = events['divergence_type'].value_counts()
            for etype, count in event_types.items():
                console.item(f"  {etype}", count)
        console.success(f"Divergence analysis complete")

        # ── Walk-Forward Validation (durability check, runs every analysis) ──
        # Re-calibrates on each expanding window and scores IC on the next
        # unseen block. Many genuine OOS grades → distinguishes a durable edge
        # from a lucky recent regime. Results power the Diagnostics tab.
        console.section("Walk-Forward Validation")
        progress_bar(progress_container, 93, "Walk-Forward Validation", "Rolling OOS IC · Re-Calibration")
        try:
            _wf_frame = _intel_mod._build_calibration_frame(
                convergence_df, aarambh_ts, target_col="Price", horizons=_intel_mod.HOLD_HORIZONS,
            )
            _wf_results = _intel_mod.walk_forward_ic(_wf_frame, horizons=_intel_mod.HOLD_HORIZONS)
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
        progress_bar(progress_container, 94, "Convergence Phase Complete", _conv_complete_sub)

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 95, "Storing Results", "Session State · Cache")
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

        # Reuse the normalized convergence computed in the Conviction Model
        # section above — single source of truth from convergence/normalization.py,
        # shared with the metric cards and the Unified Signal plot.
        st.session_state["nishkarsh_conv_normalized"] = _nishkarsh_norm

        # Display signal = what the UI cards show (normalized if available,
        # else fall back to the DDM-derived signal).
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
        _rcache[cache_key] = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
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

    # ─── Primary Signal (Above Tabs, Always Visible) ───────────────────────
    _render_primary_signal(nishkarsh_norm, agreement, signal)

    # ─── Sidebar Discovery Hint ─────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-hint" onclick="document.querySelector('[data-testid=stSidebarCollapse]').click()" title="Open sidebar for configuration">
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
    TIMEFRAMES = {'3M': 63, '6M': 126, '1Y': 252, '2Y': 504, 'ALL': None}

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
            from core.config import TIMEFRAME_TRADING_DAYS
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days):]
    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ─── Keyboard Shortcuts Hint ─────────────────────────────────────────
    from streamlit.components.v1 import html as st_html
    st_html(
        """
        <div class="kbd-shortcuts" id="kbd-shortcuts">
            <div class="shortcut-row"><kbd>1</kbd>–<kbd>5</kbd> Switch tabs</div>
            <div class="shortcut-row"><kbd>R</kbd> Run analysis</div>
            <div class="shortcut-row"><kbd>?</kbd> Toggle shortcuts</div>
        </div>
        <script>
        (function() {
            var shortcutsVisible = false;
            var kbdEl = document.getElementById('kbd-shortcuts');
            function showKbd() {
                if (kbdEl) { kbdEl.classList.toggle('visible'); shortcutsVisible = !shortcutsVisible; }
            }
            document.addEventListener('keydown', function(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
                if (e.key === '?') { e.preventDefault(); showKbd(); return; }
                if (shortcutsVisible) return;
                var tabKeys = Object();
                tabKeys['1'] = 0; tabKeys['2'] = 1; tabKeys['3'] = 2; tabKeys['4'] = 3; tabKeys['5'] = 4;
                if (e.key in tabKeys) {
                    e.preventDefault();
                    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs[tabKeys[e.key]]) tabs[tabKeys[e.key]].click();
                }
                if (e.key === 'r' || e.key === 'R') {
                    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                        var runBtn = document.querySelector('button[kind="primary"]');
                        if (runBtn) runBtn.click();
                    }
                }
            });
        })();
        </script>
        """,
        height=0,
    )

    # ─── Theme Toggle ────────────────────────────────────────────────────
    from ui.components import render_theme_toggle
    render_theme_toggle()

    # ─── Tabs with Lazy Loading + Error Boundaries ─────────────────────────
    # Track which tabs have been rendered (lazy loading)
    if 'rendered_tabs' not in st.session_state:
        st.session_state.rendered_tabs = set()

    # Get current active tab from URL hash or default to 0
    active_tab_idx = 0  # Streamlit doesn't expose active tab index directly, so we render on demand

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "CONVERGENCE", "AARAMBH", "NIRNAY", "DIAGNOSTICS", "DATA",
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
        st.session_state.rendered_tabs.add(0)
        _safe_render("Convergence", lambda: render_convergence_tab(ts_filtered))
    with tab2:
        st.session_state.rendered_tabs.add(1)
        _safe_render("Aarambh", lambda: render_aarambh_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target))
    with tab3:
        st.session_state.rendered_tabs.add(2)
        _safe_render("Nirnay", lambda: render_nirnay_tab(selected_tf=selected_tf))
    with tab4:
        st.session_state.rendered_tabs.add(3)
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats))
    with tab5:
        st.session_state.rendered_tabs.add(4)
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    _render_footer()


if __name__ == "__main__":
    main()
