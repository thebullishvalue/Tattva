"""
Tattva — Diagnostics tab: ML diagnostics from both engines.
तत्त्व (Tattva) — "Principle / Essence"

UI — Model quality assessment: feature importance, residuals, walk-forward performance.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_section_header, section_gap
from core.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_MUTED,
)
from data.cache import all_caches
from data.circuit_breaker import all_circuits, CircuitState

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
CYAN = COLOR_CYAN
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "ou_half_life": (
        "Expected time (in days) for the pricing residual to close halfway back to fair value "
        "after a shock. Shorter half-lives = faster mean reversion = more frequent opportunities."
    ),
    "adf_pvalue": (
        "Tests whether the pricing residual has a unit root (drifts away from fair value). "
        "p < 0.05 rejects the unit root, confirming mean-reversion."
    ),
    "kpss_pvalue": (
        "Corroborating test: checks whether the residual is stationary around a trend. "
        "p > 0.05 fails to reject stationarity — second confirmation of mean-reversion."
    ),
    "hmm_cov_shrinkage": (
        "Covariance regularization for the regime detection model. "
        "Prevents overfitting when estimating regime volatility from limited data."
    ),
    "viterbi_persist": (
        "Probability the current regime (bull/bear) persists into the next period. "
        "Near 1.0 = stable regimes; below 0.9 = frequent switching, lower signal confidence."
    ),
}


def render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats):
    """ML Diagnostics — sections ordered by decision priority (edge first)."""

    # System identity background
    st.markdown(
        '<div class="tab-bg diagnostics"></div>',
        unsafe_allow_html=True,
    )
    _is_forward = bool(getattr(engine, "forward_signal", False))

    # ═══════════════════════════════════════════════════════════════════════
    # 1. EDGE & TRUST — Intelligence Center (calibration + walk-forward)
    #    The out-of-sample IC and durability are the headline diagnostics, so
    #    they lead.
    # ═══════════════════════════════════════════════════════════════════════
    _render_intelligence_center()
    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. RESIDUAL STATIONARITY (OU) — only meaningful in relative-value mode;
    #    in predictive (forecast) mode it runs on the forecast series, so it is
    #    deprioritized and flagged.
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "OU Mean-Reversion Diagnostics",
        ("Computed on the forecast series in predictive mode — informational only, not the signal foundation."
         if _is_forward else
         "Tests whether the pricing residual is stationary — the foundation all mean-reversion signals depend on."),
        icon="crosshair",
        accent="cyan",
    )

    theta_status = "Stable" if signal.get("theta_stable", True) else "Unstable"
    stationarity = "Stationary" if signal["adf_pvalue"] < 0.05 and signal["kpss_pvalue"] > 0.05 else "Non-Stationary"

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("OU HALF-LIFE", f"{signal['ou_half_life']:.0f}d", "Days to close half the pricing gap", "info",
                           tooltip=TOOLTIPS["ou_half_life"])
    with c2:
        adf_class = "success" if signal["adf_pvalue"] < 0.05 else "danger"
        render_metric_card("ADF P-VALUE", f"{signal['adf_pvalue']:.3f}", "Rejects drift if p < 0.05", adf_class,
                           tooltip=TOOLTIPS["adf_pvalue"])
    with c3:
        kpss_class = "success" if signal["kpss_pvalue"] > 0.05 else "danger"
        render_metric_card("KPSS P-VALUE", f"{signal['kpss_pvalue']:.3f}", "Confirms mean-reversion if p > 0.05", kpss_class,
                           tooltip=TOOLTIPS["kpss_pvalue"])

    # Status indicators
    ok_svg = f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{COLOR_GREEN}" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    warn_svg = f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{COLOR_AMBER}" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    stat_icon = ok_svg if "Stationary" in stationarity else warn_svg
    theta_icon = ok_svg if "Stable" in theta_status else warn_svg

    st.markdown(
        f'<div style="display:flex;gap:var(--sp-6);margin-top:var(--sp-3);">'
        f'<span style="font-family:var(--data);font-size:0.78rem;color:var(--ink-secondary);display:inline-flex;align-items:center;gap:0.4rem;">Stationarity: {stat_icon} {stationarity}</span>'
        f'<span style="font-family:var(--data);font-size:0.78rem;color:var(--ink-secondary);display:inline-flex;align-items:center;gap:0.4rem;">\u03b8 Stability: {theta_icon} {theta_status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. FEATURE IMPACT
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Feature Impact on Fair Value",
        "How much each predictor shifts the fair-value estimate now. Top features drive the signal — if they go stale, the signal degrades.",
        icon="bar-chart",
        accent="violet",
    )

    feature_history = engine.get_feature_impact_history()
    if not feature_history.empty:
        if hasattr(engine, "latest_feature_impacts") and engine.latest_feature_impacts:
            impacts = engine.latest_feature_impacts
            _total_feats = len(impacts)
            _top_n = 15
            _items = list(impacts.items())[:_top_n]  # already sorted by contribution desc
            labels = [k for k, _v in _items][::-1]
            vals = [v for _k, v in _items][::-1]

            # Gradient color scale from light slate to bright slate based on relative contribution
            colors = []
            max_val = max(vals) if vals else 1
            for v in vals:
                intensity = v / max_val
                # Light slate (148,163,184) to brighter slate (180,195,215)
                r = int(148 + (180 - 148) * intensity)
                g = int(163 + (195 - 163) * intensity)
                b = int(184 + (215 - 184) * intensity)
                alpha = 0.75 + 0.25 * intensity
                colors.append(f"rgba({r},{g},{b},{alpha:.2f})")

            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
            ))
            fig_imp.update_layout(**chart_layout(height=max(240, len(labels) * 26), show_legend=False))
            fig_imp.update_xaxes(
                showgrid=True, gridcolor="rgba(255,255,255,0.035)", gridwidth=0.5,
                title_text="Contribution %", zeroline=True,
                zerolinecolor="rgba(255,255,255,0.06)", zerolinewidth=0.5,
            )
            fig_imp.update_yaxes(showgrid=False)
            st.plotly_chart(fig_imp, width='stretch', key="diagnostics_feature_impact")
            st.caption(f"Top {len(labels)} of {_total_feats} predictors by current contribution.")

        if not feature_history.empty and len(feature_history) > 0:
            st.markdown(
                '<div style="font-family:var(--display);font-size:0.72rem;font-weight:600;color:var(--ink-tertiary);'
                'text-transform:uppercase;letter-spacing:0.08em;margin:var(--sp-4) 0 var(--sp-2) 0;">Impact History (last 10)</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(feature_history.tail(10), width='stretch', height=200)
    else:
        st.info("Feature impact data not available.")

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SIGNAL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Signal Performance",
        "Walk-forward hit rates across 5D, 10D, 20D forward return horizons.",
        icon="trending",
        accent="emerald",
    )

    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        buy_sig = "\u2713" if p["buy_p_value"] < 0.05 else "~" if p["buy_p_value"] < 0.10 else "\u2014"
        sell_sig = "\u2713" if p["sell_p_value"] < 0.05 else "~" if p["sell_p_value"] < 0.10 else "\u2014"
        perf_rows.append({
            "Period": f"{period}D",
            "Buy HR": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "\u2014",
            "Buy Avg \u0394": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "\u2014",
            "Buy t": f"{p['buy_t_stat']:.2f} {buy_sig}" if p["buy_count"] > 0 else "\u2014",
            "Buy N": p["buy_count"],
            "Sell HR": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "\u2014",
            "Sell Avg \u0394": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "\u2014",
            "Sell t": f"{p['sell_t_stat']:.2f} {sell_sig}" if p["sell_count"] > 0 else "\u2014",
            "Sell N": p["sell_count"],
        })
    st.dataframe(pd.DataFrame(perf_rows), width='stretch', height=160)

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. HMM TELEMETRY
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Regime Detection (HMM)",
        "How the Hidden Markov Model classifies the market over time. Sustained P > 0.5 = confident. Frequent crossings = uncertainty.",
        icon="eye",
        accent="rose",
    )

    c1, c2 = st.columns(2)
    try:
        from analytics.regime import GARCHState, HMMState
        _hmm_persist = f"{HMMState().transition_matrix[0, 0]:.2f}"
        _garch_shrink = f"{GARCHState().omega:.0e}"
    except Exception:
        _hmm_persist, _garch_shrink = "0.98", "1e-4"
    with c1:
        render_metric_card("COVARIANCE SHRINKAGE", _garch_shrink, "Regularization strength", "warning",
                           tooltip=TOOLTIPS["hmm_cov_shrinkage"])
    with c2:
        render_metric_card("REGIME PERSISTENCE", _hmm_persist, "Probability regime holds next period", "info",
                           tooltip=TOOLTIPS["viterbi_persist"])

    nirnay_df = st.session_state.get("nirnay_results", pd.DataFrame())
    if not nirnay_df.empty and "avg_hmm_bull" in nirnay_df.columns:
        fig_hmm = go.Figure()
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bull"],
            name="P(Bull)", line=dict(color=EMERALD, width=1.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        ))
        fig_hmm.add_trace(go.Scatter(
            x=nirnay_df.index, y=nirnay_df["avg_hmm_bear"],
            name="P(Bear)", line=dict(color=ROSE, width=1.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
        ))
        fig_hmm.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.08)", line_width=0.5)

        fig_hmm.update_layout(**chart_layout(height=300))
        style_axes(fig_hmm, y_title="State Probability", x_title=x_title, y_range=[0, 1])
        st.plotly_chart(fig_hmm, width='stretch', key="diagnostics_hmm_plot")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DATA LAYER HEALTH — cache hit rate + circuit breaker state per source
    # ═══════════════════════════════════════════════════════════════════════
    section_gap()
    render_section_header(
        "Data Layer Health",
        "Two-tier cache statistics and circuit-breaker state for each external service.",
        icon="database",
        accent="emerald",
    )

    # ── Caches ────────────────────────────────────────────────────────────
    cache_cols = st.columns(len(all_caches()))
    for col, cache in zip(cache_cols, all_caches()):
        s = cache.stats()
        hit_pct = s["hit_rate"] * 100.0
        total = s["hits"] + s["misses"]
        # Color: green ≥70% hit rate, amber 30-70%, rose <30% (or no data)
        if total == 0:
            color_cls = "neutral"
        elif hit_pct >= 70:
            color_cls = "success"
        elif hit_pct >= 30:
            color_cls = "warning"
        else:
            color_cls = "danger"
        last_fetch = s["last_fetch_time"]
        if last_fetch:
            mins = (pd.Timestamp.now().timestamp() - last_fetch) / 60.0
            sub = f"{s['disk_entries']} disk · last fetch {mins:.0f}m ago"
        else:
            sub = f"{s['disk_entries']} disk · no fetch this run"
        with col:
            render_metric_card(
                f"CACHE · {s['namespace'].upper()}",
                f"{hit_pct:.0f}%" if total else "—",
                sub,
                color_cls,
                tooltip=(
                    f"{s['hits']} hits / {s['misses']} misses · "
                    f"{s['stale_hits']} stale-fallback · "
                    f"{s['writes']} writes · TTL {s['ttl_seconds']}s · version {s['version']}"
                ),
            )

    # ── Circuit Breakers ──────────────────────────────────────────────────
    circ_cols = st.columns(len(all_circuits()))
    for col, cb in zip(circ_cols, all_circuits()):
        st_dict = cb.get_state()
        state = st_dict["state"]
        if state == CircuitState.CLOSED.value:
            color_cls = "success"
            label_val = "CLOSED"
        elif state == CircuitState.HALF_OPEN.value:
            color_cls = "warning"
            label_val = "HALF-OPEN"
        else:
            color_cls = "danger"
            label_val = "OPEN"
        last_fail = st_dict["last_failure"]
        if last_fail:
            mins = (pd.Timestamp.now().timestamp() - last_fail) / 60.0
            sub = f"{st_dict['failure_count']} fails · last {mins:.0f}m ago"
        else:
            sub = f"{st_dict['success_count']} successful calls"
        with col:
            render_metric_card(
                f"CIRCUIT · {st_dict['name'].upper()}",
                label_val,
                sub,
                color_cls,
                tooltip=(
                    f"Threshold: {st_dict['failure_threshold']} failures · "
                    f"Recovery: {st_dict['recovery_timeout']:.0f}s · "
                    f"OPEN blocks calls; HALF-OPEN allows 1 test call after recovery timeout."
                ),
            )


def _render_intelligence_center() -> None:
    """Intelligence Center — read-only diagnostic dashboard.

    Calibration is now automatic during every **Run Analysis** when the
    Intelligence Mode toggle is ON in the sidebar. This panel surfaces:
      • Current calibrated state (Train IC, Val IC, Stability, Trials)
      • Learned weights vs factory defaults (bar chart)
      • Learned classification thresholds (4 cards)
      • Optuna fANOVA factor sensitivity (top drivers)
      • All saved profiles on disk

    There is no calibrate button here — the loop is the single Run
    Analysis flow. Reset / Import / Export controls live in the sidebar
    Passport.
    """
    from convergence import intelligence as intel

    render_section_header(
        "Intelligence Center",
        "Self-Training Convergence Calibration · auto-runs every analysis · diagnostics only",
        icon="cpu",
        accent="violet",
    )

    profile = st.session_state.get("intelligence_active_profile")
    is_calibrated = bool(profile)
    intel_enabled = bool(st.session_state.get("intelligence_mode", True))

    # Top-line status banner
    if not intel_enabled:
        st.markdown(
            '<div style="font-family:var(--data); font-size:0.72rem; color:var(--ink-secondary);'
            'background:rgba(148,163,184,0.05); border:1px solid var(--border);'
            'border-radius:6px; padding:0.7rem 0.9rem; margin-bottom:1rem;">'
            '<b>Intelligence Mode is OFF.</b> Toggle it ON in the sidebar Passport to enable '
            'automatic calibration on the next Run Analysis. The engine is currently using '
            'factory weights (0.30 / 0.25 / 0.25 / 0.20) and symmetric ±0.3 / ±0.5 thresholds.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not is_calibrated:
        st.markdown(
            '<div style="font-family:var(--data); font-size:0.72rem; color:var(--amber);'
            'background:rgba(212,168,83,0.08); border:1px solid rgba(212,168,83,0.22);'
            'border-radius:6px; padding:0.7rem 0.9rem; margin-bottom:1rem;">'
            '<b>No profile yet.</b> Intelligence Mode is ON but no calibrated profile exists '
            'for this universe yet. Click <b>Run Analysis</b> to trigger the first calibration; '
            'a profile will be saved automatically and used on every subsequent run.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Status summary row ──────────────────────────────────────────────
    train_ic = float(profile.get("train_ic", 0.0)) if profile else 0.0
    val_ic   = float(profile.get("val_ic", 0.0)) if profile else 0.0
    n_trials = int(profile.get("n_trials", 0)) if profile else 0
    n_train  = int(profile.get("n_train_dates", 0)) if profile else 0
    n_val    = int(profile.get("n_val_dates", 0)) if profile else 0
    stability = (val_ic / train_ic * 100) if abs(train_ic) > 1e-9 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "STATE",
            "Calibrated" if is_calibrated else ("Disabled" if not intel_enabled else "Default"),
            "engine state",
            "success" if is_calibrated else ("neutral" if not intel_enabled else "warning"),
        )
    with c2:
        render_metric_card(
            "VAL IC", f"{val_ic:+.3f}" if is_calibrated else "—",
            "out-of-sample skill",
            "success" if (is_calibrated and val_ic > 0) else "neutral",
        )
    with c3:
        render_metric_card(
            "STABILITY",
            f"{stability:+.0f}%" if (is_calibrated and abs(train_ic) > 1e-9) else "—",
            "val / train ratio",
            "success" if (is_calibrated and 50 <= stability <= 110) else "warning",
        )
    with c4:
        render_metric_card(
            "TRIALS",
            f"{n_trials}" if is_calibrated else "—",
            "Optuna iterations · last run",
            "info" if is_calibrated else "neutral",
        )

    # ── Walk-Forward Validation (on demand) ─────────────────────────────
    # The single train/holdout split shows the signal works once, recently.
    # This re-calibrates on each expanding window and tests IC on the next
    # unseen block — many genuinely out-of-sample grades across time, so a
    # durable edge (consistently positive) is distinguishable from a lucky
    # recent regime (a couple of big spikes). Computed automatically during
    # each analysis (Convergence phase); this panel just displays it.
    section_gap()
    render_section_header(
        "Walk-Forward Validation",
        "Re-calibrates on each expanding window, tests IC on the next unseen block. "
        "Consistently positive bars = durable edge; a few spikes = lucky regime.",
        icon="activity",
        accent="violet",
    )
    if True:
        _res = st.session_state.get("wf_results")
        if _res:
            _ics = [r["ic"] for r in _res if not pd.isna(r["ic"])]
            if _ics:
                _mean = sum(_ics) / len(_ics)
                _pos = sum(1 for v in _ics if v > 0)
                w1, w2, w3 = st.columns(3)
                with w1:
                    render_metric_card(
                        "MEAN OOS IC", f"{_mean:+.3f}", f"across {len(_ics)} windows",
                        "success" if _mean > 0.02 else "warning" if _mean > 0 else "danger",
                    )
                with w2:
                    render_metric_card(
                        "WINDOWS POSITIVE", f"{_pos}/{len(_ics)}", "IC > 0",
                        "success" if _pos / len(_ics) >= 0.6 else "warning" if _pos / len(_ics) >= 0.4 else "danger",
                    )
                with w3:
                    render_metric_card(
                        "WORST / BEST", f"{min(_ics):+.2f} / {max(_ics):+.2f}", "IC range", "neutral",
                    )
                _xs = [str(r["test_start"])[:10] for r in _res if not pd.isna(r["ic"])]
                _colors = ["rgba(52,211,153,0.85)" if v > 0 else "rgba(251,113,133,0.85)" for v in _ics]
                _fig = go.Figure(go.Bar(x=_xs, y=_ics, marker_color=_colors))
                _fig.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=0.6)
                _fig.update_layout(**chart_layout(height=280))
                style_axes(_fig, y_title="Out-of-sample IC", x_title="Test window start")
                st.plotly_chart(_fig, width='stretch', key="wf_chart")
                st.caption(
                    "Each bar re-calibrates the convergence weights on all prior data, then scores "
                    "rank-IC on the following unseen window. Overlapping forward returns make single "
                    "windows noisy — read the consistency, not any one bar."
                )
        elif _res is not None:
            st.caption("Not enough aligned history for walk-forward (need ~250+ dates).")
        else:
            st.info("Walk-forward runs automatically during each analysis — run an analysis to populate this.")

    # ── Calibration diagnostics (when calibrated) ───────────────────────
    if is_calibrated and profile:
        section_gap()
        render_section_header(
            "Calibration Diagnostics",
            "Train vs validation scores · split size · last calibration timestamp",
            icon="bar-chart",
            accent="violet",
        )
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            render_metric_card(
                "TRAIN IC", f"{train_ic:+.3f}", "in-sample IC vs forward PE",
                "success" if train_ic > 0 else "danger",
            )
        with d2:
            render_metric_card(
                "VAL IC", f"{val_ic:+.3f}", "out-of-sample IC vs forward PE",
                "success" if val_ic > 0 else "danger",
            )
        with d3:
            render_metric_card(
                "TRAIN / VAL",
                f"{n_train} / {n_val}" if (n_train or n_val) else "—",
                "chronological 70/30 split", "info",
            )
        with d4:
            render_metric_card(
                "UPDATED", profile.get("timestamp", "—"),
                "last calibration", "info",
            )

        # ── Learned weights ─────────────────────────────────────────────
        weights = profile.get("weights", {})
        if weights:
            section_gap()
            render_section_header(
                "Learned Weights",
                "Calibrated dimension weights vs factory defaults (0.30 / 0.25 / 0.25 / 0.20)",
                icon="scale",
                accent="cyan",
            )
            from convergence.intelligence import DEFAULT_WEIGHTS, _normalize_weights
            wkeys   = ["w_direction", "w_breadth", "w_magnitude", "w_regime"]
            wlabels = ["Direction", "Breadth", "Magnitude", "Regime"]
            cal_vals = [float(v) for v in _normalize_weights(weights)]
            def_vals = [float(DEFAULT_WEIGHTS[k]) for k in wkeys]
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                x=wlabels, y=def_vals, name="Default",
                marker=dict(color="rgba(148,163,184,0.35)"),
            ))
            fig_w.add_trace(go.Bar(
                x=wlabels, y=cal_vals, name="Calibrated",
                marker=dict(color=AMBER),
            ))
            fig_w.update_layout(**chart_layout(height=260), barmode="group")
            style_axes(fig_w, y_title="Weight share", y_range=[0, max(0.5, max(cal_vals + def_vals) * 1.15)])
            st.plotly_chart(fig_w, width='stretch', key="intel_weights_plot")

        # ── Learned thresholds ──────────────────────────────────────────
        thresholds = profile.get("thresholds", {})
        if thresholds:
            section_gap()
            render_section_header(
                "Learned Thresholds",
                "Calibrated classification cut-points · normalized [-1, +1] scale · "
                "calibrated thresholds may be asymmetric",
                icon="crosshair",
                accent="amber",
            )
            tcols = st.columns(4)
            for col, (k, label, base) in zip(tcols, [
                ("buy_strong",    "STRONG BUY ≤", -0.5),
                ("buy_moderate",  "BUY ≤",         -0.3),
                ("sell_moderate", "SELL ≥",        +0.3),
                ("sell_strong",   "STRONG SELL ≥", +0.5),
            ]):
                val = float(thresholds.get(k, base))
                with col:
                    render_metric_card(
                        label, f"{val:+.3f}",
                        f"factory {base:+.2f} → cal {val:+.3f}",
                        "success" if "BUY" in label else "danger",
                    )

        # ── Factor sensitivity (Optuna fANOVA) ──────────────────────────
        sensitivity = profile.get("sensitivity", {})
        if sensitivity:
            section_gap()
            render_section_header(
                "Factor Sensitivity",
                "Optuna fANOVA importance — which parameters drove the most variance in the objective",
                icon="zap",
                accent="rose",
            )
            sorted_items = sorted(sensitivity.items(), key=lambda kv: kv[1], reverse=True)[:10]
            sens_df = pd.DataFrame(sorted_items, columns=["parameter", "importance_pct"])
            fig_sens = go.Figure(go.Bar(
                x=sens_df["importance_pct"], y=sens_df["parameter"],
                orientation="h", marker=dict(color=CYAN),
            ))
            fig_sens.update_layout(**chart_layout(height=max(240, len(sorted_items) * 28), show_legend=False))
            fig_sens.update_xaxes(title_text="% importance")
            fig_sens.update_yaxes(showgrid=False)
            st.plotly_chart(fig_sens, width='stretch', key="intel_sensitivity_plot")

    # ── Saved profiles list ─────────────────────────────────────────────
    section_gap()
    saved = intel.list_profiles()
    if saved:
        render_section_header(
            "Saved Profiles",
            f"{len(saved)} profile(s) on disk · ~/.cache/tattva/intelligence/",
            icon="database",
            accent="emerald",
        )
        rows = []
        for p in saved:
            rows.append({
                "Universe": p.universe,
                "Index": p.selected_index or "—",
                "Train IC": f"{p.train_ic:+.3f}",
                "Val IC": f"{p.val_ic:+.3f}",
                "Trials": p.n_trials,
                "Updated": p.timestamp,
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', height=min(200, 60 + 35 * len(rows)))
    else:
        render_section_header(
            "Saved Profiles",
            "No profiles on disk yet · run an analysis with Intelligence Mode ON to create one",
            icon="database",
            accent="emerald",
        )
