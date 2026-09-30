"""
Tattva — Diagnostics tab: is the machinery behind the signal actually sound?

Everything here answers a question the other tabs assume the answer to. The
signal tabs show what the system concludes; this one shows whether the
conclusions rest on anything — whether the mispricing reverts, which drivers
move it, whether the signal has held up out of sample, and whether the data
layer feeding all of it is healthy.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Intelligence Center (OOS IC)
  2 ANCHOR    what is the underlying claim?      OU mean-reversion of the gap
  3 SIGNAL    what does it say to do?            Driver importance
  4 STATE     how does that sit historically?    Signal performance · regime
  5 DETAIL    the evidence behind it             Data layer health
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (render_metric_card, render_section_header,                            render_chip, render_empty_state, render_sub_header,
                           render_chart_panel, render_table_panel, render_note)
from core.config import (
    UI_CHART_HEIGHT_MEDIUM,
)
from data.cache import all_caches
from data.circuit_breaker import all_circuits, CircuitState

# (Tab-local colour aliases stood here as module-level constants. They were
# evaluated ONCE at import, when there is no session to read a theme from,
# so every chart drawn through them was frozen to whichever theme happened
# to be active at first import — the same import-time binding that made the
# original COLOR_* constants unable to follow Paper mode. Colours are
# resolved at the call site now, per render.

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
}


def render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats):
    """ML Diagnostics — sections ordered by decision priority (edge first)."""

    # System identity background
    # ═══════════════════════════════════════════════════════════════════════
    # 1. EDGE & TRUST — Intelligence Center (learned weights + walk-forward)
    #    The out-of-sample IC and durability are the headline diagnostics, so
    #    they lead.
    # ═══════════════════════════════════════════════════════════════════════
    _render_intelligence_center()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. RESIDUAL STATIONARITY (OU) — the foundation of the whole signal stack.
    #    These run on the Mūla mispricing gap, which is a candidate mean-
    #    reverting spread, so every statistic here is interpretable. Under the
    #    forecast engine this replaced they ran on a forecast series and had to
    #    be flagged as informational (audit finding F20); that caveat is gone
    #    with the engine that earned it.
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "OU Mean-Reversion Diagnostics",
        "Tests whether the mispricing gap is stationary — the foundation all mean-reversion signals depend on.",
        icon="crosshair",
        accent="cyan",
    )

    theta_status = "Stable" if signal.get("theta_stable", True) else "Unstable"
    stationarity = "Stationary" if signal["adf_pvalue"] < 0.05 and signal["kpss_pvalue"] > 0.05 else "Non-Stationary"

    c1, c2, c3 = st.columns(3)
    with c1:
        # Two half-lives are published. This one is the OU fit over the whole
        # valued history; the engine also carries an ONLINE AR(1) estimate
        # (`gap_half_life`) that tracks the current regime. Showing the
        # historical fit here and the online one on the Mūla tab's regime card
        # is deliberate — a wide split between them means reversion speed
        # today is not the long-run average.
        render_metric_card(
            "OU HALF-LIFE", f"{signal['ou_half_life']:.0f}d",
            "Days to close half the pricing gap",
            "info",
            tooltip=TOOLTIPS["ou_half_life"],
        )
    with c2:
        adf_class = "success" if signal["adf_pvalue"] < 0.05 else "danger"
        render_metric_card("ADF P-VALUE", f"{signal['adf_pvalue']:.3f}", "Rejects drift if p < 0.05", adf_class,
                           tooltip=TOOLTIPS["adf_pvalue"])
    with c3:
        kpss_class = "success" if signal["kpss_pvalue"] > 0.05 else "danger"
        render_metric_card("KPSS P-VALUE", f"{signal['kpss_pvalue']:.3f}", "Confirms mean-reversion if p > 0.05", kpss_class,
                           tooltip=TOOLTIPS["kpss_pvalue"])

    # Status chips \u2014 the shared badge system (ui.components.render_chip)
    # rather than a one-off hand-rolled SVG check/warning icon.
    stat_tone = "success" if "Stationary" in stationarity else "warning"
    theta_tone = "success" if "Stable" in theta_status else "warning"
    st.markdown(
        f'<div class="chip-row">'
        f'<span class="cr-item">Stationarity{render_chip(stationarity, stat_tone, as_html=True)}</span>'
        f'<span class="cr-item">\u03b8 Stability{render_chip(theta_status, theta_tone, as_html=True)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


    # ═══════════════════════════════════════════════════════════════════════
    # 2. FEATURE IMPACT
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Driver Importance on Fair Value",
        "How far the fair-value estimate moves when each asset-class block is removed and the "
        "valuation refit — an ablation, not a coefficient read-off.",
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
            # Contribution intensity is carried by OPACITY against the panel,
            # not by lightening the hue. The previous version interpolated
            # between two hardcoded slates (148,163,184 → 180,195,215) — i.e.
            # it got LIGHTER as contribution rose, which on the Paper theme
            # means the most important driver is the one closest to invisible.
            max_val = max(vals) if vals else 1
            colors = [chart_rgba("slate", round(0.35 + 0.55 * (v / max_val), 2))
                      for v in vals]

            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
            ))
            fig_imp.update_layout(**chart_layout(height=max(240, len(labels) * 26), show_legend=False))
            # Axis type comes from the shared grammar, not from a bespoke
            # pair of update_*axes calls — those set grid and title text but
            # never a tick font, so the block labels and the axis title fell
            # back to Plotly's default ink and were invisible on both grounds.
            style_axes(fig_imp, x_title="Contribution %")
            fig_imp.update_xaxes(gridcolor=grid_rgba(0.035), gridwidth=0.5,
                                 zeroline=True, zerolinecolor=grid_rgba(0.06),
                                 zerolinewidth=0.5)
            fig_imp.update_yaxes(showgrid=False)
            render_chart_panel(fig_imp, "diagnostics_feature_impact", units="contribution", window=True)
            render_note(f"{len(labels)} of {_total_feats} asset-class blocks by current contribution.")

        if not feature_history.empty and len(feature_history) > 0:
            # Unlike the single end-of-run snapshot the previous engine could
            # produce (its attribution existed only for the last walk-forward
            # chunk's fitted models — audit finding C4), this IS a genuine time
            # series: the leave-one-block-out ablations run at every published
            # session, so importance can be watched rotating between blocks.
            # Subsampled to ~120 rows by the engine for render cost.
            render_sub_header("Importance Over Time")
            render_table_panel(feature_history.tail(10), "diag-importance-history",
                               max_rows=10, max_height=240)
    else:
        render_empty_state(
            "Driver contributions unavailable",
            "The engine has not published per-block contributions for this run — "
            "they appear once the valuation pass completes with an admitted cross-section.",
            eyebrow="Diagnostics",
        )


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
    render_table_panel(pd.DataFrame(perf_rows), "diag-signal-performance",
                       units="hit-rate · avg move · t-stat",
                       label_col="Period", max_height=220)


    # ═══════════════════════════════════════════════════════════════════════
    # 4. HMM TELEMETRY
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Regime Detection (HMM)",
        "How the Hidden Markov Model classifies the market over time. Sustained P > 0.5 = confident. Frequent crossings = uncertainty.",
        icon="eye",
        accent="rose",
    )

    # (Two metric cards previously shown here — "Covariance Shrinkage" and
    # "Regime Persistence" — displayed the HMM/GARCH INITIAL PRIOR constants
    # (GARCHState().omega, HMMState().transition_matrix[0,0]), not live
    # telemetry: "Covariance Shrinkage" was actually the GARCH intercept
    # omega (unrelated to covariance shrinkage — nothing in this pipeline
    # regularizes a covariance matrix), and "Regime Persistence" was the
    # transition matrix's INITIAL value, which each constituent then adapts
    # online per-instrument (analytics/regime.py's _adapt_transitions) — the
    # basket-wide adapted value isn't currently returned by run_regime_loop.
    # Removed rather than left displaying constants mislabeled as measured
    # state (audit finding E4).

    # app.py stores the aggregated basket time-series under "swayam_daily"
    # (produced by engines.swayam.aggregate_constituent_timeseries, which
    # carries avg_hmm_bull/avg_hmm_bear) — "swayam_results" was never written
    # anywhere, so this chart previously never rendered (audit finding C3).
    swayam_df = st.session_state.get("swayam_daily", pd.DataFrame())
    if not swayam_df.empty and "avg_hmm_bull" in swayam_df.columns:
        fig_hmm = go.Figure()
        fig_hmm.add_trace(go.Scatter(
            x=swayam_df.index, y=swayam_df["avg_hmm_bull"],
            name="P(Bull)", line=dict(color=chart_color("emerald"), width=1.5),
            fill="tozeroy", fillcolor=chart_rgba("emerald", 0.08),
        ))
        fig_hmm.add_trace(go.Scatter(
            x=swayam_df.index, y=swayam_df["avg_hmm_bear"],
            name="P(Bear)", line=dict(color=chart_color("rose"), width=1.5),
            fill="tozeroy", fillcolor=chart_rgba("rose", 0.08),
        ))
        fig_hmm.add_hline(y=0.5, line_dash="dot", line_color=grid_rgba(0.08), line_width=0.5)

        fig_hmm.update_layout(**chart_layout(height=300))
        style_axes(fig_hmm, y_title="State Probability", x_title=x_title, y_range=[0, 1])
        render_chart_panel(fig_hmm, "diagnostics_hmm_plot", units="probability")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DATA LAYER HEALTH — cache hit rate + circuit breaker state per source
    # ═══════════════════════════════════════════════════════════════════════
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
    """Intelligence Center — what the model learned, and whether it held up.

    This panel used to report a calibration: train IC, val IC, Optuna trial
    count, fANOVA parameter sensitivity, and the list of profiles saved on
    disk. All of it described a full-history fit whose result was applied back
    across that history and persisted between runs, which is what made the
    published record repaint.

    Weights are now learned forward from resolved outcomes, so there is no
    trial count, no saved profile and no train-vs-val split to report — the
    weights at any date ARE out-of-sample with respect to everything after it.
    What is left is the pair of things worth knowing: where the learner ended
    up relative to its prior, and whether the resulting signal actually held up
    out-of-sample across time (the walk-forward IC, which was always the honest
    number here and is now the only one).
    """
    render_section_header(
        "Intelligence Center",
        "Online dimension weighting · learned forward, never refitted · diagnostics only",
        icon="cpu",
        accent="violet",
    )

    from convergence.intelligence import PRIOR_WEIGHTS

    weights = st.session_state.get("intelligence_active_weights") or {}
    wf = st.session_state.get("wf_results") or []

    if not weights:
        render_empty_state(
            "No weights yet",
            "Dimension weights are learned forward from resolved outcomes — "
            "run an analysis in the sidebar to populate them.",
            eyebrow="Intelligence Center",
        )
        return

    # ── Learned vs prior ────────────────────────────────────────────────
    names = [k for k in PRIOR_WEIGHTS if k in weights]
    learned = [float(weights[k]) for k in names]
    prior = [float(PRIOR_WEIGHTS[k]) for k in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=prior, name="Prior",
                         marker=dict(color=chart_rgba("slate", 0.45))))
    fig.add_trace(go.Bar(x=names, y=learned, name="Learned",
                         marker=dict(color=chart_color("accent"))))
    fig.update_layout(**chart_layout(height=260), barmode="group")
    style_axes(fig, y_title="weight")
    render_chart_panel(fig, "intel_weights_plot", units="weight")

    _moved = max(abs(l - p) for l, p in zip(learned, prior))
    _top = names[int(np.argmax(learned))]
    render_note(f"Largest move from prior {_moved:+.3f} · dominant dimension {_top}")

    # ── Out-of-sample durability ────────────────────────────────────────
    render_section_header(
        "Walk-Forward Durability",
        "Expanding-window out-of-sample IC — each window learns on the past and is scored on the next purged block",
        icon="trending",
        accent="emerald",
    )
    if not wf:
        render_empty_state(
            "Not enough scored history",
            "Walk-forward IC needs roughly 250+ scored dates before a window can be "
            "evaluated out of sample. This target has fewer.",
            eyebrow="Durability",
        )
        return

    ics = [r["ic"] for r in wf if np.isfinite(r.get("ic", float("nan")))]
    if not ics:
        render_empty_state(
            "No finite walk-forward windows",
            "Every window scored non-finite — usually a degenerate overlap between the "
            "two engines rather than a model failure.",
            eyebrow="Durability",
        )
        return
    mean_ic = float(np.mean(ics))
    pos = sum(1 for v in ics if v > 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "MEAN OOS IC", f"{mean_ic:+.3f}",
            "Average rank correlation with forward returns across windows.",
            "success" if mean_ic > 0.05 else "warning" if mean_ic > 0 else "danger",
        )
    with c2:
        render_metric_card(
            "POSITIVE WINDOWS", f"{pos}/{len(ics)}",
            "Durability across regimes. A high mean carried by one window is not an edge.",
            "success" if pos > len(ics) * 0.6 else "warning" if pos >= len(ics) * 0.5 else "danger",
        )
    with c3:
        render_metric_card(
            "IC STABILITY", f"{float(np.std(ics)):.3f}",
            "Dispersion of window ICs. Lower = the edge is consistent, not regime-specific.",
            "neutral",
        )

    fig_wf = go.Figure(go.Bar(
        x=[str(r["test_start"])[:10] for r in wf],
        y=[r["ic"] for r in wf],
        marker=dict(color=[chart_color("emerald") if r["ic"] > 0 else chart_color("rose") for r in wf]),
    ))
    fig_wf.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
    fig_wf.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig_wf, y_title="OOS IC")
    render_chart_panel(fig_wf, "intel_wf_plot", units="OOS IC")
