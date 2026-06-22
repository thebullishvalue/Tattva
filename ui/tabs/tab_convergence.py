"""
Tattva — Convergence tab: Unified signal with timeframe filtering.
तत्त्व (Tattva) — "Principle / Essence"

UI — Cross-system convergence visualization: conviction scores with DDM filtering.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_section_header, section_gap
from convergence.normalization import (
    align_aarambh_nirnay,
    compute_norm_params,
    zscore_clip,
)
from core.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_MUTED,
    UI_AGREEMENT_STRONG,
    UI_AGREEMENT_MODERATE,
    UI_CONVICTION_STRONG,
    UI_CONVICTION_MODERATE,
    UI_NIRNAY_BULLISH,
    UI_NIRNAY_BEARISH,
    UI_CHART_HEIGHT_STACKED,
)

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
CYAN = COLOR_CYAN
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "nishkarsh_conviction": (
        "Composite score combining Aarambh (top-down) and Nirnay (bottom-up) into a single "
        "signal. Near 0 = both systems uncertain — avoid new positions. Large absolute values "
        "= high-conviction opportunities."
    ),
    "aarambh_conviction": (
        "Aarambh's fair-value breadth: how many lookback windows see the market as overbought "
        "vs. oversold. Below -20 = most stocks cheap (bullish); above +20 = most expensive (bearish)."
    ),
    "nirnay_avg": (
        "Average technical signal across all basket instruments, computed bottom-up from each "
        "instrument's price action. Negative = net bullish; positive = net bearish. Moves "
        "slowly and confirms (or contradicts) Aarambh's top-down view."
    ),
    "agreement": (
        "How often Aarambh and Nirnay point in the same direction. Above 70% = both systems "
        "agree — trust the signal. Below 50% = they disagree — stay flat until alignment improves."
    ),
}


def _dynamic_range(vals, padding=0.15):
    """Compute a padded y-axis range from a list of values."""
    valid = [v for v in vals if v is not None and not np.isnan(v)]
    if not valid:
        return (-1, 1)
    mn, mx = min(valid), max(valid)
    span = mx - mn if mx != mn else 1.0
    pad = span * padding
    return (round(mn - pad, 2), round(mx + pad, 2))


def render_convergence_tab(ts_filtered=None):
    """Render the convergence dashboard tab with amber-gold system identity."""
    convergence_df = st.session_state.get("convergence_df")
    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    aarambh_ts = st.session_state.get("aarambh_ts")
    nirnay_daily = st.session_state.get("nirnay_daily")

    if convergence_df is None or convergence_df.empty:
        st.info("No convergence data available. Run the analysis first.")
        return

    # System identity background
    st.markdown(
        '<div class="tab-bg convergence"></div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER + METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Convergence Analysis",
        "Aarambh top-down vs Nirnay bottom-up. Agreement = reliable signal. Divergence = stand aside.",
        icon="target",
    )

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        # Mirrors Row 1 of the Unified Signal plot: average of normalized Aarambh
        # + Nirnay z-scores, in [-1, +1].
        if nishkarsh_norm:
            score = nishkarsh_norm["value"]
            sig = nishkarsh_norm["signal"]
            color = "success" if "BUY" in sig else "danger" if "SELL" in sig else "neutral"
            render_metric_card("TATTVA CONVICTION", f"{score:+.2f}", sig, color, tooltip=TOOLTIPS["nishkarsh_conviction"])
        else:
            render_metric_card("TATTVA CONVICTION", "N/A", "Not computed", "neutral")

    with col2:
        # Mirrors Row 2 of the Unified Signal plot: raw Aarambh ConvictionRaw.
        if aarambh_ts is not None and "ConvictionRaw" in aarambh_ts.columns:
            a_conv = aarambh_ts["ConvictionRaw"].iloc[-1]
            render_metric_card("AARAMBH CONVICTION", f"{a_conv:+.2f}", "Market breadth: oversold vs overbought",
                               "success" if a_conv < -UI_CONVICTION_MODERATE else "danger" if a_conv > UI_CONVICTION_MODERATE else "neutral",
                               tooltip=TOOLTIPS["aarambh_conviction"])
        else:
            render_metric_card("AARAMBH CONVICTION", "N/A", "", "neutral")

    with col3:
        # Mirrors Row 3 of the Unified Signal plot: raw Nirnay Avg Signal.
        if nirnay_daily is not None and not nirnay_daily.empty:
            df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")]
            n_avg = 0.0
            for candidate in ("avg_unified_osc", "Avg_Signal", "avg_signal"):
                if candidate in df_n.columns:
                    n_avg = df_n[candidate].iloc[-1]
                    break
            render_metric_card("NIRNAY AVG SIGNAL", f"{n_avg:.2f}", "Bottom-up constituent momentum",
                               "success" if n_avg < UI_NIRNAY_BULLISH else "danger" if n_avg > UI_NIRNAY_BEARISH else "neutral",
                               tooltip=TOOLTIPS["nirnay_avg"])
        else:
            render_metric_card("NIRNAY AVG SIGNAL", "N/A", "No constituent data", "neutral")

    with col4:
        agreement = convergence_df["agreement_ratio"].iloc[-1]
        render_metric_card("AGREEMENT", f"{agreement:.0%}", "Aarambh and Nirnay alignment",
                           "success" if agreement > UI_AGREEMENT_STRONG else "warning" if agreement > UI_AGREEMENT_MODERATE else "neutral",
                           tooltip=TOOLTIPS["agreement"])

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # UNIFIED NORMALIZED SIGNAL — 3-row stacked chart
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Unified Signal — Normalized Convergence",
        "Z-scored to [−1, 1]. Combined signal (top) decomposed into constituent inputs (below).",
        icon="layers",
        accent="cyan",
    )

    # Build filtered date set
    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    # Align Aarambh + Nirnay on overlapping dates (respecting the user's filter)
    aligned_dates, aligned_aarambh_raw, aligned_nirnay_raw = align_aarambh_nirnay(
        aarambh_ts, nirnay_daily, filter_dates=filtered_dates,
    )

    if not aligned_dates:
        st.warning("No overlapping dates between Aarambh and Nirnay data sources.")
        return

    # ── Normalization params: computed once from the FULL dataset, cached, ──
    #    then applied to the filtered slice for plotting.
    if "conv_norm_params" not in st.session_state:
        _, full_a, full_n = align_aarambh_nirnay(aarambh_ts, nirnay_daily)
        st.session_state["conv_norm_params"] = compute_norm_params(full_a, full_n)

    params = st.session_state["conv_norm_params"]

    arr_a = np.array(aligned_aarambh_raw, dtype=np.float64)
    arr_n = np.array(aligned_nirnay_raw, dtype=np.float64)
    norm_a = zscore_clip(arr_a, params["mu_a"], params["sigma_a"])
    norm_n = zscore_clip(arr_n, params["mu_n"], params["sigma_n"])
    norm_avg = (norm_a + norm_n) / 2.0

    # Pre-compute conviction raw for row 2
    aligned_conv_raw = []
    if aarambh_ts is not None:
        at_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
        for d in aligned_dates:
            d_str = str(d.date()) if hasattr(d, "date") else str(d)
            if d in at_dedup.index:
                aligned_conv_raw.append(float(at_dedup.loc[d]["ConvictionRaw"]))
            elif "Date" in at_dedup.columns:
                mask = at_dedup["Date"].astype(str).str.contains(d_str)
                if mask.any():
                    aligned_conv_raw.append(float(at_dedup.loc[mask, "ConvictionRaw"].iloc[0]))
                else:
                    aligned_conv_raw.append(None)
            else:
                aligned_conv_raw.append(None)
    else:
        aligned_conv_raw = [None] * len(aligned_dates)

    # Compute ranges
    unified_y = _dynamic_range(norm_avg)
    conv_y = _dynamic_range(aligned_conv_raw)
    nirnay_y = _dynamic_range(aligned_nirnay_raw)

    # ── Build 3-row chart ───────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # Convergence color mapping
    avg_colors, avg_sizes = [], []
    for v in norm_avg:
        if v < -0.40:
            avg_colors.append(EMERALD); avg_sizes.append(8)
        elif v <= -0.25:
            avg_colors.append("rgba(52,211,153,1.0)"); avg_sizes.append(6)
        elif v > 0.40:
            avg_colors.append(ROSE); avg_sizes.append(8)
        elif v >= 0.25:
            avg_colors.append("rgba(251,113,133,1.0)"); avg_sizes.append(6)
        else:
            avg_colors.append("rgba(148,163,184,0.95)"); avg_sizes.append(5)

    # ── Row 1: Unified normalized ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_a, mode="lines", name="Aarambh",
        line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_n, mode="lines", name="Nirnay",
        line=dict(color="rgba(34,211,238,0.2)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_avg, mode="lines+markers", name="Consensus (50/50)",
        line=dict(color=SLATE, width=2),
        marker=dict(size=avg_sizes, color=avg_colors),
    ), row=1, col=1)
    # Overlay the CALIBRATED convergence (the hero headline's model line, ±100→[-1,1])
    # on top of the normalized 50/50 consensus, so the plot base and the headline
    # reconcile on the same object. Amber = the validated model; slate = consensus.
    _calib_series = st.session_state.get("calibrated_conv_series")
    if _calib_series is not None and len(_calib_series):
        _clut = {str(k): float(v) for k, v in _calib_series.items()}
        _cal_y = [
            _clut.get(str(d.date()) if hasattr(d, "date") else str(d)) for d in aligned_dates
        ]
        if any(v is not None for v in _cal_y):
            fig.add_trace(go.Scatter(
                x=aligned_dates, y=_cal_y, mode="lines", name="Calibrated model",
                line=dict(color=AMBER, width=2), connectgaps=True,
            ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(251,113,133,0.15)", line_width=0.5, row=1, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="rgba(52,211,153,0.15)", line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=1, col=1)

    # ── Row 2: Base Conviction ────────────────────────────────────────
    conv_vals = [v if v is not None else None for v in aligned_conv_raw]
    conv_colors, conv_sizes = [], []
    for v in aligned_conv_raw:
        if v is None:
            conv_colors.append("rgba(148,163,184,0.90)"); conv_sizes.append(5)
        elif v > 40:
            conv_colors.append(ROSE); conv_sizes.append(7)
        elif v >= 20:
            conv_colors.append("rgba(251,113,133,1.0)"); conv_sizes.append(6)
        elif v < -40:
            conv_colors.append(EMERALD); conv_sizes.append(7)
        elif v <= -20:
            conv_colors.append("rgba(52,211,153,1.0)"); conv_sizes.append(6)
        else:
            conv_colors.append("rgba(148,163,184,0.95)"); conv_sizes.append(5)

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.05)", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.05)", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=conv_vals, mode="lines+markers", name="Base Conviction",
        line=dict(color=SLATE, width=1.5),
        marker=dict(size=conv_sizes, color=conv_colors),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="rgba(251,113,133,0.12)", line_width=0.5, row=2, col=1)
    fig.add_hline(y=-40, line_dash="dot", line_color="rgba(52,211,153,0.12)", line_width=0.5, row=2, col=1)

    # ── Row 3: Nirnay Avg Signal ──────────────────────────────────────
    nirnay_colors = [EMERALD if v < -2 else ROSE if v > 2 else "rgba(148,163,184,0.95)" for v in aligned_nirnay_raw]

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_nirnay_raw, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.05)", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_nirnay_raw, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.05)", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=aligned_nirnay_raw, mode="lines+markers", name="Avg Signal",
        line=dict(color=SLATE, width=1.2),
        marker=dict(size=5, color=nirnay_colors),
    ), row=3, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="rgba(251,113,133,0.15)", line_width=0.5, row=3, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="rgba(52,211,153,0.15)", line_width=0.5, row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Normalized", y_range=unified_y, row=1, col=1)
    style_axes(fig, y_title="Conviction", y_range=conv_y, row=2, col=1)
    style_axes(fig, y_title="Avg Signal", y_range=nirnay_y, row=3, col=1)

    st.plotly_chart(fig, width='stretch', key="convergence_overlay")
    st.caption(f"{len(aligned_dates)} overlapping trading days")
