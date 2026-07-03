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
from ui.components import render_metric_card, render_section_header, section_gap, render_info_box
from convergence.normalization import align_aarambh_nirnay
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
    UI_CONSENSUS_STRONG,
    UI_CONSENSUS_MODERATE,
    UI_CONVRAW_STRONG,
    UI_CONVRAW_MODERATE,
    UI_NIRNAY_AVG_THRESHOLD,
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

    # ── SINGLE SOURCE OF TRUTH ───────────────────────────────────────────────
    # Align Aarambh + Nirnay ONCE, here, before anything renders. The metric cards
    # AND the 3-row plot below both read these exact arrays, so a card can never
    # disagree with the plot point it mirrors (the old bug: card read the raw last
    # ts row, the plot read the Nirnay-aligned last row → drift on calendar gaps).
    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    aligned_dates, aligned_aarambh_raw, aligned_nirnay_raw = align_aarambh_nirnay(
        aarambh_ts, nirnay_daily, filter_dates=filtered_dates,
    )
    has_overlap = bool(aligned_dates)

    norm_a = norm_n = norm_avg = np.array([], dtype=np.float64)
    aligned_conv_raw: list = []
    if has_overlap:
        # Key by the full engine config (target + features + horizon + date range) so
        # switching predictor sets with the same target never reuses stale z-scores.
        # Also fold in content (row count + latest raw Aarambh/Nirnay reading): a
        # "Refresh Data" that updates the LAST bar's value without changing the
        # date-range fingerprint that engine_cache is built from would otherwise
        # keep this key unchanged and silently reuse pre-refresh z-scores against
        # the post-refresh raw series (audit finding C1).
        _last_a = aligned_aarambh_raw[-1] if aligned_aarambh_raw else 0.0
        _last_n = aligned_nirnay_raw[-1] if aligned_nirnay_raw else 0.0
        _np_key = (
            f"conv_norm_causal::{st.session_state.get('engine_cache', st.session_state.get('active_target', ''))}"
            f"|{len(aligned_dates)}|{_last_a:.6g}|{_last_n:.6g}"
        )
        if _np_key not in st.session_state:
            # Compute per-date CAUSAL expanding-window z-scores over the FULL aligned
            # series.  Applying terminal-point μ/σ to a historical slice is look-ahead
            # bias: earlier bars appear less extreme than they were at the time because
            # σ is estimated from data that didn't yet exist.
            _full_dates, full_a, full_n = align_aarambh_nirnay(aarambh_ts, nirnay_daily)
            fa = np.array(full_a, dtype=np.float64)
            fn = np.array(full_n, dtype=np.float64)
            sa, sn = pd.Series(fa), pd.Series(fn)
            na_full = np.clip(
                (fa - sa.expanding().mean().to_numpy())
                / sa.expanding().std().clip(lower=1e-10).fillna(1.0).to_numpy()
                / 3.0, -1.0, 1.0,
            )
            nn_full = np.clip(
                (fn - sn.expanding().mean().to_numpy())
                / sn.expanding().std().clip(lower=1e-10).fillna(1.0).to_numpy()
                / 3.0, -1.0, 1.0,
            )
            def _dk(d):
                return str(d.date()) if hasattr(d, "date") else str(d)
            _p = {
                "a": {_dk(d): v for d, v in zip(_full_dates, na_full)},
                "n": {_dk(d): v for d, v in zip(_full_dates, nn_full)},
                "_n": len(full_a),
            }
            st.session_state[_np_key] = _p
        params = st.session_state[_np_key]
        def _dk(d):
            return str(d.date()) if hasattr(d, "date") else str(d)
        norm_a = np.array([params["a"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_n = np.array([params["n"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_avg = (norm_a + norm_n) / 2.0
        at_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")] if aarambh_ts is not None else None
        for d in aligned_dates:
            d_str = str(d.date()) if hasattr(d, "date") else str(d)
            val = None
            if at_dedup is not None:
                if d in at_dedup.index:
                    val = float(at_dedup.loc[d]["ConvictionRaw"])
                elif "Date" in at_dedup.columns:
                    mask = at_dedup["Date"].astype(str).str.contains(d_str)
                    if mask.any():
                        val = float(at_dedup.loc[mask, "ConvictionRaw"].iloc[0])
            aligned_conv_raw.append(val)

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
        # Mirrors Row 2 of the plot — reads the SAME aligned ConvictionRaw last point
        # (falls back to the raw last ts row only when there is no Nirnay overlap).
        a_conv = None
        if has_overlap and aligned_conv_raw and aligned_conv_raw[-1] is not None:
            a_conv = aligned_conv_raw[-1]
        elif aarambh_ts is not None and "ConvictionRaw" in aarambh_ts.columns:
            a_conv = float(aarambh_ts["ConvictionRaw"].iloc[-1])
        if a_conv is not None:
            render_metric_card("AARAMBH CONVICTION", f"{a_conv:+.2f}", "Market breadth: oversold vs overbought",
                               "success" if a_conv < -UI_CONVICTION_MODERATE else "danger" if a_conv > UI_CONVICTION_MODERATE else "neutral",
                               tooltip=TOOLTIPS["aarambh_conviction"])
        else:
            render_metric_card("AARAMBH CONVICTION", "N/A", "", "neutral")

    with col3:
        # Mirrors Row 3 of the plot — reads the SAME aligned Nirnay Avg Signal point.
        if has_overlap and len(aligned_nirnay_raw):
            n_avg = float(aligned_nirnay_raw[-1])
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

    # Aligned series already computed once at the top (single source of truth with
    # the metric cards). Aarambh-only targets (no Nirnay basket) have no overlap →
    # the cards above still rendered; the plot just can't be drawn.
    if not has_overlap:
        st.warning("No overlapping dates between Aarambh and Nirnay data sources "
                   "(this target runs Aarambh-only — see the cards above).")
        return

    # Short-history guard: z-scoring needs a stable σ. When the FULL Aarambh∩Nirnay
    # overlap is tiny (brand-new sheet target, freshly-listed basket constituents),
    # σ collapses to its 1e-10 floor and the whole normalized plot flat-lines at 0 —
    # which misreads as a confident "neutral". The cards above already show the raw
    # latest reads honestly; here we suppress the misleading plot and say why.
    MIN_CONV_NORM_POINTS = 10
    _n_full = int(params.get("_n", len(aligned_dates)))
    if _n_full < MIN_CONV_NORM_POINTS:
        render_info_box(
            "Building convergence history",
            f"Only {_n_full} overlapping session{'s' if _n_full != 1 else ''} between Aarambh and Nirnay "
            f"so far — too few to z-score into a stable convergence view (the plot would flat-line at zero "
            f"and misread as neutral). The cards above reflect the latest raw reads; this view populates "
            f"once {MIN_CONV_NORM_POINTS}+ shared sessions accrue.",
            color="cyan",
        )
        return

    # Honesty for the carry-forward: when the basket's native data ends before the
    # latest plotted session (its markets closed / haven't posted), say so — those
    # trailing breadth points are carried forward, so they're provisional.
    _nn_last = st.session_state.get("nirnay_native_last")
    _plot_last = aligned_dates[-1] if aligned_dates else None
    try:
        if _nn_last is not None and _plot_last is not None \
                and pd.Timestamp(_nn_last).normalize() < pd.Timestamp(_plot_last).normalize():
            render_info_box(
                "Breadth carried forward",
                f"The constituent basket's data ends {pd.Timestamp(_nn_last):%d %b %Y}; later sessions "
                f"(through {pd.Timestamp(_plot_last):%d %b %Y}) carry its last reads forward — the "
                f"constituents' markets are closed or haven't posted yet, so bottom-up breadth on those "
                f"bars is provisional.",
                color="amber",
            )
    except Exception:
        pass

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
        if v < -UI_CONSENSUS_STRONG:
            avg_colors.append(EMERALD); avg_sizes.append(8)
        elif v <= -UI_CONSENSUS_MODERATE:
            avg_colors.append("rgba(52,211,153,1.0)"); avg_sizes.append(6)
        elif v > UI_CONSENSUS_STRONG:
            avg_colors.append(ROSE); avg_sizes.append(8)
        elif v >= UI_CONSENSUS_MODERATE:
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
    # Calibrated-model overlay (amber trace) intentionally omitted here — the
    # normalized 50/50 consensus (slate) is the plot's own read; the hero card
    # above already shows the calibrated model value separately.
    fig.add_hline(y=UI_CONSENSUS_STRONG, line_dash="dot", line_color="rgba(251,113,133,0.15)", line_width=0.5, row=1, col=1)
    fig.add_hline(y=-UI_CONSENSUS_STRONG, line_dash="dot", line_color="rgba(52,211,153,0.15)", line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=1, col=1)

    # ── Row 2: Base Conviction ────────────────────────────────────────
    conv_vals = [float(v) if v is not None else np.nan for v in aligned_conv_raw]
    conv_colors, conv_sizes = [], []
    for v in aligned_conv_raw:
        if v is None:
            conv_colors.append("rgba(148,163,184,0.90)"); conv_sizes.append(5)
        elif v > UI_CONVRAW_STRONG:
            conv_colors.append(ROSE); conv_sizes.append(7)
        elif v >= UI_CONVRAW_MODERATE:
            conv_colors.append("rgba(251,113,133,1.0)"); conv_sizes.append(6)
        elif v < -UI_CONVRAW_STRONG:
            conv_colors.append(EMERALD); conv_sizes.append(7)
        elif v <= -UI_CONVRAW_MODERATE:
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
    fig.add_hline(y=UI_CONVRAW_STRONG, line_dash="dot", line_color="rgba(251,113,133,0.12)", line_width=0.5, row=2, col=1)
    fig.add_hline(y=-UI_CONVRAW_STRONG, line_dash="dot", line_color="rgba(52,211,153,0.12)", line_width=0.5, row=2, col=1)

    # ── Row 3: Nirnay Avg Signal ──────────────────────────────────────
    nirnay_colors = [EMERALD if v < -UI_NIRNAY_AVG_THRESHOLD else ROSE if v > UI_NIRNAY_AVG_THRESHOLD else "rgba(148,163,184,0.95)" for v in aligned_nirnay_raw]

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
    fig.add_hline(y=UI_NIRNAY_AVG_THRESHOLD, line_dash="dot", line_color="rgba(251,113,133,0.15)", line_width=0.5, row=3, col=1)
    fig.add_hline(y=-UI_NIRNAY_AVG_THRESHOLD, line_dash="dot", line_color="rgba(52,211,153,0.15)", line_width=0.5, row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Normalized", y_range=unified_y, row=1, col=1)
    style_axes(fig, y_title="Conviction", y_range=conv_y, row=2, col=1)
    style_axes(fig, y_title="Avg Signal", y_range=nirnay_y, row=3, col=1)

    st.plotly_chart(fig, width='stretch', key="convergence_overlay")
    st.caption(f"{len(aligned_dates)} overlapping trading days")
