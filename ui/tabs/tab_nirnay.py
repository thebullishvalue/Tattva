"""
Tattva — Nirnay tab: Constituent regime intelligence, zone distribution, signals, HMM.
तत्त्व (Tattva) — "Principle / Essence"

UI — NIRNAY engine visualization: per-constituent MSF + MMR with regime classification.

Section order (logical analytical flow):
  Phase 1 — State:      Metric Cards (current snapshot)
  Phase 2 — Regime:     HMM State Probabilities
  Phase 3 — Composition: Zone Distribution → Raw Zone Counts
  Phase 4 — Signals:    Signal Counts → Average Unified Signal
  Phase 5 — Drill-down: Individual Constituents
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import chart_layout, style_axes
from ui.components import render_metric_card, render_section_header, section_gap
from core.config import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_MUTED,
    UI_BREADTH_HIGH,
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_LARGE,
)

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "oversold_pct": (
        "Share of basket instruments whose MSF and MMR oscillators are in the oversold zone. "
        "Above 60% often precedes short-term bounce opportunities."
    ),
    "overbought_pct": (
        "Share of basket instruments whose MSF and MMR oscillators are in the overbought zone. "
        "Above 60% signals elevated pullback risk across the basket."
    ),
    "avg_signal": (
        "Mean of the unified oscillator (MSF + MMR) across all constituents. "
        "Below -2 = broad bullish pressure; above +2 = broad bearish pressure; near zero = mixed."
    ),
    "buy_signals": (
        "Count of constituents where the unified oscillator crossed from oversold into neutral, "
        "triggering a regime-change buy signal. More buy signals = broader reversal participation."
    ),
    "sell_signals": (
        "Count of constituents where the unified oscillator crossed from overbought into neutral, "
        "triggering a regime-change sell signal. More sell signals = broader distribution pressure."
    ),
    "trading_days": (
        "Number of trading days in the Nirnay lookback window. "
        "Longer histories produce more stable regime estimates and HMM calibration."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _render_hmm_regime_chart(df_n, dates):
    """Section: HMM State Probabilities — bull/bear regime classification."""
    fig_hmm = go.Figure()
    if "avg_hmm_bull" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=df_n["avg_hmm_bull"].values,
            mode="lines", name="P(Bull)",
            line=dict(color=EMERALD, width=1.5),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        ))
    if "avg_hmm_bear" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=df_n["avg_hmm_bear"].values,
            mode="lines", name="P(Bear)",
            line=dict(color=ROSE, width=1.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.08)",
        ))
    if "avg_hmm_bull" in df_n.columns and "avg_hmm_bear" in df_n.columns:
        neutral_vals = 1.0 - df_n["avg_hmm_bull"].values - df_n["avg_hmm_bear"].values
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=neutral_vals,
            mode="lines", name="P(Neutral)",
            line=dict(color=SLATE, width=1, dash="dot"),
        ))
    fig_hmm.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.08)", line_width=0.5)

    fig_hmm.update_layout(**chart_layout(height=300))
    style_axes(fig_hmm, y_title="Probability", y_range=[0, 1])
    st.plotly_chart(fig_hmm, width='stretch', key="nirnay_hmm_regime")


def _render_zone_distribution_chart(df_n, dates):
    """Section: Zone Distribution Over Time — oversold/overbought share."""
    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Oversold_Pct"].values,
        mode="lines", name="Oversold %",
        fill="tozeroy", fillcolor="rgba(52, 211, 153, 0.12)",
        line=dict(color=EMERALD, width=1.5),
    ))
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Overbought_Pct"].values,
        mode="lines", name="Overbought %",
        fill="tozeroy", fillcolor="rgba(251, 113, 133, 0.12)",
        line=dict(color=ROSE, width=1.5),
    ))
    ymax = max(df_n["Oversold_Pct"].max(), df_n["Overbought_Pct"].max()) * 1.15

    fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
    style_axes(fig_zones, y_title="% of Constituents", y_range=[0, ymax])
    st.plotly_chart(fig_zones, width='stretch', key="nirnay_os_ob")


def _render_raw_zone_counts_chart(df_n, dates):
    """Section: Raw Zone Counts — absolute count of constituents per zone."""
    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Oversold"].values, name="Oversold",
        marker=dict(color="rgba(52,211,153,0.85)"),
    ))
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Overbought"].values, name="Overbought",
        marker=dict(color="rgba(251,113,133,0.85)"),
    ))

    fig_counts.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode="group")
    style_axes(fig_counts, y_title="Count")
    st.plotly_chart(fig_counts, width='stretch', key="nirnay_counts")


def _render_signal_counts_chart(df_n, dates):
    """Section: Signal Counts Over Time — regime-change buy/sell triggers."""
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Buy_Signals"].values,
        mode="lines+markers", name="Buy Signals",
        line=dict(color=EMERALD, width=1.5),
        marker=dict(size=3, color=EMERALD),
    ))
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Sell_Signals"].values,
        mode="lines+markers", name="Sell Signals",
        line=dict(color=ROSE, width=1.5),
        marker=dict(size=3, color=ROSE),
    ))

    fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_signals, y_title="Signal Count")
    st.plotly_chart(fig_signals, width='stretch', key="nirnay_signal_counts")


def _render_avg_unified_signal_chart(df_n, dates):
    """Section: Average Unified Signal — cross-sectional oscillator mean."""
    avg_vals = df_n["Avg_Signal"].values
    colors = [EMERALD if v < -2 else ROSE if v > 2 else "rgba(148,163,184,0.75)" for v in avg_vals]

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, 0, None),
        fill="tozeroy", fillcolor="rgba(251,113,133,0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, None, 0),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.05)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=avg_vals,
        mode="lines+markers", name="Avg Signal",
        line=dict(color=SLATE, width=1.5),
        marker=dict(size=3, color=colors),
    ))
    fig_n.add_hline(y=2, line_color="rgba(251,113,133,0.2)", line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=-2, line_color="rgba(52,211,153,0.2)", line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)

    fig_n.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_n, y_title="Avg Signal", y_range=[-6, 6])
    st.plotly_chart(fig_n, width='stretch', key="nirnay_avg_signal")


def _render_individual_constituents(nirnay_constituent_dfs):
    """Section: Individual Constituents — per-stock oscillator and regime."""
    if nirnay_constituent_dfs:
        sym = st.selectbox("Select Symbol", sorted(nirnay_constituent_dfs.keys()), key="nirnay_sym_select")
        if sym and sym in nirnay_constituent_dfs:
            cdf = nirnay_constituent_dfs[sym].iloc[-100:].copy()
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = [c[0] for c in cdf.columns]
            cols_show = [c for c in ["Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition", "Regime"] if c in cdf.columns]
            st.dataframe(cdf[cols_show] if cols_show else cdf, width='stretch')


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_nirnay_tab(selected_tf: str | None = None) -> None:
    """Nirnay tab — constituent regime intelligence with cyan system identity.

    Analytical flow:
      1. Metric Cards        — "What's the current snapshot?"
      2. HMM Regime          — "What's the hidden regime probability?"
      3. Zone Distribution   — "How many stocks are oversold vs overbought?"
      4. Raw Zone Counts     — "Absolute counts per zone."
      5. Signal Counts       — "Where are the regime-change triggers?"
      6. Avg Unified Signal  — "What's the broad oscillator consensus?"
      7. Individual Stocks   — "What does each stock look like?"
    """

    st.markdown(
        '<div class="tab-bg nirnay"></div>',
        unsafe_allow_html=True,
    )
    nirnay_daily = st.session_state.get("nirnay_daily")
    nirnay_constituent_dfs = st.session_state.get("nirnay_constituent_dfs", {})

    if nirnay_daily is None or nirnay_daily.empty:
        st.info("No Nirnay constituent data available.")
        return

    # ── Normalize columns ───────────────────────────────────────────────
    df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
    col_map = {}
    for c in df_n.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("oversold_pct",):          col_map[c] = "Oversold_Pct"
        elif cl in ("overbought_pct",):      col_map[c] = "Overbought_Pct"
        elif cl in ("neutral_pct",):         col_map[c] = "Neutral_Pct"
        elif cl in ("buy_signals", "buy_signal_count"): col_map[c] = "Buy_Signals"
        elif cl in ("sell_signals", "sell_signal_count"): col_map[c] = "Sell_Signals"
        elif cl in ("avg_signal", "avg_unified_osc"):   col_map[c] = "Avg_Signal"
        elif cl in ("oversold",):            col_map[c] = "Oversold"
        elif cl in ("overbought",):          col_map[c] = "Overbought"
        elif cl in ("neutral",):             col_map[c] = "Neutral"
        elif cl in ("total_analyzed", "num_constituents"): col_map[c] = "Total_Analyzed"
        elif cl in ("avg_hmm_bull",):        col_map[c] = "avg_hmm_bull"
        elif cl in ("avg_hmm_bear",):        col_map[c] = "avg_hmm_bear"
    df_n = df_n.rename(columns=col_map)

    for col, default in [
        ("Oversold_Pct", 0), ("Overbought_Pct", 0), ("Neutral_Pct", 0),
        ("Buy_Signals", 0), ("Sell_Signals", 0), ("Avg_Signal", 0),
        ("Oversold", 0), ("Overbought", 0), ("Neutral", 0),
        ("Total_Analyzed", 0), ("avg_hmm_bull", 0.33), ("avg_hmm_bear", 0.33),
    ]:
        if col not in df_n.columns:
            df_n[col] = default

    # ── Apply the global timeframe selector (3M/6M/1Y/2Y/ALL) ───────────────
    if selected_tf and selected_tf != "ALL":
        try:
            _idx = pd.to_datetime(df_n.index)
            offsets = {
                "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
                "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2),
            }
            _cutoff = _idx.max() - offsets.get(selected_tf, pd.DateOffset(years=1))
            df_n = df_n[_idx >= _cutoff]
        except Exception:
            pass

    dates = list(df_n.index)

    # ── Phase 1: STATE — metric cards ──────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        v = df_n["Oversold_Pct"].iloc[-1]
        render_metric_card("OVERSOLD INSTRUMENTS", f"{v:.0f}%", "of basket", "success" if v > UI_BREADTH_HIGH else "neutral",
                           tooltip=TOOLTIPS["oversold_pct"])
    with c2:
        v = df_n["Overbought_Pct"].iloc[-1]
        render_metric_card("OVERBOUGHT INSTRUMENTS", f"{v:.0f}%", "of basket", "danger" if v > UI_BREADTH_HIGH else "neutral",
                           tooltip=TOOLTIPS["overbought_pct"])
    with c3:
        v = df_n["Avg_Signal"].iloc[-1]
        render_metric_card("AVG UNIFIED SIGNAL", f"{v:.2f}", "<-2 bullish · >+2 bearish", "success" if v < -1 else "danger" if v > 1 else "neutral",
                           tooltip=TOOLTIPS["avg_signal"])
    with c4:
        v = int(df_n["Buy_Signals"].iloc[-1])
        render_metric_card("BUY SIGNALS", str(v), "Oversold-to-neutral crosses", "success" if v > 0 else "neutral",
                           tooltip=TOOLTIPS["buy_signals"])
    with c5:
        v = int(df_n["Sell_Signals"].iloc[-1])
        render_metric_card("SELL SIGNALS", str(v), "Overbought-to-neutral crosses", "danger" if v > 0 else "neutral",
                           tooltip=TOOLTIPS["sell_signals"])
    with c6:
        render_metric_card("LOOKBACK WINDOW", str(len(df_n)), "Trading days", "info",
                           tooltip=TOOLTIPS["trading_days"])

    section_gap()

    # ── Phase 2: REGIME ────────────────────────────────────────────────
    render_section_header(
        "HMM State Probabilities",
        "Probability the index is in a bull or bear regime. P > 0.5 = regime confidence. Frequent crossings = uncertainty.",
        icon="eye",
        accent="violet",
    )
    _render_hmm_regime_chart(df_n, dates)

    section_gap()

    # ── Phase 3: COMPOSITION ───────────────────────────────────────────
    render_section_header(
        "Zone Distribution Over Time",
        "Daily share of basket instruments oversold vs overbought. Rising oversold = accumulation setup. Rising overbought = distribution risk.",
        icon="layers",
        accent="emerald",
    )
    _render_zone_distribution_chart(df_n, dates)

    section_gap()

    render_section_header(
        "Raw Zone Counts",
        "Raw count of constituents per regime zone.",
        icon="bar-chart",
        accent="cyan",
    )
    _render_raw_zone_counts_chart(df_n, dates)

    section_gap()

    # ── Phase 4: SIGNALS ───────────────────────────────────────────────
    render_section_header(
        "Signal Counts Over Time",
        "Daily regime-change signal count. Clusters across basket instruments often precede target reversals.",
        icon="zap",
        accent="rose",
    )
    _render_signal_counts_chart(df_n, dates)

    section_gap()

    render_section_header(
        "Average Unified Signal",
        "Cross-sectional mean of all oscillators. Sustained moves beyond ±2 = broad participation. Whipsaws near zero = no consensus.",
        icon="activity",
    )
    _render_avg_unified_signal_chart(df_n, dates)

    section_gap()

    # ── Phase 5: DRILL-DOWN ────────────────────────────────────────────
    render_section_header(
        "Individual Constituents",
        "Per-stock MSF, MMR, unified signal, and regime. Verify index-level signals are backed by individual names.",
        icon="database",
    )
    _render_individual_constituents(nirnay_constituent_dfs)
