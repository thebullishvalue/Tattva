"""
Tattva — Swayam tab: self-referential breadth, regime composition, per-view detail.
तत्त्व (Tattva) — "Principle / Essence"

UI — the Swayam view bank: MSF + MMR + regime, read across timescales,
information sets and mechanisms on the target's OWN price.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Effective view count
  2 ANCHOR    what is the underlying claim?      Breadth snapshot cards
  3 SIGNAL    what does it say to do?            Zone distribution → counts
  4 STATE     how does that sit historically?    HMM regime probabilities
  5 DETAIL    the evidence behind it             Signal counts → per-view drill-down

Trust leads because breadth across 15 views of ONE price is more internally
correlated than a genuine cross-section, and the effective-view count is what
tells you how much independent evidence is actually behind the number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.adaptive import tier_now

from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (render_metric_card, render_section_header, render_empty_state,
                           render_chart_panel, render_table_panel)
from core.config import (
    get_instrument_config, InstrumentConfig, # per-instrument breadth tier
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_LARGE,
)

# (Tab-local colour aliases stood here as module-level constants. They were
# evaluated ONCE at import, when there is no session to read a theme from,
# so every chart drawn through them was frozen to whichever theme happened
# to be active at first import — the same import-time binding that made the
# original COLOR_* constants unable to follow Paper mode. Colours are
# resolved at the call site now, per render.

# ── Tooltip definitions ────────────────────────────────────────────────────
# Tooltips describe the view bank: every reading here counts Swayam views
# (timescale x information-set x mechanism reads of the target's OWN OHLCV).
# A second wording set existed for the basket read, which counted constituent
# instruments; it went with that engine.
TOOLTIPS = {
    "oversold_pct": (
        "Share of Swayam views (timescale/information-set/mechanism reads of the target's OWN "
        "OHLCV) whose MSF and MMR oscillators are in the oversold zone. Above 60% often precedes "
        "short-term bounce opportunities."
    ),
    "overbought_pct": (
        "Share of Swayam views whose MSF and MMR oscillators are in the overbought zone. "
        "Above 60% signals elevated pullback risk."
    ),
    "avg_signal": (
        "Mean of the unified oscillator (MSF + MMR) across all Swayam views. "
        "Below -2 = broad bullish pressure; above +2 = broad bearish pressure; near zero = mixed."
    ),
    "buy_signals": (
        "Count of views where the unified oscillator crossed from oversold into neutral, "
        "triggering a regime-change buy signal. More = broader multi-scale reversal agreement."
    ),
    "sell_signals": (
        "Count of views where the unified oscillator crossed from overbought into neutral, "
        "triggering a regime-change sell signal. More = broader multi-scale distribution agreement."
    ),
    "trading_days": (
        "Number of trading days in the Swayam lookback window. "
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
            line=dict(color=chart_color("emerald"), width=1.5),
            fill="tozeroy", fillcolor=chart_rgba("emerald", 0.08),
        ))
    if "avg_hmm_bear" in df_n.columns:
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=df_n["avg_hmm_bear"].values,
            mode="lines", name="P(Bear)",
            line=dict(color=chart_color("rose"), width=1.5),
            fill="tozeroy", fillcolor=chart_rgba("rose", 0.08),
        ))
    if "avg_hmm_bull" in df_n.columns and "avg_hmm_bear" in df_n.columns:
        neutral_vals = 1.0 - df_n["avg_hmm_bull"].values - df_n["avg_hmm_bear"].values
        fig_hmm.add_trace(go.Scatter(
            x=dates, y=neutral_vals,
            mode="lines", name="P(Neutral)",
            line=dict(color=chart_rgba("slate", 0.4), width=1, dash="dot"),
        ))
    fig_hmm.add_hline(y=0.5, line_dash="dot", line_color=grid_rgba(0.08), line_width=0.5)

    fig_hmm.update_layout(**chart_layout(height=300))
    style_axes(fig_hmm, y_title="Probability", y_range=[0, 1])
    render_chart_panel(fig_hmm, "swayam_hmm_regime", units="probability", window=True)


def _render_zone_distribution_chart(df_n, dates):
    """Section: Zone Distribution Over Time — oversold/overbought share."""
    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Oversold_Pct"].values,
        mode="lines", name="Oversold %",
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.12),
        line=dict(color=chart_color("emerald"), width=1.5),
    ))
    fig_zones.add_trace(go.Scatter(
        x=dates, y=df_n["Overbought_Pct"].values,
        mode="lines", name="Overbought %",
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.12),
        line=dict(color=chart_color("rose"), width=1.5),
    ))
    ymax = max(df_n["Oversold_Pct"].max(), df_n["Overbought_Pct"].max()) * 1.15

    fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE))
    style_axes(fig_zones, y_title="% of Constituents", y_range=[0, ymax])
    render_chart_panel(fig_zones, "swayam_os_ob", units="% of views")


def _render_raw_zone_counts_chart(df_n, dates):
    """Section: Raw Zone Counts — absolute count of views per zone."""
    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Oversold"].values, name="Oversold",
        marker=dict(color=chart_rgba("emerald", 0.85)),
    ))
    fig_counts.add_trace(go.Bar(
        x=dates, y=df_n["Overbought"].values, name="Overbought",
        marker=dict(color=chart_rgba("rose", 0.85)),
    ))

    fig_counts.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode="group")
    style_axes(fig_counts, y_title="Count")
    render_chart_panel(fig_counts, "swayam_counts", units="count")


def _render_signal_counts_chart(df_n, dates):
    """Section: Signal Counts Over Time — regime-change buy/sell triggers."""
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Buy_Signals"].values,
        mode="lines+markers", name="Buy Signals",
        line=dict(color=chart_color("emerald"), width=1.5),
        marker=dict(size=3, color=chart_color("emerald")),
    ))
    fig_signals.add_trace(go.Scatter(
        x=dates, y=df_n["Sell_Signals"].values,
        mode="lines+markers", name="Sell Signals",
        line=dict(color=chart_color("rose"), width=1.5),
        marker=dict(size=3, color=chart_color("rose")),
    ))

    fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_signals, y_title="Signal Count")
    render_chart_panel(fig_signals, "swayam_signal_counts", units="count")


def _render_avg_unified_signal_chart(df_n, dates):
    """Section: Average Unified Signal — cross-sectional oscillator mean."""
    avg_vals = df_n["Avg_Signal"].values
    colors = [chart_color("emerald") if v < -2 else chart_color("rose") if v > 2 else chart_rgba("slate", 0.75) for v in avg_vals]

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.05),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=np.clip(avg_vals, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.05),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_n.add_trace(go.Scatter(
        x=dates, y=avg_vals,
        mode="lines+markers", name="Avg Signal",
        line=dict(color=chart_rgba("slate", 0.4), width=1.5),
        marker=dict(size=3, color=colors),
    ))
    fig_n.add_hline(y=2, line_color=chart_rgba("rose", 0.2), line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=-2, line_color=chart_rgba("emerald", 0.2), line_width=0.5, line_dash="dot")
    fig_n.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5)

    fig_n.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_n, y_title="Avg Signal", y_range=[-6, 6])
    render_chart_panel(fig_n, "swayam_avg_signal", units="oscillator")


def _render_individual_views(swayam_view_dfs):
    """Section: Individual Views — per-view oscillator and regime.

    The drop-down lists views in member-bank order so timescale groups stay
    adjacent (L10·FULL next to L10·PRICE) rather than being scattered by an
    alphabetical sort. Any view not in the declared bank sorts after it.
    """
    if swayam_view_dfs:
        from engines.swayam.ensemble import SWAYAM_MEMBER_ORDER
        keys = [k for k in SWAYAM_MEMBER_ORDER if k in swayam_view_dfs]
        keys += sorted(k for k in swayam_view_dfs if k not in SWAYAM_MEMBER_ORDER)
        # In a filter strip, like every other panel-scoped control.
        with st.container(key="filterbar"):
            sym = st.selectbox("View", keys, key="swayam_sym_select")
        if sym and sym in swayam_view_dfs:
            cdf = swayam_view_dfs[sym].iloc[-100:].copy()
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = [c[0] for c in cdf.columns]
            # Surface the full regime-intelligence stack per view: MSF/MMR
            # oscillators + the HMM/GARCH(Vol_Regime)/CUSUM(Change_Point) outputs that
            # the engine computes but the aggregate view doesn't expose.
            cols_show = [c for c in [
                "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
                "Regime", "Vol_Regime", "Change_Point", "Confidence",
            ] if c in cdf.columns]
            _shown = cdf[cols_show] if cols_show else cdf
            # Oscillators are signed [-10,+10] — colour them emerald/rose by sign,
            # matching Pragyam's per-signal columns.
            render_table_panel(
                _shown, "swayam-view-drilldown", units="last 60 sessions",
                index_label="Date", max_rows=60, max_height=520,
                sign_color_cols={"MSF_Osc", "MMR_Osc", "Unified_Osc"},
            )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_swayam_tab(selected_tf: str | None = None) -> None:
    """Swayam tab — self-referential breadth with cyan system identity.

    Analytical flow:
      1. Metric Cards        — "What's the current snapshot?"
      2. HMM Regime          — "What's the hidden regime probability?"
      3. Zone Distribution   — "How many stocks are oversold vs overbought?"
      4. Raw Zone Counts     — "Absolute counts per zone."
      5. Signal Counts       — "Where are the regime-change triggers?"
      6. Avg Unified Signal  — "What's the broad oscillator consensus?"
      7. Individual Stocks   — "What does each stock look like?"
    """

    # System identity background
    swayam_daily = st.session_state.get("swayam_daily")
    swayam_view_dfs = st.session_state.get("swayam_view_dfs", {})
    tooltips = TOOLTIPS

    if swayam_daily is None or swayam_daily.empty:
        render_empty_state(
            "No Swayam breadth data available",
            "The self-referential view bank hasn't produced a breadth read for this target yet.",
            eyebrow="Swayam",
            action_label="Run analysis in the sidebar, then return to this page.",
        )
        return

    # Correlation disclosure (SWAYAM_PLAN.md §6.4): the basket source line
    # becomes the swayam src_msg naturally (set in app.py); append the
    # honest correlation caveat — self-ensemble views share one price
    # series, so this reads breadth across independent CAUSAL ANGLES on the
    # target, not independent cross-sectional names.
    # ── Normalize columns ───────────────────────────────────────────────
    df_n = swayam_daily[~swayam_daily.index.duplicated(keep="last")].copy()
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

    # The noun for "what's being counted". Kept as locals because the schema
    # is shared with the removed basket read, which counted instruments.
    _unit, _unit_of, _plural = "views", "of views", "views"

    # ── Phase 1: STATE — metric cards ──────────────────────────────────
    try:   # per-instrument breadth-alert tier (shadow module global)
        _prior_bh = get_instrument_config(st.session_state.get("active_target", "")).ui_breadth_high
        # Causal p90 of this instrument's own oversold-breadth history — a
        # 15-view bank on a quiet name simply does not reach the pooled 60%.
        _os = (df_n["Oversold_Pct"].to_numpy(dtype=float)
               if "Oversold_Pct" in df_n.columns else None)
        UI_BREADTH_HIGH = (tier_now(_os, _prior_bh, q=0.90) if _os is not None and len(_os)
                           else _prior_bh)
    except KeyError:
        UI_BREADTH_HIGH = InstrumentConfig().ui_breadth_high
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        v = df_n["Oversold_Pct"].iloc[-1]
        render_metric_card("OVERSOLD INSTRUMENTS", f"{v:.0f}%", _unit_of, "success" if v > UI_BREADTH_HIGH else "neutral",
                           tooltip=tooltips["oversold_pct"])
    with c2:
        v = df_n["Overbought_Pct"].iloc[-1]
        render_metric_card("OVERBOUGHT INSTRUMENTS", f"{v:.0f}%", _unit_of, "danger" if v > UI_BREADTH_HIGH else "neutral",
                           tooltip=tooltips["overbought_pct"])
    with c3:
        v = df_n["Avg_Signal"].iloc[-1]
        render_metric_card("AVG UNIFIED SIGNAL", f"{v:.2f}", "<-2 bullish · >+2 bearish", "success" if v < -2 else "danger" if v > 2 else "neutral",
                           tooltip=tooltips["avg_signal"])
    with c4:
        v = int(df_n["Buy_Signals"].iloc[-1])
        render_metric_card("BUY SIGNALS", str(v), "Oversold-to-neutral crosses", "success" if v > 0 else "neutral",
                           tooltip=tooltips["buy_signals"])
    with c5:
        v = int(df_n["Sell_Signals"].iloc[-1])
        render_metric_card("SELL SIGNALS", str(v), "Overbought-to-neutral crosses", "danger" if v > 0 else "neutral",
                           tooltip=tooltips["sell_signals"])
    with c6:
        render_metric_card("LOOKBACK WINDOW", str(len(df_n)), "Trading days", "info",
                           tooltip=tooltips["trading_days"])


    # ── Phase 2: REGIME ────────────────────────────────────────────────
    render_section_header(
        "HMM State Probabilities",
        f"{_unit.capitalize()}-average probability of a bull vs bear regime (mean of per-{_plural[:-1]} HMM states). P > 0.5 = regime confidence. Frequent crossings = uncertainty.",
        icon="eye",
        accent="violet",
    )
    _render_hmm_regime_chart(df_n, dates)


    # ── Phase 3: COMPOSITION ───────────────────────────────────────────
    render_section_header(
        "Zone Distribution Over Time",
        f"Daily share of {_plural} oversold vs overbought. Rising oversold = accumulation setup. Rising overbought = distribution risk.",
        icon="layers",
        accent="emerald",
    )
    _render_zone_distribution_chart(df_n, dates)


    render_section_header(
        "Raw Zone Counts",
        f"Raw count of {_plural} per regime zone.",
        icon="bar-chart",
        accent="cyan",
    )
    _render_raw_zone_counts_chart(df_n, dates)


    # ── Phase 4: SIGNALS ───────────────────────────────────────────────
    render_section_header(
        "Signal Counts Over Time",
        f"Daily regime-change signal count. Clusters across {_plural} often precede target reversals.",
        icon="zap",
        accent="rose",
    )
    _render_signal_counts_chart(df_n, dates)


    render_section_header(
        "Average Unified Signal",
        "Cross-sectional mean of all oscillators. Sustained moves beyond ±2 = broad participation. Whipsaws near zero = no consensus.",
        icon="activity",
    )
    _render_avg_unified_signal_chart(df_n, dates)


    # ── Phase 5: DRILL-DOWN ────────────────────────────────────────────
    render_section_header(
        "Individual Views",
        "Per-view MSF, MMR, unified signal, and regime. Verify the bank's "
        "breadth read is backed by individual causal angles on the target.",
        icon="database",
    )
    _render_individual_views(swayam_view_dfs)
