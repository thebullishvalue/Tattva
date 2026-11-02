"""
Tattva — Swayam tab: what the instrument's own price says about itself.
तत्त्व (Tattva) — "Principle / Essence"

UI — the Swayam view bank: MSF + MMR + regime, read across timescales,
information sets and mechanisms on the target's OWN price.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    what does Swayam say?              Net breadth · effective views · edge
  2 ANCHOR    how has that moved?                Breadth over time
  3 TRUST     has it been worth anything?        Walk-forward edge · view weights
  4 STATE     the regime underneath it           HMM state probabilities · average signal
  5 DETAIL    the evidence behind it             Per-view drill-down

The unit everywhere is a VIEW — one timescale × information-set × mechanism
read of a single price series — never an instrument, a constituent or a
stock. Fifteen views of one price are more correlated than fifteen
instruments would be, which is why the effective view count sits in the
answer row: it is how much independent evidence the breadth number carries.

Sign convention, as everywhere in Tattva (core/semantics.py): the oscillator
reads positive when the views see the instrument stretched UP; the call that
implies is contrarian, so OVERSOLD views are bullish and net breadth is
reported bullish-positive (oversold share minus overbought share).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics.adaptive import tier_now
from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (render_kpi_strip, render_section_header, render_empty_state,
                           render_chart_panel, render_table_panel, render_edge_panel,
                           window_cutoff)
from core.config import (
    get_instrument_config, InstrumentConfig,
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_LARGE,
)

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "net_breadth": (
        "Share of Swayam views reading the instrument oversold, minus the share reading it "
        "overbought, skill-weighted. Positive = its own price looks depressed (a bullish lean "
        "under Tattva's contrarian read); negative = stretched up."
    ),
    "oversold_pct": (
        "Share of Swayam views (timescale × information-set × mechanism reads of the target's "
        "OWN price) whose unified oscillator is in the oversold zone, weighted by each view's "
        "own realised skill."
    ),
    "overbought_pct": (
        "Share of Swayam views whose unified oscillator is in the overbought zone, "
        "skill-weighted."
    ),
    "avg_signal": (
        "Skill-weighted mean of the unified oscillator (MSF + MMR) across the views, ±10. "
        "Coloured against this instrument's own p75 of the reading — the same cut-point the "
        "Convergence page uses for it."
    ),
    "n_eff": (
        "How many genuinely independent views the bank amounts to: the eigenvalue "
        "participation ratio of the views' correlation over the full history. Fifteen views "
        "of one price are never fifteen independent opinions; this is how many they are worth."
    ),
}


def _avg_tier(full: pd.DataFrame) -> float:
    """The Avg_Signal display cut-point — causal p75 of the instrument's own history.

    One rule for one reading: the Convergence page colours the same series
    with the same tier, so the two pages cannot call one value bullish and
    neutral at once (the Swayam page used to hard-code ±2).
    """
    try:
        prior = get_instrument_config(st.session_state.get("active_target", "")).ui_swayam_avg_threshold
    except KeyError:
        prior = InstrumentConfig().ui_swayam_avg_threshold
    if full is None or "Avg_Signal" not in getattr(full, "columns", ()):
        return float(prior)
    return tier_now(pd.to_numeric(full["Avg_Signal"], errors="coerce").to_numpy(dtype=float),
                    float(prior), q=0.75)


# ═══════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def _render_breadth_chart(df_n, dates):
    """Net breadth (top) over the oversold / overbought split it is made of."""
    os_ = df_n["Oversold_Pct"].to_numpy(dtype=float)
    ob_ = df_n["Overbought_Pct"].to_numpy(dtype=float)
    net = os_ - ob_
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], shared_xaxes=True,
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=dates, y=np.clip(net, 0, None), fill="tozeroy",
                             fillcolor=chart_rgba("emerald", 0.12), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=np.clip(net, None, 0), fill="tozeroy",
                             fillcolor=chart_rgba("rose", 0.12), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=net, mode="lines", name="Net breadth",
                             line=dict(color=chart_color("cyan"), width=1.4),
                             hovertemplate="%{y:+.0f} pts<extra>Net breadth</extra>"), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=os_, mode="lines", name="Views oversold",
                             fill="tozeroy", fillcolor=chart_rgba("emerald", 0.12),
                             line=dict(color=chart_color("emerald"), width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ob_, mode="lines", name="Views overbought",
                             fill="tozeroy", fillcolor=chart_rgba("rose", 0.12),
                             line=dict(color=chart_color("rose"), width=1.2)), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE + 40))
    style_axes(fig, y_title="Net (pts)", y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title="% of views", y_range=[0, 100], row=2, col=1)
    render_chart_panel(fig, "swayam_breadth", units="% of views", window=True)


def _render_view_weights(weights: dict | None, n_eff: float | None):
    """Which views the breadth number is actually listening to, today."""
    if not weights:
        render_empty_state(
            "No view weights yet",
            "Each view is weighted by its own realised skill once enough of its calls have "
            "resolved; until then every view counts equally.",
            eyebrow="Swayam",
        )
        return
    from engines.swayam.ensemble import SWAYAM_MEMBER_ORDER
    keys = [k for k in SWAYAM_MEMBER_ORDER if k in weights] + \
           sorted(k for k in weights if k not in SWAYAM_MEMBER_ORDER)
    vals = [float(weights[k]) for k in keys]
    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        marker=dict(color=[chart_color("cyan") if v >= 1.0 else chart_rgba("cyan", 0.45) for v in vals]),
        hovertemplate="%{x}: %{y:.2f}× equal weight<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=grid_rgba(0.25), line_width=0.6)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM - 60, show_legend=False))
    style_axes(fig, y_title="× equal weight")
    render_chart_panel(
        fig, "swayam_view_weights", units="today's skill weight per view", windowed=False,
        footer=(f"Dotted line = equal weight. The bank behaves like ≈{n_eff:.1f} independent "
                f"views out of {len(keys)}." if n_eff else "Dotted line = equal weight."),
    )


def _render_regime_chart(df_n, dates):
    """HMM state probabilities, named for what the states are.

    Each view's HMM has a high-oscillator state and a low-oscillator state.
    They used to be labelled Bull and Bear, which reads as a call — and under
    Tattva's contrarian read the "bull" state (views persistently stretched
    UP) is the bearish one. Named for what they measure instead.
    """
    fig = go.Figure()
    if "avg_hmm_bull" in df_n.columns:
        fig.add_trace(go.Scatter(x=dates, y=df_n["avg_hmm_bull"].values, mode="lines",
                                 name="P(stretched up)", line=dict(color=chart_color("rose"), width=1.4),
                                 fill="tozeroy", fillcolor=chart_rgba("rose", 0.08)))
    if "avg_hmm_bear" in df_n.columns:
        fig.add_trace(go.Scatter(x=dates, y=df_n["avg_hmm_bear"].values, mode="lines",
                                 name="P(stretched down)", line=dict(color=chart_color("emerald"), width=1.4),
                                 fill="tozeroy", fillcolor=chart_rgba("emerald", 0.08)))
    if {"avg_hmm_bull", "avg_hmm_bear"} <= set(df_n.columns):
        fig.add_trace(go.Scatter(x=dates, y=1.0 - df_n["avg_hmm_bull"].values - df_n["avg_hmm_bear"].values,
                                 mode="lines", name="P(neutral)",
                                 line=dict(color=chart_rgba("slate", 0.5), width=1, dash="dot")))
    fig.add_hline(y=0.5, line_dash="dot", line_color=grid_rgba(0.08), line_width=0.5)
    fig.update_layout(**chart_layout(height=300))
    style_axes(fig, y_title="Probability", y_range=[0, 1])
    render_chart_panel(fig, "swayam_hmm_regime", units="probability")


def _render_avg_signal_chart(df_n, dates, tier):
    """The views' average oscillator, against this instrument's own p75."""
    v = df_n["Avg_Signal"].to_numpy(dtype=float)
    colors = [chart_color("emerald") if x < -tier else chart_color("rose") if x > tier
              else chart_rgba("slate", 0.6) for x in v]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=np.clip(v, 0, None), fill="tozeroy",
                             fillcolor=chart_rgba("rose", 0.05), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=np.clip(v, None, 0), fill="tozeroy",
                             fillcolor=chart_rgba("emerald", 0.05), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=v, mode="lines+markers", name="Average signal",
                             line=dict(color=chart_color("cyan"), width=1.3),
                             marker=dict(size=3, color=colors)))
    for y, nm in ((tier, "rose"), (-tier, "emerald")):
        fig.add_hline(y=y, line_color=chart_rgba(nm, 0.25), line_width=0.5, line_dash="dot")
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig, y_title="Avg oscillator")
    render_chart_panel(fig, "swayam_avg_signal", units=f"oscillator · tier ±{tier:.2f}")


def _render_individual_views(swayam_view_dfs):
    """Per-view oscillators and regime, in member-bank order."""
    if not swayam_view_dfs:
        return
    from engines.swayam.ensemble import SWAYAM_MEMBER_ORDER
    keys = [k for k in SWAYAM_MEMBER_ORDER if k in swayam_view_dfs]
    keys += sorted(k for k in swayam_view_dfs if k not in SWAYAM_MEMBER_ORDER)
    with st.container(key="filterbar"):
        sym = st.selectbox("View", keys, key="swayam_sym_select")
    if sym and sym in swayam_view_dfs:
        cdf = swayam_view_dfs[sym].iloc[-60:].copy()
        if isinstance(cdf.columns, pd.MultiIndex):
            cdf.columns = [c[0] for c in cdf.columns]
        cols_show = [c for c in ["Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
                                 "Regime", "Vol_Regime", "Change_Point", "Confidence"]
                     if c in cdf.columns]
        render_table_panel(
            cdf[cols_show] if cols_show else cdf, "swayam-view-drilldown",
            units="last 60 sessions", windowed=False, index_label="Date", max_rows=60, max_height=520,
            sign_color_cols={"MSF_Osc", "MMR_Osc", "Unified_Osc"},
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_swayam_tab(selected_tf: str | None = None, edge: dict | None = None,
                      horizon: int = 10) -> None:
    """Swayam tab — self-referential breadth, its record, and its views."""
    swayam_daily = st.session_state.get("swayam_daily")
    swayam_view_dfs = st.session_state.get("swayam_view_dfs", {}) or {}
    if swayam_daily is None or swayam_daily.empty:
        render_empty_state(
            "No Swayam breadth for this target",
            "Swayam reads the target's own OHLCV, and none was available for it this run "
            "(sheet-sourced targets have no OHLCV feed).",
            eyebrow="Swayam",
        )
        return

    full = swayam_daily[~swayam_daily.index.duplicated(keep="last")].copy()
    for col, default in (("Oversold_Pct", 0.0), ("Overbought_Pct", 0.0), ("Avg_Signal", 0.0)):
        if col not in full.columns:
            full[col] = default
    df_n = full
    _cut = window_cutoff(pd.to_datetime(full.index).max(), selected_tf)
    if _cut is not None:
        df_n = full[pd.to_datetime(full.index) >= _cut]
    dates = list(df_n.index)

    last = full.iloc[-1]
    os_, ob_ = float(last["Oversold_Pct"]), float(last["Overbought_Pct"])
    net = os_ - ob_
    avg = float(last["Avg_Signal"])
    tier = _avg_tier(full)
    n_views = len(swayam_view_dfs) or int(last.get("Total_Analyzed", 0) or 0)
    n_eff = st.session_state.get("swayam_n_eff")
    ic = (edge or {}).get("ic")

    stance = ("depressed — its own views lean oversold" if net > 10 else
              "stretched up — its own views lean overbought" if net < -10 else
              "balanced — its own views are split")
    render_section_header(
        "Swayam — The Instrument's Own Read",
        f"Net breadth {net:+.0f} pts: {os_:.0f}% of {n_views} views read the instrument "
        f"oversold and {ob_:.0f}% overbought, so its own price looks {stance}.",
        icon="activity", accent="cyan",
    )
    render_kpi_strip([
        {"label": "Net Breadth", "value": f"{net:+.0f} pts",
         "subtext": "oversold − overbought · bullish-positive",
         "color_class": "success" if net > 10 else "danger" if net < -10 else "neutral",
         "tooltip": TOOLTIPS["net_breadth"]},
        {"label": "Views Oversold", "value": f"{os_:.0f}%", "subtext": f"of {n_views} views",
         "color_class": "success" if os_ >= 50 else "neutral", "tooltip": TOOLTIPS["oversold_pct"]},
        {"label": "Views Overbought", "value": f"{ob_:.0f}%", "subtext": f"of {n_views} views",
         "color_class": "danger" if ob_ >= 50 else "neutral", "tooltip": TOOLTIPS["overbought_pct"]},
        {"label": "Average Signal", "value": f"{avg:+.2f}",
         "subtext": f"tier ±{tier:.2f} (own p75)",
         "color_class": "success" if avg < -tier else "danger" if avg > tier else "neutral",
         "tooltip": TOOLTIPS["avg_signal"]},
        {"label": "Effective Views", "value": f"≈{n_eff:.1f}" if n_eff else "—",
         "subtext": f"independent evidence in {n_views} views", "color_class": "info",
         "tooltip": TOOLTIPS["n_eff"]},
        {"label": "Walk-Forward IC", "value": f"{ic:+.3f}" if ic is not None else "—",
         "subtext": f"net breadth, non-overlapping {horizon}d" if ic is not None else "not yet graded",
         "color_class": ("success" if ic is not None and ic >= 0.05 else
                         "warning" if ic is not None and ic > 0 else
                         "danger" if ic is not None else "neutral")},
    ], max_cols=6, key="swayam-read")

    render_section_header(
        "Breadth Over Time",
        "Top: net breadth, bullish-positive. Bottom: the share of views oversold and overbought "
        "it is made of.",
        icon="layers", accent="cyan",
    )
    _render_breadth_chart(df_n, dates)

    render_section_header(
        "Walk-Forward Edge",
        f"When Swayam's net breadth leaned one way, did the instrument move that way over the "
        f"next {horizon} sessions? Scored exactly as Mūla's call is.",
        icon="trending", accent="cyan",
    )
    render_edge_panel(edge, "swayam", what="Swayam's net breadth")

    render_section_header(
        "Which Views Carry Weight",
        "Each view is weighted by its own realised skill, learned causally; a timescale that "
        "has stopped predicting this instrument fades instead of padding the breadth.",
        icon="bar-chart", accent="cyan",
    )
    _render_view_weights(st.session_state.get("swayam_view_weights"), n_eff)

    render_section_header(
        "Regime",
        "Views-average probability of each view's HMM state. Sustained P > 0.5 = a confident "
        "regime; frequent crossings = none.",
        icon="eye", accent="cyan",
    )
    _render_regime_chart(df_n, dates)

    render_section_header(
        "Average Signal",
        "The views' mean oscillator. Beyond this instrument's own p75 (dotted) = broad "
        "participation; whipsaws near zero = no consensus.",
        icon="activity", accent="cyan",
    )
    _render_avg_signal_chart(df_n, dates, tier)

    render_section_header(
        "Individual Views",
        "Per-view MSF, MMR, unified signal and regime — the readings the breadth is counted from.",
        icon="database",
    )
    _render_individual_views(swayam_view_dfs)
