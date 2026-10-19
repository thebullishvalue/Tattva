"""
Tattva — Convergence tab: where the two engines agree, and where they do not.

Evidence about the call, not a second call. This page used to open with a
card titled "TATTVA CONVICTION" — the 50/50 consensus with BUY/SELL words of
its own — which could, and did, contradict the Overview's verdict while
wearing the product's name. The verdict lives on Overview; here every number
is a reading of how far Mūla and Swayam concur, in lean words (bullish /
bearish), never in action words.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    do they agree, and which way?      Agreement · consensus · the two leans
  2 ANCHOR    how has that moved?                The consensus and its two inputs
  3 SIGNAL    what did the learner build?        The weighted composite (+ its DDM)
  4 TRUST     has agreement been worth anything? Walk-forward edge of the composite
  5 DETAIL    where they disagreed               Recent divergences

Sign convention (core/semantics.py): every series here is a READING —
negative = stretched down (cheap / oversold), which the call reads as bullish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics.adaptive import tier_now
from core.config import DIV_LOOKBACK, UI_CHART_HEIGHT_STACKED, UI_CHART_HEIGHT_MEDIUM
from core.config import get_instrument_config
from core.semantics import CLAIM_SIGMA, valuation_state
from convergence.normalization import consensus_series, lean_label
from ui.theme import chart_layout, style_axes, chart_color, chart_rgba, grid_rgba, panel_bg
from ui.components import (render_kpi_strip, render_section_header, render_empty_state,
                           render_info_box, render_chart_panel, render_table_panel,
                           render_edge_panel, render_note)

TOOLTIPS = {
    "agreement": (
        "Today's weighted agreement between Mūla and Swayam across four dimensions — "
        "direction, breadth, magnitude and regime — with the weights the learner has "
        "earned for each. Two causal transforms of ONE price agree more often than chance "
        "by construction, so read it against this instrument's own history (the tier), "
        "not against 50%."
    ),
    "consensus": (
        "The average of the two engines' readings, each scaled by its own history without "
        "re-centring, so zero still means 'no reading': Mūla's oscillator and Swayam's "
        "average signal. Negative = both lean stretched down (bullish under the contrarian "
        "read). Classified at this instrument's own p90 / p75."
    ),
    "composite": (
        "The convergence learner's construction: agreement strength, weighted by each "
        "dimension's learned signed skill, pointed in the direction the two engines lean. "
        "±100; negative = bullish agreement. This is the series its walk-forward edge grades."
    ),
}


def _lean_tone(label: str) -> str:
    return "success" if "BULLISH" in label else "danger" if "BEARISH" in label else "neutral"


def _window_index(ts_filtered) -> pd.Timestamp | None:
    """First date of the active chart window (None = all)."""
    if ts_filtered is None or ts_filtered.empty or "Date" not in ts_filtered.columns:
        return None
    first = pd.to_datetime(ts_filtered["Date"], errors="coerce").min()
    return first.normalize() if pd.notna(first) else None


def _marker_style(values, strong, moderate):
    """Colour carries direction, size carries tier — from each date's own tier."""
    colors, sizes = [], []
    for v, s, m in zip(values, strong, moderate):
        if not np.isfinite(v):
            colors.append(chart_rgba("slate", 0.4)); sizes.append(3)
        elif v <= -s:
            colors.append(chart_color("emerald")); sizes.append(8)
        elif v <= -m:
            colors.append(chart_rgba("emerald", 0.9)); sizes.append(6)
        elif v >= s:
            colors.append(chart_color("rose")); sizes.append(8)
        elif v >= m:
            colors.append(chart_rgba("rose", 0.9)); sizes.append(6)
        else:
            colors.append(chart_rgba("slate", 0.5)); sizes.append(4)
    return dict(size=sizes, color=colors, line=dict(width=1, color=panel_bg()))


def _render_consensus_chart(cs: pd.DataFrame, sw_tier: float) -> None:
    """Consensus (top) over the two readings it averages."""
    d = list(cs.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.46, 0.27, 0.27],
                        vertical_spacing=0.05)
    c = cs["Consensus"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=d, y=np.clip(c, 0, None), fill="tozeroy",
                             fillcolor=chart_rgba("rose", 0.06), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=np.clip(c, None, 0), fill="tozeroy",
                             fillcolor=chart_rgba("emerald", 0.06), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=c, mode="lines+markers", name="Consensus",
                             line=dict(color=chart_rgba("slate", 0.6), width=1.2),
                             marker=_marker_style(c, cs["Strong"], cs["Moderate"]),
                             customdata=cs["Label"],
                             hovertemplate="%{y:+.2f} · %{customdata}<extra>Consensus</extra>"),
                  row=1, col=1)
    # The tier a marker was judged against, drawn as a step so the dots and the
    # line always tell the same story about the same date.
    _step = dict(mode="lines", line=dict(width=0.5, dash="dot"), hoverinfo="skip", showlegend=False)
    fig.add_trace(go.Scatter(x=d, y=cs["Strong"], line_color=chart_rgba("rose", 0.2), **_step), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=-cs["Strong"], line_color=chart_rgba("emerald", 0.2), **_step), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=1, col=1)

    a = cs["RawA"].to_numpy(dtype=float)
    fig.add_hrect(y0=-CLAIM_SIGMA, y1=CLAIM_SIGMA, fillcolor=grid_rgba(0.05), line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=a, mode="lines", name="Mūla oscillator",
                             line=dict(color=chart_color("violet"), width=1.2),
                             hovertemplate="%{y:+.2f}σ<extra>Mūla</extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=2, col=1)

    n = cs["RawN"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=d, y=n, mode="lines", name="Swayam average signal",
                             line=dict(color=chart_color("cyan"), width=1.2),
                             hovertemplate="%{y:+.2f}<extra>Swayam</extra>"), row=3, col=1)
    for y, nm in ((sw_tier, "rose"), (-sw_tier, "emerald")):
        fig.add_hline(y=y, line_dash="dot", line_color=chart_rgba(nm, 0.25), line_width=0.5, row=3, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=3, col=1)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Consensus", row=1, col=1)
    style_axes(fig, y_title="Mūla σ", row=2, col=1)
    style_axes(fig, y_title="Swayam", row=3, col=1)
    render_chart_panel(fig, "convergence_overlay", units="consensus · Mūla σ · Swayam signal", window=True)


def _render_composite_chart(cdf: pd.DataFrame) -> None:
    """The learner's weighted composite with its drift-diffusion filter, and agreement."""
    d = list(cdf.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.07)
    if {"ddm_upper", "ddm_lower"} <= set(cdf.columns):
        fig.add_trace(go.Scatter(x=d, y=cdf["ddm_upper"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d, y=cdf["ddm_lower"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=chart_rgba("slate", 0.07),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
    sc = pd.to_numeric(cdf["convergence_score"], errors="coerce")
    fig.add_trace(go.Scatter(x=d, y=sc, mode="lines", name="Weighted composite",
                             line=dict(color=chart_rgba("accent", 0.45), width=1),
                             hovertemplate="%{y:+.1f}<extra>Composite</extra>"), row=1, col=1)
    if "ddm_smoothed" in cdf.columns:
        fig.add_trace(go.Scatter(x=d, y=cdf["ddm_smoothed"], mode="lines", name="Drift-diffusion filtered",
                                 line=dict(color=chart_color("accent"), width=1.6),
                                 customdata=cdf.get("ddm_label"),
                                 hovertemplate="%{y:+.1f} · %{customdata}<extra>Filtered</extra>"),
                      row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.08), line_width=0.5, row=1, col=1)
    ag = pd.to_numeric(cdf["agreement_ratio"], errors="coerce")
    fig.add_trace(go.Scatter(x=d, y=ag, mode="lines", name="Agreement",
                             line=dict(color=chart_color("slate"), width=1.2),
                             hovertemplate="%{y:.0%}<extra>Agreement</extra>"), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM + 100))
    style_axes(fig, y_title="Composite (±100)", row=1, col=1)
    style_axes(fig, y_title="Agreement", y_range=[0, 1], row=2, col=1)
    render_chart_panel(fig, "convergence_composite", units="composite · agreement")


def _render_divergences(window: int) -> None:
    """The divergence events the hero counts, as a table — the last few sessions' worth."""
    ev = st.session_state.get("divergence_events")
    if ev is None or not len(ev):
        render_note("No divergence events: wherever both engines read, they did not point "
                    "opposite ways.")
        return
    e = ev.copy()
    e.index = pd.to_datetime(e.index, errors="coerce")
    e = e[~e.index.isna()].sort_index(ascending=False)
    cdf = st.session_state.get("convergence_df")
    sessions = (pd.DatetimeIndex(pd.to_datetime(cdf.index, errors="coerce")).dropna().sort_values()
                if cdf is not None and not cdf.empty else pd.DatetimeIndex([]))
    start = sessions[max(0, len(sessions) - window)] if len(sessions) else None
    recent = e[e.index >= start] if start is not None else e.iloc[:0]
    shown = (recent if len(recent) else e).head(15)
    tbl = pd.DataFrame({
        "Type": shown["divergence_type"].str.replace("_", " ").str.title(),
        "Mūla": shown["mula_stance"], "Swayam": shown["swayam_stance"],
        "Severity": shown["severity"].round(2),
        "Persistent": np.where(shown["persistent"].astype(bool), "yes", ""),
        "What happened": shown["description"],
    }, index=shown.index)
    render_table_panel(
        tbl, "convergence-divergences", index_label="Date", windowed=False,
        units=(f"{len(recent)} in the last {window} sessions" if len(recent)
               else f"none in the last {window} sessions — most recent shown"),
        max_rows=15, max_height=420,
    )


def render_convergence_tab(ts_filtered=None, edge: dict | None = None, horizon: int = 10):
    """Convergence — agreement between the engines, measured and graded."""
    cdf = st.session_state.get("convergence_df")
    fvo_ts = st.session_state.get("fvo_ts")
    swayam_daily = st.session_state.get("swayam_daily")
    if cdf is None or cdf.empty:
        render_empty_state(
            "No convergence read for this target",
            "Convergence needs sessions where both engines published. Swayam reads the "
            "target's own OHLCV, and none was available this run.",
            eyebrow="Convergence",
        )
        return
    _icfg = get_instrument_config(st.session_state.get("active_target", ""))

    cs = consensus_series(fvo_ts, swayam_daily)
    cdf = cdf.copy()
    cdf.index = pd.to_datetime(cdf.index, errors="coerce")
    cdf = cdf[~cdf.index.isna()]
    first = _window_index(ts_filtered)

    # ── 1 · The answer ────────────────────────────────────────────────────
    last = cdf.iloc[-1]
    agree = float(last.get("agreement_ratio", np.nan))
    ag_hist = pd.to_numeric(cdf["agreement_ratio"], errors="coerce").to_numpy(dtype=float)
    ag_p90 = tier_now(ag_hist, _icfg.ui_agreement_strong, q=0.90)
    ag_p75 = tier_now(ag_hist, _icfg.ui_agreement_moderate, q=0.75)
    cons_v = float(cs["Consensus"].iloc[-1]) if not cs.empty else None
    cons_lbl = str(cs["Label"].iloc[-1]) if not cs.empty else "—"
    m_fvo = float(cs["RawA"].iloc[-1]) if not cs.empty else None
    s_avg = float(cs["RawN"].iloc[-1]) if not cs.empty else None
    comp = float(last.get("convergence_score", np.nan))
    comp_lbl = str(last.get("ddm_label", "")) or lean_label(comp, 15.9, 9.2)
    ic = (edge or {}).get("ic")
    _sw_tier = tier_now(pd.to_numeric(swayam_daily.get("Avg_Signal"), errors="coerce").to_numpy(dtype=float)
                        if swayam_daily is not None and "Avg_Signal" in swayam_daily.columns else np.array([]),
                        _icfg.ui_swayam_avg_threshold, q=0.75)

    summary = (f"Mūla reads the instrument {valuation_state(m_fvo).lower()} ({m_fvo:+.2f}σ) and Swayam's "
               f"views average {s_avg:+.2f}; the consensus leans {cons_lbl.lower()}, and the engines "
               f"agree {agree:.0%} across four dimensions." if cons_v is not None and np.isfinite(agree)
               else "The engines share too little history for a consensus yet.")
    render_section_header("Convergence — Do the Engines Agree?", summary, icon="target")
    render_kpi_strip([
        {"label": "Agreement", "value": f"{agree:.0%}" if np.isfinite(agree) else "—",
         "subtext": f"own p75 {ag_p75:.0%} · p90 {ag_p90:.0%}",
         "color_class": "success" if np.isfinite(agree) and agree >= ag_p90 else "neutral",
         "tooltip": TOOLTIPS["agreement"]},
        {"label": "Consensus", "value": f"{cons_v:+.2f}" if cons_v is not None else "—",
         "subtext": cons_lbl.title() if cons_v is not None else "not enough overlap",
         "color_class": _lean_tone(cons_lbl), "tooltip": TOOLTIPS["consensus"]},
        {"label": "Mūla Reads", "value": f"{m_fvo:+.2f}σ" if m_fvo is not None else "—",
         "subtext": valuation_state(m_fvo).title() if m_fvo is not None else "—",
         "color_class": ("success" if m_fvo is not None and m_fvo <= -CLAIM_SIGMA else
                         "danger" if m_fvo is not None and m_fvo >= CLAIM_SIGMA else "neutral")},
        {"label": "Swayam Reads", "value": f"{s_avg:+.2f}" if s_avg is not None else "—",
         "subtext": f"average signal · tier ±{_sw_tier:.2f}",
         "color_class": ("success" if s_avg is not None and s_avg < -_sw_tier else
                         "danger" if s_avg is not None and s_avg > _sw_tier else "neutral")},
        {"label": "Weighted Composite", "value": f"{comp:+.1f}" if np.isfinite(comp) else "—",
         "subtext": comp_lbl.title(), "color_class": _lean_tone(comp_lbl),
         "tooltip": TOOLTIPS["composite"]},
        {"label": "Walk-Forward IC", "value": f"{ic:+.3f}" if ic is not None else "—",
         "subtext": f"composite, non-overlapping {horizon}d" if ic is not None else "not yet graded",
         "color_class": ("success" if ic is not None and ic >= 0.05 else
                         "warning" if ic is not None and ic > 0 else
                         "danger" if ic is not None else "neutral")},
    ], max_cols=6, key="conv-read")

    # Breadth carried forward: the Swayam tail can end before the model spine.
    _nn_last = st.session_state.get("swayam_native_last")
    if _nn_last is not None and len(cdf) and pd.Timestamp(_nn_last).normalize() < cdf.index[-1].normalize():
        render_info_box(
            "Swayam carried forward",
            f"The instrument's own OHLCV ends {pd.Timestamp(_nn_last):%d %b %Y}; later sessions "
            f"(through {cdf.index[-1]:%d %b %Y}) carry Swayam's last reading forward, so its side of "
            f"the comparison is provisional on those dates.",
            color="amber",
        )

    # ── 2 · The consensus and its inputs ─────────────────────────────────
    render_section_header(
        "Consensus — The Two Readings, Averaged",
        "Top: the consensus, each marker classified at the tier knowable on its own date. "
        f"Below: Mūla's oscillator (shaded ±{CLAIM_SIGMA:g}σ = no claim) and Swayam's average "
        "signal. Negative = stretched down.",
        icon="layers", accent="cyan",
    )
    if cs.empty or len(cs) < 10:
        render_empty_state(
            "Building consensus history",
            "Each reading is scaled by its own history, which needs a few weeks of overlap "
            "before the scale is trusted.",
            eyebrow="Convergence",
        )
    else:
        view = cs[cs.index >= first] if first is not None else cs
        _render_consensus_chart(view if len(view) else cs, _sw_tier)

    # ── 3 · What the learner built ───────────────────────────────────────
    render_section_header(
        "Weighted Composite — What the Learner Built",
        "Agreement strength weighted by each dimension's learned signed skill, pointed the way "
        "the engines lean, and its drift-diffusion filter. Below: agreement over time.",
        icon="cpu", accent="accent",
    )
    cview = cdf[cdf.index >= first] if first is not None else cdf
    _render_composite_chart(cview if len(cview) else cdf)

    # ── 4 · Has agreement been worth anything? ───────────────────────────
    render_section_header(
        "Walk-Forward Edge",
        f"When the composite leaned one way, did the instrument move that way over the next "
        f"{horizon} sessions? Graded exactly as Mūla's call and Swayam's breadth are.",
        icon="trending", accent="accent",
    )
    render_edge_panel(edge, "composite", what="the weighted composite")

    # ── 5 · Where they disagreed ──────────────────────────────────────────
    render_section_header(
        "Divergences",
        "Sessions where the engines pointed opposite ways, or one read strongly while the other "
        "did not. The Overview counts the recent ones as a standing risk.",
        icon="alert-triangle", accent="amber",
    )
    _render_divergences(DIV_LOOKBACK)
