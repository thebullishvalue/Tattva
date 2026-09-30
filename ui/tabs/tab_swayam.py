"""
Tattva — Swayam tab: the target read by itself through the Pragati indicator.
तत्त्व (Tattva) — "Principle / Essence"

UI — Swayam (engines/swayam): the conviction ladder (Daily · Weekly) × Samanvaya's value on
the target's own OHLCV, the 3 × 3 Conviction-Value Grid they meet in, and the ▲ capitulation
turn / ▼ distribution events read from it.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      what the engine measured, ladder coverage
  2 ANCHOR    what is the underlying claim?      today's state · conviction · value · push
  3 SIGNAL    what does it say to do?            price with every ▲ / ▼
  4 STATE     how does that sit historically?    the two tapes against their zones · the grid
  5 DETAIL    the evidence behind it             the trace's push · rung breadth · event ledger
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_SMALL, get_instrument_config
from engines.pragati import grid as cg
from engines.pragati import samanvaya as sv
from engines.swayam import CELL_LEAN, RUNG_NAMES
from ui.components import (render_empty_state, render_chart_panel, render_metric_card, render_note,
                           render_section_header, render_table_panel)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes

Z1, Z2 = 30.0, 60.0                 # the conviction tape's inner / outer zone
THETA = float(sv.THETA_OSC)         # value's θ on the ±100 scale (≈ 42.9)

TOOLTIPS = {
    "state": ("The 3 × 3 Conviction-Value Grid cell the target stands in today — rows: who "
              "controls (sellers / faint / buyers, the conviction tape past ±30, moved only by a "
              "confirmed push of conviction's own histogram); columns: where price stands (cheap / "
              "fair / rich, the value tape past ±θ). The units are Pragyam's measured weights."),
    "conviction": ("The conviction tape: the participation-weighted share of each bar's effort "
                   "that became displacement, averaged over the ladder's rungs (Daily, Weekly), "
                   "±100. Past −30 sellers control; past +30 buyers. Tattva reads sellers in "
                   "control as OVERSOLD — the side measured to lead at its 10-day horizon."),
    "value": ("Samanvaya's value tape: price against what ~20 macro drivers explain (RV leg) "
              "blended with price-only breadth, on ±100. Past −θ cheap, past +θ rich."),
    "push": ("The trace's histogram, as the grid reads it: IMPULSE past k·σ, pressing, "
             "decelerating, turning. A row changes only on a confirmed push; QUIET means "
             "conviction's own σ is in the bottom fifth of its history, where a small imbalance "
             "scales into a large reading — the ▲ does not fire there."),
    "last_up": ("The last ▲ CAPITULATION TURN: the first bar the grid stood in Buy · capitulation "
                "(sellers in control of a price cheap past θ) with value momentum already turning "
                "back toward fair, outside the quiet regime. Measured on Tattva's own targets: "
                "+0.19σ / +0.25σ excess over the next 10 days in 2022-23 / 2024-26 (commodities, "
                "FX and indices; ≈ 0 on crypto)."),
    "rungs": ("Swayam's views: each rung of the conviction ladder on ±100. Oversold / "
              "overbought breadth — what Convergence reads — is the share of rungs past ∓30."),
}

_LEAN_TONE = {1.0: "emerald", 0.5: "emerald", -1.0: "rose", 0.0: "slate"}


def _window(df: pd.DataFrame, tf: str | None) -> pd.DataFrame:
    if not tf or tf == "ALL" or df.empty:
        return df
    off = {"3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
           "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2)}.get(tf, pd.DateOffset(years=1))
    idx = pd.to_datetime(df.index)
    return df[idx >= idx.max() - off]


def _cards(cur: dict, f: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    lean = float(cur.get("lean", 0.0))
    with c1:
        render_metric_card("STATE", cur["state"], f"{cur['units']:.2f} units · {cur['family'].lower()}",
                           "success" if lean > 0 else "danger" if lean < 0 else "neutral",
                           tooltip=TOOLTIPS["state"])
    with c2:
        cv = cur["conviction"]
        render_metric_card("CONVICTION", f"{cv:+.0f}",
                           "sellers in control" if cv <= -Z1 else "buyers in control" if cv >= Z1 else "not decided",
                           "success" if cv <= -Z1 else "danger" if cv >= Z1 else "neutral",
                           tooltip=TOOLTIPS["conviction"])
    with c3:
        v = cur["value"]
        txt = "—" if v != v else f"{v:+.0f}"
        sub = "no value reading" if v != v else "cheap past θ" if v <= -THETA else "rich past θ" if v >= THETA else "fair"
        render_metric_card("VALUE", txt, sub,
                           "success" if v == v and v <= -THETA else "danger" if v == v and v >= THETA else "neutral",
                           tooltip=TOOLTIPS["value"])
    with c4:
        tier = cur["push_tier"] or "—"
        render_metric_card("PUSH", tier.split(" · ")[0].upper(), "quiet regime" if cur["quiet"] else "live",
                           "warning" if cur["quiet"] else "info", tooltip=TOOLTIPS["push"])
    with c5:
        lu = cur["last_capitulation"]
        if lu is not None:
            ago = int((f.index > lu).sum())
            render_metric_card("LAST ▲", str(pd.Timestamp(lu).date()), f"{ago} bars ago", "success" if ago <= 10 else "neutral",
                               tooltip=TOOLTIPS["last_up"])
        else:
            render_metric_card("LAST ▲", "—", "none in the window", "neutral", tooltip=TOOLTIPS["last_up"])
    with c6:
        r = cur["rungs"]
        render_metric_card("LADDER", " · ".join(f"{(r.get(k) or 0):+.0f}" for k in RUNG_NAMES),
                           " · ".join(RUNG_NAMES), "neutral", tooltip=TOOLTIPS["rungs"])


def _price_events(f: pd.DataFrame, target: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index, y=f["close"], mode="lines", name="Close",
                             line=dict(color=chart_rgba("slate", 0.9), width=1.4)))
    up = f[f["turn_buy"].fillna(False).astype(bool)]
    dn = f[f["turn_sell"].fillna(False).astype(bool)]
    if len(up):
        fig.add_trace(go.Scatter(x=up.index, y=up["close"], mode="markers", name="▲ capitulation turn",
                                 marker=dict(symbol="triangle-up", size=11, color=chart_color("emerald"),
                                             line=dict(width=1, color=grid_rgba(0.5)))))
    if len(dn):
        fig.add_trace(go.Scatter(x=dn.index, y=dn["close"], mode="markers", name="▼ distribution",
                                 marker=dict(symbol="triangle-down", size=11, color=chart_color("rose"),
                                             line=dict(width=1, color=grid_rgba(0.5)))))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title="Price")
    render_chart_panel(fig, "swayam_price_events", units="price", context=target)


def _tapes(f: pd.DataFrame) -> None:
    fig = go.Figure()
    ct = f["c_tape"].where(f["c_ready"].fillna(False).astype(bool))
    vt = f["v_tape"].where(f["v_ready"].fillna(False).astype(bool))
    fig.add_trace(go.Scatter(x=f.index, y=ct, name="Conviction tape", line=dict(color=chart_color("cyan"), width=1.6)))
    fig.add_trace(go.Scatter(x=f.index, y=vt, name="Value tape", line=dict(color=chart_color("amber"), width=1.3)))
    for y, c in ((Z1, "rose"), (-Z1, "emerald")):
        fig.add_hline(y=y, line_color=chart_rgba(c, 0.35), line_width=0.8, line_dash="dot")
    for y, c in ((THETA, "rose"), (-THETA, "emerald")):
        fig.add_hline(y=y, line_color=chart_rgba(c, 0.25), line_width=0.8, line_dash="dash")
    fig.add_hline(y=0, line_color=grid_rgba(0.08), line_width=0.5)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title="±100", y_range=[-100, 100])
    render_chart_panel(fig, "swayam_tapes", units="±100 · zones ±30 (dotted) · θ ±43 (dashed)")


def _grid_timeline(f: pd.DataFrame) -> None:
    cell = f["cvg_cell"].fillna(cg.UNREAD).astype(int).clip(0, cg.UNREAD)
    lean = np.asarray(CELL_LEAN)[cell.to_numpy()]
    names = np.asarray(cg.NAMES)[cell.to_numpy()]
    colors = [chart_color(_LEAN_TONE.get(float(l), "slate")) if c != cg.UNREAD else grid_rgba(0.15)
              for l, c in zip(lean, cell)]
    units = pd.to_numeric(f["cvg_units"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=f.index, y=units, marker=dict(color=colors, line=dict(width=0)),
                         customdata=names, name="Units",
                         hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<br>%{y:.2f} units<extra></extra>"))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
    fig.update_layout(bargap=0)
    style_axes(fig, y_title="Units")
    render_chart_panel(fig, "swayam_grid_timeline",
                       units="grid units · green = capitulation / washout lean · red = distribution")


def _push(f: pd.DataFrame) -> None:
    h = pd.to_numeric(f["hist"], errors="coerce")
    thr = pd.to_numeric(f["thr"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=f.index, y=h, name="Histogram",
                         marker=dict(color=[chart_rgba("emerald" if v >= 0 else "rose", 0.7) for v in h.fillna(0)],
                                     line=dict(width=0))))
    fig.add_trace(go.Scatter(x=f.index, y=thr, name="+k·σ", line=dict(color=grid_rgba(0.35), width=0.8, dash="dot")))
    fig.add_trace(go.Scatter(x=f.index, y=-thr, name="−k·σ", line=dict(color=grid_rgba(0.35), width=0.8, dash="dot")))
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL))
    fig.update_layout(bargap=0)
    style_axes(fig, y_title="Push")
    render_chart_panel(fig, "swayam_push", units="the trace's histogram · impulse band ±k·σ")


def _breadth(d: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Oversold_Pct"], name="Oversold (sellers past −30)",
                             line=dict(color=chart_color("emerald"), width=1.4), fill="tozeroy",
                             fillcolor=chart_rgba("emerald", 0.06)))
    fig.add_trace(go.Scatter(x=d.index, y=-d["Overbought_Pct"], name="Overbought (buyers past +30)",
                             line=dict(color=chart_color("rose"), width=1.4), fill="tozeroy",
                             fillcolor=chart_rgba("rose", 0.06)))
    fig.add_hline(y=0, line_color=grid_rgba(0.08), line_width=0.5)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL))
    style_axes(fig, y_title="% of rungs", y_range=[-105, 105])
    render_chart_panel(fig, "swayam_breadth", units="share of the ladder's rungs · what Convergence reads")


def _ledger(f: pd.DataFrame, horizon: int) -> None:
    ev = f[f["turn_buy"].fillna(False).astype(bool) | f["turn_sell"].fillna(False).astype(bool)]
    if ev.empty:
        render_note("No ▲ or ▼ in the loaded history.")
        return
    close = f["close"]
    rows = []
    for t, r in ev.iterrows():
        i = f.index.get_loc(t)
        j = i + horizon
        fwd = (float(close.iloc[j] / close.iloc[i] - 1.0) * 100) if j < len(f) else np.nan
        up = bool(r["turn_buy"])
        rows.append({"Date": str(pd.Timestamp(t).date()), "Event": "▲ capitulation" if up else "▼ distribution",
                     "Close": f"{r['close']:,.2f}",
                     "Conviction": f"{r['c_tape']:+.0f}", "Value": f"{r['v_tape']:+.0f}",
                     f"{horizon}d move": "open" if fwd != fwd else f"{fwd:+.2f}%",
                     "Worked": "—" if fwd != fwd else ("✓" if (fwd > 0) == up else "✗")})
    tbl = pd.DataFrame(rows[::-1])
    render_table_panel(tbl, "swayam_event_ledger", units=f"close-to-close move over {horizon} bars",
                       label_col="Date", max_height=320)


def render_swayam_tab(selected_tf: str | None = None) -> None:
    """Swayam tab — the Pragati read of the target."""
    frame = st.session_state.get("swayam_frame")
    daily = st.session_state.get("swayam_daily")
    cur = st.session_state.get("swayam_current") or {}
    if frame is None or frame.empty or daily is None or daily.empty or not cur:
        render_empty_state(
            "No Swayam read available",
            "Swayam needs ~2 years of the target's own daily bars to calibrate its conviction ladder.",
            eyebrow="Swayam",
            action_label="Run analysis in the sidebar, then return to this page.",
        )
        return
    target = st.session_state.get("active_target", "")
    try:
        horizon = int(get_instrument_config(target).forecast_horizon)
    except KeyError:
        horizon = 10
    f = frame[frame.index <= pd.Timestamp(cur["date"])]
    fw = _window(f, selected_tf)
    dw = _window(daily, selected_tf)

    # 1 · TRUST
    n_drv = int(cur.get("drivers", 0))
    render_note(
        f"Swayam reads {target} by itself: the conviction ladder ({' · '.join(RUNG_NAMES)}, built from "
        f"daily bars so no past value is ever rewritten) × Samanvaya's value "
        f"({'hedged against ' + str(n_drv) + ' macro drivers' if n_drv else 'UNHEDGED — no value drivers arrived'}). "
        f"Measured on Tattva's 25 targets at the 10-day horizon, it replaced the MSF/MMR view bank, whose "
        f"oscillator anti-predicted in two of three periods. Crypto is the stated exception.")

    # 2 · ANCHOR
    _cards(cur, f)

    # 3 · SIGNAL
    render_section_header("Price and Events",
                          "Every ▲ capitulation turn and ▼ distribution in the window, on the target's close.",
                          icon="zap", accent="emerald")
    _price_events(fw, target)

    # 4 · STATE
    render_section_header("Conviction and Value Tapes",
                          "Who controls (conviction) and where price stands (value), each on ±100 against its "
                          "own zones. The grid's rows follow conviction past ±30; its columns follow value past ±θ.",
                          icon="activity", accent="cyan")
    _tapes(fw)
    render_section_header("Grid State",
                          "The Conviction-Value Grid cell through time, drawn as its units. Green: the cells "
                          "measured to lead (capitulation, washout); red: distribution; grey: no measured edge.",
                          icon="grid", accent="violet")
    _grid_timeline(fw)

    # 5 · DETAIL
    render_section_header("The Push",
                          "The trace's histogram — the push that moves the grid's rows. A column past the "
                          "dotted band is an impulse; a row changes only on a confirmed push.",
                          icon="bar-chart")
    _push(fw)
    render_section_header("Rung Breadth",
                          "The share of the ladder's rungs with sellers (up) or buyers (down) past the inner "
                          "zone — the oversold / overbought breadth Convergence reads from Swayam.",
                          icon="layers", accent="emerald")
    _breadth(dw)
    render_section_header("Event Ledger",
                          f"Every ▲ / ▼ in the loaded history and the {horizon}-bar move that followed. "
                          "A ledger, not a backtest: no costs, overlapping windows.",
                          icon="database", accent="rose")
    _ledger(f, horizon)
