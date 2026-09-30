"""
Charts. One colour per role, everywhere: Mūla blue, Swayam orange, Tattva aqua (the reference
palette's first three categorical slots, validated all-pairs in light and dark), price a
neutral grey, and BUY/SELL the fixed status green/red, always with ▲/▼ and a label. One
y-axis per chart; Streamlit's plotly theme supplies the surface for light and dark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MULA, SWAYAM, TATTVA = "#2a78d6", "#eb6834", "#1baf7a"
PRICE, MUTED = "#8a8984", "#b5b4ae"
BUY, SELL = "#0ca30c", "#d03b3b"

_LAYOUT = dict(margin=dict(l=8, r=8, t=36, b=8), height=340, hovermode="x unified",
               legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
               xaxis=dict(showgrid=False), yaxis=dict(zeroline=False))


def _fig(title: str, height: int | None = None) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=dict(text=title, x=0, font=dict(size=14)), **_LAYOUT)
    if height:
        f.update_layout(height=height)
    return f


def _window(s: pd.Series | pd.DataFrame, years: float) -> pd.Series | pd.DataFrame:
    if len(s) == 0:
        return s
    return s[s.index >= s.index[-1] - pd.DateOffset(days=int(365.25 * years))]


def call_starts(call: pd.Series) -> pd.Series:
    """The first day of each run of identical non-zero calls."""
    c = call.fillna(0).astype(int)
    return c[(c != 0) & (c != c.shift())]


def price_with_calls(run, years: float = 2.0) -> go.Figure:
    close = _window(run.bars["close"], years)
    starts = call_starts(run.out["call"]).reindex(close.index).dropna()
    f = _fig("Price and Tattva's calls")
    f.add_scatter(x=close.index, y=close, name="Close", line=dict(color=PRICE, width=2),
                  hovertemplate="%{y:,.4g}<extra>Close</extra>")
    for side, sym, col, lab in ((1, "triangle-up", BUY, "▲ BUY"), (-1, "triangle-down", SELL, "▼ SELL")):
        d = starts[starts == side].index
        if len(d):
            f.add_scatter(x=d, y=close.loc[d], mode="markers", name=lab,
                          marker=dict(symbol=sym, size=11, color=col, line=dict(width=2, color="white")),
                          hovertemplate="%{x|%d %b %Y}<extra>" + lab + "</extra>")
    return f


def engines(run, years: float = 2.0) -> go.Figure:
    o = _window(run.out, years)
    f = _fig("The two engines (+ = expected to rise relative to its norm)")
    f.add_hline(y=0, line=dict(color=MUTED, width=1))
    f.add_scatter(x=o.index, y=o["M"], name="Mūla — world-relative", line=dict(color=MULA, width=2))
    f.add_scatter(x=o.index, y=o["S"], name="Swayam — self", line=dict(color=SWAYAM, width=2))
    return f


def conviction(run, call_z: float, years: float = 2.0) -> go.Figure:
    o = _window(run.out, years)
    f = _fig("Conviction z — timing, in units of its own typical size")
    f.add_hrect(y0=-call_z, y1=call_z, fillcolor=MUTED, opacity=0.15, line_width=0)
    f.add_scatter(x=o.index, y=o["z"], name="Tattva z", line=dict(color=TATTVA, width=2),
                  hovertemplate="%{y:+.2f}<extra>z</extra>")
    f.add_annotation(x=o.index[0] if len(o) else None, y=call_z, text="call threshold",
                     showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11))
    return f


def fair_value(run, years: float = 2.0) -> go.Figure:
    close = _window(run.bars["close"], years)
    fv = run.mula["fair_value"].reindex(close.index)
    f = _fig("Price against the price its world implied (last 60 days)")
    f.add_scatter(x=close.index, y=close, name="Close", line=dict(color=PRICE, width=2))
    f.add_scatter(x=fv.index, y=fv, name="World-implied", line=dict(color=MULA, width=2, dash="dot"))
    return f


def contributions(run, top: int = 10) -> go.Figure:
    c = run.mula["contribution"].dropna(how="all")
    if c.empty:
        return _fig("World push — no exposures estimated yet")
    last = c.iloc[-1]
    last = last[last.abs() > 0].reindex(last.abs().sort_values(ascending=False).index).head(top)[::-1]
    f = _fig("World push on the next 10 days, by factor (exposure × factor's recent trend)",
             height=max(240, 26 * len(last) + 80))
    f.add_bar(x=100 * last.to_numpy(), y=list(last.index), orientation="h",
              marker=dict(color=[MULA if v >= 0 else SWAYAM for v in last], cornerradius=4),
              hovertemplate="%{y}: %{x:+.2f}%<extra></extra>", showlegend=False)
    f.update_layout(hovermode="closest", xaxis=dict(title="% over 10 days", zeroline=True))
    return f


def swayam_parts(parts: dict[str, float]) -> go.Figure:
    names = {"tsm60": "60-day trend", "tsm250": "250-day trend", "range60": "place in 60-day range"}
    keys = list(parts)
    f = _fig("Swayam's parts now (+ = stretched down → leans up)", height=220)
    f.add_bar(x=[parts[k] for k in keys], y=[names.get(k, k) for k in keys], orientation="h",
              marker=dict(color=SWAYAM, cornerradius=4), showlegend=False,
              hovertemplate="%{y}: %{x:+.2f}<extra></extra>")
    f.update_layout(hovermode="closest", xaxis=dict(zeroline=True))
    return f


def range_band(run, years: float = 1.0) -> go.Figure:
    b = _window(run.bars, years)
    hi = b["high"].rolling(60, min_periods=48).max()
    lo = b["low"].rolling(60, min_periods=48).min()
    f = _fig("Price inside its 60-day range")
    f.add_scatter(x=b.index, y=hi, line=dict(color=MUTED, width=1), name="60-day high")
    f.add_scatter(x=b.index, y=lo, line=dict(color=MUTED, width=1), name="60-day low",
                  fill="tonexty", fillcolor="rgba(181,180,174,0.18)")
    f.add_scatter(x=b.index, y=b["close"], line=dict(color=PRICE, width=2), name="Close")
    return f


def ic_by_period(summary: pd.DataFrame) -> go.Figure:
    f = _fig("Timing skill by period — mean IC across targets (non-overlapping 10-day outcomes)")
    for col, name, colr in (("M", "Mūla", MULA), ("S", "Swayam", SWAYAM), ("F", "Tattva", TATTVA)):
        f.add_bar(x=list(summary.index), y=summary[col], name=name,
                  marker=dict(color=colr, cornerradius=4),
                  hovertemplate="%{x}: %{y:+.3f}<extra>" + name + "</extra>")
    f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
    f.add_hline(y=0, line=dict(color=MUTED, width=1))
    return f


def hit_rates(summary: pd.DataFrame) -> go.Figure:
    f = _fig("Hit rate of BUY/SELL calls, against doing without Tattva's timing")
    for col, name, colr in (("tattva_hit", "Tattva calls", TATTVA), ("drift_hit", "Drift only", PRICE),
                            ("long_hit", "Always long", MUTED)):
        f.add_bar(x=list(summary.index), y=100 * summary[col], name=name,
                  marker=dict(color=colr, cornerradius=4),
                  hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>")
    f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest",
                    yaxis=dict(range=[40, max(65, float(np.nanmax(100 * summary[['tattva_hit', 'drift_hit', 'long_hit']].to_numpy())) + 3)],
                               title="%"))
    f.add_hline(y=50, line=dict(color=MUTED, width=1, dash="dot"))
    return f
