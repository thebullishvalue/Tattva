"""
Tattva — Mūla tab: where this instrument should trade, and how far to believe it.
तत्त्व (Tattva) — "Principle / Essence"

UI — Mūla engine visualization: the fair-value LEVEL the macro cross-section
implies, the oscillator the call is made on, whether PRICE (not fair value)
closes the gap, how well the model locates the level at all, and what the
call has been worth.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    what does Mūla say?                The read (state · fair value · move)
  2 ANCHOR    what is the claim made on?         Price & fair value · the oscillator
  3 SIGNAL    does acting on it make sense?      Error correction: does price revert?
  4 TRUST     can the level be believed?         Model quality · walk-forward edge
  5 DETAIL    the evidence behind it             Drivers · window agreement

The answer leads. Every page in this app now opens with the sentence it
exists to say and the number behind it, then shows why; the Mūla page used to
open with five model-quality cards and never plotted the oscillator its own
call was made on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics.adaptive import tier_now
from core.semantics import CLAIM_SIGMA, DEEP_SIGMA, valuation_state
from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (
    render_kpi_strip,
    render_metric_card,
    render_interpretation_card,
    render_section_header,
    render_sub_header,
    render_chart_panel,
    render_table_panel,
    render_empty_state,
    render_edge_panel,
    render_note,
    provisional_note,
)
from core.config import (
    get_instrument_config, InstrumentConfig,
    UI_R2_STRONG,
    UI_R2_ACCEPTABLE,
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_XLARGE,
)

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "valuation": (
        "The Mūla oscillator: how far price sits from the fair-value level the macro "
        "cross-section implies, in units of the gap's own typical size. Inside ±0.5σ is the "
        "normal day-to-day noise of price around fair value and no claim is made; ±2σ is a "
        "deep mispricing. This is the one directional claim Tattva makes."
    ),
    "fair_value": (
        "The price level the whole traded cross-section implies today, from a recursive "
        "cointegrating regression of log price on the integrated factors of ~200 instruments. "
        "The band is where price itself would be expected 95% of the time if it traded at "
        "fair value plus normal noise."
    ),
    "reversion": (
        "Does PRICE move to close the gap? The error-correction layer regresses each day's "
        "price change on yesterday's gap. κ̂ > 0 means price moves toward fair value; this is "
        "the probability that it does, from the coefficient's own uncertainty. It matters "
        "because a gap can also close by fair value moving to price — which is stationary, "
        "and is not a trade."
    ),
    "drift": (
        "The price move the error-correction regression expects over the horizon, given "
        "today's gap and how persistent gaps are, with its one-SD range. A gap that is real "
        "but small relative to normal price noise shows up here as a small expected move."
    ),
    "oos_r2": (
        "Share of log-price variance explained by the fitted level. A level regression on "
        "integrated factors reads high by construction (both sides trend), so treat this as "
        "a sanity check that the cross-section tracks the target at all — not as evidence of "
        "edge. R² vs Trailing Mean and the walk-forward IC are the discriminating numbers."
    ),
    "r2_vs_anchor": (
        "The published fair value versus the honest competing hypothesis: that price simply "
        "reverts to its own 252-day trailing mean. Positive = the global cross-section locates "
        "the level better than the asset's own history does, which is the whole claim of a "
        "cross-sectional valuation. Note this is deliberately NOT the random-walk test: one "
        "step ahead, yesterday's close beats any valuation of a near-integrated price, so "
        "that comparison measures a claim this engine never makes."
    ),
    "xs_consistency": (
        "Leave-one-block-out agreement: the valuation is refitted with each asset-class block "
        "(rates, credit, energy, …) removed in turn, and this is how far those independent "
        "slices of the world agree on the SIGN of today's mispricing. Low = the call depends "
        "on which part of the world you ask."
    ),
    "stationarity": (
        "Augmented Dickey-Fuller test on the gap itself, with its online half-life. A "
        "stationary gap is necessary for a valuation to mean anything — but not sufficient for "
        "a trade, because the gap can close by fair value moving rather than price. The "
        "reversion read above is the price-side test."
    ),
    "model_spread": (
        "How precisely the engine locates the fair-value LEVEL today: the standard deviation "
        "of the fair value itself (coefficient and view uncertainty), in basis points — not "
        "the price noise around it. Wide = the cross-section does not pin the level, so treat "
        "the mispricing with caution."
    ),
    "window_agreement": (
        "The gap measured against its own trailing distribution over three windows (~1 "
        "month, ~1 quarter, ~1 year): how many of them read today's gap as unusually cheap "
        "versus unusually rich, combined and smoothed through a drift-diffusion filter. It "
        "says how persistent and broad the mispricing is, not which way to trade — that is "
        "the oscillator's job."
    ),
}


def _series_tier(col: str, prior: float, q: float = 0.90) -> float:
    """Display cut-point for ``col``, from that series' own causal past.

    Reads the full (unfiltered) engine frame out of session state so the tier
    reflects the whole history rather than whatever window the user is looking
    at — a p90 recomputed over the last 3 months would move every time the
    window changed, which is not a threshold. Falls back to the config prior
    whenever the series is missing or too short.
    """
    ts = st.session_state.get("fvo_ts")
    if ts is None or col not in getattr(ts, "columns", ()):
        return float(prior)
    v = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=float)
    if "Valid" in ts.columns:
        v = v[ts["Valid"].astype(bool).to_numpy()]
    return tier_now(v, prior, q=q)


def _active_tiers() -> InstrumentConfig:
    """This render's instrument config (per-instrument display priors)."""
    try:
        return get_instrument_config(st.session_state.get("active_target", ""))
    except KeyError:
        return InstrumentConfig()


def _num(d: dict, key: str):
    """A finite float from the signal dict, else None — never a silent 0."""
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _state_tone(state: str) -> str:
    return "success" if "CHEAP" in state else "danger" if "RICH" in state else "neutral"


# ═══════════════════════════════════════════════════════════════════════
#  1 · THE READ
# ═══════════════════════════════════════════════════════════════════════

def _render_read(signal: dict, ts: pd.DataFrame, edge: dict | None, horizon: int) -> None:
    fvo = _num(signal, "fvo")
    pct = (_num(signal, "pct_mispricing") or 0.0) * 100.0
    state = valuation_state(fvo)
    fv = _num(signal, "fair_value")
    lo = hi = None
    if {"FairValueLo", "FairValueHi"} <= set(ts.columns):
        _b = ts[["FairValueLo", "FairValueHi"]].dropna()
        if len(_b):
            lo, hi = float(_b["FairValueLo"].iloc[-1]), float(_b["FairValueHi"].iloc[-1])
    rev = _num(signal, "mr_prob")
    hl = _num(signal, "mula_half_life")
    drift = _num(signal, "expected_drift_pct")
    dsd = _num(signal, "drift_sd_pct")
    ic = (edge or {}).get("ic")

    render_kpi_strip([
        {"label": "Valuation", "value": state if fvo is not None else "—",
         "subtext": (f"{fvo:+.2f}σ · {abs(pct):.1f}% {'below' if pct < 0 else 'above'} fair value"
                     if fvo is not None else "no published valuation"),
         "color_class": _state_tone(state), "tooltip": TOOLTIPS["valuation"]},
        {"label": "Fair Value", "value": f"{fv:,.2f}" if fv is not None else "—",
         "subtext": (f"95% band {lo:,.2f} – {hi:,.2f}" if lo is not None else "level the world implies"),
         "color_class": "violet", "tooltip": TOOLTIPS["fair_value"]},
        {"label": "Price Reverts", "value": f"{rev:.0%}" if rev is not None else "—",
         "subtext": (f"price-side half-life {hl:.0f}d" if hl is not None and hl < 2000
                     else "price does not close the gap" if rev is not None and rev < 0.5
                     else "P(price moves toward fair value)"),
         "color_class": ("success" if rev is not None and rev >= 0.8 else
                         "warning" if rev is not None and rev >= 0.5 else "danger"),
         "tooltip": TOOLTIPS["reversion"]},
        {"label": f"Expected {horizon}D Move", "value": f"{drift:+.2f}%" if drift is not None else "—",
         "subtext": f"± {dsd:.2f}% (1 SD) from error correction" if dsd is not None else "error-correction read",
         "color_class": ("success" if drift is not None and drift > 0 else
                         "danger" if drift is not None and drift < 0 else "neutral"),
         "tooltip": TOOLTIPS["drift"]},
        {"label": "Walk-Forward IC", "value": f"{ic:+.3f}" if ic is not None else "—",
         "subtext": f"Mūla's call, non-overlapping {horizon}d" if ic is not None else "not yet graded",
         "color_class": ("success" if ic is not None and ic >= 0.05 else
                         "warning" if ic is not None and ic > 0 else
                         "danger" if ic is not None else "neutral")},
    ], max_cols=5, key="mula-read")


# ═══════════════════════════════════════════════════════════════════════
#  2 · PRICE & FAIR VALUE
# ═══════════════════════════════════════════════════════════════════════

def _render_fair_value_chart(ts_filtered, x_axis, active_target):
    """Price against fair value (top) and the oscillator the call is made on (bottom).

    The bottom pane used to be the raw log gap with an OU projection of it
    closing. Two things were wrong with that. The call is made on the
    OSCILLATOR — the gap in units of its own typical size, against the ±0.5σ
    no-claim band — and the page never drew it. And the projection showed the
    gap closing as if price would do the closing, while the engine's own
    evidence says gaps mostly close by fair value moving to price; the
    error-correction section below is the honest version of that forecast.
    """
    has_price = "Price" in ts_filtered.columns
    has_band = {"FairValueLo", "FairValueHi"}.issubset(ts_filtered.columns)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        shared_xaxes=True, vertical_spacing=0.06)

    if has_band:
        fig.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["FairValueHi"], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["FairValueLo"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=chart_rgba("violet", 0.10), name="95% band",
            hoverinfo="skip",
        ), row=1, col=1)

    _pn = provisional_note(st.session_state.get("fvo_ts"))
    if _pn:
        render_note(_pn)

    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair value",
        line=dict(color=chart_color("violet"), width=1.4),
        hovertemplate="%{y:,.2f}<extra>Fair value</extra>",
    ), row=1, col=1)
    # Price is the instrument — the primary series — so it is drawn in full
    # ink. It used to be the faintest line on the chart (slate at 40%).
    top_series = ts_filtered["Price"] if has_price else ts_filtered["Actual"]
    fig.add_trace(go.Scatter(
        x=x_axis, y=top_series, mode="lines", name=f"{active_target}",
        line=dict(color=chart_color("accent"), width=1.6),
        hovertemplate="%{y:,.2f}<extra>Price</extra>",
    ), row=1, col=1)

    # ── Bottom: the oscillator, against the bands the call is made on ──
    osc = pd.to_numeric(ts_filtered["FVO"], errors="coerce")
    gap_pct = (pd.to_numeric(ts_filtered.get("PctMispricing"), errors="coerce") * 100.0
               if "PctMispricing" in ts_filtered.columns else osc * np.nan)
    fig.add_hrect(y0=-CLAIM_SIGMA, y1=CLAIM_SIGMA, fillcolor=grid_rgba(0.05), line_width=0,
                  row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=osc.clip(lower=0), fill="tozeroy", fillcolor=chart_rgba("rose", 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=osc.clip(upper=0), fill="tozeroy", fillcolor=chart_rgba("emerald", 0.12),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=osc, mode="lines", name="Oscillator",
        line=dict(color=chart_color("violet"), width=1.4),
        customdata=gap_pct,
        hovertemplate="%{y:+.2f}σ · %{customdata:+.2f}% vs fair value<extra>Mūla</extra>",
    ), row=2, col=1)
    for y, nm in ((DEEP_SIGMA, "rose"), (-DEEP_SIGMA, "emerald")):
        fig.add_hline(y=y, line_dash="dot", line_color=chart_rgba(nm, 0.35), line_width=0.6,
                      row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=2, col=1)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE))
    style_axes(fig, y_title=f"{active_target}", row=1, col=1)
    style_axes(fig, y_title="Oscillator (σ)", row=2, col=1)
    render_chart_panel(fig, "fvo_fairvalue", units="price · oscillator σ", window=True)


# ═══════════════════════════════════════════════════════════════════════
#  3 · ERROR CORRECTION
# ═══════════════════════════════════════════════════════════════════════

def _render_error_correction(ts_filtered, x_axis, signal, horizon):
    """Does PRICE close the gap — and which forecast family has been right?

    The valuation regression says where fair value is; it cannot say whether
    price goes there. That is what the error-correction layer measures, on
    differenced data where fast memories are legitimate: the response of each
    day's price change to yesterday's gap (κ̂), and a pooled contest between
    a valuation-led, a momentum-led and a full forecast of the price change,
    weighted by their own discounted predictive likelihood.
    """
    need = {"WValuation", "WMomentum", "WFull"}
    if not need <= set(ts_filtered.columns) or ts_filtered[list(need)].dropna().empty:
        render_empty_state(
            "No error-correction read yet",
            "The error-correction layer publishes once it has learned from enough sessions "
            "after the valuation's own burn-in.",
            eyebrow="Mūla · ECM",
        )
        return

    w_led = (_num(signal, "w_valuation") or 0.0) + (_num(signal, "w_full") or 0.0)
    kappa = _num(signal, "mula_kappa")
    gap_hl = _num(signal, "gap_half_life")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        render_metric_card(
            "Price-Side κ̂", f"{kappa:+.4f}/d" if kappa is not None else "—",
            "Response of each day's price change to yesterday's gap. Positive = price "
            "moves toward fair value.",
            "success" if kappa is not None and kappa > 0 else "danger" if kappa is not None else "neutral",
            tooltip=TOOLTIPS["reversion"])
    with c2:
        render_metric_card(
            "Valuation-Led Weight", f"{w_led:.0%}",
            "Predictive weight on the forecasts that let the gap move price, against a "
            "momentum-only forecast. Below ~40%: momentum has been the better forecaster.",
            "success" if w_led >= 0.55 else "warning" if w_led >= 0.4 else "danger")
    with c3:
        render_metric_card(
            "Gap Half-Life", f"{gap_hl:.0f}d" if gap_hl else "—",
            "How long the gap itself persists, however it closes — mostly by fair value "
            "moving on most instruments. Not the speed price converges at.",
            "info", tooltip=TOOLTIPS["stationarity"])

    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], shared_xaxes=True,
                        vertical_spacing=0.08)
    for col, name, colour in (("WValuation", "Valuation-led", "violet"),
                              ("WFull", "Full (valuation + momentum)", "cyan"),
                              ("WMomentum", "Momentum-led", "slate")):
        fig.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered[col], mode="lines", stackgroup="pool", name=name,
            line=dict(width=0.5, color=chart_color(colour)),
            fillcolor=chart_rgba(colour, 0.35),
            hovertemplate="%{y:.0%}<extra>" + name + "</extra>",
        ), row=1, col=1)
    if "MulaDriftPct" in ts_filtered.columns:
        d = pd.to_numeric(ts_filtered["MulaDriftPct"], errors="coerce")
        sd = (pd.to_numeric(ts_filtered["MulaSdPct"], errors="coerce")
              if "MulaSdPct" in ts_filtered.columns else d * 0)
        fig.add_trace(go.Scatter(x=x_axis, y=d + sd, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=d - sd, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=chart_rgba("slate", 0.10),
                                 showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=d, mode="lines", name=f"Expected {horizon}d move",
            line=dict(color=chart_color("accent"), width=1.4),
            hovertemplate="%{y:+.2f}%<extra>Expected move</extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM + 80))
    style_axes(fig, y_title="Pool weight", y_range=[0, 1], row=1, col=1)
    style_axes(fig, y_title=f"{horizon}d move %", row=2, col=1)
    render_chart_panel(fig, "mula_ecm", units="pool weight · expected move")


# ═══════════════════════════════════════════════════════════════════════
#  4 · MODEL QUALITY
# ═══════════════════════════════════════════════════════════════════════

def _render_model_quality_cards(model_stats, signal):
    """Model Quality — does the cross-section locate this instrument's level?

    Read left to right as a chain of increasingly demanding questions. Does the
    cross-section track this asset at all (OOS R²)? Does it locate the level
    better than the asset's own trailing mean does (R² vs Trailing Mean)? Do
    independent slices of the world agree on the mispricing's sign
    (Cross-Section)? Is the gap stationary (Stationarity)? And how precisely
    is the level pinned today (Model Spread)?

    A high R² with a negative R²-vs-anchor is the diagnostic worth knowing: the
    regression is fitting the trend, not the mispricing.
    """
    _t = _active_tiers()
    UI_MODEL_SPREAD_LOW = _series_tier("ModelSpread", _t.ui_model_spread_low / 1e4, 0.10) * 1e4
    UI_MODEL_SPREAD_HIGH = _series_tier("ModelSpread", _t.ui_model_spread_high / 1e4, 0.90) * 1e4
    q1, q2, q3, q4, q5 = st.columns(5, gap="small")

    with q1:
        r2 = float(model_stats.get("r2_oos", np.nan))
        render_metric_card(
            "OOS R²", f"{r2:.3f}",
            "Log-price variance explained by the fitted level. High is expected for a level "
            "regression — the next card is the discriminating one.",
            "success" if r2 > UI_R2_STRONG else "warning" if r2 > UI_R2_ACCEPTABLE else "danger",
            tooltip=TOOLTIPS["oos_r2"],
        )
    with q2:
        r2_anchor = float(model_stats.get("r2_vs_anchor", 0.0) or 0.0)
        render_metric_card(
            "R² vs Trailing Mean", f"{r2_anchor:+.3f}",
            "Edge over 'price reverts to its own 252d mean'. Negative = a moving average "
            "locates fair value better than the global cross-section does.",
            "success" if r2_anchor > 0.05 else "warning" if r2_anchor > -0.05 else "danger",
            tooltip=TOOLTIPS["r2_vs_anchor"],
        )
    with q3:
        xs = _num(signal, "xs_consistency")
        render_metric_card(
            "Cross-Section", f"{xs:.2f}" if xs is not None else "—",
            "How far independent asset-class slices agree on the sign of today's gap.",
            "success" if xs is not None and xs >= 0.75 else
            "warning" if xs is not None and xs >= 0.5 else "danger",
            tooltip=TOOLTIPS["xs_consistency"],
        )
    with q4:
        adf_p = _num(signal, "adf_pvalue")
        render_metric_card(
            "Gap Stationarity", f"p={adf_p:.3f}" if adf_p is not None else "—",
            ("The gap is stationary — a valuation is meaningful here." if adf_p is not None and adf_p < 0.05
             else "Unit root not rejected — the gap may be a permanent re-rating."
             if adf_p is not None and adf_p > 0.10 else "Borderline — treat the valuation as unproven."),
            "success" if adf_p is not None and adf_p < 0.05 else
            "warning" if adf_p is not None and adf_p <= 0.10 else "danger",
            tooltip=TOOLTIPS["stationarity"],
        )
    with q5:
        sp = (_num(signal, "model_spread") or 0.0) * 10000
        render_metric_card(
            "Model Spread", f"{sp:.0f} bps",
            f"SD of the fair-value level itself. Above {UI_MODEL_SPREAD_HIGH:.0f} bps "
            f"(p90 of history) = the level is poorly pinned.",
            "success" if sp < UI_MODEL_SPREAD_LOW else "warning" if sp < UI_MODEL_SPREAD_HIGH else "danger",
            tooltip=TOOLTIPS["model_spread"],
        )


# ═══════════════════════════════════════════════════════════════════════
#  5 · DRIVERS
# ═══════════════════════════════════════════════════════════════════════

def _render_drivers(engine) -> None:
    """Which asset-class blocks the fair value leans on (leave-one-block-out).

    An ablation, not a coefficient read-off: each block is removed, the
    valuation refitted, and the move in fair value recorded. Lives here
    rather than on the System page because it is evidence about THIS
    engine's claim — every page now carries its own evidence.
    """
    impacts = getattr(engine, "latest_feature_impacts", None) or {}
    if not impacts:
        render_empty_state(
            "Driver contributions unavailable",
            "Per-block contributions appear once the valuation pass completes with an "
            "admitted cross-section.",
            eyebrow="Mūla · Drivers",
        )
        return
    items = list(impacts.items())[:15]          # already sorted by contribution
    labels = [k for k, _v in items][::-1]
    vals = [float(v) for _k, v in items][::-1]
    mx = max(vals) if vals else 1.0
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[chart_rgba("violet", round(0.35 + 0.55 * (v / mx), 2)) for v in vals]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**chart_layout(height=max(240, len(labels) * 26), show_legend=False))
    style_axes(fig, x_title="Contribution %")
    fig.update_yaxes(showgrid=False)
    render_chart_panel(fig, "mula_drivers", units="contribution to fair value", windowed=False,
                       footer=f"{len(labels)} of {len(impacts)} asset-class blocks by current contribution.")
    hist = engine.get_feature_impact_history() if hasattr(engine, "get_feature_impact_history") else None
    if hist is not None and not hist.empty:
        render_sub_header("Contribution over time")
        render_table_panel(hist.tail(10), "mula-drivers-history", units="latest sessions",
                           windowed=False, max_rows=10, max_height=240)


# ═══════════════════════════════════════════════════════════════════════
#  6 · WINDOW AGREEMENT
# ═══════════════════════════════════════════════════════════════════════

def _render_window_agreement(ts_filtered, x_axis, signal, windows):
    """How broad and persistent the mispricing is across 1m / 1q / 1y windows.

    Evidence about the claim, not a second claim: no BUY/SELL words here. The
    stack used to be built on 3/5/10-day windows, which made it a two-week
    timing read wearing valuation labels ("OVERSOLD" while the oscillator said
    fair). On month / quarter / year windows it describes the same mispricing
    the oscillator does, from a different angle — how unusual it is for this
    instrument.
    """
    _t = _active_tiers()
    strong = _series_tier("ConvictionBounded", _t.conviction_strong, 0.90)
    moderate = _series_tier("ConvictionBounded", _t.conviction_moderate, 0.75)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], shared_xaxes=True,
                        vertical_spacing=0.08)
    if {"ConvictionUpper", "ConvictionLower"} <= set(ts_filtered.columns):
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["ConvictionUpper"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["ConvictionLower"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=chart_rgba("slate", 0.06),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
    cb = pd.to_numeric(ts_filtered["ConvictionBounded"], errors="coerce")
    fig.add_trace(go.Scatter(x=x_axis, y=cb.clip(lower=0), fill="tozeroy",
                             fillcolor=chart_rgba("rose", 0.08), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=cb.clip(upper=0), fill="tozeroy",
                             fillcolor=chart_rgba("emerald", 0.08), line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=cb, mode="lines", name="Window agreement (DDM)",
                             line=dict(color=chart_color("violet"), width=1.4)), row=1, col=1)
    for y, nm in ((strong, "rose"), (-strong, "emerald")):
        fig.add_hline(y=y, line_dash="dot", line_color=chart_rgba(nm, 0.25), line_width=0.5,
                      row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["OversoldBreadth"], name="Windows reading cheap",
                             fill="tozeroy", fillcolor=chart_rgba("emerald", 0.1),
                             line=dict(color=chart_color("emerald"), width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["OverboughtBreadth"], name="Windows reading rich",
                             fill="tozeroy", fillcolor=chart_rgba("rose", 0.1),
                             line=dict(color=chart_color("rose"), width=1.2)), row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM + 80))
    style_axes(fig, y_title="Agreement", y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title="% of windows", y_range=[0, 100], row=2, col=1)
    render_chart_panel(fig, "fvo_ddm", units=f"windows {'/'.join(str(w) for w in windows)}d")

    cv = _num(signal, "conviction_score")
    if cv is None:
        return
    os_, ob_ = signal.get("oversold_breadth", 0) or 0, signal.get("overbought_breadth", 0) or 0
    n_w = len(windows)
    if abs(cv) >= strong:
        title, tone = ("BROADLY CHEAP" if cv < 0 else "BROADLY RICH"), ("success" if cv < 0 else "danger")
        body = (f"Window agreement {cv:+.0f} — beyond this instrument's own p90. The gap is unusually "
                f"{'low' if cv < 0 else 'high'} against its 1m, 1q and 1y history alike.")
    elif abs(cv) >= moderate:
        title, tone = ("LEANS CHEAP" if cv < 0 else "LEANS RICH"), ("success" if cv < 0 else "danger")
        body = (f"Window agreement {cv:+.0f} — past p75. Most windows read the gap as "
                f"{'cheap' if cv < 0 else 'rich'} for this instrument, not all.")
    else:
        title, tone = "NOT UNUSUAL", "neutral"
        body = f"Window agreement {cv:+.0f} — the gap is within its ordinary range on these windows."
    body += (f" Right now {os_:.0f}% of the {n_w} windows read it cheap and {ob_:.0f}% rich.")
    render_interpretation_card(title=title, body=body, color=tone)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_mula_tab(engine, ts_filtered, x_axis, signal, model_stats, ts, active_target,
                    edge: dict | None = None, horizon: int = 10):
    """Mūla tab — the valuation call, its evidence, and its record."""
    fvo = _num(signal, "fvo")
    pct = (_num(signal, "pct_mispricing") or 0.0) * 100.0
    _n_inst = int(model_stats.get("n_features", 0) or 0)
    _n_blocks = int(model_stats.get("n_blocks", 0) or 0)
    if fvo is None:
        claim = "No valuation has been published for this instrument yet."
    elif abs(fvo) < CLAIM_SIGMA:
        claim = (f"{active_target} trades {abs(pct):.1f}% {'below' if pct < 0 else 'above'} the level "
                 f"{_n_inst} instruments across {_n_blocks} asset-class blocks imply — within the "
                 f"normal noise around fair value, so no claim is made.")
    else:
        claim = (f"{active_target} trades {abs(pct):.1f}% {'below' if pct < 0 else 'above'} the level "
                 f"{_n_inst} instruments across {_n_blocks} asset-class blocks imply "
                 f"({fvo:+.2f}σ): {valuation_state(fvo).lower()}.")
    if signal.get("asof_row_withheld"):
        claim += (f" The latest session was not valued ({signal.get('withheld_reason') or 'withheld'}); "
                  f"this is the last session that was.")

    render_section_header("Mūla — The Valuation Call", claim, icon="target", accent="violet")
    _render_read(signal, ts, edge, horizon)

    render_section_header(
        "Price & Fair Value",
        "Top: price against the fair-value level and its 95% band. Bottom: the oscillator the "
        f"call is made on — inside the shaded ±{CLAIM_SIGMA:g}σ band no claim is made; beyond "
        f"±{DEEP_SIGMA:g}σ (dotted) is a deep mispricing. Below zero (green) = cheap.",
        icon="trending", accent="violet",
    )
    _render_fair_value_chart(ts_filtered, x_axis, active_target)

    render_section_header(
        "Error Correction — Does Price Close the Gap?",
        "A stationary gap is not yet a trade: it can close by fair value moving to price. This "
        "is the price-side read — how today's price change responds to yesterday's gap, and "
        "which forecast family has actually predicted it.",
        icon="crosshair", accent="violet",
    )
    _render_error_correction(ts_filtered, x_axis, signal, horizon)

    render_section_header(
        "Model Quality",
        "Does the cross-section locate this instrument's level at all — and better than its "
        "own trailing mean?",
        icon="cpu", accent="violet",
    )
    _render_model_quality_cards(model_stats, signal)

    render_section_header(
        "Walk-Forward Edge",
        f"When Mūla called this instrument cheap or rich, did it move that way over the next "
        f"{horizon} sessions? Scored on non-overlapping windows, period by period.",
        icon="trending", accent="violet",
    )
    render_edge_panel(edge, "mula", what="Mūla's call")

    render_section_header(
        "What Drives Fair Value",
        "How far fair value moves when each asset-class block is removed and the valuation "
        "refitted — an ablation, not a coefficient read-off.",
        icon="bar-chart", accent="violet",
    )
    _render_drivers(engine)

    _windows = tuple(getattr(engine, "lookback_windows", ()) or ())
    render_section_header(
        "Window Agreement",
        "The gap against its own 1-month, 1-quarter and 1-year history: how broad and how "
        "unusual the mispricing is for this instrument. Evidence about the call, not a second "
        "call.",
        icon="layers", accent="violet",
    )
    _render_window_agreement(ts_filtered, x_axis, signal, _windows)
