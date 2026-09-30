"""
Tattva — FVO tab: Price vs fair value, the mispricing gap, conviction, breadth, model quality.
तत्त्व (Tattva) — "Principle / Essence"

UI — FVO engine visualization: the published fair-value LEVEL with its
predictive band, the mispricing gap that drives the signal stack, and rolling
robust-quantile bounds on that gap.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Model Quality
  2 ANCHOR    what is the underlying claim?      Price vs Fair Value
  3 SIGNAL    what does it say to do?            DDM Conviction → Breadth
  4 STATE     how does that sit historically?    Market State → Lookback States
  5 DETAIL    the evidence behind it             Signal Frequency → Average Z

Trust leads deliberately. A conviction reading with no idea whether the model
tracks this instrument is worse than no reading, so the card that says whether
to believe the tab comes before the tab's own conclusions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics.adaptive import tier_now
from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (
    render_metric_card,
    render_interpretation_card,
    render_section_header,
    render_chart_panel,
    render_note,
    provisional_note,
)
from core.config import (
    OU_PROJECTION_DAYS,
    get_instrument_config, InstrumentConfig,  # per-instrument display tiers (resolved
    # at render via _active_tiers(); conviction/breadth/model-spread tiers are NOT
    # imported as module globals — they're read off the active instrument's config).
    UI_R2_STRONG,
    UI_R2_ACCEPTABLE,
    UI_CHART_HEIGHT_MEDIUM,
    UI_CHART_HEIGHT_XLARGE,
    UI_CHART_HEIGHT_SMALL,
)

# (Tab-local colour aliases stood here as module-level constants. They were
# evaluated ONCE at import, when there is no session to read a theme from,
# so every chart drawn through them was frozen to whichever theme happened
# to be active at first import — the same import-time binding that made the
# original COLOR_* constants unable to follow Paper mode. Colours are
# resolved at the call site now, per render.

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "conviction_raw": (
        "Difference between the percentage of lookback windows calling the market oversold "
        "vs. overbought. Positive = more windows see overbought (bearish); negative = more "
        "see oversold (bullish). Values beyond +/-40 are extremes. This is the unsmoothed "
        "signal before any temporal filtering."
    ),
    "ddm_conviction": (
        "Conviction score smoothed through a Drift-Diffusion Model that accumulates evidence "
        "over time and reverts toward zero when signals are inconsistent. The shaded band is "
        "a heuristic uncertainty band (not a statistical confidence interval) — narrow bands "
        "mean recent evidence has been consistent, wide bands mean it has been noisy/conflicting. "
        "Use this (not raw conviction) for trade decisions."
    ),
    "oversold_breadth": (
        "Share of lookback windows where the idiosyncratic spread is stretched low — the target "
        "has underperformed its macro-implied path. Above 60% = broadly cheap vs macro, a bullish "
        "signal when it starts to turn up."
    ),
    "overbought_breadth": (
        "Share of lookback windows where the idiosyncratic spread is stretched high — the target "
        "has outrun its macro-implied path. Above 60% = broadly rich vs macro, a caution signal "
        "when it starts to turn down."
    ),
    "current_regime": (
        "Classification derived from OU half-life (speed of mean reversion) and DDM "
        "conviction score. 'Oversold' = market prices in excessive pessimism; 'Overbought' "
        "= excessive optimism. Half-life tells you how fast reversion is expected."
    ),
    "oos_r2": (
        "Share of log-price variance explained by the fitted level. A level regression on "
        "integrated factors reads high by construction (both sides trend), so treat this as "
        "a sanity check that the cross-section tracks the target at all — not as evidence of "
        "edge. R² vs Trailing Mean and Valuation Confidence are the discriminating numbers."
    ),
    "r2_vs_anchor": (
        "The published fair value versus the honest competing hypothesis: that price simply "
        "reverts to its own 252-day trailing mean. Positive = the global cross-section locates "
        "the level better than the asset's own history does, which is the whole claim of a "
        "cross-sectional valuation. Negative = a moving average would have served you better. "
        "Note this is deliberately NOT the random-walk test: one step ahead, yesterday's close "
        "beats any valuation of a near-integrated price, so that comparison measures a claim "
        "this engine never makes."
    ),
    "valuation_confidence": (
        "The engine's own gate on whether today's valuation is worth acting on: the geometric "
        "mean of (a) the online probability that the mispricing is mean-reverting rather than a "
        "permanent re-rating, and (b) how far independent slices of the cross-section agree on "
        "the sign of the mispricing. It usually sits near 1 — after thousands of sessions the "
        "residual's stationarity is a settled question — so it only bites in the regime where "
        "the relationship has genuinely broken down. That is exactly when you want it to."
    ),
    "mean_reversion": (
        "Augmented Dickey-Fuller test on the mispricing gap: is the gap stationary — does price "
        "actually come back to fair value — or does it wander off, meaning the 'mispricing' is a "
        "permanent re-rating you would be fading forever? p < 0.05 rejects the unit root and is "
        "what licenses trading the gap at all. Paired with the engine's ONLINE half-life estimate, "
        "which says how fast, and unlike the ADF statistic tracks a regime change rather than "
        "averaging over all history."
    ),
    "model_spread": (
        "Predictive standard deviation of the fair-value level, in basis points — how tightly "
        "the engine can locate fair value today, blending the two valuation views (latent "
        "factors and named asset-class blocks) and their disagreement with each other. Narrow "
        "= the cross-section pins the level; wide = it does not, so treat the mispricing with "
        "caution even when conviction is high."
    ),
}


def _series_tier(col: str, prior: float, q: float = 0.90) -> float:
    """Display cut-point for ``col``, from that series' own causal past.

    Reads the full (unfiltered) engine frame out of session state so the tier
    reflects the whole history rather than whatever timeframe the user is
    currently looking at — a p90 recomputed over the last 3 months would move
    every time the timeframe selector changed, which is not a threshold.

    Falls back to the config prior whenever the series is missing or too short.
    """
    ts = st.session_state.get("fvo_ts")
    if ts is None or col not in getattr(ts, "columns", ()):
        return float(prior)
    v = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=float)
    if "Valid" in ts.columns:
        v = v[ts["Valid"].astype(bool).to_numpy()]
    return tier_now(v, prior, q=q)


def _active_tiers() -> InstrumentConfig:
    """This render's instrument config — per-instrument display tiers (conviction /
    breadth / model-spread), resolved from the active target with a defaults
    fallback. Sub-render helpers call this and shadow the module-global tier names
    with per-instrument values, so a _PER_INSTRUMENT_OVERRIDES entry retunes how
    THIS target's already-computed signal is displayed."""
    try:
        return get_instrument_config(st.session_state.get("active_target", ""))
    except KeyError:
        return InstrumentConfig()


# (``_conviction_colors`` lived here — marker colouring for the raw-conviction
# scatter that was removed with it.)

# ═══════════════════════════════════════════════════════════════════════
#  CHART BUILDERS (extracted so they can be reused in any section order)
# ═══════════════════════════════════════════════════════════════════════

# (``_render_raw_conviction_chart`` lived here. It plotted ConvictionRaw — the
# same series the DDM chart below plots, only unsmoothed and without the
# uncertainty band, and NOT the series any card or downstream consumer reads.
# Two charts of one signal is not two pieces of information.)


def _render_ddm_conviction_chart(ts_filtered, x_axis, signal):
    """Section: DDM-Filtered Conviction with Uncertainty Band + interpretation card.

    Interpretation copy is valuation language throughout — "prices the market
    above/below fair value" is now literally what the underlying series says,
    since conviction is driven by the mispricing gap rather than by a forward-
    return forecast. (The forecast-mode wording this used to carry is gone with
    the engine it described; see the audit's E3 finding for why the two had to
    be kept apart.)
    """
    fig_conv = go.Figure()
    if "ConvictionUpper" in ts_filtered.columns:
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=chart_rgba("slate", 0.06),
            showlegend=False, hoverinfo="skip",
        ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionBounded"].clip(lower=0),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionBounded"].clip(upper=0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionBounded"], mode="lines", name="DDM Conviction",
        line=dict(color=chart_rgba("slate", 0.4), width=1.5),
    ))
    fig_conv.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5)

    fig_conv.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM, show_legend=False))
    style_axes(fig_conv, y_title="Conviction", y_range=[-100, 100])
    render_chart_panel(fig_conv, "fvo_ddm", units="conviction, ±100")

    # Interpretation card
    cv = signal["conviction_score"]
    _t = _active_tiers()   # warm-up priors, superseded once history allows
    CONVICTION_STRONG = _series_tier("ConvictionBounded", _t.conviction_strong, 0.90)
    CONVICTION_MODERATE = _series_tier("ConvictionBounded", _t.conviction_moderate, 0.75)

    if cv > CONVICTION_STRONG:
        regime_title = "STRONG OVERBOUGHT"
        regime_color = "danger"
        regime_body = (
            f"Conviction {cv:+.0f} — top decile. Most windows price the market above fair value. "
            f"Elevated drawdown risk from this zone. "
        )
    elif cv > CONVICTION_MODERATE:
        regime_title = "MODERATE OVERBOUGHT"
        regime_color = "warning"
        regime_body = (
            f"Conviction {cv:+.0f} — tilts overbought. Not at extremes, but evidence suggests "
            f"fair value sits below current price. "
        )
    elif cv > -CONVICTION_MODERATE:
        regime_title = "NEUTRAL"
        regime_color = "neutral"
        regime_body = (
            f"Conviction {cv:+.0f} — noise band. Windows are split — no reliable directional signal. "
            f"Stand aside or maintain current allocation. "
        )
    elif cv > -CONVICTION_STRONG:
        regime_title = "MODERATE OVERSOLD"
        regime_color = "success"
        regime_body = (
            f"Conviction {cv:+.0f} — tilts oversold. Market prices in more pessimism than the "
            f"ensemble justifies. Watch for conviction to roll over before entering. "
        )
    else:
        regime_title = "STRONG OVERSOLD"
        regime_color = "success"
        regime_body = (
            f"Conviction {cv:+.0f} — bottom decile. Most windows agree the market is below fair value. "
            f"Historically the most favorable return regime. "
        )

    # (The CI band-width tier sentence was REMOVED 2026-07-12: the DDM's
    # mean-reverting variance pins band width to a ~2-point range (p1–p99 =
    # 40.2–42.5, research/ui_anchors_study.py), so the NARROW/WIDE tiers could
    # never fire and the sentence permanently read "Band moderate — some
    # uncertainty" — a dead indicator, not information. The band itself is
    # still drawn on the conviction chart, where its shape is visible.)
    render_interpretation_card(title=regime_title, body=regime_body, color=regime_color)


def _render_market_breadth_chart(ts_filtered, x_axis):
    """Section: Market Breadth — oversold/overbought zone convergence."""
    UI_BREADTH_HIGH = _series_tier("OversoldBreadth", _active_tiers().ui_breadth_high, 0.90)
    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OversoldBreadth"],
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.1),
        line=dict(color=chart_color("emerald"), width=1.5), name="Oversold",
    ))
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OverboughtBreadth"],
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.1),
        line=dict(color=chart_color("rose"), width=1.5), name="Overbought",
    ))
    fig_zones.add_hline(y=UI_BREADTH_HIGH, line_dash="dot", line_color=chart_rgba("amber", 0.18), line_width=0.5)

    fig_zones.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig_zones, y_title="Breadth %", y_range=[0, 100])
    render_chart_panel(fig_zones, "fvo_breadth", units="% of windows")


def _render_market_state_cards(signal, regime_stats, ts):
    """Section: Market State — metric cards + regime distribution interpretation.

    The breadth cards read literally now: each lookback window scores how
    stretched the mispricing gap is against its own trailing distribution, so
    "cheap/expensive valuation" describes exactly what is being counted. Under
    the forecast engine this replaced, the same windows were measuring how
    extreme a forward-return FORECAST was, and the valuation wording had to be
    swapped out (the audit's E3 finding); that branch is gone with the engine.
    """
    UI_BREADTH_HIGH = _series_tier("OversoldBreadth", _active_tiers().ui_breadth_high, 0.90)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "OVERSOLD BREADTH", f'{signal["oversold_breadth"]:.0f}%',
            "Fraction of lookback windows that see cheap valuation. Rising = bullish pressure building.",
            "success" if signal["oversold_breadth"] > UI_BREADTH_HIGH else "neutral",
            tooltip=TOOLTIPS["oversold_breadth"],
        )
    with c2:
        render_metric_card(
            "OVERBOUGHT BREADTH", f'{signal["overbought_breadth"]:.0f}%',
            "Fraction of lookback windows that see expensive valuation. Rising = caution strengthening.",
            "danger" if signal["overbought_breadth"] > UI_BREADTH_HIGH else "neutral",
            tooltip=TOOLTIPS["overbought_breadth"],
        )
    with c3:
        curr_regime = signal["regime"]
        regime_short = curr_regime.replace("STRONGLY ", "").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        regime_color = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else "neutral"
        # Two half-lives, deliberately both shown. The OU estimate is a
        # full-history fit on the gap; `gap_half_life` is the engine's own
        # ONLINE AR(1), which is what the confidence gate actually uses and
        # what tracks a regime change. A wide split between them is itself
        # information: the reversion speed today is not the historical average.
        _gap_hl = float(signal.get("gap_half_life", 0.0) or 0.0)
        _desc = f"Mean reversion half-life: {signal['ou_half_life']:.0f}d. Shorter = faster snap-back to fair value."
        if _gap_hl > 0:
            _desc += f" Online estimate right now: {_gap_hl:.0f}d."
        render_metric_card(
            "CURRENT REGIME", regime_short, _desc, regime_color,
            tooltip=TOOLTIPS["current_regime"],
        )

    # Regime distribution interpretation. Denominator = the sum of the five
    # regime counts, NOT len(ts): get_regime_stats excludes the engine's
    # burn-in rows (no published valuation there), so dividing by the full row
    # count would silently deflate every percentage by the burn-in share.
    os_total = regime_stats["strongly_oversold"] + regime_stats["oversold"]
    ob_total = regime_stats["strongly_overbought"] + regime_stats["overbought"]
    neutral_count = regime_stats["neutral"]
    total = os_total + ob_total + neutral_count
    os_pct = os_total / total * 100 if total > 0 else 0
    ob_pct = ob_total / total * 100 if total > 0 else 0
    neutral_pct = neutral_count / total * 100 if total > 0 else 0

    if os_pct > 50:
        interp_title = "MARKET LEANS CHEAP"
        interp_color = "success"
        interp_text = (
            f"{os_pct:.0f}% of history ({os_total}/{total} periods) classified oversold. "
            f"Valuation sits near the lower end of its range. "
            f"Neutral {neutral_pct:.0f}%, overbought {ob_pct:.0f}%."
        )
    elif ob_pct > 50:
        interp_title = "MARKET LEANS EXPENSIVE"
        interp_color = "danger"
        interp_text = (
            f"{ob_pct:.0f}% of history ({ob_total}/{total} periods) classified overbought. "
            f"Valuation sits near the upper end of its range. "
            f"Neutral {neutral_pct:.0f}%, oversold {os_pct:.0f}%."
        )
    else:
        interp_title = "MARKET OSCILLATES EVENLY"
        interp_color = "neutral"
        interp_text = (
            f"No dominant regime. Neutral {neutral_pct:.0f}% ({neutral_count} periods), "
            f"oversold {os_pct:.0f}%, overbought {ob_pct:.0f}%. "
            f"Mean-reversion signals equally likely in both directions."
        )
    render_interpretation_card(title=interp_title, body=interp_text, color=interp_color)


def _render_model_quality_cards(model_stats, signal):
    """Section: Model Quality — five metric cards.

    Read left to right as a chain of increasingly demanding questions. Does the
    cross-section track this asset at all (OOS R²)? Does it locate the level
    better than the asset's own trailing mean does (R² vs Trailing Mean)? Do
    independent slices of the world agree on the mispricing's sign, and does
    the engine believe it reverts (Valuation Confidence)? Does the gap test
    stationary, and how fast (Mean Reversion)? And how tightly can fair value
    be pinned today at all (Model Spread)?

    A high R² with a negative R²-vs-anchor is the diagnostic worth knowing: the
    regression is fitting the trend, not the mispricing.
    """
    # Model-spread tiers in BASIS POINTS, from this instrument's own spread
    # history (the raw column is a log-level SD, hence the 1e4).
    _t = _active_tiers()
    UI_MODEL_SPREAD_LOW = _series_tier("ModelSpread", _t.ui_model_spread_low / 1e4, 0.10) * 1e4
    UI_MODEL_SPREAD_HIGH = _series_tier("ModelSpread", _t.ui_model_spread_high / 1e4, 0.90) * 1e4
    q1, q2, q3, q4, q5 = st.columns(5)

    with q1:
        r2 = model_stats["r2_oos"]
        render_metric_card(
            "OOS R²", f"{r2:.3f}",
            "Log-price variance explained by the fitted level. High is expected for a level "
            "regression — the discriminating numbers are the two cards to the right.",
            "success" if r2 > UI_R2_STRONG else "warning" if r2 > UI_R2_ACCEPTABLE else "danger",
            tooltip=TOOLTIPS["oos_r2"],
        )

    with q2:
        r2_anchor = model_stats.get("r2_vs_anchor", 0.0)
        render_metric_card(
            "R² vs Trailing Mean", f"{r2_anchor:+.3f}",
            "Edge over 'price reverts to its own 252d mean'. Negative = a moving average "
            "locates fair value better than the global cross-section does.",
            "success" if r2_anchor > 0.05 else "warning" if r2_anchor > -0.05 else "danger",
            tooltip=TOOLTIPS["r2_vs_anchor"],
        )

    with q3:
        conf = float(signal.get("valuation_confidence", 0.0) or 0.0)
        mr = float(signal.get("mr_prob", 0.0) or 0.0)
        xs = float(signal.get("xs_consistency", 0.0) or 0.0)
        render_metric_card(
            "Valuation Confidence", f"{conf:.2f}",
            f"Mean-reversion evidence {mr:.2f} × cross-sectional agreement {xs:.2f}. "
            "Low = the mispricing may be a permanent re-rating, or the cross-section "
            "disagrees with itself about its sign.",
            "success" if conf > 0.70 else "warning" if conf > 0.45 else "danger",
            tooltip=TOOLTIPS["valuation_confidence"],
        )

    with q4:
        adf_p = float(signal.get("adf_pvalue", 1.0) or 1.0)
        gap_hl = float(signal.get("gap_half_life", 0.0) or 0.0)
        hl_txt = f"Half-life {gap_hl:.0f}d." if gap_hl > 0 else "Half-life not estimable."
        mr_label = (
            f"Gap is stationary — price does return to fair value. {hl_txt}"
                if adf_p < 0.05
            else f"Unit root not rejected — the gap may be a permanent re-rating, "
                 f"not a mispricing to fade. {hl_txt}"
                if adf_p > 0.10
            else f"Borderline stationarity. Treat reversion as unproven. {hl_txt}"
        )
        render_metric_card(
            "Mean Reversion", f"p={adf_p:.3f}", mr_label,
            "success" if adf_p < 0.05 else "warning" if adf_p <= 0.10 else "danger",
            tooltip=TOOLTIPS["mean_reversion"],
        )

    with q5:
        sp = signal["model_spread"] * 10000  # log-level predictive SD → basis points
        render_metric_card(
            "Model Spread", f"{sp:.1f} bps",
            f"Predictive SD of the fair-value level. Above {UI_MODEL_SPREAD_HIGH:.0f} bps "
            f"(p90 of history) = fair value is poorly pinned — distrust the mispricing.",
            "success" if sp < UI_MODEL_SPREAD_LOW else "warning" if sp < UI_MODEL_SPREAD_HIGH else "danger",
            tooltip=TOOLTIPS["model_spread"],
        )


def _render_fair_value_chart(engine, ts_filtered, x_axis, ts, active_target):
    """Section: Price vs Fair Value + the mispricing gap.

    Top panel: the traded price against the fair-value LEVEL the global
    cross-section implies, inside its 95% predictive band. Where the price line
    sits relative to the band is the whole reading — inside it, the asset is
    trading where the opportunity set says it should; outside, the mispricing
    is larger than the engine's own uncertainty about where fair value is.

    Bottom panel: the mispricing gap (log price − log fair value), which is the
    series the entire signal stack runs on. Green below zero = cheap versus the
    cross-section; red above = rich. The OU projection extends the gap forward
    at its estimated reversion speed, with a cone from the dispersion of the
    rolling theta estimates.
    """
    has_price = "Price" in ts_filtered.columns
    has_band = {"FairValueLo", "FairValueHi"}.issubset(ts_filtered.columns)

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.06,
    )

    # ── Top: 95% predictive band around fair value, then fair value, then price.
    # Band first so it renders underneath both lines.
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
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value",
        line=dict(color=chart_color("violet"), width=1.2, dash="dot"),
        hovertemplate="%{y:.2f}<extra>Fair value</extra>",
    ), row=1, col=1)

    top_series = ts_filtered["Price"] if has_price else ts_filtered["Actual"]
    fig.add_trace(go.Scatter(
        x=x_axis, y=top_series, mode="lines", name=f"{active_target} Price",
        line=dict(color=chart_rgba("slate", 0.4), width=1.5),
    ), row=1, col=1)

    # ── Bottom: the mispricing gap ──
    # Sign convention: gap = log(price) − log(fair value). Positive = rich
    # (red above zero), negative = cheap (green below zero) — the same polarity
    # the zone/breadth/conviction stack downstream reads.
    series = ts_filtered["Residual"]
    bottom_name, bottom_title = "Mispricing Gap", "log(price / fair value)"
    pos_fill, neg_fill = chart_rgba("rose", 0.12), chart_rgba("emerald", 0.12)

    fig.add_trace(go.Scatter(
        x=x_axis, y=series.clip(lower=0), fill="tozeroy",
        fillcolor=pos_fill, line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=series.clip(upper=0), fill="tozeroy",
        fillcolor=neg_fill, line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=series, mode="lines", name=bottom_name,
        line=dict(color=chart_color("accent"), width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=2, col=1)

    # OU mean-reversion projection of the gap. Genuinely applicable here — the
    # gap IS a candidate mean-reverting spread, which is what the OU model
    # describes. (Under the forecast engine this replaced, the same projection
    # was suppressed because the series was a forecast, not a spread.)
    if (hasattr(engine, "ou_projection") and len(engine.ou_projection) > 0
            and pd.api.types.is_datetime64_any_dtype(ts["Date"])):
        from pandas import bdate_range
        last_date = ts["Date"].iloc[-1]
        proj_dates = bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection, mode="lines", name="OU Projection",
            line=dict(color=chart_rgba("slate", 0.4), width=1, dash="dot"), opacity=0.5,
        ), row=2, col=1)
        if len(engine.ou_projection_upper) > 0:
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig.add_trace(go.Scatter(x=proj_dates, y=engine.ou_projection_lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=chart_rgba("slate", 0.08), showlegend=False, hoverinfo="skip"), row=2, col=1)

    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_XLARGE))
    style_axes(fig, y_title=f"{active_target} Price", row=1, col=1)
    style_axes(fig, y_title=bottom_title, row=2, col=1)

    render_chart_panel(fig, "fvo_fairvalue", units="price · gap", window=True)


def _render_signal_frequency_chart(ts_filtered, x_axis):
    """Section: Signal Frequency — buy/sell threshold crossings."""
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=chart_color("emerald"), opacity=0.85),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=chart_color("rose"), opacity=0.85),
    ))

    fig_signals.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False), barmode="relative")
    style_axes(fig_signals, y_title="Count")
    render_chart_panel(fig_signals, "fvo_signal_freq", units="crossings")


def _render_avg_zscore_chart(ts_filtered, x_axis):
    """Section: Average Z-Score — statistical extremes across windows."""
    fig_z = go.Figure()
    bar_colors = [chart_color("emerald") if z < -1 else chart_color("rose") if z > 1 else chart_rgba("slate", 0.75) for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, opacity=0.85, showlegend=False))
    fig_z.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5)
    fig_z.add_hline(y=2, line_dash="dot", line_color=chart_rgba("rose", 0.18), line_width=0.5)
    fig_z.add_hline(y=-2, line_dash="dot", line_color=chart_rgba("emerald", 0.18), line_width=0.5)

    fig_z.update_layout(**chart_layout(height=UI_CHART_HEIGHT_SMALL, show_legend=False))
    style_axes(fig_z, y_title="Z-Score")
    render_chart_panel(fig_z, "fvo_avg_z", units="z-score")


def _render_lookback_states(ts_filtered):
    """Section: Current Lookback States — per-window z-score and zone.

    Zone is a categorical read (Under/Over/neutral), not a signed numeric
    column, so it doesn't map cleanly onto render_data_table's sign-based
    cell coloring — this stays a small purpose-built list, but through the
    same tokens (--surface-1/--line/--r-lg) every other panel uses, not an
    undefined --glass var.
    """
    from core.config import LOOKBACK_WINDOWS
    rows_html = []
    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_filtered.columns:
            continue
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        zone_cls = "long" if "Under" in zone else "short" if "Over" in zone else "flat"
        rows_html.append(
            f'<div class="lookback-row">'
            f'<span class="lbl">{lb}-Day Lookback</span>'
            f'<span class="val {zone_cls}">{zone} ({z:+.2f})</span>'
            f'</div>'
        )
    if rows_html:
        st.markdown(
            f'<div class="lookback-panel">{"".join(rows_html)}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION — sections arranged in logical analytical flow
# ═══════════════════════════════════════════════════════════════════════

def render_fvo_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target):
    """FVO tab — walk-forward valuation with violet system identity.

    Analytical flow:
      1. Model Quality        — "Can I trust this?"
      2. Actual vs Fair Value  — "What are we valuing?"
      3. Base Conviction       — "What's the raw signal?"
      4. DDM-Filtered          — "What's the processed signal?"
      5. Market Breadth        — "How much agreement?"
      6. Market State          — "What's the regime?"
      7. Lookback States       — "What do individual windows say?"
      8. Signal Frequency      — "Where are the crossings?"
      9. Average Z-Score       — "What's statistically extreme?"
    """

    # ── Phase 1: TRUST ─────────────────────────────────────────────────
    render_section_header(
        "Model Quality",
        "Is the ensemble reliable? Weak metrics here mean treat all fair-value estimates with caution.",
        icon="cpu",
        accent="violet",
    )
    _render_model_quality_cards(model_stats, signal)


    # ── Phase 2: ANCHOR ────────────────────────────────────────────────
    _n_inst = int(model_stats.get("n_features", 0) or 0)
    _n_blocks = int(model_stats.get("n_blocks", 0) or 0)
    render_section_header(
        "Price & Fair Value",
        (
            f"Top: the target's price against the level implied by {_n_inst} macro instruments "
            f"across {_n_blocks} asset-class blocks, inside its 95% predictive band. "
            f"Bottom: the mispricing gap — how far price sits from that level. Below zero "
            f"(green) = cheap versus the cross-section; above (red) = rich."
        ),
        icon="trending",
        accent="cyan",
    )
    _render_fair_value_chart(engine, ts_filtered, x_axis, ts, active_target)


    # ── Phase 3: SIGNAL ────────────────────────────────────────────────

    render_section_header(
        "DDM-Filtered Conviction with Uncertainty Band",
        "Evidence-accumulated signal with a heuristic band — narrow means consistent recent "
        "evidence. Not a statistical confidence interval.",
        icon="shield",
        accent="cyan",
    )
    _render_ddm_conviction_chart(ts_filtered, x_axis, signal)


    render_section_header(
        "Market Breadth",
        "Windows that agree the market is cheap (green) vs expensive (red) versus its "
        "macro-implied level. Convergence near zero = fairly valued.",
        icon="bar-chart",
        accent="emerald",
    )
    _render_market_breadth_chart(ts_filtered, x_axis)


    # ── Phase 5: EXTREMES ──────────────────────────────────────────────
    render_section_header(
        "Signal Frequency",
        "Buy/sell threshold crossings per window. Clusters = conviction building.",
        icon="zap",
        accent="rose",
    )
    _render_signal_frequency_chart(ts_filtered, x_axis)


    render_section_header(
        "Average Z-Score",
        "Mean z-score across all windows. Beyond ±2 (dotted) is statistically extreme. Green = cheap; red = expensive.",
        icon="target",
    )
    _render_avg_zscore_chart(ts_filtered, x_axis)
