"""
Tattva — Precedent view (historical analog matching + forward returns + backtest).

Ports Arthagati's Similar-Periods view: covariance-aware Mahalanobis analog cards,
a forward-return base-rate summary, and a descriptive state→forward-return
backtest. Inputs are Tattva's engine.ts_data state features; forward-return
horizons follow the active Signal-Horizon lens.
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.analogs import find_similar_periods, summarize_forward
from core.config import (
    PRECEDENT_HONORARY_HORIZON,
    COLOR_GREEN, COLOR_RED, COLOR_GOLD, COLOR_CYAN, COLOR_MUTED,
)
from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    section_gap,
)
from ui.theme import chart_layout, style_axes


def _classify_state(avgz: float) -> tuple[str, str, str, str]:
    """Map the conformal extension (AvgZ) to (tier, badge, label, fill) classes.

    Descriptive of WHERE the target sat (oversold↔overbought), not a forecast —
    the forward tiles carry the realized outcome colouring.
    """
    if avgz <= -2.0:
        return "tier-strong-buy", "badge-strong-buy", "Deep Oversold", "fill-strong-buy"
    if avgz <= -1.0:
        return "tier-buy", "badge-buy", "Oversold", "fill-buy"
    if avgz >= 1.0:
        return "tier-caution", "badge-caution", "Overbought", "fill-caution"
    return "tier-hold", "badge-hold", "Neutral", "fill-hold"


def _render_fwd_tile(horizon: int, val: float | None) -> str:
    """One forward-return tile in the analog card footer grid."""
    if val is None:
        return (
            f'<div class="analog-fwd-tile neutral">'
            f'<span class="analog-fwd-label">+{horizon}D</span>'
            f'<span class="analog-fwd-value">—</span>'
            f"</div>"
        )
    cls = "pos" if val > 0 else "neg"
    return (
        f'<div class="analog-fwd-tile {cls}">'
        f'<span class="analog-fwd-label">+{horizon}D</span>'
        f'<span class="analog-fwd-value">{val:+.1f}%</span>'
        f"</div>"
    )


def _render_period_card(period: dict, target: str, hold_horizons: tuple[int, ...]) -> None:
    """Render one analog-period card — Obsidian Quant fidelity (ported)."""
    avgz = period["avgz"]
    similarity_pct = period["similarity"] * 100
    price_val = period["price"]
    tier_cls, badge_cls, badge_label, bar_cls = _classify_state(avgz)

    fwd = period["fwd"]
    fwd_tiles = "".join(_render_fwd_tile(int(h), fwd.get(int(h))) for h in hold_horizons)
    # Grid adapts to the lens hold count (2 for the finalized lenses) so tiles fill
    # the row instead of left-packing into the legacy 4-column track.
    grid_style = f"grid-template-columns:repeat({max(1, len(hold_horizons))},1fr);"

    z_color = "pos" if avgz < 0 else "neg" if avgz > 0 else "neutral"

    # NOTE: the f-string below must stay flush-left with NO blank lines — Streamlit
    # feeds it to a CommonMark parser; a blank line closes the HTML block and the
    # rest renders as raw text. (Same rule as Arthagati's analog card.)
    st.markdown(
        f"""\
<div class="position-card analog-card {tier_cls}">
  <div class="analog-card-head">
    <div class="analog-card-id">
      <div class="analog-eyebrow">Analog · Historical Match</div>
      <div class="analog-symbol">{html_mod.escape(period['date'])}</div>
    </div>
    <span class="position-card-badge {badge_cls}">{badge_label}</span>
  </div>
  <div class="analog-stat-row">
    <div class="analog-stat">
      <span class="analog-stat-label">Similarity</span>
      <span class="analog-stat-value amber">{similarity_pct:.1f}%</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Extension (Z)</span>
      <span class="analog-stat-value {z_color}">{avgz:+.2f}</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">{html_mod.escape(target)} at T</span>
      <span class="analog-stat-value">{price_val:,.2f}</span>
    </div>
  </div>
  <div class="analog-fwd-block">
    <div class="analog-fwd-block-label">Forward {html_mod.escape(target)} Return</div>
    <div class="analog-fwd-grid" style="{grid_style}">{fwd_tiles}</div>
  </div>
  <div class="analog-card-foot">
    <span class="analog-foot-label">Similarity</span>
    <div class="conviction-bar">
      <div class="conviction-bar-fill {bar_cls}" style="width:{similarity_pct:.0f}%;"></div>
    </div>
    <span class="analog-foot-pct">{similarity_pct:.0f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_precedent_tab(
    ts,
    active_target: str,
    hold_horizons: tuple[int, ...],
    mom_window: int,
    fwd_horizon: int,
) -> None:
    """Render the Precedent view — analog cards + base-rate summary + backtest."""

    # Display horizons = the lens hold grid + an honorary +1d reference tile (the
    # analog has no edge at 1d; shown with a caveat, NOT part of calibration).
    if PRECEDENT_HONORARY_HORIZON is not None and PRECEDENT_HONORARY_HORIZON not in hold_horizons:
        display_hold = tuple(sorted(set((PRECEDENT_HONORARY_HORIZON,) + tuple(hold_horizons))))
    else:
        display_hold = tuple(hold_horizons)

    render_section_header(
        title="Similar Historical Periods",
        description=(f"Covariance-aware Mahalanobis state-matching · forward {active_target} "
                     f"returns from each analog · lens horizons {'/'.join(str(h) for h in hold_horizons)}d"),
        icon="compass",
        accent="emerald",
    )

    if ts is None or len(ts) == 0 or "Price" not in getattr(ts, "columns", []):
        st.warning("No engine time-series available — run an analysis first.")
        return

    periods = find_similar_periods(
        ts, active_target, hold_horizons=display_hold, mom_window=mom_window,
    )
    if not periods:
        st.warning("Not enough historical data to find similar periods.")
        return

    # ── Forward-return base-rate summary (one card per horizon) ──────────────
    summary = summarize_forward(periods, display_hold)
    if summary:
        cols = st.columns(len(summary), gap="small")
        for col, (h, s) in zip(cols, summary.items()):
            _hon = (h == PRECEDENT_HONORARY_HORIZON)
            with col:
                render_metric_card(
                    label=f"+{h}D Median Return" + ("  · honorary" if _hon else ""),
                    value=f"{s['median']:+.1f}%",
                    subtext=("reference only — no edge at 1d" if _hon
                             else f"{s['positive_pct']:.0f}% positive ({s['n']} analogs)"),
                    color_class="neutral" if _hon else ("success" if s["median"] > 0 else "danger"),
                    icon="help-circle" if _hon else ("trending-up" if s["median"] > 0 else "trending-down"),
                )

    render_interpretation_card(
        title="Empirical Base Rate",
        body=(
            "This is a <strong>non-parametric base rate</strong> — what "
            f"{html_mod.escape(active_target)} actually did after the most "
            "statistically-similar historical states (covariance-aware Mahalanobis "
            "on Tattva's own state features). Read it <strong>alongside</strong> the "
            "Aarambh forecast: agreement strengthens conviction, disagreement is a "
            "divergence worth respecting."
        ),
        color="info",
    )

    section_gap()

    # ── Analog period cards (2-column grid) ─────────────────────────────────
    render_section_header(
        title="Top Analog Periods",
        description=f"Top {len(periods)} historical matches by similarity score",
        icon="layers",
    )

    analog_cols = st.columns(2, gap="medium")
    for i, period in enumerate(periods):
        with analog_cols[i % 2]:
            _render_period_card(period, active_target, display_hold)
            st.markdown('<div style="height: var(--sp-3);"></div>', unsafe_allow_html=True)

    # ── Backtest: state extension vs forward return (descriptive) ───────────
    section_gap()
    render_section_header(
        title=f"Backtest · Extension (Z) vs Forward {active_target} Return",
        description=f"Each dot = one independent (non-overlapping) {fwd_horizon}d window · honest OOS IC",
        icon="chart",
        accent="rose",
    )

    render_interpretation_card(
        title="Descriptive, Not Predictive",
        body=(
            "Points are evaluated with the engine's current state features — treat the "
            "relationship as <strong>descriptive</strong> context, not a standalone "
            "<strong>predictive</strong> signal. The calibrated edge lives in the "
            "Diagnostics walk-forward IC."
        ),
        color="warning",
    )

    df = ts.reset_index(drop=True)
    price = pd.to_numeric(df["Price"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(price)
    horizon = int(fwd_horizon)
    if n <= horizon + 20 or "AvgZ" not in df.columns:
        st.caption("Insufficient data points for backtest.")
        return

    avgz_full = np.asarray(df["AvgZ"], dtype=np.float64)[: n - horizon]
    fwd_full = (price[horizon:] / price[: n - horizon] - 1) * 100
    # NON-OVERLAPPING sampling (stride = horizon): adjacent forward windows don't
    # overlap, so the correlation isn't overlap-inflated (the honest IC for smooth
    # multi-day returns). Every dot below is one independent window.
    avgz = avgz_full[::horizon]
    fwd_ret = fwd_full[::horizon]

    valid = np.isfinite(avgz) & np.isfinite(fwd_ret)
    x = avgz[valid]
    y = fwd_ret[valid]
    if len(x) <= 20:
        st.caption("Insufficient data points for backtest.")
        return

    from scipy.stats import spearmanr as _spearmanr

    split = int(len(x) * 0.7)
    tr_x, tr_y = x[:split], y[:split]
    te_x, te_y = x[split:], y[split:]

    tr_p = np.corrcoef(tr_x, tr_y)[0, 1] if len(tr_x) > 2 else 0.0
    tr_s, _ = _spearmanr(tr_x, tr_y) if len(tr_x) > 2 else (0.0, 1.0)
    oos_p = np.corrcoef(te_x, te_y)[0, 1] if len(te_x) > 2 else 0.0
    oos_s, _ = _spearmanr(te_x, te_y) if len(te_x) > 2 else (0.0, 1.0)
    tr_s = 0.0 if not np.isfinite(tr_s) else tr_s
    oos_s = 0.0 if not np.isfinite(oos_s) else oos_s
    oos_p = 0.0 if not np.isfinite(oos_p) else oos_p

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tr_x, y=tr_y, mode="markers",
        marker=dict(size=4, color=np.where(tr_x > 0, COLOR_GREEN, COLOR_RED), opacity=0.4),
        hovertemplate="Z: %{x:.2f}<br>Fwd: %{y:.1f}%<extra></extra>",
        name=f"Train (70%, n={len(tr_x)})",
    ))
    fig.add_trace(go.Scattergl(
        x=te_x, y=te_y, mode="markers",
        marker=dict(size=6, color=np.where(te_x > 0, COLOR_GREEN, COLOR_RED),
                    opacity=0.85, symbol="diamond"),
        hovertemplate="Z: %{x:.2f}<br>Fwd: %{y:.1f}%<extra></extra>",
        name=f"Test (30%, n={len(te_x)})",
    ))
    if len(tr_x) > 10:
        xr = np.linspace(x.min(), x.max(), 50)
        z1 = np.polyfit(tr_x, tr_y, 1)
        fig.add_trace(go.Scatter(
            x=xr, y=z1[0] * xr + z1[1], mode="lines",
            line=dict(color=COLOR_GOLD, width=2, dash="dash"),
            name=f"Linear (train ρ={tr_p:.2f}, test ρ={oos_p:.2f})",
        ))
        z2 = np.polyfit(tr_x, tr_y, 2)
        fig.add_trace(go.Scatter(
            x=xr, y=z2[0] * xr ** 2 + z2[1] * xr + z2[2], mode="lines",
            line=dict(color=COLOR_CYAN, width=2, dash="dot"),
            name=f"Quadratic (train ρ_s={tr_s:.2f}, test ρ_s={oos_s:.2f})",
        ))

    fig.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")

    layout = chart_layout(height=420, show_legend=True)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    style_axes(fig, y_title=f"{active_target} Return T+{horizon}d (%)", x_title="Extension Z at T")

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False})

    oos_strong = oos_s if abs(oos_s) > abs(oos_p) else oos_p
    if abs(oos_strong) > 0.3:
        strength = "strong" if abs(oos_strong) > 0.5 else "moderate"
        direction = "positive" if oos_strong > 0 else "negative"
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_p:.2f} · Spearman {oos_s:.2f} — "
            f"{strength} {direction} relationship holds on unseen data.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {tr_p:.2f} · "
            f"Spearman {tr_s:.2f}</span>"
        )
        render_interpretation_card("Relationship Holds Out-of-Sample", body, color="success")
    else:
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_p:.2f} · Spearman {oos_s:.2f} — "
            f"weak out-of-sample relationship at the {horizon}d horizon.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {tr_p:.2f} · "
            f"Spearman {tr_s:.2f}</span><br><br>"
            "The extension→return link may be non-linear (see the quadratic curve) or "
            "stronger at a different lens horizon."
        )
        render_interpretation_card("Weak Out-of-Sample Fit", body, color="warning")
