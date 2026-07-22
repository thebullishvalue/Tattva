"""
Tattva — Precedent view (historical analog matching + forward returns + backtest).

Ports Arthagati's Similar-Periods view: covariance-aware Mahalanobis analog cards,
a forward-return base-rate summary, and a descriptive state→forward-return
backtest. Inputs are Tattva's engine.ts_data state features; forward-return
horizons are a fixed term structure (core.config.PRECEDENT_HORIZONS).
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.analogs import (
    find_similar_periods, summarize_forward, analog_prediction_series,
    analog_skill_by_horizon,
)
from core.config import (
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
    """Map the robust-quantile extension (AvgZ) to (tier, badge, label, fill) classes.

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
    # Grid adapts to the horizon count (6 for PRECEDENT_HORIZONS) so tiles fill
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
    precomputed_periods: list[dict] | None = None,
) -> None:
    """Render the Precedent view — analog cards + base-rate summary + backtest.

    ``precomputed_periods``: an already-computed analog list (from
    ``analytics.analogs.find_similar_periods``, called with the SAME
    ``display_hold`` this function would derive below) that the caller may
    pass in to skip recomputation. app.py computes this once for the hero
    card's precedent read and passes it here too — the tab used to call
    ``find_similar_periods`` a second time for the identical
    (ts, target, mom_window), redoing the expensive feature-frame build
    (incl. rolling Hurst) and Mahalanobis distance/Theiler selection work
    (audit finding F18). ``None`` (e.g. a cache-key mismatch) falls back to
    computing it here exactly as before.
    """

    # Display horizons = the FIXED precedent term structure passed in by app.py
    # (core.config.PRECEDENT_HORIZONS = 1/3/5/10/20/60d), horizon-independent. 1d is
    # a normal member here — the former "honorary" caveat is gone; the
    # Analog-Skill walk-forward chart below discloses per-horizon edge (incl.
    # where 1d/60d are weak) honestly via IC + p-value rather than a blanket note.
    display_hold = tuple(hold_horizons)

    render_section_header(
        title="Similar Historical Periods",
        description=(f"Covariance-aware Mahalanobis state-matching · forward {active_target} "
                     f"returns from each analog · term structure {'/'.join(str(h) for h in display_hold)}d"),
        icon="compass",
        accent="emerald",
    )

    if ts is None or len(ts) == 0 or "Price" not in getattr(ts, "columns", []):
        st.warning("No engine time-series available — run an analysis first.")
        return

    periods = precomputed_periods if precomputed_periods is not None else find_similar_periods(
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
            with col:
                render_metric_card(
                    label=f"+{h}D Median Return",
                    value=f"{s['median']:+.1f}%",
                    subtext=f"{s['positive_pct']:.0f}% positive ({s['n']} analogs)",
                    color_class="success" if s["median"] > 0 else "danger",
                    icon="trending-up" if s["median"] > 0 else "trending-down",
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

    # ── Analog SKILL — term structure across ALL precedent horizons ─────────
    # The base-rate cards above are a *snapshot* (today's analog pool). This
    # asks the harder, walk-forward question at EVERY horizon at once: across
    # history, how well did the analog matcher's prediction track what actually
    # happened +Hd later? Reading the whole term structure (1/3/5/10/20/60d)
    # shows WHERE the analog has edge — typically strongest at ~10-20d and
    # fading at the 1d and 60d ends, which the per-horizon IC/p-value below
    # discloses honestly. Cached per config (O(n·grid·|H|)).
    _askk = (f"analog_skill::{active_target}|{'/'.join(map(str, hold_horizons))}|"
             f"{mom_window}|{len(ts)}|"
             f"{float(pd.to_numeric(ts['Price'], errors='coerce').iloc[-1]):.6g}")
    _askc = st.session_state.get("_analog_skill_cache")
    if _askc is None or _askc.get("key") != _askk:
        try:
            _skill = analog_skill_by_horizon(
                ts, active_target, tuple(hold_horizons), mom_window=mom_window,
            )
        except Exception:
            _skill = {}
        _askc = {"key": _askk, "skill": _skill}
        st.session_state["_analog_skill_cache"] = _askc
    _skill = _askc["skill"]

    _scored = {h: s for h, s in _skill.items() if np.isfinite(s.get("ic", np.nan))}
    if _scored:
        render_section_header(
            title="Analog Skill — Term Structure",
            description=("Walk-forward IC (predicted vs realized) at each horizon (1→60d) — "
                         "the analog matcher's forward skill is NOT one number; it varies "
                         "by holding period. Bars = rank IC; hover for hit-rate and n."),
            icon="bar-chart",
            accent="cyan",
        )
        _hs_sorted = sorted(_scored.keys())
        _ic_vals = [_scored[h]["ic"] for h in _hs_sorted]
        _bar_colors = [COLOR_GREEN if v > 0 else COLOR_RED for v in _ic_vals]
        _cust = [[_scored[h]["hit"], _scored[h]["n"], _scored[h]["pval"]] for h in _hs_sorted]
        _fig_ts = go.Figure()
        _fig_ts.add_trace(go.Bar(
            x=[f"+{h}d" for h in _hs_sorted], y=_ic_vals,
            marker=dict(color=_bar_colors, opacity=0.85),
            customdata=_cust,
            hovertemplate=("Horizon %{x}<br>IC %{y:+.2f}<br>Hit-rate %{customdata[0]:.0f}%"
                           "<br>n=%{customdata[1]} · p=%{customdata[2]:.3f}<extra></extra>"),
        ))
        _fig_ts.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1)
        _layout_ts = chart_layout(height=300, show_legend=False)
        _fig_ts.update_layout(**_layout_ts)
        style_axes(_fig_ts, y_title="Walk-forward IC (Spearman)", x_title="Holding horizon")
        st.plotly_chart(_fig_ts, width='stretch', key="analog_skill_term_structure")

        # Plain-language read: best horizon + whether the forecast horizon is
        # where the edge actually lives.
        _best_h = max(_scored, key=lambda h: _scored[h]["ic"])
        _best = _scored[_best_h]
        _lens_note = ""
        if fwd_horizon in _scored and _best_h != fwd_horizon:
            _lens_note = (f" The forecast horizon ({fwd_horizon}d, IC "
                          f"{_scored[fwd_horizon]['ic']:+.2f}) is NOT the strongest — "
                          f"the analog edge concentrates at {_best_h}d.")
        elif fwd_horizon in _scored:
            _lens_note = f" The forecast horizon ({fwd_horizon}d) is also the strongest horizon."
        st.caption(
            f"Strongest at +{_best_h}d: IC {_best['ic']:+.2f} (p={_best['pval']:.3f}), "
            f"hit-rate {_best['hit']:.0f}% over {_best['n']} non-overlapping windows.{_lens_note} "
            "Positive IC = the matcher's directional call held out of sample."
        )
        section_gap()

    # ── Analog prediction history: predicted vs realized over time ──────────
    # The term structure above is the SUMMARY; this is the DETAIL for the active
    # forecast horizon — what the matcher would have predicted at each PAST as-of
    # date, using only information available then (candidate outcomes completed
    # by the as-of date, warm-up excluded, pool-only median cleaning — see
    # analytics.analogs.analog_prediction_series). Strided every `fwd_horizon`
    # rows so consecutive points are non-overlapping (the honest sampling for
    # smooth multi-day returns). Cached per config — Streamlit renders every tab
    # each rerun, and this is an O(n·grid) computation worth doing once.
    # Reuse the walk-forward the term structure already ran for this horizon
    # (identical parameters) instead of recomputing it; only fall back to a
    # standalone compute if the forecast horizon wasn't in the skill grid.
    _pred_df = None
    if fwd_horizon in _skill:
        _pred_df = _skill[fwd_horizon].get("df")
    if _pred_df is None:
        _apk = (f"analog_pred::{active_target}|{fwd_horizon}|{mom_window}|{len(ts)}|"
                f"{float(pd.to_numeric(ts['Price'], errors='coerce').iloc[-1]):.6g}")
        _apc = st.session_state.get("_analog_pred_cache")
        if _apc is None or _apc.get("key") != _apk:
            try:
                _pred_df = analog_prediction_series(
                    ts, active_target, fwd_horizon, mom_window=mom_window,
                )
            except Exception:
                _pred_df = pd.DataFrame(columns=["Date", "Predicted", "Realized"])
            _apc = {"key": _apk, "df": _pred_df}
            st.session_state["_analog_pred_cache"] = _apc
        _pred_df = _apc["df"]

    if len(_pred_df) >= 5:
        render_section_header(
            title=f"Analog Predictions Over Time · +{fwd_horizon}d",
            description=(f"What the matcher predicted at each past as-of date (using only "
                         f"data available then) vs what {active_target} actually did — "
                         f"non-overlapping every {fwd_horizon} sessions"),
            icon="activity",
            accent="cyan",
        )
        _pd_pred = _pred_df["Predicted"].to_numpy(dtype=float)
        _pd_real = _pred_df["Realized"].to_numpy(dtype=float)
        _pd_dates = _pred_df["Date"]
        # Hit coloring: prediction direction vs realized direction; amber =
        # window not yet complete (the live predictions at the right edge).
        _mk_colors = []
        for p, r in zip(_pd_pred, _pd_real):
            if not np.isfinite(r):
                _mk_colors.append(COLOR_GOLD)
            elif (p > 0) == (r > 0):
                _mk_colors.append(COLOR_GREEN)
            else:
                _mk_colors.append(COLOR_RED)

        _fig_ap = go.Figure()
        _fig_ap.add_trace(go.Scatter(
            x=_pd_dates, y=_pd_real, mode="lines", name="Realized",
            line=dict(color=COLOR_MUTED, width=1.3),
            connectgaps=False,
        ))
        _fig_ap.add_trace(go.Scatter(
            x=_pd_dates, y=_pd_pred, mode="lines+markers", name="Analog prediction",
            line=dict(color=COLOR_CYAN, width=1.6),
            marker=dict(size=6, color=_mk_colors,
                        line=dict(color="rgba(10,14,23,0.8)", width=1)),
        ))
        _fig_ap.add_hline(y=0, line_color="rgba(148,163,184,0.25)", line_width=0.8,
                          line_dash="dot")
        _layout_ap = chart_layout(height=340, show_legend=True)
        _fig_ap.update_layout(**_layout_ap)
        style_axes(_fig_ap, y_title=f"+{fwd_horizon}d return (%)")
        st.plotly_chart(_fig_ap, width='stretch', key="analog_pred_history")

        _done = np.isfinite(_pd_real)
        _cap = (f"{len(_pred_df)} as-of dates · marker green = predicted direction was "
                f"right, red = wrong, gold = window still open (live prediction).")
        if _done.sum() >= 10:
            from scipy.stats import spearmanr as _sp
            _ic, _pv = _sp(_pd_pred[_done], _pd_real[_done])
            _hit = float(np.mean(np.sign(_pd_pred[_done]) == np.sign(_pd_real[_done]))) * 100
            _cap += (f" Completed windows: IC {_ic:+.2f} (p={_pv:.3f}), directional hit "
                     f"{_hit:.0f}% over {int(_done.sum())} non-overlapping windows.")
        st.caption(_cap)
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
    oos_s, oos_s_pval = _spearmanr(te_x, te_y) if len(te_x) > 2 else (0.0, 1.0)
    tr_s = 0.0 if not np.isfinite(tr_s) else tr_s
    oos_s = 0.0 if not np.isfinite(oos_s) else oos_s
    oos_p = 0.0 if not np.isfinite(oos_p) else oos_p
    oos_s_pval = 1.0 if not np.isfinite(oos_s_pval) else oos_s_pval

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
    # Quadratic curve needs more points than the linear fit to avoid a 2nd-degree
    # polynomial chasing noise in a small training split — 40 is a low bar,
    # chosen only to exclude the smallest/thinnest backtests, not a claim of
    # genuine statistical power at 40 either.
    if len(tr_x) >= 40:
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

    st.plotly_chart(fig, width='stretch',
                    config={"displayModeBar": False, "displaylogo": False})

    # Gate the verdict on the test-split Spearman p-value, not a bare |rho|
    # magnitude threshold. After non-overlapping striding the test split is
    # typically only ~20-30 points (e.g. h=20 on ~2200 rows), where |rho|=0.3
    # has a two-sided p-value ~0.20 — a 1-in-5 fluke rate under the null that
    # was previously being branded "holds out-of-sample" regardless of n.
    # scipy's spearmanr already returns this p-value; use it directly rather
    # than a fixed correlation-magnitude cutoff that ignores sample size.
    n_test = len(te_x)
    if oos_s_pval < 0.10:
        strength = "strong" if abs(oos_s) > 0.5 else "moderate"
        direction = "positive" if oos_s > 0 else "negative"
        body = (
            f"<strong>Out-of-sample (30%, n={n_test}):</strong> Pearson {oos_p:.2f} · "
            f"Spearman {oos_s:.2f} (p={oos_s_pval:.3f}) — "
            f"{strength} {direction} relationship holds on unseen data.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {tr_p:.2f} · "
            f"Spearman {tr_s:.2f}</span>"
        )
        render_interpretation_card("Relationship Holds Out-of-Sample", body, color="success")
    else:
        body = (
            f"<strong>Out-of-sample (30%, n={n_test}):</strong> Pearson {oos_p:.2f} · "
            f"Spearman {oos_s:.2f} (p={oos_s_pval:.3f}, not significant at 10%) — "
            f"no reliable out-of-sample relationship at the {horizon}d horizon.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {tr_p:.2f} · "
            f"Spearman {tr_s:.2f}</span><br><br>"
            "The extension→return link may be non-linear (see the quadratic curve, when shown), "
            "stronger at a different horizon, or the test split (n above) may simply be "
            "too small to detect a real but modest effect."
        )
        render_interpretation_card("Weak Out-of-Sample Fit", body, color="warning")
