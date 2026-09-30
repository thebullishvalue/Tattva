"""
Tattva — Precedent tab: what happened last time the state looked like this.

Covariance-aware Mahalanobis analog matching over the state the SYSTEM
measures — momentum, realised volatility, Swayam breadth, the Mūla valuation
oscillator, market stress and valuation confidence — under a Theiler exclusion
window so the returned analogs are distinct episodes rather than adjacent days
of one episode. The output is an empirical base rate, independent of the model:
it is the one read on the app that does not depend on the engines being right.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Analog skill term structure
  2 ANCHOR    what is the underlying claim?      Forward-return base rate
  3 SIGNAL    what does it say to do?            Predicted vs realised history
  4 STATE     how does that sit historically?    The analog cards themselves
  5 DETAIL    the evidence behind it             State → forward-return backtest
"""

from __future__ import annotations

import html as html_mod

#: Analog cards rendered. The base rate is computed over every returned
#: episode, not just these — see the section description.
MAX_ANALOG_CARDS = 10

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.analogs import (
    find_similar_periods, summarize_forward,
    analog_skill_by_horizon,
)
from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    render_empty_state,
    render_chart_panel,
    render_note,
    )
from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba)


def _classify_state(fvo: float) -> tuple[str, str, str, str]:
    """Map the valuation oscillator to (tier, chip tone, label, fill) classes.

    Describes WHERE the target sat versus its macro-implied level — cheap or
    rich — not what happened next; the forward tiles carry the realised outcome.

    Keyed on Mūla because that is what the analogs are MATCHED on. It previously
    keyed on AvgZ, which the tuning study had already removed from the matching
    feature set, so the badge described a dimension the similarity score
    ignored: two cards labelled "Deep Oversold" could have been selected for
    reasons unrelated to being oversold. Thresholds are in units of the
    engine's own predictive SD, so +/-1 and +/-2 read as the usual sigma bands.

    The chip tone is one of the app's shared ``.chip-*`` tones (``render_chip``
    in ui/components.py); ``"success-outline"``/``"danger-outline"`` render as
    the lighter, unfilled variant for the weaker "Cheap"/"Rich" reads vs the
    filled "Deep Value"/"Very Rich" extremes.
    """
    if fvo <= -2.0:
        return "tier-strong-buy", "success", "Deep Value", "fill-strong-buy"
    if fvo <= -1.0:
        return "tier-buy", "success-outline", "Cheap", "fill-buy"
    if fvo >= 2.0:
        return "tier-caution", "danger", "Very Rich", "fill-caution"
    if fvo >= 1.0:
        return "tier-caution", "danger", "Rich", "fill-caution"
    return "tier-hold", "neutral", "Fair", "fill-hold"


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
    fvo = period.get("fvo", 0.0)
    stress = period.get("stress", 0.5)
    similarity_pct = period["similarity"] * 100
    price_val = period["price"]
    tier_cls, chip_tone, badge_label, bar_cls = _classify_state(fvo)
    tone_base, outline = (chip_tone[:-8], True) if chip_tone.endswith("-outline") else (chip_tone, False)
    badge_html = (
        f'<span class="chip chip-{tone_base}{" chip-outline" if outline else ""}">'
        f'{html_mod.escape(badge_label)}</span>'
    )
    fwd = period["fwd"]
    fwd_tiles = "".join(_render_fwd_tile(int(h), fwd.get(int(h))) for h in hold_horizons)
    # Grid adapts to the horizon count (6 for PRECEDENT_HORIZONS) so tiles fill
    # the row instead of left-packing into the legacy 4-column track.
    grid_style = f"grid-template-columns:repeat({max(1, len(hold_horizons))},1fr);"

    # Cheap (negative Mūla) is the bullish pole, so it takes the positive colour.
    z_color = "pos" if fvo < 0 else "neg" if fvo > 0 else "neutral"

    # NOTE: the f-string below must stay flush-left with NO blank lines — Streamlit
    # feeds it to a CommonMark parser; a blank line closes the HTML block and the
    # rest renders as raw text. (Same rule as Arthagati's analog card.)
    st.markdown(
        f"""\
<div class="analog-card {tier_cls}">
  <div class="analog-card-head">
    <div class="analog-card-id">
      <div class="analog-eyebrow">Analog · Historical Match</div>
      <div class="analog-symbol">{html_mod.escape(period['date'])}</div>
    </div>
    {badge_html}
  </div>
  <div class="analog-stat-row">
    <div class="analog-stat">
      <span class="analog-stat-label">Similarity</span>
      <span class="analog-stat-value accent">{similarity_pct:.1f}%</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Valuation (Mūla)</span>
      <span class="analog-stat-value {z_color}">{fvo:+.2f}&sigma;</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Market Stress</span>
      <span class="analog-stat-value">{stress:.0%}</span>
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
    and Mahalanobis distance/Theiler selection work
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
        render_empty_state(
            "No engine time-series available",
            "Precedent matching needs a fitted engine run to draw its state features from.",
            eyebrow="Precedent",
            action_label="Run analysis in the sidebar, then return to this page.",
        )
        return

    periods = precomputed_periods if precomputed_periods is not None else find_similar_periods(
        ts, active_target, hold_horizons=display_hold, mom_window=mom_window,
    )
    if not periods:
        render_empty_state(
            "Not enough historical data",
            f"Too few distinct historical episodes matched {html_mod.escape(active_target)}'s "
            "current state closely enough to form a base rate.",
            eyebrow="Precedent",
        )
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
            "Mūla forecast: agreement strengthens conviction, disagreement is a "
            "divergence worth respecting."
        ),
        color="info",
    )


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
        # COLOUR BY EDGE, NOT BY IC. They disagree, and colouring by IC paints
        # the worst horizon green: measured 2026-08-17, Gold +60d had IC +0.19
        # (positive, green) while its directional edge was -11.1pp — the matcher
        # called direction worse than a constant guess. IC ranks magnitudes;
        # edge is what a directional read is worth. A "can this be believed?"
        # panel must be coloured by the second.
        _edges = [_scored[h].get("edge", float("nan")) for h in _hs_sorted]
        _bar_colors = [
            chart_color("emerald") if (np.isfinite(e) and e > 0)
            else chart_color("rose") if np.isfinite(e)
            else (chart_color("emerald") if v > 0 else chart_color("rose"))
            for e, v in zip(_edges, _ic_vals)
        ]
        # Carry the hit-rate's NULL alongside it. A bare "hit-rate 50%" reads as
        # a coin flip when the benchmark is the unconditional majority direction
        # of the same windows — 57.8% on Gold, so 50% is materially WORSE than
        # ignoring the state entirely. `edge` = hit - base is the honest number.
        _cust = [[_scored[h]["hit"], _scored[h]["n"], _scored[h]["pval"],
                  _scored[h].get("base", float("nan")),
                  _scored[h].get("edge", float("nan"))] for h in _hs_sorted]
        _fig_ts = go.Figure()
        _fig_ts.add_trace(go.Bar(
            x=[f"+{h}d" for h in _hs_sorted], y=_ic_vals,
            marker=dict(color=_bar_colors, opacity=0.85),
            customdata=_cust,
            hovertemplate=("Horizon %{x}<br>IC %{y:+.2f}<br>Hit-rate %{customdata[0]:.0f}%"
                           " vs base %{customdata[3]:.0f}%"
                           "<br>Edge %{customdata[4]:+.1f}pp"
                           "<br>n=%{customdata[1]} · p=%{customdata[2]:.3f}<extra></extra>"),
        ))
        _fig_ts.add_hline(y=0, line_color=chart_rgba("slate", 0.35), line_width=1)
        _layout_ts = chart_layout(height=300, show_legend=False)
        _fig_ts.update_layout(**_layout_ts)
        style_axes(_fig_ts, y_title="Walk-forward IC (Spearman)", x_title="Holding horizon")
        # Bar HEIGHT is IC, bar COLOUR is edge, and they can disagree — a tall
        # green-looking bar with a negative edge is exactly the case this panel
        # exists to expose, so the mismatch is deliberate and has to be stated
        # or it reads as a rendering bug.
        render_note(
            "Bar height = rank IC (does the prediction order outcomes?). "
            "Bar colour = directional edge vs the base rate (is acting on the "
            "direction worth anything?). A tall bar in red means the matcher "
            "ranks magnitudes but calls direction worse than assuming the "
            "majority — hover for both numbers."
        )
        render_chart_panel(_fig_ts, "analog_skill_term_structure", units="Spearman IC", window=True)

        # Plain-language read: best horizon + whether the forecast horizon is
        # where the edge actually lives.
        # "Strongest" must mean strongest AT DECIDING, so rank by edge where it
        # exists. Ranking by IC announced +60d as strongest on Gold — the
        # horizon with the most negative edge in the whole term structure.
        def _rank_key(h):
            e = _scored[h].get("edge", float("nan"))
            return (0, e) if np.isfinite(e) else (-1, _scored[h]["ic"])
        _best_h = max(_scored, key=_rank_key)
        _best = _scored[_best_h]
        _lens_note = ""
        if fwd_horizon in _scored and _best_h != fwd_horizon:
            _lens_note = (f" · forecast horizon {fwd_horizon}d reads "
                          f"{_scored[fwd_horizon]['ic']:+.2f}, not the strongest")
        elif fwd_horizon in _scored:
            _lens_note = f" · forecast horizon {fwd_horizon}d is also the strongest"
        _edge = _best.get("edge", float("nan"))
        _base = _best.get("base", float("nan"))
        _edge_txt = (f" · hit-rate {_best['hit']:.0f}% vs base rate {_base:.0f}% "
                     f"({_edge:+.1f}pp)") if np.isfinite(_edge) else (
                     f" · hit-rate {_best['hit']:.0f}%")
        render_note(
            f"Strongest at +{_best_h}d · IC {_best['ic']:+.2f} (p={_best['pval']:.3f})"
            f"{_edge_txt} over {_best['n']} windows{_lens_note}"
        )
        # State it plainly when the matching is not adding anything. Measured
        # 2026-08-17 the edge was negative at every horizon on Gold and Copper,
        # so this is the expected reading, not an exceptional warning.
        if np.isfinite(_edge) and _edge <= 0:
            render_note(
                "Negative edge: at this horizon the analog matching did not beat "
                "simply assuming the majority direction of the same windows. Read "
                "the base rate below as context for what the state has historically "
                "preceded, not as a directional forecast."
            )

    # (The 'Analog Predictions Over Time' section stood here: a per-as-of-date
    # predicted-vs-realised scatter for the active horizon. It restated, at
    # much greater length, what the term-structure chart above already says
    # — and its own summary line was the term structure's headline number.)

    # ── Analog period cards (2-column grid) ─────────────────────────────────
    render_section_header(
        title="Top Analog Periods",
        description=(f"The {min(len(periods), MAX_ANALOG_CARDS)} closest of "
                     f"{len(periods)} distinct episodes behind the base rate above"),
        icon="layers",
    )

    # Cards show the closest few; the base rate above is computed over ALL
    # returned episodes (see analytics.analogs.find_similar_periods).
    # "small" like every other grid in the app — this was the one call site
    # using a wider gutter, so the analog cards sat further apart than the
    # metric cards directly above them.
    analog_cols = st.columns(2, gap="small")
    for i, period in enumerate(periods[:MAX_ANALOG_CARDS]):
        with analog_cols[i % 2]:
            _render_period_card(period, active_target, display_hold)

    # ── Backtest: state extension vs forward return (descriptive) ───────────
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
        render_note("Insufficient data points for backtest.")
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
        render_note("Insufficient data points for backtest.")
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
        marker=dict(size=4, color=np.where(tr_x > 0, chart_color("emerald"), chart_color("rose")), opacity=0.4),
        hovertemplate="Z: %{x:.2f}<br>Fwd: %{y:.1f}%<extra></extra>",
        name=f"Train (70%, n={len(tr_x)})",
    ))
    fig.add_trace(go.Scattergl(
        x=te_x, y=te_y, mode="markers",
        marker=dict(size=6, color=np.where(te_x > 0, chart_color("emerald"), chart_color("rose")),
                    opacity=0.85, symbol="diamond"),
        hovertemplate="Z: %{x:.2f}<br>Fwd: %{y:.1f}%<extra></extra>",
        name=f"Test (30%, n={len(te_x)})",
    ))
    if len(tr_x) > 10:
        xr = np.linspace(x.min(), x.max(), 50)
        z1 = np.polyfit(tr_x, tr_y, 1)
        fig.add_trace(go.Scatter(
            x=xr, y=z1[0] * xr + z1[1], mode="lines",
            line=dict(color=chart_color("amber"), width=1.2, dash="dash"),
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
            line=dict(color=chart_color("cyan"), width=1.2, dash="dot"),
            name=f"Quadratic (train ρ_s={tr_s:.2f}, test ρ_s={oos_s:.2f})",
        ))

    fig.add_hline(y=0, line_color=chart_rgba("slate", 0.35), line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color=chart_rgba("slate", 0.35), line_width=1, line_dash="dot")

    layout = chart_layout(height=420, show_legend=True)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    style_axes(fig, y_title=f"{active_target} Return T+{horizon}d (%)", x_title="Extension Z at T")

    # This was the one chart in the app that passed its own Plotly config —
    # a private, hand-rolled second opinion on chrome. It now takes the same
    # PLOTLY_CONFIG as the other seventeen, via the same panel.
    render_chart_panel(fig, "analog_extension_scatter", units="% return")

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
