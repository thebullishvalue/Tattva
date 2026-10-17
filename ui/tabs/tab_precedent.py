"""
Tattva — Precedent tab: what happened last time the state looked like this.

Covariance-aware Mahalanobis analog matching over the state the SYSTEM
measures — momentum, realised volatility, the Mūla valuation oscillator and its
window agreement, market stress and valuation confidence — under a Theiler
exclusion window so the returned analogs are distinct episodes rather than
adjacent days of one episode. The output is an empirical base rate: no model
is fitted to forward returns to produce it.

It is NOT independent of the engines, and the page no longer says it is: the
state it matches on is the engines' own output, so "the last times the state
looked like this" means "the last times Mūla read it like this". That is what
makes it useful — it asks what this exact reading has historically preceded —
and also why it is one gate of six rather than a second opinion that could
outvote the call.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    what happened after states like today's?   Forward-return base rate
  2 TRUST     has the matcher been worth anything?       Analog skill term structure
  3 DETAIL    the evidence behind it                     The analog cards themselves

(The extension-vs-return backtest that closed this page graded Mūla's AvgZ on a
70/30 split — a Mūla diagnostic on the wrong page, superseded by the
walk-forward edge every engine page now carries.)
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
from core.semantics import valuation_state
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


def _classify_state(fvo: float) -> tuple[str, str, str]:
    """Map the valuation oscillator to (tier class, chip tone, label).

    Describes WHERE the target sat versus its macro-implied level — cheap or
    rich — not what happened next; the forward tiles carry the realised outcome.
    The word comes from ``core.semantics.valuation_state``, the one classifier
    the hero and the Mūla page use too: this page used to cut at ±1σ while the
    hero cut at ±0.5σ, so a −0.7σ day was "cheap" on Overview and "Fair" here.
    """
    state = valuation_state(fvo)
    return {
        "VERY CHEAP": ("tier-strong-buy", "success", "Very cheap"),
        "CHEAP": ("tier-buy", "success-outline", "Cheap"),
        "RICH": ("tier-caution", "danger-outline", "Rich"),
        "VERY RICH": ("tier-caution", "danger", "Very rich"),
    }.get(state, ("tier-hold", "neutral", "Fair"))


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
    tier_cls, chip_tone, badge_label = _classify_state(fvo)
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
        title="Precedent — What Followed States Like Today's",
        description=(f"The most similar distinct historical states of the system (Mahalanobis on "
                     f"Mūla's valuation, its window agreement, stress, confidence, momentum and "
                     f"volatility) and what {active_target} did next, across "
                     f"{'/'.join(str(h) for h in display_hold)}d."),
        icon="compass",
        accent="accent",
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
    from analytics.analogs import MIN_EFFECTIVE_N as _MIN_N
    summary = summarize_forward(periods, display_hold)
    _notes: list[str] = []
    if summary:
        cols = st.columns(len(summary), gap="small")
        for col, (h, s) in zip(cols, summary.items()):
            n_raw = int(s.get("n", 0) or 0)
            n_eff = float(s.get("n_eff", n_raw) or 0.0)
            # The SAME usability rule the hero's precedent gate applies, so this
            # page cannot print a confident green base rate the Overview just
            # called "too few effective analogs to read".
            usable = bool(s.get("usable", True)) and n_eff >= _MIN_N
            if not usable and s.get("note"):
                _notes.append(f"+{h}d: {s['note']}")
            with col:
                render_metric_card(
                    label=f"+{h}D Median Return",
                    value=f"{s['median']:+.1f}%",
                    subtext=(f"{s['positive_pct']:.0f}% rose · ≈{n_eff:.0f} effective of {n_raw}"
                             if usable else f"insufficient precedent · ≈{n_eff:.0f} effective"),
                    color_class=("neutral" if not usable else
                                 "success" if s["median"] > 0 else "danger"),
                )
    for _n in dict.fromkeys(_notes):
        render_note(_n)

    render_interpretation_card(
        title="A Base Rate, Conditioned on the System's Own State",
        body=(
            "What " + html_mod.escape(active_target) + " actually did after the most "
            "similar past states — no model is fitted to returns to produce it. The state is "
            "the engines' own reading, so this asks what <strong>this</strong> reading has "
            "historically preceded; it is one gate of the conviction chain, read with Mūla's "
            "call rather than against it. Every analog is a distinct episode (a Theiler "
            "window keeps adjacent days of one episode out); the \u201ceffective\u201d "
            "count discounts further for how unevenly the closest of them are weighted."
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
        render_chart_panel(_fig_ts, "analog_skill_term_structure", units="Spearman IC", windowed=False)

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
