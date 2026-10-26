"""
Tattva — Mūla tab: the one claim, its grade, its evidence, and the reversion sizer.
तत्त्व (Tattva) — "Principle / Essence"

UI — Mūla makes the one CLAIM Tattva issues: price against the fair-value
level the macro cross-section implies, in units of the gap's own typical size.
Its trust is earned only by its graded price-side skill, and its
error-correction layer feeds the one other sizer that rides on the claim,
Reversion (ECM). This page shows each of those in the role it plays.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    what does the call read off Mūla?   The chain rows: claim · preconditions · trust
  2 TRUST     has the claim paid on price?        GRADE→claim: t_post path · own vs prior · periods · sides
  3 ANCHOR    what is the claim made on?          EVIDENCE→claim: who closes the gap · price & fair value ·
                                                  the claim's unit · gap stationarity · model fit · drivers
  4 SIGNAL    what does the ECM do to the call?   SIZER Reversion · EVIDENCE→sizer:reversion

Every number is read off the pipeline — the verdict (``call_ts`` at d*),
``edges``, ``evidence`` and Mūla's published frame. Nothing is estimated,
thresholded or re-derived here, and evidence is neutral-toned with its own
grade beside it: directional colour is reserved for the claim's state.
"""

from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.knobs import knob
from core.semantics import CLAIM_SIGMA, DEEP_SIGMA, edge_tier, valuation_state
from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (
    render_kpi_strip,
    render_section_header,
    render_sub_header,
    render_chart_panel,
    render_table_panel,
    render_empty_state,
    render_edge_panel,
    render_gate_rows,
    render_note,
    provisional_note,
)
from core.config import UI_CHART_HEIGHT_MEDIUM, UI_CHART_HEIGHT_XLARGE

#: The evidence chip every EVIDENCE section carries (design §11).
_EVIDENCE = "EVIDENCE → {target} · does not move the call."

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "fair_value": (
        "The price level the whole traded cross-section implies today, from a recursive "
        "cointegrating regression of log price on the integrated factors of ~200 instruments. "
        "The band is where price itself would be expected 95% of the time if it traded at "
        "fair value plus normal noise."
    ),
    "no_claim": (
        f"Inside ±{CLAIM_SIGMA:g}σ of fair value is the normal day-to-day noise of price around "
        f"it and no claim is made. σ is the gap's own typical size today, so the band in "
        f"percent of price moves with it."
    ),
    "gap_half_life": (
        "How long the gap itself persists, however it closes: the online AR(1) of the gap. "
        "On most instruments fair value moving toward price does most of the closing, so "
        "this is not the speed price converges at."
    ),
    "price_half_life": (
        "How long price alone would take to close half the gap if fair value stood still: "
        "ln 2 / −ln(1 − κ̂), from the error-correction layer's price-side κ̂. Undefined when "
        "κ̂ ≤ 0 (price has not been moving toward fair value)."
    ),
    "reading": (
        "MRProb, the error-correction layer's evidence that PRICE responds to the gap: "
        "P(κ̂ > 0) under the coefficient's own uncertainty, where κ̂ is the share of the gap "
        "price closes per session. 0.5 carries no information."
    ),
    "z": (
        "The sizer's input: z = 2·MRProb − 1, unsigned because evidence that price responds "
        "to the gap supports either side of the claim. 0 when there is no claim to size."
    ),
    "effect": (
        "The factor the sizer applies: exp(β·z). β = f(t_post) is 0 until the sizer's "
        "posterior t clears the bar, so an unskilled sizer is exactly ×1.00. A factor is "
        "positive: it scales the claim and can never veto or invert it."
    ),
    "ecm_move": (
        "The price move over the horizon that today's gap implies under the error-correction "
        "layer: b·z·Σφᵏ, with the regression's intercept (the drift) deliberately left out, "
        "so it is not the expected move. Shown beside its own grade."
    ),
}


def _num(d, key=None):
    """A finite float (``d[key]``, or ``d`` itself), else None — never a silent 0."""
    try:
        v = float(d.get(key) if key is not None else d)
    except (TypeError, ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def _state_tone(state: str) -> str:
    return "success" if "CHEAP" in state else "danger" if "RICH" in state else "neutral"


def _grade_sub(e: dict | None) -> str:
    """'± 0.075 · t 1.06 · n/h 136 · unproven' for an edge, or why it has none."""
    e = e or {}
    se, t = _num(e, "se"), _num(e, "t")
    if _num(e, "ic") is None or se is None:
        return str(e.get("status") or "not yet graded")
    return f"± {se:.3f} · t {t:.2f} · n/h {int(e.get('n_win') or 0)} · {edge_tier(t)[0]}"


def _row_at(frame: pd.DataFrame | None, asof):
    """d*'s row of a date-indexed frame (call_ts, Mūla's fvo_ts), or None."""
    if frame is None or asof is None or asof not in frame.index:
        return None
    return frame.loc[asof]


# ═══════════════════════════════════════════════════════════════════════
#  2 · GRADE → claim
# ═══════════════════════════════════════════════════════════════════════

def _render_t_path(call_ts: pd.DataFrame | None, since) -> None:
    """The claim's as-of posterior t at every session, against the bars f opens and fills at."""
    if call_ts is None or call_ts.empty or "claim_t" not in call_ts.columns:
        return
    ct = call_ts if since is None else call_ts[call_ts.index >= since]
    t = pd.to_numeric(ct["claim_t"], errors="coerce")
    if t.notna().sum() == 0:
        render_empty_state(
            "No as-of grade yet",
            f"The claim's as-of grade appears once {knob('GRADE_WARMUP_WIN')} independent "
            f"windows have resolved in this window.",
            eyebrow="GRADE → claim",
        )
        return
    t0, t1 = knob("STRENGTH_T0"), knob("STRENGTH_T1")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ct.index, y=t, mode="lines", name="Claim t_post",
                             line=dict(color=chart_color("violet"), width=1.4),
                             hovertemplate="t_post %{y:.2f}<extra></extra>"))
    for y, dash in ((t0, "dot"), (t1, "dash")):
        fig.add_hline(y=y, line_dash=dash, line_color=chart_rgba("slate", 0.55), line_width=0.8)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title="t_post")
    render_chart_panel(fig, "mula_t_path", units="posterior t, as of each session",
                       footer=html.escape(f"Trust opens at t_post {t0:g} (dotted) and is full at "
                                          f"{t1:g} (dashed); below {t0:g} the claim is a HOLD."))


def _render_own_vs_prior(edge: dict | None, row, verdict: dict) -> None:
    """What trust read at d*: the own as-of grade, the class prior, and their combination."""
    g = (edge or {}).get("asof_grade") or {}
    ic, se, n_win = _num(g, "ic"), _num(g, "se"), int(_num(g, "n_win") or 0)
    mu, sd, scope = _num(g, "prior_mu"), _num(g, "prior_sd"), str(g.get("prior_scope") or "none")
    has_prior = scope not in ("none", "ungraded") and mu is not None and sd is not None
    w_own = _num(row, "claim_w_own") if row is not None else None
    t = verdict.get("claim_t")
    prior_label = (st.session_state.get("priors_meta") or {}).get("label", "no pooled prior")
    render_kpi_strip([
        {"label": "Own grade · as of d*", "value": f"{ic:+.3f}" if ic is not None else "warming up",
         "subtext": (f"± {se:.3f} · {n_win} windows" if ic is not None and se is not None
                     else f"{n_win} of {knob('GRADE_WARMUP_WIN')} windows"),
         "color_class": "neutral"},
        {"label": "Class prior", "value": scope if has_prior else "none",
         "subtext": f"{mu:+.3f} ± {sd:.3f}" if has_prior else prior_label, "color_class": "neutral"},
        {"label": "Weight on own", "value": f"{w_own:.2f}" if w_own is not None else "—",
         "subtext": "1.00 with no prior; the rest is the class's", "color_class": "neutral"},
        {"label": "t_post", "value": f"{t:.2f}" if t is not None else "ungraded",
         "subtext": edge_tier(t)[0], "color_class": "neutral"},
        {"label": "Trust", "value": f"{verdict['trust']:.2f}",
         "subtext": (f"f(t_post): 0 below {knob('STRENGTH_T0'):g}, full at "
                     f"{knob('STRENGTH_T1'):g}"), "color_class": "neutral"},
    ], max_cols=5, key="mula-own-prior")


def _render_sides(edges: dict) -> None:
    """The claim by side: hit against the side-matched no-skill rate, and payoff."""
    def _side(e):
        hit, base, pay = _num(e, "hit"), _num(e, "base"), _num(e, "payoff")
        value = f"{hit:.0f}% vs {base:.0f}%" if hit is not None and base is not None else "—"
        sub = (f"drift-neutral payoff {pay * 100:+.2f}% · n/h {int(e.get('n_win') or 0)}"
               if pay is not None else "not yet graded")
        return value, sub

    items = []
    for key, name in (("claim:buy", "BUY claims hit"), ("claim:sell", "SELL claims hit")):
        value, sub = _side(edges.get(key) or {})
        items.append({"label": name, "value": value,
                      "subtext": f"vs the side-matched no-skill rate · {sub}",
                      "color_class": "neutral"})
    render_kpi_strip(items, max_cols=2, key="mula-sides")


# ═══════════════════════════════════════════════════════════════════════
#  3 · EVIDENCE → claim
# ═══════════════════════════════════════════════════════════════════════

def _render_gap_split(evidence: dict, mula, horizon: int) -> None:
    """Who closes the gap, and the gap's two half-lives (d*'s published row)."""
    gs = evidence.get("gap_split") or {}
    share, se_share = _num(gs, "price_share"), _num(gs, "se_share")
    b_fv, se_fv = _num(gs, "b_fv"), _num(gs, "se_b_fv")
    gap_hl, price_hl = _num(mula, "GapHalfLife"), _num(mula, "MulaHalfLife")
    render_kpi_strip([
        {"label": "Price's share", "value": f"{share * 100:+.0f}%" if share is not None else "—",
         "subtext": (f"± {se_share * 100:.0f}% · of the gap closed in {horizon}D · "
                     f"{gs.get('n_win', 0)} windows" if se_share is not None else "not measured"),
         "color_class": "neutral"},
        {"label": "Fair value toward price", "value": f"{b_fv:+.2f}" if b_fv is not None else "—",
         "subtext": (f"± {se_fv:.2f} of each unit of gap, over {horizon}D" if se_fv is not None
                     else f"of each unit of gap, over {horizon}D"), "color_class": "neutral"},
        {"label": "Gap half-life", "value": f"{gap_hl:.0f}d" if gap_hl is not None else "—",
         "subtext": "the gap's own persistence, however it closes", "color_class": "neutral",
         "tooltip": TOOLTIPS["gap_half_life"]},
        {"label": "Price-side half-life",
         "value": f"{price_hl:.0f}d" if price_hl is not None else "—",
         "subtext": ("price alone, if fair value stood still" if price_hl is not None
                     else "undefined: price-side κ̂ ≤ 0"), "color_class": "neutral",
         "tooltip": TOOLTIPS["price_half_life"]},
    ], max_cols=4, key="mula-gap-split")
    if share is not None and se_share is not None:
        render_note(
            f"While the gap was open, price closed {share:.0%} (± {se_share:.0%}) of it within "
            f"{horizon} sessions; fair value moving toward price did the rest. Tattva therefore "
            f"does not assume price does the closing: it acts only when the claim's price-side "
            f"grade clears the bar.")


def _render_fair_value_kpis(mula) -> None:
    """The claim's reading at d*: the gap, the level, and the no-claim band in price terms."""
    fvo, pct = _num(mula, "FVO"), _num(mula, "PctMispricing")
    fv, gs = _num(mula, "FairValue"), _num(mula, "GapScale")
    lo, hi = _num(mula, "FairValueLo"), _num(mula, "FairValueHi")
    state = valuation_state(fvo)
    render_kpi_strip([
        {"label": "Valuation", "value": state if fvo is not None else "—",
         "subtext": (f"{fvo:+.2f}σ · {abs(pct) * 100:.1f}% {'below' if pct < 0 else 'above'} fair value"
                     if fvo is not None and pct is not None else "no published valuation"),
         "color_class": _state_tone(state)},
        {"label": "Fair Value", "value": f"{fv:,.2f}" if fv is not None else "—",
         "subtext": (f"95% band {lo:,.2f} – {hi:,.2f}" if lo is not None and hi is not None
                     else "level the world implies"),
         "color_class": "neutral", "tooltip": TOOLTIPS["fair_value"]},
        {"label": "No-claim band", "value": f"±{CLAIM_SIGMA:g}σ",
         "subtext": (f"±{CLAIM_SIGMA * gs * 100:.1f}% of fair value at today's gap scale"
                     if gs is not None else "the gap's own typical size"),
         "color_class": "neutral", "tooltip": TOOLTIPS["no_claim"]},
    ], max_cols=3, key="mula-read")


def _render_fair_value_chart(ts_filtered, x_axis, active_target):
    """Price against fair value (top) and the oscillator the claim is made on (bottom).

    The bottom pane is the OSCILLATOR — the gap in units of its own typical
    size — against the ±CLAIM_SIGMA no-claim band and the ±DEEP_SIGMA deep
    lines. (It used to be the raw log gap with an OU projection of it closing,
    drawn as if price would do the closing; the gap split above measures who
    does.)
    """
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
    # Price is the instrument — the primary series — so it is drawn in full ink.
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Price"], mode="lines", name=f"{active_target}",
        line=dict(color=chart_color("accent"), width=1.6),
        hovertemplate="%{y:,.2f}<extra>Price</extra>",
    ), row=1, col=1)

    # ── Bottom: the oscillator, against the bands the claim is made on ──
    osc = pd.to_numeric(ts_filtered["FVO"], errors="coerce")
    gap_pct = pd.to_numeric(ts_filtered["PctMispricing"], errors="coerce") * 100.0
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


def _render_claim_unit(evidence: dict) -> None:
    """The claim's unit against the C12 contract, and Decision A's status."""
    cu = evidence.get("claim_unit") or {}
    c = cu.get("contract") or {}
    mean, sd, rc, sell = (_num(cu, k) for k in ("mean", "sd", "rich_cheap", "sell_share"))

    def _holds(k):
        return "holds" if c.get(k) else "fails"

    render_kpi_strip([
        {"label": "Mean", "value": f"{mean:+.2f}σ" if mean is not None else "—",
         "subtext": f"contract |mean| < 0.15 · {_holds('mean')}", "color_class": "neutral"},
        {"label": "SD", "value": f"{sd:.2f}" if sd is not None else "—",
         "subtext": f"contract 0.8 – 1.2 · {_holds('sd')}", "color_class": "neutral"},
        {"label": "RICH : CHEAP", "value": f"{rc:.1f}" if rc is not None else "—",
         "subtext": f"contract 0.67 – 1.5 · {_holds('rich_cheap')}", "color_class": "neutral"},
        {"label": "SELL share", "value": f"{sell:.0%}" if sell is not None else "—",
         "subtext": f"of {int(cu.get('n') or 0)} valued sessions' claims", "color_class": "neutral"},
    ], max_cols=4, key="mula-claim-unit")
    render_note(
        f"C12 contract {'holds' if cu.get('passes') else 'fails'} on the valued sessions through "
        f"d*. Decision A (a trailing-year centring of the claim) is pre-registered with the pooled-"
        f"prior study and reads '{knob('CLAIM_CENTRING')}': the study records it, and nothing "
        f"applies it without an owner-signed change.")


def _render_gap_stationarity(evidence: dict) -> None:
    """The one gap-stationarity word, from ADF and KPSS on the published gap."""
    gt = evidence.get("gap_stationarity") or {}
    adf, kp = _num(gt, "adf_pvalue"), _num(gt, "kpss_pvalue")
    render_kpi_strip([
        {"label": "Full-history diagnostic", "value": str(gt.get("state") or "—"),
         "subtext": f"{int(gt.get('n') or 0)} valued sessions through d*", "color_class": "neutral"},
        {"label": "ADF p-value", "value": f"{adf:.3f}" if adf is not None else "—",
         "subtext": (f"null: a unit root; rejected below {knob('STATIONARITY_ALPHA'):g}"),
         "color_class": "neutral"},
        {"label": "KPSS p-value", "value": f"{kp:.3f}" if kp is not None else "—",
         "subtext": (f"null: a constant level; rejected below {knob('STATIONARITY_ALPHA'):g} "
                     f"(reported within 0.01 – 0.10)"), "color_class": "neutral"},
    ], max_cols=3, key="mula-gap-tests")


def _render_model_fit(model_stats: dict, mula) -> None:
    """What the fit is made of and how closely it tracks the level — facts, no advice."""
    r2 = _num(model_stats, "r2_oos")
    n_inst = int(model_stats.get("n_features", 0) or 0)
    n_blocks = int(model_stats.get("n_blocks", 0) or 0)
    k = int(_num(mula, "KFactors") or 0)
    sp = _num(mula, "ModelSpread")
    render_kpi_strip([
        {"label": "OOS R²", "value": f"{r2:.3f}" if r2 is not None else "—",
         "subtext": "log-price variance the one-step fitted level tracks", "color_class": "neutral"},
        {"label": "Cross-section", "value": f"{n_inst} instruments",
         "subtext": f"in {n_blocks} asset-class blocks · {k} latent factors on d*",
         "color_class": "neutral"},
        {"label": "Model spread", "value": f"{sp * 1e4:.0f} bps" if sp is not None else "—",
         "subtext": "SD of the fair-value level itself on d*", "color_class": "neutral"},
    ], max_cols=3, key="mula-model-fit")


def _render_drivers(engine) -> None:
    """Which asset-class blocks the fitted level is made of (the engine's own decomposition)."""
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
    style_axes(fig, x_title="Share of the fitted level %")
    fig.update_yaxes(showgrid=False)
    render_chart_panel(fig, "mula_drivers", units="share of the fitted level", windowed=False,
                       footer=f"{len(labels)} of {len(impacts)} asset-class blocks by current share.")
    hist = engine.get_feature_impact_history()
    if hist is not None and not hist.empty:
        render_sub_header("Share over time")
        render_table_panel(hist.tail(10), "mula-drivers-history", units="every 21 sessions",
                           windowed=False, index_label="Date", max_rows=10, max_height=240)


# ═══════════════════════════════════════════════════════════════════════
#  4 · SIZER Reversion · EVIDENCE → sizer:reversion
# ═══════════════════════════════════════════════════════════════════════

def _render_reversion_sizer(verdict: dict, edge: dict | None, row) -> None:
    """The reversion sizer on d*, as the verdict's chain row reads it — Overview's numbers."""
    chain = next((r for r in verdict.get("chain") or [] if r["key"] == "sizer:reversion"), None)
    render_section_header(
        "Reversion (ECM) — P(price-side κ > 0)",
        ("SIZER → call. The sizer scales the claim by exp(β·z), z = 2·MRProb − 1, and is ×1.00 "
         "until its own grade earns a weight; it can never veto or invert the claim, and a "
         "grade measured against it leaves it neutral." if chain else
         "SIZER → call. No published valuation yet, so there is no claim to scale."),
        icon="activity", accent="violet",
    )
    if chain is None:
        return
    render_gate_rows(verdict, ["sizer:reversion"])
    if row is None:
        return
    mr, kappa = _num(row, "mr_prob"), _num(row, "kappa")
    z, t = _num(row, "reversion_z"), _num(row, "reversion_t")
    beta, factor = _num(row, "reversion_beta") or 0.0, _num(row, "reversion_factor") or 1.0
    g = (edge or {}).get("asof_grade") or {}
    ic, se, n_win = _num(g, "ic"), _num(g, "se"), int(_num(g, "n_win") or 0)
    mu, sd, scope = _num(g, "prior_mu"), _num(g, "prior_sd"), str(g.get("prior_scope") or "none")
    has_prior = scope not in ("none", "ungraded") and mu is not None and sd is not None
    prior_label = (st.session_state.get("priors_meta") or {}).get("label", "no pooled prior")
    render_kpi_strip([
        {"label": "Reading on d*", "value": f"{mr:.2f}" if mr is not None else "—",
         "subtext": (f"MRProb · κ̂ {kappa:+.4f}/d" if kappa is not None
                     else "the error-correction layer is warming up"),
         "color_class": "neutral", "tooltip": TOOLTIPS["reading"]},
        {"label": "z = 2·MRProb − 1", "value": f"{z:+.2f}" if z is not None else "—",
         "subtext": ("the reading the sizer scales by" if verdict.get("direction")
                     else "0: no claim to size"), "color_class": "neutral",
         "tooltip": TOOLTIPS["z"]},
        {"label": "Own grade · as of d*",
         "value": f"{ic:+.3f}" if ic is not None else "warming up",
         "subtext": (f"± {se:.3f} · {n_win} windows" if ic is not None and se is not None
                     else f"{n_win} of {knob('GRADE_WARMUP_WIN')} windows"),
         "color_class": "neutral"},
        {"label": "Prior", "value": scope if has_prior else "none",
         "subtext": f"{mu:+.3f} ± {sd:.3f}" if has_prior else prior_label, "color_class": "neutral"},
        {"label": "t_post", "value": f"{t:.2f}" if t is not None else "ungraded",
         "subtext": edge_tier(t)[0], "color_class": "neutral"},
        {"label": "Effect", "value": f"×{factor:.2f}",
         "subtext": (f"β {beta:.2f}; opens at t_post {knob('STRENGTH_T0'):g}, full at "
                     f"{knob('STRENGTH_T1'):g}"),
         "color_class": "neutral", "tooltip": TOOLTIPS["effect"]},
    ], max_cols=6, key="mula-reversion-sizer")


def _render_ecm_move(evidence: dict, edge: dict | None, ts_filtered, x_axis, horizon: int) -> None:
    """The ECM's gap-implied move (drift excluded), neutral, beside its own grade."""
    em = evidence.get("ecm_drift") or {}
    label = str(em.get("label") or f"gap-implied {horizon}D move (excludes drift)")
    v, sd = _num(em, "value"), _num(em, "sd")
    ic = _num(edge, "ic") if edge else None
    render_kpi_strip([
        {"label": label, "value": f"{v:+.2f}%" if v is not None else "—",
         "subtext": (f"± {sd:.2f}% (1 SD, {horizon} sessions) on d*" if sd is not None
                     else "the error-correction layer is warming up"),
         "color_class": "neutral", "tooltip": TOOLTIPS["ecm_move"]},
        {"label": "Its grade · IC", "value": f"{ic:+.3f}" if ic is not None else "—",
         "subtext": _grade_sub(edge), "color_class": "neutral"},
    ], max_cols=2, key="mula-ecm-move")
    if "MulaDriftPct" not in ts_filtered.columns:
        return
    d = pd.to_numeric(ts_filtered["MulaDriftPct"], errors="coerce")
    band = pd.to_numeric(ts_filtered["MulaSdPct"], errors="coerce")
    if d.notna().sum() == 0:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=d + band, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_axis, y=d - band, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=chart_rgba("slate", 0.10),
                             name="± 1 SD", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_axis, y=d, mode="lines", name=label,
                             line=dict(color=chart_color("slate"), width=1.4),
                             hovertemplate="%{y:+.2f}%<extra>gap-implied move</extra>"))
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM))
    style_axes(fig, y_title=f"{horizon}D move %")
    render_chart_panel(fig, "mula_ecm", units="% over the horizon, drift excluded")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_mula_tab(engine, ts_filtered, x_axis, signal, model_stats, active_target, *,
                    verdict: dict, edges: dict, evidence: dict, call_ts: pd.DataFrame | None,
                    horizon: int):
    """Mūla tab — the claim and what it has earned, its evidence, the reversion sizer."""
    edges, evidence = edges or {}, evidence or {}
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

    # ── 1 · the chain rows Overview reads off Mūla ──────────────────────
    render_section_header("Mūla — The Claim", claim, icon="target", accent="violet")
    render_gate_rows(verdict, ["claim", "precondition", "trust"])

    # ── 2 · GRADE → claim ───────────────────────────────────────────────
    asof = verdict.get("asof")
    row = _row_at(call_ts, asof)
    since = (pd.to_datetime(x_axis).min() if x_axis is not None and len(x_axis) else None)
    render_section_header(
        "Has the Claim Paid on Price?",
        f"GRADE → claim. The claim's sign against the drift-neutral {horizon}-session return from "
        f"the next close, on its claim sessions: its posterior t as of every session (what trust "
        f"read), its own grade beside the class prior at d*, period by period over the full "
        f"history, and by side.",
        icon="trending", accent="violet",
    )
    _render_t_path(call_ts, since)
    _render_own_vs_prior(edges.get("claim"), row, verdict)
    render_edge_panel(edges.get("claim"), "mula", what="the claim's sign on its claim sessions")
    rule = (edges.get("claim") or {}).get("stated_rule")
    if rule:
        render_note(f"Stated rule (C12): {rule}.")
    _render_sides(edges)

    # ── 3 · EVIDENCE → claim ────────────────────────────────────────────
    ev = _EVIDENCE.format(target="claim")
    render_section_header(
        "Who Closes the Gap?",
        f"{ev} When Mūla's gap was open, how much of its closing over the next {horizon} sessions "
        f"did price do, and how much did fair value do by moving toward price? Beside it, the "
        f"gap's own half-life and price's.",
        icon="layers", accent="violet",
    )
    mula = _row_at(st.session_state.get("fvo_ts"), signal.get("asof"))
    _render_gap_split(evidence, mula, horizon)

    render_section_header(
        "Price & Fair Value",
        f"{ev} Top: price against the fair-value level and its 95% band. Bottom: the oscillator "
        f"the claim is made on — inside the shaded ±{CLAIM_SIGMA:g}σ band no claim is made; beyond "
        f"±{DEEP_SIGMA:g}σ (dotted) is a deep mispricing. Below zero (green) = cheap.",
        icon="trending", accent="violet",
    )
    _render_fair_value_kpis(mula)
    _render_fair_value_chart(ts_filtered, x_axis, active_target)

    render_section_header(
        "The Claim's Unit",
        f"{ev} Is the oscillator in the unit its ±{CLAIM_SIGMA:g}σ and ±{DEEP_SIGMA:g}σ cuts "
        f"assume? Centred, unit SD, RICH and CHEAP claims balanced (charter C12), on the valued "
        f"sessions through d*.",
        icon="crosshair", accent="violet",
    )
    _render_claim_unit(evidence)

    render_section_header(
        "Gap Stationarity",
        f"{ev} ADF (null: a unit root) and KPSS (null: a constant level) on the published gap, "
        f"read into one word. One test of the whole record, not an as-of series, and no "
        f"precondition of the call.",
        icon="crosshair", accent="violet",
    )
    _render_gap_stationarity(evidence)

    render_section_header(
        "Model Fit",
        f"{ev} What the valuation is fitted on and how closely its level tracks log price.",
        icon="cpu", accent="violet",
    )
    _render_model_fit(model_stats, mula)

    render_section_header(
        "What Drives Fair Value",
        f"{ev} Each asset-class block's share of the fitted fair-value level: the engine's own "
        f"exact decomposition of the level into the instruments it is made of, summed by block "
        f"— not a refit and not a coefficient read-off.",
        icon="bar-chart", accent="violet",
    )
    _render_drivers(engine)

    # ── 4 · SIZER Reversion · EVIDENCE → sizer:reversion ────────────────
    _render_reversion_sizer(verdict, edges.get("sizer:reversion"), row)
    em_label = str((evidence.get("ecm_drift") or {}).get("label")
                   or f"gap-implied {horizon}D move (excludes drift)")
    render_section_header(
        em_label[:1].upper() + em_label[1:],
        f"{_EVIDENCE.format(target='sizer:reversion')} The move over the next {horizon} sessions "
        f"today's gap implies under the error-correction layer, with its drift left out: the "
        f"reversion sizer's mechanism in price units, neutral-toned, graded on its own on the "
        f"valued sessions (drift-neutral).",
        icon="activity", accent="violet",
    )
    _render_ecm_move(evidence, edges.get("evidence:ecm_drift"), ts_filtered, x_axis, horizon)
