"""
Tattva — Swayam tab: the Swayam confirmation sizer, its grade and its evidence.
तत्त्व (Tattva) — "Principle / Essence"

UI — Swayam is a SIZER: its published lean, aligned with the claim, scales the
call by exp(β·z), and β is earned by its own grade (×1.00 until it is). What
it reads is a short-window macro-residual breadth: a bank of MSF + MMR views
of the target's price, the ones that can vote anchored by a short rolling
regression to the same macro panel Mūla values against, every view counted
once.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    what does the sizer do to the call?  The sizer row, as the verdict reads it
  2 TRUST     has Swayam's read paid?              The claim's payoff by contradicts / split / confirms
  3 ANCHOR    how has the breadth moved?           Net breadth over time, carried sessions marked
  4 STATE     what is the breadth made of?         The view bank · the standalone lean's grade
  5 DETAIL    the evidence behind it               Per-view drill-down

The unit everywhere is a VIEW — one timescale × information-set × mechanism
read of a single price series. Views of one price are correlated by
construction, which is why the effective view count is shown beside the bank.

Sign convention, as everywhere in Tattva (core/semantics.py): the oscillator
reads positive when a view sees the instrument stretched UP; the lean that
implies is contrarian, so OVERSOLD views are bullish and the published
``Lean`` (net breadth, in points) is bullish-positive. Every number here is
read off the pipeline — the verdict, ``call_ts``, ``edges`` and Swayam's
published frame; nothing is re-derived or re-thresholded on this page. All
of it is the sizer's input or its evidence, so all of it is neutral-toned:
directional colour belongs to the claim's state and the call. Each section
and KPI names its card (``core.roles.CARDS``), whose chip states its role.
"""

from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import UI_CHART_HEIGHT_LARGE
from core.knobs import knob
from core.semantics import SWAYAM_DEAD, edge_tier
from ui.theme import chart_layout, style_axes, chart_color, chart_rgba, grid_rgba
from ui.components import (render_kpi_strip, render_section_header, render_empty_state,
                           render_chart_panel, render_table_panel, render_edge_panel,
                           render_gate_rows, edge_summary_text, window_cutoff)

#: The sizer's three words, in the order the payoff table reads them.
_WORDS = ("contradicts", "split", "confirms")

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "reading": (
        "Swayam's published lean on d*: the share of its views reading the instrument "
        "oversold minus the share reading it overbought, in points. Positive = its views see "
        "it stretched down, the bullish reading under Tattva's contrarian convention. A "
        "carried session repeats an earlier reading and is never a new one."
    ),
    "z": (
        "The sizer's input: z = d·Lean, the lean aligned with the claim's direction d, "
        "clipped to ±1. Positive = Swayam confirms the claim, negative = it contradicts it. "
        "0 on a carried or absent session, or when there is no claim to scale."
    ),
    "effect": (
        "The factor the sizer applies: exp(β·z). β = f(t_post) is 0 until the sizer's "
        "posterior t clears the bar, so an unskilled sizer is exactly ×1.00. A factor is "
        "positive: it scales the claim and can never veto or invert it."
    ),
    "n_eff": (
        "How many uncorrelated views the bank amounts to: the eigenvalue participation ratio "
        "of the views' correlation over the full history. A dozen views of one price are "
        "worth far fewer than a dozen votes."
    ),
}


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ═══════════════════════════════════════════════════════════════════════
#  1. THE SIZER ROW
# ═══════════════════════════════════════════════════════════════════════

def _render_sizer_row(verdict: dict, edge: dict | None, lean_edge: dict | None) -> None:
    """The Swayam sizer on d*, as the verdict's chain row reads it — Overview's numbers.

    The chain row first (the same reading, grade and effect Overview prints),
    then what it is made of: the reading in points (a lean, so with its own
    grade as a lean), native or carried, z, the own as-of grade and the class
    prior, t_post, and the factor.
    """
    row = next((r for r in verdict.get("chain") or [] if r["key"] == "sizer:swayam"), None)
    call_ts = st.session_state.get("call_ts")
    asof = verdict.get("asof")
    render_section_header(
        "Swayam confirmation — short-window macro-residual breadth",
        ("The sizer scales the claim by exp(β·z), z = d·Lean on a native row; it can never "
         "veto or invert the claim." if row else
         "No published valuation yet, so there is no claim for Swayam to scale."),
        icon="activity", accent="cyan", card="swayam:sizer",
    )
    if row is None:
        return
    render_gate_rows(verdict, ["sizer:swayam"])
    if call_ts is None or asof is None or asof not in call_ts.index:
        return
    r = call_ts.loc[asof]
    lean, native = _num(r["swayam_lean"]), bool(r["swayam_native"])
    z, t = _num(r["swayam_z"]), _num(r["swayam_t"])
    beta, factor = _num(r["swayam_beta"]) or 0.0, _num(r["swayam_factor"]) or 1.0
    g = (edge or {}).get("asof_grade") or {}
    ic, se, n_win = _num(g.get("ic")), _num(g.get("se")), int(_num(g.get("n_win")) or 0)
    mu, sd, scope = _num(g.get("prior_mu")), _num(g.get("prior_sd")), str(g.get("prior_scope") or "none")
    has_prior = scope not in ("none", "ungraded") and mu is not None and sd is not None
    prior_label = (st.session_state.get("priors_meta") or {}).get("label", "no pooled prior")
    # The newest native view-bank session at or before d* (pipeline.run.build_swayam).
    last = st.session_state.get("swayam_native_last")
    since = f"the {last:%d %b %Y} session" if last is not None else "an earlier session"
    sizer, grade = "swayam:sizer", "swayam:sizer_grade"
    render_kpi_strip([
        {"label": "Reading on d*", "value": f"{lean * 100:+.0f} pts" if lean is not None else "—",
         "subtext": ("a native row" if native else
                     f"carried from {since}: ×1.00, not graded" if lean is not None
                     else "no Swayam row on d*: ×1.00, nothing filled in"),
         "card": sizer, "lean": True, "edge": lean_edge or {}, "tooltip": TOOLTIPS["reading"]},
        {"label": "z = d·Lean", "value": f"{z:+.2f}" if z is not None else "—",
         "subtext": "the claim-aligned reading", "card": sizer, "tooltip": TOOLTIPS["z"]},
        {"label": "Own grade · as of d*",
         "value": f"{ic:+.3f}" if ic is not None else "warming up",
         "subtext": (f"± {se:.3f} · {n_win} windows" if ic is not None and se is not None
                     else f"{n_win} of {knob('GRADE_WARMUP_WIN')} windows"), "card": grade},
        {"label": "Prior", "value": scope if has_prior else "none",
         "subtext": f"{mu:+.3f} ± {sd:.3f}" if has_prior else prior_label, "card": grade},
        {"label": "t_post", "value": f"{t:.2f}" if t is not None else "ungraded",
         "subtext": edge_tier(t)[0], "card": grade},
        {"label": "Effect", "value": f"×{factor:.2f}",
         "subtext": (f"β {beta:.2f}; opens at t_post {knob('STRENGTH_T0'):g}, full at "
                     f"{knob('STRENGTH_T1'):g}"), "card": sizer, "tooltip": TOOLTIPS["effect"]},
    ], max_cols=6, key="swayam-sizer")


# ═══════════════════════════════════════════════════════════════════════
#  2. GRADE → sizer:swayam — the payoff by what Swayam said
# ═══════════════════════════════════════════════════════════════════════

def _render_buckets(edge: dict | None, horizon: int) -> None:
    """The claim's drift-neutral payoff by contradicts / split / confirms (edges' buckets)."""
    render_section_header(
        "Does Swayam's Read Pay?",
        f"The claim's drift-neutral {horizon}-session payoff from the next close, on its "
        f"sessions with a native Swayam row, split by what Swayam said: "
        f"claim-aligned net breadth below −{SWAYAM_DEAD * 100:.0f} pts contradicts, above "
        f"+{SWAYAM_DEAD * 100:.0f} confirms, and between is split. Confirmation has paid when "
        f"the confirms row pays more than the contradicts row; the sizer's grade ranks the "
        f"same reading.",
        icon="trending", accent="cyan", card="swayam:buckets",
    )
    buckets = (edge or {}).get("buckets") or {}
    if not any((buckets.get(w) or {}).get("n") for w in _WORDS):
        render_empty_state(
            "Not yet graded",
            "No claim session has a native Swayam row with a resolved outcome yet.",
            eyebrow="Swayam",
        )
        return
    rows = []
    for w in _WORDS:
        b = buckets.get(w) or {}
        m, se = _num(b.get("mean")), _num(b.get("se"))
        rows.append({
            "Swayam": w, "Sessions": int(b.get("n") or 0), "Windows (n/h)": int(b.get("n_win") or 0),
            "Mean payoff": f"{m * 100:+.2f}%" if m is not None else "—",
            "± SE": f"{se * 100:.2f}%" if se is not None else "—",
        })
    render_table_panel(pd.DataFrame(rows), "swayam-buckets", units=f"drift-neutral {horizon}D payoff",
                       windowed=False, label_col="Swayam", max_height=160,
                       footer=html.escape(edge_summary_text(
                           edge, "the Swayam sizer's z (d·Lean) on claim sessions")))


# ═══════════════════════════════════════════════════════════════════════
#  3–5. EVIDENCE
# ═══════════════════════════════════════════════════════════════════════

def _render_breadth_chart(df_n: pd.DataFrame) -> None:
    """Net breadth (top, the published Lean in pts) over the split it is made of.

    Evidence, so neutral-toned: the net keeps its native sign on a one-colour
    fill, and the two shares are told apart by line, not by a trade colour.
    """
    dates = df_n.index
    net = df_n["Lean"].to_numpy(dtype=float) * 100.0
    carried = ~df_n["_Native"].astype(bool).to_numpy()
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], shared_xaxes=True,
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=dates, y=net, fill="tozeroy", fillcolor=chart_rgba("cyan", 0.10),
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=net, mode="lines", name="Net breadth",
                             line=dict(color=chart_color("cyan"), width=1.4),
                             hovertemplate="%{y:+.0f} pts<extra>Net breadth</extra>"), row=1, col=1)
    if carried.any():
        fig.add_trace(go.Scatter(x=dates[carried], y=net[carried], mode="markers",
                                 name="Carried (not a new reading)",
                                 marker=dict(symbol="x-thin", size=7, color=chart_color("slate"),
                                             line=dict(width=1.2, color=chart_color("slate"))),
                                 hovertemplate="%{y:+.0f} pts carried<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_n["Oversold_Pct"], mode="lines", name="Views oversold",
                             line=dict(color=chart_color("cyan"), width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_n["Overbought_Pct"], mode="lines", name="Views overbought",
                             line=dict(color=chart_color("slate"), width=1.2, dash="dot")),
                  row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE + 40))
    style_axes(fig, y_title="Net (pts)", y_range=[-100, 100], row=1, col=1)
    style_axes(fig, y_title="% of views", y_range=[0, 100], row=2, col=1)
    render_chart_panel(fig, "swayam_breadth", units="pts · % of views", window=True)


def _render_bank(cfg, row: pd.Series | None, n_eff: float | None) -> None:
    """What the breadth is made of: who can vote, how they are weighted, how many they are worth.

    The counts are the published ones (``aggregate_views``' ``N_Voting`` and
    ``Total_Analyzed``) on d*'s Swayam row, so the page counts the views the
    breadth was actually made of.
    """
    n_vote = int(row["N_Voting"]) if row is not None and pd.notna(row["N_Voting"]) else 0
    n_run = int(row["Total_Analyzed"]) if row is not None and pd.notna(row["Total_Analyzed"]) else 0
    lengths = sorted(cfg.swayam_lengths)
    mid = lengths[len(lengths) // 2]
    card = "swayam:bank"
    render_kpi_strip([
        {"label": "Views that can vote", "value": f"{n_vote} of {n_run}",
         "subtext": (f"on d*, of the views that reported, the ones that can reach the "
                     f"±{cfg.swayam_overbought:g} vote: the macro-anchored views; a price-only "
                     f"view cannot at the shipped blend weight, and sits in the shares' "
                     f"denominator"), "card": card},
        {"label": "Weighting", "value": "equal",
         "subtext": (f"every view counted once; L{mid}·MOM / STRUCT / FLOW, which duplicated "
                     f"L{mid}·FULL's vote, removed"), "card": card},
        {"label": "Effective views", "value": f"≈{n_eff:.1f}" if n_eff else "—",
         "subtext": f"of the {n_run} views run here, over the full history", "card": card,
         "tooltip": TOOLTIPS["n_eff"]},
    ], max_cols=3, key="swayam-bank")


def _render_individual_views(view_dfs: dict) -> None:
    """Per-view oscillators and condition, in bank order (the order the views were run)."""
    if not view_dfs:
        return
    keys = list(view_dfs)
    with st.container(key="filterbar"):
        sym = st.selectbox("View", keys, key="swayam_sym_select")
    if sym and sym in view_dfs:
        cdf = view_dfs[sym].iloc[-60:]
        cols = [c for c in ("Close", "Unified_Osc", "MSF_Osc", "MMR_Osc", "Condition")
                if c in cdf.columns]
        render_table_panel(
            cdf[cols], "swayam-view-drilldown",
            units="last 60 sessions", windowed=False, index_label="Date", max_rows=60, max_height=520,
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_swayam_tab(verdict: dict, edges: dict, cfg, *, selected_tf: str | None,
                      horizon: int) -> None:
    """Swayam tab — the sizer row, its graded audit, then the evidence behind it.

    The sizer row opens the page whatever the target: one with no OHLCV
    (a sheet series such as Jeera) has no Swayam reading, and its row says
    ×1.00 before the page says why there is nothing more to show.
    """
    swayam_daily = st.session_state.get("swayam_daily")
    view_dfs = st.session_state.get("swayam_view_dfs") or {}
    edges = edges or {}
    _render_sizer_row(verdict, edges.get("sizer:swayam"), edges.get("evidence:swayam_lean"))
    if swayam_daily is None or swayam_daily.empty:
        render_empty_state(
            "No Swayam breadth for this target",
            "Swayam reads the target's own OHLCV, and none was available for it this run "
            "(sheet-sourced targets have no OHLCV feed). The sizer is ×1.00.",
            eyebrow="Swayam",
        )
        return

    _render_buckets(edges.get("sizer:swayam"), horizon)

    full = swayam_daily[~swayam_daily.index.duplicated(keep="last")]
    df_n = full
    _cut = window_cutoff(pd.to_datetime(full.index).max(), selected_tf)
    if _cut is not None:
        df_n = full[pd.to_datetime(full.index) >= _cut]
    render_section_header(
        "Breadth Over Time",
        "Top: net breadth, the published lean in points, bullish-positive; × marks a carried "
        "session (the bars ended before the panel did, so the row repeats an earlier reading "
        "and the sizer leaves it at ×1.00). Bottom: the shares of views oversold and "
        "overbought it is made of.",
        icon="layers", accent="cyan", card="swayam:breadth",
    )
    _render_breadth_chart(df_n)

    render_section_header(
        "The View Bank",
        "Which views can vote, how they are counted, and how many uncorrelated views they "
        "amount to.",
        icon="grid", accent="cyan", card="swayam:bank",
    )
    asof = verdict.get("asof")
    _render_bank(cfg, full.loc[asof] if asof is not None and asof in full.index else full.iloc[-1],
                 st.session_state.get("swayam_n_eff"))

    render_section_header(
        "Swayam's Lean, Standalone",
        f"When the published lean leaned one way on a valued session with a native row, did "
        f"price move that way over the next {horizon} sessions from the next close, net of "
        f"drift? It audits the sizer's input; the sizer's own grade is the claim-aligned one "
        f"above.",
        icon="trending", accent="cyan", card="swayam:lean",
    )
    render_edge_panel(edges.get("evidence:swayam_lean"), "swayam-lean",
                      what="Swayam's published lean on its own", title="Lean grade",
                      card="swayam:lean")

    render_section_header(
        "Individual Views",
        "Per view, the unified oscillator its vote is cast on, its MSF and MMR halves and the "
        "condition it voted. Oscillators keep their native sign (positive = stretched up, the "
        "bearish reading under the contrarian convention), in the neutral ink, as on the chart.",
        icon="database", card="swayam:views",
    )
    _render_individual_views(view_dfs)
