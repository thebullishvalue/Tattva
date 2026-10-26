"""
Tattva — Grades tab: is the loop working?
तत्त्व (Tattva) — "Principle / Essence"

UI — every grade the call is built from or audited by, in one place, read off
the pipeline: ``edges`` (each object graded over the full history),
``grades_ts`` (the as-of grades, shrunk to the class prior, that the call read
at each session) and ``call_ts`` (what trust and each β were at each
session). Nothing is estimated here.

It replaces the Convergence page. Its question — do the two engines agree, and
has their agreement paid? — is answered on the issued object: the Swayam
sizer's grade and its contradicts / split / confirms payoff (the Swayam page),
and the call's own grade on the sessions it was issued (this one).

Reading order — the house convention every analysis tab follows:

  1 ANSWER    every grade, one table               call · claim · two sizers · evidence
  2 ANCHOR    what has each been allowed to do?    t_post at every session; trust and β
  3 TRUST     does a bigger size pay more?         Issued-call payoff by size tercile

One grader (core.grade.series_edge) on one target — the drift-neutral
h-session return from the next close — on the call's own decision sample. A
role object (the claim, a sizer) is labelled on its posterior t as of d*, the
number that set its trust or β; the call and the evidence on their own t.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import UI_CHART_HEIGHT_LARGE
from core.knobs import knob
from core.roles import ROLES
from core.semantics import edge_tier
from ui.theme import chart_layout, style_axes, chart_color, chart_rgba, grid_rgba
from ui.components import (render_section_header, render_table_panel, render_chart_panel,
                           render_empty_state, render_note)

#: The table's rows, in reading order: ``(edges key, ROLES key, row name)``.
#: The call first, then what it is made of (the claim, the two sizers), then
#: the evidence that audits them.
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("call", "grade:call", "The call, as issued"),
    ("call:prospective", "grade:call:prospective", "The call, prospective"),
    ("claim", "grade:claim", "Mūla claim"),
    ("sizer:swayam", "grade:sizer:swayam", "Swayam confirmation"),
    ("sizer:reversion", "grade:sizer:reversion", "Reversion (ECM)"),
    ("claim:buy", "evidence:claim_side", "The claim, BUY side"),
    ("claim:sell", "evidence:claim_side", "The claim, SELL side"),
    ("evidence:swayam_lean", "evidence:swayam_lean", "Swayam's lean, standalone"),
)

#: The paths drawn in section 2: ``(call_ts column, legend name, colour)``.
_T_PATHS = (("claim_t", "Claim", "violet"), ("swayam_t", "Swayam", "cyan"),
            ("reversion_t", "Reversion", "amber"))
_SIZE_PATHS = (("trust", "Trust (claim)", "violet"), ("swayam_beta", "β Swayam", "cyan"),
               ("reversion_beta", "β Reversion", "amber"))


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x) -> str:
    v = _num(x)
    return f"{v * 100:+.2f}%" if v is not None else "—"


# ═══════════════════════════════════════════════════════════════════════
#  1. EVERY GRADE, ONE TABLE
# ═══════════════════════════════════════════════════════════════════════

def _row(key: str, role_key: str, name: str, edges: dict, grades_ts: dict, d_star,
         verdict: dict) -> dict:
    """One table row: the full-history grade, and for a role object what it read at d*."""
    e = edges.get(key) or {}
    role = ROLES[role_key]
    ic, se, t = _num(e.get("ic")), _num(e.get("se")), _num(e.get("t"))
    hit, base = _num(e.get("hit")), _num(e.get("base"))
    prior = t_post = power = "—"
    label = edge_tier(t)[0]
    g = grades_ts.get(key)
    if g is not None and d_star is not None and d_star in g.index:
        a = g.loc[d_star]
        mu, sd = _num(a["prior_mu"]), _num(a["prior_sd"])
        scope = str(a["prior_scope"])
        prior = (f"{scope} · k {int(a['prior_k'])} · {mu:+.3f} ± {sd:.3f}"
                 if scope != "none" and mu is not None and sd is not None
                 else "none")
        tp = _num(a["t_post"])
        t_post = f"{tp:.2f}" if tp is not None else "ungraded"
        label = edge_tier(tp)[0]
        if key == "claim":
            power = f"trust {float(a['strength']):.2f}"
        else:
            power = f"β {_num(e.get('beta')) or 0.0:.2f}"
    if key == "call":
        prior = verdict["call_grade"]["pooled"]
        if not e.get("graded"):
            label = str(e.get("status") or "ungraded")
    elif key == "call:prospective" and not e.get("n_win"):
        label = str(e.get("status") or "no prospective record")
    elif role.kind == "EVIDENCE" and key.startswith("claim:"):
        label = "one side: read hit vs base"
    return {
        "Object": name, "Role": f"{role.kind} → {role.target}",
        "Own IC": f"{ic:+.3f}" if ic is not None else "—",
        "± SE": f"{se:.3f}" if se is not None else "—",
        "n_win": int(e.get("n_win") or 0),
        "Hit vs base": f"{hit:.0f}% vs {base:.0f}%" if hit is not None and base is not None else "—",
        "Payoff": _pct(e.get("payoff")),
        "Prior": prior, "t_post": t_post, "Trust / β": power, "Label": label,
        "Consumer": role.consumer,
    }


def _render_table(edges: dict, grades_ts: dict, d_star, verdict: dict, horizon: int,
                  priors_meta: dict) -> None:
    render_section_header(
        "Every Grade",
        f"GRADE. Each object's rank IC against the drift-neutral {horizon}-session return from "
        f"the next close, on non-overlapping windows over the full history, with its hit "
        f"against the side-matched no-skill rate and its mean drift-neutral payoff. For the "
        f"claim and each sizer, the prior and the posterior t it read on d* — what set its "
        f"trust or β. Priors: {priors_meta.get('label', 'none')}.",
        icon="grid", accent="accent",
    )
    rows = [_row(k, rk, name, edges, grades_ts, d_star, verdict) for k, rk, name in _ROWS
            if k in edges]
    if not rows:
        render_empty_state("Nothing graded yet", "The run produced no grades.", eyebrow="Grades")
        return
    render_table_panel(pd.DataFrame(rows), "grades-table", units=f"non-overlapping {horizon}D",
                       windowed=False, label_col="Object", max_height=320)
    render_note(f"|t| below 1 is never called an edge; a role object opens at t_post "
                f"{knob('STRENGTH_T0'):g} and is at full strength by {knob('STRENGTH_T1'):g}. "
                f"A grade measured against its lean leaves the object neutral — nothing is "
                f"ever inverted on its grade.")


# ═══════════════════════════════════════════════════════════════════════
#  2. THE TRUST AND β PATHS
# ═══════════════════════════════════════════════════════════════════════

def _render_paths(call_ts: pd.DataFrame) -> None:
    render_section_header(
        "What Each Has Been Allowed to Do",
        "GRADE. Top: the posterior t each role object carried at every session, against the "
        "bar it must clear. Bottom: what that bought — the claim's trust and each sizer's β, "
        "both f(t_post) — as the call read them, session by session.",
        icon="trending", accent="accent",
    )
    t0, t1 = knob("STRENGTH_T0"), knob("STRENGTH_T1")
    dates = call_ts.index
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], shared_xaxes=True,
                        vertical_spacing=0.07)
    for col, name, colour in _T_PATHS:
        fig.add_trace(go.Scatter(x=dates, y=call_ts[col], mode="lines", name=f"t_post {name}",
                                 line=dict(color=chart_color(colour), width=1.3),
                                 hovertemplate="%{y:.2f}<extra>" + name + "</extra>"),
                      row=1, col=1)
    for y, dash in ((t0, "dot"), (t1, "dash")):
        fig.add_hline(y=y, line_color=chart_rgba("slate", 0.55), line_width=0.7, line_dash=dash,
                      row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6, row=1, col=1)
    for col, name, colour in _SIZE_PATHS:
        fig.add_trace(go.Scatter(x=dates, y=call_ts[col], mode="lines", name=name,
                                 line=dict(color=chart_color(colour), width=1.3),
                                 hovertemplate="%{y:.2f}<extra>" + name + "</extra>"),
                      row=2, col=1)
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_LARGE + 60))
    style_axes(fig, y_title="t_post", row=1, col=1)
    style_axes(fig, y_title="trust · β", y_range=[-0.05, 1.05], row=2, col=1)
    render_chart_panel(fig, "grades-paths", units="t_post · trust and β", windowed=False,
                       footer=(f"Dotted: t_post {t0:g}, where trust and β open. Dashed: {t1:g}, "
                               f"where they are full. Blank stretches are warm-up: fewer than "
                               f"{knob('GRADE_WARMUP_WIN')} resolved windows."))


# ═══════════════════════════════════════════════════════════════════════
#  3. PAYOFF BY SIZE TERCILE
# ═══════════════════════════════════════════════════════════════════════

def _render_terciles(edge: dict | None, horizon: int) -> None:
    render_section_header(
        "Does a Bigger Size Pay More?",
        f"GRADE → call. The issued calls split by their size's causal tercile (each against the "
        f"issued sizes up to its own session), with the claim's mean drift-neutral "
        f"{horizon}-session payoff in each, and the claim sessions it held beside them.",
        icon="bar-chart", accent="accent",
    )
    e = edge or {}
    terc, ivh = e.get("terciles") or {}, e.get("issued_vs_held") or {}
    if not int(e.get("issued_n") or 0):
        render_empty_state(
            "Nothing issued yet",
            "No call has been issued on this history, so there is no size to split: every "
            "claim session was held (its payoff is in the table above).",
            eyebrow="Grades",
        )
        return
    rows = []
    for name, b in (("Smallest third", terc.get("T1")), ("Middle third", terc.get("T2")),
                    ("Largest third", terc.get("T3")), ("All issued", ivh.get("issued")),
                    ("Claim held (not issued)", ivh.get("held"))):
        b = b or {}
        rows.append({"Sessions": name, "n": int(b.get("n") or 0),
                     "Windows (n/h)": int(b.get("n_win") or 0),
                     "Mean payoff": _pct(b.get("mean")),
                     "± SE": f"{b['se'] * 100:.2f}%" if _num(b.get("se")) is not None else "—"})
    render_table_panel(pd.DataFrame(rows), "grades-terciles", units=f"drift-neutral {horizon}D payoff",
                       windowed=False, label_col="Sessions", max_height=220)
    render_note(f"Decision C: the size carries no tier word (TIER_RULE "
                f"'{knob('TIER_RULE')}') until the pooled study shows this payoff rising with "
                f"size, with at least {knob('TIER_MIN_WIN')} windows in every tercile.")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_grades_tab(edges: dict, grades_ts: dict, call_ts: pd.DataFrame | None, verdict: dict,
                      *, horizon: int, priors_meta: dict) -> None:
    """Grades — every grade in one table, the trust and β paths, the size terciles."""
    edges, grades_ts = edges or {}, grades_ts or {}
    d_star = verdict.get("asof")
    _render_table(edges, grades_ts, d_star, verdict, horizon, priors_meta or {})
    if call_ts is not None and not call_ts.empty:
        _render_paths(call_ts)
    _render_terciles(edges.get("call"), horizon)
