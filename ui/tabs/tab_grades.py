"""
Tattva — Grades tab: is the loop working?
तत्त्व (Tattva) — "Principle / Essence"

UI — every grade the call is built from or audited by, in one place, read off
the pipeline: ``edges`` (each object graded over the full history),
``grades_ts`` (the as-of grades, shrunk to the class prior, that the call read
at each session) and ``call_ts`` (what trust and each β were at each
session), then the pooled-prior study's own results, read-only, from the
bundle's ``priors_meta``. Nothing is estimated here.

It replaces the Convergence page. Its question — do the two engines agree, and
has their agreement paid? — is answered on the issued object: the Swayam
sizer's grade and its contradicts / split / confirms payoff (the Swayam page),
and the call's own grade on the sessions it was issued (this one).

Reading order — the house convention every analysis tab follows:

  1 ANSWER    every grade, one table               call · claim · two sizers · evidence
  2 ANCHOR    what has each been allowed to do?    t_post at every session; trust and β
  3 TRUST     does a larger size pay more?         Issued-call payoff by size tercile
  4 RECORD    what did the pooled study find?      R4 · decisions A/B/C · R5 power · arms E/P/F

One grader (core.grade, run in the pipeline) on one target — the drift-neutral
h-session return from the next close — on the call's own decision sample. A
role object (the claim, a sizer) is labelled on its posterior t as of d*, the
number that set its trust or β; the call and the evidence on their own t. Each
row's name is its role's title in ``core.roles``, so the table words every
object as the registry does.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import UI_CHART_HEIGHT_LARGE
from core.instruments import asset_class_name
from core.knobs import knob
from core.roles import ROLES
from core.semantics import edge_tier
from ui.theme import chart_layout, style_axes, chart_color, chart_rgba, grid_rgba
from ui.components import (class_pool, render_section_header, render_table_panel,
                           render_chart_panel, render_empty_state, render_kpi_strip,
                           render_note, render_size_terciles)

#: The table's rows, in reading order: ``(edges key, ROLES key, row name)``;
#: a name of None is the role's own title. The call first, then what it is
#: made of (the claim, the two sizers), then the evidence that audits them.
_ROWS: tuple[tuple[str, str, str | None], ...] = (
    ("call", "grade:call", None),
    ("call:prospective", "grade:call:prospective", None),
    ("claim", "grade:claim", None),
    ("sizer:swayam", "grade:sizer:swayam", None),
    ("sizer:reversion", "grade:sizer:reversion", None),
    ("claim:buy", "evidence:claim_side", "The claim, CHEAP side"),
    ("claim:sell", "evidence:claim_side", "The claim, RICH side"),
    ("evidence:swayam_lean", "evidence:swayam_lean", None),
    ("evidence:ecm_drift", "evidence:ecm_drift", None),
)

#: The paths drawn in section 2: ``(call_ts column, legend name, colour)``.
_T_PATHS = (("claim_t", "Claim", "violet"), ("swayam_t", "Swayam", "cyan"),
            ("reversion_t", "Reversion", "slate"))
_SIZE_PATHS = (("trust", "Trust (claim)", "violet"), ("swayam_beta", "β Swayam", "cyan"),
               ("reversion_beta", "β Reversion", "slate"))

#: The study's objects, in the order it pools them.
_OBJECTS = ("claim", "sizer:swayam", "sizer:reversion", "call")


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x) -> str:
    v = _num(x)
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _f(x, fmt: str = "+.3f") -> str:
    v = _num(x)
    return format(v, fmt) if v is not None else "—"


# ═══════════════════════════════════════════════════════════════════════
#  1. EVERY GRADE, ONE TABLE
# ═══════════════════════════════════════════════════════════════════════

def _class(asset_class: str | None) -> str:
    """An asset class as the page names it (the key stays in the data)."""
    return asset_class_name(asset_class) if asset_class else "—"


def _pool_text(pool: dict | None, asset_class: str | None) -> str:
    if not pool:
        return "none"
    return (f"class-pooled · {_class(asset_class)} · k {int(pool['k'])} · "
            f"{pool['mu']:+.3f} ± {pool['se_mu']:.3f} · vintage {pool.get('asof') or '—'}")


def _row(key: str, role_key: str, name: str | None, edges: dict, grades_ts: dict, d_star,
         priors_meta: dict, asset_class: str | None) -> dict:
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
        vintage = pd.Timestamp(a["prior_vintage"]) if pd.notna(a.get("prior_vintage")) else None
        prior = (f"{scope} · k {int(a['prior_k'])} · {mu:+.3f} ± {sd:.3f} · vintage "
                 f"{f'{vintage:%Y-%m-%d}' if vintage is not None else '—'}"
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
        prior = _pool_text(class_pool(priors_meta, asset_class, "call"), asset_class)
        if not e.get("graded"):
            label = str(e.get("status") or "ungraded")
    elif key == "call:prospective" and not e.get("n_win"):
        label = str(e.get("status") or "no prospective record")
    elif role.kind == "EVIDENCE" and key.startswith("claim:"):
        label = "one side: read hit vs base"
    return {
        "Object": name or role.title, "Role": f"{role.kind} → {role.target}",
        "Own IC": f"{ic:+.3f}" if ic is not None else "—",
        "± SE": f"{se:.3f}" if se is not None else "—",
        "n_win": int(e.get("n_win") or 0),
        "Hit vs base": f"{hit:.0f}% vs {base:.0f}%" if hit is not None and base is not None else "—",
        "Payoff": _pct(e.get("payoff")),
        "Prior": prior, "t_post": t_post, "Trust / β": power, "Label": label,
        "Consumer": role.consumer,
    }


def _render_table(edges: dict, grades_ts: dict, d_star, horizon: int, priors_meta: dict,
                  asset_class: str | None) -> None:
    render_section_header(
        "Every Grade",
        f"Each object's rank IC against the drift-neutral {horizon}-session return from the "
        f"next close, on non-overlapping windows over the full history, with its hit against "
        f"the side-matched no-skill rate and its mean drift-neutral payoff. For the claim and "
        f"each sizer, the prior and the posterior t it read on d* — what set its trust or β. "
        f"Priors: {priors_meta.get('label', 'none')}.",
        icon="grid", accent="accent", card="grades:table",
    )
    rows = [_row(k, rk, name, edges, grades_ts, d_star, priors_meta, asset_class)
            for k, rk, name in _ROWS if k in edges]
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
        "Top: the posterior t each role object carried at every session, against the bar it "
        "must clear. Bottom: what that bought — the claim's trust and each sizer's β, both "
        "f(t_post) — as the composer read them, session by session.",
        icon="trending", accent="accent", card="grades:paths",
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
#  4. THE POOLED-PRIOR STUDY, READ-ONLY
# ═══════════════════════════════════════════════════════════════════════

def _render_decisions(pm: dict) -> None:
    """The study's provenance and its three decisions, as it recorded them."""
    d = pm.get("decisions") or {}
    a, b, c = d.get("A") or {}, d.get("B") or {}, d.get("C") or {}
    db = b.get("delta_ic") or {}
    card = "grades:study"
    render_kpi_strip([
        {"label": "Study", "value": str(pm.get("run_date") or "—"),
         "subtext": (f"{pm.get('study') or '—'} · {pm.get('mode') or 'network'} · "
                     f"{len(pm.get('members') or [])} members · "
                     f"{len((pm.get('held_out') or {}).get('members') or [])} held out"),
         "card": card},
        {"label": "Decision A · claim centring", "value": str(a.get("value", "—")),
         "subtext": (f"raw fails C12 on {len(a.get('raw_fails') or [])}, the candidate passes "
                     f"on {len(a.get('candidate_passes') or [])}; recorded, never applied"),
         "card": card},
        {"label": "Decision B · SIZERS_ENABLED", "value": str(b.get("value", "—")),
         "subtext": (f"ΔIC(d·∏factor − d) {_f(db.get('delta'))} ± {_f(db.get('se'), '.3f')} "
                     f"on {int(db.get('n_members') or 0)} held-out members; released iff ≥ −1 SE"),
         "card": card},
        {"label": "Decision C · TIER_RULE", "value": str(c.get("value", "—")),
         "subtext": (f"pooled tercile payoff {'rises' if c.get('monotone') else 'does not rise'}; "
                     f"{'every' if c.get('enough_windows') else 'not every'} tercile has n/h ≥ "
                     f"{knob('TIER_MIN_WIN')}"), "card": card},
    ], max_cols=4, key="grades-study")


def _render_checks(pm: dict) -> None:
    """R4 (the SE's calibration, an abort check) and R5 (power, recorded)."""
    checks = pm.get("checks") or {}
    r4 = checks.get("R4") or {}
    rows = [{"Object": obj, "Median FFT / brute": _f(r.get("median"), ".3f"),
             "Band": " – ".join(f"{x:g}" for x in r.get("band") or []) or "—",
             "Members": int(r.get("n") or 0), "Pass": "pass" if r.get("pass") else "FAIL"}
            for obj, r in r4.items()]
    if rows:
        render_table_panel(pd.DataFrame(rows), "grades-r4",
                           context="catalogue · R4 · shift-null SE calibration",
                           windowed=False, label_col="Object", max_height=180)
    r5 = checks.get("R5") or {}
    rows = [{"Object": obj, "Class": _class(cls), "k": int(r.get("k") or 0),
             "n/h (median)": _f(r.get("n_win_median"), ".0f"),
             "ρ (mean)": _f(r.get("rho_mean"), ".2f"),
             "τ": _f(r.get("tau"), ".3f"), "False-open": _f(r.get("false_open"), ".3f"),
             "E[f] null": _f(r.get("e_strength_null"), ".3f"),
             "P(open) at 0.05": _f(r.get("power_0.05"), ".2f"),
             "P(open) at 0.08": _f(r.get("power_0.08"), ".2f"), "Scope": str(r.get("scope") or "—")}
            for obj in _OBJECTS for cls, r in (r5.get(obj) or {}).items()]
    if rows:
        render_table_panel(pd.DataFrame(rows), "grades-r5",
                           context="catalogue · R5 · one look's false-open rate and power per class",
                           windowed=False, label_col="Object", max_height=320)


def _render_arms(pm: dict) -> None:
    """Arm E (what the execution lag removes), arm P (path ranges), arm F (prospective)."""
    arms = pm.get("arms") or {}
    e = (arms.get("E") or {}).get("per_class") or {}
    rows = [{"Class": _class(cls), "Object": obj, "μ at L = 0": _f((r.get("L0") or {}).get("mu")),
             "μ at L = 1": _f((r.get("L1") or {}).get("mu")),
             "k": int((r.get("L1") or r.get("L0") or {}).get("k") or 0)}
            for cls, by_obj in e.items() for obj, r in by_obj.items()]
    if rows:
        render_table_panel(pd.DataFrame(rows), "grades-arm-e",
                           context="catalogue · arm E · class-pooled grade at lag 0 and lag 1",
                           windowed=False, label_col="Class", max_height=320)
    p = arms.get("P") or {}
    rows = [{"Member": m, "Base claim IC": _f(r.get("base_ic")),
             "Range low": _f((r.get("range") or [None, None])[0]),
             "Range high": _f((r.get("range") or [None, None])[1]),
             "Jackknife draws": len(r.get("jackknife") or [])} for m, r in p.items()]
    if rows:
        render_table_panel(pd.DataFrame(rows), "grades-arm-p",
                           context="catalogue · arm P · start shifts and panel jackknife",
                           windowed=False, label_col="Member", max_height=260)
    f = arms.get("F") or {}
    per = f.get("per_member") or {}
    if per:
        rows = [{"Member": m, "Issued sessions": int(r.get("issued_n") or 0),
                 "n_win": int(r.get("n_win") or 0), "IC": _f(r.get("ic")),
                 "± SE": _f(r.get("se"), ".3f"),
                 "Hit vs base": (f"{r['hit']:.0f}% vs {r['base']:.0f}%"
                                 if _num(r.get("hit")) is not None and _num(r.get("base")) is not None
                                 else "—")} for m, r in per.items()]
        render_table_panel(pd.DataFrame(rows), "grades-arm-f",
                           context="catalogue · arm F · prospective record",
                           windowed=False, label_col="Member", max_height=260)
    render_note(f"Arm F: {f.get('status') or 'no prospective record'}.")


def _render_study(priors_meta: dict) -> None:
    """The pooled-prior study's results, read-only from the bundle; or that it has not run."""
    render_section_header(
        "The Pooled-Prior Study",
        "The committed catalogue study every posterior shrinks toward, read as data: its "
        "decisions, the calibration of the SE it pooled (R4), the power each class has (R5), "
        "what the execution lag removes (arm E), how far a member's start moves its grade "
        "(arm P) and the prospective record (arm F). Nothing here is recomputed.",
        icon="layers", accent="accent", card="grades:study",
    )
    pm = priors_meta or {}
    if pm.get("status") != "loaded":
        render_empty_state(
            "study not yet run",
            f"No pooled-prior file is loaded for this horizon and lag "
            f"({pm.get('label') or 'no pooled prior'}): every grade above is the instrument's "
            f"own. <strong>studies/pooled_priors.py</strong> writes it.",
            eyebrow="Pooled-prior study",
        )
        return
    _render_decisions(pm)
    _render_checks(pm)
    _render_arms(pm)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def render_grades_tab(edges: dict, grades_ts: dict, call_ts: pd.DataFrame | None, verdict: dict,
                      *, horizon: int, priors_meta: dict) -> None:
    """Grades — every grade, the trust and β paths, the size terciles, the study's results."""
    edges, grades_ts, priors_meta = edges or {}, grades_ts or {}, priors_meta or {}
    d_star = verdict.get("asof")
    asset_class = ((edges.get("call") or {}).get("attrs") or {}).get("asset_class")
    _render_table(edges, grades_ts, d_star, horizon, priors_meta, asset_class)
    if call_ts is not None and not call_ts.empty:
        _render_paths(call_ts)
    render_size_terciles(edges.get("call"), horizon)
    _render_study(priors_meta)
