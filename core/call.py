"""
Tattva — THE call: one claim, sized only by measured skill, composed at every session.
तत्त्व (Tattva) — "Principle / Essence"

    w_t = d_t · size_t,      size_t = trust_t · ∏_i exp(β_i,t · z_i,t)

on Y_t = log P(t+L+h) − log P(t+L): the h sessions from the first close a call
published after the close of t can trade at (L = EXECUTION_LAG).

* ``d_t`` — the claim, ``core.semantics.claim_direction(FVO_t)`` when the four
  preconditions hold, else 0. It is the only source of sign anywhere.

    P1 published — FVO finite and the session not withheld (NO CALL, with the
                   engine's reason);
    P2 printed   — the target printed at t and, at d*, is fewer than h
                   sessions behind (NO CALL);
    P3 tradeable — a valuation ratio is never called (NO CALL);
    P4 |FVO| ≥ CLAIM_SIGMA — otherwise NO CLAIM, inside the noise.

* ``trust_t = f(t_post of the claim, as of t)`` — the claim's EARNED base
  size. It is 0 until the claim's price-side grade clears STRENGTH_T0, and a
  claim with trust 0 is a HOLD whose reason names that grade. Trust is never
  the 'binding' factor: it is not something that limits the call, it is what
  the call has earned.
* each sizer i — its reading z_i, clipped to ±SIZER_Z_CLIP (absent → 0), at
  ``β_i = SIZER_BETA_MAX · f(t_post_i) · SIZERS_ENABLED``. An unskilled sizer
  (β = 0) or an absent reading is exactly ×1.00, and a factor is positive and
  unsigned, so a sizer scales the claim and can never veto or invert it.

:func:`compose_series` evaluates this at EVERY session from date-keyed inputs
and as-of grades, so the call shown today is one row of the same history that
is graded as issued (``pipeline.assemble.grade_all``). :func:`verdict_from_row`
reads d*'s row into the one object every page renders.

No statistics here: every grade arrives in ``grades_ts`` (pipeline.assemble,
from core.grade and core.trust), and this module is arithmetic and words on
it. No Streamlit dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.instruments import asset_class_name
from core.knobs import knob
from core.roles import CHAIN, ROLES, chain_roles
from core.semantics import CLAIM_SIGMA, claim_direction, edge_tier, swayam_word
from core.trust import strength

SIZER_BETA_MAX: float = knob("SIZER_BETA_MAX")
SIZER_Z_CLIP: float = knob("SIZER_Z_CLIP")
STRENGTH_T0: float = knob("STRENGTH_T0")
STRENGTH_T1: float = knob("STRENGTH_T1")
GRADE_WARMUP_WIN: int = knob("GRADE_WARMUP_WIN")


@dataclass(frozen=True)
class Sizer:
    """One sizer: its key (``sizer:<key>`` in ROLES and edges), its title and
    the inputs column that holds its reading z."""

    key: str
    title: str
    z_col: str


#: The two sizers. Their inputs are disjoint from the claim's.
SIZERS: tuple[Sizer, ...] = (
    Sizer("swayam", "Swayam confirmation", "z_swayam"),
    Sizer("reversion", "Reversion (ECM)", "z_reversion"),
)

#: The five actions, in the order they are decided.
ACTIONS: tuple[str, ...] = ("NO CALL", "NO CLAIM", "HOLD", "BUY", "SELL")

#: P3's reason. The only untradeable target is a valuation ratio (core.instruments).
NOT_TRADEABLE = "the target is a valuation ratio, not a traded price"

#: The raw readings behind each sizer's z, carried into call_ts so the verdict
#: (and the Data export) can name them.
_SIZER_READINGS: dict[str, tuple[str, ...]] = {
    "swayam": ("swayam_lean", "swayam_native"),
    "reversion": ("mr_prob", "kappa"),
}

#: call_ts's columns, in order (sizer blocks from SIZERS).
CALL_COLUMNS: tuple[str, ...] = (
    "published", "valid", "provisional", "printed", "behind", "precondition",
    "withheld_reason", "fvo", "pct", "d",
    "claim_ic", "claim_se", "claim_n_win", "claim_prior_mu", "claim_prior_sd",
    "claim_prior_scope", "claim_prior_k", "claim_w_own", "claim_t", "trust",
    *(c for s in SIZERS for c in (*_SIZER_READINGS[s.key],
                                  f"{s.key}_z", f"{s.key}_t", f"{s.key}_beta", f"{s.key}_factor")),
    "size", "w", "action", "reason", "binding", "grid_date",
)

#: grades_ts columns the claim block of call_ts copies (call_ts name ← grades_ts name).
_CLAIM_GRADE = (("claim_ic", "ic"), ("claim_se", "se"), ("claim_n_win", "n_win"),
                ("claim_prior_mu", "prior_mu"), ("claim_prior_sd", "prior_sd"),
                ("claim_prior_scope", "prior_scope"), ("claim_prior_k", "prior_k"),
                ("claim_w_own", "w_own"), ("claim_t", "t_post"))
_NO_PRIOR = ("none", "ungraded")


def _num(x) -> float | None:
    """A finite float, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _strengths(t: np.ndarray) -> np.ndarray:
    """THE map (core.trust.strength), element by element."""
    return np.array([strength(v) for v in t], dtype=np.float64)


def format_size(size) -> str:
    """The size as printed: a unitless index to two decimals, never a '%'."""
    v = _num(size)
    return "—" if v is None else f"{v:.2f}"


def edge_periods(edge: dict | None) -> tuple[int, int]:
    """``(positive, scored)`` periods of a grade.

    The denominator is the periods that had enough windows to score
    (``scored_blocks``), not every period drawn: a short history leaves its
    first blocks unscored, and dividing by all of them understated durability.
    """
    e = edge or {}
    blocks = [b for b in (e.get("blocks") or [])
              if b.get("ic") is not None and np.isfinite(b.get("ic"))]
    scored = int(e.get("scored_blocks", len(blocks)) or 0)
    return int(e.get("positive_blocks", 0) or 0), scored


# ═══════════════════════════════════════════════════════════════════════
#  The composer
# ═══════════════════════════════════════════════════════════════════════

def _claim_grade_text(ic, se, n_win, mu, sd, scope, k, *, asset_class: str | None = None,
                      study_date: str | None = None) -> str:
    """'own +0.070 ± 0.062, n_win 140; no pooled prior' — the claim grade in words.

    Given the instrument's ``asset_class`` key and the study's ``study_date``
    (the verdict's words), the prior is named as the pool it is, by the
    class's page name — 'Commodities class-LOO prior −0.001 ± 0.028, k 6,
    study 2026-10-27', or 'global-LOO prior …, study …' for a catalogue-wide
    pool. call_ts's ``reason`` keeps the short form.
    """
    ic, se, mu, sd = _num(ic), _num(se), _num(mu), _num(sd)
    n = int(_num(n_win) or 0)
    own = (f"own {ic:+.3f} ± {se:.3f}, n_win {n}" if ic is not None and se is not None
           else f"own grade warming up: {n} of {GRADE_WARMUP_WIN} windows")
    if str(scope) in _NO_PRIOR or mu is None or sd is None:
        return f"{own}; no pooled prior"
    pool = (f"{asset_class_name(asset_class)} {scope}"
            if asset_class and str(scope).startswith("class") else str(scope))
    study = f", study {study_date}" if study_date else ""
    return f"{own}; {pool} prior {mu:+.3f} ± {sd:.3f}, k {int(_num(k) or 0)}{study}"


def _reason(action: str, precondition: str, r: dict, h: int, graded: bool,
            text: str | None = None) -> str:
    """Why this session's action is what it is, in numbers.

    ``text`` is the claim grade in words; by default the short form call_ts
    stores (:func:`_claim_grade_text` without the pool's class and study).
    """
    if action == "NO CALL":
        if precondition == "published":
            return r["withheld_reason"] or "no valuation published"
        if precondition == "printed":
            b = int(r["behind"])
            return (f"{b} sessions behind: the {h}-session horizon has run" if b >= h
                    else "the target did not print this session")
        return NOT_TRADEABLE
    if action == "NO CLAIM":
        return f"|FVO| {abs(r['fvo']):.2f}σ is inside the ±{CLAIM_SIGMA:g}σ noise"
    if text is None:
        text = _claim_grade_text(r["claim_ic"], r["claim_se"], r["claim_n_win"],
                                 r["claim_prior_mu"], r["claim_prior_sd"], r["claim_prior_scope"],
                                 r["claim_prior_k"])
    t = _num(r["claim_t"])
    if action == "HOLD":
        if not graded:
            return "claim ungraded: its grade could not be computed"
        if t is None:
            return f"claim ungraded: {text}"
        return f"claim not shown to pay on price: t_post {t:.2f} does not clear {STRENGTH_T0:g} ({text})"
    return f"claim t_post {t:.2f} clears {STRENGTH_T0:g} ({text}): trust {r['trust']:.2f}"


def compose_series(inputs: pd.DataFrame, grades_ts: dict[str, pd.DataFrame], *,
                   tradeable: bool, sizers_enabled: bool) -> pd.DataFrame:
    """The call at every session: ``call_ts`` (see CALL_COLUMNS), one row per input date.

    ``inputs`` is ``pipeline.assemble.gather_inputs``'s frame (every column
    read by date; ``attrs['h']`` is THE horizon). ``grades_ts`` maps
    ``'claim'`` and ``'sizer:<key>'`` to their as-of grades on the same dates
    (``pipeline.assemble.grade_series``); a missing entry is a grade that
    failed: the claim is then 'ungraded' with trust 0 and its direction kept,
    a sizer gets β = 0.

    Pure and causal: row t reads only row t of each input, so it is unchanged
    by anything dated after t.
    """
    idx = inputs.index
    n = len(idx)
    h = int(inputs.attrs["h"])
    nan = np.full(n, np.nan)

    published = inputs["published"].to_numpy(dtype=bool)
    p2 = inputs["printed"].to_numpy(dtype=bool) & (inputs["behind"].to_numpy(dtype=np.int64) < h)
    p3 = np.full(n, bool(tradeable))
    claim = inputs["claim"].to_numpy(dtype=np.int64)
    called = published & p2 & p3
    precondition = np.select([~published, ~p2, ~p3, claim == 0],
                             ["published", "printed", "tradeable", "claim_sigma"], "")
    d = np.where(called, claim, 0)

    out = pd.DataFrame(index=idx)
    for c in ("published", "valid", "provisional", "printed", "behind"):
        out[c] = inputs[c].to_numpy()
    out["precondition"] = precondition
    for c in ("withheld_reason", "fvo", "pct"):
        out[c] = inputs[c].to_numpy()
    out["d"] = d

    # ── Trust: the claim's earned base size ──────────────────────────────
    g = (grades_ts or {}).get("claim")
    graded = g is not None
    g = g.reindex(idx) if graded else None
    for col, src in _CLAIM_GRADE:
        if graded:
            out[col] = g[src].to_numpy()
        else:
            out[col] = "ungraded" if col == "claim_prior_scope" else nan
    out["trust"] = _strengths(out["claim_t"].to_numpy(dtype=np.float64))
    trust = out["trust"].to_numpy()

    # ── The sizers: exp(β·z), exactly 1.0 when β = 0 or z is absent ─────
    factors = []
    for s in SIZERS:
        for c in _SIZER_READINGS[s.key]:
            out[c] = inputs[c].to_numpy()
        z = inputs[s.z_col].to_numpy(dtype=np.float64)
        # A sizer sizes a call: with no call (d = 0) there is nothing to scale.
        z = np.where(np.isfinite(z) & (d != 0), np.clip(z, -SIZER_Z_CLIP, SIZER_Z_CLIP), 0.0)
        gs = (grades_ts or {}).get(f"sizer:{s.key}")
        t = gs["t_post"].reindex(idx).to_numpy(dtype=np.float64) if gs is not None else nan
        beta = SIZER_BETA_MAX * _strengths(t) * (1.0 if sizers_enabled else 0.0)
        factor = np.exp(beta * z)
        out[f"{s.key}_z"], out[f"{s.key}_t"] = z, t
        out[f"{s.key}_beta"], out[f"{s.key}_factor"] = beta, factor
        factors.append(factor)

    size = np.where(d != 0, trust * np.prod(factors, axis=0), 0.0)
    out["size"] = size
    out["w"] = d * size
    action = np.select([~called, d == 0, trust <= 0, d > 0], list(ACTIONS[:4]), ACTIONS[4])
    out["action"] = action

    # Binding: the sizer with the smallest factor below 1 — never trust.
    fx = np.vstack(factors)
    low = fx.argmin(axis=0)
    keys = np.array([s.key for s in SIZERS])
    out["binding"] = np.where(d == 0, "", np.where(fx.min(axis=0) < 1.0, keys[low], "none"))

    rows = out.to_dict("records")
    out["reason"] = [_reason(a, p, r, h, graded) for a, p, r in zip(action, precondition, rows)]
    out["grid_date"] = g["grid_date"].to_numpy() if graded else pd.NaT
    out = out[list(CALL_COLUMNS)]
    out.attrs = {**inputs.attrs, "tradeable": bool(tradeable), "sizers_enabled": bool(sizers_enabled)}
    return out


# ═══════════════════════════════════════════════════════════════════════
#  The verdict: d*'s row, in words
# ═══════════════════════════════════════════════════════════════════════

def _chain_row(key: str, reading: str, grade: str, effect: str, detail: str) -> dict:
    role = ROLES[key]
    return {"role": role.kind, "key": key, "title": role.title, "reading": reading,
            "grade": grade, "effect": effect, "detail": detail}


def _gap_words(target: str, fvo: float, pct: float) -> str:
    side = "below" if pct < 0 else "above"
    return (f"{target} is {abs(fvo):.2f}σ ({abs(pct):.1f}%) {side} the level the macro "
            f"cross-section implies")


def _sizer_reading(key: str, r: dict, d: int) -> tuple[str, str]:
    """(reading, detail) of one sizer on d*'s row."""
    if key == "swayam":
        lean, native = _num(r["swayam_lean"]), bool(r["swayam_native"])
        z = r["swayam_z"]
        if lean is None:
            return "no Swayam read", "no Swayam row on d*: ×1.00, nothing is filled in"
        if not native:
            return ("carried: no native row on d*",
                    f"net breadth {lean * 100:+.0f} pts carried from an earlier session: a carried "
                    f"row is ×1.00 and is not graded")
        word = swayam_word(d * lean) if d else "no claim to confirm"
        return (f"net breadth {lean * 100:+.0f} pts · {word}",
                f"z = d·Lean = {z:+.2f} on a native row (±{SIZER_Z_CLIP:g} clip)")
    mr, kappa = _num(r["mr_prob"]), _num(r["kappa"])
    if mr is None:
        return "no ECM read", "the error-correction layer is warming up: ×1.00"
    kap = f"κ̂ {kappa:+.4f}/d" if kappa is not None else "κ̂ —"
    z = f"z = 2·MRProb − 1 = {r['reversion_z']:+.2f}" if d else "no claim to size: z = 0"
    return f"P(price-side κ > 0) {mr:.2f}", f"MRProb {mr:.2f} · {kap} · {z}"


def _payoff(p: dict | None) -> dict | None:
    if not p or _num(p.get("mean")) is None:
        return None
    return {"mean": _num(p.get("mean")), "se": _num(p.get("se")),
            "n": int(p.get("n") or 0), "n_win": int(p.get("n_win") or 0)}


def _call_grade(e: dict | None, prospective: dict | None) -> dict:
    """The GRADE→call summary: the call graded on the sessions it was issued.

    ``graded`` is True only with a defined IC, SE and t beside it, so a reader
    that formats them on ``graded`` never meets a None.
    """
    e, p = e or {}, prospective or {}
    label = edge_tier(e.get("t"))[0]
    ivh = e.get("issued_vs_held") or {}
    ic, se, t = _num(e.get("ic")), _num(e.get("se")), _num(e.get("t"))
    graded = bool(e.get("graded", False)) and None not in (ic, se, t)
    return {
        "status": str(e.get("status") or "ungraded"), "graded": graded,
        "ic": ic, "se": se, "t": t,
        "n_win": int(e.get("n_win") or 0), "label": label,
        "coverage": _num(e.get("coverage")), "issued_n": int(e.get("issued_n") or 0),
        "claim_n": int(e.get("claim_n") or 0),
        "hit": _num(e.get("hit")), "base": _num(e.get("base")),
        "issued_payoff": _payoff(ivh.get("issued")), "held_payoff": _payoff(ivh.get("held")),
        "prospective": {"status": str(p.get("status") or "no prospective record"),
                        "ic": _num(p.get("ic")), "se": _num(p.get("se")),
                        "n_win": int(p.get("n_win") or 0)},
    }


def verdict_from_row(call_ts: pd.DataFrame, edges: dict, *, asof, horizon: int,
                     exec_lag: int, target: str) -> dict:
    """The verdict: ``call_ts`` read at d* (``asof``), in the words every page prints.

    Returns ``{action, word, direction, size, trust, claim_t, headline, limit,
    binding, chain, size_formula, call_grade, asof, horizon_days, exec_lag,
    provisional, elapsed, caveat}``. ``chain`` is one row per ``CHAIN`` key,
    each ``{role, key, title, reading, grade, effect, detail}``. Nothing here
    is re-derived from anything but the row and the grades in ``edges``, and
    no value is NaN (a missing number is None), so two verdicts of one row
    compare equal.
    """
    h, edges = int(horizon), edges or {}
    call_grade = _call_grade(edges.get("call"), edges.get("call:prospective"))
    base = {"horizon_days": h, "exec_lag": int(exec_lag), "call_grade": call_grade}
    when = pd.Timestamp(asof) if asof is not None else None
    if when is None or call_ts is None or when not in call_ts.index:
        return {**base, "action": "NO CALL", "word": "NO CALL", "direction": 0, "size": 0.0,
                "trust": 0.0, "claim_t": None,
                "headline": f"NO CALL: {target} has no published valuation yet.",
                "limit": "no valuation published", "binding": None, "chain": [],
                "size_formula": f"no claim to size: size {format_size(0.0)}",
                "asof": None, "provisional": False, "elapsed": 0, "caveat": None}

    r = call_ts.loc[when].to_dict()
    action, d = str(r["action"]), int(r["d"])
    fvo, pct = _num(r["fvo"]), _num(r["pct"]) or 0.0
    size, trust, t = float(r["size"]), float(r["trust"]), _num(r["claim_t"])
    tradeable = bool(call_ts.attrs.get("tradeable", True))

    # ── The chain, in decision order ────────────────────────────────────
    raw = claim_direction(fvo)
    claim_side = {1: "BUY side (cheap)", -1: "SELL side (rich)"}.get(raw, "no claim")
    claim_reading = ("no valuation" if fvo is None else
                     f"{abs(fvo):.2f}σ {'cheap' if fvo < 0 else 'rich'}" if raw else
                     "inside the noise")
    claim_detail = (f"{abs(pct):.1f}% {'below' if pct < 0 else 'above'} the level the macro "
                    f"cross-section implies (FVO {fvo:+.2f}σ; no claim inside ±{CLAIM_SIGMA:g}σ)"
                    if fvo is not None else "Mūla has published no valuation for this session")

    behind = int(r["behind"])
    holds = {"precondition:published": bool(r["published"]),
             "precondition:printed": bool(r["printed"]) and behind < h,
             "precondition:tradeable": tradeable,
             "precondition:claim_sigma": raw != 0}
    pre = chain_roles("precondition")
    failed = [p.title for p in pre if not holds[p.key]]
    pre_detail = " · ".join(f"{p.title} {'✓' if holds[p.key] else '✗'}" for p in pre)
    pre_detail += f" · {behind} of {h} sessions behind"
    pre_effect = ("the claim stands" if not failed else
                  "NO CLAIM" if failed == [ROLES["precondition:claim_sigma"].title] else "NO CALL")

    e_claim = edges.get("claim") or {}
    pos, nb = edge_periods(e_claim)
    # The claim's grade, its prior named as the pool it is (the instrument's
    # class and the study's date, from call_ts's provenance stamp); the
    # reason is rebuilt from it, so the headline, the limit and the trust row
    # say one thing. call_ts's own reason keeps the short form.
    grade_text = _claim_grade_text(r["claim_ic"], r["claim_se"], r["claim_n_win"],
                                   r["claim_prior_mu"], r["claim_prior_sd"],
                                   r["claim_prior_scope"], r["claim_prior_k"],
                                   asset_class=call_ts.attrs.get("asset_class"),
                                   study_date=call_ts.attrs.get("priors_run_date"))
    reason = _reason(action, str(r["precondition"]), r, h,
                     str(r["claim_prior_scope"]) != "ungraded", text=grade_text)
    periods = f"; full history {pos} of {nb} periods positive" if nb else ""
    trust_grade = (f"{grade_text}{periods} → t_post {t:.2f} · {edge_tier(t)[0]}" if t is not None
                   else f"{grade_text}{periods} → ungraded")

    chain = []
    for key in CHAIN:
        if key == "claim":
            chain.append(_chain_row(key, claim_reading, "its grade sets trust", claim_side, claim_detail))
        elif key == "precondition":
            chain.append(_chain_row(key, "all hold" if not failed else f"{', '.join(failed)} fails",
                                    "", pre_effect, pre_detail))
        elif key == "trust":
            chain.append(_chain_row(
                key, f"t_post {t:.2f}" if t is not None else "ungraded", trust_grade,
                f"trust {trust:.2f}",
                f"f(t) opens at {STRENGTH_T0:g} and is full at {STRENGTH_T1:g}; trust is the "
                f"claim's earned base size and is never the binding factor"))
        elif key == "size":
            pass                                     # appended after the formula below
        else:
            s = next(x for x in SIZERS if f"sizer:{x.key}" == key)
            reading, detail = _sizer_reading(s.key, r, d)
            st = _num(r[f"{s.key}_t"])
            grade = f"t_post {st:.2f} · {edge_tier(st)[0]}" if st is not None else "ungraded"
            chain.append(_chain_row(key, reading, grade,
                                    f"β {r[f'{s.key}_beta']:.2f} → ×{r[f'{s.key}_factor']:.2f}", detail))

    terms = " × ".join(f"{s.title.split()[0]} {r[f'{s.key}_factor']:.2f}" for s in SIZERS)
    size_formula = (f"trust {trust:.2f} × {terms} = {format_size(size)}" if d
                    else f"no claim to size: size {format_size(0.0)}")
    chain.append(_chain_row("size", size_formula, "", f"size {format_size(size)}",
                            "size = trust × ∏ factor: 1.00 is a fully trusted claim with neutral "
                            "sizers; an index, never a percentage"))

    # ── Headline and limit ──────────────────────────────────────────────
    gap = _gap_words(target, fvo, pct) if fvo is not None else f"{target} has no valuation"
    if action == "NO CALL":
        headline = f"NO CALL on {target}: {reason}."
        limit = reason
    elif action == "NO CLAIM":
        headline = f"NO CLAIM: {gap}, inside the ±{CLAIM_SIGMA:g}σ noise."
        limit = reason
    elif action == "HOLD":
        clause = (f"the claim has not been shown to pay on price: t_post {t:.2f} ({grade_text})"
                  if t is not None else f"the claim is ungraded ({reason.removeprefix('claim ungraded: ')})")
        headline = f"HOLD: {gap}, but {clause}; acting needs a t_post above {STRENGTH_T0:g}."
        limit = f"Limited by the claim's grade: {reason}"
    else:
        headline = (f"{action} · size {format_size(size)}: {gap}; the claim's price-side grade "
                    f"(t_post {t:.2f}) earns trust {trust:.2f}.")
        b = str(r["binding"])
        if b in {s.key for s in SIZERS}:
            s = next(x for x in SIZERS if x.key == b)
            limit = (f"Limited by {s.title}: ×{r[f'{b}_factor']:.2f} "
                     f"({_sizer_reading(b, r, d)[0]})")
        else:
            limit = "none: sizers neutral or confirming"

    last = call_ts.index[-1]
    caveat = None
    if bool(r["provisional"]):
        caveat = (f"PROVISIONAL: {when:%d %b %Y} is a session still forming; the call is "
                  f"published and may be revised once when it closes.")
    elif last > when:
        why = str(call_ts["withheld_reason"].iloc[-1] or "withheld")
        caveat = (f"The newest session ({last:%d %b %Y}) was not valued ({why}); the call is "
                  f"as of {when:%d %b %Y}.")

    return {
        **base, "action": action, "word": action, "direction": d, "size": size,
        "trust": trust, "claim_t": t, "headline": headline, "limit": limit,
        "binding": str(r["binding"]) or None, "chain": chain, "size_formula": size_formula,
        "asof": when, "provisional": bool(r["provisional"]), "elapsed": min(behind, h),
        "caveat": caveat,
    }
