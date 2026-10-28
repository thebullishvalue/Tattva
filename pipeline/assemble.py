"""
Tattva — the loop, closed: inputs by date → as-of grades → the call at every
session → the call graded as issued → the verdict at d*.
तत्त्व (Tattva) — "Principle / Essence"

    inputs    = gather_inputs(fvo_ts, swayam_daily, …)     every value read by date
    grades_ts = grade_series(inputs, price, …)             core.grade.expanding_edge,
                                                           shrunk (core.trust), per object
    call_ts   = core.call.compose_series(inputs, grades_ts, …)
    evidence  = pipeline.evidence.build_evidence(fvo_ts, …)
    edges     = grade_all(call_ts, inputs, price, evidence, …)
    verdict   = core.call.verdict_from_row(call_ts, edges, asof=d*, …)

:func:`assemble` runs it. Every grade here is THE grade (core.grade) on the
drift-neutral target, with the execution lag, on ONE decision sample: the
Valid, target-printed sessions on which Mūla made a claim (``claim rows``),
and of those the native Swayam rows for the Swayam sizer (its z is absent on
a carried row). Each grade is computed in its own ``try``: one that fails is
reported and leaves the others, and the call, standing.

No Streamlit dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.call import SIZERS, compose_series, verdict_from_row
from core.config import VERSION
from core.grade import causal_drift, expanding_edge, forward_return, series_edge
from core.knobs import knob
from core.logger_config import console
from core.semantics import claim_direction, swayam_word
from core.trust import PriorTable, posterior, strength
from data.calendars import trading_days_behind
from pipeline.evidence import build_evidence

EXECUTION_LAG: int = knob("EXECUTION_LAG")
DRIFT_MIN_RESOLVED: int = knob("DRIFT_MIN_RESOLVED")
GRADE_WARMUP_WIN: int = knob("GRADE_WARMUP_WIN")

#: The label of a run without the study's pooled priors.
NO_PRIOR_LABEL = "no pooled prior: per-instrument grade only"


def _by_date(frame: pd.DataFrame | None) -> pd.DataFrame:
    """``frame`` on normalised, de-duplicated (last wins), sorted dates."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    f = frame.copy()
    f.index = pd.DatetimeIndex(pd.to_datetime(f.index, errors="coerce")).normalize()
    f = f[~f.index.isna()]
    return f[~f.index.duplicated(keep="last")].sort_index()


def _col(f: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column, NaN where absent."""
    if name not in f.columns:
        return pd.Series(np.nan, index=f.index)
    return pd.to_numeric(f[name], errors="coerce").astype(float)


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ═══════════════════════════════════════════════════════════════════════
#  Inputs, by date
# ═══════════════════════════════════════════════════════════════════════

def gather_inputs(fvo_ts: pd.DataFrame, swayam_daily: pd.DataFrame | None, *, instrument,
                  h: int, today) -> pd.DataFrame:
    """The composer's inputs, one row per Mūla session, every value read BY DATE.

    Columns: ``price, fvo, pct`` (gap in %), ``valid, provisional``;
    ``published`` (FVO finite and the session not withheld) and
    ``withheld_reason``; ``printed`` (the target printed: the prepared spine
    keeps only its own sessions, so a row's price is its print); ``behind``
    (the target's sessions from d* to ``today``, on d*'s row only — a past
    session was issued on its own close, 0 behind); ``claim``
    (``claim_direction(FVO)``, before any precondition); the Swayam sizer's
    ``swayam_lean`` (the session's published ``Lean``, the one Swayam lean
    ``aggregate_views`` emits; NaN, never filled, when Swayam has no row that
    day), ``swayam_native`` and ``z_swayam = claim · Lean`` on a native row
    (NaN on a carried or absent one); the reversion sizer's ``mr_prob, kappa``
    and ``z_reversion = 2·MRProb − 1``.

    ``attrs``: ``h``, ``target``, ``asset_class`` and ``asof`` (d*, the
    newest published session).
    """
    f = _by_date(fvo_ts)
    idx = f.index
    fvo = _col(f, "FVO")
    reason = (f["WithheldReason"].fillna("").astype(str) if "WithheldReason" in f.columns
              else pd.Series("", index=idx))
    published = fvo.notna() & (reason == "")
    price = _col(f, "Price")
    pub = idx[published.to_numpy()]
    d_star = pd.Timestamp(pub[-1]) if len(pub) else None
    behind = pd.Series(0, index=idx, dtype=np.int64)
    if d_star is not None:
        behind.loc[d_star] = int(trading_days_behind(instrument.ticker, d_star.date(),
                                                     pd.Timestamp(today).date()))

    sw = _by_date(swayam_daily).reindex(idx)
    lean = _col(sw, "Lean")
    native = (sw["_Native"].astype("boolean").fillna(False).astype(bool)
              if "_Native" in sw.columns else pd.Series(True, index=idx)) & lean.notna()
    claim = fvo.map(claim_direction).astype(np.int64)
    mr = _col(f, "MRProb")

    out = pd.DataFrame({
        "price": price, "fvo": fvo, "pct": _col(f, "PctMispricing") * 100.0,
        "valid": f["Valid"].astype(bool), "provisional": f["Provisional"].astype(bool),
        "published": published, "withheld_reason": reason, "printed": price.notna() & (price > 0),
        "behind": behind, "claim": claim,
        "swayam_lean": lean, "swayam_native": native, "z_swayam": (claim * lean).where(native),
        "mr_prob": mr, "kappa": _col(f, "MulaKappa"), "z_reversion": 2.0 * mr - 1.0,
    }, index=idx)
    out.attrs = {"h": int(h), "target": instrument.name, "asset_class": instrument.asset_class,
                 "asof": d_star}
    return out


def _claim_rows(inputs: pd.DataFrame) -> pd.Series:
    """The decision sample: Valid, target-printed sessions carrying a claim."""
    return inputs["valid"] & inputs["printed"] & (inputs["claim"] != 0)


# ═══════════════════════════════════════════════════════════════════════
#  As-of grades, shrunk to the class prior
# ═══════════════════════════════════════════════════════════════════════

def _shrunk(ee: pd.DataFrame, obj: str, priors: PriorTable | None, instrument) -> pd.DataFrame:
    """An as-of grade combined with the prior valid at each grid date (core.trust)."""
    at = ee[ee.index == pd.DatetimeIndex(ee["grid_date"])]
    rows = []
    for gd, r in at.iterrows():
        prior = (priors.lookup(obj, instrument.name, instrument.asset_class, gd)
                 if priors is not None else None)
        post = posterior(r["ic"], r["se"], prior)
        rows.append({
            "prior_mu": prior.mu if prior else np.nan,
            "prior_sd": math.sqrt(prior.tau2 + prior.var_mu) if prior else np.nan,
            "prior_scope": prior.scope if prior else "none",
            "prior_k": prior.k if prior else 0,
            "prior_vintage": prior.vintage if prior else pd.NaT,
            "ic_post": post.ic, "se_post": post.se, "t_post": post.t, "w_own": post.w_own,
            "strength": strength(post.t),
        })
    post = pd.DataFrame(rows, index=at.index).reindex(pd.DatetimeIndex(ee["grid_date"]))
    post.index = ee.index
    return pd.concat([ee[["ic", "se", "n_win", "grid_date"]], post], axis=1)


def grade_series(inputs: pd.DataFrame, price: pd.Series, *, h: int, exec_lag: int,
                 priors: PriorTable | None, instrument) -> dict[str, pd.DataFrame]:
    """The as-of grade of every role object, on ``price``'s dates.

    ``'claim'`` grades the claim's sign against the drift-neutral forward
    return on the claim rows; ``'sizer:<key>'`` grades the sizer's z against
    claim · (Y − μ) on the claim rows (rows where its z is absent drop out).
    Each is ``core.grade.expanding_edge`` (only outcomes resolved by each grid
    date), combined with the prior valid at that date by
    ``core.trust.posterior``: columns ``ic, se, n_win, grid_date, prior_mu,
    prior_sd, prior_scope, prior_k, prior_vintage, ic_post, se_post, t_post,
    w_own, strength`` (``prior_vintage`` the study vintage the prior was read
    from, NaT with none). An object whose grade raises is left out (the composer then
    reads it as ungraded) and the failure is reported.
    """
    rows = _claim_rows(inputs)
    claim = inputs["claim"].astype(float)
    specs = (("claim", claim, None),
             *((f"sizer:{s.key}", inputs[s.z_col], claim) for s in SIZERS))
    out: dict[str, pd.DataFrame] = {}
    for obj, lean, sign in specs:
        try:
            ee = expanding_edge(lean, price, h, mask=rows, sign=sign, drift="causal",
                                exec_lag=exec_lag, warmup_win=GRADE_WARMUP_WIN)
            out[obj] = _shrunk(ee, obj, priors, instrument)
        except Exception as e:                       # noqa: BLE001 - one grade, not the call
            console.warning(f"{obj} as-of grade failed ({type(e).__name__}: {e}) — ungraded")
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Every role graded, and the call graded as issued
# ═══════════════════════════════════════════════════════════════════════

def _empty_edge(h: int, exec_lag: int) -> dict:
    """series_edge's shape with nothing graded."""
    return series_edge(pd.Series(dtype=float), pd.Series(dtype=float), h,
                       drift="causal", exec_lag=exec_lag)


def _mean_se(y: pd.Series, h: int) -> dict:
    """Mean of a drift-neutral payoff, its SE on n/h non-overlapping windows."""
    y = y.dropna()
    n = len(y)
    n_win = n // max(1, h)
    se = float(y.std(ddof=1) / math.sqrt(n_win)) if n_win >= 2 else math.nan
    return {"mean": float(y.mean()) if n else math.nan, "se": se, "n": n, "n_win": n_win}


def _with_asof(edge: dict, grade: pd.DataFrame | None, d_star, **extra) -> dict:
    """Attach the as-of, shrunk grade the call read at d* (what trust or β used)."""
    if grade is not None and d_star is not None and d_star in grade.index:
        g = grade.loc[d_star]
        edge["asof_grade"] = {**{k: _num(g[k]) for k in ("ic", "se", "n_win", "prior_mu", "prior_sd",
                                                           "ic_post", "se_post", "t_post", "w_own",
                                                           "strength")},
                              "prior_scope": str(g["prior_scope"])}
    edge.update(extra)
    return edge


def _stated_rule(edge: dict) -> dict:
    """C12's stated rule, where the pooled IC and the median period disagree in sign."""
    blocks = [b["ic"] for b in edge.get("blocks") or [] if _num(b.get("ic")) is not None]
    ic, se = _num(edge.get("ic")), _num(edge.get("se"))
    if blocks and ic is not None and se is not None:
        med = float(np.median(blocks))
        if np.sign(med) != np.sign(ic):
            edge["stated_rule"] = (f"pooled {ic:+.2f} ± {se:.2f}; median period {med:+.2f}: "
                                   f"the gap level shifts across periods")
    return edge


def _terciles(size: pd.Series, y: pd.Series, h: int) -> dict:
    """Payoff by causal expanding size tercile of the issued calls.

    A call's tercile is the share of issued sizes up to and including its own
    session that are at or below it — known when it was issued.
    """
    s = size.to_numpy(dtype=np.float64)
    terc = np.array([min(3, max(1, math.ceil(3 * float(np.mean(s[:i + 1] <= v)))))
                     for i, v in enumerate(s)], dtype=np.int64)
    return {f"T{k}": _mean_se(y[terc == k], h) for k in (1, 2, 3)}


def _payoff(d: pd.Series, price: pd.Series, *, h: int, exec_lag: int) -> pd.Series:
    """The claim's drift-neutral payoff d · (Y − μ) on ``d``'s dates.

    Y is the h-session return from the execution close, μ its causal drift
    (only returns resolved by each date; NaN until DRIFT_MIN_RESOLVED have).
    """
    Y = forward_return(price, h, exec_lag)
    mu, _ = causal_drift(Y, h, exec_lag, DRIFT_MIN_RESOLVED)
    return d.astype(float) * (Y - mu).reindex(d.index)


def _swayam_buckets(inputs: pd.DataFrame, y: pd.Series, rows: pd.Series, h: int) -> dict:
    """The Swayam sizer's graded audit: the claim's payoff by what Swayam said.

    On the claim rows with a native Swayam row (the sizer's own sample), each
    session goes to ``core.semantics.swayam_word(d · Lean)`` — contradicts,
    split or confirms at ±SWAYAM_DEAD — and the claim's drift-neutral payoff
    is averaged within each, with its SE on n/h windows.
    """
    z = inputs["z_swayam"]
    graded = rows & z.notna()
    word = z.map(swayam_word)
    return {w: _mean_se(y[graded & (word == w)], h) for w in ("contradicts", "split", "confirms")}


def _call_edge(call_ts: pd.DataFrame, price: pd.Series, *, h: int, exec_lag: int) -> dict:
    """The call graded on the sessions it was issued (w ≠ 0), with its coverage."""
    w, d = call_ts["w"].astype(float), call_ts["d"].astype(float)
    issued, claimed = w != 0, d != 0
    e = series_edge(w, price, h, mask=issued, drift="causal", exec_lag=exec_lag)
    y = _payoff(d, price, h=h, exec_lag=exec_lag)
    # Graded only past the warm-up AND with a defined IC and SE: issued calls
    # that are all the same (one side at one size) have nothing to rank.
    defined = bool(np.isfinite(e["ic"]) and np.isfinite(e["se"]))
    graded = e["n_win"] >= GRADE_WARMUP_WIN and defined
    if graded:
        status = "graded"
    elif e["n_win"] < GRADE_WARMUP_WIN:
        status = f"not yet graded: {e['n_win']} of {GRADE_WARMUP_WIN} non-overlapping issued windows"
    elif w[issued].nunique() <= 1:
        status = "ungraded: every issued call is the same, nothing to rank"
    else:
        status = "ungraded: the issued calls' IC is undefined"
    e.update(
        coverage=float(issued.sum() / claimed.sum()) if claimed.any() else math.nan,
        issued_n=int(issued.sum()), claim_n=int(claimed.sum()),
        issued_vs_held={"issued": _mean_se(y[issued], h), "held": _mean_se(y[claimed & ~issued], h)},
        terciles=_terciles(call_ts["size"][issued], y[issued], h),
        graded=bool(graded), status=status,
    )
    return e


def _prospective(call_ts: pd.DataFrame, price: pd.Series, *, h: int, exec_lag: int,
                 priors_meta: dict) -> dict:
    """The call graded only on sessions after the priors file's run date."""
    run_date = (priors_meta or {}).get("run_date")
    if not run_date:
        return {**_empty_edge(h, exec_lag),
                "status": "no prospective record: the pooled-prior study has not run"}
    w = call_ts["w"].astype(float)
    issued = (w != 0) & (call_ts.index > pd.Timestamp(run_date))
    e = series_edge(w, price, h, mask=issued, drift="causal", exec_lag=exec_lag)
    e.update(since=str(run_date),
             status=f"{e['n_win']} non-overlapping issued windows since {run_date}")
    return e


def grade_all(call_ts: pd.DataFrame, inputs: pd.DataFrame, price: pd.Series, evidence: dict, *,
              h: int, exec_lag: int, grades_ts: dict[str, pd.DataFrame], priors_meta: dict) -> dict:
    """``edges``: the full-history grade of every role object, each in its own try.

    * ``claim`` — the claim's sign on the claim rows, drift-neutral (with the
      as-of shrunk grade trust read at d*, and C12's stated rule where the
      pooled IC and the median period disagree in sign); ``claim:buy`` /
      ``claim:sell`` the same on each side (their IC is undefined — one sign —
      so they report hit against the side-matched base, and payoff);
    * ``sizer:<key>`` — the sizer's z against claim · (Y − μ) on the claim
      rows, with the β the call applied at d*; ``sizer:swayam`` also carries
      ``buckets``, the claim's drift-neutral payoff on its rows by
      contradicts / split / confirms (``_swayam_buckets``);
    * ``call`` — w on the sessions it was issued, plus coverage, issued-vs-held
      payoff, causal size-tercile payoff and the warm-up status;
    * ``call:prospective`` — the same on sessions after the priors' run date;
    * ``evidence:swayam_lean`` — Swayam's published Lean on its own, as a
      lean, on the valued sessions with a native Swayam row: EVIDENCE about
      the sizer's input, read by the Swayam page; it moves nothing;
    * ``evidence:ecm_drift`` — the reversion sizer's gap-implied h-session
      move (``evidence['ecm_drift']['lean']``, MulaDriftPct) as a lean, on the
      valued sessions: EVIDENCE→sizer:reversion, read by the Mūla page; it
      moves nothing.

    ``evidence`` is ``build_evidence``'s output; its one lean, the ECM's
    gap-implied move, is graded here (the other pieces are not leans). A key
    that raises reads as an empty grade with ``status 'failed: …'``; every key
    carries ``horizon``.
    """
    rows = _claim_rows(inputs)
    claim = inputs["claim"].astype(float)
    d_star = inputs.attrs.get("asof")
    grades_ts = grades_ts or {}

    def grade(lean, mask, sign=None):
        return series_edge(lean, price, h, mask=mask, sign=sign, drift="causal", exec_lag=exec_lag)

    def sizer(s):
        beta = _num(call_ts[f"{s.key}_beta"].get(d_star)) if d_star is not None else None
        e = _with_asof(grade(inputs[s.z_col], rows, claim), grades_ts.get(f"sizer:{s.key}"),
                       d_star, beta=beta)
        if s.key == "swayam":
            e["buckets"] = _swayam_buckets(inputs, _payoff(claim, price, h=h, exec_lag=exec_lag),
                                           rows, h)
        return e

    specs = {
        "claim": lambda: _stated_rule(_with_asof(grade(claim, rows), grades_ts.get("claim"), d_star)),
        "claim:buy": lambda: grade(claim, rows & (claim > 0)),
        "claim:sell": lambda: grade(claim, rows & (claim < 0)),
        **{f"sizer:{s.key}": (lambda s=s: sizer(s)) for s in SIZERS},
        "call": lambda: _call_edge(call_ts, price, h=h, exec_lag=exec_lag),
        "call:prospective": lambda: _prospective(call_ts, price, h=h, exec_lag=exec_lag,
                                                 priors_meta=priors_meta),
        "evidence:swayam_lean": lambda: grade(inputs["swayam_lean"],
                                              inputs["valid"] & inputs["printed"]
                                              & inputs["swayam_native"]),
        "evidence:ecm_drift": lambda: grade(evidence["ecm_drift"]["lean"].reindex(price.index),
                                            inputs["valid"] & inputs["printed"]),
    }
    edges: dict = {}
    for key, fn in specs.items():
        try:
            edges[key] = fn()
        except Exception as e:                       # noqa: BLE001 - one grade, not the run
            console.warning(f"{key} grade failed ({type(e).__name__}: {e})")
            edges[key] = {**_empty_edge(h, exec_lag), "status": f"failed: {type(e).__name__}: {e}"}
    return edges


# ═══════════════════════════════════════════════════════════════════════
#  The whole loop
# ═══════════════════════════════════════════════════════════════════════

def describe_priors(priors: PriorTable | None) -> dict:
    """The bundle's ``priors_meta``: the priors file's provenance, or the no-prior label."""
    if priors is None:
        return {"status": "none", "label": NO_PRIOR_LABEL, "study": None, "run_date": None,
                "sha": None}
    m = dict(priors.meta)
    study, run_date = m.get("study"), m.get("run_date")
    return {**m, "status": "loaded", "label": f"pooled prior: {study} ({run_date})",
            "study": study, "run_date": run_date, "sha": m.get("script_sha256")}


@dataclass
class Assembled:
    """One target's closed loop: the call at every session, its grades, its evidence, d*'s verdict."""

    call_ts: pd.DataFrame
    grades_ts: dict[str, pd.DataFrame]
    edges: dict
    evidence: dict
    verdict: dict


def assemble(fvo_ts: pd.DataFrame, swayam_daily: pd.DataFrame | None, *, instrument, cfg, asof,
             today, priors: PriorTable | None, epoch=None, fingerprint: str = "") -> Assembled:
    """Compose, grade and read the call for one target.

    ``asof`` is d* (``MulaEngine.asof``) and must be the newest published
    session of ``fvo_ts``; ``today`` is the target's current session date
    (``data.calendars.session_today``). h is ``cfg.forecast_horizon`` and L is
    ``EXECUTION_LAG``, once, for every grade. ``epoch`` and ``fingerprint``
    name the prepared history (``data.prep``); with the priors' provenance and
    the code version they are stamped on ``call_ts.attrs`` and on every edge's
    ``attrs``.
    """
    h, lag = int(cfg.forecast_horizon), int(EXECUTION_LAG)
    inputs = gather_inputs(fvo_ts, swayam_daily, instrument=instrument, h=h, today=today)
    want = pd.Timestamp(asof) if asof is not None else None
    if inputs.attrs["asof"] != want:
        raise ValueError(f"d* disagrees: the engine's {want} vs the newest published session "
                         f"{inputs.attrs['asof']}")
    price = inputs["price"]
    grades_ts = grade_series(inputs, price, h=h, exec_lag=lag, priors=priors, instrument=instrument)
    call_ts = compose_series(inputs, grades_ts, tradeable=instrument.tradeable,
                             sizers_enabled=bool(knob("SIZERS_ENABLED")))
    evidence = build_evidence(fvo_ts, h=h, exec_lag=lag)
    meta = describe_priors(priors)
    edges = grade_all(call_ts, inputs, price, evidence, h=h, exec_lag=lag,
                      grades_ts=grades_ts, priors_meta=meta)

    stamp = {"h": h, "exec_lag": lag, "epoch": epoch, "fingerprint": str(fingerprint or ""),
             "priors_study": meta["study"], "priors_run_date": meta["run_date"],
             "priors_sha": meta["sha"], "target": instrument.name,
             "asset_class": instrument.asset_class, "code_version": VERSION}
    call_ts.attrs.update(stamp)
    for e in edges.values():
        e["attrs"] = dict(stamp)
    verdict = verdict_from_row(call_ts, edges, asof=want, horizon=h, exec_lag=lag,
                               target=instrument.name)
    return Assembled(call_ts=call_ts, grades_ts=grades_ts, edges=edges, evidence=evidence,
                     verdict=verdict)

