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
from data.calendars import printed_sessions_behind
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
                  h: int, today, exec_lag: int | None = None) -> pd.DataFrame:
    """The composer's inputs, one row per Mūla session, every value read BY DATE.

    Columns: ``price, fvo, pct`` (gap in %), ``valid, provisional``;
    ``published`` (FVO finite and the session not withheld) and
    ``withheld_reason``; ``printed`` (the target printed: the prepared spine
    keeps only its own sessions, so a row's price is its print); ``behind``
    (the target's sessions after d* that have closed by ``today`` —
    ``data.calendars.printed_sessions_behind``: its own printed sessions where
    the spine has them, the calendar's beyond — on d*'s row only; a past
    session was issued on its own close, 0 behind); ``claim``
    (``claim_direction(FVO)``, before any precondition); the Swayam sizer's
    ``swayam_lean`` (the session's published ``Lean``, the one Swayam lean
    ``aggregate_views`` emits; NaN, never filled, when Swayam has no row that
    day), ``swayam_native`` and ``z_swayam = claim · Lean`` on a native row
    (NaN on a carried or absent one); the reversion sizer's ``mr_prob, kappa``
    and ``z_reversion = 2·MRProb − 1``.

    ``today`` is the newest date whose session has CLOSED
    (``data.calendars.session_closed_through``), or a replay's end date.

    ``attrs``: ``h``, ``exec_lag`` (L, default EXECUTION_LAG), ``target``,
    ``asset_class``, ``has_ohlcv`` and ``asof`` (d*, the newest published
    session).
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
        behind.loc[d_star] = int(printed_sessions_behind(instrument.ticker, idx, d_star, today))

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
    out.attrs = {"h": int(h), "exec_lag": int(EXECUTION_LAG if exec_lag is None else exec_lag),
                 "target": instrument.name, "asset_class": instrument.asset_class,
                 "has_ohlcv": bool(instrument.has_ohlcv), "asof": d_star}
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
    reads it as ungraded) and the failure is reported. An object the target
    cannot have — the Swayam sizer of a target with no OHLCV feed, which never
    reads anything — is not graded at all: no class prior predictive stands in
    for a reading that cannot exist (its β is 0 and its factor ×1.00).
    """
    rows = _claim_rows(inputs)
    claim = inputs["claim"].astype(float)
    specs = (("claim", claim, None),
             *((f"sizer:{s.key}", inputs[s.z_col], claim) for s in SIZERS
               if s.key != "swayam" or getattr(instrument, "has_ohlcv", True)))
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
    """Attach the as-of, shrunk grade the call read at d* (what trust or β used).

    ``asof_grade`` is the ONE grade of a role object every page prints: the
    own as-of grade (``ic``, ``se``, ``n_win``) at the grid date it was graded
    through (``grid_date``), the prior, the posterior and its strength. The
    edge's own full-history fields are a diagnostic beside it.
    """
    if grade is not None and d_star is not None and d_star in grade.index:
        g = grade.loc[d_star]
        gd = pd.Timestamp(g["grid_date"]) if pd.notna(g.get("grid_date")) else None
        edge["asof_grade"] = {**{k: _num(g[k]) for k in ("ic", "se", "n_win", "prior_mu", "prior_sd",
                                                           "ic_post", "se_post", "t_post", "w_own",
                                                           "strength")},
                              "prior_scope": str(g["prior_scope"]), "grid_date": gd}
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

def describe_priors(priors: PriorTable | None, refused: str | None = None) -> dict:
    """The bundle's ``priors_meta``: the priors file's provenance, or why there is none.

    ``status`` is ``'loaded'``, ``'refused'`` (a file exists and was refused —
    ``refused`` is ``core.trust.priors_for_run``'s reason, e.g. studied at
    another h or L) or ``'none'`` (no file: the study has not run). A loaded
    file carries ``knobs_match`` (False when it was studied under another
    knobs registry; ``core.trust.load_priors``).
    """
    if priors is None:
        if refused:
            return {"status": "refused", "label": f"priors file refused: {refused}",
                    "refusal": refused, "study": None, "run_date": None, "sha": None}
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
    session of ``fvo_ts``; ``today`` is the newest date whose session has
    closed on the target's exchange (``data.calendars.session_closed_through``)
    or a replay's end date: the call's recency (``behind``), P2 and the elapsed
    horizon are counted to it. h is ``cfg.forecast_horizon`` and L is
    ``EXECUTION_LAG``, once, for every grade. ``epoch`` and ``fingerprint``
    name the prepared history (``data.prep``); with the priors' provenance and
    the code version they are stamped on ``call_ts.attrs`` and on every edge's
    ``attrs``.
    """
    h, lag = int(cfg.forecast_horizon), int(EXECUTION_LAG)
    inputs = gather_inputs(fvo_ts, swayam_daily, instrument=instrument, h=h, today=today,
                           exec_lag=lag)
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
                               target=instrument.name, closed_through=today)
    return Assembled(call_ts=call_ts, grades_ts=grades_ts, edges=edges, evidence=evidence,
                     verdict=verdict)


# ═══════════════════════════════════════════════════════════════════════
#  A stored run, read again on a later day
# ═══════════════════════════════════════════════════════════════════════

#: call_ts columns a one-row recomposition rebuilds the composer's inputs from.
_INPUT_COPY = ("published", "valid", "provisional", "printed", "withheld_reason", "fvo", "pct",
               "swayam_lean", "swayam_native", "mr_prob", "kappa")


def restate(call_ts: pd.DataFrame, edges: dict, grades_ts: dict[str, pd.DataFrame], *,
            ticker: str | None, today, price: pd.Series | None) -> tuple[pd.DataFrame, dict, dict]:
    """A stored run's ``(call_ts, edges, verdict)``, read against ``today``.

    ``behind`` at d* — and with it P2, the elapsed horizon and the action — is
    measured to a date (``data.calendars.session_closed_through``), so a run
    kept open across sessions must be read again against the current one, not
    the one it was computed on. Nothing is refitted: d*'s row is recomposed by
    the composer itself (``core.call.compose_series``) from that row's own
    inputs, which call_ts carries, with ``behind`` re-counted
    (``data.calendars.printed_sessions_behind`` on call_ts's spine); if d* now
    carries a different claim or call (its coverage counts), ``edges['call']``
    is re-graded on the restated call_ts. The verdict is rebuilt with ``closed_through=today``, so a
    PROVISIONAL d* whose session has closed since says so. When nothing moved,
    the stored ``call_ts`` and ``edges`` are returned as they are.
    """
    attrs = dict(call_ts.attrs)
    d_star = attrs.get("asof")
    h, lag = int(attrs["h"]), int(attrs.get("exec_lag", EXECUTION_LAG))
    target = str(attrs.get("target") or "")
    if d_star is None or pd.Timestamp(d_star) not in call_ts.index:
        verdict = verdict_from_row(call_ts, edges, asof=d_star, horizon=h, exec_lag=lag,
                                   target=target, closed_through=today)
        return call_ts, edges, verdict
    d_star = pd.Timestamp(d_star)
    behind = int(printed_sessions_behind(ticker, call_ts.index, d_star, today))
    if behind != int(call_ts.at[d_star, "behind"]):
        row = call_ts.loc[[d_star]]
        inp = pd.DataFrame({c: row[c].to_numpy() for c in _INPUT_COPY}, index=row.index)
        inp["behind"] = np.array([behind], dtype=np.int64)
        claim = inp["fvo"].map(claim_direction).astype(np.int64)
        inp["claim"] = claim
        native = inp["swayam_native"].astype(bool)
        inp["z_swayam"] = (claim * inp["swayam_lean"].astype(float)).where(native)
        inp["z_reversion"] = 2.0 * inp["mr_prob"].astype(float) - 1.0
        inp.attrs = {"h": h, "exec_lag": lag}
        g1 = {k: g.reindex(row.index) for k, g in (grades_ts or {}).items()}
        new = compose_series(inp, g1, tradeable=bool(attrs.get("tradeable", True)),
                             sizers_enabled=bool(attrs.get("sizers_enabled", True)))
        old_w, old_d = float(call_ts.at[d_star, "w"]), int(call_ts.at[d_star, "d"])
        call_ts = call_ts.copy()
        for c in call_ts.columns:
            call_ts.at[d_star, c] = new.at[d_star, c]
        call_ts.attrs = attrs
        moved = (float(call_ts.at[d_star, "w"]) != old_w or int(call_ts.at[d_star, "d"]) != old_d)
        # The call's grade counts d*'s row in its coverage (issued and claim sessions).
        if moved and price is not None and "call" in (edges or {}):
            edges = dict(edges)
            e = _call_edge(call_ts, price.reindex(call_ts.index), h=h, exec_lag=lag)
            e["attrs"] = dict((edges["call"] or {}).get("attrs") or {})
            edges["call"] = e
    verdict = verdict_from_row(call_ts, edges, asof=d_star, horizon=h, exec_lag=lag,
                               target=target, closed_through=today)
    return call_ts, edges, verdict

