"""
Tattva — trust: an instrument's own grade, shrunk toward its asset class, mapped once.
तत्त्व (Tattva) — "Principle / Essence"

Per-instrument grading at h = 10 cannot certify even a claim that works as its
own model predicts: with about 140 independent windows, a true IC of 0.06
reaches t ≥ 1.645 a minority of the time. So the owner's ruling is to shrink
each instrument's as-of grade toward a pooled prior of its asset class,
produced offline by one committed catalogue study (``studies/pooled_priors.py``)
whose JSON the app reads as data. This module holds the arithmetic:

  * :class:`PriorTable` / :func:`load_priors` — the study's priors, looked up
    by object, instrument, class and date. A member's prior is built entirely
    WITHOUT that member (leave-one-out) and only from study vintages dated on
    or before the date it is used at, so it never contains the instrument's
    own rows or any future.
  * :func:`posterior` — the GLS combination of two correlated measurements of
    the same IC: the own grade and the class prior, whose errors correlate
    (``kappa``) because the class pool shares dates with the instrument.
  * :func:`strength` — THE map from a t to a strength in [0, 1], used for the
    claim's trust and every sizer's β alike.

No Streamlit dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.knobs import knob

#: f(t) opens at T0 (a one-sided 5% false-open per look) and is full at T1.
STRENGTH_T0: float = knob("STRENGTH_T0")
STRENGTH_T1: float = knob("STRENGTH_T1")

#: Where the study writes its output: one file per (h, L).
PRIORS_DIR = Path(__file__).resolve().parent.parent / "data" / "priors"


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class Prior:
    """The class prior for one object, instrument and vintage.

    ``mu`` the prior IC; ``var_mu`` the variance of that estimate; ``tau2`` the
    between-instrument variance of true ICs; ``kappa`` the covariance of the
    prior's error with the instrument's own grade error; ``se_floor`` the SE of
    the class pool INCLUDING the instrument (no posterior is more precise);
    ``k`` the members pooled; ``scope`` 'class-LOO', 'global-LOO' or
    'class-all'; ``vintage`` the study date it is valid from.
    """

    mu: float
    var_mu: float
    tau2: float
    kappa: float
    se_floor: float
    k: int
    scope: str
    vintage: pd.Timestamp
    study: str
    run_date: str


class PriorTable:
    """The study's priors: ``priors[obj][member | '__class__:<class>'][vintage]``.

    ``meta`` holds the file's provenance (study, git SHA, script hash, run
    date, epoch, h, L, ...): everything but the priors themselves.
    """

    def __init__(self, priors: dict, meta: dict):
        self.meta = dict(meta)
        self._priors = {
            obj: {who: sorted((pd.Timestamp(v), dict(e)) for v, e in vintages.items())
                  for who, vintages in by_who.items()}
            for obj, by_who in (priors or {}).items()
        }

    def lookup(self, obj: str, instrument: str, asset_class: str, t) -> Prior | None:
        """The latest vintage on or before ``t`` for this object and instrument.

        A study member gets its own leave-one-out entry; any other instrument
        gets its class's all-member entry. None when the object, the entry or
        a vintage on or before ``t`` is missing.
        """
        by_who = self._priors.get(obj) or {}
        entries = by_who.get(instrument) or by_who.get(f"__class__:{asset_class}")
        if not entries:
            return None
        t = pd.Timestamp(t)
        valid = [(v, e) for v, e in entries if v <= t]
        if not valid:
            return None
        v, e = valid[-1]
        return Prior(mu=float(e["mu"]), var_mu=float(e["var_mu"]), tau2=float(e["tau2"]),
                     kappa=float(e["kappa"]), se_floor=float(e["se_floor"]), k=int(e["k"]),
                     scope=str(e["scope"]), vintage=v, study=str(self.meta.get("study", "")),
                     run_date=str(self.meta.get("run_date", "")))


def load_priors(h: int, exec_lag: int, path: str | None = None) -> PriorTable | None:
    """The pooled priors for horizon ``h`` and lag ``exec_lag``, or None.

    None when the study has not been run (no file). A file studied at a
    different h or L is REFUSED (``ValueError``): its priors grade a different
    estimand, and shrinking toward them would mix two questions.
    """
    p = Path(path) if path is not None else PRIORS_DIR / f"pooled_priors_h{int(h)}_L{int(exec_lag)}.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    if int(data.get("h", -1)) != int(h) or int(data.get("exec_lag", -1)) != int(exec_lag):
        raise ValueError(f"{p.name} was studied at h={data.get('h')}, L={data.get('exec_lag')}; "
                         f"this run grades h={h}, L={exec_lag}")
    meta = {k: v for k, v in data.items() if k != "priors"}
    return PriorTable(data.get("priors") or {}, meta)


@dataclass
class Posterior:
    """The grade trust is read from: IC, SE and t after shrinkage.

    ``w_own`` is the weight on the instrument's own grade (1 with no prior, 0
    for the prior predictive); ``scope`` names what was combined ('own',
    the prior's scope, or 'ungraded').
    """

    ic: float
    se: float
    t: float
    w_own: float
    scope: str


def posterior(ic, se, prior: Prior | None) -> Posterior:
    """GLS combination of the own grade ``(ic, se)`` with ``prior``.

    With ``a = ic``, ``V_a = se²``, ``V_b = τ² + var_μ`` and error covariance
    ``c = κ``::

        w   = clip((V_b − c) / (V_a + V_b − 2c), 0, 1)     (1 if the denominator ≤ 0)
        m   = w·a + (1 − w)·μ
        v   = w²·V_a + (1 − w)²·V_b + 2w(1 − w)·c
        se  = max(√v, se_floor),  t = m / se

    No prior: the own grade. No own grade (warm-up, NaN): the prior
    predictive ``(μ, √V_b)``, so a skilled class can open a young instrument.
    Neither: 'ungraded', with t NaN (strength 0).
    """
    a, sa = _finite(ic), _finite(se)
    own = a is not None and sa is not None and sa > 0
    if prior is None:
        if not own:
            return Posterior(float("nan"), float("nan"), float("nan"), 0.0, "ungraded")
        return Posterior(a, sa, a / sa, 1.0, "own")
    v_b = prior.tau2 + prior.var_mu
    if not own:
        se_p = max(math.sqrt(v_b), prior.se_floor)
        return Posterior(prior.mu, se_p, prior.mu / se_p if se_p > 0 else float("nan"),
                         0.0, prior.scope)
    v_a, c = sa * sa, prior.kappa
    denom = v_a + v_b - 2 * c
    w = 1.0 if denom <= 0 else min(1.0, max(0.0, (v_b - c) / denom))
    m = w * a + (1 - w) * prior.mu
    v = w * w * v_a + (1 - w) * (1 - w) * v_b + 2 * w * (1 - w) * c
    se_p = max(math.sqrt(max(v, 0.0)), prior.se_floor)
    return Posterior(m, se_p, m / se_p if se_p > 0 else float("nan"), w, prior.scope)


def strength(t) -> float:
    """THE map: ``clip((t − T0) / (T1 − T0), 0, 1)``; an undefined t is 0."""
    v = _finite(t)
    if v is None:
        return 0.0
    return min(1.0, max(0.0, (v - STRENGTH_T0) / (STRENGTH_T1 - STRENGTH_T0)))
