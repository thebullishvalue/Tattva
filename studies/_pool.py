"""
Tattva — the pooled prior's arithmetic: one catalogue τ², a leave-one-out
class mean, its correlated variance, and how often the posterior opens.
तत्त्व (Tattva) — "Principle / Essence"

Pure functions on arrays of one object's own grades at one vintage — member
``j``'s IC ``ic[j]``, its shift-null SE ``se[j]`` and its asset class
``cls[j]`` (NaN where the member has no grade yet) — and ``rho``, the
members' error-correlation matrix. ``studies/pooled_priors.py`` drives them;
``tests/test_study.py`` holds them to their properties.

  * :func:`dl_tau2`        — the between-instrument variance τ² of true ICs,
                             catalogue-wide, by the DerSimonian–Laird moment
                             estimator with class fixed effects;
  * :func:`loo_prior`      — member i's prior, built entirely WITHOUT i;
  * :func:`class_all_prior`— a class's all-member prior, for an instrument
                             outside the study;
  * :func:`payoff_rho`     — the members' error correlation, from their
                             payoff-product series;
  * :func:`simulate_looks` — how often the posterior opens, per look, on a
                             catalogue whose truth is planted (check R5 and
                             tests/test_study.py).

A prior entry is ``{mu, var_mu, tau2, kappa, se_floor, k, scope}``, the
fields of ``core.trust.Prior`` the study writes. No Streamlit dependency.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.knobs import knob
from core.trust import STRENGTH_T0, Prior, posterior, strength

#: A class-LOO prior needs this many OTHER graded members of the class.
POOL_MIN_MEMBERS: int = knob("POOL_MIN_MEMBERS")


def _arrays(ic, se, cls) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(ic, se, cls, graded)``: float, float and object arrays, and which members
    carry a usable grade (finite IC, finite positive SE)."""
    ic = np.asarray(ic, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    cls = np.asarray(cls, dtype=object)
    graded = np.isfinite(ic) & np.isfinite(se) & (se > 0)
    return ic, se, cls, graded


def dl_tau2(ic, se, cls) -> float:
    """τ²: the variance of true ICs around their class means, catalogue-wide.

    DerSimonian–Laird with class fixed effects, on the members given (all of
    them must be graded): with ``w_j = 1/se_j²`` and ``μ_c`` the precision-
    weighted mean of class c,

        Q  = Σ_c Σ_{j∈c} w_j (ic_j − μ_c)²,   df = K − C,
        D  = Σ_c (Σ_{j∈c} w_j − Σ_{j∈c} w_j² / Σ_{j∈c} w_j),
        τ² = max(0, (Q − df) / D).

    One τ² across classes: a per-class estimate collapses to 0 with three to
    six members. 0 when no class has two members (D = 0).
    """
    ic, se, cls, _ = _arrays(ic, se, cls)
    w = 1.0 / (se * se)
    q = d = 0.0
    df = len(ic)
    for c in dict.fromkeys(cls):
        m = cls == c
        wc, xc = w[m], ic[m]
        sw = wc.sum()
        q += float(wc @ (xc - (wc @ xc) / sw) ** 2)
        d += float(sw - (wc @ wc) / sw)
        df -= 1
    if d <= 0:
        return 0.0
    return max(0.0, (q - df) / d)


def _pool(pool: np.ndarray, ic, se, rho: np.ndarray, tau2: float) -> tuple[float, float, np.ndarray]:
    """``(μ, var_μ, w*)`` of the members in ``pool`` at between-variance ``tau2``.

    ``w*_j = 1/(se_j² + τ²)``, ``μ = Σ w*_j ic_j / Σ w*_j`` and

        var_μ = Σ_jk w*_j w*_k (τ² δ_jk + ρ_jk se_j se_k) / (Σ w*)²,

    the variance of μ about the class mean, with the members' estimation
    errors correlated (twins, shared dates).
    """
    s = se[pool]
    w = 1.0 / (s * s + tau2)
    mu = float(w @ ic[pool]) / float(w.sum())
    cov = rho[np.ix_(pool, pool)] * np.outer(s, s) + tau2 * np.eye(len(s))
    var = float(w @ cov @ w) / float(w.sum()) ** 2
    return mu, max(var, 0.0), w


def loo_prior(i: int, ic, se, cls, rho, *, min_members: int = POOL_MIN_MEMBERS) -> dict | None:
    """Member ``i``'s prior, built entirely without member ``i``.

    Of the OTHER graded members: τ² is :func:`dl_tau2` on all of them
    (τ²₋ᵢ); the pool is i's class when it holds at least ``min_members`` of
    them (``scope 'class-LOO'``), else all of them (``'global-LOO'``); and

      * ``mu``, ``var_mu`` — the pool's mean and its variance (:func:`_pool`);
      * ``kappa`` — the covariance of the prior's error with i's own grade
        error, ``Σ_j w*_j ρ_ij se_i se_j / Σ w*_j`` (0 while i has no grade);
      * ``se_floor`` — the SE of the same pool WITH i added (at τ²₋ᵢ): no
        posterior is more precise than the whole pool it was drawn from;
      * ``k`` — the members pooled.

    ``ic[i]`` is never read, so a member's own skill cannot raise its prior.
    None when no other member is graded.
    """
    ic, se, cls, graded = _arrays(ic, se, cls)
    rho = np.asarray(rho, dtype=np.float64)
    idx = np.arange(len(ic))
    others = graded & (idx != i)
    if not others.any():
        return None
    tau2 = dl_tau2(ic[others], se[others], cls[others])
    in_class = others & (cls == cls[i])
    scope = "class-LOO" if in_class.sum() >= int(min_members) else "global-LOO"
    pool = idx[in_class if scope == "class-LOO" else others]
    mu, var_mu, w = _pool(pool, ic, se, rho, tau2)
    if graded[i]:
        kappa = float(w @ (rho[i, pool] * se[i] * se[pool])) / float(w.sum())
        _, var_with, _ = _pool(np.append(pool, i), ic, se, rho, tau2)
        se_floor = math.sqrt(var_with)
    else:
        kappa, se_floor = 0.0, math.sqrt(var_mu)
    return {"mu": mu, "var_mu": var_mu, "tau2": tau2, "kappa": kappa,
            "se_floor": se_floor, "k": int(len(pool)), "scope": scope}


def class_all_prior(c: str, ic, se, cls, rho, *, kappa, min_members: int = POOL_MIN_MEMBERS) -> dict | None:
    """The prior for an instrument of class ``c`` that is not a study member.

    Every graded member of ``c`` is pooled (``scope 'class-all'``), or every
    graded member of the catalogue when ``c`` has fewer than ``min_members``
    (``'global-all'``), at the catalogue-wide τ² of all graded members. The
    instrument is in neither, so ``se_floor`` is the pool's own SE and
    ``kappa`` is the median of the pooled members' leave-one-out κ (``kappa``,
    one per member). None when nothing is graded.
    """
    ic, se, cls, graded = _arrays(ic, se, cls)
    rho = np.asarray(rho, dtype=np.float64)
    if not graded.any():
        return None
    idx = np.arange(len(ic))
    tau2 = dl_tau2(ic[graded], se[graded], cls[graded])
    in_class = graded & (cls == c)
    scope = "class-all" if in_class.sum() >= int(min_members) else "global-all"
    pool = idx[in_class if scope == "class-all" else graded]
    mu, var_mu, _ = _pool(pool, ic, se, rho, tau2)
    k = np.asarray(kappa, dtype=np.float64)[pool]
    k = k[np.isfinite(k)]
    return {"mu": mu, "var_mu": var_mu, "tau2": tau2,
            "kappa": float(np.median(k)) if len(k) else 0.0,
            "se_floor": math.sqrt(var_mu), "k": int(len(pool)), "scope": scope}


def payoff_rho(u: dict[str, pd.Series], min_common: int = 60) -> pd.DataFrame:
    """The members' error correlation, from their payoff-product series.

    ``u[member]`` is the member's rank-payoff product r̃(lean)·r̃(target) on a
    window grid shared by every member (one value per window of h sessions;
    NaN where the member graded nothing). ``ρ_jk`` is the Pearson correlation
    of ``u_j`` and ``u_k`` over their common windows, 0 below ``min_common``
    of them, clipped to [−1, 1]; the diagonal is 1. Rows and columns follow
    ``u``'s order.
    """
    names = list(u)
    if not names:
        return pd.DataFrame()
    frame = pd.DataFrame({n: pd.Series(u[n], dtype=float) for n in names})
    r = frame.corr(min_periods=int(min_common)).reindex(index=names, columns=names)
    r = r.fillna(0.0).clip(-1.0, 1.0)
    vals = r.to_numpy(copy=True)
    np.fill_diagonal(vals, 1.0)
    return pd.DataFrame(vals, index=names, columns=names)


def _psd_factor(rho: np.ndarray) -> np.ndarray:
    """A factor L with L·Lᵀ the nearest PSD matrix to ``rho`` (eigenvalues clipped at 0)."""
    vals, vecs = np.linalg.eigh((rho + rho.T) / 2)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def simulate_looks(se, cls, rho, *, theta, tau: float = 0.0, n_sim: int = 1000, seed: int = 0,
                   min_members: int = POOL_MIN_MEMBERS, targets=None) -> dict:
    """How often one look at the posterior opens, on a catalogue with a planted truth.

    Each draw plants the members' true ICs ``θ_cls(j) + η_j`` (``theta`` a
    number, or ``{class: θ}`` with 0 elsewhere; ``η ~ N(0, tau²)``) and their
    own grades ``ic = truth + e``, ``e ~ N(0, Σ)`` with ``Σ_jk = ρ_jk se_j se_k``.
    Each member in ``targets`` (default: all) is then graded as the app grades
    it: its :func:`loo_prior` from the others, ``core.trust.posterior`` and
    ``core.trust.strength``. It OPENS when its posterior t reaches
    STRENGTH_T0.

    Returns ``{open, strength, scope}``: the per-member open rate and mean
    strength over ``n_sim`` draws (fixed ``seed``), and each member's scope.
    """
    se = np.asarray(se, dtype=np.float64)
    cls = np.asarray(cls, dtype=object)
    rho = np.asarray(rho, dtype=np.float64)
    n = len(se)
    targets = np.arange(n) if targets is None else np.asarray(list(targets), dtype=np.int64)
    if isinstance(theta, dict):
        base = np.array([float(theta.get(c, 0.0)) for c in cls])
    else:
        base = np.full(n, float(theta))
    lf = _psd_factor(rho)
    rng = np.random.default_rng(seed)
    opened = np.zeros(len(targets))
    sizes = np.zeros(len(targets))
    scopes: list[str] = [""] * len(targets)
    for _ in range(int(n_sim)):
        truth = base + float(tau) * rng.standard_normal(n)
        ic = truth + se * (lf @ rng.standard_normal(n))
        for a, i in enumerate(targets):
            e = loo_prior(int(i), ic, se, cls, rho, min_members=min_members)
            prior = None if e is None else Prior(vintage=pd.NaT, study="", run_date="", **e)
            t = posterior(ic[i], se[i], prior).t
            opened[a] += float(t >= STRENGTH_T0)
            sizes[a] += strength(t)
            scopes[a] = e["scope"] if e else "own"
    return {"open": opened / n_sim, "strength": sizes / n_sim, "scope": scopes}
