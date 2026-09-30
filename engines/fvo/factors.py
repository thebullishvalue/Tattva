"""
Online latent-factor extraction from the global cross-section.
=================================================================

The explanatory panel is ~200 instruments whose correlation matrix is mostly
noise: with N assets and T effective observations, the sample eigenvalues of
a *pure noise* correlation matrix fill the Marchenko-Pastur bulk
[(1-sqrt(q))^2, (1+sqrt(q))^2] with q = N/T.  Only eigenvalues above the
upper edge carry information.  That gives a parameter-free answer to "how
many factors?" -- the count is read off the spectrum, it is not chosen.

Design decisions and why
------------------------
*Why eigenvalue clipping rather than Bai-Ng ICp?*  Bai & Ng (2002) is the
standard for a fixed panel estimated in one shot.  It requires a penalty
calibrated to (N, T) of the full panel and a batch SVD, which is awkward to
make strictly recursive.  The Marchenko-Pastur edge (Laloux et al. 1999;
Plerou et al. 2002; Bouchaud & Potters 2003) gives the same answer
asymptotically, needs only the current spectrum, and simultaneously produces
a *cleaned, positive-definite* correlation matrix -- which the downstream
Woodbury likelihood needs anyway.  It is the cheaper of two equivalent
answers.

*Why a bank of memories instead of one window?*  Correlation memory is
itself a latent, regime-dependent quantity.  Three exponentially weighted
estimators with different half-lives are carried in parallel and scored by
their own causal one-step predictive likelihood.  The winner at time t was
the winner on evidence available at t; no window length is ever imposed.

References
----------
Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for
    some sets of random matrices." *Sbornik: Mathematics* 1(4).
Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). "Noise
    Dressing of Financial Correlation Matrices." *Phys. Rev. Lett.* 83.
Plerou, V. et al. (2002). "Random matrix approach to cross correlations in
    financial data." *Phys. Rev. E* 65.
Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate
    Factor Models." *Econometrica* 70(1).
Ledoit, O. & Wolf, O. (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices." *J. Multivariate Analysis* 88.
"""

from __future__ import annotations

import math

import numpy as np

from analytics.causal import EPS, VAR_FLOOR, halflife_to_lambda

#: Hard ceiling on retained factors.  Storage bound, not a modelling choice:
#: the Marchenko-Pastur edge decides the actual count and only exceeds this
#: in degenerate panels.  (Baked into downstream array shapes — changing it
#: at runtime breaks the latent-factor bank's broadcast.)
K_MAX = 20

#: Width of the band around the Marchenko-Pastur edge inside which `k` is held
#: at its previous value (scaled by 1/sqrt(t_eff) — see _clip_spectrum). Small
#: enough that a genuine factor still enters promptly; large enough that an
#: eigenvalue sitting on the edge stops rewriting the regressor set each run.
#:
#: MEASURED, 2026-08-17: this matters far less than it looks. Refitting the
#: real 10y panel with one predictor dropped moved `k` on 1 row out of ~2350,
#: yet fair values still drifted a median 0.56% (p95 4.6%). Composition
#: repainting is eigenvector ROTATION, not factor-count churn — so do not
#: expect changes here to fix it.  See the note on truncation in
#: _clip_spectrum before proposing shrinkage as an improvement.
MP_EDGE_HYSTERESIS: float = 2.0

#: Winsorisation of standardised returns, in sigmas.  A robustness device
#: from outlier-resistant covariance estimation, not a signal threshold: a
#: single 20-sigma print would otherwise rotate the entire eigenbasis.
Z_CLIP = 6.0

#: Correlation memories entered in the bank, in trading days.  This is the
#: support of the prior over memory, not a selected value.
MEMORY_GRID = (126.0, 252.0, 504.0)


# ===========================================================================
# Per-asset adaptive volatility (vectorised dynamic model averaging)
# ===========================================================================
class AdaptiveVolPanel:
    """Causal volatility for every column of a return panel, no fixed window.

    Each asset carries a bank of exponentially weighted variance estimators
    with different half-lives; the weights are updated by the Gaussian
    one-step predictive likelihood of the return actually realised.  This is
    dynamic model averaging (Raftery et al. 2010) applied per asset, executed
    as dense array arithmetic so the cost is negligible for N ~ 200.

    Reading :meth:`sigma` before :meth:`update` gives the volatility that was
    knowable strictly before the observation -- which is what standardisation
    requires if it is not to leak.
    """

    HALFLIVES = (10.0, 21.0, 63.0, 252.0)

    def __init__(self, n: int, mean_halflife: float = 504.0) -> None:
        self.n = n
        self.h = np.array(self.HALFLIVES)
        self.lam = np.array([halflife_to_lambda(x) for x in self.HALFLIVES])
        self.V = np.zeros((len(self.h), n))       # EW variance per model/asset
        self.W = np.zeros((len(self.h), n))       # accumulated weight
        self.logw = np.zeros((len(self.h), n))    # log model probability
        self.mu = np.zeros(n)                     # slow EW mean of returns
        self.muw = np.zeros(n)
        self.mu_lam = halflife_to_lambda(mean_halflife)
        self.count = np.zeros(n)

    def sigma(self) -> np.ndarray:
        """Model-averaged standard deviation per asset (pre-update)."""
        w = self._weights()
        var = np.sum(w * np.where(self.W > EPS, self.V / np.maximum(self.W, EPS), 0.0),
                     axis=0)
        return np.sqrt(np.maximum(var, VAR_FLOOR))

    def mean(self) -> np.ndarray:
        return np.where(self.muw > EPS, self.mu / np.maximum(self.muw, EPS), 0.0)

    def _weights(self) -> np.ndarray:
        lw = self.logw - self.logw.max(axis=0, keepdims=True)
        w = np.exp(lw)
        return w / np.maximum(w.sum(axis=0, keepdims=True), EPS)

    def effective_halflife(self) -> np.ndarray:
        return np.sum(self._weights() * self.h[:, None], axis=0)

    def update(self, r: np.ndarray) -> None:
        ok = np.isfinite(r)
        x = np.where(ok, r, 0.0)
        mu = self.mean()
        dev = np.where(ok, x - mu, 0.0)

        # score each memory on the return it did not see, then absorb it
        var = np.where(self.W > EPS, self.V / np.maximum(self.W, EPS), 0.0)
        var = np.maximum(var, VAR_FLOOR)
        ll = -0.5 * (np.log(2.0 * math.pi * var) + (dev[None, :] ** 2) / var)
        ready = (self.W > 1e-3) & ok[None, :]
        # 0.99: model-probability forgetting of Raftery et al. (2010),
        # the literature-standard value, applied uniformly.
        self.logw = np.where(ready, 0.99 * self.logw + ll, self.logw)
        self.logw -= self.logw.max(axis=0, keepdims=True)

        lam = self.lam[:, None]
        upd = ok[None, :]
        self.V = np.where(upd, lam * self.V + (1.0 - lam) * dev[None, :] ** 2, self.V)
        self.W = np.where(upd, lam * self.W + (1.0 - lam), self.W)
        self.mu = np.where(ok, self.mu_lam * self.mu + (1.0 - self.mu_lam) * x, self.mu)
        self.muw = np.where(ok, self.mu_lam * self.muw + (1.0 - self.mu_lam), self.muw)
        self.count += ok


# ===========================================================================
# Exponentially weighted pairwise-complete correlation
# ===========================================================================
class _CorrMemory:
    """One EW correlation estimator with its cleaned factor representation."""

    __slots__ = ("lam", "S", "W", "V", "lam_eig", "delta", "k", "C_ref",
                 "last_recompute", "logw", "n", "wbar")

    def __init__(self, n: int, halflife: float) -> None:
        self.lam = halflife_to_lambda(halflife)
        self.n = n
        self.wbar = 0.0
        self.S = np.zeros((n, n))
        self.W = np.zeros((n, n))
        self.V = np.zeros((n, K_MAX))     # top eigenvectors (columns)
        self.lam_eig = np.zeros(K_MAX)    # corresponding eigenvalues
        self.delta = 1.0                  # bulk (noise) eigenvalue
        self.k = 0
        self.C_ref = None
        self.last_recompute = -10**9
        self.logw = 0.0

    def absorb(self, outer: np.ndarray, mask_outer: np.ndarray,
               frac: float) -> None:
        lam = self.lam
        self.S *= lam
        self.S += (1.0 - lam) * outer
        self.W *= lam
        self.W += (1.0 - lam) * mask_outer
        self.wbar = lam * self.wbar + (1.0 - lam) * frac

    def correlation(self) -> np.ndarray:
        W = np.maximum(self.W, EPS)
        C = self.S / W
        d = np.sqrt(np.maximum(np.diag(C), VAR_FLOOR))
        C = np.clip(C / np.outer(d, d), -0.999, 0.999)
        np.fill_diagonal(C, 1.0)
        return C

    @property
    def t_eff(self) -> float:
        """Effective sample size implied by the weights actually accumulated.

        1/(1-lambda) is the asymptotic effective sample size of an
        exponentially weighted estimator; multiplying by the accumulated
        weight fraction handles the transient at the start of the record,
        where the estimator has seen far less than that.
        """
        cap = 1.0 / max(1.0 - self.lam, EPS)
        return float(min(max(self.wbar * cap, 1.0), cap))


def _clip_spectrum(C: np.ndarray, t_eff: float,
                   active: np.ndarray | None = None,
                   k_prev: int = 0,
                   ) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Marchenko-Pastur eigenvalue clipping.

    Returns (eigenvectors, eigenvalues, bulk level, k).  Eigenvalues above
    the MP upper edge are retained; the remainder are collapsed to their
    mean, which preserves the trace and leaves a positive-definite matrix.

    `active` restricts the decomposition to the instruments admitted so far.
    This matters for more than tidiness: the edge is (1 + sqrt(N/T))^2, so
    counting instruments that have never traded inflates N and moves the
    threshold.  Eigenvectors are scattered back to the full index with zeros
    on the inactive rows, so downstream indexing is unaffected.

    DO NOT REPLACE THE TRUNCATION WITH SHRINKAGE.  It was tried, measured, and
    reverted on 2026-08-17.  The argument for it is seductive: the edge moves
    with the panel's composition, so a hard rule is discontinuous in its own
    input, and weighting each eigenvalue by a sigmoid of its distance from the
    edge — lam~ = delta + rho * (lam - delta) — makes that discontinuity
    vanish.  It does.  It also makes the repainting WORSE.

    Refitting the real 10y macro panel with one predictor dropped:

                        median drift    p95      max
        hard truncation     0.56%      4.55%    13.0%
        sigmoid shrinkage   1.15%      7.74%    17.9%

    The reason is that truncation is not only a threshold, it is a NOISE
    FILTER.  Eigenvalues near the edge are poorly separated, so their
    eigenvectors have tiny eigengaps and rotate violently under any
    perturbation of the panel.  Admitting them at partial weight imports
    exactly that rotation into the fitted values; zeroing them suppressed it.

    The same measurement showed `k` differing on ONE row out of ~2350 between
    the two panels, so factor-count churn was never the mechanism.  Composition
    repainting here is eigenvector rotation, and it is a property of refitting
    a basis on a different panel — no estimator is invariant to its own input
    set.  The lever that works is the data contract (a fixed, declared
    universe), not the spectrum.
    """
    n_full = C.shape[0]
    if active is not None:
        idx = np.flatnonzero(active)
        if len(idx) < 3:
            return np.zeros((n_full, 0)), np.zeros(0), 1.0, 0
        Vs, ls, delta, k = _clip_spectrum(C[np.ix_(idx, idx)], t_eff, None, k_prev)
        V = np.zeros((n_full, Vs.shape[1]))
        V[idx, :] = Vs
        return V, ls, delta, k

    n = C.shape[0]
    C = 0.5 * (C + C.T)
    try:
        w, v = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        return np.zeros((n, 0)), np.zeros(0), 1.0, 0
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    q = n / max(t_eff, 1.0)
    edge = (1.0 + math.sqrt(q)) ** 2
    # HYSTERESIS. `k` selects the regressor set, so a bare `w > edge` makes the
    # model discontinuous in its own input: an eigenvalue resting ON the edge
    # flips k on a perturbation far too small to mean anything, and every
    # date's factor set changes with it. Requiring an eigenvalue to clear the
    # edge by a margin — and, once counted, to fall a matching margin below
    # before it is dropped — turns a knife-edge into a band. `k_prev` carries
    # the previous step's count so the band can be applied asymmetrically;
    # without it this is just a stricter threshold.
    #
    # The margin scales with the edge's own sampling noise, so it tightens as
    # the effective sample grows rather than being a hand-set constant.
    margin = MP_EDGE_HYSTERESIS / math.sqrt(max(t_eff, 1.0))
    if k_prev <= 0:
        k = int(np.sum(w > edge * (1.0 + margin)))
    else:
        n_up = int(np.sum(w > edge * (1.0 + margin)))     # decisively above
        n_dn = int(np.sum(w > edge * (1.0 - margin)))     # not yet decisively below
        k = min(max(k_prev, n_up), n_dn)
    # cap by estimability of the downstream regression, and by storage
    k = max(1, min(k, K_MAX, int(max(2, t_eff // 25))))

    total = float(np.sum(w))
    delta = (total - float(np.sum(w[:k]))) / max(n - k, 1)
    delta = max(delta, 1e-6)
    lam = w[:k].copy()

    # deterministic sign convention: the largest-magnitude loading is positive
    Vk = v[:, :k].copy()
    for j in range(k):
        idx = int(np.argmax(np.abs(Vk[:, j])))
        if Vk[idx, j] < 0:
            Vk[:, j] *= -1.0
    return Vk, lam, delta, k


def _align(V_new: np.ndarray, lam_new: np.ndarray, V_old: np.ndarray,
           k_old: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match new eigenvectors to the previous basis (order and sign).

    Without this, an eigenvalue crossing would silently relabel factor 2 as
    factor 3 and corrupt the cumulative factor levels.  Matching is greedy on
    |cos| similarity -- deterministic, and O(k^2) with k <= 20.
    """
    k_new = V_new.shape[1]
    if k_old == 0 or V_old.shape[1] == 0:
        return V_new, lam_new, np.arange(k_new)

    sim = np.abs(V_new.T @ V_old[:, :k_old])       # k_new x k_old
    order = np.full(k_new, -1, dtype=int)
    used_new, used_old = set(), set()
    pairs = sorted(
        ((sim[i, j], i, j) for i in range(k_new) for j in range(k_old)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    for s, i, j in pairs:
        if i in used_new or j in used_old or s < 0.3:
            continue
        order[i] = j
        used_new.add(i)
        used_old.add(j)
    free = [j for j in range(max(k_new, k_old)) if j not in used_old]
    fi = 0
    for i in range(k_new):
        if order[i] < 0:
            order[i] = free[fi] if fi < len(free) else i
            fi += 1

    slots = np.argsort(order)
    V_sorted = V_new[:, slots]
    lam_sorted = lam_new[slots]
    tgt = order[slots]
    for c in range(V_sorted.shape[1]):
        j = tgt[c]
        if j < k_old and float(V_sorted[:, c] @ V_old[:, j]) < 0:
            V_sorted[:, c] *= -1.0
    return V_sorted, lam_sorted, tgt


class OnlineFactorModel:
    """Recursive statistical factor model of the global cross-section.

    Per step the model emits, using information up to and including t:

    * ``scores``  -- unit-variance factor returns f_t
    * ``k``       -- number of factors above the Marchenko-Pastur edge
    * ``loadings``-- the current eigenbasis (columns)
    * ``contrib`` -- cumulative attribution of each factor level back to the
                     individual instruments that produced it

    The attribution is exact rather than approximate: factor levels are
    linear in standardised returns, so F_k,t = sum_i contrib[k, i] holds
    identically, and no post-hoc explainer is needed.
    """

    def __init__(self, n: int, memories: tuple[float, ...] = MEMORY_GRID) -> None:
        self.n = n
        self.mem = [_CorrMemory(n, h) for h in memories]
        self.memories = memories
        self.active = len(memories) - 1        # start at the longest memory
        #: Instruments admitted at least once so far.  Monotone by
        #: construction, so it can only grow forward in time -- which is what
        #: keeps the cross-sectional dimension free of hindsight.
        self.ever_active = np.zeros(n, dtype=bool)
        self.t = 0
        self.k = 0
        self.levels = np.zeros(K_MAX)          # cumulative factor levels
        self.contrib = np.zeros((K_MAX, n))    # per-instrument attribution
        self.last_scores = np.zeros(K_MAX)
        self._logw = np.zeros(len(memories))

    # -- internals ----------------------------------------------------------
    def _maybe_recompute(self, m: _CorrMemory) -> None:
        """Refresh the eigenbasis when the correlation matrix has moved by
        more than sampling noise, subject to a compute floor/ceiling."""
        # Floor on how often the eigenbasis may be rebuilt.  This is a compute
        # budget, not a model choice, and it scales with the cross-section
        # because eigendecomposition is O(N^3): at N ~ 330 an unthrottled
        # rebuild would dominate the run.  It is applied one-sidedly -- a held
        # basis is always a *past* one -- so it cannot repaint.
        gap = self.t - m.last_recompute
        if gap < max(5, int(self.ever_active.sum()) // 40) and m.k > 0:
            return
        C = m.correlation()
        t_eff = m.t_eff
        if m.C_ref is not None and gap < 63:
            denom = max(np.linalg.norm(m.C_ref), EPS)
            drift = float(np.linalg.norm(C - m.C_ref) / denom)
            if drift < 1.0 / math.sqrt(max(t_eff, 1.0)):
                return
        Vk, lk, delta, k = _clip_spectrum(C, t_eff, self.ever_active, m.k)
        Vk, lk, _ = _align(Vk, lk, m.V, m.k)
        m.V = np.zeros((self.n, K_MAX))
        m.V[:, :Vk.shape[1]] = Vk
        m.lam_eig = np.zeros(K_MAX)
        m.lam_eig[:len(lk)] = lk
        m.delta = delta
        m.k = int(Vk.shape[1])
        m.C_ref = C
        m.last_recompute = self.t

    @staticmethod
    def _loglik(m: _CorrMemory, z: np.ndarray, n_avail: int) -> float:
        """Gaussian log density under the cleaned factor covariance.

        Evaluated by the Woodbury identity, O(N k).  Used only to weight the
        memory bank; it never enters a published quantity, so the mild
        approximation of scaling the log-determinant by the available
        fraction is immaterial.
        """
        if m.k == 0 or n_avail < 5:
            return 0.0
        k = m.k
        V = m.V[:, :k]
        lam = np.maximum(m.lam_eig[:k], m.delta + 1e-9)
        p = V.T @ z
        quad = (float(z @ z) - float(np.sum((lam - m.delta) / lam * p * p))) / m.delta
        logdet = n_avail * math.log(m.delta) + float(np.sum(np.log(lam / m.delta)))
        return -0.5 * (logdet + quad + n_avail * math.log(2.0 * math.pi))

    # -- public -------------------------------------------------------------
    def update(self, z: np.ndarray, avail: np.ndarray) -> dict:
        """Absorb one cross-section of standardised returns.

        `z` is winsorised in place by the caller's contract; entries where
        `avail` is False are ignored entirely.
        """
        zz = np.where(avail, np.clip(z, -Z_CLIP, Z_CLIP), 0.0)
        zz = np.nan_to_num(zz, nan=0.0, posinf=0.0, neginf=0.0)
        mask = avail.astype(float)
        n_avail = int(mask.sum())
        self.ever_active |= avail
        n_active = max(int(self.ever_active.sum()), 1)

        # score each memory on data it has not yet absorbed
        if self.t > 0:
            lls = np.array([self._loglik(m, zz, n_avail) for m in self.mem])
            if np.all(np.isfinite(lls)):
                self._logw = 0.99 * self._logw + lls / max(n_avail, 1)
                self._logw -= self._logw.max()

        outer = np.outer(zz, zz)
        mask_outer = np.outer(mask, mask)
        # weight accumulation is measured against the admitted cross-section,
        # not the nominal universe width
        frac = n_avail / n_active
        for m in self.mem:
            m.absorb(outer, mask_outer, frac)
            self._maybe_recompute(m)

        w = np.exp(self._logw - self._logw.max())
        w = w / max(w.sum(), EPS)
        self.active = int(np.argmax(w))
        m = self.mem[self.active]

        k = m.k
        scores = np.zeros(K_MAX)
        if k > 0:
            V = m.V[:, :k]
            lam = np.maximum(m.lam_eig[:k], 1e-6)
            # Fraction of each factor's squared loading mass that actually
            # traded today.  Renormalising by it compensates for a closed
            # market rather than letting the factor shrink toward zero --
            # but only while enough of the factor is observable.  Below a
            # quarter of its mass, dividing through would amplify whatever
            # did trade into a move the factor did not make; the honest
            # reading of a factor whose constituents are shut is that it did
            # not move, so the score is zero and the level simply holds.
            observable = (V ** 2 * mask[:, None]).sum(axis=0)
            usable = observable >= 0.25
            norm = np.sqrt(np.maximum(observable, 1e-9))
            proj = (V * mask[:, None]).T @ zz
            denom = np.where(usable, norm * np.sqrt(lam), np.inf)
            s = proj / denom
            scores[:k] = s
            # exact per-instrument attribution of the factor increment
            inc = (V * mask[:, None]) * zz[:, None] / denom[None, :]
            self.contrib[:k] += inc.T
            self.levels[:k] += s

        self.k = k
        self.last_scores = scores
        self.t += 1
        return {
            "k": k,
            "scores": scores,
            "levels": self.levels.copy(),
            "memory": self.memories[self.active],
            "memory_weights": w,
            "t_eff": m.t_eff,
            "n_active": n_active,
            "explained": float(np.sum(m.lam_eig[:k]) / n_active) if k else 0.0,
        }

    def loadings(self) -> np.ndarray:
        m = self.mem[self.active]
        return m.V[:, :max(m.k, 1)].copy()

    def instrument_contribution(self) -> np.ndarray:
        """(K_MAX, N) cumulative attribution of factor levels to instruments."""
        return self.contrib.copy()
