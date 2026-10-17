"""
Causal online inference primitives.
=================================================================

Every estimator in this module is a *pure forward recursion*: its state at
time t is a deterministic function of observations 1..t only.  No estimator
here ever revises a value it has already emitted.  This is the mechanical
guarantee behind the system-level properties of causality, non-repainting
and revision invariance.

References
----------
West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*,
    2nd ed., Springer.  Chapters 4 & 6: the discounted Normal-Gamma DLM used
    by :class:`BatchDLM` (unknown observation variance, variance discounting
    in place of an explicit state-noise covariance Q).
Raftery, A. E., Karny, M. & Ettler, P. (2010). "Online Prediction Under
    Model Uncertainty via Dynamic Model Averaging." *Technometrics* 52(1).
    The recursive model-probability update used by :class:`DynamicModelAverage`.
Koop, G. & Korobilis, D. (2012). "Forecasting Inflation Using Dynamic Model
    Averaging." *International Economic Review* 53(3).
"""

from __future__ import annotations

import bisect
import math

import numpy as np

# ---------------------------------------------------------------------------
# Numerical floors.  These are machine-precision guards, not model parameters:
# they exist so that divisions and logs are defined, and their value cannot
# influence any statistically meaningful output.
# ---------------------------------------------------------------------------
EPS = 1e-12
VAR_FLOOR = 1e-10


def halflife_to_lambda(halflife: float) -> float:
    """EWMA decay with the given half-life in observations."""
    return float(np.exp(-np.log(2.0) / max(halflife, EPS)))


# ===========================================================================
# 1. Elementary streaming moments
# ===========================================================================
class EWMA:
    """Exponentially weighted mean and variance with bias correction.

    Bias correction (dividing by the accumulated weight) removes the
    initialisation transient without ever looking forward.
    """

    __slots__ = ("lam", "_m", "_v", "_w", "n")

    def __init__(self, halflife: float) -> None:
        self.lam = halflife_to_lambda(halflife)
        self._m = 0.0
        self._v = 0.0
        self._w = 0.0
        self.n = 0

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        lam = self.lam
        self._w = lam * self._w + (1.0 - lam)
        prev = self.mean
        self._m = lam * self._m + (1.0 - lam) * x
        self._v = lam * self._v + (1.0 - lam) * (x - prev) * (x - self.mean)
        self.n += 1

    @property
    def mean(self) -> float:
        return self._m / self._w if self._w > EPS else 0.0

    @property
    def var(self) -> float:
        return max(self._v / self._w, 0.0) if self._w > EPS else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


class ExpandingRank:
    """Causal empirical CDF: the rank of x among *strictly prior* observations.

    Distribution-free normalisation.  Because the reference set only ever
    grows forward in time, a value published at t is never re-scaled later.
    Insertion is O(n) worst case, which is irrelevant at daily frequency.
    """

    __slots__ = ("_sorted",)

    def __init__(self) -> None:
        self._sorted: list[float] = []

    def cdf(self, x: float) -> float:
        """P(X <= x) under the empirical law of the history seen so far."""
        n = len(self._sorted)
        if n == 0 or not np.isfinite(x):
            return 0.5
        lo = bisect.bisect_left(self._sorted, x)
        hi = bisect.bisect_right(self._sorted, x)
        # mid-rank handles ties symmetrically; (n+1) keeps the result in (0,1)
        return (0.5 * (lo + hi) + 0.5) / (n + 1.0)

    def update(self, x: float) -> None:
        if np.isfinite(x):
            bisect.insort(self._sorted, float(x))

    def quantile(self, q: float) -> float:
        n = len(self._sorted)
        if n == 0:
            return float("nan")
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return self._sorted[idx]

    @property
    def n(self) -> int:
        return len(self._sorted)


class OnlineAR1:
    """Recursive least squares for x_t = c + phi * x_{t-1} + e, with forgetting.

    Returns the coefficient together with its sampling standard error, which
    is what downstream engines need to form a probability statement about
    persistence or mean reversion.
    """

    __slots__ = ("lam", "_xx", "_xy", "_yy", "_w", "prev", "n")

    def __init__(self, halflife: float) -> None:
        self.lam = halflife_to_lambda(halflife)
        self._xx = np.zeros((2, 2))
        self._xy = np.zeros(2)
        self._yy = 0.0
        self._w = 0.0
        self.prev = None
        self.n = 0

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if self.prev is not None:
            u = np.array([1.0, self.prev])
            lam = self.lam
            self._xx = lam * self._xx + np.outer(u, u)
            self._xy = lam * self._xy + u * x
            self._yy = lam * self._yy + x * x
            self._w = lam * self._w + 1.0
            self.n += 1
        self.prev = float(x)

    def solve(self) -> tuple[float, float]:
        """(phi, standard error of phi).  NaN until identifiable."""
        if self.n < 8 or self._w < 4.0:
            return float("nan"), float("nan")
        xx = self._xx + np.eye(2) * 1e-8 * max(np.trace(self._xx), 1.0)
        try:
            inv = np.linalg.inv(xx)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta = inv @ self._xy
        rss = self._yy - beta @ self._xy
        dof = max(self._w - 2.0, 1.0)
        s2 = max(rss, 0.0) / dof
        se = math.sqrt(max(s2 * inv[1, 1], 0.0))
        return float(beta[1]), float(se)


# ===========================================================================
# 2. Discounted dynamic linear model  (West & Harrison 1997)
# ===========================================================================
class BatchDLM:
    """A bank of M discounted dynamic linear models, advanced together.

    Model (West & Harrison 1997, ch. 4), per member:

        y_t     = F_t' theta_t + nu_t ,      nu_t ~ N(0, V)
        theta_t = G theta_{t-1} + omega_t ,  omega_t ~ N(0, W_t)

    The state-noise covariance W_t is not estimated directly; it is induced
    by *discounting* the prior scale, R_t = G C_{t-1} G' / delta.  A single
    scalar delta in (0,1] therefore indexes the whole family from "the
    coefficients are constant" (delta = 1, i.e. recursive least squares) to
    "they move freely", which is what lets adaptation speed be inferred
    rather than declared.  V is unknown and handled conjugately
    (Normal-Gamma), so the one-step predictive is Student-t -- heavy tails
    come for free, and financial residuals have them.

    G = I gives a time-varying-parameter regression; G = [[1,1],[0,1]] with
    F = (1,0) gives the local linear trend of Harvey (1989).

    Every member may carry its own discount factor *and* its own regressor
    vector, which is what the leave-one-block-out ablation needs.  The
    batching is not an optimisation detail: the engines advance banks of
    5-15 filters at each of several thousand timestamps, and a Python-level
    loop over them dominates the runtime.

    Everything emitted at time t is a function of y_{1:t-1} and F_t.
    """

    def __init__(self, m: int, k: int, deltas, prior_scale: float = 1.0,
                 prior_var: float = 1.0, prior_mean: np.ndarray | None = None,
                 n0: float = 1.0, G: np.ndarray | None = None) -> None:
        self.m = m
        self.k = k
        #: Optional state-evolution matrix.  G = I is a time-varying-parameter
        #: regression; G = [[1,1],[0,1]] with F = (1,0) is the local linear
        #: trend of Harvey (1989), which is how the dynamics engine separates
        #: level from slope.
        self.G = None if G is None else np.asarray(G, dtype=float)
        self.delta = np.asarray(deltas, dtype=float).reshape(m)
        self.M = np.zeros((m, k))
        if prior_mean is not None:
            self.M[:] = np.asarray(prior_mean, dtype=float).reshape(1, k)
        self.C = np.tile(np.eye(k) * float(prior_scale), (m, 1, 1))
        self.n = np.full(m, float(n0))
        self.S = np.full(m, float(prior_var))
        self._c_max = float(prior_scale) * 1e6
        self.log_pred_lik = np.zeros(m)

    def _F(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        return np.broadcast_to(F, (self.m, self.k)) if F.ndim == 1 else F

    def _evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """Prior moments for time t given the posterior at t-1."""
        if self.G is None:
            return self.M, self.C / self.delta[:, None, None]
        a = self.M @ self.G.T
        R = np.einsum("ij,mjk,lk->mil", self.G, self.C, self.G)
        return a, R / self.delta[:, None, None]

    def forecast(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = self._F(F)
        a, R = self._evolve()
        f = np.einsum("mk,mk->m", a, F)
        RF = np.einsum("mij,mj->mi", R, F)
        Q = np.maximum(np.einsum("mi,mi->m", F, RF) + self.S, VAR_FLOOR)
        return f, Q, self.n

    def update(self, F: np.ndarray, y: float) -> tuple[np.ndarray, np.ndarray]:
        F = self._F(F)
        a, R = self._evolve()
        f = np.einsum("mk,mk->m", a, F)
        RF = np.einsum("mij,mj->mi", R, F)
        Q = np.maximum(np.einsum("mi,mi->m", F, RF) + self.S, VAR_FLOOR)

        if not np.isfinite(y):
            self.log_pred_lik = np.full(self.m, -np.inf)
            d = np.einsum("mii->mi", R)
            np.clip(d, VAR_FLOOR, self._c_max, out=d)
            self.M = a
            self.C = R
            return f, Q

        e = y - f
        dof = self.n
        # Huberised innovation.  The conjugate Normal-Gamma variance update
        # multiplies S by (1 + (z^2 - 1)/n), so one wild standardised
        # innovation can inflate the variance estimate by orders of magnitude
        # and, through the (S_new/S) factor on the scale matrix, take the
        # whole filter with it -- an overflow observed on real data before
        # this bound was added.  Capping the influence of a single
        # observation at ten predictive standard deviations is the standard
        # robustification (Masreliez 1975; West & Harrison's monitoring and
        # intervention). Under the model an innovation that large is not
        # evidence about the variance, it is an outlier.
        z = e / np.sqrt(Q)
        z = np.clip(z, -10.0, 10.0)
        e = z * np.sqrt(Q)
        z2 = z * z
        self.log_pred_lik = (
            _lgamma(0.5 * (dof + 1.0)) - _lgamma(0.5 * dof)
            - 0.5 * np.log(np.pi * dof * Q)
            - 0.5 * (dof + 1.0) * np.log1p(z2 / dof)
        )
        A = RF / Q[:, None]
        n_new = dof + 1.0
        S_new = np.maximum(self.S + (self.S / n_new) * (z2 - 1.0), VAR_FLOOR)
        self.M = a + A * e[:, None]
        C_new = (S_new / self.S)[:, None, None] * (
            R - A[:, :, None] * A[:, None, :] * Q[:, None, None])
        C_new = 0.5 * (C_new + np.transpose(C_new, (0, 2, 1)))
        d = np.einsum("mii->mi", C_new)
        np.clip(d, VAR_FLOOR, self._c_max, out=d)
        # Bound the whole matrix, not just its diagonal.  Clipping only the
        # diagonal of a matrix whose off-diagonals keep growing destroys
        # positive-definiteness; rescaling preserves the correlation
        # structure while keeping the magnitude finite.  A regressor that is
        # identically zero for years -- a latent factor that has not yet
        # emerged -- otherwise sees its prior variance grow like delta^-t.
        scale = np.max(np.abs(C_new), axis=(1, 2))
        over = scale > self._c_max
        if np.any(over):
            C_new[over] *= (self._c_max / scale[over])[:, None, None]
        self.C = C_new
        self.n = n_new
        self.S = S_new
        return f, Q


_lgamma = np.vectorize(math.lgamma, otypes=[float])


class DynamicModelAverage:
    """Bank of discounted DLMs averaged by predictive likelihood.

    Implements the recursive model-probability update of Raftery, Karny &
    Ettler (2010):

        pi_{t|t-1,i} proportional to pi_{t-1|t-1,i}^alpha
        pi_{t|t,i}   proportional to pi_{t|t-1,i} * p_i(y_t | y_{1:t-1})

    The exponent alpha < 1 lets the weighting adapt when the appropriate
    degree of parameter variation changes with the regime.  This removes the
    lookback/adaptation-speed choice from the analyst: the data selects it,
    forward in time, and the selection at t never rewrites the selection at
    t-1.
    """

    #: Discount grid.  Spacing is logarithmic in the implied memory
    #: 1/(1-delta): ~20, 33, 50, 100, 200, 1000 observations, plus the
    #: constant-coefficient limit.  A grid is not a tuned parameter -- it is
    #: the support of the prior over adaptation speed.
    DEFAULT_GRID = (0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0)

    def __init__(self, k: int, grid: tuple[float, ...] | None = None,
                 alpha: float = 0.99, prior_scale: float = 1.0,
                 prior_var: float = 1.0,
                 prior_mean: np.ndarray | None = None,
                 G: np.ndarray | None = None) -> None:
        self.grid = tuple(grid) if grid is not None else self.DEFAULT_GRID
        self.bank = BatchDLM(len(self.grid), k, self.grid, prior_scale,
                             prior_var, prior_mean, G=G)
        self.w = np.full(len(self.grid), 1.0 / len(self.grid))
        self.alpha = float(alpha)
        self.k = k
        self.n_obs = 0

    def forecast(self, F: np.ndarray) -> tuple[float, float]:
        """Mixture predictive mean and variance (law of total variance)."""
        fs, qs, dofs = self.bank.forecast(F)
        # Student-t variance inflation (dof/(dof-2)); finite for dof > 2
        infl = np.where(dofs > 2.0, dofs / np.maximum(dofs - 2.0, EPS), 3.0)
        mean = float(self.w @ fs)
        var = float(self.w @ (qs * infl) + self.w @ (fs - mean) ** 2)
        return mean, max(var, VAR_FLOOR)

    def update(self, F: np.ndarray, y: float) -> tuple[float, float]:
        mean, var = self.forecast(F)
        self.bank.update(F, y)
        if np.isfinite(y):
            ll = np.where(np.isfinite(self.bank.log_pred_lik),
                          self.bank.log_pred_lik, -1e6)
            logw = self.alpha * np.log(np.maximum(self.w, 1e-300)) + ll
            logw -= logw.max()
            w = np.exp(logw)
            # forgetting floor keeps every model recoverable (Raftery et al.)
            w = np.maximum(w / w.sum(), 1e-6)
            self.w = w / w.sum()
            self.n_obs += 1
        return mean, var

    # -- diagnostics --------------------------------------------------------
    @property
    def log_pred_lik(self) -> np.ndarray:
        return self.bank.log_pred_lik

    @property
    def coef(self) -> np.ndarray:
        """Model-averaged coefficient vector."""
        return self.w @ self.bank.M

    @property
    def coef_var(self) -> np.ndarray:
        """Model-averaged marginal posterior variances of the coefficients.

        Law of total variance across the bank: each member's own posterior
        variance plus the spread of the members' means. ``BatchDLM.C`` is
        already the posterior scale in the data's units — the conjugate update
        rescales it by ``S_new / S`` every step (West & Harrison §4.6) — so it
        is NOT multiplied by ``S`` again. (It was, before this property had a
        consumer: harmless while nothing read it, a variance in squared units
        of the observation variance the moment anything did.)
        """
        mu = self.coef
        diag = np.einsum("mii->mi", self.bank.C)
        return self.w @ (diag + (self.bank.M - mu) ** 2)

    def forecast_parts(self, F: np.ndarray) -> tuple[float, float, float]:
        """Mixture predictive ``(mean, total variance, parameter variance)``.

        The total is what :meth:`forecast` returns: the uncertainty about the
        next OBSERVATION. The parameter part drops each member's observation
        noise ``S`` and keeps only ``F'RF`` — the uncertainty about the
        fitted LEVEL itself — plus the members' disagreement. For a valuation
        regression those are different claims: one says how far price
        typically sits from fair value, the other how precisely fair value is
        known.
        """
        fs, qs, dofs = self.bank.forecast(F)
        infl = np.where(dofs > 2.0, dofs / np.maximum(dofs - 2.0, EPS), 3.0)
        mean = float(self.w @ fs)
        spread = float(self.w @ (fs - mean) ** 2)
        var = float(self.w @ (qs * infl)) + spread
        par = np.maximum(qs - self.bank.S, 0.0)
        var_par = float(self.w @ (par * infl)) + spread
        return mean, max(var, VAR_FLOOR), max(var_par, 0.0)

    @property
    def effective_memory(self) -> float:
        """Posterior-mean adaptation memory 1/(1-delta), in observations."""
        mem = np.array([1.0 / max(1.0 - d, 1e-4) for d in self.grid])
        return float(self.w @ mem)


def norm_cdf_scalar(x: float) -> float:
    """Standard normal CDF of a scalar (no SciPy in the hot loops)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
