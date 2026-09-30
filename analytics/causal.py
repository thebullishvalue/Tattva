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
Opper, M. (1998). "A Bayesian Approach to Online Learning." In *Online
    Learning in Neural Networks*.  Assumed-density filtering, the basis of
    :class:`OnlineLogistic`.
Spiegelhalter, D. J. & Lauritzen, S. L. (1990). "Sequential updating of
    conditional probabilities on directed graphical structures." *Networks* 20.
Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of
    Squares and Products." *Technometrics* 4(3).
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


def safe_div(a, b, fill=0.0):
    """Elementwise division that returns `fill` where the denominator vanishes."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    out = np.full(np.broadcast(a, b).shape, fill, dtype=float)
    ok = np.abs(b) > EPS
    np.divide(a, b, out=out, where=ok)
    return out


# ===========================================================================
# 1. Elementary streaming moments
# ===========================================================================
class Welford:
    """Expanding-window mean/variance (Welford 1962), numerically stable.

    ``mean``/``var`` always reflect observations already absorbed, so reading
    them *before* calling :meth:`update` yields a strictly causal statistic.
    """

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))


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
        """Model-averaged marginal coefficient variances (incl. model spread)."""
        mu = self.coef
        diag = np.einsum("mii->mi", self.bank.C)
        return self.w @ (diag * self.bank.S[:, None] + (self.bank.M - mu) ** 2)

    @property
    def effective_memory(self) -> float:
        """Posterior-mean adaptation memory 1/(1-delta), in observations."""
        mem = np.array([1.0 / max(1.0 - d, 1e-4) for d in self.grid])
        return float(self.w @ mem)

    @property
    def obs_var(self) -> float:
        return float(self.w @ self.bank.S)


# ===========================================================================
# 3. Bayesian online logistic regression (assumed density filtering)
# ===========================================================================
class OnlineLogistic:
    """Sequential Bayesian logistic regression with parameter drift.

    Gaussian posterior on the weights, propagated by a discount factor and
    updated by a single assumed-density-filtering step (Opper 1998;
    Spiegelhalter & Lauritzen 1990).  Predictions are made by the standard
    probit approximation to the logistic-Gaussian integral (MacKay 1992),
    which propagates *parameter* uncertainty into the emitted probability --
    the quantity the decision layer needs.
    """

    def __init__(self, k: int, delta: float = 0.995, prior_scale: float = 1.0) -> None:
        self.k = k
        self.delta = float(delta)
        self.m = np.zeros(k)
        self.P = np.eye(k) * float(prior_scale)
        self.n_obs = 0

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """(calibrated probability, latent-score variance)."""
        P = self.P / self.delta
        mu = float(x @ self.m)
        s2 = float(x @ P @ x)
        kappa = 1.0 / math.sqrt(1.0 + math.pi * s2 / 8.0)
        p = 1.0 / (1.0 + math.exp(-np.clip(kappa * mu, -30.0, 30.0)))
        return p, max(s2, 0.0)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        p, s2 = self.predict(x)
        self.P = self.P / self.delta
        if np.isfinite(y):
            var = max(p * (1.0 - p), 1e-6)
            Px = self.P @ x
            denom = 1.0 / var + float(x @ Px)
            K = Px / max(denom, EPS)
            self.m = self.m + K * (y - p)
            self.P = self.P - np.outer(K, Px)
            self.P = 0.5 * (self.P + self.P.T)
            d = np.diag(self.P).copy()
            np.fill_diagonal(self.P, np.maximum(d, 1e-9))
            self.n_obs += 1
        return p, s2


# ===========================================================================
# 4. Streaming skill accounting
# ===========================================================================
class OnlineCorr:
    """Forgetting-weighted correlation between a signal and a later outcome.

    Used to turn "confidence" from an assertion into a measurement: the
    signal at t-1 is correlated with the outcome at t, so the statistic is a
    demonstrated association, evaluated out of sample by construction.
    """

    __slots__ = ("lam", "_xy", "_xx", "_yy", "_x", "_y", "_w")

    def __init__(self, halflife: float = 504.0) -> None:
        self.lam = halflife_to_lambda(halflife)
        self._xy = self._xx = self._yy = self._x = self._y = self._w = 0.0

    def update(self, x: float, y: float) -> None:
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        lam = self.lam
        self._xy = lam * self._xy + x * y
        self._xx = lam * self._xx + x * x
        self._yy = lam * self._yy + y * y
        self._x = lam * self._x + x
        self._y = lam * self._y + y
        self._w = lam * self._w + 1.0

    @property
    def n_eff(self) -> float:
        return float(self._w)

    @property
    def rho(self) -> float:
        w = self._w
        if w < 5.0:
            return 0.0
        cxy = self._xy - self._x * self._y / w
        cxx = self._xx - self._x * self._x / w
        cyy = self._yy - self._y * self._y / w
        den = math.sqrt(max(cxx, 0.0) * max(cyy, 0.0))
        return float(cxy / den) if den > EPS else 0.0

    @property
    def evidence(self) -> float:
        """P(rho > 0) under the Fisher-z sampling law: a calibrated
        confidence in the association, not a rescaled magnitude."""
        w = self.n_eff
        if w < 10.0:
            return 0.5
        r = min(max(self.rho, -0.999), 0.999)
        z = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(w - 3.0, 1.0))
        return norm_cdf_scalar(z)


class OnlineSkill:
    """Discounted out-of-sample skill of a stream of one-step predictions.

    Tracks a forgetting-weighted R^2 against the unconditional-mean benchmark
    and a directional hit rate.  Both are computed from predictions that were
    made strictly before the corresponding outcome, so the resulting
    "confidence" is a demonstrated, not assumed, quantity.
    """

    __slots__ = ("lam", "_se", "_sv", "_w", "_hit", "_hw", "_ymean")

    def __init__(self, halflife: float = 252.0) -> None:
        self.lam = halflife_to_lambda(halflife)
        self._se = 0.0
        self._sv = 0.0
        self._w = 0.0
        self._hit = 0.0
        self._hw = 0.0
        self._ymean = EWMA(halflife)

    def update(self, pred: float, actual: float) -> None:
        if not (np.isfinite(pred) and np.isfinite(actual)):
            return
        lam = self.lam
        base = self._ymean.mean
        self._se = lam * self._se + (actual - pred) ** 2
        self._sv = lam * self._sv + (actual - base) ** 2
        self._w = lam * self._w + 1.0
        if abs(pred) > EPS:
            self._hit = lam * self._hit + (1.0 if np.sign(pred) == np.sign(actual) else 0.0)
            self._hw = lam * self._hw + 1.0
        self._ymean.update(actual)

    @property
    def r2(self) -> float:
        if self._sv <= EPS:
            return 0.0
        return float(1.0 - self._se / self._sv)

    @property
    def hit_rate(self) -> float:
        return float(self._hit / self._hw) if self._hw > EPS else 0.5

    @property
    def n_eff(self) -> float:
        return float(self._w)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def norm_cdf(x):
    """Standard normal CDF (vectorised, no SciPy dependency in hot loops)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def norm_cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probit(u: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 in the central region, which is far beyond what an
    empirical CDF built from a few thousand points can resolve.
    """
    u = min(max(float(u), 1e-6), 1.0 - 1e-6)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl, ph = 0.02425, 1.0 - 0.02425
    if u < pl:
        q = math.sqrt(-2.0 * math.log(u))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if u > ph:
        q = math.sqrt(-2.0 * math.log(1.0 - u))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = u - 0.5
    rr = q * q
    return (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q / \
           (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1.0)
