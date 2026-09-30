"""
Tattva — the adaptive layer: constants replaced by causal online estimates.
तत्त्व (Tattva) — "Principle / Essence"

Every quantity this module produces used to be a hand-set number in
``core.config``: a classification cut-point ("STRONG BUY below −0.404"), a
display tier ("model spread above 29.92 bps is red"), a dimension weight
("direction counts 0.50"). Between them they carried most of the system's
behaviour, and they had two problems that turn out to be the same problem.

**They had to be tuned.** Each one was anchored by a research study that swept
it over the full history and read off a p75 or an argmax. That is a lot of
machinery to maintain, it goes stale the moment the instrument's distribution
moves, and — as the tuning reports themselves kept concluding — most of the
sweeps were reading noise: a grid whose entire column sat within one standard
error of zero still produces an argmax, and an argmax still looks like an
answer.

**They repainted.** A constant fitted on all of history and then applied to all
of history means today's data decided how 2019 was classified. Add one
session, re-run the study, and the historical record changes underneath you.
The same is true of anything learned by fitting the whole sample at once —
which is precisely what the Optuna calibration this module replaced was doing.

The fix for both is one idea, and it is the idea the FVO valuation engine
already runs on: **estimate, don't tune, and estimate one-sidedly.** A
threshold becomes the causal empirical quantile of the signal's own past. A
weight becomes the exponentially-discounted predictive skill of the thing being
weighted, accumulated forward. Nothing is fitted to data that had not happened
yet, so a value published on day *t* is never revised, and the same input
prefix always produces the same output prefix — the reproducibility property
``research/test_reproducibility.py`` asserts.

What this deliberately does NOT auto-derive: horizons (``forecast_horizon``,
``precedent_horizons``) are product decisions about what you intend to trade,
not statistical estimates; the FVO burn-in and print floor are estimability
floors argued from first principles; and the view/discount *banks* stay
declared, because a bank is a hypothesis space to average over, not a value to
pick. The distinction throughout is between choosing a number and choosing a
question.

References
----------
Raftery, A. E., Kárný, M. & Ettler, P. (2010). "Online Prediction Under Model
    Uncertainty via Dynamic Model Averaging." *Technometrics* 52(1).
Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and Games.*
"""

from __future__ import annotations

import math

import numpy as np

# One implementation of each causal primitive, in analytics/ where the generic
# math lives. They arrived with the Mūla port and briefly sat under engines/fvo/
# (now engines/mula/),
# which pointed the dependency the wrong way — analytics is the layer engines
# are built ON, so anything in it importing from an engine is a cycle waiting
# to happen, and duly was.
from analytics.causal import EPS, ExpandingRank

__all__ = [
    "AdaptiveThreshold",
    "OnlineSkillWeights",
    "adaptive_tiers",
    "tier_now",
    "expanding_quantile_series",
    "expanding_symmetric_tiers",
]


class AdaptiveThreshold:
    """A cut-point that is the causal empirical quantile of its own history.

    Feed it the signal one observation at a time; ask for the threshold BEFORE
    absorbing the current value. The reference set only grows forward, so a
    threshold used at *t* depends on nothing after *t*.

    ``min_obs`` guards the opening stretch, where an empirical quantile of a
    handful of points is noise pretending to be a level. Below it the caller's
    ``fallback`` is returned — the former hand-set constant is exactly the
    right thing to pass, so the system's early behaviour is unchanged and the
    estimate takes over only once it is better informed than the prior.

    A count guard is not sufficient on its own, which is why ``quantile`` also
    refuses a DEGENERATE estimate. Several signals here begin life pinned at
    exactly zero — a DDM state before evidence arrives, breadth before any
    lookback window has formed — so a p50 taken over 252 such observations is
    a perfectly well-sampled 0.0. Returned as a threshold it collapses the
    neutral band to the single point {0}, and every nonzero reading is then
    classified as extreme. Measured on Gold: 270 rows where the estimated
    tiers were 0.0, of which 36 published a "STRONGLY" regime label they had
    not earned. A non-positive quantile of an absolute signal means "this has
    not moved yet", and the honest answer to that is the prior.
    """

    __slots__ = ("_rank", "min_obs")

    def __init__(self, min_obs: int = 252) -> None:
        self._rank = ExpandingRank()
        self.min_obs = int(min_obs)

    def quantile(self, q: float, fallback: float = float("nan")) -> float:
        if self._rank.n < self.min_obs:
            return fallback
        est = self._rank.quantile(float(np.clip(q, 0.0, 1.0)))
        # Degenerate estimate (see class docstring): the signal has been flat,
        # so its quantile carries no scale information. Defer to the prior.
        if not np.isfinite(est) or est <= 0.0:
            return fallback
        return est

    def cdf(self, x: float) -> float:
        """Where does x sit in its own past distribution? In (0, 1)."""
        return self._rank.cdf(x)

    def update(self, x: float) -> None:
        self._rank.update(x)

    @property
    def n(self) -> int:
        return self._rank.n


class OnlineSkillWeights:
    """Weight a bank of views by their exponentially-discounted predictive skill.

    Each member proposes a signed signal; some time later the outcome arrives.
    A member's weight is driven by the sign agreement between what it said and
    what happened, accumulated with a forgetting factor so a view that has
    stopped working fades instead of being averaged over forever.

    Why sign agreement and not squared error: the members here are oscillators
    on different scales (an MSF at a 8-day lookback and one at 52 days do not
    produce comparable magnitudes), and the decision the system takes from them
    is directional. Scoring magnitude would mostly rank the members by their
    volatility. This is the same reason the system reports rank IC rather than
    R² for a directional signal.

    Weights are a softmax over each member's discounted MEAN agreement, scaled
    by the square root of its effective sample size. That scaling is the whole
    design and it is worth being explicit about, because the obvious
    alternative is wrong: accumulating a running SUM of per-round evidence (the
    textbook dynamic-model-averaging recursion) makes the log-weight gap grow
    like ``mean_edge / (1 - lambda)``, which at a 252-day half-life is a factor
    of ~364. A member hitting a thoroughly ordinary 55% of calls against
    another's 45% would then be weighted ~e^36 higher — total collapse onto one
    view off a difference that is not close to significant.

    Scaling the mean by ``sqrt(n_eff)`` instead means the weight gap tracks the
    *t-statistic* of the skill difference rather than the raw round count, so
    concentration requires evidence rather than merely time. ``kappa`` sets how
    many nats one standard error of edge is worth.

    Weights are floored so no member is ever fully silenced — a bank that
    collapses onto one view has stopped being an ensemble, and the member that
    looks worst over one regime is often the one that carries the next.
    """

    __slots__ = ("names", "_abar", "_wsum", "_wsq", "lam", "floor", "kappa",
                 "_n_scored")

    def __init__(self, names: list[str], halflife: float = 252.0,
                 floor: float = 0.02, kappa: float = 0.5) -> None:
        self.names = list(names)
        self._abar = np.zeros(len(self.names))   # discounted MEAN agreement
        self._wsum = 0.0                          # sum of discount weights
        self._wsq = 0.0                           # sum of squared weights
        self.lam = math.exp(-math.log(2.0) / max(float(halflife), 1.0))
        self.floor = float(floor)
        self.kappa = float(kappa)
        self._n_scored = 0

    @property
    def effective_obs(self) -> float:
        """Kish effective sample size of the discounted evidence."""
        return float(self._wsum ** 2 / self._wsq) if self._wsq > EPS else 0.0

    def weights(self) -> np.ndarray:
        """Current weights — a function of evidence absorbed so far only."""
        n = len(self._abar)
        if not n:
            return np.array([])
        logw = self.kappa * math.sqrt(max(self.effective_obs, 0.0)) * self._abar
        w = np.exp(logw - logw.max())
        w = w / max(w.sum(), EPS)
        if self.floor > 0:
            # Mix with uniform rather than clipping-then-renormalising: the
            # latter pushes the clipped members back BELOW the floor when the
            # sum is rescaled, so the floor it advertises would not hold. This
            # form gives w_i >= floor exactly, and sums to 1 by construction.
            k = min(self.floor * n, 1.0)
            w = (1.0 - k) * w + k / n
        return w

    def weight_map(self) -> dict[str, float]:
        return dict(zip(self.names, self.weights()))

    def observe(self, signals: np.ndarray, outcome: float) -> None:
        """Absorb one scored round: what each member said, and what happened.

        ``signals`` is the members' signed calls at the decision time,
        ``outcome`` the realised signed move over the horizon. Non-finite or
        flat entries score neutral rather than being punished — an abstaining
        view is not a wrong view.
        """
        if not np.isfinite(outcome) or outcome == 0.0:
            return
        s = np.asarray(signals, dtype=np.float64)
        if s.shape != self._abar.shape:
            return
        agree = np.sign(s) * np.sign(outcome)
        agree = np.where(np.isfinite(agree), agree, 0.0)
        # Discounted mean: abar <- (lam*wsum*abar + agree) / (lam*wsum + 1)
        prev_w = self.lam * self._wsum
        new_w = prev_w + 1.0
        self._abar = (prev_w * self._abar + agree) / new_w
        self._wsum = new_w
        self._wsq = (self.lam ** 2) * self._wsq + 1.0
        self._n_scored += 1

    @property
    def n_scored(self) -> int:
        return self._n_scored

    @property
    def mean_agreement(self) -> dict[str, float]:
        """Each member's discounted mean directional agreement, in [-1, +1]."""
        return dict(zip(self.names, self._abar))

    @property
    def effective_n(self) -> float:
        """Participation ratio of the weights — how many views are really voting."""
        w = self.weights()
        return float(1.0 / max(np.sum(w ** 2), EPS)) if len(w) else 0.0


def expanding_quantile_series(
    values: np.ndarray, q: float, min_obs: int = 252,
    fallback: float | None = None,
) -> np.ndarray:
    """Batch causal expanding quantile: out[t] = quantile(values[:t], q).

    Note the exclusive slice — the threshold at *t* is built from values
    STRICTLY before *t*, so a point is never compared against a distribution it
    is itself a member of. Rows before ``min_obs`` take ``fallback`` (or NaN).

    This is the batch form of :class:`AdaptiveThreshold` and it is implemented
    by literally running that class, so the two agree by construction rather
    than by inspection. A pandas ``.expanding().quantile()`` would be faster
    but uses linear interpolation between order statistics where
    ``ExpandingRank`` uses nearest-rank; the two disagree by ~1e-2 on a unit-
    variance signal, which is enough to move a classification near a cut-point.
    One definition of "the p90 of the past" is worth more than the speed.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    out = np.full(n, np.nan if fallback is None else float(fallback))
    if n == 0:
        return out
    fb = np.nan if fallback is None else float(fallback)
    thr = AdaptiveThreshold(min_obs=min_obs)
    for t in range(n):
        out[t] = thr.quantile(q, fallback=fb)
        thr.update(v[t])
    return out


def adaptive_tiers(
    values: np.ndarray,
    priors: dict[str, float],
    quantiles: dict[str, float] | None = None,
    min_obs: int = 252,
) -> dict[str, np.ndarray]:
    """Per-row tier levels on ``|values|``, each from that row's own past.

    ``priors`` maps a tier name to the constant that used to BE that tier; the
    same keys come back mapped to arrays. ``quantiles`` says which quantile of
    the absolute signal each tier represents — defaulting to the p90/p75/p50
    convention the hand-set constants were anchored to, so an instrument with
    a distribution like the pooled one lands in the same place the constant
    did, and one with a different distribution does not.

    This is the batch primitive behind every classification the system
    publishes. Because each row's level is built from strictly earlier rows,
    re-running on more data cannot move a label that was already published.
    """
    q = {"strong": 0.90, "moderate": 0.75, "weak": 0.50}
    if quantiles:
        q.update(quantiles)
    a = np.abs(np.asarray(values, dtype=np.float64))
    out = {
        name: expanding_quantile_series(a, q.get(name, 0.75), min_obs,
                                        fallback=float(prior))
        for name, prior in priors.items()
    }
    # Enforce the tier ordering. Quantiles of one sample are monotone in q by
    # construction, but a row where one tier fell back to its prior and another
    # did not can invert — and an inverted pair is not a degraded
    # classification, it is an incoherent one (a reading simultaneously above
    # "strong" and below "weak"). Ordered high→low, each tier held at or below
    # the one above it.
    ordered = sorted(out, key=lambda k: -q.get(k, 0.75))
    for hi, lo in zip(ordered, ordered[1:]):
        out[lo] = np.minimum(out[lo], out[hi])
    return out


def tier_now(
    values: np.ndarray,
    prior: float,
    q: float = 0.90,
    min_obs: int = 252,
) -> float:
    """The tier level as of the LAST row — built from everything before it.

    For display code that colours a single current reading. Falls back to
    ``prior`` while history is short. Kept separate from
    :func:`adaptive_tiers` because a card needs one number, not a column, and
    routing it through the array form would invite someone to index ``[-1]``
    of a series that included the very value being classified.
    """
    a = np.abs(np.asarray(values, dtype=np.float64))
    a = a[np.isfinite(a)]
    if len(a) < min_obs:
        return float(prior)
    thr = AdaptiveThreshold(min_obs=min_obs)
    for v in a[:-1]:          # strictly prior to the value being classified
        thr.update(v)
    # quantile() already refuses a degenerate (non-positive) estimate.
    return float(thr.quantile(q, fallback=float(prior)))


def expanding_symmetric_tiers(
    values: np.ndarray, q_strong: float = 0.90, q_moderate: float = 0.75,
    min_obs: int = 252, fallback_strong: float = float("nan"),
    fallback_moderate: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """Causal ``(strong, moderate)`` tiers from |signal|'s own past.

    Tiers are taken on the ABSOLUTE signal and applied symmetrically, which is
    the convention the hand-set constants already followed (they were quoted as
    "p75/p90 of |x|"). Using |x| rather than the two tails separately means the
    opening stretch needs half as much history to be informative, and it cannot
    produce the pathological state where the buy tier sits above the sell tier
    because one side happened to be quiet.
    """
    a = np.abs(np.asarray(values, dtype=np.float64))
    return (
        expanding_quantile_series(a, q_strong, min_obs, fallback_strong),
        expanding_quantile_series(a, q_moderate, min_obs, fallback_moderate),
    )
