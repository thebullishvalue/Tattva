"""
Tattva — the adaptive layer: constants replaced by causal online estimates.
तत्त्व (Tattva) — "Principle / Essence"

Every quantity this module produces used to be a hand-set number in
``core.config``: a classification cut-point ("STRONG BUY below −0.404"), a
display tier ("model spread above 29.92 bps is red"). Between them they
carried much of the system's behaviour, and they had two problems that turn
out to be the same problem.

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
threshold becomes the causal empirical quantile of the signal's own past.
Nothing is fitted to data that had not happened yet, so a value published on
day *t* is never revised, and the same input prefix always produces the same
output prefix — the reproducibility property the test suite asserts.

What this deliberately does NOT auto-derive: the horizon
(``forecast_horizon``) is a product decision about what you intend to trade,
not a statistical estimate; the FVO burn-in and print floor are estimability
floors argued from first principles; and the view/discount *banks* stay
declared, because a bank is a hypothesis space to average over, not a value to
pick. The distinction throughout is between choosing a number and choosing a
question. (An online skill-weighter of view banks lived here too; its
sign-hit loss rewarded the drift side, it added no measurable IC over equal
weights, and it went with the learners it served.)
"""

from __future__ import annotations

import numpy as np

# One implementation of each causal primitive, in analytics/ where the generic
# math lives. They arrived with the Mūla port and briefly sat under engines/fvo/
# (now engines/mula/),
# which pointed the dependency the wrong way — analytics is the layer engines
# are built ON, so anything in it importing from an engine is a cycle waiting
# to happen, and duly was.
from analytics.causal import ExpandingRank

__all__ = [
    "AdaptiveThreshold",
    "adaptive_tiers",
    "tier_now",
    "expanding_quantile_series",
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
