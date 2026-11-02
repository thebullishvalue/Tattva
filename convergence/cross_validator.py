"""
Tattva — CrossValidator: Cross-referencing Mūla and Swayam outputs.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — how far the two engines agree, dimension by dimension, and which
way the agreement points.

Every input is first turned into a LEAN (``core.semantics``): a bullish-positive
number in [-1, +1] that says which call the reading implies. Mūla's lean is its
valuation oscillator (cheap = bullish); Swayam's are its net view breadth, its
average oscillator and its HMM regime split (oversold / low / bearish-regime =
bullish, the system's contrarian reading). The four dimensions then measure
agreement between leans, never between raw readings that happen to share a
name:

1. **Direction** — do the two calls point the same way? Mūla's claim (|FVO| past
   the claim threshold) against the sign of Swayam's net breadth.
2. **Breadth** — does Mūla's window breadth (how many of its 1m/3m/1y windows
   read the gap cheap, minus rich) match Swayam's view breadth (oversold views
   minus overbought)? Both are NET splits — the old dimension compared only
   the oversold shares, so a Mūla reading 100% rich against a Swayam reading 0%
   of anything scored as perfect agreement.
3. **Magnitude** — are the two leans equally strong?
4. **Regime** — does Mūla's valuation state agree with Swayam's HMM regime
   split? Swayam's "bull" state is the high-oscillator state, i.e. stretched
   up, so under the contrarian reading it is a BEARISH lean; the old code
   scored "Mūla oversold + Swayam bull" as consistent, which mixed a contrarian
   reading with a momentum one inside one dimension.

A date without a Swayam row is not scored at all. It used to be scored against
a stand-in of 50% oversold / 50% overbought / 33-33-34 regime — an impossible
state (the shares sum to 100%) that then fed the weight learner, the
walk-forward frame and the divergence detector as if Swayam had said something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import (
    CONV_WEIGHT_DIRECTION,
    CONV_WEIGHT_BREADTH,
    CONV_WEIGHT_MAGNITUDE,
    CONV_WEIGHT_REGIME,
)
from core.semantics import (
    breadth_lean,
    claim_direction,
    oscillator_lean,
    state_lean,
    swayam_regime_lean,
    valuation_lean,
)

#: The four agreement dimensions, in the order every frame carries them.
DIMENSIONS: tuple[str, ...] = ("direction", "breadth", "magnitude", "regime")

#: Prior dimension weights — the starting belief before any outcome resolves.
PRIOR_WEIGHTS: dict[str, float] = {
    "direction": CONV_WEIGHT_DIRECTION,
    "breadth": CONV_WEIGHT_BREADTH,
    "magnitude": CONV_WEIGHT_MAGNITUDE,
    "regime": CONV_WEIGHT_REGIME,
}

#: Net-breadth dead band for Swayam's DIRECTION: a view split within ±10 points
#: (e.g. 0% oversold vs 0% overbought, the commonest day) is no call, not a
#: coin-flip sign. Same band as ``core.semantics.lean_word``.
SWAYAM_DIRECTION_DEAD = 0.10
#: Regime dead band: an HMM split within ±20 points is no regime call (the
#: band the old NEUTRAL branch used).
SWAYAM_REGIME_DEAD = 0.20


def normalise_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Dimension weights, normalised to sum to 1, from either key style.

    Accepts the bare dimension names (``direction``) the learner publishes and
    the ``w_``-prefixed names ``InstrumentConfig.weights_seed()`` returns, so a
    per-instrument prior and a learned weight vector can be passed the same
    way. Missing keys fall back to the global prior.
    """
    w = {}
    for d in DIMENSIONS:
        v = PRIOR_WEIGHTS[d]
        if weights:
            v = weights.get(d, weights.get(f"w_{d}", v))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = PRIOR_WEIGHTS[d]
        w[d] = max(v, 0.0) if math.isfinite(v) else PRIOR_WEIGHTS[d]
    tot = sum(w.values())
    if tot <= 1e-12:
        return {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS}
    return {d: v / tot for d, v in w.items()}


def _sign_agreement(a: int, b: int) -> float:
    """1 when two calls agree, 0 when they oppose, 0.5 when either abstains."""
    if a == 0 or b == 0:
        return 0.5
    return 1.0 if a == b else 0.0


def _dead_sign(x: float, dead: float) -> int:
    return 0 if abs(x) < dead else (1 if x > 0 else -1)


@dataclass
class ConvergenceSignal:
    """Cross-system convergence for a single date.

    Attributes
    ----------
    convergence_score : float
        Weighted composite in [-100, +100], READING-oriented like every other
        engine output: negative = the engines agree it is stretched DOWN
        (bullish), positive = stretched UP (bearish). Magnitude = how strongly
        they agree times how far the shared lean points.
    agreement_ratio : float
        Weighted mean of the four dimension scores, in [0, 1].
    consensus_direction : float
        Bullish-positive mean of the two engines' leans, in [-1, +1].
    mula_lean, swayam_lean : float
        The two leans behind it (Mūla valuation; Swayam net breadth).
    dimension_scores / dimension_weights : dict[str, float]
        Per-dimension agreement in [0, 1], and the weights applied.
    swayam_native : bool
        False when the Swayam row was carried forward onto this date rather
        than observed on it — kept so learning can skip stale evidence.
    """

    convergence_score: float
    agreement_ratio: float
    consensus_direction: float
    mula_lean: float
    swayam_lean: float
    date: str
    dimension_scores: dict[str, float] = field(default_factory=dict)
    dimension_weights: dict[str, float] = field(default_factory=dict)
    swayam_native: bool = True


def score_dimensions(mula: dict, swayam: dict) -> tuple[dict[str, float], float, float]:
    """The four dimension scores for one date, plus the two engines' leans.

    Pure function of the two inputs — :class:`CrossValidator` and
    ``convergence.intelligence`` both rebuild the composite from what this
    returns, so there is one definition of "agreement".
    """
    fvo = mula.get("fvo")
    m_lean = valuation_lean(fvo)
    m_call = claim_direction(fvo)
    w_lean = breadth_lean(mula.get("oversold_breadth"), mula.get("overbought_breadth"))
    m_state = state_lean(mula.get("regime", ""))

    s_lean = breadth_lean(swayam.get("oversold_pct"), swayam.get("overbought_pct"))
    s_osc = oscillator_lean(swayam.get("avg_unified_osc"))
    s_reg = swayam_regime_lean(swayam.get("regime_bull_pct"), swayam.get("regime_bear_pct"))

    dims = {
        "direction": _sign_agreement(m_call, _dead_sign(s_lean, SWAYAM_DIRECTION_DEAD)),
        "breadth": float(np.clip(1.0 - abs(w_lean - s_lean) / 2.0, 0.0, 1.0)),
        "magnitude": float(np.clip(1.0 - abs(abs(m_lean) - abs(s_osc)), 0.0, 1.0)),
        "regime": _sign_agreement(m_state, _dead_sign(s_reg, SWAYAM_REGIME_DEAD)),
    }
    return dims, m_lean, s_lean


def composite_score(dims: dict[str, float], weights: dict[str, float],
                    consensus_direction: float) -> tuple[float, float]:
    """``(convergence_score, agreement_ratio)`` for given dims and weights.

    ``agreement`` is the weighted mean dimension score; the score orients it:
    ``-consensus_direction × agreement × 100``. Agreement on its own has no
    direction (it is as high at a top as at a bottom), which is why the sign
    has to come from the engines' leans and not from the dimensions.
    """
    agreement = float(sum(weights[d] * dims[d] for d in DIMENSIONS))
    return -consensus_direction * agreement * 100.0, agreement


class CrossValidator:
    """Cross-references Mūla and Swayam per trading date.

    ``active_weights`` are the dimension weights to apply (either key style,
    see :func:`normalise_weights`); ``None`` means the prior. There is no
    heuristic re-weighting inside: the weights that matter are learned forward
    by ``convergence.intelligence.apply_online_weights``, which rebuilds the
    composite from the stored dimension scores.
    """

    def __init__(self, active_weights: dict[str, float] | None = None) -> None:
        self.history: list[ConvergenceSignal] = []
        self.weights = normalise_weights(active_weights)

    def compute_convergence(
        self,
        mula: dict[str, object],
        swayam: dict[str, object] | None,
        date: str,
    ) -> ConvergenceSignal | None:
        """Score one date, or return ``None`` when there is nothing to score.

        Parameters
        ----------
        mula : dict
            ``{"fvo", "oversold_breadth", "overbought_breadth", "regime"}`` —
            the Mūla oscillator (σ), its window-breadth split (0-100 each) and
            its valuation-state word.
        swayam : dict | None
            ``{"oversold_pct", "overbought_pct", "avg_unified_osc",
            "regime_bull_pct", "regime_bear_pct"}`` and optionally ``"native"``
            (False when carried forward). ``None`` = no Swayam row: the date is
            skipped, not scored against a placeholder.
        date : str
            ``YYYY-MM-DD``.
        """
        if swayam is None:
            return None
        fvo = mula.get("fvo")
        try:
            if fvo is None or not math.isfinite(float(fvo)):
                return None          # no published valuation on this date
        except (TypeError, ValueError):
            return None

        dims, m_lean, s_lean = score_dimensions(mula, swayam)
        cd = (m_lean + s_lean) / 2.0
        score, agreement = composite_score(dims, self.weights, cd)

        result = ConvergenceSignal(
            convergence_score=round(float(score), 4),
            agreement_ratio=round(float(agreement), 4),
            consensus_direction=float(cd),
            mula_lean=float(m_lean),
            swayam_lean=float(s_lean),
            date=date,
            dimension_scores={k: round(float(v), 4) for k, v in dims.items()},
            dimension_weights={k: round(float(v), 4) for k, v in self.weights.items()},
            swayam_native=bool(swayam.get("native", True)),
        )
        self.history.append(result)
        return result

    def get_convergence_series(self) -> pd.DataFrame:
        """Convergence history as a DataFrame indexed by date string.

        Columns: ``convergence_score``, ``agreement_ratio``,
        ``consensus_direction``, ``mula_lean``, ``swayam_lean``,
        ``swayam_native``, ``dim_<d>`` and ``w_<d>`` for each dimension.
        """
        if not self.history:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for h in self.history:
            rows.append(
                {
                    "date": h.date,
                    "convergence_score": h.convergence_score,
                    "agreement_ratio": h.agreement_ratio,
                    "consensus_direction": h.consensus_direction,
                    "mula_lean": h.mula_lean,
                    "swayam_lean": h.swayam_lean,
                    "swayam_native": h.swayam_native,
                    **{f"dim_{k}": v for k, v in h.dimension_scores.items()},
                    **{f"w_{k}": v for k, v in h.dimension_weights.items()},
                }
            )
        return pd.DataFrame(rows).set_index("date")
