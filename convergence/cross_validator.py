"""
Tattva — CrossValidator: Cross-referencing Aarambha and Nirnay outputs.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.

Computes a convergence score from four adaptive-weighted dimensions:
direction agreement, breadth confirmation, magnitude alignment, and
regime consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import (
    CONV_WEIGHT_DIRECTION,
    CONV_WEIGHT_BREADTH,
    CONV_WEIGHT_MAGNITUDE,
    CONV_WEIGHT_REGIME,
    CONV_ADAPTIVE_SHIFT_MAX,
    CONV_STRONG_BULLISH,
    CONV_MODERATE_BULLISH,
    CONV_WEAK_BULLISH,
    CONV_WEAK_BEARISH,
    CONV_MODERATE_BEARISH,
    CONV_STRONG_BEARISH,
)


@dataclass
class ConvergenceSignal:
    """Cross-system convergence signal for a single date.

    Attributes
    ----------
    convergence_score : float
        Composite score in [-100, +100]. Negative = bullish convergence.
    convergence_zone : str
        Categorical zone label.
    agreement_ratio : float
        Fraction of dimensions that agree [0, 1].
    lead_lag_indicator : str
        Which system is leading, or ``ALIGNED`` / ``CONTRADICTION``.
    confidence : float
        Composite confidence [0, 1].
    dimension_scores : dict[str, float]
        Per-dimension raw scores [0, 1].
    dimension_weights : dict[str, float]
        Adaptive weights actually used for this observation.
    date : str
        Date string.
    """

    convergence_score: float
    convergence_zone: str
    agreement_ratio: float
    lead_lag_indicator: str
    confidence: float
    dimension_scores: dict[str, float]
    dimension_weights: dict[str, float]
    date: str
    consensus_direction: float = 0.0  # +1 bullish, -1 bearish, 0 disagree/neutral


class CrossValidator:
    """Cross-references Aarambh and Nirnay outputs per trading date.

    Computes an adaptive-weighted composite of four convergence dimensions:

    1. **Direction** (30% base): Both systems pointing the same way.
    2. **Breadth** (25% base): Oversold breadth alignment.
    3. **Magnitude** (25% base): Signal strengths are comparable.
    4. **Regime** (20% base): OU regime aligns with HMM regime.

    Dimensions with higher clarity (stronger signals) receive up to
    ±10% additional weight at the expense of weaker dimensions.
    """

    def __init__(
        self,
        active_weights: dict[str, float] | None = None,
        expected_constituents: int | None = None,
    ) -> None:
        """Args:
            active_weights: Optional override for the base dimension weights.
                Must be a dict with keys ``direction``, ``breadth``,
                ``magnitude``, ``regime``. Used by Intelligence Mode to
                inject calibrated weights from a persisted profile. When
                ``None``, falls back to the static CONV_WEIGHT_* defaults
                with the heuristic ±10% adaptive shift.
            expected_constituents: Full size of the Nirnay basket for the
                active universe. Confidence is down-weighted only when a day
                has *fewer* analyzed instruments than this (a data-coverage
                penalty), rather than against a hardcoded Nifty-50 count.
                When ``None``, no coverage penalty is applied.
        """
        self.history: list[ConvergenceSignal] = []
        self.active_weights = active_weights
        self.expected_constituents = expected_constituents

    def compute_convergence(
        self,
        aarambh_signal: dict[str, object],
        nirnay_day_stats: dict[str, object],
        date: str,
    ) -> ConvergenceSignal:
        """Compute convergence score for a single date.

        Parameters
        ----------
        aarambh_signal : dict
            Output from ``FairValueEngine.get_current_signal()``.
        nirnay_day_stats : dict
            Aggregated Nirnay stats for the date.
        date : str
            Date string for this observation.
        """
        # ── Dimension 1: Direction Agreement ────────────────────────────
        # Aarambh: conviction < 0 = bullish (oversold), > 0 = bearish
        # Nirnay: oversold_pct > overbought_pct = bullish bias
        aarambh_direction = -np.sign(float(aarambh_signal.get("conviction_score", 0)))
        nirnay_os = float(nirnay_day_stats.get("oversold_pct", 50))
        nirnay_ob = float(nirnay_day_stats.get("overbought_pct", 50))
        nirnay_direction = np.sign(nirnay_os - nirnay_ob)

        if aarambh_direction == nirnay_direction and aarambh_direction != 0:
            direction_score = 1.0
        elif aarambh_direction == 0 or nirnay_direction == 0:
            direction_score = 0.5
        else:
            direction_score = 0.0

        # ── Dimension 2: Breadth Confirmation ───────────────────────────
        aarambh_os_breadth = float(aarambh_signal.get("oversold_breadth", 50))
        breadth_agreement = 1.0 - abs(aarambh_os_breadth - nirnay_os) / 100.0
        breadth_score = max(0.0, min(1.0, breadth_agreement))

        # ── Dimension 3: Magnitude Alignment ────────────────────────────
        aarambh_mag = abs(float(aarambh_signal.get("conviction_score", 0)))
        nirnay_mag = abs(float(nirnay_day_stats.get("avg_unified_osc", 0))) * 10
        aarambh_mag_norm = min(aarambh_mag / 100.0, 1.0)
        nirnay_mag_norm = min(nirnay_mag / 10.0, 1.0)
        magnitude_alignment = 1.0 - abs(aarambh_mag_norm - nirnay_mag_norm)
        magnitude_score = max(0.0, min(1.0, magnitude_alignment))

        # ── Dimension 4: Regime Consistency ─────────────────────────────
        aarambh_regime = str(aarambh_signal.get("regime", "NEUTRAL"))
        nirnay_bull_pct = float(nirnay_day_stats.get("regime_bull", 0))
        nirnay_bear_pct = float(nirnay_day_stats.get("regime_bear", 0))

        aarambh_bullish = "OVERSOLD" in aarambh_regime
        aarambh_bearish = "OVERBOUGHT" in aarambh_regime
        nirnay_bullish = nirnay_bull_pct > nirnay_bear_pct
        nirnay_bearish = nirnay_bear_pct > nirnay_bull_pct

        if aarambh_bullish and nirnay_bullish:
            regime_score = 1.0
        elif aarambh_bearish and nirnay_bearish:
            regime_score = 1.0
        elif aarambh_regime == "NEUTRAL" and abs(nirnay_bull_pct - nirnay_bear_pct) < 20:
            regime_score = 0.8
        else:
            regime_score = 0.2

        # ── Weighting ───────────────────────────────────────────────────
        # Two paths:
        #   (a) Intelligence Mode active → use calibrated weights from the
        #       persisted profile verbatim. Skip the adaptive shift heuristic
        #       (the calibration already learned the optimum from data).
        #   (b) Factory defaults → apply the ±10% adaptive shift heuristic
        #       on top of the CONV_WEIGHT_* base allocation.
        if self.active_weights is not None:
            # Calibrated path: trust the profile.
            base_weights = {
                "direction": float(self.active_weights.get("w_direction", CONV_WEIGHT_DIRECTION)),
                "breadth":   float(self.active_weights.get("w_breadth",   CONV_WEIGHT_BREADTH)),
                "magnitude": float(self.active_weights.get("w_magnitude", CONV_WEIGHT_MAGNITUDE)),
                "regime":    float(self.active_weights.get("w_regime",    CONV_WEIGHT_REGIME)),
            }
            total_w = sum(base_weights.values()) or 1.0
            adaptive_weights = {k: v / total_w for k, v in base_weights.items()}
        else:
            # Default path: heuristic adaptive shift.
            base_weights = {
                "direction": CONV_WEIGHT_DIRECTION,
                "breadth": CONV_WEIGHT_BREADTH,
                "magnitude": CONV_WEIGHT_MAGNITUDE,
                "regime": CONV_WEIGHT_REGIME,
            }
            clarities = {
                "direction": abs(float(aarambh_direction)) * 0.5 + abs(float(nirnay_direction)) * 0.5 + 0.001,
                "breadth": abs(aarambh_os_breadth - 50) / 50.0 + abs(nirnay_os - 50) / 50.0 + 0.001,
                "magnitude": (aarambh_mag_norm + nirnay_mag_norm) / 2.0 + 0.001,
                "regime": abs(nirnay_bull_pct - nirnay_bear_pct) / 100.0 + 0.001,
            }
            avg_clarity = np.mean(list(clarities.values()))
            adaptive_weights = {}
            for key in base_weights:
                clarity_ratio = clarities[key] / avg_clarity
                shift = CONV_ADAPTIVE_SHIFT_MAX * (clarity_ratio - 1.0)
                adaptive_weights[key] = float(np.clip(base_weights[key] + shift, 0.10, 0.40))
            total_w = sum(adaptive_weights.values())
            adaptive_weights = {k: v / total_w for k, v in adaptive_weights.items()}

        # Composite convergence score [-100, +100]
        composite = (
            adaptive_weights["direction"] * (direction_score * 2 - 1)
            + adaptive_weights["breadth"] * (2 * breadth_score - 1)
            + adaptive_weights["magnitude"] * (2 * magnitude_score - 1)
            + adaptive_weights["regime"] * (2 * regime_score - 1)
        )
        # ── Orient the score directionally ──────────────────────────────
        # `composite` measures AGREEMENT strength (the four dims are agreement
        # scores, not bull/bear), so on its own it has no direction — a high
        # value could be agreement on a top or a bottom. Multiply by the
        # consensus direction so the score is a genuine directional conviction
        # (zone convention: negative = bullish), consistent with the normalized
        # convergence signal. When the two systems disagree, the directional
        # conviction collapses to ~0 (DIVERGENT).
        if aarambh_direction == nirnay_direction and aarambh_direction != 0:
            consensus_direction = float(aarambh_direction)  # +1 bullish, -1 bearish
        else:
            consensus_direction = 0.0
        agreement_strength = (composite + 1.0) / 2.0  # [-1,1] agreement → [0,1]
        convergence_score = -consensus_direction * agreement_strength * 100.0

        # Agreement ratio
        # When Intelligence Mode injected calibrated weights, the agreement
        # metric uses those weights too — so the user-visible AGREEMENT
        # percentage stays semantically consistent with the calibrated
        # convergence score. When no calibration is active, fall back to
        # the plain 4-way mean (the historical definition).
        if self.active_weights is not None:
            agreement_ratio = (
                adaptive_weights["direction"] * direction_score
                + adaptive_weights["breadth"]   * breadth_score
                + adaptive_weights["magnitude"] * magnitude_score
                + adaptive_weights["regime"]    * regime_score
            )
        else:
            agreement_ratio = (direction_score + breadth_score + magnitude_score + regime_score) / 4.0

        # Lead-lag indicator
        if aarambh_direction != nirnay_direction and abs(aarambh_direction) > abs(nirnay_direction) * 1.5:
            lead_lag = "AARAMBH_LEADS"
        elif aarambh_direction != nirnay_direction and abs(nirnay_direction) * 1.5 > abs(aarambh_direction):
            lead_lag = "NIRNAY_LEADS"
        elif aarambh_direction == nirnay_direction:
            lead_lag = "ALIGNED"
        else:
            lead_lag = "CONTRADICTION"

        # Confidence — agreement scaled by data coverage (analyzed vs full
        # basket). With no expected size, apply no coverage penalty.
        n_analyzed = float(nirnay_day_stats.get("num_constituents", 0))
        if self.expected_constituents and self.expected_constituents > 0:
            coverage = min(1.0, n_analyzed / float(self.expected_constituents))
        else:
            coverage = 1.0
        confidence = agreement_ratio * coverage

        # Zone classification
        if convergence_score <= CONV_STRONG_BULLISH:
            zone = "STRONG_CONVERGENCE_BULLISH"
        elif convergence_score <= CONV_MODERATE_BULLISH:
            zone = "MODERATE_BULLISH"
        elif convergence_score <= CONV_WEAK_BULLISH:
            zone = "WEAK_BULLISH"
        elif convergence_score <= CONV_WEAK_BEARISH:
            zone = "DIVERGENT"
        elif convergence_score <= CONV_MODERATE_BEARISH:
            zone = "WEAK_BEARISH"
        elif convergence_score <= CONV_STRONG_BEARISH:
            zone = "MODERATE_BEARISH"
        else:
            zone = "STRONG_CONVERGENCE_BEARISH"

        result = ConvergenceSignal(
            convergence_score=round(float(convergence_score), 2),
            convergence_zone=zone,
            agreement_ratio=round(float(agreement_ratio), 3),
            lead_lag_indicator=lead_lag,
            confidence=round(float(confidence), 3),
            dimension_scores={
                "direction": round(float(direction_score), 3),
                "breadth": round(float(breadth_score), 3),
                "magnitude": round(float(magnitude_score), 3),
                "regime": round(float(regime_score), 3),
            },
            dimension_weights={k: round(v, 3) for k, v in adaptive_weights.items()},
            date=date,
            consensus_direction=consensus_direction,
        )
        self.history.append(result)
        return result

    def get_convergence_series(self) -> pd.DataFrame:
        """Return convergence history as a DataFrame indexed by date."""
        if not self.history:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for h in self.history:
            rows.append(
                {
                    "date": h.date,
                    "convergence_score": h.convergence_score,
                    "convergence_zone": h.convergence_zone,
                    "agreement_ratio": h.agreement_ratio,
                    "lead_lag_indicator": h.lead_lag_indicator,
                    "confidence": h.confidence,
                    "consensus_direction": h.consensus_direction,
                    **{f"dim_{k}": v for k, v in h.dimension_scores.items()},
                    **{f"w_{k}": v for k, v in h.dimension_weights.items()},
                }
            )
        return pd.DataFrame(rows).set_index("date")
