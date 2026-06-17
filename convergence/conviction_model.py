"""
Tattva — UnifiedConvictionModel: DDM filtering on convergence scores.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.

Applies the same Drift-Diffusion Model (DDM) primitive used by the
Aarambh engine on raw conviction, but operates on the cross-system
convergence score time-series. The result is a bounded, smoothed
signal with confidence bands reflecting agreement between both
analytical engines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import (
    CONV_DDM_LEAK_RATE,
    CONV_DDM_DRIFT_SCALE,
    CONV_DDM_LONG_RUN_VAR,
    CONVICTION_STRONG,
    CONVICTION_MODERATE,
    CONVICTION_WEAK,
)
from analytics.ddm_filter import drift_diffusion_filter
from analytics.utils import _apply_conviction_bounds


@dataclass
class UnifiedConvictionResult:
    """Output of the unified conviction model for a single date.

    Attributes
    ----------
    nishkarsh_conviction : float
        DDM-filtered convergence score, bounded to [-100, +100].
    nishkarsh_signal : str
        Classified signal (``STRONG BUY`` → ``STRONG SELL``).
    confidence_upper : float
        Upper bound of the 95% confidence band.
    confidence_lower : float
        Lower bound of the 95% confidence band.
    confidence_bandwidth : float
        Width of the confidence band (upper - lower).
    """

    nishkarsh_conviction: float
    nishkarsh_signal: str
    confidence_upper: float
    confidence_lower: float
    confidence_bandwidth: float


class UnifiedConvictionModel:
    """Applies DDM filtering to a convergence score time-series.

    Parameters
    ----------
    leak_rate : float
        DDM memory decay rate. Higher = faster forgetting.
    drift_scale : float
        Scaling factor for new evidence contribution.
    long_run_var : float
        Long-run variance toward which the filter mean-reverts.
    """

    def __init__(
        self,
        leak_rate: float = CONV_DDM_LEAK_RATE,
        drift_scale: float = CONV_DDM_DRIFT_SCALE,
        long_run_var: float = CONV_DDM_LONG_RUN_VAR,
    ) -> None:
        self.leak_rate = leak_rate
        self.drift_scale = drift_scale
        self.long_run_var = long_run_var
        self._filtered: np.ndarray = np.array([])
        self._variances: np.ndarray = np.array([])
        self._dates: list[str] = []

    def fit(
        self, convergence_scores: list[float], dates: list[str]
    ) -> list[UnifiedConvictionResult]:
        """Filter a series of convergence scores through the DDM.

        Parameters
        ----------
        convergence_scores : list[float]
            Raw convergence scores from ``CrossValidator``.
        dates : list[str]
            Corresponding date strings.

        Returns
        -------
        list[UnifiedConvictionResult]
            One result per observation.
        """
        scores = np.array(convergence_scores, dtype=np.float64)
        self._dates = dates

        filtered, _gains, variances = drift_diffusion_filter(
            scores,
            leak_rate=self.leak_rate,
            drift_scale=self.drift_scale,
            long_run_var=self.long_run_var,
        )
        self._filtered = filtered
        self._variances = variances

        results: list[UnifiedConvictionResult] = []
        for i in range(len(scores)):
            ddm_std = float(np.sqrt(max(self._variances[i], 0)))
            bounded = _apply_conviction_bounds(filtered[i])
            upper = _apply_conviction_bounds(filtered[i] + 1.96 * ddm_std)
            lower = _apply_conviction_bounds(filtered[i] - 1.96 * ddm_std)
            results.append(
                UnifiedConvictionResult(
                    nishkarsh_conviction=round(bounded, 2),
                    nishkarsh_signal=self._classify_signal(bounded),
                    confidence_upper=round(upper, 2),
                    confidence_lower=round(lower, 2),
                    confidence_bandwidth=round(upper - lower, 2),
                )
            )
        return results

    def get_latest(self) -> UnifiedConvictionResult | None:
        """Return the most recent conviction result, or ``None``."""
        if len(self._filtered) == 0:
            return None
        i = len(self._filtered) - 1
        ddm_std = float(np.sqrt(max(self._variances[i], 0)))
        bounded = _apply_conviction_bounds(self._filtered[i])
        upper = _apply_conviction_bounds(self._filtered[i] + 1.96 * ddm_std)
        lower = _apply_conviction_bounds(self._filtered[i] - 1.96 * ddm_std)
        return UnifiedConvictionResult(
            nishkarsh_conviction=round(bounded, 2),
            nishkarsh_signal=self._classify_signal(bounded),
            confidence_upper=round(upper, 2),
            confidence_lower=round(lower, 2),
            confidence_bandwidth=round(upper - lower, 2),
        )

    def get_series(self) -> pd.DataFrame:
        """Return the full filtered series as a DataFrame indexed by date."""
        if len(self._filtered) == 0:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for i in range(len(self._filtered)):
            ddm_std = float(np.sqrt(max(self._variances[i], 0)))
            bounded = _apply_conviction_bounds(self._filtered[i])
            upper = _apply_conviction_bounds(self._filtered[i] + 1.96 * ddm_std)
            lower = _apply_conviction_bounds(self._filtered[i] - 1.96 * ddm_std)
            rows.append(
                {
                    "date": self._dates[i] if i < len(self._dates) else "",
                    "nishkarsh_conviction": round(bounded, 2),
                    "signal": self._classify_signal(bounded),
                    "confidence_upper": round(upper, 2),
                    "confidence_lower": round(lower, 2),
                    "confidence_bandwidth": round(upper - lower, 2),
                }
            )
        return pd.DataFrame(rows).set_index("date")

    @staticmethod
    def _classify_signal(conviction: float) -> str:
        """Classify a conviction score into a human-readable signal label."""
        if conviction < -CONVICTION_STRONG:
            return "STRONG BUY"
        if conviction < -CONVICTION_MODERATE:
            return "BUY"
        if conviction < -CONVICTION_WEAK:
            return "WEAK BUY"
        if conviction > CONVICTION_STRONG:
            return "STRONG SELL"
        if conviction > CONVICTION_MODERATE:
            return "SELL"
        if conviction > CONVICTION_WEAK:
            return "WEAK SELL"
        return "NEUTRAL"
