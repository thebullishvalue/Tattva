"""
Tattva v2.0.0 — Conformal prediction z-scores with fat-tail adjustment.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Conformal prediction bounds for walk-forward regression residuals.
"""

from __future__ import annotations

import numpy as np


def compute_conformal_zscores(
    series: np.ndarray,
    window: int,
    min_periods: int = 5,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformal prediction-based z-scores with fat-tail adjustment.

    Uses empirical quantiles instead of mean/std for robustness
    to fat-tailed distributions.

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    window : int
        Rolling window size for conformal intervals.
    min_periods : int
        Minimum valid observations required within the window.
    alpha : float
        Significance level for conformal intervals (default 0.05 → 95%).

    Returns
    -------
    z_scores : np.ndarray
        Quantile-normalized z-scores.
    lower_bounds : np.ndarray
        Lower conformal interval bound at level ``1 - alpha``.
    upper_bounds : np.ndarray
        Upper conformal interval bound at level ``1 - alpha``.
    """
    n = len(series)
    z_scores = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    upper_bounds = np.full(n, np.nan)

    for i in range(window, n):
        # Exclude current point to prevent look-ahead bias
        window_data = series[i - window : i]

        if np.sum(np.isfinite(window_data)) < min_periods:
            continue

        q_lower = np.nanpercentile(window_data, alpha / 2 * 100)
        q_upper = np.nanpercentile(window_data, (1 - alpha / 2) * 100)
        q_median = np.nanmedian(window_data)

        # Z-score via quantile normalization (1.35 ≈ normal IQR/σ)
        iqr = q_upper - q_lower
        if iqr > 1e-10:
            z_scores[i] = (series[i] - q_median) / (iqr / 1.35)

        lower_bounds[i] = q_lower
        upper_bounds[i] = q_upper

    return z_scores, lower_bounds, upper_bounds
