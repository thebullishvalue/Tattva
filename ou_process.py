"""
Ornstein-Uhlenbeck process estimation with Andrews (1993) median-unbiased AR(1).

Extracted from correl.py lines 389-448, 560-589.
"""

from __future__ import annotations

import numpy as np


def ornstein_uhlenbeck_estimate(
    series: np.ndarray,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate OU process parameters via AR(1) regression with Andrews MU correction.

    Models: ``dx = θ(μ − x)dt + σdW``

    Uses Andrews (1993) median-unbiased estimator for near-unit-root cases,
    which handles persistent series better than jackknife methods.

    Parameters
    ----------
    series : np.ndarray
        Input time-series (typically residuals from the fair-value model).
    dt : float
        Time step (default 1.0 for daily data).

    Returns
    -------
    theta : float
        Mean-reversion speed.
    mu : float
        Equilibrium level.
    sigma : float
        Volatility.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 20:
        if len(x) > 1:
            return 0.05, 0.0, max(float(np.std(x)), 1e-6)
        return 0.05, 0.0, 1.0

    x_lag = x[:-1]
    x_curr = x[1:]
    n = len(x_lag)

    sx = np.sum(x_lag)
    sy = np.sum(x_curr)
    sxx = np.dot(x_lag, x_lag)
    sxy = np.dot(x_lag, x_curr)

    denom = n * sxx - sx**2
    if abs(denom) < 1e-12:
        return 0.05, float(np.mean(x)), max(float(np.std(x)), 1e-6)

    a = (n * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom
    a = np.clip(a, 1e-6, 0.999)

    # Andrews (1993) median-unbiased correction
    if a > 0.95:
        a_corrected = a - (1 + 3 * a) / n - 3 * (1 + 3 * a) / (n**2)
    else:
        a_corrected = a - (1 + 3 * a) / n

    a = np.clip(a_corrected, 0.0, 0.999)

    theta = -np.log(a) / dt if a > 1e-6 else 0.05
    mu = b / (1 - a) if abs(1 - a) > 1e-6 else float(np.mean(x))

    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)

    if a > 0.98:
        sigma = max(float(np.std(residuals)) * np.sqrt(2 * max(theta, 1e-4)), 1e-6)
    else:
        sigma = np.sqrt(max(sigma_sq * 2 * theta / (1 - a**2), 1e-12))

    return max(float(theta), 1e-4), float(mu), max(float(sigma), 1e-6)


def andrews_median_unbiased_ar1(series: np.ndarray) -> tuple[float, float]:
    """Andrews (1993) median-unbiased AR(1) estimator with half-life.

    Parameters
    ----------
    series : np.ndarray
        Input time-series.

    Returns
    -------
    ar_coef : float
        Median-unbiased AR(1) coefficient.
    half_life : float
        Half-life of mean reversion (``log(0.5) / log(ar_coef)``).
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 20:
        return 0.0, np.inf

    x_lag = x[:-1]
    x_curr = x[1:]

    a_ols = np.corrcoef(x_lag, x_curr)[0, 1]

    if a_ols > 0.95:
        a_mu = a_ols - (1 + 3 * a_ols) / n - 3 * (1 + 3 * a_ols) / (n**2)
    else:
        a_mu = a_ols - (1 + 3 * a_ols) / n

    a_mu = np.clip(a_mu, 0.0, 0.999)

    half_life = np.log(0.5) / np.log(a_mu) if a_mu > 0.01 else np.inf

    return float(a_mu), float(half_life)
