"""
Ornstein-Uhlenbeck process estimation with a Kendall (1954) / Orcutt & Winokur
(1969) first-order bias-corrected AR(1) coefficient.

Extracted from correl.py lines 389-448, 560-589.
"""

from __future__ import annotations

import numpy as np


def ornstein_uhlenbeck_estimate(
    series: np.ndarray,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate OU process parameters via AR(1) regression with a bias correction.

    Models: ``dx = θ(μ − x)dt + σdW``

    Uses the Kendall (1954) / Orcutt & Winokur (1969) first-order bias
    correction for the OLS AR(1) coefficient (additive: a_hat + (1+3a)/n),
    which is more robust near-unit-root than leaving the OLS estimate
    uncorrected.

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

    # Kendall (1954, Biometrika 41) / Orcutt & Winokur (1969) first-order bias
    # correction. The OLS AR(1) coefficient is biased DOWNWARD:
    # E[a_hat] ~= a - (1+3a)/n, so the correction is ADDITIVE (a_hat + (1+3a)/n
    # moves the estimate back toward the true a). The correction here used to
    # SUBTRACT, which doubles the downward bias instead of removing it
    # (verified by simulation: at a=0.90, n=300, mean OLS a_hat=0.887; the
    # subtractive form yields 0.875 — further from 0.90 — while the additive
    # form yields 0.899). This also was not Andrews (1993) — that is a
    # quantile-table/simulation-based median-unbiased estimator, a different
    # (and more involved) method than this closed-form first-order correction.
    if a > 0.95:
        a_corrected = a + (1 + 3 * a) / n + 3 * (1 + 3 * a) / (n**2)
    else:
        a_corrected = a + (1 + 3 * a) / n

    a = np.clip(a_corrected, 0.0, 0.999)

    theta = -np.log(a) / dt if a > 1e-6 else 0.05
    mu = b / (1 - a) if abs(1 - a) > 1e-6 else float(np.mean(x))

    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)

    sigma = np.sqrt(max(sigma_sq * 2 * theta / max(1 - a**2, 0.001), 1e-12))

    return max(float(theta), 1e-4), float(mu), max(float(sigma), 1e-6)


def andrews_median_unbiased_ar1(series: np.ndarray) -> tuple[float, float]:
    """Kendall (1954) / Orcutt & Winokur (1969) first-order bias-corrected AR(1)
    estimator with half-life.

    (Despite the function name — kept for import-site compatibility — this is
    NOT Andrews (1993); that is a quantile-table/simulation-based median-
    unbiased estimator. This is the simpler closed-form first-order bias
    correction: E[a_hat] ~= a - (1+3a)/n, so the correction is ADDITIVE.)

    Parameters
    ----------
    series : np.ndarray
        Input time-series.

    Returns
    -------
    ar_coef : float
        Bias-corrected AR(1) coefficient.
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
        a_mu = a_ols + (1 + 3 * a_ols) / n + 3 * (1 + 3 * a_ols) / (n**2)
    else:
        a_mu = a_ols + (1 + 3 * a_ols) / n

    a_mu = np.clip(a_mu, 0.0, 0.999)

    half_life = np.log(0.5) / np.log(a_mu) if a_mu > 0.01 else np.inf

    return float(a_mu), float(half_life)
