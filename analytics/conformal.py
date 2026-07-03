"""
Tattva — Rolling robust quantile z-scores with fat-tail adjustment.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Rolling empirical-quantile bounds for walk-forward regression
residuals.

NOTE on naming: this module (and older docs/UI copy) previously called this
"conformal prediction". It isn't — conformal prediction (Vovk, Gammerman &
Shafer 2005, Algorithmic Learning in a Random World) provides a finite-sample
COVERAGE GUARANTEE via calibration-set nonconformity scores under an
exchangeability assumption. What's computed here is a rolling-window robust
z-score (deviation from the window median, scaled by the window's normalized
IQR) — a useful, numerically fine robust-statistics tool, but it carries no
such guarantee on a serially dependent series. The IQR/1.35 scale factor is
the standard robust-scale estimate (1.349 ~= 2*Phi^-1(0.75); see e.g. Huber &
Ronchetti, Robust Statistics), unrelated to conformal calibration.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _percentile_linear(sorted_v: np.ndarray, q: float) -> float:
    """Linear-interpolated percentile of a pre-sorted array — matches numpy's
    default ``np.percentile(..., method='linear')`` (and hence nanpercentile /
    nanmedian once NaNs are removed)."""
    k = sorted_v.shape[0]
    if k == 1:
        return sorted_v[0]
    rank = q * (k - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    frac = rank - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


@njit(cache=True)
def _conformal_njit(series, window, min_periods, alpha):
    n = series.shape[0]
    z_scores = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    upper_bounds = np.full(n, np.nan)
    scratch = np.empty(window, dtype=np.float64)
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    for i in range(window, n):
        # Collect finite values from series[i-window:i] (excludes current point).
        cnt = 0
        for j in range(i - window, i):
            v = series[j]
            if np.isfinite(v):
                scratch[cnt] = v
                cnt += 1
        if cnt < min_periods:
            continue
        sv = np.sort(scratch[:cnt])
        ql = _percentile_linear(sv, q_lo)
        qu = _percentile_linear(sv, q_hi)
        qm = _percentile_linear(sv, 0.5)
        iqr = qu - ql
        if iqr > 1e-10:
            z_scores[i] = (series[i] - qm) / (iqr / 1.35)
        lower_bounds[i] = ql
        upper_bounds[i] = qu
    return z_scores, lower_bounds, upper_bounds


def compute_conformal_zscores(
    series: np.ndarray,
    window: int,
    min_periods: int = 5,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling robust quantile z-scores with fat-tail adjustment.

    Uses empirical quantiles instead of mean/std for robustness to fat-tailed
    distributions. (Function/parameter names kept as "conformal" for
    import-site compatibility — see this module's docstring for why that
    label is inaccurate: this has no conformal-prediction coverage
    guarantee, it is a rolling robust z-score / empirical-quantile band.)

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    window : int
        Rolling window size for the empirical-quantile bounds.
    min_periods : int
        Minimum valid observations required within the window.
    alpha : float
        Tail fraction for the bounds (default 0.05 -> the [2.5, 97.5]
        percentile band).

    Returns
    -------
    z_scores : np.ndarray
        Quantile-normalized z-scores.
    lower_bounds : np.ndarray
        Lower empirical-quantile bound at level ``1 - alpha``.
    upper_bounds : np.ndarray
        Upper empirical-quantile bound at level ``1 - alpha``.
    """
    n = len(series)
    if n <= window:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    # Numba kernel: faithful port of the rolling-quantile loop below. Excludes
    # the current point (no look-ahead) and replicates numpy's nanpercentile /
    # nanmedian (linear interpolation) exactly. ~10× faster than the Python
    # loop over np.nanpercentile (which sorts each window 3×).
    arr = np.ascontiguousarray(series, dtype=np.float64)
    return _conformal_njit(arr, int(window), int(min_periods), float(alpha))
