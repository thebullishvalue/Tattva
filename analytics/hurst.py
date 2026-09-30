"""
Tattva — Hurst exponent via Detrended Fluctuation Analysis (DFA).
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Measures long-term memory and mean-reversion in time series.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def hurst_dfa(
    series: np.ndarray,
    min_scale: int = 8,
    max_scale: int | None = None,
) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis.

    DFA is more robust than classical R/S analysis for short,
    noisy series (Peng et al., 1994, Phys. Rev. E 49).

    Lag range: ``min_scale`` (default 8) to ``max_scale = n // 4``, LOG-spaced
    (``np.geomspace``) rather than linear. Standard DFA practice (Peng et al.
    1994; Kantelhardt et al. 2001) fits the log-log slope over box sizes
    spanning >= 1.5-2 decades; the previous ``min_scale = max(4, n/10)``
    default put both endpoints close together at the LARGE-scale end for
    typical n (e.g. n=1500 -> scales 150..375, a 0.4-decade range with as few
    as ~4 non-overlapping windows at the largest scale) — a narrow, noisy,
    linearly-spaced fit whose slope estimate has much higher variance than
    the literature's prescription, and is biased toward whichever regime
    dominates that narrow band.

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    min_scale : int
        Minimum window size for detrending.
    max_scale : int | None
        Maximum window size (default ``n // 4``).

    Returns
    -------
    float
        Hurst exponent: ``H < 0.5`` = mean-reverting,
        ``H ≈ 0.5`` = random walk, ``H > 0.5`` = trending.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 30:
        return 0.5

    if max_scale is None:
        max_scale = n // 4

    if min_scale >= max_scale:
        return 0.5

    # Cumulative sum (profile)
    y = np.cumsum(x - np.mean(x))

    scales = np.unique(np.round(np.geomspace(min_scale, max_scale, 20)).astype(int))
    log_scales: list[float] = []
    log_fluctuations: list[float] = []

    for scale in scales:
        n_windows = n // scale
        if n_windows < 2:
            continue

        rms_errors: list[float] = []
        for i in range(n_windows):
            segment = y[i * scale : (i + 1) * scale]
            x_seg = np.arange(scale)

            # Linear detrending
            slope, intercept = np.polyfit(x_seg, segment, 1)
            trend = slope * x_seg + intercept

            # Fluctuation around trend
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            if rms > 1e-10:
                rms_errors.append(rms)

        if rms_errors:
            fluctuation = np.sqrt(np.mean(np.array(rms_errors) ** 2))
            if fluctuation > 1e-10:
                log_scales.append(np.log(scale))
                log_fluctuations.append(np.log(fluctuation))

    if len(log_scales) < 3:
        return 0.5

    slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
    return float(np.clip(slope, 0.01, 0.99))
