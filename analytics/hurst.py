"""
Tattva v2.0.0 — Hurst exponent via Detrended Fluctuation Analysis (DFA).
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Measures long-term memory and mean-reversion in time series.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def hurst_dfa(
    series: np.ndarray,
    min_scale: int = 4,
    max_scale: int | None = None,
) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis.

    DFA is more robust than classical R/S analysis for short,
    noisy series (Peng et al., 1994).

    Proper lag range: ``min_scale = max(4, n/10)`` to ``max_scale = n/4``.

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
    min_scale = max(min_scale, n // 10)

    if min_scale >= max_scale:
        return 0.5

    # Cumulative sum (profile)
    y = np.cumsum(x - np.mean(x))

    scales = range(min_scale, max_scale + 1, max(1, (max_scale - min_scale) // 15))
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
