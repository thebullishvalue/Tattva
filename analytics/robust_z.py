"""
Tattva — Rolling robust z-scores.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — How unusual is today's value against its own trailing window?

Two centrings, because "unusual" has two meanings:

``center="median"`` — unusual relative to where the series has RECENTLY been:

    z_t = (x_t - median(w)) / (IQR(w) / 1.349)

``center="zero"`` — unusual relative to a MEANINGFUL ZERO, scaled by the
window's typical distance from it:

    z_t = x_t / (median(|w|) / 0.6745)

``w`` is the trailing window ``x_{t-w..t-1}``. It EXCLUDES the current point,
so a value is never scored against a distribution it is itself a member of,
and the statistic is causal. Both scales are standard robust estimates of a
standard deviation (``IQR / 1.349`` with 1.349 = 2·Φ⁻¹(0.75), and the median
absolute value over Φ⁻¹(0.75) = 0.6745; Huber & Ronchetti, *Robust
Statistics*), so on a Gaussian series ``z`` has unit spread and zone
cut-points at ±1 and ±2 mean one and two standard deviations.

The Mūla gap needs the second. Its zero IS the claim — price equal to the
level the cross-section implies — and median-centring discards it: after a
year in which the gap sat persistently rich, a gap of exactly zero (price AT
fair value) scores as unusually CHEAP against that year, and the window stack
then disagrees in sign with the oscillator it is meant to corroborate. Scaled
but not re-centred, every window shares the gap's sign and says only how
large it is against that window's own typical mispricing.

THE SCALE WAS WRONG BEFORE, by a factor of ~2.9. The module divided by the
2.5–97.5% range over 1.35 — the 95% range is 3.92σ for a normal, not the
1.349σ of the interquartile range the 1.35 belongs to — so every z it published
was ~0.34× a z-score (measured: sd 0.352 on iid N(0,1) at window 252), and the
"±1 / ±2" zones were really ±2.9σ / ±5.8σ. On the 3-to-10-point windows it was
used on, the "2.5/97.5% quantiles" were simply the window's min and max.

This module used to be called ``conformal`` and its function
``compute_conformal_zscores``. It is not conformal prediction (Vovk, Gammerman
& Shafer 2005): there is no calibration set and no coverage guarantee on a
serially dependent series. The engine's actual conformal band — adaptive
conformal inference on the fair-value level — lives in
``engines/mula/valuation.py``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

#: Φ⁻¹(0.75) - Φ⁻¹(0.25): the interquartile range of a unit normal.
IQR_TO_SD = 1.3489795003921634
#: Φ⁻¹(0.75): the median absolute value of a unit normal.
MAD0_TO_SD = 0.6744897501960817


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
def _robust_z_njit(series, window, min_periods, alpha, zero_center):
    n = series.shape[0]
    z_scores = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    upper_bounds = np.full(n, np.nan)
    scratch = np.empty(window, dtype=np.float64)
    absval = np.empty(window, dtype=np.float64)
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    for i in range(window, n):
        # Finite values from series[i-window:i] — the current point excluded.
        cnt = 0
        for j in range(i - window, i):
            v = series[j]
            if np.isfinite(v):
                scratch[cnt] = v
                absval[cnt] = abs(v)
                cnt += 1
        if cnt < min_periods:
            continue
        sv = np.sort(scratch[:cnt])
        if zero_center:
            center = 0.0
            scale = _percentile_linear(np.sort(absval[:cnt]), 0.5) / MAD0_TO_SD
        else:
            center = _percentile_linear(sv, 0.5)
            scale = (_percentile_linear(sv, 0.75) - _percentile_linear(sv, 0.25)) \
                / IQR_TO_SD
        if scale > 1e-10 and np.isfinite(series[i]):
            z_scores[i] = (series[i] - center) / scale
        lower_bounds[i] = _percentile_linear(sv, q_lo)
        upper_bounds[i] = _percentile_linear(sv, q_hi)
    return z_scores, lower_bounds, upper_bounds


def rolling_robust_z(
    series: np.ndarray,
    window: int,
    min_periods: int = 5,
    alpha: float = 0.05,
    center: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal rolling robust z-score of ``series`` against its trailing window.

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    window : int
        Trailing window length (the current point is excluded).
    min_periods : int
        Minimum finite observations in the window before a z is published.
        Clamped to ``window`` — a minimum the window cannot hold would leave the
        output permanently empty, which is how the engine's 3-day window spent
        its whole life as NaN (``min_periods`` 5 in a window of 3).
    alpha : float
        Tail mass of the empirical band returned alongside the z-score
        (0.05 → the [2.5, 97.5] percentiles). Display only; it does not enter
        the z-score.
    center : {"median", "zero"}
        What "usual" means — see the module docstring. ``"zero"`` keeps a
        meaningful zero (a mispricing gap's), scaling by the window's typical
        distance from it.

    Returns
    -------
    z_scores, lower_bounds, upper_bounds : np.ndarray
    """
    n = len(series)
    window = int(window)
    if n <= window or window < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if center not in ("median", "zero"):
        raise ValueError(f"center must be 'median' or 'zero', not {center!r}")
    min_periods = int(min(max(int(min_periods), 2), window))
    arr = np.ascontiguousarray(series, dtype=np.float64)
    return _robust_z_njit(arr, window, min_periods, float(alpha), center == "zero")
