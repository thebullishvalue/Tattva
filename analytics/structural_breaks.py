"""
Tattva — Structural break detection via Bai-Perron multiple breakpoint test.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Identifies regime shifts in time series data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.regime_switching.bai_perron import BaiPerronTest  # type: ignore[import-not-found]

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def detect_structural_breaks(
    series: np.ndarray,
    max_breaks: int = 3,
    trim: float = 0.15,
) -> list[int]:
    """Bai-Perron multiple breakpoint detection.

    Falls back to a rolling-mean change-point heuristic if statsmodels
    does not expose ``BaiPerronTest`` (available in unreleased 0.15+).

    Parameters
    ----------
    series : np.ndarray
        Input time-series.
    max_breaks : int
        Maximum number of structural breaks to detect.
    trim : float
        Trim fraction for each segment (default 15%).

    Returns
    -------
    list[int]
        Break indices relative to the series start.
    """
    if len(series) < 50:
        return []

    # ── Primary: statsmodels Bai-Perron ───────────────────────────────
    if _HAS_STATSMODELS:
        try:
            bp_test = BaiPerronTest(series)
            result = bp_test.test_breaks(max_breaks, trim=trim)
            if hasattr(result, "break_dates") and result.break_dates is not None:
                return [int(bd) for bd in result.break_dates]
        except Exception as e:
            logging.warning("Bai-Perron test failed, using fallback: %s", e)

    # ── Fallback: rolling mean change-point detection ─────────────────
    try:
        return _rolling_mean_breaks(series, max_breaks, trim)
    except Exception as e:
        logging.warning("Fallback structural break detection failed: %s", e)
        return []


def _rolling_mean_breaks(
    series: np.ndarray,
    max_breaks: int,
    trim: float,
) -> list[int]:
    """Heuristic change-point detection via trailing rolling mean divergence.

    Uses a strictly causal trailing window so each break position is
    determined solely from data up to and including that point.
    """
    n = len(series)
    window = max(int(n * trim), 5)
    trim_n = int(n * trim)

    # Causal trailing mean: index t uses only obs[t-window+1 … t].
    s = pd.Series(series)
    rolling_mean = s.rolling(window, min_periods=window).mean().to_numpy()

    # diff[t] = |mean[t] - mean[t-1]|; NaN rows (warm-up) → 0.
    diffs = np.abs(np.diff(np.nan_to_num(rolling_mean, nan=0.0)))

    # Mask out trim regions (both ends of the usable range).
    diffs[:trim_n] = 0
    if trim_n > 0:
        diffs[-trim_n:] = 0

    # Pick top-k non-adjacent peaks; break index maps directly back to the
    # original series (diff[t] reflects the change arriving at t+1).
    break_indices = []
    diffs_copy = diffs.copy()
    for _ in range(max_breaks):
        peak = int(np.argmax(diffs_copy))
        if diffs_copy[peak] == 0:
            break
        break_indices.append(peak + 1)
        lo = max(0, peak - window)
        hi = min(len(diffs_copy), peak + window)
        diffs_copy[lo:hi] = 0

    return sorted(break_indices)
