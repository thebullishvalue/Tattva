"""
Tattva v2.0.0 — Structural break detection via Bai-Perron multiple breakpoint test.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Identifies regime shifts in time series data.
"""

from __future__ import annotations

import logging

import numpy as np

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
    """Heuristic change-point detection via rolling mean divergence."""
    n = len(series)
    window = max(int(n * trim), 5)
    trim_n = int(n * trim)

    rolling_mean = np.convolve(series, np.ones(window) / window, mode="valid")
    diffs = np.abs(np.diff(rolling_mean))

    # Mask out trim regions
    pad = window // 2
    diffs[: trim_n - pad] = 0
    diffs[-(trim_n - pad):] = 0

    # Pick top-k peaks
    break_indices = []
    diffs_copy = diffs.copy()
    for _ in range(max_breaks):
        peak = int(np.argmax(diffs_copy))
        if diffs_copy[peak] == 0:
            break
        break_indices.append(peak + pad)
        # Suppress nearby peaks
        lo = max(0, peak - window)
        hi = min(len(diffs_copy), peak + window)
        diffs_copy[lo:hi] = 0

    return sorted(break_indices)
