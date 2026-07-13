"""
Tattva — Math utilities: pure mathematical functions.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Stateless, side-effect free functions operating on NumPy arrays and pandas structures.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

# ─── Array operations ────────────────────────────────────────────────────────


def _safe_array_operation(
    arr: np.ndarray,
    operation: Literal["mean", "std", "min", "max", "sum"],
    default: float = 0.0,
) -> float:
    """Compute common array operations with NaN/Inf handling.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    operation : str
        One of ``mean``, ``std``, ``min``, ``max``, ``sum``.
    default : float
        Value returned when no valid data exists.
    """
    arr = np.asarray(arr)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return default
    clean = arr[valid]
    ops: dict[str, callable] = {
        "mean": lambda c: float(np.mean(c)),
        "std": lambda c: float(np.std(c)) if len(c) > 1 else default,
        "min": lambda c: float(np.min(c)),
        "max": lambda c: float(np.max(c)),
        "sum": lambda c: float(np.sum(c)),
    }
    fn = ops.get(operation)
    return fn(clean) if fn else default


# ─── Transformations ─────────────────────────────────────────────────────────
# CANONICAL versions of Nirnay's private helpers (audit finding F11): this
# module previously carried its OWN copies of sigmoid/zscore/ATR that were
# dead (engines/nirnay.py always called its private _sigmoid/_zscore_clipped/
# _calculate_atr instead) and, worse, NOT equivalent — zscore_clipped here
# lacked Nirnay's ffill/min_periods=1/causal-shift(1) semantics, so a future
# caller picking THIS module's version would silently get different numbers.
# Nirnay now imports these; there is exactly one implementation of each.


def sigmoid(x: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    """Sigmoid transformation bounding values to [-1, 1].

    Uses the original Nirnay formula: ``2 / (1 + exp(-x/scale)) - 1``.

    Parameters
    ----------
    x : np.ndarray | float
        Input values.
    scale : float
        Divisor controlling the curve steepness. Larger = gentler slope.
    """
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def zscore_clipped(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    """Rolling CAUSAL z-score with outlier clipping.

    Uses ``shift(1)`` so today's own value never biases the mean/std it is
    scored against, and ffills/zero-fills leading gaps so the oscillator
    stack (which consumes this on wide, sparsely-populated macro frames)
    doesn't propagate NaN indefinitely.

    Parameters
    ----------
    series : pd.Series
        Input time-series.
    window : int
        Rolling window size.
    clip : float
        Maximum absolute z-score before clipping.
    """
    series_filled = series.ffill().fillna(0)
    roll_mean = series_filled.rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
    roll_std = series_filled.rolling(window=window, min_periods=1).std().shift(1).fillna(0)
    z = (series_filled - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range — exponential moving average variant.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``High``, ``Low``, ``Close`` columns.
    length : int
        Smoothing period.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ─── Classification helpers ──────────────────────────────────────────────────


def _classify_zones(
    z_scores: np.ndarray, z_threshold: float = 1.0, z_extreme: float = 2.0
) -> np.ndarray:
    """Map z-scores to valuation zone labels.

    Parameters
    ----------
    z_scores : np.ndarray
        Raw z-scores.
    z_threshold : float
        Boundary between fair value and over/undervalued.
    z_extreme : float
        Boundary between over/undervalued and extreme zones.
    """
    condlist = [
        z_scores > z_extreme,
        z_scores > z_threshold,
        z_scores > -z_threshold,
        z_scores > -z_extreme,
    ]
    choicelist = ["Extreme Over", "Overvalued", "Fair Value", "Undervalued"]
    zones = np.select(condlist, choicelist, default="Extreme Under")
    np.putmask(zones, np.isnan(z_scores), "N/A")
    return zones


def _detect_crossover_signals(
    z_scores: np.ndarray, threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Detect z-score threshold crossings as boolean signal arrays.

    A buy signal fires when z crosses below ``-threshold``.
    A sell signal fires when z crosses above ``+threshold``.
    """
    n = len(z_scores)
    if n < 2:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    z_curr = z_scores[1:]
    z_prev = z_scores[:-1]
    valid = np.isfinite(z_curr) & np.isfinite(z_prev)
    buy_cond = valid & (z_curr < -threshold) & (z_prev >= -threshold)
    sell_cond = valid & (z_curr > threshold) & (z_prev <= threshold)
    buy_signals = np.zeros(n, dtype=bool)
    sell_signals = np.zeros(n, dtype=bool)
    buy_signals[1:] = buy_cond
    sell_signals[1:] = sell_cond
    return buy_signals, sell_signals


def _compute_significance(values: list[float]) -> dict[str, float]:
    """Compute t-statistic and p-value for a list of values.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``t_stat``, ``p_value``, ``n``.
    """
    n = len(values)
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "p_value": 1.0, "n": n}
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    if std_val < 1e-10:
        return {"mean": mean_val, "std": std_val, "t_stat": np.inf, "p_value": 0.0, "n": n}
    se = std_val / np.sqrt(n)
    t_stat = mean_val / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return {
        "mean": mean_val,
        "std": std_val,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": n,
    }


def _apply_conviction_bounds(score: np.ndarray | float, max_bound: float = 100.0) -> np.ndarray | float:
    """Apply soft bounds to conviction score via tanh transformation.

    Parameters
    ----------
    score : np.ndarray | float
        Raw conviction score(s).
    max_bound : float
        Asymptotic bound (default ±100).
    """
    return max_bound * np.tanh(np.asarray(score, dtype=np.float64) / max_bound)
