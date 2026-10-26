"""
Tattva — Math utilities: pure mathematical functions.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Stateless, side-effect free functions operating on NumPy arrays and pandas structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Array operations ────────────────────────────────────────────────────────


def sigmoid(x: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    """Sigmoid transformation bounding values to [-1, 1].

    Uses the original Swayam formula: ``2 / (1 + exp(-x/scale)) - 1``.

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
