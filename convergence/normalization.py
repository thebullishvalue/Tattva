"""
Tattva — Shared normalization math for the Unified Convergence Signal.
तत्त्व (Tattva) — "Principle / Essence"

Single source of truth for the math behind the Convergence Analysis cards and
the Unified Signal — Normalized Convergence plot. Both call into here, so the
card values are guaranteed to match the plot.

Pipeline:
  align(aarambh_ts, nirnay_daily)   →  dates, raw_a[], raw_n[]
  compute_norm_params(raw_a, raw_n) →  {mu_a, sigma_a, mu_n, sigma_n}
  zscore_clip(arr, mu, sigma)       →  (arr - mu) / sigma / 3 clipped to [-1, +1]
  classify_normalized_signal(v)     →  STRONG BUY / BUY / HOLD / SELL / STRONG SELL
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ── Signal classification thresholds (factory defaults) ─────────────────────
# These match the plot marker thresholds and are also the fallback when
# Intelligence Mode is not active. The calibrated thresholds in a saved
# profile may be asymmetric (`buy_strong != -sell_strong`).
_STRONG = 0.5
_MODERATE = 0.3

DEFAULT_THRESHOLDS: dict[str, float] = {
    "buy_strong":     -_STRONG,
    "buy_moderate":   -_MODERATE,
    "sell_moderate":  +_MODERATE,
    "sell_strong":    +_STRONG,
}


def classify_normalized_signal(
    v: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map a normalized convergence value (in ``[-1, +1]``) to a signal label.

    Args:
        v: the normalized convergence value.
        thresholds: optional dict with keys ``buy_strong``, ``buy_moderate``,
            ``sell_moderate``, ``sell_strong``. When ``None``, falls back to
            the symmetric factory defaults (``±0.3`` / ``±0.5``). Used by
            Intelligence Mode to apply calibrated thresholds from a saved
            profile.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    if v <= t["buy_strong"]:
        return "STRONG BUY"
    if v <= t["buy_moderate"]:
        return "BUY"
    if v >= t["sell_strong"]:
        return "STRONG SELL"
    if v >= t["sell_moderate"]:
        return "SELL"
    return "HOLD"


def _nirnay_signal_column(df: pd.DataFrame) -> str | None:
    """Return the first available Nirnay average-signal column, or None."""
    for c in ("avg_unified_osc", "Avg_Signal", "avg_signal"):
        if c in df.columns:
            return c
    # Case-insensitive fallback (Avg-Signal, AVG_SIGNAL, etc.)
    for c in df.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("avg_signal", "avg_unified_osc"):
            return c
    return None


def align_aarambh_nirnay(
    aarambh_ts: pd.DataFrame | None,
    nirnay_daily: pd.DataFrame | None,
    filter_dates: Iterable[str] | None = None,
) -> tuple[list, list[float], list[float]]:
    """Align Aarambh ``ConvictionRaw`` and Nirnay average signal on overlapping dates.

    Args:
        aarambh_ts: DataFrame with a ``ConvictionRaw`` column and either a
            ``Date`` column or a DatetimeIndex.
        nirnay_daily: DataFrame indexed by date with a Nirnay avg-signal column
            (``avg_unified_osc`` / ``Avg_Signal`` / ``avg_signal``).
        filter_dates: Optional iterable of date-strings (``YYYY-MM-DD``); rows
            whose date is not in this set are skipped (used by the plot to
            honour the user's lookback selection).

    Returns:
        ``(dates, aarambh_raw, nirnay_raw)`` — three parallel lists. Empty
        lists if either input is missing or there are no overlapping dates.
    """
    if aarambh_ts is None or "ConvictionRaw" not in aarambh_ts.columns:
        return [], [], []
    if nirnay_daily is None or nirnay_daily.empty:
        return [], [], []

    df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
    avg_col = _nirnay_signal_column(df_n)
    if avg_col is None:
        return [], [], []

    nirnay_lookup: dict[str, float] = {}
    for idx in df_n.index:
        key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
        nirnay_lookup[key] = float(df_n.loc[idx][avg_col])

    a_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
    date_series = a_dedup["Date"] if "Date" in a_dedup.columns else a_dedup.index

    filter_set = set(filter_dates) if filter_dates is not None else None

    dates: list = []
    raw_a: list[float] = []
    raw_n: list[float] = []
    for d_val in date_series:
        ts_key = str(d_val.date()) if hasattr(d_val, "date") else str(pd.Timestamp(d_val).date())
        if filter_set is not None and ts_key not in filter_set:
            continue
        if ts_key not in nirnay_lookup:
            continue
        try:
            raw_a.append(float(a_dedup.loc[d_val, "ConvictionRaw"]))
            raw_n.append(nirnay_lookup[ts_key])
            dates.append(d_val if hasattr(d_val, "date") else pd.Timestamp(ts_key))
        except Exception:
            pass
    return dates, raw_a, raw_n


def compute_norm_params(raw_a: list[float], raw_n: list[float]) -> dict[str, float]:
    """Compute mean/std for z-scoring using only history up to the last point.

    Uses an expanding window anchored at the first observation so the stats
    at each point are computed only from data available at that time.
    The returned mu/sigma reflect the state at the END of the series (i.e.
    the most recent causal estimate) and are used to normalize the latest
    value for the metric cards.  ``sigma`` is floored at 1e-10.
    """
    arr_a = np.array(raw_a, dtype=np.float64) if raw_a else np.array([])
    arr_n = np.array(raw_n, dtype=np.float64) if raw_n else np.array([])

    def _expanding_stats(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return 0.0, 1.0
        s = pd.Series(arr)
        mu = float(s.expanding().mean().iloc[-1])
        sigma = max(float(s.expanding().std().iloc[-1]), 1e-10) if len(arr) > 1 else 1.0
        return mu, sigma

    mu_a, sigma_a = _expanding_stats(arr_a)
    mu_n, sigma_n = _expanding_stats(arr_n)
    return {"mu_a": mu_a, "sigma_a": sigma_a, "mu_n": mu_n, "sigma_n": sigma_n}


def zscore_clip(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Z-score with ``/3`` and clip to ``[-1, +1]`` (matches plot convention)."""
    if sigma < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - mu) / sigma / 3.0, -1.0, 1.0)


def compute_normalized_convergence(
    aarambh_ts: pd.DataFrame | None,
    nirnay_daily: pd.DataFrame | None,
    thresholds: dict[str, float] | None = None,
) -> dict | None:
    """Latest normalized convergence value + per-system contributions.

    Mirrors what the Unified Signal plot's top row displays at its last point.
    Returns ``None`` if alignment yields no rows.

    Args:
        aarambh_ts, nirnay_daily: time-series inputs.
        thresholds: optional calibrated thresholds (from Intelligence Mode).
            When ``None``, the symmetric factory defaults are used for the
            ``signal`` label.
    """
    _, raw_a, raw_n = align_aarambh_nirnay(aarambh_ts, nirnay_daily)
    if not raw_a:
        return None
    arr_a = np.array(raw_a, dtype=np.float64)
    arr_n = np.array(raw_n, dtype=np.float64)
    # Causal expanding-window z-scores: each point is normalised using only
    # the history available up to that date (no future data leakage).
    s_a, s_n = pd.Series(arr_a), pd.Series(arr_n)
    exp_mu_a = s_a.expanding().mean()
    exp_sigma_a = s_a.expanding().std().clip(lower=1e-10).fillna(1.0)
    exp_mu_n = s_n.expanding().mean()
    exp_sigma_n = s_n.expanding().std().clip(lower=1e-10).fillna(1.0)
    norm_a = np.clip((arr_a - exp_mu_a.to_numpy()) / exp_sigma_a.to_numpy() / 3.0, -1.0, 1.0)
    norm_n = np.clip((arr_n - exp_mu_n.to_numpy()) / exp_sigma_n.to_numpy() / 3.0, -1.0, 1.0)
    norm_avg = (norm_a + norm_n) / 2.0
    latest = float(norm_avg[-1])
    return {
        "value": latest,
        "signal": classify_normalized_signal(latest, thresholds),
        "aarambh_norm": float(norm_a[-1]),
        "nirnay_norm": float(norm_n[-1]),
        "aarambh_raw": float(arr_a[-1]),
        "nirnay_raw": float(arr_n[-1]),
    }
