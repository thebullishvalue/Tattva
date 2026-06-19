"""
Tattva — Analog (Similar-Period) matcher.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Covariance-aware historical analog matching, ported from Arthagati's
``find_similar_periods`` (Mahalanobis + trajectory cosine + recency). The matcher
itself is unchanged; only its INPUTS are adapted to Tattva:

  • Feature vector is built from the quantities Tattva already computes per day
    (``engine.ts_data``) — conformal extension (AvgZ), net internal breadth, target
    momentum, realized volatility, and rolling Hurst — instead of Arthagati's mood
    features.
  • Forward-return horizons are the ACTIVE Signal-Horizon hold grid (so the
    precedent read inherits the lens chosen in the sidebar), not a fixed 5/20/60/90.

It answers an empirical, non-parametric question that complements the model
forecast: "when the target's state looked statistically like today, what did the
target do next?" — a base rate, descriptive not predictive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.hurst import hurst_dfa

# ── Blend weights (ported verbatim from Arthagati) ───────────────────────────
ANALOG_W_MAHA = 0.55   # Mahalanobis distance weight (covariance-aware state match)
ANALOG_W_TRAJ = 0.35   # trajectory cosine-similarity weight (path-shape match)
ANALOG_W_RECV = 0.10   # exponential recency-decay weight (prefer recent analogs)


# ════════════════════════════════════════════════════════════════════════════
# Core matcher — ported verbatim from Arthagati (arthagati.py)
# ════════════════════════════════════════════════════════════════════════════

def _ledoit_wolf_shrinkage(S: np.ndarray, n: int) -> np.ndarray:
    """Ledoit & Wolf (2004) analytical shrinkage estimator.
    Σ* = δ·F + (1−δ)·S  where F = (tr(S)/p)·I  (scaled identity target).
    Optimal δ minimises E[‖Σ*−Σ‖²_F] under standard asymptotics.
    Returns the shrunk covariance matrix — always well-conditioned.
    """
    p = S.shape[0]
    if p == 0 or n < 2:
        return S
    trace_S = np.trace(S)
    mu = trace_S / p                       # target = μ·I
    delta_mat = S - mu * np.eye(p)
    sum_sq = np.sum(delta_mat ** 2)        # ‖S − μI‖²_F
    # Optimal shrinkage intensity (OAS closed-form, Chen et al. 2010)
    rho_num = ((1.0 - 2.0 / p) * sum_sq + trace_S ** 2)
    rho_den = ((n + 1.0 - 2.0 / p) * (sum_sq + trace_S ** 2 / p))
    rho = np.clip(rho_num / max(rho_den, 1e-12), 0.0, 1.0)
    return (1.0 - rho) * S + rho * mu * np.eye(p)


def mahalanobis_distance_batch(features: np.ndarray, center: np.ndarray,
                               cov_matrix: np.ndarray) -> np.ndarray:
    """Mahalanobis distance: d_M = √((x−μ)ᵀ Σ⁻¹ (x−μ)).
    Uses Ledoit-Wolf analytical shrinkage for a well-conditioned covariance
    inverse, replacing ad-hoc diagonal regularization.
    """
    diff = features - center
    n_samples = features.shape[0]
    shrunk_cov = _ledoit_wolf_shrinkage(cov_matrix, n_samples)
    try:
        cov_inv = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(shrunk_cov)
    left = diff @ cov_inv
    d_sq = np.maximum(np.sum(left * diff, axis=1), 0)
    return np.sqrt(d_sq)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity — trajectory shape match irrespective of magnitude."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# ════════════════════════════════════════════════════════════════════════════
# Tattva adaptation — feature vector from engine.ts_data
# ════════════════════════════════════════════════════════════════════════════

def _rolling_hurst(price: np.ndarray, window: int = 120, step: int = 5) -> np.ndarray:
    """Rolling Hurst (DFA) over a trailing window, computed every ``step`` points
    and forward-filled — a regime persistence/mean-reversion feature. Mirrors
    Arthagati's stepped rolling-Hurst approach to keep it cheap on ~2k-row series.
    """
    n = len(price)
    out = np.full(n, 0.5, dtype=np.float64)
    if n < window:
        return out
    log_p = np.log(np.where(price > 0, price, np.nan))
    rets = np.diff(log_p)
    for i in range(window, n, step):
        seg = rets[i - window:i]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 30:
            out[i] = hurst_dfa(seg)
    return pd.Series(out).replace(0.0, np.nan).ffill().fillna(0.5).to_numpy()


def _build_feature_frame(ts: pd.DataFrame, mom_window: int) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the per-day analog state matrix from Tattva's engine.ts_data.

    Features (each guarded by availability, like Arthagati):
      • Momentum     — trailing ``mom_window``-day log-return of the target Price
      • Realized Vol — rolling σ of daily log-returns over ``mom_window``
      • AvgZ         — conformal multi-lookback z-state (extension), if present
      • NetBreadth   — OversoldBreadth − OverboughtBreadth, if present
      • Hurst        — rolling DFA Hurst of the target Price

    Returns ``(frame, feature_cols)`` where ``frame`` carries the feature columns
    plus ``Price`` and ``Date`` (forward returns and recency are derived from these).
    """
    if ts is None or ts.empty or "Price" not in ts.columns:
        return pd.DataFrame(), []

    df = ts.reset_index(drop=True).copy()
    price = pd.to_numeric(df["Price"], errors="coerce").to_numpy(dtype=np.float64)
    log_ret = pd.Series(np.log(np.where(price > 0, price, np.nan))).diff()

    feat = pd.DataFrame(index=df.index)
    feat["Momentum"] = log_ret.rolling(mom_window, min_periods=mom_window).sum()
    feat["RealizedVol"] = log_ret.rolling(mom_window, min_periods=mom_window).std()

    if "AvgZ" in df.columns:
        feat["AvgZ"] = pd.to_numeric(df["AvgZ"], errors="coerce")
    if {"OversoldBreadth", "OverboughtBreadth"}.issubset(df.columns):
        feat["NetBreadth"] = (pd.to_numeric(df["OversoldBreadth"], errors="coerce")
                              - pd.to_numeric(df["OverboughtBreadth"], errors="coerce"))
    feat["Hurst"] = _rolling_hurst(price, window=max(60, mom_window * 3))

    feature_cols = [c for c in ("Momentum", "RealizedVol", "AvgZ", "NetBreadth", "Hurst")
                    if c in feat.columns]
    feat["Price"] = price
    feat["Date"] = df["Date"].to_numpy() if "Date" in df.columns else df.index.to_numpy()
    return feat, feature_cols


def find_similar_periods(
    ts: pd.DataFrame,
    target_col: str,
    hold_horizons: tuple[int, ...] = (3, 5, 10, 20),
    *,
    mom_window: int = 20,
    top_n: int = 10,
    recency_weight: float = ANALOG_W_RECV,
) -> list[dict]:
    """Tattva analog finder — ported scoring, Tattva inputs.

    3-part scoring (Arthagati blend, unchanged):
      1. Mahalanobis distance (55%) — covariance-aware state match
      2. Trajectory cosine similarity (35%) — detrended Price-path shape
      3. Exponential recency decay (10%) — prefer recent analogs

    Returns top-N analogs as dicts with the target's forward returns at each
    ``hold_horizons`` value (% Price change t→t+h).
    """
    feat, feature_cols = _build_feature_frame(ts, mom_window)
    if feat.empty or len(feature_cols) < 2:
        return []

    n = len(feat)
    purge = int(max(hold_horizons)) if hold_horizons else 20
    # Exclude the tail: those rows have no realized forward path yet.
    historical = feat.iloc[:n - purge].copy()
    if len(historical) < 30:
        return []

    latest = feat.iloc[-1]
    current_vec = latest[feature_cols].to_numpy(dtype=np.float64)
    hist_matrix = historical[feature_cols].to_numpy(dtype=np.float64)

    # Clean NaN/Inf → column medians (port of Arthagati's cleaning)
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)

    # ── Part 1: Mahalanobis (55%) ───────────────────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)

    # ── Part 2: Trajectory cosine similarity (35%) ──────────────────────────
    traj_window = mom_window
    traj_sim = np.zeros(len(historical))
    price_all = feat["Price"].to_numpy(dtype=np.float64)
    if n > traj_window:
        _x = np.arange(traj_window, dtype=np.float64)
        _xm = _x - _x.mean()
        _xvar = np.sum(_xm ** 2)

        def _ls_detrend(traj: np.ndarray) -> np.ndarray:
            if _xvar < 1e-12:
                return traj - traj.mean()
            slope = np.sum(_xm * (traj - traj.mean())) / _xvar
            return traj - (traj.mean() + slope * _xm)

        cur_traj = price_all[-traj_window:]
        ct = _ls_detrend(cur_traj)
        for j in range(len(historical)):
            pos = j  # historical is feat.iloc[:n-purge] → positional index aligns
            if pos >= traj_window:
                ht = _ls_detrend(price_all[pos - traj_window:pos])
                traj_sim[j] = (cosine_similarity(ct, ht) + 1) / 2

    # ── Part 3: Exponential recency decay (10%) ─────────────────────────────
    dates = historical["Date"]
    if np.issubdtype(np.asarray(dates).dtype, np.datetime64):
        days_since = (pd.Timestamp(latest["Date"]) - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    else:
        days_since = (float(latest["Date"]) - np.asarray(dates, dtype=float))
    recency = np.exp(-np.log(2) * np.clip(days_since, 0, None) / 365.0) * recency_weight
    recency_norm = recency / max(recency.max(), 1e-6)

    # ── Combined ────────────────────────────────────────────────────────────
    combined = ANALOG_W_MAHA * maha_sim + ANALOG_W_TRAJ * traj_sim + ANALOG_W_RECV * recency_norm
    historical = historical.copy()
    historical["similarity"] = combined
    top = historical.nlargest(top_n, "similarity")

    results: list[dict] = []
    for pos, row in zip(top.index, top.to_dict("records")):
        price_at = float(row["Price"]) if row["Price"] and row["Price"] > 0 else None
        fwd: dict[int, float | None] = {}
        for h in hold_horizons:
            fi = int(pos) + h
            if fi < len(price_all) and price_at:
                fwd[h] = (price_all[fi] / price_at - 1) * 100
            else:
                fwd[h] = None
        date_val = row["Date"]
        date_str = (pd.Timestamp(date_val).strftime("%Y-%m-%d")
                    if not isinstance(date_val, (int, np.integer)) else f"t={int(date_val)}")
        results.append({
            "date": date_str,
            "similarity": float(row["similarity"]),
            "price": price_at or 0.0,
            "momentum": float(row.get("Momentum", 0.0) or 0.0) * 100,  # → %
            "realized_vol": float(row.get("RealizedVol", 0.0) or 0.0) * 100,
            "avgz": float(row.get("AvgZ", 0.0) or 0.0),
            "net_breadth": float(row.get("NetBreadth", 0.0) or 0.0),
            "hurst": float(row.get("Hurst", 0.5) or 0.5),
            "fwd": {int(h): fwd[h] for h in hold_horizons},
        })
    return results


def summarize_forward(periods: list[dict], hold_horizons: tuple[int, ...]) -> dict[int, dict]:
    """Per-horizon base-rate summary across the analogs: median return, % positive, n."""
    out: dict[int, dict] = {}
    for h in hold_horizons:
        vals = [p["fwd"].get(int(h)) for p in periods]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            out[int(h)] = {
                "median": float(np.median(vals)),
                "positive_pct": sum(1 for v in vals if v > 0) / len(vals) * 100.0,
                "n": len(vals),
            }
    return out
