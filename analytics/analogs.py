"""
Tattva — Analog (Similar-Period) matcher.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Covariance-aware historical analog matching, ported from Arthagati's
``find_similar_periods`` (Mahalanobis + trajectory cosine + recency). The matcher
itself is unchanged; only its INPUTS are adapted to Tattva:

  • Feature vector is built from the quantities Tattva already computes per day
    (``engine.ts_data``) — robust-quantile extension (AvgZ), net internal breadth,
    target momentum, realized volatility, and rolling Hurst — instead of
    Arthagati's mood features.
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
# Blend re-tuned for Tattva (2026-06-20, research/analog_tuning_study.py + research/analog_confirm.py:
# 13 targets, non-overlapping OOS IC full + recent-half). The ported Arthagati blend
# (.55/.35/.10) was actively HURTING: trajectory adds ~nothing and recency degrades
# the recent regime. PURE Mahalanobis state-matching is the clear winner — it
# recovers the decayed recent edge (10d recent IC −0.010 → +0.079; 20d −0.083 →
# +0.095) while holding full-sample IC. So trajectory + recency are dropped (weight
# 0 → their computation is skipped entirely, also a live speedup).
ANALOG_W_MAHA = 1.0    # Mahalanobis distance weight (covariance-aware state match)
ANALOG_W_TRAJ = 0.0    # trajectory cosine-similarity — DROPPED (no lift, hurt recent)
ANALOG_W_RECV = 0.0    # exponential recency-decay — DROPPED (degraded recent regime)


# ════════════════════════════════════════════════════════════════════════════
# Core matcher — ported verbatim from Arthagati (arthagati.py)
# ════════════════════════════════════════════════════════════════════════════

def _ledoit_wolf_shrinkage(S: np.ndarray, n: int) -> np.ndarray:
    """Oracle Approximating Shrinkage estimator (Chen, Wiesel, Eldar & Hero
    2010, IEEE Trans. Signal Processing 58(10), eq. 23, with the O(1/p)
    ``2/p`` correction term omitted — negligible for large p, but that omission
    is what makes this match the reference OAS implementation used to verify
    it (``sklearn.covariance.OAS``: "The factor 2/p is omitted since it does
    not impact the value of the estimator for large p"). Verified to agree
    with ``sklearn.covariance.OAS`` to ~1e-16 on random SPD inputs, including
    the small p (3-4 feature) regime this module actually runs in.
    Σ* = ρ·F + (1−ρ)·S  where F = (tr(S)/p)·I  (scaled identity target).
    Returns the shrunk covariance matrix — always well-conditioned.

    (Name kept for import-site compatibility; this is OAS, not Ledoit & Wolf
    2004 — a related but distinct, non-OAS shrinkage intensity. The formula
    below was previously mis-transcribed: it had tr(S^2) - tr(S)^2/p in the
    NUMERATOR and (1-2/p)*(tr(S^2)) + tr(S)^2 arrangement swapped relative to
    the denominator, which under-shrinks exactly where shrinkage matters most
    — near-isotropic S, where the true rho should approach 1.)
    """
    p = S.shape[0]
    if p == 0 or n < 2:
        return S
    trace_S = np.trace(S)
    mu = trace_S / p                       # target = μ·I
    alpha = np.mean(S * S)                 # tr(S^2)/p^2 via the Frobenius-norm identity
    mu_sq = mu ** 2
    rho_num = alpha + mu_sq
    rho_den = (n + 1.0) * (alpha - mu_sq / p)
    rho = min(max(rho_num / rho_den, 0.0), 1.0) if rho_den != 0 else 1.0
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


def select_analogs_theiler(
    scores: np.ndarray, top_n: int, gap: int,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy top-N selection under a Theiler exclusion window.

    Theiler (1986, Phys. Rev. A 34), adopted for analog/nearest-neighbor
    forecasting by Farmer & Sidorowich (1987, Phys. Rev. Lett. 59): candidates
    within `gap` rows of an already-accepted analog are excluded, so the
    returned indices are drawn from genuinely distinct episodes rather than a
    run of adjacent days whose rolling-window state (and h-day forward
    outcome) is nearly identical. Plain top-N-by-score (``argpartition`` /
    ``nlargest``) does not have this property and can return "top_n analogs"
    that are really 1-3 independent observations repeated.

    ``positions``: the TEMPORAL row position of each candidate (same length as
    ``scores``). Required when the candidate pool has been filtered (e.g. the
    engine warm-up rows removed) — array offsets then no longer measure time,
    and the exclusion window must be applied on the original row positions.
    Defaults to 0..n-1 (unfiltered pool: offset == time).

    Returns up to `top_n` integer offsets INTO ``scores``, best-first.
    """
    if positions is None:
        positions = np.arange(len(scores))
    order = np.argsort(scores)[::-1]
    accepted: list[int] = []
    accepted_time: list[int] = []
    for pos in order:
        p = int(pos)
        t = int(positions[p])
        if all(abs(t - a) >= gap for a in accepted_time):
            accepted.append(p)
            accepted_time.append(t)
            if len(accepted) >= top_n:
                break
    return np.array(accepted, dtype=np.int64)


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

    Feature set re-tuned for Tattva (research/analog_tuning_study.py): AvgZ was DROPPED —
    it degraded the recent regime (10d recent IC −0.010 → +0.034 without it) while
    NetBreadth proved critical and the candidate extras (ModelSpread, ExtremeBreadth,
    SignalBreadth, ConvictionRaw, MomentumLong) added nothing. Kept (availability-
    guarded):
      • Momentum     — trailing ``mom_window``-day log-return of the target Price
      • Realized Vol — rolling σ of daily log-returns over ``mom_window``
      • NetBreadth   — OversoldBreadth − OverboughtBreadth, if present (key feature)
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

    if {"OversoldBreadth", "OverboughtBreadth"}.issubset(df.columns):
        feat["NetBreadth"] = (pd.to_numeric(df["OversoldBreadth"], errors="coerce")
                              - pd.to_numeric(df["OverboughtBreadth"], errors="coerce"))
        # The engine's own [0, MIN_TRAIN_SIZE) warm-up (no genuine walk-forward
        # forecast yet — see A3 in the audit) reads OversoldBreadth ==
        # OverboughtBreadth == 0, so NetBreadth would otherwise be a fabricated
        # 0.0 "neutral" reading rather than genuinely missing. Force it to NaN
        # there so the analog historical pool excludes the warm-up (median-fill
        # in find_similar_periods then treats it as missing, matching every
        # other NaN-guarded consumer of this engine output).
        if "Valid" in df.columns:
            feat.loc[~df["Valid"].astype(bool), "NetBreadth"] = np.nan
    feat["Hurst"] = _rolling_hurst(price, window=max(60, mom_window * 3))

    feature_cols = [c for c in ("Momentum", "RealizedVol", "NetBreadth", "Hurst")
                    if c in feat.columns]
    feat["Price"] = price
    feat["Date"] = df["Date"].to_numpy() if "Date" in df.columns else df.index.to_numpy()
    # Carried DISPLAY-ONLY columns (not in feature_cols, never matched on):
    #   • AvgZ — the engine's extension z-score at the analog's date. It was
    #     dropped from the MATCHING feature set in the 2.2 re-tune, but the
    #     Precedent tab's analog cards still display "Extension (Z)" and key
    #     their tier badge/color off it; without carrying it, every card
    #     silently read the dict default 0.0 → permanently "Neutral" badges
    #     (round-2 audit finding M1).
    #   • ValidRow — the engine's Valid flag (False through the walk-forward
    #     warm-up). Lets find_similar_periods exclude rows whose NetBreadth
    #     would be median-filled fabrication from the candidate pool (M2).
    if "AvgZ" in df.columns:
        feat["AvgZ"] = pd.to_numeric(df["AvgZ"], errors="coerce").to_numpy(dtype=np.float64)
    if "Valid" in df.columns:
        feat["ValidRow"] = df["Valid"].astype(bool).to_numpy()
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
    """Tattva analog finder — ported scoring machinery, Tattva-tuned weights.

    Scoring = ANALOG_W_MAHA · Mahalanobis + ANALOG_W_TRAJ · trajectory-cosine
    + ANALOG_W_RECV · recency. The SHIPPED weights are **1 / 0 / 0** (pure
    covariance-aware Mahalanobis state match) — the ported Arthagati blend
    (.55/.35/.10) was re-tuned for Tattva and the trajectory/recency parts
    were dropped as harmful to the recent regime (see module header; their
    computation is skipped entirely at weight 0). Candidates are drawn from
    genuinely-forecast history only (engine warm-up rows excluded) and
    selected under a Theiler exclusion window so each returned analog is a
    distinct episode.

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
    # Exclude the engine's walk-forward WARM-UP rows from the candidate pool
    # (ValidRow False, see _build_feature_frame): their NetBreadth is genuinely
    # missing, and matching against a median-filled fabrication is not a state
    # match. The frame keeps its original RangeIndex labels after this filter,
    # so `historical.index` remains the TEMPORAL row position — the forward-
    # return lookups and the Theiler gap below both rely on that.
    if "ValidRow" in historical.columns:
        historical = historical[historical["ValidRow"]]
    if len(historical) < 30:
        return []

    latest = feat.iloc[-1]
    current_vec = latest[feature_cols].to_numpy(dtype=np.float64)
    # .copy() is load-bearing: on Streamlit Cloud DataFrame.to_numpy() can return a
    # READ-ONLY view (consolidated/cached buffers), and the median-fill below writes
    # in place → "assignment destination is read-only" without a writable copy.
    hist_matrix = historical[feature_cols].to_numpy(dtype=np.float64).copy()

    # Clean NaN/Inf → column medians (port of Arthagati's cleaning)
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)

    # ── Part 1: Mahalanobis (the signal — covariance-aware state match) ─────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)

    # ── Part 2: Trajectory cosine similarity (DROPPED → skipped when weight 0) ─
    traj_window = mom_window
    traj_sim = np.zeros(len(historical))
    price_all = feat["Price"].to_numpy(dtype=np.float64)
    if ANALOG_W_TRAJ > 0 and n > traj_window:
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
        # historical's RangeIndex labels ARE the temporal row positions (the
        # frame was built with reset_index, and the ValidRow filter above
        # preserves labels) — use them, not the filtered array offset j, to
        # slice the price path.
        _orig_pos = historical.index.to_numpy()
        for j in range(len(historical)):
            pos = int(_orig_pos[j])
            if pos >= traj_window:
                ht = _ls_detrend(price_all[pos - traj_window:pos])
                traj_sim[j] = (cosine_similarity(ct, ht) + 1) / 2

    # ── Part 3: Exponential recency decay (DROPPED → skipped when weight 0) ──
    recency_norm = np.zeros(len(historical))
    if ANALOG_W_RECV > 0:
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

    # Theiler exclusion window (see select_analogs_theiler docstring): without
    # it, top-N-by-similarity typically returns a RUN of adjacent days from
    # 1-3 historical episodes whose h-day forward outcomes overlap almost
    # completely — "10 analogs" that are really 1-3 independent observations,
    # inflating summarize_forward's apparent median/positive_pct precision.
    # gap = the longer of the momentum window (how far back the STATE feature
    # vector looks) and the longest requested forward horizon (how far the
    # OUTCOME extends) — below that, either the state or the outcome (or
    # both) would overlap a neighbor's.
    gap = max(int(mom_window), int(max(hold_horizons)) if hold_horizons else 0, 1)
    # positions= carries the ORIGINAL temporal row positions (RangeIndex labels)
    # because the pool above may be filtered (warm-up rows removed) — the
    # Theiler gap must be measured in trading days, not filtered-array offsets.
    accepted = select_analogs_theiler(
        combined, top_n, gap, positions=historical.index.to_numpy(),
    )
    top = historical.iloc[accepted]

    def _num(row: dict, key: str, default: float) -> float:
        """Finite-or-default coercion — carried columns can be NaN."""
        v = row.get(key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return default
        return fv if np.isfinite(fv) else default

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
            "momentum": _num(row, "Momentum", 0.0) * 100,  # → %
            "realized_vol": _num(row, "RealizedVol", 0.0) * 100,
            "avgz": _num(row, "AvgZ", 0.0),
            "net_breadth": _num(row, "NetBreadth", 0.0),
            "hurst": _num(row, "Hurst", 0.5),
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


def analog_prediction_series(
    ts: pd.DataFrame,
    target_col: str,
    hold_horizon: int,
    *,
    mom_window: int = 20,
    top_n: int = 10,
    step: int | None = None,
) -> pd.DataFrame:
    """Historical analog predictions over time — what the matcher would have
    predicted at each past as-of date, using only information available then.

    At each as-of position ``t`` (strided every ``step`` rows, default =
    ``hold_horizon`` so consecutive evaluations are NON-overlapping):
      • Candidate pool = rows with position ``p <= t - hold_horizon`` — the
        analog's forward outcome window ``[p, p+H]`` has fully COMPLETED by
        ``t``, so the prediction never peeks at an unrealized outcome
        (mirrors research/hero_study.py's convention).
      • Engine warm-up rows (``ValidRow`` False — fabricated NetBreadth) are
        excluded from both the pool and the as-of grid, matching
        ``find_similar_periods``.
      • NaN cleaning uses POOL-ONLY column medians per as-of date — a
        full-sample median would leak future distribution shape into past
        cleaning (the look-ahead class audit finding F14 removed elsewhere).
      • Scoring/selection = the SHIPPED config: pure Mahalanobis (ANALOG_W_*
        1/0/0) under the same Theiler exclusion gap as the live matcher.

    Returns a DataFrame with columns:
      ``Date``      — the as-of date,
      ``Predicted`` — analog-median +``hold_horizon``d forward return (%),
      ``Realized``  — the target's actual +``hold_horizon``d return from that
                      date (%; NaN for the last as-of dates whose window
                      hasn't completed — the live predictions).
    The final row is always the LATEST valid as-of date (appended off-stride
    if needed) so the series ends at the same prediction the Precedent tab's
    live cards show. Empty DataFrame when there is insufficient history.
    """
    feat, feature_cols = _build_feature_frame(ts, mom_window)
    if feat.empty or len(feature_cols) < 2:
        return pd.DataFrame(columns=["Date", "Predicted", "Realized"])

    H = int(hold_horizon)
    step = int(step) if step else H
    n = len(feat)
    F_all = feat[feature_cols].to_numpy(dtype=np.float64)
    price = feat["Price"].to_numpy(dtype=np.float64)
    dates = feat["Date"].to_numpy()
    valid_row = (feat["ValidRow"].to_numpy(dtype=bool)
                 if "ValidRow" in feat.columns else np.ones(n, dtype=bool))
    gap = max(int(mom_window), H, 1)

    start = max(mom_window + 30, H + 30)
    as_of_grid = list(range(start, n, step))
    if as_of_grid and as_of_grid[-1] != n - 1:
        as_of_grid.append(n - 1)   # always include the latest as-of date

    rows: list[dict] = []
    for t in as_of_grid:
        if not valid_row[t]:
            continue                       # as-of state itself is warm-up fabrication
        pool_end = t + 1 - H               # outcomes completed by t (p + H <= t)
        if pool_end < 30:
            continue
        pool_pos = np.flatnonzero(valid_row[:pool_end])
        if len(pool_pos) < 30:
            continue

        Fp = F_all[pool_pos].copy()
        cur = F_all[t].copy()
        # Pool-only median fill (causal cleaning).
        for j in range(Fp.shape[1]):
            col = Fp[:, j]
            ok = np.isfinite(col)
            med = float(np.median(col[ok])) if ok.any() else 0.0
            Fp[~ok, j] = med
            if not np.isfinite(cur[j]):
                cur[j] = med

        cov = np.cov(Fp, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dist = mahalanobis_distance_batch(Fp, cur, cov)
        dmax = dist.max() if dist.max() > 0 else 1.0
        sim = 1.0 - dist / dmax

        accepted = select_analogs_theiler(sim, top_n, gap, positions=pool_pos)
        sel = pool_pos[accepted]
        fwd = [(price[p + H] / price[p] - 1) * 100.0
               for p in sel if price[p] > 0]          # p + H <= t < n always
        if not fwd:
            continue

        realized = ((price[t + H] / price[t] - 1) * 100.0
                    if (t + H < n and price[t] > 0) else np.nan)
        rows.append({
            "Date": pd.Timestamp(dates[t]) if not isinstance(dates[t], (int, np.integer)) else dates[t],
            "Predicted": float(np.median(fwd)),
            "Realized": float(realized) if np.isfinite(realized) else np.nan,
        })

    return pd.DataFrame(rows, columns=["Date", "Predicted", "Realized"])
