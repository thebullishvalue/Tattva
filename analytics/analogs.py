"""
Tattva — Analog (Similar-Period) matcher.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Covariance-aware historical analog matching, ported from Arthagati's
``find_similar_periods`` (Mahalanobis + trajectory cosine + recency). The matcher
itself is unchanged; only its INPUTS are adapted to Tattva:

  • Feature vector is built from the quantities Tattva already computes per day
    (``engine.ts_data``) — robust-quantile extension (AvgZ), net internal breadth,
    target momentum, realized volatility, breadth and the valuation
    oscillator — instead of
    Arthagati's mood features.
  • Forward-return horizons are the FIXED precedent term structure
    (core.config.PRECEDENT_HORIZONS = 1/3/5/10/20/60d) — a complete span the
    Precedent tab shows end-to-end, independent of the sidebar lens. (Callers
    pass the horizon set explicitly; the default below mirrors that constant.)

It answers an empirical, non-parametric question that complements the model
forecast: "when the target's state looked statistically like today, what did the
target do next?" — a base rate, descriptive not predictive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ── Blend weights (ported verbatim from Arthagati) ───────────────────────────
# Blend re-tuned for Tattva (2026-06-20, research/analog_tuning_study.py + research/analog_confirm.py:
# 13 targets, non-overlapping OOS IC full + recent-half). The ported Arthagati blend
# (.55/.35/.10) was actively HURTING: trajectory adds ~nothing and recency degrades
# the recent regime. PURE Mahalanobis state-matching is the clear winner — it
# recovers the decayed recent edge (10d recent IC −0.010 → +0.079; 20d −0.083 →
# +0.095) while holding full-sample IC. So trajectory + recency are dropped (weight
# 0 → their computation is skipped entirely, also a live speedup).


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


# (``cosine_similarity`` lived here — it served only the trajectory-cosine
# scoring term, which shipped at weight 0.0 and is gone with it.)


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

# (``_rolling_hurst`` lived here. It fed a "Hurst" matching feature that was
# effectively constant on a price series — see _build_feature_frame — and had no
# other caller. The DFA estimator itself remains in analytics.hurst, where its
# short-memory bias is documented.)


def _build_feature_frame(ts: pd.DataFrame, mom_window: int) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the per-day analog state matrix from Tattva's engine.ts_data.

    "Similar" means similar in the state the SYSTEM measures, so the feature set
    tracks what the engines publish (availability-guarded throughout):
      • Momentum     — trailing ``mom_window``-day log-return of the target Price
      • Realized Vol — rolling σ of daily log-returns over ``mom_window``
      • NetBreadth   — OversoldBreadth − OverboughtBreadth (Swayam agreement)
      • FVO          — the valuation oscillator: how rich or cheap the target is
                       versus the level the global cross-section implies, in units
                       of the engine's own predictive SD
      • Stress       — where global realised volatility sits in its own history
                       [0,1]. The same mispricing in a crisis and in a calm tape
                       are not the same state, and nothing else here carries that
      • Confidence   — the engine's own gate on whether the valuation is worth
                       acting on (mean-reversion evidence x cross-sectional
                       agreement). Distinguishes a -2 sigma gap the cross-section
                       agrees about from one it is split on

    FVO is the addition that matters. The matcher previously read only price
    dynamics and breadth, so it would happily call two dates analogous while the
    asset was two standard deviations rich on one and two cheap on the other —
    which is the single most decision-relevant difference the system computes.
    It is a standardised quantity by construction (a z-score against the
    engine's own uncertainty), so it needs no rescaling to sit alongside the
    others under a Mahalanobis metric.

    Deliberately NOT matched on, each for a measured reason (Gold, 1354 valued
    rows) — because a Mahalanobis metric pays for every column in covariance
    estimation noise, so a feature has to earn its dimension:

      • GapPercentile — r = +0.94 with FVO. The same information twice, which
        does not add a dimension so much as double-weight an existing one.
      • MRProb        — sd 0.006 around a mean of 1.00. After thousands of
        sessions the gap's stationarity is a settled question, so the column is
        a constant; a constant contributes nothing to a distance except a
        near-singular covariance direction.
      • ExplainedVar / KFactors — cv 0.10 and 13 distinct values. Properties of
        the factor model's internal state, not of the market's.
      • Rolling DFA Hurst — dropped outright. ``analytics.hurst.hurst_dfa``
        saturates near its 0.99 clip for anything close to a random walk, so
        across a price series it was very nearly a constant too.
      • AvgZ — dropped from MATCHING by the tuning study (10d recent IC
        -0.010 → +0.034 without it), but still carried for display.

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
        # The engine's own burn-in (no published valuation yet) reads
        # OversoldBreadth ==
        # OverboughtBreadth == 0, so NetBreadth would otherwise be a fabricated
        # 0.0 "neutral" reading rather than genuinely missing. Force it to NaN
        # there so the analog historical pool excludes the warm-up (median-fill
        # in find_similar_periods then treats it as missing, matching every
        # other NaN-guarded consumer of this engine output).
        if "Valid" in df.columns:
            feat.loc[~df["Valid"].astype(bool), "NetBreadth"] = np.nan
    # The FVO oscillator — the valuation state. Masked to published rows for
    # the same reason NetBreadth is: the engine's burn-in carries no valuation,
    # and a fabricated 0.0 there would read as "fairly valued" rather than as
    # "not yet valued".
    if "FVO" in df.columns:
        _fvo = pd.to_numeric(df["FVO"], errors="coerce")
        if "Valid" in df.columns:
            _fvo = _fvo.where(df["Valid"].astype(bool))
        feat["FVO"] = _fvo.to_numpy(dtype=np.float64)

    # Regime context + valuation trustworthiness. Both are already [0,1] and
    # both are masked to published rows, for the same reason FVO is: the
    # burn-in carries no valuation, and a fabricated 0.0 would read as
    # "calm, and certain about it".
    for _src, _dst in (("Stress", "Stress"), ("Confidence", "Confidence")):
        if _src in df.columns:
            _c = pd.to_numeric(df[_src], errors="coerce")
            if "Valid" in df.columns:
                _c = _c.where(df["Valid"].astype(bool))
            feat[_dst] = _c.to_numpy(dtype=np.float64)

    # ── V2 engine state (MŪLA), availability-guarded ───────────────────
    # Appended only when the columns exist (i.e. the run used the MŪLA
    # valuation core), so a legacy ts_data produces the exact legacy set.
    #   ExpertW   — MŪLA's pooled weight on valuation-containing designs:
    #               distinguishes "2σ rich AND the ECM agrees it pays to fade"
    #               from "2σ rich while momentum owns this tape"
    #   GapSpeed  — the ECM reversion speed κ̂: fast-κ gaps are states whose
    #               analogs resolve quickly; slow-κ ones are not the same
    #               state even at identical FVO
    for _src, _dst in (("WValuation", "ExpertW"), ("MulaKappa", "GapSpeed")):
        if _src in df.columns:
            _c = pd.to_numeric(df[_src], errors="coerce")
            if "Valid" in df.columns:
                _c = _c.where(df["Valid"].astype(bool))
            feat[_dst] = _c.to_numpy(dtype=np.float64)

    feature_cols = [c for c in ("Momentum", "RealizedVol", "NetBreadth", "FVO",
                                "Stress", "Confidence", "ExpertW", "GapSpeed")
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
    #   • ValidRow — the engine's Valid flag (False through the burn-in).
    #     Lets find_similar_periods exclude rows whose NetBreadth
    #     would be median-filled fabrication from the candidate pool (M2).
    if "AvgZ" in df.columns:
        feat["AvgZ"] = pd.to_numeric(df["AvgZ"], errors="coerce").to_numpy(dtype=np.float64)
    if "Valid" in df.columns:
        feat["ValidRow"] = df["Valid"].astype(bool).to_numpy()
    return feat, feature_cols


#: Effective sample size below which the base rate is not reported as a rate.
#: Kish ESS, not a raw count — ten analogs of which one carries most of the
#: weight is not ten observations, and reporting "70% positive, n=10" for it
#: would be the single most misleading thing this module could say.
MIN_EFFECTIVE_N = 4.0

#: Kernel bandwidth as a fraction of the pool's own median distance. Weights
#: fall to ~0.6 at the median distance and ~0.1 at twice it, so "similar"
#: is defined relative to how similar things ever get for this instrument
#: rather than by an absolute Mahalanobis number that means different things
#: on different targets.
#:
#: Half the median rather than the whole of it: at h = median, the analogs that
#: survive Theiler selection all sit far inside one bandwidth and receive
#: near-identical weights, which defeats the point — the kernel has to
#: discriminate ACROSS the accepted set, not merely between it and the bulk of
#: history it already beat.
KERNEL_BANDWIDTH = 0.5

#: Tail probability at which a state is declared to have no precedent.
#:
#: The test is on the CHI-SQUARE scale, not a ratio to the pool. Under a
#: Mahalanobis metric a point drawn from the same distribution has d^2 ~ chi2
#: with df = number of features, so "is the nearest analog implausibly far?"
#: has a calibrated answer that does not move with today's own position.
#:
#: That last property is the whole reason for it. The obvious test — nearest
#: distance versus the median distance from today to history — cannot work:
#: when today is extreme, EVERY distance inflates, the median inflates with
#: them, and the ratio stays small however unprecedented the state is. It was
#: implemented that way first and silently passed a synthetic state sitting 19
#: Mahalanobis units from anything in the record.
UNPRECEDENTED_P = 0.999


def find_similar_periods(
    ts: pd.DataFrame,
    target_col: str,
    hold_horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 60),
    *,
    mom_window: int = 20,
    top_n: int = 40,
    bandwidth: float = KERNEL_BANDWIDTH,
) -> list[dict]:
    """Find historical states resembling today's, weighted by how close they are.

    Every returned analog carries a ``weight`` in (0, 1] from a Gaussian kernel
    on its Mahalanobis distance. This replaces a hard top-N cut, and the
    difference is not cosmetic: under a top-N the tenth-nearest analog counted
    exactly as much as the nearest, so a state with two genuine precedents and
    eight loose ones reported a ten-observation base rate that was mostly noise
    from the eight. The kernel lets the data say how many precedents there
    really are, and :func:`summarize_forward` reports that as an effective
    sample size.

    ``top_n`` bounds how many distinct episodes are returned, and it defaults
    high (40) on purpose: the base rate should be computed over as many genuine
    precedents as the kernel finds material, and the DISPLAY should then show
    the closest handful of those. Truncating at ten before summarising would
    make the effective sample size meaningless — the ten nearest analogs are
    near-equally weighted by construction, so n_eff would read ~10 no matter
    how thin the real evidence was. The caller slices for the cards.

    Candidates are drawn from genuinely-valued history only (burn-in rows
    excluded), exclude the tail whose forward window has not closed, and are
    selected under a Theiler window so each is a distinct episode rather than
    ten adjacent days of one.

    What was dropped: the trajectory-cosine and recency-decay scoring terms.
    Both had shipped at weight 0.0 since the analog re-tune — ~50 lines of
    arithmetic multiplied by zero on every call — and a blend weight that is
    always zero is not a tuning knob, it is a deleted feature that still costs
    a code path.
    """
    feat, feature_cols = _build_feature_frame(ts, mom_window)
    if feat.empty or len(feature_cols) < 2:
        return []

    n = len(feat)
    purge = int(max(hold_horizons)) if hold_horizons else 20
    # Exclude the tail: those rows have no realized forward path yet.
    historical = feat.iloc[:n - purge].copy()
    # Exclude the engine's burn-in rows from the candidate pool (ValidRow
    # False): their breadth and valuation columns are genuinely missing, and
    # matching against a median-filled fabrication is not a state match. The
    # frame keeps its original RangeIndex labels, so `historical.index` remains
    # the TEMPORAL row position — the forward-return lookups and the Theiler
    # gap below both rely on that.
    if "ValidRow" in historical.columns:
        historical = historical[historical["ValidRow"]]
    if len(historical) < 30:
        return []

    latest = feat.iloc[-1]
    current_vec = latest[feature_cols].to_numpy(dtype=np.float64)
    # .copy() is load-bearing: on Streamlit Cloud DataFrame.to_numpy() can return
    # a READ-ONLY view (consolidated/cached buffers), and the median-fill below
    # writes in place → "assignment destination is read-only" without it.
    hist_matrix = historical[feature_cols].to_numpy(dtype=np.float64).copy()
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)

    # ── Covariance-aware distance ───────────────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    dist = np.where(np.isfinite(dist), dist, np.inf)

    # ── Kernel weights, scaled to the pool's own distance distribution ──
    finite = dist[np.isfinite(dist)]
    if not len(finite):
        return []
    med_d = float(np.median(finite))
    h = max(bandwidth * med_d, 1e-9)
    weights = np.exp(-0.5 * (dist / h) ** 2)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    # Is even the closest historical state implausibly far? Calibrated against
    # chi2(df = n features), so the verdict is about today's position in the
    # historical DISTRIBUTION rather than relative to its own distance spread.
    nearest = float(np.min(dist))
    _crit = float(chi2.ppf(UNPRECEDENTED_P, df=max(1, len(feature_cols))))
    unprecedented = bool(nearest ** 2 > _crit)

    historical = historical.copy()
    historical["similarity"] = weights
    historical["distance"] = dist

    # ── Theiler exclusion: distinct EPISODES, not adjacent days ─────────
    gap = max(int(mom_window), int(max(hold_horizons)) if hold_horizons else 20)
    positions = historical.index.to_numpy()
    accepted = select_analogs_theiler(weights, top_n, gap, positions=positions)

    price_all = feat["Price"].to_numpy(dtype=np.float64)
    results: list[dict] = []
    for j in accepted:
        row = historical.iloc[j] if not isinstance(j, (np.integer, int)) else historical.iloc[int(j)]
        pos = int(historical.index[int(j)])
        fwd: dict[int, float | None] = {}
        for hh in hold_horizons:
            hh = int(hh)
            tgt = pos + hh
            if tgt < len(price_all) and price_all[pos] > 0 and np.isfinite(price_all[tgt]):
                fwd[hh] = float((price_all[tgt] / price_all[pos] - 1.0) * 100.0)
            else:
                fwd[hh] = None

        def _num(r, key, default=0.0):
            v = r.get(key)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return v if np.isfinite(v) else default

        d = row["Date"]
        price_at = float(row["Price"]) if row["Price"] and row["Price"] > 0 else None
        results.append({
            "date": (pd.Timestamp(d).strftime("%Y-%m-%d")
                     if not isinstance(d, (int, np.integer)) else str(d)),
            "similarity": float(row["similarity"]),
            "distance": float(row["distance"]),
            "price": price_at or 0.0,
            "momentum": _num(row, "Momentum", 0.0) * 100,
            "realized_vol": _num(row, "RealizedVol", 0.0) * 100,
            "avgz": _num(row, "AvgZ", 0.0),
            "net_breadth": _num(row, "NetBreadth", 0.0),
            "fvo": _num(row, "FVO", 0.0),
            "stress": _num(row, "Stress", 0.5),
            "confidence": _num(row, "Confidence", 0.0),
            "fwd": {int(hh): fwd[int(hh)] for hh in hold_horizons},
            # Pool-level context, identical on every returned analog — carried
            # here so summarize_forward needs no second pass over the pool.
            "_pool_median_distance": med_d,
            "_unprecedented": unprecedented,
            "_nearest_distance": nearest,
        })
    return results


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile — the median/IQR of a kernel-weighted analog set."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    cutoff = q * cw[-1]
    return float(v[np.searchsorted(cw, cutoff)])


def summarize_forward(periods: list[dict], hold_horizons: tuple[int, ...]) -> dict[int, dict]:
    """Kernel-weighted forward-return base rate, with its own uncertainty.

    Per horizon: ``{median, positive_pct, n, n_eff, p25, p75, usable, note}``.

      • ``median`` / ``positive_pct`` are WEIGHTED by analog closeness, so a
        near-exact precedent counts for more than a loose one.
      • ``n_eff`` is the Kish effective sample size of those weights. It is the
        number to read, not ``n``: they are equal only when every analog is
        equally close, and they diverge exactly when the base rate is being
        carried by one or two episodes.
      • ``p25`` / ``p75`` bound the outcomes. A base rate whose analogs
        disagree violently is not the same evidence as one whose analogs
        agree, and a bare median cannot tell them apart.
      • ``usable`` is False when the effective sample is too thin or the
        current state has no genuine precedent, with ``note`` saying which.
        "There is no precedent for this" is a real finding; manufacturing a
        percentage from the ten least-dissimilar days in the record is not.
    """
    out: dict[int, dict] = {}
    if not periods:
        return out
    unprecedented = bool(periods[0].get("_unprecedented", False))

    for h in hold_horizons:
        h = int(h)
        pairs = [(p["fwd"].get(h), float(p.get("similarity", 0.0))) for p in periods]
        pairs = [(v, w) for v, w in pairs if v is not None and np.isfinite(v) and w > 0]
        if not pairs:
            continue
        vals = np.array([v for v, _ in pairs], dtype=np.float64)
        wts = np.array([w for _, w in pairs], dtype=np.float64)
        wsum = float(wts.sum())
        n_eff = float(wsum ** 2 / max(float((wts ** 2).sum()), 1e-12))

        pos_pct = float((wts[vals > 0].sum() / wsum) * 100.0) if wsum > 0 else 0.0
        usable = (n_eff >= MIN_EFFECTIVE_N) and not unprecedented
        note = ""
        if unprecedented:
            note = "No close precedent — the current state is unlike anything in the record."
        elif n_eff < MIN_EFFECTIVE_N:
            note = (f"Effective sample {n_eff:.1f} (of {len(vals)} analogs) — "
                    "too concentrated on one episode to read as a base rate.")

        out[h] = {
            "median": _weighted_quantile(vals, wts, 0.50),
            "p25": _weighted_quantile(vals, wts, 0.25),
            "p75": _weighted_quantile(vals, wts, 0.75),
            "positive_pct": pos_pct,
            "n": len(vals),
            "n_eff": n_eff,
            "usable": usable,
            "note": note,
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


def analog_skill_by_horizon(
    ts: pd.DataFrame,
    target_col: str,
    hold_horizons: tuple[int, ...],
    *,
    mom_window: int = 20,
    top_n: int = 10,
    min_windows: int = 10,
) -> dict[int, dict]:
    """Walk-forward analog SKILL at every hold horizon — the term structure the
    Precedent tab plots so the read isn't pinned to a single lens horizon.

    For each ``H`` in ``hold_horizons`` runs :func:`analog_prediction_series`
    (the same causal, non-overlapping, pool-only-cleaned walk-forward the
    single-horizon plot uses) and scores its COMPLETED windows:

      ``ic``   — Spearman rank IC of analog-predicted vs realized +H returns,
      ``hit``  — directional hit-rate (% of windows where sign matched), in [0,100],
      ``n``    — number of completed (realized) non-overlapping windows,
      ``pval`` — two-sided Spearman p-value (``nan`` when ``n < min_windows``),
      ``df``   — the full predicted/realized frame (for the detail timeseries).

    ``ic``/``hit``/``pval`` are ``nan`` until at least ``min_windows`` completed
    windows exist, so a horizon with too little realized history reads as "no
    estimate" rather than a noise number. Keyed by horizon; horizons that yield
    an empty walk-forward are omitted.
    """
    from scipy.stats import spearmanr as _spearmanr

    out: dict[int, dict] = {}
    for h in hold_horizons:
        H = int(h)
        df = analog_prediction_series(
            ts, target_col, H, mom_window=mom_window, top_n=top_n,
        )
        if df.empty:
            continue
        pred = df["Predicted"].to_numpy(dtype=np.float64)
        real = df["Realized"].to_numpy(dtype=np.float64)
        done = np.isfinite(real) & np.isfinite(pred)
        n_done = int(done.sum())
        ic = hit = pval = float("nan")
        base = edge = float("nan")
        if n_done >= int(min_windows):
            _ic, _pv = _spearmanr(pred[done], real[done])
            ic = float(_ic) if np.isfinite(_ic) else float("nan")
            pval = float(_pv) if np.isfinite(_pv) else float("nan")
            hit = float(np.mean(np.sign(pred[done]) == np.sign(real[done])) * 100.0)

            # THE HIT RATE'S OWN NULL, reported beside it.
            #
            # `hit` alone is unreadable, and reads FLATTERINGLY wrong. A 50%
            # directional hit rate looks like a coin flip — neutral, no harm
            # done — but the benchmark is not 50%. It is the unconditional
            # majority direction of the SAME realized windows, which on a
            # trending asset is well above half: measured 2026-08-17 the base
            # rate was 57.8% (Gold) and 53.5-57.0% (Copper), while the matcher
            # scored 43.0-51.8%. Every horizon tested was WORSE than always
            # predicting the majority direction, and nothing on screen said so.
            #
            # This is the same failure the convergence agreement tooltip had —
            # a number judged against an assumed 50% null that was never its
            # null — so the null now travels with the number rather than being
            # something a reader is expected to know.
            #
            # `edge` is the honest headline: hit minus what a constant
            # prediction would have scored. Negative means the matching added
            # nothing over ignoring the state entirely.
            up = float(np.mean(real[done] > 0))
            base = float(max(up, 1.0 - up) * 100.0)
            edge = float(hit - base)
        out[H] = {"ic": ic, "hit": hit, "base": base, "edge": edge,
                  "n": n_done, "pval": pval, "df": df}
    return out
