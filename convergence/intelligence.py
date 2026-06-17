"""
Tattva v2.0.0 — Intelligence Mode: self-calibrating convergence profile.
तत्त्व (Tattva) — "Principle / Essence"

A Bayesian (Optuna TPE) optimizer that learns per-universe optimal values
for two structural decisions in the Convergence layer:

  1. **Dimension weights** for the agreement ratio in
     ``convergence/cross_validator.py``: ``(w_direction, w_breadth,
     w_magnitude, w_regime)``. The defaults are a hand-set prior
     (0.30 / 0.25 / 0.25 / 0.20). Intelligence mode replaces the
     adaptive ±10% heuristic with evidence learned from data.

  2. **Asymmetric classification thresholds** in
     ``convergence/normalization.py``: the four cut-points that map a
     normalized convergence value (in [-1, +1]) to STRONG BUY / BUY /
     HOLD / SELL / STRONG SELL. The static defaults are symmetric
     (±0.3 moderate, ±0.5 strong). Real return distributions are
     skewed — optimal buy/sell thresholds usually aren't symmetric.

Objective: maximize the mean Spearman Information Coefficient of the
calibrated signal vs the target commodity's forward returns across hold
horizons ``[3, 5, 10, 20]`` days, evaluated on a PURGED K-FOLD CV
objective (robust across time, not a single slice) with a held-out tail
and L2 regularization on weights. A separate expanding-window
``walk_forward_ic`` provides genuinely out-of-sample durability grades.

Persistence: one profile per universe key. Loaded automatically on
universe switch, saved to ``~/.cache/tattva/intelligence/profiles.json``.

Mechanics: Bayesian TPE search over per-dimension agreement-score weights
and asymmetric thresholds, keyed per commodity universe, persisted to disk.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import optuna
    _OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # graceful degradation when optuna isn't installed
    optuna = None  # type: ignore
    _OPTUNA_AVAILABLE = False

from core.config import (
    CONV_WEIGHT_DIRECTION,
    CONV_WEIGHT_BREADTH,
    CONV_WEIGHT_MAGNITUDE,
    CONV_WEIGHT_REGIME,
)

log = logging.getLogger(__name__)

PROFILE_VERSION = "v1-tattva-convergence"

# Disk location: ~/.cache/tattva/intelligence/profiles.json
_PROFILE_DIR = Path.home() / ".cache" / "tattva" / "intelligence"
_PROFILE_FILE = _PROFILE_DIR / "profiles.json"

# ── Factory defaults ─────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "w_direction": CONV_WEIGHT_DIRECTION,
    "w_breadth":   CONV_WEIGHT_BREADTH,
    "w_magnitude": CONV_WEIGHT_MAGNITUDE,
    "w_regime":    CONV_WEIGHT_REGIME,
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Symmetric defaults from convergence/normalization.py
    "buy_strong":     -0.5,   # below this → STRONG BUY
    "buy_moderate":   -0.3,   # below this (but above buy_strong) → BUY
    "sell_moderate":  +0.3,   # above this (but below sell_strong) → SELL
    "sell_strong":    +0.5,   # above this → STRONG SELL
}

# Forward-return horizons for IC scoring (trading days)
HOLD_HORIZONS = (3, 5, 10, 20)


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════


@dataclass
class IntelligenceProfile:
    """A learned (weights, thresholds) bundle plus calibration metadata.

    Persisted to disk per universe. The metadata captures everything needed
    to surface a Sanket-style "Model Passport" in the UI: which universe
    the profile was fit on, train/val IC scores, when it was last updated,
    and what version of the calibration pipeline produced it.
    """

    weights:    dict[str, float]
    thresholds: dict[str, float]
    universe:   str
    selected_index: str | None = None
    train_score:    float = 0.0
    val_score:      float = 0.0
    train_ic:       float = 0.0
    val_ic:         float = 0.0
    n_train_dates:  int = 0
    n_val_dates:    int = 0
    n_trials:       int = 0
    sensitivity:    dict[str, float] = field(default_factory=dict)
    timestamp:      str = ""
    version:        str = PROFILE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntelligenceProfile":
        # Tolerate older v1 dicts that may not carry every field.
        safe = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        safe.setdefault("weights", DEFAULT_WEIGHTS.copy())
        safe.setdefault("thresholds", DEFAULT_THRESHOLDS.copy())
        safe.setdefault("universe", "—")
        return cls(**safe)


def default_profile(universe: str = "default", index: str | None = None) -> IntelligenceProfile:
    """A neutral profile holding the factory weights/thresholds."""
    return IntelligenceProfile(
        weights=DEFAULT_WEIGHTS.copy(),
        thresholds=DEFAULT_THRESHOLDS.copy(),
        universe=universe,
        selected_index=index,
    )


# ════════════════════════════════════════════════════════════════════════
# Persistence layer
# ════════════════════════════════════════════════════════════════════════


def _ensure_dir() -> None:
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _profile_key(universe: str, index: str | None) -> str:
    """One profile per (universe, index) — matches Sanket's keying convention."""
    return f"{universe or '—'} · {index or '—'}"


def _read_all() -> dict[str, dict]:
    _ensure_dir()
    if not _PROFILE_FILE.exists():
        return {}
    try:
        with open(_PROFILE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Profile file read failed: %s", e)
        return {}


def _write_all(profiles: dict[str, dict]) -> None:
    _ensure_dir()
    tmp = _PROFILE_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(profiles, f, indent=2, default=str)
        tmp.replace(_PROFILE_FILE)
    except Exception as e:
        log.error("Profile file write failed: %s", e)
        if tmp.exists():
            tmp.unlink()


def save_profile(profile: IntelligenceProfile) -> None:
    """Persist a calibrated profile under its (universe, index) key."""
    profiles = _read_all()
    key = _profile_key(profile.universe, profile.selected_index)
    profiles[key] = profile.to_dict()
    _write_all(profiles)


def load_profile_for(universe: str, index: str | None) -> IntelligenceProfile | None:
    """Load the profile fit for this (universe, index), or None."""
    profiles = _read_all()
    key = _profile_key(universe, index)
    raw = profiles.get(key)
    if not raw:
        return None
    try:
        return IntelligenceProfile.from_dict(raw)
    except Exception as e:
        log.warning("Profile decode failed for %s: %s", key, e)
        return None


def list_profiles() -> list[IntelligenceProfile]:
    """All persisted profiles, newest first."""
    profiles = _read_all()
    out: list[IntelligenceProfile] = []
    for raw in profiles.values():
        try:
            out.append(IntelligenceProfile.from_dict(raw))
        except Exception:
            continue
    out.sort(key=lambda p: p.timestamp, reverse=True)
    return out


def delete_profile(universe: str, index: str | None = None) -> bool:
    """Remove the profile for this (universe, index). Returns True if deleted."""
    profiles = _read_all()
    key = _profile_key(universe, index)
    if key not in profiles:
        return False
    del profiles[key]
    _write_all(profiles)
    return True


def delete_all_profiles() -> int:
    """Wipe every profile. Returns count removed."""
    profiles = _read_all()
    n = len(profiles)
    _write_all({})
    return n


# ════════════════════════════════════════════════════════════════════════
# Calibration data assembly
# ════════════════════════════════════════════════════════════════════════


def _build_calibration_frame(
    convergence_df: pd.DataFrame,
    aarambh_ts: pd.DataFrame,
    target_col: str = "Actual",
    horizons: tuple[int, ...] = HOLD_HORIZONS,
) -> pd.DataFrame:
    """Assemble the per-date calibration matrix.

    Columns:
      - dim_direction, dim_breadth, dim_magnitude, dim_regime: per-day
        agreement sub-scores (already in [0, 1])
      - convergence_score: legacy signed composite (used for sign anchor)
      - Ret_{h}b for h in horizons: forward log-returns of the target
        column (typically NIFTY50_PE from the Aarambh sheet, used as a
        return proxy — PE expansion/compression correlates with price
        moves since earnings are slow-moving)

    Returns an empty frame if any input is missing.
    """
    if convergence_df is None or convergence_df.empty:
        return pd.DataFrame()
    needed = {"dim_direction", "dim_breadth", "dim_magnitude", "dim_regime", "convergence_score"}
    if not needed.issubset(convergence_df.columns):
        return pd.DataFrame()
    if aarambh_ts is None or target_col not in aarambh_ts.columns:
        return pd.DataFrame()

    conv = convergence_df.copy()
    conv.index = pd.to_datetime(conv.index, errors="coerce")
    conv = conv[~conv.index.isna()].sort_index()

    a_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")].copy()
    if "Date" in a_dedup.columns:
        a_dedup["Date"] = pd.to_datetime(a_dedup["Date"], errors="coerce", dayfirst=True)
        a_dedup = a_dedup.dropna(subset=["Date"]).set_index("Date")
    else:
        a_dedup.index = pd.to_datetime(a_dedup.index, errors="coerce")
        a_dedup = a_dedup[~a_dedup.index.isna()]
    a_dedup = a_dedup.sort_index()

    target = a_dedup[target_col].astype(float).reindex(conv.index, method="ffill")

    # Forward log-returns at each horizon
    log_target = np.log(target.replace(0, np.nan)).ffill()
    out = conv.copy()
    for h in horizons:
        out[f"Ret_{h}b"] = log_target.shift(-h) - log_target

    # Drop dates with NaN at any horizon (tail of the series)
    ret_cols = [f"Ret_{h}b" for h in horizons]
    out = out.dropna(subset=ret_cols + list(needed))
    return out


# ════════════════════════════════════════════════════════════════════════
# Fast scoring kernel
# ════════════════════════════════════════════════════════════════════════


def _normalize_weights(w: dict[str, float]) -> np.ndarray:
    """Renormalize a 4-tuple of weights to sum to 1.0 (Dirichlet-style)."""
    arr = np.array([w["w_direction"], w["w_breadth"], w["w_magnitude"], w["w_regime"]], dtype=np.float64)
    s = float(arr.sum())
    if s <= 1e-12:
        return np.full(4, 0.25, dtype=np.float64)
    return arr / s


def _composite_signal(frame: pd.DataFrame, w: dict[str, float]) -> np.ndarray:
    """Recompute the signed, DIRECTIONAL convergence composite from per-dim scores.

    Mirrors cross_validator.py:
        agreement  = Σ wₖ · (2·score_k − 1)            ∈ [−1, +1]  (agreement strength)
        directional = −consensus_direction · (agreement + 1) / 2   (negative = bullish)
    The per-dim scores are AGREEMENT scores (not bull/bear), so the consensus
    direction is what orients the signal. Falls back to the legacy agreement-only
    composite if a frame predates the ``consensus_direction`` column.
    """
    weights = _normalize_weights(w)
    M = np.column_stack([
        frame["dim_direction"].to_numpy(dtype=np.float64),
        frame["dim_breadth"].to_numpy(dtype=np.float64),
        frame["dim_magnitude"].to_numpy(dtype=np.float64),
        frame["dim_regime"].to_numpy(dtype=np.float64),
    ])
    agreement = (M @ weights) * 2.0 - weights.sum()  # = M @ w * 2 - 1 ∈ [-1, 1]
    if "consensus_direction" in frame.columns:
        d = frame["consensus_direction"].to_numpy(dtype=np.float64)
        return -d * (agreement + 1.0) / 2.0          # directional, negative = bullish
    return agreement                                 # legacy agreement-only fallback


def _classify(values: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Apply thresholds to map signal → {-2, -1, 0, +1, +2}.

    -2 STRONG BUY, -1 BUY, 0 HOLD, +1 SELL, +2 STRONG SELL.
    Sign convention: in the existing system, *negative* composite =
    bullish convergence (the metric cards use this convention).
    """
    bs = thresholds["buy_strong"]
    bm = thresholds["buy_moderate"]
    sm = thresholds["sell_moderate"]
    ss = thresholds["sell_strong"]
    out = np.zeros_like(values, dtype=np.int8)
    out[values <= bs] = -2
    out[(values > bs) & (values <= bm)] = -1
    out[(values > bm) & (values < sm)] = 0
    out[(values >= sm) & (values < ss)] = +1
    out[values >= ss] = +2
    return out


from scipy.stats import rankdata  # module-level: avoid a re-import on every call


def _centered_rank(a: np.ndarray) -> np.ndarray:
    """Average-method ranks, mean-centered (matches scipy rankdata default)."""
    r = rankdata(a).astype(np.float64)
    r -= r.mean()
    return r


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN on degenerate input."""
    if len(x) < 5 or len(x) != len(y):
        return float("nan")
    rx = _centered_rank(x)
    ry = _centered_rank(y)
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


# Bin-monotonicity reference ranks (constant): index order [2,1,0,-1,-2].
_BIN_IDX = np.array([2, 1, 0, -1, -2], dtype=np.float64)
_BIN_IDX_R = _centered_rank(_BIN_IDX)
_BIN_IDX_SS = float((_BIN_IDX_R * _BIN_IDX_R).sum())


class _PreparedFrame:
    """Pre-extracted, partially pre-ranked view of a scoring frame.

    Everything that does NOT depend on (weights, thresholds) is computed once
    here: the dim matrix, consensus direction, and each horizon's forward
    returns together with their mean-centered ranks. Across an Optuna study the
    return ranks are reused for every trial instead of being recomputed — the
    single biggest cost in the calibration objective. Scores are bit-identical
    to ``_score_frame`` (same rankdata, same formulas)."""

    __slots__ = ("n", "M", "cons", "ret_arrays", "ret_ranks", "ret_ss")

    def __init__(self, frame: pd.DataFrame, horizons: tuple[int, ...]) -> None:
        self.n = len(frame)
        self.M = np.column_stack([
            frame["dim_direction"].to_numpy(dtype=np.float64),
            frame["dim_breadth"].to_numpy(dtype=np.float64),
            frame["dim_magnitude"].to_numpy(dtype=np.float64),
            frame["dim_regime"].to_numpy(dtype=np.float64),
        ]) if self.n else np.empty((0, 4))
        self.cons = (frame["consensus_direction"].to_numpy(dtype=np.float64)
                     if "consensus_direction" in frame.columns else None)
        self.ret_arrays: dict[int, np.ndarray] = {}
        self.ret_ranks: dict[int, np.ndarray] = {}
        self.ret_ss: dict[int, float] = {}
        for h in horizons:
            col = f"Ret_{h}b"
            if col not in frame.columns:
                continue
            r = frame[col].to_numpy(dtype=np.float64)
            self.ret_arrays[h] = r
            if self.n >= 5:
                rr = _centered_rank(r)
                self.ret_ranks[h] = rr
                self.ret_ss[h] = float((rr * rr).sum())

    def composite(self, w: dict[str, float]) -> np.ndarray:
        weights = _normalize_weights(w)
        agreement = (self.M @ weights) * 2.0 - weights.sum()
        if self.cons is not None:
            return -self.cons * (agreement + 1.0) / 2.0
        return agreement


def _score_prepared(
    prep: "_PreparedFrame",
    weights: dict[str, float],
    thresholds: dict[str, float],
    horizons: tuple[int, ...],
) -> tuple[float, float]:
    """Exact equivalent of ``_score_frame`` operating on a ``_PreparedFrame``."""
    if prep.n == 0:
        return float("nan"), -1e6
    composite = prep.composite(weights)
    bins = _classify(composite, thresholds)

    horizon_ics: list[float] = []
    rx = None
    ss_rx = 0.0
    if prep.n >= 5:
        rx = _centered_rank(composite)
        ss_rx = float((rx * rx).sum())
    for h in horizons:
        if h not in prep.ret_ranks or rx is None:
            continue
        ry = prep.ret_ranks[h]
        denom = np.sqrt(ss_rx * prep.ret_ss[h])
        if denom < 1e-12:
            continue
        ic = float((rx * ry).sum() / denom)
        horizon_ics.append(-ic)
    if not horizon_ics:
        return float("nan"), -1e6
    mean_ic = float(np.mean(horizon_ics))

    bin_quality = 0.0
    n_used = 0
    for h in horizons:
        if h not in prep.ret_arrays:
            continue
        rets = prep.ret_arrays[h]
        means = []
        for b in (-2, -1, 0, +1, +2):
            mask = bins == b
            if mask.sum() >= 3:
                means.append(rets[mask].mean())
            else:
                means.append(np.nan)
        means_arr = np.array(means)
        if np.isnan(means_arr).any():
            continue
        ry = _centered_rank(means_arr)
        denom = np.sqrt(_BIN_IDX_SS * float((ry * ry).sum()))
        if denom < 1e-12:
            continue
        bin_quality += float((_BIN_IDX_R * ry).sum() / denom)
        n_used += 1
    if n_used:
        bin_quality /= n_used

    score = mean_ic + 0.25 * bin_quality
    return mean_ic, score


def _score_frame(
    frame: pd.DataFrame,
    weights: dict[str, float],
    thresholds: dict[str, float],
    horizons: tuple[int, ...],
) -> tuple[float, float]:
    """Score a frame at one (weights, thresholds) point.

    Returns (mean_IC, weighted_score).

    The composite signal direction is what matters for IC. In Nishkarsh
    the *negative* composite corresponds to bullish convergence (more
    expected return). So a *useful* signal has IC(composite, forward
    return) < 0 — we flip the sign so the objective reads naturally
    as "higher is better."

    The classification thresholds don't affect IC directly (IC is on
    the continuous signal). They influence the *bin-mean-return*
    profile, which we mix in as a secondary objective rewarding
    well-separated buckets.
    """
    if frame.empty:
        return float("nan"), -1e6

    composite = _composite_signal(frame, weights)        # signed, ~[-1, +1]
    bins = _classify(composite, thresholds)              # {-2, -1, 0, +1, +2}

    # Primary: mean Spearman IC across horizons, sign-flipped so bullish
    # composite (negative) predicts positive forward returns → positive IC.
    horizon_ics: list[float] = []
    for h in horizons:
        col = f"Ret_{h}b"
        if col not in frame.columns:
            continue
        rets = frame[col].to_numpy(dtype=np.float64)
        ic = _spearman_ic(composite, rets)
        if not np.isnan(ic):
            horizon_ics.append(-ic)  # negative composite = bullish
    if not horizon_ics:
        return float("nan"), -1e6
    mean_ic = float(np.mean(horizon_ics))

    # Secondary: bin separation — is the mean return monotonic in the bin?
    bin_quality = 0.0
    n_used = 0
    for h in horizons:
        col = f"Ret_{h}b"
        if col not in frame.columns:
            continue
        rets = frame[col].to_numpy(dtype=np.float64)
        means = []
        for b in (-2, -1, 0, +1, +2):
            mask = bins == b
            if mask.sum() >= 3:
                means.append(rets[mask].mean())
            else:
                means.append(np.nan)
        means_arr = np.array(means)
        if np.isnan(means_arr).any():
            continue
        # Monotonicity reward: bullish bins should have higher returns
        # (we negate index order so STRONG BUY (-2) → highest expected ret).
        idx = np.array([2, 1, 0, -1, -2], dtype=np.float64)
        bq_ic = _spearman_ic(idx, means_arr)
        if not np.isnan(bq_ic):
            bin_quality += bq_ic
            n_used += 1
    if n_used:
        bin_quality /= n_used

    # Combined: IC (primary) + 0.25 × bin monotonicity (secondary)
    score = mean_ic + 0.25 * bin_quality
    return mean_ic, score


# ════════════════════════════════════════════════════════════════════════
# Rolling walk-forward validation
# ════════════════════════════════════════════════════════════════════════


def _optimize_frame(
    train: pd.DataFrame, horizons: tuple[int, ...], n_trials: int,
    l2_alpha: float, n_cv_folds: int, seed: int = 42,
) -> tuple[dict, dict]:
    """Mini TPE search on one train slice (k-fold CV objective). Returns (w, t).

    Standalone twin of ConvergenceTuner's objective, used by walk_forward_ic so
    each expanding window is re-calibrated independently (no leakage from the
    full-history calibration).
    """
    if not _OPTUNA_AVAILABLE or len(train) < 60:
        return DEFAULT_WEIGHTS.copy(), DEFAULT_THRESHOLDS.copy()

    # Pre-rank the fold returns ONCE (constant across all trials).
    n = len(train)
    if n < n_cv_folds * 10:
        _folds = [_PreparedFrame(train, horizons)]
        _single = True
    else:
        _edges = np.linspace(0, n, n_cv_folds + 1).astype(int)
        _folds = [_PreparedFrame(train.iloc[_edges[i]:_edges[i + 1]], horizons)
                  for i in range(n_cv_folds)]
        _single = False

    def _cv_score(w: dict, t: dict) -> float:
        if _single:
            return _score_prepared(_folds[0], w, t, horizons)[1]
        scores = []
        for fold in _folds:
            _, sc = _score_prepared(fold, w, t, horizons)
            if not np.isnan(sc):
                scores.append(sc)
        return float(np.mean(scores) - 0.5 * np.std(scores)) if scores else -1e6

    def _obj(trial: "optuna.Trial") -> float:
        w = {k: trial.suggest_float(k, 0.05, 1.0)
             for k in ("w_direction", "w_breadth", "w_magnitude", "w_regime")}
        bs = trial.suggest_float("buy_strong", -0.85, -0.30)
        bm = trial.suggest_float("buy_moderate", bs + 0.05, -0.05)
        sm = trial.suggest_float("sell_moderate", 0.05, 0.50)
        ssg = trial.suggest_float("sell_strong", sm + 0.05, 0.85)
        t = {"buy_strong": bs, "buy_moderate": bm, "sell_moderate": sm, "sell_strong": ssg}
        sc = _cv_score(w, t)
        if np.isnan(sc):
            return -1e6
        wa = _normalize_weights(w)
        return float(sc - l2_alpha * float(np.sum((wa - 0.25) ** 2)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_obj, n_trials=n_trials)
    b = study.best_params
    wn = _normalize_weights({k: b[k] for k in ("w_direction", "w_breadth", "w_magnitude", "w_regime")})
    weights = {"w_direction": float(wn[0]), "w_breadth": float(wn[1]),
               "w_magnitude": float(wn[2]), "w_regime": float(wn[3])}
    thresholds = {k: b[k] for k in ("buy_strong", "buy_moderate", "sell_moderate", "sell_strong")}
    return weights, thresholds


def walk_forward_ic(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HOLD_HORIZONS,
    n_splits: int = 6,
    min_train_frac: float = 0.45,
    n_trials: int = 20,
    l2_alpha: float = 0.10,
    n_cv_folds: int = 4,
) -> list[dict]:
    """Expanding-window walk-forward: re-calibrate on each expanding train block,
    measure IC on the NEXT purged test block. Every test IC is genuinely
    out-of-sample, so the sequence reveals whether the edge is durable across
    time or just a recent-regime artifact.

    Returns a list of ``{"test_start", "n_test", "ic"}`` (one per window).
    """
    n = len(frame)
    if n < 250:
        return []
    purge = int(max(horizons))
    start = max(60, int(n * min_train_frac))
    span = n - purge - start
    if span < 60:
        return []
    n_splits = max(2, min(int(n_splits), span // 30))
    step = max(30, span // n_splits)

    out: list[dict] = []
    cut = start
    while cut + purge + 20 <= n:
        train = frame.iloc[:cut]
        test = frame.iloc[cut + purge: cut + purge + step]
        if len(test) >= 20:
            w, t = _optimize_frame(train, horizons, n_trials, l2_alpha, n_cv_folds)
            ic, _ = _score_frame(test, w, t, horizons)
            try:
                ts = frame.index[cut + purge]
            except Exception:
                ts = cut + purge
            out.append({
                "test_start": ts,
                "n_test": int(len(test)),
                "ic": float(ic) if not np.isnan(ic) else float("nan"),
            })
        cut += step
    return out


# ════════════════════════════════════════════════════════════════════════
# Tuner
# ════════════════════════════════════════════════════════════════════════


class ConvergenceTuner:
    """Optuna TPE optimizer for (weights, thresholds).

    Usage:
        tuner = ConvergenceTuner(convergence_df, aarambh_ts, universe="NIFTY 50")
        profile, study = tuner.optimize(n_trials=50)
        tuner.evaluate_validation()  # populates val_score on the profile
        save_profile(profile)
    """

    def __init__(
        self,
        convergence_df: pd.DataFrame,
        aarambh_ts: pd.DataFrame,
        universe: str = "NIFTY 50",
        selected_index: str | None = None,
        train_frac: float = 0.70,
        horizons: tuple[int, ...] = HOLD_HORIZONS,
        l2_alpha: float = 0.10,
        target_col: str = "Actual",
        n_cv_folds: int = 5,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError(
                "optuna is not installed. Add `optuna>=3.5.0` to requirements.txt "
                "and `pip install -r requirements.txt`."
            )
        self.universe = universe
        self.selected_index = selected_index
        self.train_frac = float(train_frac)
        self.horizons = tuple(horizons)
        self.l2_alpha = float(l2_alpha)

        # Forward returns for IC scoring must come from a PRICE LEVEL column.
        # In returns-space modeling "Actual" holds per-period returns, so the
        # caller passes the level column (e.g. "Price") explicitly.
        frame = _build_calibration_frame(
            convergence_df, aarambh_ts, target_col=target_col, horizons=self.horizons,
        )
        if frame.empty:
            raise ValueError(
                "Cannot calibrate: no aligned dates after computing forward returns. "
                "Need convergence_df with dim_* columns AND aarambh_ts with the target column."
            )

        # Purged train/holdout split (chronological — preserves causality):
        #   • opt_frame = first train_frac, used by the objective as K rolling
        #     CV folds, so the search is rewarded for params robust ACROSS time
        #     rather than ones that fit a single slice's noise.
        #   • val_frame = the last (1 - train_frac), a TRUE holdout the optimiser
        #     never sees → an honest generalization estimate.
        #   • a purge gap of max(horizon) between the two prevents the
        #     overlapping forward-return target from leaking across the boundary.
        n = len(frame)
        self.n_cv_folds = max(2, int(n_cv_folds))
        purge = int(max(self.horizons))
        n_holdout = max(20, int(n * (1.0 - self.train_frac)))
        opt_end = max(20, n - n_holdout - purge)
        self.opt_frame   = frame.iloc[:opt_end].copy()
        self.val_frame   = frame.iloc[n - n_holdout:].copy()
        self.train_frame = self.opt_frame  # alias for reporting/logging
        self.full_frame  = frame
        self.study: optuna.Study | None = None
        self.best_weights = DEFAULT_WEIGHTS.copy()
        self.best_thresholds = DEFAULT_THRESHOLDS.copy()
        self.train_score = 0.0
        self.val_score   = 0.0
        self.train_ic    = 0.0
        self.val_ic      = 0.0

    def _suggest_params(self, trial: "optuna.Trial") -> tuple[dict, dict]:
        # Weights — Dirichlet-style: sample 4 unbounded reals in [0.05, 1],
        # then normalize. The L2 penalty keeps them close to uniform.
        w = {
            "w_direction": trial.suggest_float("w_direction", 0.05, 1.0),
            "w_breadth":   trial.suggest_float("w_breadth",   0.05, 1.0),
            "w_magnitude": trial.suggest_float("w_magnitude", 0.05, 1.0),
            "w_regime":    trial.suggest_float("w_regime",    0.05, 1.0),
        }
        # Thresholds — ordered: buy_strong < buy_moderate < 0 < sell_moderate < sell_strong
        buy_strong   = trial.suggest_float("buy_strong",   -0.85, -0.30)
        buy_moderate = trial.suggest_float("buy_moderate", buy_strong + 0.05, -0.05)
        sell_moderate = trial.suggest_float("sell_moderate", 0.05, 0.50)
        sell_strong   = trial.suggest_float("sell_strong", sell_moderate + 0.05, 0.85)
        t = {
            "buy_strong":    buy_strong,
            "buy_moderate":  buy_moderate,
            "sell_moderate": sell_moderate,
            "sell_strong":   sell_strong,
        }
        return w, t

    def _score_cv(self, frame: pd.DataFrame, w: dict, t: dict) -> tuple[float, float]:
        """Score (w, t) across K contiguous time folds.

        Returns (mean_IC, robust_score) where robust_score = mean fold score
        minus a 0.5·std consistency penalty. Evaluating on multiple disjoint
        time blocks stops the optimiser from exploiting one slice's noise — a
        param set must work across the whole history to score well.
        """
        # Hot path: the optimiser scores ``self.opt_frame`` thousands of times.
        # Pre-rank its fold returns once and reuse (bit-identical to _score_frame).
        if frame is self.opt_frame:
            folds, single = self._opt_folds()
            if single:
                return _score_prepared(folds[0], w, t, self.horizons)
            ics: list[float] = []
            scores: list[float] = []
            for fold in folds:
                ic, sc = _score_prepared(fold, w, t, self.horizons)
                if not np.isnan(sc):
                    ics.append(ic)
                    scores.append(sc)
            if not scores:
                return float("nan"), -1e6
            return float(np.mean(ics)), float(np.mean(scores) - 0.5 * np.std(scores))

        # Generic fallback (arbitrary frames): original per-call path.
        n = len(frame)
        if n < self.n_cv_folds * 10:
            return _score_frame(frame, w, t, self.horizons)
        edges = np.linspace(0, n, self.n_cv_folds + 1).astype(int)
        ics = []
        scores = []
        for i in range(self.n_cv_folds):
            sub = frame.iloc[edges[i]:edges[i + 1]]
            ic, sc = _score_frame(sub, w, t, self.horizons)
            if not np.isnan(sc):
                ics.append(ic)
                scores.append(sc)
        if not scores:
            return float("nan"), -1e6
        return float(np.mean(ics)), float(np.mean(scores) - 0.5 * np.std(scores))

    def _opt_folds(self):
        """Lazily build & cache the prepared CV folds for ``opt_frame``."""
        cached = getattr(self, "_opt_folds_cache", None)
        if cached is not None:
            return cached
        frame = self.opt_frame
        n = len(frame)
        if n < self.n_cv_folds * 10:
            folds = ([_PreparedFrame(frame, self.horizons)], True)
        else:
            edges = np.linspace(0, n, self.n_cv_folds + 1).astype(int)
            folds = (
                [_PreparedFrame(frame.iloc[edges[i]:edges[i + 1]], self.horizons)
                 for i in range(self.n_cv_folds)],
                False,
            )
        self._opt_folds_cache = folds
        return folds

    def _objective(self, trial: "optuna.Trial") -> float:
        w, t = self._suggest_params(trial)
        _, score = self._score_cv(self.opt_frame, w, t)
        if np.isnan(score):
            return -1e6
        # L2 regularization on weights deviation from uniform (0.25 each).
        weights_arr = _normalize_weights(w)
        l2 = float(np.sum((weights_arr - 0.25) ** 2))
        return float(score - self.l2_alpha * l2)

    def optimize(
        self,
        n_trials: int = 50,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> tuple[IntelligenceProfile, "optuna.Study"]:
        """Run TPE search. Returns (best_profile, study)."""
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        def _cb(study: "optuna.Study", trial: "optuna.FrozenTrial") -> None:
            if progress_callback is not None:
                progress_callback(trial.number + 1, n_trials, float(study.best_value))

        self.study.optimize(self._objective, n_trials=n_trials, callbacks=[_cb] if progress_callback else None)

        # Extract best
        best = self.study.best_params
        weights = {k: best[k] for k in ("w_direction", "w_breadth", "w_magnitude", "w_regime")}
        thresholds = {k: best[k] for k in ("buy_strong", "buy_moderate", "sell_moderate", "sell_strong")}
        # Normalize stored weights so they sum to 1.0 (Dirichlet)
        wn = _normalize_weights(weights)
        weights = {
            "w_direction": float(wn[0]),
            "w_breadth":   float(wn[1]),
            "w_magnitude": float(wn[2]),
            "w_regime":    float(wn[3]),
        }
        self.best_weights = weights
        self.best_thresholds = thresholds
        # Train IC = cross-validated mean over the optimisation folds (what the
        # search actually optimised), so it is directly comparable to val_ic.
        self.train_ic, self.train_score = self._score_cv(self.opt_frame, weights, thresholds)
        return self._make_profile(), self.study

    def evaluate_validation(self) -> tuple[float, float]:
        """Score the best params on the held-out validation set."""
        self.val_ic, self.val_score = _score_frame(
            self.val_frame, self.best_weights, self.best_thresholds, self.horizons,
        )
        return self.val_ic, self.val_score

    def sensitivity(self) -> dict[str, float]:
        """Optuna fANOVA importance over trial history (percent share)."""
        if self.study is None or len(self.study.trials) < 4:
            return {}
        try:
            imp = optuna.importance.get_param_importances(self.study)
            total = float(sum(imp.values()))
            if total <= 0:
                return {}
            return {k: float(v / total) * 100.0 for k, v in imp.items()}
        except Exception as e:
            log.debug("fANOVA importance failed: %s", e)
            return {}

    def _make_profile(self) -> IntelligenceProfile:
        return IntelligenceProfile(
            weights=self.best_weights,
            thresholds=self.best_thresholds,
            universe=self.universe,
            selected_index=self.selected_index,
            train_score=float(self.train_score) if not np.isnan(self.train_score) else 0.0,
            val_score=float(self.val_score) if not np.isnan(self.val_score) else 0.0,
            train_ic=float(self.train_ic) if not np.isnan(self.train_ic) else 0.0,
            val_ic=float(self.val_ic) if not np.isnan(self.val_ic) else 0.0,
            n_train_dates=int(len(self.train_frame)),
            n_val_dates=int(len(self.val_frame)),
            n_trials=int(len(self.study.trials)) if self.study else 0,
            sensitivity=self.sensitivity(),
            timestamp=time.strftime("%Y-%m-%d %H:%M"),
            version=PROFILE_VERSION,
        )


# ════════════════════════════════════════════════════════════════════════
# Convenience API for the rest of the app
# ════════════════════════════════════════════════════════════════════════


def calibrate(
    convergence_df: pd.DataFrame,
    aarambh_ts: pd.DataFrame,
    universe: str = "NIFTY 50",
    selected_index: str | None = None,
    n_trials: int = 50,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> IntelligenceProfile:
    """End-to-end calibration: fit, validate, persist, return the profile."""
    tuner = ConvergenceTuner(
        convergence_df, aarambh_ts,
        universe=universe, selected_index=selected_index,
    )
    profile, _ = tuner.optimize(n_trials=n_trials, progress_callback=progress_callback)
    tuner.evaluate_validation()
    profile = tuner._make_profile()  # refresh with val scores
    save_profile(profile)
    return profile


def resolve_active(universe: str, index: str | None) -> tuple[dict, dict, IntelligenceProfile | None]:
    """Resolve which weights + thresholds the app should use right now.

    Returns ``(weights, thresholds, loaded_profile_or_None)``. Falls back
    to factory defaults if no profile exists for this universe.
    """
    profile = load_profile_for(universe, index)
    if profile is not None:
        return profile.weights, profile.thresholds, profile
    return DEFAULT_WEIGHTS.copy(), DEFAULT_THRESHOLDS.copy(), None


def apply_calibrated_weights(
    convergence_df: pd.DataFrame,
    weights: dict[str, float],
    *,
    strong_bullish: float = -60.0,
    moderate_bullish: float = -30.0,
    weak_bullish: float = -10.0,
    weak_bearish: float = +10.0,
    moderate_bearish: float = +30.0,
    strong_bearish: float = +60.0,
) -> pd.DataFrame:
    """Re-weight an existing convergence_df from its per-dimension columns.

    The per-dimension agreement scores (``dim_direction``, ``dim_breadth``,
    ``dim_magnitude``, ``dim_regime``) are *independent of weights*. The
    composite ``convergence_score`` and ``convergence_zone`` depend on the
    weights. After auto-calibration, this function applies the new weights
    via vectorized recomputation — no need to re-run the full CrossValidator
    loop over every date.

    Returns a new DataFrame (does not mutate the input).
    """
    if convergence_df is None or convergence_df.empty:
        return convergence_df
    needed = {"dim_direction", "dim_breadth", "dim_magnitude", "dim_regime"}
    if not needed.issubset(convergence_df.columns):
        return convergence_df

    out = convergence_df.copy()
    composite = _composite_signal(out, weights)  # in [-1, +1]
    out["convergence_score"] = (composite * 100.0).round(2)

    # Recompute agreement_ratio with calibrated weights too, so the
    # user-visible AGREEMENT % stays consistent with the calibrated
    # convergence_score (instead of being a plain 4-way mean of dim scores
    # while the convergence_score reflects weighted importance).
    w_norm_arr = _normalize_weights(weights)
    M = np.column_stack([
        out["dim_direction"].to_numpy(dtype=np.float64),
        out["dim_breadth"].to_numpy(dtype=np.float64),
        out["dim_magnitude"].to_numpy(dtype=np.float64),
        out["dim_regime"].to_numpy(dtype=np.float64),
    ])
    out["agreement_ratio"] = (M @ w_norm_arr).round(3)

    # `confidence` = agreement_ratio × data-coverage (analyzed / basket size).
    # We can't recover the per-row coverage factor from convergence_df alone,
    # but the stored `confidence` already encodes it, so we rebase it using the
    # new agreement_ratio while preserving the per-row scaling factor
    # (= old_confidence / old_agreement when old_agreement > 0).
    if "confidence" in out.columns and "agreement_ratio" in convergence_df.columns:
        old_agreement = convergence_df["agreement_ratio"].astype(float).to_numpy()
        old_conf = out["confidence"].astype(float).to_numpy()
        scale = np.where(old_agreement > 1e-9, old_conf / np.maximum(old_agreement, 1e-9), 1.0)
        out["confidence"] = (out["agreement_ratio"].to_numpy() * scale).round(3)

    # Reclassify zones using the same band layout as CrossValidator
    def _zone(cs: float) -> str:
        if cs <= strong_bullish:
            return "STRONG_CONVERGENCE_BULLISH"
        if cs <= moderate_bullish:
            return "MODERATE_BULLISH"
        if cs <= weak_bullish:
            return "WEAK_BULLISH"
        if cs <= weak_bearish:
            return "DIVERGENT"
        if cs <= moderate_bearish:
            return "WEAK_BEARISH"
        if cs <= strong_bearish:
            return "MODERATE_BEARISH"
        return "STRONG_CONVERGENCE_BEARISH"

    out["convergence_zone"] = out["convergence_score"].apply(_zone)

    # Update the per-row weight columns to reflect what was actually applied
    w_norm = _normalize_weights(weights)
    out["w_direction"] = float(round(w_norm[0], 3))
    out["w_breadth"]   = float(round(w_norm[1], 3))
    out["w_magnitude"] = float(round(w_norm[2], 3))
    out["w_regime"]    = float(round(w_norm[3], 3))
    return out
