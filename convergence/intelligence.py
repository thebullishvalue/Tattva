"""
Tattva — Intelligence: online convergence weighting, learned forward only.
तत्त्व (Tattva) — "Principle / Essence"

This module used to run an Optuna TPE search over the dimension weights and
classification thresholds, fit on the whole history, and then apply the winner
back across that same history. It produced a genuinely useful number — a
per-target Val IC — and one serious defect: **the published record repainted.**
Adding a single session re-ran the search, and every convergence score ever
computed changed with it. Worse, the winning profile was persisted to disk and
reloaded on the next run, so the output depended not only on the data but on
when you had last calibrated. Two people with identical data could disagree.

The replacement keeps the goal and drops the mechanism. Dimension weights are
still LEARNED — the four agreement dimensions are still not equally
informative, and which one matters still varies by instrument — but they are
learned the way the Mūla engine learns which of its two valuation views to
trust: recursively, from evidence that had already resolved, discounted so the
recent past counts for more. At date *t* the weights reflect outcomes through
``t - h`` and nothing later. A score published on *t* is never revised.

Concretely, for each dimension the learner asks the only question that matters:
when this dimension leaned bullish, did the target subsequently go up? That is
scored per dimension, accumulated with a forgetting factor, and turned into
weights whose sharpness tracks the *significance* of the skill difference
rather than the raw number of observations (see
:class:`analytics.adaptive.OnlineSkillWeights`).

What is retained from the old module: ``_build_calibration_frame`` (assembling
the per-date dimension matrix and forward returns) and ``walk_forward_ic``,
which was always causal — it re-fit on expanding blocks and scored the next
purged block — and is now the honest read-only durability diagnostic rather
than something whose output feeds back into the signal.

What is gone: the Optuna dependency, the TPE search, the profile JSON on disk,
and the "Intelligence Mode" on/off toggle. There is no mode any more, because
there is no expensive optional step to gate — the learner costs one pass over a
frame the pipeline had already built.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analytics.adaptive import OnlineSkillWeights
from core.config import (CONV_WEIGHT_BREADTH, CONV_WEIGHT_DIRECTION,
                         CONV_WEIGHT_MAGNITUDE, CONV_WEIGHT_REGIME,
                         HOLD_HORIZONS)

log = logging.getLogger(__name__)

#: The four agreement dimensions the convergence composite is built from.
DIMENSIONS: tuple[str, ...] = ("direction", "breadth", "magnitude", "regime")

#: Prior weights — where the learner starts before any outcome has resolved.
PRIOR_WEIGHTS: dict[str, float] = {
    "direction": CONV_WEIGHT_DIRECTION,
    "breadth": CONV_WEIGHT_BREADTH,
    "magnitude": CONV_WEIGHT_MAGNITUDE,
    "regime": CONV_WEIGHT_REGIME,
}


def _build_calibration_frame(
    convergence_df: pd.DataFrame,
    fvo_ts: pd.DataFrame,
    target_col: str = "Actual",
    horizons: tuple[int, ...] = HOLD_HORIZONS,
) -> pd.DataFrame:
    """Assemble the per-date calibration matrix.

    Columns:
      - dim_direction, dim_breadth, dim_magnitude, dim_regime: per-day
        agreement sub-scores (already in [0, 1])
      - convergence_score: legacy signed composite (used for sign anchor)
      - Ret_{h}b for h in horizons: forward log-returns of the target
        column (the active target's price level — commodity/FX/index)

    Returns an empty frame if any input is missing.
    """
    if convergence_df is None or convergence_df.empty:
        return pd.DataFrame()
    needed = {"dim_direction", "dim_breadth", "dim_magnitude", "dim_regime", "convergence_score"}
    if not needed.issubset(convergence_df.columns):
        return pd.DataFrame()
    if fvo_ts is None or target_col not in fvo_ts.columns:
        return pd.DataFrame()

    conv = convergence_df.copy()
    conv.index = pd.to_datetime(conv.index, errors="coerce")
    conv = conv[~conv.index.isna()].sort_index()

    a_dedup = fvo_ts[~fvo_ts.index.duplicated(keep="last")].copy()
    if "Date" in a_dedup.columns:
        a_dedup["Date"] = pd.to_datetime(a_dedup["Date"], errors="coerce", dayfirst=True)
        a_dedup = a_dedup.dropna(subset=["Date"]).set_index("Date")
    else:
        a_dedup.index = pd.to_datetime(a_dedup.index, errors="coerce")
        a_dedup = a_dedup[~a_dedup.index.isna()]
    a_dedup = a_dedup.sort_index()

    # Reindex without forward-fill first so we can detect carry-forward rows.
    target_raw = a_dedup[target_col].astype(float).reindex(conv.index)
    is_carried = target_raw.isna()  # True = date has no genuine Mūla price
    target = target_raw.ffill()

    # Forward log-returns at each horizon; mask rows where price was carried
    # forward (no genuine observation) so mis-stated labels are excluded.
    log_target = np.log(target.replace(0, np.nan)).ffill()
    log_target = log_target.where(~is_carried)
    out = conv.copy()
    for h in horizons:
        out[f"Ret_{h}b"] = log_target.shift(-h) - log_target

    # Drop dates with NaN at any horizon (tail of the series)
    ret_cols = [f"Ret_{h}b" for h in horizons]
    out = out.dropna(subset=ret_cols + list(needed))
    return out


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

#: Alias kept for the cross-check in research/test_convergence_integrity.py,
#: which asserts CrossValidator's composite matches an INDEPENDENT
#: implementation of the same formula. Two implementations that must agree is
#: the point — it is what catches a silent divergence between what the
#: convergence layer computes and what this module scores.
DEFAULT_WEIGHTS: dict[str, float] = dict(PRIOR_WEIGHTS)


def _agreement_strength(frame: pd.DataFrame, w: dict[str, float]) -> np.ndarray:
    """Weighted AGREEMENT across the four dimensions, mapped to [0, 1].

    ``sum_d w_d * (2*dim_d - 1)`` re-centred to [0, 1]. This is a magnitude,
    not a direction: the dim_* scores measure how strongly the two engines
    AGREE, and a high value could be agreement about a top or about a bottom.
    """
    ws = np.array([float(w.get(d, 0.0)) for d in DIMENSIONS], dtype=np.float64)
    tot = ws.sum()
    ws = ws / tot if tot > 1e-12 else np.full(len(DIMENSIONS), 1.0 / len(DIMENSIONS))
    D = frame[[f"dim_{d}" for d in DIMENSIONS]].to_numpy(dtype=np.float64)
    composite = np.sum(ws * (2.0 * D - 1.0), axis=1)
    return (composite + 1.0) / 2.0


def _composite_signal(frame: pd.DataFrame, w: dict[str, float]) -> np.ndarray:
    """The published convergence signal in [-1, +1] — direction AND strength.

    Mirrors ``CrossValidator`` exactly:

        score = -consensus_direction * agreement_strength

    The sign comes from ``consensus_direction`` (the engines' own signed
    leans), the magnitude from weighted agreement. Getting this wrong is easy
    and consequential — weighting only the agreement term produces a number
    that looks like a signal, moves like a signal, and has no direction in it —
    which is why ``research/test_convergence_integrity.py`` re-derives the
    published score from the stored columns through this independent
    implementation and requires the two to match.
    """
    strength = _agreement_strength(frame, w)
    if "consensus_direction" not in frame.columns:
        return strength * 0.0
    cd = pd.to_numeric(frame["consensus_direction"], errors="coerce").to_numpy(dtype=np.float64)
    return -cd * strength


# ════════════════════════════════════════════════════════════════════════
# Online dimension weighting
# ════════════════════════════════════════════════════════════════════════


def online_dimension_weights(
    frame: pd.DataFrame,
    horizon: int,
    halflife: float = 252.0,
    floor: float = 0.05,
    warmup: int = 126,
) -> pd.DataFrame:
    """Causal per-date weights for the four convergence dimensions.

    ``frame`` must carry ``dim_*`` columns in [0, 1] and a ``Ret_{horizon}b``
    forward return. Each dimension's signed lean is ``2*dim - 1`` (the same
    re-centring the composite uses), scored against the realised forward move.

    Ordering, again, is the point: the outcome of a call made at ``s`` is only
    knowable at ``s + horizon``, so step ``t`` absorbs ``s = t - horizon``. The
    first ``warmup`` rows return the prior unchanged — a weight estimated from
    twenty resolved outcomes is noise, and the prior is a better answer than a
    confident wrong one.

    Returns a frame indexed like ``frame`` with one column per dimension,
    each row summing to 1.
    """
    cols = [f"dim_{d}" for d in DIMENSIONS]
    if frame is None or frame.empty or not set(cols).issubset(frame.columns):
        return pd.DataFrame()
    ret_col = f"Ret_{horizon}b"
    if ret_col not in frame.columns:
        return pd.DataFrame()

    D = frame[cols].to_numpy(dtype=np.float64)
    lean = 2.0 * D - 1.0                       # [0,1] -> [-1,+1]
    fwd = frame[ret_col].to_numpy(dtype=np.float64)

    prior = np.array([PRIOR_WEIGHTS[d] for d in DIMENSIONS], dtype=np.float64)
    prior = prior / max(prior.sum(), 1e-12)

    skill = OnlineSkillWeights(list(DIMENSIONS), halflife=halflife, floor=floor)
    h = max(1, int(horizon))
    W = np.empty((len(frame), len(DIMENSIONS)), dtype=np.float64)
    for t in range(len(frame)):
        s = t - h
        if s >= 0:
            skill.observe(lean[s], fwd[s])     # resolved at t; never before
        W[t] = prior if skill.n_scored < warmup else skill.weights()
    return pd.DataFrame(W, index=frame.index, columns=list(DIMENSIONS))


def apply_online_weights(
    convergence_df: pd.DataFrame,
    fvo_ts: pd.DataFrame,
    horizon: int,
    target_col: str = "Price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute ``convergence_score`` with per-date, causally-learned weights.

    Returns ``(convergence_df with the score replaced, weight history)``. Rows
    the learner could not reach (no resolved forward return — the tail, and any
    date missing a dimension) keep their original prior-weighted score rather
    than being dropped: the newest rows are the ones being traded, and a live
    signal that vanishes because its outcome has not happened yet would be
    worse than one carrying the prior.

    The recomputation mirrors ``CrossValidator`` exactly — direction from the
    stored ``consensus_direction``, magnitude from the re-weighted agreement:
    ``-consensus_direction * agreement_strength * 100``. Re-weighting only the
    agreement term (and publishing THAT as the score) would silently strip the
    direction out of the signal; the convergence integrity test exists to catch
    precisely that substitution.
    """
    if convergence_df is None or convergence_df.empty:
        return convergence_df, pd.DataFrame()
    cols = [f"dim_{d}" for d in DIMENSIONS]
    if not set(cols).issubset(convergence_df.columns):
        return convergence_df, pd.DataFrame()

    frame = _build_calibration_frame(convergence_df, fvo_ts, target_col=target_col,
                                     horizons=(horizon,))
    if frame.empty:
        return convergence_df, pd.DataFrame()

    W = online_dimension_weights(frame, horizon=horizon)
    if W.empty:
        return convergence_df, pd.DataFrame()

    out = convergence_df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    Wr = W.reindex(out.index)
    # Rows past the learner's reach keep the LAST learned weights (a forward
    # carry, which is causal) rather than reverting to the prior mid-series.
    Wr = Wr.ffill()
    covered = Wr.notna().all(axis=1)
    if not covered.any():
        return convergence_df, W

    if "consensus_direction" not in out.columns:
        return convergence_df, W
    lean = 2.0 * out.loc[covered, cols].to_numpy(dtype=np.float64) - 1.0
    w = Wr.loc[covered].to_numpy(dtype=np.float64)
    strength = (np.sum(w * lean, axis=1) + 1.0) / 2.0        # agreement, [0,1]
    cd = pd.to_numeric(out.loc[covered, "consensus_direction"],
                       errors="coerce").to_numpy(dtype=np.float64)
    out.loc[covered, "convergence_score"] = -cd * strength * 100.0
    for d in DIMENSIONS:
        out[f"w_{d}"] = Wr[d]
    return out, W


# ════════════════════════════════════════════════════════════════════════
# Read-only durability diagnostic
# ════════════════════════════════════════════════════════════════════════


def walk_forward_ic(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HOLD_HORIZONS,
    n_splits: int = 6,
    min_train_frac: float = 0.45,
) -> list[dict]:
    """Expanding-window out-of-sample IC of the ONLINE-weighted composite.

    Each window learns weights on the expanding train block and scores the next
    purged test block, so every reported IC is genuinely out-of-sample. This is
    now a pure diagnostic: nothing it returns feeds back into the signal, which
    is what lets it be read as evidence rather than as a fitted result.

    Returns a list of ``{"test_start", "n_test", "ic"}`` (one per window).
    """
    n = len(frame)
    if n < 250:
        return []
    purge = int(max(horizons))
    h_score = int(min(horizons))
    ret_col = f"Ret_{h_score}b"
    if ret_col not in frame.columns:
        return []

    out: list[dict] = []
    start = int(n * min_train_frac)
    step = max(1, (n - start) // max(1, n_splits))
    for k in range(n_splits):
        tr_end = start + k * step
        te_start = tr_end + purge
        te_end = min(te_start + step, n)
        if te_end - te_start < 30 or tr_end < 200:
            break
        W = online_dimension_weights(frame.iloc[:tr_end], horizon=h_score)
        if W.empty:
            continue
        w = W.iloc[-1].to_numpy(dtype=np.float64)
        test = frame.iloc[te_start:te_end]
        lean = 2.0 * test[[f"dim_{d}" for d in DIMENSIONS]].to_numpy(dtype=np.float64) - 1.0
        sig = np.sum(w * lean, axis=1)
        ic = _spearman_ic(sig, test[ret_col].to_numpy(dtype=np.float64))
        if np.isfinite(ic):
            out.append({"test_start": test.index[0], "n_test": int(len(test)), "ic": float(ic)})
    return out
