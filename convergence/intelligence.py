"""
Tattva — Intelligence: online convergence weighting, and the edge every engine is graded by.
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

WHAT A DIMENSION IS SCORED ON. A dimension score is AGREEMENT — 0.9 means the
engines agree, at a top or at a bottom alike — so it has no direction of its
own, and scoring ``2·dim − 1`` against the forward return (which this module
did until 2026-10) rewarded whichever dimension happened to sit above 0.5 in a
rising market. Planted in a steady up-trend, it put 0.85 on a dimension that
carried no information. The published composite is
``-consensus_direction × agreement``, so each dimension's contribution to the
BULLISH score is ``consensus_direction × (2·dim − 1)``: agreement, signed by
the way the two engines lean. That is what is scored now — "when this
dimension's agreement pointed bullish, did the target go up?" — which is the
question the module docstring always claimed to ask.

THE EDGE. :func:`series_edge` is the one measurement of directional skill in
the system, used for every engine alike: the rank IC of a bullish-positive
lean against the realised h-day forward log return, NON-overlapping. It
replaced ``walk_forward_ic``, which scored the unsigned agreement (no
direction in it at all — a planted composite of IC +0.31 came back as −0.01),
at the shortest hold horizon rather than the instrument's own, on every
overlapping daily row. Because every lean it is given is already causal (Mūla
is fit to no label; the dimension weights are learned forward), there is
nothing to re-fit per window: an expanding "train, then score the next block"
loop over a causal series is the same thing as scoring each block of the
published series, and this does the latter directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analytics.adaptive import OnlineSkillWeights
from convergence.cross_validator import DIMENSIONS, PRIOR_WEIGHTS, normalise_weights
from core.config import HOLD_HORIZONS

log = logging.getLogger(__name__)

__all__ = [
    "DIMENSIONS",
    "PRIOR_WEIGHTS",
    "online_dimension_weights",
    "apply_online_weights",
    "series_edge",
]


def _forward_log_return(price: pd.Series, horizon: int) -> pd.Series:
    """h-SESSION forward log return on the price's own calendar.

    Computed on the price series' own index, before any alignment, so ``h``
    always means h trading sessions of the target — never "h rows of whatever
    frame it is later joined to", which is longer wherever that frame has gaps
    (e.g. dates with no Swayam row).
    """
    p = pd.to_numeric(price, errors="coerce").astype(float)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    lp = np.log(p.where(p > 0))
    return lp.shift(-int(horizon)) - lp


def _datetime_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _price_series(fvo_ts: pd.DataFrame, target_col: str) -> pd.Series:
    """The target's price level from a Mūla frame, date-indexed."""
    a = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
    if "Date" in a.columns:
        s = pd.Series(pd.to_numeric(a[target_col], errors="coerce").to_numpy(),
                      index=pd.to_datetime(a["Date"], errors="coerce", dayfirst=True))
    else:
        s = pd.to_numeric(a[target_col], errors="coerce")
    return _datetime_index(s)


def _build_calibration_frame(
    convergence_df: pd.DataFrame,
    fvo_ts: pd.DataFrame,
    target_col: str = "Price",
    horizons: tuple[int, ...] = HOLD_HORIZONS,
) -> pd.DataFrame:
    """Assemble the per-date learning matrix.

    Columns: the ``dim_*`` agreement scores, ``consensus_direction``, and
    ``Ret_{h}b`` — the h-session forward log return of the target — for each
    ``h``. Rows are dates the convergence layer actually scored (a date with no
    Swayam row is absent from ``convergence_df`` by construction) AND on which
    the Swayam evidence was observed rather than carried forward
    (``swayam_native``): a run of forward-filled breadth is one observation,
    not a new one per day. Empty if any input is missing.

    The newest rows KEEP a NaN return rather than being dropped. Their outcome
    has not happened, so they contribute nothing to learning — but they still
    need a weight of their own. Dropping them (as this used to) left the
    newest h sessions on a forward-filled weight that was replaced by the
    properly learned one once their outcomes arrived: the newest published
    scores were revised by data that came after them.
    """
    if convergence_df is None or convergence_df.empty:
        return pd.DataFrame()
    needed = {f"dim_{d}" for d in DIMENSIONS} | {"consensus_direction"}
    if not needed.issubset(convergence_df.columns):
        return pd.DataFrame()
    if fvo_ts is None or target_col not in fvo_ts.columns:
        return pd.DataFrame()

    conv = _datetime_index(convergence_df)
    if "swayam_native" in conv.columns:
        conv = conv[conv["swayam_native"].astype(bool)]
    price = _price_series(fvo_ts, target_col)

    out = conv.copy()
    for h in horizons:
        out[f"Ret_{h}b"] = _forward_log_return(price, h).reindex(out.index)
    return out.dropna(subset=sorted(needed))


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN on degenerate input."""
    if len(x) < 5 or len(x) != len(y):
        return float("nan")
    rx = rankdata(x).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


# ════════════════════════════════════════════════════════════════════════
# Online dimension weighting
# ════════════════════════════════════════════════════════════════════════


def _prior_vector(prior: dict[str, float] | None) -> np.ndarray:
    w = normalise_weights(prior)
    return np.array([w[d] for d in DIMENSIONS], dtype=np.float64)


#: Resolved outcomes the learner absorbs before its weights replace the prior.
#: A weight estimated from twenty resolved calls is noise. app.py gates on the
#: same number, so "learning skipped" means exactly "the learner would only
#: have returned its prior".
LEARNER_WARMUP = 126


def online_dimension_weights(
    frame: pd.DataFrame,
    horizon: int,
    prior: dict[str, float] | None = None,
    halflife: float = 252.0,
    floor: float = 0.05,
    warmup: int = LEARNER_WARMUP,
) -> pd.DataFrame:
    """Causal per-date weights for the four convergence dimensions.

    ``frame`` must carry ``dim_*`` in [0, 1], ``consensus_direction`` and a
    ``Ret_{horizon}b`` forward return. Each dimension's call on a date is its
    agreement signed by the engines' shared lean, ``cd · (2·dim − 1)`` — its
    contribution to the bullish composite — scored against the realised move.

    Ordering is the point: the outcome of a call made at ``s`` is only
    knowable at ``s + horizon``, so step ``t`` absorbs ``s = t - horizon``. The
    frame may skip dates (no Swayam row, carried-forward breadth); a skipped
    date only lengthens the gap between a call and the step that scores it, so
    the ordering stays causal. The first ``warmup`` scored outcomes return the
    prior unchanged — a weight estimated from twenty resolved outcomes is
    noise, and the prior is a better answer than a confident wrong one.

    ``prior`` is the instrument's own starting belief (either key style — see
    ``cross_validator.normalise_weights``); ``None`` is the global prior.

    Returns a frame indexed like ``frame`` with one column per dimension,
    each row summing to 1.
    """
    cols = [f"dim_{d}" for d in DIMENSIONS]
    ret_col = f"Ret_{horizon}b"
    if (frame is None or frame.empty or not set(cols).issubset(frame.columns)
            or ret_col not in frame.columns or "consensus_direction" not in frame.columns):
        return pd.DataFrame()

    D = frame[cols].to_numpy(dtype=np.float64)
    cd = pd.to_numeric(frame["consensus_direction"], errors="coerce").to_numpy(dtype=np.float64)
    lean = np.nan_to_num(cd)[:, None] * (2.0 * D - 1.0)
    fwd = frame[ret_col].to_numpy(dtype=np.float64)

    p0 = _prior_vector(prior)
    skill = OnlineSkillWeights(list(DIMENSIONS), halflife=halflife, floor=floor)
    h = max(1, int(horizon))
    W = np.empty((len(frame), len(DIMENSIONS)), dtype=np.float64)
    for t in range(len(frame)):
        s = t - h
        if s >= 0 and np.isfinite(fwd[s]):
            skill.observe(lean[s], fwd[s])     # resolved at t; never before
        W[t] = p0 if skill.n_scored < warmup else skill.weights()
    return pd.DataFrame(W, index=frame.index, columns=list(DIMENSIONS))


def apply_online_weights(
    convergence_df: pd.DataFrame,
    fvo_ts: pd.DataFrame,
    horizon: int,
    target_col: str = "Price",
    prior: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild ``convergence_score`` and ``agreement_ratio`` with learned weights.

    Returns ``(convergence_df with both columns and w_* replaced, weight
    history)``. The weight a row receives is the one learned from outcomes
    resolved by that row's date — including the newest rows, whose own outcome
    is still open — and is carried forward only across dates the learner does
    not score (carried-forward Swayam breadth), which is causal. Rows before
    the learner's first date keep the prior-weighted values ``CrossValidator``
    already wrote.

    Both columns move together on purpose. ``agreement_ratio`` is the number
    the Overview and the Convergence page print as "how far the engines agree";
    leaving it on the prior while the composite beside it used learned weights
    made the two disagree about what agreement meant.
    """
    if convergence_df is None or convergence_df.empty:
        return convergence_df, pd.DataFrame()
    cols = [f"dim_{d}" for d in DIMENSIONS]
    if not set(cols).issubset(convergence_df.columns) or "consensus_direction" not in convergence_df.columns:
        return convergence_df, pd.DataFrame()

    frame = _build_calibration_frame(convergence_df, fvo_ts, target_col=target_col,
                                     horizons=(horizon,))
    if frame.empty:
        return convergence_df, pd.DataFrame()

    W = online_dimension_weights(frame, horizon=horizon, prior=prior)
    if W.empty:
        return convergence_df, pd.DataFrame()

    out = _datetime_index(convergence_df)
    Wr = W.reindex(out.index).ffill()
    covered = Wr.notna().all(axis=1)
    if not covered.any():
        return convergence_df, W

    D = out.loc[covered, cols].to_numpy(dtype=np.float64)
    w = Wr.loc[covered].to_numpy(dtype=np.float64)
    agreement = np.sum(w * D, axis=1)
    cd = pd.to_numeric(out.loc[covered, "consensus_direction"],
                       errors="coerce").to_numpy(dtype=np.float64)
    out.loc[covered, "agreement_ratio"] = agreement
    out.loc[covered, "convergence_score"] = -cd * agreement * 100.0
    for d in DIMENSIONS:
        out.loc[covered, f"w_{d}"] = Wr.loc[covered, d]
    # Keep the caller's index type: the pipeline indexes convergence_df by
    # 'YYYY-MM-DD' strings and joins on them.
    if not isinstance(convergence_df.index, pd.DatetimeIndex):
        out.index = out.index.strftime("%Y-%m-%d")
        out.index.name = convergence_df.index.name
    return out, W


# ════════════════════════════════════════════════════════════════════════
# The edge — one measurement of directional skill, for every engine
# ════════════════════════════════════════════════════════════════════════


#: Fewer non-overlapping observations than this in a block and its IC is
#: reported as missing rather than as a number: eight points is already an IC
#: standard error of ~0.4.
MIN_BLOCK_OBS = 8


def _offset_averaged_ic(lean: np.ndarray, fwd: np.ndarray, h: int) -> float:
    """Mean of the h non-overlapping ICs, one per phase offset.

    Sampling every h-th row makes consecutive forward windows disjoint, which
    is what makes the observations independent and the IC's precision honest
    (a daily-sampled IC on overlapping h-day returns overstates it by ~√h).
    Any single offset throws away (h−1)/h of the data and depends on which
    day the sample happens to start; averaging the h offsets uses all of it
    and removes that arbitrariness without re-introducing the overlap inside
    any one of them.
    """
    ics = []
    for o in range(h):
        x, y = lean[o::h], fwd[o::h]
        if len(x) >= 5 and np.nanstd(x) > 0:
            ic = _spearman_ic(x, y)
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def series_edge(
    lean: pd.Series,
    price: pd.Series,
    horizon: int,
    n_blocks: int = 6,
) -> dict:
    """Out-of-sample directional skill of a bullish-positive lean.

    ``lean`` is dated *t* and uses only data available at *t* (every lean the
    system publishes is causal); it is scored against ``log P(t+h) − log P(t)``
    computed on ``price``'s own session calendar. Rows where either is missing
    are dropped. Nothing is fitted here, so every number is out-of-sample with
    respect to everything after the date it was published.

    Returns::

        {"ic", "se", "t", "n", "n_obs", "horizon",
         "hit", "base", "positive_blocks", "scored_blocks", "n_blocks",
         "blocks": [{"start", "end", "n", "ic"}, ...]}

    * ``ic`` — the offset-averaged non-overlapping Spearman IC over the whole
      history (:func:`_offset_averaged_ic`).
    * ``n`` — independent observations, ``n_obs // h``; ``se`` —
      ``1/√(n−3)``, the single-offset standard error (conservative: the
      offset average is somewhat more precise than any one offset, by an
      amount that depends on the series' own autocorrelation); ``t = ic/se``.
    * ``hit`` / ``base`` — directional hit rate on rows with a non-zero lean,
      beside the rate a constant majority-direction call would have scored on
      the same rows. A hit rate is only readable against its own null.
    * ``blocks`` — the same IC within ``n_blocks`` consecutive stretches of
      history: consistently positive is a durable edge, one lucky stretch is
      not. ``positive_blocks`` counts those above zero among the
      ``scored_blocks`` with at least :data:`MIN_BLOCK_OBS` observations.
    """
    h = max(1, int(horizon))
    empty = {"ic": float("nan"), "se": float("nan"), "t": float("nan"), "n": 0,
             "n_obs": 0, "horizon": h, "hit": float("nan"), "base": float("nan"),
             "positive_blocks": 0, "scored_blocks": 0, "n_blocks": int(n_blocks),
             "blocks": []}
    if lean is None or price is None or len(lean) == 0 or len(price) == 0:
        return empty

    lean_s = _datetime_index(pd.to_numeric(pd.Series(lean), errors="coerce").astype(float))
    fwd = _forward_log_return(_datetime_index(pd.Series(price)), h)
    df = pd.DataFrame({"lean": lean_s, "fwd": fwd.reindex(lean_s.index)}).dropna()
    if len(df) < 5 * h:
        out = dict(empty)
        out["n_obs"] = int(len(df))
        return out

    x = df["lean"].to_numpy(dtype=np.float64)
    y = df["fwd"].to_numpy(dtype=np.float64)
    ic = _offset_averaged_ic(x, y, h)
    n_ind = len(df) // h
    se = float(1.0 / np.sqrt(max(n_ind - 3, 1)))

    called = x != 0
    if called.any():
        hit = float(np.mean(np.sign(x[called]) == np.sign(y[called])) * 100.0)
        up = float(np.mean(y[called] > 0))
        base = float(max(up, 1.0 - up) * 100.0)
    else:
        hit = base = float("nan")

    blocks = []
    for part in np.array_split(np.arange(len(df)), max(1, int(n_blocks))):
        if not len(part):
            continue
        nb = len(part) // h
        b_ic = (_offset_averaged_ic(x[part], y[part], h) if nb >= MIN_BLOCK_OBS
                else float("nan"))
        blocks.append({"start": df.index[part[0]], "end": df.index[part[-1]],
                       "n": int(nb), "ic": b_ic})
    scored = [b["ic"] for b in blocks if np.isfinite(b["ic"])]

    return {
        "ic": ic, "se": se, "t": float(ic / se) if np.isfinite(ic) else float("nan"),
        "n": int(n_ind), "n_obs": int(len(df)), "horizon": h,
        "hit": hit, "base": base,
        "positive_blocks": int(sum(1 for v in scored if v > 0)),
        "scored_blocks": int(len(scored)), "n_blocks": int(len(blocks)),
        "blocks": blocks,
    }
