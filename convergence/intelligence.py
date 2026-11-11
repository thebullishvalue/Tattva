"""
Tattva — grading: the lean every engine publishes, and the call as it is issued.

:func:`series_edge` is the rank IC of a bullish-positive lean against the
h-session forward return, on non-overlapping windows — one measurement, used
for every engine's lean alike. Nothing is fitted here, so every number is
out-of-sample with respect to everything after the date it was published.

:func:`call_record` grades the CALL: claim sessions only, entered at the next
close, in excess of always holding the claimed direction. That is what the
verdict is licensed by (``core.trust``). The two answer different questions —
a lean can rank well while its thresholded call loses to the drift.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from core.config import EXECUTION_LAG
from core.semantics import claim_direction

log = logging.getLogger(__name__)

__all__ = ["series_edge", "claim_outcomes", "call_outcomes", "call_record", "summarise_outcomes"]


def _forward_log_return(price: pd.Series, horizon: int, lag: int = EXECUTION_LAG) -> pd.Series:
    """Log return over h sessions entered ``lag`` sessions after the row's date.

    ``log P(t+lag+h) − log P(t+lag)``, on the price series' own calendar and
    before any alignment, so ``h`` always means h sessions of the target.
    """
    p = pd.to_numeric(price, errors="coerce").astype(float)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    lp = np.log(p.where(p > 0))
    return lp.shift(-(int(lag) + int(horizon))) - lp.shift(-int(lag))


def _datetime_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


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
    lag: int = EXECUTION_LAG,
) -> dict:
    """Out-of-sample directional skill of a bullish-positive lean.

    ``lean`` is dated *t* and uses only data available at *t* (every lean the
    system publishes is causal); it is scored against the h-session return
    entered ``lag`` sessions later, ``log P(t+lag+h) − log P(t+lag)``, computed
    on ``price``'s own session calendar. Rows where either is missing
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
    fwd = _forward_log_return(_datetime_index(pd.Series(price)), h, lag)
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


# ════════════════════════════════════════════════════════════════════════
# The call record — the call as it is issued, not the lean behind it
# ════════════════════════════════════════════════════════════════════════


def claim_outcomes(direction: pd.Series, published: pd.Series, price: pd.Series,
                   horizon: int, lag: int = EXECUTION_LAG) -> pd.DataFrame:
    """One row per CLAIM session whose forward window has closed.

    ``direction`` is +1 (buy), −1 (sell) or 0 (no claim) per session;
    ``published`` marks the sessions on which a claim could have been made.
    Columns: ``direction``; ``fwd`` — the h-session log return entered ``lag``
    sessions later; ``y = direction × fwd``; and the benchmark the claim has to
    beat, taken over EVERY published session of the same target: ``base`` — how
    often a position in that direction won on any day — and ``ex = direction ×
    (fwd − mean fwd)``, the return in excess of holding that direction always.

    The benchmark is the point. On an asset that drifts up, a buy claim wins
    more than half the time whatever it is conditioned on, so a raw hit rate or
    a raw mean return grades the drift, not the call.
    """
    d = _datetime_index(pd.to_numeric(pd.Series(direction), errors="coerce").astype(float))
    pub = _datetime_index(pd.Series(published).astype(bool)).reindex(d.index, fill_value=False)
    fwd = _forward_log_return(_datetime_index(pd.Series(price)), horizon, lag).reindex(d.index)
    ok = pub & fwd.notna()
    mu = float(fwd[ok].mean()) if ok.any() else float("nan")
    up = float((fwd[ok] > 0).mean()) if ok.any() else float("nan")
    out = pd.DataFrame({"direction": np.sign(d.fillna(0.0)), "fwd": fwd})
    out = out[ok & (out["direction"] != 0)].copy()
    out["y"] = out["direction"] * out["fwd"]
    out["ex"] = out["direction"] * (out["fwd"] - mu)
    out["base"] = np.where(out["direction"] > 0, up, 1.0 - up)
    return out


def call_outcomes(fvo: pd.Series, price: pd.Series, horizon: int,
                  lag: int = EXECUTION_LAG) -> pd.DataFrame:
    """:func:`claim_outcomes` for the shipped rule: contrarian past the claim threshold."""
    f = pd.to_numeric(pd.Series(fvo), errors="coerce").astype(float)
    return claim_outcomes(f.map(claim_direction), f.notna(), price, horizon, lag)


_EMPTY_SIDE = {"n_claims": 0, "n_independent": 0, "hit": float("nan"), "base": float("nan"),
               "mean_bps": float("nan"), "excess_bps": float("nan"), "se_bps": float("nan"),
               "t": float("nan")}


def _side(out: pd.DataFrame, h: int) -> dict:
    n = int(len(out))
    if n == 0:
        return dict(_EMPTY_SIDE)
    ex = out["ex"].to_numpy(dtype=np.float64)
    n_ind = max(n // h, 1)
    se = float(np.std(ex, ddof=1) / np.sqrt(n_ind)) if n > 1 else float("nan")
    mean_ex = float(np.mean(ex))
    return {"n_claims": n, "n_independent": int(n // h),
            "hit": float((out["y"] > 0).mean() * 100.0),
            "base": float(out["base"].mean() * 100.0),
            "mean_bps": float(out["y"].mean() * 1e4),
            "excess_bps": mean_ex * 1e4, "se_bps": se * 1e4,
            "t": float(mean_ex / se) if se and np.isfinite(se) and se > 0 else float("nan")}


def summarise_outcomes(out: pd.DataFrame, horizon: int, n_blocks: int = 6) -> dict:
    """The record of a set of claim outcomes (see :func:`call_outcomes`).

    ``hit`` is the share of claims that made money and ``base`` what the same
    directions would have hit on any day. ``mean_bps`` is the raw mean signed
    return per claim; ``excess_bps`` ± ``se_bps`` is that return in excess of
    always holding the direction, and ``t`` is the excess over its SE. The SE
    counts one independent observation per h claims, because claims on
    adjacent sessions share almost all of their forward window.
    """
    h = max(1, int(horizon))
    rec = _side(out, h)
    rec["horizon"] = h
    rec["long"] = _side(out[out["direction"] > 0], h)
    rec["short"] = _side(out[out["direction"] < 0], h)
    blocks = []
    if len(out):
        for part in np.array_split(np.arange(len(out)), max(1, int(n_blocks))):
            if len(part):
                b = _side(out.iloc[part], h)
                blocks.append({"start": out.index[part[0]], "end": out.index[part[-1]],
                               "n": b["n_claims"], "hit": b["hit"], "base": b["base"],
                               "excess_bps": b["excess_bps"]})
    rec["blocks"] = blocks
    scored = [b for b in blocks if b["n"] >= MIN_BLOCK_OBS * h]
    rec["positive_blocks"] = int(sum(1 for b in scored if b["excess_bps"] > 0))
    rec["scored_blocks"] = int(len(scored))
    return rec


def call_record(fvo: pd.Series, price: pd.Series, horizon: int,
                lag: int = EXECUTION_LAG, n_blocks: int = 6) -> dict:
    """What the call has been worth, graded exactly as it is issued.

    A claim is a session where ``|FVO|`` is past the claim threshold; it is
    entered at the close ``lag`` sessions later and held ``horizon`` sessions.
    This is the number the verdict is licensed by. :func:`series_edge` grades
    the continuous lean on every row, which is a different question: a lean
    can rank well and its thresholded call still lose to the drift.
    """
    rec = summarise_outcomes(call_outcomes(fvo, price, horizon, lag), horizon, n_blocks)
    rec["lag"] = int(lag)
    return rec
