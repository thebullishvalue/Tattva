"""
Tattva — THE grade: one measurement of directional skill, for every object.
तत्त्व (Tattva) — "Principle / Essence"

Every skill number Tattva shows or consumes comes from here: the claim's grade
behind trust, each sizer's grade behind its β, the call graded as issued, and
the grade printed beside every lean shown as evidence. One estimator, so the
same IC means the same thing on every page.

THE TARGET. ``Y_s = log P(s+L+h) − log P(s+L)`` on the price's own session
calendar (:func:`forward_return`): h sessions of the target, starting at the
first close a call published after the close of s can trade at (``L``,
``EXECUTION_LAG``). Under ``drift='causal'`` the target is drift-neutral,
``Y_s − μ_s``, where μ_s is the mean of the forward returns already RESOLVED
at s (:func:`causal_drift`). On a rising asset the raw target is higher on
BUY rows than on SELL rows, so anything that reads higher on BUY claims would
earn IC from the trend alone; the drift-neutral target removes that credit.

THE IC. The offset-averaged non-overlapping Spearman correlation
(:func:`_offset_averaged_ic`): every h-th graded row, so no two forward
windows overlap, averaged over the h phase offsets so no data is discarded
and no start day is privileged. A ``mask`` grades a subset (claim rows,
native rows) on the compressed sequence of those rows; a ``sign`` grades a
lean against ``sign · target`` (a sizer against d·Y).

THE SE. A circular-shift null (:func:`shift_null_se`): the spread of the
rank correlation of the graded pairs when the lean is rotated against the
target by every admissible shift, computed at once by FFT. It keeps both
series' own persistence. The textbook 1/√(n−3) it replaces credits neither
that nor the offset average: it overstated the claim's SE by 15–30% and was
off by up to ×2.6 elsewhere. A signed grade's null keeps each row's side with
its row and adds the side-aligned mean that no rotation can see.
``studies/se_calibration.py`` checks the closed form against a brute-force
shift null of the offset-averaged IC, and the SE against the true sampling
spread.

THE AS-OF GRADE. :func:`expanding_edge` evaluates the same grade on a fixed
grid of every h-th row, using at each grid date only pairs whose outcome had
resolved by then: appending data never moves the grid or a past value.

Every constant here is a registered knob (``core.knobs``). No Streamlit
dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from core.knobs import knob

__all__ = [
    "forward_return",
    "causal_drift",
    "shift_null_se",
    "series_edge",
    "expanding_edge",
]

#: Resolved forward returns before the causal drift (and the side-matched base) exist.
DRIFT_MIN_RESOLVED: int = knob("DRIFT_MIN_RESOLVED")
#: Shifts within this many horizons of the true alignment are not null.
GRADE_SHIFT_MARGIN_H: int = knob("GRADE_SHIFT_MARGIN_H")
#: Graded pairs, in horizons, below which the shift-null SE is not estimated.
GRADE_MIN_PAIRS_H: int = knob("GRADE_MIN_PAIRS_H")
#: Non-overlapping windows (pairs per phase offset) below which no IC is computed.
GRADE_MIN_WIN: int = knob("GRADE_MIN_WIN")
#: Independent windows before an as-of grade is reported.
GRADE_WARMUP_WIN: int = knob("GRADE_WARMUP_WIN")
#: Consecutive stretches of history the durability blocks split it into.
GRADE_N_BLOCKS: int = knob("GRADE_N_BLOCKS")
#: Fewer windows than this in a block and its IC is reported as missing.
MIN_BLOCK_OBS: int = knob("MIN_BLOCK_OBS")

Drift = Literal["none", "causal"]
_DRIFTS = ("none", "causal")


def _datetime_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Normalised, de-duplicated (last wins), sorted DatetimeIndex."""
    out = obj.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def forward_return(price: pd.Series, h: int, exec_lag: int = 0) -> pd.Series:
    """``log P(t+L+h) − log P(t+L)``: h SESSIONS forward, from the first tradeable close.

    Computed on the price series' own index, before any alignment, so ``h``
    always means h sessions of the target — never "h rows of whatever frame it
    is later joined to", which is longer wherever that frame has gaps.
    """
    p = pd.to_numeric(pd.Series(price), errors="coerce").astype(float)
    p = p[~p.index.duplicated(keep="last")].sort_index()
    lp = np.log(p.where(p > 0))
    lag = int(exec_lag)
    return lp.shift(-(int(h) + lag)) - lp.shift(-lag)


def causal_drift(Y: pd.Series, h: int, exec_lag: int,
                 min_resolved: int) -> tuple[pd.Series, pd.Series]:
    """``(μ, p_up)``: the expanding mean and up-rate of the RESOLVED forward returns.

    ``Y_u`` is known at row ``u + L + h``, so the value at ``s`` uses exactly
    the ``Y_u`` with ``u + L + h <= s`` and nothing later. Both are NaN until
    ``min_resolved`` returns have resolved.
    """
    y = pd.to_numeric(pd.Series(Y), errors="coerce").astype(float)
    resolved = y.shift(int(h) + int(exec_lag))
    mu = resolved.expanding(min_periods=int(min_resolved)).mean()
    up = ((resolved > 0).astype(float).where(resolved.notna())
          .expanding(min_periods=int(min_resolved)).mean())
    return mu, up


def _rank(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged — ``scipy.stats.rankdata(a)``, without
    its per-call overhead (the as-of grade ranks thousands of short arrays)."""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    srt = a[order]
    new = np.empty(n, dtype=bool)
    new[:1] = True
    np.not_equal(srt[1:], srt[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    out = np.empty(n, dtype=np.float64)
    out[order] = (0.5 * (starts + ends - 1) + 1)[np.cumsum(new) - 1]
    return out


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN on degenerate input."""
    if len(x) < GRADE_MIN_WIN or len(x) != len(y):
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


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
        if len(x) >= GRADE_MIN_WIN and np.nanstd(x) > 0:
            ic = _spearman_ic(x, y)
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _null_parts(x: np.ndarray, y: np.ndarray, h: int,
                sign: np.ndarray | None) -> tuple[float, float] | None:
    """``(rotation SD, side-mean SD)`` of the IC's no-skill spread; None when undefined.

    See :func:`shift_null_se`. The side-mean part is exactly 0 without a sign.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sg = np.ones_like(x) if sign is None else np.asarray(sign, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sg)
    x, y, sg = x[ok], y[ok], sg[ok]
    h = max(1, int(h))
    n = len(x)
    if n < GRADE_MIN_PAIRS_H * h:
        return None
    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return None
    rx = (rx - rx.mean()) / sx
    ry = (ry - ry.mean()) / sy
    side_mean = 0.0
    if sign is not None:
        rx, ry = sg * rx, sg * ry
        a_bar = rx.mean()
        rx, ry = rx - a_bar, ry - ry.mean()
        side_mean = float(abs(a_bar) * np.sqrt(ry.var() / (n // h)))
    # c[s] = (1/n) Σ_i rx[i] · ry[(i + s) mod n]
    c = np.fft.irfft(np.conj(np.fft.rfft(rx)) * np.fft.rfft(ry), n) / n
    s = np.arange(n)
    margin = GRADE_SHIFT_MARGIN_H * h
    return float(np.std(c[(s >= margin) & (s <= n - margin)])), side_mean


def shift_null_se(x: np.ndarray, y: np.ndarray, h: int, *, sign: np.ndarray | None = None) -> float:
    """SE of the offset-averaged IC from a circular-shift null, exactly, by FFT.

    The pairs are rank-standardised once; ``c(s)`` is the circular
    cross-correlation of the two rank series at shift ``s``, for every ``s`` at
    once (``c(0)`` is the full-sample rank correlation itself). A shift of at
    least ``GRADE_SHIFT_MARGIN_H·h`` either way breaks the true pairing while
    keeping each series' own persistence, so the spread of ``c`` over
    ``[5h, N − 5h]`` is the IC's sampling spread under no skill. NaN below
    ``GRADE_MIN_PAIRS_H·h`` pairs or for a constant series.

    ``sign`` is the side a signed grade multiplies its target by (a sizer is
    graded against d·Y). The statistic is then ``mean(a·b)`` with ``a``, ``b``
    the rank series times the side, and its no-skill spread has two parts:

    * the rotation of the CENTRED ``a`` against ``b``, so every rotated product
      keeps the side of its own row — rotating the signed pair as is rotates
      the side out of the products of a lean that carries it (``d · z``) and
      understates the SE (0.73 of the truth on a claim whose side runs ~13
      rows, as on the audited instruments);
    * ``|mean a| · sd(b) / √n_win`` — the part of the lean that only says
      which side the row is on scores the mean of the side-aligned target,
      which every rotation leaves unchanged and no rotation can see. It is
      what grades a factor that is merely higher on BUY rows.

    Simulated over sizers that carry the side, ignore it, lean to one side,
    and only say the side, on two claim processes, the SE is 0.95–1.03 of the
    true sampling SD (the rotation alone: 0.18–1.03; ``studies/se_calibration.py``).
    """
    parts = _null_parts(x, y, h, sign)
    if parts is None:
        return float("nan")
    rotation, side_mean = parts
    return rotation if sign is None else float(np.sqrt(rotation * rotation + side_mean * side_mean))


def _ic_se(x: np.ndarray, y: np.ndarray, sign: np.ndarray | None, h: int) -> tuple[float, float]:
    """The IC of the graded pairs and its shift-null SE (``sign`` None when unsigned)."""
    return _offset_averaged_ic(x, y, h), shift_null_se(x, y, h, sign=sign)


def _aligned(values, like: pd.Series) -> pd.Series:
    """A date-indexed series; a bare array is taken to be aligned with ``like``."""
    if not isinstance(values, pd.Series):
        values = pd.Series(np.asarray(values), index=like.index)
    return _datetime_index(values)


def _graded_frame(lean, price: pd.Series, h: int, *, mask, sign, drift: str,
                  exec_lag: int) -> pd.DataFrame:
    """The graded pairs, date-indexed and in order.

    Columns: ``x`` the lean; ``y`` the graded target (drift-neutral under
    ``'causal'``, times ``sign``); ``Y`` the raw forward return; ``p`` the
    causal up-rate; ``s`` the sign (1 when none); ``pos`` the row of the date
    on the price's own calendar. Only rows where both ``x`` and ``y`` exist.
    ``price`` is already date-indexed.
    """
    if drift not in _DRIFTS:
        raise ValueError(f"drift must be one of {_DRIFTS}, not {drift!r}")
    Y = forward_return(price, h, exec_lag)
    mu, up = causal_drift(Y, h, exec_lag, DRIFT_MIN_RESOLVED)
    target = Y - mu if drift == "causal" else Y

    raw = lean if isinstance(lean, pd.Series) else pd.Series(lean)
    df = pd.DataFrame({"x": _datetime_index(pd.to_numeric(raw, errors="coerce").astype(float))})
    df["y"] = target.reindex(df.index)
    df["Y"] = Y.reindex(df.index)
    df["p"] = up.reindex(df.index)
    df["pos"] = pd.Series(np.arange(len(price)), index=price.index).reindex(df.index)
    if sign is None:
        df["s"] = 1.0
    else:
        # A row with no side has no signed target and is not graded.
        s = np.sign(pd.to_numeric(_aligned(sign, raw), errors="coerce").astype(float))
        df["s"] = s.where(s != 0).reindex(df.index)
    if mask is not None:
        m = _aligned(mask, raw)
        m = m.where(m.notna(), False).astype(bool)
        df = df[m.reindex(df.index, fill_value=False).to_numpy(dtype=bool)]
    df["y"] = df["y"] * df["s"]
    return df.dropna(subset=["x", "y"])


def series_edge(
    lean: pd.Series,
    price: pd.Series,
    horizon: int,
    *,
    mask: pd.Series | None = None,
    sign: pd.Series | None = None,
    drift: Drift = "none",
    exec_lag: int = 0,
    n_blocks: int = GRADE_N_BLOCKS,
) -> dict:
    """Out-of-sample directional skill of a bullish-positive lean.

    ``lean`` is dated *t* and uses only data available at *t* (every lean the
    system publishes is causal); it is scored against the forward return
    :func:`forward_return` computes on ``price``'s own session calendar —
    drift-neutral under ``drift='causal'``, times ``sign`` when given, on the
    ``mask`` rows when given. Rows where either side is missing are dropped.
    Nothing is fitted here, so every number is out-of-sample with respect to
    everything after the date it was published.

    Returns::

        {"ic", "se", "t", "n", "n_win", "n_obs", "horizon", "exec_lag", "drift",
         "se_kind", "se_naive", "hit", "base", "base_majority", "payoff", "span",
         "positive_blocks", "scored_blocks", "n_blocks",
         "blocks": [{"start", "end", "n", "ic"}, ...]}

    * ``ic`` — the offset-averaged non-overlapping Spearman IC
      (:func:`_offset_averaged_ic`) on the graded rows.
    * ``n`` = ``n_win`` — independent windows, ``n_obs // h``; ``se`` — the
      circular-shift null (:func:`shift_null_se`, ``se_kind``); ``t = ic/se``.
      ``se_naive`` — the old ``1/√(n−3)``, kept as a disclosure only.
    * ``hit`` — on rows with a non-zero lean, the share whose implied side
      (``sign(lean · sign)``) matched the sign of the raw forward return, in %.
      ``base`` — the side-matched causal no-skill hit beside it: the mean, over
      the same rows, of the resolved up-rate p_s on BUY rows and 1 − p_s on
      SELL rows. A hit rate is only readable against its own null.
      ``base_majority`` — the old constant majority-direction rate, secondary.
    * ``payoff`` — the mean of ``sign(lean) · target`` on those rows (a log
      return; drift-neutral under ``'causal'``).
    * ``span`` — ``(first, last)`` graded date.
    * ``blocks`` — the same IC within ``n_blocks`` consecutive stretches of
      history: consistently positive is a durable edge, one lucky stretch is
      not. ``positive_blocks`` counts those above zero among the
      ``scored_blocks`` with at least :data:`MIN_BLOCK_OBS` windows.
    """
    h = max(1, int(horizon))
    lag = int(exec_lag)
    nan = float("nan")
    out = {"ic": nan, "se": nan, "t": nan, "n": 0, "n_win": 0, "n_obs": 0, "horizon": h,
           "exec_lag": lag, "drift": drift, "se_kind": "shift-null", "se_naive": nan,
           "hit": nan, "base": nan, "base_majority": nan, "payoff": nan, "span": None,
           "positive_blocks": 0, "scored_blocks": 0, "n_blocks": int(n_blocks), "blocks": []}
    if lean is None or price is None or len(lean) == 0 or len(price) == 0:
        return out

    p = _datetime_index(pd.to_numeric(pd.Series(price), errors="coerce").astype(float))
    df = _graded_frame(lean, p, h, mask=mask, sign=sign, drift=drift, exec_lag=lag)
    n_win = len(df) // h
    out.update(n=int(n_win), n_win=int(n_win), n_obs=int(len(df)),
               span=(df.index[0], df.index[-1]) if len(df) else None)
    if len(df) < GRADE_MIN_WIN * h:
        return out

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    sg = df["s"].to_numpy(dtype=np.float64)
    ic, se = _ic_se(x, y, None if sign is None else sg, h)

    side = np.sign(x * sg)   # the call, in Y's orientation
    called = side != 0
    hit = base = base_majority = payoff = nan
    if called.any():
        sc, Yc, pc = side[called], df["Y"].to_numpy()[called], df["p"].to_numpy()[called]
        hit = float(np.mean(sc == np.sign(Yc)) * 100)
        up = float(np.mean(Yc > 0))
        base_majority = float(max(up, 1 - up) * 100)
        known = np.isfinite(pc)
        if known.any():
            base = float(np.mean(np.where(sc[known] > 0, pc[known], 1 - pc[known])) * 100)
        payoff = float(np.mean(np.sign(x[called]) * y[called]))

    blocks = []
    for part in np.array_split(np.arange(len(df)), max(1, int(n_blocks))):
        if not len(part):
            continue
        nb = len(part) // h
        b_ic = (_offset_averaged_ic(x[part], y[part], h) if nb >= MIN_BLOCK_OBS
                else nan)
        blocks.append({"start": df.index[part[0]], "end": df.index[part[-1]],
                       "n": int(nb), "ic": b_ic})
    scored = [b["ic"] for b in blocks if np.isfinite(b["ic"])]

    out.update(
        ic=ic, se=se, t=float(ic / se) if np.isfinite(ic) and np.isfinite(se) else nan,
        se_naive=float(1 / np.sqrt(max(n_win - 3, 1))),
        hit=hit, base=base, base_majority=base_majority, payoff=payoff,
        positive_blocks=int(sum(1 for v in scored if v > 0)),
        scored_blocks=int(len(scored)), n_blocks=int(len(blocks)), blocks=blocks,
    )
    return out


def expanding_edge(
    lean: pd.Series,
    price: pd.Series,
    horizon: int,
    *,
    mask: pd.Series | None = None,
    sign: pd.Series | None = None,
    drift: Drift = "causal",
    exec_lag: int = 0,
    warmup_win: int = GRADE_WARMUP_WIN,
) -> pd.DataFrame:
    """The as-of grade: :func:`series_edge`'s IC and SE as they stood at each date.

    The grid is every h-th row of ``price`` counted from its FIRST row, so
    appending data never moves it. At grid row ``g`` only the pairs whose
    outcome had resolved by then count (``s + L + h <= g``), and their target
    uses only prices up to ``g``. Each grid value is carried to the dates up to
    the next grid row (``grid_date`` says which), and ``ic`` / ``se`` / ``t``
    are NaN while fewer than ``warmup_win`` independent windows have resolved
    (``n_win`` is reported throughout).

    Returns a frame on ``price``'s (normalised) index with columns
    ``ic, se, t, n_win, grid_date``. Rows up to any date T are identical
    whether the inputs end at T or later.
    """
    h = max(1, int(horizon))
    lag = int(exec_lag)
    cols = ["ic", "se", "t", "n_win", "grid_date"]
    if price is None or len(price) == 0:
        return pd.DataFrame(columns=cols)
    p = _datetime_index(pd.to_numeric(pd.Series(price), errors="coerce").astype(float))
    if lean is None or len(lean) == 0:
        df = pd.DataFrame(columns=["x", "y", "s", "pos"])
    else:
        df = _graded_frame(lean, p, h, mask=mask, sign=sign, drift=drift, exec_lag=lag)
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    sg = df["s"].to_numpy(dtype=np.float64)
    known_at = df["pos"].to_numpy(dtype=np.float64) + lag + h   # the row each outcome resolves

    nan = float("nan")
    rows, last_k, last = [], -1, (nan, nan, nan)
    grid = np.arange(0, len(p), h)
    for g in grid:
        k = int(np.searchsorted(known_at, g, side="right"))
        n_win = k // h
        if k != last_k:
            if n_win >= warmup_win:
                ic, se = _ic_se(x[:k], y[:k], None if sign is None else sg[:k], h)
                last = (ic, se, ic / se if np.isfinite(ic) and np.isfinite(se) else nan)
            else:
                last = (nan, nan, nan)
            last_k = k
        rows.append((*last, n_win, p.index[g]))
    at_grid = pd.DataFrame(rows, columns=cols, index=p.index[grid])
    return at_grid.reindex(p.index, method="ffill")
