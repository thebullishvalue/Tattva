"""
Features — what each engine is allowed to see (DESIGN.md §3.2-3.3).

MŪLA sees the target only *relative to its world*: a recursive, exponentially weighted ridge
regression of the target's daily return on the world factors' returns (lags 0 and 1), whose
one-step-ahead residuals measure how far the target has moved beyond or short of what the
world implied, and whose exposures carry the world's current push onto the target.

SWAYAM sees the target's *own* bars and nothing else: trend, reversal, stretch, range, volatility
regime and participation.

Every value at row t uses bars up to and including t. The label (next-open 10-day return) is
built here too, with the date on which it becomes known, so the learner can purge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 10
EXPOSURE_HALFLIFE = 250        # days
EXPOSURE_RIDGE = 0.01          # × mean diagonal of the decayed Gram matrix
EXPOSURE_WARMUP = 120          # rows before exposures are used
LEAK_CORR = 0.95               # |ρ| of daily returns above which a factor is the target itself
LABEL_CLIP = 4.0

MULA_FEATURES = ("gap20", "gap60", "drift60")
SWAYAM_FEATURES = ("tsm20", "tsm60", "tsm250", "rev5", "stretch", "range60", "volreg", "flow20")


def _vol(r: pd.Series, n: int) -> pd.Series:
    return r.rolling(n, min_periods=int(n * 0.8)).std()


# ── the target's own bars → Swayam ─────────────────────────────────────────────────────────────
def swayam_features(bars: pd.DataFrame) -> pd.DataFrame:
    c, h, lo = bars["close"], bars["high"], bars["low"]
    lr = np.log(c)
    r = lr.diff()
    s60 = _vol(r, 60)
    f = pd.DataFrame(index=bars.index)
    for n in (20, 60, 250):
        f[f"tsm{n}"] = (lr - lr.shift(n)) / (s60 * np.sqrt(n))
    f["rev5"] = (lr - lr.shift(5)) / (s60 * np.sqrt(5))
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=16).mean()
    atr = atr.where(atr > 0, c * s60)                  # close-only series: ATR from return vol
    f["stretch"] = (c - c.ewm(span=20, adjust=False).mean()) / atr
    hi60, lo60 = h.rolling(60, min_periods=48).max(), lo.rolling(60, min_periods=48).min()
    f["range60"] = ((c - lo60) / (hi60 - lo60).where(hi60 > lo60) - 0.5)
    f["volreg"] = np.log(_vol(r, 20) / _vol(r, 250))
    v = bars["volume"].fillna(0.0)
    if v.gt(0).mean() > 0.8:
        sv = (np.sign(r) * v).rolling(20, min_periods=16).sum()
        f["flow20"] = sv / v.rolling(20, min_periods=16).sum().where(lambda x: x > 0)
    else:
        f["flow20"] = 0.0                               # no usable volume: participation is silent
    return f.replace([np.inf, -np.inf], np.nan)


# ── the target against its world → Mūla ────────────────────────────────────────────────────────
def world_returns(world_close: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Factor log returns between the target's own consecutive dates (holidays aggregate)."""
    lp = np.log(world_close).ffill()
    lp = lp.reindex(lp.index.union(dates)).ffill().reindex(dates)
    return lp.diff()


def _leak_mask(r: np.ndarray, X0: np.ndarray, dates: pd.DatetimeIndex, own: list[bool]) -> np.ndarray:
    """(T, K) bool: factor k usable at row t. Judged at each month's first row on the trailing
    250 rows (DESIGN.md §3.1) and held for the month; a factor that IS the target is never used."""
    T, K = X0.shape
    ok = np.ones((T, K), bool)
    month = dates.to_period("M")
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    bounds = list(starts) + [T]
    cur = np.ones(K, bool)
    for i, s in enumerate(starts):
        lo = max(0, s - 250)
        if s - lo >= 60:
            y, Xw = r[lo:s], X0[lo:s]
            m = np.isfinite(y)[:, None] & np.isfinite(Xw)
            cur = np.ones(K, bool)
            for k in range(K):
                mk = m[:, k]
                if mk.sum() >= 60:
                    rho = np.corrcoef(y[mk], Xw[mk, k])[0, 1]
                    cur[k] = not (np.isfinite(rho) and abs(rho) > LEAK_CORR)
        ok[s:bounds[i + 1]] = cur
    ok[:, np.asarray(own, bool)] = False
    return ok


def mula_features(bars: pd.DataFrame, world_close: pd.DataFrame, own_ticker: str,
                  factor_tickers: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Returns (features, detail). ``detail`` carries what the Mūla page shows: the implied
    fair-value path, the residual, the latest exposures and each factor's contribution."""
    dates = bars.index
    lr = np.log(bars["close"])
    r = lr.diff().to_numpy()
    W = world_returns(world_close, dates)
    names = list(W.columns)
    X0 = W.to_numpy()
    X1 = np.vstack([np.full((1, X0.shape[1]), np.nan), X0[:-1]])
    own = [factor_tickers.get(n) == own_ticker for n in names]
    usable = _leak_mask(r, X0, dates, own)

    T, K = X0.shape
    P = 2 * K
    decay = 0.5 ** (1.0 / EXPOSURE_HALFLIFE)
    Sxx, Sxy = np.zeros((P, P)), np.zeros(P)
    beta_prev = np.zeros(P)
    betas = np.full((T, P), np.nan)
    resid = np.full(T, np.nan)
    n_obs = 0
    for t in range(T):
        u = np.r_[usable[t], usable[t]]
        x = np.r_[X0[t], X1[t]]
        x = np.where(np.isfinite(x) & u, x, 0.0)
        y = r[t]
        if n_obs >= EXPOSURE_WARMUP and np.isfinite(y):
            resid[t] = y - beta_prev @ x                # priced with yesterday's exposures
        if np.isfinite(y):
            Sxx = decay * Sxx + np.outer(x, x)
            Sxy = decay * Sxy + x * y
            n_obs += 1
            lam = EXPOSURE_RIDGE * max(np.trace(Sxx) / P, 1e-12)
            beta_prev = np.linalg.solve(Sxx + lam * np.eye(P), Sxy)
        betas[t] = beta_prev * u            # an unusable factor carries no exposure

    e = pd.Series(resid, index=dates)
    sd_e = e.ewm(halflife=EXPOSURE_HALFLIFE, min_periods=60).std()
    f = pd.DataFrame(index=dates)
    for n in (20, 60):
        f[f"gap{n}"] = e.rolling(n, min_periods=n).sum() / (sd_e * np.sqrt(n))
    B = betas[:, :K] + betas[:, K:]                     # total exposure (lag 0 + lag 1)
    m60 = W.rolling(60, min_periods=48).mean().fillna(0.0).to_numpy()
    push = pd.Series(np.nansum(B * m60, axis=1), index=dates) * HORIZON
    s60 = _vol(pd.Series(r, index=dates), 60)
    f["drift60"] = (push / (s60 * np.sqrt(HORIZON))).where(np.isfinite(B).all(axis=1)
                                                          & (np.arange(T) >= EXPOSURE_WARMUP))
    detail = {
        "resid": e,
        "fair_value": bars["close"] * np.exp(-e.rolling(60, min_periods=60).sum()),
        "exposure": pd.DataFrame(B, index=dates, columns=names),
        "contribution": pd.DataFrame(B * m60 * HORIZON, index=dates, columns=names),
        "usable": pd.DataFrame(usable, index=dates, columns=names),
    }
    return f.replace([np.inf, -np.inf], np.nan), detail


# ── the label ─────────────────────────────────────────────────────────────────────────────────
def label(bars: pd.DataFrame) -> pd.DataFrame:
    """y = log(open[t+1+H] / open[t+1]) ÷ (σ60·√H), clipped; ``known`` = the date of open[t+1+H];
    ``ret`` = the raw log return (for scoring)."""
    o = np.log(bars["open"])
    ret = o.shift(-(1 + HORIZON)) - o.shift(-1)
    s60 = _vol(np.log(bars["close"]).diff(), 60)
    y = (ret / (s60 * np.sqrt(HORIZON))).clip(-LABEL_CLIP, LABEL_CLIP)
    known = pd.Series(bars.index, index=bars.index).shift(-(1 + HORIZON))
    return pd.DataFrame({"y": y, "ret": ret, "known": known, "sigma": s60})
