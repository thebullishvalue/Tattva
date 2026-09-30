"""
The learner shared by both engines (DESIGN.md §3.4): a pooled, purged, hierarchical ridge,
refitted on a fixed monthly calendar.

At each refit date R the model sees only samples whose label was already known on R. The
forecast on any date uses the latest refit on or before it. A global fit across every pool is
the prior each pool's fit is shrunk toward: at 10 days one target has too few independent
outcomes to estimate anything alone, but the sign of a signal still differs by asset class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DECAY_YEARS = 3.0
WARMUP_YEARS = 2.0
RIDGE_PER_FEATURE = 10.0
POOL_SHRINK_PER_FEATURE = 100.0


@dataclass
class Model:
    refit: pd.Timestamp
    std: np.ndarray                      # scale of the within-target (centred) features
    coef: dict[str, np.ndarray]          # pool → standardised coefficients
    n: dict[str, int] = field(default_factory=dict)


def refit_dates(dates: pd.DatetimeIndex, first: pd.Timestamp) -> pd.DatetimeIndex:
    """First weekday of every month from ``first`` through the last date."""
    months = pd.date_range(first.to_period("M").to_timestamp(), dates.max(), freq="MS")
    return pd.DatetimeIndex([m + pd.offsets.BDay(0) for m in months])


def _wridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float,
            prior: np.ndarray | None = None) -> np.ndarray:
    """Weighted ridge on centred data (no intercept), shrunk toward ``prior`` (default 0)."""
    p = X.shape[1]
    b0 = np.zeros(p) if prior is None else prior
    A = (X * w[:, None]).T @ X + lam * np.eye(p)
    return np.linalg.solve(A, (X * w[:, None]).T @ y + lam * b0)


def target_means(feats: pd.DataFrame, cols: list[str], R: pd.Timestamp) -> np.ndarray:
    """A target's own decay-weighted feature mean over its rows dated on or before R — the
    centre its forecasts are read from (features only: known at R, no label needed)."""
    x = feats.loc[feats.index <= R, list(cols)].dropna()
    if x.empty:
        return np.full(len(cols), np.nan)
    w = 0.5 ** (((R - x.index).days.to_numpy() / 365.25) / DECAY_YEARS)
    return (w[:, None] * x.to_numpy(float)).sum(0) / w.sum()


def fit(panel: pd.DataFrame, cols: list[str], R: pd.Timestamp,
        tfeats: dict[str, pd.DataFrame] | None = None) -> Model | None:
    """One refit at R. ``panel`` rows: date, target, pool, known, y and the feature columns.

    WITHIN-TARGET (target fixed effects): features are centred on each target's own mean and
    labels on each target's own mean drift, so the fit learns how a target's return varies with
    ITS OWN state over time — the timing question Tattva answers — and not which targets
    happened to trend more than others (a between-target effect that can carry the opposite
    sign; research/tattva_v3_study.md §2)."""
    tr = panel[(panel["known"] <= R)]
    tr = tr.dropna(subset=cols + ["y"])
    if tr.empty or (R - tr["date"].min()).days < WARMUP_YEARS * 365.25:
        return None
    age = (R - tr["date"]).dt.days.to_numpy() / 365.25
    w = 0.5 ** (age / DECAY_YEARS)
    X = tr[cols].to_numpy(float).copy()
    y = tr["y"].to_numpy(float).copy()
    tgt = tr["target"].to_numpy()
    for t in np.unique(tgt):
        m = tgt == t
        feats_t = (tfeats[t] if tfeats is not None
                   else panel.loc[panel["target"] == t, ["date"] + cols].set_index("date"))
        X[m] -= target_means(feats_t, cols, R)
        y[m] -= (w[m] * y[m]).sum() / w[m].sum()
    std = np.sqrt((w[:, None] * X ** 2).sum(0) / w.sum())
    std = np.where(std > 1e-12, std, 1.0)
    Z = X / std
    p = len(cols)
    bg = _wridge(Z, y, w, RIDGE_PER_FEATURE * p)
    coef, n = {"_global": bg}, {}
    pools = tr["pool"].to_numpy()
    for pool in np.unique(pools):
        m = pools == pool
        coef[pool] = _wridge(Z[m], y[m], w[m], POOL_SHRINK_PER_FEATURE * p, prior=bg)
        n[pool] = int(m.sum())
    return Model(R, std, coef, n)


def fit_all(panel: pd.DataFrame, cols: list[str]) -> list[Model]:
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    first = dates.min() + pd.DateOffset(years=int(WARMUP_YEARS))
    tfeats = {t: g.set_index("date")[cols] for t, g in panel.groupby("target")}
    out = []
    for R in refit_dates(dates, first):
        m = fit(panel, cols, R, tfeats)
        if m is not None:
            out.append(m)
    return out


def predict(models: list[Model], feats: pd.DataFrame, cols: list[str], pool: str) -> pd.Series:
    """Forecast for every row of one target's ``feats`` (DatetimeIndex) with the latest refit on
    or before each date, read from the target's own centre as of that refit. A pool the model
    has not seen uses the global coefficients."""
    out = pd.Series(np.nan, index=feats.index)
    if not models:
        return out
    starts = pd.DatetimeIndex([m.refit for m in models])
    pos = starts.searchsorted(feats.index, side="right") - 1
    X = feats[list(cols)].to_numpy(float)
    vals = np.full(len(feats), np.nan)
    for i in np.unique(pos[pos >= 0]):
        m = models[i]
        rows = pos == i
        key = pool if pool in m.coef else "_global"
        Z = (X[rows] - target_means(feats, cols, m.refit)) / m.std
        vals[rows] = Z @ m.coef[key]
    out[:] = vals
    return out
