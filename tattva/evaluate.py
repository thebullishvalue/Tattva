"""
The track record: every forecast Tattva has made, scored once its 10 days are up.

- IC: Spearman correlation of a forecast with the realised next-open 10-day log return, on
  non-overlapping dates (every 10th row), per target per period, averaged across targets.
- Calls: the hit rate and the mean signed 10-day return of BUY/SELL, against two baselines.
  One is the target's drift alone (the call a model with no timing would make); the other is
  always long.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tattva.features import HORIZON

PERIODS = (("P0", "2012-01-01", "2019-07-01"), ("P1", "2019-07-01", "2022-01-01"),
           ("P2", "2022-01-01", "2024-01-01"), ("P3", "2024-01-01", "2100-01-01"))
MIN_N = 15


def scorable(run) -> pd.DataFrame:
    d = run.out[["M", "S", "F", "mu", "call"]].copy()
    d["ret"] = run.label["ret"]
    return d.dropna(subset=["F", "mu", "ret"])


def _period_of(idx: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(idx), None, dtype=object)
    for per, a, b in PERIODS:
        out[(idx >= a) & (idx < b)] = per
    return out


def table(runs: dict, pool_of: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per target × period: ICs of M, S, F and call statistics (Tattva, drift, long)."""
    rows = []
    for name, run in runs.items():
        d = scorable(run)
        for per, a, b in PERIODS:
            s = d[(d.index >= a) & (d.index < b)].iloc[::HORIZON]
            if len(s) < MIN_N:
                continue
            r = {"target": name, "pool": (pool_of or {}).get(name, ""), "period": per, "n": len(s)}
            for k in ("M", "S", "F"):
                x = s[[k, "ret"]].dropna()
                r[k] = spearmanr(x[k], x["ret"]).correlation if len(x) >= MIN_N else np.nan
            for tag, sig in (("tattva", s["call"]), ("drift", np.sign(s["mu"])),
                             ("long", pd.Series(1, index=s.index))):
                c = sig != 0
                r[f"{tag}_calls"] = int(c.sum())
                r[f"{tag}_hits"] = int((np.sign(s["ret"][c]) == sig[c]).sum())
                r[f"{tag}_pnl"] = float((sig[c] * s["ret"][c]).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def summary(T: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    keys = ["period"] if by is None else [by, "period"]
    g = T.groupby(keys)
    out = g[["M", "S", "F"]].mean()
    out["targets"] = g.size()
    for tag in ("tattva", "drift", "long"):
        calls = g[f"{tag}_calls"].sum()
        out[f"{tag}_hit"] = g[f"{tag}_hits"].sum() / calls.where(calls > 0)
        out[f"{tag}_ret%"] = 100 * g[f"{tag}_pnl"].sum() / calls.where(calls > 0)
    out["calls"] = g["tattva_calls"].sum()
    return out


def target_record(run) -> pd.DataFrame:
    """One target's scored calls (every call, overlapping), newest first — the ledger."""
    d = scorable(run)
    c = d[d["call"] != 0].copy()
    c["period"] = _period_of(c.index)
    c["hit"] = np.sign(c["ret"]) == c["call"]
    c["signed_ret"] = c["call"] * c["ret"]
    return c.sort_index(ascending=False)
