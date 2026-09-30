"""
The Tattva pipeline: universe → features → Mūla and Swayam forecasts → convergence → the call.

``run(universe)`` computes every catalogue target at once (the learner is pooled, so a target's
forecast depends on its pool). ``score_symbol(result, bars)`` applies the fitted equity models to
a free-form stock without refitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tattva import features as ft
import learner_v3_rejected as lr  # the pre-registered learner (rejected; see the study)
from tattva.universe import FACTORS, TARGETS

CALL_Z = 0.5
Z_WINDOW = 250
STACK_PRIOR_N = 2000        # pseudo-samples behind the ½/½ prior
STACK_PRIOR_SKILL = 0.05    # the prior's assumed correlation for either engine
STACK_MIN_YEARS = 1.0


@dataclass
class TargetRun:
    name: str
    pool: str
    bars: pd.DataFrame
    feats: pd.DataFrame           # Mūla + Swayam features
    label: pd.DataFrame
    mula: dict                    # Mūla detail (fair value, exposures, contributions)
    out: pd.DataFrame | None = None


def world_close(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {n: universe[t]["close"] for n, t in FACTORS.items() if t in universe}
    return pd.DataFrame(cols).sort_index()


def build_target(name: str, pool: str, ticker: str, bars: pd.DataFrame,
                 world: pd.DataFrame) -> TargetRun:
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    mf, detail = ft.mula_features(bars, world, ticker, FACTORS)
    sf = ft.swayam_features(bars)
    return TargetRun(name, pool, bars, pd.concat([mf, sf], axis=1), ft.label(bars), detail)


def _panel(runs: list[TargetRun]) -> pd.DataFrame:
    parts = []
    for r in runs:
        d = pd.concat([r.feats, r.label[["y", "known"]]], axis=1)
        d["date"], d["target"], d["pool"] = d.index, r.name, r.pool
        parts.append(d.reset_index(drop=True))
    p = pd.concat(parts, ignore_index=True)
    p["known"] = pd.to_datetime(p["known"])
    return p


def _stack_weights(oos: pd.DataFrame, refits: pd.DatetimeIndex) -> dict[pd.Timestamp, dict]:
    """Per refit, per pool: engine weights from their own past out-of-sample skill (decay-
    weighted correlation with the label), shrunk toward ½/½; no skill → weight 0."""
    out = {}
    for R in refits:
        tr = oos[oos["known"] <= R].dropna(subset=["F_mula", "F_swayam", "y"])
        wr = {}
        if not tr.empty and (R - tr["date"].min()).days >= STACK_MIN_YEARS * 365.25:
            for pool, g in tr.groupby("pool"):
                age = (R - g["date"]).dt.days.to_numpy() / 365.25
                w = 0.5 ** (age / lr.DECAY_YEARS)
                y = g["y"].to_numpy()
                sk = {}
                for k in ("F_mula", "F_swayam"):
                    x = g[k].to_numpy()
                    xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
                    cov = (w * (x - xm) * (y - ym)).sum()
                    den = np.sqrt((w * (x - xm) ** 2).sum() * (w * (y - ym) ** 2).sum())
                    c = cov / den if den > 0 else 0.0
                    n = len(g)
                    sk[k] = max(0.0, (n * c + STACK_PRIOR_N * STACK_PRIOR_SKILL) / (n + STACK_PRIOR_N))
                s = sum(sk.values())
                wr[pool] = {k: (v / s if s > 0 else 0.0) for k, v in sk.items()} | {"n": len(g)}
        out[R] = wr
    return out


def _converge(r: TargetRun, weights: dict, refits: pd.DatetimeIndex) -> pd.DataFrame:
    o = r.out
    pos = refits.searchsorted(o.index, side="right") - 1
    wm = np.full(len(o), 0.5)
    ws = np.full(len(o), 0.5)
    for i in np.unique(pos[pos >= 0]):
        w = weights[refits[i]].get(r.pool)
        if w:
            wm[pos == i], ws[pos == i] = w["F_mula"], w["F_swayam"]
    o["w_mula"], o["w_swayam"] = wm, ws
    o["F"] = wm * o["F_mula"].fillna(0) + ws * o["F_swayam"].fillna(0)
    o.loc[o["F_mula"].isna() & o["F_swayam"].isna(), "F"] = np.nan
    sd = o["F"].rolling(Z_WINDOW, min_periods=60).std()
    o["z"] = o["F"] / sd.where(sd > 0)
    agree = np.sign(o["F_mula"]) == np.sign(o["F_swayam"])
    o["agree"] = agree
    call = np.where(o["z"] >= CALL_Z, 1, np.where(o["z"] <= -CALL_Z, -1, 0))
    o["call"] = np.where(agree, call, 0)
    o.loc[o["z"].isna(), "call"] = 0
    o["exp_move"] = o["F"] * r.label["sigma"] * np.sqrt(ft.HORIZON)       # log-return units
    return o


@dataclass
class Result:
    runs: dict[str, TargetRun]
    models_mula: list
    models_swayam: list
    stack: dict
    refits: pd.DatetimeIndex
    world: pd.DataFrame


def run(universe: dict[str, pd.DataFrame]) -> Result:
    world = world_close(universe)
    runs: dict[str, TargetRun] = {}
    for t in TARGETS:
        bars = universe.get(t.ticker)
        if bars is None or len(bars) < 300:
            continue
        runs[t.name] = build_target(t.name, t.pool, t.ticker, bars, world)
    panel = _panel(list(runs.values()))
    mcols, scols = list(ft.MULA_FEATURES), list(ft.SWAYAM_FEATURES)
    models_m = lr.fit_all(panel, mcols)
    models_s = lr.fit_all(panel, scols)
    for r in runs.values():
        r.out = pd.DataFrame({
            "F_mula": lr.predict(models_m, r.feats, mcols, r.pool),
            "F_swayam": lr.predict(models_s, r.feats, scols, r.pool),
        })
    oos = pd.concat([pd.DataFrame({"date": r.out.index, "pool": r.pool,
                                   "F_mula": r.out["F_mula"].to_numpy(),
                                   "F_swayam": r.out["F_swayam"].to_numpy(),
                                   "y": r.label["y"].to_numpy(),
                                   "known": pd.to_datetime(r.label["known"]).to_numpy()})
                     for r in runs.values()], ignore_index=True)
    refits = pd.DatetimeIndex([m.refit for m in models_m])
    stack = _stack_weights(oos, refits)
    for r in runs.values():
        r.out = _converge(r, stack, refits)
    return Result(runs, models_m, models_s, stack, refits, world)


def score_symbol(res: Result, name: str, ticker: str, bars: pd.DataFrame) -> TargetRun:
    """A free-form stock, scored with the fitted equity models and equity stack weights."""
    r = build_target(name, "equity", ticker, bars, res.world)
    mcols, scols = list(ft.MULA_FEATURES), list(ft.SWAYAM_FEATURES)
    r.out = pd.DataFrame({"F_mula": lr.predict(res.models_mula, r.feats, mcols, "equity"),
                          "F_swayam": lr.predict(res.models_swayam, r.feats, scols, "equity")})
    r.out = _converge(r, res.stack, res.refits)
    return r
