"""
The Tattva pipeline for one target: world + own bars → Mūla and Swayam → convergence → the call.

MŪLA (world-relative). The target's 60-day residual against what its world implied (the
exposure regression in ``features.mula_features``). Overshooting the world reverts:
    M = −gap60, scaled by its own trailing RMS.

SWAYAM (self). The target's own overextension: 60- and 250-day trend and its place in the
60-day range. Overextension reverts:
    S = −mean(tsm60, tsm250, range60), each scaled by its own trailing RMS.

CONVERGENCE. Timing F = (M + S) / 2 and its conviction z = F / RMS₂₅₀(F). The call adds timing to
the target's own drift, because a directional call is judged on the absolute move:
    E[y] = μ + K·z     (y = next-open 10-day log return in units of σ₆₀·√10)
μ is the decay-weighted mean of the target's realised y, using only outcomes already known.
BUY / SELL = the sign of E[y], called only when |z| ≥ 0.5 and Mūla and Swayam agree; otherwise
NO CALL.

Why this shape and these features: research/tattva_v3_study.md. It covers the pre-registered
learned design that failed, the diagnosis, and the P0-only selection behind this one. Every
value at a date uses bars up to that date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tattva import features as ft
from tattva.universe import FACTORS, TARGETS

K_TIMING = 0.05        # σ-units of y per unit of z — the composite's P0 cross-target IC (0.042)
CALL_Z = 0.5
RMS_WINDOW = 250
DRIFT_HALFLIFE_DAYS = int(3 * 365.25)
SWAYAM_PARTS = ("tsm60", "tsm250", "range60")


def _rms_scale(s: pd.Series, n: int = RMS_WINDOW) -> pd.Series:
    """s ÷ its trailing RMS (causal; not centred, so the sign is kept)."""
    return s / np.sqrt((s ** 2).rolling(n, min_periods=60).mean())


def _drift(lab: pd.DataFrame) -> pd.Series:
    """The target's own decay-weighted mean y at each date, from outcomes already known then."""
    known = pd.to_datetime(lab["known"])
    kn = pd.Series(lab["y"].to_numpy(), index=known.to_numpy()).dropna().sort_index()
    if kn.empty:
        return pd.Series(np.nan, index=lab.index)
    ew = kn.ewm(halflife=pd.Timedelta(days=DRIFT_HALFLIFE_DAYS), times=kn.index).mean()
    ew = ew[~ew.index.duplicated(keep="last")]
    return ew.reindex(ew.index.union(lab.index)).ffill().reindex(lab.index)


@dataclass
class TargetRun:
    name: str
    ticker: str
    category: str
    bars: pd.DataFrame
    feats: pd.DataFrame
    label: pd.DataFrame
    mula: dict
    out: pd.DataFrame = field(default_factory=pd.DataFrame)

    def current(self) -> dict:
        o = self.out.dropna(subset=["F"])
        if o.empty:
            return {}
        t = o.index[-1]
        r = o.loc[t]
        sig = float(self.label["sigma"].loc[t]) * np.sqrt(ft.HORIZON)
        last_calls = o.index[o["call"] != 0]
        return {
            "date": t, "close": float(self.bars["close"].loc[t]),
            "call": int(r["call"]), "M": float(r["M"]), "S": float(r["S"]), "F": float(r["F"]),
            "z": float(r["z"]), "mu": float(r["mu"]), "Ey": float(r["Ey"]),
            "exp_move": float(r["exp_move"]), "sigma10": sig, "agree": bool(r["agree"]),
            "fair_value": float(self.mula["fair_value"].loc[t]) if pd.notna(self.mula["fair_value"].loc[t]) else None,
            "gap60": float(self.feats["gap60"].loc[t]),
            "parts": {k: float(-_rms_scale(self.feats[k]).loc[t]) for k in SWAYAM_PARTS},
            "last_call_date": last_calls[-1] if len(last_calls) else None,
        }


def analyze(name: str, ticker: str, category: str, bars: pd.DataFrame,
            world: pd.DataFrame) -> TargetRun:
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    mf, detail = ft.mula_features(bars, world, ticker, FACTORS)
    sf = ft.swayam_features(bars)
    feats = pd.concat([mf, sf], axis=1)
    lab = ft.label(bars)
    r = TargetRun(name, ticker, category, bars, feats, lab, detail)

    M = -_rms_scale(feats["gap60"])
    S = -sum(_rms_scale(feats[k]) for k in SWAYAM_PARTS) / len(SWAYAM_PARTS)
    F = (M.fillna(0.0) + S.fillna(0.0)) / 2
    F[M.isna() & S.isna()] = np.nan
    z = _rms_scale(F)
    mu = _drift(lab)
    Ey = mu + K_TIMING * z
    agree = (np.sign(M) == np.sign(S)) & M.notna() & S.notna()
    strong = (z.abs() >= CALL_Z) & agree & Ey.notna()
    call = pd.Series(np.sign(Ey), index=bars.index).where(strong, 0.0).fillna(0.0).astype(int)
    r.out = pd.DataFrame({"M": M, "S": S, "F": F, "z": z, "mu": mu, "Ey": Ey, "agree": agree,
                          "call": call, "exp_move": Ey * lab["sigma"] * np.sqrt(ft.HORIZON)})
    return r


def world_close(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({n: universe[t]["close"] for n, t in FACTORS.items()
                         if t in universe}).sort_index()


def run_catalogue(universe: dict[str, pd.DataFrame]) -> dict[str, TargetRun]:
    """Every catalogue target with enough history — the cross-target track record."""
    world = world_close(universe)
    out = {}
    for t in TARGETS:
        bars = universe.get(t.ticker)
        if bars is not None and len(bars) >= 300:
            out[t.name] = analyze(t.name, t.ticker, t.category, bars, world)
    return out
