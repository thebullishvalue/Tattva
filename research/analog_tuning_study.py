"""
Tattva — comprehensive tuning of the PRECEDENT (analog) engine.

The analog is now the system's primary directional edge (post-purge the Aarambh
model IC ≈ 0). Every analog knob is ported from Arthagati & un-validated for Tattva.
This sweeps them honestly (non-overlapping OOS IC, full + recent-half) across the
universe, at horizons 1 / 10 / 20d.

Cheap: the analog only needs the Aarambh ts_data (one fit per target) — then every
config is a fast re-walk on the same series (no refits).

Dimensions (one-factor-at-a-time around the current defaults):
  • blend weights  W_MAHA / W_TRAJ / W_RECV   (currently .55/.35/.10)
  • TOP_N          analogs averaged           (currently 10)
  • recency        half-life days             (currently 365)
  • features       ablation + new Tattva features (ModelSpread, ExtremeBreadth,
                   SignalBreadth, ConvictionRaw, MomentumLong)
  • aggregation    equal-weight median vs similarity-weighted mean

Run: python3 -u analog_tuning_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from analytics.analogs import _rolling_hurst, mahalanobis_distance_batch
from markers_study import _aarambh_ts, _load

TARGETS = ["Gold", "Silver", "Copper", "Cotton", "Brent Crude", "USD/INR", "Jeera",
           "Nifty 50", "Nifty Bank", "Nifty IT", "Nifty Metal", "S&P 500", "Nasdaq 100"]
HORIZONS = [1, 10, 20]
MOM = {1: 10, 10: 20, 20: 40}
BASE_FEATS = ["Momentum", "RealizedVol", "AvgZ", "NetBreadth", "Hurst"]
NEW_FEATS = ["ModelSpread", "ExtremeBreadth", "SignalBreadth", "ConvictionRaw", "MomentumLong"]
DEF = dict(wm=0.55, wt=0.35, wr=0.10, top_n=10, hl=365.0, sim=False, feats=tuple(BASE_FEATS))

_TS = {}
def _ts(target):
    if target not in _TS:
        try:
            _TS[target] = _aarambh_ts(target)
        except Exception:
            _TS[target] = None
    return _TS[target]


_FS = {}
def _superset(target, mom):
    key = (target, mom)
    if key in _FS:
        return _FS[key]
    ts = _ts(target)
    if ts is None:
        _FS[key] = None; return None
    price = pd.to_numeric(ts["Price"], errors="coerce").to_numpy(float)
    lr = pd.Series(np.log(np.where(price > 0, price, np.nan))).diff()
    def col(name, default=0.0):
        return pd.to_numeric(ts[name], errors="coerce").to_numpy(float) if name in ts.columns \
            else np.full(len(price), default)
    F = {
        "Momentum": lr.rolling(mom, min_periods=mom).sum().to_numpy(),
        "RealizedVol": lr.rolling(mom, min_periods=mom).std().to_numpy(),
        "AvgZ": col("AvgZ"),
        "NetBreadth": col("OversoldBreadth") - col("OverboughtBreadth"),
        "Hurst": _rolling_hurst(price, window=max(60, mom * 3)),
        "ModelSpread": col("ModelSpread"),
        "ExtremeBreadth": col("ExtremeOversold") - col("ExtremeOverbought"),
        "SignalBreadth": col("BuySignalBreadth") - col("SellSignalBreadth"),
        "ConvictionRaw": col("ConvictionRaw"),
        "MomentumLong": lr.rolling(mom * 2, min_periods=mom * 2).sum().to_numpy(),
    }
    # trajectory matrix (detrended unit price path), tw = mom
    n = len(price); tw = mom
    xm = np.arange(tw, dtype=float); xm -= xm.mean(); xv = np.sum(xm ** 2)
    Tn = np.zeros((n, tw))
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xv if xv > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm); nm = np.linalg.norm(d)
        if nm > 1e-12:
            Tn[i] = d / nm
    _FS[key] = (F, Tn, price, pd.to_datetime(ts.index).to_numpy())
    return _FS[key]


def _walk_ic(target, h, cfg):
    sup = _superset(target, MOM[h])
    if sup is None:
        return np.nan, np.nan
    F, Tn, price, dates = sup
    feats = cfg["feats"]
    M = np.column_stack([F[f] for f in feats])
    for j in range(M.shape[1]):
        c = M[:, j]; ok = np.isfinite(c)
        M[~ok, j] = np.median(c[ok]) if ok.any() else 0.0
    n = len(price); tw = MOM[h]
    wm, wt, wr, top_n, hl, sim = cfg["wm"], cfg["wt"], cfg["wr"], cfg["top_n"], cfg["hl"], cfg["sim"]
    preds, reals = [], []
    for t in range(max(250, tw + 30), n - h, h):
        he = t + 1 - h
        if he < 30:
            continue
        Fh = M[:he]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dd = mahalanobis_distance_batch(Fh, M[t], cov)
        dmax = dd.max() if dd.max() > 0 else 1.0
        maha = 1.0 - dd / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0; traj[:tw] = 0.0
        ds = (pd.Timestamp(dates[t]) - pd.to_datetime(dates[:he])).days.to_numpy(float)
        rec = np.exp(-np.log(2) * np.clip(ds, 0, None) / hl); rec /= max(rec.max(), 1e-6)
        sc = wm * maha + wt * traj + wr * rec
        top = np.argpartition(sc, -top_n)[-top_n:]
        valid = [p for p in top if price[p] > 0 and p + h < n]
        if not valid:
            continue
        fr = np.array([(price[p + h] / price[p] - 1) * 100 for p in valid])
        if sim:
            w = np.clip(sc[valid], 0, None)
            pred = float(np.sum(w * fr) / max(np.sum(w), 1e-9))
        else:
            pred = float(np.median(fr))
        if price[t] > 0 and t + h < n:
            preds.append(pred); reals.append((price[t + h] / price[t] - 1) * 100)
    p, r = np.array(preds), np.array(reals)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 12:
        return np.nan, np.nan
    pp, rr = p[m], r[m]
    full = spearmanr(pp, rr)[0]
    half = len(pp) // 2
    recent = spearmanr(pp[half:], rr[half:])[0] if len(pp) - half >= 8 else np.nan
    return full, recent


def agg(cfg, h):
    fs, rs = [], []
    for t in TARGETS:
        f, r = _walk_ic(t, h, cfg)
        if np.isfinite(f):
            fs.append(f)
        if np.isfinite(r):
            rs.append(r)
    return (np.mean(fs) if fs else np.nan), (np.mean(rs) if rs else np.nan)


def _cfg(**kw):
    c = dict(DEF); c.update(kw); return c


def line(label, cfg, horizons, mark=""):
    cells = []
    for h in horizons:
        f, r = agg(cfg, h)
        cells.append(f"{f:+.3f}/{r:+.3f}")
    print(f"  {label:<22} " + "  ".join(f"{c:>14}" for c in cells) + mark, flush=True)


def main():
    _load()
    print(f"Analog tuning · {len(TARGETS)} targets · cells = full_IC / recent_IC", flush=True)

    print("\n### BASELINE (current defaults) — horizons 1 / 10 / 20", flush=True)
    print(f"  {'config':<22} " + "  ".join(f"{('+'+str(h)+'d'):>14}" for h in HORIZONS), flush=True)
    line("current", _cfg(), HORIZONS, "  ←current")

    H2 = [10, 20]
    print(f"\n### BLEND  W_MAHA/W_TRAJ/W_RECV  (horizons 10 / 20)", flush=True)
    print(f"  {'config':<22} " + "  ".join(f"{('+'+str(h)+'d'):>14}" for h in H2), flush=True)
    for wm, wt, wr in [(1, 0, 0), (.7, .2, .1), (.55, .35, .10), (.4, .5, .1), (.5, .25, .25), (.45, .30, .25)]:
        line(f"{wm}/{wt}/{wr}", _cfg(wm=wm, wt=wt, wr=wr), H2,
             "  ←current" if (wm, wt, wr) == (.55, .35, .10) else "")

    print(f"\n### TOP_N  (horizons 10 / 20)", flush=True)
    for tn in [5, 10, 15, 20, 30]:
        line(f"top_n={tn}", _cfg(top_n=tn), H2, "  ←current" if tn == 10 else "")

    print(f"\n### RECENCY half-life days  (horizons 10 / 20)", flush=True)
    for hl in [120, 250, 365, 730]:
        line(f"halflife={hl}", _cfg(hl=hl, wr=0.20), H2, "")   # at wr=0.20 so recency bites
    print("    (tested at W_RECV=0.20 so the half-life actually matters)", flush=True)

    print(f"\n### FEATURES  (horizons 10 / 20)", flush=True)
    line("base5", _cfg(), H2, "  ←current")
    for f in BASE_FEATS:
        line(f"drop {f}", _cfg(feats=tuple(x for x in BASE_FEATS if x != f)), H2)
    for f in NEW_FEATS:
        line(f"+ {f}", _cfg(feats=tuple(BASE_FEATS) + (f,)), H2)
    line("base + all-new", _cfg(feats=tuple(BASE_FEATS) + tuple(NEW_FEATS)), H2)

    print(f"\n### AGGREGATION  (horizons 10 / 20)", flush=True)
    line("median (equal)", _cfg(sim=False), H2, "  ←current")
    line("similarity-weighted", _cfg(sim=True), H2)

    print("\n  Read each cell as full_IC / recent_IC. recent_IC is the honest test of"
          "\n  whether the analog still works in the CURRENT regime.", flush=True)


if __name__ == "__main__":
    main()
