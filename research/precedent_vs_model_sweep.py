"""
Tattva — MODEL vs ANALOG across the universe, at the two finalized lenses (10d/20d).

Now that the Aarambh walk-forward is purged (no train-label leakage), this maps
where the honest model forecast beats the precedent base rate — and vice-versa —
per asset class. Refits the engine at EACH horizon (so FairValue forecasts that
horizon) with purge = horizon, then scores both predictors on NON-OVERLAPPING
windows.

Run: python3 -u precedent_vs_model_sweep.py
"""

from __future__ import annotations

import warnings
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← → · and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import (
    MIN_DATA_POINTS, MIN_TRAIN_SIZE, TARGET_EXCLUDED_PREDICTORS, ALL_TARGETS, COMMODITY_TARGETS,
)
from data.fetcher import fetch_commodity_dataset
from engines.aarambh import FairValueEngine
from analytics.analogs import _build_feature_frame, mahalanobis_distance_batch, select_analogs_theiler

# SPEED basket for the sweep (drop Huber → ~8× faster; analog ranking robust).
import engines.aarambh as _aa
_aa.ENSEMBLE_MODELS = ("ridge", "ols")

HORIZONS = [10, 20]
MOM = {10: 20, 20: 40}
TOP_N = 10
W_MAHA, W_TRAJ, W_RECV = 0.55, 0.35, 0.10
_DF = {}


def _df():
    if "df" not in _DF:
        end = pd.Timestamp.today(); start = end - pd.Timedelta(days=365 * 9)
        d, err = fetch_commodity_dataset(start, end)
        if d is None:
            raise SystemExit(err)
        _DF["df"] = d
    return _DF["df"]


def fit_ts(target: str, h: int, mom: int):
    df = _df()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    excl = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    feats = [c for c in numeric if c not in excl]
    data = df[[c for c in [target] + feats + ["DATE"] if c in df.columns]].copy()
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    data = data.dropna(subset=["DATE"]).sort_values("DATE")
    for c in [target] + feats:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[[target] + feats] = data[[target] + feats].ffill()
    # Causal fill only — NO bfill (backfilling leading NaNs injects FUTURE values, a
    # look-ahead). Drop predictors lacking real history so the dropna doesn't collapse
    # the window — matches the live app pipeline.
    _win = min(MIN_DATA_POINTS, len(data)) if len(data) else 0
    feats = [c for c in feats if _win and data[c].tail(_win).notna().all()]
    data = data.dropna(subset=[target] + feats).reset_index(drop=True)
    if len(data) < MIN_DATA_POINTS:
        return None
    lvl = data[[target] + feats].astype(float)
    ret = np.log(lvl.where(lvl > 0)).diff().replace([np.inf, -np.inf], np.nan)
    momx = ret[feats].rolling(mom, min_periods=mom).sum()
    fwd = ret[target].rolling(h, min_periods=h).sum().shift(-h)
    valid = momx.notna().all(axis=1).to_numpy()
    data = data.loc[valid].reset_index(drop=True)
    X = momx.loc[valid].to_numpy()
    y = np.nan_to_num(fwd.loc[valid].to_numpy(), nan=0.0)
    price_level = data[target].to_numpy(dtype=np.float64)
    eng = FairValueEngine()
    # price= so FwdChg_*/divergence detection use the real price, not the
    # overlapping-label pseudo-price (audit finding A1).
    eng.fit(X, y, feature_names=feats, forward_signal=True, n_pca_components=20, purge=h, price=price_level)
    ts = eng.ts_data.copy()
    ts["Price"] = price_level
    ts["Date"] = pd.to_datetime(data["DATE"].values)
    return ts


def both_ic(ts: pd.DataFrame, h: int, mom: int):
    """Non-overlapping IC for TATTVA (FairValue) and ANALOG at horizon h."""
    feat, fcols = _build_feature_frame(ts, mom)
    F = feat[fcols].to_numpy(dtype=np.float64)
    for j in range(F.shape[1]):
        c = F[:, j]; ok = np.isfinite(c)
        F[~ok, j] = np.median(c[ok]) if ok.any() else 0.0
    price = feat["Price"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(feat["Date"]).to_numpy()
    tattva = pd.to_numeric(ts["FairValue"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(F); tw = mom
    xm = np.arange(tw, dtype=np.float64); xm -= xm.mean(); xv = np.sum(xm ** 2)
    Tn = np.zeros((n, tw))
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xv if xv > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm); nm = np.linalg.norm(d)
        if nm > 1e-12:
            Tn[i] = d / nm
    # predictions[:MIN_TRAIN_SIZE] is genuinely NaN post-A3-fix (no more
    # look-ahead-tainted expanding-mean placeholder), so the first finite
    # FairValue already starts at (>=) MIN_TRAIN_SIZE.
    vp = np.where(np.isfinite(tattva))[0]
    start = int(max(MIN_TRAIN_SIZE, vp[0] if len(vp) else MIN_TRAIN_SIZE))
    tv, an, re = [], [], []
    for t in range(start, n - h, h):                  # stride h → non-overlapping
        he = t + 1 - h
        if he < 30:
            continue
        Fh = F[:he]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dd = mahalanobis_distance_batch(Fh, F[t], cov)
        dmax = dd.max() if dd.max() > 0 else 1.0
        maha = 1.0 - dd / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0; traj[:tw] = 0.0
        ds = (pd.Timestamp(dates[t]) - pd.to_datetime(dates[:he])).days.to_numpy(dtype=float)
        rec = np.exp(-np.log(2) * np.clip(ds, 0, None) / 365.0) * W_RECV
        rec /= max(rec.max(), 1e-6)
        score = W_MAHA * maha + W_TRAJ * traj + W_RECV * rec
        # Theiler exclusion window (audit finding A5) — see
        # analytics.analogs.select_analogs_theiler's docstring.
        top = select_analogs_theiler(score, TOP_N, max(tw, h, 1))
        fa = [(price[p + h] / price[p] - 1) * 100 for p in top if price[p] > 0]
        if not fa or price[t] <= 0:
            continue
        re.append((price[t + h] / price[t] - 1) * 100)
        an.append(float(np.median(fa)))
        tv.append(tattva[t])

    def ic(pred):
        p = np.array(pred); r = np.array(re)
        m = np.isfinite(p) & np.isfinite(r) & (p != 0)
        p, r = p[m], r[m]
        if len(p) < 12:
            return np.nan, np.nan, len(p)
        full = spearmanr(p, r)[0]
        half = len(p) // 2
        rec_ic = spearmanr(p[half:], r[half:])[0] if len(p) - half >= 8 else np.nan
        return full, rec_ic, len(p)

    tvi, tvr, ntv = ic(tv)
    ani, anr, nan_ = ic(an)
    return dict(tv_ic=tvi, tv_rec=tvr, an_ic=ani, an_rec=anr, n=ntv)


def main():
    df = _df()
    targets = [t for t in ALL_TARGETS if t in df.columns and df[t].notna().mean() >= 0.5]
    cls = {t: ("Commodity/FX" if t in COMMODITY_TARGETS else "Equity Index") for t in targets}
    print(f"\nTattva — MODEL vs ANALOG (purged) · {len(targets)} targets · horizons {HORIZONS}")
    rows = []
    t0 = time.time()
    for k, tgt in enumerate(targets, 1):
        line = f"  [{k:>2}/{len(targets)}] {tgt:<22}"
        for h in HORIZONS:
            try:
                ts = fit_ts(tgt, h, MOM[h])
                r = both_ic(ts, h, MOM[h]) if ts is not None else None
            except Exception as e:
                r = None
            if r and np.isfinite(r["tv_ic"]) and np.isfinite(r["an_ic"]):
                rows.append({"target": tgt, "class": cls[tgt], "h": h, **r})
                line += f"  {h}d[M{r['tv_ic']:+.2f}/A{r['an_ic']:+.2f}]"
            else:
                line += f"  {h}d[—]"
        print(line, flush=True)
    print(f"\n  swept in {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  MODEL (purged Aarambh) vs ANALOG (precedent) — mean rank IC by class")
    print("=" * 80)
    print(f"  {'class':<14} {'h':>4} {'N':>4} {'MODEL ic':>10} {'ANALOG ic':>10} "
          f"{'M_recent':>9} {'A_recent':>9} {'model wins':>11}")
    print("  " + "-" * 76)
    for c in ["Commodity/FX", "Equity Index"]:
        for h in HORIZONS:
            sub = res[(res["class"] == c) & (res["h"] == h)]
            if sub.empty:
                continue
            mwins = int((sub["tv_ic"] > sub["an_ic"]).sum())
            print(f"  {c:<14} {h:>4} {len(sub):>4} {sub['tv_ic'].mean():>+10.3f} "
                  f"{sub['an_ic'].mean():>+10.3f} {sub['tv_rec'].mean():>+9.3f} "
                  f"{sub['an_rec'].mean():>+9.3f} {mwins:>6}/{len(sub):<4}")
    print("  " + "-" * 76)
    print("  Overall (all targets):")
    for h in HORIZONS:
        sub = res[res["h"] == h]
        mwins = int((sub["tv_ic"] > sub["an_ic"]).sum())
        print(f"    +{h}d  MODEL {sub['tv_ic'].mean():+.3f} | ANALOG {sub['an_ic'].mean():+.3f}"
              f"   model beats analog in {mwins}/{len(sub)} targets")


if __name__ == "__main__":
    main()
