"""
Tattva — Precedent (analog) potency sweep across the WHOLE target universe.

Goal: let real computational evidence pick the two horizons to keep — a SHORT
(Tactical) and a LONG (Positional) — and decide which horizons the analog
precedent is actually reliable at (so we trim the Precedent hold grid to those).

Method (honest, fully causal):
  • Fit the real FairValueEngine once per target (20d/40d) → causal ts_data
    (Price, AvgZ, breadth) that the analog matcher consumes.
  • For each candidate horizon h ∈ {5,10,20,40,60}, build the analog feature
    frame (momentum window = 2h capped) and walk forward, scoring the analog
    median prediction vs realized on NON-OVERLAPPING windows (stride = h) — the
    only honest IC for smooth multi-day forward returns.
  • Aggregate per horizon ACROSS targets: mean/median per-target rank IC (full &
    recent half), fraction of targets with a positive & significant IC, and mean
    independent sample size.

Run: python3 precedent_universe_sweep.py
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
from core.config import MIN_DATA_POINTS, MIN_TRAIN_SIZE, TARGET_EXCLUDED_PREDICTORS, ALL_TARGETS
from data.fetcher import fetch_commodity_dataset
from engines.aarambh import FairValueEngine
from analytics.analogs import _build_feature_frame, mahalanobis_distance_batch, select_analogs_theiler

# SPEED basket for the universe sweep: drop Huber (the dominant cost) → ~8× faster
# per fit. The analog ranking is driven by price-based features + breadth, so it is
# robust to the ensemble choice; this only affects the AvgZ feature marginally.
import engines.aarambh as _aa
_aa.ENSEMBLE_MODELS = ("ridge", "ols")

HORIZONS = [5, 10, 20, 40, 60]
MOM = {5: 10, 10: 20, 20: 40, 40: 60, 60: 90}
FIT_H, FIT_MOM = 20, 40          # single engine fit per target
TOP_N = 10
W_MAHA, W_TRAJ, W_RECV = 0.55, 0.35, 0.10

_DF = {}


def _df():
    if "df" not in _DF:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=365 * 9)
        d, err = fetch_commodity_dataset(start, end)
        if d is None:
            raise SystemExit(err)
        _DF["df"] = d
    return _DF["df"]


def fit_target(target: str):
    df = _df()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    excl = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    feats = [c for c in numeric if c not in excl]
    cols = [target] + feats + ["DATE"]
    data = df[[c for c in cols if c in df.columns]].copy()
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
    mom = ret[feats].rolling(FIT_MOM, min_periods=FIT_MOM).sum()
    fwd = ret[target].rolling(FIT_H, min_periods=FIT_H).sum().shift(-FIT_H)
    valid = mom.notna().all(axis=1).to_numpy()
    data = data.loc[valid].reset_index(drop=True)
    X = mom.loc[valid].to_numpy()
    y = np.nan_to_num(fwd.loc[valid].to_numpy(), nan=0.0)
    price_level = data[target].to_numpy(dtype=np.float64)
    eng = FairValueEngine()
    # price= so FwdChg_*/divergence detection use the real price, not a
    # pseudo-price reconstructed from overlapping forward-return labels
    # (see the audit's A1 finding) — matches the live app pipeline.
    eng.fit(X, y, feature_names=feats, forward_signal=True, n_pca_components=20,
            purge=FIT_H, price=price_level)   # purge gap = horizon → no train-label leakage
    ts = eng.ts_data.copy()
    ts["Price"] = price_level
    ts["Date"] = pd.to_datetime(data["DATE"].values)
    return ts


def analog_ic(ts: pd.DataFrame, h: int):
    """Non-overlapping analog IC at horizon h. Returns (ic, ic_recent, p, n_indep)."""
    mom_window = MOM[h]
    feat, fcols = _build_feature_frame(ts, mom_window)
    F = feat[fcols].to_numpy(dtype=np.float64)
    for j in range(F.shape[1]):
        c = F[:, j]; ok = np.isfinite(c)
        F[~ok, j] = np.median(c[ok]) if ok.any() else 0.0
    price = feat["Price"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(feat["Date"]).to_numpy()
    n = len(F)
    tw = mom_window
    xm = np.arange(tw, dtype=np.float64); xm -= xm.mean()
    xvar = np.sum(xm ** 2)
    Tn = np.zeros((n, tw))
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xvar if xvar > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm)
        nm = np.linalg.norm(d)
        if nm > 1e-12:
            Tn[i] = d / nm
    purge = h
    preds, reals = [], []
    # Start no earlier than MIN_TRAIN_SIZE: the engine's own warm-up rows
    # (see the audit's A3 fix) carry NaN forecast/breadth features, so
    # anything before that is an unfit region, not a genuine as-of point.
    for t in range(max(MIN_TRAIN_SIZE, tw + 30), n - h, h):     # stride = h → non-overlapping
        he = t + 1 - purge
        if he < 30:
            continue
        Fh = F[:he]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dd = mahalanobis_distance_batch(Fh, F[t], cov)
        dmax = dd.max() if dd.max() > 0 else 1.0
        maha = 1.0 - dd / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0
        traj[:tw] = 0.0
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
        preds.append(float(np.median(fa)))
        reals.append((price[t + h] / price[t] - 1) * 100)
    p = np.array(preds); r = np.array(reals)
    m = np.isfinite(p) & np.isfinite(r) & (p != 0)
    p, r = p[m], r[m]
    if len(p) < 12:
        return np.nan, np.nan, np.nan, len(p)
    ic, pv = spearmanr(p, r)
    half = len(p) // 2
    icr = spearmanr(p[half:], r[half:])[0] if len(p) - half >= 8 else np.nan
    return ic, icr, pv, len(p)


def main():
    df = _df()
    targets = [t for t in ALL_TARGETS if t in df.columns and df[t].notna().mean() >= 0.5]
    print(f"\nTattva — Precedent universe sweep · {len(targets)} targets · horizons {HORIZONS}")
    agg = {h: {"ic": [], "icr": [], "nsig": 0, "nsign": 0, "nind": [], "ntar": 0} for h in HORIZONS}
    t0 = time.time()
    for k, tgt in enumerate(targets, 1):
        ts = None
        try:
            ts = fit_target(tgt)
        except Exception as e:
            print(f"  [{k}/{len(targets)}] {tgt:<24} FIT FAILED: {e}")
            continue
        if ts is None:
            print(f"  [{k}/{len(targets)}] {tgt:<24} skipped (insufficient rows)")
            continue
        cells = []
        for h in HORIZONS:
            ic, icr, pv, nind = analog_ic(ts, h)
            if np.isfinite(ic):
                agg[h]["ic"].append(ic); agg[h]["icr"].append(icr if np.isfinite(icr) else np.nan)
                agg[h]["nind"].append(nind); agg[h]["ntar"] += 1
                if ic > 0: agg[h]["nsign"] += 1
                if ic > 0 and pv < 0.05: agg[h]["nsig"] += 1
            cells.append(f"{h}d={ic:+.2f}")
        print(f"  [{k}/{len(targets)}] {tgt:<24} " + " ".join(cells))
    print(f"\n  swept in {time.time()-t0:.0f}s")

    print("\n" + "=" * 80)
    print("  ANALOG POTENCY ACROSS THE UNIVERSE — which horizons to rely on")
    print("=" * 80)
    print(f"  {'horizon':>8} {'meanIC':>8} {'medIC':>8} {'mean_recent':>12} "
          f"{'IC>0':>8} {'sig(p<.05)':>11} {'mean n_ind':>11}")
    print("  " + "-" * 74)
    for h in HORIZONS:
        a = agg[h]
        N = a["ntar"]
        if N == 0:
            continue
        mic = np.nanmean(a["ic"]); med = np.nanmedian(a["ic"]); mr = np.nanmean(a["icr"])
        print(f"  {('+'+str(h)+'d'):>8} {mic:>+8.3f} {med:>+8.3f} {mr:>+12.3f} "
              f"{a['nsign']:>4}/{N:<3} {a['nsig']:>6}/{N:<3} {np.mean(a['nind']):>11.0f}")
    print("  " + "-" * 74)
    print("  meanIC/medIC = avg/median per-target rank IC (full sample)")
    print("  mean_recent  = avg per-target IC on the recent half")
    print("  IC>0 = # targets with positive IC ; sig = # with positive AND p<0.05")


if __name__ == "__main__":
    main()
