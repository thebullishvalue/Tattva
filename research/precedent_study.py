"""
Tattva — Model vs Precedent walk-forward potency study on Jeera.

Question: at each horizon, do the predictions actually predict — and which is more
potent, TATTVA'S OWN model forecast or the analog/PRECEDENT base rate?

For each horizon config we fit the real FairValueEngine at that horizon (so its
forecast AND the residual→AvgZ/breadth state features are native to the lens),
then walk forward and score TWO causal predictors against realized returns:

  • TATTVA  — engine.ts_data["FairValue"]: the model's predicted forward return
              (walk-forward OOS, higher = more bullish).
  • ANALOG  — median forward return of the top-N Mahalanobis analogs (purge =
              horizon, so each analog's outcome is known as of the as-of date).

Horizons studied: 5d (honorary), Tactical 10d, Swing 20d, Positional 90d.

Run: python3 precedent_study.py
"""

from __future__ import annotations

import warnings
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← ◀ and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import MIN_DATA_POINTS, MIN_TRAIN_SIZE, TARGET_EXCLUDED_PREDICTORS
from data.fetcher import fetch_commodity_dataset
from engines.aarambh import FairValueEngine
from analytics.analogs import _build_feature_frame, mahalanobis_distance_batch, select_analogs_theiler

TARGET = "Jeera"
TOP_N = 10
W_MAHA, W_TRAJ, W_RECV = 0.55, 0.35, 0.10

# (label, forecast horizon d, momentum/feature window d)
CONFIGS = [
    ("5d (honorary)",  5,  10),
    ("Tactical 10d",  10,  20),
    ("Swing 20d",     20,  40),
    ("Positional 90d", 90, 90),
]

_DATASET = {}


def _load_dataset():
    if "df" not in _DATASET:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=365 * 9)
        df, err = fetch_commodity_dataset(start, end)
        if df is None:
            raise SystemExit(f"Dataset load failed: {err}")
        _DATASET["df"] = df
    return _DATASET["df"]


def build_engine_ts(fwd_horizon: int, fwd_mom_k: int) -> pd.DataFrame:
    """Replicate app.py's Jeera pipeline for one horizon → ts_data + Price/Date."""
    df = _load_dataset()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {TARGET, *TARGET_EXCLUDED_PREDICTORS.get(TARGET, [])}
    features = [c for c in numeric_cols if c not in excluded]

    cols = [TARGET] + features + ["DATE"]
    data = df[[c for c in cols if c in df.columns]].copy()
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    data = data.dropna(subset=["DATE"]).sort_values("DATE")
    for c in [TARGET] + features:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[[TARGET] + features] = data[[TARGET] + features].ffill()
    # Causal fill only — NO bfill (backfilling leading NaNs injects FUTURE values, a
    # look-ahead). Drop predictors lacking real history so the dropna doesn't collapse
    # the window — matches the live app pipeline.
    _win = min(MIN_DATA_POINTS, len(data)) if len(data) else 0
    features = [c for c in features if _win and data[c].tail(_win).notna().all()]
    data = data.dropna(subset=[TARGET] + features).reset_index(drop=True)
    if len(data) < MIN_DATA_POINTS:
        raise SystemExit(f"Only {len(data)} rows (<{MIN_DATA_POINTS}).")

    lvl = data[[TARGET] + features].astype(float)
    ret = np.log(lvl.where(lvl > 0)).diff().replace([np.inf, -np.inf], np.nan)
    mom = ret[features].rolling(fwd_mom_k, min_periods=fwd_mom_k).sum()
    fwd = ret[TARGET].rolling(fwd_horizon, min_periods=fwd_horizon).sum().shift(-fwd_horizon)
    valid = mom.notna().all(axis=1).to_numpy()
    data = data.loc[valid].reset_index(drop=True)
    X = mom.loc[valid].to_numpy()
    y = np.nan_to_num(fwd.loc[valid].to_numpy(), nan=0.0)

    print(f"  Fitting FairValueEngine (fit {fwd_horizon}d / mom {fwd_mom_k}d) "
          f"on {len(data)} rows × {len(features)} predictors ...")
    price_level = data[TARGET].to_numpy(dtype=np.float64)
    eng = FairValueEngine()
    # price= so FwdChg_*/divergence detection use the real price, not the
    # overlapping-label pseudo-price (audit finding A1).
    eng.fit(X, y, feature_names=features, forward_signal=True, n_pca_components=20,
            purge=fwd_horizon, price=price_level)   # purge gap = horizon → no train-label leakage
    ts = eng.ts_data.copy()
    ts["Price"] = price_level
    ts["Date"] = pd.to_datetime(data["DATE"].values)
    return ts


def walk_forward(ts: pd.DataFrame, h: int, mom_window: int) -> pd.DataFrame:
    """Score Tattva (engine forecast) and Analog (precedent) at a single horizon h."""
    feat, fcols = _build_feature_frame(ts, mom_window)
    F = feat[fcols].to_numpy(dtype=np.float64)
    for j in range(F.shape[1]):
        col = F[:, j]; ok = np.isfinite(col)
        F[~ok, j] = np.median(col[ok]) if ok.any() else 0.0
    price = feat["Price"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(feat["Date"]).to_numpy()
    tattva = pd.to_numeric(ts["FairValue"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(F)

    tw = mom_window
    xm = np.arange(tw, dtype=np.float64); xm -= xm.mean()
    xvar = np.sum(xm ** 2)
    Tn = np.zeros((n, tw), dtype=np.float64)
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xvar if xvar > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm)
        nrm = np.linalg.norm(d)
        if nrm > 1e-12:
            Tn[i] = d / nrm

    purge = h
    # Start where the engine's walk-forward forecast is live (finite) AND there
    # is enough analog history — same date set for both predictors (fair compare).
    # predictions[:MIN_TRAIN_SIZE] is genuinely NaN post-A3-fix, so the first
    # finite FairValue already starts at (>=) MIN_TRAIN_SIZE.
    valid_pred = np.where(np.isfinite(tattva))[0]
    start = int(max(MIN_TRAIN_SIZE, valid_pred[0] if len(valid_pred) else MIN_TRAIN_SIZE))

    rows = []
    for t in range(start, n - h):
        he = t + 1 - purge
        if he < 30:
            continue
        Fh = F[:he]; cur = F[t]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        d = mahalanobis_distance_batch(Fh, cur, cov)
        dmax = d.max() if d.max() > 0 else 1.0
        maha_sim = 1.0 - d / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0
        traj[:tw] = 0.0
        ds = (pd.Timestamp(dates[t]) - pd.to_datetime(dates[:he])).days.to_numpy(dtype=float)
        rec = np.exp(-np.log(2) * np.clip(ds, 0, None) / 365.0) * W_RECV
        rec /= max(rec.max(), 1e-6)
        score = W_MAHA * maha_sim + W_TRAJ * traj + W_RECV * rec
        # Theiler exclusion window (audit finding A5) — see
        # analytics.analogs.select_analogs_theiler's docstring.
        top = select_analogs_theiler(score, TOP_N, max(tw, h, 1))
        fwd_analog = [(price[p + h] / price[p] - 1) * 100 for p in top if price[p] > 0]
        analog_pred = float(np.median(fwd_analog)) if fwd_analog else np.nan
        real = (price[t + h] / price[t] - 1) * 100 if price[t] > 0 else np.nan
        rows.append({"t": t, "tattva": tattva[t], "analog": analog_pred, "real": real})
    return pd.DataFrame(rows)


def _metrics(pred, real, stride: int):
    """Honest, NON-OVERLAPPING metrics: keep one obs every `stride` (=horizon) days
    so adjacent forward windows don't overlap (overlap inflates IC on smooth series).
    """
    m = np.isfinite(pred) & np.isfinite(real) & (pred != 0)
    p_all, r_all = pred[m], real[m]
    if len(p_all) < 20:
        return None
    # Non-overlapping subsample (independent windows)
    p, r = p_all[::stride], r_all[::stride]
    overlap_ic = spearmanr(p_all, r_all)[0]   # inflated, for reference only
    if len(p) < 12:
        return dict(n_indep=len(p), n_overlap=len(p_all), hit=np.nan, base=np.nan,
                    ic=np.nan, pv=np.nan, spread=np.nan, ic_recent=np.nan, overlap_ic=overlap_ic)
    hit = np.mean(np.sign(p) == np.sign(r)) * 100
    base = max(np.mean(r > 0), np.mean(r < 0)) * 100
    ic, pv = spearmanr(p, r)
    up, dn = r[p > 0], r[p < 0]
    spread = (np.mean(up) if len(up) else np.nan) - (np.mean(dn) if len(dn) else np.nan)
    half = len(p) // 2
    ic_recent = spearmanr(p[half:], r[half:])[0] if len(p) - half >= 8 else np.nan
    return dict(n_indep=len(p), n_overlap=len(p_all), hit=hit, base=base, ic=ic,
                pv=pv, spread=spread, ic_recent=ic_recent, overlap_ic=overlap_ic)


def report(res: pd.DataFrame, label: str, h: int):
    print("\n" + "=" * 84)
    print(f"  {label}  ·  horizon +{h}d  ·  {TARGET}  (NON-OVERLAPPING, stride={h}d)")
    print("=" * 84)
    print(f"  {'predictor':<10} {'n_ind':>6} {'hit%':>6} {'base%':>6} {'edge':>6} "
          f"{'IC':>8} {'p':>7} {'IC_rec':>8} {'spread%':>8} {'[overlapIC]':>12}")
    print("  " + "-" * 80)
    out = {}
    for name, col in (("TATTVA", "tattva"), ("ANALOG", "analog")):
        mt = _metrics(res[col].to_numpy(), res["real"].to_numpy(), stride=h)
        out[name] = mt
        if mt:
            print(f"  {name:<10} {mt['n_indep']:>6} {mt['hit']:>6.1f} {mt['base']:>6.1f} "
                  f"{mt['hit']-mt['base']:>+6.1f} {mt['ic']:>+8.3f} {mt['pv']:>7.4f} "
                  f"{mt['ic_recent']:>+8.3f} {mt['spread']:>+8.2f} {mt['overlap_ic']:>+12.3f}")
    return out


def main():
    print(f"\nTattva — Model vs Precedent walk-forward · target={TARGET}")
    all_res = {}
    for label, h, mom in CONFIGS:
        print(f"\n### {label}  (fit {h}d / mom {mom}d)")
        ts = build_engine_ts(h, mom)
        t0 = time.time()
        res = walk_forward(ts, h, mom)
        print(f"  Walk-forward: {len(res)} as-of dates in {time.time()-t0:.1f}s")
        all_res[(label, h)] = report(res, label, h)

    # ── Head-to-head matrix ─────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  TATTVA vs ANALOG — HONEST non-overlapping rank IC  [full | recent]   (winner ◀)")
    print("=" * 84)
    print(f"  {'horizon':<16} {'n_ind':>6} {'TATTVA (model)':>22} {'ANALOG (precedent)':>22}")
    print("  " + "-" * 74)
    for (label, h), m in all_res.items():
        tv, an = m.get("TATTVA"), m.get("ANALOG")
        def fmt(x): return f"{x['ic']:+.3f} | {x['ic_recent']:+.3f}" if x and np.isfinite(x['ic']) else "   —"
        n_ind = tv["n_indep"] if tv else 0
        win = ""
        if tv and an and np.isfinite(tv["ic"]) and np.isfinite(an["ic"]):
            win = "  ◀ TATTVA" if tv["ic"] >= an["ic"] else "  ◀ ANALOG"
        print(f"  {label:<16} {n_ind:>6} {fmt(tv):>22} {fmt(an):>22}{win}")
    print("  " + "-" * 74)
    print("  IC on independent (non-overlapping) windows — n_ind = # independent obs.")
    print("  At +90d only ~20 independent windows exist in 9y → treat as low-confidence.")


if __name__ == "__main__":
    main()
