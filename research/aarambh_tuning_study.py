"""
Tattva — POST-PURGE re-tuning of the Aarambh engine defaults (FULL / exhaustive).

The shipped defaults (REFIT=5, ENSEMBLE=ols+huber, MIN/MAX_TRAIN=500/750, PCA=20,
RIDGE_ALPHAS=…) were chosen on the OLD walk-forward that leaked future labels into
training. fit() now purges that overlap, so the optimal defaults may have moved.
This re-runs the choice honestly.

Scope (one-factor-at-a-time around current defaults):
  • REFIT_INTERVAL, ENSEMBLE_MODELS (incl. 4-model + elasticnet re-test),
    MAX_TRAIN_SIZE, MIN_TRAIN_SIZE, PCA, RIDGE_ALPHAS (with a ridge ensemble)
  • Scored at BOTH lenses (10d & 20d) — engine defaults are global, so we pick what
    is robust across both.
  • Across ALL ~33 targets (every asset class).

Metric: mean NON-OVERLAPPING (stride = horizon) OOS Spearman IC of the Aarambh
forward-return forecast (ts_data["FairValue"]) vs realized return.

Robustness: results stream to a CSV (resumable — re-run to continue; skips done
rows). Aggregate anytime with:  python3 aarambh_tuning_study.py --agg

Run: python3 -u aarambh_tuning_study.py
"""

from __future__ import annotations

import sys, os, csv, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import (
    MIN_DATA_POINTS, TARGET_EXCLUDED_PREDICTORS, COMMODITY_TARGETS, ALL_TARGETS,
)
from data.fetcher import fetch_commodity_dataset
import engines.aarambh as aa
from engines.aarambh import FairValueEngine

RESULTS_CSV = "/tmp/aarambh_tune_results.csv"
HORIZONS = {10: 20, 20: 40}            # horizon : momentum window

# OFAT base. ens defaults to the FAST (ridge+ols) basket so the non-ensemble levers
# (REFIT/MAX/MIN/PCA/RIDGE_ALPHAS) sweep ~3× faster — their *winner* is decided by
# relative IC holding the ensemble fixed, so the ranking is unchanged. The
# ENSEMBLE_MODELS lever overrides `ens` explicitly and DOES test the real baskets
# (ols+huber, 4-model, elasticnet) — that's the only place the ensemble is the question.
BASE = dict(refit=5, ens=("ridge", "ols"), maxt=750, mint=500, pca=20,
            ralpha=(0.01, 0.1, 1.0, 10.0, 100.0))

# Each lever → list of (value, full-cfg-override-dict)
def _cfgs():
    levers = {}
    levers["REFIT_INTERVAL"] = [(v, {"refit": v}) for v in (3, 5, 7, 10)]
    levers["ENSEMBLE_MODELS"] = [(("+".join(e)), {"ens": e}) for e in (
        ("ols",), ("ols", "huber"), ("ridge", "ols"),
        ("ridge", "ols", "huber"), ("ols", "huber", "elasticnet"))]
    levers["MAX_TRAIN_SIZE"] = [(v, {"maxt": v}) for v in (500, 750, 1000, 1500)]
    levers["MIN_TRAIN_SIZE"] = [(v, {"mint": v}) for v in (300, 500, 750)]
    levers["PCA_COMPONENTS"] = [(v, {"pca": v}) for v in (10, 20, 30)]
    # RIDGE_ALPHAS only bites with a ridge ensemble → evaluate on (ridge, ols).
    levers["RIDGE_ALPHAS"] = [
        ("narrow(0.1,1,10)", {"ens": ("ridge", "ols"), "ralpha": (0.1, 1.0, 10.0)}),
        ("default(.01..100)", {"ens": ("ridge", "ols"), "ralpha": (0.01, 0.1, 1.0, 10.0, 100.0)}),
        ("wide(1,10,100,1k)", {"ens": ("ridge", "ols"), "ralpha": (1.0, 10.0, 100.0, 1000.0)}),
    ]
    return levers


def _class(t):
    if t in COMMODITY_TARGETS:
        return "Cmdty/FX"
    if t in ("S&P 500", "Nasdaq 100", "Dow Jones"):
        return "US-Eq"
    return "India-Eq"


_DF = {}
def _df():
    if "df" not in _DF:
        end = pd.Timestamp.today(); start = end - pd.Timedelta(days=365 * 9)
        d, err = fetch_commodity_dataset(start, end)
        if d is None:
            raise SystemExit(err)
        _DF["df"] = d
    return _DF["df"]


_MAT = {}
def _matrix(target, h, mom):
    key = (target, h, mom)
    if key in _MAT:
        return _MAT[key]
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
    # look-ahead that biases early-window OOS IC). Drop predictors lacking real history
    # so the dropna below doesn't collapse the window — matches the live app pipeline.
    _win = min(MIN_DATA_POINTS, len(data)) if len(data) else 0
    feats = [c for c in feats if _win and data[c].tail(_win).notna().all()]
    data = data.dropna(subset=[target] + feats).reset_index(drop=True)
    if len(data) < MIN_DATA_POINTS:
        _MAT[key] = None; return None
    lvl = data[[target] + feats].astype(float)
    ret = np.log(lvl.where(lvl > 0)).diff().replace([np.inf, -np.inf], np.nan)
    momx = ret[feats].rolling(mom, min_periods=mom).sum()
    fwd = ret[target].rolling(h, min_periods=h).sum().shift(-h)
    valid = momx.notna().all(axis=1).to_numpy()
    data = data.loc[valid].reset_index(drop=True)
    out = (momx.loc[valid].to_numpy(), np.nan_to_num(fwd.loc[valid].to_numpy(), nan=0.0),
           feats, data[target].to_numpy(dtype=np.float64))
    _MAT[key] = out
    return out


_ICCACHE = {}
def fit_ic(cfg, target, h, mom):
    # Dedupe identical configs across levers (the shared base appears in 5 levers).
    sig = (cfg["refit"], cfg["ens"], cfg["maxt"], cfg["mint"], cfg["pca"], cfg["ralpha"], target, h)
    if sig in _ICCACHE:
        return _ICCACHE[sig]
    m = _matrix(target, h, mom)
    if m is None:
        _ICCACHE[sig] = (np.nan, 0); return np.nan, 0
    X, y, feats, price = m
    aa.REFIT_INTERVAL = cfg["refit"]; aa.ENSEMBLE_MODELS = cfg["ens"]
    aa.MAX_TRAIN_SIZE = cfg["maxt"]; aa.MIN_TRAIN_SIZE = cfg["mint"]
    aa.RIDGE_ALPHAS = cfg["ralpha"]
    eng = FairValueEngine()
    eng.fit(X, y, feature_names=feats, forward_signal=True,
            n_pca_components=cfg["pca"], purge=h)
    fv = pd.to_numeric(eng.ts_data["FairValue"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(price)
    vp = np.where(np.isfinite(fv) & (fv != 0))[0]
    start = int(vp[0]) if len(vp) else cfg["mint"]
    p, r = [], []
    for t in range(max(start, 250), n - h, h):
        if price[t] > 0 and np.isfinite(fv[t]) and fv[t] != 0:
            p.append(fv[t]); r.append((price[t + h] / price[t] - 1) * 100)
    p, r = np.array(p), np.array(r)
    mk = np.isfinite(p) & np.isfinite(r)
    if mk.sum() < 12:
        res = (np.nan, int(mk.sum())); _ICCACHE[sig] = res; return res
    res = (float(spearmanr(p[mk], r[mk])[0]), int(mk.sum())); _ICCACHE[sig] = res
    return res


def _done_keys():
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        d = pd.read_csv(RESULTS_CSV)
        return set(zip(d.lever, d.value.astype(str), d.target, d.horizon.astype(int)))
    except Exception:
        return set()


def run():
    df = _df()
    targets = [t for t in ALL_TARGETS if t in df.columns and df[t].notna().mean() >= 0.5]
    levers = _cfgs()
    done = _done_keys()
    new_file = not os.path.exists(RESULTS_CSV)
    f = open(RESULTS_CSV, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["lever", "value", "target", "class", "horizon", "ic", "n"]); f.flush()

    total = sum(len(vs) for vs in levers.values()) * len(targets) * len(HORIZONS)
    print(f"Aarambh post-purge tuning · {len(targets)} targets · {len(HORIZONS)} lenses · "
          f"{sum(len(v) for v in levers.values())} configs → {total} fits "
          f"({len(done)} already done)", flush=True)
    t0 = time.time(); k = 0
    for lever, vals in levers.items():
        for vlabel, override in vals:
            cfg = dict(BASE); cfg.update(override)
            for tgt in targets:
                for h, mom in HORIZONS.items():
                    k += 1
                    if (lever, str(vlabel), tgt, h) in done:
                        continue
                    try:
                        ic, npts = fit_ic(cfg, tgt, h, mom)
                    except Exception as e:
                        ic, npts = float("nan"), 0
                    w.writerow([lever, vlabel, tgt, _class(tgt), h, f"{ic:.4f}", npts]); f.flush()
            el = time.time() - t0
            print(f"  [{k}/{total}] {lever}={vlabel}  done  ({el:.0f}s elapsed)", flush=True)
    f.close()
    aggregate()


def aggregate():
    d = pd.read_csv(RESULTS_CSV)
    d = d[np.isfinite(d["ic"])]
    print("\n" + "=" * 78)
    print("  POST-PURGE AARAMBH TUNING — mean OOS IC (non-overlapping)")
    print("  current defaults: REFIT=5 · ENS=ols+huber · MAX=750 · MIN=500 · PCA=20")
    print("  NOTE: non-ENSEMBLE levers use a fast ridge+ols base (relative ranking is")
    print("  what matters); ENSEMBLE_MODELS is tested with the real baskets.")
    print("=" * 78)
    cur = {"REFIT_INTERVAL": "5", "ENSEMBLE_MODELS": "ols+huber", "MAX_TRAIN_SIZE": "750",
           "MIN_TRAIN_SIZE": "500", "PCA_COMPONENTS": "20", "RIDGE_ALPHAS": "default(.01..100)"}
    recs = {}
    for lever in d["lever"].unique():
        sub = d[d["lever"] == lever]
        print(f"\n  {lever}")
        print(f"    {'value':<18} {'10d':>7} {'20d':>7} {'combined':>9} "
              f"{'Cmdty/FX':>9} {'India-Eq':>9} {'US-Eq':>7}")
        best = (None, -9)
        for v in sub["value"].unique():
            s = sub[sub["value"].astype(str) == str(v)]
            ic10 = s[s.horizon == 10]["ic"].mean()
            ic20 = s[s.horizon == 20]["ic"].mean()
            comb = np.nanmean([ic10, ic20])
            cf = s[s["class"] == "Cmdty/FX"]["ic"].mean()
            ie = s[s["class"] == "India-Eq"]["ic"].mean()
            us = s[s["class"] == "US-Eq"]["ic"].mean()
            mark = "  ←current" if str(v) == cur.get(lever, "") else ""
            print(f"    {str(v):<18} {ic10:>+7.3f} {ic20:>+7.3f} {comb:>+9.3f} "
                  f"{cf:>+9.3f} {ie:>+9.3f} {us:>+7.3f}{mark}")
            if np.isfinite(comb) and comb > best[1]:
                best = (str(v), comb)
        recs[lever] = best
    print("\n" + "=" * 78)
    print("  RECOMMENDED post-purge defaults (best by combined 10d+20d IC):")
    for lever, (v, ic) in recs.items():
        chg = "" if str(v) == cur.get(lever, "") else f"   ← CHANGE from {cur.get(lever)}"
        print(f"    {lever:<18} {str(v):<18} (IC {ic:+.3f}){chg}")


if __name__ == "__main__":
    if "--agg" in sys.argv:
        aggregate()
    else:
        run()
