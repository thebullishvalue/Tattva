"""
Tattva — data-driven study of the Unified-Signal PLOT MARKERS.

The 3-row Unified Signal plot draws hand-set reference lines / marker-color tiers:
  • Row 1  norm_avg  (consensus, [-1,1])   markers ±0.5 (line), ±0.40/±0.25 (color)
  • Row 2  ConvictionRaw (Aarambh, ~[-100,100])  markers ±40 (line), ±40/±20 (color)
  • Row 3  Avg_Signal (Nirnay, [-10,10])    markers ±2
These are display guide-lines (the *decision* boundary is the Optuna-calibrated
convergence threshold). Question: what do the data say these guide-lines should be?

The three series are exactly what the plot builds (no CrossValidator needed):
  raw_a = Aarambh ts_data["ConvictionRaw"];  raw_n = Nirnay Avg_Signal;
  norm_avg = mean( causal_normalize(raw_a), causal_normalize(raw_n) )   (real
  normalization fn — causal expanding-window z-score, matches the live app;
  see convergence.normalization's module docstring. A previous revision here
  used the terminal-point compute_norm_params/zscore_clip pair, which applies
  the FULL-SAMPLE mean/std to every historical point — a look-ahead: earlier
  bars appear less extreme than they were at the time, because sigma is
  estimated from data that didn't yet exist at that point in history. That
  inflates this study's apparent marker/IC precision (audit finding F14).)

Method: build the 3 series across a representative target set, pool them, and report
(a) the empirical distribution — what percentile the current markers sit at — and
(b) forward-return by signal quintile (does the tier ordering actually hold?). Then
recommend percentile-anchored marker levels (robust; forward-return calibration is
unreliable given these signals' near-zero edge).

Run: python3 -u markers_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← → · and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import (
    MIN_DATA_POINTS, TARGET_EXCLUDED_PREDICTORS,
    UI_CONSENSUS_STRONG, UI_CONSENSUS_MODERATE,
    UI_CONVRAW_STRONG, UI_CONVRAW_MODERATE, UI_NIRNAY_AVG_THRESHOLD,
)
import engines.aarambh as aa
aa.ENSEMBLE_MODELS = ("ridge", "ols")          # fast base for the conviction series
from engines.aarambh import FairValueEngine
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries
from convergence.normalization import align_aarambh_nirnay, causal_normalize
from nirnay_tuning_study import _load, _basket_ohlcv

# Commodity/FX + small India sectors (skip Nifty 50/US — large baskets).
TARGETS = ["Gold", "Copper", "Cotton", "USD/INR", "Jeera", "Nifty Bank", "Nifty IT", "Nifty Auto"]
H, MOM = 10, 20


def _aarambh_ts(target):
    d = _load(); df = d["df"]
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
    mom = ret[feats].rolling(MOM, min_periods=MOM).sum()
    fwd = ret[target].rolling(H, min_periods=H).sum().shift(-H)
    valid = mom.notna().all(axis=1).to_numpy()
    data = data.loc[valid].reset_index(drop=True)
    X = mom.loc[valid].to_numpy(); y = np.nan_to_num(fwd.loc[valid].to_numpy(), nan=0.0)
    price_level = data[target].to_numpy(dtype=np.float64)
    eng = FairValueEngine()
    # price= so FwdChg_*/divergence detection use the real price, not the
    # overlapping-label pseudo-price (audit finding A1).
    eng.fit(X, y, feature_names=feats, forward_signal=True, n_pca_components=20, purge=H, price=price_level)
    ts = eng.ts_data.copy()
    ts["Price"] = price_level
    ts.index = pd.to_datetime(data["DATE"].values)
    return ts


def _nirnay_daily(target):
    d = _load(); ohlcv = _basket_ohlcv(target)
    if not ohlcv:
        return None
    results = {}
    for sym, odf in ohlcv.items():
        merged = odf.copy()
        if not d["macro"].empty:
            merged = merged.join(d["macro"], how="left")
            merged[d["macro_cols"]] = merged[d["macro_cols"]].ffill()
        try:
            res, _ = run_full_analysis(merged, length=20, roc_len=14, regime_sensitivity=1.5,
                                       base_weight=0.6, num_vars=5, macro_columns=d["macro_cols"])
            results[sym] = res
        except Exception:
            pass
    if not results:
        return None
    return aggregate_constituent_timeseries(results)


def target_series(target):
    ts = _aarambh_ts(target)
    nd = _nirnay_daily(target)
    if ts is None or nd is None or nd.empty:
        return None
    dates, raw_a, raw_n = align_aarambh_nirnay(ts, nd)
    if len(dates) < 100:
        return None
    raw_a = np.array(raw_a, float); raw_n = np.array(raw_n, float)
    norm_a = causal_normalize(raw_a)
    norm_n = causal_normalize(raw_n)
    norm_avg = (norm_a + norm_n) / 2.0
    # forward returns of the target aligned to these dates
    price = ts["Price"].astype(float)
    pidx = {d: i for i, d in enumerate(ts.index)}
    pr = price.to_numpy()
    fwd = np.full(len(dates), np.nan)
    for j, d in enumerate(dates):
        i = pidx.get(d)
        if i is not None and i + H < len(pr) and pr[i] > 0:
            fwd[j] = (pr[i + H] / pr[i] - 1) * 100
    return dict(S1=norm_avg, S2=raw_a, S3=raw_n, fwd=fwd)


def _pct_at(absvals, thr):
    """% of observations with |signal| >= thr."""
    return float(np.mean(absvals >= thr) * 100)


def _report_signal(name, vals, fwd, cur_strong, cur_mod, scale):
    a = np.abs(vals)
    finite = np.isfinite(vals)
    a = a[finite]; v = vals[finite]; f = fwd[finite]
    print(f"\n  {name}  (n={len(v)}, scale {scale})")
    print(f"    current markers: strong ±{cur_strong}, moderate ±{cur_mod}")
    print(f"    |signal| ≥ {cur_strong}: {_pct_at(a, cur_strong):5.1f}% of days  "
          f"| ≥ {cur_mod}: {_pct_at(a, cur_mod):5.1f}%")
    qs = {p: np.quantile(a, p / 100) for p in (25, 40, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5)}
    print("    |signal| percentiles:  " + "  ".join(f"p{p:g}={qs[p]:.2f}"
                                                    for p in (25, 40, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5)))
    print(f"    → data-anchored:  strong = p90 ≈ {qs[90]:.2f}   moderate = p75 ≈ {qs[75]:.2f}")
    # forward-return by signal quintile (does tier ordering hold?)
    mf = np.isfinite(f)
    if mf.sum() > 50:
        vv, ff = v[mf], f[mf]
        qcut = pd.qcut(vv, 5, labels=False, duplicates="drop")
        means = [np.mean(ff[qcut == k]) for k in range(int(np.nanmax(qcut)) + 1)]
        ic = spearmanr(vv, ff)[0]
        print(f"    fwd +{H}d by signal quintile (low→high): " +
              " ".join(f"{m:+.2f}%" for m in means) + f"   (IC {ic:+.3f})")


def main():
    _load()
    print(f"Tattva — Unified-Signal marker study · {len(TARGETS)} targets", flush=True)
    S1, S2, S3, FW = [], [], [], []
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        try:
            r = target_series(tgt)
        except Exception as e:
            r = None
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
        if r is None:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True)
            continue
        S1.append(r["S1"]); S2.append(r["S2"]); S3.append(r["S3"]); FW.append(r["fwd"])
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(r['S1'])} days", flush=True)
    if not S1:
        raise SystemExit("no data")
    S1 = np.concatenate(S1); S2 = np.concatenate(S2); S3 = np.concatenate(S3); FW = np.concatenate(FW)
    print(f"\n  pooled {len(S1)} observations in {time.time()-t0:.0f}s")
    print("\n" + "=" * 72)
    print("  UNIFIED-SIGNAL MARKERS — what the data says (pooled across targets)")
    print("=" * 72)
    # "current markers" read LIVE config — hardcoded values here silently misreport
    # once config is re-anchored (they had: 0.5/0.25, 40/20, 2/2 vs live values).
    _report_signal("Row 1 · norm_avg (consensus)", S1, FW,
                   UI_CONSENSUS_STRONG, UI_CONSENSUS_MODERATE, "[-1,1]")
    _report_signal("Row 2 · ConvictionRaw (Aarambh)", S2, FW,
                   UI_CONVRAW_STRONG, UI_CONVRAW_MODERATE, "~[-100,100]")
    _report_signal("Row 3 · Avg_Signal (Nirnay)", S3, FW,
                   UI_NIRNAY_AVG_THRESHOLD, UI_NIRNAY_AVG_THRESHOLD, "[-10,10]")
    print("\n  NOTE: percentiles describe how EXTREME a reading is vs history (robust).")
    print("  Forward-return-by-quintile is the (noisy) edge check — near-zero IC means")
    print("  the markers are interpretive guides, not actionable thresholds.")


if __name__ == "__main__":
    main()
