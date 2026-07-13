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

Robustness: results stream to a repo-local CSV under research/.tune_cache/
(resumable — re-run to continue; skips done rows). Aggregate anytime with
  python3 aarambh_tuning_study.py --agg
Force a clean recompute (wipe the cache first) with
  python3 aarambh_tuning_study.py --fresh

Run: python3 -u aarambh_tuning_study.py
"""

from __future__ import annotations

import sys, os, csv, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which can't encode the → · glyphs used
# in status prints below; force UTF-8 so this runs the same on Windows/Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import (
    MIN_DATA_POINTS, TARGET_EXCLUDED_PREDICTORS, COMMODITY_TARGETS, ALL_TARGETS,
)
from data.fetcher import fetch_commodity_dataset
import engines.aarambh as aa
from engines.aarambh import FairValueEngine

# Resumable results cache. Repo-local + OS-independent ON PURPOSE: the old
# "/tmp/…" path resolved to DIFFERENT files under Windows Python (C:\tmp) vs
# git-bash (/tmp), so a post-fix re-run silently resumed a STALE cache and skipped
# every recompute (the 2026-07-08 11:18 report's stale aarambh_full section).
_CACHE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".tune_cache")
RESULTS_CSV = _os.path.join(_CACHE_DIR, "aarambh_tune_results.csv")

# Mechanical leakage guard. |non-overlapping OOS forward-return IC| this large is a
# money-printer, i.e. forward-LABEL leakage — never real skill at these horizons.
# aggregate() flags such rows (⚠LEAK) and refuses to recommend them. This is an
# impossibility check, NOT an assumption about the tuning answer.
_LEAK_IC = 0.35

# Structural degenerate-window guard. A train window smaller than this cannot fit the
# 219-predictor PCA(20) ensemble: after the purge gap only a handful of rows remain
# (see the "n_samples=1/5/10" PCA-reduce warnings), the fit collapses to a near-constant
# train-mean forecast, and its IC is small-sample NOISE — not a leak (stays < _LEAK_IC),
# but not skill either, and it can spuriously "win" a lever. aggregate() flags such
# window-lever rows (⚠small-win) and excludes them from the recommendation. This is a
# structural floor (a 15-row model is incoherent), NOT a tuning opinion; the rows are
# still DISPLAYED so the small windows the study was asked to probe remain visible.
_MIN_SANE_WINDOW = 100

HORIZONS = {10: 20, 20: 40}            # horizon : momentum window

# OFAT base. ens defaults to the FAST (ridge+ols) basket so the non-ensemble levers
# (REFIT/MAX/MIN/PCA/RIDGE_ALPHAS) sweep ~3× faster — their *winner* is decided by
# relative IC holding the ensemble fixed, so the ranking is unchanged. The
# ENSEMBLE_MODELS lever overrides `ens` explicitly and DOES test the real baskets
# (ols+huber, 4-model, elasticnet) — that's the only place the ensemble is the question.
BASE = dict(refit=5, ens=("ridge", "ols"), maxt=750, mint=500, pca=20,
            ralpha=(0.01, 0.1, 1.0, 10.0, 100.0),
            heps=1.35, lookb=(5, 10, 20, 50, 100))

# Each lever → list of (value, full-cfg-override-dict).
# GRID DEPTH NOTE (2026-07-12): grids densified for a finer optimum search. The
# results CSV is keyed (lever, value, target, horizon), so previously-computed
# values are resumed from cache and only the NEW values cost compute. For that
# resume to stay valid the OFAT BASE above must NOT change — if you ever change
# BASE, wipe the cache (--fresh), or every old row silently becomes stale while
# still being skipped as "done".
def _cfgs():
    levers = {}
    levers["REFIT_INTERVAL"] = [(v, {"refit": v})
                                for v in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18,
                                          21, 25, 30, 40, 50, 63)]
    levers["ENSEMBLE_MODELS"] = [(("+".join(e)), {"ens": e}) for e in (
        ("ols",), ("ridge",), ("huber",), ("elasticnet",),
        ("ols", "huber"), ("ridge", "ols"), ("ridge", "huber"),
        ("ridge", "elasticnet"), ("ols", "elasticnet"), ("huber", "elasticnet"),
        ("ridge", "ols", "huber"), ("ols", "huber", "elasticnet"),
        ("ridge", "ols", "elasticnet"), ("ridge", "huber", "elasticnet"),
        ("ridge", "ols", "huber", "elasticnet"))]
    # Window levers (2026-07-13: widened to 10..3000). <100 is the degenerate
    # region (⚠small-win excludes it from recommendations but it is shown so the
    # full curve is visible from a 10-row window up). DATA CEILING: the shared
    # 9-year sample is ~2346 rows, so (a) MAX_TRAIN ≥ ~the sample saturates
    # (2000/2500/3000 read identically — kept to CONFIRM the plateau), and
    # (b) MIN_TRAIN above ~2000 starves the out-of-sample window (no scoreable
    # rows → NaN), so MIN is capped at 2000. Raise both if a longer history is
    # ever fetched.
    levers["MAX_TRAIN_SIZE"] = [(v, {"maxt": v}) for v in (
        10, 15, 20, 30, 50, 75, 100, 150, 200, 252, 350, 500, 625, 750, 875,
        1000, 1250, 1500, 1750, 2000, 2500, 3000)]
    levers["MIN_TRAIN_SIZE"] = [(v, {"mint": v}) for v in (
        10, 15, 20, 30, 50, 75, 100, 150, 200, 252, 350, 500, 625, 750, 875,
        1000, 1250, 1500, 1750, 2000)]
    # PCA capped by min(n_samples, n_features)≈219, so 2..150 spans trivial→rich.
    levers["PCA_COMPONENTS"] = [(v, {"pca": v}) for v in (
        2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150)]
    # RIDGE_ALPHAS only bites with a ridge ensemble → evaluate on (ridge, ols).
    levers["RIDGE_ALPHAS"] = [
        ("ultra-narrow(0.1..10)", {"ens": ("ridge", "ols"), "ralpha": (0.1, 1.0, 10.0)}),
        ("narrow(0.01..10)", {"ens": ("ridge", "ols"), "ralpha": (0.01, 0.1, 1.0, 10.0)}),
        ("default(.01..100)", {"ens": ("ridge", "ols"), "ralpha": (0.01, 0.1, 1.0, 10.0, 100.0)}),
        ("wide(0.01..1k)", {"ens": ("ridge", "ols"), "ralpha": (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)}),
        ("ultra-wide(0.001..10k)", {"ens": ("ridge", "ols"), "ralpha": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)}),
        ("low(0.001..1)", {"ens": ("ridge", "ols"), "ralpha": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)}),
        ("high(1..1k)", {"ens": ("ridge", "ols"), "ralpha": (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)}),
        ("dense(0.01..100·9pt)", {"ens": ("ridge", "ols"), "ralpha": (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)}),
        ("extreme(1e-4..1e5)", {"ens": ("ridge", "ols"), "ralpha": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)}),
        ("dense-low(1e-4..1)", {"ens": ("ridge", "ols"), "ralpha": (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)}),
    ]
    # HUBER_EPSILON only bites with huber in the basket → evaluate on the real
    # skill basket (ols+huber). 1.35 is sklearn's default (95% Gaussian efficiency).
    levers["HUBER_EPSILON"] = [(v, {"ens": ("ols", "huber"), "heps": v})
                               for v in (1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75,
                                         2.0, 2.5, 3.0, 4.0)]
    # LOOKBACK_WINDOWS drives the Z_lb bands → AvgZ/breadth internals (the
    # engine's state features). Tuple variants probe shorter/longer/denser sets.
    levers["LOOKBACK_WINDOWS"] = [
        ("ultra-short(3-10)", {"lookb": (3, 5, 10)}),
        ("short(5-20)",   {"lookb": (5, 10, 20)}),
        ("mid(10-50)",    {"lookb": (10, 20, 50)}),
        ("long(20-100)",  {"lookb": (20, 50, 100)}),
        ("current(5-100)", {"lookb": (5, 10, 20, 50, 100)}),
        ("no-short(10-100)", {"lookb": (10, 20, 50, 100)}),
        ("wide(5-200)",   {"lookb": (5, 10, 20, 50, 100, 200)}),
        ("dense(5-150)",  {"lookb": (5, 10, 20, 30, 50, 75, 100, 150)}),
        ("ultra-wide(5-250)", {"lookb": (5, 10, 20, 50, 100, 150, 200, 250)}),
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
    sig = (cfg["refit"], cfg["ens"], cfg["maxt"], cfg["mint"], cfg["pca"], cfg["ralpha"],
           cfg.get("heps", 1.35), cfg.get("lookb", (5, 10, 20, 50, 100)), target, h)
    if sig in _ICCACHE:
        return _ICCACHE[sig]
    m = _matrix(target, h, mom)
    if m is None:
        _ICCACHE[sig] = (np.nan, 0); return np.nan, 0
    X, y, feats, price = m
    aa.REFIT_INTERVAL = cfg["refit"]; aa.ENSEMBLE_MODELS = cfg["ens"]
    aa.MAX_TRAIN_SIZE = cfg["maxt"]; aa.MIN_TRAIN_SIZE = cfg["mint"]
    aa.RIDGE_ALPHAS = cfg["ralpha"]
    aa.HUBER_EPSILON = cfg.get("heps", 1.35)
    aa.LOOKBACK_WINDOWS = cfg.get("lookb", (5, 10, 20, 50, 100))
    eng = FairValueEngine()
    eng.fit(X, y, feature_names=feats, forward_signal=True,
            n_pca_components=cfg["pca"], purge=h)
    fv = pd.to_numeric(eng.ts_data["FairValue"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(price)
    # Since engines/aarambh.py leaves predictions[:MIN_TRAIN_SIZE] as NaN (no
    # look-ahead-tainted expanding-mean placeholder — see the audit's A3 fix),
    # the first finite FairValue already correctly starts at (>=) MIN_TRAIN_SIZE.
    # The floor at cfg["mint"] here is a belt-and-suspenders guard, not the
    # primary mechanism (previously it was, filtering on fv != 0 as a *proxy*
    # for validity against a warm-up that was silently 0.0-ish, not NaN).
    vp = np.where(np.isfinite(fv))[0]
    start = int(vp[0]) if len(vp) else cfg["mint"]
    p, r = [], []
    for t in range(max(start, cfg["mint"]), n - h, h):
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
    os.makedirs(_CACHE_DIR, exist_ok=True)
    done = _done_keys()
    new_file = not os.path.exists(RESULTS_CSV)
    f = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
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


def _live_cur():
    """Build the '←current' marker map from LIVE config, so the ←current row and the
    '← CHANGE from' flags reflect what the app ACTUALLY runs — not a hardcoded snapshot
    that silently goes stale (as it had: it read 5/750/500 while config was 7/1000/750)."""
    import importlib
    cfg = importlib.import_module("core.config")
    g = lambda name, default: getattr(cfg, name, default)
    cur = {
        "REFIT_INTERVAL":  str(g("REFIT_INTERVAL", BASE["refit"])),
        "ENSEMBLE_MODELS": "+".join(g("ENSEMBLE_MODELS", BASE["ens"])),
        "MAX_TRAIN_SIZE":  str(g("MAX_TRAIN_SIZE", BASE["maxt"])),
        "MIN_TRAIN_SIZE":  str(g("MIN_TRAIN_SIZE", BASE["mint"])),
        # PCA has no config constant — it is a literal at the live engine.fit call site
        # (app.py sets n_pca_components at the engine.fit call site). Reference the study baseline.
        "PCA_COMPONENTS":  str(BASE["pca"]),
        "HUBER_EPSILON":   str(g("HUBER_EPSILON", BASE["heps"])),
    }
    live_lb = tuple(g("LOOKBACK_WINDOWS", BASE["lookb"]))
    cur["LOOKBACK_WINDOWS"] = next(
        (label for label, ov in _cfgs()["LOOKBACK_WINDOWS"] if tuple(ov["lookb"]) == live_lb),
        "current(5-100)")
    # RIDGE_ALPHAS is a labelled grid here; match the live tuple to its label.
    live_ra = tuple(g("RIDGE_ALPHAS", BASE["ralpha"]))
    cur["RIDGE_ALPHAS"] = next(
        (label for label, ov in _cfgs()["RIDGE_ALPHAS"] if tuple(ov.get("ralpha", ())) == live_ra),
        "default(.01..100)")
    return cur


def aggregate():
    d = pd.read_csv(RESULTS_CSV)
    d = d[np.isfinite(d["ic"])]
    cur = _live_cur()
    print("\n" + "=" * 78)
    print("  POST-PURGE AARAMBH TUNING — mean OOS IC (non-overlapping)")
    print(f"  current defaults (live config): REFIT={cur['REFIT_INTERVAL']} · "
          f"ENS={cur['ENSEMBLE_MODELS']} · MAX={cur['MAX_TRAIN_SIZE']} · "
          f"MIN={cur['MIN_TRAIN_SIZE']} · PCA={cur['PCA_COMPONENTS']}")
    print("  NOTE: non-ENSEMBLE levers use a fast ridge+ols base (relative ranking is")
    print("  what matters); ENSEMBLE_MODELS is tested with the real baskets.")
    print("=" * 78)
    recs = {}
    any_leak = False
    any_degen = False
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
            leak = any(np.isfinite(x) and abs(x) > _LEAK_IC for x in (ic10, ic20, comb, cf, ie, us))
            any_leak = any_leak or leak
            # Degenerate window: a MAX/MIN_TRAIN value below the model's fitting floor.
            degen = (lever in ("MAX_TRAIN_SIZE", "MIN_TRAIN_SIZE")
                     and str(v).isdigit() and int(v) < _MIN_SANE_WINDOW)
            any_degen = any_degen or degen
            mark = "  ←current" if str(v) == cur.get(lever, "") else ""
            flag = "  ⚠LEAK" if leak else ("  ⚠small-win" if degen else "")
            print(f"    {str(v):<18} {ic10:>+7.3f} {ic20:>+7.3f} {comb:>+9.3f} "
                  f"{cf:>+9.3f} {ie:>+9.3f} {us:>+7.3f}{mark}{flag}")
            # A ⚠LEAK (leaked) or ⚠small-win (degenerate) row is NEVER recommendable.
            if np.isfinite(comb) and not leak and not degen and comb > best[1]:
                best = (str(v), comb)
        recs[lever] = best
    if any_degen:
        print(f"\n  NOTE: window values < {_MIN_SANE_WINDOW} are ⚠small-win — the post-purge fit "
              "collapses to a near-\n  constant forecast (noise, not skill); shown for reference "
              "but EXCLUDED from the\n  recommendation. Trust only windows large enough to fit the model.")
    if any_leak:
        print("\n" + "!" * 78)
        print(f"  LEAKAGE DETECTED — one or more rows had |IC| > {_LEAK_IC:.2f}, impossible as")
        print("  real forward-return skill at these horizons (the honest edge is ~0). It is")
        print("  forward-LABEL leakage — almost always a too-small MAX/MIN_TRAIN_SIZE starving")
        print("  the walk-forward purge gap. ⚠LEAK rows are EXCLUDED from the recommendations")
        print("  below; fix the engine/window and re-run before trusting anything here.")
        print("!" * 78)
    print("\n" + "=" * 78)
    print("  RECOMMENDED post-purge defaults (best NON-LEAKED row by combined 10d+20d IC):")
    for lever, (v, ic) in recs.items():
        if v is None:
            print(f"    {lever:<18} {'—':<18} (no clean row — every value tripped the leak guard)")
            continue
        chg = "" if str(v) == cur.get(lever, "") else f"   ← CHANGE from {cur.get(lever)}"
        print(f"    {lever:<18} {str(v):<18} (IC {ic:+.3f}){chg}")


if __name__ == "__main__":
    if "--agg" in sys.argv:
        aggregate()
    else:
        if "--fresh" in sys.argv and os.path.exists(RESULTS_CSV):
            os.remove(RESULTS_CSV)
            print(f"--fresh: cleared results cache {RESULTS_CSV}", flush=True)
        run()
