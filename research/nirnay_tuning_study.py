"""
Tattva — Nirnay STRUCTURAL knob study (now that the knobs are wired to config).

Question: do the Nirnay structural params (MSF length, ROC length, regime
sensitivity, base weight, MMR num_vars) actually change the predictive content of
the cross-sectional BREADTH signal — and are the current hand-set values reasonable?

Method: for each config (one-factor-at-a-time around the current defaults), run the
REAL Nirnay pipeline per constituent (MSF + MMR + regime loop), aggregate to a daily
breadth oscillator (mean Unified_Osc across the basket), and score its NON-OVERLAPPING
OOS rank IC vs the TARGET's forward 10d return. Run on the 7 commodity/FX targets
(small baskets); equity indices skipped (50+ constituents each → too heavy).

Honest caveat: breadth is ONE dimension of an already-modest convergence signal, so
big swings are not expected — the goal is "do the knobs matter / are defaults sane".

Run: python3 -u nirnay_tuning_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from data.fetcher import fetch_commodity_dataset, fetch_macro_live, fetch_constituent_ohlcv
from data.constituents import get_commodity_basket
from engines.nirnay import run_full_analysis

TARGETS = ["Gold", "Silver", "Copper", "Cotton", "USD/INR", "Brent Crude", "Jeera"]
H = 10                                   # forward horizon (Tactical lens)
BASE = dict(length=20, roc=14, sens=1.5, bw=0.6, nv=5)
# Widened + densified 2026-07-13 (the base cfg is computed once and shared
# across all five sweeps via the _RFA per-constituent cache, so extra grid
# points are cheap). Ranges span from near-degenerate short windows to
# multi-quarter lengths so the full response curve is visible.
SWEEPS = {
    "MSF_LENGTH":         ("length", [3, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]),
    "ROC_LEN":            ("roc",    [2, 3, 5, 7, 9, 10, 12, 14, 17, 21, 25, 28, 35, 45, 60]),
    "REGIME_SENSITIVITY": ("sens",   [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]),
    "BASE_WEIGHT":        ("bw",     [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
    "MMR_NUM_VARS":       ("nv",     [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]),
}

_DATA = {}
def _load():
    if "df" not in _DATA:
        end = pd.Timestamp.today(); start = end - pd.Timedelta(days=365 * 9)
        df, err = fetch_commodity_dataset(start, end)
        if df is None:
            raise SystemExit(err)
        macro = fetch_macro_live(start, end)
        _DATA["df"] = df
        _DATA["macro"] = macro if macro is not None else pd.DataFrame()
        _DATA["macro_cols"] = list(_DATA["macro"].columns)
        _DATA["start"], _DATA["end"] = start, end
    return _DATA


_OHLCV = {}
def _basket_ohlcv(target):
    if target not in _OHLCV:
        d = _load()
        cons, _ = get_commodity_basket(target)
        _OHLCV[target] = fetch_constituent_ohlcv(cons, d["start"], d["end"]) or {}
    return _OHLCV[target]


_RFA = {}
def _constituent_osc(cfg, target, sym, ohlcv_df):
    key = (cfg["length"], cfg["roc"], cfg["sens"], cfg["bw"], cfg["nv"], target, sym)
    if key in _RFA:
        return _RFA[key]
    d = _load()
    merged = ohlcv_df.copy()
    if not d["macro"].empty:
        merged = merged.join(d["macro"], how="left")
        merged[d["macro_cols"]] = merged[d["macro_cols"]].ffill()
    try:
        res, _ = run_full_analysis(
            merged, length=cfg["length"], roc_len=cfg["roc"],
            regime_sensitivity=cfg["sens"], base_weight=cfg["bw"], num_vars=cfg["nv"],
            macro_columns=d["macro_cols"],
        )
        s = pd.to_numeric(res["Unified_Osc"], errors="coerce")
        s.index = pd.to_datetime(res.index)
        _RFA[key] = s
    except Exception:
        _RFA[key] = None
    return _RFA[key]


def breadth_ic(cfg, target):
    d = _load()
    ohlcv = _basket_ohlcv(target)
    if not ohlcv:
        return np.nan
    oscs = []
    for sym, odf in ohlcv.items():
        s = _constituent_osc(cfg, target, sym, odf)
        if s is not None and len(s):
            oscs.append(s)
    if len(oscs) < 3:
        return np.nan
    breadth = pd.concat(oscs, axis=1).mean(axis=1)        # mean osc across basket
    tgt = d["df"][["DATE", target]].dropna().copy()
    tgt["DATE"] = pd.to_datetime(tgt["DATE"])
    price = tgt.set_index("DATE")[target].astype(float)
    breadth = breadth.reindex(price.index, method="ffill")
    pr, br = price.to_numpy(), breadth.to_numpy()
    n = len(pr)
    p, r = [], []
    for t in range(60, n - H, H):                          # skip warmup; non-overlapping
        if pr[t] > 0 and np.isfinite(br[t]):
            p.append(br[t]); r.append((pr[t + H] / pr[t] - 1) * 100)
    p, r = np.array(p), np.array(r)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 12:
        return np.nan
    return float(spearmanr(p[m], r[m])[0])


def main():
    _load()
    print(f"Nirnay structural study · {len(TARGETS)} commodity/FX targets · "
          f"breadth IC vs +{H}d return (non-overlapping)", flush=True)
    recs = {}
    t0 = time.time()
    for lever, (field, vals) in SWEEPS.items():
        print(f"\n### {lever}  (base: len20/roc14/sens1.5/bw0.6/nv5)", flush=True)
        print(f"  {'value':<8} {'mean|IC|':>9} {'mean IC':>9}   per-target IC", flush=True)
        best = (None, -9)
        for v in vals:
            cfg = dict(BASE); cfg[field] = v
            ics = {t: breadth_ic(cfg, t) for t in TARGETS}
            arr = np.array([x for x in ics.values() if np.isfinite(x)])
            if not len(arr):
                continue
            mabs, msign = np.mean(np.abs(arr)), np.mean(arr)
            tag = "  ←current" if v == BASE[field] else ""
            detail = " ".join(f"{t.split()[0][:4]}={ics[t]:+.2f}" for t in TARGETS if np.isfinite(ics[t]))
            print(f"  {str(v):<8} {mabs:>9.3f} {msign:>+9.3f}   {detail}{tag}", flush=True)
            if mabs > best[1]:
                best = (v, mabs)
        recs[lever] = best
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
    print("\n" + "=" * 64)
    print("  NIRNAY STRUCTURAL — best per knob (by mean |breadth IC|)")
    print("=" * 64)
    for lever, (v, ic) in recs.items():
        chg = "  (unchanged)" if v == BASE[SWEEPS[lever][0]] else f"  ← vs current {BASE[SWEEPS[lever][0]]}"
        print(f"    {lever:<20} {str(v):<6} |IC| {ic:.3f}{chg}")


if __name__ == "__main__":
    main()
