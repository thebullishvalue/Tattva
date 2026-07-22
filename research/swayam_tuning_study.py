"""
Tattva — Nirnay-Swayam GRID tuning study.

Tunes the Swayam self-ensemble grid that self-mode targets use — the
`swayam_lengths` timescale span and the `swayam_roc_frac` ROC fraction on each
InstrumentConfig (defaults `NIRNAY_SWAYAM_LENGTHS` / `NIRNAY_SWAYAM_ROC_FRAC`).
These are the knobs that define HOW MANY and WHICH causal views the ensemble
runs (engines/nirnay_self.default_swayam_members), which the A/B efficacy study
(`nirnay_swayam`) takes as given.

Method: for each self-mode target (the commodity futures — real OHLCV + volume),
build the Swayam breadth aggregate under each candidate grid and score its
NON-OVERLAPPING OOS rank IC of breadth spread (Oversold% − Overbought%) vs the
target's own forward FORECAST_HORIZON-day return. One-factor-at-a-time around the
current defaults: length SPAN (count + spread), then ROC fraction.

Honest caveat: breadth is one dimension of an already-modest convergence signal
and Swayam views are correlated by construction, so big swings are not expected —
the goal is "does the grid matter / is the default span sane", reported per-target
and as a cross-target mean |IC|.

Run: python3 -u research/swayam_tuning_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: repo root on path so `from core...` resolves
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from data.fetcher import fetch_commodity_dataset, fetch_macro_live, fetch_constituent_ohlcv
from engines.nirnay import aggregate_constituent_timeseries
from engines.nirnay_self import build_swayam_frames, default_swayam_members
from data.constituents import get_nirnay_mode
from core.config import (
    ALL_TARGETS, TARGET_CATEGORIES, swayam_macro_columns, get_instrument_config,
    NIRNAY_SWAYAM_LENGTHS, NIRNAY_SWAYAM_ROC_FRAC,
)
from research._per_instrument import (
    per_instrument_reco, merge_overrides, print_overrides_snippet,
)

# Self-mode targets with genuine futures OHLCV + volume (the commodities). FX and
# Jeera are basket-mode; stocks are covered by per_asset_config_study.
SELF_TARGETS = [t for t in TARGET_CATEGORIES.get("Commodities", [])
                if get_nirnay_mode(t) == "self"]
H = 10   # forward horizon for the breadth-IC objective (the fixed FORECAST_HORIZON)

# Candidate grids. LENGTH SPANS vary count + spread around the tuned default
# (10,14,20,28,40); ROC fractions bracket the 0.7 default.
LENGTH_SPANS = {
    "tight-3":     (14, 20, 28),
    "tight-5":     (12, 16, 20, 26, 34),
    "default-5":   (10, 14, 20, 28, 40),          # current default
    "wide-5":      (8, 14, 22, 34, 52),
    "wide-7":      (8, 12, 18, 26, 36, 50, 70),
    "fast-5":      (5, 8, 12, 18, 28),
    "slow-5":      (14, 20, 30, 45, 70),
}
ROC_FRACS = [0.4, 0.55, 0.7, 0.85, 1.0]

_DATA: dict = {}


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


_OHLCV: dict = {}
def _target_ohlcv(target):
    if target not in _OHLCV:
        d = _load()
        tkr = ALL_TARGETS.get(target)
        m = fetch_constituent_ohlcv([tkr], d["start"], d["end"]) or {}
        _OHLCV[target] = m.get(tkr)
    return _OHLCV[target]


def _target_price(target) -> pd.Series:
    d = _load()
    tgt = d["df"][["DATE", target]].dropna().copy()
    tgt["DATE"] = pd.to_datetime(tgt["DATE"])
    return tgt.set_index("DATE")[target].astype(float)


_CACHE: dict = {}
def _breadth_ic(target, lengths, roc_frac) -> float:
    key = (target, tuple(lengths), float(roc_frac))
    if key in _CACHE:
        return _CACHE[key]
    d = _load()
    ohlcv = _target_ohlcv(target)
    if ohlcv is None or ohlcv.empty:
        _CACHE[key] = np.nan
        return np.nan
    cfg = get_instrument_config(target)
    members = default_swayam_members(tuple(lengths), float(roc_frac))
    cols = swayam_macro_columns(target, d["macro_cols"])
    try:
        frames = build_swayam_frames(
            ohlcv, d["macro"], cols, members=members,
            regime_sensitivity=cfg.nirnay_regime_sensitivity, base_weight=cfg.nirnay_base_weight,
            num_vars=cfg.nirnay_mmr_num_vars, oversold=cfg.nirnay_oversold, overbought=cfg.nirnay_overbought,
        )
        daily = aggregate_constituent_timeseries(frames)
    except Exception:
        _CACHE[key] = np.nan
        return np.nan
    if daily.empty:
        _CACHE[key] = np.nan
        return np.nan

    spread = (pd.to_numeric(daily.get("Oversold_Pct", 0), errors="coerce")
              - pd.to_numeric(daily.get("Overbought_Pct", 0), errors="coerce"))
    spread.index = pd.to_datetime(daily.index)
    price = _target_price(target)
    spread = spread.reindex(price.index, method="ffill")
    pr, br = price.to_numpy(), spread.to_numpy()
    n = len(pr)
    p, r = [], []
    for t in range(60, n - H, H):        # skip warmup; non-overlapping stride
        if pr[t] > 0 and np.isfinite(br[t]):
            p.append(br[t]); r.append((pr[t + H] / pr[t] - 1) * 100)
    p, r = np.array(p), np.array(r)
    m = np.isfinite(p) & np.isfinite(r)
    ic = float(spearmanr(p[m], r[m])[0]) if m.sum() >= 12 else np.nan
    _CACHE[key] = ic
    return ic


def _sweep(name, variants, fixed_lengths=None, fixed_roc=None):
    print(f"\n### {name}", flush=True)
    print(f"  {'variant':<12} {'mean|IC|':>9} {'mean IC':>9}   per-target IC", flush=True)
    best = (None, -9.0)
    table: dict = {}       # {config_value: {target: ic}} for the per-instrument reco
    for label, val in variants.items():
        lengths = val if fixed_roc is not None else fixed_lengths
        roc = fixed_roc if fixed_roc is not None else val
        ics = {t: _breadth_ic(t, lengths, roc) for t in SELF_TARGETS}
        # key the table by the ACTUAL config value (tuple for lengths, float for roc)
        key = tuple(lengths) if fixed_roc is not None else roc
        table[key] = ics
        arr = np.array([x for x in ics.values() if np.isfinite(x)])
        if not len(arr):
            continue
        mabs, msign = float(np.mean(np.abs(arr))), float(np.mean(arr))
        tag = "  ←default" if label in ("default-5", 0.7) else ""
        detail = " ".join(f"{t.split()[0][:4]}={ics[t]:+.2f}" for t in SELF_TARGETS if np.isfinite(ics[t]))
        print(f"  {str(label):<12} {mabs:>9.3f} {msign:>+9.3f}   {detail}{tag}", flush=True)
        if mabs > best[1]:
            best = (label, mabs)
    return best, table


def main():
    _load()
    if not SELF_TARGETS:
        print("No self-mode commodity targets found — nothing to tune.")
        return
    print(f"Swayam grid study · self-mode targets: {SELF_TARGETS}", flush=True)
    print(f"Objective: breadth-spread IC vs +{H}d return (non-overlapping). "
          f"Default grid: lengths={NIRNAY_SWAYAM_LENGTHS} roc_frac={NIRNAY_SWAYAM_ROC_FRAC}", flush=True)
    t0 = time.time()

    best_len, table_len = _sweep("LENGTH SPAN (roc_frac=default)", LENGTH_SPANS,
                                 fixed_roc=NIRNAY_SWAYAM_ROC_FRAC)
    best_roc, table_roc = _sweep("ROC FRACTION (lengths=default)",
                                 {f: f for f in ROC_FRACS}, fixed_lengths=NIRNAY_SWAYAM_LENGTHS)

    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
    print("\n" + "=" * 68)
    print("  SWAYAM GRID — best per knob (by mean |breadth IC|, class-level)")
    print("=" * 68)
    if best_len[0] is not None:
        chg = "  (unchanged)" if best_len[0] == "default-5" else "  ← vs default-5"
        print(f"    swayam_lengths   {str(LENGTH_SPANS.get(best_len[0], best_len[0])):<28} |IC| {best_len[1]:.3f}{chg}")
    if best_roc[0] is not None:
        chg = "  (unchanged)" if best_roc[0] == NIRNAY_SWAYAM_ROC_FRAC else f"  ← vs {NIRNAY_SWAYAM_ROC_FRAC}"
        print(f"    swayam_roc_frac  {str(best_roc[0]):<28} |IC| {best_roc[1]:.3f}{chg}")

    # Per-commodity (each self-mode commodity is tuned per instrument).
    overrides: dict = {}
    own = set(SELF_TARGETS)
    merge_overrides(overrides, per_instrument_reco(
        "swayam_lengths", table_len, NIRNAY_SWAYAM_LENGTHS, own))
    merge_overrides(overrides, per_instrument_reco(
        "swayam_roc_frac", table_roc, NIRNAY_SWAYAM_ROC_FRAC, own))
    print_overrides_snippet(overrides)


if __name__ == "__main__":
    main()
