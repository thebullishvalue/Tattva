"""
Tattva — DDM SMOOTHING study: are the drift-diffusion filter constants right?

Two DDM parameter sets ship hand-set:
  • CONV_DDM_LEAK_RATE / CONV_DDM_DRIFT_SCALE / CONV_DDM_LONG_RUN_VAR
    (0.10 / 0.12 / 50) — smooths the CONSENSUS (hero TREND row, hero-history
    dashed trend, conviction model).
  • DDM_LEAK_RATE / DDM_DRIFT_SCALE / DDM_LONG_RUN_VAR (0.08 / 0.15 / 100) —
    smooths the engine's ConvictionRaw into ConvictionBounded + CI band.

This sweeps the LEAK (memory) on a grid while CO-SCALING drift to hold the
steady-state GAIN (drift/leak) at each set's shipped ratio — the F3 invariant:
leak controls memory, gain controls response strength; sweeping leak alone
would silently change gain and make magnitudes incomparable across the grid.

Scoring per (target, leak): the smoothed series' NON-OVERLAPPING Spearman IC
vs +10d (the fixed forecast horizon) and +20d (a robustness horizon) forward
return (sign-flipped; negative = bullish), plus two cost metrics — sign
flips/year (whipsaw) and lag (days by which smoothed's cross-correlation with
raw peaks). A smoother filter should cut whipsaw without destroying what little
IC the raw series carries or adding lag beyond the forecast horizon.

Run: python -u research/ddm_smoothing_study.py   (script mode from repo root)
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from core.config import (
    CONV_DDM_LEAK_RATE, CONV_DDM_DRIFT_SCALE, CONV_DDM_LONG_RUN_VAR,
    DDM_LEAK_RATE, DDM_DRIFT_SCALE, DDM_LONG_RUN_VAR,
)
from analytics.ddm_filter import drift_diffusion_filter
from calibration_lift_study import _ts_nd, _consensus_series, TARGETS

# Widened 2026-07-13: 0.01 (very long memory / heavy smoothing) → 0.80 (barely
# filtering). Finer through the 0.05–0.20 plateau where the shipped values sit.
LEAKS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
         0.25, 0.30, 0.40, 0.50, 0.65, 0.80]
HORIZONS = (10, 20)


def _metrics(raw: np.ndarray, price: np.ndarray, leak: float, gain: float,
             lrv: float) -> dict:
    """Smooth `raw` at (leak, drift=gain*leak, lrv); score IC/whipsaw/lag."""
    sm, _, _ = drift_diffusion_filter(raw, leak_rate=leak,
                                      drift_scale=gain * leak, long_run_var=lrv)
    n = len(sm)
    ics = []
    for h in HORIZONS:
        idx = np.arange(0, n - h, h)          # non-overlapping stride = h
        s, p0, p1 = sm[idx], price[idx], price[idx + h]
        m = np.isfinite(s) & np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
        if m.sum() < 12:
            continue
        r = (p1[m] / p0[m] - 1) * 100
        ic = spearmanr(s[m], r)[0]
        if not np.isnan(ic):
            ics.append(-ic)                    # negative = bullish
    ic = float(np.mean(ics)) if ics else float("nan")
    sgn = np.sign(sm[np.isfinite(sm)])
    flips = float(np.sum(sgn[1:] != sgn[:-1])) / max(1, len(sgn)) * 252
    # lag: argmax of cross-corr smoothed vs raw over 0..30 days
    lag = np.nan
    fin = np.isfinite(sm) & np.isfinite(raw)
    if fin.sum() > 100:
        s0, r0 = sm[fin] - np.mean(sm[fin]), raw[fin] - np.mean(raw[fin])
        cc = [np.corrcoef(s0[k:], r0[:len(r0) - k])[0, 1] if k else np.corrcoef(s0, r0)[0, 1]
              for k in range(0, 31)]
        lag = int(np.nanargmax(cc))
    return {"ic": ic, "flips": flips, "lag": lag}


def _sweep(name: str, series_by_tgt: dict[str, tuple[np.ndarray, np.ndarray]],
           cur_leak: float, gain: float, lrv: float) -> None:
    print(f"\n### {name} — gain held at {gain:.2f}, LRV {lrv:g} "
          f"(shipped leak {cur_leak})")
    print(f"  {'leak':>6} {'drift':>7} {'mean IC':>8} {'flips/yr':>9} {'lag(d)':>7}")
    print("  " + "-" * 44)
    best = (None, -9)
    for leak in LEAKS:
        ics, fls, lgs = [], [], []
        for raw, price in series_by_tgt.values():
            m = _metrics(raw, price, leak, gain, lrv)
            if np.isfinite(m["ic"]):
                ics.append(m["ic"])
            fls.append(m["flips"]); lgs.append(m["lag"])
        if not ics:
            continue
        mic, mfl, mlg = np.mean(ics), np.nanmean(fls), np.nanmean(lgs)
        mark = "  ←current" if abs(leak - cur_leak) < 1e-9 else ""
        print(f"  {leak:>6.2f} {gain*leak:>7.3f} {mic:>+8.3f} {mfl:>9.1f} {mlg:>7.1f}{mark}")
        if mic > best[1]:
            best = (leak, mic)
    print(f"  best mean IC: leak={best[0]} ({best[1]:+.3f}) — adopt ONLY if the IC")
    print("  separation is materially outside noise AND lag stays within the horizon;")
    print("  otherwise the shipped leak stands (memory/whipsaw is a product choice).")


def main() -> None:
    from markers_study import _load
    _load()
    print(f"Tattva — DDM smoothing study · {len(TARGETS)} targets · "
          f"leak grid {LEAKS}", flush=True)

    cons_by, conv_by = {}, {}
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        ts, _nd = _ts_nd(tgt)
        if ts is None:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True); continue
        ok = ts.get("Valid")
        tsv = ts[ok.astype(bool)] if ok is not None else ts
        price = pd.to_numeric(tsv["Price"], errors="coerce").to_numpy(float)
        raw_conv = pd.to_numeric(tsv["ConvictionRaw"], errors="coerce").to_numpy(float)
        conv_by[tgt] = (np.nan_to_num(raw_conv, nan=0.0), price)
        cs = _consensus_series(tgt)
        if cs is not None:
            cs = cs.reindex(pd.to_datetime(tsv.index)).to_numpy(float)
            cons_by[tgt] = (np.nan_to_num(cs, nan=0.0), price)
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ready", flush=True)

    print(f"\n  built in {time.time()-t0:.0f}s")
    print("\n" + "=" * 70)
    print("  DDM SMOOTHING — leak sweep at constant gain (the F3 invariant)")
    print("=" * 70)
    _sweep("CONSENSUS smoothing (CONV_DDM_*: hero TREND / conviction model)",
           cons_by, CONV_DDM_LEAK_RATE,
           CONV_DDM_DRIFT_SCALE / CONV_DDM_LEAK_RATE, CONV_DDM_LONG_RUN_VAR)
    _sweep("ENGINE conviction smoothing (DDM_*: ConvictionBounded + CI band)",
           conv_by, DDM_LEAK_RATE,
           DDM_DRIFT_SCALE / DDM_LEAK_RATE, DDM_LONG_RUN_VAR)


if __name__ == "__main__":
    main()
