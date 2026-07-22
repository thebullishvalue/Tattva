"""Confirmatory MAX_TRAIN_SIZE sweep at the LIVE MIN_TRAIN_SIZE.

The main study sweeps MAX_TRAIN at its fixed OFAT base MIN; this re-checks it at
the MIN the app actually runs (read from core.config, not hardcoded — a stale
hardcoded 750 would silently confirm the wrong interaction once config moves) to
rule out a MAX×MIN interaction. Reuses the same fit/IC machinery (fast ridge+ols
base, purged, non-overlapping OOS IC). Grid starts at 100 — the sub-100 windows
are structurally degenerate (see the main study's _MIN_SANE_WINDOW) and add noise,
not information, to a confirmation sweep.
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import ALL_TARGETS, MIN_TRAIN_SIZE, MAX_TRAIN_SIZE
from aarambh_tuning_study import _df, fit_ic, _class, BASE, HORIZONS

# Widened 2026-07-13 to span the fittable range through the sample ceiling
# (≥ ~2346 rows saturates; kept to confirm the plateau). Sub-100 is degenerate
# and left to the main aarambh_full sweep's ⚠small-win display.
MAXT = [100, 150, 200, 252, 350, 500, 625, 750, 875, 1000, 1250, 1500, 1750,
        2000, 2500, 3000]


def main():
    df = _df()
    targets = [t for t in ALL_TARGETS if t in df.columns and df[t].notna().mean() >= 0.5]
    print(f"Confirmatory MAX_TRAIN sweep @ live MIN={MIN_TRAIN_SIZE} · {len(targets)} targets · "
          f"horizon(s) {list(HORIZONS)}", flush=True)
    print(f"  {'MAX_TRAIN':<10} {'IC':>8} {'Cmdty/FX':>9} {'India-Eq':>9} {'US-Eq':>7}", flush=True)
    print("  " + "-" * 52, flush=True)
    t0 = time.time()
    rows = []
    for maxt in MAXT:
        cfg = dict(BASE); cfg["mint"] = MIN_TRAIN_SIZE; cfg["maxt"] = maxt   # ens=ridge+ols (fast base)
        rec: dict = {"Cmdty/FX": [], "India-Eq": [], "US-Eq": [], "ic": []}
        for tgt in targets:
            for h, mom in HORIZONS.items():
                ic, _ = fit_ic(cfg, tgt, h, mom)
                if np.isfinite(ic):
                    rec["ic"].append(ic); rec[_class(tgt)].append(ic)
        ic_mean = np.mean(rec["ic"]) if rec["ic"] else np.nan
        cf, ie, us = (np.mean(rec[c]) if rec[c] else np.nan for c in ("Cmdty/FX", "India-Eq", "US-Eq"))
        rows.append((maxt, ic_mean))
        mark = "  ←current" if maxt == MAX_TRAIN_SIZE else ""
        print(f"  {maxt:<10} {ic_mean:>+8.3f} {cf:>+9.3f} {ie:>+9.3f} {us:>+7.3f}{mark}", flush=True)
    best = max(rows, key=lambda x: x[1])
    print("  " + "-" * 64, flush=True)
    print(f"  best @ MIN={MIN_TRAIN_SIZE}: MAX_TRAIN={best[0]} (mean IC {best[1]:+.3f})  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
