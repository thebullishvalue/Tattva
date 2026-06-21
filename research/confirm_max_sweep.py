"""Confirmatory MAX_TRAIN_SIZE sweep at the NEW MIN_TRAIN_SIZE=750.

The main study swept MAX_TRAIN at the old MIN=500; this re-checks it at MIN=750
(the change just applied) to rule out an interaction. Reuses the same fit/IC
machinery (fast ridge+ols base, purged, non-overlapping OOS IC).
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import ALL_TARGETS
from aarambh_tuning_study import _df, fit_ic, _class, BASE, HORIZONS

MAXT = [500, 750, 1000, 1500]


def main():
    df = _df()
    targets = [t for t in ALL_TARGETS if t in df.columns and df[t].notna().mean() >= 0.5]
    print(f"Confirmatory MAX_TRAIN sweep @ MIN=750 · {len(targets)} targets · lenses {list(HORIZONS)}",
          flush=True)
    print(f"  {'MAX_TRAIN':<10} {'10d':>8} {'20d':>8} {'combined':>9} "
          f"{'Cmdty/FX':>9} {'India-Eq':>9} {'US-Eq':>7}", flush=True)
    print("  " + "-" * 64, flush=True)
    t0 = time.time()
    rows = []
    for maxt in MAXT:
        cfg = dict(BASE); cfg["mint"] = 750; cfg["maxt"] = maxt   # ens=ridge+ols (fast base)
        rec = {"Cmdty/FX": [], "India-Eq": [], "US-Eq": [], 10: [], 20: []}
        for tgt in targets:
            for h, mom in HORIZONS.items():
                ic, _ = fit_ic(cfg, tgt, h, mom)
                if np.isfinite(ic):
                    rec[h].append(ic); rec[_class(tgt)].append(ic)
        ic10, ic20 = np.mean(rec[10]), np.mean(rec[20])
        comb = np.nanmean([ic10, ic20])
        cf, ie, us = (np.mean(rec[c]) if rec[c] else np.nan for c in ("Cmdty/FX", "India-Eq", "US-Eq"))
        rows.append((maxt, comb))
        mark = "  ←current" if maxt == 750 else ""
        print(f"  {maxt:<10} {ic10:>+8.3f} {ic20:>+8.3f} {comb:>+9.3f} "
              f"{cf:>+9.3f} {ie:>+9.3f} {us:>+7.3f}{mark}", flush=True)
    best = max(rows, key=lambda x: x[1])
    print("  " + "-" * 64, flush=True)
    print(f"  best @ MIN=750: MAX_TRAIN={best[0]} (combined IC {best[1]:+.3f})  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
