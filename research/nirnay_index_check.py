"""MSF_LENGTH validation on EQUITY indices (does 10 > 20 hold beyond commodities?).

Reuses nirnay_tuning_study's breadth_ic. India sectoral/broad indices only —
US indices (S&P 500 ~500, Nasdaq ~100 constituents, uncapped) are too heavy.
"""
from __future__ import annotations
import warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from nirnay_tuning_study import breadth_ic, BASE, _load

INDICES = ["Nifty 50", "Nifty Bank", "Nifty IT", "Nifty Auto", "Nifty Metal"]


def main():
    _load()
    print(f"MSF_LENGTH check on {len(INDICES)} India equity indices · breadth IC vs +10d "
          f"(non-overlapping)", flush=True)
    print(f"  {'MSF_LEN':<8} {'mean|IC|':>9} {'mean IC':>9}   per-index IC", flush=True)
    t0 = time.time()
    # Same grid as nirnay_tuning_study's MSF_LENGTH sweep so the commodity and
    # equity-index tables are directly comparable row-for-row.
    for v in [3, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]:
        cfg = dict(BASE); cfg["length"] = v
        ics = {t: breadth_ic(cfg, t) for t in INDICES}
        arr = np.array([x for x in ics.values() if np.isfinite(x)])
        mabs = np.mean(np.abs(arr)) if len(arr) else np.nan
        msign = np.mean(arr) if len(arr) else np.nan
        detail = " ".join(f"{t.replace('Nifty ','N')[:6]}={ics[t]:+.2f}"
                          for t in INDICES if np.isfinite(ics[t]))
        tag = "  ←current" if v == 20 else ""
        print(f"  {v:<8} {mabs:>9.3f} {msign:>+9.3f}   {detail}{tag}", flush=True)
    print(f"\n  done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
