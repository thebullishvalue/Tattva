"""PER-INSTRUMENT MSF_LENGTH tuning on the India EQUITY indices.

india_index is a per-instrument class: every India index carries its own
`nirnay_msf_length`. This study sweeps MSF length on each index's constituent
breadth and emits a per-index override (gated by the per-instrument noise rule).
US indices (S&P 500 ~500, Nasdaq ~100 constituents) are too heavy here — per_asset
covers us_index / etf per instrument.

Reuses nirnay_tuning_study's breadth_ic + live-config BASE. Only MSF length is
tuned per India index; the other Nirnay knobs (roc/sensitivity/base_weight/
num_vars) stay at the india_index class default (tuned cross-universe by `nirnay`).
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
from core.config import TARGET_CATEGORIES
from research._per_instrument import per_instrument_reco, print_overrides_snippet

# Every India index in the catalogue (per-instrument class). Some broad indices
# may not resolve a constituent basket → their row reads n/a and they keep the
# class default; that is expected and honest.
INDICES = list(TARGET_CATEGORIES.get("India Indices", []))
MSF_GRID = [3, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]


def main():
    _load()
    print(f"PER-INSTRUMENT MSF_LENGTH on {len(INDICES)} India indices · breadth IC vs "
          f"+10d (non-overlapping)", flush=True)
    print(f"  {'MSF_LEN':<8} {'mean|IC|':>9} {'mean IC':>9}   per-index IC", flush=True)
    t0 = time.time()
    table: dict = {}       # {msf_value: {index: ic}} for the per-instrument reco
    # Same grid as nirnay_tuning_study's MSF_LENGTH sweep so the commodity and
    # equity-index tables are directly comparable row-for-row.
    for v in MSF_GRID:
        cfg = dict(BASE); cfg["length"] = v
        ics = {t: breadth_ic(cfg, t) for t in INDICES}
        table[v] = ics
        arr = np.array([x for x in ics.values() if np.isfinite(x)])
        mabs = np.mean(np.abs(arr)) if len(arr) else np.nan
        msign = np.mean(arr) if len(arr) else np.nan
        detail = " ".join(f"{t.replace('Nifty ','N')[:6]}={ics[t]:+.2f}"
                          for t in INDICES if np.isfinite(ics[t]))
        tag = "  ←current" if v == BASE["length"] else ""
        print(f"  {v:<8} {mabs:>9.3f} {msign:>+9.3f}   {detail}{tag}", flush=True)
    print(f"\n  done in {time.time()-t0:.0f}s", flush=True)

    overrides = per_instrument_reco("nirnay_msf_length", table, BASE["length"], set(INDICES))
    print_overrides_snippet(overrides)


if __name__ == "__main__":
    main()
