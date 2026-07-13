"""Confirm the COMBINED analog config (OFAT winners together): Mahalanobis-only +
drop AvgZ. Reuses analog_tuning_study machinery."""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from analog_tuning_study import agg, _cfg, _load
from analytics.analogs import ANALOG_W_MAHA, ANALOG_W_TRAJ, ANALOG_W_RECV

NO_AVGZ = ("Momentum", "RealizedVol", "NetBreadth", "Hurst")


def main():
    _load()
    # Label the current row from LIVE weights — a hardcoded ".55/.35/.10" kept
    # mislabelling the row after the shipped blend moved to maha-only.
    _cur = f"current {ANALOG_W_MAHA:g}/{ANALOG_W_TRAJ:g}/{ANALOG_W_RECV:g} base5"
    cfgs = {
        _cur:                             _cfg(),
        "maha-only base5":                _cfg(wm=1, wt=0, wr=0),
        "maha-only drop-AvgZ":            _cfg(wm=1, wt=0, wr=0, feats=NO_AVGZ),
        "maha-only drop-AvgZ top5":       _cfg(wm=1, wt=0, wr=0, top_n=5, feats=NO_AVGZ),
        "maha-only drop-AvgZ top15":      _cfg(wm=1, wt=0, wr=0, top_n=15, feats=NO_AVGZ),
        "maha-only drop-AvgZ top20":      _cfg(wm=1, wt=0, wr=0, top_n=20, feats=NO_AVGZ),
        "maha.9/traj.1 drop-AvgZ":        _cfg(wm=.9, wt=.1, wr=0, feats=NO_AVGZ),
        "maha.85/traj.15 drop-AvgZ":      _cfg(wm=.85, wt=.15, wr=0, feats=NO_AVGZ),
        "maha-only drop-AvgZ trim":       _cfg(wm=1, wt=0, wr=0, feats=NO_AVGZ, agg="trim"),
    }
    print(f"  {'config':<28} " + "  ".join(f"{('+'+str(h)+'d'):>15}" for h in (1, 10, 20)), flush=True)
    print("  (cells = full_IC / recent_IC)", flush=True)
    for name, cfg in cfgs.items():
        cells = []
        for h in (1, 10, 20):
            f, r = agg(cfg, h)
            cells.append(f"{f:+.3f}/{r:+.3f}")
        mark = "  ←current" if name == _cur else ""
        print(f"  {name:<28} " + "  ".join(f"{c:>15}" for c in cells) + mark, flush=True)


if __name__ == "__main__":
    main()
