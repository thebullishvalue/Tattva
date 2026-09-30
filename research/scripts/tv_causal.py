"""Swayam causality check: refit on data truncated 250 / 600 bars early; the published
contract, grid cells and events must match the full fit exactly on every shared date.
Reads the per-target pickles tv_run.py writes (SP below)."""
import sys, os, glob, pickle, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/home/user/tattva"); os.chdir("/home/user/tattva")
from engines.swayam import SwayamEngine
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad/tv"
DRV = pickle.load(open(f"{SP}/drivers.pkl", "rb"))
worst = 0
for p in sorted(glob.glob(f"{SP}/base/*.pkl"))[:8]:
    d = pickle.load(open(p, "rb")); o = d["ohlcv"]
    full = SwayamEngine().fit(o, DRV, d["ticker"])
    for cut in (250, 600):
        T = o.index[-cut]
        part = SwayamEngine().fit(o[o.index <= T], DRV[DRV.index <= T], d["ticker"])
        a, b = full.daily, part.daily
        idx = a.index.intersection(b.index)
        diff = (a.loc[idx] - b.loc[idx]).abs().max().max()
        missing = len(b.index.difference(a.index))
        ev = (full.frame.loc[idx, ["turn_buy","turn_sell","cvg_cell"]].fillna(-1) != part.frame.loc[idx, ["turn_buy","turn_sell","cvg_cell"]].fillna(-1)).sum().sum()
        worst = max(worst, diff)
        print(f"{d['target']:<14} cut-{cut}: shared={len(idx)} maxdiff={diff:.2e} event/cell mismatches={ev} only-in-part={missing}")
print("WORST", worst)
