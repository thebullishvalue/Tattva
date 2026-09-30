import pickle, sys, time, warnings; warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, "/home/user/tattva")
from tattva.data import fetch_universe
from tattva import engine, evaluate as ev
from tattva.universe import TARGETS_BY_NAME
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad"
tag = sys.argv[1]
t0 = time.time()
res = engine.run(fetch_universe(end="2026-09-30"))
pickle.dump(res, open(f"{SP}/res_{tag}.pkl", "wb"))
pools = {n: TARGETS_BY_NAME[n].pool for n in res.runs}
T = ev.ic_table(res.runs, pools); T.to_pickle(f"{SP}/ic_{tag}.pkl")
pd.set_option("display.width", 200)
print("secs", round(time.time() - t0))
print(ev.summary(T).round(3).to_string()); print(); print(ev.summary(T, "pool").round(3).to_string())
print("overall hit", round(T.hits.sum() / T.calls.sum(), 4), "calls", T.calls.sum())
O = pd.read_pickle(f"{SP}/tv/e2e_ic.pkl").rename(columns={"per": "period"})
M = O.merge(T, on=["target", "period"])
print("vs old hero, shared", M.target.nunique()); print(M.groupby("period")[["E0_HERO", "F"]].mean().round(3).to_string())
print("stack latest", res.stack[res.refits[-1]])
