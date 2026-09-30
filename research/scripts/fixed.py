import pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/tattva")
from scipy.stats import spearmanr
from tattva import evaluate as ev
from tattva.universe import TARGETS_BY_NAME
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad"
res = pickle.load(open(f"{SP}/res_fe.pkl", "rb"))

def cz(s, n=250):  # causal scale: divide by trailing RMS (no centring → sign kept)
    return s / np.sqrt((s ** 2).rolling(n, min_periods=60).mean())

rows = []
for name, r in res.runs.items():
    f = r.feats
    M = -cz(f["gap60"])
    S = -(cz(f["tsm60"]) + cz(f["tsm250"]) + cz(f["range60"])) / 3
    F = (M.fillna(0) + S.fillna(0)) / 2
    F[M.isna() & S.isna()] = np.nan
    z = F / np.sqrt((F ** 2).rolling(250, min_periods=60).mean())
    agree = np.sign(M) == np.sign(S)
    call = pd.Series(np.where(z >= .5, 1, np.where(z <= -.5, -1, 0)), index=f.index).where(agree, 0)
    d = pd.DataFrame({"M": M, "S": S, "F": F, "call": call, "ret": r.label["ret"]}).dropna(subset=["F", "ret"])
    for per, a, b in ev.PERIODS:
        s = d[(d.index >= a) & (d.index < b)].iloc[::10]
        if len(s) < 15: continue
        x = {"target": name, "pool": TARGETS_BY_NAME[name].pool, "period": per}
        for k in ("M", "S", "F"):
            q = s[[k, "ret"]].dropna(); x[k] = spearmanr(q[k], q.ret).correlation
        c = s[s.call != 0]; x["calls"] = len(c); x["hits"] = int((np.sign(c.ret) == c.call).sum())
        x["call_ret"] = (c.call * c.ret).sum()
        rows.append(x)
T = pd.DataFrame(rows)
def summ(g):
    o = g[["M", "S", "F"]].mean(); o["pos_share_F"] = (g.F > 0).mean(); o["calls"] = g.calls.sum()
    o["hit"] = g.hits.sum() / max(g.calls.sum(), 1); o["avg_call_ret%"] = 100 * g.call_ret.sum() / max(g.calls.sum(), 1); return o
pd.set_option("display.width", 200)
print(T.groupby("period").apply(summ).round(3).to_string())
print(T.groupby(["pool", "period"]).apply(summ).round(3).to_string())
O = pd.read_pickle(f"{SP}/tv/e2e_ic.pkl").rename(columns={"per": "period"})
print(O.merge(T, on=["target", "period"]).groupby("period")[["E0_HERO", "F"]].mean().round(3).to_string())
T.to_pickle(f"{SP}/ic_fixed.pkl")
