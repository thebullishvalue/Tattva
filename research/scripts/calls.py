import pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/tattva")
from tattva import evaluate as ev
from tattva.universe import TARGETS_BY_NAME
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad"
res = pickle.load(open(f"{SP}/res_fe.pkl", "rb"))
K = 0.05   # timing slope, = the P0 cross-target IC of the composite (0.042 → 0.05)
def cz(s, n=250): return s / np.sqrt((s ** 2).rolling(n, min_periods=60).mean())
def drift(lab):
    """Causal decay-weighted mean of y using only labels already known at each date."""
    y = lab["y"]; known = pd.to_datetime(lab["known"])
    kn = pd.Series(y.to_numpy(), index=known.to_numpy()).dropna().sort_index()
    # EW mean with 3y half-life on the 'known' clock, then as-of each date
    ew = kn.ewm(halflife=pd.Timedelta(days=int(3 * 365.25)), times=kn.index).mean()
    ew = ew[~ew.index.duplicated(keep="last")]
    return ew.reindex(ew.index.union(lab.index)).ffill().reindex(lab.index)
rows = []
for name, r in res.runs.items():
    f = r.feats
    M = -cz(f["gap60"]); S = -(cz(f["tsm60"]) + cz(f["tsm250"]) + cz(f["range60"])) / 3
    F = (M.fillna(0) + S.fillna(0)) / 2; F[M.isna() & S.isna()] = np.nan
    z = F / np.sqrt((F ** 2).rolling(250, min_periods=60).mean())
    mu = drift(r.label)
    Fa = mu + K * z
    agree = np.sign(M) == np.sign(S)
    strong = (z.abs() >= 0.5) & agree
    V = {
      "V0_timing": np.sign(z).where(strong, 0),                        # what failed
      "V1_abs": np.sign(Fa).where(strong, 0),                          # direction from drift+timing
      "V2_aligned": np.sign(z).where(strong & (np.sign(z) == np.sign(mu)), 0),  # timing agrees with drift
      "V3_abs_all": np.sign(Fa).where(Fa.abs() >= 0.05, 0),            # any |E[y]| ≥ 0.05σ
      "B_drift": np.sign(mu),                                          # baseline: drift sign always
      "B_long": pd.Series(1.0, index=f.index),                         # baseline: always long
    }
    d = pd.DataFrame(V); d["ret"] = r.label["ret"]; d["ok"] = F.notna() & mu.notna()
    d = d[d.ok].dropna(subset=["ret"])
    for per, a, b in ev.PERIODS:
        s = d[(d.index >= a) & (d.index < b)].iloc[::10]
        if len(s) < 15: continue
        for k in V:
            c = s[s[k] != 0]
            rows.append({"target": name, "pool": TARGETS_BY_NAME[name].pool, "period": per, "v": k,
                         "calls": len(c), "hits": int((np.sign(c.ret) == c[k]).sum()),
                         "pnl": float((c[k] * c.ret).sum())})
T = pd.DataFrame(rows)
g = T.groupby(["v", "period"])[["calls", "hits", "pnl"]].sum()
g["hit"] = g.hits / g.calls; g["avg_ret%"] = 100 * g.pnl / g.calls
pd.set_option("display.width", 200)
print(g[["calls", "hit", "avg_ret%"]].round(3).unstack("period").to_string())
g2 = T.groupby(["pool", "v", "period"])[["calls", "hits", "pnl"]].sum(); g2["hit"] = g2.hits / g2.calls
print(g2["hit"].unstack("period").round(3).to_string())
