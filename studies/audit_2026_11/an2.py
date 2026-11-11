import harness, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from core.semantics import valuation_lean
from core.config import ALL_TARGETS, get_instrument_config, MIN_DATA_POINTS, STOCK_TARGET_MARKETS
from data.calendars import session_mask
fits = pickle.load(open("fits.pkl","rb"))
df = harness.load_panel()
print("=== E. the next history cliff: features currently dropped for <1500 rows, and when they cross")
tgt="Gold"; cfg=get_instrument_config(tgt)
num=[c for c in df.select_dtypes(include=[np.number]).columns if c!=tgt and c not in cfg.excluded_predictors]
d=df[[tgt]+num+["DATE"]].copy(); sm=session_mask(ALL_TARGETS[tgt], d["DATE"]); d=d[sm].reset_index(drop=True)
d[num]=d[num].ffill()
rows=[]
for c in num:
    fi=d[c].first_valid_index()
    n=len(d)-fi if fi is not None else 0
    if n<MIN_DATA_POINTS: rows.append((c, str(d['DATE'].iloc[fi].date()) if fi is not None else None, n, MIN_DATA_POINTS-n))
print(sorted(rows, key=lambda r:-r[2])[:8])
print("total columns", len(num), "kept", sum(1 for c in num if d[c].tail(1500).notna().all()))

print("\n=== F. IC term structure with 1-session lag (non-overlap offset-avg), and pooled")
from convergence.intelligence import series_edge
names=["Gold","Copper","Nifty 50","Nifty Bank","S&P 500","Nasdaq 100","USD/INR","Bitcoin"]
tab={}
for k in names:
    ts=fits[k]; lean=ts["FVO"].map(valuation_lean).where(ts["FVO"].notna()).shift(1)
    tab[k]={h: series_edge(lean, ts["Price"], h)["ic"] for h in (1,3,5,10,20)}
t=pd.DataFrame(tab).T; t.loc["MEAN"]=t.mean(); print(t.round(3))

print("\n=== G. do the gates discriminate? claim rows only, y = direction x fwd10 (entered next close), pooled z-scored per target")
rows=[]
for k in names:
    ts=fits[k]; v=ts[ts["FVO"].notna()].copy()
    lp=np.log(ts["Price"]); fwd=(lp.shift(-11)-lp.shift(-1)).reindex(v.index)
    dirn=-np.sign(v["FVO"]); y=dirn*fwd; y=y/ fwd.std()
    g_mis=((v["FVO"].abs()-0.5)/1.5).clip(0.02,1); g_rev=((v["MRProb"]-0.5)/0.4).clip(0.02,1).fillna(0.5); g_xs=((v["XSConsistency"]-0.3)/0.6).clip(0.02,1).fillna(.5)
    x=pd.DataFrame({"t":k,"y":y,"mis":g_mis,"rev":g_rev,"xs":g_xs,"prod3":g_mis*g_rev*g_xs,"absfvo":v["FVO"].abs()})[v["FVO"].abs()>=0.5].dropna()
    rows.append(x)
X=pd.concat(rows)
print("claim rows", len(X), "mean y (in fwd-sd units)", round(X.y.mean(),4), "hit", round((X.y>0).mean(),3))
for g in ("mis","rev","xs","prod3"):
    q=pd.qcut(X[g].rank(method="first"),4,labels=["Q1 low","Q2","Q3","Q4 high"])
    m=X.groupby(q).y.agg(["mean",lambda s:(s>0).mean(),"size"]); 
    print(g, "| mean y by gate quartile:", " ".join(f"{a:+.3f}" for a in m["mean"]), "| hit:", " ".join(f"{a:.2f}" for a in m.iloc[:,1]), "| rank corr", round(X[g].corr(X.y,method='spearman'),3))
print("per target mean y on claim rows:", X.groupby("t").y.mean().round(3).to_dict())
print("\n=== H. always-long benchmark: mean fwd10/sd on all rows per target vs on claim rows signed")
for k in names:
    ts=fits[k]; lp=np.log(ts["Price"]); f=(lp.shift(-11)-lp.shift(-1)); v=ts["FVO"]
    print(k, "drift/sd", round(f.mean()/f.std(),3), "| long-claims", round(f[v<=-0.5].mean()/f.std(),3), f"(n={int((v<=-0.5).sum())})", "| short-claims fwd", round(f[v>=0.5].mean()/f.std(),3), f"(n={int((v>=0.5).sum())})")
