import pickle, numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore"); sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from convergence.intelligence import series_edge
from core.semantics import valuation_lean
cat = pickle.load(open("cat.pkl","rb"))
cat = {k:v for k,v in cat.items() if k!="Nifty 50 - PE"}
H=10
rows=[]; ics={}; icl={}
for k,ts in cat.items():
    fvo=ts["FVO"]; lp=np.log(ts["Price"]); fwd=lp.shift(-(H+1))-lp.shift(-1)
    lean=fvo.map(valuation_lean).where(fvo.notna())
    ics[k]=series_edge(lean,ts["Price"],H)["ic"]; icl[k]=series_edge(lean.shift(1),ts["Price"],H)["ic"]
    sd=fwd.std()
    mom=(lp-lp.shift(126))                      # causal 6m trend
    ctr=fvo-fvo.ewm(halflife=63,min_periods=63).mean().shift(1)   # causally de-meaned oscillator
    rows.append(pd.DataFrame({"t":k,"fvo":fvo,"ctr":ctr,"fwd":fwd/sd,"mom":mom,"rich_share":0}).dropna(subset=["fvo","fwd"]))
X=pd.concat(rows); X["date"]=X.index
s=pd.Series(ics); l=pd.Series(icl)
print(f"targets {len(s)}; per-target 10d IC as shipped: mean {s.mean():+.3f} median {s.median():+.3f} share>0 {(s>0).mean():.0%}; with 1-session lag: mean {l.mean():+.3f} median {l.median():+.3f} share>0 {(l>0).mean():.0%}; se of mean (naive, indep) {l.std()/np.sqrt(len(l)):.3f}")
print("worst/best lag1:", l.sort_values().round(3).head(5).to_dict(), l.sort_values().round(3).tail(5).to_dict())
C=X[X.fvo.abs()>=0.5].copy(); C["y"]=-np.sign(C.fvo)*C.fwd
print(f"\nCALL AS ISSUED (|FVO|>=0.5, enter next close, 10d): rows {len(C)}, mean signed outcome {C.y.mean():+.3f} sd-units, hit {(C.y>0).mean():.1%}; always-long on same rows hit {(C.fwd>0).mean():.1%}")
print(f"  cheap claims: n={int((C.fvo<0).sum())} mean fwd {C[C.fvo<0].fwd.mean():+.3f} | rich claims: n={int((C.fvo>0).sum())} mean fwd {C[C.fvo>0].fwd.mean():+.3f} | all rows mean fwd {X.fwd.mean():+.3f}")
print(f"  share of published days rich(>=.5) {(X.fvo>=.5).mean():.0%} cheap(<=-.5) {(X.fvo<=-.5).mean():.0%}; mean FVO {X.fvo.mean():+.2f}; corr(FVO, 126d momentum) {X.fvo.corr(X.mom,method='spearman'):+.2f}")
for lo,hi in ((.5,1),(1,2),(2,99)):
    b=C[(C.fvo.abs()>=lo)&(C.fvo.abs()<hi)]; print(f"  |FVO| in [{lo},{hi}): n={len(b)} mean signed {b.y.mean():+.3f} hit {(b.y>0).mean():.1%}")
# cross-sectional
def xs_ic(col):
    out=[]
    for d,g in X.groupby("date"):
        g=g.dropna(subset=[col])
        if len(g)>=12: out.append(( -g[col]).corr(g.fwd,method="spearman"))
    o=pd.Series(out); nn=len(o)/H
    return o.mean(), o.std()/np.sqrt(nn), len(o)
m,se,n=xs_ic("fvo"); print(f"\nCROSS-SECTIONAL rank IC (cheapest-vs-richest across catalogue each day, 10d fwd): mean {m:+.3f} ± {se:.3f} (t≈{m/se:.1f}) over {n} days")
m,se,n=xs_ic("ctr"); print(f"  same, causally de-meaned oscillator: {m:+.3f} ± {se:.3f} (t≈{m/se:.1f})")
# quintile spread
def spread(col):
    r=[]
    for d,g in X.groupby("date"):
        g=g.dropna(subset=[col])
        if len(g)>=12:
            q=g[col].rank(pct=True); r.append(g.fwd[q<=.2].mean()-g.fwd[q>=.8].mean())
    r=pd.Series(r); return r.mean(), r.std()/np.sqrt(len(r)/H)
print("  cheapest-quintile minus richest-quintile 10d fwd (sd units): fvo %+.3f ± %.3f ; de-meaned %+.3f ± %.3f"%(*spread("fvo"),*spread("ctr")))
# exploratory time-series variants
D=X.dropna(subset=["ctr"]); Cc=D[D.ctr.abs()>=0.5]; y=-np.sign(Cc.ctr)*Cc.fwd
print(f"\nEXPLORATORY de-meaned call: n={len(Cc)} mean signed {y.mean():+.3f} hit {(y>0).mean():.1%}; rich share {(D.ctr>=.5).mean():.0%} cheap {(D.ctr<=-.5).mean():.0%}")
T=C.dropna(subset=["mom"]); al=T[np.sign(T.mom)==-np.sign(T.fvo)]; ag=T[np.sign(T.mom)==np.sign(T.fvo)]
print(f"EXPLORATORY trend split: claims WITH 6m trend (buy cheap in uptrend / sell rich in downtrend) n={len(al)} mean {(-np.sign(al.fvo)*al.fwd).mean():+.3f} hit {((-np.sign(al.fvo)*al.fwd)>0).mean():.1%} | AGAINST trend n={len(ag)} mean {(-np.sign(ag.fvo)*ag.fwd).mean():+.3f} hit {((-np.sign(ag.fvo)*ag.fwd)>0).mean():.1%}")
by=C.groupby("t").y.mean(); print("per-target mean signed outcome: share of targets > 0:", f"{(by>0).mean():.0%}", "| by class:", C.assign(c=C.t.map(lambda n: 'crypto' if n in ('Bitcoin','Ethereum','Solana','XRP','BNB','Cardano','Dogecoin') else 'india' if 'Nifty' in n else 'us' if n in ('S&P 500','Nasdaq 100','Dow Jones') else 'cmdty/fx')).groupby('c').y.agg(['mean','size']).round(3).to_dict('index'))
