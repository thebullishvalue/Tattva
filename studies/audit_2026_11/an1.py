import harness, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from convergence.intelligence import series_edge
from core.semantics import valuation_lean
fits = pickle.load(open("fits.pkl","rb"))
pd.set_option("display.width", 250)
print("=== A. edge: lag0 (as shipped) vs 1-session execution lag; claim-only rows; first-day share")
for k in ["Gold","Nifty 50","Nifty Bank","S&P 500","Nasdaq 100","USD/INR","Bitcoin","Copper"]:
    ts = fits[k]; fvo = ts["FVO"]; px = ts["Price"]
    lean = fvo.map(valuation_lean).where(fvo.notna())
    e0 = series_edge(lean, px, 10)
    e1 = series_edge(lean.shift(1), px, 10)     # call issued at t-1 scored from t: i.e. enter next close
    lp = np.log(px); r1 = (lp.shift(-1) - lp); r2_10 = (lp.shift(-11) - lp.shift(-1))
    d = pd.DataFrame({"l": lean, "r1": r1, "rr": r2_10}).dropna()
    ic1 = d["l"].corr(d["r1"], method="spearman"); icr = d["l"].corr(d["rr"], method="spearman")
    claim = lean.where(fvo.abs() >= 0.5)
    ec = series_edge(claim, px, 10)
    print(f"{k:11s} IC lag0 {e0['ic']:+.3f} (n={e0['n']}, {e0['positive_blocks']}/{e0['scored_blocks']})  lag1 {e1['ic']:+.3f}  | daily IC next-1d {ic1:+.3f}  days2-11 {icr:+.3f} | claim-only IC {ec['ic']:+.3f} n={ec['n']} hit {ec['hit']:.0f}% base {ec['base']:.0f}%  claim-share {(fvo.abs()>=0.5).mean():.0%}")
print("\n=== B. gate saturation: distribution of gate inputs on published rows")
for k in ["Gold","Nifty 50","S&P 500","USD/INR","Bitcoin"]:
    ts = fits[k]; v = ts[ts["FVO"].notna()]
    q = lambda c: " ".join(f"{x:.2f}" for x in v[c].dropna().quantile([.05,.25,.5,.75,.95]))
    print(f"{k:9s} XS[{q('XSConsistency')}]  MRProb[{q('MRProb')}]  |FVO|[{' '.join(f'{x:.2f}' for x in v['FVO'].abs().quantile([.25,.5,.75,.95]))}]  sd(FVO) {v['FVO'].std():.2f} halflife med {v['GapHalfLife'].median():.1f}d  kappa med {v['MulaKappa'].median():+.4f}")
print("\n=== C. sliding start repaint (Gold): same code, panel start moved by 1 / 21 rows")
a = fits["Gold"]
for dh in (1,21):
    b = fits[f"Gold_drop{dh}"]
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    both = j["FVO_a"].notna() & j["FVO_b"].notna()
    d = (j["FVO_a"]-j["FVO_b"])[both].abs()
    fv = (np.log(j["FairValue_a"])-np.log(j["FairValue_b"]))[both].abs()
    flips = (np.sign(j["FVO_a"])!=np.sign(j["FVO_b"]))[both].mean()
    st = (j["Regime_a"]!=j["Regime_b"])[both].mean()
    print(f"drop {dh:2d}: rows {both.sum()}  |dFVO| med {d.median():.3f} p95 {d.quantile(.95):.3f} max {d.max():.2f}; last-250 med {d.tail(250).median():.3f}; |dlogFV| med {fv.median()*1e4:.0f}bps p95 {fv.quantile(.95)*1e4:.0f}bps; sign flips {flips:.1%}; state word differs {st:.1%}; first-pub {a['FVO'].first_valid_index().date()} vs {b['FVO'].first_valid_index().date()}")
print("\n=== D. spine truncation: rows fetched vs rows used")
df = harness.load_panel()
for k in ["Gold","Nifty 50","S&P 500"]:
    data, feats, cfg = harness.prep(df, k)
    tgt_rows = df[k].notna().sum()
    first = df[feats].apply(lambda s: s.first_valid_index())
    late = df.loc[first.sort_values().tail(5).values, "DATE"].dt.date.tolist()
    print(f"{k}: target has {tgt_rows} rows; spine used {len(data)} ({len(data)/tgt_rows:.0%}), starts {data['DATE'].iloc[0].date()}; youngest kept features: {dict(zip(first.sort_values().tail(5).index, late))}")
