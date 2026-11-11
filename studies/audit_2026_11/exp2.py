import harness, pickle, numpy as np, pandas as pd
from core.config import CATALOGUE_TARGETS
df = harness.load_panel()
out = {}
for tgt in CATALOGUE_TARGETS:
    if tgt not in df.columns: print(tgt, "no column", flush=True); continue
    try:
        eng, ts, dt, nf = harness.fit(df, tgt)
        out[tgt] = ts[["Price","FVO","FairValue","Residual","GapScale","MRProb","XSConsistency","Valid","Provisional","MulaDriftPct","GapHalfLife"]]
        print(tgt, f"{dt:.0f}s", len(ts), int(ts['FVO'].notna().sum()), flush=True)
    except Exception as e:
        print(tgt, "FAIL", type(e).__name__, str(e)[:100], flush=True)
pickle.dump(out, open("cat.pkl", "wb")); print("DONE", flush=True)
