import harness, pickle, sys, numpy as np, pandas as pd
df = harness.load_panel()
out = {}
for tgt in ["Gold", "Nifty 50", "S&P 500", "USD/INR", "Bitcoin", "Nifty Bank", "Copper", "Nasdaq 100"]:
    try:
        eng, ts, dt, nf = harness.fit(df, tgt)
        out[tgt] = ts
        print(tgt, f"{dt:.0f}s", nf, "feats", len(ts), "rows", flush=True)
    except Exception as e:
        print(tgt, "FAIL", type(e).__name__, e, flush=True)
pickle.dump(out, open("fits.pkl", "wb"))
# sliding-start repaint test on Gold: drop first 1 and 21 calendar rows
for dh in (1, 21):
    eng, ts, dt, nf = harness.fit(df, "Gold", drop_head=dh)
    out[f"Gold_drop{dh}"] = ts
    print("Gold drop", dh, nf, flush=True)
pickle.dump(out, open("fits.pkl", "wb"))
