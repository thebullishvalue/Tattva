"""PRE-REGISTERED (2026-09-30, before any number): current Tattva engines vs the Pragati axes,
on Tattva's own targets and data, at Tattva's 10-day horizon.

Signals (every one oriented BULLISH-POSITIVE by its own semantics, never by the data):
  CURRENT   mula_conv  = −ConvictionBounded      (Mūla's published conviction; negative = bullish)
            mula_gap   = −FVO                    (Mūla's gap level, σ; cheap = bullish)
            swayam     = −Avg_Signal             (Swayam's breadth oscillator; oversold = bullish)
  NEW       val_sam    = −v_tape  (Samanvaya)    val_mula = −Mūla value tape    val_mix = −blend
            conv_tape  = +c_tape  (reported, not a directional claim)
            grid_{sam,mula,mix} = lean by cell: capitulation +1, washout +½, distribution −1, else 0
            ▲ / ▼ events (grid_sam and grid_mula variants), scored as events
Scoring: Spearman IC of the signal vs the next-open h=10 log return, non-overlapping (stride 10),
per target per period (P1 2019-07…2021, P2 2022-23, P3 2024…), on the dates every compared signal
is valid; plus the causal vol-scaled time-series edge. RULE: a new piece replaces the current one
only if it beats it in ALL THREE periods on the cross-target mean."""
import os, sys, glob, pickle, warnings, dataclasses
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/Sanket")
import intraday; intraday.ENABLED = False                 # history has no intrabars: W·D (the ↺ read)
import engine as en, samanvaya as sv
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad/tv"
H = 10
PER = (("P1", "2019-07-01", "2022-01-01"), ("P2", "2022-01-01", "2024-01-01"), ("P3", "2024-01-01", "2027-01-01"))
LEAN = np.array([1.0, 0.5, -1.0, 0, 0, 0, 0, 0, 0, np.nan])


def drivers():
    f = f"{SP}/drivers.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    import yfinance as yf
    raw = yf.download(sv.DRIVER_TICKERS, start="2016-01-01", end="2026-10-01", progress=False, auto_adjust=True)
    c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    c = pd.DataFrame(c).dropna(how="all", axis=1); c.index = pd.to_datetime(c.index).tz_localize(None)
    d = sv.prepare_drivers(c, "D"); pickle.dump(d, open(f, "wb")); return d


def mula_value(sam_val, mula):
    """Samanvaya's value frame with Mūla's gap as the value reading (same ±100 scale, same θ)."""
    v = sam_val.copy()
    m = mula.reindex(v.index)
    z = pd.to_numeric(m["FVO"], errors="coerce")
    ok = m["Valid"].fillna(False).astype(bool) & z.notna()
    osc = 100.0 * sv.softbound(sv.GAIN * z) if hasattr(sv, "GAIN") else 100.0 * sv.softbound(0.5 * z)
    v["value_z"], v["value"], v["tape"] = z, osc, osc
    v["model_built"], v["tape_ready"] = ok, ok
    return v


def mix_value(sam_val, mv):
    v = sam_val.copy()
    ok = sam_val["tape_ready"].astype(bool) & mv["tape_ready"].astype(bool)
    v["tape"] = (sam_val["tape"] + mv["tape"]) / 2
    v["value"] = (sam_val["value"] + mv["value"]) / 2
    v["value_z"] = (sam_val["value_z"] + mv["value_z"]) / 2
    v["model_built"] = sam_val["model_built"].astype(bool) & mv["model_built"].astype(bool)
    v["tape_ready"] = ok
    return v


def build(p):
    d = pickle.load(open(p, "rb"))
    o = d["ohlcv"]
    if o is None or len(o) < 600:
        return None
    o = o[["Open", "High", "Low", "Close", "Volume"]].copy(); o.index = pd.to_datetime(o.index).tz_localize(None)
    o = o[~o.index.duplicated(keep="last")].sort_index()
    st = en.settings_for(None, None, "Daily")
    st = dataclasses.replace(st, params=dataclasses.replace(st.params, ladder="up"))
    lo = en._chart_bars(o)
    sam = sv.compute_value(lo, drivers(), d["ticker"], chart="D")
    mv = mula_value(sam, d["mula"]); xv = mix_value(sam, mv)
    F = {k: en.compute_frame(o, None, d["ticker"], st, value=v, intraday={}).reindex(o.index)
         for k, v in (("sam", sam), ("mula", mv), ("mix", xv))}
    m = d["mula"].reindex(o.index); s = d["swayam"]
    s = s.reindex(o.index) if len(s) else pd.DataFrame(index=o.index)
    S = pd.DataFrame(index=o.index)
    S["mula_conv"] = -pd.to_numeric(m.get("ConvictionBounded"), errors="coerce").where(m["Valid"].fillna(False).astype(bool))
    S["mula_gap"] = -pd.to_numeric(m.get("FVO"), errors="coerce").where(m["Valid"].fillna(False).astype(bool))
    S["swayam"] = -pd.to_numeric(s.get("Avg_Signal"), errors="coerce") if "Avg_Signal" in s else np.nan
    for k, f in F.items():
        ok = f["v_ready"].fillna(False).astype(bool) & f["c_ready"].fillna(False).astype(bool)
        S[f"val_{k}"] = (-f["v_tape"]).where(ok)
        cell = f["cvg_cell"].fillna(9).astype(int).clip(0, 9).to_numpy()
        S[f"grid_{k}"] = pd.Series(LEAN[cell], index=o.index).where(ok)
        S[f"up_{k}"] = f["turn_buy"].fillna(False).astype(bool)
        S[f"dn_{k}"] = f["turn_sell"].fillna(False).astype(bool)
    S["conv_tape"] = F["sam"]["c_tape"].where(F["sam"]["c_ready"].fillna(False).astype(bool))
    lr = np.log(o["Open"])
    S["fwd"] = lr.shift(-(1 + H)) - lr.shift(-1)
    S["vol"] = np.log(o["Close"]).diff().rolling(60, min_periods=40).std() * np.sqrt(H)
    return d["target"], S


CONT = ["mula_conv", "mula_gap", "swayam", "val_sam", "val_mula", "val_mix", "conv_tape", "grid_sam", "grid_mula", "grid_mix"]


def ic_table(all_S):
    rows = []
    for tgt, S in all_S.items():
        common = S[CONT[:6] + ["fwd"]].dropna().index           # every CURRENT + VALUE signal valid
        for per, a, b in PER:
            idx = common[(common >= a) & (common < b)][::H]         # non-overlapping
            if len(idx) < 15:
                continue
            r = {"target": tgt, "per": per, "n": len(idx)}
            for c in CONT:
                x = S.loc[idx, c]; y = S.loc[idx, "fwd"]; m = x.notna() & y.notna()
                r[c] = spearmanr(x[m], y[m]).correlation if m.sum() >= 10 and x[m].nunique() > 1 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def events(all_S):
    rows = []
    for tgt, S in all_S.items():
        z = (S["fwd"] / S["vol"])
        for per, a, b in PER:
            m = (S.index >= a) & (S.index < b) & z.notna()
            base = z[m].mean()
            for k in ("up_sam", "up_mula", "dn_sam", "dn_mula"):
                e = S[k].to_numpy(bool) & m
                if e.sum():
                    rows.append({"target": tgt, "per": per, "event": k, "n": int(e.sum()),
                                 "excess": float(z[e].mean() - base) * (1 if k.startswith("up") else -1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_S = {}
    for p in sorted(glob.glob(f"{SP}/base/*.pkl")):
        try:
            r = build(p)
        except Exception as e:
            print("FAIL", os.path.basename(p), type(e).__name__, e, flush=True); continue
        if r:
            all_S[r[0]] = r[1]; print("built", r[0], flush=True)
    pickle.dump(all_S, open(f"{SP}/measure_S.pkl", "wb"))
    T = ic_table(all_S); T.to_pickle(f"{SP}/measure_ic.pkl")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    g = T.groupby("per")[CONT]
    print("\n== IC (10d, non-overlapping) — cross-target MEAN by period")
    print(g.mean().round(3).to_string())
    print("\n== share of targets with IC > 0")
    print(g.apply(lambda x: (x > 0).mean()).round(2).to_string())
    print("\n== targets per period"); print(T.groupby("per").size().to_string())
    E = events(all_S)
    if len(E):
        E.to_pickle(f"{SP}/measure_ev.pkl")
        print("\n== events: vol-scaled 10d excess over the target's period mean (▼ sign-flipped; + = worked)")
        print(E.groupby(["event", "per"]).apply(lambda x: pd.Series({"n": x.n.sum(), "excess_w": np.average(x.excess, weights=x.n)})).round(3).to_string())
