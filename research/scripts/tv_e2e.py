"""PRE-REGISTERED END-TO-END TEST (2026-09-30, before any number): Tattva's own Convergence
pipeline — the CrossValidator, the online dimension weights, the hero consensus — run exactly as
app.py Phase 4 runs it, on each target, for three engine pairings:

  E0  current Mūla + current Swayam (MSF/MMR view bank)                 the baseline
  E1  current Mūla + NEW Swayam (engines/swayam/engine.py)              adopt if > E0 in all 3 periods
  E2  E1, with Mūla's published conviction taken from the value blend    adopt over E1 if > E1 in all 3
      (Mūla's gap osc and Samanvaya's value tape, averaged, on ±100; + = rich = bearish)

Headlines scored (both, bullish-positive): HERO = −Consensus (consensus_series — the hero card's
value); CONV = −convergence_score after the online dimension weights (the calibrated composite).
Score: Spearman IC vs the next-open 10d log return, non-overlapping (stride 10), per target per
period (P1 2019-07…2021, P2 2022-23, P3 2024…), on dates all three pairings publish; cross-target
mean, and ex-crypto."""
import os, sys, glob, pickle, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/home/user/tattva"); os.chdir("/home/user/tattva")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from core.config import get_instrument_config
from convergence.cross_validator import CrossValidator
from convergence import intelligence as intel
from convergence.normalization import consensus_series
from engines.swayam.engine import SwayamEngine
from engines.pragati import samanvaya as sv
from data.fetcher import fetch_value_drivers
SP = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad/tv"
H = 10
PER = (("P1", "2019-07-01", "2022-01-01"), ("P2", "2022-01-01", "2024-01-01"), ("P3", "2024-01-01", "2027-01-01"))
CR = {"Bitcoin", "Ethereum", "Solana", "XRP", "BNB", "Cardano", "Dogecoin"}
DRV = fetch_value_drivers("2017-09-30", "2026-09-29")


def on_calendar(sw, cal):
    if sw is None or sw.empty:
        return pd.DataFrame(index=cal)
    s = sw.copy(); s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    nat = set(s.index)
    out = s.reindex(cal, method="ffill").dropna(how="all")
    out["_Native"] = out.index.isin(nat)
    return out


def headline(fvo_ts, sw, cfg):
    v = CrossValidator(active_weights=cfg.weights_seed(), expected_constituents=None)
    by_date = {str(i.date()): r for i, r in sw.iterrows()}
    native = 0
    for ts_idx, row_a in fvo_ts.iterrows():
        if not bool(row_a.get("Valid", True)):
            continue
        ds = str(ts_idx.date())
        fsig = {"conviction_score": float(row_a.get("ConvictionBounded", 0)),
                "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
                "regime": str(row_a.get("Regime", "NEUTRAL"))}
        r = by_date.get(ds)
        if r is not None:
            st = {"oversold_pct": float(r.get("Oversold_Pct", 50)), "overbought_pct": float(r.get("Overbought_Pct", 50)),
                  "avg_unified_osc": float(r.get("Avg_Signal", 0)), "regime_bull_pct": float(r.get("Regime_Bull_Pct", 33)),
                  "regime_bear_pct": float(r.get("Regime_Bear_Pct", 33)), "regime_neutral": float(r.get("Regime_Neutral", 34)),
                  "num_constituents": int(r.get("Total_Analyzed", 0))}
            native += bool(r.get("_Native", True))
        else:
            st = {"oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0, "regime_bull_pct": 33,
                  "regime_bear_pct": 33, "regime_neutral": 34, "num_constituents": 0}
        v.compute_convergence(fsig, st, ds)
    cdf = v.get_convergence_series()
    if native >= 60:
        cdf, _ = intel.apply_online_weights(cdf, fvo_ts, horizon=H, target_col="Price")
    cons = consensus_series(fvo_ts, sw)
    cdf.index = pd.to_datetime(cdf.index)
    return (-cons["Consensus"]).rename("HERO"), (-cdf["convergence_score"]).rename("CONV")


def run(p):
    d = pickle.load(open(p, "rb"))
    tgt = d["target"]; cfg = get_instrument_config(tgt)
    m = d["mula"].copy(); m.index = pd.to_datetime(m.index)
    cal = m.index
    old = on_calendar(d["swayam"], cal)
    eng = SwayamEngine().fit(d["ohlcv"], DRV, d["ticker"])
    new = on_calendar(eng.daily, cal)
    # E2: Mūla's published conviction from the value blend (Mūla gap osc ⊕ Samanvaya value tape)
    m2 = m.copy()
    mula_osc = 100.0 * sv.softbound(sv.GAIN * pd.to_numeric(m["FVO"], errors="coerce"))
    vt = pd.to_numeric(eng.frame["v_tape"], errors="coerce").reindex(cal)
    blend = ((mula_osc + vt) / 2).where(vt.notna(), mula_osc)
    m2["ConvictionRaw"] = blend; m2["ConvictionBounded"] = blend.clip(-100, 100)
    H0 = headline(m, old, cfg); H1 = headline(m, new, cfg); H2 = headline(m2, new, cfg)
    o = d["ohlcv"].copy(); o.index = pd.to_datetime(o.index).tz_localize(None) if o.index.tz is None else o.index.tz_convert(None)
    lr = np.log(o["Open"]); fwd = (lr.shift(-(1 + H)) - lr.shift(-1)).reindex(cal)
    S = pd.DataFrame({"E0_HERO": H0[0], "E0_CONV": H0[1], "E1_HERO": H1[0], "E1_CONV": H1[1],
                      "E2_HERO": H2[0], "E2_CONV": H2[1]}).reindex(cal)
    S["fwd"] = fwd
    return tgt, S


if __name__ == "__main__":
    rows, allS = [], {}
    for p in sorted(glob.glob(f"{SP}/base/*.pkl")):
        try:
            tgt, S = run(p)
        except Exception as e:
            print("FAIL", os.path.basename(p), type(e).__name__, e, flush=True); continue
        allS[tgt] = S
        common = S.dropna().index
        for per, a, b in PER:
            idx = common[(common >= a) & (common < b)][::H]
            if len(idx) < 15:
                continue
            r = {"target": tgt, "per": per, "n": len(idx), "crypto": tgt in CR}
            for c in [c for c in S.columns if c != "fwd"]:
                r[c] = spearmanr(S.loc[idx, c], S.loc[idx, "fwd"]).correlation
            rows.append(r)
        print("done", tgt, flush=True)
    pickle.dump(allS, open(f"{SP}/e2e_S.pkl", "wb"))
    T = pd.DataFrame(rows); T.to_pickle(f"{SP}/e2e_ic.pkl")
    cols = ["E0_HERO", "E1_HERO", "E2_HERO", "E0_CONV", "E1_CONV", "E2_CONV"]
    pd.set_option("display.width", 220)
    print("\n== headline IC (10d, non-overlapping) — cross-target mean"); print(T.groupby("per")[cols].mean().round(3).to_string())
    print("\n== ex-crypto"); print(T[~T.crypto].groupby("per")[cols].mean().round(3).to_string())
    print("\n== share of targets IC > 0"); print(T.groupby("per")[cols].apply(lambda x: (x > 0).mean()).round(2).to_string())
    print("\n== targets"); print(T.groupby("per").size().to_string())
