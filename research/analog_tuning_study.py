"""
Tattva — comprehensive tuning of the PRECEDENT (analog) engine.

The analog is now the system's primary directional edge (post-purge the Aarambh
model IC ≈ 0). Every analog knob is ported from Arthagati & un-validated for Tattva.
This sweeps them honestly (non-overlapping OOS IC, full + recent-half) across the
universe, at horizons 1 / 10 / 20d.

Cheap: the analog only needs the Aarambh ts_data (one fit per target) — then every
config is a fast re-walk on the same series (no refits).

Dimensions (one-factor-at-a-time around the current defaults):
  • blend weights  W_MAHA / W_TRAJ / W_RECV   (currently .55/.35/.10)
  • TOP_N          analogs averaged           (currently 10)
  • recency        half-life days             (currently 365)
  • features       ablation + new Tattva features (ModelSpread, ExtremeBreadth,
                   SignalBreadth, ConvictionRaw, MomentumLong)
  • aggregation    equal-weight median vs similarity-weighted mean

Run: python3 -u analog_tuning_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, trim_mean
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import MIN_TRAIN_SIZE
from analytics.analogs import (
    _rolling_hurst, mahalanobis_distance_batch, select_analogs_theiler,
    ANALOG_W_MAHA, ANALOG_W_TRAJ, ANALOG_W_RECV,
)
from markers_study import _aarambh_ts, _load

TARGETS = ["Gold", "Silver", "Copper", "Cotton", "Brent Crude", "USD/INR", "Jeera",
           "Nifty 50", "Nifty Bank", "Nifty IT", "Nifty Metal", "S&P 500", "Nasdaq 100"]
HORIZONS = [1, 10, 20]
MOM = {1: 10, 10: 20, 20: 40}
BASE_FEATS = ["Momentum", "RealizedVol", "AvgZ", "NetBreadth", "Hurst"]
NEW_FEATS = ["ModelSpread", "ExtremeBreadth", "SignalBreadth", "ConvictionRaw", "MomentumLong"]
# Baseline blend weights read from LIVE config (analytics.analogs), NOT hardcoded —
# a hardcoded 0.55/0.35/0.10 here silently mislabelled the ←current row and its sweeps
# once the shipped weights moved to maha-only (1/0/0). top_n=10 matches the analogs
# default; hl only bites in the recency sweep (production W_RECV=0 turns recency off).
# agg ∈ {"median", "mean", "trim", "sim"} — how the top-N analog outcomes are
# reduced to one prediction ("median" is the shipped behavior; "sim" is the old
# sim=True similarity-weighted mean; "trim" is a 20% trimmed mean).
DEF = dict(wm=ANALOG_W_MAHA, wt=ANALOG_W_TRAJ, wr=ANALOG_W_RECV,
           top_n=10, hl=365.0, agg="median", feats=tuple(BASE_FEATS))
# Round for a robust float match when marking the ←current blend row.
_CUR_BLEND = (round(ANALOG_W_MAHA, 3), round(ANALOG_W_TRAJ, 3), round(ANALOG_W_RECV, 3))

_TS = {}
def _ts(target):
    if target not in _TS:
        try:
            _TS[target] = _aarambh_ts(target)
        except Exception:
            _TS[target] = None
    return _TS[target]


_FS = {}
def _superset(target, mom):
    key = (target, mom)
    if key in _FS:
        return _FS[key]
    ts = _ts(target)
    if ts is None:
        _FS[key] = None; return None
    price = pd.to_numeric(ts["Price"], errors="coerce").to_numpy(float)
    lr = pd.Series(np.log(np.where(price > 0, price, np.nan))).diff()
    def col(name, default=0.0):
        return pd.to_numeric(ts[name], errors="coerce").to_numpy(float) if name in ts.columns \
            else np.full(len(price), default)
    F = {
        "Momentum": lr.rolling(mom, min_periods=mom).sum().to_numpy(),
        "RealizedVol": lr.rolling(mom, min_periods=mom).std().to_numpy(),
        "AvgZ": col("AvgZ"),
        "NetBreadth": col("OversoldBreadth") - col("OverboughtBreadth"),
        "Hurst": _rolling_hurst(price, window=max(60, mom * 3)),
        "ModelSpread": col("ModelSpread"),
        "ExtremeBreadth": col("ExtremeOversold") - col("ExtremeOverbought"),
        "SignalBreadth": col("BuySignalBreadth") - col("SellSignalBreadth"),
        "ConvictionRaw": col("ConvictionRaw"),
        "MomentumLong": lr.rolling(mom * 2, min_periods=mom * 2).sum().to_numpy(),
    }
    # trajectory matrix (detrended unit price path), tw = mom
    n = len(price); tw = mom
    xm = np.arange(tw, dtype=float); xm -= xm.mean(); xv = np.sum(xm ** 2)
    Tn = np.zeros((n, tw))
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xv if xv > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm); nm = np.linalg.norm(d)
        if nm > 1e-12:
            Tn[i] = d / nm
    _FS[key] = (F, Tn, price, pd.to_datetime(ts.index).to_numpy())
    return _FS[key]


def _walk_ic(target, h, cfg):
    sup = _superset(target, MOM[h])
    if sup is None:
        return np.nan, np.nan
    F, Tn, price, dates = sup
    feats = cfg["feats"]
    M = np.column_stack([F[f] for f in feats])
    for j in range(M.shape[1]):
        c = M[:, j]; ok = np.isfinite(c)
        M[~ok, j] = np.median(c[ok]) if ok.any() else 0.0
    n = len(price); tw = MOM[h]
    wm, wt, wr, top_n, hl, agg_mode = (cfg["wm"], cfg["wt"], cfg["wr"],
                                       cfg["top_n"], cfg["hl"], cfg["agg"])
    preds, reals = [], []
    # Start no earlier than MIN_TRAIN_SIZE: before that the engine's own
    # forecast/breadth features are unfit (NaN) — see the audit's A3 fix.
    for t in range(max(MIN_TRAIN_SIZE, tw + 30), n - h, h):
        he = t + 1 - h
        if he < 30:
            continue
        Fh = M[:he]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dd = mahalanobis_distance_batch(Fh, M[t], cov)
        dmax = dd.max() if dd.max() > 0 else 1.0
        maha = 1.0 - dd / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0; traj[:tw] = 0.0
        ds = (pd.Timestamp(dates[t]) - pd.to_datetime(dates[:he])).days.to_numpy(float)
        rec = np.exp(-np.log(2) * np.clip(ds, 0, None) / hl); rec /= max(rec.max(), 1e-6)
        sc = wm * maha + wt * traj + wr * rec
        # Theiler exclusion window (audit finding A5): argpartition's plain
        # top-N returns adjacent-day runs from 1-3 episodes whose h-day
        # forward outcomes overlap almost completely — "top_n analogs" that
        # are really far fewer independent observations.
        top = select_analogs_theiler(sc, top_n, max(tw, h, 1))
        valid = [p for p in top if price[p] > 0 and p + h < n]
        if not valid:
            continue
        fr = np.array([(price[p + h] / price[p] - 1) * 100 for p in valid])
        if agg_mode == "sim":
            w = np.clip(sc[valid], 0, None)
            pred = float(np.sum(w * fr) / max(np.sum(w), 1e-9))
        elif agg_mode == "mean":
            pred = float(np.mean(fr))
        elif agg_mode == "trim":
            # 20% trimmed mean; falls back to median when too few analogs to trim.
            pred = float(trim_mean(fr, 0.2)) if len(fr) >= 5 else float(np.median(fr))
        else:
            pred = float(np.median(fr))
        if price[t] > 0 and t + h < n:
            preds.append(pred); reals.append((price[t + h] / price[t] - 1) * 100)
    p, r = np.array(preds), np.array(reals)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 12:
        return np.nan, np.nan
    pp, rr = p[m], r[m]
    full = spearmanr(pp, rr)[0]
    half = len(pp) // 2
    recent = spearmanr(pp[half:], rr[half:])[0] if len(pp) - half >= 8 else np.nan
    return full, recent


def agg(cfg, h):
    fs, rs = [], []
    for t in TARGETS:
        f, r = _walk_ic(t, h, cfg)
        if np.isfinite(f):
            fs.append(f)
        if np.isfinite(r):
            rs.append(r)
    return (np.mean(fs) if fs else np.nan), (np.mean(rs) if rs else np.nan)


def _cfg(**kw):
    c = dict(DEF); c.update(kw); return c


def line(label, cfg, horizons, mark=""):
    cells = []
    for h in horizons:
        f, r = agg(cfg, h)
        cells.append(f"{f:+.3f}/{r:+.3f}")
    print(f"  {label:<22} " + "  ".join(f"{c:>14}" for c in cells) + mark, flush=True)


def main():
    _load()
    print(f"Analog tuning · {len(TARGETS)} targets · cells = full_IC / recent_IC", flush=True)

    print("\n### BASELINE (current defaults) — horizons 1 / 10 / 20", flush=True)
    print(f"  {'config':<22} " + "  ".join(f"{('+'+str(h)+'d'):>14}" for h in HORIZONS), flush=True)
    line("current", _cfg(), HORIZONS, "  ←current")

    H2 = [10, 20]
    print(f"\n### BLEND  W_MAHA/W_TRAJ/W_RECV  (horizons 10 / 20)", flush=True)
    print(f"  {'config':<22} " + "  ".join(f"{('+'+str(h)+'d'):>14}" for h in H2), flush=True)
    # Densified 2026-07-12: fine steps around the shipped maha-only blend plus
    # the traj/recency-heavy corners, so the optimum is bracketed, not guessed.
    for wm, wt, wr in [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.95, 0.05, 0.0), (0.9, 0.1, 0.0), (0.9, 0.05, 0.05),
        (0.85, 0.15, 0.0), (0.8, 0.2, 0.0), (0.8, 0.1, 0.1),
        (0.7, 0.3, 0.0), (0.7, 0.2, 0.1), (0.6, 0.4, 0.0), (0.6, 0.3, 0.1),
        (0.55, 0.35, 0.10), (0.5, 0.5, 0.0), (0.5, 0.25, 0.25),
        (0.45, 0.30, 0.25), (0.4, 0.5, 0.1), (0.33, 0.33, 0.34),
        (0.25, 0.5, 0.25), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)]:
        line(f"{wm}/{wt}/{wr}", _cfg(wm=wm, wt=wt, wr=wr), H2,
             "  ←current" if (round(wm, 3), round(wt, 3), round(wr, 3)) == _CUR_BLEND else "")

    print(f"\n### TOP_N  (horizons 10 / 20)", flush=True)
    # 2026-07-13: 1..10 every step (the sensitive region), then out to 150.
    for tn in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150]:
        line(f"top_n={tn}", _cfg(top_n=tn), H2, "  ←current" if tn == 10 else "")

    print(f"\n### RECENCY half-life days  (horizons 10 / 20)", flush=True)
    # 2026-07-13: widened 15d (very recent-weighted) → 3000d (near-flat).
    for hl in [15, 30, 45, 60, 90, 120, 150, 180, 250, 300, 365, 450, 500, 600,
               730, 1000, 1500, 2000, 3000]:
        line(f"halflife={hl}", _cfg(hl=hl, wr=0.20), H2, "")   # at wr=0.20 so recency bites
    print("    (tested at W_RECV=0.20 so the half-life actually matters)", flush=True)

    print(f"\n### FEATURES  (horizons 10 / 20)", flush=True)
    line("base5", _cfg(), H2, "  ←current")
    for f in BASE_FEATS:
        line(f"drop {f}", _cfg(feats=tuple(x for x in BASE_FEATS if x != f)), H2)
    # Pairwise drops: OFAT single-drops can't see redundancy between two features
    # (dropping either alone looks costless when they duplicate each other).
    for i in range(len(BASE_FEATS)):
        for j in range(i + 1, len(BASE_FEATS)):
            fi, fj = BASE_FEATS[i], BASE_FEATS[j]
            line(f"drop {fi[:6]}+{fj[:6]}",
                 _cfg(feats=tuple(x for x in BASE_FEATS if x not in (fi, fj))), H2)
    for f in NEW_FEATS:
        line(f"+ {f}", _cfg(feats=tuple(BASE_FEATS) + (f,)), H2)
    line("base + all-new", _cfg(feats=tuple(BASE_FEATS) + tuple(NEW_FEATS)), H2)

    print(f"\n### AGGREGATION  (horizons 10 / 20)", flush=True)
    line("median (equal)", _cfg(agg="median"), H2, "  ←current")
    line("mean (equal)", _cfg(agg="mean"), H2)
    line("trimmed mean 20%", _cfg(agg="trim"), H2)
    line("similarity-weighted", _cfg(agg="sim"), H2)

    print("\n  Read each cell as full_IC / recent_IC. recent_IC is the honest test of"
          "\n  whether the analog still works in the CURRENT regime.", flush=True)

    # ── PER-INSTRUMENT analog blend (gated: best blend beats this target's OWN
    #    current-blend IC beyond noise) → analog_w_maha/traj/recv overrides ──
    from core.config import InstrumentConfig as _IC
    from research._per_instrument import IC_FLOOR, IC_MARGIN, print_overrides_snippet
    _blends = [(1.0, 0.0, 0.0), (0.9, 0.1, 0.0), (0.8, 0.2, 0.0), (0.8, 0.1, 0.1),
               (0.7, 0.3, 0.0), (0.6, 0.4, 0.0), (0.55, 0.35, 0.10), (0.5, 0.5, 0.0),
               (0.5, 0.25, 0.25), (0.33, 0.33, 0.34), (0.2, 0.6, 0.2)]
    _cur = (round(_IC().analog_w_maha, 3), round(_IC().analog_w_traj, 3), round(_IC().analog_w_recv, 3))
    print("\n" + "=" * 74)
    print(f"  PER-INSTRUMENT ANALOG BLEND @ +10d (gate |IC|≥{IC_FLOOR} & beat current by ≥{IC_MARGIN})")
    print("=" * 74)
    overrides: dict = {}
    for t in TARGETS:
        ics = {(wm, wt, wr): _walk_ic(t, 10, _cfg(wm=wm, wt=wt, wr=wr))[0]
               for (wm, wt, wr) in _blends}
        ics = {k: v for k, v in ics.items() if np.isfinite(v)}
        if not ics:
            continue
        cur_ic = ics.get(_cur, np.nan)
        (bwm, bwt, bwr), bic = max(ics.items(), key=lambda kv: kv[1])
        adopt = ((bwm, bwt, bwr) != _cur and bic >= IC_FLOOR
                 and (not np.isfinite(cur_ic) or bic - cur_ic >= IC_MARGIN))
        flag = "  ADOPT" if adopt else "  (noise → current)"
        print(f"    {t:<22} best={bwm}/{bwt}/{bwr}  |IC|={bic:+.3f}  current|IC|={cur_ic:+.3f}{flag}")
        if adopt:
            overrides[t] = {"analog_w_maha": bwm, "analog_w_traj": bwt, "analog_w_recv": bwr}
    print_overrides_snippet(overrides)


if __name__ == "__main__":
    main()
