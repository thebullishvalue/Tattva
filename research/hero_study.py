"""
Tattva — does the HERO CARD actually predict, and does richer interpretation help?

The hero headline is a direction = sign of the convergence conviction. We test three
nested interpretation logics as bullish-oriented directional scores (positive = the
card should read BULLISH → expect a positive forward return), scored by HONEST
NON-OVERLAPPING OOS rank IC + directional hit-rate vs the target's +10d return,
pooled across the universe:

  • A (current):        convergence consensus only            bull = −norm_avg
  • B (+ markers):      blend the 3 plot-marker rows           bull = z-mean(−norm_avg, −ConvictionRaw, −Avg_Signal)
  • C (+ precedent):    B + the analog base-rate prediction    bull = z-mean(B-parts, analog_fwd)

Honest notes baked in:
  • The markers ARE the convergence's own inputs (ConvictionRaw=Aarambh breadth,
    Avg_Signal=Nirnay breadth, norm_avg=their blend) → B is expected ≈ A. The
    precedent analog is the one genuinely independent signal.
  • Model A here is the uncalibrated 50/50 consensus (the hero's fallback / a faithful
    DIRECTION proxy); the live calibrated conviction reweights the dims but its
    directional content is similar. We avoid re-running Optuna ×N.
  • Non-overlapping (stride=10) → no overlap inflation.
  • norm_avg uses convergence.normalization's causal_normalize (expanding-
    window z-score) — a previous revision here used the terminal-point
    compute_norm_params/zscore_clip pair, a look-ahead: applying the FULL-
    SAMPLE mean/std to every historical point means earlier bars appear less
    extreme than they were at the time, since sigma is estimated from data
    that didn't yet exist then (audit finding F14). The within-target `_z()`
    re-standardization applied on top does NOT remove this — it operates on
    the already-look-ahead-biased norm_avg values, not the raw series.

Run: python3 -u hero_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← → · and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from core.config import MIN_TRAIN_SIZE
from convergence.normalization import align_aarambh_nirnay, causal_normalize
from analytics.analogs import _build_feature_frame, mahalanobis_distance_batch, select_analogs_theiler
from markers_study import _aarambh_ts, _nirnay_daily, _load, TARGETS

H, MOM = 10, 20
TOP_N = 10
W_MAHA, W_TRAJ, W_RECV = 0.55, 0.35, 0.10


def _z(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-9 else np.zeros_like(x)


def _analog_by_pos(ts):
    """Non-overlapping analog median +Hd prediction, keyed by ts row position."""
    feat, fcols = _build_feature_frame(ts, MOM)
    F = feat[fcols].to_numpy(float)
    for j in range(F.shape[1]):
        c = F[:, j]; ok = np.isfinite(c)
        F[~ok, j] = np.median(c[ok]) if ok.any() else 0.0
    price = feat["Price"].to_numpy(float)
    dates = pd.to_datetime(feat["Date"]).to_numpy()
    n = len(F); tw = MOM
    xm = np.arange(tw, dtype=float); xm -= xm.mean(); xv = np.sum(xm ** 2)
    Tn = np.zeros((n, tw))
    for i in range(tw, n):
        seg = price[i - tw:i]
        slope = np.sum(xm * (seg - seg.mean())) / xv if xv > 1e-12 else 0.0
        d = seg - (seg.mean() + slope * xm); nm = np.linalg.norm(d)
        if nm > 1e-12:
            Tn[i] = d / nm
    out = {}
    # Start no earlier than MIN_TRAIN_SIZE: before that the engine's own
    # forecast/breadth features are unfit (NaN) — see the audit's A3 fix.
    for t in range(max(MIN_TRAIN_SIZE, tw + 30), n - H, H):
        he = t + 1 - H
        if he < 30:
            continue
        Fh = F[:he]
        cov = np.cov(Fh, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[max(float(cov), 1e-6)]])
        dd = mahalanobis_distance_batch(Fh, F[t], cov)
        dmax = dd.max() if dd.max() > 0 else 1.0
        maha = 1.0 - dd / dmax
        traj = (Tn[:he] @ Tn[t] + 1.0) / 2.0; traj[:tw] = 0.0
        ds = (pd.Timestamp(dates[t]) - pd.to_datetime(dates[:he])).days.to_numpy(float)
        rec = np.exp(-np.log(2) * np.clip(ds, 0, None) / 365.0) * W_RECV
        rec /= max(rec.max(), 1e-6)
        score = W_MAHA * maha + W_TRAJ * traj + W_RECV * rec
        # Theiler exclusion window (audit finding A5) — see
        # analytics.analogs.select_analogs_theiler's docstring.
        top = select_analogs_theiler(score, TOP_N, max(tw, H, 1))
        fa = [(price[p + H] / price[p] - 1) * 100 for p in top if price[p] > 0]
        if fa:
            out[t] = float(np.median(fa))
    return out


def target_rows(target):
    ts = _aarambh_ts(target)
    nd = _nirnay_daily(target)
    if ts is None or nd is None or nd.empty:
        return None
    ts = ts.copy(); ts["Date"] = ts.index
    dates, raw_a, raw_n = align_aarambh_nirnay(ts, nd)
    if len(dates) < 100:
        return None
    raw_a = np.array(raw_a, float); raw_n = np.array(raw_n, float)
    norm_avg = (causal_normalize(raw_a) + causal_normalize(raw_n)) / 2.0
    # date → marker values
    mk = {pd.Timestamp(d): (norm_avg[i], raw_a[i], raw_n[i]) for i, d in enumerate(dates)}
    # within-target z of each marker (bullish-oriented = negated)
    zA = _z(-norm_avg); zRA = _z(-raw_a); zRN = _z(-raw_n)
    zmk = {pd.Timestamp(d): (zA[i], zRA[i], zRN[i]) for i, d in enumerate(dates)}

    analog = _analog_by_pos(ts)                      # ts-position → analog +Hd pred
    pr = ts["Price"].astype(float).to_numpy()
    idx = list(ts.index)

    rows = []
    for t, apred in analog.items():
        d = pd.Timestamp(idx[t])
        if d not in zmk or not (pr[t] > 0 and t + H < len(pr)):
            continue
        za, zra, zrn = zmk[d]
        fwd = (pr[t + H] / pr[t] - 1) * 100
        rows.append((za, zra, zrn, apred, fwd))
    return rows


def _ic(score, fwd):
    score = np.asarray(score); fwd = np.asarray(fwd)
    m = np.isfinite(score) & np.isfinite(fwd)
    s, f = score[m], fwd[m]
    if len(s) < 20:
        return np.nan, np.nan, 0
    ic = spearmanr(s, f)[0]
    hit = np.mean(np.sign(s) == np.sign(f)) * 100
    return ic, hit, len(s)


def main():
    _load()
    print(f"Tattva — Hero interpretation study · {len(TARGETS)} targets · "
          f"non-overlapping +{H}d", flush=True)
    allrows = []
    per_tgt = {}
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        try:
            r = target_rows(tgt)
        except Exception as e:
            r = None
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
        if not r:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True)
            continue
        per_tgt[tgt] = r
        allrows += r
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(r)} as-of dates", flush=True)

    arr = np.array(allrows, float)   # cols: zA, zRA, zRN, apred, fwd
    zA, zRA, zRN, apred, fwd = arr.T
    bull_A = zA
    bull_B = np.nanmean(np.vstack([zA, zRA, zRN]), axis=0)
    bull_C = np.nanmean(np.vstack([zA, zRA, zRN, _z(apred)]), axis=0)

    print(f"\n  pooled {len(arr)} as-of obs in {time.time()-t0:.0f}s")
    print("\n" + "=" * 66)
    print("  HERO INTERPRETATION — honest non-overlapping OOS (pooled)")
    print("=" * 66)
    print(f"  {'model':<26} {'IC':>8} {'hit%':>7} {'n':>7}")
    print("  " + "-" * 52)
    for name, sc in (("A · convergence (current)", bull_A),
                     ("B · + markers", bull_B),
                     ("C · + markers + precedent", bull_C),
                     ("  (precedent alone)", _z(apred))):
        ic, hit, n = _ic(sc, fwd)
        print(f"  {name:<26} {ic:>+8.3f} {hit:>7.1f} {n:>7}")
    print("  " + "-" * 52)
    print("  IC>0 = bullish score predicts positive forward return (correct direction).")
    print("  base rate hit% ≈ 50; markers ARE convergence inputs so B≈A is expected.")


if __name__ == "__main__":
    main()
