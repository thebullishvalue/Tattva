"""
Tattva — CALIBRATION LIFT study, THREE paired arms on identical test windows:
does the Optuna-CALIBRATED composite beat the RAW factory composite — and does
either beat the NORMALIZED CONSENSUS the hero card actually headlines?

Context: the hero card headlines the NORMALIZED CONSENSUS (a separate
construction — see convergence/normalization.py), with the Optuna-calibrated
composite demoted to an evidence row. That product decision was architectural
(no fitted layer between engines and verdict, hero/plot/card reconcile by
construction) — THIS study is the empirical check: all three signals scored
on the SAME purged test blocks with the SAME non-overlapping IC, so "which
construction should headline" has a paired answer, not a cross-study guess.

Method (per target):
  1. Build the live pipeline's convergence frame: Aarambh walk-forward engine
     (Valid-gated, purged) + Nirnay basket aggregation + the CrossValidator
     per-date loop — the same construction app.py runs. Alongside it, build
     the live CONSENSUS series (causal_normalize of ConvictionRaw and
     Avg_Signal, averaged) and align it to the frame by date.
  2. TRIPLE expanding-window walk-forward over the calibration frame: at each
     cut, (a) CALIBRATED re-fits (weights, thresholds) on train via the same
     mini-TPE used by convergence.intelligence.walk_forward_ic, then scores
     the NEXT purged test block; (b) RAW scores the SAME test block with
     DEFAULT_WEIGHTS / DEFAULT_THRESHOLDS (no fitting); (c) CONSENSUS scores
     the SAME strided test rows with the same sign-flipped Spearman
     (negative = bullish, matching _score_frame's convention). All three use
     the honest NON-OVERLAPPING sampling (stride = min(hold)).
  3. Report per-target means, pooled means, and paired win-rates
     (cal vs raw, consensus vs cal, consensus vs raw).

Honest notes:
  • Same test blocks, same purge, same scoring → the comparisons are paired;
    the only differences are the constructions themselves.
  • The consensus needs no fitting, so its "train" period is unused — but it
    IS causal (expanding-z), so scoring it only on the test blocks is fair.
  • n windows per target is small (~4-6); read the POOLED verdict across
    targets, not any single target's.
  • The secondary bin-monotonicity term in the score is dropped for the
    verdict — mean IC is the primary comparison.

Run: python -u research/calibration_lift_study.py   (from the repo root —
SCRIPT mode, not -m: the study imports its research/ siblings flat, which
needs the script directory on sys.path; needs network for the data fetch,
~minutes per target).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from convergence.intelligence import (
    _build_calibration_frame,
    _optimize_frame,
    _score_frame_nonoverlap,
    DEFAULT_WEIGHTS,
    DEFAULT_THRESHOLDS,
)

# Same target set / lens as the marker study (commodities/FX + small India
# sectors; large baskets skipped for runtime).
TARGETS = ["Gold", "Copper", "Cotton", "USD/INR", "Jeera", "Nifty Bank", "Nifty IT", "Nifty Auto"]
HOLD = (5, 10)          # Tactical lens hold grid (the default lens)
N_TRIALS = 20           # per-window mini-TPE budget (matches walk_forward_ic)
L2_ALPHA = 0.10
N_CV_FOLDS = 4


def _consensus_ic(cons_block: np.ndarray, test: pd.DataFrame,
                  horizons: tuple[int, ...]) -> float:
    """Consensus arm scored EXACTLY like _score_frame_nonoverlap scores the
    composite: stride = min(horizons) subsample, per-horizon Spearman vs
    Ret_{h}b, sign-flipped (negative signal = bullish), mean over horizons."""
    stride = max(1, int(min(horizons)))
    sub = test.iloc[::stride]
    cv = np.asarray(cons_block, dtype=np.float64)[::stride]
    ics = []
    for h in horizons:
        col = f"Ret_{h}b"
        if col not in sub.columns:
            continue
        r = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(cv) & np.isfinite(r)
        if m.sum() < 8:
            continue
        ic = spearmanr(cv[m], r[m])[0]
        if not np.isnan(ic):
            ics.append(-ic)          # negative = bullish → flip, as _score_frame does
    return float(np.mean(ics)) if ics else float("nan")


def dual_walk_forward(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HOLD,
    n_splits: int = 6,
    min_train_frac: float = 0.45,
    n_trials: int = N_TRIALS,
    l2_alpha: float = L2_ALPHA,
    n_cv_folds: int = N_CV_FOLDS,
    cons: np.ndarray | None = None,
) -> list[dict]:
    """Paired expanding-window walk-forward: calibrated vs raw (vs consensus,
    when ``cons`` — positionally aligned to ``frame`` — is given) on IDENTICAL
    purged test blocks. Mirrors convergence.intelligence.walk_forward_ic's
    windowing exactly (same start/step/purge).

    Returns rows: {"test_start", "n_test", "ic_cal", "ic_raw", "ic_cons"}.
    """
    n = len(frame)
    if n < 250:
        return []
    purge = int(max(horizons))
    start = max(60, int(n * min_train_frac))
    span = n - purge - start
    if span < 60:
        return []
    n_splits = max(2, min(int(n_splits), span // 30))
    step = max(30, span // n_splits)

    out: list[dict] = []
    cut = start
    while cut + purge + 20 <= n:
        train = frame.iloc[:cut]
        test = frame.iloc[cut + purge: cut + purge + step]
        if len(test) >= 20:
            w_cal, t_cal = _optimize_frame(train, horizons, n_trials, l2_alpha, n_cv_folds)
            ic_cal, _ = _score_frame_nonoverlap(test, w_cal, t_cal, horizons)
            ic_raw, _ = _score_frame_nonoverlap(
                test, DEFAULT_WEIGHTS.copy(), DEFAULT_THRESHOLDS.copy(), horizons)
            ic_cons = (_consensus_ic(cons[cut + purge: cut + purge + step], test, horizons)
                       if cons is not None else float("nan"))
            try:
                ts = frame.index[cut + purge]
            except Exception:
                ts = cut + purge
            out.append({
                "test_start": ts, "n_test": int(len(test)),
                "ic_cal": float(ic_cal) if not np.isnan(ic_cal) else float("nan"),
                "ic_raw": float(ic_raw) if not np.isnan(ic_raw) else float("nan"),
                "ic_cons": float(ic_cons) if not np.isnan(ic_cons) else float("nan"),
            })
        cut += step
    return out


# Shared per-target (ts, nd) memo — the frame AND the consensus arm both need
# them; without this the engine is fit twice per target.
_TSND: dict[str, tuple] = {}


def _ts_nd(target: str):
    if target not in _TSND:
        from markers_study import _aarambh_ts, _nirnay_daily
        _TSND[target] = (_aarambh_ts(target), _nirnay_daily(target))
    return _TSND[target]


def _consensus_series(target: str) -> pd.Series | None:
    """The hero's live headline object: causal expanding-z of ConvictionRaw and
    Avg_Signal, averaged — date-indexed for alignment to the frame."""
    from convergence.normalization import align_aarambh_nirnay, causal_normalize
    ts, nd = _ts_nd(target)
    if ts is None or nd is None or nd.empty:
        return None
    dates, raw_a, raw_n = align_aarambh_nirnay(ts, nd)
    if len(dates) < 100:
        return None
    v = (causal_normalize(np.array(raw_a, dtype=float))
         + causal_normalize(np.array(raw_n, dtype=float))) / 2.0
    return pd.Series(v, index=pd.to_datetime(dates))


def _convergence_frame(target: str) -> pd.DataFrame | None:
    """Build the live pipeline's calibration frame for a target: Aarambh engine
    → Nirnay aggregate → CrossValidator per-date loop (Valid-gated, matching
    app.py's convergence scoring) → dim_*/consensus/Ret_{h}b frame."""
    # Imported lazily so dual_walk_forward stays importable/testable without
    # the heavy data/engine stack.
    from convergence.cross_validator import CrossValidator

    ts, nd = _ts_nd(target)
    if ts is None or nd is None or nd.empty:
        return None

    nd = nd[~nd.index.duplicated(keep="last")]
    nirnay_by_date = {
        (str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())): nd.loc[idx]
        for idx in nd.index
    }

    validator = CrossValidator(expected_constituents=None)
    ts = ts[~ts.index.duplicated(keep="last")]
    for idx in ts.index:
        row_a = ts.loc[idx]
        if isinstance(row_a, pd.DataFrame):
            row_a = row_a.iloc[-1]
        if not bool(row_a.get("Valid", True)):   # engine warm-up — no genuine forecast
            continue
        date_str = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
        aarambh_sig = {
            "conviction_score": float(row_a.get("ConvictionBounded", 0)),
            "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
            "regime": str(row_a.get("Regime", "NEUTRAL")),
        }
        row_n = nirnay_by_date.get(date_str)
        if row_n is not None:
            nirnay_stats = {
                "oversold_pct": float(row_n.get("Oversold_Pct", 50)),
                "overbought_pct": float(row_n.get("Overbought_Pct", 50)),
                "avg_unified_osc": float(row_n.get("Avg_Signal", 0)),
                "regime_bull_pct": float(row_n.get("Regime_Bull_Pct", 33)),
                "regime_bear_pct": float(row_n.get("Regime_Bear_Pct", 33)),
                "regime_neutral": float(row_n.get("Regime_Neutral", 34)),
                "num_constituents": int(row_n.get("Total_Analyzed", 0)),
            }
        else:
            nirnay_stats = {
                "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                "regime_bull_pct": 33, "regime_bear_pct": 33,
                "regime_neutral": 34, "num_constituents": 0,
            }
        validator.compute_convergence(aarambh_sig, nirnay_stats, date_str)

    conv_df = validator.get_convergence_series()
    if conv_df.empty:
        return None
    a_ts = ts.copy()
    a_ts["Date"] = a_ts.index
    frame = _build_calibration_frame(conv_df, a_ts, target_col="Price", horizons=HOLD)
    return frame if not frame.empty else None


def main() -> None:
    from markers_study import _load
    _load()
    print(f"Tattva — CALIBRATION LIFT study · {len(TARGETS)} targets · "
          f"hold {HOLD} · {N_TRIALS} trials/window", flush=True)

    all_rows: list[dict] = []
    per_tgt: dict[str, list[dict]] = {}
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        try:
            frame = _convergence_frame(tgt)
        except Exception as e:
            frame = None
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
        if frame is None or len(frame) < 250:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped "
                  f"({0 if frame is None else len(frame)} frame rows)", flush=True)
            continue
        cons_arr = None
        cs = _consensus_series(tgt)
        if cs is not None:
            fidx = pd.to_datetime(frame.index, errors="coerce")
            cons_arr = cs.reindex(fidx).to_numpy(dtype=float)
            if not np.isfinite(cons_arr).any():
                cons_arr = None          # index alignment failed — cons arm NaN
        rows = dual_walk_forward(frame, HOLD, cons=cons_arr)
        rows = [r for r in rows
                if not (np.isnan(r["ic_cal"]) or np.isnan(r["ic_raw"]))]
        if not rows:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} no scorable windows", flush=True)
            continue
        per_tgt[tgt] = rows
        all_rows += rows
        m_cal = np.mean([r["ic_cal"] for r in rows])
        m_raw = np.mean([r["ic_raw"] for r in rows])
        m_con = np.nanmean([r["ic_cons"] for r in rows])
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(rows)} windows · "
              f"cal {m_cal:+.3f} vs raw {m_raw:+.3f} vs cons {m_con:+.3f}", flush=True)

    if not all_rows:
        raise SystemExit("no data")

    ic_cal = np.array([r["ic_cal"] for r in all_rows])
    ic_raw = np.array([r["ic_raw"] for r in all_rows])
    ic_con = np.array([r["ic_cons"] for r in all_rows])
    diff = ic_cal - ic_raw
    wins = int((diff > 0).sum())
    n = len(diff)

    print(f"\n  pooled {n} paired windows in {time.time()-t0:.0f}s")
    print("\n" + "=" * 72)
    print("  HEADLINE ARMS — consensus vs raw vs calibrated, identical purged test windows")
    print("=" * 72)
    print(f"  {'target':<14} {'win':>4} {'cons IC':>9} {'raw IC':>9} {'cal IC':>9} {'cal-raw':>9}")
    print("  " + "-" * 60)
    for tgt, rows in per_tgt.items():
        mc = np.mean([r["ic_cal"] for r in rows])
        mr = np.mean([r["ic_raw"] for r in rows])
        mo = np.nanmean([r["ic_cons"] for r in rows])
        print(f"  {tgt:<14} {len(rows):>4} {mo:>+9.3f} {mr:>+9.3f} {mc:>+9.3f} {mc - mr:>+9.3f}")
    print("  " + "-" * 60)
    print(f"  {'POOLED':<14} {n:>4} {np.nanmean(ic_con):>+9.3f} {ic_raw.mean():>+9.3f} "
          f"{ic_cal.mean():>+9.3f} {diff.mean():>+9.3f}")

    print(f"\n  paired win-rate (calibrated > raw): {wins}/{n} windows ({wins / n:.0%})")
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"  mean cal-raw lift {diff.mean():+.3f} ± {se:.3f} (SE, {n} paired windows — "
          f"windows overlap in training data, so treat the SE as optimistic)")
    mfin = np.isfinite(ic_con)
    if mfin.sum() >= 8:
        d_cc = ic_con[mfin] - ic_cal[mfin]
        d_cr = ic_con[mfin] - ic_raw[mfin]
        se_cc = d_cc.std(ddof=1) / np.sqrt(len(d_cc))
        se_cr = d_cr.std(ddof=1) / np.sqrt(len(d_cr))
        print(f"  paired win-rate (consensus > calibrated): "
              f"{int((d_cc > 0).sum())}/{len(d_cc)} ({(d_cc > 0).mean():.0%}) · "
              f"mean diff {d_cc.mean():+.3f} ± {se_cc:.3f}")
        print(f"  paired win-rate (consensus > raw):        "
              f"{int((d_cr > 0).sum())}/{len(d_cr)} ({(d_cr > 0).mean():.0%}) · "
              f"mean diff {d_cr.mean():+.3f} ± {se_cr:.3f}")
    print("\n  VERDICT GUIDE — the hero currently headlines the CONSENSUS (product decision):")
    print("  • calibrated clearly > consensus across targets → calibration earns the")
    print("    headline back; revisit the consensus-primary default.")
    print("  • otherwise → consensus-primary stands; the calibrated composite stays an")
    print("    evidence row (its Val IC still reported, attributed).")


if __name__ == "__main__":
    main()
