"""
Tattva — UI/TIER ANCHOR study: data-anchor EVERY remaining hand-set tier constant.

The markers study anchored the Unified-Signal plot rows; the hero-threshold
study anchored the action classifiers. This study covers the REST of the
hand-set tier constants — the last "mental assertion" numbers in the display/
classification layer — by pooling each constant's OWN live signal distribution
across targets and reporting percentiles + current occupancy + the p75/p90
convention anchor (single-tier constants anchor at the percentile matching
their role). Where a forward return exists, a quintile edge check is printed
(informational — these are extremeness tiers, not edges).

Constants covered (signal → constant(s)):
  ConvictionBounded |x|      → CONVICTION_WEAK/MODERATE/STRONG
  ModelSpread (bps)          → UI_MODEL_SPREAD_LOW / UI_MODEL_SPREAD_HIGH
  CI band width (U−L)        → (informational only — the UI_BAND_* tier was
                                removed 2026-07-12 after this study measured
                                the distribution degenerate; the row remains
                                so a future regime change would be visible)
  OversoldBreadth/Overbought → UI_BREADTH_HIGH (60%)
  agreement_ratio            → UI_AGREEMENT_MODERATE / UI_AGREEMENT_STRONG (.5/.7)
  convergence_score |x|      → CONV_WEAK/MODERATE/STRONG ±10/±30/±60 (legacy tiers)
  Nirnay Avg_Signal          → UI_NIRNAY_BULLISH/BEARISH (±2)
  Nirnay per-instrument osc  → NIRNAY_OVERSOLD/OVERBOUGHT (±5)

Run: python -u research/ui_anchors_study.py   (script mode from repo root)
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from core.config import (
    CONVICTION_MODERATE, CONVICTION_STRONG,
    UI_MODEL_SPREAD_LOW, UI_MODEL_SPREAD_HIGH,
    UI_BREADTH_HIGH,
    UI_AGREEMENT_MODERATE, UI_AGREEMENT_STRONG,
    CONV_WEAK_BULLISH, CONV_MODERATE_BULLISH, CONV_STRONG_BULLISH,
    UI_NIRNAY_BULLISH, UI_NIRNAY_BEARISH,
    NIRNAY_OVERSOLD, NIRNAY_OVERBOUGHT,
)
from calibration_lift_study import _ts_nd, _convergence_frame, TARGETS

H = 10  # forward lens for the (informational) edge check

_PCTS = (25, 40, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5)


def _dist_report(name: str, vals: np.ndarray, cur: dict[str, float],
                 fwd: np.ndarray | None = None, absolute: bool = True) -> None:
    """Percentile table + current-tier occupancy (+ optional forward-IC check)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        print(f"\n  {name}: NO DATA"); return
    a = np.abs(v) if absolute else v
    qs = {p: float(np.quantile(a, p / 100)) for p in _PCTS}
    print(f"\n  {name}  (n={len(v)})")
    # %.4g, not %.2f — a fixed 2-decimal print flattened the ModelSpread scale
    # (~1e-3) to a wall of 0.00 and hid a dead-scale tier constant.
    print("    |x| percentiles:  " + "  ".join(f"p{p:g}={qs[p]:.4g}" for p in _PCTS))
    for label, thr in cur.items():
        occ = float(np.mean(a >= abs(thr)) * 100)
        # nearest percentile of the current threshold
        pnear = float(np.mean(a <= abs(thr)) * 100)
        print(f"    current {label} = {thr}  → {occ:5.1f}% of obs beyond (≈p{pnear:.0f})")
    if fwd is not None:
        f = np.asarray(fwd, dtype=float)
        m = np.isfinite(v[:len(f)]) & np.isfinite(f[:len(v)])
        if m.sum() > 100:
            ic = spearmanr(v[:len(f)][m], f[:len(v)][m])[0]
            print(f"    forward +{H}d IC (informational): {ic:+.3f}")


def main() -> None:
    from markers_study import _load
    _load()
    print(f"Tattva — UI/tier anchor study · {len(TARGETS)} targets · "
          f"pooled live distributions", flush=True)

    conv_b, spread, bandw, osb, agree, cscore = [], [], [], [], [], []
    avg_sig = []
    fwd_conv = []
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        ts, nd = _ts_nd(tgt)
        if ts is None:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True); continue
        ok = ts.get("Valid")
        tsv = ts[ok.astype(bool)] if ok is not None else ts
        conv_b.append(pd.to_numeric(tsv.get("ConvictionBounded"), errors="coerce").to_numpy(float))
        # ×1e4: the UI tile (tab_aarambh) displays ModelSpread in BASIS POINTS;
        # the raw column is a log-return std (~1e-3). Pooling the raw units
        # against the bps-scaled constants produced a false "dead scale" read
        # on the first run of this study. CAVEAT: this pool inherits markers'
        # fast ridge+ols basket; the live ols+huber spread is ~2× wider —
        # re-anchor UI_MODEL_SPREAD_* only from an ols+huber measurement.
        spread.append(pd.to_numeric(tsv.get("ModelSpread"), errors="coerce").to_numpy(float) * 1e4)
        if "ConvictionUpper" in tsv.columns and "ConvictionLower" in tsv.columns:
            bandw.append((pd.to_numeric(tsv["ConvictionUpper"], errors="coerce")
                          - pd.to_numeric(tsv["ConvictionLower"], errors="coerce")).to_numpy(float))
        for c in ("OversoldBreadth", "OverboughtBreadth"):
            if c in tsv.columns:
                osb.append(pd.to_numeric(tsv[c], errors="coerce").to_numpy(float))
        pr = pd.to_numeric(tsv.get("Price"), errors="coerce").to_numpy(float)
        fr = np.full(len(pr), np.nan)
        fr[:-H] = (pr[H:] / pr[:-H] - 1) * 100
        fwd_conv.append(fr)
        if nd is not None and not nd.empty and "Avg_Signal" in nd.columns:
            avg_sig.append(pd.to_numeric(nd["Avg_Signal"], errors="coerce").to_numpy(float))
        # convergence frame → agreement_ratio + convergence_score distributions
        try:
            from convergence.cross_validator import CrossValidator  # noqa: F401
            frame = _convergence_frame(tgt)
        except Exception:
            frame = None
        if frame is not None:
            for col, sink in (("agreement_ratio", agree), ("convergence_score", cscore)):
                if col in frame.columns:
                    sink.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(float))
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} pooled", flush=True)

    print(f"\n  built in {time.time()-t0:.0f}s")
    print("\n" + "=" * 74)
    print("  UI/TIER ANCHORS — pooled live distributions vs current constants")
    print("  Convention: two-tier constants anchor at p75 (moderate) / p90 (strong);")
    print("  single-tier at the percentile matching the tier's role.")
    print("=" * 74)

    _dist_report("ConvictionBounded → CONVICTION_MODERATE/STRONG",
                 np.concatenate(conv_b) if conv_b else np.array([]),
                 {"MODERATE": CONVICTION_MODERATE, "STRONG": CONVICTION_STRONG},
                 fwd=np.concatenate(fwd_conv) if fwd_conv else None)
    _dist_report("ModelSpread → UI_MODEL_SPREAD_LOW/HIGH",
                 np.concatenate(spread) if spread else np.array([]),
                 {"LOW": UI_MODEL_SPREAD_LOW, "HIGH": UI_MODEL_SPREAD_HIGH})
    _dist_report("CI band width (Upper−Lower) — informational (tier removed; degenerate)",
                 np.concatenate(bandw) if bandw else np.array([]), {})
    _dist_report("Oversold/Overbought breadth %% → UI_BREADTH_HIGH",
                 np.concatenate(osb) if osb else np.array([]),
                 {"HIGH": UI_BREADTH_HIGH})
    _dist_report("agreement_ratio → UI_AGREEMENT_MODERATE/STRONG",
                 np.concatenate(agree) if agree else np.array([]),
                 {"MODERATE": UI_AGREEMENT_MODERATE, "STRONG": UI_AGREEMENT_STRONG})
    _dist_report("convergence_score (±100) → CONV_WEAK/MODERATE/STRONG tiers",
                 np.concatenate(cscore) if cscore else np.array([]),
                 {"WEAK": abs(CONV_WEAK_BULLISH), "MODERATE": abs(CONV_MODERATE_BULLISH),
                  "STRONG": abs(CONV_STRONG_BULLISH)})
    _dist_report("Nirnay Avg_Signal → UI_NIRNAY_BULLISH/BEARISH",
                 np.concatenate(avg_sig) if avg_sig else np.array([]),
                 {"BULLISH/BEARISH": abs(UI_NIRNAY_BULLISH)})

    # Per-instrument oscillator (NIRNAY_OVERSOLD/OVERBOUGHT) — commodity baskets
    # only (small; equity baskets too heavy for an anchor check).
    try:
        from nirnay_tuning_study import _constituent_osc, _basket_ohlcv, BASE as _NBASE
        osc_pool = []
        for tgt in ("Gold", "Copper", "Cotton", "USD/INR", "Jeera"):
            for sym, odf in (_basket_ohlcv(tgt) or {}).items():
                s = _constituent_osc(dict(_NBASE), tgt, sym, odf)
                if s is not None and len(s):
                    osc_pool.append(pd.to_numeric(s, errors="coerce").to_numpy(float))
        _dist_report("Per-instrument Unified_Osc (±10) → NIRNAY_OVERSOLD/OVERBOUGHT",
                     np.concatenate(osc_pool) if osc_pool else np.array([]),
                     {"OVERSOLD/OVERBOUGHT": abs(NIRNAY_OVERSOLD)})
    except Exception as e:
        print(f"\n  per-instrument osc pool failed: {e}")

    print("\n  NOTE: these are EXTREMENESS tiers (display/alert vocabulary), not")
    print("  actionable edges. Re-anchor a constant when its current occupancy is")
    print("  far from its tier's convention percentile; leave it when within a few")
    print("  points (re-anchoring to chase small drift is churn, not rigor).")


if __name__ == "__main__":
    main()
