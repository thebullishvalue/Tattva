"""
Integrity test for the convergence signal chain — written in response to a
real defect: the hard {-1,0,+1} consensus gate zeroed the ENTIRE composite on
60.7% of real days (Gold, 810 scored days), so the hero card read 0.00 and
its trend flatlined while every module faithfully computed on the degenerate
object. "What it says vs what it does" checks, pinned:

  1. NON-DEGENERACY — on realistic engine streams the composite must not be
     exactly zero on a material fraction of days (the old gate: 60%+).
  2. SIGN CORRECTNESS — both engines bullish → negative score (bullish
     convention); both bearish → positive.
  3. GRACEFUL DISAGREEMENT — opposed engines partially cancel (small |score|,
     sign of the stronger engine), never a hard snap to 0.
  4. TIE HANDLING — a nirnay breadth tie contributes zero from that engine
     but must NOT silence Aarambh.
  5. SAYS-VS-DOES IDENTITY — the stored convergence_score must be
     reproducible from the stored dim_*/consensus columns via
     intelligence._composite_signal (two independent implementations of the
     same formula, reconciled numerically).
  6. DOWNSTREAM LIVENESS — the raw product chain (composite → DDM trend)
     must produce a non-flat trend on a signal with genuine direction.

Run: python -m research.test_convergence_integrity  (from the repo root)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from convergence.cross_validator import CrossValidator
from convergence.intelligence import _composite_signal, DEFAULT_WEIGHTS
from analytics.ddm_filter import drift_diffusion_filter
from analytics.utils import _apply_conviction_bounds


def _feed(validator, conviction, os_pct, ob_pct, date):
    """One scored day with otherwise-neutral auxiliary stats."""
    return validator.compute_convergence(
        {"conviction_score": conviction, "oversold_breadth": 50.0,
         "regime": "NEUTRAL"},
        {"oversold_pct": os_pct, "overbought_pct": ob_pct,
         "avg_unified_osc": 0.0, "regime_bull_pct": 33.0,
         "regime_bear_pct": 33.0, "regime_neutral": 34.0,
         "num_constituents": 16},
        date,
    )


def run() -> None:
    checks = 0
    rng = np.random.default_rng(21)

    # ── 1. NON-DEGENERACY on realistic streams ────────────────────────────
    # Small-basket breadth (16 names) with frequent ties, DDM-like conviction.
    v = CrossValidator(active_weights=DEFAULT_WEIGHTS.copy(), expected_constituents=16)
    n = 600
    conv_stream = np.clip(rng.normal(0, 40, n), -95, 95)
    os_counts = rng.binomial(16, 0.12, n)
    ob_counts = rng.binomial(16, 0.12, n)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    for i in range(n):
        _feed(v, float(conv_stream[i]), os_counts[i] / 16 * 100,
              ob_counts[i] / 16 * 100, str(dates[i].date()))
    frame = v.get_convergence_series()
    score = frame["convergence_score"].to_numpy(dtype=float)
    tie_frac = float(np.mean(os_counts == ob_counts))
    zero_frac = float(np.mean(score == 0))
    assert tie_frac > 0.15, f"stream not realistic enough (ties {tie_frac:.0%})"
    assert zero_frac < 0.02, (
        f"composite exactly zero on {zero_frac:.0%} of days (ties were "
        f"{tie_frac:.0%}) — the hard-gate degeneracy is back")
    checks += 1

    # ── 2. SIGN CORRECTNESS ──────────────────────────────────────────────
    v2 = CrossValidator(active_weights=DEFAULT_WEIGHTS.copy())
    r = _feed(v2, -60.0, 40.0, 5.0, "2024-01-01")     # both bullish
    assert r.convergence_score < 0, "both-bullish day must score negative (bullish)"
    assert r.consensus_direction > 0
    r = _feed(v2, +60.0, 5.0, 40.0, "2024-01-02")     # both bearish
    assert r.convergence_score > 0 and r.consensus_direction < 0
    checks += 1

    # ── 3. GRACEFUL DISAGREEMENT (no hard zero; stronger engine wins) ────
    r_bal = _feed(v2, -50.0, 10.0, 60.0, "2024-01-03")   # aarambh bull 0.5 vs nirnay bear 0.5
    assert abs(r_bal.consensus_direction) < 1e-9, "equal-and-opposite must cancel to ~0"
    r_skew = _feed(v2, -80.0, 10.0, 30.0, "2024-01-04")  # aarambh bull 0.8 vs nirnay bear 0.2
    assert r_skew.consensus_direction > 0 and r_skew.convergence_score < 0, \
        "stronger engine must set the sign under disagreement"
    assert 0 < abs(r_skew.convergence_score) < 100
    checks += 1

    # ── 4. TIE HANDLING — nirnay tie must not silence aarambh ────────────
    r_tie = _feed(v2, -70.0, 0.0, 0.0, "2024-01-05")     # 0% vs 0% breadth tie
    assert r_tie.convergence_score < 0 and abs(r_tie.convergence_score) > 1, (
        f"nirnay tie hard-zeroed the composite again (score "
        f"{r_tie.convergence_score}) — old-gate behaviour")
    # Both engines exactly neutral → and only then → zero.
    r_zero = _feed(v2, 0.0, 0.0, 0.0, "2024-01-08")
    assert r_zero.convergence_score == 0.0
    checks += 1

    # ── 5. SAYS-VS-DOES IDENTITY across modules ──────────────────────────
    # Stored convergence_score (CrossValidator) vs recomputation from the
    # stored dim_*/consensus columns (intelligence._composite_signal).
    # Tolerance covers the dataclass's display rounding (dims 3dp, score 2dp).
    recomputed = _composite_signal(frame, DEFAULT_WEIGHTS) * 100.0
    max_err = float(np.max(np.abs(recomputed - score)))
    assert max_err < 0.2, (
        f"stored convergence_score diverges from dim_* recomputation by "
        f"{max_err:.4f} — the two implementations of the composite disagree")
    checks += 1

    # ── 6. DOWNSTREAM LIVENESS — DDM trend of the raw product chain ──────
    raw = _composite_signal(frame, DEFAULT_WEIGHTS)
    filt, _, _ = drift_diffusion_filter(raw * 100.0, leak_rate=0.10,
                                        drift_scale=0.12, long_run_var=50.0)
    smoothed = _apply_conviction_bounds(filt) / 100.0
    assert float(np.std(smoothed)) > 0.01, (
        "DDM trend of the composite is flat — the signal chain is degenerate")
    assert float(np.mean(np.abs(smoothed))) > 0.005
    checks += 1

    # ── 7. THRESHOLD-DISTRIBUTION PAIRING (the F1 principle, post-anchor) ──
    # The composite classifier's defaults must be the composite's OWN
    # data-anchored cut-points, not the consensus's (wider) set — a score at
    # the composite's real p90 (~0.33, ×100 = 33) must label directional, not HOLD.
    from convergence.normalization import (
        classify_convergence_score, COMPOSITE_THRESHOLDS,
        DEFAULT_THRESHOLDS as CONSENSUS_THRESHOLDS,
    )
    from convergence.intelligence import DEFAULT_THRESHOLDS as INTEL_THRESHOLDS
    assert INTEL_THRESHOLDS == COMPOSITE_THRESHOLDS, (
        "intelligence's calibration seed/fallback thresholds diverged from "
        "the composite's factory set — single-source violation")
    assert abs(COMPOSITE_THRESHOLDS["buy_moderate"]) < abs(CONSENSUS_THRESHOLDS["buy_moderate"]), (
        "composite thresholds should be tighter than consensus thresholds "
        "(different distributions)")
    # Anchored to COMPOSITE p75/p90 = ±0.092/±0.159 (×100 = ±9.2/±15.9).
    assert classify_convergence_score(-25.0) == "STRONG BUY"   # beyond p90 (15.9)
    assert classify_convergence_score(-12.0) == "BUY"          # p75-p90 (9.2-15.9)
    assert classify_convergence_score(-5.0) == "HOLD"          # sub-moderate (<9.2)
    assert classify_convergence_score(+25.0) == "STRONG SELL"
    # A composite-directional day (-12 → BUY on composite) reads HOLD under the
    # WIDER consensus thresholds (moderate ±27.9) — the exact mispairing this
    # check exists to prevent:
    assert classify_convergence_score(-12.0, CONSENSUS_THRESHOLDS) == "HOLD"
    checks += 1

    print(f"convergence integrity: ALL {checks} CHECK GROUPS PASSED "
          f"(zero-days {zero_frac:.1%} on realistic stream with {tie_frac:.0%} "
          f"breadth ties; says-vs-does max error {max_err:.4f})")


if __name__ == "__main__":
    run()
