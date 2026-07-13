"""
Equivalence check: analytics.regime's object-oriented reference classes
(AdaptiveKalmanFilter, GARCHDetector, AdaptiveHMM, CUSUMDetector) vs the
njit-compiled kernel (_regime_loop_njit) that engines.nirnay actually calls.

Why this exists (audit finding F12): the object classes are dead code in the
live app — engines.nirnay only ever called run_regime_loop's njit kernel —
but they're kept as a readable, step-by-step reference for the kernel's dense
single-pass logic. That means the SAME fix (e.g. the HMM ordering-constraint
label-switching guard) has to be applied to both independently, with no
automated check that they stay identical. This script drives both paths over
the same random signal and asserts bit-for-bit (within float tolerance)
agreement on every output the kernel produces, so a future edit to either
side that silently diverges from the other fails loudly here instead of
drifting unnoticed.

Run: python -m research.test_regime_equivalence  (from the repo root)
"""
from __future__ import annotations

import numpy as np

from analytics.regime import (
    AdaptiveKalmanFilter,
    AdaptiveHMM,
    GARCHDetector,
    CUSUMDetector,
    run_regime_loop,
    _REGIME_NAMES,
    _VOL_NAMES,
)


def _object_driver(sig: np.ndarray):
    """Faithful step-by-step replay of _regime_loop_njit using the object
    classes, matching the kernel's exact call order and inputs:
      1. Kalman.update(raw signal)          -> filtered
      2. GARCH.update(shock = raw - prev_raw, using PREVIOUS raw, not filtered)
      3. HMM.update(filtered)
      4. CUSUM.update(filtered)
    """
    n = len(sig)
    kf = AdaptiveKalmanFilter()
    garch = GARCHDetector()
    hmm = AdaptiveHMM()
    cusum = CUSUMDetector()

    regimes: list[str] = []
    hmm_bulls = np.zeros(n)
    hmm_bears = np.zeros(n)
    vol_regimes: list[str] = []
    change_points: list[bool] = []
    confidences = np.zeros(n)

    prev_sig = 0.0
    has_prev = False

    for i in range(n):
        s = float(sig[i])

        filtered = kf.update(s)

        shock = (s - prev_sig) if has_prev else 0.0
        garch.update(shock)
        vol_regime, _sensitivity = garch.get_regime()

        probs = hmm.update(filtered)
        bull_p, neut_p, bear_p = probs["BULL"], probs["NEUTRAL"], probs["BEAR"]

        changed = cusum.update(filtered)

        if changed:
            regime = "TRANSITION"
        elif bull_p > 0.6:
            regime = "BULL"
        elif bear_p > 0.6:
            regime = "BEAR"
        elif bull_p > 0.4:
            regime = "WEAK_BULL"
        elif bear_p > 0.4:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"

        regimes.append(regime)
        hmm_bulls[i] = bull_p
        hmm_bears[i] = bear_p
        vol_regimes.append(vol_regime)
        change_points.append(bool(changed))
        confidences[i] = max(bull_p, bear_p, neut_p)

        prev_sig = s
        has_prev = True

    return regimes, hmm_bulls, hmm_bears, vol_regimes, change_points, confidences


def run() -> None:
    rng = np.random.default_rng(11)
    n = 400
    # A mix of trend + noise + a few large shocks — exercises both the HMM's
    # regime classification and the CUSUM change-point / GARCH vol-regime
    # branches, not just steady-state small-signal behaviour.
    sig = np.cumsum(rng.normal(0, 0.15, n))
    shock_idx = rng.choice(n, size=8, replace=False)
    sig[shock_idx] += rng.normal(0, 3.0, len(shock_idx))

    kernel_regimes, kernel_hb, kernel_hbe, kernel_vol, kernel_cp, kernel_conf = run_regime_loop(sig)
    obj_regimes, obj_hb, obj_hbe, obj_vol, obj_cp, obj_conf = _object_driver(sig)

    assert kernel_regimes == obj_regimes, (
        f"Regime label mismatch at index "
        f"{next(i for i in range(n) if kernel_regimes[i] != obj_regimes[i])}"
    )
    assert kernel_vol == obj_vol, (
        f"Vol-regime label mismatch at index "
        f"{next(i for i in range(n) if kernel_vol[i] != obj_vol[i])}"
    )
    assert kernel_cp == obj_cp, (
        f"Change-point flag mismatch at index "
        f"{next(i for i in range(n) if kernel_cp[i] != obj_cp[i])}"
    )
    np.testing.assert_allclose(kernel_hb, obj_hb, atol=1e-9, err_msg="HMM_Bull mismatch")
    np.testing.assert_allclose(kernel_hbe, obj_hbe, atol=1e-9, err_msg="HMM_Bear mismatch")
    np.testing.assert_allclose(kernel_conf, obj_conf, atol=1e-9, err_msg="Confidence mismatch")

    # Sanity: every emitted label is a genuine member of the kernel's enums
    # (guards against the object driver's classification branch silently
    # drifting from the kernel's own if/elif ladder).
    assert set(kernel_regimes) <= set(_REGIME_NAMES)
    assert set(kernel_vol) <= set(_VOL_NAMES)

    print(f"regime_equivalence: ALL CHECKS PASSED ({n} steps, "
          f"{len(shock_idx)} injected shocks)")
    print(f"  regime label distribution: "
          f"{ {r: kernel_regimes.count(r) for r in sorted(set(kernel_regimes))} }")


if __name__ == "__main__":
    run()
