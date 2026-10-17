"""
Tattva — Regime detection: Kalman Filter, HMM, GARCH, CUSUM.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — the per-series regime loop behind every Swayam view: an adaptive
Kalman smoother on the unified oscillator, a three-state HMM over it (BULL /
NEUTRAL / BEAR emission means +0.6 / 0 / -0.6), a GARCH(1,1) volatility
regime and a CUSUM change-point test, all advanced one bar at a time.

READ THE STATE NAMES AS STRETCH, NOT AS A CALL. "BULL" is the HMM state whose
emission mean is a HIGH oscillator — the views reading the instrument as
persistently stretched UP. Under the system's contrarian reading (see
``core.semantics``) that is a bearish lean, which is why the convergence layer
scores Swayam's regime split through ``semantics.swayam_regime_lean`` rather
than treating "bull" as "buy".

(Object-oriented reference copies of these four models — AdaptiveKalmanFilter,
AdaptiveHMM, GARCHDetector, CUSUMDetector and their state dataclasses — lived
below the kernel. Nothing called them; they were kept as a "readable
reference" that a parity test was meant to hold to the kernel, and that test
was never in the repository. An unverified duplicate of a numerical kernel is a
second place for a fix to be missed, so they were removed: the kernel below is
the one implementation.)
"""

from __future__ import annotations

import numpy as np
from numba import njit

@njit(cache=True)
def _njit_forward_step(transition_matrix: np.ndarray, state_probabilities: np.ndarray,
                       emissions: np.ndarray) -> np.ndarray:
    predicted = transition_matrix.T @ state_probabilities
    updated = emissions * predicted
    total = np.sum(updated)
    if total > 1e-10:
        updated = updated / total
    else:
        updated = np.array([0.33, 0.34, 0.33])
    return updated

@njit(cache=True)
def _njit_gaussian_pdf(x: float, mean: float, std: float) -> float:
    var = std ** 2
    denom = (2 * np.pi * var) ** 0.5
    num = np.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


@njit(cache=True)
def _pop_var(buf: np.ndarray, lo: int, hi: int) -> float:
    """Population variance (ddof=0) of buf[lo:hi] — mirrors np.var default."""
    k = hi - lo
    if k <= 0:
        return 0.0
    s = 0.0
    for j in range(lo, hi):
        s += buf[j]
    m = s / k
    v = 0.0
    for j in range(lo, hi):
        d = buf[j] - m
        v += d * d
    return v / k


@njit(cache=True)
def _regime_loop_njit(sig: np.ndarray):
    """Faithful single-pass port of the Swayam regime-intelligence loop.

    Runs the four sequential filters (Kalman → GARCH → HMM → CUSUM) exactly
    as the object versions below, but with zero per-step Python/NumPy dispatch
    overhead. Returns, per step:
        regime_code  0=NEUTRAL 1=BULL 2=BEAR 3=WEAK_BULL 4=WEAK_BEAR 5=TRANSITION
        hmm_bull, hmm_bear, confidence  (floats)
        vol_code     0=LOW 1=NORMAL 2=HIGH 3=EXTREME
        change       (0/1)
    """
    n = sig.shape[0]
    regime_code = np.zeros(n, dtype=np.int64)
    vol_code = np.zeros(n, dtype=np.int64)
    hmm_bull = np.zeros(n, dtype=np.float64)
    hmm_bear = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    change_pts = np.zeros(n, dtype=np.int64)

    # ── Kalman state ──
    k_est = 0.0
    k_cov = 1.0
    k_proc = 0.01
    k_meas = 0.1
    innov = np.empty(n, dtype=np.float64)

    # ── GARCH state ──
    g_var = 0.04
    g_omega = 0.0001
    g_alpha = 0.1
    g_beta = 0.85
    g_ltm = 0.04
    shocks = np.empty(n, dtype=np.float64)

    # ── HMM state ──
    trans = np.array([[0.98, 0.01, 0.01],
                      [0.01, 0.98, 0.01],
                      [0.01, 0.01, 0.98]])
    em_mean = np.array([0.6, 0.0, -0.6])
    em_std = np.array([0.3, 0.25, 0.3])
    sp = np.array([0.33, 0.34, 0.33])
    obs_hist = np.empty(n, dtype=np.float64)   # observation_history (= filtered)
    state_hist = np.empty(n, dtype=np.int64)

    # ── CUSUM state ──
    pos_cusum = 0.0
    neg_cusum = 0.0
    c_thr = 4.0
    c_drift = 0.5
    cvals = np.empty(n, dtype=np.float64)
    run_mean = 0.0
    run_std = 1.0

    prev_sig = 0.0
    has_prev = False

    for i in range(n):
        s = sig[i]

        # ── Kalman update(s) ──
        pred_est = k_est
        pred_cov = k_cov + k_proc
        innovation = s - pred_est
        innov[i] = innovation
        innov_cov = pred_cov + k_meas
        gain = pred_cov / innov_cov
        k_est = pred_est + gain * innovation
        k_cov = (1.0 - gain) * pred_cov
        ni = i + 1  # innovation_history length
        if ni >= 5:
            w = 20 if ni >= 20 else ni
            iv = _pop_var(innov, i + 1 - w, i + 1)
            k_meas = 0.9 * k_meas + 0.1 * iv
        filtered = k_est

        # ── GARCH update(shock) ──
        shock = (s - prev_sig) if has_prev else 0.0
        shocks[i] = shock
        new_var = g_omega + g_alpha * (shock * shock) + g_beta * g_var
        if new_var < 0.001:
            new_var = 0.001
        elif new_var > 1.0:
            new_var = 1.0
        g_var = new_var
        ns = i + 1
        if ns >= 10:
            w = 50 if ns >= 50 else ns
            realized = _pop_var(shocks, i + 1 - w, i + 1)
            g_ltm = 0.95 * g_ltm + 0.05 * realized
        # get_regime()
        cur_vol = np.sqrt(g_var)
        lt_vol = np.sqrt(g_ltm)
        ratio = cur_vol / lt_vol if lt_vol > 0 else 1.0
        if ratio < 0.6:
            vol_code[i] = 0
        elif ratio < 0.9:
            vol_code[i] = 1
        elif ratio < 1.4:
            vol_code[i] = 2
        else:
            vol_code[i] = 3

        # ── HMM update(filtered) ──
        obs_hist[i] = filtered
        # forward step
        e0 = _njit_gaussian_pdf(filtered, em_mean[0], em_std[0] + 1e-4)
        e1 = _njit_gaussian_pdf(filtered, em_mean[1], em_std[1] + 1e-4)
        e2 = _njit_gaussian_pdf(filtered, em_mean[2], em_std[2] + 1e-4)
        # predicted = trans.T @ sp
        p0 = trans[0, 0] * sp[0] + trans[1, 0] * sp[1] + trans[2, 0] * sp[2]
        p1 = trans[0, 1] * sp[0] + trans[1, 1] * sp[1] + trans[2, 1] * sp[2]
        p2 = trans[0, 2] * sp[0] + trans[1, 2] * sp[1] + trans[2, 2] * sp[2]
        u0 = e0 * p0
        u1 = e1 * p1
        u2 = e2 * p2
        total = u0 + u1 + u2
        if total > 1e-10:
            sp[0] = u0 / total
            sp[1] = u1 / total
            sp[2] = u2 / total
        else:
            sp[0] = 0.33
            sp[1] = 0.34
            sp[2] = 0.33
        # most likely
        if sp[0] >= sp[1] and sp[0] >= sp[2]:
            ml = 0
        elif sp[1] >= sp[2]:
            ml = 1
        else:
            ml = 2
        state_hist[i] = ml
        nobs = i + 1
        # _adapt_emissions (uses last 50 obs)
        if nobs >= 10:
            w = 50 if nobs >= 50 else nobs
            lo = i + 1 - w
            for st in range(3):
                cnt = 0
                ssum = 0.0
                for j in range(lo, i + 1):
                    if state_hist[j] == st:
                        cnt += 1
                        ssum += obs_hist[j]
                if cnt >= 2:
                    nm = ssum / cnt
                    vv = 0.0
                    for j in range(lo, i + 1):
                        if state_hist[j] == st:
                            d = obs_hist[j] - nm
                            vv += d * d
                    nsd = np.sqrt(vv / cnt)
                    if nsd < 1e-4:
                        nsd = 1e-4
                    em_mean[st] = 0.9 * em_mean[st] + 0.1 * nm
                    em_std[st] = 0.9 * em_std[st] + 0.1 * nsd
            # Ordering constraint (state 0=Bull > 1=Neutral > 2=Bear): without
            # it the online EM update has no identifiability constraint on
            # which state is "which", so a long one-sided regime can drift
            # em_mean[0] (Bull) below em_mean[2] (Bear) — the canonical
            # label-switching failure of unconstrained online HMM/mixture
            # estimation (Jasra, Holmes & Stephens 2005, Statistical Science
            # 20(1)). After that crossing, hmm_bull silently reports the BEAR
            # state's probability. A small margin (not a bare >=) stops the
            # constraint itself from flip-flopping on floating-point noise
            # when two means are nearly equal.
            if em_mean[0] < em_mean[1] + 0.01:
                em_mean[0] = em_mean[1] + 0.01
            if em_mean[2] > em_mean[1] - 0.01:
                em_mean[2] = em_mean[1] - 0.01
        # _adapt_transitions (uses last 30 states)
        if nobs >= 5:
            w = 30 if nobs >= 30 else nobs
            lo = i + 1 - w
            counts = np.zeros((3, 3))
            for j in range(lo, i):
                counts[state_hist[j], state_hist[j + 1]] += 1.0
            for r in range(3):
                rs = counts[r, 0] + counts[r, 1] + counts[r, 2]
                if rs >= 2:
                    for cc in range(3):
                        np_ = (counts[r, cc] + 1.0) / (rs + 3.0)
                        trans[r, cc] = 0.8 * trans[r, cc] + 0.2 * np_

        bull_p = sp[0]
        neut_p = sp[1]
        bear_p = sp[2]

        # ── CUSUM update(filtered) ──
        cvals[i] = filtered
        nc = i + 1
        if nc >= 3:
            # Window [lo, i) excludes current point — causal z-score
            w = min(20, nc - 1)
            lo = i - w
            csum = 0.0
            for j in range(lo, i):
                csum += cvals[j]
            run_mean = csum / w
            cv = 0.0
            for j in range(lo, i):
                d = cvals[j] - run_mean
                cv += d * d
            sd = np.sqrt(cv / w)
            run_std = sd if sd > 0.1 else 0.1
        z = (filtered - run_mean) / run_std
        pos_cusum = pos_cusum + z - c_drift
        if pos_cusum < 0.0:
            pos_cusum = 0.0
        neg_cusum = neg_cusum - z - c_drift
        if neg_cusum < 0.0:
            neg_cusum = 0.0
        change = (pos_cusum > c_thr) or (neg_cusum > c_thr)
        if change:
            pos_cusum = 0.0
            neg_cusum = 0.0
        change_pts[i] = 1 if change else 0

        # ── Regime classification ──
        if change:
            regime_code[i] = 5  # TRANSITION
        elif bull_p > 0.6:
            regime_code[i] = 1  # BULL
        elif bear_p > 0.6:
            regime_code[i] = 2  # BEAR
        elif bull_p > 0.4:
            regime_code[i] = 3  # WEAK_BULL
        elif bear_p > 0.4:
            regime_code[i] = 4  # WEAK_BEAR
        else:
            regime_code[i] = 0  # NEUTRAL

        hmm_bull[i] = bull_p
        hmm_bear[i] = bear_p
        c = bull_p
        if bear_p > c:
            c = bear_p
        if neut_p > c:
            c = neut_p
        confidence[i] = c

        prev_sig = s
        has_prev = True

    return regime_code, hmm_bull, hmm_bear, vol_code, change_pts, confidence


_REGIME_NAMES = ("NEUTRAL", "BULL", "BEAR", "WEAK_BULL", "WEAK_BEAR", "TRANSITION")
_VOL_NAMES = ("LOW", "NORMAL", "HIGH", "EXTREME")


def run_regime_loop(unified_vals: np.ndarray):
    """Vectorized driver for the per-series regime loop.

    Returns ``(regimes, hmm_bulls, hmm_bears, vol_regimes, change_points,
    confidences)`` as Python lists/arrays matching the original loop's output
    exactly. NaNs in ``unified_vals`` are treated as 0.0 (as before).
    """
    sig = np.nan_to_num(np.asarray(unified_vals, dtype=np.float64), nan=0.0)
    rc, hb, hbe, vc, cp, conf = _regime_loop_njit(sig)
    regimes = [_REGIME_NAMES[c] for c in rc]
    vol_regimes = [_VOL_NAMES[c] for c in vc]
    change_points = [bool(x) for x in cp]
    return regimes, hb, hbe, vol_regimes, change_points, conf
