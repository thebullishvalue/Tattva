"""
Tattva — Drift-Diffusion Model (DDM) filter with mean-reverting variance.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Bounded signal filtering with a heuristic uncertainty band via a
stochastic DDM.

NOTE on "confidence bands": ``±1.96*sqrt(variances)`` (computed by callers,
e.g. convergence/conviction_model.py) LOOKS like a 95% Gaussian confidence
interval, but ``variances`` here is not a state-space posterior variance
estimated from data — the state-space model itself is not specified/fit; the
level is instead a designed, bounded heuristic that mean-reverts toward
``long_run_var`` (a fixed constant, e.g. 50 or 100) with a drift-dependent
expansion term. A real 95% CI requires the variance to be an actual sampling-
distribution estimate (see Durbin & Koopman, "Time Series Analysis by State
Space Methods", for the Kalman-filter case that WOULD justify this).
Practically the band is still a useful WIDTH signal (narrow = the filter's
recent evidence has been consistent; wide = it has been noisy/conflicting) —
just not a calibrated coverage interval, so it should be read and labeled as
an uncertainty band, not a confidence interval.
"""

from __future__ import annotations

import numpy as np


def drift_diffusion_filter(
    observations: np.ndarray,
    leak_rate: float = 0.08,
    drift_scale: float = 0.15,
    long_run_var: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D Drift-Diffusion filter with mean-reverting variance.

    State equation: ``state_t = (1-λ)×state_{t-1} + λ×drift_t``

    Variance equation: ``var_t = (1-λ)×var_{t-1} + λ×σ²_LR + 0.5×|drift_t|``

    The variance mean-reverts toward ``long_run_var`` with a drift-dependent
    expansion term capped at ``long_run_var × 0.5`` to prevent ballooning
    during prolonged regimes.

    Parameters
    ----------
    observations : np.ndarray
        Input time-series.
    leak_rate : float
        Memory decay rate (``λ``). Higher = faster forgetting.
    drift_scale : float
        Scaling factor for new evidence contribution.
    long_run_var : float
        Long-run variance toward which the filter mean-reverts.

    Returns
    -------
    filtered : np.ndarray
        Filtered state values.
    gains : np.ndarray
        Dummy array of zeros (gains are implicit in the DDM formulation).
    variances : np.ndarray
        Estimate variances at each step.
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # A NaN seed (the engine's own warm-up region, see aarambh.py's A3 fix, can
    # leave obs[0] non-finite) would propagate NaN through the leaky state for
    # the whole series via state*(1-leak_rate). Seed a NEUTRAL 0.0 belief in
    # that case — matching the per-step `evidence = 0.0` fallback below.
    # (An earlier revision seeded from the first FINITE observation instead;
    # that was a look-ahead: for a series whose leading segment is NaN, the
    # first finite value lives hundreds of rows in the future, so the filter's
    # t=0 state reflected data that did not yet exist.)
    state = float(obs[0]) if np.isfinite(obs[0]) else 0.0
    var_est = long_run_var

    filtered = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = var_est

    for i in range(1, n):
        evidence = obs[i] if np.isfinite(obs[i]) else 0.0
        drift = evidence * drift_scale

        # Leaky integration with mean-reversion
        state = state * (1 - leak_rate) + drift

        # Mean-reverting variance with capped drift-dependent expansion
        var_est = (
            var_est * (1 - leak_rate)
            + leak_rate * long_run_var
            + min(abs(drift) * 0.5, long_run_var * 0.5)
        )
        var_est = max(var_est, 1e-6)  # Prevent collapse

        filtered[i] = state
        variances[i] = var_est

    return filtered, np.zeros(n), variances
