"""
Tattva v2.0.0 — Drift-Diffusion Model (DDM) filter with mean-reverting variance.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Bounded signal filtering with confidence bands via stochastic DDM.
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

    state = float(np.mean(obs[: min(20, n)])) if n > 0 else 0.0
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
