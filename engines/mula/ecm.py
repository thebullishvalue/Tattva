"""
Tattva — MŪLA core: the error-correction / expert-pooling layer.
तत्त्व (Tattva) — "Principle / Essence"

MŪLA (मूल — "root"). This module ADDS the short-run error-correction read on
top of the incumbent valuation core instead of replacing it: the long-run
equilibrium relation (the recursive cointegrating regression whose residual
is the mispricing gap ``z_t``) is kept exactly as production computes it,
while everything this file adds is NEW information the incumbent never
published:

  1. THE ECM READ. Engle & Granger (1987): if z_t is a valid equilibrium
     gap, Δp_t responds to z_{t-1} with speed −κ. Regressing the differenced
     target on the LAGGED LEVEL gap identifies κ with fast memories — which
     is precisely what the level regression could not do (one-step
     likelihood is degenerate on integrated regressands, not on differenced
     ones). The slow-memory restriction stops being a modelling commitment
     and becomes a checked assumption: κ̂ > 0 says the gap mean-reverts;
     κ̂ ≈ 0 says it does not, whatever the level model hopes.

  2. EXPERT POOLING. Forecast families for Δp_t, combined by exponentially
     discounted cumulative one-step log predictive density (Geweke &
     Amisano 2011):
       VALUATION-led : {1, z_{t-1}}
       MOMENTUM-led  : {1, r_{t-1}}          (r = one-session log return)
     plus a FULL design {1, z, Δz, r} scored alongside. The pool weight
     WValuation IS the honest answer to "is valuation informative today?" —
     learned forward, never persisted, never refitted on history.

Each family is a BatchDLM bank over a discount grid, so coefficients are
time-varying with data-selected memory and every one-step predictive is
Student-t. All designs use quantities dated t−1 to explain Δp_t.

Output contract (per session, aligned to the engine's ``series`` frame):
  MulaKappa / MulaDriftPct / MulaSdPct / WValuation / WMomentum / WFull.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.causal import DynamicModelAverage, EWMA

#: Memory of the internal scale trackers (≈ 2 years).
_SCALE_HALFLIFE = 504.0
#: Floors on the tracked scales — below these the input is treated as
#: degenerate and the session publishes nothing rather than amplifying noise.
_SCALE_FLOOR = 1e-4

#: Discount grid for the SHORT-RUN regressions. Unlike the level regression,
#: fast memories are legitimate here (differenced data), so the grid spans
#: ~1 week-equivalent adaptation to permanent and the DATA selects.
_ECM_GRID = (0.90, 0.98, 0.995, 0.999, 1.0)

#: Forgetting exponent of the expert-pool weight update (Raftery et al.).
_POOL_ALPHA = 0.99

#: Warm-up before any MŪLA output publishes.
WARMUP = 302


def _softmax_pool(logw: np.ndarray) -> np.ndarray:
    z = logw - logw.max()
    w = np.exp(z)
    w = np.maximum(w / max(w.sum(), 1e-300), 1e-6)
    return w / w.sum()


class MulaLayer:
    """Causal ECM + expert-pooling read on an existing equilibrium gap."""

    def __init__(self, horizon: int = 10, grid: tuple = _ECM_GRID) -> None:
        self.h = int(horizon)
        g = [float(x) for x in grid]
        self.val = DynamicModelAverage(k=2, grid=tuple(g))   # {1, z_lag}
        self.mom = DynamicModelAverage(k=2, grid=tuple(g))   # {1, r_lag}
        self.full = DynamicModelAverage(k=4, grid=tuple(g))  # {1, z, dz, r}
        # Scale trackers. The discounted DLM carries unit-variance priors,
        # so the regression runs on STANDARDISED inputs and the published
        # κ̂ / drift are unscaled back to log-price units. Without this, a
        # 0.03-scale gap against a 0.01-scale Δp sends the filter's gain
        # structure unstable (measured: κ̂ → 2e5 on synthetic OU data).
        self._sd_z = EWMA(halflife=_SCALE_HALFLIFE)
        self._sd_r = EWMA(halflife=_SCALE_HALFLIFE)
        self._sd_p = EWMA(halflife=_SCALE_HALFLIFE)
        self.ll = np.zeros(3)
        self.n = 0
        self.out = {k: [] for k in ("MulaKappa", "MulaDriftPct", "MulaSdPct",
                                    "WValuation", "WMomentum", "WFull")}

    def step(self, z_lag: float, dz_lag: float, r_lag: float,
             dp_now: float) -> None:
        """Advance one session. All regressors are dated t−1; dp_now = Δp_t."""
        if not all(np.isfinite(v) for v in (z_lag, r_lag, dp_now)):
            for k in self.out:
                self.out[k].append(np.nan)
            return

        # Update scale trackers with TODAY's observations (they describe the
        # data up to t, used for the t+1 design — causal).
        self._sd_z.update(abs(z_lag))
        self._sd_r.update(abs(r_lag))
        self._sd_p.update(abs(dp_now))
        sz = max(self._sd_z.mean, _SCALE_FLOOR)
        sr = max(self._sd_r.mean, _SCALE_FLOOR)
        sp = max(self._sd_p.mean, _SCALE_FLOOR)

        # Standardised designs — the DLM's unit-variance priors live here.
        f_val = np.array([1.0, z_lag / sz])
        f_mom = np.array([1.0, r_lag / sr])
        f_full = np.array([1.0, z_lag / sz,
                           (dz_lag / sz) if np.isfinite(dz_lag) else 0.0,
                           r_lag / sr])
        y_std = dp_now / sp

        p_val, v_val = self.val.forecast(f_val)
        p_mom, v_mom = self.mom.forecast(f_mom)
        p_full, v_full = self.full.forecast(f_full)

        ll = np.array([float(np.mean(self.val.bank.log_pred_lik)),
                       float(np.mean(self.mom.bank.log_pred_lik)),
                       float(np.mean(self.full.bank.log_pred_lik))])
        ll = np.where(np.isfinite(ll), ll, -1e6)
        self.ll = _POOL_ALPHA * self.ll + ll
        w = _softmax_pool(self.ll)

        self.val.update(f_val, y_std)
        self.mom.update(f_mom, y_std)
        self.full.update(f_full, y_std)

        kappa = float(self.full.coef[1]) * sp / sz       # per day, log units
        kappa_val = float(self.val.coef[1]) * sp / sz
        preds = np.array([float(p_val), float(p_mom), float(p_full)])
        sds2 = np.array([max(float(v_val), 1e-18),
                         max(float(v_mom), 1e-18),
                         max(float(v_full), 1e-18)])
        mu = float(w @ preds)
        var = float(w @ (sds2 + (preds - mu) ** 2))
        sd = float(np.sqrt(max(var, 1e-18))) * sp

        # Expected h-day gap-closure move: κ acting over h sessions with the
        # estimated persistence folded in (geometric decay approximation).
        phi = max(min(kappa_val, 0.499), -0.499)
        geom = (1.0 - phi ** self.h) / max(1.0 - phi, 1e-6)
        drift_pct = float(-kappa_val * z_lag * geom * 100.0)

        self.out["MulaKappa"].append(kappa)
        self.out["MulaDriftPct"].append(drift_pct)
        self.out["MulaSdPct"].append(sd * 100.0)
        self.out["WValuation"].append(float(w[0]))
        self.out["WMomentum"].append(float(w[1]))
        self.out["WFull"].append(float(w[2]))
        self.n += 1

    def frame(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({k: pd.Series(v[:len(index)], index=index)
                             for k, v in self.out.items()})

