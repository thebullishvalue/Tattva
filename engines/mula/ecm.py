"""
Tattva — MŪLA error correction: does PRICE close the gap?
तत्त्व (Tattva) — "Principle / Essence"

The level regression (``valuation.py``) publishes the mispricing gap
``z_t = p_t - fv_t`` and shows that it is stationary. Stationarity alone does
not license the trade, because a gap closes when EITHER side moves — and on
real data most of the closing is fair value moving toward price, not price
moving toward fair value (measured over 10 sessions: price covers 1-3% of a
gap that closes by 41-59%; Gold, S&P 500, USD/INR). A gap that only fair value
closes is a valuation that keeps catching up with the market, not a price that
is going anywhere. This module asks the price-side question directly.

1. THE ECM READ. Engle & Granger (1987): if ``z`` is an equilibrium gap that
   price responds to, then

       Δp_t = a + b · z_{t-1} + e_t ,      b < 0

   and ``κ̂ = -b`` is the share of the gap price closes per session. Regressing
   the DIFFERENCED target on the lagged LEVEL gap identifies ``b`` with fast
   memories, which is precisely what the level regression could not do
   (one-step likelihood is degenerate on integrated regressands, not on
   differenced ones). Published:

     MulaKappa     κ̂ = -b, per session, log units. Positive ⇒ price reverts.
     MulaRevProb   P(κ̂ > 0) under the coefficient posterior — the evidence,
                   not the point estimate, that price responds at all.
     MulaHalfLife  ln2 / -ln(1-κ̂): how long price alone would take to close
                   half the gap, if fair value stood still. NaN when κ̂ ≤ 0.

2. THE IMPLIED MOVE. If the gap persists as ``E[z_{t+k}] = φ_z^k z_t`` (φ_z is
   the gap's own AR(1) coefficient, estimated in ``valuation.py``), the
   expected price change over the next h sessions is

       E[p_{t+h} - p_t | z_t] = b · z_t · Σ_{k<h} φ_z^k

   — MulaDriftPct, in percent, signed like the move itself (negative when the
   asset is rich and price reverts). MulaSdPct is the h-session predictive SD
   it should be read against.

   The previous implementation published ``-b · z_{t-1}`` with the geometric
   sum taken over ``b`` itself (≈ -0.007, so the sum was ≈ 1): the opposite
   sign of the model's own expected move, about one session's worth of it, from
   yesterday's gap, labelled as h sessions. On a synthetic OU gap its
   correlation with the true expected move was -0.89; on real data it agreed in
   sign with the corrected formula 16-28% of the time. The hero's "does the ECM
   agree with the call?" check therefore fired backwards.

3. EXPERT POOLING. Three forecast families for Δp_t, combined by exponentially
   discounted cumulative one-step log predictive density (Geweke & Amisano
   2011):
       VALUATION-led : {1, z_{t-1}}
       MOMENTUM-led  : {1, r_{t-1}}          (r = one-session log return)
       FULL          : {1, z_{t-1}, Δz_{t-1}, r_{t-1}}
   WValuation + WFull — the weight on every design that lets the gap move
   price — is the honest answer to "is valuation informative today?": learned
   forward, never persisted, never refitted on history. Each family is scored
   by its MIXTURE predictive density (the same ``_mix_ll`` the level
   regression's two views are scored by), not by the mean of its members'
   densities.

Each family is a discounted-DLM bank over a grid of memories, so coefficients
are time-varying with data-selected adaptation and every one-step predictive
is Student-t. All regressors are dated t-1; everything published at t uses
data through t and nothing later.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analytics.causal import DynamicModelAverage, EWMA, norm_cdf_scalar
from .valuation import _mix_ll

#: Memory of the internal scale trackers (≈ 2 years).
_SCALE_HALFLIFE = 504.0
#: Floor on the tracked scales — below it the input is treated as degenerate.
_SCALE_FLOOR = 1e-4
#: Sessions spent only learning the scales before the regressions start. The
#: DLMs carry unit-variance priors, so they must see STANDARDISED inputs from
#: their first step; a scale tracker that has seen nothing would divide by its
#: floor and hand the filter a 300σ regressor.
_SCALE_WARMUP = 20

#: Discount grid for the SHORT-RUN regressions. Unlike the level regression,
#: fast memories are legitimate here (differenced data), so the grid spans
#: ~10-session adaptation to permanent and the DATA selects.
_ECM_GRID = (0.90, 0.98, 0.995, 0.999, 1.0)

#: Forgetting exponent of the expert-pool weight update (Raftery et al.).
#: Memory ≈ 1 / (1 - 0.99) = 100 sessions.
_POOL_ALPHA = 0.99

#: Sessions of ABSORBED evidence before anything publishes — counted from the
#: layer's own first regression step, i.e. from the first published gap, not
#: from row 0 of the frame. (It used to be 302 rows from row 0; the gap itself
#: starts after a 252-row burn-in, so the pool and κ̂ went live on ~45 sessions
#: of evidence.) Half a year is enough for a two-coefficient regression and
#: for the pool's log-score differences to reflect evidence rather than the
#: uniform start.
WARMUP = 126

#: Published columns, in order.
COLUMNS: tuple[str, ...] = ("MulaKappa", "MulaRevProb", "MulaHalfLife",
                            "MulaDriftPct", "MulaSdPct",
                            "WValuation", "WMomentum", "WFull")


def _softmax_pool(logw: np.ndarray) -> np.ndarray:
    z = logw - logw.max()
    w = np.exp(z)
    w = np.maximum(w / max(w.sum(), 1e-300), 1e-6)
    return w / w.sum()


def gap_closure_sum(phi_z: float, horizon: int) -> float:
    """Σ_{k<h} φ_z^k — how much of today's gap is still open, summed over h.

    φ_z ≥ 1 (a gap with no measured decay) sums to h; φ_z ≤ 0 to 1 (the gap is
    expected to be gone after one session).
    """
    h = max(int(horizon), 1)
    if not np.isfinite(phi_z):
        return float("nan")
    phi = min(max(float(phi_z), 0.0), 1.0)
    if phi >= 1.0 - 1e-12:
        return float(h)
    return float((1.0 - phi ** h) / (1.0 - phi))


class MulaLayer:
    """Causal ECM + expert-pooling read on an existing equilibrium gap."""

    def __init__(self, horizon: int = 10, grid: tuple = _ECM_GRID) -> None:
        self.h = max(int(horizon), 1)
        g = tuple(float(x) for x in grid)
        self.val = DynamicModelAverage(k=2, grid=g)    # {1, z_lag}
        self.mom = DynamicModelAverage(k=2, grid=g)    # {1, r_lag}
        self.full = DynamicModelAverage(k=4, grid=g)   # {1, z, dz, r}
        # Scale trackers. The discounted DLM carries unit-variance priors, so
        # the regressions run on STANDARDISED inputs and the published κ̂ /
        # drift are unscaled back to log-price units. Without this, a
        # 0.03-scale gap against a 0.01-scale Δp sends the filter's gain
        # structure unstable (measured: κ̂ → 2e5 on synthetic OU data).
        # Each step standardises with the scales as they stood BEFORE that
        # step's observations, then absorbs them — so no one-step density is
        # ever evaluated on a scale that contains its own outcome.
        self._sd_z = EWMA(halflife=_SCALE_HALFLIFE)
        self._sd_r = EWMA(halflife=_SCALE_HALFLIFE)
        self._sd_p = EWMA(halflife=_SCALE_HALFLIFE)
        self.ll = np.zeros(3)
        self.n = 0
        self.out = {k: [] for k in COLUMNS}

    # ── one session ───────────────────────────────────────────────────────

    def skip(self) -> None:
        """A session the layer does not absorb (no valuation published)."""
        for k in self.out:
            self.out[k].append(np.nan)

    def step(self, z_lag: float, dz_lag: float, r_lag: float, dp_now: float,
             z_now: float, phi_z: float) -> None:
        """Absorb one session and publish the read as of its close.

        ``z_lag``, ``dz_lag``, ``r_lag`` are dated t-1; ``dp_now`` is Δp_t;
        ``z_now`` is today's gap (the one the h-session projection starts
        from) and ``phi_z`` the gap's persistence estimated through t.
        """
        if not all(np.isfinite(v) for v in (z_lag, r_lag, dp_now)):
            self.skip()
            return

        if self._sd_p.n < _SCALE_WARMUP:
            self._absorb_scales(z_lag, r_lag, dp_now)
            self.skip()
            return

        sz = max(self._sd_z.mean, _SCALE_FLOOR)
        sr = max(self._sd_r.mean, _SCALE_FLOOR)
        sp = max(self._sd_p.mean, _SCALE_FLOOR)

        dzs = (dz_lag / sz) if np.isfinite(dz_lag) else 0.0
        f_val = np.array([1.0, z_lag / sz])
        f_mom = np.array([1.0, r_lag / sr])
        f_full = np.array([1.0, z_lag / sz, dzs, r_lag / sr])
        y_std = dp_now / sp

        # One-step predictives of Δp_t, from data through t-1.
        preds, vars_ = np.empty(3), np.empty(3)
        ll = np.empty(3)
        for i, (fam, f) in enumerate(((self.val, f_val), (self.mom, f_mom),
                                      (self.full, f_full))):
            preds[i], vars_[i] = fam.forecast(f)
            w_pre = fam.w.copy()
            fam.update(f, y_std)
            # The family's MIXTURE density of what happened, under the member
            # weights that were current before it was seen.
            ll[i] = _mix_ll(w_pre, fam.log_pred_lik)
        ll = np.where(np.isfinite(ll), ll, -1e6)
        self.ll = _POOL_ALPHA * self.ll + ll
        w = _softmax_pool(self.ll)                 # evidence through t

        self._absorb_scales(z_lag, r_lag, dp_now)
        self.n += 1

        if self.n < WARMUP:
            self.skip()
            return

        # ── the price-side response, in log-price units ──────────────────
        b_std = float(self.val.coef[1])
        b_var = float(self.val.coef_var[1])
        b = b_std * sp / sz                         # Δp per unit of gap
        kappa = -b
        rev_prob = (norm_cdf_scalar(-b_std / math.sqrt(b_var))
                    if b_var > 0 and np.isfinite(b_var) else float("nan"))
        if 0.0 < kappa < 1.0:
            half_life = math.log(2.0) / -math.log(1.0 - kappa)
        else:
            half_life = float("nan")

        # ── the h-session move it implies, from TODAY's gap ──────────────
        geom = gap_closure_sum(phi_z, self.h)
        drift_pct = (b * z_now * geom * 100.0
                     if np.isfinite(z_now) and np.isfinite(geom) else float("nan"))

        # Pooled one-step predictive SD of Δp, scaled to h sessions. ECM
        # residuals are close to white, so √h is the honest first-order
        # aggregation; it is what the drift should be read against.
        mu = float(w @ preds)
        var1 = float(w @ (np.maximum(vars_, 1e-18) + (preds - mu) ** 2))
        sd_h_pct = math.sqrt(max(var1, 1e-18)) * sp * math.sqrt(self.h) * 100.0

        self.out["MulaKappa"].append(kappa)
        self.out["MulaRevProb"].append(rev_prob)
        self.out["MulaHalfLife"].append(half_life)
        self.out["MulaDriftPct"].append(drift_pct)
        self.out["MulaSdPct"].append(sd_h_pct)
        self.out["WValuation"].append(float(w[0]))
        self.out["WMomentum"].append(float(w[1]))
        self.out["WFull"].append(float(w[2]))

    def _absorb_scales(self, z_lag: float, r_lag: float, dp_now: float) -> None:
        self._sd_z.update(abs(z_lag))
        self._sd_r.update(abs(r_lag))
        self._sd_p.update(abs(dp_now))

    def frame(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({k: pd.Series(v[:len(index)], index=index)
                             for k, v in self.out.items()})
