"""
Tattva -- MŪLA core: Market Valuation Engine (the long-run equilibrium leg).
=================================================================

Question answered: *where should this asset be trading, given the state of
the world?*

Formulation
-----------
Log price is regressed on the integrated common factors of the global
cross-section with time-varying coefficients:

    p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t ,   F_{j,t} = sum_{s<=t} f_{j,s}

This is a *dynamic cointegrating regression* in the sense of Bierens &
Martins (2010): p and F are both integrated, and e is the deviation from the
time-varying long-run relation -- i.e. mispricing.  Two properties follow
that a return-space regression cannot deliver:

1. The residual is a *level*, so "fair value" is a price, not a forecast.
2. If the relation is genuinely cointegrating, the residual is stationary
   and its mean reversion is testable online -- which is exactly the gate
   the decision layer needs to know whether valuation is informative today.

Why a level regression and not a valuation model on fundamentals?  Because
the question is relative pricing against the traded opportunity set, which
is what an arbitrageur can actually act on (Ross 1976; Ross's APT prices an
asset by its exposures to priced factors, not by its cash flows).

Two independent valuation views are maintained and averaged by their own
out-of-sample evidence:

* **Latent view** -- integrated principal factors of the cross-section.
  Maximum statistical efficiency, weak economic labels.
* **Block view** -- integrated asset-class aggregates (equity beta, term
  structure, credit, commodities, dollar, volatility, ...).  Lower
  efficiency, but every coefficient has a name, which is what makes the
  output auditable.

Leave-one-block-out refits give ablation-based driver importance and a
cross-sectional consistency score: independent slices of the world either
agree about the mispricing or they do not, and that agreement is itself a
decision-relevant quantity.

References
----------
Ross, S. A. (1976). "The arbitrage theory of capital asset pricing."
    *Journal of Economic Theory* 13(3).
Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error
    Correction." *Econometrica* 55(2).
Bierens, H. J. & Martins, L. F. (2010). "Time-Varying Cointegration."
    *Econometric Theory* 26(5).
Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
    Kalman Filter*. Cambridge University Press.
West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic Models*.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analytics.adaptive import AdaptiveThreshold
from analytics.causal import (EPS, BatchDLM, DynamicModelAverage, EWMA,
                              ExpandingRank, OnlineAR1, norm_cdf_scalar)
from .factors import K_MAX, AdaptiveVolPanel, OnlineFactorModel
from .blocks import block_membership
from .regime import RegimeFilter

#: Observations absorbed before valuation is published.  One year is the
#: point at which an exponentially weighted correlation matrix over ~200
#: instruments has accumulated enough weight for the Marchenko-Pastur edge
#: to be meaningful; publishing earlier would be publishing the prior.
BURN_IN = 252

#: Prints an instrument must have accumulated *by time t* before it may
#: enter the cross-section at time t.  A second moment is not estimable from
#: less, and the gate is applied forward in time so admission never
#: retroactively changes.
MIN_PRINTS = 250

#: Fraction of the ADMITTED cross-section that must actually have printed on a
#: date for that date's valuation to be published Valid. A settled session
#: sits near 1.0; a half-open one (observed: 61 of ~209 during a live run)
#: falls far below and is withheld rather than published as if complete.
#: 0.60 leaves room for genuine holiday overlap across the global panel — many
#: dates legitimately have a fraction of markets shut — while still catching a
#: session that has merely not finished posting.
PANEL_MIN_COVERAGE: float = 0.60

#: Absolute floor on the number of instruments that actually printed, applied
#: alongside PANEL_MIN_COVERAGE. A RATIO CANNOT EXPRESS THIS. Christmas Day 2025
#: had 10 of the 53 instruments whose exchanges were open — if only 10 had been
#: scheduled and all 10 printed, the ratio would read 1.00 and admit a row that
#: cannot pin a factor structure at all. Realised `k` runs to 13, so ~2.3x that
#: is the minimum cross-section worth estimating one from.
PANEL_MIN_PRINTS: int = 30

#: Two-sided coverage the FairValueLo/Hi band targets, and the minimum number of
#: resolved residuals before the empirical multiplier replaces the Gaussian one.
#: The band is conformal (see the note in `run`): `pred_sd` sets the scale and
#: this quantile of realised |fvo| sets the width, so coverage is calibrated to
#: the engine's own errors instead of assuming they are normal. Measured before
#: this change, the nominal-95% band covered 81.0% / 89.6% / 87.0% on
#: Gold / Copper / Silver.
#: Share of the ADMITTED cross-section that must be scheduled-open for a session
#: to count as a normal trading day. Below it the market was shut; at or above
#: it, a withheld row means the DATA was incomplete, not the market.
#: Contemporaneous by design — measured against admitted, never against a
#: panel-wide median, because early history has few instruments admitted and all
#: of them open, which a median comparison misreads as a holiday.
SESSION_SPARSE_FRACTION: float = 0.60

CI_LEVEL: float = 0.95
CI_MIN_OBS: int = 252

#: Adaptive-conformal step size. Each published interval nudges the working
#: miscoverage level by at most this much, so the band chases realised coverage
#: without lurching on any single outcome. 0.01 moves the level ~1pp per miss,
#: i.e. it re-calibrates over ~100 observations — fast enough to follow a
#: dispersion regime, slow enough that a fortnight of quiet does not collapse
#: the band. The clamps stop a pathological stretch driving the level to a
#: degenerate 0% or 50% interval.
CI_ADAPT_GAMMA: float = 0.01
CI_ALPHA_MIN: float = 0.005      # never wider than a ~99.5% interval
CI_ALPHA_MAX: float = 0.20       # never tighter than an ~80% interval

#: Discount grid for the valuation regression, spanning implied coefficient
#: memories of ~4 years, ~8 years, ~40 years and permanent.
#:
#: The restriction to the slow end of the family is a modelling commitment,
#: not a tuned choice, and it is the single most consequential decision in
#: this engine.  Scoring discount factors by one-step predictive likelihood
#: is degenerate for a *level* regression: the model that tracks price most
#: closely always wins, and its limit is the useless statement "fair value =
#: price".  Empirically, admitting delta = 0.99 (~5 months of memory)
#: collapses the mispricing by roughly a factor of three and its half-life
#: from weeks to a few days -- a residual, not a valuation.  A long-run
#: relation that re-estimates
#: itself in months is not a long-run relation.  Within the slow family the
#: data still selects, and the selection is stable: shifting the grid one
#: notch slower moves the reported mispricing by well under a percentage
#: point.
VALUATION_DELTAS = (0.999, 0.9995, 0.9999, 1.0)


def _mix_ll(w: np.ndarray, ll: np.ndarray) -> float:
    """log of a weighted mixture of one-step predictive densities."""
    ll = np.where(np.isfinite(ll), ll, -1e6)
    mx = float(ll.max())
    return float(mx + math.log(max(float(np.sum(w * np.exp(ll - mx))), 1e-300)))


def _blocks(tickers: list[str]) -> tuple[list[str], np.ndarray]:
    """Block membership matrix (n_blocks x n_assets) from the asset classes."""
    names, mapping = block_membership(list(tickers))
    M = np.zeros((len(names), len(tickers)))
    idx = {b: i for i, b in enumerate(names)}
    for j, t in enumerate(tickers):
        M[idx[mapping[t]], j] = 1.0
    return names, M


class MarketValuationEngine:
    """Recursive fair-value inference against the global opportunity set."""

    def __init__(self, tickers: list[str], burn_in: int = BURN_IN,
                 min_prints: int = MIN_PRINTS,
                 deltas: tuple[float, ...] = VALUATION_DELTAS) -> None:
        self.tickers = list(tickers)
        self.n = len(tickers)
        self.block_names, self.block_M = _blocks(self.tickers)
        self.n_blocks = len(self.block_names)
        # Per-instrument overrides (Tattva threads these from InstrumentConfig).
        # They are estimability floors and a memory grid, not free parameters:
        # see the module constants for why each is set where it is.
        self.burn_in = int(burn_in)
        self.min_prints = int(min_prints)
        self.deltas = tuple(deltas)

    # -----------------------------------------------------------------
    def run(self, target_px: pd.Series, expl_px: pd.DataFrame,
            printed: pd.DataFrame, progress_cb=None) -> dict:
        idx = target_px.index
        T = len(idx)
        cols = list(expl_px.columns)
        assert cols == self.tickers, "panel columns must match engine universe"

        P = np.log(np.asarray(target_px.values, dtype=float))
        X = np.asarray(expl_px.values, dtype=float)                 # (T, n)
        PR = np.asarray(printed.values, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            LX = np.log(np.where(X > 0, X, np.nan))
        R = np.vstack([np.full((1, self.n), np.nan), np.diff(LX, axis=0)])
        # Causal admission: an instrument joins the cross-section on the day
        # its own accumulated print count first reaches the estimability
        # floor, and contributes only on days it actually printed.
        n_prints = np.cumsum(PR, axis=0)

        vol = AdaptiveVolPanel(self.n)
        fac = OnlineFactorModel(self.n)
        regime = RegimeFilter(d=2)

        block_var = np.ones(self.n_blocks)
        block_w = np.zeros(self.n_blocks)
        block_levels = np.zeros(self.n_blocks)
        block_contrib = np.zeros((self.n_blocks, self.n))
        blam = math.exp(-math.log(2.0) / 252.0)

        gvol = EWMA(halflife=42.0)
        glogvol = EWMA(halflife=252.0)
        # Continuous market stress.  The HMM posterior is a *classification*
        # and on daily data it saturates at 0/1, which makes it useless as a
        # graded input to the decision layer.  The empirical percentile of
        # realised global volatility against its own history is graded,
        # distribution-free, and equally causal.
        stress_rank = ExpandingRank()
        # Cross-sectional dispersion of standardised returns.  A liquidity /
        # market-quality proxy that needs no volume or spread data: when
        # liquidity provision withdraws, the cross-section fans out even
        # after each name has been normalised by its own volatility
        # (Ang et al. 2006; Herskovic et al. 2016 on common idiosyncratic
        # volatility as a priced liquidity-linked factor).
        disp_rank = ExpandingRank()

        lat_bank: DynamicModelAverage | None = None
        blk_bank: DynamicModelAverage | None = None
        loo: BatchDLM | None = None
        # row j is the block-view design with block j ablated
        loo_mask = np.ones((self.n_blocks, self.n_blocks + 1))
        for j in range(self.n_blocks):
            loo_mask[j, 1 + j] = 0.0
        view_logw = np.zeros(2)

        gap_ar = OnlineAR1(halflife=504.0)
        gap_rank = ExpandingRank()
        # Causal empirical quantile of |fvo| — the conformal band multiplier.
        # min_obs guards the opening stretch where a quantile of a handful of
        # residuals is noise; below it the Gaussian constant is used unchanged.
        fvo_rank = AdaptiveThreshold(min_obs=CI_MIN_OBS)
        # Working miscoverage level for the adaptive-conformal band; starts at
        # the nominal and is driven by realised coverage from there.
        ci_alpha = 1.0 - CI_LEVEL
        resid_ewma = EWMA(halflife=252.0)
        agree_ew = 0.0
        agree_w = 0.0

        # -- output buffers ------------------------------------------------
        out = {k: np.full(T, np.nan) for k in (
            "fair_value", "gap", "pct_mispricing", "fvo", "pred_sd", "ci_lo",
            "ci_hi", "resid_rmse", "k_factors", "explained_var", "corr_memory",
            "t_eff", "confidence", "xs_consistency", "gap_halflife",
            "mr_prob", "stress", "regime_stress", "regime_persistence",
            "switch_prob", "regime_drift", "regime_entropy",
            "w_latent", "w_block", "gap_percentile", "adapt_memory",
            "factor_var_share", "loo_dispersion", "xs_dispersion",
            "xs_dispersion_pct",
        )}
        out["regime_label"] = np.array(["initialising"] * T, dtype=object)
        lat_beta = np.full((T, K_MAX + 1), np.nan)
        blk_beta = np.full((T, self.n_blocks + 1), np.nan)
        blk_level_hist = np.full((T, self.n_blocks), np.nan)
        fac_level_hist = np.full((T, K_MAX), np.nan)
        fac_contrib_hist = np.full((T, K_MAX), np.nan)
        blk_contrib_hist = np.full((T, self.n_blocks), np.nan)
        blk_importance = np.full((T, self.n_blocks), np.nan)
        attrib_dates: list[int] = []
        attrib_rows: list[np.ndarray] = []

        avail_hist = np.zeros(T, dtype=int)
        # How many instruments were ELIGIBLE to print on t (past the print
        # floor), as distinct from how many actually did. The ratio of the two
        # is the only stable measure of panel completeness: a raw count cannot
        # separate "early history, few instruments admitted yet" from "live
        # session, most of the panel has not posted", and those need opposite
        # treatment. Admission ramps from ~0 to the full universe over the
        # record, so gating on a fraction of the UNIVERSE would invalidate
        # legitimate early history; gating on a fraction of the ADMITTED set
        # does not.
        admit_hist = np.zeros(T, dtype=int)

        # -----------------------------------------------------------------
        for t in range(T):
            r = R[t]
            sig = vol.sigma()                       # uses data through t-1
            mu = vol.mean()
            vol.update(r)                           # accumulate before admission
            avail = (np.isfinite(r) & PR[t] & (n_prints[t] >= self.min_prints)
                     & (sig > 1e-7))
            avail_hist[t] = int(avail.sum())
            admit_hist[t] = int(np.sum(n_prints[t] >= self.min_prints))

            z = np.where(avail, (r - mu) / np.maximum(sig, 1e-8), 0.0)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- global risk factor in raw return units (for the regime) --
            V = fac.loadings()
            if V.shape[1] > 0 and avail.any():
                w0 = V[:, 0] * avail
                nrm = float(np.abs(w0).sum())
                g = float(np.dot(np.nan_to_num(r, nan=0.0), w0) / nrm) if nrm > EPS else 0.0
            else:
                g = float(np.nanmean(np.where(avail, r, np.nan))) if avail.any() else 0.0
                g = 0.0 if not np.isfinite(g) else g
            gs = gvol.std
            gz = g / gs if gs > EPS else 0.0
            lv = math.log(max(gs, 1e-8))
            lvz = (lv - glogvol.mean) / max(glogvol.std, 1e-6) if glogvol.n > 20 else 0.0
            reg = regime.update(np.array([np.clip(gz, -8, 8), np.clip(lvz, -8, 8)]))
            stress_c = stress_rank.cdf(lv) if glogvol.n > 60 else np.nan
            stress_rank.update(lv)
            gvol.update(g)
            glogvol.update(lv)

            # ---- factors ---------------------------------------------------
            finfo = fac.update(z, avail)
            k = finfo["k"]

            if avail_hist[t] >= 5:
                xs_disp = float(np.std(z[avail]))
                xs_disp_pct = disp_rank.cdf(xs_disp)
                disp_rank.update(xs_disp)
            else:
                xs_disp, xs_disp_pct = np.nan, np.nan

            # ---- asset-class blocks ---------------------------------------
            cnt = self.block_M @ avail.astype(float)
            braw = np.where(cnt > 0, (self.block_M @ z) / np.maximum(cnt, 1.0), 0.0)
            bsd = np.sqrt(np.where(block_w > EPS, block_var / np.maximum(block_w, EPS), 1.0))
            bsd = np.maximum(bsd, 1e-6)
            braw_z = braw / bsd
            bz = np.clip(braw_z, -6.0, 6.0)
            # winsorisation is applied as a scale factor so the attribution
            # below stays an exact decomposition of the published level
            wins = np.where(np.abs(braw_z) > EPS, bz / np.where(
                np.abs(braw_z) > EPS, braw_z, 1.0), 1.0)
            block_levels = block_levels + bz
            # exact attribution of each block increment to its members
            per_member = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1.0), 0.0)
            block_contrib += (self.block_M * (per_member * wins / bsd)[:, None]
                              ) * z[None, :]
            block_var = blam * block_var + (1.0 - blam) * braw ** 2
            block_w = blam * block_w + (1.0 - blam)

            p = P[t]
            if not np.isfinite(p):
                continue

            # ---- regressors -------------------------------------------------
            # A factor that drops below the Marchenko-Pastur edge stops
            # *accumulating* -- its level freezes -- but it is not zeroed out
            # of the design.  Zeroing it would delete beta_j * F_j from the
            # fitted level in a single session and put a step change in fair
            # value at every point where the factor count moved, which the
            # oscillator would then report as a mispricing.  A frozen level is
            # a constant regressor: nearly collinear with the intercept, which
            # the discounted prior handles, and continuous, which is what
            # matters.
            Flat = np.concatenate([[1.0], finfo["levels"][:K_MAX]])
            Fblk = np.concatenate([[1.0], block_levels])

            if t < self.burn_in:
                continue
            if lat_bank is None:
                # priors anchored on the burn-in sample: strictly past data
                hist = P[max(0, t - self.burn_in):t]
                hist = hist[np.isfinite(hist)]
                p0 = float(np.mean(hist)) if len(hist) else float(p)
                v0 = float(np.var(np.diff(hist))) if len(hist) > 2 else 1e-4
                v0 = max(v0, 1e-6)
                pm_l = np.zeros(K_MAX + 1)
                pm_l[0] = p0
                pm_b = np.zeros(self.n_blocks + 1)
                pm_b[0] = p0
                lat_bank = DynamicModelAverage(
                    K_MAX + 1, grid=self.deltas, prior_scale=v0 * 25.0,
                    prior_var=v0 * 25.0, prior_mean=pm_l)
                blk_bank = DynamicModelAverage(
                    self.n_blocks + 1, grid=self.deltas, prior_scale=v0 * 25.0,
                    prior_var=v0 * 25.0, prior_mean=pm_b)
                loo = BatchDLM(self.n_blocks, self.n_blocks + 1,
                               np.full(self.n_blocks, 0.999),
                               prior_scale=v0 * 25.0, prior_var=v0 * 25.0,
                               prior_mean=pm_b)

            # ---- predict, then absorb (strict order) ------------------------
            f_lat, q_lat = lat_bank.forecast(Flat)
            f_blk, q_blk = blk_bank.forecast(Fblk)

            w = np.exp(view_logw - view_logw.max())
            w = w / max(w.sum(), EPS)
            fv = float(w[0] * f_lat + w[1] * f_blk)
            qv = float(w[0] * q_lat + w[1] * q_blk
                       + w[0] * (f_lat - fv) ** 2 + w[1] * (f_blk - fv) ** 2)
            sd = math.sqrt(max(qv, 1e-12))

            gap = p - fv
            fvo = gap / sd

            Floo = loo_mask * Fblk[None, :]
            f_loo, _, _ = loo.forecast(Floo)
            loo_gaps = p - f_loo

            # ---- absorb -----------------------------------------------------
            w_lat_pre = lat_bank.w.copy()
            w_blk_pre = blk_bank.w.copy()
            lat_bank.update(Flat, p)
            blk_bank.update(Fblk, p)
            loo.update(Floo, p)
            # mixture predictive log-likelihood of each view, scored with the
            # weights that were current *before* p_t was seen
            ll = np.array([
                _mix_ll(w_lat_pre, lat_bank.log_pred_lik),
                _mix_ll(w_blk_pre, blk_bank.log_pred_lik),
            ])
            ll = np.where(np.isfinite(ll), ll, -1e6)
            view_logw = 0.99 * view_logw + ll
            view_logw -= view_logw.max()

            gap_ar.update(gap)
            phi, se = gap_ar.solve()
            resid_ewma.update(gap * gap)

            if np.isfinite(phi) and np.isfinite(se) and se > EPS:
                mr_prob = norm_cdf_scalar((1.0 - phi) / se)
                hl = (math.log(2.0) / -math.log(min(max(phi, 1e-6), 0.999999))
                      if 0 < phi < 1 else np.nan)
            else:
                mr_prob, hl = np.nan, np.nan

            sgn = np.sign(gap)
            agree = float(np.mean(np.sign(loo_gaps) == sgn)) if sgn != 0 else 0.5
            # Daily agreement across twelve ablations is a twelve-point
            # proportion and jumps between discrete levels session to session.
            # It is averaged over the *mispricing's own inferred half-life* --
            # the timescale on which a disagreement would actually matter --
            # rather than over an arbitrary window.  Exponential averaging is
            # one-sided, so this smooths without reaching forward.
            hl_s = float(np.clip(hl, 5.0, 126.0)) if np.isfinite(hl) else 21.0
            lam_s = math.exp(-math.log(2.0) / hl_s)
            agree_ew = lam_s * agree_ew + (1.0 - lam_s) * agree
            agree_w = lam_s * agree_w + (1.0 - lam_s)
            xs_cons = max(0.0, 2.0 * (agree_ew / max(agree_w, EPS)) - 1.0)
            loo_disp = float(np.std(loo_gaps) / max(abs(gap), sd))

            # The mean-reversion term is usually saturated at 1 -- after
            # thousands of sessions the residual's stationarity is a settled
            # question -- so it acts as a gate that only bites in the regime
            # where the cointegrating relation has genuinely broken down.
            conf = float(np.sqrt(max(mr_prob, 0.0) * max(xs_cons, 0.0))
                         ) if np.isfinite(mr_prob) else np.nan

            beta_l = lat_bank.coef
            beta_b = blk_bank.coef
            contrib_l = beta_l[1:] * finfo["levels"][:K_MAX]
            contrib_b = beta_b[1:] * block_levels

            pct = math.expm1(gap)

            # CONFORMAL BAND WIDTH, not a Gaussian one.
            #
            # `sd` is the model-averaging MIXTURE sd: it carries the two views'
            # own uncertainty and their disagreement, but nothing about how far
            # price actually lands from the fitted value. Multiplying it by the
            # Gaussian 1.96 therefore assumes the standardised residual `fvo` is
            # unit-normal, and it is not. Measured 2026-08-17 on 10y panels, the
            # realised coverage of the nominal-95% band was:
            #
            #     Gold 81.0%   ·   Copper 89.6%   ·   Silver 87.0%
            #
            # Under-covering by up to 14 points is not a cosmetic problem: these
            # bands are what a position would be sized against, so the interval
            # understates risk exactly when it is being relied on.
            #
            # The fix keeps `sd` as the SCALE and calibrates only the multiplier,
            # from the engine's own realised |fvo| — a conformal interval.
            # Distribution-free, so fat tails are absorbed rather than assumed
            # away, and strictly causal: `quantile()` is read BEFORE `update()`,
            # so the width used at t reflects only residuals resolved before t.
            # Below `min_obs` it returns the Gaussian constant, so early history
            # is unchanged and the estimate takes over only once better informed.
            # ADAPTIVE level, because a plain conformal quantile still misses.
            #
            # A fixed-level empirical quantile is only calibrated if the |fvo|
            # distribution is STATIONARY, and residual dispersion plainly is not:
            # `ExpandingRank` never forgets, so a calm early stretch holds the
            # quantile down and the band lags every rise in dispersion. Measured
            # with a fixed level, coverage came in at 91.7% / 93.3% / 94.0%
            # (Gold / Copper / Silver) POST warm-up — better than the Gaussian
            # 81.0% / 89.6% / 87.0%, but still short, and Copper was worse after
            # warm-up than during it, which rules out the opening year as the
            # cause.
            #
            # So the level tracks realised coverage instead of being fixed
            # (adaptive conformal inference; Gibbs & Candes 2021). Miss too often
            # and alpha falls, widening the next band; cover too often and it
            # rises, tightening. The update uses only whether the interval
            # published at t contained the outcome AT t, so it stays causal, and
            # it drives long-run coverage to CI_LEVEL under drift rather than
            # assuming the dispersion that held a decade ago still holds.
            zc = fvo_rank.quantile(1.0 - ci_alpha, fallback=1.959963984540054)
            _missed = 1.0 if abs(fvo) > zc else 0.0
            ci_alpha = float(np.clip(
                ci_alpha + CI_ADAPT_GAMMA * ((1.0 - CI_LEVEL) - _missed),
                CI_ALPHA_MIN, CI_ALPHA_MAX))
            fvo_rank.update(abs(fvo))
            gap_pct = gap_rank.cdf(gap)                 # rank among prior gaps
            gap_rank.update(gap)

            out["fair_value"][t] = math.exp(fv)
            out["gap"][t] = gap
            out["pct_mispricing"][t] = pct
            out["fvo"][t] = fvo
            out["pred_sd"][t] = sd
            out["ci_lo"][t] = math.exp(fv - zc * sd)
            out["ci_hi"][t] = math.exp(fv + zc * sd)
            out["resid_rmse"][t] = math.sqrt(max(resid_ewma.mean, 0.0))
            out["k_factors"][t] = k
            out["explained_var"][t] = finfo["explained"]
            out["corr_memory"][t] = finfo["memory"]
            out["t_eff"][t] = finfo["t_eff"]
            out["confidence"][t] = conf
            out["xs_consistency"][t] = xs_cons
            out["loo_dispersion"][t] = loo_disp
            out["gap_halflife"][t] = hl
            out["mr_prob"][t] = mr_prob
            out["stress"][t] = stress_c
            out["xs_dispersion"][t] = xs_disp
            out["xs_dispersion_pct"][t] = xs_disp_pct
            out["regime_stress"][t] = reg["stress"]
            out["regime_persistence"][t] = reg["persistence"]
            out["switch_prob"][t] = reg["switch_prob"]
            out["regime_drift"][t] = reg["drift"]
            out["regime_entropy"][t] = reg["entropy"]
            out["regime_label"][t] = reg["label"]
            out["w_latent"][t] = w[0]
            out["w_block"][t] = w[1]
            out["gap_percentile"][t] = gap_pct
            out["adapt_memory"][t] = (w[0] * lat_bank.effective_memory
                                      + w[1] * blk_bank.effective_memory)
            tot_c = float(np.sum(np.abs(contrib_l)) + np.sum(np.abs(contrib_b)))
            out["factor_var_share"][t] = (
                float(np.sum(np.abs(contrib_l))) / tot_c if tot_c > EPS else np.nan)

            lat_beta[t] = beta_l
            blk_beta[t] = beta_b
            blk_level_hist[t] = block_levels
            fac_level_hist[t] = finfo["levels"][:K_MAX]
            fac_contrib_hist[t] = contrib_l
            blk_contrib_hist[t] = contrib_b
            denom = max(float(np.sum(np.abs(loo_gaps - gap))), EPS)
            blk_importance[t] = np.abs(loo_gaps - gap) / denom

            if t % 21 == 0 or t == T - 1:
                inst = (w[0] * (beta_l[1:K_MAX + 1] @ fac.instrument_contribution())
                        + w[1] * (beta_b[1:] @ block_contrib))
                attrib_dates.append(t)
                attrib_rows.append(inst.copy())

            if progress_cb is not None and (t % 250 == 0 or t == T - 1):
                progress_cb((t + 1) / T)

        df = pd.DataFrame({k: v for k, v in out.items()}, index=idx)
        df["price"] = np.exp(P)
        df["n_available"] = avail_hist
        df["n_admitted"] = admit_hist

        return {
            "series": df,
            "latent_beta": pd.DataFrame(
                lat_beta, index=idx,
                columns=["intercept"] + [f"F{j+1}" for j in range(K_MAX)]),
            "block_beta": pd.DataFrame(
                blk_beta, index=idx, columns=["intercept"] + self.block_names),
            "block_levels": pd.DataFrame(blk_level_hist, index=idx,
                                         columns=self.block_names),
            "factor_levels": pd.DataFrame(
                fac_level_hist, index=idx,
                columns=[f"F{j+1}" for j in range(K_MAX)]),
            "factor_contrib": pd.DataFrame(
                fac_contrib_hist, index=idx,
                columns=[f"F{j+1}" for j in range(K_MAX)]),
            "block_contrib": pd.DataFrame(blk_contrib_hist, index=idx,
                                          columns=self.block_names),
            "block_importance": pd.DataFrame(blk_importance, index=idx,
                                             columns=self.block_names),
            "instrument_attribution": pd.DataFrame(
                np.array(attrib_rows) if attrib_rows else np.zeros((0, self.n)),
                index=idx[attrib_dates] if attrib_dates else pd.DatetimeIndex([]),
                columns=self.tickers),
            "loadings": pd.DataFrame(
                fac.loadings(), index=self.tickers,
                columns=[f"F{j+1}" for j in range(fac.loadings().shape[1])]),
            "burn_in": self.burn_in,
        }
