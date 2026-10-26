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
2. If the relation is genuinely cointegrating, the residual is stationary.
   Stationarity of the GAP is not the same claim as "price returns to fair
   value", though: a gap closes when either side moves, and on real data
   most of the closing is fair value moving toward price (measured: price
   covers 9% / 2% / 24% of the daily gap closure on Gold / S&P 500 /
   USD-INR). Whether PRICE responds to the gap is tested separately, by the
   error-correction read in ``engines/mula/ecm.py``; this module publishes
   the gap and its own persistence.

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

What drives the published level is read off the fit itself: factor and
block levels are linear in the standardised returns, so the fitted level
decomposes exactly into per-instrument contributions, summed here to the
asset-class blocks (``block_attribution``). (Leave-one-block-out refits used
to run beside the two views, for a driver ablation and a cross-section
"consistency" score. They were a separate model of the block view only, the
score measured block-versus-latent view mismatch rather than the agreement
its label claimed, and nothing on the call read it; they went.)

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
from analytics.causal import EPS, DynamicModelAverage, OnlineAR1
from .factors import K_MAX, AdaptiveVolPanel, OnlineFactorModel
from .blocks import block_membership

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

#: The publication warm-up after the burn-in: a session is valued once the gap
#: has printed on at least PUBLISH_WARMUP_PRINTS of the PUBLISH_WARMUP_WINDOW
#: sessions before it. It is the warm-up the retired window stack imposed (its
#: shortest window, 21 sessions, read nothing below 10 prints), kept as the
#: publication rule so that deleting the stack did not move one published value
#: (tests/test_golden_mula.py). On a prepared spine the gap prints every session
#: after the burn-in, so it withholds the first 10 of them as 'warming up'.
PUBLISH_WARMUP_PRINTS: int = 10
PUBLISH_WARMUP_WINDOW: int = 21

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
        """One causal pass: the fair value, its gap and the gap's persistence.

        Returns ``series`` (one row per session: ``fair_value``, ``gap``,
        ``pct_mispricing``, ``fvo``, ``pred_sd``, ``fv_sd``, ``ci_lo`` /
        ``ci_hi``, ``k_factors``, ``gap_halflife``, ``gap_phi``, ``price``,
        ``n_available``, ``n_admitted``) and ``block_attribution`` (every 21
        sessions and the last: each block's contribution to the fitted level).
        """
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

        block_var = np.ones(self.n_blocks)
        block_w = np.zeros(self.n_blocks)
        block_levels = np.zeros(self.n_blocks)
        block_contrib = np.zeros((self.n_blocks, self.n))
        blam = math.exp(-math.log(2.0) / 252.0)

        lat_bank: DynamicModelAverage | None = None
        blk_bank: DynamicModelAverage | None = None
        view_logw = np.zeros(2)

        gap_ar = OnlineAR1(halflife=504.0)
        # Causal empirical quantile of |fvo| — the conformal band multiplier.
        # min_obs guards the opening stretch where a quantile of a handful of
        # residuals is noise; below it the Gaussian constant is used unchanged.
        fvo_rank = AdaptiveThreshold(min_obs=CI_MIN_OBS)
        # Working miscoverage level for the adaptive-conformal band; starts at
        # the nominal and is driven by realised coverage from there.
        ci_alpha = 1.0 - CI_LEVEL

        # -- output buffers ------------------------------------------------
        out = {k: np.full(T, np.nan) for k in (
            "fair_value", "gap", "pct_mispricing", "fvo", "pred_sd", "fv_sd",
            "ci_lo", "ci_hi", "k_factors", "gap_halflife", "gap_phi",
        )}
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

            # ---- factors ---------------------------------------------------
            finfo = fac.update(z, avail)
            k = finfo["k"]

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

            # ---- predict, then absorb (strict order) ------------------------
            f_lat, q_lat, qp_lat = lat_bank.forecast_parts(Flat)
            f_blk, q_blk, qp_blk = blk_bank.forecast_parts(Fblk)

            w = np.exp(view_logw - view_logw.max())
            w = w / max(w.sum(), EPS)
            fv = float(w[0] * f_lat + w[1] * f_blk)
            view_spread = float(w[0] * (f_lat - fv) ** 2 + w[1] * (f_blk - fv) ** 2)
            qv = float(w[0] * q_lat + w[1] * q_blk) + view_spread
            # TWO SCALES, and they answer different questions.
            #
            # `sd` is the predictive SD of the next OBSERVED log price around
            # the fitted level: parameter uncertainty PLUS the regression's own
            # residual noise. It is the natural unit for the gap — "how many
            # typical gaps away is price?" — so it stays the FVO denominator
            # (published as GapScale).
            #
            # `fv_sd` drops the residual noise and keeps only what is uncertain
            # about the LEVEL itself: the coefficients' posterior and the
            # disagreement between the two views. That is what "Model Spread —
            # how tightly is fair value pinned?" claims to be, and it is not
            # what it was: measured on Gold, 95.6% of the old Model Spread's
            # variance was the residual term (parameter SD ~66 bps against
            # ~274 bps displayed), so the card was reporting the size of a
            # typical gap and calling it uncertainty about fair value.
            sd = math.sqrt(max(qv, 1e-12))
            fv_sd = math.sqrt(max(float(w[0] * qp_lat + w[1] * qp_blk) + view_spread, 0.0))

            gap = p - fv
            fvo = gap / sd

            # ---- absorb -----------------------------------------------------
            w_lat_pre = lat_bank.w.copy()
            w_blk_pre = blk_bank.w.copy()
            lat_bank.update(Flat, p)
            blk_bank.update(Fblk, p)
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

            # The gap's own persistence. `phi` is published (GapPhi) because the
            # error-correction layer needs it to project the gap forward over
            # the horizon; the half-life is the same number in days.
            #
            # (A "mean-reversion probability" P(phi < 1) used to be formed here
            # and fed the Confidence score. After a few hundred sessions it read
            # 1.000 on every row of every instrument measured — the gap's
            # stationarity is not in doubt — so it gated nothing, and its label
            # invited the reading "price reverts", which is the ECM's question,
            # not this one's. Reversion evidence now comes from the price side;
            # see engines/mula/ecm.py.)
            if np.isfinite(phi) and np.isfinite(se) and se > EPS:
                hl = (math.log(2.0) / -math.log(min(max(phi, 1e-6), 0.999999))
                      if 0 < phi < 1 else np.nan)
            else:
                hl = np.nan

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

            out["fair_value"][t] = math.exp(fv)
            out["gap"][t] = gap
            out["pct_mispricing"][t] = pct
            out["fvo"][t] = fvo
            out["pred_sd"][t] = sd
            out["fv_sd"][t] = fv_sd
            out["ci_lo"][t] = math.exp(fv - zc * sd)
            out["ci_hi"][t] = math.exp(fv + zc * sd)
            out["k_factors"][t] = k
            out["gap_halflife"][t] = hl
            out["gap_phi"][t] = phi

            if t % 21 == 0 or t == T - 1:
                # Each instrument's part of the fitted level (the views' own
                # coefficients times its exact share of every factor and block
                # level), summed to its asset-class block.
                inst = (w[0] * (lat_bank.coef[1:K_MAX + 1] @ fac.instrument_contribution())
                        + w[1] * (blk_bank.coef[1:] @ block_contrib))
                attrib_dates.append(t)
                attrib_rows.append(self.block_M @ inst)

            if progress_cb is not None and (t % 250 == 0 or t == T - 1):
                progress_cb((t + 1) / T)

        df = pd.DataFrame({k: v for k, v in out.items()}, index=idx)
        df["price"] = np.exp(P)
        df["n_available"] = avail_hist
        df["n_admitted"] = admit_hist

        return {
            "series": df,
            "block_attribution": pd.DataFrame(
                np.array(attrib_rows) if attrib_rows else np.zeros((0, self.n_blocks)),
                index=idx[attrib_dates] if attrib_dates else pd.DatetimeIndex([]),
                columns=self.block_names),
        }
