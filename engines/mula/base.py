"""
Tattva — MŪLA pipeline core: recursive fair value against the global cross-section.
तत्त्व (Tattva) — "Principle / Essence"

MŪLA (मूल — "root"). The class here was the original Engine-1 rewrite
(codename FVO, "fair value oscillator") that replaced Aarambh's walk-forward
ensemble. Aarambh answered "what forward return do trailing macro momenta
predict?" — a return-space supervised regression whose residual was a forecast
error, not a valuation. This engine answers the question the rest of Tattva
actually consumes: *where should this asset be trading, given the state of
the world?*

Log price is regressed on the **integrated** common factors of the macro
cross-section with time-varying coefficients (a dynamic cointegrating
regression, Bierens & Martins 2010):

    p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t,   F_{j,t} = sum_{s<=t} f_{j,s}

Two properties follow that the return-space regression could not deliver:

1. The residual is a **level**, so fair value is a price, not a forecast, and
   the gap ``e_t = p_t - fv_t`` is a genuine mean-reverting spread. Tattva's
   OU half-life, DFA Hurst and pivot machinery are meaningful on it — under
   Aarambh they were measuring the persistence of a forecast and were
   explicitly flagged as not interpretable.
2. If the relation is cointegrating, the residual is stationary and its mean
   reversion is testable **online** (``mr_prob``), which gates whether the
   valuation is informative today rather than assuming it always is.

Everything downstream of the residual is unchanged Tattva: multi-lookback
robust-quantile z-scores, zone classification, breadth aggregation,
Drift-Diffusion conviction filtering, regime labelling, pivots, divergences
and forward-change significance. The convergence layer, Nishkarsh, the
Intelligence calibrator and the precedent analog matcher therefore consume
exactly the columns they always did — only the fair value underneath them
changed.

The public surface (``fit``/``get_current_signal``/``get_model_stats``/
``get_regime_stats``/``get_signal_performance``/``ts_data``/``ou_params``/
``hurst``/``pivots``) is deliberately identical to the engine it replaces.

Imports math primitives from analytics.* instead of inline definitions.
No Streamlit dependency.

References
----------
Ross, S. A. (1976). "The arbitrage theory of capital asset pricing."
    *Journal of Economic Theory* 13(3).
Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error
    Correction." *Econometrica* 55(2).
Bierens, H. J. & Martins, L. F. (2010). "Time-Varying Cointegration."
    *Econometric Theory* 26(5).
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from core.config import (
    CONVICTION_MODERATE,
    CONVICTION_STRONG,
    CONVICTION_WEAK,
    DDM_DRIFT_SCALE,
    DDM_LEAK_RATE,
    DDM_LONG_RUN_VAR,
    LOOKBACK_WINDOWS,
    OU_PROJECTION_DAYS,
)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover — optional dependency
    _HAS_STATSMODELS = False

from analytics.adaptive import adaptive_tiers
from analytics.conformal import compute_conformal_zscores
from analytics.ddm_filter import drift_diffusion_filter
from analytics.hurst import hurst_dfa
from analytics.ou_process import ornstein_uhlenbeck_estimate
from analytics.structural_breaks import detect_structural_breaks
from analytics.utils import (
    _apply_conviction_bounds,
    _classify_zones,
    _compute_significance,
    _detect_crossover_signals,
)

from .valuation import (BURN_IN, MIN_PRINTS, PANEL_MIN_COVERAGE, PANEL_MIN_PRINTS,
                        SESSION_SPARSE_FRACTION,
                        VALUATION_DELTAS, MarketValuationEngine)

log = logging.getLogger(__name__)


class FairValueEngine:
    """Recursive fair-value inference with Tattva's multi-lookback signal stack.

    Pipeline:
        1. Dynamic cointegrating regression of log price on the integrated
           latent factors AND the integrated asset-class block aggregates of
           the macro cross-section, each view a discount-factor mixture, the
           two views averaged by their own out-of-sample predictive evidence
        2. Multi-lookback rolling robust-quantile z-scores of the mispricing
           gap, and zone classification
        3. Breadth aggregation and raw conviction scoring
        4. Drift-Diffusion filtering of conviction with mean-reverting variance
        5. OU estimation with a Kendall/Orcutt-Winokur bias correction for
           half-life and projection
        6. Hurst exponent via DFA for mean-reversion validation
        7. Swing-based divergence detection
        8. Forward change analysis with significance testing
        9. Structural break detection for regime-aware resetting
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.ou_projection_upper: np.ndarray = np.array([])
        self.ou_projection_lower: np.ndarray = np.array([])
        self.pivots: dict = {}
        self.residual_stats: dict = {}
        self.hurst: float = 0.5
        self.latest_feature_impacts: dict = {}
        self.feature_impact_history: list[dict] = []
        self.theta_history: list[float] = []
        self.break_dates: list[int] = []
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.y: np.ndarray = np.array([])
        self.predictions: np.ndarray = np.array([])
        self.model_spread: np.ndarray = np.array([])
        self.residuals: np.ndarray = np.array([])
        self.price: np.ndarray | None = None
        #: Raw Mūla result bundle (block betas, factor levels, loadings,
        #: leave-one-block-out importance, instrument attribution). Kept whole
        #: so the diagnostics tab can render attribution without the engine
        #: having to flatten every view into ts_data.
        self.mve: dict = {}
        self.block_names: list[str] = []
        #: Per-row conviction tier levels (analytics.adaptive). Empty until fit.
        self._tiers: dict = {}
        #: History required before an estimated tier supersedes its prior. One
        #: year: below that an empirical p90 of |conviction| is a handful of
        #: points and would flap the regime labels around.
        self.tier_min_obs: int = 252
        #: The engine has exactly one mode. Its predecessor carried
        #: `forward_signal` / `cumulative_residual` flags because the same
        #: class had to serve a forward-return forecast and a relative-value
        #: spread, and every downstream reader branched on them; here the
        #: residual is always a level (a mispricing spread), so the flags and
        #: all their branches are gone rather than pinned to constants.
        #: Likewise `purge`: nothing here is a forward-window label, so there
        #: is no overlapping-label leakage to gap out.
        # Per-instrument knobs — default to the global config constants;
        # fit() overrides any the instrument's InstrumentConfig supplies.
        self.burn_in: int = int(BURN_IN)
        self.min_prints: int = int(MIN_PRINTS)
        self.valuation_deltas: tuple = tuple(VALUATION_DELTAS)
        self.lookback_windows: tuple = tuple(LOOKBACK_WINDOWS)
        #: First index carrying a published valuation. Named `min_train_size`
        #: because every downstream consumer (signal-performance burn-in, OU /
        #: Hurst slicing, the Intelligence calibration frame) already reads
        #: that attribute as "where does honest output begin".
        self.min_train_size: int = int(BURN_IN)

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        target_px: pd.Series,
        expl_px: pd.DataFrame,
        printed: pd.DataFrame | None = None,
        feature_names: list[str] | None = None,
        progress_callback: Callable | None = None,
        config: "object | None" = None,
    ) -> "FairValueEngine":
        """Run the full valuation pipeline.

        target_px: the target's price LEVEL, strictly positive. Its index is
            the publication calendar — valuation is published when the target
            prints, not when some other market does.

        expl_px: the explanatory price panel on that same calendar, one column
            per macro instrument, forward-filled (never back-filled). Columns
            are classified into asset-class blocks by ``fvo.blocks``.

        printed: boolean frame matching ``expl_px``, True where a genuine
            print occurred. An instrument contributes to the cross-section
            only on days it actually traded, so a carried-forward stale quote
            cannot masquerade as a zero return and drag a factor toward zero.
            When omitted it is inferred from where the panel changes value,
            which is the best available signal once the caller has already
            forward-filled.

        config: an optional InstrumentConfig (or any object exposing the
            ``fvo_*`` fields). When given, its per-instrument knobs (burn-in,
            print floor, discount grid, lookback windows) override the
            global-constant defaults for THIS fit. Any field it lacks falls
            back to the global default.
        """
        if config is not None:
            self.burn_in = int(getattr(config, "fvo_burn_in", self.burn_in))
            self.min_prints = int(getattr(config, "fvo_min_prints", self.min_prints))
            self.valuation_deltas = tuple(
                getattr(config, "fvo_valuation_deltas", self.valuation_deltas))
            self.lookback_windows = tuple(
                getattr(config, "fvo_lookback_windows", self.lookback_windows))

        target_px = pd.Series(target_px).astype(float)
        expl_px = pd.DataFrame(expl_px).astype(float)
        if len(expl_px) != len(target_px):
            raise ValueError("target and explanatory panel must share a calendar")
        expl_px.index = target_px.index

        self.feature_names = list(feature_names or expl_px.columns)
        expl_px.columns = self.feature_names
        self.n_samples = len(target_px)
        # Kept solely so the completeness gate can tell a STILL-FORMING session
        # from a closed one. ts_data carries a RangeIndex, so without this the
        # gate can only ask "is this the last row I was given?" — which is true
        # of the final row of a truncated backtest too, and would publish a
        # provisional value there that the untruncated run withholds. That is a
        # value published and later withdrawn, i.e. exactly the repainting the
        # gate exists to prevent.
        self._row_index = target_px.index
        self.price = target_px.to_numpy(dtype=np.float64)
        # `y` is the modelled quantity: log price. Kept under the historical
        # name because the analytics stack and the Data tab read `self.y`.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.y = np.log(np.where(self.price > 0, self.price, np.nan))

        if printed is None:
            printed = self._infer_printed(expl_px)
        else:
            printed = pd.DataFrame(printed).reindex(
                columns=self.feature_names).fillna(False).astype(bool)
            printed.index = target_px.index
        # Retained for the panel-completeness gate, which needs per-instrument
        # prints (not just the daily count) to ask "did the instruments whose
        # exchanges were OPEN actually print?" — see _compute_breadth_metrics.
        self._printed = printed

        # Full-sample retrospective diagnostic, surfaced as `break_detected`
        # in get_current_signal(). It never selects an estimation boundary —
        # the valuation regression is recursive and has no training window to
        # choose, so there is nothing here for a break date to contaminate.
        self.break_dates = detect_structural_breaks(
            np.nan_to_num(self.y, nan=0.0))

        def _mve_progress(frac: float) -> None:
            if progress_callback:
                progress_callback(
                    float(np.clip(frac, 0.0, 1.0)),
                    f"Valuing against {expl_px.shape[1]} instruments...",
                )

        core = MarketValuationEngine(
            self.feature_names,
            burn_in=self.burn_in,
            min_prints=self.min_prints,
            deltas=self.valuation_deltas,
        )
        self.mve = core.run(target_px, expl_px, printed, progress_cb=_mve_progress)
        self.block_names = list(core.block_names)

        series = self.mve["series"]
        gap = series["gap"].to_numpy(dtype=np.float64)
        # The mispricing gap IS the tradeable spread: positive = price above
        # the level the cross-section implies (rich), negative = cheap. That
        # is the same polarity the zone/breadth/conviction stack expects, so
        # no sign flip is needed (Aarambh had to negate its forecast).
        self.residuals = gap
        # Fair value as a PRICE level, and the predictive SD of that level in
        # log units — which is what `ModelSpread` has always been rendered as
        # (the tab multiplies by 1e4 and calls it basis points).
        self.predictions = series["fair_value"].to_numpy(dtype=np.float64)
        self.model_spread = np.nan_to_num(
            series["pred_sd"].to_numpy(dtype=np.float64), nan=0.0)
        # First published row — everything before it is burn-in with no
        # honest valuation, and every consumer must be able to exclude it.
        finite = np.isfinite(gap)
        self.min_train_size = int(np.argmax(finite)) if finite.any() else self.n_samples

        self._compute_model_stats(series)
        self._compute_multi_lookback_signals(series)
        self._compute_breadth_metrics()
        self._compute_ddm_conviction()
        self._compute_block_impacts()
        self._find_pivots()
        self._compute_divergences()
        self._compute_forward_changes()
        self._compute_ou_diagnostics()
        self._compute_hurst()

        if progress_callback:
            progress_callback(1.0, "Done")

        return self

    def get_current_signal(self) -> dict:
        """Derive the current composite signal from the latest observation."""
        if self.ts_data.empty:
            return {
                "signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A",
                "conviction_score": 0, "conviction_upper": 0, "conviction_lower": 0,
                "regime": "NEUTRAL", "oversold_breadth": 0, "overbought_breadth": 0,
                "residual": 0, "fair_value": 0, "actual": 0, "avg_z": 0,
                "model_spread": 0, "has_bullish_div": False, "has_bearish_div": False,
                "ou_half_life": 0, "adf_pvalue": 1.0, "kpss_pvalue": 0.0, "hurst": 0.5,
                "theta_stable": True, "break_detected": False,
                "fvo": 0.0, "pct_mispricing": 0.0, "valuation_confidence": 0.0,
                "xs_consistency": 0.0, "mr_prob": 0.0, "gap_half_life": 0.0,
                "market_regime": "initialising", "stress": 0.0, "k_factors": 0,
                "n_available": 0,
            }

        current = self.ts_data.iloc[-1]
        conviction_bounded = current["ConvictionBounded"]

        # Today's cut-points, built from this instrument's history BEFORE today.
        t = self._tiers or {}
        _hi = float(t["strong"][-1]) if "strong" in t else CONVICTION_STRONG
        _lo = float(t["weak"][-1]) if "weak" in t else CONVICTION_WEAK
        # The MODERATE band sits between them; with the pooled priors that is
        # p75, and with estimated tiers it is their midpoint — one fewer
        # quantile to estimate for a distinction that is a display gradation
        # rather than a decision boundary.
        _mid = 0.5 * (_hi + _lo)

        if conviction_bounded < -_hi:
            signal, strength = "BUY", "STRONG"
        elif conviction_bounded < -_mid:
            signal, strength = "BUY", "MODERATE"
        elif conviction_bounded < -_lo:
            signal, strength = "BUY", "WEAK"
        elif conviction_bounded > _hi:
            signal, strength = "SELL", "STRONG"
        elif conviction_bounded > _mid:
            signal, strength = "SELL", "MODERATE"
        elif conviction_bounded > _lo:
            signal, strength = "SELL", "WEAK"
        else:
            signal, strength = "HOLD", "NEUTRAL"

        oversold_breadth = current["OversoldBreadth"]
        overbought_breadth = current["OverboughtBreadth"]

        if signal == "BUY":
            confidence = "HIGH" if oversold_breadth >= 80 else "MEDIUM" if oversold_breadth >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if overbought_breadth >= 80 else "MEDIUM" if overbought_breadth >= 60 else "LOW"
        else:
            conviction_abs = abs(conviction_bounded)
            confidence = "HIGH" if conviction_abs < 10 else "MEDIUM" if conviction_abs < 20 else "LOW"

        theta_stable = True
        if len(self.theta_history) >= 10:
            theta_cv = np.std(self.theta_history[-10:]) / max(np.mean(self.theta_history[-10:]), 1e-6)
            theta_stable = theta_cv < 0.5

        def _f(col: str, default: float = 0.0) -> float:
            v = current.get(col, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return v if np.isfinite(v) else default

        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "conviction_score": conviction_bounded,
            "conviction_upper": current["ConvictionUpper"],
            "conviction_lower": current["ConvictionLower"],
            "regime": current["Regime"],
            "oversold_breadth": oversold_breadth,
            "overbought_breadth": overbought_breadth,
            "residual": current["Residual"],
            "fair_value": current["FairValue"],
            "actual": current["Actual"],
            "avg_z": current["AvgZ"],
            "model_spread": current["ModelSpread"],
            "has_bullish_div": current["BullishDiv"],
            "has_bearish_div": current["BearishDiv"],
            "ou_half_life": self.ou_params.get("half_life", 0),
            "adf_pvalue": self.ou_params.get("adf_pvalue", 1.0),
            "kpss_pvalue": self.ou_params.get("kpss_pvalue", 0.0),
            "hurst": self.hurst,
            "theta_stable": theta_stable,
            "break_detected": len(self.break_dates) > 0,
            # ── Mūla-native readings (wire keys stay "fvo"/"FVO") ──────────
            # The oscillator itself (gap in units of its own predictive SD),
            # the mispricing in percent, and the engine's internal evidence
            # that today's valuation is worth acting on.
            "fvo": _f("FVO"),
            "pct_mispricing": _f("PctMispricing"),
            "valuation_confidence": _f("Confidence"),
            "xs_consistency": _f("XSConsistency"),
            "mr_prob": _f("MRProb"),
            "gap_half_life": _f("GapHalfLife"),
            "market_regime": str(current.get("MarketRegime", "initialising")),
            "stress": _f("Stress"),
            "k_factors": int(_f("KFactors")),
            "n_available": int(_f("NAvailable")),
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        ts = self.ts_data
        # Count only rows carrying a published valuation: the burn-in region
        # (Valid == False) has a neutral-by-construction DDM state, and
        # including those rows dilutes the regime-distribution percentages the
        # Market State card reports ("X% of history classified oversold").
        if "Valid" in ts.columns:
            regimes = ts.loc[ts["Valid"].astype(bool), "Regime"]
            if regimes.empty:
                regimes = ts["Regime"]
        else:
            regimes = ts["Regime"]
        regime_counts = regimes.value_counts()
        return {
            "strongly_oversold": regime_counts.get("STRONGLY OVERSOLD", 0),
            "oversold": regime_counts.get("OVERSOLD", 0),
            "neutral": regime_counts.get("NEUTRAL", 0),
            "overbought": regime_counts.get("OVERBOUGHT", 0),
            "strongly_overbought": regime_counts.get("STRONGLY OVERBOUGHT", 0),
            "current_regime": ts["Regime"].iloc[-1],
        }

    def get_signal_performance(self) -> dict:
        """Forward change analysis with significance testing."""
        ts = self.ts_data
        results = {}
        burn_in = max(self.min_train_size + 50, 80)

        for period in (5, 10, 20):
            buy_changes: list[float] = []
            sell_changes: list[float] = []

            for i in range(burn_in, len(ts) - period, period):
                score = ts["ConvictionScore"].iloc[i]
                fwd = ts.get(f"FwdChg_{period}")
                if fwd is None:
                    continue
                fwd_val = fwd.iloc[i]
                if pd.isna(fwd_val):
                    continue
                # Per-row cut-point, so a bin means the same thing across a
                # regime change in the instrument's own conviction dispersion.
                _cut = (0.5 * (self._tiers["strong"][i] + self._tiers["weak"][i])
                        if self._tiers else CONVICTION_MODERATE)
                if score < -_cut:
                    buy_changes.append(fwd_val)
                if score > _cut:
                    sell_changes.append(-fwd_val)

            buy_stats = _compute_significance(buy_changes)
            sell_stats = _compute_significance(sell_changes)

            results[period] = {
                "buy_avg": buy_stats["mean"],
                "buy_hit": float(np.mean([c > 0 for c in buy_changes])) if buy_changes else 0.0,
                "buy_count": len(buy_changes),
                "buy_t_stat": buy_stats["t_stat"],
                "buy_p_value": buy_stats["p_value"],
                "sell_avg": sell_stats["mean"],
                "sell_hit": float(np.mean([c > 0 for c in sell_changes])) if sell_changes else 0.0,
                "sell_count": len(sell_changes),
                "sell_t_stat": sell_stats["t_stat"],
                "sell_p_value": sell_stats["p_value"],
            }

        return results

    def get_feature_impact_history(self) -> pd.DataFrame:
        if not self.feature_impact_history:
            return pd.DataFrame()
        return pd.DataFrame(self.feature_impact_history)

    def get_block_betas(self) -> pd.DataFrame:
        """Time-varying coefficient on each named asset-class block."""
        return self.mve.get("block_beta", pd.DataFrame())

    def get_instrument_attribution(self) -> pd.DataFrame:
        """Per-instrument contribution to the fitted fair-value level."""
        return self.mve.get("instrument_attribution", pd.DataFrame())

    # ── Private: input preparation ────────────────────────────────────────

    @staticmethod
    def _infer_printed(expl_px: pd.DataFrame) -> pd.DataFrame:
        """Where did each instrument genuinely print?

        Tattva forward-fills the macro panel long before the engine sees it,
        so the vendor's own NaN mask is gone by this point. A carried-forward
        quote is bit-identical to the one before it, so a *change* in value is
        a genuine print; the first finite observation of a column is one too.
        The failure mode is a real print at an unchanged price, which costs
        that instrument one day in the cross-section — the same direction of
        error as the stale-quote exclusion this mask exists to enforce, so it
        is conservative rather than optimistic. Callers holding the pre-ffill
        mask should pass it explicitly.
        """
        v = expl_px.to_numpy(dtype=np.float64)
        finite = np.isfinite(v) & (v > 0)
        changed = np.zeros_like(finite)
        changed[1:] = finite[1:] & finite[:-1] & (v[1:] != v[:-1])
        # first finite row per column counts as a print
        first = finite & (np.cumsum(finite, axis=0) == 1)
        return pd.DataFrame(changed | first, index=expl_px.index,
                            columns=expl_px.columns)

    # ── Private: Analytics Pipeline ───────────────────────────────────────

    def _compute_model_stats(self, series: pd.DataFrame) -> None:
        """Out-of-sample fit quality of the published valuation.

        A level regression on integrated regressors reports a high R² by
        construction — both sides trend — so that number alone says little.
        The comparison that carries information is ``r2_vs_anchor``, and the
        choice of baseline is the whole point.

        The random-walk null Aarambh was scored against ("fair value =
        yesterday's close") is the WRONG yardstick here and is deliberately
        not reported. One step ahead, yesterday's close is unbeatable for any
        near-integrated price, so that comparison returns a large negative
        number for a perfectly sound cointegrating relation — it measures a
        claim this engine never makes. The claim it does make is about the
        *level* price reverts to, and its live competitor is the asset's own
        trailing mean: "price reverts to where it has been" versus "price
        reverts to where the global cross-section says it should be". So the
        anchor is a causal 252-day trailing mean of log price, lagged one
        session. Positive = the cross-section locates the level better than
        the asset's own history does; negative = it does not, and the
        valuation is not adding anything over a moving average.
        """
        gap = self.residuals
        p = self.y
        valid = np.isfinite(gap) & np.isfinite(p)
        n_valid = int(valid.sum())

        if n_valid > 2:
            g = gap[valid]
            pv = p[valid]
            ss_res = float(np.sum(g ** 2))
            ss_tot = float(np.sum((pv - np.mean(pv)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            rmse = float(np.sqrt(np.mean(g ** 2)))
            mae = float(np.mean(np.abs(g)))

            anchor = (pd.Series(p).rolling(252, min_periods=60).mean()
                      .shift(1).to_numpy(dtype=np.float64))
            cmp_ok = valid & np.isfinite(anchor)
            if cmp_ok.sum() > 2:
                ss_anchor = float(np.sum((p[cmp_ok] - anchor[cmp_ok]) ** 2))
                r2_vs_anchor = 1.0 - float(np.sum(gap[cmp_ok] ** 2)) / max(ss_anchor, 1e-10)
            else:
                r2_vs_anchor = 0.0
        else:
            r2, rmse, mae, r2_vs_anchor = 0.0, 0.0, 0.0, 0.0

        def _mean(col: str) -> float:
            if col not in series.columns:
                return float("nan")
            v = pd.to_numeric(series[col], errors="coerce").to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if len(v) else float("nan")

        self.model_stats = {
            "r2_oos": r2,
            "r2_vs_anchor": r2_vs_anchor,
            "rmse_oos": rmse,
            "mae_oos": mae,
            "n_obs": n_valid,
            "n_features": len(self.feature_names),
            "avg_model_spread": float(np.mean(self.model_spread[valid])) if n_valid else 0.0,
            # ── Mūla-native quality readings ───────────────────────────────
            "avg_confidence": _mean("confidence"),
            "avg_xs_consistency": _mean("xs_consistency"),
            "avg_mr_prob": _mean("mr_prob"),
            "avg_k_factors": _mean("k_factors"),
            "avg_explained_var": _mean("explained_var"),
            "avg_n_available": _mean("n_available"),
            "n_blocks": len(self.block_names),
            "burn_in": int(self.min_train_size),
        }

    def _compute_multi_lookback_signals(self, series: pd.DataFrame) -> None:
        r = self.residuals
        n = len(r)
        self.lookback_data = {}

        for lb in self.lookback_windows:
            if n < lb:
                continue
            min_periods = max(lb // 2, 5)
            z_scores, lower_bounds, upper_bounds = compute_conformal_zscores(
                r, window=lb, min_periods=min_periods, alpha=0.05
            )
            zones = _classify_zones(z_scores)
            buy_signals, sell_signals = _detect_crossover_signals(z_scores)
            self.lookback_data[lb] = {
                "z_scores": z_scores, "zones": zones,
                "buy_signals": buy_signals, "sell_signals": sell_signals,
                "lower_bounds": lower_bounds, "upper_bounds": upper_bounds,
            }

        def _col(name: str, default=np.nan) -> np.ndarray:
            if name not in series.columns:
                return np.full(n, default)
            return series[name].to_numpy()

        self.ts_data = pd.DataFrame({
            # `Actual` is the observed price level, `FairValue` the level the
            # cross-section implies, `Residual` the log gap between them.
            "Actual": self.price,
            "FairValue": self.predictions,
            "Residual": self.residuals,
            "ModelSpread": self.model_spread,
            # ── published Mūla diagnostics (column key stays "FVO") ────────
            "FVO": _col("fvo"),
            "PctMispricing": _col("pct_mispricing"),
            "FairValueLo": _col("ci_lo"),
            "FairValueHi": _col("ci_hi"),
            "Confidence": _col("confidence"),
            "XSConsistency": _col("xs_consistency"),
            "MRProb": _col("mr_prob"),
            "GapHalfLife": _col("gap_halflife"),
            "GapPercentile": _col("gap_percentile"),
            "Stress": _col("stress"),
            "KFactors": _col("k_factors"),
            "ExplainedVar": _col("explained_var"),
            "WLatent": _col("w_latent"),
            "WBlock": _col("w_block"),
            "NAvailable": _col("n_available", 0),
            "NAdmitted": _col("n_admitted", 0),
            "MarketRegime": _col("regime_label", "initialising"),
        }, index=pd.RangeIndex(n))
        for lb, data in self.lookback_data.items():
            self.ts_data[f"Z_{lb}"] = data["z_scores"]
            self.ts_data[f"Zone_{lb}"] = data["zones"]
            self.ts_data[f"Buy_{lb}"] = data["buy_signals"]
            self.ts_data[f"Sell_{lb}"] = data["sell_signals"]

    def _compute_breadth_metrics(self) -> None:
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in self.lookback_windows if f"Z_{lb}" in self.ts_data.columns]
        num_lb = max(len(valid_lookbacks), 1)

        oversold = np.zeros(n)
        overbought = np.zeros(n)
        extreme_os = np.zeros(n)
        extreme_ob = np.zeros(n)
        buy_count = np.zeros(n)
        sell_count = np.zeros(n)
        z_scores_list = []

        for lb in valid_lookbacks:
            zones = self.ts_data[f"Zone_{lb}"].values
            z = self.ts_data[f"Z_{lb}"].values
            extreme_os += (zones == "Extreme Under")
            oversold += (zones == "Undervalued")
            extreme_ob += (zones == "Extreme Over")
            overbought += (zones == "Overvalued")
            buy_count += self.ts_data[f"Buy_{lb}"].values
            sell_count += self.ts_data[f"Sell_{lb}"].values
            z_scores_list.append(z)

        # A row with no finite z-score in ANY lookback window has no genuine
        # signal — true both of the ordinary rolling-window warm-up and of the
        # engine's own burn-in, where no valuation is published at all.
        # Without this guard the breadth/count sums are just 0 for a missing
        # row (every zone comparison is False against "N/A"), which silently
        # fabricates a confident "neutral" ConvictionRaw == 0 reading for a
        # period that was never valued — exactly the region the Intelligence
        # calibration frame and the analog feature pool must be able to drop.
        finite_stack = np.vstack([np.isfinite(z) for z in z_scores_list]) if z_scores_list else np.zeros((0, n), dtype=bool)
        valid_row = finite_stack.any(axis=0) if len(finite_stack) else np.zeros(n, dtype=bool)

        # ---- panel-completeness gate ------------------------------------
        # A row fitted on a fraction of the cross-section is not a valuation,
        # it is a valuation of whoever happened to be open. Two runs twenty
        # minutes apart during a live session saw 61 then 62 of ~209
        # instruments posted, and published fair values 1.6% apart — 45x the
        # move in spot. That is not repainting (the past was untouched) but it
        # is a number the reader has no way to discount.
        #
        # MEASURED AGAINST THE INSTRUMENTS WHOSE EXCHANGES WERE OPEN, not against
        # every admitted instrument. The panel spans ~14 venues and NYSE alone is
        # 124 of 241 columns, so a fraction-of-admitted ratio collapses whenever
        # New York is shut -- whether or not anything is actually missing. That
        # made the gate wrong on 83 of the 88 rows it withheld: Good Friday 2024
        # printed 42 of the 45 instruments that could trade (a complete session)
        # and scored 0.175 on the old denominator, indistinguishable from
        # Christmas Day's 10 of 53. Only 4 historical rows are genuinely thin
        # (Christmas, Good Friday, and two New Year's Days, at 0.19-0.22).
        #
        # Two conditions, because a ratio alone cannot express the second:
        #   coverage  -- did the instruments that COULD print actually print?
        #   headcount -- are there enough of them to estimate a factor structure?
        # A day on which 10 instruments were scheduled and all 10 printed scores
        # a perfect ratio and still cannot support k=13.
        # THIN FOR TWO DIFFERENT REASONS, and only one of them is permanent.
        # The NEWEST row is thin because its session is still filling: at 12:15
        # UTC on a Monday 105 of 240 admitted instruments had printed, and that
        # number climbs through the day. A HISTORICAL thin row is thin because
        # it was a holiday — Christmas Day 2025 posted 10 of 240, Good Friday
        # 12 — and it will never fill.
        #
        # Folding both into `Valid` withheld today's reading (which the user
        # legitimately wants to see, and which is minutes away from settling)
        # to buy protection that only the 88 holiday rows actually need. So the
        # two are separated:
        #
        #   publish   -- the value is emitted, and the UI may show it
        #   Valid     -- the value may be TRUSTED by the statistics
        #
        # `Valid` is not a display flag. It gates the analog/precedent feature
        # pool (analytics/analogs.py), adaptive tier estimation (tab_fvo), and
        # the regime distribution. A conviction computed on 4% of the panel must
        # not become a historical precedent that today gets matched against, so
        # holidays stay Valid=False AND unpublished, exactly as before. Today's
        # forming row is published, flagged `Provisional`, and still excluded
        # from all of the above.
        publish = valid_row
        # Pre-gate validity: a row can be withheld because no lookback z-score
        # is finite yet (warm-up) OR because the panel was too thin (the gate
        # below). Those are different answers to "why is there a gap here", so
        # keep them separable before the two get ANDed together.
        finite_ok = valid_row.copy()
        _pr = getattr(self, "_printed", None)
        if _pr is not None and len(_pr) == len(valid_row):
            pr = _pr.to_numpy(dtype=bool)
            # Admission the same way the factor bank does it: an instrument joins
            # once its own cumulative print count reaches the estimability floor.
            adm = np.cumsum(pr, axis=0) >= self.min_prints
            try:
                from data.calendars import panel_session_matrix
                sched = panel_session_matrix(self.feature_names, self._row_index)
            except Exception as _e:            # never let the gate break the run
                log.debug("session matrix unavailable (%s); assuming all open", _e)
                sched = np.ones_like(pr)
            open_adm = adm & sched                       # could have printed
            n_open = open_adm.sum(axis=1).astype(float)
            n_printed = (pr & open_adm).sum(axis=1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                cover = np.where(n_open > 0, n_printed / n_open, 1.0)
            complete = ((~np.isfinite(cover) | (cover >= PANEL_MIN_COVERAGE))
                        & (n_printed >= PANEL_MIN_PRINTS))
            self.ts_data["NScheduled"] = n_open

            # "Forming" means the session has not closed yet — NOT merely "the
            # last row of this slice". A truncated backtest's final row is a
            # closed session, and publishing a provisional value there would make
            # the untruncated run withdraw it. So this requires the row to
            # actually be the current date; when it is not, the publish mask
            # collapses to `valid_row` and behaviour is byte-identical to a plain
            # completeness gate.
            forming = np.zeros(len(valid_row), dtype=bool)
            _ix = getattr(self, "_row_index", None)
            if len(forming) and _ix is not None and len(_ix) == len(forming):
                try:
                    _last = pd.Timestamp(_ix[-1]).normalize()
                    forming[-1] = _last >= pd.Timestamp.today().normalize()
                except (TypeError, ValueError):
                    pass                        # non-dated index: never forming
            publish = valid_row & (complete | forming)
            valid_row = valid_row & complete

        # nanmean legitimately warns "Mean of empty slice" on the burn-in rows
        # (all-NaN across every lookback window) — expected and handled (the
        # result is NaN, masked out below via valid_row), so scope the
        # suppression to this exact expected case rather than relying on a
        # blanket global RuntimeWarning filter.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_z = np.nan_to_num(np.nanmean(np.vstack(z_scores_list), axis=0), nan=0.0) if z_scores_list else np.zeros(n)

        self.ts_data["OversoldBreadth"] = (oversold + extreme_os) / num_lb * 100
        self.ts_data["OverboughtBreadth"] = (overbought + extreme_ob) / num_lb * 100
        self.ts_data["ExtremeOversold"] = extreme_os / num_lb * 100
        self.ts_data["ExtremeOverbought"] = extreme_ob / num_lb * 100
        self.ts_data["BuySignalBreadth"] = buy_count
        self.ts_data["SellSignalBreadth"] = sell_count
        self.ts_data["AvgZ"] = np.where(publish, avg_z, np.nan)
        conviction_raw = (
            (overbought - oversold) / num_lb * 100
            + (extreme_ob - extreme_os) / num_lb * 100 * 1.5
        )
        self.ts_data["ConvictionRaw"] = np.where(publish, conviction_raw, np.nan)
        self.ts_data["Valid"] = valid_row
        # Published but not trusted: the newest session, still filling. Consumers
        # that must not learn from a partial cross-section filter on `Valid`;
        # the UI reads this to label what it is showing.
        self.ts_data["Provisional"] = publish & ~valid_row

        # ---- WHY was a row withheld? holiday, or a degraded fetch? ----------
        #
        # Both look identical on coverage — Christmas 2025 printed 10 of 49
        # scheduled (0.20) and Monday 2026-08-17 printed 51 of 238 (0.21) — so
        # the gate withholds both and the chart shows an identical gap. They are
        # not the same event: one is the market being shut, the other is this
        # run's data being incomplete, and only the second is worth re-running
        # for.
        #
        # `NScheduled` separates them cleanly and is already computed here, so
        # the diagnosis costs nothing and needs no memory of previous runs —
        # which matters because the deploy target has an ephemeral disk and no
        # external store. A holiday has FEW instruments scheduled; a degraded
        # fetch has a normal schedule and few prints.
        reason = np.full(len(valid_row), "", dtype=object)
        try:
            _ns = pd.to_numeric(self.ts_data.get("NScheduled"), errors="coerce")
            _ns = _ns.to_numpy(dtype=float) if _ns is not None else None
        except Exception:                       # noqa: BLE001
            _ns = None
        _ad_arr = None
        try:
            _adm = pd.to_numeric(self.ts_data.get("NAdmitted"), errors="coerce")
            _ad_arr = _adm.to_numpy(dtype=float) if _adm is not None else None
        except Exception:                       # noqa: BLE001
            _ad_arr = None
        if (_ns is not None and _ad_arr is not None
                and len(_ns) == len(valid_row) == len(_ad_arr)):
            withheld = ~publish
            # Burn-in is neither: nothing is published there by design.
            burn = np.zeros(len(valid_row), dtype=bool)
            burn[:min(self.burn_in, len(burn))] = True
            # CONTEMPORANEOUS ratio, not a panel-wide median.
            #
            # Comparing NScheduled against the median across all history
            # misreads the ADMISSION RAMP: in 2017 only ~31 instruments had
            # cleared the print floor, so 31 of 31 were open — scheduled looked
            # "sparse" against a median of 238 and five early rows were labelled
            # "market closed" when the market was fine and the panel was just
            # young. Scheduled OVER ADMITTED has no such failure: a holiday
            # opens few of the admitted (49/240 = 0.20) while the ramp opens
            # nearly all of them (31/31 = 1.00).
            with np.errstate(invalid="ignore", divide="ignore"):
                open_share = np.where(_ad_arr > 0, _ns / _ad_arr, 1.0)
            closed = np.isfinite(open_share) & (open_share < SESSION_SPARSE_FRACTION)
            # FOUR distinct reasons, and conflating any two of them produces a
            # confidently wrong label. Two earlier versions of this classifier
            # did exactly that: comparing NScheduled to a panel-wide median
            # called the admission ramp a holiday, and then, once that was
            # fixed, the ramp became "incomplete fetch" — because those rows
            # pass BOTH the coverage test and the print floor (31 printed of 31
            # scheduled of 31 admitted) and are withheld by the finite-z
            # condition instead. A row is only a data problem if the GATE is
            # what stopped it.
            warming = withheld & ~finite_ok & ~burn
            reason = np.where(withheld & burn, "burn-in",
                      np.where(warming, "warming up",
                      np.where(withheld & closed, "market closed",
                      np.where(withheld, "incomplete fetch", ""))))
        self.ts_data["WithheldReason"] = reason

        # ---- the valuation family follows the same publish rule -------------
        # A fair value is a statement about a cross-section. On the four sessions
        # that fail the gate outright (Christmas, Good Friday, two New Year's
        # Days: 10-17 instruments printing of 49-75 scheduled) the TARGET did not
        # trade either, so `Actual` is a carried-forward quote and `FairValue` is
        # a regression against a tenth of the panel. Publishing that as a point
        # on the chart asserts a valuation that was never made.
        #
        # `Actual` is deliberately NOT masked — it is an observed price, not a
        # claim of ours, and the Data tab should still show what the vendor
        # returned. AvgZ/ConvictionRaw above already follow `publish`, so this
        # makes the rest of the family consistent with them rather than
        # introducing a new convention. Today's forming row is in `publish`, so
        # the live reading is unaffected.
        for _c in ("FairValue", "FairValueLo", "FairValueHi", "Residual",
                   "ModelSpread", "FVO", "PctMispricing", "Confidence"):
            if _c in self.ts_data.columns:
                _v = pd.to_numeric(self.ts_data[_c], errors="coerce").to_numpy(float)
                self.ts_data[_c] = np.where(publish, _v, np.nan)

        # ---- diagnostic: which instruments SHOULD have printed and did not ---
        # Now that "closed" and "missing" are distinguishable, an instrument whose
        # exchange held a session and which still did not print is either
        # genuinely untraded or a failed fetch — and the latter is the panel
        # composition channel that moves published history. Naming them turns a
        # coverage number into something actionable. Diagnosis only: the gate
        # above already decides what gets withheld.
        self.panel_absent: list[str] = []
        if _pr is not None and len(_pr) == len(valid_row) and len(valid_row):
            try:
                _miss = open_adm[-1] & ~pr[-1]
                self.panel_absent = [n for n, m in zip(self.feature_names, _miss) if m]
            except Exception:                       # never break a run to report
                self.panel_absent = []

    def _compute_ddm_conviction(self) -> None:
        raw = self.ts_data["ConvictionRaw"].values
        filtered, _gains, variances = drift_diffusion_filter(
            raw, leak_rate=DDM_LEAK_RATE, drift_scale=DDM_DRIFT_SCALE, long_run_var=DDM_LONG_RUN_VAR
        )
        ddm_std = np.sqrt(np.maximum(variances, 0))
        self.ts_data["ConvictionScore"] = filtered
        bounded = _apply_conviction_bounds(filtered)
        self.ts_data["ConvictionBounded"] = bounded
        self.ts_data["ConvictionUpper"] = _apply_conviction_bounds(filtered + 1.96 * ddm_std)
        self.ts_data["ConvictionLower"] = _apply_conviction_bounds(filtered - 1.96 * ddm_std)

        # Regime cut-points are the causal expanding quantiles of this
        # instrument's OWN |conviction| distribution, not shared constants.
        # The constants were anchored to a pooled p90/p75/p50, which is right
        # for an instrument whose conviction disperses like the pool and wrong
        # for one that does not — a quiet name would sit permanently NEUTRAL
        # and a violent one permanently STRONGLY-something. Each row uses only
        # rows before it, so a label published in 2021 is still that label
        # after 2026's data arrives (research/test_reproducibility.py).
        self._tiers = adaptive_tiers(
            bounded,
            priors={"strong": CONVICTION_STRONG, "weak": CONVICTION_WEAK},
            quantiles={"strong": 0.90, "weak": 0.50},
            min_obs=self.tier_min_obs,
        )
        t_strong, t_weak = self._tiers["strong"], self._tiers["weak"]
        regimes = []
        for i, score_bounded in enumerate(bounded):
            hi, lo = t_strong[i], t_weak[i]
            if score_bounded < -hi:
                regimes.append("STRONGLY OVERSOLD")
            elif score_bounded < -lo:
                regimes.append("OVERSOLD")
            elif score_bounded > hi:
                regimes.append("STRONGLY OVERBOUGHT")
            elif score_bounded > lo:
                regimes.append("OVERBOUGHT")
            else:
                regimes.append("NEUTRAL")
        self.ts_data["Regime"] = regimes
        self.ts_data["TierStrong"] = t_strong
        self.ts_data["TierWeak"] = t_weak

    def _compute_block_impacts(self) -> None:
        """Driver importance from the leave-one-block-out ablations.

        This replaces Aarambh's ``coef @ pca.components_`` attribution, which
        was a linear read-off of a fitted model. Here the importance of a
        block is measured by *refitting without it* and reporting how far the
        mispricing moves — an ablation, so it survives the collinearity that
        makes coefficient magnitudes uninterpretable across ~200 macro series
        that all load on the same few factors. The values are already
        normalised to sum to 1 per row by the core engine; they are scaled to
        percent here to match the contract the diagnostics tab renders.
        """
        imp = self.mve.get("block_importance")
        if imp is None or imp.empty:
            self.latest_feature_impacts = {}
            self.feature_impact_history = []
            return

        pct = imp.astype(float) * 100.0
        rows = pct.dropna(how="all")
        if rows.empty:
            self.latest_feature_impacts = {}
            self.feature_impact_history = []
            return

        latest = rows.iloc[-1].dropna()
        self.latest_feature_impacts = dict(
            sorted({str(k): float(v) for k, v in latest.items()}.items(),
                   key=lambda kv: kv[1], reverse=True))
        # One row per rebalance-scale sample rather than every session: the
        # history feeds a stacked-area chart, and ~2,300 near-identical rows
        # render no better than ~110 while costing the tab a full redraw.
        step = max(1, len(rows) // 120)
        pos = {ts: i for i, ts in enumerate(pct.index)}
        self.feature_impact_history = [
            {"index": pos[ts], **{str(k): float(v) for k, v in row.items()
                                  if np.isfinite(v)}}
            for ts, row in rows.iloc[::step].iterrows()
        ]

    def _compute_divergences(self) -> None:
        n = len(self.ts_data)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        order = 5
        if n < order * 3:
            self.ts_data["BullishDiv"] = bull_div
            self.ts_data["BearishDiv"] = bear_div
            return

        # Local extrema on the log price level — always genuinely available
        # here (the engine is handed the price series directly), so there is
        # no cumsum reconstruction and no horizon inflation to guard against.
        price = np.nan_to_num(self.y, nan=0.0)
        residual = np.asarray(self.residuals)
        # NaN residuals (the burn-in region) must not participate in extrema
        # comparisons: NumPy's argmax/argmin treat NaN as the maximum, which
        # would misplace pivots into the unvalued window. Comparisons against
        # NaN already evaluate False, so only argmax/argmin need guarding.
        finite_res = np.isfinite(residual)
        last_low_idx = -1
        last_high_idx = -1
        expanding_std = pd.Series(residual).expanding(min_periods=min(20, max(2, len(residual) // 3))).std().ffill().fillna(0.0).values

        for i in range(order * 2, n):
            window_price = price[i - 2 * order : i + 1]
            window_ok = finite_res[i - 2 * order : i + 1].all()
            if not window_ok:
                continue
            if np.argmin(window_price) == order:
                curr_low = i - order
                if last_low_idx != -1 and price[curr_low] < price[last_low_idx] and residual[curr_low] > residual[last_low_idx]:
                    if residual[curr_low] < -expanding_std[curr_low] * 0.5:
                        bull_div[i] = True
                last_low_idx = curr_low
            if np.argmax(window_price) == order:
                curr_high = i - order
                if last_high_idx != -1 and price[curr_high] > price[last_high_idx] and residual[curr_high] < residual[last_high_idx]:
                    if residual[curr_high] > expanding_std[curr_high] * 0.5:
                        bear_div[i] = True
                last_high_idx = curr_high

        self.ts_data["BullishDiv"] = bull_div
        self.ts_data["BearishDiv"] = bear_div

    def _find_pivots(self, order: int = 5) -> None:
        r = np.asarray(self.residuals)
        n = len(r)
        conf_tops, conf_bottoms, top_vals, bottom_vals = [], [], [], []
        # NaN residuals (burn-in) sort as the max under NumPy's argmax/argmin —
        # guard windows touching them so pivots are never placed in the
        # unvalued region.
        finite_r = np.isfinite(r)

        for i in range(order * 2, n):
            if not finite_r[i - 2 * order : i + 1].all():
                continue
            window = r[i - 2 * order : i + 1]
            if np.argmax(window) == order:
                conf_tops.append(i)
                top_vals.append(r[i - order])
            if np.argmin(window) == order:
                conf_bottoms.append(i)
                bottom_vals.append(r[i - order])

        r_finite_only = r[finite_r]
        fallback_top = float(pd.Series(r_finite_only).ewm(alpha=0.05).mean().iloc[-1] + pd.Series(r_finite_only).ewm(alpha=0.05).std().iloc[-1]) if len(r_finite_only) > 0 else 0.0
        fallback_bottom = float(pd.Series(r_finite_only).ewm(alpha=0.05).mean().iloc[-1] - pd.Series(r_finite_only).ewm(alpha=0.05).std().iloc[-1]) if len(r_finite_only) > 0 else 0.0

        self.pivots = {
            "tops": conf_tops, "bottoms": conf_bottoms,
            "avg_top": float(np.mean(top_vals)) if top_vals else fallback_top,
            "avg_bottom": float(np.mean(bottom_vals)) if bottom_vals else fallback_bottom,
        }
        self.ts_data["IsPivotTop"] = False
        self.ts_data["IsPivotBottom"] = False
        if conf_tops:
            self.ts_data.loc[conf_tops, "IsPivotTop"] = True
        if conf_bottoms:
            self.ts_data.loc[conf_bottoms, "IsPivotBottom"] = True

    def _compute_forward_changes(self) -> None:
        price = pd.Series(self.price)
        for period in (5, 10, 20):
            fwd = (price.shift(-period) / price - 1.0) * 100.0
            self.ts_data[f"FwdChg_{period}"] = np.clip(fwd.values, -100, 100)

    def _compute_ou_diagnostics(self) -> None:
        # Unlike the forecast series this replaced, the gap IS a candidate
        # mean-reverting spread, so theta / half-life / ADF / KPSS are
        # interpretable here rather than being reported with a caveat.
        r = self.residuals
        oos_r = r[self.min_train_size:]
        oos_r = oos_r[np.isfinite(oos_r)]

        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            try:
                adf_pvalue = float(adfuller(oos_r, autolag="AIC")[1]) if _HAS_STATSMODELS else 1.0
            except Exception:
                adf_pvalue = 1.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_pvalue = float(kpss(oos_r, regression="c", nlags="auto")[1]) if _HAS_STATSMODELS else 0.0
            except Exception:
                kpss_pvalue = 0.0

            vol_multiplier = 1.0
            window_size = min(60, len(oos_r) // 3)
            self.theta_history = []
            for i in range(window_size, len(oos_r)):
                theta_roll, _, _ = ornstein_uhlenbeck_estimate(oos_r[i - window_size : i])
                self.theta_history.append(theta_roll)

            theta_std = np.std(self.theta_history) if len(self.theta_history) > 1 else 0.0
            dynamic_theta = self.theta_history[-1] if self.theta_history else theta
        else:
            _finite = r[np.isfinite(r)]
            theta, mu, sigma = 0.05, 0.0, max(float(np.std(_finite)) if len(_finite) else 0.0, 1e-6)
            adf_pvalue, kpss_pvalue, vol_multiplier = 1.0, 0.0, 1.0
            dynamic_theta, theta_std = theta, 0.0

        self.ou_params = {
            "theta": theta, "dynamic_theta": dynamic_theta, "mu": mu, "sigma": sigma,
            "half_life_base": np.log(2) / max(theta, 1e-4),
            "half_life": np.log(2) / max(dynamic_theta, 1e-4),
            "stationary_std": sigma / np.sqrt(2 * max(theta, 1e-4)),
            "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue,
            "vol_multiplier": vol_multiplier, "theta_std": theta_std,
        }

        current_r = float(oos_r[-1]) if len(oos_r) else 0.0
        proj_days = np.arange(1, OU_PROJECTION_DAYS + 1)
        self.ou_projection = mu + (current_r - mu) * np.exp(-dynamic_theta * proj_days)

        if theta_std > 0:
            self.ou_projection_upper = mu + (current_r - mu) * np.exp(-(dynamic_theta - theta_std) * proj_days)
            self.ou_projection_lower = mu + (current_r - mu) * np.exp(-(dynamic_theta + theta_std) * proj_days)
        else:
            self.ou_projection_upper = self.ou_projection.copy()
            self.ou_projection_lower = self.ou_projection.copy()

    def _compute_hurst(self) -> None:
        """DFA Hurst of the mispricing gap — retained, but NOT a headline.

        ``analytics.hurst.hurst_dfa`` fits a single log-log slope over box
        sizes from 8 up to n/4. That is textbook DFA and it is unbiased for a
        long-memory series, but it is badly biased upward for a SHORT-memory
        one, which is exactly what a mean-reverting valuation gap is. Below the
        correlation time the profile is smooth and the local slope runs toward
        1.5; above it the slope is 0.5; a single fit over a range that straddles
        the crossover returns something in between. Measured on synthetic AR(1)
        with the gap's own ~7-day half-life: H reads 0.96 at n=1354 and is
        still 0.78 at n=20000, against a true asymptotic 0.5. So a strongly
        stationary gap reports as "trending".

        The value is therefore computed (the signal contract and the analog
        feature pool both read it) but the Mūla tab surfaces ``adf_pvalue`` and
        the online ``gap_halflife`` instead, which measure the same property
        and are correct. Fixing the estimator — fitting only the large-scale
        asymptote, or scaling min_scale to the series' own correlation time —
        would change the analog matcher's Hurst feature too, so it belongs in
        its own change with its own validation, not here.
        """
        oos_r = self.residuals[self.min_train_size:]
        oos_r = oos_r[np.isfinite(oos_r)]
        # DFA log-log regression needs >=200 points for >=5 scale pairs.
        # Below that, the CI is ~±0.3 — meaningless — so report a neutral 0.5.
        if len(oos_r) > 200:
            if self.ou_params.get("adf_pvalue", 1.0) > 0.05:
                self.hurst = 0.5
            else:
                self.hurst = hurst_dfa(oos_r)
        else:
            self.hurst = 0.5
