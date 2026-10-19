"""
Tattva — MŪLA: recursive fair value against the global cross-section.
तत्त्व (Tattva) — "Principle / Essence"

MŪLA (मूल — "root") answers the one question the rest of Tattva consumes:
*where should this asset be trading, given the state of the world?* It
replaced Aarambh's walk-forward ensemble, which asked "what forward return do
trailing macro momenta predict?" — a return-space regression whose residual
was a forecast error, not a valuation.

Log price is regressed on the **integrated** common factors of the macro
cross-section with time-varying coefficients (a dynamic cointegrating
regression, Bierens & Martins 2010):

    p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t,   F_{j,t} = sum_{s<=t} f_{j,s}

The residual is a **level**, so fair value is a price, not a forecast, and the
gap ``e_t = p_t - fv_t`` is a genuine spread. Everything Mūla publishes is a
reading of that one gap:

* **The oscillator** ``FVO`` — the gap in units of its own typical size
  (``GapScale``). Positive = rich, negative = cheap. The system's directional
  claim and the Cheap / Fair / Rich state (``core.semantics``) come from it,
  and from nothing else.
* **Its persistence** — ``GapHalfLife`` online, the OU half-life and ADF /
  KPSS on the full history. These say the GAP is short-lived; they do not say
  which side closes it.
* **The error-correction read** (``ecm.py``) — whether PRICE responds to the
  gap (κ̂, its evidence ``MRProb``, the implied h-session move), and whether
  valuation-led forecasts are winning the predictive pool today.
* **Cross-sectional agreement** — whether independent slices of the world
  (leave-one-block-out refits) agree on the gap's sign.
* **The window stack** — robust z-scores of the gap against its own trailing
  1-month / 1-quarter / 1-year history, the share of windows reading it
  cheap or rich, and a drift-diffusion filter of their agreement. Evidence
  about how unusual today's gap is; never a rival direction.

The public surface is ``fit`` / ``get_current_signal`` / ``get_model_stats``
/ ``get_regime_stats`` / ``ts_data`` / ``ou_params`` / ``mula_summary``, plus
``get_block_betas`` / ``get_instrument_attribution`` /
``get_feature_impact_history`` for the auditable block view.

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
    DDM_DRIFT_SCALE,
    DDM_LEAK_RATE,
    DDM_LONG_RUN_VAR,
    LOOKBACK_WINDOWS,
    OU_PROJECTION_DAYS,
)
from core.semantics import VALUATION_STATES, claim_direction, valuation_state

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover — optional dependency
    _HAS_STATSMODELS = False

from analytics.ddm_filter import drift_diffusion_filter
from analytics.ou_process import ornstein_uhlenbeck_estimate
from analytics.robust_z import rolling_robust_z
from analytics.utils import (
    _apply_conviction_bounds,
    _classify_zones,
    _detect_crossover_signals,
)

from .ecm import COLUMNS as ECM_COLUMNS
from .ecm import MulaLayer
from .valuation import (BURN_IN, MIN_PRINTS, PANEL_MIN_COVERAGE, PANEL_MIN_PRINTS,
                        SESSION_SPARSE_FRACTION,
                        VALUATION_DELTAS, MarketValuationEngine)

log = logging.getLogger(__name__)


class MulaEngine:
    """Recursive fair-value inference and the readings built on its gap.

    Pipeline:
        1. Dynamic cointegrating regression of log price on the integrated
           latent factors AND the integrated asset-class block aggregates of
           the macro cross-section, each view a discount-factor mixture, the
           two views averaged by their own out-of-sample predictive evidence
        2. Rolling robust z-scores of the gap against its own 1m / 3m / 1y
           history, and zone classification
        3. The panel-completeness gate (what is published, what is trusted),
           window breadth and raw window agreement
        4. Drift-diffusion filtering of that agreement
        5. The error-correction read: does price respond to the gap?
        6. Fit quality, block-ablation driver importance, OU diagnostics
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.ou_projection_upper: np.ndarray = np.array([])
        self.ou_projection_lower: np.ndarray = np.array([])
        self.latest_feature_impacts: dict = {}
        self.feature_impact_history: list[dict] = []
        self.theta_history: list[float] = []
        self.feature_names: list[str] = []
        self.n_samples: int = 0
        self.y: np.ndarray = np.array([])
        self.predictions: np.ndarray = np.array([])
        #: Uncertainty about the fair-value LEVEL (log units) — see
        #: valuation.py's `fv_sd`. Published as ModelSpread.
        self.model_spread: np.ndarray = np.array([])
        #: The FVO denominator: predictive SD of price around fair value.
        self.gap_scale: np.ndarray = np.array([])
        self.residuals: np.ndarray = np.array([])
        self.price: np.ndarray | None = None
        #: The error-correction read as of the last published session — what
        #: the hero's reversion gate and the Precedent state vector consume.
        #: Keys: kappa, half_life, rev_prob, expected_drift_pct,
        #: drift_sd_pct, w_valuation, w_momentum, w_full.
        self.mula_summary: dict = {}
        #: Instruments whose exchange was open on the newest row and that did
        #: not print — the panel-composition channel that moves history.
        self.panel_absent: list[str] = []
        #: Raw Mūla result bundle (block betas, factor levels, loadings,
        #: leave-one-block-out importance, instrument attribution). Kept whole
        #: so the diagnostics tab can render attribution without the engine
        #: having to flatten every view into ts_data.
        self.mve: dict = {}
        self.block_names: list[str] = []
        #: The engine has exactly one mode. Its predecessor carried
        #: `forward_signal` / `cumulative_residual` flags because the same
        #: class had to serve a forward-return forecast and a relative-value
        #: spread, and every downstream reader branched on them; here the
        #: residual is always a level (a mispricing spread), so the flags and
        #: all their branches are gone rather than pinned to constants.
        #: Likewise `purge`: nothing here is a forward-window label, so there
        #: is no overlapping-label leakage to gap out.
        # Per-instrument knobs — placeholders until fit() reads them from the
        # instrument's InstrumentConfig.
        self.burn_in: int = int(BURN_IN)
        self.min_prints: int = int(MIN_PRINTS)
        self.valuation_deltas: tuple = tuple(VALUATION_DELTAS)
        self.lookback_windows: tuple = tuple(LOOKBACK_WINDOWS)
        #: THE horizon h (InstrumentConfig.forecast_horizon) — the error-
        #: correction read projects over it. Set by fit(); there is no default.
        self.horizon: int | None = None
        #: d*, Mūla's as-of session: the date of the newest published
        #: valuation. Every verdict input is read at this date.
        self.asof: pd.Timestamp | None = None
        #: The run's analysis epoch and the prepared history's fingerprint
        #: (data.prep), so the fit says which history it valued.
        self.epoch: pd.Timestamp | None = None
        self.fingerprint: str = ""
        #: First index carrying a published valuation. Named `min_train_size`
        #: because downstream consumers (OU slicing, the run console) read
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
        *,
        config,
        exchange: str = "",
        epoch: pd.Timestamp | None = None,
        fingerprint: str = "",
    ) -> "FairValueEngine":
        """Run the full valuation pipeline.

        target_px: the target's price LEVEL, strictly positive. Its index is
            the publication calendar — valuation is published when the target
            prints, not when some other market does.

        expl_px: the explanatory price panel on that same calendar, one column
            per macro instrument, forward-filled (never back-filled). Columns
            are classified into asset-class blocks by ``engines.mula.blocks``.

        printed: boolean frame matching ``expl_px``, True where a genuine
            print occurred. An instrument contributes to the cross-section
            only on days it actually traded, so a carried-forward stale quote
            cannot masquerade as a zero return and drag a factor toward zero.
            When omitted it is inferred from where the panel changes value,
            which is the best available signal once the caller has already
            forward-filled.

        config: the instrument's InstrumentConfig — required. Its knobs
            (burn-in, print floor, discount grid, lookback windows) and its
            ``forecast_horizon``, the one h, are read from it directly.

        exchange: the target's exchange (``data.calendars.resolve_exchange``);
            its local date decides whether the newest row is a session still
            forming (``data.calendars.session_today``).

        epoch / fingerprint: the analysis epoch and the prepared history's
            fingerprint (``data.prep``), stored on the engine as provenance.
        """
        self.burn_in = int(config.fvo_burn_in)
        self.min_prints = int(config.fvo_min_prints)
        self.valuation_deltas = tuple(config.fvo_valuation_deltas)
        self.lookback_windows = tuple(config.fvo_lookback_windows)
        self.horizon = int(config.forecast_horizon)
        self._exchange = str(exchange or "")
        self.epoch = pd.Timestamp(epoch) if epoch is not None else None
        self.fingerprint = str(fingerprint or "")

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

        # (A "structural break" test ran here, surfaced as `break_detected`.
        # It imported a statsmodels Bai-Perron class that has never existed,
        # so every run fell through to a rolling-mean heuristic that reported
        # three breaks on any input — pure noise included — and nothing read
        # the result. Removed rather than repaired: the valuation regression
        # is recursive and has no estimation boundary for a break to select.)

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
        # Fair value as a PRICE level; the uncertainty about that level, in
        # log units (ModelSpread — the tab renders it ×1e4 as basis points);
        # and the scale the gap is measured in (GapScale, the FVO denominator).
        self.predictions = series["fair_value"].to_numpy(dtype=np.float64)
        self.model_spread = series["fv_sd"].to_numpy(dtype=np.float64)
        self.gap_scale = series["pred_sd"].to_numpy(dtype=np.float64)
        # First published row — everything before it is burn-in with no
        # honest valuation, and every consumer must be able to exclude it.
        finite = np.isfinite(gap)
        self.min_train_size = int(np.argmax(finite)) if finite.any() else self.n_samples

        self._compute_multi_lookback_signals(series)
        self._compute_breadth_metrics()
        self._compute_ddm_conviction()
        self._compute_error_correction(series)
        self._compute_model_stats(series)
        self._compute_block_impacts()
        self._compute_ou_diagnostics()

        r = self._last_published_row()
        self.asof = (pd.Timestamp(self._row_index[r])
                     if r is not None and isinstance(self._row_index, pd.DatetimeIndex)
                     else None)

        if progress_callback:
            progress_callback(1.0, "Done")

        return self

    def _last_published_row(self) -> int | None:
        """Index of the newest row carrying a published valuation."""
        if self.ts_data.empty or "FVO" not in self.ts_data.columns:
            return None
        fvo = pd.to_numeric(self.ts_data["FVO"], errors="coerce").to_numpy(dtype=float)
        ok = np.flatnonzero(np.isfinite(fvo))
        return int(ok[-1]) if len(ok) else None

    def get_current_signal(self) -> dict:
        """Mūla's reading as of the newest PUBLISHED valuation.

        "Newest published", not "newest row". When the latest session fails
        the completeness gate (an incomplete fetch on a closed session), its
        valuation is NaN by design. This used to be read through a NaN → 0
        coercion, so a missing valuation came back as FVO = 0.00 — "fairly
        valued, within noise" — and the hero rendered a confident HOLD on a
        number the engine had declined to publish. The last valuation the
        engine DID publish is returned instead, flagged so the reader can say
        it is not today's: ``asof_row_withheld`` and ``withheld_reason``
        describe the newest row, ``asof`` is the returned row's date (d*) and
        ``provisional`` says whether that row is a session still forming.
        """
        r = self._last_published_row()
        if r is None:
            return {
                "valuation_state": "NO VALUATION", "direction": 0,
                "regime": "NO VALUATION", "fvo": float("nan"),
                "pct_mispricing": float("nan"), "fair_value": float("nan"),
                "actual": float("nan"), "residual": float("nan"),
                "gap_scale": float("nan"), "model_spread": float("nan"),
                "valuation_confidence": float("nan"), "xs_consistency": float("nan"),
                "mr_prob": float("nan"), "gap_half_life": float("nan"),
                "ou_half_life": float("nan"), "adf_pvalue": 1.0, "kpss_pvalue": 0.0,
                "theta_stable": True, "conviction_score": 0.0,
                "conviction_upper": 0.0, "conviction_lower": 0.0,
                "oversold_breadth": float("nan"), "overbought_breadth": float("nan"),
                "avg_z": float("nan"), "market_regime": "initialising",
                "stress": float("nan"), "k_factors": 0, "n_available": 0,
                "mula_kappa": float("nan"), "mula_half_life": float("nan"),
                "expected_drift_pct": float("nan"), "drift_sd_pct": float("nan"),
                "w_valuation": float("nan"), "w_momentum": float("nan"),
                "w_full": float("nan"), "asof_row_withheld": True,
                "withheld_reason": "no valuation published", "published_row": None,
                "asof": None, "provisional": False,
            }

        row = self.ts_data.iloc[r]
        last = self.ts_data.iloc[-1]

        def _f(col: str) -> float:
            """A float, or NaN — never a made-up zero."""
            try:
                v = float(row.get(col, np.nan))
            except (TypeError, ValueError):
                return float("nan")
            return v if np.isfinite(v) else float("nan")

        theta_stable = True
        if len(self.theta_history) >= 10:
            theta_cv = np.std(self.theta_history[-10:]) / max(np.mean(self.theta_history[-10:]), 1e-6)
            theta_stable = bool(theta_cv < 0.5)

        fvo = _f("FVO")
        withheld = r != len(self.ts_data) - 1
        return {
            # ── the claim ────────────────────────────────────────────────
            "valuation_state": valuation_state(fvo),
            "direction": claim_direction(fvo),
            "regime": valuation_state(fvo),
            "fvo": fvo,
            "pct_mispricing": _f("PctMispricing"),
            "fair_value": _f("FairValue"),
            "actual": _f("Actual"),
            "residual": _f("Residual"),
            "gap_scale": _f("GapScale"),
            "model_spread": _f("ModelSpread"),
            # ── evidence about the claim ────────────────────────────────
            "valuation_confidence": _f("Confidence"),
            "xs_consistency": _f("XSConsistency"),
            "mr_prob": _f("MRProb"),
            "gap_half_life": _f("GapHalfLife"),
            "ou_half_life": float(self.ou_params.get("half_life", np.nan)),
            "adf_pvalue": float(self.ou_params.get("adf_pvalue", 1.0)),
            "kpss_pvalue": float(self.ou_params.get("kpss_pvalue", 0.0)),
            "theta_stable": theta_stable,
            # ── the window stack (evidence, not a second direction) ─────
            "conviction_score": _f("ConvictionBounded"),
            "conviction_upper": _f("ConvictionUpper"),
            "conviction_lower": _f("ConvictionLower"),
            "oversold_breadth": _f("OversoldBreadth"),
            "overbought_breadth": _f("OverboughtBreadth"),
            "avg_z": _f("AvgZ"),
            # ── the world it was valued in ──────────────────────────────
            "market_regime": str(row.get("MarketRegime", "initialising")),
            "stress": _f("Stress"),
            "k_factors": int(np.nan_to_num(_f("KFactors"))),
            "n_available": int(np.nan_to_num(_f("NAvailable"))),
            # ── error correction: does price close the gap? ─────────────
            "mula_kappa": _f("MulaKappa"),
            "mula_half_life": _f("MulaHalfLife"),
            "expected_drift_pct": _f("MulaDriftPct"),
            "drift_sd_pct": _f("MulaSdPct"),
            "w_valuation": _f("WValuation"),
            "w_momentum": _f("WMomentum"),
            "w_full": _f("WFull"),
            # ── which session this is ───────────────────────────────────
            "asof_row_withheld": bool(withheld),
            "withheld_reason": str(last.get("WithheldReason", "")) if withheld else "",
            "published_row": r,
            "asof": self.asof,
            "provisional": bool(row.get("Provisional", False)),
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        """How this instrument's history divides across the valuation states.

        Counted on trusted (``Valid``) rows only: the burn-in and the withheld
        holiday sessions carry no valuation, and including them would dilute
        every share by the unvalued fraction.
        """
        ts = self.ts_data
        if ts.empty or "Regime" not in ts.columns:
            return {s.lower().replace(" ", "_"): 0 for s in VALUATION_STATES} | {
                "current_regime": "NO VALUATION"}
        regimes = ts["Regime"]
        if "Valid" in ts.columns:
            _v = regimes[ts["Valid"].astype(bool)]
            regimes = _v if not _v.empty else regimes
        counts = regimes.value_counts()
        out = {s.lower().replace(" ", "_"): int(counts.get(s, 0)) for s in VALUATION_STATES}
        out["current_regime"] = self.get_current_signal()["valuation_state"]
        return out

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

        # Averages over TRUSTED rows (`Valid`), the same population every
        # other statistic of this engine is quoted on.
        trusted = (self.ts_data["Valid"].to_numpy(dtype=bool)
                   if "Valid" in self.ts_data.columns else valid)

        def _mean(col: str) -> float:
            if col not in self.ts_data.columns:
                return float("nan")
            v = pd.to_numeric(self.ts_data[col], errors="coerce").to_numpy(dtype=np.float64)
            v = v[trusted & np.isfinite(v)]
            return float(np.mean(v)) if len(v) else float("nan")

        self.model_stats = {
            "r2_oos": r2,
            "r2_vs_anchor": r2_vs_anchor,
            "rmse_oos": rmse,
            "mae_oos": mae,
            "n_obs": n_valid,
            "n_features": len(self.feature_names),
            # Uncertainty about the fair-value level, and the typical size of
            # the gap — two different numbers since ModelSpread stopped
            # including the residual noise (valuation.py, `fv_sd`).
            "avg_model_spread": _mean("ModelSpread"),
            "avg_gap_scale": _mean("GapScale"),
            # ── Mūla-native quality readings ───────────────────────────────
            "avg_confidence": _mean("Confidence"),
            "avg_xs_consistency": _mean("XSConsistency"),
            "avg_mr_prob": _mean("MRProb"),
            "avg_k_factors": _mean("KFactors"),
            "avg_explained_var": _mean("ExplainedVar"),
            "avg_n_available": _mean("NAvailable"),
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
            # Half the window, floored at 5 — and never more than the window
            # itself. The floor alone gave the old 3-day window a minimum of 5
            # observations it could not hold, so Z_3 was NaN on every row while
            # still being counted as one of three windows: breadth topped out
            # at 66.7% and the "HIGH" confidence tier was unreachable.
            min_periods = min(lb, max(lb // 2, 5))
            # Zero-centred: the gap's zero is fair value, and the window says
            # only how LARGE today's gap is against its own typical size there
            # (analytics/robust_z.py explains why median-centring broke this).
            z_scores, lower_bounds, upper_bounds = rolling_robust_z(
                r, window=lb, min_periods=min_periods, alpha=0.05, center="zero"
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
            # `Price` is the same level under the name the grader, the analog
            # matcher and the pages key off — published here, by the engine,
            # rather than written onto its output by the orchestrator.
            "Actual": self.price,
            "Price": self.price,
            "FairValue": self.predictions,
            "Residual": self.residuals,
            "ModelSpread": self.model_spread,
            "GapScale": self.gap_scale,
            # ── published Mūla diagnostics (column key stays "FVO") ────────
            "FVO": _col("fvo"),
            "PctMispricing": _col("pct_mispricing"),
            "FairValueLo": _col("ci_lo"),
            "FairValueHi": _col("ci_hi"),
            "XSConsistency": _col("xs_consistency"),
            "GapHalfLife": _col("gap_halflife"),
            "GapPhi": _col("gap_phi"),
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
        # Breadth is a share of the windows that HAVE a reading on that row.
        # The 1-year window needs ~6 months of published gap before it reads
        # at all; counting it as a silent "not cheap" in the meantime would cap
        # every early breadth reading at two-thirds.
        num_lb = np.maximum(finite_stack.sum(axis=0), 1).astype(float) if len(finite_stack) else np.ones(n)

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
        # pool (analytics/analogs.py), the error-correction regressions, the
        # adaptive tier estimation on the Mūla page, and the regime
        # distribution. A conviction computed on 4% of the panel must
        # not become a historical precedent that today gets matched against, so
        # holidays stay Valid=False AND unpublished, exactly as before. Today's
        # forming row is published, flagged `Provisional`, and still excluded
        # from all of the above — the error-correction read included: it is
        # READ from the state at the last Valid row, never absorbed.
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
            if len(forming) and isinstance(_ix, pd.DatetimeIndex) and len(_ix) == len(forming):
                from data.calendars import session_today
                forming[-1] = (pd.Timestamp(_ix[-1]).normalize()
                               >= session_today(getattr(self, "_exchange", "")))
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

        # Wire names kept ("Oversold"/"Overbought" — every consumer reads
        # them): the share of the 1m / 3m / 1y windows that read the gap as
        # unusually CHEAP / RICH against that window's own history.
        self.ts_data["OversoldBreadth"] = np.where(
            publish, (oversold + extreme_os) / num_lb * 100, np.nan)
        self.ts_data["OverboughtBreadth"] = np.where(
            publish, (overbought + extreme_ob) / num_lb * 100, np.nan)
        self.ts_data["BuySignalBreadth"] = buy_count
        self.ts_data["SellSignalBreadth"] = sell_count
        self.ts_data["AvgZ"] = np.where(publish, avg_z, np.nan)
        # Window agreement: +100 when every window reads the gap rich, an
        # extra half again for each window reading it extreme. Positive =
        # rich, the oscillator's own orientation.
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
                   "ModelSpread", "GapScale", "FVO", "PctMispricing"):
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
        self.panel_absent = []
        if _pr is not None and len(_pr) == len(valid_row) and len(valid_row):
            try:
                _miss = open_adm[-1] & ~pr[-1]
                self.panel_absent = [n for n, m in zip(self.feature_names, _miss) if m]
            except Exception:                       # never break a run to report
                self.panel_absent = []

    def _compute_ddm_conviction(self) -> None:
        """Drift-diffusion filter of window agreement, and the state word.

        ``ConvictionRaw`` (how many windows read the gap cheap or rich) is
        noisy session to session; the filter integrates it into a persistent
        reading with an uncertainty band. It is evidence about the claim — how
        consistently the gap has looked stretched against its own history — and
        it is deliberately NOT a direction of its own any more.

        It used to be one. Its adaptive tiers drove the engine's BUY / SELL /
        STRENGTH and the ``Regime`` labels (STRONGLY OVERSOLD … OVERBOUGHT),
        while the hero took Mūla's direction from the oscillator — so one
        engine published two directions, and on the 3/5/10-day windows they
        regularly disagreed (Gold, 2026-10-16: "BUY · OVERSOLD" beside an
        oscillator of +0.01σ). ``Regime`` is now the valuation state of the
        oscillator itself (``core.semantics.valuation_state``), the same word
        every page shows.

        The DDM constants are the global ``DDM_*`` (config): a filter of a
        bounded agreement score, not an instrument property.
        """
        raw = self.ts_data["ConvictionRaw"].values
        filtered, _gains, variances = drift_diffusion_filter(
            raw, leak_rate=DDM_LEAK_RATE, drift_scale=DDM_DRIFT_SCALE, long_run_var=DDM_LONG_RUN_VAR
        )
        ddm_std = np.sqrt(np.maximum(variances, 0))
        self.ts_data["ConvictionScore"] = filtered
        self.ts_data["ConvictionBounded"] = _apply_conviction_bounds(filtered)
        self.ts_data["ConvictionUpper"] = _apply_conviction_bounds(filtered + 1.96 * ddm_std)
        self.ts_data["ConvictionLower"] = _apply_conviction_bounds(filtered - 1.96 * ddm_std)
        self.ts_data["Regime"] = [valuation_state(v) for v in self.ts_data["FVO"].to_numpy(dtype=float)]

    def _compute_error_correction(self, series: pd.DataFrame) -> None:
        """Run the price-side ECM read (``ecm.MulaLayer``) over the published gap.

        The layer absorbs ONLY trusted (``Valid``) sessions. A withheld holiday
        has a carried price (Δp = 0) and a gap fitted on a tenth of the panel;
        regressing one on the other would teach the ECM that price ignores the
        gap exactly on the days the gap is least real. A skipped session simply
        does not happen from the regression's point of view: the next absorbed
        session's Δp and lagged regressors are taken against the previous
        absorbed one. Today's ``Provisional`` row is published but not learned
        from: its read comes from the layer's state as of the last Valid row
        (``MulaLayer.read``), with its own gap used only for the projection —
        so the live κ̂ is estimated on the population the graded history is.

        Publishes MulaKappa / MulaRevProb / MulaHalfLife / MulaDriftPct /
        MulaSdPct / WValuation / WMomentum / WFull, and from them:

          MRProb      := MulaRevProb — P(κ̂ > 0), the evidence that PRICE
                         responds to the gap.
          Confidence  := sqrt(MRProb × XSConsistency) — price-side reversion
                         evidence AND cross-sectional agreement on the sign.

        Failure posture: if the layer raises, its columns (and MRProb /
        Confidence) publish as NaN and the valuation itself is untouched —
        the read may fail, the valuation must not.
        """
        n = len(self.ts_data)
        nan = np.full(n, np.nan)
        cols = {c: nan.copy() for c in ECM_COLUMNS}
        try:
            y = np.asarray(self.y, dtype=np.float64)
            z = pd.to_numeric(self.ts_data["Residual"], errors="coerce").to_numpy(dtype=np.float64)
            phi = pd.to_numeric(series["gap_phi"], errors="coerce").to_numpy(dtype=np.float64)
            absorb = self.ts_data["Valid"].to_numpy(dtype=bool)
            provisional = self.ts_data["Provisional"].to_numpy(dtype=bool)
            layer = MulaLayer(horizon=self.horizon)
            last = prev = None            # newest / second-newest absorbed rows
            last_dp = np.nan
            for t in range(n):
                finite = np.isfinite(y[t]) and np.isfinite(z[t])
                if provisional[t] and finite and last is not None:
                    dz_lag = (z[last] - z[prev]) if prev is not None else np.nan
                    layer.read(z[last], dz_lag, last_dp, z[t], phi[t])
                    continue
                if not (absorb[t] and finite):
                    layer.skip()
                    continue
                if last is None:
                    layer.skip()
                else:
                    dp = y[t] - y[last]
                    dz_lag = (z[last] - z[prev]) if prev is not None else np.nan
                    layer.step(z[last], dz_lag, last_dp, dp, z[t], phi[t])
                    last_dp = dp
                prev, last = last, t
            frame = layer.frame(pd.RangeIndex(n))
            cols = {c: frame[c].to_numpy(dtype=np.float64) for c in ECM_COLUMNS}
        except Exception:                                    # pragma: no cover
            log.exception("Mūla error-correction read failed; publishing the valuation without it")

        for c, v in cols.items():
            self.ts_data[c] = v
        rev = cols["MulaRevProb"]
        xs = pd.to_numeric(self.ts_data["XSConsistency"], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore"):
            conf = np.sqrt(np.clip(rev, 0.0, None) * np.clip(xs, 0.0, None))
        publish = np.isfinite(pd.to_numeric(self.ts_data["FVO"], errors="coerce").to_numpy(dtype=float))
        self.ts_data["MRProb"] = np.where(publish, rev, np.nan)
        self.ts_data["Confidence"] = np.where(publish & np.isfinite(rev) & np.isfinite(xs), conf, np.nan)

        r = self._last_published_row()
        if r is None:
            self.mula_summary = {}
            return
        row = self.ts_data.iloc[r]

        def _g(col: str) -> float:
            v = float(row.get(col, np.nan))
            return v if np.isfinite(v) else float("nan")

        self.mula_summary = {
            "kappa": _g("MulaKappa"),
            "half_life": _g("MulaHalfLife"),
            "rev_prob": _g("MulaRevProb"),
            "expected_drift_pct": _g("MulaDriftPct"),
            "drift_sd_pct": _g("MulaSdPct"),
            "w_valuation": _g("WValuation"),
            "w_momentum": _g("WMomentum"),
            "w_full": _g("WFull"),
        }

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

    def _compute_ou_diagnostics(self) -> None:
        """OU fit of the published gap: full-history half-life, ADF / KPSS.

        These describe the GAP's persistence — how long a mispricing of this
        size typically survives — not which side closes it (that is the ECM's
        question; see ``_compute_error_correction``).

        The headline half-life and the decay projection use the FULL-history
        fit. They used to use the last 60-session rolling estimate while the
        Mūla page called the number "a full-history fit": Gold read 4.3d on
        the window against 8.2d on its history, S&P 500 2.0d against 6.2d. A
        60-observation AR(1) is too noisy to headline. The rolling estimates
        are kept for one purpose — ``theta_stable``, whether the reversion
        speed itself has been steady lately.
        """
        r = self.residuals
        oos_r = r[self.min_train_size:]
        oos_r = oos_r[np.isfinite(oos_r)]

        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            try:
                with warnings.catch_warnings():
                    # statsmodels 0.15 announces a return-type change; the
                    # p-value is element [1] under both.
                    warnings.simplefilter("ignore", category=FutureWarning)
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
            "half_life": np.log(2) / max(theta, 1e-4),
            "half_life_recent": np.log(2) / max(dynamic_theta, 1e-4),
            "stationary_std": sigma / np.sqrt(2 * max(theta, 1e-4)),
            "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue,
            "vol_multiplier": vol_multiplier, "theta_std": theta_std,
        }

        current_r = float(oos_r[-1]) if len(oos_r) else 0.0
        proj_days = np.arange(1, OU_PROJECTION_DAYS + 1)
        self.ou_projection = mu + (current_r - mu) * np.exp(-theta * proj_days)

        # The band brackets the decay at the speeds the rolling fits have
        # actually wandered across (±1 sd of the 60-session theta).
        if theta_std > 0:
            self.ou_projection_upper = mu + (current_r - mu) * np.exp(-max(theta - theta_std, 1e-4) * proj_days)
            self.ou_projection_lower = mu + (current_r - mu) * np.exp(-(theta + theta_std) * proj_days)
        else:
            self.ou_projection_upper = self.ou_projection.copy()
            self.ou_projection_lower = self.ou_projection.copy()


#: The engine's former name (codename FVO, "fair value oscillator"). Kept so
#: ``isinstance`` checks and imports written against it keep working; it is
#: the same class, not a reduced core — the error-correction read is part of
#: every fit.
FairValueEngine = MulaEngine
