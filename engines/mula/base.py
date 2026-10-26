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
* **Its persistence** — ``GapPhi`` / ``GapHalfLife``, the online AR(1) of the
  gap. They say how long a gap survives however it closes; they do not say
  which side closes it.
* **The error-correction read** (``ecm.py``) — whether PRICE responds to the
  gap (κ̂, its evidence ``MRProb``, the reversion sizer's reading) and the
  h-session move the gap implies.

``PUBLISHED_COLUMNS`` is the whole published frame (``ts_data``), and each of
its columns has a named reader outside the engine (tests/test_consumers.py).
The window stack, its drift-diffusion filter, the leave-one-block-out
cross-section score, the HMM regime label and the OU projection that used to
ride along were near-copies of the claim, write-only, or read by nothing that
moves the call, and went.

The public surface is ``fit`` / ``get_current_signal`` / ``get_model_stats``
/ ``ts_data`` / ``latest_feature_impacts`` / ``get_feature_impact_history``.

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
from typing import Callable

import numpy as np
import pandas as pd

from .ecm import COLUMNS as ECM_COLUMNS
from .ecm import MulaLayer
from .valuation import (BURN_IN, MIN_PRINTS, PANEL_MIN_COVERAGE, PANEL_MIN_PRINTS,
                        PUBLISH_WARMUP_PRINTS, PUBLISH_WARMUP_WINDOW,
                        SESSION_SPARSE_FRACTION, VALUATION_DELTAS,
                        MarketValuationEngine)

log = logging.getLogger(__name__)

#: The published frame, ``ts_data``, in order: the valuation and its band, the
#: claim's oscillator and the gap it is made of, the gap's persistence, the
#: error-correction read, the cross-section it was valued on, and the
#: publication flags.
PUBLISHED_COLUMNS: tuple[str, ...] = (
    "Price", "FairValue", "FairValueLo", "FairValueHi", "FVO", "PctMispricing",
    "GapScale", "ModelSpread", "GapPhi", "Residual", "GapHalfLife",
    *ECM_COLUMNS,
    "KFactors", "NAvailable", "NAdmitted", "NScheduled",
    "Valid", "Provisional", "WithheldReason",
)

#: The valuation family: the columns that are a claim about the cross-section
#: and so follow the publication rule (``Price`` is an observed print and does
#: not).
_VALUATION_FAMILY: tuple[str, ...] = (
    "FairValue", "FairValueLo", "FairValueHi", "Residual", "ModelSpread", "GapScale",
    "FVO", "PctMispricing")


class MulaEngine:
    """Recursive fair-value inference and the readings built on its gap.

    Pipeline:
        1. Dynamic cointegrating regression of log price on the integrated
           latent factors AND the integrated asset-class block aggregates of
           the macro cross-section, each view a discount-factor mixture, the
           two views averaged by their own out-of-sample predictive evidence
        2. The publication rule: the warm-up and the panel-completeness gate
           (what is published, what is trusted, and why a session is not)
        3. The error-correction read: does price respond to the gap?
        4. Fit quality and the blocks the fitted level is made of
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.model_stats: dict = {}
        self.latest_feature_impacts: dict = {}
        self.feature_impact_history: pd.DataFrame = pd.DataFrame()
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
        #: Instruments whose exchange was open on the newest row and that did
        #: not print — the panel-composition channel that moves history. The
        #: System page names them beside the newest row's coverage.
        self.panel_absent: list[str] = []
        #: Raw Mūla result bundle: the per-session series and the blocks the
        #: fitted level is made of (``block_attribution``).
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
        #: First index carrying a valuation (the end of the burn-in), for the
        #: run console.
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
    ) -> "MulaEngine":
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
            (burn-in, print floor, discount grid) and its ``forecast_horizon``,
            the one h, are read from it directly.

        exchange: the target's exchange (``data.calendars.resolve_exchange``);
            its local date decides whether the newest row is a session still
            forming (``data.calendars.session_today``).

        epoch / fingerprint: the analysis epoch and the prepared history's
            fingerprint (``data.prep``), stored on the engine as provenance.
        """
        self.burn_in = int(config.fvo_burn_in)
        self.min_prints = int(config.fvo_min_prints)
        self.valuation_deltas = tuple(config.fvo_valuation_deltas)
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
        # `y` is the modelled quantity: log price, which the error-correction
        # read differences.
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
        # exchanges were OPEN actually print?" — see _apply_publication_rule.
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
        # the level the cross-section implies (rich), negative = cheap — the
        # oscillator's own orientation (Aarambh had to negate its forecast).
        self.residuals = gap
        # Fair value as a PRICE level; the uncertainty about that level, in
        # log units (ModelSpread — the tab renders it ×1e4 as basis points);
        # and the scale the gap is measured in (GapScale, the FVO denominator).
        self.predictions = series["fair_value"].to_numpy(dtype=np.float64)
        self.model_spread = series["fv_sd"].to_numpy(dtype=np.float64)
        self.gap_scale = series["pred_sd"].to_numpy(dtype=np.float64)
        # First row carrying a valuation — everything before it is burn-in.
        finite = np.isfinite(gap)
        self.min_train_size = int(np.argmax(finite)) if finite.any() else self.n_samples

        self._publish_frame(series)
        self._apply_publication_rule()
        self._compute_error_correction(series)
        self.ts_data = self.ts_data[list(PUBLISHED_COLUMNS)]
        self._compute_model_stats()
        self._compute_block_impacts()

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

        It is the session's identity and the claim's reading, nothing more:
        every other value of d* is read off the published frame by date
        (``fvo_ts.loc[d*]``). Every key has a reader: the Mūla page's claim
        sentence and the app's as-of status and session notices. The state
        word is ``core.semantics.valuation_state(fvo)``, derived where it is
        shown.
        """
        r = self._last_published_row()
        if r is None:
            return {
                "fvo": float("nan"), "pct_mispricing": float("nan"), "n_available": 0,
                "asof_row_withheld": True, "withheld_reason": "no valuation published",
                "published_row": None, "asof": None, "provisional": False,
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

        withheld = r != len(self.ts_data) - 1
        return {
            # ── the claim ────────────────────────────────────────────────
            "fvo": _f("FVO"),
            "pct_mispricing": _f("PctMispricing"),
            # ── how much of the cross-section has printed on it ─────────
            "n_available": int(np.nan_to_num(_f("NAvailable"))),
            # ── which session this is ───────────────────────────────────
            "asof_row_withheld": bool(withheld),
            "withheld_reason": str(last.get("WithheldReason", "")) if withheld else "",
            "published_row": r,
            "asof": self.asof,
            "provisional": bool(row.get("Provisional", False)),
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_feature_impact_history(self) -> pd.DataFrame:
        """Each block's share of the fitted level at every attribution session."""
        return self.feature_impact_history

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

    def _publish_frame(self, series: pd.DataFrame) -> None:
        """The valuation pass's readings, on the engine's row positions.

        ``Price`` is the observed level the grader, the gap split and the
        pages key off — published here, by the engine, rather than written
        onto its output by the orchestrator. ``Residual`` is the log gap
        ``p − fv`` the oscillator divides by ``GapScale``.
        """
        n = self.n_samples

        def _col(name: str) -> np.ndarray:
            return series[name].to_numpy()

        self.ts_data = pd.DataFrame({
            "Price": self.price,
            "FairValue": self.predictions,
            "FairValueLo": _col("ci_lo"),
            "FairValueHi": _col("ci_hi"),
            "FVO": _col("fvo"),
            "PctMispricing": _col("pct_mispricing"),
            "GapScale": self.gap_scale,
            "ModelSpread": self.model_spread,
            "GapPhi": _col("gap_phi"),
            "Residual": self.residuals,
            "GapHalfLife": _col("gap_halflife"),
            "KFactors": _col("k_factors"),
            "NAvailable": _col("n_available"),
            "NAdmitted": _col("n_admitted"),
        }, index=pd.RangeIndex(n))

    def _apply_publication_rule(self) -> None:
        """Which sessions are published, which are trusted, and why one is not.

        A session is VALUED once the gap has printed on PUBLISH_WARMUP_PRINTS of
        the PUBLISH_WARMUP_WINDOW sessions before it (the warm-up after the
        burn-in, valuation.py), and published if it also passes the panel-
        completeness gate below. Writes ``NScheduled``, ``Valid``,
        ``Provisional`` and ``WithheldReason``, masks the valuation family to
        the published rows, and records ``panel_absent``.
        """
        gap = self.residuals
        n = len(gap)
        finite = np.isfinite(gap)
        # Prints of the gap in the WARMUP_WINDOW sessions strictly before each
        # row: the count a session's own value is never part of.
        prior = np.concatenate([[0], np.cumsum(finite)])
        lag = np.arange(n) - PUBLISH_WARMUP_WINDOW
        before = prior[np.arange(n)] - np.where(lag >= 0, prior[np.maximum(lag, 0)], 0)
        valid_row = finite & (lag >= 0) & (before >= PUBLISH_WARMUP_PRINTS)

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
        # `Valid` is not a display flag. It gates the error-correction
        # regressions, every grade and every piece of evidence (they are all
        # computed on Valid rows). A valuation computed on 4% of the panel
        # must not become history the call is graded on, so holidays stay
        # Valid=False AND unpublished. Today's forming row is published,
        # flagged `Provisional`, and still excluded from all of the above —
        # the error-correction read included: it is READ from the state at the
        # last Valid row, never absorbed.
        publish = valid_row
        # Pre-gate validity: a row can be withheld because the warm-up has not
        # run (no published gap history yet) OR because the panel was too thin
        # (the gate below). Those are different answers to "why is there a gap
        # here", so keep them separable before the two get ANDed together.
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
            # scheduled of 31 admitted) and are withheld by the warm-up
            # instead. A row is only a data problem if the GATE is what
            # stopped it.
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
        # trade either, so `Price` is a carried-forward quote and `FairValue` is
        # a regression against a tenth of the panel. Publishing that as a point
        # on the chart asserts a valuation that was never made.
        #
        # `Price` is deliberately NOT masked — it is an observed price, not a
        # claim of ours, and the Data tab should still show what the vendor
        # returned. Today's forming row is in `publish`, so the live reading is
        # unaffected.
        for _c in _VALUATION_FAMILY:
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

        Publishes the layer's columns (``ecm.COLUMNS``): MulaKappa, MRProb —
        P(κ̂ > 0), the evidence that PRICE responds to the gap and the
        reversion sizer's reading — MulaHalfLife, MulaDriftPct and MulaSdPct.
        Each is NaN off the absorbed and provisional rows, which are published
        rows by construction.

        Failure posture: if the layer raises, its columns publish as NaN and
        the valuation itself is untouched — the read may fail, the valuation
        must not.
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

    def _compute_model_stats(self) -> None:
        """How much of log price the fitted level tracks, and what it is fitted on.

        ``r2_oos`` is the share of log-price variance the one-step-ahead fitted
        level explains. A level regression on integrated regressors reads high
        by construction — both sides trend — so it is a check that the
        cross-section tracks the target at all, never a skill number: the
        claim's only skill number is its grade on price (``edges['claim']``).
        (An 'R² vs trailing mean' used to be shown here as the discriminating
        number. A fair value of yesterday's close beat it on every instrument,
        so it measured level tracking, not skill, and went.)
        """
        gap = self.residuals
        p = self.y
        valid = np.isfinite(gap) & np.isfinite(p)
        if int(valid.sum()) > 2:
            ss_res = float(np.sum(gap[valid] ** 2))
            ss_tot = float(np.sum((p[valid] - np.mean(p[valid])) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        else:
            r2 = float("nan")
        self.model_stats = {
            "r2_oos": r2,
            "n_features": len(self.feature_names),
            "n_blocks": len(self.block_names),
        }

    def _compute_block_impacts(self) -> None:
        """What the fitted level is made of, by asset-class block.

        Factor and block levels are linear in the standardised returns, so the
        fitted level decomposes exactly into per-instrument contributions
        (valuation.py); summed to their blocks they say which parts of the
        world the published fair value leans on. Each attribution session's
        blocks are shown as their share (%) of the absolute contributions, so
        a block's weight reads on one scale whatever the level's sign. The
        engine's own decomposition, not a refit — the leave-one-block-out
        ablation that used to produce this was a separate model and went.
        """
        att = self.mve.get("block_attribution")
        if att is None or att.empty:
            self.latest_feature_impacts = {}
            self.feature_impact_history = pd.DataFrame()
            return
        mag = att.astype(float).abs()
        tot = mag.sum(axis=1)
        share = mag.div(tot.where(tot > 0), axis=0) * 100.0
        rows = share.dropna(how="all")
        if rows.empty:
            self.latest_feature_impacts = {}
            self.feature_impact_history = pd.DataFrame()
            return
        latest = rows.iloc[-1].dropna()
        self.latest_feature_impacts = dict(
            sorted({str(k): float(v) for k, v in latest.items()}.items(),
                   key=lambda kv: kv[1], reverse=True))
        hist = rows.copy()
        hist.index.name = "Date"
        self.feature_impact_history = hist


#: The engine's former name (codename FVO, "fair value oscillator"). Kept so
#: ``isinstance`` checks and imports written against it keep working; it is
#: the same class, not a reduced core — the error-correction read is part of
#: every fit.
FairValueEngine = MulaEngine
