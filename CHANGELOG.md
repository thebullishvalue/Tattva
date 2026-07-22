# Changelog

All notable changes to **Tattva** (formerly **Nishkarsh**) are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Sections used: **Added · Changed · Deprecated · Removed · Fixed · Security · Performance · Docs**.

---

## [Unreleased]

## [2.7.0] — 2026-07-20 — *Nirnay-Swayam self mode · individual stocks · full per-instrument configuration · research-suite overhaul*

### Added
- **Interpretation layer is now per-instrument too** — markers, UI display tiers,
  classification thresholds, and analog blend weights, matching the engines.
  25 data-anchored interpretation constants (Unified-Signal markers, agreement /
  breadth / model-spread tiers, conviction + convergence display tiers,
  consensus/composite thresholds, `analog_w_*`) became `InstrumentConfig` fields
  (defaults mirror the module globals — pinned by a new sync check in
  `test_instrument_configs`). Read sites now resolve the active instrument's
  values: the convergence-classification threshold SEED is `_icfg.composite_thresholds()`
  (weights seed / nirnay oversold-overbought were already per-instrument); the
  Convergence / Aarambh / Nirnay tabs shadow their marker & tier globals with the
  active instrument's config; the analog matcher's maha/trajectory weights are
  parameters fed from `_icfg`. Structural constants (R²/ADF/KPSS/HMM cut-points,
  chart dims) stay global — statistical definitions, not distribution anchors.
  Behaviour-preserving until an override is wired via `_PER_INSTRUMENT_OVERRIDES`.
  The interpretation STUDIES (`markers`, `hero_thresholds`, `ui_anchors`,
  `conv_weights`, `analog`) now emit gated per-instrument recommendations too —
  percentile-anchor studies adopt a target-specific anchor only when its own
  distribution diverges from the pooled default by ≥25% with ≥250 obs; the IC
  studies use the engine gate — so every study is at full per-instrument parity.
- **Aarambh forecast engine is now tunable PER INSTRUMENT / asset class**, like
  Nirnay (`nirnay_*`) and Swayam (`swayam_*`). Seven training knobs moved onto
  `InstrumentConfig` — `aarambh_refit_interval`, `aarambh_min_train_size`,
  `aarambh_max_train_size`, `aarambh_ensemble_models`, `aarambh_ridge_alphas`,
  `aarambh_huber_epsilon`, `aarambh_lookback_windows` (defaulting to the global
  constants, so behaviour-preserving). `FairValueEngine.fit(..., config=<cfg>)`
  reads them off the instrument's config into per-run instance attributes
  (replacing the former module-global references throughout the walk-forward; the
  `@staticmethod` ensemble path takes them as parameters). `app.py` threads
  `get_instrument_config(target)` into the fit call, so any instrument or asset
  class can retune the forecast in isolation via `_PER_INSTRUMENT_OVERRIDES` /
  `STOCK_CONFIGS`. `aarambh_tuning_study` now emits a gated per-target
  `_PER_INSTRUMENT_OVERRIDES` snippet (via `research/_per_instrument.py`) alongside
  its class-level block, and a new `test_aarambh_config` (suite key `t_aarambh`)
  pins the threading contract.
- **Orchestrator "from scratch" run option** (`research/run_tuning.py`). A new
  interactive action `f) Run EVERYTHING end-to-end FROM SCRATCH` and a `--fresh`
  CLI flag wipe the persistent study-result cache (`research/.tune_cache/` — the
  aarambh resume CSV) and pass `--fresh` through, so a full re-run carries nothing
  from a previous report and recomputes every result. Segment / specific-study
  selections that include a cache-bearing study prompt for it too. Deliberately
  does NOT clear the raw market-data cache (`~/.cache/tattva`) — that is shared and
  expensive to refetch, and the preflight re-warms it. `--fresh` no-ops (with a
  note) when the selection has no persistent cache, and for the tests-only run.
- **Per-instrument configuration for the five catalogue classes.** Commodities,
  the currency, every India index, every US index and the ETF target now each
  carry their OWN tuned InstrumentConfig knobs, layered on their class default via
  a new explicit `PER_INSTRUMENT_TUNING` map in `core/config.py` (auto-seeded with
  a slot for every per-instrument-class target, so it can't drift from the
  catalogue). Hand-wired values live in `_PER_INSTRUMENT_OVERRIDES` and are
  validated at import (tunable fields only, real targets only). The India/US STOCK
  classes deliberately stay ASSET-LEVEL (one config per market via `STOCK_CONFIGS`),
  since free-form symbols can't be pre-tuned. `PER_INSTRUMENT_CLASSES` /
  `ASSET_LEVEL_CLASSES` name the split, and the derived-view invariant (each
  config == class default ⊕ per-instrument override) is pinned by
  `test_instrument_configs`. Behaviour-preserving until values are wired.
- **Studies now emit PER-INSTRUMENT recommendations.** `swayam` (per-commodity
  Swayam grid), `nirnay_index` (per-index MSF, expanded to all 24 India indices),
  `nirnay` (per-target basket knobs for USD/INR + Jeera) and `per_asset` (per-index
  MSF for S&P 500 / Nasdaq 100 / Dow Jones + the ETF target) each print a gated,
  copy-paste-ready `_PER_INSTRUMENT_OVERRIDES` snippet via a shared helper
  (`research/_per_instrument.py`). The gate adopts a per-instrument value ONLY when
  that target's own best |IC| clears an absolute floor AND beats its class-default
  |IC| by a margin — otherwise the instrument keeps its class default (no
  overfitting per-target grid noise). `per_asset`'s stock classes stay pooled
  (asset-level), matching the config split.

### Changed
- **Per-instrument values wired from the 2026-07-20 from-scratch suite run**
  (`reports/tuning_20260720_025604.txt`; full 19-study run, `--fresh`, on the
  fixed aarambh baseline). The per-instrument layer fired as designed — the
  studies gate-passed a systematic set of overrides now in
  `core.config._PER_INSTRUMENT_OVERRIDES` (24 instruments):
  - **19 India indices → long MSF** (`nirnay_index`). India equity indices
    systematically prefer MSF 25–60 over the commodity-tuned class default of 5,
    with |IC| up 3–7× (Nifty Energy 0.164 vs 0.061; Nifty Services 0.148 vs
    0.022; Nifty Metal 0.145 vs 0.045). Nifty PSU Bank is the lone short-window
    winner (MSF 3). FMCG/IT/MNC/Media wanted short windows but within noise →
    kept at default; Smallcap 100 had no basket.
  - **Copper → swayam_lengths (8,14,22,34,52)** (`swayam`; |IC| 0.075 vs 0.044).
    The other commodities' grid stayed at the class default (within noise).
  - **US indices + ETF** (`per_asset`): S&P 500 MSF 40, Nasdaq 100 MSF 18, Dow
    Jones MSF 8, India Sector ETFs MSF 12 (each |IC| 0.11–0.16 on its basket).
- **Class-level / global constants applied LITERALLY from the same 2026-07-20
  report, per explicit user directive** (track the report's recommendation outputs
  rather than the "adopt only beyond noise" default). Applied:
  - **Aarambh** (`aarambh_full` RECOMMENDED block): `REFIT_INTERVAL` 63→**40**,
    `ENSEMBLE_MODELS` →**("ols",)**, `MAX_TRAIN_SIZE` 350→**150**, `MIN_TRAIN_SIZE`
    150→**1000**, `RIDGE_ALPHAS` →**(0.1,1,10)**, `HUBER_EPSILON` 1.1→**4.0**,
    `InstrumentConfig.pca_components` 2→**20**. ⚠ MIN>MAX is an OFAT interaction —
    the engine uses `max(MAX,MIN)` so the window is 1000 and MAX=150 is INERT;
    flagged at the constants and worth a joint MIN×MAX grid. A single-member "ols"
    ensemble also zeroes the Model Spread indicator. (These ICs are small and some
    non-monotonic — applied as directed, not as an endorsement of separation.)
  - **Nirnay class-level** (`nirnay` best-per-knob): `NIRNAY_MSF_LENGTH` 5→**20**,
    `NIRNAY_ROC_LEN` 2→**60**, `NIRNAY_REGIME_SENSITIVITY` 6.0→**8.0**,
    `NIRNAY_MMR_NUM_VARS` 4→**15** (base_weight 0.0 unchanged). The per-instrument
    India/US-index overrides above still supersede the MSF default for those targets.
  - **DDM** (`ddm` best-mean-IC leak, gain held): consensus `CONV_DDM_LEAK_RATE`
    0.15→**0.01** (drift 0.012); engine `DDM_LEAK_RATE` 0.65→**0.03** (drift 0.056).
    Much lighter smoothing of the hero trend / conviction display (a product-character
    change; lag stays within the 10d horizon).
  - **Interp markers/thresholds** (`markers` / `hero_thresholds` anchors):
    `UI_CONSENSUS_STRONG/MODERATE` →**0.41/0.27**, `UI_CONVRAW_STRONG/MODERATE`
    →**66.67/33.33**, consensus `DEFAULT_THRESHOLDS` →**±0.284/±0.428** (p75/p90).
    COMPOSITE_THRESHOLDS already on-anchor (±0.19/±0.33 ≈ p75/p90) → unchanged.
  - **stock_us** (asset-level, `per_asset`): `swayam_roc_frac` 0.7→**0.55**
    (stock_india unchanged; both stay pooled at market level).
  - **Not changed — the report recommended KEEPING current:** `conv_weights`
    (factory vector stands; best alt scored negative), `calibration_lift`
    (consensus-primary stands), `analog`/`analog_confirm` (current 1/0/0 wins),
    `LOOKBACK_WINDOWS` (already at the recommended ultra-short(3-10)), the analog
    horizon/precedent structure (validated).
- **Re-tune from the 2026-07-18 full suite run** (post per-instrument / self-mode
  refactor; `research/reports/tuning_20260718_173837.txt`). Each result was read
  against its study's own "adopt only beyond noise" rule — most knobs stood; two
  changed:
  - `NIRNAY_MSF_LENGTH` **18 → 5**. Short MSF windows won CROSS-UNIVERSE at the
    corrected baseline — `nirnay` (commodity/FX baskets: |IC| 0.083 @5 vs 0.039
    @18) and `nirnay_index` (India indices: 0.103 @5 vs 0.050 @18) both peak at
    the short end, with 18 in the worst part of each curve. Basket-mode only
    (self-mode Swayam members carry their own `swayam_lengths`).
  - `COMPOSITE_THRESHOLDS` **±0.11/±0.16 → ±0.19/±0.33** (p75/p90). The composite
    distribution shifted once commodities moved to Swayam self mode, and the old
    pair had drifted to ~p58/p69 (moderate firing 42% of days); re-anchored to the
    house occupancy convention. Cascades to the intelligence seed thresholds and
    the conviction-model display tiers (both derived), which auto-tracked.
  - **Stood, per each study's rule:** Nirnay ROC/sensitivity (inert), base_weight /
    MMR_num_vars (gains concentrated in self-mode or single targets), the Swayam
    grid (within noise), both DDM filters (the lower-leak IC "gain" is just less
    smoothing + more lag — a product choice), convergence dim-weights (inert), and
    the UI/marker anchors (within convention, or an unexplained ConvictionBounded
    shift not worth chasing on one run).
- **Research: fixed the stale `aarambh_tuning_study` baseline.** Its `BASE` held
  `refit=5/maxt=750/mint=500/pca=20` while live config is `63/350/150/2`, so the
  2026-07-18 aarambh sweeps measured interactions off-baseline (their numbers are
  not authoritative). `BASE` now pulls from live config (as `nirnay`'s already
  did), and the stale hardcoded "base: len20/…" header renders from `BASE`.
- **Research: hardened `per_asset` stock-universe fetching against rate-limiting.**
  The self stock classes fetched one ticker at a time (~200 individual yfinance
  round-trips across Nifty 100 + Nasdaq 100), which tripped the rate-limiter and
  opened the circuit breaker on the 2026-07-18 run — Nasdaq 100 collapsed to a
  ~40-name snapshot, so nothing from those two classes was adoptable. Each
  universe is now batch-fetched in ONE call up front (`_prefetch_ohlcv`); a
  prefetch below `MIN_UNIVERSE_COVERAGE` (75%) is flagged `[UNRELIABLE: thin
  universe]` in the recommendations instead of silently trusted, and
  `STOCK_UNIVERSE_CAP` is now env-overridable for a fast smoke pass. No config
  values change from this — it makes the *next* `per_asset` re-run trustworthy.

### Fixed
- **Nirnay-Swayam self-mode copy propagated to the rest of the UI.** When self
  mode was added the Nirnay tab was made mode-aware, but the Convergence tab, the
  cross-system divergence messages, and the hero fallback still assumed a
  constituent basket — so a commodity/stock target (which runs Swayam self mode)
  read "Nirnay constituents haven't turned", "Bottom-up constituent momentum",
  "The constituent basket's data ends…", and "Aarambh only (no basket
  convergence)". These now read mode-aware/neutral wording ("views" vs
  "constituents", "bottom-up breadth", "the instrument's own price data"). All
  the underlying LOGIC was already in sync — the numbers were correct (identical
  breadth schema, `expected_constituents`/coverage handled, polarity a no-op in
  self mode); only the descriptive copy lagged.

### Changed
- **Every instrument now carries its own full config (`InstrumentConfig`).** The
  system moved from global engine constants + a few sparse per-target maps to an
  explicit per-instrument config registry (`INSTRUMENT_CONFIGS` in
  `core/config.py`): each instrument's config holds BOTH its routing (Nirnay mode /
  basket / polarity / excluded predictors) AND every tunable engine knob — Nirnay
  (MSF length, ROC, regime sensitivity, base weight, MMR vars, oversold/overbought),
  the Swayam grid (lengths + ROC fraction), Aarambh forecast horizon/momentum/PCA,
  convergence DDM (leak/drift/lrv) and dimension weights, and the precedent term
  structure. `app.py` reads `get_instrument_config(active_target)` once and drives
  the whole pipeline from that instrument's fields, so any instrument can be retuned
  in isolation without touching the others. **"Defining them is a must":** every
  named catalogue target (6 commodities, USD/INR, all 24 India indices incl. Nifty 50
  & Nifty 50 - PE, 3 US indices, the ETF universe) has an EXPLICIT registry entry —
  the 22 non-Nifty-50 India indices copy the Nifty 50 baseline tuning by design but
  are each present as their own entry; `get_instrument_config` raises for an
  unregistered target rather than falling back silently. India/US **stocks** are
  configured per asset class (`STOCK_CONFIGS`), and each free-form symbol gets its own
  registry entry (cloned from its market's class config) at resolution time via
  `register_stock_target`. Per-class default configs (`CLASS_CONFIG_DEFAULTS`) let a
  whole asset class be retuned in one place. **Behaviour is byte-identical on
  introduction** — every field defaults to the exact former global constant, so the
  signal only changes once a specific instrument's (or class's) config is edited to
  diverge. The legacy per-target maps (`TARGET_ARCHETYPE`, `TARGET_POLARITY`,
  `TARGET_EXCLUDED_PREDICTORS`, `COMMODITY_BASKETS`, `NIRNAY_BASKET_ALIAS`) are
  retained as the routing source the configs are built from, so downstream consumers
  are untouched. New test suite `research/test_instrument_configs.py` (completeness,
  defaults-equal-former-globals, routing parity, the India-index copy rule, stock
  asset-class registration, and per-instrument tuning isolation).

### Removed
- **The Signal-Horizon selector (Tactical 10d / Positional 20d) is gone —
  Tattva reads one fixed horizon.** By the system's own walk-forward evidence
  (`precedent_univ` + `precedent_model`) the leakage-free directional edge lives
  at 1–10d and fades by 15–20d, so the 20d "Positional" option was a
  slower-turnover re-expression, not an independent edge; it doubled the
  Intelligence calibration surface (a second profile per target) and asked the
  user a question the evidence already answered. `SIGNAL_HORIZONS` /
  `DEFAULT_SIGNAL_HORIZON` are replaced by flat constants — `FORECAST_HORIZON`
  (10), `FORECAST_MOMENTUM` (20), `HOLD_HORIZONS` (5/10) — and the convergence
  DDM now reads the shared `CONV_DDM_*` consensus-filter tuning directly (the
  per-lens DDM override only ever differed for the removed Positional lens).
  Behaviour is byte-identical to the previous **default** (Tactical) run: same
  forecast horizon, momentum window, DDM parameters, calibration grid, and
  precedent hero read (10d). The Intelligence profile key drops its lens tag
  (now one profile per target); existing on-disk profiles keyed `"<target> ·
  Tactical (10d)"` won't match `"<target>"` and recalibrate once on next run.
  The Precedent tab is unaffected — it already shows the fixed 1/3/5/10/20/60d
  term structure independent of any lens.

### Changed
- **Liquid commodity futures now run Nirnay in Swayam self-mode.** Gold, Silver,
  Copper, Cotton, and Brent Crude are re-classified `TARGET_ARCHETYPE = "self"`,
  so Nirnay is formulated on each commodity's OWN front-month futures OHLCV (the
  self-referential timescale × information-set × mechanism ensemble,
  `engines/nirnay_self.py`) instead of a curated basket of related miners /
  agribusiness names. The futures carry real yfinance volume, so the MSF
  microstructure/flow components stay genuine. Nothing downstream changes — the
  ensemble emits the identical breadth schema the basket did, and the leakage
  guard (`swayam_macro_columns`) already drops the commodity's own macro column
  from the self-MMR driver pool. **Two commodity-group targets stay on baskets
  by data necessity, not preference:** *Jeera* (NCDEX cumin has no yfinance
  OHLCV — sheet-sourced Close only — so a Swayam ensemble can't be built; keeps
  its hybrid Indian-agribusiness basket) and *USD/INR* (FX, its own sidebar
  category and volume-less on yfinance; keeps its dollar-strength proxy basket).
  The retained `COMMODITY_BASKETS` for the now-self targets are still resolved
  by `research/nirnay_swayam_study.py` for its self-vs-basket A/B IC comparison.
  The sidebar shows a "Nirnay · Swayam self-ensemble" hint for these targets.
- **Precedent analog term structure expanded to a fixed 1/3/5/10/20/60d span,
  1d promoted to a normal horizon.** The Precedent tab previously showed the
  active lens's hold grid plus an "honorary" +1d reference tile (so Tactical
  read 1/5/10d, Positional 1/10/20d), with 1d caveated as "no edge, reference
  only." It now shows a **fixed, lens-independent** term structure —
  `PRECEDENT_HORIZONS = (1, 3, 5, 10, 20, 60)` in `core/config.py` — across the
  base-rate cards, the analog-card forward-return tiles, and the walk-forward
  **Analog Skill — Term Structure** chart. **1d is now a first-class horizon**
  (no honorary caveat); where the analog edge is genuinely weak (the 1d and
  60d ends), the per-horizon walk-forward IC + p-value on the skill chart
  disclose it honestly rather than a blanket note. The `PRECEDENT_HONORARY_HORIZON`
  constant is removed. The hero card's precedent second-opinion is unchanged —
  it still reads the base rate at the active lens's forecast horizon (10d
  Tactical / 20d Positional, both members of the new set). Because the analog
  cards now display a 60d column, `find_similar_periods`' Theiler exclusion gap
  and tail purge widen to 60d for that card set (analogs are drawn ≥60 trading
  days apart so the 60d outcome column is non-overlapping) — fewer but more
  genuinely-independent analogs; the per-horizon walk-forward chart is
  unaffected (it already uses each horizon's own gap).

### Fixed
- **Individual-stock targets failed with "data is currently unavailable".**
  `STOCK_TARGETS` tickers were registered into the sidebar (`ALL_TARGETS`)
  but never added to the macro batch `fetch_commodity_dataset` pulls, so the
  target's own price column never reached the Aarambh model matrix and the
  target-column guard in `app.py` failed clean with a misleading "pick
  another target" message. Fixed with a single-ticker injection path
  (`data.fetcher.fetch_stock_target_series`, called from
  `app._ensure_stock_target_column` before the guard) — deliberately NOT a
  macro-batch addition, since that batch's cache is keyed on `(start, end)`
  only and a per-target ticker set would break cache coherence. The guard's
  error copy now distinguishes a genuine stock-fetch failure (names the
  ticker) from a dead sheet/macro source (unchanged copy).

### Added
- **Free-form individual-stock symbol entry.** Selecting **India Stocks** /
  **US Stocks** as the sidebar Asset Class now shows a symbol text box
  instead of a fixed drop-down. India symbols probe `SYMBOL.NS` (NSE) first,
  then `SYMBOL.BO` (BSE) — an explicit suffix skips the probe; US symbols are
  used as typed (`.`→`-`). A resolved symbol registers as a first-class
  target at runtime (`core.config.register_stock_target`, idempotent —
  replayed from `st.session_state["dynamic_stock_targets"]` on every rerun)
  with the same Aarambh predictor-exclusion policy, Nirnay-Swayam self mode,
  and per-`(target, lens)` calibration as every other target. New
  `data.universe.resolve_stock_symbol` (7-day disk cache for successful
  resolutions only — a not-found symbol is session-memoized but never
  disk-cached, so a transient yfinance outage can't brand it invalid for a
  week). The `STOCK_TARGETS` static seed registry is now empty (kept for any
  future pinned defaults); the 7 hard-coded names it used to list are
  superseded by free-form entry. New test suite
  `research/test_stock_targets.py` (4 groups: fetch extraction, column
  injection/alignment, symbol resolution probe order + caching, runtime
  registration idempotency).
- **Nirnay-Swayam (स्वयम् — "self")** — a self-referential Nirnay mode for
  targets with no constituent basket (individual stocks). Instead of
  cross-sectional breadth over a basket of related instruments, breadth is
  formulated on the **target's own OHLCV**: a deterministic 15-member
  ensemble of causal views spanning three diversity axes — timescale (MSF
  length 10/14/20/28/40), information set (macro-anchored MSF+MMR vs
  pure-price-action MSF-only), and mechanism (MSF's momentum/structure/flow
  components promoted to standalone voters). Each member runs through the
  UNCHANGED per-instrument pipeline (`engines.nirnay.run_full_analysis`) and
  the UNCHANGED aggregator (`aggregate_constituent_timeseries`), so nothing
  downstream (polarity, calendar reindex, cross-validator, calibration,
  precedent, UI) needed to change — only a new instrument-selection mode.
  New module `engines/nirnay_self.py`; mode resolution via
  `TARGET_ARCHETYPE == "self"` (`data.constituents.get_nirnay_mode`) —
  individual stocks register into this archetype via free-form symbol entry
  (see "Free-form individual-stock symbol entry" above). A leakage
  guard (`swayam_macro_columns`) drops the target's own macro column from the
  MMR driver pool — without it, MMR would silently "explain" the target with
  itself and zero its own deviation oscillator. An eigenvalue-based
  effective-member-count diagnostic is surfaced in the Nirnay tab to disclose
  that self-ensemble views are correlated by construction (unlike an
  independent-name basket). Purpose-preservation invariants (documented in
  `NIRNAY_SWAYAM_PLAN.md`): no forward-return forecasting inside Nirnay (so
  Convergence stays a genuine state-read-vs-forecast agreement), strict
  causality/no-repainting, and byte-identical output for every existing
  basket-mode target (the only `engines/nirnay.py` change — an optional MSF
  component mask — reduces to the pre-existing combine formula when unset).
  Gated on a new A/B efficacy study (`research/nirnay_swayam_study.py`,
  registered in `run_tuning.py` as `nirnay_swayam`) comparing self-ensemble
  vs basket-mode breadth IC on targets where both are runnable; the
  basket-empty → Swayam fallback (`NIRNAY_SWAYAM_FALLBACK`) ships `False`
  until that study's acceptance gates pass. New integrity tests:
  `research/test_nirnay_swayam.py` (schema parity, byte-identity, causality/
  no-repainting, leakage guard, volume degeneracy, determinism).

---

## [2.6.0] — 2026-07-13 — *Signal tables · hero decision synthesis · full system re-tune · universe expansion*

Ships the Obsidian-Quant data tables and 2-decimal hover across every plot, a
hero-card decision-synthesis layer that weighs all evidence rows (not just the
raw consensus), a chart-palette single source of truth, an accessibility lift on
the muted text tier, an instrument-universe expansion (11 macro predictors + 7
Nirnay basket members), and a full from-scratch re-tune of every engine/interp
constant against widened 10→3000 grids — with config now tracking the tuning
report's recommendation outputs literally. `ENSEMBLE_MODELS` kept at
`("ols","huber")` (the report's single-member "huber" pick is declined to
preserve the Model Spread indicator).

### Added
- **Obsidian-Quant signal tables everywhere (`render_data_table`).** Ported the
  Position-Guide table design from the sibling Pragyam app into a single reusable
  component in `ui/components.py` and wired it into **every** tabular view — the
  Data-tab dataset viewer, Convergence "Recent Divergences", the Nirnay
  constituent board, and the three Diagnostics tables (feature impact, signal
  performance, saved profiles) — replacing all six `st.dataframe` calls. Rounded
  glass card, uppercase amber-ruled sticky header, zebra rows with amber hover,
  right-aligned tabular numerics, and a bold Space-Grotesk label column. It
  renders via `components.html` (an iframe that can't see `theme.css` `:root`
  vars), so the Obsidian-Quant tokens are resolved and inlined; the component is
  generic (auto numeric detection, per-column precision, NaN em-dash, optional
  sign-colouring of signed columns like the Nirnay oscillators) and scrolls
  horizontally/vertically under a fixed height, so it is safe on both the 10-row
  divergence table and the full dataset (capped to the most recent 300 rows on
  screen; full set still exportable via CSV).
- **Hero card DECISION synthesis.** The hero card's top-line label was previously
  read straight off the raw normalized-consensus value (factory ±0.3/±0.5
  thresholds); MODEL, CALIBRATED, TREND, PRECEDENT, INTERNALS and RISK were
  display-only commentary that never changed the recommendation — a CALIBRATED
  conflict could say "stand aside" in small print under a headline still shouting
  BUY. `build_hero_verdict` now folds the trust tier and every evidence row into
  an explicit action tier (`_synthesize_action`): HIGH / MODERATE / LOW
  CONVICTION · STAND ASIDE · NO ACTION, with documented ordinal weights
  (calibrated conflict −2, coherent precedent divergence −2, engines split −2,
  trend contradiction −1, recent divergences −1, confirmations +1), a hard
  no-edge gate (validated Val IC ≤ 0 → stand aside regardless of soft evidence),
  and a cap at MODERATE when the edge is unvalidated. The headline itself remains
  the normalized consensus (reconciliation invariant with the Unified Signal plot
  and the TATTVA CONVICTION card) — the new DECISION line, rendered below the
  evidence with its itemised drivers, is what the card now recommends acting on.
  Pinned by a new decision-table check group in `research/test_hero_verdict.py`.

### Removed
- **Hero MODEL-row noise copy.** Dropped the `Calibrated <timestamp>.` suffix and
  the `(Val IC is measured on the calibrated variant — see CALIBRATED row.)`
  attribution note from the MODEL evidence row, along with the now-dead
  `profile_age` plumbing (`_trust_tier` → `build_hero_verdict` → `app.py`).

### Changed
- **Tuning suite grid depth ~doubled + full deep-grid re-run (2026-07-12).**
  All 11 studies' parameter grids densified: Aarambh REFIT 7→13 values,
  ENSEMBLE 9→12 baskets, MAX/MIN_TRAIN 9→16/15 values (dense 100–2000),
  PCA 8→14 (2–60), RIDGE_ALPHAS 5→8 grids; Nirnay 32→57 OFAT values; analog
  blend 12→22 / TOP_N 8→15 / recency 10→16 + pairwise feature drops + two new
  aggregation modes (equal mean, 20% trimmed mean); precedent sweeps +7/15/30d
  horizons; markers percentiles p50–p99. The aarambh study resumes its
  (lever,value,target,horizon)-keyed CSV cache, so only new grid values cost
  compute — with the explicit warning that changing the OFAT BASE requires
  `--fresh`. Suite re-run clean (11/11 exit 0, 273 min, zero ⚠LEAK rows):
  `research/reports/tuning_20260712_010302.txt`. Verdict: every engine lever's
  variation sits inside ±0.01–0.04 IC noise — the live config is confirmed
  optimal within resolution; no engine constant moved.
- **`UI_NIRNAY_AVG_THRESHOLD` 2.5 → 2.9.** The Row-3 (Nirnay Avg_Signal) marker
  tier re-anchored to the freshly measured p75 (2.88, n=12135 pooled) per the
  block's stated p90/p75 anchoring policy; 2.5 sat at ~p65 and colored ~35% of
  days as extreme. Rows 1–2 already matched their anchors exactly (0.39/0.26,
  50/20).
- **Config rationale notes refreshed from the clean deep-grid report.** The
  "VALUES PROVISIONAL — under active re-tuning" placeholders on the Aarambh
  window/ensemble blocks, the Nirnay knobs, and the marker thresholds were
  replaced with validated 2026-07-12 figures. The `SIGNAL_HORIZONS` comment no
  longer cites pre-purge leaked analog ICs (+20d 0.162 "peak") as justification —
  post-purge the analog is ≈0 at 1–5d and negative ≥7d; the 1–10d band is carried
  by the purged model forecast (+1d IC +0.069, 31/34 targets), and Positional 20d
  is documented as a turnover lens, not an edge claim.

### Added
- **Hero classification-threshold study (`research/hero_threshold_study.py`,
  suite key `hero_thresholds`).** The hero card's BUY/SELL/STRONG cut-points had
  never had a dedicated study — the markers study anchors the *plot guides*, not
  the *action classifier*. The new study pools the exact live constructions
  (causal consensus at 10d/20d; CrossValidator composite at 5d/10d) across 8
  targets on non-overlapping windows, sweeps 30 percentile-anchored
  (moderate, strong) pairs plus the current pair on forward-return tier
  separation with occupancy floors, and applies an explicit decision rule:
  adopt a sweep winner only if its separation is believable (|t|≳2 on both
  horizons), otherwise anchor at the p75/p90 occupancy convention. Wired into
  `run_tuning.py` (T5) with both threshold dicts in the tuned-config reference.
- **`DEFAULT_THRESHOLDS` (hero consensus classifier) re-anchored ±0.3/±0.5 →
  ±0.26/±0.39.** First run (reports/tuning_20260712_141326.txt): no threshold
  pair shows believable forward-return separation (max |t|≈0.9, spreads flip
  sign across horizons) → occupancy anchoring applies. The old hand-set pair sat
  at p82/p93 and disagreed with the Row-1 plot markers (0.26/0.39 = p75/p90 of
  the same distribution); the hero classifier, plot markers, and hero-history
  bands now share one extremeness vocabulary. `COMPOSITE_THRESHOLDS`
  (±0.11/±0.18) re-validated on 8 targets (p75/p90 = 0.107/0.174, matches
  within rounding) — PROVISIONAL tag removed, values unchanged.

- **Three-arm hero-headline comparison (`calibration_lift_study.py`, suite key
  `calibration_lift`).** Extended the calibrated-vs-raw lift study with a paired
  CONSENSUS arm (the hero's live headline object), all three scored on identical
  purged test blocks with the same non-overlapping sign-flipped Spearman, one
  engine fit per target (shared ts/nd memo). First run (48 paired windows,
  8 targets, `reports/calibration_lift_20260712.txt`): pooled IC consensus
  +0.039 vs raw +0.022 vs calibrated +0.022; cal−raw lift +0.000 ± 0.004 (the
  Optuna layer adds nothing out-of-sample); consensus > calibrated in 48% of
  windows, +0.017 ± 0.031. Verdict: the consensus-primary hero headline stands
  on evidence, not just architecture; the calibrated composite remains an
  evidence row. Registered in `run_tuning.py` (T5) so every re-run re-answers
  this.

- **Full tuning-coverage audit — every constant classified, three new studies
  (`research/TUNING_COVERAGE.md`).** Every tunable constant is now study-tuned,
  data-anchored, structural, or budget — none justified by assertion. New suite
  members: `ddm` (both drift-diffusion filter sets swept on a leak grid at
  constant gain — both shipped sets sit on the IC plateau, kept), `conv_weights`
  (17 unfitted dim-weight vectors — the whole grid within ±0.01 IC of the
  factory 0.30/0.25/0.25/0.20, kept as validated-insensitive), `ui_anchors`
  (pooled live distributions for every remaining tier constant), plus
  HUBER_EPSILON and LOOKBACK_WINDOWS levers in `aarambh_full`.

### Changed
- **Six tier families re-anchored to their measured distributions
  (`ui_anchors`, 2026-07-12).** The audit found several tier vocabularies
  unreachable or meaningless in practice:
  `CONVICTION_WEAK/MODERATE/STRONG` 20/40/60 → 9/17/27 (old STRONG = p100:
  printed on 0.5% of days); `UI_AGREEMENT_MODERATE/STRONG` 0.5/0.7 → 0.82/0.91
  (old STRONG = p50: half of all days "strong agreement");
  `CONV_*_BULLISH/BEARISH` label tiers ±10/30/60 → ±11/18/27 (old STRONG
  unreachable — max observed |score| ≈ 35; new values = COMPOSITE_THRESHOLDS×100);
  `UI_NIRNAY_BULLISH/BEARISH` ±2 → ±2.9 (p75, unified with
  UI_NIRNAY_AVG_THRESHOLD); `UI_MODEL_SPREAD_HIGH` 50 → 35 bps (p90 under the
  LIVE ols+huber basket). Validated-and-kept: NIRNAY_OVERSOLD/OVERBOUGHT (±5 =
  p81), UI_BREADTH_HIGH (≈p96 alert), both DDM sets, CONV_WEIGHT_*.

### Changed
- **Config now tracks the tuning report's recommendation outputs literally
  (policy change, per user directive).** The latest from-scratch report's
  explicit RECOMMENDED/best/anchor lines were applied as-is — no sub-SE
  judgment holds: Aarambh `MIN/MAX_TRAIN` 750/1000 → **150/350**,
  `REFIT_INTERVAL` 7 → **63**, `ENSEMBLE_MODELS` → **("huber",)**,
  `RIDGE_ALPHAS` → high(1..1k), `HUBER_EPSILON` 1.35 → **1.1**,
  `LOOKBACK_WINDOWS` → **(3,5,10)** (breadth now quantized in 33% steps),
  app.py PCA literal 20 → **2**; Nirnay `ROC_LEN` 14 → **2**,
  `REGIME_SENSITIVITY` 1.5 → **6.0**, `BASE_WEIGHT` 0.6 → **0.0** (fixed half
  of the blend goes fully to MMR), `MMR_NUM_VARS` 5 → **4**, and `MSF_LENGTH`
  20 → **18** (universe-share-weighted winner across the commodity + equity
  result tables; the commodity-slice winner 10 is universe-weighted WORSE than
  20); DDM engine filter → **leak 0.65/drift 1.219**, consensus filter →
  **leak 0.15/drift 0.18** (lens dict co-updated at the 1.2× gain invariant);
  `CONV_WEIGHT_*` → **0.50/0.20/0.20/0.10**; consensus `DEFAULT_THRESHOLDS`
  strong → **±0.41**, `COMPOSITE_THRESHOLDS` strong → **±0.16** (per
  hero_thresholds' printed occupancy anchors; conviction-model tiers derive
  automatically). Markers/UI tiers/CONV label tiers unchanged (their studies
  measured current values on-anchor). Operational notes: with a single-member
  ensemble the ModelSpread tile loses its cross-member meaning if the engine
  fits only huber, and several adopted values sit on noise-level margins by
  the reports' own spreads — the next re-run may legitimately move them again.

### Docs
- **Config comment hygiene — measurement snapshots removed.** `core/config.py`
  (and the factory-threshold block in `convergence/normalization.py`) carried
  run-dated tuning snapshots (ICs, percentiles, correlation values, report
  filenames) that go stale with every re-run. All tuned/anchored constants now
  follow one convention, stated once at the top of the config: comments carry
  the constant's ROLE, its validating study key, and any hard GUARD that must
  survive a re-tune (cross-universe MSF rule, F1 threshold-distribution pairing,
  F3 constant-gain DDM sweeps, same-barrel Brent exclusions, live-basket
  ModelSpread anchoring, USD/INR rejected-alternative). Measurements, dates and
  report paths live in `research/TUNING_COVERAGE.md` and this CHANGELOG. No
  constant values changed — verified by assertion and both regression suites.

### Added
- **Instrument-universe audit → 11 macro predictors + 7 basket members
  (2026-07-13).** Audited GLOBAL_MACRO_MAP / MACRO_SYMBOLS_YF for factor-space
  coverage and every Nirnay commodity/FX basket for cohesion; every addition was
  verified fetchable on yfinance (5y) and, for basket members, co-directional
  with its target. Macro spine: EM FX legs `USD/MXN·BRL·ZAR` (LatAm/Africa were
  absent) + `USD/THB·TWD·MYR` (used in the USD/INR basket but missing from the
  spine), ag depth `Cocoa`/`Soybean Oil` (edible-oil import complex), refined
  products `RBOB Gasoline`/`Heating Oil` (added to the Brent target's exclusion
  list — same-barrel logic as WTI), and `^MOVE` bond volatility. Baskets: Copper
  +FM.TO/+LUN.TO (r +0.49/+0.52 — the largest missing pure-plays), Silver
  +AYA.TO/+USAS (r +0.53/+0.49; scarcity is structural, GATO/SILV delisted via
  M&A), Brent +SU/+CNQ (r +0.56/+0.60; refiner exclusion now documented —
  crack-spread businesses are not co-directional), Cotton +ZW=F (completes the
  acreage-competition triangle). USD/INR and Jeera baskets validated as-is
  (both already data-curated). New symbols enter the predictor pool on the next
  data refresh; the tuning suite should be re-run after that refresh, not before.

### Changed
- **Full tuning suite re-run FROM SCRATCH — zero config changes warranted
  (2026-07-13, `reports/tuning_20260713_004005.txt`).** Wiped the Aarambh result
  cache and ran all 16 studies on the widened grids (~12.5h; aarambh_full alone
  ~8.4h with no cache to resume). Every study reproduced its prior verdict and no
  constant moved: Aarambh window/refit/PCA/huber/ridge/lookback all noise-level
  (the widened 10→3000 window sweep confirms saturation past ~625 and the same
  sub-SE short-window hump, not adopted; the new 10-row window correctly trips the
  ⚠LEAK guard and is excluded); Nirnay MSF=20 holds on the cross-universe equity
  check (commodity winner MSF=10 stays worse on the 27-index majority); both DDM
  sets sit on their plateaus; SIGNAL_HORIZONS unchanged (analog ≈0 standalone, the
  purged model carries 1–10d); analog/blend/TOP_N/recency all noise; markers
  anchors byte-match the shipped p75/p90 (consensus ±0.26/±0.39, ConvictionRaw
  ±20/±50, Nirnay ±2.9); hero thresholds find no separation (occupancy-anchored,
  unchanged); calibration_lift reconfirms consensus (+0.039) ≥ calibrated (+0.023)
  with cal−raw lift −0.001; conv_weights flat; every ui_anchors tier matches its
  occupancy target. The recompute also validated the cache was never stale (≥100-row
  Aarambh rows byte-match). Config comments updated with the from-scratch provenance.

### Changed
- **Tuning grids widened again (depth pass 2, 2026-07-13).** Every study's test
  grid was expanded to span from near-degenerate to well past the current optima
  so the full response curve is visible: Aarambh windows now sweep **10 → 3000**
  (MIN capped at 2000 by the 9-yr sample; MAX ≥ sample saturates and is kept to
  confirm the plateau), REFIT 1–63, PCA 2–150, HUBER_EPSILON 1.0–4.0, plus two
  wider ridge grids and two denser lookback sets; Nirnay knobs densified (e.g.
  MSF 3–100, BASE_WEIGHT in 0.05 steps); analog TOP_N 1–150 and recency 15–3000d;
  both precedent horizon sweeps get a fine 1–25d band out to 120–180d;
  `conv_weights` 17 → 31 vectors (every simplex face); `ddm` leaks 0.01–0.80;
  `hero_thresholds` 30 → 96 percentile pairs; markers/ui_anchors percentiles down
  to p25/up to p99.5. The Aarambh resumable cache means only ~33 new configs
  (~2h) recompute; `hero_threshold`'s quantile set is now derived from its P_MOD/
  P_STR grids so a widened grid can't reference a missing key. Suite ETAs updated.
- **Chart palette centralized to a single source of truth (zero pixel change).**
  Design-system audit found the app shipped *two* palettes for the same semantic
  colors — the Plotly charts (`config.COLOR_*` + inline `rgba()`) used the
  brighter Tailwind-400 family while the CSS surfaces used a -500/custom family
  (only amber-gold was shared). Introduced `core.config._PALETTE_RGB` + an
  `rgba(name, alpha)` helper as the one definition for the chart palette, and
  migrated the 70 scattered inline `rgba(r,g,b,a)` literals across the four chart
  tabs to it. **Values are byte-identical** — this is centralization, not a
  recolor, so a future palette reconciliation is now a one-line-per-color edit.
  The chart↔CSS hue divergence is documented at `_PALETTE_RGB` for a later,
  deliberate decision.
- **Muted text tier lifted to clear the accessibility floor.** `--ink-tertiary`
  `#4B5563` (2.55:1 on `--bg-base` — below the WCAG 3:1 UI-contrast minimum, and
  it drives table headers, section subtitles, and chart annotations) → `#5B6675`
  (3.31:1). Still clearly the de-emphasized tier (secondary is 7.5:1); the
  hardcoded `.kv-table` header color now tracks the token instead of duplicating
  it.

### Fixed
- **False `COLOR_PURPLE` comment.** It claimed "matches CSS `--violet`" while the
  two differ by Δ47 (`#A78BFA` vs `#8B5CF6`); comment corrected as part of the
  palette centralization.
- **Hover values across every plot now clip to 2 decimals.** Plotly silently
  ignores a d3 number format inside a hovertemplate (`%{y:.2f}`) — and the axis
  `hoverformat` — under `hovermode="x unified"`, so hover boxes leaked full float
  precision (e.g. `Consensus (50/50): -0.3687992004699925`). Fixed centrally in
  `ui/theme.apply_default_hover`, called from `style_axes` (which every chart
  invokes right before `st.plotly_chart`): each visible trace's y-values are
  pre-formatted to strings in Python and inserted via `%{customdata[0]}`, which
  has no client-side format to ignore. Preserves `hoverinfo="skip"` fills,
  intentional `%{x…}` templates (the precedent Z-vs-forward scatter), custom
  `%{customdata}` labels (the fair-value "Implied target"), and marker `text`
  labels (the hero "S. Buy"/"Hold"); idempotent across multi-row subplots.
- **conviction_model binned the smoothed COMPOSITE with the ENGINE's tiers
  (F1 pairing violation).** `UnifiedConvictionModel._classify_signal` used
  CONVICTION_MODERATE/STRONG (anchored on ConvictionBounded) to bin the
  DDM-filtered convergence score — on that distribution the old 40/60 sat at
  ~p98/p100, so the TATTVA CONVICTION card read NEUTRAL ~96% of days and never
  printed STRONG. It now bins on the composite's own anchored set
  (COMPOSITE_THRESHOLDS × 100 → ±11/±18, WEAK at measured p50 ≈ 6).
- **CI band-width tier removed (measured degenerate).** The DDM's mean-reverting
  variance pins band width to 40.2–42.5 (p1–p99), so UI_BAND_NARROW/WIDE (30/60)
  could never fire — the interpretation-card sentence permanently read "Band
  moderate — some uncertainty". The sentence and both constants are deleted
  (a percentile-vs-history replacement would be a new unvalidated indicator —
  the class this audit exists to eliminate); the CI band itself remains drawn
  on the conviction chart, and ui_anchors keeps an informational distribution
  row so a future regime change in band width would still be visible.
- **Stale "current" markers in three studies.** `confirm_max_sweep.py` hardcoded
  `MIN=750` and flagged `←current` at MAX=750 (live: MAX=1000) — now reads
  `core.config` live; `markers_study.py` printed hardcoded current markers
  (0.5/0.25, 40/20, 2/2) instead of the live `UI_*` constants; `analog_confirm.py`
  labelled the live maha-only blend as ".55/.35/.10". All three now self-report
  from live config.

---

## [2.5.0] — 2026-07-04 — *Audit hardening · hero verdict rebuild · UI/UX polish*

Resolves an end-to-end statistical/correctness/infra/docs audit spanning the
walk-forward engine, the Intelligence calibration, the Precedent analog matcher,
regime detection, and the data/cache layer; rebuilds the hero convergence card
from a pure, unit-tested verdict function; and completes an institutional-grade
UI/UX polish pass (grid rhythm, motion, focus/a11y) within the existing design
language.

### Fixed
- **Warm-up look-ahead in the walk-forward signal stack.** The first
  `MIN_TRAIN_SIZE` rows of `predictions` were filled with an expanding mean of
  `y[:t]` — in `forward_signal` mode `y` is the h-day FORWARD label, so that mean
  drew on labels overlapping the forecast window it was standing in for. Those
  rows fed the Intelligence calibration frame, the analog matcher's `NetBreadth`
  feature, and several research sweeps. `predictions[:MIN_TRAIN_SIZE]` (and the
  breadth/conviction columns derived from it) are now left genuinely NaN/missing;
  every consumer (calibration loop, analog feature pool, DDM seed, pivot/divergence
  detection) now explicitly excludes or gracefully handles that region instead of
  reading a fabricated "neutral" reading from it.
- **Forward-change table and divergence detection used a corrupted pseudo-price.**
  `_compute_forward_changes`/`_compute_divergences` reconstructed a "price" via
  `cumsum`/`exp(cumsum)` of `y`, which in `forward_signal` mode is the h-day
  FORWARD return — summing it double-counts each daily return up to h times,
  inflating `FwdChg_*` (and the Diagnostics "Signal Performance" hit-rates/t-stats
  built on it) by roughly h×. `FairValueEngine.fit` now accepts an optional
  `price=` series (the app passes the real target price) and uses it for both.
- **AR(1) bias correction had the wrong sign.** The OU/half-life estimator's
  finite-sample correction subtracted `(1+3a)/n` from the OLS AR(1) coefficient;
  the correction is additive (Kendall 1954; Orcutt & Winokur 1969) — subtracting
  doubled the downward bias instead of removing it. Also corrected the citation:
  this is not Andrews (1993), which is a different (quantile-table-based)
  estimator.
- **Val IC / walk-forward IC were scored on overlapping daily forward returns.**
  Adjacent h-day forward-return observations share up to h-1 days, so a
  daily-sampled IC's effective sample size is ~n/h, not n — overstating precision
  by ~√h. The *reported* Val IC and each walk-forward-IC window are now scored
  non-overlapping (stride = the shortest hold horizon); the Optuna objective
  itself is unchanged (still scores the full overlapping frame for a smoother
  search surface). Hero/Aarambh-tab trust-chip thresholds recalibrated to the
  non-overlapping scale.
- **Intelligence calibration ran (and persisted a profile) on a degenerate
  all-constant convergence signal** when a target's Nirnay basket had little/no
  date overlap with Aarambh — every date got the same neutral default, so
  `convergence_score` was constant and the "calibrated" profile was an arbitrary
  fit to noise. Calibration (and the walk-forward IC) now skip when overlap is
  below a small floor.
- **Analog/precedent "N analogs" could be 1-3 historical episodes counted as N.**
  `find_similar_periods` picked the top-N by raw similarity with no exclusion
  window, so adjacent days from the same episode (near-identical rolling-window
  state) dominated the returned set — inflating the apparent independence of
  `summarize_forward`'s median/positive_pct. Now enforces a Theiler exclusion
  window (Theiler 1986) between accepted analogs; propagated to the research
  analog walkers too.
- **Precedent backtest verdict used a bare `|ρ| > 0.3` cutoff with no
  sample-size awareness** — on the ~20-30-point non-overlapping test split this
  is common under pure noise. Now gated on the Spearman p-value (`p < 0.10`).
- **"Ledoit-Wolf" shrinkage formula was mis-transcribed** (numerator/denominator
  terms swapped relative to the OAS closed form it was meant to implement),
  under-shrinking exactly where shrinkage matters most (near-isotropic state
  covariance). Corrected to match `sklearn.covariance.OAS` exactly.
- **Online HMM regime detector had no state-ordering constraint** — a long
  one-sided regime could drift the Bull emission mean below the Bear mean (label
  switching), after which `HMM_Bull`/`HMM_Bear` silently swapped meaning. Both
  the Numba kernel and the Python class now enforce Bull > Neutral > Bear after
  every emission-parameter update.
- **DFA Hurst exponent used a narrow, linearly-spaced scale range**
  (`min_scale = max(4, n/10)`), giving a noisy slope fit at typical `n`. Now
  `min_scale=8` with `max_scale=n//4`, log-spaced (`np.geomspace`) per standard
  DFA practice — cuts RMSE roughly 4x in simulation at H=0.5.
- **`_process_wf_chunk`'s causal break-point search used unpurged data** near the
  forecast boundary in `forward_signal` mode. Now excludes the last `purge` rows,
  matching the purge already applied to the ensemble's training rows.
- **NSE constituent-list fetch disabled TLS verification** (`verify=False`) with
  no corresponding need — removed; both archive hosts present valid certificate
  chains.
- **Snapshot-backfilled macro columns had no staleness ceiling** — a rate-limited
  ticker was silently refilled from whatever prior snapshot had it, however old.
  Now dropped instead of filled past a staleness threshold, and columns that
  are filled are surfaced in-app (previously console-only).
- **"Refresh Data" didn't clear the convergence tab's per-config normalization
  cache** (`conv_norm_causal::*` — the cleanup still targeted an old
  `conv_norm_params` prefix from before a rename), and that cache key didn't
  account for an intraday last-bar update. Cleanup now targets the live prefix;
  the key folds in row count + latest raw values.
- **Force-refresh window was a single process-global deadline** — on a
  multi-session Streamlit deploy, one user's "Refresh Data" forced every
  concurrent session's next fetch to bypass cache too. Now scoped per session.
- **`lead_lag_indicator`'s AARAMBH_LEADS/NIRNAY_LEADS comparison was degenerate**
  — it compared `abs(sign)` values (always 0 or 1) with a 1.5x margin, which can
  only ever resolve when one side's *direction* is exactly zero, never from a
  genuine magnitude dominance. Now compares the actual normalized magnitudes.
- **A blanket `RuntimeWarning` suppression hid every numeric warning
  process-wide**, not just the one legitimate source it was meant to cover
  (`nanmean`'s "empty slice" on the engine's own warm-up rows). That source is
  now scoped locally at its call site; the global filter is removed. Removing
  it surfaced a second, genuine warning it had been masking (next item).
- **MMR driver selection could be hijacked by a constant/pegged macro
  column.** `rolling().corr()` emits `±inf` (not NaN) on near-zero-variance
  windows — routine at 200+ macro columns (price-pegged ETFs, forward-filled
  holiday runs). `+inf` passed the old `~np.isnan` validity mask and always
  sorted last in the top-N driver selection, so a constant, information-free
  column could permanently occupy a "top driver" slot and its downstream
  `inf/inf` produced the `RuntimeWarning: invalid value encountered in scalar
  divide` at `engines/nirnay.py`'s MMR row loop. Correlations are now
  sanitized (non-finite → NaN, clipped to `[-1, 1]`) at the source, the row
  validity mask uses `np.isfinite`, and the division guard rejects a
  non-finite denominator.
- **MMR "top driver" list emitted integer column positions as the driver
  name**, and reported `|r|` instead of the signed correlation (an inversely
  related driver showed as positively correlated). Both corrected — positions
  now map back to the macro ticker name, and the signed last-bar correlation
  is reported.
- **Analog cards' "Extension (Z)" always read `+0.00`.** `AvgZ` was dropped
  from the analog *matching* feature set in the 2.2 re-tune but the Precedent
  tab's cards still read it from the per-analog dict, which never carried it
  — every card's tier badge/color was keyed off a permanent default. `AvgZ`
  is now carried as a display-only field (not matched on).
- **Analog candidate pool included the engine's own warm-up rows**, whose
  `NetBreadth` is genuinely missing (not neutral) — matching against a
  median-filled fabrication is not a real state match. The pool now excludes
  those rows; the Theiler exclusion gap is measured on the original temporal
  row position (not the post-filter array offset) so it still means "N
  trading days apart" after filtering.
- **`r2_vs_rw`, in forward-return mode, benchmarked against the wrong null.**
  It compared the forecast to "yesterday's realized forward return" as the
  naive baseline — for a *return* forecast the martingale null is zero, not
  the previous label; scored against the previous-label baseline, a
  skill-less (always-zero) forecaster registered `r2_vs_rw ≈ +0.5` on pure
  noise (measured by simulation). Forward mode now benchmarks against the
  zero forecast; level/residual modes are unchanged (last-value RW baseline
  is the correct null there).
- **Regime-distribution stats (`get_regime_stats`) counted the engine's
  warm-up rows** as `NEUTRAL`, diluting the Aarambh tab's "% of history
  classified oversold/overbought" by roughly the warm-up's share of history.
  Now excludes rows with no genuine forecast; the tab's percentage
  denominator was updated to match.
- **The `Actual` display column showed a literal `0.0000`** for the last
  `FWD_HORIZON` rows — those rows' forward-return label doesn't exist yet and
  was zero-filled only so the regression wouldn't choke on it, but the same
  zero-filled array was reused for display. Now masked to NaN for display.
- **DDM warm-up seeding had a look-ahead.** The drift-diffusion filter's
  initial state, when its own leading input is NaN (the engine's warm-up
  region), was seeded from the first *finite* observation — for a series
  whose leading segment is NaN that value can be hundreds of rows in the
  future. Now seeds a neutral 0.0 in that case, matching the filter's own
  per-step NaN handling.
- **Phase-3 terminal log printed hardcoded MSF/ROC/regime-sensitivity/
  base-weight literals** instead of the `core/config.py` constants actually
  passed to the engine — the console would silently disagree with a tuned
  config. Now reads the same `NIRNAY_*` constants the engine call uses; two
  previously-unlogged knobs (MMR top-N, oversold/overbought thresholds) are
  now printed too.
- **The Intelligence-calibration overlap gate decided *after* the "First-Pass
  Conviction Model" console/progress label was already chosen** — a
  low-overlap run that the gate correctly skipped still announced a "first
  pass" that would never have a second. The gate (unchanged logic) now runs
  immediately after the convergence-scoring summary, before any pass label
  is chosen.
- **Two distinct late-pipeline stages ("Detecting Divergences", "Walk-Forward
  Validation") both posted 93% on the progress bar**, reading as a stall.
  Re-sequenced the tail: 93 → 94 → 95 → 96 → 100, one number per stage.
- **`theme.css` could crash the app on startup on non-UTF-8-locale Windows
  systems.** `Path.read_text()` with no explicit encoding falls back to the
  OS locale (commonly cp1252 on Windows), which cannot decode the
  stylesheet's embedded Devanagari string — `UnicodeDecodeError` before
  anything renders. Now reads with `encoding="utf-8"` explicitly.
- **The app required Streamlit ≥1.42 but declared `>=1.30.0`.** `width=`
  on `st.button`/`st.plotly_chart`/`st.dataframe` was introduced in 1.42; on
  the declared minimum the app crashed at first render
  (`TypeError: unexpected keyword 'width'`, reproduced on 1.37.1). Pin raised.

### Added
- **Hero verdict rebuilt as a pure function.** `ui.components.build_hero_verdict`
  now holds 100% of the hero card's interpretation logic — headline-object
  resolution (calibrated model → normalized consensus → Aarambh-only), signal-
  label normalization, trust tiering, agreement tiering, and the precedent
  evidence row — as a side-effect-free function returning a plain dict,
  unit-tested independently of Streamlit. `render_hero_card` renders that
  verdict as a structured, tagged evidence list (MODEL / PRECEDENT /
  INTERNALS / RISK, each confirm/conflict/neutral/info) instead of a single
  run-on interpretation paragraph. The precedent evidence row is gated on a
  minimum of 5 *distinct* (post-Theiler) analogs before claiming agreement or
  divergence — below that it reports "thin sample, not probative" instead of
  a base rate built on 1-2 repeated episodes.

### Changed
- Non-overlapping Val IC changes the hero/Aarambh-tab trust-chip thresholds
  (previously 0.02/0.05, now 0.10/0.20) — same qualitative tiers, recalibrated
  to the less-inflated scale.
- `_RESULTS_CACHE_MAX` (session result cache) raised 3 → 6; the universe has
  grown well past "3 commodities".
- Aarambh-tab and Diagnostics-tab copy is now mode-aware: forward-return
  forecast mode no longer describes itself with relative-value language
  ("cheap/expensive valuation") it doesn't mean, and the OOS R² / ADF / KPSS
  cards no longer grade a magnitude-R²-near-zero forecast as if it were a
  failing level-regression model.
- **UI/UX polish pass** (grid rhythm, motion, focus/a11y — no change to the
  Obsidian Quant Terminal palette, typography, or card anatomy):
  - Two silently-conflicting duplicate `@keyframes` (`pulse`, `shimmer`,
    each defined twice with different motion) resolved by scoping the
    earlier pair (`pulseDot`, `shimmerSkeleton`) to their actual consumers.
  - Every `transition: all` in `theme.css` (21 sites) replaced with explicit
    property lists — `all` was animating unintended properties and, at two
    sites where a later same-selector rule's `transition` silently won the
    whole property outright, had been masking which properties the earlier
    rule's hover state actually needed.
  - The metric-card entrance stagger (`.metric-card:nth-child(N)`) never
    matched in practice — a metric card is always its own Streamlit column's
    only child, so it was permanently `nth-child(1)`. Replaced with a stagger
    keyed off the column's position in the row.
  - Metric-tooltip: dead duplicate `white-space` declaration removed;
    last-column tooltips in a row no longer clip past the viewport edge.
  - Sidebar "CONFIGURE" hint's `onclick` handler removed — Streamlit
    sanitizes inline handlers in markdown HTML, so it never fired; the
    element is now honestly styled as the non-interactive pointer it is.
  - `.stPlotlyChart`'s bottom margin no longer double-stacks with
    `.section-gap` when a chart is the last element before a section
    boundary (was 48px there vs. 32px everywhere else).
  - Content width capped at 1720px on ultra-wide displays (was unbounded
    98%); no visible change ≤1720px.
  - `tabular-nums` applied to the remaining un-covered numeric readouts
    (metric-card values, spec rows, hero score) so ticking values don't
    jitter their container width.
  - `:focus-visible` rings added for buttons/tabs (previously no visible
    keyboard focus indicator anywhere in the app).
  - Mobile: removed a rule that force-clamped every chart to 300px height
    (was crushing the 680px 3-row Unified Signal stack); tab bar now scrolls
    horizontally instead of wrapping; interactive touch targets raised to
    the ≥44px WCAG 2.5.8 floor.
  - z-index stacking order documented inline (-1 decorative glows / 0 ambient
    textures / 1 card content / 10 sticky headers / 100 tooltips / 999 fixed
    hint).

### Removed
- Dead UI components with zero call sites: `render_collapsible_section(_close)`,
  `render_conviction_signal`, `render_signal_card`, `render_system_card`,
  `render_kv_table`, `get_signal_badge`, `render_chart_skeleton`,
  `render_export_button_row`; the theme toggle and keyboard-shortcuts hint
  (rendered inside 0-height component iframes whose scripts targeted the
  iframe's own document, not the app's — inert, and the light-mode CSS they
  gated had accordingly never rendered); the unused `MathUtils` namespace class;
  the `rendered_tabs` session-state set (written every run, never read, under a
  comment claiming lazy loading Streamlit doesn't actually do); dead
  `regime_weak_bull`/`regime_weak_bear` fields (always hardcoded 0, never read).

### Docs
- Corrected "conformal prediction" → "rolling robust quantile z-scores"
  throughout (the module never had a conformal-prediction coverage guarantee)
  and "Andrews (1993)" → "Kendall (1954) / Orcutt & Winokur (1969)" for the
  AR(1) bias correction. DDM "confidence bands" relabeled "uncertainty bands"
  (heuristic width, not a calibrated statistical interval). Diagnostics'
  "Covariance Shrinkage"/"Regime Persistence" cards (which displayed HMM/GARCH
  initial-prior constants mislabeled as measured telemetry) removed rather than
  left misleading. Stale "IC vs forward PE" copy (pre-2.0.0 target) corrected to
  "forward return".

---

## [2.4.0] — 2026-06-23 — *Data-pipeline recovery · forward price projection · UI fidelity, motion & logging*

Restores end-to-end runnability after the leakage hardening (removing a look-ahead
`bfill` had a latent consequence that collapsed datasets), gets **every** asset class
across the walk-forward floor, turns the Aarambh price panel into a genuine forecast,
and runs a full **fidelity pass** so the UI describes the system exactly as it behaves.
Backward compatible — no config or data-format changes.

### Fixed
- **Total dataset collapse to a single row ("Need 1500+ data points").** Removing the
  look-ahead `.bfill()` from the predictor fill (a correctness fix) left near-empty
  columns all-NaN; a target retained for selection but returned ~empty by yfinance
  (e.g. `^CNXSC` / Nifty Smallcap 100) leaked into the predictor set, and
  `dropna(subset=all features)` then wiped every row. Added a **per-feature history
  guard**: after the causal `ffill`, any predictor still NaN within the most recent
  `MIN_DATA_POINTS` target-session rows is dropped (it can't support the walk-forward) —
  without ever re-introducing `bfill`. The window now settles on real data above the floor.
- **India indices and the ETF universe failed the 1500-row floor while US/commodity
  targets passed.** The history guard and `dropna` were measured on the US-weekday spine,
  but the walk-forward runs on the target's exchange sessions — NSE's heavier holiday
  calendar turned ~1582 weekdays into only ~1496 sessions. The Phase-3 `session_mask`
  restriction now runs **first**, so all row-filtering happens in target-session space;
  the guard then drops predictors too young for *that* calendar (e.g. SGOV vs an NSE
  target), extending the usable window back (Nifty 50: 1496 → 1598 sessions).
- **`AttributeError: 'int' object has no attribute 'date'` while building the result
  cache key.** The date-range fingerprint called `.date()` on the frame index, which is a
  `RangeIndex` (integers) after load. It now derives the range from the `DATE` column,
  with a row-count surrogate fallback when no date column is present.
- **Dead macro ticker.** `BUNL.L` (delisted) → `IBGL.L` (iShares € Govt Bond 15–30yr,
  the renamed ISIN-equivalent) — restores the Germany-Bunds/long predictor (history to 2008).
- **CUSUM change-point self-reference (look-ahead).** Both the Numba kernel
  (`analytics/regime.py`) and the Python `CUSUMDetector` included the current observation
  in the running mean/σ used to z-score that same observation. The window now excludes the
  current point.
- **Nirnay MMR warm-up leak.** The per-driver `x_std` warm-up fill used the full-series
  std (`fillna(x.std())`) — a look-ahead; replaced with a neutral `1.0`.
- **Convergence magnitude dimension saturated.** A spurious `×10` on the Nirnay oscillator
  magnitude in `CrossSystemValidator` pinned the magnitude-alignment score at its ceiling;
  removed.
- **OU σ under-estimated near a unit root.** The `a > 0.98` branch dropped the `1/(1−a²)`
  factor; unified to a single formula across all `a`.
- **Convergence tab crash guard.** `None` values in the conviction series are coerced to
  `np.nan` before `np.clip` (previously a `TypeError`).
- **Convergence regime-key naming.** `regime_bull`/`regime_bear` → `regime_bull_pct`/
  `regime_bear_pct` on both writer (`app.py`) and reader (`cross_validator.py`), so the
  percentage scale is explicit.

### Added
- **Forward expected-price projection (Aarambh).** The Price & Forecast panel turns the
  latest expected forward return into a price path — an implied target *h* days out with
  an OOS-RMSE uncertainty cone, anchored at the last close (emerald = bullish, rose =
  bearish). The section is now a genuine forecast, not a history of past leans.
- **Full per-constituent regime stack surfaced (Nirnay).** The engine runs
  Kalman→GARCH→HMM→CUSUM but only HMM reached the UI. The constituent drill-down now also
  shows `Vol_Regime` (GARCH), `Change_Point` (CUSUM) and `Confidence` — making the
  advertised "HMM · GARCH · CUSUM" true at the tab level.
- **Continuous progress bar + data-prep telemetry.** A single main-area progress bar now
  drives the whole run from the first click (fetch → data spine → engines → convergence),
  replacing the sidebar spinners. A "DATA PREPARATION" terminal trace logs the row
  evolution (fetched → session spine → features dropped → final), and every early-exit now
  prints a specific failure reason instead of a silent "Need 1500+".

### Changed
- **Landing/tab copy corrected to the actual system.** Aarambh ensemble shown as
  **PCA-OLS + Huber** (the default; Ridge/ENet/WLS are off); the Nirnay card spec fixed to
  MSF+MMR · Oversold/Overbought % · HMM·GARCH·CUSUM (the OU-90d projection and DFA Hurst it
  had listed are *Aarambh* features); Convergence card now "Fusion: Aarambh + Nirnay".
  Tagline broadened to **Cross-Asset**; generic "commodity" copy → "target" across tabs.
- **Aarambh breadth cards** now read "fraction of **lookback windows**" (not "models" —
  breadth is across windows); the **Nirnay HMM** header is clarified as the basket-average
  of per-constituent states; the **Avg Unified Signal** card's color thresholds align to
  ±2 (matching its label and chart).
- **Unified control-hint typography.** A single `.control-hint` style + `render_control_hint`
  helper replaces ad-hoc `st.caption` sidebar hints, so the sidebar/tab fine-print is one
  coherent tier; a subtle "working" sheen was added to the progress fill.

### Docs
- README: version, and the Aarambh-tab description now notes the forward price projection.
- Refreshed the `_render_fair_value_chart` docstring to describe the new projection.

---

## [2.3.0] — 2026-06-22 — *Per-exchange calendars · global macro universe · leakage & freshness hardening*

Takes data freshness from heuristic to calendar-exact: holiday-aware **per-exchange
trading calendars** now drive the "days-behind" notices, the partial-session gate, and
the per-target model spine. Adds a **preflight data check** to the tuning orchestrator,
and resolves a round of **leakage / data-integrity issues** surfaced by auditing a newly
expanded *global* macro universe (European/Asian/AU indices + bonds). Backward
compatible — `exchange_calendars` is an OPTIONAL dependency that degrades to the prior
Mon–Fri behaviour when absent, so a lib-less deploy is byte-identical to 2.2.0.

### Added
- **Per-exchange trading calendars — holiday-aware data freshness (Phase 1).** New
  `data/calendars.py` resolves each target ticker to its exchange and counts "trading
  days behind" on that exchange's actual session calendar via the `exchange_calendars`
  library. This retires the ~1-day over-count the old Mon–Fri `busday_count` produced
  across market holidays (Diwali, Thanksgiving, Juneteenth, 15 Aug…) — the freshness
  notices no longer flash a premature "behind" across a closed session. Wired into both
  the dataset and per-target freshness notices in `app.py`.
  - **Graceful, optional dependency.** The import is guarded: if `exchange_calendars`
    is absent (or a calendar can't be built / a date is out of bounds), every count
    degrades to the *exact* legacy `busday_count` — never broken, never worse than
    before, only better when the lib is present. Added to `requirements.txt`.
- **Exchange-aware partial-session gate (Phase 2).** The "partial latest session"
  warning previously counted *every* numeric column equally, so a one-region holiday
  (e.g. US Thanksgiving) made the whole row look half-stale and could trip a false
  warning. It now (via `is_session`) judges only the columns whose exchange was actually
  OPEN on the latest date — a closed market's forward-filled value is legitimate, not
  stale. Degrades to the prior all-column weekday check without the lib.
- **Target-exchange session spine (Phase 3).** The fetched matrix is a Mon–Fri spine
  (FX trades every weekday), so each target carried a forward-filled fake "no-change"
  bar on its own market holidays (~1% CME, ~4% US equity, ~6% NSE; 0% FX). The
  per-target model frame is now restricted to the target exchange's real sessions
  (`session_mask`), so the walk-forward trains on genuine bars and the forward-return
  horizon counts true target trading days. Nirnay/Convergence already reindex onto the
  target's dates, so they stay aligned. No-op for FX and under the weekday fallback, and
  a `.any()` guard refuses to blank the frame if a ticker→calendar mapping misfires.
  - *Note:* the `research/` tuning scripts still build on the weekday spine; the 1–6%
    row delta is immaterial to the row-count hyperparameters (`MIN_TRAIN_SIZE` 750 ≈ 3y
    of sessions either way), so the validated configs carry over.
- **Tuning preflight — data sufficiency check (`research/run_tuning.py`).** Every study
  shares the same 9-year `fetch_commodity_dataset` pull, so a thin/rate-limited fetch
  would silently corrupt a multi-hour run. The orchestrator now warms that fetch ONCE
  and asserts it's deep/broad enough (`≥ MIN_DATA_POINTS` rows, `≥ 50` numeric columns,
  `≥ 60%` of targets present) before any tier runs — aborting with a clear rate-limit /
  "Refresh Data" hint instead. Flags: `--skip-preflight`, `--preflight-warn`.

### Fixed
- **Predictor→target leakage from the expanded universe.** Two targets could be
  "explained by themselves", contaminating the Aarambh fair-value residual the whole
  system trades:
  - `USD/INR` now excludes **every INR-leg cross** — `INR/USD` (its exact reciprocal,
    near-perfect collinearity) plus `AUD/NZD/CAD/CHF/CNY/SGD/HKD-INR` (each =
    `X/USD × USD/INR`). Computed from `MACRO_SYMBOLS_YF` so future INR crosses are
    covered automatically. (Previously only `EUR/GBP/JPY-INR`.)
  - `Copper` had **no** exclusion list (the only commodity target without one); now
    excludes `Base Metals (DBB)` (~⅓ copper). DBC/GSG are kept — only a few % copper.
- **Dead tickers silently dropped by the ≥20% coverage filter** removed from
  `GLOBAL_MACRO_MAP`: `^TPX` (TOPIX — yfinance returns nothing; `^N225` covers Japan)
  and `JGBL.L` (Japan Gov Bonds — ~13% coverage). They produced no column, just
  confusion; omitted with a note rather than feigned.
- **Exchange resolution wrong for the new global universe** (`data/calendars.py`). The
  `^`-index heuristic assumed US-or-India, so European/Asian indices resolved to the
  *Indian* calendar (`^GDAXI, ^FCHI, ^STOXX50E, ^FTSE, ^IBEX, ^AEX, ^SSMI, ^N225,
  ^KS11, ^KQ11`) and China/Australia to *US* (`000001.SS, 399001.SZ, VGB.AX`) — skewing
  the Phase-2 gate for those columns. Added a suffix→MIC table (`.DE→XETR, .SS→XSHG,
  .AX→XASX, .HK→XHKG, .T→XTKS, …`) and a foreign-index map; **unknown `^` symbols /
  unmapped suffixes now return the safe weekday fallback instead of a guessed calendar**,
  so a future foreign add can never again be silently mis-dated. Verified against real
  DE/JP/CN/UK closures in `research/test_calendars.py`.
- **Aarambh tab — conviction card vs DDM plot now agree (single source of truth).**
  The DDM-Filtered Conviction chart drew the *unbounded* `ConvictionScore` line/fills
  while its own confidence bands and the interpretation card used the *bounded* series
  (`ConvictionBounded`) — a three-way mismatch on a `[-100,100]` axis, visible at
  extreme conviction. The line/fills now read `ConvictionBounded`, so the plot's last
  point equals the card value and matches the bands and axis.
- **Data-freshness off-by-one across timezones.** "Trading days behind" anchored
  `today` to UTC while the data date is exchange-local, so a UTC-hosted deploy rolling
  past midnight over-counted a current bar as "1 day behind". Now anchored to the
  earlier of UTC and machine-local and clamped `≥ 0` (this is additionally subsumed by
  the per-exchange calendar count above).
- **Convergence — short-history degeneracy.** When the full Aarambh∩Nirnay overlap is
  tiny (new sheet target / freshly-listed constituents), the z-score σ collapsed to its
  `1e-10` floor and the normalized plot flat-lined at zero, misreading as a confident
  "neutral". A guard now suppresses the plot below 10 shared sessions with a calm
  "building convergence history" note; the metric cards still show the raw latest reads.

### Tested
- **Inverse-basket polarity path** (`engines.nirnay.apply_polarity`, `TARGET_POLARITY = -1`)
  execution-verified for the first time via `research/test_polarity.py` — full 27-column
  schema: pair swaps, sign negation, neutral-column preservation, no-op for `+1/0/None`,
  involution (double-flip = identity), breadth-% conservation. No live target sets `-1`
  yet, so the path previously shipped unexercised.
- **Per-exchange calendars** covered by `research/test_calendars.py`: ticker→exchange
  resolution (incl. the global universe + safe fallback), holiday-aware counts vs the
  naive mask, `is_session`/`session_mask` against real foreign closures, and the forced
  no-library path reproducing `busday_count` byte-for-byte.

---

## [2.2.0] — 2026-06-19 — *Signal Horizons · Precedent engine · leakage-free walk-forward*

Adds a two-lens forecast-horizon selector, a new **Precedent** analog-matching tab
(ported from Arthagati and validated across the full 33-target universe), and a
correctness fix that purges future-leakage from the Aarambh walk-forward. The two
horizons and the analog hold grid were chosen by computational study, not by hand.
Backward compatible — no data-format or cache-key breakage (the lens is a new cache
dimension; old profiles simply recalibrate).

### Added
- **"Refresh Data" control — force a live re-pull.** A sidebar button below Reset
  Analysis that force-fetches the whole universe live, then recomputes (Reset =
  re-run on cached data, fast; Refresh = re-fetch + re-run, slower). Implemented
  snapshot-preserving: `Cache.begin_force_refresh()` opens a window in which
  `Cache.get` misses (→ live fetch) **without deleting the disk snapshot**, so a
  failed forced refresh (rate-limit / circuit-open) degrades to last-good data, not
  to an empty app — the safety the naive "clear cache" lacks. Also fixed
  `all_caches()` to include the constituents cache (was missed). The data-freshness
  notices now point users to it. UI matches the existing sidebar button idiom (no
  new visual language).
- **`research/` suite + `run_tuning.py` orchestrator.** All the session's tuning &
  validation harnesses (Aarambh/Nirnay/analog sweeps, marker & hero studies — 11
  scripts) moved out of the repo root into `research/` (path-shimmed so each still
  runs standalone). A single `python3 research/run_tuning.py` re-runs the whole
  suite, tees one consolidated timestamped report to `research/reports/`, and prints
  a **current-vs-validated reference** for every tuned constant. By design it
  *reports only* — config stays applied-by-hand after review (auto-tuning would
  invite overfitting / regime-chasing). `research/README.md` documents the suite.
- **Nifty 50 — PE target (sheet-sourced), under India Indices.** A second
  non-yfinance target after Jeera: its daily P/E series is pulled from a published
  Google Sheet (column `NIFTY50_PE`) via the same `data/sheets.py` contract (cache +
  circuit-breaker + stale fallback) and injected into the model matrix. The sheet
  fetcher/parser were generalized for an arbitrary value column + auto-detected date
  column, and `_fetch_exogenous_targets` now loops every registered `SHEET_SOURCES`
  entry. Registered via a new `SHEET_TARGETS` map (sentinel ticker kept out of the
  yfinance maps; category "India Indices"; ~20y / 4.9k rows of history). It borrows
  the **Nifty 50 constituents** as its Nirnay basket via a new `NIRNAY_BASKET_ALIAS`
  (the PE co-moves with constituent strength, polarity +1), so it runs the FULL
  pipeline — Aarambh forecast + Nirnay breadth + Convergence + Precedent.
- **Signal Horizon lenses (2).** A sidebar selector picks how far ahead the engine
  reads — **Tactical (10d)** for hedging / short-term and **Positional (20d)** for
  positioning — on daily bars throughout (no weekly resampling, which would starve
  the walk-forward / conformal / Optuna machinery: ~9y ≈ 470 weekly rows < the
  500/750 train windows and the 1500-point floor). Each lens sets the forecast
  horizon (`FWD_HORIZON`), the predictor-momentum window, lens-scaled DDM smoothing,
  and its **own** calibrated Intelligence profile (keyed per `(target, lens)` so they
  never clobber). The engine cache keys on the lens, so a Tactical and a Positional
  read coexist in one session — position on the long lens, hedge on the short one.
  Both `d` values were finalized from a 33-target study (see *Changed*), not guessed.
  (`SIGNAL_HORIZONS` in `core/config.py`.)
- **PRECEDENT tab — historical analog matcher.** A non-parametric base rate ported
  from Arthagati: covariance-aware **Mahalanobis** matching (Ledoit-Wolf shrinkage) +
  detrended-trajectory cosine + recency, run over Tattva's own causal state features
  (`AvgZ`, net internal breadth, target momentum/volatility, rolling Hurst). For the
  most statistically-similar historical states it reports what the target *actually
  did next*, at the active lens's hold horizons — an empirical complement to the
  Aarambh forecast, framed as descriptive (the calibrated edge still lives in the
  Diagnostics walk-forward IC). Rendered with the ported Obsidian-Quant analog cards,
  a forward-return base-rate summary, and a state→forward-return backtest.
  (`analytics/analogs.py`, `ui/tabs/tab_precedent.py`; new `analog-*` CSS reusing the
  existing `position-card`/`conviction-bar` system.)
- **Reproducible research harnesses** behind the horizon/engine decisions, all using
  honest **non-overlapping** (stride = horizon) IC to avoid overlap inflation on
  smooth multi-day forward returns: `precedent_study.py` (model-vs-analog at multiple
  horizons on one target), `precedent_universe_sweep.py` (analog potency across all
  33 targets), `precedent_vs_model_sweep.py` (purged model vs analog by asset class).
- **Jeera (NCDEX cumin) target** — the first non-yfinance commodity. Its daily
  price is pulled from a published Google Sheet via a new `data/sheets.py`
  fetcher (same cache + circuit-breaker + stale + committed-CSV-snapshot
  resilience contract as the yfinance path), injected into the Aarambh matrix
  reindexed onto the macro calendar. ~11y of clean daily history.
- **Data-backed Nirnay basket for Jeera.** Curated from an 11-year daily
  return-correlation study (`/tmp/jeera_research.py` methodology) over a ~70-name
  candidate universe (Indian agri/FMCG/agrochem/seed/sugar/farm-equipment
  equities + global ag-soft futures + macro refs). Findings:
  - Jeera is idiosyncratic/domestic/supply-driven → all single-name linkages are
    modest (max daily r ≈ 0.08), but the **equal-weight basket aggregates to
    r ≈ +0.087 daily / +0.082 weekly** — and unlike Nifty (whose +0.076 daily
    decays to +0.010 weekly) it **persists at the weekly horizon**, i.e. genuine
    agri-regime signal rather than market beta. Recent-3y r ≈ +0.111.
  - **Global ag-soft futures are excluded** (the key divergence from Cotton's
    basket): CT=F/ZC=F/ZW=F/CC=F are decoupled from jeera (daily r ≈ 0 to
    negative). Domestic soft-commodity exposure is captured via sugar equities
    instead. True sibling spices (coriander/turmeric/guar) would fit but are
    NCDEX-only → future enhancement via the same sheets pipeline.
  - **International spice/ingredient majors also excluded** (McCormick, Olam/ofi,
    IFF, Symrise, Kerry, ADM, Bunge, Nestlé, Unilever): flat-to-negative vs jeera
    (Olam weekly r = −0.10, McCormick 3y r = −0.08). They are cumin *buyers*, so
    a price spike is a margin headwind (inverse exposure, not co-directional),
    and they trade async non-Indian calendars.
  - Final 17 names span agri-inputs/fertilizer, sugar, FMCG-foods, spice-direct,
    farm equipment, grain processing, and seeds (`COMMODITY_BASKETS["Jeera"]`).
- **Expanded `GLOBAL_MACRO_MAP` by 37 predictors** (≈98 → 135), all verified with
  ~11y yfinance history, filling orthogonal gaps in the Aarambh predictor pool:
  a full **FX complex** (UDN, USDU, FXE/FXY/FXB/FXF/FXA/FXC, CEW), **REITs**
  (VNQ/VNQI/REET — previously no real-estate asset class), **inflation
  expectations** (RINF), the **remaining GICS sectors** (XLU/XLP/XLY/XLK/XLV/
  XLRE/XLC + XHB/IYT/SMH), **equity style factors** (VTV/VUG/MTUM/USMV/SPHB/VYM),
  **regional equity** (EWJ/EZU/EWY/EWW/EWT/EWU), and real assets (WOOD/IGF). The
  universe is PCA-reduced inside each walk-forward window, so added inputs broaden
  factor coverage without destabilising the ensemble. (Cold-fetch verified:
  model matrix 143 → 180 columns.)

### Changed
- **Forecast lenses finalized to two (10d / 20d), data-driven.** A universe-wide
  walk-forward of the analog engine (33 targets, non-overlapping IC;
  `precedent_universe_sweep.py`) showed analog edge peaking at **+20d** (mean rank
  IC 0.162, positive in 28/33 targets) and collapsing beyond it (+40d: 1/33
  significant; +60d: **0/33**). So the prior three-lens set (Tactical 10d / Swing 20d
  / Positional 40d) was **cut to two — Tactical 10d and Positional 20d** (20d being
  the longest horizon the analog actually supports), and the Precedent hold grids
  were trimmed to only the validated horizons: **Tactical reads 5d + 10d**,
  **Positional reads 10d + 20d**. 40/60/90d were dropped as non-predictive. (Honest
  caveat in the code: even 20d's edge is full-sample; recent-half IC has decayed,
  so the precedent is a *fading* base rate strongest as a 10d confirmer.)
- **Precedent (analog) engine re-tuned for Tattva — fixes the recent decay.** Since
  the analog is now the system's primary directional edge (post-purge the Aarambh
  model IC ≈ 0), its ported-from-Arthagati knobs were swept honestly (33-target
  non-overlapping OOS IC, full + recent-half; `analog_tuning_study.py` +
  `analog_confirm.py`). Findings → changes:
  - **Blend → pure Mahalanobis (1/0/0).** Trajectory cosine added nothing and
    recency decay *hurt* the recent regime; dropping both **recovered the decayed
    recent-half IC** (10d −0.010 → +0.079, 20d −0.083 → +0.095) while holding
    full-sample IC. (Their computation is now skipped when weight = 0 — a live
    speedup for the Precedent tab + hero.)
  - **Dropped the `AvgZ` feature** — it degraded the recent regime; `NetBreadth`
    proved critical (kept). **No new feature helped** — ModelSpread, ExtremeBreadth,
    SignalBreadth, ConvictionRaw, MomentumLong all tested, none added lift; so
    "do we need more features?" → no. Similarity-weighting ≈ median (no gain).
    1d horizon is noise (IC ≈ 0) → surfaced on the Precedent tab as an **honorary
    reference tile** (`PRECEDENT_HONORARY_HORIZON`), clearly caveated and excluded
    from calibration so it can't dilute the Val-IC.
  - **Precedent tab backtest IC is now NON-OVERLAPPING** (stride = horizon) — the
    old overlapping daily IC was inflated; the chart now reports the honest OOS number.
  - **Hero reliability gate:** when the analogs are internally split (~coin-flip
    on direction) the hero now says "Precedent is split — low conviction" instead of
    claiming agreement/divergence.
- **Hero card now reads the precedent as a co-equal second opinion.** A 3-model
  non-overlapping study (`hero_study.py`, 8 targets) found the hero's convergence
  signal *does* predict (OOS IC +0.158, 55% hit) but the **analog precedent is the
  stronger directional read** (IC +0.226, 58% hit) and adds genuine independent
  value — while the plot markers add **nothing** (they ARE the convergence's own
  inputs; the +markers model was byte-identical to current). So the hero now
  computes the precedent base rate for the active target/lens and folds it into the
  interpretation: **agreement** confirms the read, **disagreement** is flagged as a
  divergence (the analog being the historically stronger predictor). Markers were
  deliberately NOT added (proven redundant). Cached per (target, lens, length) to
  avoid recompute on rerun.
- **Unified-Signal plot markers re-anchored to the data.** The 3-row plot's
  reference lines + marker-color tiers were hand-set magic numbers that were badly
  mis-scaled (a marker-distribution study across 8 targets / 17.6k days,
  `markers_study.py`, found Row-1 ±0.5 fired only **3%** of days while Row-2 ±20 and
  Row-3 ±2 fired **51% / 41%**). Re-set to each signal's p90 (strong) / p75
  (moderate) quantiles so "strong/moderate" means the same extremeness on every row,
  and lifted into named config constants (`UI_CONSENSUS_*`, `UI_CONVRAW_*`,
  `UI_NIRNAY_AVG_THRESHOLD`): Row 1 ±0.5→**±0.40/±0.25**, Row 2 ±40/±20→**±60/±40**,
  Row 3 ±2→**±2.5**. (The forward-return check showed the conviction rows are
  mean-reverting and Nirnay-avg is flat — these are extremeness guides, not
  actionable thresholds.)
- **DDM smoothing, the Intelligence hold-grid, and the Val-IC are now lens-aware.**
  DDM `leak_rate` scales ~(10 / horizon) so a longer lens turns over slower
  (Tactical 0.10 → Positional 0.05); calibration/durability IC is scored at the
  lens's own hold horizons rather than a fixed grid.
- **Aarambh engine defaults re-tuned post-purge.** Once the walk-forward stopped
  leaking (see *Fixed*), the old leaky-study defaults were re-validated across 33
  targets × both lenses on honest non-overlapping OOS IC (`aarambh_tuning_study.py`,
  `confirm_max_sweep.py`): **`MIN_TRAIN_SIZE` 500 → 750** (the one real gain: combined
  IC −0.004 → +0.019, mostly lifting the 20d lens + India equities) and
  **`REFIT_INTERVAL` 5 → 10** (equal skill at ~2× lower walk-forward cost — the prior
  "more refit = more skill" ladder was a pure leakage artifact). `MAX_TRAIN_SIZE`=750,
  `ENSEMBLE_MODELS`=ols+huber, and PCA=20 were re-confirmed (PCA=30 overfits;
  elasticnet remains worst). Config rationale comments updated to match.
- **USD/INR Nirnay basket refined (data-backed).** An 11y daily+weekly
  return-correlation study confirmed the **co-directional dollar/USD-Asia design**
  (polarity +1) and upgraded its members: now 11 names — UUP, **USDU** (added,
  volume-bearing), **DX-Y.NYB**, the strongest USD/Asia crosses (SGD/KRW/IDR/THB/
  PHP/TWD), plus **CNY=X** (China/EM-Asia anchor) and MYR=X. Equal-weight tracks
  USD/INR at daily r ≈ +0.354 / weekly +0.404 with low intra-basket redundancy
  (0.21), and now carries 2 volume-bearing members (UUP+USDU) for Nirnay's
  microstructure vs 1 before. An inverse India/EM-equity basket (polarity −1) was
  tested and **rejected** — daily signal broken by US-calendar async (r ≈ +0.05),
  redundancy 0.48, and its weekly signal is mostly Nifty beta, not INR.

### Fixed
- **Nirnay basket carried forward onto the target's calendar (cross-calendar fix).**
  The constituents often trade a different calendar than the target (US-listed
  miners vs an Indian target; or a Monday-morning IST run where global EOD bars
  haven't posted). Previously the convergence/breadth signal & plots **truncated to
  the slowest constituent** (e.g. Gold's plots stopped at Jun 18 while Gold itself
  was fresh to Jun 22), so an India-based user couldn't see the latest session. The
  basket is now reindexed onto the target's trading calendar with forward-fill — a
  closed market's last close *is* its current value — so the SIGNAL (CrossValidator),
  cards and plots all reach the target's latest session. Honesty preserved, not
  truncation: a "Breadth carried forward" notice (Convergence tab) states the
  basket's true last-native date and that those trailing bars are provisional, and
  the partial-session gate still flags the row-level staleness. (Verified: Gold
  basket Jun 18 → carried to Jun 22.)
- **Edge-case audit resolutions (verified by execution).** A rigorous pass found and
  fixed several real defects (and refuted a few suspected ones — constant-series
  div-by-zero, single-feature crash, and the MIN_DATA_POINTS pre-warmup failure were
  all proven *non*-issues):
  - **Partial-session gate.** The latest row can be a partial session (e.g. Indian
    markets posted Jun 19 but US hasn't on a publish lag) that ff-fill makes *look*
    complete — measured at **27% native-fresh** on such a day. The forecast, breadth
    and analog then rest on stale predictors. A calendar-agnostic check (native
    coverage = fraction of columns that changed vs the prior row) now flags it
    prominently ("Partial latest session — only N% of inputs posted… provisional");
    full sessions run ~97% so the 0.6 floor separates cleanly.
  - **`conv_norm_params` now target-keyed** — was cached once and reused across
    target switches, silently mis-normalizing every later target's Row-1 plot + card.
  - **Missing-target guard** — a selected target whose source fetch fails no longer
    KeyErrors the run; it fails clean with a message.
  - **Content-aware precedent cache** (keyed on latest price, not just row count) so
    an intraday refresh recomputes. **Non-positive-target guard** (returns-based engine
    needs a strictly positive series — defensive). **`render_info_box` color** is now
    applied (was a dead param). Documented two deliberate limits: the macro calendar
    is the spine (rare cross-calendar sheet dates dropped, ~4/6y), and `busday_count`
    has no holiday table (the partial-session gate is the calendar-agnostic primary).
- **Convergence cards & plot are now a single source of truth.** The "Aarambh
  Conviction" card read `aarambh_ts[...].iloc[-1]` (raw last row) while the plot Row 2
  read the Nirnay-aligned series — two sources, one label, so they could disagree.
  `render_convergence_tab` now aligns Aarambh+Nirnay **once, up front**, and both the
  metric cards and the 3-row plot read those exact arrays → a card can never drift
  from the point it mirrors (Aarambh-only targets fall back gracefully and still
  render their cards).
- **Weekend artifact rows removed from the dataset.** A few weekend-trading tickers
  (FX) created Sat/Sun index dates that the upstream `combined.ffill()` back-filled
  across the other ~180 columns — a fully ff-filled, entirely-stale weekend row that
  *looked* complete but carried Friday's values (which drove the illogical signals
  and exposed the card/plot split above). `fetch_commodity_dataset` now drops
  weekend rows so the latest row is always a real trading day.
- **Per-source, trading-day-aware data-freshness notices.** Staleness is now counted
  in TRADING days (weekends ignored — Friday data on a weekend reads *current*, not
  stale), tiered and design-consistent: a calm info note when 1–2 trading days
  behind, a prominent warning ("Latest data unavailable") once genuinely stale. Plus
  a **per-target** check: the active target can lag the macro universe (sheet behind,
  or its market shut on a holiday) with the gap forward-filled — detected exactly for
  sheet targets (from the source) and via ff-filled-tail detection for the rest — so
  the user is told when *that target's* latest signal is stale even if the dataset
  isn't. Every notice states the as-of date; signals reflect that date, not today.
- **Wired previously-dead `NIRNAY_*` config.** The nine Nirnay constants were
  referenced nowhere — the engine ran on hardcoded literals in `app.py` /
  `engines/nirnay.py`, and `NIRNAY_REGIME_SENSITIVITY = 1.0` actively *disagreed*
  with the `1.5` the engine really used. Now `core/config.py` is the single source
  of truth: `app.py` reads the constants and passes them into `run_full_analysis`
  (which gained `num_vars`/`oversold`/`overbought` params). Corrected the
  sensitivity drift (1.0 → 1.5, behaviour-preserving) and removed the no-op `±7`
  `NIRNAY_STRONG_BUY/SELL` (no code path). Output is byte-identical; the knobs are
  now honest and tunable.
- **Walk-forward label-overlap leakage purged (Aarambh).** Each forward-return label
  spans `(t, t+h]`, so training rows within `h` of the prediction point had targets
  overlapping the forecast window — future leakage that inflated out-of-sample skill,
  worse at longer horizons. `FairValueEngine.fit(…, purge=h)` now drops the last `h`
  training rows per walk-forward chunk (and keeps the ensemble-weighting validation
  slice behind the gap); the live app passes `purge=FWD_HORIZON`. Impact is large and
  measured: on Jeera the honest non-overlapping model IC at +90d fell from **+0.86 →
  +0.04** (it was almost entirely leakage), and short-horizon IC settled near the
  documented ~0.2 Val IC. The displayed **Val IC / walk-forward IC are now
  leakage-free** (lower but honest). A purged universe sweep (`precedent_vs_model_
  sweep.py`) confirmed the de-leaked model's directional edge is modest at 10/20d
  (overall IC +0.03 / −0.05), with the leakage-free **analog leading in 27–29 of 33
  targets**. `purge=0` (default) preserves legacy behaviour for non-forward modes.

---

## [2.1.0] — 2026-06-17 — *Multi-asset universe + robustness*

Minor release. Broadens the target universe well beyond commodities, retunes the
Aarambh ensemble from a study, hardens data fetching against rate limits, and
removes dead code. Backward compatible — no data-format or cache changes.

### Added
- **Equity-index targets** alongside commodities/FX: India broad & sectoral
  (Nifty 50/Next 50/100, Midcap 50, Smallcap 100, Bank, IT, Auto, FMCG, Pharma,
  Metal, Energy, Fin Services, Pvt/PSU Bank, Realty, Media, Infra, PSE, Consumption,
  Commodities, MNC, Services), US benchmarks (S&P 500, Nasdaq 100, Dow Jones), and an
  India sector-ETF universe. Each is its own Aarambh target with the index's own
  constituents as the Nirnay basket. (`INDEX_TARGETS` in `data/universe.py`.)
- **Snapshot fallbacks for S&P 500 and Nasdaq 100** — previously only Dow Jones had one,
  so the other two broke whenever the Wikipedia scrape was blocked.
- `lxml` and `html5lib` pinned in `requirements.txt` (parsers for the constituent scrape;
  `lxml` was previously only an implicit transitive dependency).

### Changed
- **Default ensemble `("ridge", "ols")` → `("ols", "huber")`.** An offline study across
  all six headline targets (Cotton/USD-INR/Nifty 50/Gold/Silver/Copper), scored by OOS
  rank-IC of the 10-day-forward forecast, found `ols+huber` best on both mean IC and
  worst-target robustness; the old `ridge+ols` ranked last (Ridge ≈ OLS on PCA-20).
- **No constituent cap by default** (`_DEFAULT_CAP` 40 → 0). Indices now use their full
  constituent set — capping a "Nifty 50" to 40 dropped real members. Re-enable by setting
  a positive cap.

### Fixed
- **Macro fetch column backfill.** yfinance rate-limits a few tickers per batch (e.g.
  `GC=F`, `BUNL.L`); the partial frame bypassed the all-or-nothing stale fallback and got
  cached, silently dropping a target column (Gold = `GC=F`) and failing the run with
  "Need 1500+ data points." Missing/all-NaN columns are now refilled from the most recent
  prior snapshot (scanning newest→oldest), re-healing the cache.

### Removed
- Dead code: `analytics/signals.py` (unused `MSFCalculator`/`MMRCalculator`, superseded by
  Nirnay's functional `calculate_msf`/`calculate_mmr`), `data/schema.py` +
  `build_unified_dataset` (uncalled Google-Sheet-era plumbing), and the unused
  `DEFAULT_PREDICTORS` / `DEFAULT_SHEET_URL` config constants.

### Docs
- **Single source of truth for the version.** `VERSION`/`PRODUCT_NAME`/`COMPANY` now live
  only in `core/config.py`; `ui/theme.py` re-exports them (ends the recurring config↔theme
  drift). Removed the duplicated `vX.Y.Z` token from every module-header docstring.
- README and docstrings corrected from "commodity-only" to the multi-asset reality, and
  the ensemble description updated to the configurable default; stale Google-Sheet /
  "Nifty 50 PE" references removed.

---

## [2.0.0] — 2026-06-11 — *Tattva — Commodity Convergence*

Major release. The system pivots from a Sheets-fed Nifty 50 PE valuation tool to
a **single-source (yfinance) commodity convergence engine** for **Gold, Silver,
and Copper**, and is renamed **Nishkarsh → Tattva** (तत्त्व, "principle/essence").
This is a breaking change: data source, target universe, model formulation, on-disk
cache/profile location, and product identity all change.

### Added
- **User-selectable commodity target** (Gold / Silver / Copper) in the sidebar; each
  keys its own calibrated intelligence profile.
- **Predictive Aarambh engine.** Forecasts the forward 10-day return from trailing
  20-day macro momentum (ex-ante), driving a directional conviction signal.
- **Causal per-window PCA** inside the walk-forward ensemble (≈20 components) — keeps
  all ~112 inputs "on" while stabilising the ensemble; fit per training window so it
  never repaints.
- **Walk-forward validation** (purged, expanding-window, re-calibrated per window) that
  runs automatically each analysis and renders in the Diagnostics tab — a durability
  grade distinguishing real edge from a lucky regime.
- **Per-commodity Nirnay baskets** of related miners/streamers (pure single names).
- Curated macro universe expansion (equities, volatility, EM/China, sector and
  real-asset ETFs) usable as predictors.

### Changed
- **Renamed the product Nishkarsh → Tattva** across UI, console, docstrings, cache dir
  (`~/.cache/tattva`), and profile version (`v1-tattva-convergence`).
- **Directional convergence score.** The cross-validator score now carries the consensus
  direction, so the hero card and DDM no longer contradict each other in sign.
- **Robustified Intelligence calibration:** purged **k-fold CV objective** + a held-out
  tail (replaces the single 70/30 split), stronger L2 — eliminates single-slice overfit.
- Aarambh now models **log-returns**, not price levels (ends the spurious levels regression).
- Diagnostics tab reordered by decision priority (edge metrics first); Feature Impact
  compacted to top-15.

### Removed
- **Google Sheets ingestion** and all sheet/secret plumbing — the system is now pure yfinance.
- Predictive-mode metrics that were misleading (R²-vs-RW, OU/Hurst on the forecast series)
  removed from the cards/console in favour of Val IC.

### Performance
- Aarambh phase ~**4× faster** (≈3.5 min → ≈50 s) via the causal PCA dimensionality cut.

---

## [1.4.0] — 2026-05-27 — *Self-Calibrating Convergence*

Headline release: **Intelligence Mode** is now the default pipeline path —
auto-calibrated profiles applied on every Run Analysis in a single flow, with
the progress bar and sidebar rewritten to expose the new behaviour clearly.

### Added
- **Intelligence-aware progress bar.** The CONVERGENCE phase now surfaces its
  sub-stages explicitly so users can see what the calibration loop is doing:
  - `83%` First-Pass Conviction Model (DDM filter · prior weights)
  - `84%` Intelligence Mode · Setup (tuner build · 70/30 split · N trials)
  - `84 → 90%` Intelligence Mode · Calibrating (live Optuna trial counter
    and best score)
  - `90%` Intelligence Mode · Profile Saved (Train IC · Val IC)
  - `91%` Applying Calibrated Profile (vectorized re-weight)
  - `92%` Re-Fitting Conviction Model (post-calibration DDM pass)
  - `93%` Detecting Divergences
  - `94%` Convergence Phase Complete (with `calibrated profile applied`
    or `factory defaults` suffix so the user can see which path ran)
  - `95%` Storing Results · `100%` Analysis Complete
  When Intelligence Mode is OFF, the `84–92%` band is skipped end-to-end.

### Changed
- **Progress bar typography** — every progress label and subtitle migrated
  to Title Case for consistency across the pipeline.
- **Sidebar rhythm tightened.** The vertical gap between consecutive setting
  groups (Data Source ↔ Model Configuration ↔ Model Passport ↔ System Spec)
  reduced from `1.5rem` / `3rem` to `0.5–0.75rem` so the right rail reads
  as a compact toolset rather than three scattered panels:
  - `.section-divider` margin: `var(--sp-6)` → `var(--sp-3)` (globally);
    `var(--sp-2)` inside `[data-testid="stSidebar"]`.
  - `.sidebar-title` margin: `var(--sp-6) 0 var(--sp-3) 0` →
    `var(--sp-3) 0 var(--sp-2) 0`.
  - Pre-`system-spec` `<hr>` margin: `3.00rem 0` → `1rem 0 0.75rem 0`.
- **Reset Analysis** button uses `use_container_width=True` — full-width like
  Export Profile and Reset to Defaults below it in the Model Passport.

### Docs
- README now reflects the Intelligence Mode pipeline as the default flow:
  new `Intelligence Mode` section explaining what gets calibrated, the
  Optuna-TPE objective, and the per-universe profile persistence path.
- Pipeline-flow diagram updated to include the **Phase 4 calibration loop**
  (first-pass → tuner → apply → re-fit).
- `What You See` table now lists the Intelligence Center and Model Passport.
- Configuration table notes the Intelligence Mode toggle and trial count.
- Module headers and version constants unified at **`v1.4.0`** across all
  Python files, `requirements.txt`, `README.md`, `LICENSE.md`, and CHANGELOG.

---

## [1.3.0] — 2026-05-26 — *Resilient Convergence*

Production-grade data layer, refactored convergence wiring, self-calibrating
**Intelligence Mode**, and full UI parity with the sibling **Pragyam** terminal.

### Added (Intelligence Mode — new in this release)
- **Self-calibrating convergence profile.** New module `convergence/intelligence.py`
  ports Sanket's Bayesian-TPE calibration pattern to Nishkarsh's convergence
  layer. An Optuna search finds the per-universe optimum for:
  - Four dimension weights (`w_direction`, `w_breadth`, `w_magnitude`, `w_regime`)
    used inside `CrossValidator.compute_convergence`, replacing the static
    `0.30 / 0.25 / 0.25 / 0.20` defaults and the ±10% adaptive shift heuristic.
  - Four asymmetric classification thresholds (`buy_strong`, `buy_moderate`,
    `sell_moderate`, `sell_strong`) used inside `convergence/normalization.py`
    `classify_normalized_signal`, replacing the symmetric ±0.3 / ±0.5 defaults.
- **Calibration objective:** maximize the Spearman Information Ratio of the
  composite convergence signal against forward NIFTY-50-PE returns at
  horizons `[3, 5, 10, 20]` trading days, with L2 regularization toward
  uniform weights to discourage overfit. Validates via a chronological
  70/30 train/val split (no shuffling — preserves causality).
- **Disk persistence** at `~/.cache/nishkarsh/intelligence/profiles.json`,
  one profile per `(universe · selected_index)` key, with versioning
  (`PROFILE_VERSION = "v1-nishkarsh-convergence"`).
- **Sidebar "Model Passport" card** ported faithfully from Sanket
  (`_render_model_passport_sidebar` in `app.py`). Shows Default / Calibrated
  / Calibrated · ⚠ profile state, Trained-on label, Train IC, Val IC, last
  updated timestamp, plus universe-mismatch warnings, an Import / Export /
  Reset control group, and an **Intelligence Mode toggle** (default ON).
- **Intelligence Center** section in the Diagnostics tab — read-only
  diagnostic dashboard surfacing Train IC, Val IC, Stability %, Trials,
  learned-weights bar chart (calibrated vs default), threshold values,
  Optuna fANOVA factor sensitivity, and a list of all saved profiles
  on disk. No calibrate button — calibration is auto-triggered by
  Run Analysis (see below).
- **Single-flow auto-calibration.** The `CONVERGENCE` phase of every
  Run Analysis now runs the full calibration loop end-to-end with no
  manual user input:
  1. First-pass `CrossValidator` with the **prior profile** (loaded from
     disk for this universe, or factory defaults if none exists).
  2. Initial `UnifiedConvictionModel` fit to populate the convergence
     time-series (which the calibrator needs as its input).
  3. **Auto-calibration** — `ConvergenceTuner` runs Optuna TPE on the
     fresh `convergence_df` + `aarambh_ts`, with live progress on the
     pipeline progress bar (85% → 90%) and per-trial console output.
  4. **Apply in same run** — `intelligence.apply_calibrated_weights()`
     does vectorized recomputation of `convergence_score` and
     `convergence_zone` from the existing `dim_*` columns and the
     newly-learned weights (no need to re-loop CrossValidator).
  5. **Conviction model re-fit** on the recomputed scores, so the
     final DDM-filtered conviction reflects the calibrated state.
  6. **Normalized convergence** classified with the calibrated
     asymmetric thresholds.
  7. **Profile persisted** to disk so the next run starts from this
     calibration (warm path), and the **Passport sidebar updates** on
     the post-analysis rerun to show the freshly-saved profile.

  When the Intelligence Mode toggle is OFF, steps 3–5 are skipped —
  the system runs on factory defaults end-to-end (this is also the
  fall-back if calibration raises an exception).

### Dependencies
- Added `optuna>=3.5.0` to `requirements.txt` for the Bayesian TPE search.

### Added
- **Smart data layer.** Three production-grade primitives now sit between the
  app and every external API:
  - **Circuit breaker** (`data/circuit_breaker.py`) — `CLOSED → OPEN → HALF_OPEN`
    state machine per service, thread-safe, with two module-level breakers
    (`yfinance_circuit`, `sheets_circuit`) and a shared `all_circuits()` helper.
  - **Retry-with-backoff** — exponential decorator (1s → 2s → 4s, capped at 60s,
    max 3 retries), `@yfinance_circuit.protect` / `@RetryWithBackoff` stack.
  - **Two-tier cache** (`data/cache.py`, revived from dead code) — memory + disk,
    TTL expiry, **versioned keys** (`version="v1"` bump invalidates a namespace
    atomically), `get_stale()` last-good-snapshot fallback used automatically
    when a fetch fails *and* the circuit is open, full `stats()` snapshot
    (hits / misses / stale_hits / writes / hit_rate / namespace / TTL).
- **Data Layer Health diagnostics** — new section in the Diagnostics tab
  showing per-namespace cache hit rate, disk entry count, last-fetch
  timestamp, and per-service circuit-breaker state (CLOSED / HALF-OPEN / OPEN).
- **Global Macro bond-ETF universe.** Ported from the sibling **Sanket**
  project — 66 yfinance-available bond ETF tickers covering US Treasuries
  (full curve + raw yields ^IRX / ^FVX / ^TNX / ^TYX), TIPS, aggregate
  bonds, corporate IG / HY, mortgage-backed, municipals, developed-markets
  sovereign (Europe + Asia-Pacific), India fixed income, and emerging
  markets. Replaces the broken Stooq endpoints.
- **Shared normalisation module** (`convergence/normalization.py`) — single
  source of truth for the math behind the Convergence Analysis cards *and*
  the Unified Signal plot. Five small pure functions:
  `align_aarambh_nirnay`, `compute_norm_params`, `zscore_clip`,
  `classify_normalized_signal`, `compute_normalized_convergence`.
- **Pragyam-style UI uplift.** Full port of the *Obsidian Quant Terminal*
  design system from Pragyam:
  - `ui/theme.css` expanded from 47KB to 114KB — adds backdrop-filter glass,
    SVG noise + grid underlays, 11+ entrance / shimmer / gradient-shift
    keyframes, premium springy easing `cubic-bezier(0.16, 1, 0.3, 1)`,
    expanded palette (`--orange`, `--slate-warm`, `--card-base` DRY tokens).
  - `ui/components.py` expanded from 18KB to 27KB — new helpers
    `get_icon(name, size, stroke_width)` (dynamic-sized SVG), `get_signal_badge`
    (5-tier conviction badge), `render_conviction_signal`, `render_system_card`,
    `render_kv_table`. Icon library grew from 18 → 34.
  - Signal card design language unified with metric cards: corner accent dot
    + bottom gradient sweep + tinted background gradients per variant.
  - **Equal-height metric cards** — replaced the brittle `height: 100%` cascade
    with a `flex: 1 1 auto` propagation that's robust against Streamlit DOM
    changes (e.g. inserted `stVerticalBlock` wrappers).
- **System info card** in the sidebar — adopts Pragyam's `system-spec` markup
  with `.spec-row` / `.spec-label` / `.spec-value` flex layout (replaces the
  earlier `info-box` paragraph version).

### Changed
- **Convergence Analysis cards re-wired to the Unified Signal plot.** The four
  metric cards now show *exactly* what the Unified Signal plot rows display:
  - **NISHKARSH CONVICTION** ← normalized convergence (`norm_avg[-1]` in
    `[−1, +1]`), formatted `+0.42`. Signal classification re-thresholded for
    the new scale (`±0.3` moderate, `±0.5` strong).
  - **AARAMBH CONVICTION** ← `aarambh_ts["ConvictionRaw"]` (was
    `ConvictionBounded`), formatted `+0.42`.
  - **NIRNAY AVG SIGNAL** ← unchanged source, format upgraded from 1 to
    2 decimal places.
  - **AGREEMENT** ← unchanged.
- **Hero signal card** (`_render_primary_signal` in `app.py`) now reads the
  normalized value too. Interpretation paragraph rewritten to surface
  Aarambh and Nirnay contributions independently.
- **`render_nishkarsh_signal_card`** conviction format changed from `:+.0f`
  to `:+.2f` to render the new `[−1, +1]` scale meaningfully.
- **Sidebar masthead** typography tightened to match Pragyam exactly
  (`1.35rem` brand size, `0.04em` letter-spacing, `<hr>` divider instead of
  `.section-divider`).

### Removed
- **Stooq direct-yield endpoints.** Stooq started returning HTML error pages
  instead of CSV in late 2025, causing a cascade of `ParserError` retry
  loops. Replaced wholesale with the Global Macro yfinance universe.
- **`MACRO_SYMBOLS_STOOQ`** and dead **`MACRO_COLUMN_MAP`** removed from
  `core/config.py`.
- **`stooq_circuit`** breaker removed — no longer reachable.

### Fixed
- **DataFrame fragmentation `PerformanceWarning`** in `engines/nirnay.py` —
  two batches of column-by-column assignments (12 + 6 columns) collapsed into
  single `df.assign(...)` calls. `Series.shift(1)` replaced with
  `np.concatenate(([nan], arr[:-1]))` to keep behaviour byte-identical
  (verified on a 5-element test sequence).
- **Metric tooltip ring/dot misalignment** — the `::before` glow dot and the
  `.metric-tooltip` help-circle were both anchored at `top: var(--sp-3); right:
  var(--sp-3)`, but their centres differed by 3px each axis. Tooltip now has
  explicit `width/height: 12px` and adjusted position so the ring is
  concentric with the dot.
- **Header text alignment** inside metric cards — the equal-height flex rule
  was inadvertently forcing `flex-direction: column` on `<h4>`, which combined
  with the inherited `align-items: center` to horizontally centre the label.
  Narrowed the flex propagation selector to `div`-only descendants.
- **Sanskrit name on the landing page** — `.premium-header h1::after` content
  was still "प्रज्ञम" (left over from the Pragyam CSS port); corrected to
  "निष्कर्ष".

### Performance
- **Macro fetch concurrency** is now driven by yfinance's internal `threads=True`
  pool — one batch call for all 84 unique macro tickers (66 Global Macro +
  18 commodities/FX), one circuit hit, no manual `ThreadPoolExecutor` loop.

### Docs
- Module headers and version constants unified at **`v1.3.0`** across all
  35 Python files, `requirements.txt`, `README.md`, `LICENSE.md`, and the
  CHANGELOG.
- `data/cache.py`, `data/circuit_breaker.py`, and `convergence/normalization.py`
  carry full module-level docstrings explaining the lifecycle, state machine,
  and pipeline respectively.

---

## [1.2.0] — 2026-04-13 — *Obsidian Quant Terminal*

Full standardisation pass: module headers, documentation, and system integrity fixes.

### Added
- **Standardised module headers** across all 33 Python files. Every module now carries
  the `Nishkarsh v1.2.0` header with the निष्कर्ष tagline and a system-level description
  (AARAMBH, NIRNAY, CONVERGENCE, DATA, ANALYTICS, UI, CORE).
- **Structural break fallback** — `BaiPerronTest` is only in unreleased statsmodels 0.15+.
  A rolling-mean change-point heuristic is now used as a fallback when the native
  implementation is unavailable.

### Changed
- **UI aesthetic** rewritten as *"Obsidian Quant Terminal"* design language:
  - **Typography:** Syne (geometric, authoritative) for display, JetBrains Mono for data.
  - **Palette:** Obsidian (#0A0E17 → #050810), Amber Gold (#D4A853),
    Cyan (#22D3EE), Emerald (#34D399), Rose (#FB7185).
  - **Surfaces:** Frameless glass panels with thin border strokes.
  - **Plotly defaults** centralised in `ui/theme.py` — single source of truth for
    font, grid, hover, legend, and margin config across all tab renderers.
- **`VERSION` unified** — `ui/theme.py` VERSION corrected from `3.1.0` to `1.2.0`
  to match `core/config.py`. All module headers now reference `v1.2.0`.
- **`README.md`** rewritten from scratch with Obsidian Quant Terminal branding,
  full architecture diagram, pipeline flow, and troubleshooting section.
- **`CHANGELOG.md`** restructured — removed legacy Aarambh-only version entries
  that predate the Nishkarsh unification.

### Fixed
- **statsmodels compatibility** — `BaiPerronTest` import no longer crashes on
  statsmodels 0.14.x. Graceful fallback to heuristic detection.
- **VERSION consistency** — `core/config.py` (`1.2.0`) and `ui/theme.py` (`3.1.0`)
  now both report `1.2.0`.

---

## [1.1.0] — 2026-04-07 — *Nishkarsh Production Release*

The first release under the **Nishkarsh** name. Replaces the prior
"Samyoga" branding and ships the unified two-system convergence engine
end-to-end.

### Added
- **Unified Convergence Engine.** Two orthogonal systems (Aarambh top-down +
  Nirnay bottom-up) merged into a single convergence pipeline with adaptive
  4-dimension scoring (Direction 30% · Breadth 25% · Magnitude 25% · Regime 20%).
- **Cross-system divergence detection.** Three event types:
  - `AARAMBH_LEADS` — valuation extreme, constituents lagging (early warning).
  - `NIRNAY_LEADS` — momentum-first move (breadth turning before valuation).
  - `CONTRADICTION` — persistent disagreement (uncertain regime).
- **Terminal logging system** — direct console output with timed phases,
  per-constituent analysis logs, and formatted run summaries.
- **Progressive UI** — animated pulse-dot progress cards with gradient bar
  and real-time status during pipeline execution.
- **Nifty 50 constituent analysis.** All 50 symbols processed through
  MSF + MMR + the four-method regime ensemble (Adaptive Kalman / 3-state HMM /
  GARCH-like / CUSUM), aggregated to daily breadth statistics.
- **Macro data integration.** 18 yfinance commodities/FX symbols + 6 bond
  yields from Google Sheets = 24 macro indicators feeding the MMR regression.
- **Timeframe filtering.** 3M / 6M / 1Y / 2Y / ALL buttons with synchronised
  x-axis zoom across all convergence charts.
- **Hover templates.** All Plotly charts show date + value tooltips.

### Changed
- **System renamed** from *Samyoga* to **Nishkarsh** (निष्कर्ष — *"Conclusion"*).
  Complete branding update across every file.
- **Default timeframe** is now 6M (was 1Y); 3M added (replacing 1M).
- **Sigmoid formula** corrected to the original Nirnay form
  `2 / (1 + exp(−x / scale)) − 1` (was `2 / (1 + exp(−scale·x)) − 1`).
- **Market markers** on the unified signal plot now use conviction-plot-style
  markers (size 7/8/10 by signal strength, colour-coded green/red/grey).
- **Nirnay tab** completely rewritten to match the original Nirnay charts:
  oversold/overbought distribution, raw counts, buy/sell signal counts,
  HMM regime probabilities.

### Fixed
- **Stooq bond yields fallback.** Integrated Google Sheets bond yields
  (IN10Y / IN02Y / IN30Y / US10Y / US30Y / US02Y) when Stooq blocks
  automated access.
- **Column-name normalisation.** All Nirnay aggregation outputs use exact
  original column names (`Oversold_Pct`, `Overbought_Pct`, `Buy_Signals`, …).
- **Convergence date alignment.** Proper inner-join between Aarambh
  calendar dates and Nirnay trading dates.
- **Warning suppression.** Silenced yfinance, urllib3, pandas, and numpy
  warnings for clean terminal output.

### Removed
- All `Samyoga` branding references (sidebar, headers, signal cards,
  console output, metric cards).
- Progress-bar jargon — replaced with clean progress cards.
- Dead code paths from multi-phase development iterations.

---

## [1.0.0] — 2026-04-05 — *Initial Nishkarsh Release*

The first unified Nishkarsh release, inheriting the Aarambh-only lineage.

### Added
- **Aarambh FairValueEngine** — walk-forward ensemble regression (Ridge / Huber /
  OLS / ElasticNet / PCA-WLS) on Nifty 50 PE ratio with conformal prediction
  intervals and DDM smoothing.
- **Nirnay constituent engine** — per-stock MSF + MMR with four-method regime
  ensemble (Adaptive Kalman / 3-state HMM / GARCH-like / CUSUM).
- **Convergence cross-validator** — 4-dimension adaptive scoring with
  Drift-Diffusion filtering and divergence classification.
- **Streamlit UI** — five-tab interface with Obsidian Quant Terminal styling,
  timeframe filtering, and CSV export.
- **Google Sheets + yfinance data pipeline** — unified fetcher with TTL caching.
- **Nifty 50 live constituent fetching** from niftyindices.com with Wikipedia fallback.

### Inherited from Aarambh lineage (pre-unification)

The following mathematical foundations were hardened during the Aarambh-only
development cycle (versions 2.0.0–3.2.2) and carried forward:

- **True conformal quantiles** — empirical `compute_conformal_zscores` replacing
  pseudo-Gaussian z-scores.
- **Bai-Perron regime binding** — expanding windows bound to the most recent
  structural break.
- **DDM variance capping** — geometric variance scaling to prevent ballooning
  standard errors during prolonged regimes.
- **Andrews (1993) median-unbiased AR(1)** — jackknife correction for near-unit-root.
- **DFA Hurst exponent** — Peng et al. (1994) replacing biased R/S estimator.
- **Look-ahead bias elimination** — rolling mean/std with `shift(1)`.
- **Conviction soft bounds** — `tanh` transformation to `[−100, +100]`.
- **Thread-safe walk-forward** — lock-protected sequential execution.

---

© 2026 Nishkarsh · [@thebullishvalue](https://twitter.com/thebullishvalue)
