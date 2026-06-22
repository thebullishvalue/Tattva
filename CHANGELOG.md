# Changelog

All notable changes to **Tattva** (formerly **Nishkarsh**) are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Sections used: **Added · Changed · Deprecated · Removed · Fixed · Security · Performance · Docs**.

---

## [Unreleased]

### Added
- **Per-exchange trading calendars — holiday-aware data freshness (Phase 1).** New
  `data/calendars.py` resolves each target ticker to its exchange (`.NS/.BO`→XBOM,
  `=F`→CMES, `=X`→FX weekday, `^`→XNYS/XBOM, sheet/NCDEX sentinels→XBOM, bare→XNYS)
  and counts "trading days behind" on that exchange's actual session calendar via the
  `exchange_calendars` library. This retires the ~1-day over-count the old Mon–Fri
  `busday_count` produced across market holidays (Diwali, Thanksgiving, Juneteenth,
  15 Aug…) — the freshness notices no longer flash a premature "behind" across a
  closed session. Wired into both the dataset and per-target freshness notices in
  `app.py`. The partial-session gate remains the calendar-agnostic *primary* signal.
  - **Graceful, optional dependency.** The import is guarded: if `exchange_calendars`
    is absent (or a calendar can't be built / a date is out of bounds), every count
    degrades to the *exact* legacy `busday_count` — the app never breaks and is never
    worse than before, only better when the lib is present. Added to `requirements.txt`.
  - Verified end-to-end by `research/test_calendars.py`: all 35 targets resolve to a
    known exchange; holiday-aware counts are strictly lower than the naive mask across
    real US/India holidays and equal on plain weeks; FX equals the weekday mask; the
    forced no-library path reproduces `busday_count` byte-for-byte.

### Fixed
- **Aarambh tab — conviction card vs DDM plot now agree (single source of truth).**
  The DDM-Filtered Conviction chart drew the *unbounded* `ConvictionScore` line/fills
  while its own confidence bands and the interpretation card used the *bounded*
  series (`signal["conviction_score"]` = `ConvictionBounded`) — a three-way mismatch
  on a `[-100,100]` axis, visible at extreme conviction. The chart line/fills now read
  `ConvictionBounded`, so the plot's last point equals the card value and matches the
  bands and axis. (The `get_signal_performance` `±CONVICTION_MODERATE` classifier sits
  far below the clip bound, so its labels are unchanged.)
- **Data-freshness off-by-one across timezones.** "Trading days behind" anchored
  `today` to UTC while the data date is exchange-local, so a UTC-hosted deploy rolling
  past midnight over-counted a current bar as "1 day behind". Now anchored to the
  earlier of UTC and machine-local (brackets the realistic tz band, never *overstates*
  staleness) and `behind`/`t_behind` are clamped `≥ 0` to guard tz-ahead/future-dated
  bars. Genuine staleness and the exact partial-session gate are unaffected.
- **Convergence — short-history degeneracy.** When the full Aarambh∩Nirnay overlap is
  tiny (new sheet target / freshly-listed constituents), the z-score σ collapsed to its
  `1e-10` floor and the normalized plot flat-lined at zero, misreading as a confident
  "neutral". A guard now suppresses the plot below 10 shared sessions with a calm
  "building convergence history" note; the metric cards still show the raw latest reads.

### Tested
- **Inverse-basket polarity path** (`engines.nirnay.apply_polarity`, `TARGET_POLARITY = -1`)
  is execution-verified for the first time via `research/test_polarity.py` — full
  27-column schema: directional-pair swaps, signed-oscillator negation, neutral/count
  columns untouched, no-op for `+1/0/None`, involution (double-flip = identity), and
  breadth-% conservation. No live target sets `-1` yet, so the path previously shipped
  unexercised.

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
