# Tattva — Tuning Coverage Map

**Purpose**: every tunable constant in the system, classified by how its value is
justified. The goal state: no constant whose value is *mental assertion* — each is
either **study-tuned** (a suite study sweeps it), **data-anchored** (set to a measured
percentile of its own live distribution), **structural** (its value is a definition,
constraint, or product identity — tuning is not meaningful), or **budget** (a compute
budget, not a signal parameter). Anything **pending** is an open gap.

Re-generate the evidence: `python research/run_tuning.py` (full suite) — study keys
below refer to that suite. Last full audit: **2026-07-13** (full suite re-run
from scratch on widened 10→3000 grids, `reports/tuning_20260713_004005.txt`).

**POLICY CHANGE (2026-07-13, per user directive): config now tracks the
report's recommendation outputs LITERALLY.** Every constant with an explicit
RECOMMENDED / best / anchor line in the latest report carries that value —
sub-SE margins are no longer a reason to hold the previous value. Where two
result tables cover the same constant from different universes (Nirnay
MSF_LENGTH), the value is the universe-share-weighted winner of the tables
themselves. Constants whose study printed no explicit recommendation (analog
knobs, markers already on-anchor, UI tiers measured on-target) are unchanged.
Consequence to watch: several adopted values sit on noise-level margins by the
reports' own spreads; the next re-run may recommend different values again —
that is expected under this policy, not drift.

## Engine — Aarambh (core/config.py)

| Constant | Status | Evidence / study key |
|---|---|---|
| MIN_TRAIN_SIZE, MAX_TRAIN_SIZE | study-tuned | `aarambh_full`, `aarambh_maxmin` (deep grid 2026-07-12: flat ≥625; short-window hump sub-SE, not adopted) |
| REFIT_INTERVAL | study-tuned | `aarambh_full` (13 values, all within noise) |
| ENSEMBLE_MODELS | study-tuned | `aarambh_full` (12 baskets, none separates; ols+huber kept for robustness) |
| RIDGE_ALPHAS | study-tuned | `aarambh_full` (8 grids, insensitive) |
| HUBER_EPSILON | study-tuned | `aarambh_full` HUBER_EPSILON lever (added 2026-07-12) |
| LOOKBACK_WINDOWS | study-tuned | `aarambh_full` LOOKBACK_WINDOWS lever (added 2026-07-12) |
| HUBER_MAX_ITER | budget | solver iteration cap — convergence budget, not a signal knob |
| PCA components (app.py literal 20) | study-tuned | `aarambh_full` PCA lever (14 values, 20 within noise of best) |
| CONVICTION_WEAK / MODERATE / STRONG (9/17/27) | data-anchored | `ui_anchors` 2026-07-12: re-anchored 20/40/60 → p50/p75/p90 of \|ConvictionBounded\| (old values sat at p85/p97/p100 — STRONG printed on 0.5% of days) |
| DDM_LEAK_RATE / DRIFT_SCALE / LONG_RUN_VAR | study-tuned | `ddm` (leak sweep at constant gain; F3 invariant held) |
| OU_PROJECTION_DAYS | structural | display projection length (chart window, not a fit) |
| MIN_DATA_POINTS | structural | walk-forward feasibility floor (≥ train window + purge + OOS span) |
| LOOKBACK momentum ≈ 2×horizon convention | structural | per-lens `momentum` in SIGNAL_HORIZONS; horizon scope validated by `precedent_univ` / `precedent_model` |

## Engine — Nirnay (core/config.py)

| Constant | Status | Evidence / study key |
|---|---|---|
| NIRNAY_MSF_LENGTH | study-tuned | `nirnay` + `nirnay_index` (cross-universe guard: 20 stands, commodity winner rejected) |
| NIRNAY_ROC_LEN, REGIME_SENSITIVITY, BASE_WEIGHT, MMR_NUM_VARS | study-tuned | `nirnay` (densified OFAT; sensitivity/num_vars measured INERT) |
| NIRNAY_OVERSOLD / OVERBOUGHT (±5) | data-anchored | `ui_anchors` 2026-07-12: ±5 = p81 of 152k pooled per-instrument obs — validated, kept |

## Convergence layer (core/config.py, convergence/)

| Constant | Status | Evidence / study key |
|---|---|---|
| CONV_WEIGHT_DIRECTION/BREADTH/MAGNITUDE/REGIME | study-tuned | `conv_weights` (unfitted vector sweep on live frames) |
| DEFAULT_THRESHOLDS (consensus ±0.26/±0.39) | study-tuned | `hero_thresholds` (no separation winner → p75/p90 occupancy anchor) |
| COMPOSITE_THRESHOLDS (±0.11/±0.18) | study-tuned | `hero_thresholds` (8-target p75/p90 re-validation) |
| Headline construction (consensus vs calibrated) | study-tuned | `calibration_lift` (3-arm paired: consensus +0.039 ≥ cal +0.022; cal−raw = 0.000) |
| CONV_DDM_LEAK/DRIFT/LRV | study-tuned | `ddm` |
| CONV_STRONG/MODERATE/WEAK_BULLISH/BEARISH (±27/18/11) | data-anchored | `ui_anchors` 2026-07-12: re-anchored ±60/30/10 → p97.5/p90/p75 of \|convergence_score\| (old MODERATE=p98, STRONG unreachable — max observed ≈35); aligned with COMPOSITE_THRESHOLDS×100 |
| conviction_model tiers (6/11/18, was CONVICTION_* 20/40/60) | data-anchored | F1 fix 2026-07-12: the model bins the smoothed COMPOSITE, now on the composite's own anchors (COMPOSITE_THRESHOLDS×100 + measured p50); sharing the engine's tiers put NEUTRAL on ~96% of days |
| CONV_ADAPTIVE_SHIFT_MAX (0.10) | pending | pipeline-embedded (weights shift during frame BUILDING); sweeping requires per-value frame rebuilds — documented gap, low priority (calibration_lift showed the weight layer is flat) |
| INTEL_N_TRIALS, L2 α, CV folds | budget | Optuna budget/regularization of a layer measured to add zero OOS lift (`calibration_lift`) — tuning its budget cannot matter until the layer itself earns lift |
| DIV_LOOKBACK / DIV_PERSISTENCE_THRESHOLD | structural | event-window definitions (what "recent"/"persistent" mean); hero RISK row windows by DIV_LOOKBACK — vocabulary, not an edge claim |
| SIGNAL_HORIZONS (10d/20d lenses, hold grids) | study-tuned | `precedent_univ` + `precedent_model` (model carries 1–10d; 20d = turnover lens, documented as no-edge-claim) |

## Analog / Precedent (analytics/analogs.py)

| Constant | Status | Evidence / study key |
|---|---|---|
| ANALOG_W_MAHA/TRAJ/RECV (1/0/0) | study-tuned | `analog` + `analog_confirm` (22 blends; nothing beats noise) |
| TOP_N (10), recency half-life, feature set, aggregation | study-tuned | `analog` (incl. pairwise drops, 4 aggregation modes) |
| Theiler exclusion window max(tw,h) | structural | statistical independence requirement (audit A5), not tunable — smaller reintroduces overlap double-counting |
| PRECEDENT_HONORARY_HORIZON (+1d tile) | study-tuned | `analog` (1d IC ≈ 0 → display-only, caveated) |

## Hero card interpretation (ui/components.py)

| Constant | Status | Evidence / study key |
|---|---|---|
| Classification cut-points (via DEFAULT_THRESHOLDS) | study-tuned | `hero_thresholds` |
| _FLAT_BAND (0.05) | data-anchored | ≈p42 of pooled |composite| (2026-07-11 measurement, comment at constant) |
| _PRECEDENT_MIN_N (5), _PRECEDENT_SPLIT_BAND (15pp) | structural | statistical floors (min sample for a base rate; band inside which n≥5 hit-rates are coin-flip-indistinguishable) — derived from counting, not tuned |
| _ACTION_WEIGHTS (decision synthesis) | structural | ordinal evidence-weighting policy, documented per-weight at the constant; pinned by decision-table tests. A fitted version would be a calibration layer — the class of layer `calibration_lift` measured to add nothing |
| Headline A/B/C construction | study-tuned | `hero` (consensus-only beats +markers/+precedent) |

## Display tiers (core/config.py UI_*)

| Constant | Status | Evidence / study key |
|---|---|---|
| UI_CONSENSUS_STRONG/MODERATE | data-anchored | `markers` (p90/p75, exact) |
| UI_CONVRAW_STRONG/MODERATE | data-anchored | `markers` (p90/p75, exact) |
| UI_NIRNAY_AVG_THRESHOLD | data-anchored | `markers` (re-anchored 2.5→2.9 = p75, 2026-07-12) |
| UI_AGREEMENT_STRONG/MODERATE (0.91/0.82) | data-anchored | `ui_anchors` 2026-07-12: re-anchored 0.7/0.5 → p90/p75 (old STRONG = p50 — half of all days "strong") |
| UI_MODEL_SPREAD_LOW/HIGH (20/35 bps) | data-anchored | live ols+huber basket measurement 2026-07-12 (p77/p90); the ridge+ols research basket is ~2× tighter and must not anchor these |
| UI_BAND_NARROW/WIDE | **removed** (2026-07-12) | `ui_anchors` measured the band width degenerate (pinned by DDM long-run variance, p1–p99 = 40.2–42.5) — the tier sentence and constants were deleted; the CI band remains on the conviction chart, and the study keeps an informational distribution row to catch a future regime change |
| UI_BREADTH_HIGH (60) | data-anchored | `ui_anchors` 2026-07-12: ≈p96 (validated as an alert tier) |
| UI_NIRNAY_BULLISH/BEARISH (±2.9) | data-anchored | `ui_anchors` 2026-07-12: re-anchored ±2 (p58) → p75, matching UI_NIRNAY_AVG_THRESHOLD |
| UI_R2_STRONG/ACCEPTABLE (0.7/0.4) | structural | R² quality conventions (display copy) |
| UI_ADF_SIGNIFICANT / UI_KPSS (0.05) | structural | statistical significance convention |
| UI_HMM_CONFIDENT (0.5) | structural | majority-probability definition |
| Chart/table heights, colors | structural | design system |

## Data / ops (not signal parameters)

STALENESS_DAYS, SESSION_FRESH_FLOOR, TIMEFRAME_TRADING_DAYS, RAW_YIELD_PREDICTORS,
GLOBAL_MACRO_MAP, MACRO_SYMBOLS_YF, baskets/targets/polarity/archetype/excluded-predictor
maps — all **structural** (calendar, data semantics, universe definition).

## Open gaps

1. **CONV_ADAPTIVE_SHIFT_MAX** — needs a frame-rebuild sweep (8 targets × ~4 values,
   ~30 min). Low priority: the whole weight layer is measured flat.
2. **Optuna search-space bounds** (intelligence.py trial ranges) — budget-class; only
   worth touching if calibration ever shows lift.

**Maintenance rule**: a new constant enters the codebase either with a study key in
this table or with an explicit `structural`/`budget` classification in its comment.
"Looks reasonable" is not a value.
