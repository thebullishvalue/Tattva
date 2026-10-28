# ARCHITECTURE — Tattva as one machine

Tattva 2.9.0. This document says what the shipped system is: the one thing it
computes, the role every part plays in computing it, and the tests that fail
the build when a part drifts. The measured numbers — what the catalogue study
found, what the app says today — are in [README.md](README.md), generated from
the committed study; nothing here is a measurement.

---

## The objective

For each instrument, at Mūla's latest published session d*, Tattva issues one
signed call

    w_t = d_t · size_t        on   Y_t = log P(t+L+h) − log P(t+L)

- **h** is `InstrumentConfig.forecast_horizon` (10 sessions for every catalogue
  target), resolved once and passed explicitly.
- **L** is `KNOBS['EXECUTION_LAG']` = 1: a call published after the close of t
  first trades at the close of t+1.
- **t** runs on the target's own printed-session spine, from `ANALYSIS_EPOCH`.

The parts of the call:

- **Direction.** d_t = `core.semantics.claim_direction(FVO_t)` when every
  precondition holds, else 0. It is the only source of sign anywhere.
- **Size.** size_t = trust_t · ∏_i exp(β_i,t · z_i,t), a unitless index (1.0 is a
  fully trusted claim with neutral sizers), never a percentage.
  - trust_t = f(t_post of the claim, as of t) — the claim's earned base size;
  - β_i,t = SIZER_BETA_MAX · f(t_post of sizer i, as of t) · SIZERS_ENABLED.
- **The one map.** f(t) = clip((t − STRENGTH_T0)/(STRENGTH_T1 − STRENGTH_T0), 0, 1),
  with f(undefined) = 0. Trust and every β go through it.
- **Posterior t.** Each t_post is the GLS combination of the object's own causal
  as-of grade (`core.grade`) and its leave-one-out asset-class prior from the
  committed pooled-prior study (`core.trust`).
- **Graded as issued.** The call is composed at every past session
  (`call_ts`); `edges['call']` grades w on the sessions it was issued, by the
  same grader, and that grade heads Overview.

Every module is one of: an input to w, a grade of w or of an input, evidence
tagged to one of those, or infrastructure that guarantees one as-of row, one
horizon and causality. Anything else was deleted (see
[Admission and re-entry](#admission-and-re-entry-rules)).

---

## Roles

`core/roles.py` declares every role once. The kinds:

| Kind | Holder | Invariant |
|---|---|---|
| CALL | the call and its size (`core/call.py`) | one object, `verdict_from_row(call_ts at d*)`, read by every page |
| CLAIM | Mūla's FVO → d | the only sign source |
| PRECONDITION | P1 published · P2 target printed and fewer than h sessions behind · P3 tradeable · P4 \|FVO\| ≥ CLAIM_SIGMA | voids the claim (NO CALL / NO CLAIM), never sizes it; one definition each |
| SIZER | Swayam confirmation (z = d·Lean on a native Swayam row, else 0) · Reversion, ECM (z = 2·MRProb − 1, MRProb = P(price-side κ̂ > 0): the evidence that price responds to the gap at all, not how much of the closing it does — that is the gap split) | β = 0 while unskilled; an absent or carried reading is ×1.00; a factor is positive and unsigned, so it never vetoes or inverts |
| GRADE | `core.grade` (series_edge, expanding_edge) and `core.trust` (posterior, strength); trust is the claim's grade mapped once | the only skill numbers shown or consumed |
| EVIDENCE | the gap-closure split, the claim's unit, the claim by side, gap stationarity, fair value, model fit, drivers, Swayam's standalone lean, the view bank, the ECM's gap-implied move | never moves the call; neutral tone; a lean carries its own grade |
| INFRASTRUCTURE | fetch from the epoch, `data/prep.py`, `data/calendars.py`, the completeness gate, caches, the registries, the record | one as-of row; causal; holds no trust constant |

No learner ships on the decision path: Mūla's internal model averaging and
error-correction pool are graded through what they produce (the claim and the
reversion sizer), and Swayam's views count equally.

Every registered role (`ROLES`):

| Role | Kind | Serves | Title |
|---|---|---|---|
| `call` | CALL | — | The call |
| `claim` | CLAIM | `call` | Mūla claim |
| `precondition` | PRECONDITION | `claim` | Preconditions |
| `precondition:published` | PRECONDITION | `claim` | Published |
| `precondition:printed` | PRECONDITION | `claim` | Target printed |
| `precondition:tradeable` | PRECONDITION | `claim` | Tradeable |
| `precondition:claim_sigma` | PRECONDITION | `claim` | Outside the noise |
| `trust` | GRADE | `claim` | Trust |
| `sizer:swayam` | SIZER | `call` | Swayam confirmation |
| `sizer:reversion` | SIZER | `call` | Reversion (ECM) |
| `size` | CALL | `call` | Size |
| `grade:call` | GRADE | `call` | Call grade (as issued) |
| `grade:call:prospective` | GRADE | `call` | Call grade (prospective) |
| `grade:claim` | GRADE | `claim` | Claim grade |
| `grade:sizer:swayam` | GRADE | `sizer:swayam` | Swayam sizer grade |
| `grade:sizer:reversion` | GRADE | `sizer:reversion` | Reversion sizer grade |
| `evidence:gap_split` | EVIDENCE | `claim` | Who closes the gap? |
| `evidence:claim_unit` | EVIDENCE | `claim` | The claim's unit |
| `evidence:gap_stationarity` | EVIDENCE | `claim` | Gap stationarity |
| `evidence:claim_side` | EVIDENCE | `claim` | The claim by side |
| `evidence:fair_value` | EVIDENCE | `claim` | Price and fair value |
| `evidence:model_fit` | EVIDENCE | `claim` | Model fit |
| `evidence:drivers` | EVIDENCE | `claim` | What drives fair value |
| `evidence:swayam_lean` | EVIDENCE | `sizer:swayam` | Swayam's lean, standalone |
| `evidence:view_bank` | EVIDENCE | `sizer:swayam` | The view bank |
| `evidence:ecm_drift` | EVIDENCE | `sizer:reversion` | Gap-implied move |
| `grades` | GRADE | `call` | Every grade |
| `infra:asof` | INFRASTRUCTURE | — | As of d* |
| `infra:data` | INFRASTRUCTURE | — | The data |
| `infra:priors` | INFRASTRUCTURE | `trust` | Pooled priors |
| `infra:registry` | INFRASTRUCTURE | — | The registries |
| `infra:record` | INFRASTRUCTURE | — | The record |

`CARDS` registers every card a page draws — each KPI item, section and Data
column — to (role, target); its chip is fixed text per kind (`CHIPS`), so an
EVIDENCE card always reads 'EVIDENCE→<target> · does not move the call'.
`ui.components` refuses to draw an unregistered card. `CHAIN` is the decision
order of the verdict's chain rows on every page: claim, precondition, trust,
sizer:swayam, sizer:reversion, size.

---

## Data flow

```
INFRASTRUCTURE  cfg = get_instrument_config(target); inst = core.instruments.instrument(target)
                panel = pipeline.run.fetch_panel(target, epoch=ANALYSIS_EPOCH)   (+ join_target)
                bars  = pipeline.run.target_bars(target)        target OHLCV on its own sessions
                prep  = pipeline.run.prepare(panel, target)     data.prep.prepare_panel: the target-
                                                                printed spine, the registered cross-section
CLAIM           engine = pipeline.run.fit_mula(prep, cfg)  →  fvo_ts, d* = engine.asof
SIZER INPUT     swayam_daily = pipeline.run.build_swayam(...)   equal-weight breadth, Lean, _Native, ≤ d*
COMPOSE+GRADE   pipeline.assemble.assemble(fvo_ts, swayam_daily, instrument, cfg, asof=d*, priors):
                  inputs    = gather_inputs(...)           every value read by date
                  grades_ts = grade_series(...)            core.grade.expanding_edge ⊕ core.trust.PriorTable
                                                           → posterior → strength, per object
                  call_ts   = core.call.compose_series(inputs, grades_ts, tradeable, SIZERS_ENABLED)
                  evidence  = pipeline.evidence.build_evidence(fvo_ts, h, L)
                  edges     = grade_all(call_ts, ...)      the call graded on the sessions it was issued
                  verdict   = core.call.verdict_from_row(call_ts, edges, asof=d*)
PRESENTATION    ui/ renders the bundle and the verdict only: no estimator, no threshold, no inline lean
OFFLINE         studies/pooled_priors.py → pipeline.run.run_target over the catalogue
                → data/priors/pooled_priors_h{h}_L{L}.json → core.trust.PriorTable + KNOBS (measured)
                studies/render_docs_table.py → the README's edge table
```

app.py's phases call the same `pipeline.run` step functions and add only
progress display, caching and reruns; `run_target` chains them without
Streamlit for the study and the tests. `tests/test_pipeline_equivalence.py`
holds the two to the same `call_ts`, byte for byte, on the offline fixture.

The bundle (`app._BUNDLE_KEYS`) carries `call_ts`, `grades_ts`, `edges`,
`evidence`, `verdict`, d*, h, L, the epoch, the history fingerprint and the
priors' provenance; `call_ts` and every `edges` entry carry the same attrs
(h, L, epoch, fingerprint, priors study and run date, target, asset class,
code version).

---

## The grader

`core/grade.py` holds the one estimator, used for every role object and every
piece of graded evidence:

- **Target.** The drift-neutral h-session return from the execution close:
  sign · (Y_s − μ_s), where μ_s is the causal expanding mean of the returns
  resolved by s (s' + L + h ≤ s), undefined until `DRIFT_MIN_RESOLVED` have.
- **IC.** The Spearman rank IC on the masked rows, sampled every h sessions so
  no two windows overlap, averaged over the h phase offsets.
- **SE.** The circular-shift null computed over every admissible shift by FFT
  (shifts within `GRADE_SHIFT_MARGIN_H`·h of the true alignment excluded;
  undefined below `GRADE_MIN_PAIRS_H`·h pairs). The legacy SE is kept in the
  dict as a disclosure only.
- **Hit and base.** Hit on the called rows beside the side-matched causal
  no-skill rate; payoff, period blocks (`GRADE_N_BLOCKS`, `MIN_BLOCK_OBS`),
  n_win = rows // h.
- **As of each session.** `expanding_edge` grades on a grid of every h-th row
  anchored at the epoch spine's first row, using only pairs resolved by the
  grid date; it is undefined below `GRADE_WARMUP_WIN` windows, so appending data
  never moves a past grade.

Labels (`core.semantics.edge_tier`), on the posterior t of a role object and on
the own t of evidence: ≤ −1.645 measured against (neutral, never inverted);
(−1.645, −1] negative, unproven; |t| < 1 no measured edge; [1, 1.645)
unproven; [1.645, 2.5) edge; ≥ 2.5 solid edge.

---

## The pooled prior

Owner decision 2: an instrument's as-of grade is shrunk toward a leave-one-out
prior of its asset class, produced by one committed catalogue study.

- **The study** (`studies/pooled_priors.py`) is pre-registered: its docstring
  fixed the universe (every tradeable catalogue target plus ten
  `STUDY_STOCKS`), the held-out set, the objects, decisions A–C, the arms and
  the abort checks before its first run, and its SHA-256 is written into the
  output (`tests/test_priors.py` holds the committed output to the committed
  script). Each member runs the normal pipeline (`run_target`, no prior).
- **Pooling,** per object (claim, sizer:swayam, sizer:reversion, call) and
  quarter-end vintage: one catalogue-wide τ² by DerSimonian–Laird with class
  fixed effects; a leave-one-out class mean; its variance var_μ and the error
  covariance κ from the members' pairwise payoff-product correlations;
  se_floor, the SE of the class pool including the member. A class with fewer
  than `POOL_MIN_MEMBERS` other graded members pools the whole catalogue
  instead (global-LOO).
- **Lookup** (`core.trust.PriorTable.lookup`): the latest vintage on or before
  the session. A study member reads only its own leave-one-out entry — never
  the class's all-member entry, which contains it; an instrument outside the
  study reads its class's all-member entry. No file, or no vintage yet, is no
  prior.
- **Posterior** (`core.trust.posterior`), with own grade a, V_a = se²; prior μ,
  V_b = τ² + var_μ, covariance κ:
  w = clip((V_b − κ)/(V_a + V_b − 2κ), 0, 1); m = w·a + (1 − w)·μ;
  se_post = max(√(w²V_a + (1 − w)²V_b + 2w(1 − w)κ), se_floor); t_post = m/se_post.
  No own grade yet: the prior predictive. No prior: the own grade. Neither:
  ungraded, strength 0.
- **Decisions,** on held-out members only (the catalogue minus the four
  audited instruments and anything correlated ≥ 0.8 with them): A — claim
  centring, recorded and never applied without an owner-signed change; B —
  `SIZERS_ENABLED`, the sizers isolated at trust ≡ 1; C — `TIER_RULE`, tier
  words only if the payoff rises across causal size terciles.
- **Checks.** R4 (the FFT SE against a brute-force null) and R6 (the pool
  variance against a bootstrap) abort the study and write nothing; R5 (power
  and false-open per class) is recorded.

`core/knobs.py` reads the committed file at import and registers its results as
measured knobs; `core.trust.load_priors` refuses a file studied at another h
or L.

---

## Registries

**KNOBS** (`core/knobs.py`): every constant that can move the call, with its
kind — structural (fixed by an argument, never tuned), structural-inherited
(an older value kept without a study behind it), or measured (set by the
committed study, named with its run date) — its consumer, its horizon
coupling and its reason. `register_measured` is the only writer of a measured
value, and the decision-path modules hold no unregistered literal.

| Knob | Kind | Value | Moves |
|---|---|---|---|
| `EXECUTION_LAG` | structural | `1` | estimand |
| `DRIFT_MIN_RESOLVED` | structural | `60` | grade |
| `GRADE_SHIFT_MARGIN_H` | structural | `5` | grade |
| `GRADE_MIN_PAIRS_H` | structural | `12` | grade |
| `GRADE_MIN_WIN` | structural | `5` | grade |
| `GRADE_WARMUP_WIN` | structural | `30` | grade |
| `GRADE_N_BLOCKS` | structural | `6` | grade |
| `MIN_BLOCK_OBS` | structural | `8` | grade |
| `STRENGTH_T0` | structural | `1.645` | trust and every sizer β |
| `STRENGTH_T1` | structural | `2.5` | trust and every sizer β |
| `POOL_MIN_MEMBERS` | structural | `3` | trust: the prior's scope |
| `SIZER_BETA_MAX` | structural | `1.0` | every sizer's β |
| `SIZER_Z_CLIP` | structural | `1.0` | every sizer's z |
| `TIER_MIN_WIN` | structural | `20` | composer: Decision C |
| `CLAIM_SIGMA` | structural | `0.5` | precondition P4 and the claim |
| `DEEP_SIGMA` | structural | `2.0` | claim: the deep state words |
| `SWAYAM_DEAD` | structural | `0.1` | sizer:swayam words |
| `STATIONARITY_ALPHA` | structural | `0.05` | evidence:gap_stationarity |
| `CLAIM_CENTRING_WINDOW` | structural | `252` | claim: Decision A's dormant candidate |
| `SWAYAM_OVERSOLD` | structural-inherited | `-5` | sizer:swayam input |
| `SWAYAM_OVERBOUGHT` | structural-inherited | `5` | sizer:swayam input |
| `SWAYAM_LENGTHS` | structural-inherited | `8, 14, 22, 34, 52` | sizer:swayam input |
| `SWAYAM_ROC_FRAC` | structural-inherited | `0.85` | sizer:swayam input |
| `SWAYAM_MMR_NUM_VARS` | structural-inherited | `4` | sizer:swayam input |
| `SWAYAM_REGIME_SENSITIVITY` | structural-inherited | `8.0` | sizer:swayam input |
| `SWAYAM_BASE_WEIGHT` | structural-inherited | `0.0` | sizer:swayam input |
| `FVO_BURN_IN` | structural | `252` | claim |
| `FVO_MIN_PRINTS` | structural | `250` | claim |
| `FVO_VALUATION_DELTAS` | structural | `0.999, 0.9995, 0.9999, 1.0` | claim |
| `MIN_DATA_POINTS` | structural | `1500` | infrastructure: target history |
| `ANALYSIS_EPOCH` | structural | `2017-01-02` | infrastructure: every history's first session |
| `SIZERS_ENABLED` | measured | `True` | composer: Decision B |
| `TIER_RULE` | measured | `none` | composer: Decision C |
| `CLAIM_CENTRING` | measured | `none` | claim: Decision A, recorded only |
| `POOLED_PRIOR` | measured | `data/priors/pooled_priors_h10_L1.json` | trust: the class priors |
| `PRIOR:<object>:<class>` | measured | one per object and class | trust: the class summary {μ, se_μ, τ, k, as-of}, generated from the study file, never typed |

The four measured placeholders keep structural defaults (`True`, `none`,
`none`, no file) when no study file is present.

**ROLES / CARDS / CHIPS / CHAIN** (`core/roles.py`): listed above.

**INSTRUMENTS** (`core/instruments.py`): each target's ticker, asset class,
quote currency, tradeable flag, OHLCV availability and declared twins; the one
registry `engines.mula.blocks`' ticker table, the twin exclusions and the
registered predictor universe are built from. The asset classes and the names
the pages print them by (`ASSET_CLASS_NAMES`, the control rail's own
categories; data keep the key): `commodity` Commodities, `fx` Currency (FX),
`crypto` Crypto, `india_index` India Indices, `us_index` US Indices,
`stock_india` India Stocks, `stock_us` US Stocks, `sheet` Sheet series.

---

## Contracts and the tests that enforce them

Each charter clause, and the test files that hold the shipped system to it.
`tests/test_charter.py` is this table as code: one test per clause asserts that
each enforcing test exists and passed, and `tests/test_docs.py` holds this
table to that index.

| Clause | Contract | Tests |
|---|---|---|
| C1 | one h (from the instrument's config) and one L (EXECUTION_LAG) in every grade, the verdict and the bundle; an h = 20 config reaches every object | tests/test_contracts_asof_horizon.py, tests/test_call.py, tests/test_grade.py, tests/test_trust.py |
| C2 | every verdict input is read at d* | tests/test_contracts_asof_horizon.py, tests/test_call.py, tests/test_causality.py |
| C3 | only claim_direction(FVO) sets d; no sizer input changes it; one sign vocabulary | tests/test_call.py, tests/test_semantics.py, tests/test_swayam.py |
| C4 | one grader, in the pipeline; it cannot see the future; no estimator in ui/ | tests/test_roles.py, tests/test_app_smoke.py, tests/test_grade.py |
| C5 | the call at every session, resolved rows never rewritten, graded as issued and heading Overview | tests/test_causality.py, tests/test_call.py, tests/test_app_smoke.py |
| C6 | an unskilled or failed sizer is ×1.00; binding is never trust; every knob registered, no unregistered literal | tests/test_call.py, tests/test_knobs.py |
| C7 | trust from the causal as-of grade, its calibrated SE and a leave-one-out prior, through the one map | tests/test_grade.py, tests/test_trust.py, tests/test_study.py, tests/test_call.py |
| C8 | sizer inputs disjoint from the claim, \|ρ\| < 0.5 on claim days; twins dropped, and pooled at their true precision | tests/test_lineage.py, tests/test_instruments.py, tests/test_study.py |
| C9 | drift-neutral targets and side-matched causal bases | tests/test_grade.py |
| C10 | live value = graded value: the app's call_ts is run_target's; pages print the chain's own values | tests/test_pipeline_equivalence.py, tests/test_call.py, tests/test_swayam.py, tests/test_roles.py |
| C11 | the size is a unitless index; no '%' and no tier word unless Decision C passes; breadth in points | tests/test_copy.py, tests/test_call.py, tests/test_study.py |
| C12 | the claim's unit: reported as evidence; a strict xfail until Decision A is applied by an owner-signed change | tests/test_charter.py |
| C13 | one definition per concept, one copy per quantity; no retired or banned copy on a page or in the docs | tests/test_semantics.py, tests/test_app_smoke.py, tests/test_copy.py, tests/test_docs.py |
| C14 | no learner on the decision path; Swayam's views equal-weighted and causal; drift is not skill | tests/test_app_smoke.py, tests/test_swayam.py, tests/test_grade.py |
| C15 | every card registered to a role; every lean beside its grade; Overview in decision order; the rail on the call path | tests/test_roles.py, tests/test_render.py |
| C16 | every published column, edge, key, config field and knob has a reader | tests/test_consumers.py |

Beyond the clauses, the pinned history: `tests/test_prep_asof.py` (rows
through T never depend on the run date), `tests/test_golden_mula.py` (Mūla's
published frame against a golden baseline), `tests/test_priors.py` and
`tests/test_study.py` (the study's schema, leave-one-out by construction,
byte-reproducible output), and `tests/test_render.py` (every page in every
state the verdict can be in).

---

## Admission and re-entry rules

- **Evidence (the E-rule).** A displayed quantity is admitted only if it audits
  a named role-holder and is computed from that holder's own published inputs
  or outputs without a separate model — or is a required graded audit of one
  (the claim by side, the Swayam bucket table, the call's coverage and payoff).
  A lean carries its own grade. It must not be a second copy of a quantity
  already shown (C13).
- **Sizer.** A quantity becomes a sizer only by passing `tests/test_lineage.py`
  (inputs disjoint from the claim and the other sizers, |ρ| < 0.5 on claim days)
  and then entering the pooled study as a pre-registered candidate that earns β
  through f like any sizer.
- **Learner.** A learner must be causal, train on d·(Y − μ) at h with ESS/h, beat
  its registered baseline (equal weights, or the prior) in the pooled study,
  and pass a planted-drift ranking test: an informative member outranks an
  always-bullish one, and an abstainer cannot win.
- **Re-entry of what 2.9.0 deleted.** The code is recoverable at commit
  2755b4c (2.8.0 with its verification fixes); each part returns only this way:
  - Precedent (analog matching): only as a sizer on causally FVO-residualised
    Momentum, RealizedVol and Stress, through the sizer rule.
  - Convergence (agreement, the learned composite, the divergences): only on
    inputs that exclude Mūla's lean and |FVO|, and only by beating
    `sizer:swayam` in the pooled study.
  - Mūla's cross-section consistency: only as a pre-registered pooled
    candidate; its robustness question is the study's panel jackknife (arm P).
  - Mūla's window stack and its smoothers: not as a second copy of FVO (C13); a
    different claim unit only through Decision A and an owner-signed change.
  - Swayam's regime engine and learned view weights: through the learner rule;
    a change to the view bank is one pre-registered change, graded once.
  - Tier words and a '%' size: only when Decision C passes (`TIER_RULE`).

---

## Changing history

Each of these rewrites every published grade and needs the study re-run; each
is a versioned change, never a tweak:

- `ANALYSIS_EPOCH`, the first session of every history;
- the predictor universe or the instrument registry (`core/instruments.py`);
- `EXECUTION_LAG` (the priors file records L, and a file for another L is
  refused);
- the priors file itself.

The history's fingerprint is a content hash of the prepared rows up to the
session the study hashed each member to; System shows 'history rewritten —
history differs from the panel the priors were studied on' when the live
panel's differs from the study's (a failed fetch, an adjusted-price revision).
A Provisional newest row may be revised once and is flagged; nothing else that
has resolved changes when data are appended.

---

## What it will usually say

HOLD — or NO CLAIM inside the ±CLAIM_SIGMA noise — with the reason in numbers:
the claim's posterior t against STRENGTH_T0, its own grade, and the pool it was
shrunk toward, named with the study's date. Trust opens only where the claim
has a measured, price-side edge: an own grade strong enough, or a class pool
skilled enough, to carry the posterior over the bar. The committed study found
no asset class whose pool does that at h = 10 (README.md, generated from the
study), so a BUY or SELL appears only where a member's own grade carries it,
and the call's own grade, coverage and prospective record are always on screen
beside it.
