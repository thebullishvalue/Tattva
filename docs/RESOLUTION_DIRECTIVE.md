# Tattva — Audit and Resolution Directive

**Audited:** Tattva 2.8.2 at `dde6647`, 2026-11-10/11.
**Audience:** the implementing model. Read Part A once, then execute Part C in order.
**Reproduce every number here:** `studies/audit_2026_11/` (see its section at the end).

---

## Part A — Verdict

### What Tattva is

A Streamlit research terminal that makes one graded 10-session call on a target
(commodity, FX, crypto, index, stock). Mūla values log price against the
integrated factors of ~220 macro instruments and publishes the gap as an
oscillator (FVO). The call is contrarian on FVO. Six gates multiply into a
"conviction" that sets the word (STRONG BUY … HOLD). Swayam, Convergence and
Precedent are supporting evidence.

### Is it acceptable?

**As engineering: largely yes.** The estimators are genuinely causal, the
truncation test is real, the sign vocabulary is written once, the data layer
is defensive, and the code says what it measured. This is unusually honest work.

**As a system that issues BUY and SELL: no.** I fitted Mūla on 28 catalogue
targets from one live panel and graded the call the way a user would act on it
(claim when |FVO| ≥ 0.5σ, enter at the next close, hold 10 sessions):

| Measurement (28 targets, 2021-06 → 2026-11) | Result |
|---|---|
| Claims graded | 23,175 |
| Hit rate of the call as issued | **47.7%** |
| Hit rate of always-long on the same rows | **52.9%** |
| Mean signed outcome per claim | **−0.043** forward-return SDs |
| Targets where the call made money on average | 32% |
| Per-target 10d rank IC, mean across targets | −0.003 ± 0.012 |
| Share of targets with positive IC | 50% |
| Cross-sectional rank IC (cheapest vs richest each day) | −0.019 ± 0.023 |

The call has no measurable edge and is slightly worse than doing nothing. The
README's per-target table (ICs within one SE of zero) already hinted at this;
pooling removes the doubt. Three structural reasons, each measured:

1. **The oscillator is not centred.** It reads rich on 39% of days and cheap on
   24%. Gold's rich-to-cheap ratio is 3.2, the US indices 2.7–2.8. FVO correlates
   +0.21 with 126-day momentum. A slow-memory level regression lags a trending
   target, so "rich" largely means "has gone up", and the call shorts drift.
2. **The grade is not the call.** `series_edge` is a rank IC of a continuous
   lean over every row. The call is a sign at a fixed threshold on claim rows.
   Nifty Bank has IC +0.11 and a call hit rate of 54% against a 53% base.
3. **The gates do not discriminate.** On 7,105 claim rows across 8 targets the
   rank correlation between the product of the mispricing, reversion and
   cross-section gates and the realised signed outcome is **−0.046**. Deeper
   mispricing did worse, not better (mean outcome −0.009 in the lowest
   mispricing-gate quartile, −0.061 in the highest). "Conviction 22%" is an
   uncalibrated product, not a probability.

I also tried two obvious rescues — a causally de-meaned oscillator and
splitting claims by 6-month trend. Neither produced an edge (−0.013 and
−0.065 / −0.031). These were exploratory, on the same sample; treat them as
"not obviously fixable", not as settled.

### What I want the system to be

1. **A valuation terminal that never claims more than it has measured.** The
   call word is earned by the as-issued record, pooled across the asset class.
   Where that record is not positive, the page says "Rich by 1.4σ — no graded
   call" and shows the record. This is not the same as printing HOLD everywhere:
   the valuation, its drivers and its history remain the product.
2. **A system with a legitimate way to find an edge.** A pre-registered study
   harness with a holdout, so a signal variant can be promoted on evidence
   instead of by editing constants.
3. **Correct on its own terms.** Several real defects below, one of them a
   dated time bomb.
4. **A universe view.** Mūla fits in about 5 seconds per target. Ranking the
   whole catalogue takes four minutes and the product does not offer it.
5. **Smaller.** Roughly a third of the pipeline produces numbers that reach no
   decision.

### What happened before, and what to do differently

Git history shows two earlier rebuilds (3.0.0, 2.9.0), both reverted. 2.9.0
reached the same statistical conclusion and shipped it as eight work packages
with roles, registries and a charter. **Do not repeat that shape.** This
directive is incremental on 2.8.2, each package ships alone, and nothing here
adds a registry, a role grammar or a new abstraction layer.

---

## Part B — Findings

Severity: **S1** wrong output or a claim the data contradicts · **S2** real
defect, bounded effect · **S3** debt or cleanup. Every "measured" figure comes
from the audit scripts.

### B1. The call and its grade

| # | Sev | Finding | Where |
|---|---|---|---|
| B1.1 | S1 | The call as issued loses to always-long (table above). The page prints BUY/SELL anyway whenever the gate product clears 0.06. | `ui/components.py` `build_hero_verdict` |
| B1.2 | S1 | The edge gate and every "edge" chip grade a continuous lean over all rows, not the call. | `app.py` §4e, `convergence/intelligence.py` `series_edge` |
| B1.3 | S1 | Conviction is a product of six hand-shaped ramps (`_gate(x, lo, hi)`). It is not calibrated and is anti-correlated with outcomes (−0.046). Tier cut-points 0.30 / 0.15 / 0.06 are hand-set. | `ui/components.py` |
| B1.4 | S1 | FVO is biased rich (39% vs 24% of days; Gold 3.2:1). The claim threshold is applied to the raw level. | `engines/mula/valuation.py`, `core/semantics.py` |
| B1.5 | S2 | No execution lag. A lean dated *t* is scored from the close of *t*. For targets that close before New York (India, FX fixings) the panel row for *t* contains US closes that happened after the target's close. Measured: a 1-session lag moves 10d IC from +0.092 to +0.076 (Nifty 50), +0.033 to +0.005 (USD/INR), +0.090 to +0.067 (Copper); next-day IC is the largest single-day component on exactly those targets. | `series_edge`, `view_skill_weights`, `online_dimension_weights`, `analogs.py` |
| B1.6 | S2 | Per-target grading cannot resolve an edge: about 130 independent calls gives SE ≈ 0.09. Only pooling across targets has power. Nothing pools. | whole grading path |
| B1.7 | S2 | The precedent gate reads the analog base rate without consulting the matcher's own walk-forward skill, which the Precedent page computes and which was below its base rate on Gold and Copper per the code's own comments. | `build_hero_verdict`, `tab_precedent.py` |
| B1.8 | S2 | Gap half-life is 4–5.5 sessions (median, five targets) against a 10-session horizon. The horizon was declared, never tested against the signal's life. | `core/config.py` `FORECAST_HORIZON` |
| B1.9 | S3 | "Six independent conditions" is not true: reversion and cross-section are both functions of the same gap; convergence and precedent both condition on FVO. | README, landing page copy |

### B2. Data and history

| # | Sev | Finding | Where |
|---|---|---|---|
| B2.1 | S1 | **The rectangular-panel truncation.** `data.dropna(subset=[target] + features)` cuts every target's spine to the listing date of the youngest surviving predictor. Today that is SGOV, 2020-06-01. Measured: Gold has 2,347 fetched rows and uses 1,664 (71%); first published valuation is **2021-06-07**. The "~9y" claim on the landing page and README is 5.4 years of graded history. The engine was built for a ragged panel (per-instrument admission by cumulative prints, `ever_active`, NaN-safe returns) and never receives one. | `app.py` lines ~1196–1218 |
| B2.2 | S1 | **A dated history cliff.** The short-history guard keeps a predictor once it is non-null over the last `MIN_DATA_POINTS` (1500) target sessions. "India 5Y G-Sec" (first print 2021-04-07) has 1,445 CME-calendar rows today. In about **55 sessions (late January 2027)** it crosses 1500 for every CME-calendar target; the spine then starts 2021-04-07, shrinks to exactly 1500 rows, the first published date jumps to roughly April 2022, about 220 published sessions vanish and every remaining value is refit from a different origin. The same event recurs for every predictor that ever crosses the line. Fixing B2.1 removes it. | same |
| B2.3 | S1 | **A dead target is listed and valued.** Zinc (`ZNC=F`): 12 distinct prices in the whole history, 99.2% of sessions unchanged. It publishes 97 valuations with \|FVO\| up to **2,065σ**. README says "the catalogue is what can run". Nothing checks that a target's own price is alive. | `core/config.py`, `app.py` |
| B2.4 | S2 | The fetch window is `today − 9y → today`, so the origin slides daily. Today B2.1 hides this (the spine starts at SGOV's listing regardless). Once B2.1 is fixed, a sliding origin would move every EW estimator's start each day. The origin must be pinned in the same change. | `app.py` (three `365 * 9` sites), `data/universe.py` |
| B2.5 | S2 | The panel is forward-filled at source, destroying the true print mask. The engine then infers prints from value changes, which misreads a genuine unchanged print and cannot see a stale vendor repeat. `fit(printed=…)` exists and is never passed. | `data/fetcher.py` `combined.ffill()`, `MulaEngine._infer_printed` |
| B2.6 | S2 | The universe double-counts. Exact inverses are both present (USD/INR and INR/USD; USD/JPY and JPY/USD; USD/KRW and KRW/USD; USD/CNY and CNY/USD; UUP and UDN), plus eleven `X/INR` crosses that are each USD/INR times a major. In a PCA of standardised returns this manufactures an INR factor and a dollar factor with inflated eigenvalues. | `core/config.py` `MACRO_SYMBOLS_YF` |
| B2.7 | S2 | Unadjusted closes are used for stock and ETF targets, so every ex-dividend date is scored as a loss. The repaint argument for avoiding back-adjusted prices is correct for predictors; for the *target's forward return* a forward-cumulated total-return series is non-repainting and correct. | `data/fetcher.py` |
| B2.8 | S2 | Crypto: the panel drops weekend rows but Swayam reads 7-day OHLCV, so "10 sessions" means 14 calendar days in Mūla and 10 in Swayam's skill weights. | `fetcher.py` (`dayofweek < 5`), `kernel.view_skill_weights` |
| B2.9 | S2 | `IEGA.L` no longer resolves; it logs a "dropped" warning on every fetch. Continuous futures (`=F`) carry roll gaps as returns; nowhere stated. | `core/config.py`, README |
| B2.10 | S3 | `MulaEngine` decides whether the last row is "forming" from `pd.Timestamp.today()` inside the engine, so the engine is not a pure function of its inputs. | `engines/mula/base.py` ~line 747 |

### B3. Mūla

| # | Sev | Finding | Where |
|---|---|---|---|
| B3.1 | S2 | `OnlineFactorModel` keeps **one** cumulative `levels` vector but switches the active correlation memory by `argmax` of a likelihood weight. Each memory has its own eigenbasis, aligned only against its own past (`_align`). When the active memory changes, factor *j*'s level starts accumulating scores from an unrelated direction. The integrated regressor silently changes meaning. | `engines/mula/factors.py` `update` |
| B3.2 | S2 | FVO is called σ but is fat-tailed and wide: SD 1.0–1.3, 95th percentile of \|FVO\| 2.1–3.0. `DEEP_SIGMA = 2.0` ("deep mispricing") is reached on more than 10% of Gold's days. The conformal multiplier is used for the band only, not for the state words. | `core/semantics.py`, `valuation.py` |
| B3.3 | S2 | The OU block refits a 60-point OU on every row to produce `theta_history`, whose only consumer is a `theta_stable` flag no page acts on. The 90-day OU projection and its band are computed and stored on every fit. | `base.py` `_compute_ou_diagnostics` |
| B3.4 | S3 | The window stack (Z_21/63/252, zones, `Buy_*`/`Sell_*` crossovers, `BuySignalBreadth`, `ConvictionRaw`, its DDM and bounds) feeds one convergence dimension, one Precedent feature and one chart. `BuySignalBreadth` and the crossover columns have no reader. | `base.py`, `analytics/utils.py` |
| B3.5 | S3 | The valuation loop writes ~12 series nobody reads (`regime_entropy`, `regime_drift`, `switch_prob`, `regime_persistence`, `adapt_memory`, `factor_var_share`, `loo_dispersion`, `xs_dispersion*`, `corr_memory`, `resid_rmse`). | `valuation.py` |
| B3.6 | S3 | `r2_oos` is a level R² on an integrated series and is displayed as model quality; it is near 1 by construction, as the code's own docstring says. | `base.py`, `tab_mula.py` |

### B4. Swayam

| # | Sev | Finding | Where |
|---|---|---|---|
| B4.1 | S2 | MMR regresses **price levels** on 20-day rolling level correlations, picks the top 4 of ~220 candidates per row, and fills missing macro values with 0. That is a max-statistic over spurious level correlations; the code's own caveat says the fit quality "reads high by construction". | `kernel.calculate_mmr` |
| B4.2 | S2 | The flow component's regime counter uses a fixed ±0.33% daily move for every asset. That is most days for Bitcoin and almost none for USD/INR. ATR length is a fixed 14 regardless of the view's length. | `kernel.calculate_msf` |
| B4.3 | S2 | Each view's result frame carries all ~220 joined macro columns (~270 columns × 15 views) before it is slimmed. Price-only members receive and copy the macro columns too. | `ensemble.build_swayam_frames`, `kernel.run_full_analysis` |
| B4.4 | S2 | The MMR replica screen uses a full-sample correlation. It only excludes, but it makes the output depend on future rows, so Swayam is not covered by the truncation guarantee the README states for the system. | `kernel.calculate_mmr` |
| B4.5 | S3 | "Fifteen views" is 13 for any target without volume (indices, FX): flow-only members are dropped. Copy says fifteen everywhere. | `ensemble.py`, README, landing |
| B4.6 | S3 | `msf_raw = sum/√n` then `sigmoid(msf_raw·√n)`: the scaling cancels. | `kernel.calculate_msf` |
| B4.7 | S3 | A failing member is swallowed by a bare `except: continue` with no log line. | `ensemble.build_swayam_frames` |

### B5. Convergence and Precedent

| # | Sev | Finding | Where |
|---|---|---|---|
| B5.1 | S2 | The convergence layer computes four agreement dimensions, learns their weights online, builds a composite, filters it with a DDM, builds a second "consensus", and detects divergences. **The verdict reads none of it** except a divergence count in a footnote; the convergence gate reads Swayam's net breadth directly. The composite's IC tracks Mūla's because half of it is Mūla. This is a parallel opinion carried at full cost. | `convergence/*`, `app.py` phase 4 |
| B5.2 | S2 | Phase 4 iterates ~1,400 rows with `fvo_ts.loc[...]` and `iterrows()` and updates the progress bar in the loop. | `app.py` ~1700 |
| B5.3 | S3 | DDM constants (two sets) are hand-set and admitted as such. | `core/config.py` |
| B5.4 | S3 | Precedent matches on eight features, six of which are Mūla outputs, with a pool of ~1,100 rows and a 60-session Theiler gap: at most ~18 distinct episodes exist. The page is honest about this; the gate is not (B1.7). | `analytics/analogs.py` |

### B6. Application structure

| # | Sev | Finding | Where |
|---|---|---|---|
| B6.1 | S2 | `app.py` `main()` is ~1,500 lines mixing data preparation, five pipeline phases, session caching and page rendering. None of the pipeline can run without Streamlit, which is why there is no end-to-end test and why this audit had to re-implement the prep (`studies/audit_2026_11/harness.py`). | `app.py` |
| B6.2 | S2 | Tests: 25, all synthetic, 13 seconds. Nothing tests data preparation, the fetcher's assembly, Swayam's causality, the analog matcher's causality, or the as-issued call. | `tests/` |
| B6.3 | S3 | Config carries dead structure: `PER_INSTRUMENT_TUNING` and `_PER_INSTRUMENT_OVERRIDES` (two cosmetic overrides, four empty dicts), `CONVICTION_STRONG/MODERATE`, `hold_horizons`, `ui_*` warm-up priors, five identical `CLASS_CONFIG_DEFAULTS`. | `core/config.py` |
| B6.4 | S3 | `vault/` holds 1.3 MB of zipped old trees in the repository. Git is the archive. | `vault/` |
| B6.5 | S3 | Comments narrate removed code at length (roughly half of `app.py`'s non-blank lines). History belongs in the changelog; a comment should explain the line under it. | throughout |
| B6.6 | S3 | The disk cache unpickles whatever is in `~/.cache/tattva`. Acceptable on a single-user host; say so. | `data/cache.py` |

### B7. Not audited

The 2,509-line stylesheet and the rendered pages were read as code but **not
viewed in a browser** in this audit. Part C does not change the design system.
Any UI change below must be verified in the browser on Streamlit 1.52.2 in both
Paper and Slate before it is committed.

---

## Part C — The directive

### Rules for the implementer

1. Work on branch `claude/eager-shannon-7nxvdg`. One commit per work package,
   message `WPn: <what changed>`. Run `python -m pytest tests -q` before each
   commit; it must pass.
2. Do the packages in order. WP1 is a prerequisite for everything.
3. **Do not tune anything to improve a number in Part A.** Where a package
   changes published values, record before and after with the audit scripts and
   put both in the commit message. A worse number is an acceptable result.
4. Never use a value in production code that came from a full-sample fit of
   the data it is applied to.
5. Do not add a dependency, a registry, a plugin system or a config layer. Do
   not rename engines or pages. Do not touch `ui/theme.css` except where a
   package says so.
6. Delete code instead of commenting it out. When deleting, also delete the
   comments that describe the deleted thing.
7. If a step cannot be done as written, stop that package, leave the tree
   passing, and write what blocked it in the commit message. Do not improvise a
   substitute design.
8. Keep every Streamlit pin and the `[theme]` section exactly as they are.

### WP1 — Extract the pipeline from Streamlit

**Why:** B6.1, B6.2. Every later package needs a pipeline that runs headless.

**Create `pipeline/` (new package, with `__init__.py`):**

- `pipeline/prep.py`
  - `EPOCH = "2015-01-01"` and `fetch_window() -> (pd.Timestamp, pd.Timestamp)`
    returning `(EPOCH, today)`. Replace all three `365 * 9` sites in `app.py`
    and the one in `data/universe.py` with it. (Pins the origin, B2.4.)
  - `@dataclass PreparedPanel`: `target: str`, `calendar: pd.DatetimeIndex`,
    `price: pd.Series`, `panel: pd.DataFrame`, `features: list[str]`,
    `config: InstrumentConfig`, `report: dict`.
  - `prepare(df, target) -> PreparedPanel`, raising `PrepError(message,
    report)` for each failure `app.py` currently reports with `st.error`
    (missing column, too few rows, no features, non-positive target). Move the
    logic of `app.py` lines ~1156–1295 here verbatim in this package; behaviour
    changes come in WP2.
- `pipeline/run.py`
  - `@dataclass RunResult` with exactly the fields in `_BUNDLE_KEYS` plus
    `prepared`, `signal`, `model_stats`, `precedent_summary`, `analogs`.
  - `run_target(df, target, *, ohlcv=None, progress=None, today=None) ->
    RunResult` containing phases 2–5 and the precedent block of `app.py`
    (lines ~1412–1924), with `console.*` calls kept and every `st.*` /
    `progress_bar` call replaced by the optional `progress(pct, title, detail)`
    callback.
  - `today` is passed down to `MulaEngine.fit(..., today=...)`; the engine
    uses it instead of `pd.Timestamp.today()` (B2.10). Default `None` means
    "now", resolved once in `run_target`.
- `app.py` keeps: page config, theme, rail, freshness notices, session caching,
  the page shell. It calls `prepare` and `run_target` and stores the result.
  The Swayam OHLCV fetch stays in `app.py` (it is I/O) and is passed in.

**Tests (new `tests/test_pipeline.py`):**

- Build a synthetic `df` (reuse `_panel` from `test_causality.py`, add a
  `DATE` column and register a temporary target) and assert `run_target`
  returns a populated `RunResult` with a finite `signal["fvo"]`.
- Assert `prepare` raises `PrepError` for a target with a non-positive price
  and for a frame shorter than `MIN_DATA_POINTS`.

**Accept:** app behaviour unchanged. Confirm by running
`studies/audit_2026_11/exp1.py` before and after on the same cached
`panel.pkl`: Gold's `FVO` column must be identical on every row.
`app.py` is under 900 lines.

### WP2 — Give the engine the panel it was built for

**Why:** B2.1, B2.2, B2.3, B2.5, B2.6, B2.9. This is the largest correctness
change and it moves every published number. That is expected.

1. **Ragged panel.** In `pipeline/prep.py`:
   - Delete the short-history guard (`data[f].tail(_win).notna().all()`) and
     replace `dropna(subset=[target] + features)` with
     `dropna(subset=[target])`. Leading NaNs in predictors stay NaN.
   - Drop a predictor only if it has fewer than `FVO_MIN_PRINTS` finite rows in
     total, and report it in `report["feats_dropped"]`.
   - Verify by reading, then by test, that `MarketValuationEngine.run` treats a
     NaN predictor as not available: `LX` → NaN → `R` NaN → `avail` False, and
     `_infer_printed` marks the first finite row as a print. Fix any site that
     does not (candidates: `expl_px.astype(float)` is fine; `block_contrib`
     and `AdaptiveVolPanel.update` already guard with `isfinite`).
   - `MIN_DATA_POINTS` now applies to the target's own rows only.
2. **True print mask.** In `data/fetcher._fetch_macro_live_uncached`, before
   `combined.ffill()`, compute `printed = combined.notna()`; carry it through
   `fetch_commodity_dataset` as `df.attrs["printed"]` (a bool frame with the
   friendly column names and the same row order as `df`). Catalogue and sheet
   targets get their own mask from their pre-ffill series. `prepare` slices it
   to the calendar and `run_target` passes it as `fit(printed=...)`. Keep
   `_infer_printed` as the fallback when the attribute is absent (cached
   frames from before this change, tests).
3. **Target liveness.** In `prepare`, after the session-spine filter, compute
   on the last 252 target rows the share of sessions where the price changed.
   If it is below 0.50, raise `PrepError("'<target>' price feed is not live:
   <n> distinct prices in the last 252 sessions")`. Then remove **Zinc** from
   `COMMODITY_TARGETS`, `TARGET_CATEGORIES` and `TARGET_EXCLUDED_PREDICTORS`.
4. **Universe hygiene** in `core/config.py`:
   - Remove the redundant inverses: `INR/USD`, `JPY/USD`, `KRW/USD`,
     `CNY/USD`, `US Dollar Bearish (UDN)`. Fix `TARGET_EXCLUDED_PREDICTORS`
     entries that name them.
   - Keep `USD/INR`; remove the other ten `*/INR` crosses from the predictor
     map (they are USD/INR × a major that is already present). `USD/INR`'s
     exclusion list comprehension then becomes empty of crosses; simplify it.
   - Remove `IEGA.L`.
5. **Copy.** Landing page and README: replace "~9y" with the measured span,
   computed at runtime where it is shown (first published date to last).

**Tests:**

- `test_ragged_panel_extends_history`: synthetic panel where one predictor
  starts at row 600; assert the published history starts where it would
  without that predictor (to within the burn-in), and that the truncation test
  still passes exactly on the ragged panel.
- `test_late_predictor_never_moves_earlier_rows`: fit with and without a
  predictor that first prints at row 900; assert rows before 900 are
  bit-identical. (This is the property that kills the cliff, B2.2.)
- `test_dead_target_is_refused`: a 99%-flat price raises `PrepError`.
- `test_no_inverse_pairs`: for the configured universe, assert no two tickers
  are a known inverse pair (hard-code the five removed pairs as the fixture).

**Accept:** `exp1.py` shows Gold's first published date moving from
2021-06-07 to within 300 sessions of the target's first row; record the new
row counts and the as-issued table from `an3.py` in the commit message.

### WP3 — Grade the call that is issued

**Why:** B1.1, B1.2, B1.5, B1.6. After this package the system knows its own
record. WP4 acts on it.

1. `core/config.py`: add `EXECUTION_LAG = 1` with a one-line comment: a call
   dated *t* is acted on at the close of *t+1*.
2. `convergence/intelligence.py`:
   - `_forward_log_return(price, horizon, lag=EXECUTION_LAG)` returns
     `lp.shift(-(lag + h)) - lp.shift(-lag)`. Thread `lag` through
     `series_edge` (default `EXECUTION_LAG`), `_build_calibration_frame` and
     `online_dimension_weights` (a call at `s` resolves at `s + h + lag`).
   - Apply the same lag in `engines/swayam/kernel.view_skill_weights` and in
     `analytics/analogs` (forward returns for analogs and `Realized`; the pool
     cutoff becomes `p + H + lag <= t`).
   - New `call_record(fvo, price, horizon, lag=EXECUTION_LAG) -> dict`:
     on rows with `claim_direction(fvo) != 0`, `y = direction × forward
     return`. Return `n_claims`, `n_independent` (`n_claims // horizon`),
     `hit`, `base` (majority-direction rate on the same rows),
     `mean_bps`, `se_bps` (SD of `y` / √`n_independent`), `t`,
     `long: {n, hit, mean_bps}`, `short: {...}`, and the same six-block split
     `series_edge` uses, each block with `n`, `hit`, `base`, `mean_bps`.
3. `pipeline/run.py`: compute `edges["call"] = call_record(...)` beside the
   existing three. Keep `edges["mula"]` (the lean IC) as a diagnostic.
4. **Pooled record.** New `studies/pooled_record.py` (CLI, no Streamlit):
   fetches the panel, runs Mūla only on every catalogue target, and writes
   `data/priors/call_record.json`:
   `{generated, panel_fingerprint, horizon, lag, classes: {<class>: {n_claims,
   n_independent, hit, base, mean_bps, se_bps, t, targets: [...]}}, targets:
   {<name>: call_record}}`. Class = the `_CATEGORY_TO_CLASS` value. Pool by
   concatenating the per-target `y` standardised by each target's own
   forward-return SD; `se` uses `n_independent` summed over targets divided by
   an effective-breadth factor of 2 (targets in a class co-move; state this in
   the docstring as a declared conservatism). Commit the generated file.
   Stocks use `us_index` / `india_index` as their class.
5. `core/trust.py` (new, ~60 lines, pure): `load_record()` reads the JSON
   once; `class_record(target) -> dict | None`.

**Tests:** planted-signal recovery for `call_record` (a synthetic series where
the call is right 60% of the time must report hit ≈ 60 and `t > 2`); the lag
test (a lean that is the same-day return has IC ≈ 1 at lag 0 and ≈ 0 at lag 1);
update `test_edge_and_learners.py` expectations for the lag.

**Accept:** `data/priors/call_record.json` exists, is reproducible from
`studies/pooled_record.py`, and its pooled figures are in the commit message.

### WP4 — The verdict says only what the record supports

**Why:** B1.1, B1.3, B1.7, B1.9. Rewrite `build_hero_verdict`; keep its
signature style (pure data in and out) and keep `render_hero_card`'s layout.

**New rule, in full:**

1. **State** (always shown): `valuation_state(fvo)`, the percentage gap, FVO.
   Unchanged.
2. **Licence.** The call is *licensed* if the class record for the target's
   asset class has `t >= 1.0` **and** `hit > base`, on the side being called
   (use the `long` or `short` sub-record). Otherwise it is *unlicensed*.
   There is no per-target override: one target's 130 calls cannot outvote its
   class.
3. **Word.**
   - No claim (|FVO| < 0.5): `FAIR — NO CALL`.
   - Claim, unlicensed: `CHEAP — NO GRADED CALL` / `RICH — NO GRADED CALL`.
     `signal_class = "hold"`. Headline:
     `"1.4σ rich versus the cross-section. Rich calls in <class> have hit
     46.5% against a 53% base over 9,920 claims, so no call is issued."`
   - Claim, licensed: `BUY` / `SELL`. No STRONG or LEAN tiers.
4. **Evidence rows** replace the gate chain. Same six rows, same order, same
   detail sentences, same bar. Each row shows its reading and a tone
   (supports / neutral / against) using the existing label thresholds.
   **They are not multiplied.** Remove `conviction`, `_TIERS`, `_TIER_WORD`,
   `_GATE_ABSENT`, `_GATE_UNVALIDATED`, the `binding` concept and
   `format_conviction`. The card's foot line becomes:
   `"Evidence is shown, not multiplied: on <n> past claims these readings did
   not separate good calls from bad."` Fill `<n>` from the record.
5. **Edge row** reads `edges["call"]` (this target) and the class record, both
   as hit versus base and mean bps ± SE.
6. **Precedent row** is `neutral` with the text "matcher has no measured
   skill at this horizon" unless the analog walk-forward (`analog_skill_by_
   horizon`, computed once in `run_target` at the forecast horizon only and
   stored on `RunResult`) has `edge > 0` and `n >= 20`.
7. Overview KPI strip: replace the "Conviction" tile with "Record" (class hit
   versus base). Command bar meta: `Call` shows the new word only.
8. Copy: remove "six independent conditions" from README, landing lede,
   `_SYSTEM_PANELS` and the section header. Replace with "six readings".

**Tests:** rewrite `tests/test_hero_verdict.py` for the rule above: unlicensed
claim never yields `buy`/`sell`; licensed claim does; no claim yields
`FAIR — NO CALL`; verdict dict has no `conviction` key; a missing record file
means unlicensed.

**Accept:** in the browser, Overview for Gold, Nifty 50 and S&P 500 renders in
Paper and Slate with no layout break; screenshots attached to the commit
message as file paths are not required, a one-line statement of what each page
read is.

### WP5 — A pre-registered way to earn a call

**Why:** the system needs a legitimate route from "no graded call" to a call.
This package builds the route and runs it once. It does not promote anything.

1. `studies/PREREGISTRATION.md`, committed **before** any study code runs,
   stating:
   - Sample: catalogue targets passing WP2's liveness check; history from each
     target's first published row.
   - **Holdout: the final 252 published sessions of every target. No study
     script may read them** except `--final`, which may be run once per
     candidate and whose output is committed.
   - Metric: pooled as-issued `call_record` (`mean_bps`, `t`, hit − base),
     horizon and lag as configured.
   - Promotion bar: on the development sample `t >= 2.0` and hit > base in at
     least four of six blocks; then on the holdout mean > 0 and hit > base.
   - The candidate list below, fixed. Adding a candidate requires a new dated
     section in the file and discounts the bar (`t >= 2.0 + 0.25 × extra
     candidates`).
2. Candidates (each a pure function `fvo_ts -> lean series`, in
   `studies/candidates.py`; none touches production code):
   - **C0** the shipped call (baseline).
   - **C1** causally centred: FVO minus its own EW mean (half-life 126,
     `shift(1)`), same ±0.5 threshold.
   - **C2** causal rank: FVO's expanding percentile (`GapPercentile`), claim
     below 0.2 / above 0.8.
   - **C3** cheap-only: C0 restricted to the long side.
   - **C4** horizon 5 instead of 10 (C0 otherwise).
   - **C5** cross-sectional: each day rank catalogue targets within asset
     class by FVO; long the cheapest third, short the richest third; graded
     as a spread.
   - **C6** fast-memory gap: FVO recomputed with
     `FVO_VALUATION_DELTAS = (0.995, 0.998, 0.999)`. (The code argues against
     this on likelihood grounds; the question here is the as-issued record,
     which was never measured.)
3. `studies/run_study.py` runs every candidate on the development sample and
   writes `studies/results/<date>.json` and a markdown table.
4. Run it. Commit the results. **Do not run `--final`** and do not change
   production behaviour in this package. Report the table in the commit
   message and stop.

The owner decides promotion. If a candidate is promoted later, it replaces the
claim rule in `core/semantics.py` and `studies/pooled_record.py` regenerates
the record; nothing else changes.

### WP6 — Universe scan

**Why:** the untapped product. Depends on WP1–WP4.

1. `pipeline/scan.py`: `scan(df, targets, progress=None) -> pd.DataFrame`
   running **Mūla only** (`MulaEngine.fit` + `get_current_signal` +
   `call_record`) per target, catching `PrepError` per target. Columns:
   `target, class, state, fvo, pct_gap, gap_percentile, half_life, mr_prob,
   xs_consistency, hit, base, licensed, as_of, withheld_reason, error`.
2. New page **Universe** (first under "Engines" in `st.navigation`), in
   `ui/tabs/tab_universe.py`: a "Scan catalogue" button with the existing
   progress bar; result cached in `st.session_state["universe_scan"]` keyed by
   panel fingerprint; rendered with `render_data_table`, sorted by `fvo`,
   state shown as the existing chips; a selectbox + button to make any row the
   active target (set `active_target`, clear `engine`/`engine_cache`, rerun).
3. No BUY/SELL words on this page. Columns are states and records only.
4. Memory: keep only the result frame, never the engines.

**Tests:** `scan` on a two-target synthetic frame returns two rows; a target
that raises `PrepError` yields a row with `error` set and does not abort.

**Accept:** a full scan completes in under ten minutes on the hosted tier's
single vCPU (measure locally with `OPENBLAS_NUM_THREADS=1`; it was ~5 s per
target in this audit) and the page renders in both themes.

### WP7 — Repair or remove what no decision reads

Do these as separate commits `WP7a` … `WP7f`. Each must leave the truncation
test and `an3.py`'s as-issued table within noise (state both in the message).

- **7a Factor levels (B3.1).** In `OnlineFactorModel`, give each `_CorrMemory`
  its own `levels` and `contrib` arrays, updated every step from that memory's
  own basis. `update` returns the levels and contributions of the active
  memory. When the active memory changes, nothing is re-aligned: the regressor
  switches to that memory's own continuous history. Add a test that feeds a
  stationary two-factor panel and asserts each memory's level series is
  continuous (no jump larger than 6 in one step) across forced switches.
- **7b Swayam MMR (B4.1, B4.3, B4.4).** Rewrite `calculate_mmr` to work on
  **log returns**: rolling `length`-window correlation of the target's log
  return with each macro log return (shifted one row), top-`num_vars` by |r|,
  r²-weighted predicted *return*, deviation = cumulative sum over `length` of
  (actual − predicted) return, then the existing z-score and sigmoid. Missing
  macro returns are excluded from that row's candidates, never filled with 0.
  Replace the full-sample replica screen with a causal one: exclude a
  candidate on a row when its trailing-252 return correlation (shifted) with
  the target exceeds 0.98. Make `run_full_analysis` return only kernel
  columns (drop macro columns from the output); delete `_slim_view_dfs` and
  `_SWAYAM_DRILLDOWN_COLS` handling that becomes redundant. Add
  `tests/test_swayam_causality.py`: `build_swayam_frames` +
  `view_skill_weights` + `aggregate_views` on `data[:T]` and `data[:T-200]`
  agree exactly on shared dates.
- **7c Swayam scale (B4.2, B4.6, B4.7).** Regime-count threshold becomes
  `0.5 × trailing-`length` SD of daily returns (shifted)`; ATR length becomes
  the view's `length`; remove the cancelling √n; log a warning with the member
  name when a member raises.
- **7d Convergence (B5.1, B5.2).** Delete the dimension machinery:
  `CrossValidator`, `score_dimensions`, `composite_score`,
  `online_dimension_weights`, `apply_online_weights`, `CompositeDDM`,
  `convergence/conviction_model.py`, the `CONV_*` constants and config
  fields, `weights_seed`, the rail's "Model" weight rows, the System page's
  learner panel, and their tests. Keep: `consensus_series` (the page's one
  chart), `CrossSystemDivergenceDetector`, `series_edge`, `call_record`. The
  Convergence page becomes: consensus chart, Mūla-lean versus Swayam-lean
  scatter over the window, divergences table. Replace the row loop with a
  date-indexed join. `edges["composite"]` is removed; the System edge table
  shows Mūla lean, Swayam breadth and the call.
- **7e Mūla dead weight (B3.3–B3.6).** Remove: `theta_history` and the rolling
  OU refit, `ou_projection*`, `OU_PROJECTION_DAYS`, `theta_stable`;
  `_detect_crossover_signals`, `Buy_*`/`Sell_*`, `BuySignalBreadth`,
  `SellSignalBreadth`; the unread valuation series listed in B3.5; `r2_oos`
  from the quality cards (keep `r2_vs_anchor`). Keep the window z-scores and
  breadth (one chart and one analog feature read them). Remove anything on the
  Mūla page that displayed a removed field.
- **7f Config and repo (B6.3–B6.5).** Delete `PER_INSTRUMENT_TUNING`,
  `_PER_INSTRUMENT_OVERRIDES` and their asserts, `CONVICTION_STRONG/MODERATE`,
  `hold_horizons`/`HOLD_HORIZONS`, unused `ui_*` fields (grep each; keep any
  with a live reader). Collapse `CLASS_CONFIG_DEFAULTS` to the entries that
  differ. `git rm -r vault/` and remove its README line. In every file touched
  by WP1–WP7, delete comments that describe removed code or past bugs; keep
  comments that explain the present line. Do not sweep untouched files.

### WP8 — Remaining correctness items

- **8a Total-return targets (B2.7).** For stock targets only, fetch with
  `actions=True`, build a forward-cumulated total-return index
  (`TR_t = TR_{t-1} × (Close_t + Dividend_t) / Close_{t-1}`, anchored at the
  first row so past values never change), and use it for **forward returns in
  grading only**. Valuation keeps the traded close. Test: an added dividend
  on the last row leaves every earlier `TR` value unchanged.
- **8b Crypto calendar (B2.8).** In `app.py`, before Swayam, restrict a crypto
  target's OHLCV to the model spine's dates so both engines count the same
  sessions. One line in the README's data section says crypto is read on a
  five-day week.
- **8c Sigma words (B3.2).** In `core/semantics.py` docstrings and UI copy,
  describe FVO as "gap in units of its typical size" wherever it says σ implies
  a normal tail; add the causal `GapPercentile` beside FVO on the Mūla read
  card and Overview valuation tile ("1.6 — 91st percentile of its own
  history"). Do not change `CLAIM_SIGMA`/`DEEP_SIGMA` here; that is a WP5
  candidate (C2), not a copy edit.
- **8d Stated limits.** README "Data sources": continuous futures carry roll
  gaps; the disk cache trusts its own directory; Swayam's guarantees after 7b.

### WP9 — Documents and release

- README: rewrite "What it does", the gate table, "Honest validation" and
  "Where the edge is not" from the shipped system. The edge table is replaced
  by the pooled as-issued table generated from `data/priors/call_record.json`
  by a small `studies/render_record.py` (prints markdown; paste between two
  HTML comment markers so it can be regenerated).
- CHANGELOG: one `3.1.0` entry (3.0.0 was used by a reverted build), sections
  per package, leading with what a user will see change.
- `core/config.py` `VERSION = "3.1.0"`.
- Move this file's Part A table into the README once regenerated; keep this
  directive in `docs/` unchanged as the record of why.

---

## Order, size and stop conditions

| WP | Depends on | Changes published numbers | Rough size |
|---|---|---|---|
| 1 Extract pipeline | — | no | large, mechanical |
| 2 Ragged panel, liveness, universe | 1 | **yes, all** | medium |
| 3 As-issued grading, pooled record | 2 | grades only | medium |
| 4 Verdict | 3 | words only | medium |
| 5 Study harness | 3 | no | medium |
| 6 Universe scan | 4 | no | small |
| 7 Repair / remove | 4 | 7a, 7b, 7c yes | large |
| 8 Correctness items | 4 | 8a grades | small |
| 9 Documents | all | no | small |

Stop and report to the owner after **WP4** (the product now tells the truth)
and after **WP5** (the study table decides what happens next). Do not start
WP7 before the owner has seen WP5's table: if a candidate is promoted, 7a–7c
are re-measured against it.

## Decisions that belong to the owner

1. **The licence bar** (WP4: `t >= 1.0` and hit > base on the called side).
   Lower it and the page calls more often on weaker evidence. I recommend it
   as written.
2. **Whether an unlicensed claim should still show a direction word.** I
   recommend no: "RICH — NO GRADED CALL" is the honest reading.
3. **Removing the convergence dimension machinery (7d).** It is the largest
   deletion. It is recommended because no decision reads it.
4. **The epoch** (`2015-01-01`). Earlier means more history and more early
   rows on a thin panel; the liveness and admission rules handle the latter.

## Reproducing the audit

```bash
cd studies/audit_2026_11
python exp1.py      # 8 targets + two shifted-origin fits  -> fits.pkl   (~1 min)
python exp2.py      # every catalogue target, Mūla only     -> cat.pkl    (~3 min)
python an1.py       # lag sensitivity, gate inputs, spine truncation
python an2.py       # the history cliff, horizon term structure, gate discrimination
python an3.py       # pooled as-issued call record, cross-sectional IC
```

`harness.py` is a headless copy of `app.py`'s data preparation; it caches the
fetched panel as `panel.pkl` beside itself. Numbers will differ slightly on a
different day's panel; the conclusions in Part A were far outside that noise.
The scripts are the audit's working files, not production code. WP1 makes
`harness.py` redundant; delete the directory's `harness.py` then and point the
scripts at `pipeline.prep`.

**Limits of this audit.** Swayam, Convergence and Precedent were read in full
but only Mūla was run on live data; Swayam's README edge figures were not
re-measured. The UI was not viewed in a browser. The pooled record covers one
panel vintage and one 5.4-year window that was a bull market for most of the
catalogue, which is exactly the regime in which a contrarian call does worst;
WP2 extends the window and WP3 re-measures it.
