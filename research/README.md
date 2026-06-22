# Tattva — Research & Tuning Suite

Every tuned constant in Tattva was chosen by a computational study, not by hand.
This folder holds those studies and a one-command **orchestrator** to re-run them.

> **Philosophy:** these scripts *measure and recommend* — they never auto-edit
> config. Each tuning decision needs human judgement (parameter interactions,
> regime shifts, signal redundancy). Run → read the report → apply by hand,
> guided by the rationale comments in `core/config.py` and the CHANGELOG.

## One command

```bash
python3 research/run_tuning.py --list          # show the suite + ETAs (~4h full)
python3 research/run_tuning.py                 # run everything → one report
python3 research/run_tuning.py --only analog,analog_confirm
python3 research/run_tuning.py --skip aarambh_full     # skip the ~2h sweep
python3 research/run_tuning.py --skip-preflight        # bypass the data check
python3 research/run_tuning.py --preflight-warn        # check, but warn (don't abort)
```

**Preflight.** Every study pulls the *same* 9-year universe via `fetch_commodity_dataset`
(cache → live), so a single thin/rate-limited fetch would silently corrupt the whole
multi-hour run. Before any tier runs, `run_tuning.py` warms that fetch **once** and
asserts it's deep and broad enough — `≥ MIN_DATA_POINTS` rows, `≥ 50` numeric columns,
and `≥ 60%` of targets present. On failure it aborts immediately (nothing is tuned)
with a hint to wait out a rate-limit or run the app's "Refresh Data". Bypass with
`--skip-preflight`, or downgrade to a warning with `--preflight-warn`. (Scope: the
macro/target spine; Nirnay constituent baskets are fetched per-study with their own
fallback.)

Output: a consolidated, timestamped report in `research/reports/`, ending with a
**current-vs-validated reference** for every tuned constant (live value · which
study validates it).

## The studies

The orchestrator runs them **bottom-up through the architecture** — each tier is
tuned before the layers that consume it, so reading the report top-to-bottom, every
later result rests on decisions already settled above it.

| tier | key | script | informs |
|---|---|---|---|
| **T1 · engines** | `aarambh_full` | `aarambh_tuning_study.py` | `MIN_TRAIN_SIZE`, `MAX_TRAIN_SIZE`, `REFIT_INTERVAL`, `ENSEMBLE_MODELS`, PCA |
| | `aarambh_maxmin` | `confirm_max_sweep.py` | `MAX_TRAIN_SIZE` × `MIN_TRAIN_SIZE` interaction |
| | `nirnay` | `nirnay_tuning_study.py` | `NIRNAY_MSF_LENGTH`/`ROC_LEN`/`REGIME_SENSITIVITY`/`BASE_WEIGHT`/`MMR_NUM_VARS` |
| | `nirnay_index` | `nirnay_index_check.py` | Nirnay MSF on equity indices (generalization) |
| **T2 · scope** | `precedent_univ` | `precedent_universe_sweep.py` | `SIGNAL_HORIZONS` — which horizons the analog works at (frames T3) |
| **T3 · analog** | `analog` | `analog_tuning_study.py` | `ANALOG_W_*`, `TOP_N`, recency, feature set (1/10/20d) |
| | `analog_confirm` | `analog_confirm.py` | combined analog config (maha-only + drop-AvgZ) |
| **T4 · validation** | `precedent_model` | `precedent_vs_model_sweep.py` | purged model vs tuned analog by asset class |
| | `precedent_horiz` | `precedent_study.py` | model vs analog at 5/10/20/90d |
| **T5 · interpretation** | `markers` | `markers_study.py` | `UI_CONSENSUS_*`/`UI_CONVRAW_*`/`UI_NIRNAY_AVG_THRESHOLD` |
| | `hero` | `hero_study.py` | hero interpretation (convergence vs +markers vs +precedent) |

## Correctness tests (not tuning)

Fast, assertion-based regression tests for edge-case paths that ship unexercised by
the live universe. Run directly (`PYTHONPATH=. python3 research/<name>.py`):

| Test | Covers |
| --- | --- |
| `test_polarity.py` | `engines.nirnay.apply_polarity` inverse-basket path (`TARGET_POLARITY = -1`): pair swaps, sign negation, neutral-column preservation, no-op, involution, breadth conservation — no live target sets `-1` yet. |
| `test_calendars.py` | `data.calendars` freshness counting: every target resolves to a known exchange; holiday-aware counts beat the naive Mon–Fri mask across real holidays; weekday fallback reproduces `busday_count` byte-for-byte. |

## Shared conventions

- **Honest metric:** non-overlapping (stride = horizon) OOS Spearman IC, reported
  full-sample **and** recent-half (the recent-half is the test of "does it still
  work in the current regime").
- **Speed:** sweeps that only need relative ranking use the fast `ridge+ols`
  ensemble; the real `ols+huber` is reserved for the ensemble decision itself.
- **Data:** all studies read the cached real universe (`~/.cache/tattva`) offline.
- Each script is standalone (`python3 research/<name>.py`); a path shim at the top
  puts the repo root on `sys.path` so `from core...` resolves.
