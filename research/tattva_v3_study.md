# Tattva v3 — the study behind the rebuild

**Date:** 2026-09-30

**Data**
- 27 catalogue targets with yfinance history.
- About 25 world factors.
- Unadjusted daily bars since 2008.

**Protocol:** [`DESIGN.md`](../DESIGN.md), committed before any result (`fe8ef5c`).

**Score**
- **IC:** the per-target Spearman correlation with the next-open 10-day log return, taken on
  non-overlapping dates (every 10th session) and averaged across targets.
- **Periods:**

| Period | Dates |
|---|---|
| P0 | 2012 to mid-2019 |
| P1 | mid-2019 to 2021 |
| P2 | 2022-23 |
| P3 | 2024 to date |

This document reports everything that was run, in order, including what failed.

## 1. The pre-registered design failed

The design in `DESIGN.md` has one learner for both engines: a pooled, purged, hierarchical ridge
refitted monthly. The record of it is `research/scripts/engine_v3_preregistered.py` +
`learner_v3_rejected.py`.

| mean IC | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| Mūla (learned) | −0.058 | −0.017 | −0.016 | −0.016 |
| Swayam (learned) | −0.047 | −0.081 | +0.005 | −0.068 |
| **Tattva (stacked)** | **−0.022** | **−0.006** | **−0.018** | **−0.047** |

Its hit rate was 51% / 54% / 50% / 51%. **It fails rule 2** of `DESIGN.md` §6: the IC is negative
in every period.

## 2. Diagnosis: the evidence was there; the estimator misread it

Each raw feature was scored at a fixed sign (+ = the feature's own value):

| feature | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| gap60 (Mūla) | −0.063 | −0.071 | −0.052 | 0.000 |
| tsm60 | −0.059 | −0.045 | −0.045 | −0.035 |
| tsm250 | −0.063 | −0.082 | −0.085 | −0.061 |
| range60 | −0.042 | +0.001 | −0.030 | −0.023 |
| (the other 7) | \|IC\| < 0.04 in P0 | | | |

**Overextension reverts at 10 days, consistently.**
- tsm250 is negative in all 12 pool × period cells.
- gap60 is negative in 10 of 12.

The learned model still put a *positive* weight on tsm250 in later years. Tattva's question is
*timing within one target*, but the pooled regression also learned a *between-target* effect:
the assets that trended up for years also earned more. That effect has the opposite sign.

**Post-hoc 1 — within-target (target fixed effects).**
- The IC moved to −0.021 / +0.006 / +0.023 / −0.048.
- That is better, but it still fails.
- Learned combinations of weak, correlated features stayed unstable.

## 3. The adopted design (post-hoc, selected on P0 only)

**Selection rule.** Using **P0 only**, keep features with |mean IC| ≥ 0.04 and sign them by P0.
- That gives gap60 for Mūla, and tsm60, tsm250 and range60 for Swayam.
- The weights are equal, and there are no learned coefficients.
- Each feature is scaled by its own trailing RMS.

**Periods P1-P3.** These had been *seen* in §1-2, but they were not used to select anything.

| mean IC | P0 (selection) | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| Mūla = −gap60 | +0.044 | +0.069 | +0.050 | −0.003 |
| Swayam = −(tsm60, tsm250, range60) | +0.032 | +0.047 | +0.056 | +0.043 |
| **Tattva = ½(M + S)** | **+0.042** | **+0.062** | **+0.063** | **+0.011** |

On the 25 targets shared with the old system's end-to-end test, Tattva compares with the old
hero headline as follows:

| mean IC | P1 | P2 | P3 |
|---|---:|---:|---:|
| Tattva | +0.084 | +0.069 | +0.026 |
| Old hero | −0.145 | −0.010 | −0.010 |

### Calls need drift

Timing is *relative to the target's normal drift*, and a BUY or SELL is judged on the absolute
move. Calling the timing sign alone lost money: hit rate 47% / 46% / 51% / 46%.

**The adopted rule** is the fix stated before it was run:

    E[y] = μ + 0.05·z

- **μ** is the target's own decay-weighted mean outcome, using only outcomes already known.
- **0.05** is the P0 IC.
- The call is sign(E[y]), made only when |z| ≥ 0.5 and Mūla and Swayam agree.

| calls | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| **Tattva hit rate** | **55.2%** | **53.7%** | **56.9%** | **55.0%** |
| Tattva return / call | +0.28% | +0.82% | +0.23% | +0.41% |
| drift-only hit rate | 55.0% | 52.9% | 50.0% | 52.4% |
| drift-only return / call | +0.29% | +0.81% | −0.26% | +0.29% |
| always-long hit rate | 56.5% | 56.0% | 50.3% | 52.6% |
| always-long return / call | +0.34% | +1.40% | −0.13% | +0.40% |

By pool, the mean IC and hit rate over P0-P3 are:

| pool | mean IC | hit rate |
|---|---|---|
| equity (11) | +0.02 to +0.10 | 59-61% |
| macro (9) | +0.02 to +0.07 | 49-55% |
| crypto (7) | about 0 (−0.03 to +0.12) | 50-57% |

### What this does and does not show
- **Timing.** Tattva's timing adds information beyond drift in every period: its hit rate is at or
  above drift alone in all four.
- **Bull runs.** In the 2012-2021 bull runs, holding long had a higher hit rate and return than
  any call.
- **P3.** The 2024 onward IC is thin (+0.011). It is carried by Swayam; Mūla is flat there.
- **Crypto.** There is no timing edge on crypto.
- **Not a clean out-of-sample test.** The adopted design was chosen after the pre-registered one
  failed. P1-P3 were not used to select it, but they had been looked at. **The only clean test is
  forward.** The app's record keeps scoring every new call as its 10 days end.

## 4. Pre-registered rules applied to the adopted design (post-hoc)

- **Rule 1 (an engine has skill).** Mūla has a positive IC in 3 of 4 periods, and Swayam in 4 of
  4. ✓
- **Rule 2 (the combination ships).** Its IC is positive in every period ✓. It is at or above the
  old hero in P1-P3 ✓. Its call hit rate is above 50% in every period ✓.

## 5. Causality

`tests/test_causality.py` runs each of 6 targets on data truncated 250 and 600 sessions early.
Every published value (M, S, F, z, μ, E[y], call, expected move) must equal the full run's on every
shared date. **All 12 cases pass exactly.**

## 6. Universe notes
- **Dropped.** Sixteen NSE sector indices from the old catalogue are not listed: Auto, FMCG,
  Metal, Energy, Smallcap 100, Fin Services, Pvt Bank, PSU Bank, Realty, Media, Infra, PSE,
  Consumption, Commodities, MNC and Services. yfinance serves only the current bar for them, so
  there is nothing to forecast or to score.
- **Renamed.** The old "India Sector ETFs" target was priced on `^CRSLDX`, which is the Nifty
  500. It is now listed under that name.

## Scripts (`research/scripts/`)

| script | what it is |
|---|---|
| `engine_v3_preregistered.py` + `learner_v3_rejected.py` | the pre-registered design (§1), kept as the record |
| `score.py` | scores that design (§1, and §2 post-hoc 1) |
| `fixed.py` | the P0-selected fixed-sign composite (§3) |
| `calls.py` | the call-rule comparison (§3, calls) |
| `../evaluate_catalogue.py` | reproduces the adopted design's record from the package |

The scripts in `research/scripts/` ran from a scratch directory during the study, and their paths
point there. They are the record of what was run, not maintained tools. `evaluate_catalogue.py` is
the maintained one.
