# Pre-registration — can any reading of Mūla's gap earn a call?

Written and committed **before** `studies/run_study.py` was first run.
Registered 2026-11-11.

## Question

The shipped call (claim when |FVO| ≥ 0.5, contrarian) has no pooled edge
(`data/priors/call_record.json`). Does any of a fixed, short list of
alternative readings of the same valuation have one?

## Sample

- Targets: every catalogue target that is a price (sheet-sourced valuation
  ratios excluded) and passes the liveness check in `pipeline/prep.py`.
- History: from each target's first published valuation.
- **Holdout: the final 252 published sessions of every target.** No study
  script reads them, except `run_study.py --final <candidate>`, which may be
  run **once** per candidate, only for a candidate that cleared the
  development bar, and whose output is committed with the run.

## Grading

Exactly the production grader (`convergence.intelligence`): a claim dated *t*
is entered at the close of *t + EXECUTION_LAG* and held for the candidate's
horizon; each claim is scored in excess of always holding its direction on that
target; outcomes are divided by the target's forward-return SD and pooled
across all targets with `studies/pooled_record.pool` (one independent
observation per *h* claims per target, count halved for cross-target
dependence).

## Promotion bar

Development sample, pooled over all targets:

1. `t ≥ 2.0`, **and**
2. hit rate above the any-day base, **and**
3. positive mean excess in at least 4 of 6 consecutive calendar blocks.

Then, on the holdout (`--final`): mean excess > 0 and hit rate above base.

A candidate that clears both replaces the claim rule in `core/semantics.py`;
`studies/pooled_record.py` is then re-run and the licence in `core/trust.py`
decides, per asset class and side, whether it is printed as a call.

## Candidates (fixed)

| ID | Reading | Claim rule |
|---|---|---|
| C0 | the shipped call | contrarian, \|FVO\| ≥ 0.5, h = 10 |
| C1 | causally centred | FVO minus its own EW mean (half-life 126 sessions, lagged one row); contrarian, \|x\| ≥ 0.5, h = 10 |
| C2 | causal rank | FVO's expanding percentile among its own past (`GapPercentile`); buy below 0.20, sell above 0.80, h = 10 |
| C3 | cheap only | C0 restricted to buy claims |
| C4 | shorter hold | C0 at h = 5 |
| C5 | cross-sectional | each day, within an asset class with ≥ 3 valued targets, buy the cheapest third by FVO and sell the richest third, h = 10 |
| C6 | fast memory | C0 on an FVO refit with coefficient discounts (0.995, 0.998, 0.999) |

Seven candidates, one bar. Adding a candidate requires a new dated section in
this file **before** it is run, and raises the development bar for every
candidate added after this registration to `t ≥ 2.0 + 0.25 × (number added)`.

## What is not allowed

- Changing a candidate's rule, threshold or horizon after seeing its result.
- Reading the holdout for any purpose other than the one `--final` run.
- Reporting a per-class or per-target result as a promotion. Per-class
  figures are printed for information; the bar is the pooled one.
