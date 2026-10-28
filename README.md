# TATTVA — तत्त्व

**One claim, sized only by measured skill, graded as issued** · v2.9.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality": the underlying
> truth, kept apart from what is only claimed about it.

Tattva is a research terminal for a commodity, a currency, a crypto asset, an
equity index or a listed stock. For each one it issues **one signed call** on
the log return over the next 10 sessions, counted from the close after the
call is published:

- the **direction** comes only from Mūla's claim — the instrument is cheap or
  rich against the level the macro cross-section implies;
- the **size** comes only from measured, price-side skill: the claim's own
  grade, shrunk toward a pooled prior of its asset class, and two sizers that
  count only once their own grades earn it;
- the call is composed at **every past session** and **graded as issued** by the
  same grader that grades its parts, and that grade sits beside it on screen.

On the committed catalogue study no asset class has a pooled, price-side
claim edge at 10 sessions, so the honest output is **HOLD** (or **NO CLAIM**
inside the noise) almost everywhere — and the page says why, in numbers. See
[What it will usually say](#what-it-will-usually-say-hold-and-why).

It runs on free **yfinance** data, plus two published Google Sheets for the
sheet-sourced targets. No API keys, no secrets, no database.

**Where to start.** [Quickstart](#quickstart) gets it running.
[ARCHITECTURE.md](ARCHITECTURE.md) is the system in one place: the objective,
every role, the data flow, the grader, the pooled prior, the registries and
the tests behind each contract. [CHANGELOG.md](CHANGELOG.md) says what 2.9.0
removed and why.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Test (offline, a few minutes; --runslow adds the extra full-panel fits)
pip install pytest && python -m pytest tests -q
```

In the control rail pick an **Asset Class** and a **Target** (or type a stock
symbol under India / US Stocks) and click **Run Analysis**. The first run
fetches the history from 2017-01-02 (cached for an hour); switching target
re-runs on the panel already in session. **Refresh data** forces one live pull;
**Start over** clears the session's results. Nothing about the model is
user-configurable, by design.

---

## One claim, sized only by measured skill, graded as issued

At Mūla's latest published session d*, Tattva issues

    w = d · size,      size = trust · exp(β_swayam · z_swayam) · exp(β_reversion · z_reversion)

on Y = log P(t+1+h) − log P(t+1), h = 10.

- **d** — Mūla's claim: CHEAP → +1 (BUY side), RICH → −1 (SELL side), and 0 when
  any precondition fails: the valuation was not published, the target did not
  print or is a full horizon behind, the target is not tradeable (a valuation
  ratio), or |FVO| is inside the ±0.5σ noise.
- **trust** — f(t_post of the claim), the claim's earned base size, 0 until the
  claim's price-side grade clears the bar.
- **each sizer** — exp(β·z) with β = f(t_post of the sizer): exactly ×1.00 while
  unskilled, and also whenever its reading is absent or carried. A factor is
  positive and unsigned, so a sizer scales the claim and can never veto or
  invert it. *Swayam confirmation* reads z = d · Lean, the breadth of Swayam's
  view bank on a native Swayam row; *Reversion (ECM)* reads z = 2·MRProb − 1,
  where MRProb = P(price-side κ̂ > 0) is the error-correction layer's evidence
  that price responds to the gap at all (0.5 carries no information). How
  much of the closing price does is measured separately, by the gap split
  ([Who closes the gap](#who-closes-the-gap)).
- **f(t)** = clip((t − 1.645)/(2.5 − 1.645), 0, 1): one map for trust and every β.
- **size** is a unitless index — 1.00 is a fully trusted claim with neutral
  sizers — never a percentage. The actions: NO CALL, NO CLAIM, HOLD (trust 0),
  BUY · size s, SELL · size s. No tier words until the study shows that a
  larger size pays more.

---

## The role grammar

Every part of the system has exactly one role in producing, grading or
auditing the call (`core/roles.py`), and every card on every page is
registered to one — its chip says which:

| Role | What it is | Chip |
|---|---|---|
| CALL | the call and its size | CALL |
| CLAIM | Mūla's direction, the only source of sign | CLAIM→call |
| PRECONDITION | a binary condition that voids the claim and never sizes it | PRECONDITION→claim · voids the claim, never sizes it |
| SIZER | a factor exp(β·z) whose β is earned by its own grade | SIZER→call · ×1.00 until its own grade earns a weight |
| GRADE | a measured skill number, and what it moves (trust is the claim's grade, mapped once) | GRADE→… |
| EVIDENCE | shown to audit one role-holder; a lean carries its own grade | EVIDENCE→… · does not move the call |
| INFRASTRUCTURE | what guarantees one as-of session, one horizon and causality | INFRASTRUCTURE→… |

Directional colour belongs to the claim's state and the call; evidence is drawn
neutral, with its IC ± SE beside it. Pages read the bundle and the verdict
only: they run no estimator and draw no threshold of their own.

---

## The pages

- **Overview** — the call in decision order: the action and one sentence built
  from the verdict; the call's grade as issued (IC ± SE on the sessions it was
  issued, coverage, hit against the side-matched base, issued against held
  payoff, the prospective record — or 'not yet graded: k of 30' beside the
  class's pooled call grade); the chain (claim · preconditions · trust ·
  Swayam · Reversion · the size's arithmetic); who closes the gap; the call's
  history.
- **Mūla** — the claim and what it has earned: Overview's own chain rows; the
  claim's posterior t at every session against 1.645 and 2.5, own grade beside
  the class prior, period by period and by side; the evidence (who closes the
  gap, price against fair value with its band, the claim's unit, gap
  stationarity, model fit, what drives fair value); then the reversion sizer
  and its gap-implied move.
- **Swayam** — the confirmation sizer: its reading (native or carried), z, own
  grade and prior, t_post, β and factor; the claim's drift-neutral payoff when
  Swayam contradicts, splits or confirms; the breadth over time, the view bank
  and each view's own oscillators.
- **Grades** — is the loop working? One table of every grade (object, role,
  consumer, own IC ± SE, n_win, hit against base, the prior with its scope, k
  and vintage, t_post, trust or β, label); the trust and β paths; payoff by
  issued size; and the pooled-prior study's results, read-only.
- **System** — d*, h and L, the epoch, the history's fingerprint and whether it
  is the history the priors were studied on; the priors' provenance; data
  health; the KNOBS and ROLES registries.
- **Data** — the record: Mūla's frame, Swayam's breadth and every column of the
  call at every session, joined by date with each column's role, exportable.

---

## How trust is earned

**Own grade.** One grader (`core/grade.py`) scores every role object: the rank
IC of its lean against the drift-neutral 10-session return from the next close
(the causal mean return is taken out, so a trend is not skill), sampled every
10 sessions so no two windows overlap, with an SE from a circular-shift null.
As of each session it uses only outcomes that had resolved by then, and it
says nothing until 30 non-overlapping windows have.

**⊕ class prior.** A single instrument's 10-session grade is too noisy to
certify a real but modest edge, so the owner's ruling is to shrink it toward
its asset class. One pre-registered catalogue study
(`studies/pooled_priors.py`) runs the normal pipeline on every tradeable
catalogue target and ten large stocks and pools their grades — leaving each
member out of its own prior — into `data/priors/pooled_priors_h10_L1.json`,
which the app reads as data. A class with fewer than three other members
pools the whole catalogue instead.

**→ posterior t.** The two are combined as two correlated measurements of the
same IC (GLS, with the error covariance the class shares with the member), and
the result can never be more precise than the whole class pool.

**→ f.** Trust = f(t_post): 0 below 1.645, full at 2.5. The same for each
sizer's β. Every HOLD prints its t_post, the bar, its own grade and the pool it
was shrunk toward, named with the study's date.

---

## The edge table

Generated from the committed study by `studies/render_docs_table.py`; the test
suite holds this block to that file byte for byte.

<!-- edge-table:begin (generated by studies/render_docs_table.py; do not edit) -->
Study `studies/pooled_priors.py` run 2026-10-27 on code `25880c3` (script SHA-256 `0a4fe565e974…`), h = 10, L = 1: 37 members, 26 held out for the decisions; not run: Jeera.

**Pooled grade by class** at the last vintage (2026-09-30): the class's own graded members pooled, μ ± se_μ, the catalogue-wide τ and the members k.

| Class | Members | Claim | Swayam sizer | Reversion sizer | Call | Vintage |
|---|---|---|---|---|---|---|
| Commodities | 7 | −0.012 ± 0.027 · τ 0.000 · k 7 | +0.013 ± 0.016 · τ 0.015 · k 7 | −0.021 ± 0.027 · τ 0.000 · k 7 | not graded | 2026-09-30 |
| Crypto | 7 | −0.027 ± 0.033 · τ 0.000 · k 7 | −0.023 ± 0.020 · τ 0.015 · k 7 | −0.047 ± 0.027 · τ 0.000 · k 7 | not graded | 2026-09-30 |
| Currency (FX) | 2 | +0.075 ± 0.043 · τ 0.000 · k 2 | −0.006 ± 0.026 · τ 0.015 · k 2 | −0.068 ± 0.044 · τ 0.000 · k 2 | not graded | 2026-09-30 |
| India Indices | 8 | −0.057 ± 0.039 · τ 0.000 · k 8 | +0.013 ± 0.023 · τ 0.015 · k 8 | −0.008 ± 0.031 · τ 0.000 · k 8 | not graded | 2026-09-30 |
| India Stocks | 5 | −0.027 ± 0.036 · τ 0.000 · k 5 | +0.025 ± 0.020 · τ 0.015 · k 5 | −0.001 ± 0.030 · τ 0.000 · k 5 | not graded | 2026-09-30 |
| US Stocks | 5 | −0.006 ± 0.032 · τ 0.000 · k 5 | +0.013 ± 0.019 · τ 0.015 · k 5 | +0.006 ± 0.028 · τ 0.000 · k 5 | not graded | 2026-09-30 |
| US Indices | 3 | +0.050 ± 0.046 · τ 0.000 · k 3 | +0.058 ± 0.029 · τ 0.015 · k 3 | −0.127 ± 0.050 · τ 0.000 · k 3 | not graded | 2026-09-30 |

No class has a pooled call grade: nothing graded at the last vintage.

**The claim prior each member reads** at 2026-09-30: its leave-one-out entry, and the t that prior gives on its own, μ / max(√(τ² + var_μ), se_floor). Trust opens at a posterior t of 1.645.

| Class | Scope | Prior μ | t of the prior alone | Highest t |
|---|---|---|---|---|
| Commodities | class-LOO | −0.017 … −0.001 | −0.63 … −0.05 | Gold |
| Crypto | class-LOO | −0.050 … +0.003 | −1.35 … +0.10 | Bitcoin |
| Currency (FX) | global-LOO | −0.019 … −0.012 | −1.01 … −0.67 | Dollar Index |
| India Indices | class-LOO | −0.063 … −0.051 | −1.61 … −1.30 | Nifty IT |
| India Stocks | class-LOO | −0.050 … −0.009 | −1.31 … −0.24 | ITC (NSE) |
| US Stocks | class-LOO | −0.025 … +0.011 | −0.64 … +0.35 | MSFT (US) |
| US Indices | global-LOO | −0.017 … −0.014 | −0.93 … −0.77 | Nasdaq 100 |

No member's claim prior reaches t 1.645 on its own: they run from −1.61 (Nifty 100) to +0.35 (MSFT (US)).

The Currency (FX) class's own claim summary, +0.075 ± 0.043 on k 2 (USD/INR, Dollar Index), is 1.75 standard errors from zero, but with fewer than 3 other members no member reads it: each reads the global-LOO pool.

**Who closes the gap** (arm G): over 10 sessions, the share of the open gap that price closed, and the slopes of the 10-session change in log fair value and in log price on the gap — the mean over a class's members, with the SD across them.

| Class | k | Price's share of the closing | Fair value's slope on the gap | Price's slope on the gap |
|---|---|---|---|---|
| Commodities | 7 | +5.5% (SD 9.3%) | +0.342 (SD 0.048) | −0.024 (SD 0.033) |
| Crypto | 7 | −5.6% (SD 10.4%) | +0.445 (SD 0.096) | +0.024 (SD 0.050) |
| Currency (FX) | 2 | +11.8% (SD 9.7%) | +0.353 (SD 0.035) | −0.048 (SD 0.039) |
| India Indices | 8 | −10.4% (SD 4.5%) | +0.389 (SD 0.042) | +0.036 (SD 0.013) |
| India Stocks | 5 | −2.2% (SD 11.0%) | +0.381 (SD 0.032) | +0.007 (SD 0.043) |
| US Stocks | 5 | +0.9% (SD 8.4%) | +0.339 (SD 0.053) | −0.003 (SD 0.029) |
| US Indices | 3 | +2.5% (SD 3.3%) | +0.333 (SD 0.018) | −0.009 (SD 0.011) |

Class by class, price did between −10.4% (India Indices) and +11.8% (Currency (FX)) of the closing; fair value moving toward price did the rest.

**Decisions**, on the held-out members only:

| Decision | Value | Measured |
|---|---|---|
| A · claim centring (recorded, never applied) | `none` | raw FVO fails the C12 unit contract on 21; the candidate `robust_z_252` passes on 13 of them (adoption needs 2/3); paired ΔIC −0.000 ± 0.010 |
| B · SIZERS_ENABLED | `True` | ΔIC(d·∏factor − d) +0.000 ± 0.001 on 26 members (released iff ≥ −1 SE); call vs bare claim −0.116 ± 0.091 on 1 member (information only) |
| C · TIER_RULE | `none` | 1 held-out member issued calls; n/h per tercile 0 / 0 / 0 (needs 20 each) |

Checks: R4, the FFT shift-null SE against a brute-force null (median ratio, band 0.85 – 1.15): claim 1.020, sizer:swayam 0.956, sizer:reversion 0.991 — pass; R6, the bootstrap pool variance against the closed form (median ratio, bar 1.5): claim 1.05, sizer:swayam 0.90, sizer:reversion 1.12 — pass.
<!-- edge-table:end -->

---

## Who closes the gap

Mūla's claim is a price-side claim: CHEAP means price should rise toward fair
value. Whether that happens is measured, not assumed. While the gap was open,
regress the 10-session change in log price and in log fair value on the gap:
price's share of what closed is −(price's slope)/(fair value's slope − price's
slope). The edge table's arm-G rows give it per class; class by class, price
did a small share of the closing, of either sign, and fair value moving toward
price did the rest. Overview and Mūla show the same split for the instrument
on screen, with its bootstrap SE. Tattva therefore acts only when the claim's
price-side grade clears the bar.

---

## What it will usually say: HOLD, and why

The edge table gives the reason in numbers. No member's leave-one-out claim
prior reaches t 1.645 on its own, so the posterior can open only where an
instrument's own grade is strong enough to carry it there against its pool —
and the pools are near zero or negative. The one class summary that clears
1.645 on its face (Currency (FX), two members) is not read by either member:
with fewer than three others in the class, each reads the catalogue-wide pool.

What the app said on the four audited instruments, live, on 2026-10-27 (the
network harness, committed priors):

| Target | Overview | Claim t_post (own; prior) |
|---|---|---|
| Gold | HOLD — 0.89σ (5.4%) below the level the macro cross-section implies, but the claim has not been shown to pay on price | −0.35 (own −0.059 ± 0.057, n_win 162; Commodities class-LOO prior −0.001 ± 0.028, k 6) |
| Nifty 50 | NO CLAIM — 0.37σ (2.4%) above, inside the ±0.5σ noise | −1.37 (own −0.090 ± 0.056, n_win 152; India Indices class-LOO prior −0.051 ± 0.039, k 7) |
| S&P 500 | NO CLAIM — 0.25σ (1.8%) above, inside the ±0.5σ noise | −0.86 (own +0.051 ± 0.055, n_win 158; global-LOO prior −0.016 ± 0.018, k 36) |
| USD/INR | NO CLAIM — 0.06σ (0.1%) below, inside the ±0.5σ noise | +0.03 (own +0.144 ± 0.056, n_win 140; global-LOO prior −0.019 ± 0.019, k 36) |

USD/INR shows the price of pooling plainly: its own claim grade is more than
two standard errors from zero, but a member of a two-member class reads the
catalogue-wide pool (36 others, −0.019 ± 0.019), and the posterior lands near
zero. The design accepts that trade, and the Grades page shows what it costs:
each class's power and false-open rate at its own size (the study's R5). A HOLD
here is the measured answer, not a fault.

---

## Owner sign-offs

These are the owner's calls, recorded as the design recommended; each is a
one-knob change:

- **EXECUTION_LAG = 1.** A call computed after the close of t is graded from
  the close of t+1, which amends the charter's estimand. A veto sets it to 0
  and re-runs the study (the priors file records L; a file for another L is
  refused).
- **Decision A recorded as 'none'.** The study tested a centred claim unit
  (FVO against its trailing year) on the held-out members; it did not meet
  its adoption rule, and in any case it is never applied without a separate,
  owner-signed change. The claim's unit is reported as evidence on Mūla.
- **'Nifty 50 - PE' kept as NO CALL.** A P/E is a valuation of an index, not a
  traded price: it is valued and its claim graded as evidence, and never
  called.
- **Jeera is outside the study.** Its Google-Sheets source was unreachable
  when the study ran, so it has no entry of its own; as a non-member it reads
  the Commodities class's all-member prior.
- **A near-permanent HOLD** is accepted as the honest output on today's
  evidence, not a defect.

---

## What was removed, and how it may come back

2.9.0 deleted every part that could not earn a role (the code is recoverable at
commit 2755b4c, 2.8.0 with its verification fixes):

- **The gate chain and its conviction** — the gate ramps, the tier words, the
  '%' conviction and the 'nothing limiting' rule. Their influence ran inverse
  to their skill; replaced by trust × exp(β·z).
- **Convergence** (agreement, the learned composite and its smoother, the
  consensus, the divergences, Engine Agreement) — Mūla-signed on almost every
  row, it moved nothing and contradicted the call on screen. Comes back only on
  inputs that exclude Mūla's lean and |FVO|, and only by beating the Swayam
  sizer in the pooled study.
- **Precedent** (analog matching) — a separate model on the claim's own
  features. Comes back only as a sizer on causally FVO-residualised
  momentum, realised volatility and stress, through the sizer rule.
- **Mūla's cross-section consistency, window stack, regime words, OU
  projection and smoothers** — separate models or second copies of FVO. A
  different claim unit comes only through Decision A and an owner-signed
  change; the robustness question is the study's panel jackknife.
- **Swayam's regime engine, learned view weights and averaged signal** — a
  learner whose loss rewarded drift, and duplicates. Comes back only through the
  learner rule: causal, trained on the drift-neutral payoff at h, beating its
  baseline in the pooled study, passing the planted-drift ranking test.
- **The ticker tape, the sidebar weights and the walk-forward cards.**

The rules for any new part are in
[ARCHITECTURE.md](ARCHITECTURE.md#admission-and-re-entry-rules): evidence must
audit a named role-holder from its own outputs, a sizer must be
lineage-disjoint from the claim and earn β in the study, and nothing is shown
twice.

---

## Data sources

- **Target and predictors:** the target's price plus the registered macro
  universe in `core/config.py` (`GLOBAL_MACRO_MAP`, `MACRO_SYMBOLS_YF`) and the
  index levels. Mūla's cross-section is that registered list minus raw yield
  levels, sheet series, the target, its near-replicas and its declared
  currency twins (`core/instruments.py`).
- **Swayam input:** the target's own OHLCV on its own trading sessions, with a
  macro driver pool from the same panel, minus the target and its near-replicas.
- **Sheet-sourced targets** (Jeera, Nifty 50 - PE) come from published Google
  Sheets; they have no OHLCV feed, so they run without Swayam (×1.00).

Every external call goes through a two-tier cache (memory + disk), a
per-service consecutive-failure circuit breaker, retry-with-backoff, a
partial-success re-fetch, and a stale-snapshot fallback. Freshness is counted
in trading days on the target's own exchange calendar (`data/calendars.py`).

---

## Configuration

| What | Where |
|---|---|
| Every constant that can move the call | `KNOBS` in `core/knobs.py` (kind, consumer, horizon coupling, reason) |
| Every role, card and chip | `core/roles.py` |
| What each target is (class, currency, tradeable, twins) | `core/instruments.py` |
| How each target is configured (horizon, view bank) | `InstrumentConfig` / `INSTRUMENT_CONFIGS` in `core/config.py` |
| Catalogue | `COMMODITY_TARGETS` in `core/config.py`, `INDEX_TARGETS` in `data/universe.py` |
| The pooled priors | `data/priors/pooled_priors_h10_L1.json`, written by `studies/pooled_priors.py` |

---

## Project structure

```
app.py                  Streamlit entrypoint: the run's phases and the page shell
core/                   call (the composer and the verdict), grade (the one grader),
                        trust (posterior, strength map, priors), roles / knobs /
                        instruments (the registries), semantics (signs and words), config
pipeline/               run (step functions shared by the app and the study), assemble
                        (the closed loop), evidence (admitted evidence)
data/                   fetchers, the pinned history (prep), calendars, caches, circuit
                        breakers, sheet sources; priors/ (the committed study output)
engines/mula/           valuation: online factor model, asset-class blocks, dynamic
                        cointegrating regression, error-correction layer
engines/swayam/         breadth: the per-series kernel and the equal-weight view bank
analytics/              causal online primitives
studies/                the pooled-prior study and its arithmetic, the docs' edge table,
                        the SE calibration, the fixture builder
ui/                     theme, components, one module per page
tests/                  offline: the charter's clauses and the tests behind them; fixtures/
vault/                  archives of the pre-August-2026 tree
```

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Its
outputs are statistical readings whose measured out-of-sample edge is, on
today's evidence, none that clears its own bar. Do not make trading or
investment decisions solely on this software's output. See
[LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
