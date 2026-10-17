# TATTVA — तत्त्व

**Unified Convergence Engine** · v2.8.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality": the underlying
> truth distilled from the convergence of evidence.

Tattva is a research terminal that makes **one graded call** about a target — a
commodity, a currency, a crypto asset, an equity index or any listed stock — for
roughly the next ten trading sessions. One engine values the target against the
whole traded macro cross-section; a second reads the target's own price from
fifteen angles; the call is the first engine's, and six independent conditions
say how far to trust it, with the weakest one named.

It runs entirely on free **yfinance** data. No API keys, no secrets, no database.

**Where to start.** [What it does](#what-it-does) is the one-screen version and
[Quickstart](#quickstart) gets it running. [How the model works](#how-the-model-works)
is the argument — what is estimated, what is declared, and what is deliberately
not claimed. [Reading the output](#reading-the-output) is what to look at once a
run finishes, including where the edge is not.

---

## What it does

| Engine | Question it answers | How |
|---|---|---|
| **Mūla** | *Where should this be trading, given the state of the world — and does price go there?* | A recursive **dynamic cointegrating regression** of log price on the integrated factors of ~200 macro instruments (and on named asset-class blocks), with time-varying coefficients. Publishes a fair-value **level**, the gap to it, and the **oscillator** (the gap in units of its own typical size) — the one call Tattva makes. An **error-correction** layer then asks the question a trade depends on: does the *price* close the gap, or does fair value move to price? |
| **Swayam** | *What does the target's own price say about itself?* | MSF + MMR oscillators with HMM/GARCH/CUSUM regime detection, run as a 15-view bank (timescale × information set × mechanism) on the target's **own** OHLCV, each view weighted by its own realised skill. Published as **net breadth**: the share of views reading the target oversold minus the share reading it overbought. |
| **Convergence** | *Do the two agree, and has agreement been worth anything?* | Four agreement dimensions (direction, breadth, magnitude, regime) weighted **online** by each dimension's own signed skill, a weighted composite with a drift-diffusion filter, a zero-anchored consensus, and the divergences between the engines. |
| **Precedent** | *What followed states like today's?* | Covariance-aware **Mahalanobis** analog matching (OAS shrinkage) over the system's own state, under a **Theiler exclusion window** so analogs are distinct episodes — a base rate across a fixed **1/3/5/10/20/60d** term structure. |

The headline is the **conviction chain** on the Overview page: Mūla's call
(cheap → bullish, rich → bearish, inside ±0.5σ → no claim), and six gates whose
product is the conviction —

| Gate | Question | Reads |
|---|---|---|
| Mispricing | Is it mispriced at all? | \|oscillator\|, open from 0.5σ to 2σ |
| Reversion | Does **price** close the gap? | P(κ̂ > 0), the price-side error-correction coefficient |
| Cross-section | Does the whole world agree? | leave-one-block-out agreement on the gap's sign |
| Convergence | Do the target's own internals confirm it? | Swayam net breadth, aligned with the call |
| Edge | Has this call paid out of sample? | Mūla's walk-forward IC at the horizon |
| Precedent | Did it pay the last times this state occurred? | the analog base rate, only when it is usable |

The signal word follows the arithmetic: **STRONG BUY / BUY / LEAN BUY** (and the
SELL side) at high / moderate / low conviction, and **HOLD** whenever the gates
do not support acting — with the reason on the card.

Every engine is graded the same way: the rank IC of its bullish-positive lean
against the h-day forward return, on **non-overlapping** windows, period by
period. Every engine page leads with its answer and its own grade.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Test (offline, ~15 s)
pip install pytest && python -m pytest tests -q
```

In the control rail pick an **Asset Class** and a **Target** (or type a stock
symbol under India / US Stocks) and click **Run Analysis**. The first run fetches
~9 years of daily history (cached for an hour); switching target re-runs the
engines on the panel already in session. **Refresh data** forces one live pull;
**Start over** clears the session's results.

The rail is grouped by scope — **Instrument** → **Model** (a read-only readout:
the learned dimension weights and Mūla's walk-forward edge) → **Session** →
**Appearance** (Paper, the light default; Slate, the dark one). Controls that
reframe a chart live in that chart's panel header, so a control's position
tells you its scope. Nothing about the model is user-configurable, by design.

---

## How the model works

**Valuation, in levels.** Mūla regresses **log price** on the *integrated* common
factors of the macro cross-section with time-varying coefficients:

```
p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t ,   F_{j,t} = sum_{s<=t} f_{j,s}
```

This is a dynamic cointegrating regression (Bierens & Martins 2010): `p` and `F`
are both integrated, and `e` is the deviation from the time-varying long-run
relation — the mispricing. The residual is a **level**, so fair value is a price
rather than a forecast, and the gap is a genuine mean-reverting spread. Two
independent views are averaged by their own out-of-sample predictive evidence: a
**latent** view on the principal factors of the cross-section and a **block**
view on named asset-class aggregates (equity, rates, credit, inflation, energy,
metals, agriculture, currency, volatility, real assets). Leave-one-block-out
refits give ablation-based driver importance and the **cross-section** gate.

**A stationary gap is not yet a trade.** The gap is stationary on every
instrument tested (ADF p ≈ 0, half-life ~1–2 weeks) — but measured on Gold,
S&P 500 and USD/INR, price accounts for only 2–24% of daily gap closure; the
rest is fair value moving to price. So the reversion gate does not read the
gap's own stationarity. It reads the **error-correction** layer: each day's price
change regressed on yesterday's gap (Engle & Granger 1987), on differenced data
where fast memories are legitimate. κ̂ > 0 means price moves toward fair value;
the gate is the probability that it does, from the coefficient's own posterior.
The layer also publishes the h-day price move that regression implies, with its
h-day SD, and pools a valuation-led, a momentum-led and a full forecast of the
price change by discounted predictive likelihood (Geweke & Amisano 2011).

**What is deliberately not claimed.** One step ahead, yesterday's close beats any
valuation of a near-integrated price, so the engine is *not* scored against a
random-walk null. Its level fit is scored against the honest competitor — the
asset's own 252-day trailing mean (R² vs Trailing Mean) — and its **call** is
scored by its walk-forward IC, like every engine's.

**One sign vocabulary.** Every reading keeps one orientation: positive = stretched
up (Mūla's oscillator above zero is rich; Swayam's oscillator above zero is
overbought). The call is contrarian on all of them. That inversion is written
once, in `core/semantics.py` — a *reading* (native sign, what charts show) and a
*lean* (bullish-positive, what anything scored against a forward return uses) —
and every layer imports it.

**One horizon.** Tattva reads a single **10-day** horizon on daily bars. The
Precedent page shows a fixed 1/3/5/10/20/60d term structure on purpose, with the
matcher's own walk-forward skill at each horizon, so where its edge fades is
visible rather than hidden behind a truncated grid.

**Estimated, not tuned.** A classification cut-point is the causal empirical
quantile of the signal's own past; a weight is the exponentially-discounted
realised skill of the thing being weighted (`analytics/adaptive.py`). Each
constant has a **warm-up prior**, so an instrument's first year runs on the
declared value and its own estimate takes over once it is better informed.
What stays declared is *structure* — the horizon, the view bank, the discount
grid, the estimability floors — because those are choices about the question.
Still hand-set, and said so: the drift-diffusion filter constants and the
Swayam kernel knobs.

**The engine never looks ahead, and it is asserted.** Every published value is a
function of data available at its own date, so re-running on more data extends
the record rather than rewriting it. `tests/test_causality.py` fits Mūla on
`data[:T]` and on `data[:T-200]` and requires the two to agree **exactly** on
every shared date, for fair value, the oscillator, the error-correction outputs,
the window agreement and the confidence family; `tests/test_edge_and_learners.py`
does the same for Swayam's view weights. A component that consulted the future
cannot pass, and neither can one that quietly stopped producing anything.

**Two things can still move a past reading, and neither is look-ahead.** *The
newest bar is still forming* until its session closes; it is published flagged
`Provisional` and excluded from everything that learns or matches. *The panel's
composition can change between runs* — a rate-limited fetch, a holiday, an
instrument admitted for the first time — and the factor basis is estimated from
whichever instruments are present. The panel fingerprint printed in the run
console is how a composition change is detected; a predictor refilled from a
cached snapshot is named in a warning on the page. Free-form stock symbols never
enter the shared panel, so another user's typing cannot move your history.

**Why the coefficient memory is slow.** Scoring discount factors by one-step
predictive likelihood is degenerate for a *level* regression: the model that
tracks price most closely always wins, and its limit is "fair value = price". The
grid is therefore restricted to implied memories of ~4y, ~8y, ~40y and permanent
— a modelling commitment, not a tuned choice. (The error-correction layer, on
differenced data, uses a fast grid and lets the data choose.)

**Honest validation.** Every engine's edge is `convergence.intelligence.series_edge`:
the Spearman IC of its lean against the h-day forward log return, sampled every h
sessions (averaged over the h phase offsets) so no two scored windows overlap,
reported with its standard error, its hit rate beside the majority-direction base
rate, and per-period ICs. Nothing is fitted to produce it. A durable edge is
positive in most periods, not large in one; with ~130 non-overlapping calls one
standard error is about ±0.09, and the page prints it.

---

## Data sources (all yfinance)

- **Target & predictors:** the target's price plus the macro universe in
  `core/config.py` — `GLOBAL_MACRO_MAP` (bond, rates, credit, equity, sector,
  style, volatility, real-asset ETFs) + `MACRO_SYMBOLS_YF` (commodities + FX).
  Each target excludes its own near-replicas (GLTR for gold, the dollar's
  component crosses for DXY, the whole US equity complex for a US index —
  VTV + VUG alone explain 99.5% of the S&P 500's daily returns).
- **Swayam input:** the target's **own** OHLCV. Its MMR driver pool is the same
  panel Mūla values against, minus the target and its near-replicas.
- **Sheet-sourced targets** (Jeera, Nifty 50 - PE) come from published Google
  Sheets; they have no OHLCV feed, so they run without Swayam.

Every external call goes through a two-tier cache (memory + disk), a
per-service consecutive-failure circuit breaker, retry-with-backoff, a
partial-success re-fetch, and a stale-snapshot fallback. A symbol that simply
does not exist is a distinct "no data" answer — not retried, and not counted
against the service. Freshness is counted in trading days on the target's own
exchange calendar (`data/calendars.py`, `exchange_calendars` optional).

---

## Configuration

Each catalogue target carries its own `InstrumentConfig` (`core/config.py`):
structure (leakage exclusions, view bank, horizons), estimability floors, and
the warm-up priors its own adaptive tiers start from. Targets inherit their
asset class's defaults (`CLASS_CONFIG_DEFAULTS`); `_PER_INSTRUMENT_OVERRIDES`
holds the few genuinely instrument-specific values. Free-form stocks are
configured per market (`STOCK_CONFIGS`).

| What | Where |
|---|---|
| Catalogue (commodities, FX, crypto) | `COMMODITY_TARGETS` in `core/config.py` |
| Index targets (India / US) | `INDEX_TARGETS` in `data/universe.py` |
| Per-instrument config | `InstrumentConfig` / `INSTRUMENT_CONFIGS` in `core/config.py` |
| Leakage exclusions | `TARGET_EXCLUDED_PREDICTORS` → each config's `excluded_predictors` |
| Mūla floors, memory grid, windows | `FVO_BURN_IN`, `FVO_MIN_PRINTS`, `FVO_VALUATION_DELTAS`, `LOOKBACK_WINDOWS` |
| Sign vocabulary and valuation states | `core/semantics.py` |
| Asset-class blocks for the cross-section | `engines/mula/blocks.py` |

**The catalogue is what can run.** Sixteen NSE sector indices that yfinance
serves only one or two bars for are not listed; the broad Nifty 500 is listed
under its own name.

---

## Project structure

```
app.py                  Streamlit entrypoint: data prep, the 5-phase pipeline, page shell
core/                   config (universe, targets, per-instrument config), semantics
                        (the sign vocabulary), logging
data/                   yfinance fetchers, index catalogue, two-tier cache, circuit
                        breakers, per-exchange trading calendars, sheet sources
engines/mula/           valuation: online factor model with a Marchenko-Pastur cut,
                        block map, dynamic cointegrating regression, error-correction
                        layer, market-regime filter
engines/swayam/         breadth: the per-series MSF/MMR/regime kernel and the
                        skill-weighted self-referential view bank
analytics/              causal primitives (DLMs, DMA, online AR(1)), adaptive tiers and
                        skill weights, robust z, OU, the regime kernel, analogs
convergence/            cross-validator, online weights + series_edge, consensus,
                        divergences, the composite's drift-diffusion filter
ui/                     theme, components (incl. the conviction chain), one module per page
tests/                  offline assertions: causality, signs, ECM arithmetic, edge
                        recovery, the conviction chain's rules
vault/                  archives of the pre-August-2026 tree (the git history starts later)
```

---

## Reading the output

- **Overview** — the call and its conviction, then the chain: each gate with the
  number behind it, the binding constraint named, standing risks below.
- **Mūla** — the valuation call; price against fair value with the oscillator and
  its ±0.5σ no-claim band; the error-correction read (does price close the gap?);
  model quality; the call's walk-forward edge; what drives fair value; how broad
  the mispricing is across 1-month / 1-quarter / 1-year windows.
- **Swayam** — net breadth, the effective number of independent views (fifteen
  views of one price are never fifteen opinions), its own edge, which views carry
  weight, the regime underneath.
- **Convergence** — agreement against the target's own history, the consensus,
  the learner's weighted composite and its edge, and the divergences table.
- **Precedent** — the base rate (with its distinct-episode count and a usability
  flag), the matcher's skill at every horizon, the closest analogs.
- **System** — every engine's edge side by side, what the learner concluded, the
  gap's stationarity statistics, data-layer health. **Data** — the joined record,
  exportable.

**Where the edge is not.** Measured by the app itself on live data, 2026-10-17 —
non-overlapping 10-day IC (periods positive of 6); one standard error is about
±0.09 for Mūla and the composite and ±0.08 for Swayam:

| Target | Mūla's call | Swayam net breadth | Weighted composite |
|---|---|---|---|
| Gold | −0.005 (5/6) | +0.056 (5/6) | +0.023 (5/6) |
| S&P 500 | −0.001 (4/6) | +0.055 (5/6) | +0.026 (4/6) |
| Nifty 50 | +0.078 (4/6) | +0.057 (5/6) | +0.078 (4/6) |
| Nifty 500 | +0.047 (4/6) | +0.041 (4/6) | +0.052 (4/6) |
| USD/INR | +0.044 (3/6) | −0.009 (3/6) | +0.042 (3/6) |
| Bitcoin | −0.056 (2/6) | −0.023 (2/6) | −0.062 (1/6) |
| AAPL | +0.088 (5/6) | −0.006 (2/6) | +0.064 (4/6) |

Every figure is within about one standard error of zero. What is consistent is
modest: Swayam's contrarian breadth is positive in five of six periods on gold
and both equity benchmarks, and nothing works on Bitcoin. These are the numbers
the Edge gate and every engine page print; read conviction against them, and
trust a period-by-period record that stays positive over any single reading.

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Outputs are
statistical readings with weak, regime-dependent, out-of-sample edge — not
predictions. Do not make trading or investment decisions solely on this
software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
