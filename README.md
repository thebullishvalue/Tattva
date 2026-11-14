# TATTVA — तत्त्व

**Cross-asset valuation, with a call only where its record supports one** · v3.1.9 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality".

Tattva is a research terminal. For a target — a commodity, a currency, a crypto
asset, an equity index or any listed stock — it says where the price stands
against the whole traded macro cross-section: **cheap, fair or rich**. It then
shows what that reading has been worth, graded exactly as a user would act on
it, and prints **BUY** or **SELL** only where that record supports one.

Today it supports none. Across 27 catalogue targets and eleven years the
contrarian call has no measurable edge over simply holding the same direction
([the record](#the-record)). The valuation, its drivers and its history are the
product; the record is printed beside every reading so nothing is taken on
trust.

It runs on free **yfinance** data. No API keys, no secrets, no database.

---

## What it does

| Part | Question | How |
|---|---|---|
| **Mūla** | *Where should this be trading, given the state of the world?* | A recursive dynamic cointegrating regression of log price on the integrated factors of ~190 macro instruments and on named asset-class blocks, with time-varying coefficients. Publishes a fair-value level, the gap to it, and the gap in units of its own typical size (the oscillator). An error-correction layer asks whether *price* closes the gap or fair value moves to price. |
| **Swayam** | *What does the target's own price say?* | Fifteen causal views of the target's own OHLCV (timescale × macro-anchored or price-only × mechanism), each weighted by its own realised skill. Published as net breadth: views reading it oversold minus views reading it overbought. |
| **Convergence** | *Do the two lean the same way?* | The two leans joined by date: how often they agree in sign, their consensus on a common scale, and the sessions on which they part company. Described, not graded. |
| **Precedent** | *What followed states like today's?* | Mahalanobis analog matching over the system's own state under a Theiler exclusion window — a base rate across 1/3/5/10/20/60 sessions, with the matcher's own walk-forward skill beside it. |
| **Universe** | *Where does everything stand?* | Mūla on every catalogue target in one scan, cheapest first, each with its own record. |

**The read** on the Overview page is a state and, if licensed, a call:

| Reads | Means |
|---|---|
| `FAIR` · NO CALL | the gap is within half its typical size |
| `CHEAP` / `RICH` · NO GRADED CALL | a claim, but that side's pooled record does not support acting; the record is printed |
| `BUY` / `SELL` · GRADED CALL | a claim whose side of the asset-class record is licensed |

Under it are six readings, each with its number: mispricing, price-side
reversion, cross-section agreement, Swayam's breadth, the record, and
precedent. **They are shown, not multiplied into a score.** Graded on past
claims, their product did not separate good calls from bad.

---

## Quickstart

```bash
pip install -r requirements.txt          # Python 3.10+; Streamlit is pinned exactly
streamlit run app.py
pip install pytest && python -m pytest tests -q     # offline, ~30 s
```

Pick an **Asset Class** and a **Target** in the rail (or type a stock symbol
under India / US Stocks) and click **Run Analysis**. The first run fetches daily
history from 2015; switching target re-runs the engines on the panel already in
session. **Refresh data** forces one live pull. **Universe → Scan catalogue**
values every catalogue target, about ten seconds each.

Nothing about the model is user-configurable, by design.

---

## How it works

**Valuation, in levels.** Mūla regresses log price on the integrated common
factors of the macro cross-section with time-varying coefficients:

```
p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t ,   F_{j,t} = sum_{s<=t} f_{j,s}
```

`e` is the gap. Two views — latent principal factors and named asset-class
blocks — are averaged by their own out-of-sample predictive evidence.
Leave-one-block-out refits give driver importance and the cross-section read.

**The gap is not a z-score.** It is divided by its own predictive SD, but its
tails are fat: |gap| ≥ 2 occurs about one day in ten. The pages print the gap's
percentile of its own history beside it.

**A stationary gap is not yet a trade.** A gap closes when either side moves,
and most of the closing is fair value moving to price. The reversion reading is
therefore the price-side error-correction coefficient: each session's price
change regressed on the previous gap.

**The panel is ragged on purpose.** A predictor that lists later keeps its
leading gaps and joins the cross-section when its own print count allows. No
predictor's listing date shortens the target's history, and a new predictor
does not change any row before its first print.

**The call is graded as it is issued.** A claim is a session where the gap is
at least half its typical size. It is entered at the **next** close — the panel
row for a date holds closes from markets that shut after the target did — and
held ten sessions. Each claim is scored **in excess of always holding its
direction on that target**: on an asset that drifts up, a buy claim wins more
than half the time whatever it is conditioned on, so a raw hit rate grades the
drift. Dividend-paying targets are graded on a total-return level built forward
from the close.

**One target cannot license a call.** A single target has a few hundred
independent claims at most, which cannot tell an edge from noise. Claims are
pooled within the asset class (`studies/pooled_record.py` →
`data/priors/call_record.json`), and a side is licensed when its pooled excess
is at least one standard error above zero **and** its hit rate beats the
any-day base (`core/trust.py`).

**Nothing looks ahead, and it is asserted.** Every published value is a
function of data available at its own date. `tests/test_causality.py` and
`tests/test_panel.py` fit Mūla on `data[:T]` and `data[:T-200]` and require
exact agreement on every shared date, on rectangular and ragged panels;
`tests/test_swayam.py` requires the same of Swayam's views, skill weights and
aggregate.

**What can still move a past reading.** The newest bar is still forming until
its session closes; it is published flagged `Provisional` and excluded from
everything that learns. And the panel's composition can differ between fetches
(a rate-limited ticker): the panel fingerprint in the run console says when it
has, and a predictor refilled from a cached snapshot is named in a warning.
The fetch always starts on the same date (`core.config.EPOCH`), so the
estimators' origin does not move from day to day.

---

## The record

<!-- record:start -->
Generated 2026-11-11 by `studies/pooled_record.py` from a panel through 2026-11-11 (fingerprint `439877db9eb4`). 27 targets, 10 sessions, entered 1 session after the claim. *Hit vs base* is the share of claims that made money against the share the same direction made on any day; *t* is the mean return in excess of always holding that direction, over its standard error.

| Asset class | Claims | All claims | Buy claims | Sell claims |
|---|---|---|---|---|
| **All targets** | 45,522 | 47.0% vs 48.2% · t -0.13 | 53.8% vs 55.0% · t -0.02 | 42.6% vs 43.8% · t -0.16 |
| Commodities | 12,631 | 49.7% vs 49.3% · t +0.46 | 53.9% vs 53.1% · t -0.02 | 46.6% vs 46.5% · t +0.60 |
| Currencies | 3,429 | 51.4% vs 49.1% · t +0.70 | 55.8% vs 52.8% · t +0.64 | 48.9% vs 47.0% · t +0.38 |
| Crypto | 8,827 | 47.7% vs 49.5% · t -0.91 | 46.8% vs 49.4% · t -0.88 | 48.5% vs 49.6% · t -0.48 |
| Indian equities | 14,919 | 44.3% vs 47.5% · t -0.47 | 55.4% vs 58.6% · t +0.22 | 38.0% vs 41.2% · t -0.81 |
| US equities | 5,716 | 44.5% vs 44.9% · t +0.16 | 63.5% vs 63.6% · t +0.07 | 35.7% vs 36.3% · t +0.16 |

<!-- record:end -->

No side of any class reaches the licence (t ≥ 1.0 and hit above base), so every
claim currently reads as a state with its record.

Three things the record explains:

1. **The gap leans rich.** It reads rich on about 39% of days and cheap on 26%:
   a slow-memory level regression lags a trending target, so "rich" partly
   means "has gone up", and a contrarian call shorts the drift.
2. **A good rank correlation is not a good call.** The lean's rank IC is shown
   on the System page as a diagnostic. It grades every row; the call is a sign
   at a threshold on claim rows.
3. **Swayam's breadth has no measured edge either** — its ten-session IC is
   inside one standard error of zero on every target measured.

### The candidate study

`studies/PREREGISTRATION.md` fixes seven alternative readings of the same
valuation, the grader, a 252-session holdout that no script reads, and a
promotion bar, and was committed before any of it was run.
`studies/run_study.py` runs them; results are in `studies/results/`. **No
candidate clears the bar** on the development sample, so the holdout has not
been read. Adding a candidate means a new dated section in the pre-registration
and a higher bar.

---

## Reading the pages

- **Overview** — the read, the class record and this target's own, then the
  six readings.
- **Universe** — every catalogue target's state, gap, percentile and own
  record; open any of them.
- **Mūla** — price against fair value with the oscillator and its no-claim
  band; the error-correction read; model quality; drivers; how broad the
  mispricing is across 1-month / 1-quarter / 1-year windows.
- **Swayam** — net breadth, the effective number of independent views, which
  views carry weight, the regime underneath.
- **Convergence** — sign agreement, the consensus, lean against lean, the
  divergences table.
- **Precedent** — the base rate with its distinct-episode count, the matcher's
  skill at every horizon, the closest analogs.
- **System** — each engine's lean IC, the call's record by side for this target
  and its class, the gap's stationarity statistics, data-layer health.
  **Data** — the joined record, exportable.

---

## Data

- **Predictors:** `GLOBAL_MACRO_MAP` (bond, rates, credit, equity, sector,
  style, volatility, real-asset ETFs and indices) and `MACRO_SYMBOLS_YF`
  (commodities, FX) in `core/config.py`. Reciprocals and re-denominations of an
  instrument already present are not listed. Each target excludes its own
  near-replicas.
- **Swayam input:** the target's own OHLCV; its macro driver pool is the same
  panel, minus the target and its near-replicas.
- **Sheet-sourced targets** (Jeera, Nifty 50 - PE) come from published Google
  Sheets, have no OHLCV and run without Swayam. A valuation ratio is not graded
  as a call.

Every external call goes through a two-tier cache, a consecutive-failure
circuit breaker, retry with backoff and a stale-snapshot fallback. A target
whose own price changed on fewer than half of its last 252 sessions is refused
as not live.

**Limits worth knowing.**

- Prices are the traded close, not dividend-adjusted: an adjusted history is
  rewritten backwards on every distribution, which would repaint. Only the
  *grading* of dividend-paying targets uses total return.
- Continuous futures (`=F`) carry contract rolls as returns.
- Crypto is read on a five-day week, by both engines.
- The disk cache (`~/.cache/tattva`) is unpickled as found. Use it on a host
  you control.
- The record covers one decade that was a bull market for most of the
  catalogue — the regime in which a contrarian call does worst.

---

## Configuration and structure

| What | Where |
|---|---|
| Catalogue, universe, exclusions, per-target config | `core/config.py` |
| Sign vocabulary and valuation states | `core/semantics.py` |
| The licence | `core/trust.py`, `data/priors/call_record.json` |
| Index targets, stock symbol resolution | `data/universe.py` |
| Asset-class blocks | `engines/mula/blocks.py` |

```
app.py              Streamlit shell: rail, session caching, pages
pipeline/           prep (one target's panel), run (every phase, no Streamlit),
                    scan (the universe), freshness
core/               config, semantics, trust, logging
data/               yfinance fetchers, cache, circuit breakers, calendars, sheets
engines/mula/       factor model, block map, valuation regression, error correction
engines/swayam/     the per-series kernel and the view bank
analytics/          causal primitives, adaptive tiers, analogs
convergence/        grading (lean IC, call record), consensus, divergences
ui/                 theme, components, one module per page
studies/            pooled record, pre-registered candidate study, the audit's scripts
tests/              offline: causality, the panel, grading, the verdict, the pipeline
docs/               the audit and directive this release implements
```

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Its own
record shows no edge in the call it grades. Do not make trading or investment
decisions on this software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
