<div align="center">

# TATTVA

**तत्त्व** · *principle, essence, reality*

Cross-Asset Fair Value · Self-Referential Breadth · A Graded Record

[![Version](https://img.shields.io/badge/version-3.1.20-2B5FD9)](CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](requirements.txt) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-FF4B4B)](requirements.txt) [![License](https://img.shields.io/badge/license-proprietary-6B7280)](LICENSE) [![Pragyam Family](https://img.shields.io/badge/pragyam-family-0F1217)](#the-pragyam-family)

</div>

> One engine values the instrument against the world, a second reads its own price from a bank
> of views, and Tattva says where it stands (cheap, fair or rich) and issues a call only where
> that call's own record supports one.

## Overview

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

## What It Does

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

## Quick Start

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

## The Record

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

## Project Structure

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

## Documentation

| Document | What it covers |
|---|---|
| [Guide](docs/GUIDE.md) | How it works, reading each page, the data and its limits |
| [Pre-registration](studies/PREREGISTRATION.md) | The candidate study, fixed before it was run |
| [Resolution directive](docs/RESOLUTION_DIRECTIVE.md) | The audit the current release implements |
| [Implementation notes](docs/IMPLEMENTATION_NOTES.md) | Where the build departed from the directive, and what is left |
| [Changelog](CHANGELOG.md) | Every release, newest first |

## The Pragyam Family

Ten Streamlit applications built on one design system and one set of components, so a
panel, a table, a chart or a number reads the same in each.

| Product | What it is |
|---|---|
| **Tattva** · *this repository* | Cross-asset fair value, with a call only where its own record supports one |
| [**Nishkarsh**](https://github.com/thebullishvalue/Nishkarsh) | Nifty 50 inference: a top-down forecast and bottom-up breadth, read as one |
| [**Arthagati**](https://github.com/thebullishvalue/Arthagati) | Valuation-anchored market sentiment, tested on data it never saw |
| [**Vyapti**](https://github.com/thebullishvalue/Vyapti) | Market breadth: where participation stands, which way it moves, and what that has meant |
| [**Pragyam**](https://github.com/thebullishvalue/Pragyam) | Portfolio curation that spreads risk rather than predicting returns |
| [**Sanket**](https://github.com/thebullishvalue/Sanket) | A market signal screener: what fired, what is about to, and whether it has paid |
| [**Vaaydo**](https://github.com/thebullishvalue/Vaaydo) | An F&O trade screener on option positioning, with futures tickets |
| [**Nivesa**](https://github.com/thebullishvalue/Nivesa) | A bond portfolio ledger: cost, yield, duration and cashflows across accounts |
| [**Samhita**](https://github.com/thebullishvalue/Samhita) | A portfolio tracker: live prices, performance and risk against the benchmark |
| [**Swing**](https://github.com/thebullishvalue/Swing) | A portfolio tracker for the swing book, on the same engine as Samhita |

## License

Proprietary. © 2026 @thebullishvalue. All rights reserved. The source is visible here
for reference only: viewing it grants no right to use, copy, modify or distribute it.
See [LICENSE](LICENSE).

## Disclaimer

Tattva is a research and decision-support tool, not investment advice. Nothing it shows is
a recommendation to buy or sell any security, and past or measured results do not guarantee
future ones. Every decision made with it is your own. Its own record shows no edge in the call
it grades.

---

<div align="center"><sub>© 2026 @thebullishvalue · Part of the Pragyam product family</sub></div>
