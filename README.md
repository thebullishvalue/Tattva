# TATTVA — तत्त्व

**Unified Convergence Engine** · v2.7.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality": the underlying
> truth distilled from the convergence of evidence.

Tattva is a research terminal that produces a single, reproducible directional
signal for a **target** — a commodity (Gold, Silver, Copper, Brent, Cotton), a
currency (USD/INR, the Dollar Index), or an equity **index** (Indian broad & sectoral, US benchmarks,
or an India sector-ETF universe) — by converging two independent engines: a
top-down macro **forecast** and a bottom-up **regime breadth** read, grading its
own out-of-sample edge as it goes.

It runs entirely on free **yfinance** data (plus NSE/Wikipedia for index
constituents). No API keys, no secrets, no database.

**Where to start.** [What it does](#what-it-does) is the one-screen version and
[Quickstart](#quickstart) gets it running. [How the model works](#how-the-model-works)
is the argument — what is estimated, what is declared, and what is deliberately
not claimed — and is the section to read before trusting an output.
[Interpreting the output](#interpreting-the-output) is what to look at once a run
finishes, including where the edge is not.

---

## What it does

For the selected target, Tattva runs a 5-phase pipeline and renders a Streamlit
terminal:

| Engine | Question it answers | How |
|---|---|---|
| **Mūla** | *Where should this be trading, given the state of the world?* | Recursive **dynamic cointegrating regression** of log price on the *integrated* common factors of ~200 macro instruments, with time-varying coefficients — plus the **error-correction read**: Δp regressed on the lagged level gap identifies the reversion speed κ̂ and the expected gap-closure drift over the horizon, and three forecast families (valuation-led / momentum-led / full) are pooled by discounted log predictive density into a learned valuation-informativeness weight. Publishes a fair-value **level**, the mispricing gap against it, the oscillator (gap in units of its own predictive SD), κ̂, and the drift decomposition. |
| **SWAYAM** | *Do independent views of this asset agree?* | MSF + MMR oscillators with HMM/GARCH/CUSUM regime detection, run as a 15-view ensemble (timescale × information-set × mechanism) on the target's **own** OHLCV, aggregated into breadth. Views are weighted by their own recursively-estimated skill, not counted equally. |
| **CONVERGENCE** | *Do the two agree, and how strongly?* | Adaptive-weighted, **directional** composite across Direction / Breadth / Magnitude / Regime, smoothed with a Drift-Diffusion filter. |
| **INTELLIGENCE** | *Which dimensions actually predict, and does it hold up?* | Dimension weights learned **online** from resolved outcomes — exponentially discounted directional skill, scaled by its own significance — plus a read-only expanding-window **walk-forward IC** durability check. Nothing is fitted to the whole sample and nothing is persisted. |
| **PRECEDENT** | *When the state looked like today, what happened next?* | Covariance-aware **Mahalanobis** analog matching (OAS shrinkage) over Tattva's own state features, under a **Theiler exclusion window** so returned analogs are genuinely distinct episodes → an empirical, non-parametric forward-return base rate across a fixed **1/3/5/10/20/60d** term structure, independent of the model. |

The headline output is a normalized convergence signal in `[-1, +1]`
(STRONG BUY → HOLD → STRONG SELL) with a per-window walk-forward IC chart you
can trust — and a published history that does not change when you re-run it.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Then in the control rail on the left: pick an **Asset Class** and a **Target**
(a commodity, USD/INR, an equity index, or any listed stock) and click
**Run Analysis**. First run fetches ~9 years of history (cached afterwards) and
runs the full pipeline; subsequent runs are fast. Switching target re-runs the
engines on the already-fetched macro universe.

The rail is grouped by scope — **Instrument** (what is being analysed) →
**Model** (a read-only status readout) → **Session** (Reset / Refresh) →
**Appearance** (Slate, the dark working theme; Paper, the light one for
reading and print). Controls that reframe a single chart live in that chart's
own panel header, not in the rail: the chart-window selector sits opposite the
context line on the page's primary chart, so a control's position tells you its
scope.

No configuration is required — there are no secrets or environment variables to set.

---

## How the model works

**Valuation, in levels.** Mūla regresses **log price** on the *integrated* common
factors of the macro cross-section with time-varying coefficients:

```
p_t = alpha_t + sum_j beta_{j,t} F_{j,t} + e_t ,   F_{j,t} = sum_{s<=t} f_{j,s}
```

This is a dynamic cointegrating regression (Bierens & Martins 2010), not the
spurious level regression the phrase usually implies: `p` and `F` are both
integrated, and `e` is the deviation from the time-varying long-run relation —
i.e. the mispricing. Two properties follow that a return-space regression cannot
deliver. The residual is a **level**, so fair value is a price rather than a
forecast, and the gap is a genuine mean-reverting spread. And if the relation is
really cointegrating, that residual is stationary and its reversion is testable
**online**, which is what tells the decision layer whether valuation is
informative today instead of assuming it always is.

Two independent valuation views are maintained and averaged by their own
out-of-sample predictive evidence: a **latent** view on the principal factors of
the cross-section (maximum statistical efficiency, weak economic labels), and a
**block** view on named asset-class aggregates — equity, rates, credit,
inflation, energy, metals, agriculture, currency, volatility, real assets. Every
block coefficient has a name, which is what makes the output auditable, and
leave-one-block-out refits give ablation-based driver importance plus a
cross-sectional consistency score: independent slices of the world either agree
about the mispricing or they do not, and that agreement is itself decision-relevant.

**What is deliberately not claimed.** One step ahead, yesterday's close beats any
valuation of a near-integrated price, so the engine is *not* scored against a
random-walk null — that would measure a claim it never makes. It is scored
against the honest competitor: the asset's own 252-day trailing mean. Positive
means the global cross-section locates the level better than the asset's own
history does.

**One horizon, chosen by computation.** Tattva reads a single **10-day** forecast
horizon (daily bars throughout — no weekly resampling), finalized from a 33-target
walk-forward study: the leakage-free directional edge lives at 1–10d and fades by
15–20d (analog edge peaks at +20d and collapses beyond it — zero of 33 targets
significant at +60d). There is no second horizon to choose, because a longer lens
measured on this evidence is a slower-turnover re-expression of the same edge
rather than an independent one. The Precedent tab shows a fixed **1/3/5/10/20/60d**
term structure spanning past that collapse point on purpose — its per-horizon
walk-forward IC makes the fade legible rather than hiding it behind a truncated grid.

**Estimated, not tuned.** A classification cut-point is the causal empirical
quantile of the signal's own past; a weight is the exponentially-discounted
realised skill of the thing being weighted. Both come from
`analytics/adaptive.py`, and both use only data that had already resolved. Because the
quantile is the instrument's own, the p90 conviction cut-point resolves to
15.28 on Gold, 12.26 on USD/INR and 13.10 on S&P 500 — a single pooled number
would leave quiet instruments permanently NEUTRAL and volatile ones permanently
at an extreme. Each constant has a **warm-up prior**, so an instrument's first
year runs on the declared value and the estimate takes over only once it is
better informed than the prior.

What stays declared is *structure* — horizons (what you intend to hold), the
view bank and discount grid (the hypothesis space to average over), the
estimability floors — because those are choices about the question, not
estimates of an answer. Still genuinely hand-set, and the README would rather
say so than pretend otherwise: the DDM filter constants, the analog blend
weights, and the Swayam kernel knobs. The research suite is eight studies —
one per constant that is still swept rather than estimated.

**The engine never looks ahead, and it is asserted.** Every published value is a
function of data available at its own date, so re-running on more data extends
the record rather than rewriting it. That is not a claim about intent — it is a
mechanical property with a mechanical test: `research/test_reproducibility.py`
runs the system on `data[:T]` and on `data[:T-250]` and requires the two to
agree **exactly** on every shared date, across the Mūla engine, the Swayam view
weights, the aggregated breadth, the convergence dimension weights and the
adaptive thresholds. A component that consulted the future cannot pass it. The
test also fails on all-NaN output, so a component that quietly stopped
producing anything cannot pass it either.

**Two things can still move a past reading, and neither is look-ahead.**

*The newest bar is still forming* until its session closes — continuously, for a
24/7 instrument like crypto — so the reading for today can differ from the
reading for today once today is over. Verified not to leak backwards: perturbing
the final close and re-running leaves all 2,921 earlier dates bit-identical.
A session fitted on a fraction of the cross-section is withheld outright rather
than published provisionally, so a half-open panel reads as "no value yet"
instead of a confident wrong one.

*The panel's composition can change between runs* — a rate-limited fetch, a
holiday, an instrument admitted for the first time — and the factor basis is
estimated from whichever instruments are present. Different panel, different
eigenvectors, so published history moves. Measured by dropping one predictor
from the live 242-column panel and refitting: **median 0.04-0.14%, p95
3.1-5.0%** (`research/test_composition_sensitivity.py`, which pins the size of
this exposure so a change that worsens it fails loudly). This is not fixable
inside the model — no estimator is invariant to its own input set, and
replacing Marchenko-Pastur truncation with eigenvalue shrinkage was measured to
make it 2-8x worse. Closing it requires a declared universe that the realised
panel is asserted against, which does not exist yet. Until it does, the panel
fingerprint printed in the run console is how a composition change is detected
after the fact.

**Causal factors.** The factor structure is estimated recursively
from an exponentially weighted correlation matrix, with the number of factors set
by the **Marchenko-Pastur** edge — the eigenvalues that stand above what pure
noise of that dimension would produce — and the memory chosen online from a bank
of half-lives by predictive likelihood. Everything is one-sided: an instrument
joins the cross-section on the day its own accumulated print count first reaches
the estimability floor, and contributes only on days it actually printed, so
admission never retroactively changes and a foreign market's holiday cannot enter
as a fabricated zero return. Adding new data never rewrites history — though
changing which instruments are in the panel does, per the exception above.

**Why the coefficient memory is slow.** Scoring discount factors by one-step
predictive likelihood is degenerate for a *level* regression: the model that
tracks price most closely always wins, and its limit is the useless statement
"fair value = price". The grid is therefore restricted to implied memories of
~4y, ~8y, ~40y and permanent. This is the single most consequential decision in
the engine and it is a modelling commitment, not a tuned choice — admitting a
~5-month memory collapses the measured mispricing by roughly a factor of three
and its half-life from weeks to days, which is a residual, not a valuation.

**Honest validation, leakage-free.** The durability diagnostic is an
expanding-window walk-forward: each window learns weights on the past and is
scored on the NEXT purged block, so every reported IC is genuinely out-of-sample
and nothing it returns feeds back into the signal. Scoring is
**non-overlapping** (stride = the shortest hold horizon) rather than on every
daily row — a daily-sampled IC on overlapping h-day forward returns overstates
its own precision by roughly √h, so the trust chip's SOLID/MODEST/MARGINAL tiers
are set on the non-overlapping scale. Mūla itself has no labels to leak: it is fit to no forward
target, so there is no label overlap to purge and no horizon-specific refit. Its
**burn-in** (the first `FVO_BURN_IN` rows, before an exponentially weighted
correlation matrix over ~200 instruments has enough weight for the
Marchenko-Pastur edge to mean anything) is left genuinely unpublished and flagged
`Valid = False`, rather than filled with a prior dressed up as an estimate. An expanding-window **walk-forward IC** runs every analysis and
is charted in Diagnostics — consistently positive bars = durable edge; a couple of
spikes = a lucky regime. The **Precedent** tab is a separate, non-parametric base
rate read alongside the model, not part of the calibrated convergence signal; its
analog matcher enforces a **Theiler exclusion window** (Theiler 1986) between
returned analogs so "N analogs" reflects N genuinely distinct historical episodes,
not N adjacent days of the same episode.

---

## Data sources (all yfinance)

- **Target & predictors:** the target's price series (commodity future / FX / index
  level) plus the macro universe in `core/config.py` — `GLOBAL_MACRO_MAP`
  (bond/rates/equity/risk/real-asset ETFs) + `MACRO_SYMBOLS_YF` (commodities + FX).
- **Index targets:** `INDEX_TARGETS` in `data/universe.py` (India broad/sectoral, US
  benchmarks, India sector-ETF universe).
- **Swayam input:** the target's **own** OHLCV, for every target. Swayam asks
  its breadth question of one price series read many ways — timescale ×
  information set × mechanism — so no constituent list, proxy basket or
  cross-section fetch is involved, and a large index costs the same single
  series as a commodity.

Every external call is wrapped in a two-tier cache (memory + disk), a per-service
circuit breaker, retry-with-backoff, a **partial-success re-fetch** (yfinance
rate-limits a few tickers per batch, so the missing symbols are re-fetched to
complete the set rather than cached incomplete), and a stale-snapshot fallback — so
the UI and research suite keep working through transient yfinance rate-limiting.

**Freshness is calendar-exact.** `data/calendars.py` resolves each ticker to its home
exchange and uses real trading calendars (`exchange_calendars`) to count "days behind",
judge the partial-session gate (only markets that were *open* are expected to post), and
build each target's model spine from its true sessions. The dependency is **optional** —
absent, every check degrades to a plain Mon–Fri mask.

---

## Configuration

**Everything is per-instrument.** Each instrument carries its own full
`InstrumentConfig` — routing *and* every tunable knob across ALL layers: the
Mūla valuation (burn-in / print floor / coefficient-memory grid / lookback),
Swayam + Swayam breadth, convergence DDM + dimension weights, the
classification thresholds, and the interpretation/display tiers (markers,
conviction, breadth, agreement, model-spread) — in the `INSTRUMENT_CONFIGS`
registry (`core/config.py`). The five catalogue classes (commodity, fx,
india_index, us_index, etf) are tuned **per instrument** (hand-wired values in
`_PER_INSTRUMENT_OVERRIDES`, layered on the class default); the India/US **stock**
classes are tuned at **asset-class** level via `STOCK_CONFIGS`, since free-form
symbols can't be pre-tuned. Only genuine statistical-definition constants
(R²/ADF/KPSS/HMM cut-points, chart dimensions) stay global. An instrument with
no override runs on its class default, so the registry only has to carry what
is genuinely instrument-specific: to retune one instrument, add its knob to
`_PER_INSTRUMENT_OVERRIDES`; to retune a whole class, edit its default in
`CLASS_CONFIG_DEFAULTS`.

| What | Where |
|---|---|
| Target commodities / FX | `COMMODITY_TARGETS` in `core/config.py` |
| Index targets (India / US / ETF) | `INDEX_TARGETS` in `data/universe.py` |
| **Per-instrument config (structure, floors, warm-up priors)** | `InstrumentConfig` / `INSTRUMENT_CONFIGS` in `core/config.py` |
| **Per-asset-class config defaults** | `CLASS_CONFIG_DEFAULTS` (`commodity`, `fx`, `india_index`, `us_index`, `etf`, `stock_india`, `stock_us`) + `STOCK_CONFIGS` in `core/config.py` |
| Individual-stock targets (free-form symbol, Swayam self-mode) | Sidebar **India Stocks** / **US Stocks** asset class → `data/universe.py::resolve_stock_symbol` + `core/config.py::register_stock_target` |
| Mūla valuation + scoring horizons (burn-in / print floor / discount grid / lookback / hold) | fields on each `InstrumentConfig` (`fvo_*`, `forecast_horizon`, `hold_horizons`) |
| DDM / dimension weights / thresholds / markers / display tiers / analog blend / Swayam grid | fields on each `InstrumentConfig` |
| Macro predictor universe | `GLOBAL_MACRO_MAP` + `MACRO_SYMBOLS_YF` |
| Constituent cap | `_DEFAULT_CAP` in `data/universe.py` (`0` = no cap, full index) |
| Valuation burn-in / print floor / discount grid | `core/config.py` (`FVO_BURN_IN`, `FVO_MIN_PRINTS`, `FVO_VALUATION_DELTAS`, `MIN_DATA_POINTS`) |
| Asset-class block map for the cross-section | `engines/fvo/blocks.py` |

In-app: nothing about the model is user-configurable, by design. The valuation
panel is the whole traded cross-section minus this target's self-replicating
near-duplicates, and the dimension weights are learned forward every run rather
than loaded from a profile — so there is no predictor picker to set and no
calibration artefact to go stale. The rail's **Model** group shows what the run
actually reached (dimension weights, walk-forward IC); it is a readout, not a
control.

**Individual stocks are free-form, not a drop-down.** Selecting **India Stocks** or
**US Stocks** as the Asset Class swaps the Target picker for a symbol text box.
India symbols are resolved by probing `SYMBOL.NS` (NSE) first, then `SYMBOL.BO`
(BSE) — an explicit `.NS`/`.BO` suffix skips the probe; US symbols are used as
typed (`.` → `-`, the yfinance convention — e.g. `BRK.B` → `BRK-B`). A resolved
symbol is registered as a first-class target (`RELIANCE (NSE)`, `AAPL (US)`, …) —
Mūla values it and Swayam runs Swayam self-mode on it, with the same
per-target treatment as every other target. Successful resolutions are
cached 7 days (`~/.cache/tattva/symbol_resolution/`); a not-found symbol is never
cached, so a transient yfinance outage can't permanently brand it invalid.

---

## Project structure

```
app.py                  Streamlit entrypoint + 5-phase orchestration
core/                   config — macro universe, structure, floors, priors, and the
                        per-instrument InstrumentConfig registry — + logging
data/                   yfinance fetchers, index catalogue + constituent
                        resolution (universe), two-tier cache, circuit breakers,
                        per-exchange trading calendars (calendars.py)
engines/                mula/ (valuation: recursive cointegrating regression —
                        ECM error-correction layer,
                        causal DLM/DMA primitives, online factor model with a
                        Marchenko-Pastur cut, regime filter, asset-class block
                        map), swayam/ (breadth: the per-series MSF/MMR/regime
                        kernel + the skill-weighted self-referential view bank)
analytics/              adaptive (causal thresholds + online skill weights),
                        OU, Hurst/DFA,
                        robust-quantile z-scores, HMM/GARCH/CUSUM, breaks,
                        analogs (Mahalanobis precedent matcher)
convergence/            cross-validator, conviction (DDM), divergence,
                        normalization, intelligence (online weights + walk-forward)
ui/                     theme, components, tabs (Convergence/Mūla/Swayam/
                        Precedent/Diagnostics/Data)
research/               validation harnesses (identity sweep, Mūla ECM + hero
                        verdict synthetic suite, real-data Swayam study)
```

Re-tuning: `python3 research/run_tuning.py` opens an interactive menu (run the whole
suite end-to-end, from-scratch, a single tier, or hand-picked studies); `--list`
shows the suite, `--all`/`--only`/`--segment`/`--fresh` script it. Every study
emits a **gated per-instrument** `_PER_INSTRUMENT_OVERRIDES` snippet alongside its
class-level result, and the orchestrator writes one consolidated report
(`research/reports/`) plus a current-vs-validated reference for every tuned
constant. A live heartbeat keeps long runs legible. It **reports only** — config is
applied by hand after review.

---

## Interpreting the output

- **Hero card** — normalized convergence signal and the Mūla / Swayam contributions.
- **Mūla tab** — price against the fair-value level the cross-section implies,
  inside its 95% predictive band, with the mispricing gap that drives the signal
  stack below it. Model quality reads left to right as a chain: does the
  cross-section track this asset at all (OOS R²), does it beat the asset's own
  252d trailing mean (**R² vs Trailing Mean** — the discriminating number), do
  independent slices of the world agree on the mispricing's sign (**Valuation
  Confidence**), is the gap stationary and how fast (**Mean Reversion**), and how
  tightly is fair value pinned today (**Model Spread**).
- **Precedent tab** — the most statistically-similar historical states (Mahalanobis)
  and what the target did next, across a fixed **1/3/5/10/20/60d** term structure
  (`PRECEDENT_HORIZONS`); an empirical base rate to read *alongside* Mūla
  (agreement strengthens conviction, disagreement is a divergence). The Analog Skill
  chart shows walk-forward IC at each horizon, so where the edge is genuinely present
  (typically ~10–20d) vs weak (the 1d and 60d ends) is visible, not assumed.
- **Diagnostics → Intelligence Center** — learned-vs-prior weights and the **walk-forward
  IC** chart (the durability verdict).

Rule of thumb: trust the **walk-forward consistency**, not any single conviction
reading. Across the universe the (leakage-free) directional edge is
modest and concentrated at **10–20d** — the precedent base rate is strongest as a
~10d confirmer, and is best treated as fading in the recent regime.

**Swayam's honest limitation.** Breadth is read across 15 *views of one price
series* rather than 15 independent instruments, so the bank is more internally
correlated than a genuine cross-section would be — expect lumpier breadth swings
and more synchronized regime flips than a constituent read would show. That is
the price of not needing a hand-curated proxy, and it is disclosed rather than
hidden: the Swayam tab surfaces an "effective view count" (an eigenvalue-based
diagnostic, never fed into the signal itself), and the views are skill-weighted,
so a timescale that has stopped predicting fades out of the aggregate instead of
padding the apparent agreement. The trade is deliberate — a self-referential
bank needs no hand-curated proxy basket, and a proxy is a judgement the data
never gets to overrule.

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Outputs are
statistical signals with weak, regime-dependent, out-of-sample edge — not predictions.
Markets are noisy and the validated ICs are modest. Do not make trading or investment
decisions solely on this software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
