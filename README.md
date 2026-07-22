# TATTVA — तत्त्व

**Unified Convergence Engine** · v2.7.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality": the underlying
> truth distilled from the convergence of evidence.

Tattva is a research terminal that produces a single, calibrated directional
signal for a **target** — a commodity (Gold, Silver, Copper, Brent, Cotton), an
FX pair (USD/INR), or an equity **index** (Indian broad & sectoral, US benchmarks,
or an India sector-ETF universe) — by converging two independent engines: a
top-down macro **forecast** and a bottom-up **regime breadth** read, grading its
own out-of-sample edge as it goes.

It runs entirely on free **yfinance** data (plus NSE/Wikipedia for index
constituents). No API keys, no secrets, no database.

---

## What it does

For the selected target, Tattva runs a 5-phase pipeline and renders a Streamlit
terminal:

| Engine | Question it answers | How |
|---|---|---|
| **AARAMBH** | *Is the macro setup pointing up or down?* | Walk-forward ensemble (members configurable per instrument via `aarambh_ensemble_models`) **forecasting the 10-day forward return** from trailing macro momentum. The walk-forward purges label overlap, so the out-of-sample IC is leakage-free. |
| **NIRNAY** | *What is the related complex doing bottom-up?* | Per-instrument MSF + MMR oscillators with HMM/GARCH/CUSUM regime detection, aggregated into breadth. Two formulations, same output schema: **basket mode** runs it across the target's constituents (an equity index's own members, or the dollar-strength/agri basket for USD-INR / Jeera); **Swayam self-mode** runs it as a 15-view ensemble (timescale × information-set × mechanism) on the target's *own* OHLCV — used for **individual stocks** and the **liquid commodity futures** (Gold, Silver, Copper, Cotton, Brent), which have no need of a proxy basket when their own volume-bearing futures price is available. |
| **CONVERGENCE** | *Do the two agree, and how strongly?* | Adaptive-weighted, **directional** composite across Direction / Breadth / Magnitude / Regime, smoothed with a Drift-Diffusion filter. |
| **INTELLIGENCE** | *Does the signal actually have edge?* | Optuna TPE calibration of the convergence weights/thresholds with a **purged k-fold CV objective** + held-out tail, plus an automatic **walk-forward IC** durability check (per target). |
| **PRECEDENT** | *When the state looked like today, what happened next?* | Covariance-aware **Mahalanobis** analog matching (OAS shrinkage) over Tattva's own state features, under a **Theiler exclusion window** so returned analogs are genuinely distinct episodes → an empirical, non-parametric forward-return base rate across a fixed **1/3/5/10/20/60d** term structure, independent of the model. |

The headline output is a normalized convergence signal in `[-1, +1]`
(STRONG BUY → HOLD → STRONG SELL) with an honest out-of-sample **Val IC** and a
per-window walk-forward chart you can trust.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Then in the sidebar: pick a **Target** (a commodity, USD/INR, or an equity index)
and click **Run Analysis**. First run fetches ~9 years of history (cached afterwards)
and runs the full pipeline; subsequent runs are fast. Switching target re-runs the
engines on the already-fetched macro universe (only the Nirnay basket re-pulls).

No configuration is required — there are no secrets or environment variables to set.

---

## How the model works

**Predictive, returns-based.** Aarambh does **not** regress price levels (which is a
spurious regression). It forecasts the **forward log-return** of the target — over a
fixed **10-day** horizon — from **trailing momentum** of ~135 macro/FX/commodity
series, a genuine ex-ante setup. The forecast drives a directional conviction;
out-of-sample skill is measured by rank **IC**, not R² (a price forecast's magnitude
R² is ~0 by nature; the tradeable information is in the direction).

**One horizon, chosen by computation.** Tattva reads a single **10-day** forecast
horizon (daily bars throughout — no weekly resampling), finalized from a 33-target
walk-forward study: the leakage-free directional edge lives at 1–10d and fades by
15–20d (analog edge peaks at +20d and collapses beyond it — zero of 33 targets
significant at +60d). An earlier build offered a second "Positional 20d" lens, but it
was a slower-turnover re-expression rather than an independent edge, so it was removed
along with its selector. The Precedent tab still shows a fixed **1/3/5/10/20/60d**
term structure spanning past that collapse point on purpose — its per-horizon
walk-forward IC makes the fade legible rather than hiding it behind a truncated grid.

**Causal PCA, no repainting.** The ~112 collinear macro inputs are reduced to ~20
orthogonal components **inside each walk-forward training window** — fit only on past
data, so a component's value at time *t* never depends on the future. Adding new data
never rewrites history. This stabilises the ensemble (low model spread) while keeping
every input "on."

**Honest validation, leakage-free.** The intelligence calibration optimises a
**purged k-fold cross-validation** objective (robust across time, not one slice) and
reports a Val IC on a genuinely held-out, purged tail, scored **non-overlapping**
(stride = the shortest hold horizon) rather than on every daily row — a daily-sampled
IC on overlapping h-day forward returns overstates its own precision by roughly √h, so
the trust chip's SOLID/MODEST/MARGINAL tiers are calibrated to the non-overlapping
scale. The Aarambh walk-forward itself also **purges label overlap** — each
forward-return label spans `(t, t+h]`, so training rows within `h` of the prediction
point are dropped to stop the forecast window leaking into training (this materially
lowered, and corrected, the long-horizon IC) — and the engine's own warm-up window
(the first `MIN_TRAIN_SIZE` rows, before any walk-forward chunk has been fit) is left
genuinely unscored rather than filled from an expanding mean of labels that overlap
the forecast window. An expanding-window **walk-forward IC** runs every analysis and
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
- **Nirnay input:** depends on the target's archetype (`TARGET_ARCHETYPE`,
  `core/config.py`). *Basket-mode* targets fetch a cross-section — an index's own
  constituents (resolved live via NSE archive CSV / Wikipedia, cached 24 h, with a
  hardcoded-snapshot fallback), or the curated `COMMODITY_BASKETS` for USD/INR
  (dollar-strength proxies) and Jeera (Indian agribusiness). *Swayam self-mode*
  targets fetch only their own OHLCV and run the self-referential ensemble
  (`engines/nirnay_self.py`) — this covers individual stocks (free-form symbol
  entry) and the liquid commodity futures (Gold/Silver/Copper/Cotton/Brent), whose
  volume-bearing front-month price makes a proxy basket unnecessary. (Jeera stays a
  basket target because NCDEX cumin has no yfinance OHLCV; USD/INR because FX carries
  no yfinance volume.)

Every external call is wrapped in a two-tier cache (memory + disk), a per-service
circuit breaker, retry-with-backoff, a **partial-success re-fetch** (yfinance
rate-limits a few tickers per batch, so the missing symbols are re-fetched to
complete the set rather than cached incomplete), and a stale-snapshot fallback — so
the UI and research suite keep working through transient yfinance rate-limiting.

**Freshness is calendar-exact.** `data/calendars.py` resolves each ticker to its home
exchange and uses real trading calendars (`exchange_calendars`) to count "days behind",
judge the partial-session gate (only markets that were *open* are expected to post), and
build each target's model spine from its true sessions. The dependency is **optional** —
absent, every check degrades to a plain Mon–Fri mask, identical to prior behaviour.

---

## Configuration

**Everything is per-instrument.** Each instrument carries its own full
`InstrumentConfig` — routing *and* every tunable knob across ALL layers: the
Aarambh forecast (train window / refit / ensemble / ridge / huber / lookback /
PCA), Nirnay + Swayam breadth, convergence DDM + dimension weights, the
classification thresholds, and the interpretation/display tiers (markers,
conviction, breadth, agreement, model-spread) — in the `INSTRUMENT_CONFIGS`
registry (`core/config.py`). The five catalogue classes (commodity, fx,
india_index, us_index, etf) are tuned **per instrument** (hand-wired values in
`_PER_INSTRUMENT_OVERRIDES`, layered on the class default); the India/US **stock**
classes are tuned at **asset-class** level via `STOCK_CONFIGS`, since free-form
symbols can't be pre-tuned. Only genuine statistical-definition constants
(R²/ADF/KPSS/HMM cut-points, chart dimensions) stay global. Field defaults equal
the former global constants, so an untuned instrument behaves exactly as before —
to retune one, add its knob to `_PER_INSTRUMENT_OVERRIDES`; to retune a whole
class, edit its default in `CLASS_CONFIG_DEFAULTS`.

| What | Where |
|---|---|
| Target commodities / FX | `COMMODITY_TARGETS` in `core/config.py` |
| Index targets (India / US / ETF) | `INDEX_TARGETS` in `data/universe.py` |
| **Per-instrument config (routing + all engine knobs)** | `InstrumentConfig` / `INSTRUMENT_CONFIGS` in `core/config.py` |
| **Per-asset-class config defaults** | `CLASS_CONFIG_DEFAULTS` (`commodity`, `fx`, `india_index`, `us_index`, `etf`, `stock_india`, `stock_us`) + `STOCK_CONFIGS` in `core/config.py` |
| Nirnay mode per instrument (basket vs Swayam self) | `InstrumentConfig.archetype` (`"self"` → Swayam; else basket) |
| Nirnay basket (USD/INR, Jeera, retained-for-research commodities) | `InstrumentConfig.basket` / `basket_alias` (source: `COMMODITY_BASKETS`) |
| Individual-stock targets (free-form symbol, Nirnay-Swayam self-mode) | Sidebar **India Stocks** / **US Stocks** asset class → `data/universe.py::resolve_stock_symbol` + `core/config.py::register_stock_target` |
| Aarambh forecast + training (horizon / momentum / hold / PCA / refit / min-max train / ensemble / ridge / huber / lookback) | fields on each `InstrumentConfig` (`forecast_*`, `pca_components`, `aarambh_*`) |
| DDM / dimension weights / thresholds / markers / display tiers / analog blend / Swayam grid | fields on each `InstrumentConfig` (defaults = the former globals) |
| Macro predictor universe | `GLOBAL_MACRO_MAP` + `MACRO_SYMBOLS_YF` |
| Ensemble members (per instrument) | `InstrumentConfig.aarambh_ensemble_models` (global default `ENSEMBLE_MODELS = ("ols",)`) |
| Constituent cap | `_DEFAULT_CAP` in `data/universe.py` (`0` = no cap, full index) |
| Walk-forward / train sizes | `core/config.py` (`MIN_TRAIN_SIZE`, `MAX_TRAIN_SIZE`, `MIN_DATA_POINTS`) |

In-app: the sidebar **Model Configuration** lets you deselect predictors (the full
universe is on by default). Calibrated profiles persist to
`~/.cache/tattva/intelligence/profiles.json` (one per target).

**Individual stocks are free-form, not a drop-down.** Selecting **India Stocks** or
**US Stocks** as the Asset Class swaps the Target picker for a symbol text box.
India symbols are resolved by probing `SYMBOL.NS` (NSE) first, then `SYMBOL.BO`
(BSE) — an explicit `.NS`/`.BO` suffix skips the probe; US symbols are used as
typed (`.` → `-`, the yfinance convention — e.g. `BRK.B` → `BRK-B`). A resolved
symbol is registered as a first-class target (`RELIANCE (NSE)`, `AAPL (US)`, …) —
Aarambh forecasts it and Nirnay runs Swayam self-mode on it, with the same
per-target calibration as every other target. Successful resolutions are
cached 7 days (`~/.cache/tattva/symbol_resolution/`); a not-found symbol is never
cached, so a transient yfinance outage can't permanently brand it invalid.

---

## Project structure

```
app.py                  Streamlit entrypoint + 5-phase orchestration
core/                   config — macro universe, baskets, thresholds, and the
                        per-instrument InstrumentConfig registry — + logging
data/                   yfinance fetchers, index catalogue + constituent
                        resolution (universe), two-tier cache, circuit breakers,
                        per-exchange trading calendars (calendars.py)
engines/                aarambh (forecast, purged walk-forward), nirnay (breadth),
                        nirnay_self (Swayam self-referential ensemble for
                        commodity + individual-stock targets)
analytics/              OU, Hurst/DFA, robust-quantile z-scores, HMM/GARCH/CUSUM,
                        breaks, analogs (Mahalanobis precedent matcher)
convergence/            cross-validator, conviction (DDM), divergence,
                        normalization, intelligence (calibration + walk-forward)
ui/                     theme, components, tabs (Convergence/Aarambh/Nirnay/
                        Precedent/Diagnostics/Data)
research/               tuning & validation harnesses (Aarambh/Nirnay/analog
                        sweeps, marker/hero studies) + run_tuning.py orchestrator
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

- **Hero card** — normalized convergence signal and the Aarambh / Nirnay contributions.
- **Aarambh tab** — price with the model's forward expected-price projection (an
  implied target + uncertainty cone) and the expected-forward-return forecast driving
  it; model quality shows **Val IC** and the train→val gap (overfit detector).
- **Precedent tab** — the most statistically-similar historical states (Mahalanobis)
  and what the target did next, across a fixed **1/3/5/10/20/60d** term structure
  (`PRECEDENT_HORIZONS`); an empirical base rate to read *alongside* Aarambh
  (agreement strengthens conviction, disagreement is a divergence). The Analog Skill
  chart shows walk-forward IC at each horizon, so where the edge is genuinely present
  (typically ~10–20d) vs weak (the 1d and 60d ends) is visible, not assumed.
- **Diagnostics → Intelligence Center** — calibration state and the **walk-forward
  IC** chart (the durability verdict).

Rule of thumb: trust the **Val IC** and the **walk-forward consistency**, not any
single conviction reading. Across the universe the (leakage-free) directional edge is
modest and concentrated at **10–20d** — the precedent base rate is strongest as a
~10d confirmer, and is best treated as fading in the recent regime.

**Nirnay-Swayam's honest limitation.** On individual-stock targets, breadth is read
across 15 *views of one price series* rather than 15 independent names, so the
ensemble is more internally correlated than a real constituent basket — expect
lumpier breadth swings and more synchronized regime flips than the commodity/index
tabs show. The Nirnay tab surfaces an "effective view count" (an eigenvalue-based
diagnostic, never fed into the signal itself) so this is visible rather than hidden;
the Intelligence layer recalibrates its weights/thresholds per target
against the observed distribution rather than inheriting basket-tuned defaults. Ship
status is evidence-gated by `research/nirnay_swayam_study.py` (`nirnay_swayam` in
the tuning suite), not assumed.

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Outputs are
statistical signals with weak, regime-dependent, out-of-sample edge — not predictions.
Markets are noisy and the validated ICs are modest. Do not make trading or investment
decisions solely on this software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
