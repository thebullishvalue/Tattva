# TATTVA — तत्त्व

**Unified Convergence Engine** · v2.4.0 · *@thebullishvalue*

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
| **AARAMBH** | *Is the macro setup pointing up or down?* | Walk-forward ensemble (configurable via `ENSEMBLE_MODELS`; default **PCA-OLS + Huber**) **forecasting the forward return at the selected Signal Horizon** (Tactical 10d / Positional 20d) from trailing macro momentum. The walk-forward purges label overlap, so the out-of-sample IC is leakage-free. |
| **NIRNAY** | *What is the related complex doing bottom-up?* | Per-instrument MSF + MMR oscillators with HMM/GARCH/CUSUM regime detection across the target's basket (related miners/streamers for a commodity, or the index's own constituents), aggregated into breadth. |
| **CONVERGENCE** | *Do the two agree, and how strongly?* | Adaptive-weighted, **directional** composite across Direction / Breadth / Magnitude / Regime, smoothed with a Drift-Diffusion filter. |
| **INTELLIGENCE** | *Does the signal actually have edge?* | Optuna TPE calibration of the convergence weights/thresholds with a **purged k-fold CV objective** + held-out tail, plus an automatic **walk-forward IC** durability check (per `(target, lens)`). |
| **PRECEDENT** | *When the state looked like today, what happened next?* | Covariance-aware **Mahalanobis** analog matching (Ledoit-Wolf) + trajectory + recency over Tattva's own state features → an empirical, non-parametric forward-return base rate at the lens horizons, independent of the model. |

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

Then in the sidebar: pick a **Target** (a commodity, USD/INR, or an equity index),
a **Signal Horizon** (Tactical 10d or Positional 20d), and click **Run Analysis**.
First run fetches ~9 years of history (cached afterwards) and runs the full pipeline;
subsequent runs are fast. Switching target re-runs the engines on the already-fetched
macro universe (only the Nirnay basket re-pulls); switching lens recomputes and caches
that lens separately, so both reads coexist.

No configuration is required — there are no secrets or environment variables to set.

---

## How the model works

**Predictive, returns-based.** Aarambh does **not** regress price levels (which is a
spurious regression). It forecasts the **forward log-return** of the target — over the
selected **Signal Horizon** — from **trailing momentum** of ~135 macro/FX/commodity
series, a genuine ex-ante setup. The forecast drives a directional conviction;
out-of-sample skill is measured by rank **IC**, not R² (a price forecast's magnitude
R² is ~0 by nature; the tradeable information is in the direction).

**Two horizons, chosen by computation.** The sidebar offers **Tactical (10d)** and
**Positional (20d)** lenses (daily bars throughout — no weekly resampling). The two
`d` values, and which horizons the Precedent base rate is shown at, were finalized
from a 33-target walk-forward study: analog edge peaks at +20d across the universe and
collapses beyond it (zero of 33 targets significant at +60d). Each lens carries its
own momentum window, lens-scaled DDM smoothing, and its own calibrated profile.

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
- **Nirnay basket:** per-commodity miners/streamers in `COMMODITY_BASKETS`
  (`core/config.py`); for an index target, the index's own constituents — resolved
  live (NSE archive CSV / Wikipedia), cached 24 h, with a hardcoded-snapshot fallback.

Every external call is wrapped in a two-tier cache (memory + disk), a per-service
circuit breaker, retry-with-backoff, and a stale-snapshot fallback — so the UI keeps
working through transient yfinance outages.

**Freshness is calendar-exact.** `data/calendars.py` resolves each ticker to its home
exchange and uses real trading calendars (`exchange_calendars`) to count "days behind",
judge the partial-session gate (only markets that were *open* are expected to post), and
build each target's model spine from its true sessions. The dependency is **optional** —
absent, every check degrades to a plain Mon–Fri mask, identical to prior behaviour.

---

## Configuration

| What | Where |
|---|---|
| Target commodities / FX | `COMMODITY_TARGETS` in `core/config.py` |
| Index targets (India / US / ETF) | `INDEX_TARGETS` in `data/universe.py` |
| Nirnay commodity baskets | `COMMODITY_BASKETS` in `core/config.py` |
| Macro predictor universe | `GLOBAL_MACRO_MAP` + `MACRO_SYMBOLS_YF` |
| Ensemble members | `ENSEMBLE_MODELS` in `core/config.py` (default `("ols", "huber")`) |
| Constituent cap | `_DEFAULT_CAP` in `data/universe.py` (`0` = no cap, full index) |
| Signal Horizon lenses (horizon · momentum · hold · DDM) | `SIGNAL_HORIZONS` in `core/config.py` (Tactical 10d / Positional 20d) |
| PCA components | `n_pca_components` in the `engine.fit(...)` call (`app.py`) |
| Walk-forward / train sizes | `core/config.py` (`MIN_TRAIN_SIZE`, `MAX_TRAIN_SIZE`, `MIN_DATA_POINTS`) |

In-app: the sidebar **Model Configuration** lets you deselect predictors (the full
universe is on by default) and pick the **Signal Horizon**. Calibrated profiles persist
to `~/.cache/tattva/intelligence/profiles.json` (one per `(target, lens)`).

---

## Project structure

```
app.py                  Streamlit entrypoint + 5-phase orchestration
core/                   config (macro universe, baskets, thresholds, Signal
                        Horizons), logging
data/                   yfinance fetchers, index catalogue + constituent
                        resolution (universe), two-tier cache, circuit breakers,
                        per-exchange trading calendars (calendars.py)
engines/                aarambh (forecast, purged walk-forward), nirnay (breadth)
analytics/              OU, Hurst/DFA, robust-quantile z-scores, HMM/GARCH/CUSUM,
                        breaks, analogs (Mahalanobis precedent matcher)
convergence/            cross-validator, conviction (DDM), divergence,
                        normalization, intelligence (calibration + walk-forward)
ui/                     theme, components, tabs (Convergence/Aarambh/Nirnay/
                        Precedent/Diagnostics/Data)
research/               tuning & validation harnesses (Aarambh/Nirnay/analog
                        sweeps, marker/hero studies) + run_tuning.py orchestrator
```

Re-tuning: `python3 research/run_tuning.py --list` shows the suite;
`python3 research/run_tuning.py` re-runs all studies and writes one consolidated
report (`research/reports/`) plus a current-vs-validated reference for every tuned
constant. It reports only — config is applied by hand after review.

---

## Interpreting the output

- **Hero card** — normalized convergence signal and the Aarambh / Nirnay contributions.
- **Aarambh tab** — price with the model's forward expected-price projection (an
  implied target + uncertainty cone) and the expected-forward-return forecast driving
  it; model quality shows **Val IC** and the train→val gap (overfit detector).
- **Precedent tab** — the most statistically-similar historical states (Mahalanobis)
  and what the target did next, at the lens horizons; an empirical base rate to read
  *alongside* Aarambh (agreement strengthens conviction, disagreement is a divergence).
- **Diagnostics → Intelligence Center** — calibration state and the **walk-forward
  IC** chart (the durability verdict).

Rule of thumb: trust the **Val IC** and the **walk-forward consistency**, not any
single conviction reading. Across the universe the (leakage-free) directional edge is
modest and concentrated at **10–20d** — the precedent base rate is strongest as a
~10d confirmer, and is best treated as fading in the recent regime.

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Outputs are
statistical signals with weak, regime-dependent, out-of-sample edge — not predictions.
Markets are noisy and the validated ICs are modest. Do not make trading or investment
decisions solely on this software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
