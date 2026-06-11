# TATTVA — तत्त्व

**Unified Commodity Convergence Engine** · v2.0.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — Sanskrit for "principle / essence / reality": the underlying
> truth distilled from the convergence of evidence.

Tattva is a research terminal that produces a single, calibrated directional
signal for a commodity (**Gold, Silver, or Copper**) by converging two
independent engines — a top-down macro **forecast** and a bottom-up **regime
breadth** read — and grading its own out-of-sample edge as it goes.

It runs entirely on free **yfinance** data. No API keys, no secrets, no database.

---

## What it does

For the selected commodity, Tattva runs a 5-phase pipeline and renders a Streamlit
terminal:

| Engine | Question it answers | How |
|---|---|---|
| **AARAMBH** | *Is the macro setup pointing up or down?* | Walk-forward ensemble (Ridge + Huber + ElasticNet + PCA-OLS) **forecasting the forward 10-day return** from trailing macro momentum. |
| **NIRNAY** | *What is the related complex doing bottom-up?* | Per-instrument MSF + MMR oscillators with HMM/GARCH/CUSUM regime detection across a basket of related miners/streamers, aggregated into breadth. |
| **CONVERGENCE** | *Do the two agree, and how strongly?* | Adaptive-weighted, **directional** composite across Direction / Breadth / Magnitude / Regime, smoothed with a Drift-Diffusion filter. |
| **INTELLIGENCE** | *Does the signal actually have edge?* | Optuna TPE calibration of the convergence weights/thresholds with a **purged k-fold CV objective** + held-out tail, plus an automatic **walk-forward IC** durability check. |

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

Then in the sidebar: pick a **Target Commodity** (Gold / Silver / Copper) and click
**Run Analysis**. First run fetches ~9 years of history (cached afterwards) and runs
the full pipeline; subsequent runs are fast.

No configuration is required — there are no secrets or environment variables to set.

---

## How the model works

**Predictive, returns-based.** Aarambh does **not** regress price levels (which is a
spurious regression). It forecasts the **forward 10-day log-return** of the commodity
from **trailing 20-day momentum** of ~112 macro/FX/commodity series — a genuine
ex-ante setup. The forecast drives a directional conviction; out-of-sample skill is
measured by rank **IC**, not R² (a price forecast's magnitude R² is ~0 by nature; the
tradeable information is in the direction).

**Causal PCA, no repainting.** The ~112 collinear macro inputs are reduced to ~20
orthogonal components **inside each walk-forward training window** — fit only on past
data, so a component's value at time *t* never depends on the future. Adding new data
never rewrites history. This stabilises the ensemble (low model spread) while keeping
every input "on."

**Honest validation.** The intelligence calibration optimises a **purged k-fold
cross-validation** objective (robust across time, not one slice) and reports a Val IC
on a genuinely held-out, purged tail. An expanding-window **walk-forward IC** runs
every analysis and is charted in Diagnostics — consistently positive bars = durable
edge; a couple of spikes = a lucky regime.

---

## Data sources (all yfinance)

- **Target & predictors:** the commodity front-month future plus the macro universe in
  `core/config.py` — `GLOBAL_MACRO_MAP` (bond/rates/equity/risk/real-asset ETFs) +
  `MACRO_SYMBOLS_YF` (commodities + FX).
- **Nirnay basket:** per-commodity miners/streamers in `COMMODITY_BASKETS`
  (`core/config.py`).

Every external call is wrapped in a two-tier cache (memory + disk), a per-service
circuit breaker, retry-with-backoff, and a stale-snapshot fallback — so the UI keeps
working through transient yfinance outages.

---

## Configuration

| What | Where |
|---|---|
| Target commodities | `COMMODITY_TARGETS` in `core/config.py` |
| Nirnay baskets | `COMMODITY_BASKETS` in `core/config.py` |
| Macro predictor universe | `GLOBAL_MACRO_MAP` + `MACRO_SYMBOLS_YF` |
| Forecast horizon / momentum window | `FWD_HORIZON`, `FWD_MOM_K` in `app.py` |
| PCA components | `n_pca_components` in the `engine.fit(...)` call (`app.py`) |
| Walk-forward / train sizes | `core/config.py` (`MIN_TRAIN_SIZE`, `MAX_TRAIN_SIZE`, `MIN_DATA_POINTS`) |

In-app: the sidebar **Model Configuration** lets you deselect predictors (the full
universe is on by default). Calibrated profiles persist to
`~/.cache/tattva/intelligence/profiles.json` (one per commodity).

---

## Project structure

```
app.py                  Streamlit entrypoint + 5-phase orchestration
core/                   config (universe, baskets, thresholds), logging
data/                   yfinance fetchers, two-tier cache, circuit breakers
engines/                aarambh (forecast), nirnay (regime breadth)
analytics/              OU, Hurst/DFA, conformal, HMM/GARCH/CUSUM, breaks
convergence/            cross-validator, conviction (DDM), divergence,
                        normalization, intelligence (calibration + walk-forward)
ui/                     theme, components, tabs (Convergence/Aarambh/Nirnay/
                        Diagnostics/Data)
```

---

## Interpreting the output

- **Hero card** — normalized convergence signal and the Aarambh / Nirnay contributions.
- **Aarambh tab** — price + expected-forward-return forecast; model quality shows
  **Val IC** and the train→val gap (overfit detector).
- **Diagnostics → Intelligence Center** — calibration state and the **walk-forward
  IC** chart (the durability verdict).

Rule of thumb: trust the **Val IC** and the **walk-forward consistency**, not any
single conviction reading.

---

## Disclaimer

Tattva is a **research and educational tool**, not investment advice. Outputs are
statistical signals with weak, regime-dependent, out-of-sample edge — not predictions.
Markets are noisy and the validated ICs are modest. Do not make trading or investment
decisions solely on this software's output. See [LICENSE.md](LICENSE.md).

---

*© 2026 @thebullishvalue. All rights reserved. See [LICENSE.md](LICENSE.md) and
[CHANGELOG.md](CHANGELOG.md).*
