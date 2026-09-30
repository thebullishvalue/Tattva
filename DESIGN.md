# Tattva — from-scratch design (v3)

*Written 2026-09-30, before any result from this design was computed. The rules in §6 are fixed here;
anything changed after results are seen is reported as post-hoc, next to the pre-registered result.*

## 1. Purpose

For one **target** (a commodity, an FX rate, a crypto asset, an equity index, or a listed stock), Tattva
makes **one directional call for the next 10 trading days**: BUY, SELL or NO CALL. The call comes with
an expected move and a confidence, and it is built by converging two independent bodies of evidence:

- **Mūla (मूल, "root")** — what the *world* implies for this asset: where it stands against the
  moves of the global markets it is exposed to.
- **Swayam (स्वयम्, "self")** — what the asset's *own market* says: its trend, stretch and
  participation, read from its own price and volume alone.

Tattva grades itself. Every forecast it has ever made is kept, scored when its 10 days are up, and
shown as a track record. When the evidence is weak or the engines disagree, it says NO CALL.

**Invariants.**
- No look-ahead: every value dated *t* uses only data available at the close of *t*.
- No repaint: re-running on more data only extends history. Prices are unadjusted, the universe is
  declared, and refits happen on a fixed calendar.
- Free data only: yfinance, plus the published Google Sheet for the two sheet-sourced series.

## 2. What the old system taught (why it is not reused)

1. **Too few independent observations per target.** At 10 days, nine years of one target holds about
   225 non-overlapping outcomes. Anything fitted per target is mostly noise. **Estimation must be
   pooled** across targets that behave alike.
2. **The engines spoke different languages.** Oscillators, breadth percentages and regime labels had
   to be reconciled by hand-set conventions, and one convention was measured backwards: the old
   Swayam anti-predicted. **Both new engines output the same thing, a forecast of the target's 10-day
   return in units of its own volatility.** Convergence then becomes forecast combination, weighted
   by measured skill.
3. **The sign of a signal differs by asset class.** Crypto trends where commodities and indices
   revert at 10 days. **The learner estimates the sign per class** instead of hard-coding it.
4. **Abstaining is an output.** A call is only as good as its track record, so the record is part
   of the product.

## 3. Architecture

```
data  ─►  features (Mūla: world-relative │ Swayam: self)  ─►  pooled purged learner  ─►  F_mula, F_swayam
                                                                                              │
                                              Convergence: skill-weighted stack ─► F ─► call + confidence
                                                                                              │
                                                                  track record (walk-forward, per target & pool)
```

### 3.1 Data
- Daily bars, unadjusted (`auto_adjust=False`), from 2008-01-01 or the first available date.
- **Targets:** the catalogue — commodities, FX, crypto, India and US indices, the Nifty 500 — plus
  free-form stocks.
- **World factors (Mūla's inputs):** about 20 named, liquid, long-history series:
  - equities: S&P 500, Nasdaq 100, Russell 2000, EM, Europe, Japan, China, India
  - rates: long, intermediate and short US Treasuries
  - credit: high yield, investment grade
  - currencies: the dollar index, EUR, JPY, CNY, INR
  - commodities: oil, gold, copper, broad commodities, agriculture
  - volatility: VIX
  - crypto: Bitcoin
- **Leakage guard (data-driven, causal):** a factor is excluded for a target when it is the target
  itself, or when the trailing-year correlation of their daily returns exceeds 0.95. It is re-judged
  at each refit.

### 3.2 Mūla — world-relative value and transmission
- **Exposure.** An exponentially weighted ridge regression of the target's daily log return on
  factor returns at lags 0 and 1 (Dimson; this handles India/US close asynchrony). Half-life is 250
  days. It is updated recursively, and the betas at *t−1* price day *t*.
- **Residual.** e_t = r_t − β̂_{t−1}·x_t. This is one step ahead, so it is out-of-sample by construction.
- **Features (pre-registered):**
  - `gap20`, `gap60`: the sum of residuals over 20 and 60 days, divided by the residual SD × √L. It
    measures how far the target has moved *beyond or short of* what its world implied.
  - `drift60`: β̂ · (factors' 60-day mean return) × 10, divided by the target's σ. It is the direction
    the world is currently pushing the target.
- **Display:** a fair-value path P_t·exp(−Σ₆₀ e), a "price the world implies", and the per-factor
  contributions.

### 3.3 Swayam — self-read
**Features (pre-registered)**, all volatility-normalised:
- `tsm20`, `tsm60`, `tsm250`: the L-day log return ÷ (σ₆₀ √L)
- `rev5`: the 5-day log return ÷ (σ₆₀ √5)
- `stretch`: (close − EMA₂₀) ÷ ATR₂₀
- `range60`: the close's position in its 60-day high-low range, −½ to +½
- `volreg`: log(σ₂₀ / σ₂₅₀)
- `flow20`: Σ sign(r)·volume ÷ Σ volume over 20 days (0 when there is no volume)

### 3.4 The learner (shared by both engines)
- **Label:** the next-open 10-day log return (open t+1 to open t+11) ÷ σ₆₀ at t, clipped to ±4.
  The label is known only after t+11.
- **Pools:**
  - `macro`: commodities and FX
  - `crypto`
  - `equity`: all indices; free-form stocks are scored with the equity model
- **Hierarchical ridge:** a global fit across all pools, then each pool is fitted with a penalty
  toward the global coefficients.
  - Samples are decay-weighted, with a 3-year half-life.
  - Features are standardised with statistics from the fit window only.
- **Purged, fixed-calendar refits.** The model refits on the first trading day of each month, using
  only samples whose label ended on or before that day. The forecast at t uses the latest refit on or
  before t. The first forecast comes after 2 years of pooled samples.
- The same learner produces **F_mula** from the Mūla features and **F_swayam** from the Swayam
  features.

### 3.5 Convergence — the call
- **Stack:** F = w_m·F_mula + w_s·F_swayam, with w ≥ 0.
  - The weights are fitted per pool, on the engines' own past out-of-sample forecasts (purged,
    monthly), and shrunk toward ½/½.
  - An engine with no measured skill gets weight near 0 automatically.
- **Conviction:** z = F ÷ the causal SD of F over the target's past 250 days.
- **Call:** BUY or SELL when |z| ≥ 0.5 **and** both engines agree in sign. Otherwise NO CALL, with a
  lean shown as the sign of F.
- **Expected move:** F × σ₆₀ × √10, in percent.
- **Confidence:** the target's and the pool's walk-forward record, shown as a non-overlapping IC and
  a hit rate with standard errors.

### 3.6 Product
A Streamlit terminal with five pages:
- **Call:** the verdict, expected move, engine agreement and track record.
- **Mūla:** the implied fair-value path, gap, drift and factor contributions.
- **Swayam:** the feature contributions, trend and stretch.
- **Record:** walk-forward performance by period and pool, hit rate, and the realised return by
  conviction bucket.
- **Data:** freshness and downloads.

## 4. Fixed hyperparameters (not tuned)

| setting | value |
|---|---|
| Exposure half-life | 250 days |
| Ridge λ, standardised features | 10 × the number of features |
| Pool-to-global shrinkage | 100 × the number of features |
| Sample decay half-life | 3 years |
| Warm-up | 2 years |
| Call threshold | \|z\| ≥ 0.5 |
| Clip | ±4 |
| Horizon | 10 days, from the next open |

## 5. Evaluation

- **Targets:** every catalogue target yfinance serves.
- **Periods:**
  - P0: 2012 to 2019-06
  - P1: 2019-07 to 2021
  - P2: 2022-23
  - P3: 2024 to date
- **Metric:** per-target Spearman IC of the forecast against the next-open 10-day log return,
  non-overlapping (stride 10), averaged across targets. The same is also reported per pool.
- **Calls:** the hit rate and mean realised return of BUY and SELL calls.
- **Baselines:**
  - the old Tattva hero, E0 in research/swayam_v3_study.md on the 3.0.0 branch: −0.145 / −0.010 /
    −0.010 for P1-P3 on 25 targets
  - 60-day time-series momentum, `tsm60` alone

## 6. Pre-registered rules

1. An engine counts as having skill if its cross-target mean IC is > 0 in at least 3 of the 4
   periods.
2. Tattva's combined forecast ships if:
   - its mean IC is > 0 in every period, and
   - it is ≥ the old hero in P1, P2 and P3 on the shared 25 targets, and
   - its BUY/SELL hit rate is > 50% overall.
3. If a rule fails, the failure is reported. Any change made after seeing results is labelled
   post-hoc and re-scored only on periods it was not designed on.
