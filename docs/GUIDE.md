# Tattva · Guide

How Tattva works, how to read each page, and the data it runs on. The [README](../README.md)
is the overview; this is the detail.

## How It Works

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

## Reading the Pages

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
