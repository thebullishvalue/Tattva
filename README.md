# TATTVA — तत्त्व

**One 10-day call per target, from the world and the self** · v3.0.0 · *@thebullishvalue*

> *Tattva (तत्त्व)* — "principle, essence": what remains when the evidence converges.

For one **target**, Tattva makes a single directional call for the next 10 trading days: **▲ BUY**,
**▼ SELL** or **● NO CALL**. It shows the call with its expected move, and next to it the record of
every call it has made on that target.

Targets can be:
- a commodity
- an FX rate
- a crypto asset
- an Indian or US equity index
- any listed stock

It runs on free yfinance data. There are no keys, no secrets and no database.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it decides

| | question | reads | signal |
|---|---|---|---|
| **Mūla** (मूल, root) | Has the target run beyond or short of what its world implied? | about 25 global markets: equities, Treasuries, credit, currencies, oil, metals, VIX, Bitcoin | the target's 60-day return *unexplained* by those markets; a daily-re-estimated exposure model supplies the explanation. Overshoot reverts. |
| **Swayam** (स्वयम्, self) | How stretched is the target on its own? | the target's own bars only | 60- and 250-day trend and its place in the 60-day range. Stretch reverts. |
| **Convergence** | Up or down over 10 days, and is there enough to say so? | both engines + the target's own drift | timing z = ½(Mūla + Swayam), scaled; expected move = drift + 0.05·z. It calls only when \|z\| ≥ 0.5 **and** the engines agree. |

**Why drift is part of the call.** Both engines measure timing *relative to the target's normal
behaviour*. A BUY or SELL is judged on the actual move, so on an asset that usually rises, a
moderate "down" reading lowers the expected move without reversing it. The Call page says so when
that happens.

## What it has earned

The track record below covers every catalogue target, scored on non-overlapping 10-day outcomes.
The study behind it is [`research/tattva_v3_study.md`](research/tattva_v3_study.md), and the
Record page recomputes it live.

| | 2012–mid-2019 | mid-2019–2021 | 2022–23 | 2024– |
|---|---:|---:|---:|---:|
| Timing IC (Tattva) | +0.042 | +0.062 | +0.063 | +0.011 |
| Call hit rate | 55.2% | 53.7% | 56.9% | 55.0% |
| Mean 10-day return per call | +0.28% | +0.82% | +0.23% | +0.41% |
| Hit rate of drift alone | 55.0% | 52.9% | 50.0% | 52.4% |
| Hit rate of always long | 56.5% | 56.0% | 50.3% | 52.6% |

**Read this honestly:**
- **Edge.** The edge is modest. It is strongest on equity indices (59-61% hit rate) and absent on
  crypto.
- **Against drift.** Tattva's calls are at or above drift alone in every period.
- **Against holding long.** In the 2012-2021 bull runs, simply holding long did better.
- **The recent period.** Timing skill in 2024 onward is thin, and it comes from Swayam.
- **How the design was chosen.** This design was adopted *after* a pre-registered design failed.
  It was selected on 2012-2019 only, but the later periods had been seen. The clean test is
  forward, and the app keeps scoring every new call.

## Guarantees
- **No look-ahead.** Every value on a date uses only bars up to that date.
- **No repaint.** Prices are unadjusted, so a dividend or split never rewrites published history.
  `tests/test_causality.py` re-runs on data truncated 250 and 600 sessions early and requires exact
  agreement on every shared date.

## Layout

```
app.py                     the terminal (Call · Mūla · Swayam · Record · How it works)
tattva/universe.py         the declared universe: targets, world factors, pools
tattva/data.py             unadjusted yfinance + sheet fetch, disk cache, stock symbol resolution
tattva/features.py         Mūla's exposure model and gap, Swayam's stretch, the 10-day label
tattva/engine.py           Mūla, Swayam, convergence, the call
tattva/evaluate.py         the track record
tattva/charts.py           charts (one colour per role: Mūla blue, Swayam orange, Tattva aqua)
tests/test_causality.py    no look-ahead / no repaint
research/                  the study, its scripts, and evaluate_catalogue.py
DESIGN.md                  the pre-registered design and rules (committed before any result)
```

The cache lives in `~/.cache/tattva`; set `TATTVA_CACHE` to move it. A universe download is reused
for 6 hours. If a download fails, the last good snapshot is used rather than a smaller universe.

## Data notes
- **Sheet-sourced series.** Jeera and the Nifty 50 P/E come from published Google Sheets; everything
  else comes from yfinance.
- **Dropped indices.** Sixteen NSE sector indices from earlier versions are not listed, because
  yfinance serves no history for them.
- **Stocks.** A free-form stock is read exactly like a catalogue target: against the same world, on
  its own bars.

## Disclaimer

Tattva is a research and educational tool, not investment advice. Its edge is modest and
regime-dependent. See [LICENSE.md](LICENSE.md).
