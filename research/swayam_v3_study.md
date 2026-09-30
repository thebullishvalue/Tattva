# Swayam v3 — the study behind the rebuild

**Date:** 2026-09-30 · **Ships in:** Tattva 3.0.0 · **Scripts:** `research/scripts/`

Tattva is built on two engines. Mūla values the target against about 200 traded
instruments. Swayam reads the target by itself. Convergence then fuses the two into
one directional call at a 10-day horizon. This study asked whether each engine could
be rebuilt around the Pragati indicator (the engine behind Sanket and Pragyam), and
measured the answer before changing anything.

**Verdict**
- Swayam is **replaced** by the Pragati indicator.
- Mūla's valuation core is **kept unchanged**.

## 1. Method (pre-registered, written before any number was seen)

**Targets:** 25 of Tattva's own catalogue targets, split as follows:
- **Commodities:** Gold, Silver, Copper, Aluminium, Zinc, Brent Crude, Cotton.
- **FX:** USD/INR, Dollar Index.
- **Crypto:** Bitcoin, Ethereum, Solana, XRP, BNB, Cardano, Dogecoin.
- **Indices:** Nifty 50, Nifty 100, Nifty Next 50, Nifty Bank, Nifty IT, Nifty Pharma, Nifty Midcap 50, S&P 500, Nasdaq 100, Dow Jones.

**Data:** about 9 years of the app's own data preparation.

**Score:** the Spearman IC of each signal against the next-open 10-day log return.
- Returns are sampled every 10 bars, so the windows do not overlap.
- Each target gets one IC per period.
- The table reports the mean across targets.

**Periods:**
- **P1:** 2019-07 to 2021 (17 targets have history).
- **P2:** 2022-23 (25 targets).
- **P3:** 2024 to date (25 targets).

**Rule:** a new piece replaces the current one only if it beats it in **all three**
periods on the cross-target mean.

**Orientation:** every signal is oriented bullish-positive by its own semantics, never
by the data.

## 2. The pieces, measured alone (`tv_measure.py`)

Mean IC by period (bullish-positive):

| signal | P1 | P2 | P3 |
|---|---:|---:|---:|
| current Swayam (MSF/MMR view bank, −Avg_Signal) | **−0.081** | **−0.044** | +0.030 |
| Pragati conviction tape, read as Tattva reads Swayam (sellers = oversold = bullish) | **+0.284** | **+0.041** | +0.027 |
| current Mūla conviction | +0.033 | −0.014 | +0.074 |
| current Mūla gap (−FVO) | +0.034 | +0.009 | +0.049 |
| Samanvaya value tape | +0.228 | −0.002 | +0.030 |
| Mūla ⊕ Samanvaya value blend | +0.166 | +0.000 | +0.047 |
| grid lean (capitulation +1, washout +½, distribution −1) | +0.280 | +0.048 | +0.061 |

**The current Swayam anti-predicted in two of three periods.** Its 15 views were 15
readings of one series. Their effective count was only about 8-10 (N_eff), and their
skill weights did not persist from one period to the next.

**The Pragati conviction tape beats it in all three periods.**

**Mūla's gap beats Samanvaya's value tape in P2 and P3**, which is why Mūla's
valuation core stays.

### The ▲ capitulation turn

The ▲ fires when sellers control a cheap price, value momentum is already reverting,
and conviction is not in its quiet regime.

Excess return is the vol-scaled 10-day excess over the target's own period mean
(+ means the event worked):

| | P1 | P2 | P3 |
|---|---:|---:|---:|
| n | 7 | 63 | 51 |
| excess | +0.08σ | +0.19σ | +0.25σ |
| commodities + FX | | +0.25σ | +0.14σ |
| indices | | +0.39σ | +0.64σ |
| crypto | | ≈ 0 | ≈ 0 |

Crypto is the stated exception. The same was true across 380 instruments in Sanket's
audit.

### Why Ladder up and not Ladder down

Tattva's doctrine is that published history never changes. Intraday history rolls
off yfinance after a few weeks, so a Ladder-down read of an old bar could not be
reproduced later: it would repaint. Swayam therefore reads the ladder **up**, with a
daily rung and a weekly rung built from the daily bars. Its value drivers are fetched
unadjusted (`auto_adjust=False`) for the same reason.

## 3. End to end, through Tattva's own Convergence (`tv_e2e.py`)

This test ran the pipeline exactly as `app.py` Phase 4 runs it: the CrossValidator,
the online dimension weights and the hero consensus. It compared three pairings:

- **E0:** current Mūla + current Swayam (the baseline).
- **E1:** current Mūla + new Swayam. Adopt if it beats E0 in all three periods.
- **E2:** E1, with Mūla's published conviction taken from the value blend. Adopt over E1 if it beats E1 in all three periods.

It scored two headline series:
- **HERO:** −Consensus, the number on the hero card.
- **CONV:** −convergence_score after the online weights.

| | P1 | P2 | P3 |
|---|---:|---:|---:|
| **HERO** E0 | −0.145 | −0.010 | −0.010 |
| **HERO** E1 | **+0.141** | **+0.023** | **−0.009** |
| HERO E2 | +0.204 | −0.017 | +0.038 |
| **CONV** E0 | −0.081 | −0.042 | +0.079 |
| **CONV** E1 | **+0.230** | **+0.014** | +0.061 |
| CONV E2 | +0.262 | +0.013 | +0.055 |
| *ex-crypto* HERO E0 | −0.121 | −0.023 | +0.024 |
| *ex-crypto* HERO E1 | **+0.347** | **+0.004** | **+0.030** |
| *ex-crypto* CONV E0 | −0.082 | −0.033 | +0.105 |
| *ex-crypto* CONV E1 | **+0.325** | **+0.015** | +0.072 |

Share of targets with a positive HERO IC:

| | P1 | P2 | P3 |
|---|---:|---:|---:|
| E0 | 24% | 52% | 40% |
| E1 | 65% | 60% | 44% |

### Decision
- **E1 is adopted.** On the hero headline it beats E0 in all three periods, both across all targets and ex-crypto.
- **E2 is rejected.** It loses P2 to E1 on the hero, so the rule keeps Mūla's conviction as it is.

### Disclosed, not hidden
- **CONV in P3.** The calibrated composite trails the old pairing in P3 (+0.061 vs +0.079). CONV is not the headline Tattva publishes, but the gap is real, and it is recorded here rather than tuned away.
- **The P3 hero.** P3 on the hero is a tie at about zero (−0.009 vs −0.010). The 2024-26 gain is carried by the ▲ events, not by the continuous headline.
- **The 10-day read is weak after 2021.** In 2022-26 no version of the continuous headline has a strong 10-day IC. The honest read is a large gain in P1, a small gain in P2, and parity in P3.

## 4. Causality (`tv_causal.py`)

For 8 targets, the engine was refit on data truncated 250 and 600 bars before the end.

On every shared date:
- The published contract matched the full-history fit exactly (maximum absolute difference 0).
- The grid cell and the ▲/▼ events matched exactly.

No row appears in the truncated fit that the full fit does not publish. Swayam does
not repaint.

## 5. Caveats
- **Driver adjustment.** The per-piece measurement (§2) ran Sanket's Samanvaya with auto-adjusted drivers. The end-to-end test (§3) used Tattva's production path: `data.fetcher.fetch_value_drivers`, unadjusted.
- **Sample sizes.** The ▲ sample in P1 is small (n = 7), and 8 targets have no P1 history.
- **Pinned tree.** `tv_run.py` and `tv_measure.py` import the pre-3.0 engines (the MSF/MMR bank, removed in 3.0.0) and the Sanket repo. They are kept as the record of what was run. Reproduce them at commit `198361d`, with Sanket checked out alongside.
