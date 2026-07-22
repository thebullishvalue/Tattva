# Tattva — Research & Tuning Suite

Every tuned constant in Tattva was chosen by a computational study, not by hand.
This folder holds those studies and an **interactive, segment-aware orchestrator**
to run them.

> **Philosophy:** these scripts *measure and recommend* — they never auto-edit
> config. Each tuning decision needs human judgement (parameter interactions,
> regime shifts, signal redundancy). Run → read the report → apply by hand,
> guided by the rationale comments in `core/config.py` and the CHANGELOG.

The suite reflects the current system: a **single fixed forecast horizon** (the
Signal-Horizon lens selector was removed), **commodities and individual stocks in
Nirnay-Swayam self mode**, **per-instrument / per-asset `InstrumentConfig`**, and the
fixed **1/3/5/10/20/60d** precedent term structure.

## One command — interactive or scripted

```bash
python3 research/run_tuning.py                 # interactive menu (on a terminal)
python3 research/run_tuning.py --list          # show every segment + study + ETAs
python3 research/run_tuning.py --all           # run EVERY tuning study end-to-end
python3 research/run_tuning.py --segment engines   # run one segment (tier)
python3 research/run_tuning.py --tests         # run ONLY the fast correctness tests
python3 research/run_tuning.py --only analog,nirnay    # a hand-picked subset
python3 research/run_tuning.py --skip aarambh_full     # everything except the long sweep
```

Run with **no flags on a terminal** and you get a menu: run everything end-to-end,
pick a **segment** (tier), pick specific studies, or run just the tests. (Piped /
non-interactive with no flags defaults to the fast tests, never a silent multi-hour
run.)

**Preflight.** Every *tuning* study pulls the same 9-year universe via
`fetch_commodity_dataset` (cache → live), so a single thin/rate-limited fetch would
silently corrupt a multi-hour run. Before any tuning tier runs, the orchestrator warms
that fetch **once** and asserts it's deep and broad enough (`≥ MIN_DATA_POINTS` rows,
`≥ 50` numeric columns, `≥ 60%` of targets present). On failure it aborts (nothing is
tuned). Bypass with `--skip-preflight`, downgrade to a warning with `--preflight-warn`.
The **tests** segment needs no fetch, so it skips preflight.

Output: a consolidated, timestamped report in `research/reports/`, ending with a
**current-vs-validated reference** for every tuned constant (these are the
`InstrumentConfig` defaults; per-instrument registry entries may override).

## Segments

The orchestrator walks the system **bottom-up** — each tier tuned before the layers
that consume it.

| segment | tier | key studies | informs |
|---|---|---|---|
| **engines** | T1 | `aarambh_full`, `aarambh_maxmin`, `nirnay`, `nirnay_index`, `nirnay_swayam`, **`swayam`**, `ddm` | Aarambh train sizes/ensemble/PCA; **basket-mode** Nirnay knobs; **Swayam self-ensemble grid** (`swayam_lengths`/`roc_frac`); DDM smoothing |
| **scope** | T2 | `precedent_univ` | validates the single **forecast horizon (10d)** + the **precedent term structure** (1/3/5/10/20/60d) |
| **analog** | T3 | `analog`, `analog_confirm` | `ANALOG_W_*`, `TOP_N`, recency, feature set |
| **validation** | T4 | `precedent_model`, `precedent_horiz` | purged model vs tuned analog, by asset class and across horizons |
| **interp** | T5 | `markers`, `hero_thresholds`, `hero`, `calibration_lift`, `conv_weights`, `ui_anchors` | plot markers, hero classification, dim weights, UI tiers |
| **per_asset** | T6 | **`per_asset`** | **per-asset-class `InstrumentConfig` recommendations for the classes no other tier covers** — US indices, ETF, and the stock classes computed from **India = Nifty 100**, **US = Nasdaq 100**. (commodity → `swayam`, India indices → `nirnay_index`, FX/Jeera → `nirnay` — each class is tuned exactly **once**, no recomputation) |
| **tests** | — | `t_*` | fast assertion-based regression (no fetch, no tuning) |

### The two new studies (this refactor)

- **`swayam` (`swayam_tuning_study.py`)** — sweeps the Swayam grid the self-mode
  targets use: the `swayam_lengths` timescale span (count + spread) and
  `swayam_roc_frac`, scored by breadth-spread IC on the self-mode commodities.
- **`per_asset` (`per_asset_config_study.py`)** — recommends the tuning knobs for the
  asset classes **no other study covers** (so the suite tunes each class exactly once,
  no redundant recomputation): **US indices** and the **ETF universe** (basket, MSF
  sweep) plus the two **stock** classes computed from broad universes — **India stocks =
  the Nifty 100 constituents, US stocks = the Nasdaq 100 constituents** (self, Swayam
  grid sweep) — so the asset-class stock config is representative, not fit to a handful
  of names. It prints a coverage map showing where the remaining classes are tuned
  (commodity → `swayam`, India indices → `nirnay_index`, FX/Jeera → `nirnay`). Heavy (a
  Swayam ensemble per stock); lower `STOCK_UNIVERSE_CAP` for a faster smoke pass.

## Correctness tests (the `tests` segment)

Fast, assertion-based regression tests for edge-case paths that ship unexercised by
the live universe. Run the whole segment with `--tests`, or a single file directly
(`python3 -m research.<name>`):

| Test | Covers |
| --- | --- |
| `test_instrument_configs.py` | per-instrument config registry: completeness, defaults == former globals, routing parity, India-index copy rule, stock asset-class registration, tuning isolation |
| `test_nirnay_swayam.py` | Swayam ensemble: schema parity, byte-identity, causality/no-repainting, leakage guard, volume degeneracy, determinism |
| `test_stock_targets.py` | individual-stock fetch + free-form symbol resolution (NSE→BSE), column injection, registration idempotency |
| `test_convergence_integrity.py` | convergence signal-chain non-degeneracy + says-vs-does identity |
| `test_analog_series.py` | analog prediction series causality (no repainting) |
| `test_regime_equivalence.py` | regime njit kernel == the object reference implementation |
| `test_polarity.py` | `apply_polarity` inverse-basket path |
| `test_hero_verdict.py` | hero-verdict decision table |
| `test_calendars.py` | per-exchange freshness counting |

## Shared conventions

- **Honest metric:** non-overlapping (stride = horizon) OOS Spearman IC.
- **Per-instrument:** the tuned constants are `InstrumentConfig` DEFAULTS; the
  `per_asset` study recommends per-class values, and any single instrument can diverge
  via its `INSTRUMENT_CONFIGS` entry.
- **Data:** all studies read the cached real universe (`~/.cache/tattva`) offline.
- Each script is standalone (`python3 research/<name>.py`); a path shim at the top
  puts the repo root on `sys.path` so `from core...` resolves.
