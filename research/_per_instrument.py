"""Shared PER-INSTRUMENT recommendation helper for the tuning studies.

The commodity, currency, India-index, US-index and ETF classes are tuned PER
INSTRUMENT (each target carries its own InstrumentConfig knobs; see
core.config.PER_INSTRUMENT_TUNING). The India/US stock classes stay ASSET-LEVEL
and never route through here.

Each per-instrument study sweeps a knob across a grid and records the per-TARGET
IC. This module turns that {value: {target: IC}} table into a per-instrument
recommendation with a NOISE GATE, and prints a copy-paste-ready
`_PER_INSTRUMENT_OVERRIDES` snippet for core/config.py.

Why a gate: per-instrument argmax over a grid, on the near-zero / noisy ICs these
signals produce, will pick a "best" value by chance for most targets. An override
is recommended ONLY when the target's own best |IC|
  (a) clears an absolute floor (it is a real signal, not noise), AND
  (b) beats its CLASS-DEFAULT value's |IC| by a margin (the per-instrument value
      is genuinely better than just inheriting the class default).
Everything else keeps the class default — no invented per-instrument divergence.
"""
from __future__ import annotations
import numpy as np

IC_FLOOR = 0.06     # a target's best |IC| must clear this to be worth an override
IC_MARGIN = 0.03    # best |IC| must beat the class-default value's |IC| by this


def per_instrument_reco(field, table, default_value, own_targets, *,
                        value_fmt=repr, floor=IC_FLOOR, margin=IC_MARGIN):
    """Print a per-target block for one knob and return adopted overrides.

    table         : {grid_value: {target: ic}}  (ic may be NaN / missing)
    default_value : the value that lives in the class default (the baseline)
    own_targets   : the targets this study OWNS for per-instrument wiring; other
                    targets in the table are shown as context only ("--", never
                    adopted — they are cross-sections for statistical power).
    Returns {target: {field: value}} for the targets that cleared the gate.
    """
    targets = sorted({t for v in table for t in table[v]})
    print(f"    per-instrument {field}  (gate: |IC|≥{floor} and beat default by ≥{margin})", flush=True)
    adopted: dict[str, dict] = {}
    for t in targets:
        dv = table.get(default_value, {})
        dic = abs(dv[t]) if t in dv and np.isfinite(dv.get(t, np.nan)) else np.nan
        cand = [(abs(table[v][t]), v) for v in table
                if t in table[v] and np.isfinite(table[v][t])]
        bic, bv = max(cand) if cand else (np.nan, None)
        owned = t in own_targets
        gate = (owned and np.isfinite(bic) and bic >= floor and bv != default_value
                and (not np.isfinite(dic) or bic - dic >= margin))
        if gate:
            adopted[t] = {field: bv}
        if not owned:
            flag = "  --context--"
        elif gate:
            flag = "  ADOPT"
        elif np.isfinite(bic):
            flag = "  (noise → default)"
        else:
            flag = "  (n/a)"
        print(f"      {t:<22} best {field}={str(value_fmt(bv)):<18} |IC|={bic:>6.3f}   "
              f"default={str(value_fmt(default_value)):<12} |IC|={dic:>6.3f}{flag}", flush=True)
    return adopted


def merge_overrides(dst: dict, adopted: dict) -> None:
    """Accumulate per-field adopted overrides into a per-target override dict."""
    for t, ov in adopted.items():
        dst.setdefault(t, {}).update(ov)


# ── Percentile-ANCHOR gate (interpretation layer: markers / thresholds / tiers) ─
ANCHOR_REL_GATE = 0.25   # target's own anchor must differ from pooled by ≥ 25%
ANCHOR_MIN_N = 250       # …and have ≥ this many obs (thin targets keep the pooled convention)


def per_instrument_anchor_reco(field, per_target, pooled_default, own_targets, *,
                               rel_gate=ANCHOR_REL_GATE, min_n=ANCHOR_MIN_N,
                               value_fmt=lambda v: round(float(v), 4)):
    """Per-instrument reco for a DISTRIBUTION-ANCHORED constant (a percentile of the
    target's own signal, not an IC). Unlike the IC gate, "adopt" means the target's
    OWN anchor materially DIVERGES from the pooled/class default.

    per_target     : {target: (anchor_value, n_obs)} — this target's own anchor.
    pooled_default : the pooled/class anchor it must diverge from to be worth wiring.
    Adopt only when |anchor − pooled| / |pooled| ≥ rel_gate AND n_obs ≥ min_n (thin
    samples keep the pooled convention — the whole point of the gate). Returns
    {target: {field: value}} for adopted targets; prints a per-target block.
    """
    print(f"    per-instrument {field}  (gate: |Δ vs pooled {pooled_default:.4g}| ≥ "
          f"{rel_gate:.0%} and n ≥ {min_n})", flush=True)
    adopted: dict = {}
    for t in sorted(per_target):
        av, n = per_target[t]
        owned = t in own_targets
        rel = abs(av - pooled_default) / max(abs(pooled_default), 1e-9)
        gate = owned and np.isfinite(av) and n >= min_n and rel >= rel_gate
        if gate:
            adopted[t] = {field: value_fmt(av)}
        flag = ("  ADOPT" if gate else
                "  --context--" if not owned else
                f"  (Δ {rel:.0%} < {rel_gate:.0%} → pooled)" if n >= min_n else
                f"  (n={n} < {min_n} → pooled)")
        print(f"      {t:<22} anchor={str(value_fmt(av)):<10} n={n:<6} "
              f"pooled={pooled_default:.4g}{flag}", flush=True)
    return adopted


def print_overrides_snippet(overrides_by_target: dict, value_fmt=repr) -> None:
    """Print the combined copy-paste block for core.config._PER_INSTRUMENT_OVERRIDES."""
    print("\n" + "-" * 72, flush=True)
    if not overrides_by_target:
        print("  PER-INSTRUMENT: no target cleared the gate — all keep the class "
              "default.\n  (Nothing to wire; the class-level config stands.)", flush=True)
        return
    print("  PER-INSTRUMENT OVERRIDES — paste into core.config._PER_INSTRUMENT_OVERRIDES:", flush=True)
    for t in sorted(overrides_by_target):
        body = ", ".join(f'"{k}": {value_fmt(v)}' for k, v in overrides_by_target[t].items())
        print(f'      "{t}": {{{body}}},', flush=True)
    print("  (Review against the per-target |IC| above before adopting — the gate "
          "is a floor,\n   not a proof; a human still signs off each override.)", flush=True)
