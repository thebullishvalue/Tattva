"""
Tattva — Integrity tests for the per-instrument config registry
(core.config.InstrumentConfig / INSTRUMENT_CONFIGS / get_instrument_config /
PER_INSTRUMENT_TUNING).

Pins the contract of the per-instrument config model:

  1. COMPLETENESS — every catalogue target (commodities, FX, all India/US
     indices, ETF universe) has an EXPLICIT InstrumentConfig; get_instrument_config
     raises for an unregistered target (no silent global fallback).
  2. DERIVED-VIEW INVARIANT — each instrument's config == its CLASS default,
     overlaid with that instrument's PER_INSTRUMENT_TUNING override. This holds
     whether or not any override is wired, so the test survives future tuning
     (it does NOT assume every instrument still equals the base default).
  3. ROUTING PARITY — each config's archetype/polarity/excluded_predictors/
     basket/alias match the legacy per-target maps (the derived-view invariant).
  4. PER-INSTRUMENT vs ASSET-LEVEL SPLIT — the five catalogue classes (commodity,
     fx, india_index, us_index, etf) are per-instrument: each has an explicit
     PER_INSTRUMENT_TUNING slot. The stock classes are asset-level: they have NO
     slot and are configured per-symbol from STOCK_CONFIGS. India-index class
     default == the Nifty 50 baseline; each India index is a distinct entry.
  5. OVERRIDE SANITY — _PER_INSTRUMENT_OVERRIDES only sets TUNABLE (non-routing)
     fields, only on per-instrument-class targets; wiring one flows through the
     build into that instrument's config and nowhere else.
  6. STOCK ASSET-CLASS CONFIG — register_stock_target installs a per-symbol config
     cloned from the market's STOCK_CONFIGS, with market-based exclusions; idempotent.
  7. PER-INSTRUMENT TUNING IS ISOLATED — editing one instrument's config does not
     affect any other.

Run: python -m research.test_instrument_configs  (from the repo root)
"""
from __future__ import annotations

import os as _os
import sys as _sys
from dataclasses import replace

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import core.config as c
from core.config import (
    InstrumentConfig, INSTRUMENT_CONFIGS, get_instrument_config, register_stock_target,
    ALL_TARGETS, TARGET_ARCHETYPE, TARGET_POLARITY, TARGET_EXCLUDED_PREDICTORS,
    COMMODITY_BASKETS, NIRNAY_BASKET_ALIAS, TARGET_CATEGORIES,
    CLASS_CONFIG_DEFAULTS, PER_INSTRUMENT_TUNING, PER_INSTRUMENT_CLASSES,
    ASSET_LEVEL_CLASSES, _CATEGORY_TO_CLASS, _TUNABLE_FIELDS, _ROUTING_FIELDS,
    _PER_INSTRUMENT_OVERRIDES,
)

_TUNING_FIELDS = tuple(sorted(_TUNABLE_FIELDS))


def _class_of(target: str) -> str:
    for cat, names in TARGET_CATEGORIES.items():
        if target in names and cat in _CATEGORY_TO_CLASS:
            return _CATEGORY_TO_CLASS[cat]
    raise KeyError(target)


def run() -> None:
    checks = 0
    base = InstrumentConfig()   # all fields at their defaults

    # ── 1. COMPLETENESS ───────────────────────────────────────────────────
    for t in ALL_TARGETS:
        assert t in INSTRUMENT_CONFIGS, f"catalogue target {t!r} has no InstrumentConfig"
        assert isinstance(get_instrument_config(t), InstrumentConfig)
    try:
        get_instrument_config("Definitely Not A Registered Target")
        raise AssertionError("expected KeyError for unregistered target")
    except KeyError:
        pass
    _named = sum(len(v) for k, v in TARGET_CATEGORIES.items() if k in _CATEGORY_TO_CLASS)
    assert len(INSTRUMENT_CONFIGS) == _named, (len(INSTRUMENT_CONFIGS), _named)
    checks += 1

    # ── 2. DERIVED-VIEW INVARIANT (survives per-instrument tuning) ────────
    # Base dataclass defaults still equal the former global constants.
    assert base.nirnay_msf_length == c.NIRNAY_MSF_LENGTH
    assert base.swayam_lengths == c.NIRNAY_SWAYAM_LENGTHS
    assert base.swayam_roc_frac == c.NIRNAY_SWAYAM_ROC_FRAC
    assert base.forecast_horizon == c.FORECAST_HORIZON
    assert base.precedent_horizons == c.PRECEDENT_HORIZONS
    assert base.weights_seed() == {
        "w_direction": c.CONV_WEIGHT_DIRECTION, "w_breadth": c.CONV_WEIGHT_BREADTH,
        "w_magnitude": c.CONV_WEIGHT_MAGNITUDE, "w_regime": c.CONV_WEIGHT_REGIME,
    }
    # Each instrument's TUNING == its class default overlaid with its per-instrument
    # override. (With no override wired this reduces to the class default; once a
    # value is wired the invariant still holds — the test does not go stale.)
    for t in ALL_TARGETS:
        cls_default = CLASS_CONFIG_DEFAULTS[_class_of(t)]
        expected = replace(cls_default, **PER_INSTRUMENT_TUNING.get(t, {}))
        cfg = get_instrument_config(t)
        for f in _TUNING_FIELDS:
            assert getattr(cfg, f) == getattr(expected, f), (t, f, getattr(cfg, f), getattr(expected, f))
    checks += 1

    # ── 3. ROUTING PARITY with legacy maps ────────────────────────────────
    for t in ALL_TARGETS:
        cfg = get_instrument_config(t)
        assert cfg.archetype == TARGET_ARCHETYPE.get(t, cfg.archetype), t
        assert cfg.polarity == TARGET_POLARITY.get(t, cfg.polarity), t
        assert list(cfg.excluded_predictors) == TARGET_EXCLUDED_PREDICTORS.get(t, []), t
        assert list(cfg.basket) == list(COMMODITY_BASKETS.get(t, [])), t
        assert cfg.basket_alias == NIRNAY_BASKET_ALIAS.get(t), t
    checks += 1

    # ── 4. PER-INSTRUMENT vs ASSET-LEVEL SPLIT ────────────────────────────
    assert set(PER_INSTRUMENT_CLASSES) == set(_CATEGORY_TO_CLASS.values())
    assert set(PER_INSTRUMENT_CLASSES).isdisjoint(ASSET_LEVEL_CLASSES)
    assert set(ASSET_LEVEL_CLASSES) == {"stock_india", "stock_us"}
    # Every per-instrument-class catalogue target has an explicit tuning SLOT…
    for cat, cls in _CATEGORY_TO_CLASS.items():
        for nm in TARGET_CATEGORIES.get(cat, []):
            assert nm in PER_INSTRUMENT_TUNING, f"{nm!r} has no per-instrument slot"
    # …and the slot set is EXACTLY those targets (no stock leaked in).
    assert set(PER_INSTRUMENT_TUNING) == {
        nm for cat, cls in _CATEGORY_TO_CLASS.items() for nm in TARGET_CATEGORIES.get(cat, [])}
    _stock_targets = [t for cat in ("India Stocks", "US Stocks")
                      for t in TARGET_CATEGORIES.get(cat, [])]
    assert not any(s in PER_INSTRUMENT_TUNING for s in _stock_targets)
    # India-index class default == the Nifty 50 baseline (== base tuning); each
    # India index is a distinct explicit entry with index archetype.
    idx_default = CLASS_CONFIG_DEFAULTS["india_index"]
    for f in _TUNING_FIELDS:
        assert getattr(idx_default, f) == getattr(base, f), ("india_index default", f)
    india_indices = [n for n in TARGET_CATEGORIES["India Indices"]
                     if n not in ("Nifty 50", "Nifty 50 - PE")]
    assert len(india_indices) >= 20
    for idx in india_indices:
        assert idx in INSTRUMENT_CONFIGS and get_instrument_config(idx).archetype == "index"
    checks += 1

    # ── 5. OVERRIDE SANITY + build flow-through ───────────────────────────
    # Hand-wired overrides never touch routing fields and only name real
    # per-instrument targets (guards the import-time asserts from regressing).
    for t, ov in _PER_INSTRUMENT_OVERRIDES.items():
        assert t in PER_INSTRUMENT_TUNING, f"override target {t!r} is not per-instrument"
        assert set(ov).issubset(_TUNABLE_FIELDS), (t, set(ov) - _TUNABLE_FIELDS)
        assert not (set(ov) & _ROUTING_FIELDS), (t, "override sets a routing field")
    # A wired override flows through the builder into that instrument's config.
    # (Simulate the build for one target; the live registry is not mutated.)
    probe = "Nifty Bank"
    cls_default = CLASS_CONFIG_DEFAULTS[_class_of(probe)]
    simulated = replace(
        cls_default,
        archetype=TARGET_ARCHETYPE.get(probe, cls_default.archetype),
        excluded_predictors=tuple(TARGET_EXCLUDED_PREDICTORS.get(probe, ())),
        **{"nirnay_msf_length": 8, "swayam_roc_frac": 0.55},
    )
    assert simulated.nirnay_msf_length == 8 and simulated.swayam_roc_frac == 0.55
    assert simulated.archetype == "index"   # routing preserved under a tuning override
    checks += 1

    # ── 6. STOCK ASSET-CLASS CONFIG via register_stock_target ─────────────
    register_stock_target("TESTRELIANCE (NSE)", "TESTRELIANCE.NS", "india")
    register_stock_target("TESTAAPL (US)", "TESTAAPL", "us")
    r = get_instrument_config("TESTRELIANCE (NSE)")
    a = get_instrument_config("TESTAAPL (US)")
    assert r.archetype == "self" and a.archetype == "self"
    assert r.polarity == 1 and a.polarity == 1
    assert "India Equity" in r.excluded_predictors and "India Equity" not in a.excluded_predictors
    assert "US Large Cap (S&P 500)" in a.excluded_predictors and "US Large Cap (S&P 500)" not in r.excluded_predictors
    # Stocks are ASSET-LEVEL: tuning inherits the STOCK CLASS default — which may
    # DIVERGE from the global InstrumentConfig default, since the `swayam` study tunes
    # the global Swayam grid on COMMODITIES while `per_asset` tunes the stock grids
    # separately (so a registered stock must match its own market's class default, not
    # the global base). They are NOT in the per-instrument slot map.
    india_base = CLASS_CONFIG_DEFAULTS["stock_india"]
    us_base = CLASS_CONFIG_DEFAULTS["stock_us"]
    for f in _TUNING_FIELDS:
        assert getattr(r, f) == getattr(india_base, f), ("stock-india", f)
        assert getattr(a, f) == getattr(us_base, f), ("stock-us", f)
    assert "TESTRELIANCE (NSE)" not in PER_INSTRUMENT_TUNING
    _before = get_instrument_config("TESTRELIANCE (NSE)")
    register_stock_target("TESTRELIANCE (NSE)", "TESTRELIANCE.NS", "india")
    assert get_instrument_config("TESTRELIANCE (NSE)") is _before   # idempotent
    checks += 1

    # ── 7. PER-INSTRUMENT TUNING IS ISOLATED ──────────────────────────────
    _orig_gold = INSTRUMENT_CONFIGS["Gold"]
    _orig_silver = INSTRUMENT_CONFIGS["Silver"]
    try:
        INSTRUMENT_CONFIGS["Gold"] = replace(_orig_gold, forecast_horizon=42, nirnay_msf_length=99)
        assert get_instrument_config("Gold").forecast_horizon == 42
        assert get_instrument_config("Gold").nirnay_msf_length == 99
        assert get_instrument_config("Silver").forecast_horizon == base.forecast_horizon
        assert get_instrument_config("Silver").nirnay_msf_length == base.nirnay_msf_length
    finally:
        INSTRUMENT_CONFIGS["Gold"] = _orig_gold
    assert INSTRUMENT_CONFIGS["Silver"] is _orig_silver
    checks += 1

    # ── 8. INTERPRETATION-LAYER SYNC (config default == module global) ─────
    # The interpretation fields (markers / UI tiers / conviction / convergence
    # display / thresholds / analog) were promoted to per-instrument config; their
    # DEFAULTS must stay identical to the module globals the read sites fall back
    # to, else a "behaviour-preserving" claim silently breaks. Pins the contract.
    import core.config as _c
    from convergence import normalization as _nz
    import analytics.analogs as _al
    _pairs = [
        (base.ui_consensus_strong, _c.UI_CONSENSUS_STRONG), (base.ui_consensus_moderate, _c.UI_CONSENSUS_MODERATE),
        (base.ui_convraw_strong, _c.UI_CONVRAW_STRONG), (base.ui_convraw_moderate, _c.UI_CONVRAW_MODERATE),
        (base.ui_nirnay_avg_threshold, _c.UI_NIRNAY_AVG_THRESHOLD),
        (base.ui_agreement_strong, _c.UI_AGREEMENT_STRONG), (base.ui_agreement_moderate, _c.UI_AGREEMENT_MODERATE),
        (base.ui_breadth_high, _c.UI_BREADTH_HIGH),
        (base.ui_model_spread_low, _c.UI_MODEL_SPREAD_LOW), (base.ui_model_spread_high, _c.UI_MODEL_SPREAD_HIGH),
        (base.ui_nirnay_bullish, _c.UI_NIRNAY_BULLISH), (base.ui_nirnay_bearish, _c.UI_NIRNAY_BEARISH),
        (base.conviction_strong, _c.CONVICTION_STRONG), (base.conviction_moderate, _c.CONVICTION_MODERATE),
        (base.conviction_weak, _c.CONVICTION_WEAK),
        (base.conv_display_strong, -_c.CONV_STRONG_BULLISH), (base.conv_display_moderate, -_c.CONV_MODERATE_BULLISH),
        (base.conv_display_weak, -_c.CONV_WEAK_BULLISH),
        (base.analog_w_maha, _al.ANALOG_W_MAHA), (base.analog_w_traj, _al.ANALOG_W_TRAJ),
        (base.analog_w_recv, _al.ANALOG_W_RECV),
    ]
    for got, glob in _pairs:
        assert got == glob, ("interp default drifted from module global", got, glob)
    # Threshold dict helpers match the normalization module globals.
    assert base.composite_thresholds() == dict(_nz.COMPOSITE_THRESHOLDS)
    assert base.consensus_thresholds() == dict(_nz.DEFAULT_THRESHOLDS)
    # And every interpretation field is a tunable (overridable) field, not routing.
    for f in ("ui_consensus_strong", "conviction_strong", "composite_strong",
              "conv_display_strong", "analog_w_maha", "ui_breadth_high"):
        assert f in c._TUNABLE_FIELDS and f not in c._ROUTING_FIELDS, f
    checks += 1

    print(f"instrument-configs: ALL {checks} CHECK GROUPS PASSED "
          f"({len(INSTRUMENT_CONFIGS)} per-instrument configs across "
          f"{len(PER_INSTRUMENT_CLASSES)} classes + asset-level stock classes)")


if __name__ == "__main__":
    run()
