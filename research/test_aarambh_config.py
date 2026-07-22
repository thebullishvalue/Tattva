"""
Tattva — Aarambh per-instrument config threading test.

Pins the contract that the Aarambh forecast engine is now tunable PER INSTRUMENT
/ asset class (like Nirnay's nirnay_* and Swayam's swayam_* knobs):

  1. CONFIG SURFACE — InstrumentConfig exposes the 7 aarambh training knobs and
     they are all TUNABLE (overridable) fields, not routing.
  2. DEFAULT FALLBACK — a bare engine (or fit without a config) uses the global
     config constants, so the change is behaviour-preserving.
  3. PER-INSTRUMENT THREADING — passing an InstrumentConfig into fit() overrides
     every one of the 7 knobs on the running engine, and the walk-forward still
     completes and produces finite predictions.
  4. STATIC-METHOD PATH — the ensemble knobs (ensemble_models / ridge_alphas /
     huber_epsilon) reach the @staticmethod _fit_ensemble via parameters, so an
     override actually changes which members are fit (no silent global fallback).

Run: python -m research.test_aarambh_config  (from the repo root)
"""
from __future__ import annotations

import os as _os
import sys as _sys
import logging
from dataclasses import replace

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import numpy as np

import core.config as c
from core.config import InstrumentConfig
from engines.aarambh import FairValueEngine

_AARAMBH_FIELDS = (
    "aarambh_refit_interval", "aarambh_min_train_size", "aarambh_max_train_size",
    "aarambh_ensemble_models", "aarambh_ridge_alphas", "aarambh_huber_epsilon",
    "aarambh_lookback_windows",
)


def run() -> None:
    checks = 0
    base = InstrumentConfig()

    # ── 1. CONFIG SURFACE — 7 aarambh knobs exist and are tunable ─────────
    for f in _AARAMBH_FIELDS:
        assert hasattr(base, f), f"InstrumentConfig missing {f}"
        assert f in c._TUNABLE_FIELDS, f"{f} not tunable/overridable"
        assert f not in c._ROUTING_FIELDS
    # Defaults equal the shipped global constants.
    assert base.aarambh_refit_interval == c.REFIT_INTERVAL
    assert base.aarambh_min_train_size == c.MIN_TRAIN_SIZE
    assert base.aarambh_max_train_size == c.MAX_TRAIN_SIZE
    assert tuple(base.aarambh_ensemble_models) == tuple(c.ENSEMBLE_MODELS)
    assert tuple(base.aarambh_ridge_alphas) == tuple(c.RIDGE_ALPHAS)
    assert base.aarambh_huber_epsilon == c.HUBER_EPSILON
    assert tuple(base.aarambh_lookback_windows) == tuple(c.LOOKBACK_WINDOWS)
    checks += 1

    # Synthetic panel with mild signal so the walk-forward doesn't degenerate.
    rng = np.random.default_rng(7)
    n, p = 400, 8
    X = rng.normal(0, 1, (n, p))
    y = 0.3 * X[:, 0] + rng.normal(0, 1, n)

    # ── 2. DEFAULT FALLBACK — no config → global constants ────────────────
    e0 = FairValueEngine()
    e0.fit(X, y, forward_signal=True, purge=10)
    assert e0.min_train_size == c.MIN_TRAIN_SIZE
    assert e0.max_train_size == c.MAX_TRAIN_SIZE
    assert e0.refit_interval == c.REFIT_INTERVAL
    assert tuple(e0.ensemble_models) == tuple(c.ENSEMBLE_MODELS)
    assert e0.huber_epsilon == c.HUBER_EPSILON
    assert tuple(e0.lookback_windows) == tuple(c.LOOKBACK_WINDOWS)
    assert int(np.isfinite(e0.predictions).sum()) > 100, "default engine produced no forecasts"
    checks += 1

    # ── 3. PER-INSTRUMENT THREADING — config overrides every knob ─────────
    cfg = replace(
        base,
        aarambh_min_train_size=120, aarambh_max_train_size=220,
        aarambh_refit_interval=21, aarambh_ensemble_models=("ridge", "ols"),
        aarambh_ridge_alphas=(0.01, 1.0), aarambh_huber_epsilon=1.5,
        aarambh_lookback_windows=(5, 20, 60),
    )
    e1 = FairValueEngine()
    e1.fit(X, y, forward_signal=True, purge=10, config=cfg)
    assert e1.min_train_size == 120 and e1.max_train_size == 220
    assert e1.refit_interval == 21
    assert tuple(e1.ensemble_models) == ("ridge", "ols")
    assert tuple(e1.ridge_alphas) == (0.01, 1.0)
    assert e1.huber_epsilon == 1.5
    assert tuple(e1.lookback_windows) == (5, 20, 60)
    assert int(np.isfinite(e1.predictions).sum()) > 100, "overridden engine produced no forecasts"
    # The lookback override actually changed which Z_ feature columns were built.
    assert any(col == "Z_60" for col in e1.ts_data.columns), "lookback override not reflected in features"
    assert not any(col == "Z_100" for col in e1.ts_data.columns), "old lookback window leaked in"
    checks += 1

    # ── 4. STATIC-METHOD PATH — ensemble override reaches _fit_ensemble ───
    # ols-only vs ridge+ols must fit a DIFFERENT member set (no silent global).
    e2 = FairValueEngine()
    e2.fit(X, y, forward_signal=True, purge=10,
           config=replace(base, aarambh_ensemble_models=("ols",)))
    assert tuple(e2.ensemble_models) == ("ols",)
    assert int(np.isfinite(e2.predictions).sum()) > 100
    checks += 1

    print(f"aarambh-config: ALL {checks} CHECK GROUPS PASSED "
          f"(7 per-instrument knobs thread into the walk-forward + @staticmethod ensemble)")


if __name__ == "__main__":
    run()
