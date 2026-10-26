"""The one sign vocabulary (core/semantics.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from core.knobs import knob
from core.semantics import (CLAIM_SIGMA, DEEP_SIGMA, GAP_STATES, SWAYAM_DEAD, breadth_lean,
                            claim_direction, claim_oscillator, gap_stationarity, swayam_word,
                            valuation_lean, valuation_state)


def test_valuation_state_bands():
    assert valuation_state(-DEEP_SIGMA) == "VERY CHEAP"
    assert valuation_state(-CLAIM_SIGMA) == "CHEAP"
    assert valuation_state(0.0) == "FAIR"
    assert valuation_state(CLAIM_SIGMA) == "RICH"
    assert valuation_state(DEEP_SIGMA) == "VERY RICH"
    # A missing valuation is never silently "fair".
    assert valuation_state(float("nan")) == "NO VALUATION"
    assert valuation_state(None) == "NO VALUATION"


def test_readings_and_leans_have_opposite_signs():
    # Reading positive = stretched up; the call that implies is bearish.
    assert valuation_lean(1.0) < 0 < valuation_lean(-1.0)
    # An oversold majority of Swayam's views is the bullish lean.
    assert breadth_lean(60, 10) > 0 > breadth_lean(10, 60)
    assert breadth_lean(float("nan"), 10) == 0.0


def test_the_one_swayam_word():
    assert swayam_word(SWAYAM_DEAD + 0.01) == "confirms"
    assert swayam_word(-SWAYAM_DEAD - 0.01) == "contradicts"
    assert swayam_word(SWAYAM_DEAD) == "split" and swayam_word(float("nan")) == "split"


def test_claim_direction_dead_band_and_saturation():
    assert claim_direction(-0.49) == 0 and claim_direction(0.49) == 0
    assert claim_direction(-0.5) == 1 and claim_direction(0.5) == -1
    assert claim_direction(float("nan")) == 0
    assert math.isclose(valuation_lean(-DEEP_SIGMA), 1.0) and math.isclose(valuation_lean(-10), 1.0)


def test_gap_stationarity_has_three_states():
    a = knob("STATIONARITY_ALPHA")
    assert gap_stationarity(a / 10, 0.10) == "stationary"
    assert gap_stationarity(a / 10, 0.01) == ("reverts around a drifting level: "
                                              "unproven (see claim unit)")
    assert gap_stationarity(a / 10, float("nan")) == GAP_STATES[1]
    assert gap_stationarity(a, 0.10) == "unit root not rejected"
    assert gap_stationarity(float("nan"), 0.10) == "unit root not rejected"
    assert set(GAP_STATES) == {gap_stationarity(*p) for p in ((0.0, 1.0), (0.0, 0.0), (1.0, 1.0))}


def test_the_claim_oscillator_is_dormant_causal_and_lagged():
    w = knob("CLAIM_CENTRING_WINDOW")
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=w + 200)
    fvo = pd.Series(0.4 + rng.normal(0, 1.3, len(idx)), index=idx)
    fvo.iloc[[10, w + 50]] = np.nan                     # unpublished sessions
    assert claim_oscillator(fvo, "none").equals(fvo)
    with pytest.raises(ValueError):
        claim_oscillator(fvo, "robust_z_63")
    z = claim_oscillator(fvo, f"robust_z_{w}")
    pub = fvo.dropna()
    # NaN until W published sessions precede it, and on every unpublished one.
    assert z.loc[pub.index[:w]].isna().all() and z[fvo.isna()].isna().all()
    # Row t is FVO_t against the median / MAD of the W published sessions before it.
    t = pub.index[w + 7]
    win = pub.iloc[w + 7 - w:w + 7].to_numpy()
    med = np.median(win)
    assert math.isclose(z.loc[t], (pub.loc[t] - med) / (1.4826 * np.median(np.abs(win - med))))
    # Causal: truncating the history changes no earlier value.
    part = claim_oscillator(fvo.iloc[:-60], f"robust_z_{w}")
    assert np.array_equal(part.to_numpy(), z.iloc[:-60].to_numpy(), equal_nan=True)
