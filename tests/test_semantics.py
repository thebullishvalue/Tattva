"""The one sign vocabulary (core/semantics.py)."""
import math

from core.semantics import (CLAIM_SIGMA, DEEP_SIGMA, SWAYAM_DEAD, breadth_lean, claim_direction,
                            swayam_word, valuation_lean, valuation_state)


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
