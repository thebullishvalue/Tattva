"""
Tattva — FVO (Fair Value Oscillator) engine.
तत्त्व (Tattva) — "Principle / Essence"

Ported from AMIS's Market Valuation Engine and wired into Tattva's signal
stack. The one-sided estimation primitives it is built on (discounted DLMs,
dynamic model averaging, online AR(1), expanding empirical ranks) live in
``analytics.causal`` — engines depend on analytics, never the reverse.

``FairValueEngine`` is the public entry point and keeps the exact API of the
Aarambh engine it replaces; the modules beneath it are the recursive machinery:

    factors    — adaptive volatility panel + online factor model with a
                 Marchenko-Pastur spectral cut
    regime     — two-state HMM over global return/volatility z-scores
    blocks     — Tattva macro-column → asset-class block membership
    valuation  — the dynamic cointegrating regression that publishes fair
                 value, the gap, and the oscillator itself
"""

from .engine import FairValueEngine
from .valuation import BURN_IN, MIN_PRINTS, VALUATION_DELTAS, MarketValuationEngine

__all__ = [
    "FairValueEngine",
    "MarketValuationEngine",
    "BURN_IN",
    "MIN_PRINTS",
    "VALUATION_DELTAS",
]
