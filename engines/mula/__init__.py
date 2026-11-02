"""Tattva — MŪLA: Engine 1, top-down recursive valuation.

मूल — "root". Prices the target against the level the global macro
cross-section implies: a recursive cointegrating regression of log price on
the integrated factors and asset-class blocks of ~200 instruments. Its gap is
read four ways — the oscillator (the claim), the gap's persistence, the
error-correction read of whether PRICE closes it (``ecm.py``), and the
agreement of independent slices of the world on its sign.

Public surface:
    MulaEngine        — the engine (valuation + error-correction read)
    FairValueEngine   — its former name, the same class
"""
from engines.mula.base import FairValueEngine, MulaEngine

__all__ = ["MulaEngine", "FairValueEngine"]
