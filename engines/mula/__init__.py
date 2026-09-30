"""Tattva — MŪLA: Engine 1, top-down recursive valuation.

मूल — "root". Prices the target against the level the global macro
cross-section implies: a recursive cointegrating regression of log price on
the integrated factors and asset-class blocks of ~200 instruments, plus the
error-correction read (κ̂, expected gap-closure drift) and expert-pooled
informativeness that turn the mispricing gap into a decision-grade signal.

Public surface:
    MulaEngine        — the engine app.py runs (pipeline core + ECM layer)
    FairValueEngine   — the pipeline core without the ECM layer (legacy alias)
"""
from engines.mula.engine import MulaEngine
from engines.mula.base import FairValueEngine

__all__ = ["MulaEngine", "FairValueEngine"]

