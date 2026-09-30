"""Tattva — SWAYAM: Engine 2, the target read by itself through the Pragati indicator.

स्वयम् — "self". The conviction ladder (Daily · Weekly) × Samanvaya's value on the target's own
OHLCV, meeting in the 3 × 3 Conviction-Value Grid, with the ▲ capitulation turn and ▼
distribution read from it. See engines/swayam/engine.py for the design and the measurement
that replaced the MSF / MMR view bank.

Public surface:
    SwayamEngine       — fit(ohlcv, drivers, symbol) → .daily (the Convergence contract),
                         .frame (the full per-bar read), .rungs (the ladder's views), .current()
    RUNG_NAMES         — the views: ("Daily", "Weekly")
    CELL_LEAN          — the grid's measured lean per cell
    CONTRACT_COLUMNS   — the schema Convergence reads
"""
from engines.swayam.engine import CELL_LEAN, CONTRACT_COLUMNS, RUNG_NAMES, SwayamEngine

__all__ = ["CELL_LEAN", "CONTRACT_COLUMNS", "RUNG_NAMES", "SwayamEngine"]
