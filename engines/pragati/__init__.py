"""Tattva — the Pragati indicator core (pragati.pine v9.3), shared by Swayam.

    core       conviction (Nishchaya v3), the conviction ladder (UP in Tattva), the trace, its
               histogram and push tiers, and ``signals`` — the ▲▼ read from the grid
    samanvaya  the value engine (RV leg against the macro drivers + price-only breadth leg)
    grid       the 3 × 3 Conviction-Value Grid: states, measured units, phases

Ported verbatim from Sanket (bar-for-bar with the Pine); see each module's header.
"""
from engines.pragati import core, grid, samanvaya

__all__ = ["core", "grid", "samanvaya"]
