"""
Tattva v2.0.0 — Commodity basket resolution for the Nirnay engine.
तत्त्व (Tattva) — "Principle / Essence"

DATA — Resolves the per-commodity Nirnay basket (ETFs + miners) for the active
target from the static config map.
"""

from __future__ import annotations

from core.config import COMMODITY_BASKETS


def get_commodity_basket(target: str) -> tuple[list[str], str]:
    """Resolve the Nirnay instrument basket for a commodity target.

    Returns ``(symbols, source)``. Baskets are static (miners + streamers)
    and defined in :data:`core.config.COMMODITY_BASKETS`; an unknown target
    returns an empty list, which makes Convergence fall back to Aarambh-only.
    """
    basket = list(COMMODITY_BASKETS.get(target, []))
    return basket, "config"
