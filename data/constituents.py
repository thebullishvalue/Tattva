"""
Tattva — Commodity basket resolution for the Nirnay engine.
तत्त्व (Tattva) — "Principle / Essence"

DATA — Resolves the per-commodity Nirnay basket (ETFs + miners) for the active
target from the static config map.
"""

from __future__ import annotations

from core.config import COMMODITY_BASKETS, NIRNAY_BASKET_ALIAS
from data.universe import is_index_target, resolve_index_constituents


def get_commodity_basket(target: str) -> tuple[list[str], str]:
    """Resolve the Nirnay instrument basket for a target.

    Returns ``(symbols, source)``.
      • Commodity/FX targets → static basket from
        :data:`core.config.COMMODITY_BASKETS`.
      • Index targets (India/US/ETF) → live constituent resolution via
        :func:`data.universe.resolve_index_constituents` (cached + capped).
      • Aliased sheet targets (:data:`core.config.NIRNAY_BASKET_ALIAS`) borrow
        another index's constituents (e.g. Nifty 50 PE → the Nifty 50 stocks).
    An unknown/empty result makes Convergence fall back to Aarambh-only.
    """
    target = NIRNAY_BASKET_ALIAS.get(target, target)   # borrow another index's basket
    if is_index_target(target):
        return resolve_index_constituents(target)
    basket = list(COMMODITY_BASKETS.get(target, []))
    return basket, "config"
