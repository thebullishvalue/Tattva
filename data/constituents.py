"""
Tattva — Commodity basket resolution for the Nirnay engine.
तत्त्व (Tattva) — "Principle / Essence"

DATA — Two responsibilities for the Nirnay layer:
  • get_nirnay_mode(target) — routes each target to "self" (Nirnay-Swayam runs on
    the target's OWN OHLCV: commodities and individual stocks) or "basket" (breadth
    across a correlated basket: FX, indices, ETFs). Single source of truth for the
    mode switch, read by app.py Phase 1/3 and the research studies.
  • get_commodity_basket(target) — resolves the per-target basket (producers /
    constituents / sector ETFs) for basket-mode targets from the static config map.
"""

from __future__ import annotations

from core.config import COMMODITY_BASKETS, NIRNAY_BASKET_ALIAS, TARGET_ARCHETYPE
from data.universe import is_index_target, resolve_index_constituents


def get_nirnay_mode(target: str) -> str:
    """Resolve which Nirnay mode a target runs in.

    ``'self'`` (Nirnay-Swayam, engines/nirnay_self.py) for targets whose
    archetype is ``'self'`` (individual stocks, registered at runtime via
    :func:`core.config.register_stock_target` from a free-form symbol —
    see :func:`data.universe.resolve_stock_symbol`); ``'basket'`` (the
    original constituents/correlated-basket path) for everything else.
    Single source of truth for the mode branch in app.py.
    """
    return "self" if TARGET_ARCHETYPE.get(target) == "self" else "basket"


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
