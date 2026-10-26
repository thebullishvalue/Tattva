"""
Tattva — asset-class block membership for the Mūla valuation cross-section.
=========================================================================

The engine maintains two independent valuation views (see valuation.py):
a *latent* view on the principal factors of the cross-section, and a *block*
view on named asset-class aggregates. The block view is what makes the
published fair value auditable — every coefficient carries an economic label,
and the fitted level's exact decomposition is reported by block.

That requires a map from each explanatory column to an asset class. In AMIS
the map is a curated ticker table (``amis.universe.TICKER_CLASS``). Tattva's
explanatory panel is keyed by the *display names* of ``GLOBAL_MACRO_MAP`` /
``MACRO_SYMBOLS_YF`` (e.g. "US Large Cap (S&P 500)", "USD/INR"), so the
classification runs in two passes:

1. **Ticker table** — resolve the display name back to its yfinance ticker
   via the config maps and look it up in an explicit table. Exact, and the
   only path that can distinguish e.g. "DBB" (base metals) from "DBA"
   (agriculture) when the display names are reworded.
2. **Name keywords** — an ordered rule list over the display name, for
   columns that never went through the config maps at all (a Google-Sheet
   series such as Jeera, or a user-supplied CSV predictor).

Anything unresolved lands in "Other", which is a real block rather than a
discard: an unlabelled instrument still carries cross-sectional information,
and dropping it would make the panel's width depend on the completeness of
this file.

Block count is kept deliberately coarse (~12). The block-view regression
carries ``n_blocks + 1`` time-varying coefficients against a single price
series; splitting US equity into eleven GICS sectors would buy labels at the
cost of an estimable design.
"""

from __future__ import annotations

import re

from core.config import (COMMODITY_TARGETS, GLOBAL_MACRO_MAP, INDEX_TARGETS_MAP,
                         MACRO_SYMBOLS_YF)
from core.instruments import ticker_class_map

# Blocks, in the order they are reported. Ordering is cosmetic (the engine
# sorts the names it is given) but keeping it economic makes the drivers
# table readable.
BLOCKS = (
    "Equity",
    "Rates",
    "Credit",
    "Inflation",
    "Energy",
    "Metals",
    "Agriculture",
    "Commodity Broad",
    "Currency",
    "Volatility",
    "Real Assets",
    "Other",
)

# ---------------------------------------------------------------------------
# Pass 1 — explicit ticker table
# ---------------------------------------------------------------------------
# Generated from the instrument registry (core.instruments), which holds the
# one definition of each instrument's asset class. Only tickers whose class is
# NOT recoverable from the display name have an entry; the rest fall through
# to the keyword pass.
_TICKER_CLASS: dict[str, str] = ticker_class_map()

# Everything else in Tattva's macro universe is equity (broad, sector, style,
# regional, single-country). Listing the exceptions above and defaulting the
# remainder is both shorter and more robust than enumerating ~90 equity lines.
_EQUITY_DEFAULT_PREFIXES = ("^", "XL", "EW", "IW", "V", "MTUM", "USMV", "SPHB")


# ---------------------------------------------------------------------------
# Pass 2 — ordered keyword rules over the display name
# ---------------------------------------------------------------------------
# First match wins, so the ordering is the specification. Volatility precedes
# equity because "Equity Volatility (VIX)" contains both words; the commodity
# rules precede the sector rules for the same reason ("US Energy Sector" is
# equity, "Crude Oil" is energy — disambiguated by the explicit Sector rule
# being folded into the commodity keyword set only for the commodity itself).
_NAME_RULES: tuple[tuple[str, str], ...] = (
    (r"volatilit|\bvix\b|\bmove\b", "Volatility"),
    (r"reit|real estate|infrastructure|timber|forestry", "Real Assets"),
    (r"inflation|tips|linker|breakeven", "Inflation"),
    (r"treasury|gilt|bund|schatz|g-sec|sovereign|govt bond|government bond|"
     r"aggregate bond|total bond|yield$|overnight rate|broad bond", "Rates"),
    (r"corporate|high yield|municipal|mortgage|senior loan|preferred|"
     r"convertible|fallen angel|credit|green bond|floating rate|psu bond",
     "Credit"),
    (r"gold|silver|copper|platinum|palladium|base metals|precious|"
     r"lithium|uranium|steel|rare earth|metals & mining|miners", "Metals"),
    (r"crude|brent|natural gas|gasoline|heating oil|\bwti\b", "Energy"),
    (r"corn|wheat|soybean|cotton|coffee|sugar|cocoa|jeera|cumin|"
     r"agricultur|grain", "Agriculture"),
    (r"broad commodity|commodity index", "Commodity Broad"),
    (r"dollar|\bfx\b|currency|currencies|/[A-Z]{3}\b|\b[A-Z]{3}/", "Currency"),
    (r"equity|equities|nasdaq|s&p|russell|nikkei|kospi|kosdaq|dax|cac|"
     r"stoxx|ftse|ibex|\baex\b|\bsmi\b|composite|component|shenzhen|"
     r"shanghai|china broad|sector|semiconductor|"
     r"homebuilder|transport|value|growth|momentum|low volatility|"
     r"high beta|high dividend|banks|large cap|small cap|mid cap|acwi|"
     r"index$", "Equity"),
)

_COMPILED_RULES = tuple((re.compile(p, re.IGNORECASE), b) for p, b in _NAME_RULES)


def _name_to_ticker_map() -> dict[str, str]:
    """Display name → yfinance ticker, from Tattva's own macro config."""
    return {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **COMMODITY_TARGETS,
            **INDEX_TARGETS_MAP}


_NAME_TO_TICKER: dict[str, str] | None = None


def classify(column: str) -> str:
    """Asset-class block for one explanatory column of the Tattva panel."""
    global _NAME_TO_TICKER
    if _NAME_TO_TICKER is None:
        _NAME_TO_TICKER = _name_to_ticker_map()

    # Pass 1 — the column may itself BE a ticker (research harnesses build
    # panels straight from symbols), or resolve to one via the config maps.
    for key in (_NAME_TO_TICKER.get(column), column):
        if key and key in _TICKER_CLASS:
            return _TICKER_CLASS[key]

    # Pass 2 — keyword rules on the display name.
    for pattern, block in _COMPILED_RULES:
        if pattern.search(column):
            return block

    # Pass 3 — ticker-shape fallback for symbols with no descriptive name.
    tkr = _NAME_TO_TICKER.get(column, column)
    if tkr.endswith("=X"):
        return "Currency"
    if tkr.endswith("=F"):
        return "Commodity Broad"
    if any(tkr.startswith(p) for p in _EQUITY_DEFAULT_PREFIXES):
        return "Equity"
    return "Other"


def block_membership(columns: list[str]) -> tuple[list[str], dict[str, str]]:
    """``(sorted block names present, column → block)`` for a panel.

    Only blocks with at least one member are returned: an all-zero column in
    the block design is collinear with the intercept and would add a
    coefficient the data cannot identify.
    """
    mapping = {c: classify(c) for c in columns}
    present = sorted(set(mapping.values()))
    return present, mapping
