"""
Tattva — the instrument registry: what each target IS, declared once.
तत्त्व (Tattva) — "Principle / Essence"

``core.config`` says how each target is TUNED (``InstrumentConfig``). This
module says what each one IS — its ticker, asset class, the currency its price
is quoted in, whether a call on it can be traded at all, whether it has an
OHLCV feed, and which panel columns are its same-exposure twins in another
currency. Every consumer that needs one of those facts reads it here, so each
has one definition:

  * ``instrument(target)``           — the record; raises for an unregistered
                                       target (no fallback).
  * ``predictor_universe(target, …)`` — Mūla's cross-section for that target.
  * ``ticker_class_map()``           — ticker → asset-class block, which
                                       generates ``engines.mula.blocks``' table.

The predictor universe is a REGISTERED list (``PREDICTOR_UNIVERSE``), not
"every column the shared panel happens to hold". The panel's composition used
to follow the product catalogue and Google Sheets uptime: a sheet series was a
predictor of every target while the sheet answered and vanished when it did
not, and Nifty 50's own P/E sat in the cross-section that valued Nifty 50.
Sheet series are now targets only, and a newly selectable catalogue target
joins the cross-section only by being registered here.

No Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from core.config import (
    CATALOGUE_TARGETS,
    GLOBAL_MACRO_MAP,
    INDEX_TARGETS,
    INDEX_TARGETS_MAP,
    MACRO_SYMBOLS_YF,
    RAW_YIELD_PREDICTORS,
    SHEET_TARGETS,
    TARGET_CATEGORIES,
    TARGET_EXCLUDED_PREDICTORS,
)
from data.sheets import SHEET_SOURCES

AssetClass = Literal["commodity", "fx", "crypto", "india_index", "us_index",
                     "stock_us", "stock_india", "sheet"]


@dataclass(frozen=True)
class Instrument:
    """One selectable target, as a fact rather than a tuning.

    ``tradeable`` is False for a series no position can be taken in (a
    valuation ratio); a call on it is never issued. ``has_ohlcv`` is False for
    a sheet-sourced series, which has no bar feed for Swayam to read. ``twins``
    are the panel columns that carry this target's exposure in another
    currency and are therefore dropped from its cross-section.
    """

    name: str
    ticker: str
    asset_class: AssetClass
    currency: str
    tradeable: bool
    has_ohlcv: bool
    twins: tuple[str, ...] = ()


#: Series pulled from a published Google Sheet (data/sheets.py). They are
#: joined to a run's private copy of the panel as the TARGET column only and
#: are never predictors (see the module docstring).
SHEET_SERIES: frozenset[str] = frozenset(SHEET_SOURCES)

# ─── The declared twins ──────────────────────────────────────────────────────
# A USD-listed fund on a foreign market is that market's local index times the
# exchange rate. With both in the cross-section, an FX target is rebuilt rather
# than valued: log USDINR ≈ log Nifty 500 − log INDA up to fund drift (level
# R² ≈ 0.96), and the claim then reads the tracking error. One side of each
# pair is dropped for the FX target it would rebuild, chosen STRUCTURALLY, not
# by IC: the USD-listed fund goes, the local-currency index stays as a flow
# driver. Declared pairs only — no R² screen, which flags any single trending
# series (USD/IDR alone fits USD/INR's level at R² 0.91).
#
# (USD-listed member — dropped, local-currency member — kept), per FX target.
_INDIA_PAIRS: tuple[tuple[str, str], ...] = (
    ("India Equity", "Nifty 50"),
    # IIND.L is an MSCI India EQUITY fund despite its column name (core.config
    # GLOBAL_MACRO_MAP): 10d corr 0.87 with INDA.
    ("India Gov Bonds (LSE Proxy)", "Nifty 50"),
)
TWIN_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "USD/INR": _INDIA_PAIRS,
    # The dollar index is the USD against a basket, so EVERY USD country fund
    # whose local index is in the panel rebuilds one of its crosses. Not
    # listed: the China funds (FXI, MCHI hold Hong Kong / offshore shares, not
    # the Shanghai / Shenzhen A-share indices the panel carries) and the funds
    # with no local index in the panel (EWZ, EWA, EWT, EWW and the Asia EM set).
    "Dollar Index": (
        ("Eurozone Equity (EZU)", "Euro Stoxx 50"),
        ("Japan Equity (EWJ)", "Nikkei 225"),
        ("UK Equity (EWU)", "FTSE 100 (UK)"),
        ("South Korea Equity (EWY)", "KOSPI"),
        *_INDIA_PAIRS,
    ),
}


def _twins(target: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(usd for usd, _local in TWIN_PAIRS.get(target, ())))


# ─── The registered predictor universe ───────────────────────────────────────
#: Catalogue targets outside the macro maps that ARE registered as predictors
#: (the opportunity set includes them; fetched by data.fetcher). A catalogue
#: target added later is selectable at once but enters other targets'
#: cross-sections only by being listed here.
_CATALOGUE_PREDICTORS: tuple[str, ...] = (
    "Aluminium", "Zinc",
    "Bitcoin", "Ethereum", "Solana", "XRP", "BNB", "Cardano", "Dogecoin",
)

#: Every column that may enter a Mūla cross-section, in a fixed order. The
#: macro maps are the universe by definition; the index levels and the
#: catalogue predictors above are registered members of it.
PREDICTOR_UNIVERSE: tuple[str, ...] = tuple(dict.fromkeys(
    [*GLOBAL_MACRO_MAP, *MACRO_SYMBOLS_YF, *INDEX_TARGETS_MAP, *_CATALOGUE_PREDICTORS]))
_PREDICTOR_SET: frozenset[str] = frozenset(PREDICTOR_UNIVERSE)


# ─── Building the registry from the catalogue ────────────────────────────────
_CATEGORY_CLASS: dict[str, AssetClass] = {
    "Commodities": "commodity", "Currency (FX)": "fx", "Crypto": "crypto",
    "India Indices": "india_index", "US Indices": "us_index"}
_CLASS_CURRENCY: dict[str, str] = {
    "commodity": "USD", "fx": "USD", "crypto": "USD", "india_index": "INR",
    "us_index": "USD", "stock_us": "USD", "stock_india": "INR", "sheet": "INR"}
#: Where the price's quote currency is not its class's: USD/INR is quoted in
#: rupees per dollar, and Jeera on NCDEX in rupees.
_CURRENCY_OVERRIDES: dict[str, str] = {"USD/INR": "INR", "Jeera": "INR"}
#: Series no position can be taken in. A P/E is a valuation of an index, not a
#: price of anything: it is graded as evidence and never called.
_NOT_TRADEABLE: frozenset[str] = frozenset({"Nifty 50 - PE"})


def _catalogue_class(name: str) -> AssetClass:
    if name in SHEET_TARGETS:
        return "sheet"
    for cat, names in TARGET_CATEGORIES.items():
        if name in names and cat in _CATEGORY_CLASS:
            return _CATEGORY_CLASS[cat]
    if name in INDEX_TARGETS:
        return "india_index" if INDEX_TARGETS[name]["kind"] == "india" else "us_index"
    raise KeyError(f"catalogue target {name!r} has no asset class")


def _make(name: str, ticker: str, asset_class: AssetClass) -> Instrument:
    return Instrument(
        name=name, ticker=ticker, asset_class=asset_class,
        currency=_CURRENCY_OVERRIDES.get(name, _CLASS_CURRENCY[asset_class]),
        tradeable=name not in _NOT_TRADEABLE,
        has_ohlcv=name not in SHEET_SERIES,
        twins=_twins(name),
    )


#: One record per selectable target. Built from the catalogue frozen at import;
#: free-form stocks are added by ``register_stock_instrument``.
INSTRUMENTS: dict[str, Instrument] = {
    _nm: _make(_nm, _tk, _catalogue_class(_nm)) for _nm, _tk in CATALOGUE_TARGETS.items()}


def register_stock_instrument(display_name: str, ticker: str, market: str) -> None:
    """Add a free-form stock's record (called by core.config.register_stock_target).

    Idempotent, like the registration that calls it.
    """
    cls: AssetClass = "stock_india" if market == "india" else "stock_us"
    INSTRUMENTS.setdefault(display_name, _make(display_name, ticker, cls))


def instrument(target: str) -> Instrument:
    """The registered record for ``target``; raises KeyError with no fallback."""
    try:
        return INSTRUMENTS[target]
    except KeyError:
        raise KeyError(
            f"No Instrument registered for target {target!r}. Every target must be "
            "in the catalogue or registered via register_stock_target.") from None


def predictor_universe(target: str, panel_cols: Sequence[str]) -> list[str]:
    """Mūla's cross-section for ``target``, in panel order.

    The registered universe present in ``panel_cols``, minus the raw yield
    levels (rate series, not prices), the sheet series (targets only), the
    target itself, its TARGET_EXCLUDED_PREDICTORS near-replicas and its
    declared twins.
    """
    inst = instrument(target)
    drop = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, ()), *inst.twins}
    return [c for c in panel_cols
            if c in _PREDICTOR_SET and c not in RAW_YIELD_PREDICTORS
            and c not in SHEET_SERIES and c not in drop]


# ─── Asset-class blocks of the explanatory panel ─────────────────────────────
# Only tickers whose block is NOT recoverable from the display name need an
# entry; engines.mula.blocks' keyword pass covers the rest.
_BLOCK_TICKERS: dict[str, tuple[str, ...]] = {
    # Government curve.
    "Rates": (
        "BIL", "SHV", "SGOV", "SHY", "VGSH", "IEI", "IEF", "VGIT", "TLH",
        "TLT", "VGLT", "GOVT", "BSV", "BLV", "AGG", "BND", "BNDW", "BNDX",
        "IGOV", "BWX", "IEGA.L", "IBGL.L", "SDEU.L", "IGLT.L", "VGB.AX",
        "XBB.TO", "LTGILTBEES.NS", "GILT5YBEES.NS",
        "LIQUIDBEES.NS", "CBON", "CNYB.L",
        "^IRX", "^FVX", "^TNX", "^TYX",
    ),
    "Credit": (
        "LQD", "VCSH", "VCIT", "VCLT", "HYG", "JNK", "GHYG", "BGRN", "PFF",
        "CWB", "FALN", "MBB", "VMBS", "BKLN", "MUB", "VTEB", "FLOT",
        "IEAC.L", "SLXX.L", "IBND", "EMB", "PCY", "EMLC", "EMHY",
        "EBBETF0430.NS",
    ),
    "Inflation": ("TIP", "VTIP", "SCHP", "WIP", "INXG.L", "RINF"),
    "Commodity Broad": ("DBC", "GSG"),
    # Metals, including mining / thematic-metal equity proxies.
    "Metals": (
        "DBB", "GLTR", "PALL", "LIT", "URA", "SLX", "REMX", "XME", "PICK",
        "XLB", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
    ),
    "Energy": ("XLE", "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F"),
    "Agriculture": (
        "DBA", "ZC=F", "ZW=F", "ZS=F", "ZL=F", "CT=F", "KC=F", "SB=F",
        "CC=F", "JEERA.NCDEX",
    ),
    "Volatility": ("^VIX", "VIXM", "^MOVE"),
    "Real Assets": ("VNQ", "VNQI", "REET", "XLRE", "WOOD", "IGF"),
    "Currency": (
        "DX-Y.NYB", "UUP", "UDN", "USDU", "FXE", "FXY", "FXB", "FXF",
        "FXA", "FXC", "CEW",
    ),
    # IIND.L is India equity (16-17% vol, 10d corr 0.87 with INDA, 0.18 with
    # India 8-13Y G-Sec). It sat in Rates on its column name, which put an
    # equity fund in the block the leave-one-block-out refits ablate as "rates".
    "Equity": ("IIND.L",),
}


def ticker_class_map() -> dict[str, str]:
    """Ticker → asset-class block for every explicitly classified ticker."""
    out: dict[str, str] = {}
    for block, tickers in _BLOCK_TICKERS.items():
        for t in tickers:
            assert t not in out, f"{t} is filed under two blocks ({out[t]}, {block})"
            out[t] = block
    return out
