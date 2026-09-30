"""
Tattva — Index target catalogue + free-form stock symbol resolution.
तत्त्व (Tattva) — "Principle / Essence"

Adds equity-INDEX targets (India sectoral/broad, US benchmarks, the Nifty 500 for the
India sector-ETF universe) on top of the commodity/FX targets in ``core/config.py``.
For an index target the Mūla target is the index price level (a yfinance index
ticker) and Swayam reads that same price series through the Pragati indicator — no
constituent basket is needed (Swayam v3, research/swayam_v3_study.md).

Also resolves user-typed stock symbols (India / US Stocks asset classes) to yfinance
tickers with listing auto-detect.
"""

from __future__ import annotations

import logging

import pandas as pd

from data.cache import Cache

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Index target catalogue
# ════════════════════════════════════════════════════════════════════════
# friendly name → metadata:
#   ticker  : yfinance ticker for the index PRICE level (the Mūla / Swayam target)
#   kind    : "india" | "us" | "etf" — picks which sibling equity ETFs are kept out
#             of the target's Mūla predictor set (core/config.py)
#   category: sidebar grouping

INDEX_TARGETS: dict[str, dict] = {
    # ── India — broad & sectoral ────────────────────────────
    "Nifty 50":      {"ticker": "^NSEI",     "kind": "india", "category": "India Indices"},
    "Nifty Next 50": {"ticker": "^NSMIDCP",  "kind": "india", "category": "India Indices"},
    "Nifty Bank":    {"ticker": "^NSEBANK",  "kind": "india", "category": "India Indices"},
    "Nifty IT":      {"ticker": "^CNXIT",    "kind": "india", "category": "India Indices"},
    "Nifty Auto":    {"ticker": "^CNXAUTO",  "kind": "india", "category": "India Indices"},
    "Nifty FMCG":    {"ticker": "^CNXFMCG",  "kind": "india", "category": "India Indices"},
    "Nifty Pharma":  {"ticker": "^CNXPHARMA","kind": "india", "category": "India Indices"},
    "Nifty Metal":   {"ticker": "^CNXMETAL", "kind": "india", "category": "India Indices"},
    "Nifty Energy":  {"ticker": "^CNXENERGY","kind": "india", "category": "India Indices"},
    # broad-market
    "Nifty 100":          {"ticker": "^CNX100",    "kind": "india", "category": "India Indices"},
    "Nifty Midcap 50":    {"ticker": "^NSEMDCP50", "kind": "india", "category": "India Indices"},
    "Nifty Smallcap 100": {"ticker": "^CNXSC",     "kind": "india", "category": "India Indices"},
    # additional sectoral / thematic
    "Nifty Fin Services": {"ticker": "NIFTY_FIN_SERVICE.NS", "kind": "india", "category": "India Indices"},
    "Nifty Pvt Bank":     {"ticker": "NIFTY_PVT_BANK.NS",    "kind": "india", "category": "India Indices"},
    "Nifty PSU Bank":     {"ticker": "^CNXPSUBANK", "kind": "india", "category": "India Indices"},
    "Nifty Realty":       {"ticker": "^CNXREALTY",  "kind": "india", "category": "India Indices"},
    "Nifty Media":        {"ticker": "^CNXMEDIA",   "kind": "india", "category": "India Indices"},
    "Nifty Infra":        {"ticker": "^CNXINFRA",   "kind": "india", "category": "India Indices"},
    "Nifty PSE":          {"ticker": "^CNXPSE",     "kind": "india", "category": "India Indices"},
    "Nifty Consumption":  {"ticker": "^CNXCONSUM",  "kind": "india", "category": "India Indices"},
    "Nifty Commodities":  {"ticker": "^CNXCMDT",    "kind": "india", "category": "India Indices"},
    "Nifty MNC":          {"ticker": "^CNXMNC",     "kind": "india", "category": "India Indices"},
    "Nifty Services":     {"ticker": "^CNXSERVICE", "kind": "india", "category": "India Indices"},
    # ── US — benchmark indices ──────────────────────────────
    "S&P 500":       {"ticker": "^GSPC", "kind": "us", "category": "US Indices"},
    "Nasdaq 100":    {"ticker": "^NDX",  "kind": "us", "category": "US Indices"},
    "Dow Jones":     {"ticker": "^DJI",  "kind": "us", "category": "US Indices"},
    # ── India sector-ETF universe (target = Nifty 500) ────────────
    "India Sector ETFs": {"ticker": "^CRSLDX", "kind": "etf", "category": "ETF Universe"},
}

# friendly name → yfinance ticker (for the price-fetch layer).
INDEX_TARGETS_MAP: dict[str, str] = {k: v["ticker"] for k, v in INDEX_TARGETS.items()}


# ════════════════════════════════════════════════════════════════════════
# Free-form individual-stock symbol resolution (stock targets)
# ════════════════════════════════════════════════════════════════════════

# Resolved symbol → (ticker, exchange_label), 7-day disk cache. Only SUCCESSES
# are memoized to disk — a transient yfinance outage must not brand a symbol
# invalid for a week (see the in-session failure memo below instead).
_symbol_cache = Cache(ttl=7 * 86_400, version="v1", namespace="symbol_resolution")
_symbol_fail_memo: dict[tuple[str, str], str] = {}   # (raw, market) → error message, THIS SESSION ONLY


def resolve_stock_symbol(raw: str, market: str) -> tuple[str | None, str]:
    """Resolve a user-typed symbol to a yfinance ticker with listing auto-detect.

    ``market='india'``: an explicit ``.NS``/``.BO`` suffix on ``raw`` is
    respected as-is (no probing); otherwise probes ``SYMBOL.NS`` first, then
    ``SYMBOL.BO`` (NSE takes precedence for dual-listed names). ``market='us'``:
    the bare symbol, uppercased, with ``.`` translated to ``-`` (yfinance
    convention — e.g. ``BRK.B`` -> ``BRK-B``).

    A candidate "hits" when :func:`data.fetcher.fetch_stock_target_series`
    returns a usable series (>=20 real Close rows) — the SAME definition of
    "usable data" the analysis pipeline uses, so a successful resolution here
    means the subsequent fetch is a cache hit, not a re-probe.

    Returns ``(ticker, exchange_label)`` on success — ``exchange_label`` is
    one of ``"NSE"``, ``"BSE"``, ``"US"`` — or ``(None, error_message)`` on
    failure, where the message names every candidate tried.
    """
    from data.fetcher import fetch_stock_target_series   # local: avoid a
    # data.universe <-> data.fetcher import cycle (fetcher imports
    # core.config, which imports this module for INDEX_TARGETS/_MAP).

    cleaned = (raw or "").strip().upper()
    if not cleaned or " " in cleaned or len(cleaned) > 20:
        return None, f"'{raw}' is not a valid ticker symbol."

    memo_key = (cleaned, market)
    cached = _symbol_cache.get(cleaned, market)
    if cached is not None:
        return tuple(cached)
    if memo_key in _symbol_fail_memo:
        return None, _symbol_fail_memo[memo_key]

    if market == "india":
        if cleaned.endswith(".NS") or cleaned.endswith(".BO"):
            candidates = [(cleaned, "NSE" if cleaned.endswith(".NS") else "BSE")]
        else:
            candidates = [(f"{cleaned}.NS", "NSE"), (f"{cleaned}.BO", "BSE")]
    else:
        candidates = [(cleaned.replace(".", "-"), "US")]

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * 9)
    for ticker, exch in candidates:
        try:
            s = fetch_stock_target_series(ticker, start, end)
        except Exception as e:  # noqa: BLE001
            log.warning("Stock symbol probe %s failed: %s", ticker, e)
            s = None
        if s is not None and not s.empty:
            _symbol_cache.put(cleaned, market, value=(ticker, exch))
            return ticker, exch

    tried = " or ".join(f"{t} ({e})" for t, e in candidates)
    if market == "india":
        msg = f"'{raw}' not found on NSE (.NS) or BSE (.BO) via yfinance (tried {tried})."
    else:
        msg = f"'{raw}' not found on yfinance (tried {tried})."
    _symbol_fail_memo[memo_key] = msg
    return None, msg
