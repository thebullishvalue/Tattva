"""
Tattva — the index target catalogue + free-form stock symbol resolution.
तत्त्व (Tattva) — "Principle / Essence"

Adds equity-INDEX targets (India broad & sectoral, US benchmarks) on top of the
commodity / FX / crypto targets in ``core/config.py``. An index target is its
own price level (a yfinance index ticker): Mūla values that level against the
macro cross-section, and Swayam reads breadth off the same series through its
self-referential view bank. Nothing about an index needs its constituents, so
nothing here resolves them.

This module used to be mostly constituent machinery — NSE archive CSV and
Wikipedia scrapers, a 30-name India sector-ETF basket, hardcoded snapshot
lists, a stride-cap and a day-long cache — all feeding the basket breadth
engine Swayam replaced. With that engine gone none of it had a caller, and it
still sat in the Diagnostics cache panel as a cache that was never read. It is
removed rather than kept "in case": a constituent list the system does not use
is a claim about the system that is not true.

Also here: :func:`resolve_stock_symbol`, which turns a typed symbol into a
yfinance ticker for the India / US Stocks asset classes.

Universe selection is adapted from the Sanket terminal (@thebullishvalue).
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from data.cache import Cache

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Index target catalogue
# ════════════════════════════════════════════════════════════════════════
# friendly name → metadata:
#   ticker  : yfinance ticker for the index PRICE level (the target column)
#   kind    : "india" | "us" — the market, which decides the target's leakage
#             exclusions in core/config.py (sibling indices + same-market ETFs)
#   category: sidebar grouping

INDEX_TARGETS: dict[str, dict] = {
    # ── India — broad & sectoral ─────────────────────────────────────────────
    "Nifty 50":        {"ticker": "^NSEI",      "kind": "india", "category": "India Indices"},
    "Nifty Next 50":   {"ticker": "^NSMIDCP",   "kind": "india", "category": "India Indices"},
    "Nifty 100":       {"ticker": "^CNX100",    "kind": "india", "category": "India Indices"},
    "Nifty 500":       {"ticker": "^CRSLDX",    "kind": "india", "category": "India Indices"},
    "Nifty Midcap 50": {"ticker": "^NSEMDCP50", "kind": "india", "category": "India Indices"},
    "Nifty Bank":      {"ticker": "^NSEBANK",   "kind": "india", "category": "India Indices"},
    "Nifty IT":        {"ticker": "^CNXIT",     "kind": "india", "category": "India Indices"},
    "Nifty Pharma":    {"ticker": "^CNXPHARMA", "kind": "india", "category": "India Indices"},
    # ── US — benchmark indices ───────────────────────────────────────────────
    "S&P 500":         {"ticker": "^GSPC",      "kind": "us",    "category": "US Indices"},
    "Nasdaq 100":      {"ticker": "^NDX",       "kind": "us",    "category": "US Indices"},
    "Dow Jones":       {"ticker": "^DJI",       "kind": "us",    "category": "US Indices"},
}
# Not listed, deliberately: Nifty Auto, FMCG, Metal, Energy, Smallcap 100, Fin
# Services, Pvt Bank, PSU Bank, Realty, Media, Infra, PSE, Consumption,
# Commodities, MNC and Services. yfinance serves 1-2 bars for each of them
# (re-checked 2026-10-17), against the MIN_DATA_POINTS = 1500 every run needs,
# so they could only ever fail with "Need 1500+ data points". A target the
# catalogue cannot run is not coverage. Re-add one when its history returns.
#
# "Nifty 500" was listed as "India Sector ETFs" under an "ETF Universe" class:
# the name described the sector-ETF basket Swayam used to read, but the price
# was always ^CRSLDX — the Nifty 500. With breadth read off the target's own
# price there is no basket, so it is named for what it is.

# friendly name → yfinance ticker (for the price-fetch layer).
INDEX_TARGETS_MAP: dict[str, str] = {k: v["ticker"] for k, v in INDEX_TARGETS.items()}


# ════════════════════════════════════════════════════════════════════════
# Free-form individual-stock symbol resolution
# ════════════════════════════════════════════════════════════════════════

# Resolved symbol → (ticker, exchange_label), 7-day disk cache. Only SUCCESSES
# are memoized to disk — a transient yfinance outage must not brand a symbol
# invalid for a week.
_symbol_cache = Cache(ttl=7 * 86_400, version="v1", namespace="symbol_resolution")

#: How long a "not found" answer is remembered before the symbol is probed again.
_SYMBOL_FAIL_TTL = 300.0
#: (raw, market) → (message, monotonic expiry). PROCESS-wide, which is why it is
#: short-lived and holds only definite answers: the comment here once said "this
#: session only" over a module dict, so one failure — including one caused by an
#: open circuit, which says nothing about the symbol — branded it invalid for
#: every user until the process restarted.
_symbol_fail_memo: dict[tuple[str, str], tuple[str, float]] = {}


def resolve_stock_symbol(raw: str, market: str) -> tuple[str | None, str]:
    """Resolve a user-typed symbol to a yfinance ticker with listing auto-detect.

    ``market='india'``: an explicit ``.NS``/``.BO`` suffix on ``raw`` is
    respected as-is (no probing); otherwise probes ``SYMBOL.NS`` first, then
    ``SYMBOL.BO`` (NSE takes precedence for dual-listed names). ``market='us'``:
    the bare symbol, uppercased, with ``.`` translated to ``-`` (yfinance
    convention — e.g. ``BRK.B`` -> ``BRK-B``).

    A candidate "hits" when :func:`data.fetcher.probe_symbol` finds a usable
    series (>=20 real Close rows) — the SAME definition of "usable data" the
    analysis pipeline uses, and cached under the same key, so a successful
    resolution here means the subsequent fetch is a cache hit, not a re-probe.

    Returns ``(ticker, exchange_label)`` on success — ``exchange_label`` is
    one of ``"NSE"``, ``"BSE"``, ``"US"`` — or ``(None, error_message)`` on
    failure. A symbol yfinance could not be ASKED about (rate limit, circuit
    open) gets a "could not check" message and is not remembered; only a
    definite "not found" from every candidate is, and only for a few minutes.
    """
    from data.fetcher import probe_symbol   # local: avoid a data.universe <->
    # data.fetcher import cycle (fetcher imports core.config, which imports
    # this module for INDEX_TARGETS/_MAP).

    cleaned = (raw or "").strip().upper()
    if not cleaned or " " in cleaned or len(cleaned) > 20:
        return None, f"'{raw}' is not a valid ticker symbol."

    memo_key = (cleaned, market)
    cached = _symbol_cache.get(cleaned, market)
    if cached is not None:
        return tuple(cached)
    memo = _symbol_fail_memo.get(memo_key)
    if memo is not None:
        if time.monotonic() < memo[1]:
            return None, memo[0]
        _symbol_fail_memo.pop(memo_key, None)

    if market == "india":
        if cleaned.endswith(".NS") or cleaned.endswith(".BO"):
            candidates = [(cleaned, "NSE" if cleaned.endswith(".NS") else "BSE")]
        else:
            candidates = [(f"{cleaned}.NS", "NSE"), (f"{cleaned}.BO", "BSE")]
    else:
        candidates = [(cleaned.replace(".", "-"), "US")]

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * 9)
    unreachable = False
    for ticker, exch in candidates:
        s, outcome = probe_symbol(ticker, start, end)
        if outcome == "ok":
            _symbol_cache.put(cleaned, market, value=(ticker, exch))
            return ticker, exch
        if outcome == "unavailable":
            unreachable = True

    tried = " or ".join(f"{t} ({e})" for t, e in candidates)
    if unreachable:
        return None, (f"Could not check '{raw}' — yfinance is not answering right "
                      f"now (tried {tried}). Try again in a minute.")
    if market == "india":
        msg = f"'{raw}' not found on NSE (.NS) or BSE (.BO) via yfinance (tried {tried})."
    else:
        msg = f"'{raw}' not found on yfinance (tried {tried})."
    # Size-bounded: a long-running server sees many typos.
    if len(_symbol_fail_memo) > 512:
        now = time.monotonic()
        for k in [k for k, (_, exp) in _symbol_fail_memo.items() if exp <= now]:
            _symbol_fail_memo.pop(k, None)
    _symbol_fail_memo[memo_key] = (msg, time.monotonic() + _SYMBOL_FAIL_TTL)
    return None, msg
