"""
Tattva — Index target universe + constituent resolution.
तत्त्व (Tattva) — "Principle / Essence"

Adds equity-INDEX targets (India sectoral/broad, US benchmarks, India sector-ETF
universe) on top of the commodity/FX targets in ``core/config.py``. For an index
target:

  • the Aarambh **target** is the index price level (a yfinance index ticker), and
  • the Nirnay **basket** is the index's own constituents — the natural bottom-up
    cross-section (this is exactly what Nirnay was originally built for).

Constituent lists are resolved live (NSE archive CSV for India, Wikipedia for US),
cached to disk (24 h), and fall back to a hardcoded snapshot for the headline
indices so the app keeps working if a scrape is blocked. By default the FULL
constituent set is used (no cap) — capping an index drops real members and defeats
its purpose. An optional stride-cap (``_DEFAULT_CAP``) can be re-enabled to bound
the per-index Nirnay pass on the very large indices (e.g. an uncapped S&P 500 is
~300 s of constituent analysis at ~0.6 s/name).

Universe selection is adapted from the Sanket terminal (@thebullishvalue).
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from data.cache import Cache

log = logging.getLogger(__name__)

# Constituent lists change slowly — cache a full day, serve stale on failure.
_constituent_cache = Cache(ttl=86_400, version="v1", namespace="constituents")

# Max Nirnay constituents per index (stride-sampled). 0 = NO CAP: every index uses
# its FULL constituent set — capping a "Nifty 50" down to 40 drops real members and
# defeats the index's purpose. Trade-off: Nirnay's MMR cost is ~0.6 s/constituent,
# so the per-index pass scales with size (Nifty 50 ≈ 30 s, an uncapped S&P 500 ≈
# 300 s / ~5 min). Raise this to a positive integer to re-impose an upper bound.
_DEFAULT_CAP = 0

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_NSE_BASE = "content/indices/"


# ════════════════════════════════════════════════════════════════════════
# Index target catalogue
# ════════════════════════════════════════════════════════════════════════
# friendly name → metadata:
#   ticker  : yfinance ticker for the index PRICE level (Aarambh target column)
#   kind    : "india" (NSE CSV) | "us" (Wikipedia) | "etf" (fixed basket)
#   nse     : NSE index name (kind=india) → archive CSV slug
#   wiki    : US index name (kind=us)
#   cap     : max constituents for the Nirnay basket
#   category: sidebar grouping

INDEX_TARGETS: dict[str, dict] = {
    # ── India — broad & sectoral (constituents via NSE archive CSV) ──────────
    "Nifty 50":      {"ticker": "^NSEI",     "kind": "india", "nse": "ind_nifty50list.csv",     "category": "India Indices"},
    "Nifty Next 50": {"ticker": "^NSMIDCP",  "kind": "india", "nse": "ind_niftynext50list.csv", "category": "India Indices"},
    "Nifty Bank":    {"ticker": "^NSEBANK",  "kind": "india", "nse": "ind_niftybanklist.csv",   "category": "India Indices"},
    "Nifty IT":      {"ticker": "^CNXIT",    "kind": "india", "nse": "ind_niftyitlist.csv",     "category": "India Indices"},
    "Nifty Auto":    {"ticker": "^CNXAUTO",  "kind": "india", "nse": "ind_niftyautolist.csv",   "category": "India Indices"},
    "Nifty FMCG":    {"ticker": "^CNXFMCG",  "kind": "india", "nse": "ind_niftyfmcglist.csv",   "category": "India Indices"},
    "Nifty Pharma":  {"ticker": "^CNXPHARMA","kind": "india", "nse": "ind_niftypharmalist.csv", "category": "India Indices"},
    "Nifty Metal":   {"ticker": "^CNXMETAL", "kind": "india", "nse": "ind_niftymetallist.csv",  "category": "India Indices"},
    "Nifty Energy":  {"ticker": "^CNXENERGY","kind": "india", "nse": "ind_niftyenergylist.csv", "category": "India Indices"},
    # broad-market
    "Nifty 100":          {"ticker": "^CNX100",    "kind": "india", "nse": "ind_nifty100list.csv",          "category": "India Indices"},
    "Nifty Midcap 50":    {"ticker": "^NSEMDCP50", "kind": "india", "nse": "ind_niftymidcap50list.csv",     "category": "India Indices"},
    "Nifty Smallcap 100": {"ticker": "^CNXSC",     "kind": "india", "nse": "ind_niftysmallcap100list.csv",  "category": "India Indices"},
    # additional sectoral / thematic
    "Nifty Fin Services": {"ticker": "NIFTY_FIN_SERVICE.NS", "kind": "india", "nse": "ind_niftyfinancelist.csv",     "category": "India Indices"},
    "Nifty Pvt Bank":     {"ticker": "NIFTY_PVT_BANK.NS",    "kind": "india", "nse": "ind_nifty_privatebanklist.csv", "category": "India Indices"},
    "Nifty PSU Bank":     {"ticker": "^CNXPSUBANK", "kind": "india", "nse": "ind_niftypsubanklist.csv",      "category": "India Indices"},
    "Nifty Realty":       {"ticker": "^CNXREALTY",  "kind": "india", "nse": "ind_niftyrealtylist.csv",       "category": "India Indices"},
    "Nifty Media":        {"ticker": "^CNXMEDIA",   "kind": "india", "nse": "ind_niftymedialist.csv",        "category": "India Indices"},
    "Nifty Infra":        {"ticker": "^CNXINFRA",   "kind": "india", "nse": "ind_niftyinfralist.csv",        "category": "India Indices"},
    "Nifty PSE":          {"ticker": "^CNXPSE",     "kind": "india", "nse": "ind_niftypselist.csv",          "category": "India Indices"},
    "Nifty Consumption":  {"ticker": "^CNXCONSUM",  "kind": "india", "nse": "ind_niftyconsumptionlist.csv",  "category": "India Indices"},
    "Nifty Commodities":  {"ticker": "^CNXCMDT",    "kind": "india", "nse": "ind_niftycommoditieslist.csv",  "category": "India Indices"},
    "Nifty MNC":          {"ticker": "^CNXMNC",     "kind": "india", "nse": "ind_niftymnclist.csv",          "category": "India Indices"},
    "Nifty Services":     {"ticker": "^CNXSERVICE", "kind": "india", "nse": "ind_niftyservicelist.csv",      "category": "India Indices"},
    # ── US — benchmark indices (constituents via Wikipedia) ──────────────────
    "S&P 500":       {"ticker": "^GSPC", "kind": "us", "wiki": "S&P 500",    "category": "US Indices"},
    "Nasdaq 100":    {"ticker": "^NDX",  "kind": "us", "wiki": "NASDAQ 100", "category": "US Indices"},
    "Dow Jones":     {"ticker": "^DJI",  "kind": "us", "wiki": "DOW JONES",  "category": "US Indices"},
    # ── India sector-ETF universe (target = Nifty 500; basket = sector ETFs) ─
    "India Sector ETFs": {"ticker": "^CRSLDX", "kind": "etf", "category": "ETF Universe"},
}

# friendly name → yfinance ticker (for the price-fetch layer).
INDEX_TARGETS_MAP: dict[str, str] = {k: v["ticker"] for k, v in INDEX_TARGETS.items()}


# ── Fixed India sector-ETF basket (Pragyam/Sanket universe) ──────────────────
ETF_BASKET = [
    "CHEMICAL.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "GROWWPOWER.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS",
]

# ── Hardcoded snapshots (fallback when live scrape AND cache both fail) ───────
# Only the headline / most-likely-blocked indices; everything else degrades to
# an empty basket → Aarambh-only convergence (the app already handles that).
_SNAPSHOTS: dict[str, list[str]] = {
    "Dow Jones": [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MRK", "MSFT",
        "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "MMM",
    ],
    # S&P 500 / Nasdaq 100 are uncapped (_DEFAULT_CAP = 0 — full constituent set,
    # see CHANGELOG 2.1.0) so this ~40-name snapshot is NOT a representative
    # sample of the index when live scrape + cache both fail; it's a small,
    # deliberately-incomplete fallback (~8% of the S&P 500) chosen only to keep
    # SOME cross-sectional signal alive rather than falling back to Aarambh-only
    # (audit finding B4 — a prior comment here claimed it "covers the stride-cap",
    # which stopped being true when the cap went 40 -> 0). The basket-source
    # string returned by resolve_index_constituents (e.g. "snapshot (40)")
    # documents the actual count used.
    "S&P 500": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
        "BRK-B", "JPM", "LLY", "V", "XOM", "UNH", "MA", "COST", "HD", "PG",
        "JNJ", "WMT", "NFLX", "ABBV", "BAC", "CRM", "ORCL", "CVX", "KO", "AMD",
        "PEP", "TMO", "LIN", "ADBE", "MRK", "CSCO", "ACN", "MCD", "WFC", "ABT",
        "GE",
    ],
    "Nasdaq 100": [
        "AAPL", "MSFT", "NVDA", "AMZN", "AVGO", "META", "GOOGL", "GOOG", "TSLA",
        "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "QCOM",
        "INTU", "TXN", "AMGN", "AMAT", "ISRG", "BKNG", "HON", "CMCSA", "ADP",
        "VRTX", "GILD", "MU", "LRCX", "REGN", "PANW", "ADI", "MELI", "KLAC",
        "SBUX", "MDLZ", "SNPS", "CDNS",
    ],
    "Nifty Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS", "FEDERALBNK.NS",
        "IDFCFIRSTB.NS", "CANBK.NS",
    ],
    "Nifty IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS",
    ],
}


# ════════════════════════════════════════════════════════════════════════
# Live resolvers
# ════════════════════════════════════════════════════════════════════════


def _stride_cap(symbols: list[str], cap: int) -> list[str]:
    """Evenly-spaced down-sample to ``cap`` (keeps cross-sectional spread)."""
    if cap <= 0 or len(symbols) <= cap:
        return symbols
    step = len(symbols) / float(cap)
    return [symbols[int(i * step)] for i in range(cap)]


def _fetch_nse_csv(slug: str) -> list[str]:
    """Constituents (``.NS`` tickers) from the NSE archive CSV. Static-file host,
    rarely IP-blocked — preferred over the (often-blocked) NSE JSON API."""
    import requests

    for host in ("archives.nseindia.com", "nsearchives.nseindia.com"):
        try:
            s = requests.Session()
            # TLS verification ON (default). Both hosts present valid certificate
            # chains; disabling verification let a network-path attacker splice
            # arbitrary tickers into the constituent list that's subsequently fed
            # straight to yf.download — an integrity risk with no corresponding
            # benefit (these are plain public HTTPS endpoints, no corporate-proxy
            # MITM is assumed here).
            s.get(f"https://{host}", headers=_HTTP_HEADERS, timeout=10)
            r = s.get(f"https://{host}/{_NSE_BASE}{slug}", headers=_HTTP_HEADERS, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            col = next((c for c in df.columns if c.lower() == "symbol"), None)
            if col:
                syms, seen = [], set()
                for v in df[col].tolist():
                    t = f"{str(v).strip()}.NS"
                    if v and str(v).strip() and t not in seen:
                        seen.add(t)
                        syms.append(t)
                if syms:
                    return syms
        except Exception as e:  # noqa: BLE001
            log.debug("NSE CSV %s @ %s failed: %s", slug, host, e)
    return []


_US_WIKI = {
    "S&P 500":    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ 100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "DOW JONES":  "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}


def _fetch_us_wiki(index_name: str) -> list[str]:
    """Constituent tickers for a US index from Wikipedia (plain tickers)."""
    import requests

    url = _US_WIKI.get(index_name)
    if not url:
        return []
    try:
        r = requests.get(url, headers=_HTTP_HEADERS, timeout=15)
        r.raise_for_status()
        for table in pd.read_html(io.StringIO(r.text)):
            cols = [str(c).lower() for c in table.columns]
            col = next((table.columns[i] for i, c in enumerate(cols)
                        if "symbol" in c or "ticker" in c), None)
            if col is None:
                continue
            syms = []
            for s in table[col].dropna().tolist():
                s = str(s).strip().replace(".", "-")
                if s and s.lower() not in ("symbol", "ticker", "nan") and 1 <= len(s) <= 6:
                    syms.append(s)
            if len(syms) >= 10:
                return syms
    except Exception as e:  # noqa: BLE001
        log.debug("US wiki %s failed: %s", index_name, e)
    return []


def is_index_target(target: str) -> bool:
    return target in INDEX_TARGETS


def resolve_index_constituents(target: str, cap: int = _DEFAULT_CAP) -> tuple[list[str], str]:
    """Resolve the Nirnay basket for an index target.

    Order: disk cache → live (NSE CSV / Wikipedia / fixed ETF list) → stale cache
    → hardcoded snapshot → empty (Aarambh-only). Result is stride-capped and cached.
    """
    meta = INDEX_TARGETS.get(target)
    if meta is None:
        return [], "not an index target"

    cached = _constituent_cache.get(target)
    if cached is not None:
        return cached, f"cache ({len(cached)})"

    kind = meta["kind"]
    if kind == "etf":
        syms = list(ETF_BASKET)
        src = "ETF universe"
    elif kind == "india":
        syms = _fetch_nse_csv(meta["nse"])
        src = "NSE archive"
    elif kind == "us":
        syms = _fetch_us_wiki(meta["wiki"])
        src = "Wikipedia"
    else:
        syms = []
        src = "?"

    if syms:
        syms = _stride_cap(syms, cap)
        _constituent_cache.put(target, value=syms)
        return syms, f"{src} ({len(syms)})"

    # Live failed — try stale cache, then snapshot.
    stale = _constituent_cache.get_stale(target)
    if stale:
        return stale, f"stale cache ({len(stale)})"
    snap = _SNAPSHOTS.get(target)
    if snap:
        snap = _stride_cap(snap, cap)
        return snap, f"snapshot ({len(snap)})"
    return [], "unavailable (Aarambh-only)"


# ════════════════════════════════════════════════════════════════════════
# Free-form individual-stock symbol resolution (Nirnay-Swayam targets)
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
