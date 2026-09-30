"""
The declared universe: every target Tattva calls, the world factors Mūla reads, and the pools
the learner estimates in. Declared, not discovered, so that history never moves because a
list changed underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: pool → the asset classes estimated together (DESIGN.md §3.4)
POOLS = ("macro", "crypto", "equity")


@dataclass(frozen=True)
class Target:
    name: str
    ticker: str          # yfinance ticker, or "sheet:<key>" for a sheet-sourced close series
    category: str        # sidebar grouping
    pool: str            # learner pool


def _t(category: str, pool: str, rows: dict[str, str]) -> list[Target]:
    return [Target(n, t, category, pool) for n, t in rows.items()]


TARGETS: list[Target] = [
    *_t("Commodities", "macro", {
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F", "Aluminium": "ALI=F",
        "Zinc": "ZNC=F", "Brent Crude": "BZ=F", "Cotton": "CT=F"}),
    *_t("Commodities", "macro", {"Jeera": "sheet:jeera"}),
    *_t("Currency (FX)", "macro", {"USD/INR": "INR=X", "Dollar Index": "DX-Y.NYB"}),
    *_t("Crypto", "crypto", {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD",
        "BNB": "BNB-USD", "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD"}),
    *_t("India Indices", "equity", {
        "Nifty 50": "^NSEI", "Nifty Next 50": "^NSMIDCP", "Nifty 100": "^CNX100",
        "Nifty 500": "^CRSLDX", "Nifty Midcap 50": "^NSEMDCP50", "Nifty Bank": "^NSEBANK",
        "Nifty IT": "^CNXIT", "Nifty Pharma": "^CNXPHARMA"}),
    # Not listed: Nifty Auto, FMCG, Metal, Energy, Smallcap 100, Fin Services, Pvt Bank, PSU Bank,
    # Realty, Media, Infra, PSE, Consumption, Commodities, MNC, Services — yfinance serves only
    # the current bar for these (no history), so there is nothing to forecast or to score.
    *_t("India Indices", "equity", {"Nifty 50 - PE": "sheet:nifty50_pe"}),
    *_t("US Indices", "equity", {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI"}),
]
TARGETS_BY_NAME: dict[str, Target] = {t.name: t for t in TARGETS}

#: free-form stock asset classes → market (symbol resolution) — scored with the equity pool
STOCK_CATEGORIES = {"India Stocks": "india", "US Stocks": "us"}

CATEGORIES: dict[str, list[str]] = {}
for _t_ in TARGETS:
    CATEGORIES.setdefault(_t_.category, []).append(_t_.name)
for _c in STOCK_CATEGORIES:
    CATEGORIES[_c] = []

#: Mūla's world: named, liquid, long-history series (DESIGN.md §3.1)
FACTORS: dict[str, str] = {
    "US equity": "SPY", "US tech": "QQQ", "US small cap": "IWM", "EM equity": "EEM",
    "Europe equity": "VGK", "Japan equity": "EWJ", "China equity": "FXI", "India equity": "^NSEI",
    "UST long": "TLT", "UST 7-10y": "IEF", "UST short": "SHY",
    "High yield": "HYG", "Inv. grade": "LQD",
    "Dollar": "DX-Y.NYB", "EUR": "EURUSD=X", "JPY": "JPY=X", "CNY": "CNY=X", "INR": "INR=X",
    "Oil": "CL=F", "Gold": "GC=F", "Copper": "HG=F", "Commodities": "DBC", "Agriculture": "DBA",
    "VIX": "^VIX", "Bitcoin": "BTC-USD",
}

#: sheet-sourced close series (published Google Sheets; gviz CSV endpoint)
SHEETS: dict[str, dict[str, str]] = {
    "jeera": {"sheet_id": "1WfT2EGCyqPuKXtejp2CZsFBs9T3Lsa8JCPuHUlv-rUo", "gid": "0",
              "value_col": "Price"},
    "nifty50_pe": {"sheet_id": "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c", "gid": "990005192",
                   "value_col": "NIFTY50_PE"},
}

HISTORY_START = "2008-01-01"


def yf_tickers() -> list[str]:
    """Every yfinance ticker the universe needs, once each."""
    out = {t.ticker for t in TARGETS if not t.ticker.startswith("sheet:")} | set(FACTORS.values())
    return sorted(out)
