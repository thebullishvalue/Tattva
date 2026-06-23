"""
Tattva — per-exchange trading calendars (Phase 1: freshness counting).

तत्त्व (Tattva) — "Principle / Essence"

The freshness notices count "trading days behind" the latest data bar. Historically
that used ``np.busday_count`` — a plain Mon–Fri mask with NO holiday awareness, so it
over-counts by ~1 across every market holiday (Diwali, Thanksgiving, …) and flashes a
premature stale notice. This module resolves a ticker to its exchange and counts
*actual sessions* using the ``exchange_calendars`` library when present.

Design constraints (system context):
  • Keyless / offline — ``exchange_calendars`` ships its holiday tables in-package; no
    API keys, no network. Consistent with Tattva's "no secrets" data stance.
  • OPTIONAL — the import is guarded. If the library is absent (or a calendar can't be
    built), every function degrades to the exact prior ``busday_count`` behaviour, so
    the app never breaks and is never *worse* than before — only better when the lib
    is installed (it is in requirements.txt).
  • Conservative — unknown tickers fall back to the Mon–Fri weekmask, never to an
    arbitrary calendar that could mis-state freshness.

Exposes three primitives, all with the same weekday fallback:
  • ``trading_days_behind`` — holiday-aware "trading days behind" (freshness notices).
  • ``is_session``          — was a given ticker's exchange open on a given day? (the
                              exchange-aware partial-session gate, app.py Phase 2).
  • ``session_mask``        — vectorised session membership over a date index (the
                              target-exchange model spine, app.py Phase 3).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Optional backend. Absent → weekday fallback (== legacy behaviour).
try:
    import exchange_calendars as _ec  # type: ignore
    _HAVE_EC = True
except Exception:  # pragma: no cover - depends on deploy env
    _ec = None
    _HAVE_EC = False

CALENDAR_BACKEND = "exchange_calendars" if _HAVE_EC else "weekday"

# Sentinel for instruments that trade every weekday with no exchange holidays
# (spot FX is ~24×5). These use the Mon–Fri weekmask, which is correct for them.
_FX = "FX"

# Yahoo ticker SUFFIX → exchange MIC (home market of a stock/ETF listed there).
# A MIC the library doesn't ship degrades to the weekday mask automatically
# (_get_calendar → None), so it is safe to map liberally. India (NSE/BSE) and China
# (SSE/SZSE) each share ONE national holiday calendar, so .BO→XBOM and .SZ→XSHG are
# accurate proxies (the lib lacks XNSE / XSHE).
_SUFFIX_MIC = {
    ".NS": "XBOM", ".BO": "XBOM",                 # India
    ".L":  "XLON",                                # London
    ".TO": "XTSE",                                # Toronto
    ".AX": "XASX",                                # Australia
    ".SS": "XSHG", ".SZ": "XSHG",                 # China (Shanghai / Shenzhen)
    ".HK": "XHKG",                                # Hong Kong
    ".T":  "XTKS",                                # Tokyo
    ".KS": "XKRX", ".KQ": "XKRX",                 # Korea (KOSPI / KOSDAQ)
    ".TW": "XTAI",                                # Taiwan
    ".SI": "XSES",                                # Singapore
    ".DE": "XETR", ".F": "XFRA",                  # Germany (Xetra / Frankfurt)
    ".PA": "XPAR", ".AS": "XAMS", ".BR": "XBRU",  # Euronext
    ".MC": "XMAD",                                # Madrid
    ".MI": "XMIL",                                # Milan
    ".SW": "XSWX",                                # SIX Swiss
}

# Index symbols (^...) carry no suffix, so they need an explicit home-MIC map.
# US indices → XNYS; Indian indices are recognised by prefix below; NEW foreign
# indices must be listed here, else they fall through to the SAFE weekday mask
# (never silently assigned a wrong calendar, which was the pre-global-universe bug).
_US_INDEX_SYMBOLS = {"^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^OEX"}
_FOREIGN_INDEX_MIC = {
    "^GDAXI": "XETR", "^FCHI": "XPAR", "^STOXX50E": "XPAR", "^FTSE": "XLON",
    "^IBEX": "XMAD", "^AEX": "XAMS", "^SSMI": "XSWX",
    "^N225": "XTKS", "^TPX": "XTKS", "^KS11": "XKRX", "^KQ11": "XKRX",
    "^HSI": "XHKG", "^TWII": "XTAI", "^STI": "XSES", "^AXJO": "XASX",
}
# Indian index prefixes (Nifty/BSE families) → XBOM proxy. Covers ^NSEI, ^NSEBANK,
# ^NSEMDCP50, ^CNX*, ^NSMIDCP, ^CRSLDX, etc.
_INDIA_INDEX_PREFIXES = ("^NSE", "^CNX", "^NSM", "^CRSLDX", "^BSE")


def resolve_exchange(ticker: str | None) -> str:
    """Map a yfinance ticker (or Tattva sentinel) to an exchange code.

    Returns an ``exchange_calendars`` MIC (e.g. "XBOM", "XNYS", "CMES", "XETR",
    "XTKS"), the ``_FX`` weekday sentinel, or "" when unknown (→ weekday fallback).
    Unknown ``^`` indices and unmapped suffixes return "" rather than a guessed
    calendar, so a freshly-added foreign ticker is never assigned the wrong holidays.
    Pure string logic; never raises.
    """
    if not ticker:
        return ""
    t = ticker.strip()
    tu = t.upper()

    # FX / futures first (these suffixes override any home-market notion).
    if tu.endswith("=X"):
        return _FX              # spot FX — 24×5, weekday mask
    if tu.endswith("=F"):
        return "CMES"           # CME Globex futures (proxy for ICE Brent too)

    # Tattva sentinels (non-yfinance sources) — Indian-session instruments.
    if tu.endswith(".SHEET") or tu.endswith(".NCDEX"):
        return "XBOM"

    # Suffix-encoded home exchanges (.NS, .L, .DE, .SS, .HK, …).
    for suf, mic in _SUFFIX_MIC.items():
        if tu.endswith(suf):
            return mic

    # Index symbols (^...): US explicit, foreign explicit, Indian by prefix, else
    # the safe weekday fallback (never a guessed calendar).
    if t.startswith("^"):
        if tu in _US_INDEX_SYMBOLS:
            return "XNYS"
        if tu in _FOREIGN_INDEX_MIC:
            return _FOREIGN_INDEX_MIC[tu]
        if tu.startswith(_INDIA_INDEX_PREFIXES):
            return "XBOM"
        return ""

    # Bare symbol → US equity (NYSE/NASDAQ share a holiday calendar).
    if t and t.replace(".", "").replace("-", "").isalnum():
        return "XNYS"

    return ""


_CAL_CACHE: dict[str, object | None] = {}


def _get_calendar(mic: str):
    """Lazily build & cache an exchange_calendars calendar; None on any failure.

    Failures are NOT cached so a transient cold-start error (memory pressure,
    lib hiccup) retries on the next call rather than permanently disabling the
    exchange calendar for the process lifetime.
    """
    if not _HAVE_EC or not mic or mic == _FX:
        return None
    if mic in _CAL_CACHE:
        return _CAL_CACHE[mic]
    try:
        cal = _ec.get_calendar(mic)  # type: ignore[union-attr]
        _CAL_CACHE[mic] = cal        # only cache confirmed successful builds
        return cal
    except Exception as e:           # unknown MIC / lib hiccup → weekday fallback, retryable
        log.debug("calendar %s unavailable (%s); using weekday fallback", mic, e)
        return None


def is_session(ticker: str | None, day) -> bool:
    """True if ``ticker``'s exchange held a trading session on ``day``.

    Holiday-aware when ``exchange_calendars`` is installed; otherwise (and for FX /
    unknown tickers / out-of-bounds dates) falls back to "is a weekday" — which makes
    every caller degrade to the legacy Mon–Fri behaviour with the library absent.
    Never raises.
    """
    ts = pd.Timestamp(day).normalize()
    cal = _get_calendar(resolve_exchange(ticker))
    if cal is None:
        return bool(ts.dayofweek < 5)
    try:
        if ts < cal.first_session or ts > cal.last_session:
            return bool(ts.dayofweek < 5)
        return bool(cal.is_session(ts))
    except Exception:
        return bool(ts.dayofweek < 5)


def session_mask(ticker: str | None, dates) -> np.ndarray:
    """Boolean mask over ``dates`` — True where ``ticker``'s exchange had a session.

    Vectorised companion to :func:`is_session` for filtering a whole index in one call
    (Phase 3 spine restriction). Same weekday fallback semantics: with no library the
    mask is simply "is a weekday", so a weekday-only frame is returned unchanged.
    """
    dts = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if len(dts) == 0:
        return np.zeros(0, dtype=bool)
    weekday = np.asarray(dts.dayofweek < 5)
    cal = _get_calendar(resolve_exchange(ticker))
    if cal is None:
        return weekday
    try:
        in_bounds = (dts >= cal.first_session) & (dts <= cal.last_session)
        if not in_bounds.any():
            return weekday
        lo_ib = dts[in_bounds].min()
        hi_ib = dts[in_bounds].max()
        sess = cal.sessions_in_range(lo_ib, hi_ib)
        result = weekday.copy()
        result[in_bounds] = np.asarray(dts[in_bounds].isin(sess))
        return result
    except Exception:
        return weekday


def _busday_behind(latest: date, today: date) -> int:
    """Legacy Mon–Fri count: trading days strictly after ``latest`` through ``today``."""
    return max(0, int(np.busday_count(latest + timedelta(days=1), today + timedelta(days=1))))


def trading_days_behind(ticker: str | None, latest: date, today: date) -> int:
    """Sessions strictly after ``latest`` up to & including ``today`` for ``ticker``'s
    exchange. Holiday-aware when ``exchange_calendars`` is installed; otherwise the
    exact legacy Mon–Fri ``busday_count``. Always ``>= 0``.
    """
    if today <= latest:
        return 0
    mic = resolve_exchange(ticker)
    cal = _get_calendar(mic)
    if cal is None:
        return _busday_behind(latest, today)
    try:
        start = pd.Timestamp(latest) + pd.Timedelta(days=1)
        end = pd.Timestamp(today)
        # Clamp to the calendar's known bounds so a far-future/near-epoch date can't
        # raise; outside bounds we simply fall back to the weekday count.
        if start < cal.first_session or end > cal.last_session:
            return _busday_behind(latest, today)
        return max(0, int(len(cal.sessions_in_range(start, end))))
    except Exception as e:
        log.debug("session count failed for %s/%s: %s; weekday fallback", ticker, mic, e)
        return _busday_behind(latest, today)
