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

Phase 1 scope is freshness counting only (app.py). Making the dataset spine and the
partial-session gate exchange-aware are deliberately deferred (see CHANGELOG / #5).
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

# Explicit MICs for index / sentinel tickers that carry no resolvable suffix.
# Indian equities/indices → XBOM (BSE): the lib has no XNSE, but BSE and NSE observe
# the same holiday schedule, so it is the correct proxy. Jeera (NCDEX) and the
# Nifty-50-PE sheet are Indian-session instruments → XBOM as well.
_US_INDEX_SYMBOLS = {"^GSPC", "^DJI", "^NDX", "^IXIC", "^RUT", "^VIX", "^OEX"}


def resolve_exchange(ticker: str | None) -> str:
    """Map a yfinance ticker (or Tattva sentinel) to an exchange code.

    Returns an ``exchange_calendars`` MIC ("XBOM", "XNYS", "CMES", "XLON", "XTSE"),
    the ``_FX`` weekday sentinel, or "" when unknown (→ weekday fallback). Pure string
    logic; never raises.
    """
    if not ticker:
        return ""
    t = ticker.strip()
    tu = t.upper()

    # Suffix-encoded exchanges.
    if tu.endswith(".NS") or tu.endswith(".BO"):
        return "XBOM"           # India (NSE/BSE share a holiday calendar)
    if tu.endswith(".L"):
        return "XLON"           # London
    if tu.endswith(".TO"):
        return "XTSE"           # Toronto
    if tu.endswith("=X"):
        return _FX              # spot FX — 24×5, weekday mask
    if tu.endswith("=F"):
        return "CMES"           # CME Globex futures (proxy for ICE Brent too)

    # Tattva sentinels (non-yfinance sources) — Indian-session instruments.
    if tu.endswith(".SHEET") or tu.endswith(".NCDEX"):
        return "XBOM"

    # Index symbols (^...): a few US ones, otherwise Indian in this universe.
    if t.startswith("^"):
        return "XNYS" if tu in _US_INDEX_SYMBOLS else "XBOM"

    # Bare symbol → US equity.
    if t and t.replace(".", "").replace("-", "").isalnum():
        return "XNYS"

    return ""


_CAL_CACHE: dict[str, object | None] = {}


def _get_calendar(mic: str):
    """Lazily build & cache an exchange_calendars calendar; None on any failure."""
    if not _HAVE_EC or not mic or mic == _FX:
        return None
    if mic in _CAL_CACHE:
        return _CAL_CACHE[mic]
    cal = None
    try:
        cal = _ec.get_calendar(mic)  # type: ignore[union-attr]
    except Exception as e:  # unknown MIC / lib hiccup → weekday fallback
        log.debug("calendar %s unavailable (%s); using weekday fallback", mic, e)
    _CAL_CACHE[mic] = cal
    return cal


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
