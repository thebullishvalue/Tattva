"""
Tattva — per-exchange trading calendars: the session spine, freshness and "today".

तत्त्व (Tattva) — "Principle / Essence"

The freshness notices count "trading days behind" Mūla's as-of session. Historically
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

Exposes, all with the same weekday fallback:
  • ``trading_days_behind`` — holiday-aware sessions between Mūla's as-of session d*
                              and today (the freshness chip and notices).
  • ``session_mask``        — vectorised session membership over a date index (the
                              target-exchange model spine, data.prep).
  • ``target_printed``      — did the TARGET itself print on each date? (the other
                              half of the spine: an open exchange is not a print).
  • ``session_bars``        — the target's OHLCV on its own weekday sessions (the bars
                              the spine and Swayam read).
  • ``session_today``       — the one "today", in the exchange's own time zone, for
                              the app's freshness and the engine's forming-session test.
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

# Sentinel for instruments that never close at all — crypto is 24×7×365.
# Distinct from _FX because the weekday mask is WRONG here, not approximate:
# BTC-USD prints on Saturdays and on Christmas Day. Before this existed the
# bare-symbol branch below resolved BTC-USD to XNYS, so every crypto instrument
# was modelled as shut whenever New York was.
_CRYPTO = "CRYPTO"

# yfinance quotes crypto as PAIR-QUOTE (BTC-USD, ETH-USDT). Matching on the
# quote-currency suffix keeps share-class tickers (BRK-B) out of this branch.
_CRYPTO_QUOTES = ("-USD", "-USDT", "-USDC", "-EUR", "-GBP", "-JPY", "-BTC", "-ETH")

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

    # Crypto — 24×7×365, and it must be caught BEFORE the bare-symbol branch,
    # which would otherwise read BTC-USD as a US equity.
    if tu.endswith(_CRYPTO_QUOTES):
        return _CRYPTO

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
    # FX and crypto are sentinels, not MICs: asking exchange_calendars for
    # "CRYPTO" failed on EVERY call (failures are deliberately not cached), so
    # each crypto freshness check paid a library lookup to learn nothing.
    if not _HAVE_EC or not mic or mic in (_FX, _CRYPTO):
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


def session_mask(ticker: str | None, dates) -> np.ndarray:
    """Boolean mask over ``dates`` — True where ``ticker``'s exchange had a session.

    Holiday-aware when ``exchange_calendars`` is installed; for FX, unknown tickers and
    out-of-bounds dates it falls back to "is a weekday" (crypto: every day), so with no
    library a weekday-only frame is returned unchanged. Never raises.
    """
    dts = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if len(dts) == 0:
        return np.zeros(0, dtype=bool)
    weekday = np.asarray(dts.dayofweek < 5)
    mic = resolve_exchange(ticker)
    if mic == _CRYPTO:
        return np.ones(len(dts), dtype=bool)   # 24x7x365
    cal = _get_calendar(mic)
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


def _busday_behind(asof: date, today: date) -> int:
    """Legacy Mon–Fri count: trading days strictly after ``asof`` through ``today``."""
    return max(0, int(np.busday_count(asof + timedelta(days=1), today + timedelta(days=1))))


def trading_days_behind(ticker: str | None, asof: date, today: date) -> int:
    """Sessions strictly after ``asof`` up to & including ``today`` for ``ticker``'s
    exchange. Holiday-aware when ``exchange_calendars`` is installed; otherwise the
    exact legacy Mon–Fri ``busday_count``. Always ``>= 0``.

    Measured from d*, Mūla's as-of session — the session the call is about — not
    from the panel's newest date: a withheld newest row is data, not a valuation,
    so it cannot make a call any fresher.
    """
    if today <= asof:
        return 0
    mic = resolve_exchange(ticker)
    # Crypto trades 24×7, but the model spine does not: fetch_commodity_dataset
    # drops weekend rows (nothing else in the cross-section printed), so the
    # newest bar a crypto target can have on a Sunday is Friday's. Counting
    # calendar days here would call that bar two days stale every weekend; the
    # weekday count is the honest one for a weekday spine.
    if mic == _CRYPTO:
        return _busday_behind(asof, today)
    cal = _get_calendar(mic)
    if cal is None:
        return _busday_behind(asof, today)
    try:
        start = pd.Timestamp(asof) + pd.Timedelta(days=1)
        end = pd.Timestamp(today)
        # Clamp to the calendar's known bounds so a far-future/near-epoch date can't
        # raise; outside bounds we simply fall back to the weekday count.
        if start < cal.first_session or end > cal.last_session:
            return _busday_behind(asof, today)
        return max(0, int(len(cal.sessions_in_range(start, end))))
    except Exception as e:
        log.debug("session count failed for %s/%s: %s; weekday fallback", ticker, mic, e)
        return _busday_behind(asof, today)


#: Time zones for the two sentinels, which have no calendar to supply one. The
#: FX day rolls at the New York close; crypto has no close, so it keeps UTC.
_SENTINEL_TZ = {_FX: "America/New_York", _CRYPTO: "UTC"}


def session_today(exchange: str) -> pd.Timestamp:
    """Today's date where ``exchange`` trades — normalised and tz-naive.

    ``exchange`` is what :func:`resolve_exchange` returns (a MIC, a sentinel, or
    "" for unknown, which reads UTC). There used to be two "todays": the app's
    min(UTC date, machine-local date) and the engine's machine-local
    ``Timestamp.today()``, which disagree for an hour or more every day on any
    host whose clock is not the exchange's. One date, the exchange's own, now
    decides both whether data is behind and whether a row is still forming.
    """
    tz = _SENTINEL_TZ.get(exchange)
    if tz is None:
        cal = _get_calendar(exchange)
        tz = str(getattr(cal, "tz", None) or "UTC") if cal is not None else "UTC"
    return pd.Timestamp.now(tz=tz).tz_localize(None).normalize()


def session_bars(bars: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    """The target's OHLCV bars on its own sessions, on the spine's weekday calendar.

    yfinance can return a placeholder bar on a non-session (USD/INR carried a
    Saturday bar with O=H=L=C), and crypto's weekend bars would be valued
    against Friday's macro — the shared panel is a weekday spine. Bars BEFORE
    the spine starts are kept: they are Swayam's warm-up (~640-680 per target).
    """
    idx = pd.DatetimeIndex(bars.index)
    keep = session_mask(ticker, idx) & np.asarray(idx.dayofweek < 5)
    return bars[keep]


def target_printed(price, bars: pd.DataFrame | None, dates) -> pd.Series:
    """Did the TARGET itself print on each of ``dates``? → bool Series on ``dates``.

    An open exchange is not a print. Gold's spine is the CME calendar, but on 35
    US federal holidays GC=F has no bar at all: the panel carries Friday's close
    forward and the row used to be valued, graded and counted as one of the h
    sessions of a forward return, so "10 sessions" was sometimes nine.

    With the target's own OHLCV (``bars``), a print is a bar with a finite
    Close on that date. Outside the span the bars cover — a feed that ends a
    session before the close feed, or starts later — and when there are no bars
    at all, it falls back to the repeat rule: the panel is forward-filled at
    source and continuous prices do not repeat exactly, so a value identical to
    the previous row's is a carried one. (The rule also drops the rare genuine
    unchanged close — 6 of Gold's ~1,400 valued sessions — which is why the bars
    decide wherever they exist.)

    ``price`` is positionally aligned with ``dates``. Causal: each date reads its
    own bar, or its own and the previous row's value.
    """
    dts = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    px = pd.to_numeric(pd.Series(np.asarray(price, dtype=object)), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(px)
    repeat = np.zeros(len(px), dtype=bool)
    repeat[1:] = finite[1:] & finite[:-1] & (px[1:] == px[:-1])
    out = finite & ~repeat
    if bars is not None and len(bars) and "Close" in bars.columns:
        close = pd.to_numeric(bars["Close"], errors="coerce")
        bidx = pd.DatetimeIndex(bars.index)
        if bidx.tz is not None:
            bidx = bidx.tz_localize(None)
        bidx = bidx.normalize()[np.isfinite(close.to_numpy(dtype=float))]
        if len(bidx):
            covered = np.asarray((dts >= bidx.min()) & (dts <= bidx.max()))
            out = np.where(covered, np.asarray(dts.isin(bidx)) & finite, out)
    return pd.Series(out.astype(bool), index=dts, name="TargetPrinted")


def panel_session_matrix(names, dates) -> np.ndarray:
    """``(len(dates), len(names))`` bool — was each instrument's venue in session?

    Resolves friendly panel column names to tickers through the config maps, then
    walks ONE calendar per distinct venue rather than per column: a 241-column
    panel spans ~14 venues, so this is ~14 calendar queries instead of 241.

    Exists because "what fraction of the panel printed" is the wrong question to
    ask of a global panel. Measured against every admitted instrument, Christmas
    Day looks identical to a data outage — NYSE alone is 124 of 241 columns, so
    the ratio collapses whenever New York is shut regardless of whether anything
    is actually missing. Measured against the instruments whose exchanges were
    OPEN, Good Friday 2024 turns out to be 42 of 45 printed: a complete session.

    Unmapped names fall back to the weekday mask, matching the rest of the module.
    """
    dts = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    names = list(names)
    out = np.ones((len(dts), len(names)), dtype=bool)
    if len(dts) == 0 or not names:
        return out
    try:
        from core.config import (GLOBAL_MACRO_MAP, MACRO_SYMBOLS_YF,
                                 INDEX_TARGETS_MAP, ALL_TARGETS)
        n2t = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF,
               **INDEX_TARGETS_MAP, **ALL_TARGETS}
    except Exception as e:                      # pragma: no cover - config drift
        log.debug("panel_session_matrix: no name->ticker map (%s); assuming open", e)
        return out

    groups: dict[str, list[int]] = {}
    reps: dict[str, str | None] = {}
    for j, nm in enumerate(names):
        tick = n2t.get(nm)
        mic = resolve_exchange(tick)
        groups.setdefault(mic, []).append(j)
        reps.setdefault(mic, tick)
    for mic, cols in groups.items():
        m = session_mask(reps[mic], dts)
        for j in cols:
            out[:, j] = m
    return out
