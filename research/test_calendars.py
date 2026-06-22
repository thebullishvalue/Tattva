"""
Execution test for data.calendars (Phase 1 — per-exchange freshness counting).

Verifies: (1) every live target ticker resolves to a known exchange, (2) the
holiday-aware count differs from the naive Mon–Fri count exactly across a real
market holiday and agrees elsewhere, (3) the weekday fallback reproduces the legacy
busday_count, and (4) degenerate inputs are safe. Run: python research/test_calendars.py
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

import pandas as pd

from core.config import ALL_TARGETS
from data import calendars as cal
from data.calendars import (
    resolve_exchange, trading_days_behind, is_session, session_mask, CALENDAR_BACKEND,
)


def _busday(latest: date, today: date) -> int:
    return max(0, int(np.busday_count(latest + timedelta(days=1), today + timedelta(days=1))))


def run() -> None:
    print(f"calendar backend: {CALENDAR_BACKEND}")

    # 1. every target resolves to a non-empty, expected exchange code
    known = {"XBOM", "XNYS", "XLON", "XTSE", "CMES", "FX"}
    unresolved = []
    for name, ticker in sorted(ALL_TARGETS.items()):
        mic = resolve_exchange(ticker)
        if mic not in known:
            unresolved.append((name, ticker, mic))
    assert not unresolved, f"unresolved tickers: {unresolved}"
    print(f"  all {len(ALL_TARGETS)} targets resolve to known exchanges")

    # spot-check the mapping intent
    assert resolve_exchange("GC=F") == "CMES"
    assert resolve_exchange("INR=X") == "FX"
    assert resolve_exchange("RELIANCE.NS") == "XBOM"
    assert resolve_exchange("^NSEI") == "XBOM"
    assert resolve_exchange("^GSPC") == "XNYS"
    assert resolve_exchange("AAPL") == "XNYS"
    assert resolve_exchange("NIFTY50_PE.SHEET") == "XBOM"
    assert resolve_exchange("JEERA.NCDEX") == "XBOM"
    assert resolve_exchange("") == "" and resolve_exchange(None) == ""
    # global universe: foreign indices map to home calendars, not US/India
    assert resolve_exchange("^GDAXI") == "XETR"      # was wrongly XBOM before the fix
    assert resolve_exchange("^N225") == "XTKS"
    assert resolve_exchange("000001.SS") == "XSHG"   # was wrongly XNYS
    assert resolve_exchange("399001.SZ") == "XSHG"   # Shenzhen shares China's calendar
    assert resolve_exchange("VGB.AX") == "XASX"
    assert resolve_exchange("BMW.DE") == "XETR" and resolve_exchange("7203.T") == "XTKS"
    assert resolve_exchange("^UNKNOWNIDX") == ""     # unknown ^ → safe weekday, not a guess
    print("  mapping spot-checks pass (incl. global universe)")

    # 2. degenerate inputs are safe and >= 0
    assert trading_days_behind("AAPL", date(2026, 6, 22), date(2026, 6, 22)) == 0  # same day
    assert trading_days_behind("AAPL", date(2026, 6, 23), date(2026, 6, 22)) == 0  # future-dated
    assert trading_days_behind(None, date(2026, 6, 19), date(2026, 6, 22)) >= 0    # unknown→fallback

    # 3. weekday-equivalence: for FX (24×5) the calendar count == busday count always
    fx_pairs = [(date(2026, 6, 19), date(2026, 6, 24)), (date(2026, 1, 1), date(2026, 1, 9))]
    for lo, hi in fx_pairs:
        assert trading_days_behind("INR=X", lo, hi) == _busday(lo, hi), "FX should equal weekday mask"
    print("  FX == weekday mask; degenerate inputs safe")

    # 4. holiday awareness (only meaningful with the lib): across US Thanksgiving 2025
    #    (Thu 27 Nov closed) the holiday-aware count is strictly LESS than the naive
    #    Mon–Fri count; across a plain week with no holiday they agree.
    if CALENDAR_BACKEND == "exchange_calendars":
        lo, hi = date(2025, 11, 26), date(2025, 11, 28)  # Wed -> Fri, spans Thanksgiving
        cal_n = trading_days_behind("AAPL", lo, hi)
        naive_n = _busday(lo, hi)
        assert cal_n < naive_n, f"holiday not detected: cal={cal_n} naive={naive_n}"
        print(f"  US Thanksgiving: holiday-aware={cal_n} < naive={naive_n}  (over-count avoided)")

        # plain no-holiday week → agreement (Jun 9-13 2025; Juneteenth is the 19th)
        lo, hi = date(2025, 6, 9), date(2025, 6, 13)  # Mon -> Fri, no US holiday
        assert trading_days_behind("AAPL", lo, hi) == _busday(lo, hi)
        print("  plain week: holiday-aware == naive")

        # Indian Diwali / a known NSE/BSE holiday — Independence Day 15 Aug 2025 (Fri, closed)
        lo, hi = date(2025, 8, 14), date(2025, 8, 18)  # Thu -> Mon spans 15 Aug holiday + weekend
        cal_i = trading_days_behind("^NSEI", lo, hi)
        naive_i = _busday(lo, hi)
        assert cal_i < naive_i, f"India holiday not detected: cal={cal_i} naive={naive_i}"
        print(f"  India 15-Aug: holiday-aware={cal_i} < naive={naive_i}  (over-count avoided)")
    else:
        print("  (exchange_calendars not installed — holiday assertions skipped; fallback active)")

    # 5. is_session / session_mask (Phase 2 + 3 primitives)
    days = pd.bdate_range("2024-01-01", "2024-12-31")  # 262 weekdays
    fx = session_mask("INR=X", days)
    assert fx.all(), "FX should trade every weekday"
    assert session_mask(None, days).all(), "unknown ticker → weekday mask (all weekdays)"
    assert session_mask("RANDOM", days).shape == (len(days),)
    if CALENDAR_BACKEND == "exchange_calendars":
        us = session_mask("AAPL", days)
        ind = session_mask("^NSEI", days)
        assert 0 < int((~us).sum()) < 20, f"US 2024 holidays out of range: {int((~us).sum())}"
        assert int((~ind).sum()) > int((~us).sum()), "India should have more holidays than US"
        assert not is_session("AAPL", pd.Timestamp("2024-07-04"))   # July 4
        assert is_session("AAPL", pd.Timestamp("2024-07-05"))       # next session
        assert is_session("INR=X", pd.Timestamp("2024-07-04"))      # FX open
        print(f"  session_mask: US drops {int((~us).sum())} / India drops {int((~ind).sum())} (2024); FX none")
        # foreign home-calendar holidays are now honoured (were silently wrong before)
        assert is_session("^GDAXI", pd.Timestamp("2024-10-03"))      # Xetra TRADES Unity Day
        assert not is_session("^GDAXI", pd.Timestamp("2024-12-25"))  # but closes Christmas
        assert not is_session("^N225", pd.Timestamp("2024-01-01"))   # Tokyo New Year
        assert not is_session("000001.SS", pd.Timestamp("2024-10-01"))  # China National Day
        print("  foreign holidays honoured (DE/JP/CN)")
    # empty input safe
    assert session_mask("AAPL", []).shape == (0,)
    print("  is_session/session_mask: degenerate inputs safe")

    print("data.calendars: ALL CHECKS PASSED")


if __name__ == "__main__":
    run()
