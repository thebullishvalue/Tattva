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

from core.config import ALL_TARGETS
from data import calendars as cal
from data.calendars import resolve_exchange, trading_days_behind, CALENDAR_BACKEND


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
    print("  mapping spot-checks pass")

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

    print("data.calendars: ALL CHECKS PASSED")


if __name__ == "__main__":
    run()
