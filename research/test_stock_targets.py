"""
Tattva — Integrity tests for individual-stock targets (SWAYAM_STOCKS_DIRECTIVE.md).

Two connected concerns pinned here:

  PART 1 — the stock-target price column reaches the Aarambh model matrix.
    A. fetch_stock_target_series (data/fetcher.py) — Close extraction +
       usability floor (>=20 rows).
    B. _ensure_stock_target_column (app.py) — DATE-spine alignment, ffill,
       leading-NaN-before-listing, and every no-op guard (column already
       present / non-'self' archetype / unregistered target).

  PART 2 — free-form symbol entry for India Stocks / US Stocks.
    C. resolve_stock_symbol (data/universe.py) — NSE-first/BSE-fallback probe
       order, explicit-suffix short-circuit, US dot->dash translation,
       invalid-input rejection, success-only disk memoization (a failure
       must NOT survive to a fresh session/process — only a live re-probe
       proves this, so it is session-memoized, never disk-memoized).
    D. register_stock_target (core/config.py) — idempotent wiring, market-
       based predictor exclusions, no TARGET_CATEGORIES mutation, and
       display-name collision safety across two markets.

Uses a TEMP, ISOLATED disk cache for _symbol_cache (data.universe) so this
file's assertions never read or write the real ~/.cache/tattva store and are
unaffected by (and don't affect) prior runs.

Run: python -m research.test_stock_targets  (from the repo root)
"""
from __future__ import annotations

import os as _os
import sys as _sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import data.universe as universe_mod
from data.cache import Cache
from data.fetcher import fetch_stock_target_series
import core.config as config_mod
from core.config import (
    register_stock_target, ALL_TARGETS, TARGET_POLARITY, TARGET_ARCHETYPE,
    TARGET_EXCLUDED_PREDICTORS, TARGET_CATEGORIES, FREEFORM_STOCK_CATEGORIES,
)

import sys as _sys2
_sys2.argv = ["streamlit"]   # app.py imports streamlit; harmless in bare mode
import app


def _ohlcv(n=40, start="2024-01-02", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
         "Volume": rng.integers(1_000_000, 5_000_000, n)},
        index=dates,
    )


def run() -> None:
    checks = 0

    # Isolate data.universe's symbol-resolution cache from the real disk
    # store for the whole test run (restored in a finally block below).
    _tmpdir = tempfile.mkdtemp(prefix="tattva_symcache_")
    _orig_cache = universe_mod._symbol_cache
    _orig_fail_memo = universe_mod._symbol_fail_memo
    universe_mod._symbol_cache = Cache(ttl=7 * 86_400, version="v1",
                                        namespace="test_symbol_resolution",
                                        disk_dir=Path(_tmpdir))
    universe_mod._symbol_fail_memo = {}

    try:
        # ── A. fetch_stock_target_series ────────────────────────────────
        good_ohlcv = _ohlcv(n=40)
        with patch("data.fetcher.fetch_constituent_ohlcv") as m:
            m.return_value = {"XYZ": good_ohlcv}
            s = fetch_stock_target_series("XYZ", "2024-01-01", "2024-03-01")
            assert s is not None and len(s) == 40
            assert s.index.is_monotonic_increasing

        with patch("data.fetcher.fetch_constituent_ohlcv") as m:
            m.return_value = {}
            assert fetch_stock_target_series("XYZ", "2024-01-01", "2024-03-01") is None

        with patch("data.fetcher.fetch_constituent_ohlcv") as m:
            m.return_value = {"XYZ": pd.DataFrame()}
            assert fetch_stock_target_series("XYZ", "2024-01-01", "2024-03-01") is None

        with patch("data.fetcher.fetch_constituent_ohlcv") as m:
            m.return_value = {"XYZ": _ohlcv(n=10)}   # below the 20-row floor
            assert fetch_stock_target_series("XYZ", "2024-01-01", "2024-03-01") is None
        checks += 1

        # ── B. _ensure_stock_target_column ──────────────────────────────
        register_stock_target("TESTSTOCK (NSE)", "TESTSTOCK.NS", "india")

        n = 200
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        df = pd.DataFrame({"DATE": dates, "Gold": 1900 + np.cumsum(np.random.randn(n))})

        # Shifted calendar: series starts 5 sessions in, ends 3 sessions early.
        stock_dates = dates[5:-3]
        close = 100 + np.cumsum(np.random.randn(len(stock_dates)))
        series = pd.Series(close, index=stock_dates).sort_index()

        with patch("app.fetch_stock_target_series") as m:
            m.return_value = series
            out = app._ensure_stock_target_column(df, "TESTSTOCK (NSE)")
            assert "TESTSTOCK (NSE)" in out.columns
            assert out["TESTSTOCK (NSE)"].iloc[:5].isna().all(), "leading NaN before listing expected"
            assert out["TESTSTOCK (NSE)"].iloc[5:].notna().all(), "no gaps after listing start"
            # tail 3 rows (no native data) must equal the last real observation (ffill)
            assert (out["TESTSTOCK (NSE)"].iloc[-3:] == out["TESTSTOCK (NSE)"].iloc[-4]).all()
            assert m.call_args[0][0] == "TESTSTOCK.NS"

        # No-op: column already present.
        out_noop1 = app._ensure_stock_target_column(out, "TESTSTOCK (NSE)")
        assert out_noop1 is out

        # No-op: non-'self' archetype target ABSENT from the frame — the
        # archetype guard must short-circuit before any fetch. (Commodity
        # futures are archetype 'self' now, so use an index target, which
        # stays basket-mode, and which df does not carry as a column.)
        assert TARGET_ARCHETYPE.get("Nifty 50") != "self"
        assert "Nifty 50" not in df.columns
        out_noop2 = app._ensure_stock_target_column(df, "Nifty 50")
        assert out_noop2 is df

        # No-op: 'self' archetype but no ticker registered under this name.
        assert TARGET_ARCHETYPE.get("Ghost Target") != "self"  # sanity: truly unregistered
        out_noop3 = app._ensure_stock_target_column(df, "Ghost Target")
        assert out_noop3 is df

        # Fetch returns None (symbol resolved earlier but data now unavailable)
        # -> frame unchanged, no KeyError, no column added; the app.py guard
        # right after the call site fires the honest error in that case.
        with patch("app.fetch_stock_target_series") as m:
            m.return_value = None
            out_fail = app._ensure_stock_target_column(df, "TESTSTOCK (NSE)")
            assert "TESTSTOCK (NSE)" not in out_fail.columns
        checks += 1

        # ── C. resolve_stock_symbol ──────────────────────────────────────
        _good = pd.Series(np.arange(25.0), index=pd.date_range("2024-01-01", periods=25))

        # C1: NS hits on the first probe -> BO never tried.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = _good
            ticker, exch = universe_mod.resolve_stock_symbol("RELTEST1", "india")
            assert (ticker, exch) == ("RELTEST1.NS", "NSE")
            assert m.call_count == 1

        # C2: NS misses, BO hits -> exactly 2 calls, NS probed first.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            calls = []
            def _se(tkr, start, end):
                calls.append(tkr)
                return _good if tkr.endswith(".BO") else None
            m.side_effect = _se
            ticker, exch = universe_mod.resolve_stock_symbol("BSETEST2", "india")
            assert (ticker, exch) == ("BSETEST2.BO", "BSE")
            assert calls == ["BSETEST2.NS", "BSETEST2.BO"]

        # C3/C4: explicit suffix short-circuits to a single probe.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = _good
            ticker, exch = universe_mod.resolve_stock_symbol("EXPLICIT3.BO", "india")
            assert (ticker, exch) == ("EXPLICIT3.BO", "BSE")
            assert m.call_count == 1
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = _good
            ticker, exch = universe_mod.resolve_stock_symbol("EXPLICIT4.NS", "india")
            assert (ticker, exch) == ("EXPLICIT4.NS", "NSE")
            assert m.call_count == 1

        # C5: US dot -> dash translation.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = _good
            ticker, exch = universe_mod.resolve_stock_symbol("BRK.TEST5", "us")
            assert ticker == "BRK-TEST5" and exch == "US"

        # C6: both candidates miss -> (None, msg naming both); failure is
        # SESSION-memoized (no re-probe within this process) but not disk-
        # backed — proven by clearing the session memo and observing a
        # fresh probe happen again.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = None
            ticker, msg = universe_mod.resolve_stock_symbol("BOGUS6", "india")
            assert ticker is None
            assert "BOGUS6.NS" in msg and "BOGUS6.BO" in msg
            first_calls = m.call_count
            # Same symbol again, same process -> session memo hit, no re-probe.
            ticker2, msg2 = universe_mod.resolve_stock_symbol("BOGUS6", "india")
            assert ticker2 is None and m.call_count == first_calls
        # Simulate a fresh session (memo cleared) -> re-probes for real,
        # proving the earlier failure was never written to the disk cache.
        universe_mod._symbol_fail_memo.clear()
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = None
            universe_mod.resolve_stock_symbol("BOGUS6", "india")
            assert m.call_count > 0, "failure must not be disk-memoized"

        # C7: invalid input rejected before any fetch.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            for bad in ("", "   ", "HAS SPACE", "X" * 25):
                t, e = universe_mod.resolve_stock_symbol(bad, "india")
                assert t is None
            assert m.call_count == 0

        # C8: a SUCCESSFUL resolution IS disk-memoized — a second call with a
        # brand-new mock (simulating a later process) must not re-probe.
        with patch("data.fetcher.fetch_stock_target_series") as m:
            m.return_value = _good
            universe_mod.resolve_stock_symbol("CACHEHIT8", "india")
        with patch("data.fetcher.fetch_stock_target_series") as m2:
            m2.return_value = _good
            ticker, exch = universe_mod.resolve_stock_symbol("CACHEHIT8", "india")
            assert (ticker, exch) == ("CACHEHIT8.NS", "NSE")
            assert m2.call_count == 0, "successful resolution should be cache-served"
        checks += 1

        # ── D. register_stock_target ──────────────────────────────────────
        _cats_before = {k: list(v) for k, v in TARGET_CATEGORIES.items()}
        register_stock_target("DTEST (NSE)", "DTEST.NS", "india")
        assert ALL_TARGETS["DTEST (NSE)"] == "DTEST.NS"
        assert TARGET_POLARITY["DTEST (NSE)"] == 1
        assert TARGET_ARCHETYPE["DTEST (NSE)"] == "self"
        excl_india = set(TARGET_EXCLUDED_PREDICTORS["DTEST (NSE)"])
        assert "India Equity" in excl_india      # _INDIA_INDEX_ETFS member
        assert "US Large Cap (S&P 500)" not in excl_india
        assert TARGET_CATEGORIES == _cats_before, "must not mutate TARGET_CATEGORIES"

        register_stock_target("DTEST_US (US)", "DTEST", "us")
        excl_us = set(TARGET_EXCLUDED_PREDICTORS["DTEST_US (US)"])
        assert "US Large Cap (S&P 500)" in excl_us
        assert "India Equity" not in excl_us

        # Idempotency: re-register with identical args -> no change.
        before = (dict(ALL_TARGETS), dict(TARGET_POLARITY), dict(TARGET_ARCHETYPE),
                  {k: list(v) for k, v in TARGET_EXCLUDED_PREDICTORS.items()})
        register_stock_target("DTEST (NSE)", "DTEST.NS", "india")
        after = (dict(ALL_TARGETS), dict(TARGET_POLARITY), dict(TARGET_ARCHETYPE),
                 {k: list(v) for k, v in TARGET_EXCLUDED_PREDICTORS.items()})
        assert before == after

        # Display-name collision: same base symbol, two markets -> distinct entries.
        register_stock_target("INFY (NSE)", "INFY.NS", "india")
        register_stock_target("INFY (US)", "INFY", "us")
        assert ALL_TARGETS["INFY (NSE)"] == "INFY.NS"
        assert ALL_TARGETS["INFY (US)"] == "INFY"
        assert ALL_TARGETS["INFY (NSE)"] != ALL_TARGETS["INFY (US)"]

        # FREEFORM_STOCK_CATEGORIES entries exist in TARGET_CATEGORIES so the
        # Asset Class selector lists them even with zero static members.
        for _cat in FREEFORM_STOCK_CATEGORIES:
            assert _cat in TARGET_CATEGORIES
        checks += 1

    finally:
        universe_mod._symbol_cache = _orig_cache
        universe_mod._symbol_fail_memo = _orig_fail_memo
        shutil.rmtree(_tmpdir, ignore_errors=True)

    print(f"stock-targets integrity: ALL {checks} CHECK GROUPS PASSED "
          f"(fetch_stock_target_series, _ensure_stock_target_column, "
          f"resolve_stock_symbol, register_stock_target)")


if __name__ == "__main__":
    run()
