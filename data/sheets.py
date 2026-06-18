"""
Tattva — Google Sheets price fetcher (exogenous, non-yfinance targets).
तत्त्व (Tattva) — "Principle / Essence"

Some targets are not on yfinance — e.g. **Jeera** (NCDEX cumin), an Indian spice
whose daily price lives in a published Google Sheet rather than in the macro
universe. This module fetches such a series with the same resilience contract as
``data/fetcher.py``:

  1. **Two-tier cache** (memory + disk, TTL) — ``sheets_cache``.
  2. **Circuit breaker** (CLOSED → OPEN → HALF_OPEN) — ``sheets_circuit``.
  3. **Retry-with-backoff** for transient HTTP failures.
  4. **Stale fallback** — last-good cache snapshot, then a *committed* CSV
     snapshot under ``data/snapshots/`` so the series survives even a cold start
     with the sheet offline / made private.

The returned object is a single ``pd.Series`` (datetime index, float values)
named after the target, ready to be reindexed onto the macro calendar and
injected as a column in the Aarambh model matrix.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from data.cache import Cache
from data.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    RetryWithBackoff,
)

log = logging.getLogger(__name__)

# ─── Source registry ─────────────────────────────────────────────────────────
# Each exogenous sheet target → (spreadsheet_id, gid, committed-snapshot file).
# The snapshot is the cold-start fallback shipped in the repo (data/snapshots/).
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

SHEET_SOURCES: dict[str, dict[str, str]] = {
    "Jeera": {
        "sheet_id": "1WfT2EGCyqPuKXtejp2CZsFBs9T3Lsa8JCPuHUlv-rUo",
        "gid": "0",
        "snapshot": "jeera.csv",
    },
}

# gviz endpoint (not /export): it reliably honours an explicit ?gid= for the tab
# and returns CSV, whereas /export?format=csv&gid= 400s on some sheets.
_EXPORT_URL = (
    "https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
)
_HTTP_TIMEOUT = 20  # seconds

sheets_cache = Cache(ttl=3600, version="v1", namespace="sheets")
sheets_circuit = CircuitBreaker(
    name="gsheets",
    failure_threshold=3,
    recovery_timeout=60.0,
)


# ─── Parsing ─────────────────────────────────────────────────────────────────

def _parse_price_csv(text: str, name: str) -> pd.Series:
    """Parse a ``Date,Price`` CSV (dd/mm/yyyy dates, comma-thousands prices)
    into a clean, de-duplicated, datetime-indexed float Series named ``name``.
    """
    if "<!DOCTYPE html" in text[:200] or "<html" in text[:200].lower():
        # Google served a login/error page instead of CSV (sheet went private).
        raise ValueError(f"{name}: sheet did not return CSV (HTML login/error page)")

    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"{name}: expected 'Date,Price' columns, got {list(df.columns)}")

    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    prices = pd.to_numeric(
        df["Price"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    s = pd.Series(prices.values, index=dates, name=name)
    s = s[s.index.notna() & s.notna() & (s != 0.0)]
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise ValueError(f"{name}: no valid rows after parsing")
    return s


@RetryWithBackoff(max_retries=2, initial_delay=1.5, backoff_factor=2.0)
def _download_sheet_csv(url: str) -> str:
    """Single HTTP GET of the published-CSV export — wrapped with retry."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Tattva)"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
        raw = resp.read()
    if not raw:
        raise ValueError("Empty sheet response")
    return raw.decode("utf-8", errors="replace")


def _load_snapshot(name: str) -> pd.Series | None:
    """Parse the committed CSV snapshot for ``name`` (cold-start fallback)."""
    meta = SHEET_SOURCES.get(name)
    if not meta:
        return None
    path = _SNAPSHOT_DIR / meta["snapshot"]
    if not path.exists():
        return None
    try:
        return _parse_price_csv(path.read_text(encoding="utf-8"), name)
    except Exception as e:  # noqa: BLE001
        log.warning("%s snapshot parse failed: %s", name, e)
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_sheet_series(
    name: str,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.Series | None:
    """Fetch an exogenous price Series with cache + circuit + stale + snapshot.

    ``name`` must be registered in :data:`SHEET_SOURCES`. ``start_date`` /
    ``end_date`` (optional) clip the returned series. Returns ``None`` only if
    every tier — live, cache, committed snapshot — is unavailable.
    """
    meta = SHEET_SOURCES.get(name)
    if not meta:
        log.error("No sheet source registered for target %r", name)
        return None

    url = _EXPORT_URL.format(sheet_id=meta["sheet_id"], gid=meta["gid"])

    cached = sheets_cache.get(name, url)
    series: pd.Series | None = cached if cached is not None else None

    if series is None:
        try:
            text = sheets_circuit.call(_download_sheet_csv, url)
            series = _parse_price_csv(text, name)
            sheets_cache.put(name, url, value=series)
        except CircuitBreakerError as e:
            log.warning("%s sheet circuit open: %s", name, e)
            series = sheets_cache.get_stale(name, url)
        except Exception as e:  # noqa: BLE001
            log.warning("%s sheet fetch failed: %s", name, e)
            series = sheets_cache.get_stale(name, url)

    if series is None:
        series = _load_snapshot(name)
        if series is not None:
            log.warning("%s served from committed snapshot (live + cache unavailable)", name)

    if series is None or series.empty:
        return None

    if start_date is not None:
        series = series[series.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        series = series[series.index <= pd.Timestamp(end_date)]
    return series


def fetch_jeera_series(
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.Series | None:
    """Convenience wrapper: NCDEX Jeera daily price as a ``Jeera`` Series."""
    return fetch_sheet_series("Jeera", start_date, end_date)
