"""
Tattva — Google Sheets price fetcher (exogenous, non-yfinance targets).
तत्त्व (Tattva) — "Principle / Essence"

Some targets are not on yfinance — e.g. **Jeera** (NCDEX cumin), an Indian spice
whose daily price lives in a published Google Sheet rather than in the macro
universe. This module fetches such a series with the same resilience contract as
``data/fetcher.py``:

  1. **Two-tier cache** (memory + disk, TTL) — ``sheets_cache``.
  2. **Circuit breaker** (CLOSED → OPEN → HALF_OPEN) — ``sheets_circuit``.
  3. **Retry-with-backoff** for transient HTTP failures, and a short negative
     cache so a sheet that is down is not re-tried (~4.5 s each) on every rerun.
  4. **Stale fallback** — the last-good cache entry, then an OPTIONAL committed
     CSV under ``data/snapshots/``.

On (4), plainly: no snapshot is committed today — ``data/snapshots/`` does not
exist in the repository — so a cold start with the sheet unreachable (offline,
made private, or a network that blocks Google) has no series, and the target
reports "source fetch failed". The code path stays because dropping a file named
in ``SHEET_SOURCES`` into that directory is all it takes to turn it on; until
then it is a no-op, and nothing here claims otherwise.

The returned object is a single ``pd.Series`` (datetime index, float values)
named after the target, ready to be reindexed onto the macro calendar and
injected as a column in the FVO model matrix.
"""

from __future__ import annotations

import io
import logging
import time
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
# Each exogenous sheet target → (spreadsheet_id, gid, optional snapshot file).
# A snapshot, if one is ever committed under data/snapshots/, is the cold-start
# fallback; the directory is absent today (see the module docstring).
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

SHEET_SOURCES: dict[str, dict[str, str]] = {
    "Jeera": {
        "sheet_id": "1WfT2EGCyqPuKXtejp2CZsFBs9T3Lsa8JCPuHUlv-rUo",
        "gid": "0",
        "value_col": "Price",
        "snapshot": "jeera.csv",
    },
    "Nifty 50 - PE": {
        "sheet_id": "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c",
        "gid": "990005192",
        "value_col": "NIFTY50_PE",
        "snapshot": "nifty50_pe.csv",
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

#: After a failed live fetch, how long to skip retrying that sheet. Every rerun
#: on a sheet target used to pay the full retry budget again, and the app's
#: freshness block asked a second time on the same rerun.
_FAIL_BACKOFF_SECONDS = 120.0
_fail_until: dict[str, float] = {}


# ─── Parsing ─────────────────────────────────────────────────────────────────

def _parse_price_csv(text: str, name: str, value_col: str = "Price") -> pd.Series:
    """Parse a sheet CSV into a clean, de-duplicated, datetime-indexed float Series
    named ``name``. The value column is ``value_col`` (e.g. ``Price`` for Jeera,
    ``NIFTY50_PE`` for the Nifty 50 PE sheet); the date column is auto-detected
    (first column whose header contains "date", else the first column). Dates are
    dd/mm/yyyy; values may carry comma-thousands.
    """
    if "<!DOCTYPE html" in text[:200] or "<html" in text[:200].lower():
        # Google served a login/error page instead of CSV (sheet went private).
        raise ValueError(f"{name}: sheet did not return CSV (HTML login/error page)")

    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if value_col not in df.columns:
        raise ValueError(f"{name}: value column {value_col!r} not found, got {list(df.columns)}")
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])

    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    prices = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", "", regex=False).str.strip(),
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
    """Parse the committed CSV snapshot for ``name``, or None — which is the
    answer today, since no snapshot is committed (see the module docstring)."""
    meta = SHEET_SOURCES.get(name)
    if not meta:
        return None
    path = _SNAPSHOT_DIR / meta["snapshot"]
    if not path.exists():
        return None
    try:
        return _parse_price_csv(path.read_text(encoding="utf-8"), name, meta.get("value_col", "Price"))
    except Exception as e:  # noqa: BLE001
        log.warning("%s snapshot parse failed: %s", name, e)
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_sheet_series(
    name: str,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    force: bool = False,
) -> pd.Series | None:
    """Fetch an exogenous price Series with cache + circuit + stale + snapshot.

    ``name`` must be registered in :data:`SHEET_SOURCES`. ``start_date`` /
    ``end_date`` (optional) clip the returned series. ``force`` skips the
    fresh-cache read (and the failure back-off) for this call only. Returns
    ``None`` only if every tier — live, cache, snapshot — is unavailable.
    """
    meta = SHEET_SOURCES.get(name)
    if not meta:
        log.error("No sheet source registered for target %r", name)
        return None

    url = _EXPORT_URL.format(sheet_id=meta["sheet_id"], gid=meta["gid"])

    series: pd.Series | None = None if force else sheets_cache.get(name, url)

    if series is None:
        if not force and time.monotonic() < _fail_until.get(name, 0.0):
            series = sheets_cache.get_stale(name, url)
        else:
            try:
                text = sheets_circuit.call(_download_sheet_csv, url)
                series = _parse_price_csv(text, name, meta.get("value_col", "Price"))
                sheets_cache.put(name, url, value=series)
                _fail_until.pop(name, None)
            except CircuitBreakerError as e:
                log.warning("%s sheet circuit open: %s", name, e)
                series = sheets_cache.get_stale(name, url)
            except Exception as e:  # noqa: BLE001
                log.warning("%s sheet fetch failed: %s", name, e)
                _fail_until[name] = time.monotonic() + _FAIL_BACKOFF_SECONDS
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
