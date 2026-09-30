"""
Data: unadjusted daily bars from yfinance (and the two published sheets), cached on disk.

Unadjusted (``auto_adjust=False``) on purpose: an adjusted series is rewritten backwards every
time a dividend or split lands, and a published Tattva reading must never move. Bars are
returned lower-case (open, high, low, close, volume) on a tz-naive daily index.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from tattva.universe import HISTORY_START, SHEETS, yf_tickers

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("TATTVA_CACHE", Path.home() / ".cache" / "tattva"))
CACHE_TTL = 6 * 3600          # seconds a universe download is reused
_COLS = ("open", "high", "low", "close", "volume")


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    """One ticker's raw yfinance frame → lower-case OHLCV on a tz-naive, deduplicated index."""
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)
    if "close" not in df:
        return None
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
                        for c in _COLS}, index=idx.normalize())
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["close"] > 0]
    # a bar without a usable open/high/low is carried on its close (FX and some indices)
    for c in ("open", "high", "low"):
        out[c] = out[c].where(out[c] > 0, out["close"])
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    return out if len(out) else None


def _download(tickers: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False,
                      group_by="ticker", threads=True)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        c = _clean(sub.dropna(how="all"))
        if c is not None:
            out[t] = c
    return out


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def _load(name: str, ttl: float | None) -> object | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    if ttl is not None and time.time() - p.stat().st_mtime > ttl:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:                                   # noqa: BLE001 — a bad cache is a miss
        return None


def _save(name: str, obj: object) -> None:
    tmp = _cache_path(name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(_cache_path(name))


def fetch_sheet(key: str) -> pd.DataFrame | None:
    """A published sheet's close series as OHLCV (open = high = low = close, no volume)."""
    meta = SHEETS[key]
    url = (f"https://docs.google.com/spreadsheets/d/{meta['sheet_id']}/gviz/tq"
           f"?tqx=out:csv&gid={meta['gid']}")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            text = r.read().decode("utf-8")
    except Exception as e:                              # noqa: BLE001
        log.warning("sheet %s unavailable: %s", key, e)
        return None
    if "<html" in text[:300].lower():
        log.warning("sheet %s returned HTML, not CSV", key)
        return None
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if meta["value_col"] not in df:
        return None
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    d = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    v = pd.to_numeric(df[meta["value_col"]].astype(str).str.replace(",", "", regex=False),
                      errors="coerce")
    s = pd.Series(v.to_numpy(), index=d).dropna()
    s = s[(s > 0) & s.index.notna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return None
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": np.nan})


def fetch_universe(force: bool = False, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Every universe ticker's bars (yfinance tickers, and ``sheet:<key>`` for the sheets).

    Cached for ``CACHE_TTL``. A failed or partial download falls back to the last good snapshot
    for the tickers it is missing, so a rate-limited run degrades to slightly stale data rather
    than to a different universe (which would move history).
    """
    name = "universe" if end is None else f"universe_{end}"
    if not force:
        hit = _load(name, CACHE_TTL if end is None else None)
        if hit is not None:
            return hit
    tickers = yf_tickers()
    got: dict[str, pd.DataFrame] = {}
    try:
        got = _download(tickers, HISTORY_START, end)
    except Exception as e:                              # noqa: BLE001
        log.warning("universe download failed: %s", e)
    missing = [t for t in tickers if t not in got]
    if missing:                                         # one targeted retry for the gaps
        try:
            got.update(_download(missing, HISTORY_START, end))
        except Exception as e:                          # noqa: BLE001
            log.warning("retry failed: %s", e)
    for key in SHEETS:
        s = fetch_sheet(key)
        if s is not None:
            got[f"sheet:{key}"] = s
    stale = _load(name, None) or {}
    for t, df in stale.items():                         # stale fallback for anything still missing
        got.setdefault(t, df)
    if got:
        _save(name, got)
    return got


def fetch_symbol(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """One free-form stock's bars (cached like the universe)."""
    name = "sym_" + "".join(c if c.isalnum() else "_" for c in ticker)
    if not force:
        hit = _load(name, CACHE_TTL)
        if hit is not None:
            return hit
    try:
        df = _download([ticker], HISTORY_START, None).get(ticker)
    except Exception as e:                              # noqa: BLE001
        log.warning("%s download failed: %s", ticker, e)
        df = None
    if df is None:
        return _load(name, None)
    _save(name, df)
    return df


def resolve_symbol(raw: str, market: str) -> tuple[str | None, str]:
    """User-typed symbol → (yfinance ticker, exchange) or (None, reason).

    India: an explicit .NS/.BO is used as typed, else NSE then BSE. US: the symbol with
    ``.`` → ``-`` (BRK.B → BRK-B)."""
    s = (raw or "").strip().upper()
    if not s or " " in s or len(s) > 20:
        return None, f"'{raw}' is not a valid symbol."
    if market == "india":
        cands = ([(s, "NSE" if s.endswith(".NS") else "BSE")] if s.endswith((".NS", ".BO"))
                 else [(f"{s}.NS", "NSE"), (f"{s}.BO", "BSE")])
    else:
        cands = [(s.replace(".", "-"), "US")]
    for t, ex in cands:
        df = fetch_symbol(t)
        if df is not None and len(df) >= 300:
            return t, ex
    return None, f"'{raw}' not found with enough history (tried {', '.join(t for t, _ in cands)})."
