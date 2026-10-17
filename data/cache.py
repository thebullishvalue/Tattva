"""
Tattva — TTL-based memory and disk cache for data fetching.
तत्त्व (Tattva) — "Principle / Essence"

A two-tier (memory + disk) cache with TTL expiry, versioned keys, and a
last-good-snapshot fallback that returns expired data when a downstream fetch
fails (so the UI can keep working even if APIs are down).

Cache keys are derived from a (version, *args) tuple via MD5 hashing; bumping
the `version` parameter atomically invalidates a whole namespace.

A forced refresh is NOT a property of the cache. It used to be: a per-session
five-minute window during which every ``get`` from that session missed. That
re-pulled the whole ~240-ticker universe twice per Refresh (the handler's own
fetch, then the rerun's), made every target switch inside the window download
everything again, and bypassed caches nobody had asked to refresh (symbol
resolution included). A caller that wants live data now says so at the call —
``fetch_commodity_dataset(..., force=True)`` skips ``get`` for exactly that
fetch, still ``put``s the result, and still falls back to ``get_stale`` if the
live pull fails. One request, one pull.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tattva"
DEFAULT_TTL_SECONDS = 3600  # 1 hour

# Disk snapshot retention (data.fetcher's rate-limit column-backfill scans
# these; keep enough that a multi-day gap still has something to fall back to
# without the namespace growing forever — see B3 in the audit).
_SNAPSHOT_RETENTION_DAYS = 7

class Cache:
    """TTL cache with memory tier, disk tier, and stale-fallback.

    Args:
        ttl: entry lifetime in seconds (used for the "fresh" path).
        disk_dir: directory for disk persistence. ``None`` → use default.
        version: namespace tag baked into every key (bump to invalidate all).
        namespace: optional sub-folder under ``disk_dir`` (e.g. "sheets").
    """

    def __init__(
        self,
        ttl: int = DEFAULT_TTL_SECONDS,
        disk_dir: Path | None = None,
        version: str = "v1",
        namespace: str = "",
    ) -> None:
        self.ttl = ttl
        self.version = version
        self._memory: dict[str, tuple[Any, float]] = {}
        base = disk_dir or DEFAULT_CACHE_DIR
        self._disk_dir = base / namespace if namespace else base
        self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Stats — exposed to the diagnostics view.
        self.hits = 0
        self.misses = 0
        self.stale_hits = 0   # served from expired snapshot during failure
        self.writes = 0
        self.last_fetch_time: float | None = None

    def _key(self, *args: Any) -> str:
        raw = f"{self.version}|" + "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *args: Any) -> Any | None:
        """Return cached value if fresh (within TTL), else None.

        A caller forcing a live pull simply does not call this (see the module
        docstring); the disk snapshot is untouched either way, so ``get_stale``
        still serves last-good data if that live pull fails.
        """
        key = self._key(*args)
        with self._lock:
            if key in self._memory:
                val, ts = self._memory[key]
                if time.time() - ts < self.ttl:
                    self.hits += 1
                    return val
                del self._memory[key]

        disk_path = self._disk_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    val, ts = pickle.load(f)
                if time.time() - ts < self.ttl:
                    with self._lock:
                        self._memory[key] = (val, ts)
                        self.hits += 1
                    return val
            except Exception as e:
                log.warning("Cache disk read failed for %s: %s", key[:8], e)

        with self._lock:
            self.misses += 1
        return None

    def get_stale(self, *args: Any) -> Any | None:
        """Return last-good value even if expired. Used as fetch-failure fallback."""
        key = self._key(*args)
        # Read + stats update under ONE lock acquisition (audit finding F23) —
        # every other _memory access in this class (get/put/invalidate) reads
        # under the lock; this method previously checked/read the dict first
        # and only took the lock afterward to bump stale_hits, a narrow window
        # where a concurrent put() could mutate the dict mid-read.
        with self._lock:
            if key in self._memory:
                val, _ = self._memory[key]
                self.stale_hits += 1
                return val
        disk_path = self._disk_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    val, ts = pickle.load(f)
                with self._lock:
                    self._memory[key] = (val, ts)
                    self.stale_hits += 1
                return val
            except Exception:
                pass
        return None

    def put(self, *args: Any, value: Any) -> None:
        """Store value with the current timestamp (both tiers)."""
        key = self._key(*args)
        ts = time.time()
        with self._lock:
            self._memory[key] = (value, ts)
            self.writes += 1
            self.last_fetch_time = ts
        disk_path = self._disk_dir / f"{key}.pkl"
        try:
            with open(disk_path, "wb") as f:
                pickle.dump((value, ts), f)
        except Exception as e:
            log.warning("Cache disk write failed for %s: %s", key[:8], e)
        self._prune_old_disk_entries()

    def _prune_old_disk_entries(self) -> None:
        """Delete disk entries older than ``_SNAPSHOT_RETENTION_DAYS``.

        Namespaces whose key includes a rolling date (e.g. macro_cache, whose
        args include an end-date that advances daily) accumulate one new
        ~5-10MB pickle per day forever with no eviction (audit finding B3).
        Keeps a retention window rather than pruning to just the newest file,
        since data.fetcher's rate-limit backfill (_load_macro_snapshots_
        newest_first) scans recent-but-not-latest snapshots for columns the
        current fetch is missing.
        """
        cutoff = time.time() - _SNAPSHOT_RETENTION_DAYS * 86400
        try:
            for p in self._disk_dir.glob("*.pkl"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    continue
        except Exception as e:  # noqa: BLE001
            log.debug("Cache prune skipped for %s: %s", self._disk_dir, e)

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()

    def stats(self) -> dict[str, Any]:
        """Snapshot of cache stats for diagnostics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {
                "namespace": self._disk_dir.name,
                "version": self.version,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "stale_hits": self.stale_hits,
                "writes": self.writes,
                "hit_rate": hit_rate,
                "memory_entries": len(self._memory),
                "disk_entries": len(list(self._disk_dir.glob("*.pkl"))),
                "last_fetch_time": self.last_fetch_time,
            }


# ── Module-level cache instances ─────────────────────────────────────────────
# One per data source, so namespaces stay isolated and versions are independent.

ohlcv_cache = Cache(ttl=3600, version="v1", namespace="ohlcv")
macro_cache = Cache(ttl=3600, version="v1", namespace="macro")


def all_caches() -> list[Cache]:
    """Return all module-level cache instances, for the diagnostics view."""
    from data.sheets import sheets_cache       # local imports avoid an import cycle
    from data.universe import _symbol_cache
    return [ohlcv_cache, macro_cache, sheets_cache, _symbol_cache]
