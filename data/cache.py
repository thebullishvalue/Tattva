"""
Tattva — TTL-based memory and disk cache for data fetching.
तत्त्व (Tattva) — "Principle / Essence"

A two-tier (memory + disk) cache with TTL expiry, versioned keys, and a
last-good-snapshot fallback that returns expired data when a downstream fetch
fails (so the UI can keep working even if APIs are down).

Cache keys are derived from a (version, *args) tuple via MD5 hashing; bumping
the `version` parameter atomically invalidates a whole namespace.
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

# Force-refresh window, PER SESSION. While a session's entry here is unexpired,
# that session's ``Cache.get`` calls return None (-> live re-fetch) but the disk
# snapshot is preserved as a fallback. Set via ``begin_force_refresh()`` from the
# UI "Refresh Data" action.
#
# Streamlit serves every concurrent session from ONE process, so a single global
# deadline (the previous implementation) meant one user's "Refresh Data" click
# forced EVERY other concurrent session's next fetch to bypass cache too —
# amplifying rate-limit exposure across unrelated users for no benefit to them
# (audit finding B2). Keying by session ID scopes the bypass to the session that
# actually asked for it. Falls back to a single shared key outside a Streamlit
# run context (e.g. the research/ scripts, which don't have one) — identical to
# the old global-window behaviour there, since there's only ever one "session".
_FORCE_UNTIL: dict[str, float] = {}


def _current_session_key() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            # suppress_warning=True silences Streamlit's "missing ScriptRunContext"
            # log noise when called outside a script run (the research/ scripts run
            # bare — they legitimately have no session). Kwarg exists in current
            # Streamlit; fall back if a version ever drops it.
            ctx = get_script_run_ctx(suppress_warning=True)
        except TypeError:
            ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:
        pass
    return "_no_session_"


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

        During a user-triggered force-refresh window (``begin_force_refresh``) this
        returns ``None`` so the caller re-fetches live — WITHOUT deleting the disk
        snapshot, so ``get_stale`` still serves last-good data if the live fetch
        fails (rate limit / circuit open). That's the safety the naive "clear cache"
        lacks: a failed forced refresh degrades to stale, never to empty. The window
        is scoped to the CALLING SESSION (see ``_current_session_key``), so one
        user's forced refresh doesn't force every concurrent session's fetches to
        bypass cache too.
        """
        _deadline = _FORCE_UNTIL.get(_current_session_key(), 0.0)
        if time.time() < _deadline:
            with self._lock:
                self.misses += 1
            return None
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

    def invalidate(self, *args: Any) -> None:
        key = self._key(*args)
        with self._lock:
            self._memory.pop(key, None)
        disk_path = self._disk_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                disk_path.unlink()
            except Exception:
                pass

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


def begin_force_refresh(window: float = 300.0) -> None:
    """Open a force-refresh window for the CALLING SESSION: for ``window`` seconds,
    every ``Cache.get`` call made from this session misses (→ live re-fetch) while
    disk snapshots stay intact as a failure fallback. Self-clearing — covers one
    full pipeline re-pull, then normal TTL resumes. Scoped per-session so it
    doesn't force concurrent Streamlit sessions to bypass cache too (see
    ``_current_session_key``).
    """
    # Opportunistic cleanup of expired entries so this dict doesn't grow
    # unbounded across many sessions over a long-running server process.
    now = time.time()
    for k in [k for k, deadline in _FORCE_UNTIL.items() if deadline < now]:
        del _FORCE_UNTIL[k]
    _FORCE_UNTIL[_current_session_key()] = now + window


def all_caches() -> list[Cache]:
    """Return all module-level cache instances for diagnostics + force-refresh."""
    from data.sheets import sheets_cache       # local imports avoid an import cycle
    from data.universe import _constituent_cache
    return [ohlcv_cache, macro_cache, sheets_cache, _constituent_cache]
