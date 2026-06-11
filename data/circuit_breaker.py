"""
Tattva v2.0.0 — Circuit Breaker + Retry-with-Backoff fault tolerance.
तत्त्व (Tattva) — "Principle / Essence"

Fault-tolerance primitives for external service calls (yfinance, Google Sheets,
Stooq). Adapted from Pragyam's circuit_breaker.py, using stdlib logging instead
of Pragyam's logger_config.

State machine:
    CLOSED → OPEN: when failure_count >= failure_threshold
    OPEN → HALF_OPEN: after recovery_timeout elapses since last failure
    HALF_OPEN → CLOSED: on the next successful call
    HALF_OPEN → OPEN: if the test call also fails
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a call is blocked because the circuit is OPEN."""


class CircuitBreaker:
    """Per-service circuit breaker. Thread-safe."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self.last_failure_time is None:
                    raise CircuitBreakerError(f"Circuit '{self.name}' is OPEN")
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    remaining = self.recovery_timeout - elapsed
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is OPEN — retry in {remaining:.1f}s"
                    )

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls > self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' HALF_OPEN — max test calls exceeded"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            prev_state = self.state
            self.success_count += 1
            self.last_success_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
        if prev_state == CircuitState.HALF_OPEN:
            log.info("Circuit '%s' CLOSED — service recovered", self.name)

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                log.warning("Circuit '%s' recovery failed — back to OPEN", self.name)
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    log.error(
                        "Circuit '%s' OPEN — %d failures (threshold %d)",
                        self.name,
                        self.failure_count,
                        self.failure_threshold,
                    )

    def protect(self, func: Callable) -> Callable:
        """Decorator form: wraps `func` with circuit breaker protection."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure": self.last_failure_time,
            "last_success": self.last_success_time,
        }

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.half_open_calls = 0
        log.info("Circuit '%s' manually reset", self.name)


class RetryWithBackoff:
    """Exponential-backoff retry decorator (1s → 2s → 4s → … capped)."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            delay = self.initial_delay
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exc = e
                    if attempt < self.max_retries:
                        log.warning(
                            "Attempt %d/%d failed (%s) — retrying in %.1fs",
                            attempt + 1,
                            self.max_retries + 1,
                            type(e).__name__,
                            delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * self.backoff_factor, self.max_delay)
                    else:
                        log.error(
                            "All %d attempts failed. Last error: %s",
                            self.max_retries + 1,
                            e,
                        )
            assert last_exc is not None  # for type checker
            raise last_exc
        return wrapper


# ── Global circuit breakers ──────────────────────────────────────────────────
# Instantiated once per process; shared across fetcher calls.

yfinance_circuit = CircuitBreaker(
    name="yfinance",
    failure_threshold=5,
    recovery_timeout=60.0,
)


def all_circuits() -> list[CircuitBreaker]:
    """Return all module-level circuit breakers for diagnostics."""
    return [yfinance_circuit]
