"""
Tattva — numba compatibility shim.
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — A resilient ``njit`` decorator.

Numba is an *optional accelerator* here, not a correctness dependency: every
``@njit`` kernel is written in numba-compatible Python and produces identical
results interpreted. But numba is also the most fragile piece of the native
stack — an ABI mismatch with numpy/llvmlite, or an unwritable cache dir, can
*segfault the whole process at import time* (the "Segmentation fault" seen in
Streamlit Cloud's run-streamlit.sh at boot) rather than raising a catchable
exception.

This shim isolates that risk:

  • If numba imports cleanly, ``njit`` is the real thing (JIT, fast).
  • If numba import fails, ``njit`` becomes a no-op decorator and the kernels
    run as ordinary (slower) Python — the app stays up instead of crashing.
  • Set ``TATTVA_DISABLE_NUMBA=1`` to force the pure-Python path regardless,
    which is the fastest way to confirm/deny "is numba the segfault cause?".

Usage — replace ``from numba import njit`` with::

    from analytics._njit_compat import njit
"""

from __future__ import annotations

import os


def _python_njit(*dargs, **dkwargs):
    """A no-op stand-in for ``numba.njit``.

    Supports both bare ``@njit`` and parametrized ``@njit(cache=True, ...)``
    usage, ignoring all numba-specific kwargs (``cache``, ``fastmath``,
    ``parallel``, …) and returning the undecorated function unchanged.
    """
    # Bare decorator form: @njit  (called once with the function itself)
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return dargs[0]

    # Parametrized form: @njit(cache=True)  → return the real decorator
    def _decorator(func):
        return func

    return _decorator


_FORCE_PYTHON = os.environ.get("TATTVA_DISABLE_NUMBA", "").strip() not in ("", "0", "false", "False")

if _FORCE_PYTHON:
    njit = _python_njit
    NUMBA_AVAILABLE = False
else:
    try:
        from numba import njit as _numba_njit  # noqa: F401
        njit = _numba_njit
        NUMBA_AVAILABLE = True
    except Exception:  # ImportError, or any native-init failure short of a segfault
        njit = _python_njit
        NUMBA_AVAILABLE = False
