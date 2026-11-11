"""
Tattva — the call's standing record, by asset class.

``data/priors/call_record.json`` is written by ``studies/pooled_record.py``:
every catalogue target's call graded as issued, pooled within its asset class.
The verdict reads it to decide whether a claim may be called a BUY or a SELL.
One target's own history is a few hundred independent calls at most, which
cannot tell an edge from noise; a class pools several thousand.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from core.config import asset_class

RECORD_PATH = Path(__file__).resolve().parent.parent / "data" / "priors" / "call_record.json"

#: A side of a class is licensed when its pooled record is at least this many
#: standard errors above zero AND its hit rate beats the majority-direction
#: base on the same sessions.
LICENCE_T = 1.0


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_record(path: Path | str | None = None) -> dict:
    """The committed record, or ``{}`` when it is missing or unreadable."""
    return _load(str(path or RECORD_PATH))


def class_record(target: str, record: dict | None = None) -> dict | None:
    """The pooled record of ``target``'s asset class, or None."""
    rec = load_record() if record is None else record
    cls = asset_class(target)
    out = (rec.get("classes") or {}).get(cls) if cls else None
    return dict(out, asset_class=cls) if out else None


def _num(d: dict | None, key: str) -> float | None:
    try:
        v = float((d or {}).get(key))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def is_licensed(side_record: dict | None) -> bool:
    """Does this pooled side record support issuing the call?"""
    t, hit, base = _num(side_record, "t"), _num(side_record, "hit"), _num(side_record, "base")
    return t is not None and hit is not None and base is not None and t >= LICENCE_T and hit > base
