"""
Tattva — the universe scan: Mūla's read on every target, side by side.

Runs the valuation engine only (no Swayam, no precedent) per target and
returns one row each: the state, the gap, its persistence, the target's own
call record and whether its asset class licenses a call.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from convergence.intelligence import call_record
from core.config import asset_class
from core.semantics import claim_direction
from core.trust import class_record, is_licensed

from .prep import PrepError, prepare
from .run import run_mula

COLUMNS = ("target", "class", "state", "fvo", "pct_gap", "gap_percentile", "half_life",
           "mr_prob", "xs_consistency", "hit", "base", "claims", "licensed", "as_of",
           "withheld_reason", "error")


def scan_target(df: pd.DataFrame, target: str, today: pd.Timestamp | None = None) -> dict:
    """One row of the scan. Never raises: a failure is reported in ``error``."""
    row = dict.fromkeys(COLUMNS)
    row.update(target=target, **{"class": asset_class(target)}, licensed=False)
    try:
        prep = prepare(df, target)
        engine = run_mula(prep, today=today)
    except PrepError as e:
        row["error"] = e.title
        return row
    except Exception as e:                       # noqa: BLE001 - one target, not the scan
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    sig = engine.get_current_signal()
    ts = engine.ts_data
    r = sig.get("published_row")
    rec = call_record(pd.Series(ts["FVO"].to_numpy(dtype=float), index=prep.calendar),
                      prep.price, prep.config.forecast_horizon)
    d = claim_direction(sig.get("fvo"))
    side = (class_record(target) or {}).get("long" if d > 0 else "short") if d else None
    own = rec.get("long" if d > 0 else "short") if d else rec

    def _n(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        return v if np.isfinite(v) else np.nan

    row.update(
        state=sig.get("valuation_state"), fvo=_n(sig.get("fvo")),
        pct_gap=_n(sig.get("pct_mispricing")) * 100.0,
        gap_percentile=(_n(ts["GapPercentile"].iloc[r]) * 100.0 if r is not None else np.nan),
        half_life=_n(sig.get("gap_half_life")), mr_prob=_n(sig.get("mr_prob")),
        xs_consistency=_n(sig.get("xs_consistency")),
        hit=_n((own or {}).get("hit")), base=_n((own or {}).get("base")),
        claims=int((own or {}).get("n_claims", 0) or 0),
        licensed=bool(d) and is_licensed(side),
        as_of=(prep.calendar[r].strftime("%Y-%m-%d") if r is not None else None),
        withheld_reason=(sig.get("withheld_reason") or None),
    )
    return row


def scan(df: pd.DataFrame, targets, progress: Callable[[int, int, str], None] | None = None,
         today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Scan ``targets`` on the shared model matrix ``df``; one row per target."""
    targets = list(targets)
    rows = []
    for i, t in enumerate(targets):
        if progress is not None:
            progress(i, len(targets), t)
        rows.append(scan_target(df, t, today))
    if progress is not None:
        progress(len(targets), len(targets), "")
    return pd.DataFrame(rows, columns=list(COLUMNS))
