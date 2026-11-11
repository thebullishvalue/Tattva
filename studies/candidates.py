"""
Tattva — the pre-registered candidate readings (see studies/PREREGISTRATION.md).

Each candidate maps one target's published Mūla frame to a claim series:
+1 buy, −1 sell, 0 no claim. None of this is production code; a candidate
reaches the product only by clearing the registered bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.semantics import CLAIM_SIGMA


def _contrarian(x: pd.Series, thr: float = CLAIM_SIGMA) -> pd.Series:
    return pd.Series(np.where(x <= -thr, 1.0, np.where(x >= thr, -1.0, 0.0)), index=x.index)


def c0_shipped(ts: pd.DataFrame) -> pd.Series:
    return _contrarian(ts["FVO"])


def c1_centred(ts: pd.DataFrame) -> pd.Series:
    f = ts["FVO"]
    mean = f.ewm(halflife=126, min_periods=63).mean().shift(1)
    return _contrarian(f - mean).where(mean.notna(), 0.0)


def c2_rank(ts: pd.DataFrame) -> pd.Series:
    p = ts["GapPercentile"].where(ts["FVO"].notna())
    return pd.Series(np.where(p < 0.20, 1.0, np.where(p > 0.80, -1.0, 0.0)), index=ts.index)


def c3_cheap_only(ts: pd.DataFrame) -> pd.Series:
    return _contrarian(ts["FVO"]).clip(lower=0.0)


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    horizon: int
    rule: Callable[[pd.DataFrame], pd.Series] | None = None
    fit: str = "base"            # which Mūla fit it reads: "base" or "fast"
    cross_sectional: bool = False


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("C0", "shipped call", 10, c0_shipped),
    Candidate("C1", "causally centred", 10, c1_centred),
    Candidate("C2", "causal rank 20/80", 10, c2_rank),
    Candidate("C3", "cheap only", 10, c3_cheap_only),
    Candidate("C4", "shipped, h = 5", 5, c0_shipped),
    Candidate("C5", "cross-sectional thirds", 10, None, cross_sectional=True),
    Candidate("C6", "fast memory", 10, c0_shipped, fit="fast"),
)

#: C6's coefficient discounts.
FAST_DELTAS = (0.995, 0.998, 0.999)


def cross_sectional_claims(fvo_by_target: dict[str, pd.Series], min_names: int = 3) -> dict[str, pd.Series]:
    """C5 within one asset class: each day buy the cheapest third, sell the richest."""
    wide = pd.DataFrame(fvo_by_target)
    n = wide.notna().sum(axis=1)
    rank = wide.rank(axis=1, pct=True)
    claim = pd.DataFrame(np.where(rank <= 1 / 3, 1.0, np.where(rank > 2 / 3, -1.0, 0.0)),
                         index=wide.index, columns=wide.columns)
    claim = claim.where(wide.notna(), 0.0)
    claim[n < min_names] = 0.0
    return {t: claim[t] for t in wide.columns}
