"""No look-ahead, no repaint: Tattva run on data truncated at T must publish exactly what the
full run published for every date up to T. Uses the cached universe (``tattva.data``); skips
when no data is cached.

    python -m pytest tests/test_causality.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tattva import engine
from tattva.data import fetch_universe

TARGETS = ("Gold", "USD/INR", "Bitcoin", "Nifty 50", "S&P 500", "Brent Crude")
CUTS = (250, 600)
COLS = ["M", "S", "F", "z", "mu", "Ey", "call", "exp_move"]


@pytest.fixture(scope="module")
def universe():
    try:
        u = fetch_universe()
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"no data: {e}")
    if not u:
        pytest.skip("no data")
    return u


def _cut(df: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    return df[df.index <= T]


@pytest.mark.parametrize("name", TARGETS)
@pytest.mark.parametrize("cut", CUTS)
def test_truncation_matches(universe, name, cut):
    from tattva.universe import TARGETS_BY_NAME
    t = TARGETS_BY_NAME[name]
    bars = universe.get(t.ticker)
    if bars is None:
        pytest.skip(f"{name} not in the cached universe")
    world = engine.world_close(universe)
    full = engine.analyze(name, t.ticker, t.category, bars, world).out
    T = bars.index[-cut]
    part = engine.analyze(name, t.ticker, t.category, _cut(bars, T), _cut(world, T)).out
    idx = part.index
    a, b = full.loc[idx, COLS].astype(float), part[COLS].astype(float)
    both_nan = a.isna() & b.isna()
    diff = (a - b).abs().where(~both_nan, 0.0)
    assert not (a.isna() ^ b.isna()).any().any(), "a value appears or disappears after truncation"
    assert float(np.nanmax(diff.to_numpy())) < 1e-9, f"max diff {np.nanmax(diff.to_numpy())}"
