"""
Execution test for engines.nirnay.apply_polarity (the inverse-basket path).

No live target currently sets TARGET_POLARITY = -1, so the polarity-inversion
branch ships unexercised. This test builds a full-schema aggregate frame (every
column aggregate_constituent_timeseries emits), flips it, and asserts the
inversion is correct, complete, and an involution. Run: python research/test_polarity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engines.nirnay import apply_polarity

# Every column aggregate_constituent_timeseries returns (engines/nirnay.py:498-525).
DIRECTIONAL_PAIRS = [
    ("Oversold", "Overbought"),
    ("Oversold_Pct", "Overbought_Pct"),
    ("Regime_Bull", "Regime_Bear"),
    ("Regime_Bull_Pct", "Regime_Bear_Pct"),
    ("Buy_Signals", "Sell_Signals"),
    ("Bull_Div", "Bear_Div"),
    ("avg_hmm_bull", "avg_hmm_bear"),
]
SIGNED = ["Avg_Signal", "Signal_Sum", "avg_msf_osc", "avg_mmr_osc"]
NEUTRAL = [
    "Neutral", "Neutral_Pct", "Total_Analyzed", "Regime_Neutral",
    "Regime_Transition", "Vol_High", "Vol_Low", "Vol_High_Pct", "Change_Points",
]


def _make_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", name="Date")
    cols = {}
    for a, b in DIRECTIONAL_PAIRS:
        cols[a] = rng.integers(0, 10, n).astype(float)
        cols[b] = rng.integers(0, 10, n).astype(float)
    for c in SIGNED:
        cols[c] = rng.normal(0, 3, n)
    for c in NEUTRAL:
        cols[c] = rng.integers(0, 10, n).astype(float)
    return pd.DataFrame(cols, index=idx)


def run() -> None:
    df = _make_frame()
    flipped = apply_polarity(df, polarity=-1)

    # 1. directional pairs swapped
    for a, b in DIRECTIONAL_PAIRS:
        assert (flipped[a] == df[b]).all(), f"pair {a}<->{b} not swapped"
        assert (flipped[b] == df[a]).all(), f"pair {a}<->{b} not swapped"

    # 2. signed oscillators negated
    for c in SIGNED:
        assert np.allclose(flipped[c], -df[c]), f"signed {c} not negated"

    # 3. neutral / count / vol columns untouched
    for c in NEUTRAL:
        assert (flipped[c] == df[c]).all(), f"neutral {c} mutated"

    # 4. source frame not mutated in place
    assert (df["Oversold"] != flipped["Oversold"]).any(), "source mutated in place"

    # 5. no-op for +1 / 0 / None (byte-for-byte identity)
    for p in (1, 0, None):
        assert apply_polarity(df, p).equals(df), f"polarity={p} not a no-op"

    # 6. involution: flipping twice restores the original
    twice = apply_polarity(flipped, polarity=-1)
    assert twice.equals(df), "double-flip is not identity (not an involution)"

    # 7. breadth conservation: %s that should sum to a constant still do
    pre = df["Oversold_Pct"] + df["Overbought_Pct"] + df["Neutral_Pct"]
    post = flipped["Oversold_Pct"] + flipped["Overbought_Pct"] + flipped["Neutral_Pct"]
    assert np.allclose(pre, post), "breadth %s not conserved under flip"

    # 8. empty / None inputs degrade gracefully
    assert apply_polarity(pd.DataFrame(), -1).empty
    assert apply_polarity(None, -1) is None

    print("apply_polarity: ALL 8 CHECKS PASSED")
    print(f"  schema cols covered: {len(DIRECTIONAL_PAIRS)*2 + len(SIGNED) + len(NEUTRAL)}")


if __name__ == "__main__":
    run()
