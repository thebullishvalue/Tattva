"""
Tests for analytics.analogs.analog_prediction_series — the Precedent tab's
"Analog Predictions Over Time" plot data.

The property that MATTERS here is CAUSALITY: the prediction at any past as-of
date must be a function of data available at that date only. The strongest
test of that is mutation: change the FUTURE of the series and assert every
prediction at or before the mutation point is bit-identical. Also pins the
stride/last-point conventions and Realized alignment.

Run: python -m research.test_analog_series  (from the repo root)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.analogs import analog_prediction_series


def _make_ts(n: int = 900, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    valid = np.ones(n, dtype=bool)
    valid[:120] = False   # engine warm-up region
    return pd.DataFrame({
        "Price": price,
        "OversoldBreadth": rng.uniform(0, 100, n),
        "OverboughtBreadth": rng.uniform(0, 100, n),
        "AvgZ": rng.normal(0, 1, n),
        "Valid": valid,
        "Date": pd.date_range("2019-01-01", periods=n, freq="B"),
    })


def run() -> None:
    H, MOM = 10, 20
    ts = _make_ts()
    n = len(ts)

    df = analog_prediction_series(ts, "Gold", H, mom_window=MOM)
    assert len(df) >= 10, f"too few as-of rows: {len(df)}"
    assert list(df.columns) == ["Date", "Predicted", "Realized"]
    checks = 1

    # 1. Predictions finite; Realized NaN exactly on windows that haven't
    #    completed (the last as-of dates), finite elsewhere.
    assert np.isfinite(df["Predicted"]).all()
    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(ts["Date"])}
    for _, row in df.iterrows():
        t = date_to_pos[pd.Timestamp(row["Date"])]
        if t + H < n:
            assert np.isfinite(row["Realized"]), f"Realized NaN at completed window t={t}"
        else:
            assert not np.isfinite(row["Realized"]), f"Realized set on open window t={t}"
    checks += 1

    # 2. Realized alignment: matches the actual +H return from the as-of row.
    for _, row in df.iterrows():
        t = date_to_pos[pd.Timestamp(row["Date"])]
        if t + H < n:
            want = (ts["Price"].iloc[t + H] / ts["Price"].iloc[t] - 1) * 100
            assert abs(row["Realized"] - want) < 1e-9, (t, row["Realized"], want)
    checks += 1

    # 3. Last as-of point is the LATEST row (appended off-stride if needed) —
    #    the series must end at the live prediction the tab cards show.
    assert pd.Timestamp(df["Date"].iloc[-1]) == pd.Timestamp(ts["Date"].iloc[n - 1])
    checks += 1

    # 4. CAUSALITY (mutation test): rewrite the future — every prediction at
    #    or before the mutation point must be BIT-IDENTICAL. This catches any
    #    leak: pool bounds, median cleaning, covariance, normalisation.
    cut = n - 200
    ts_mut = ts.copy()
    rng2 = np.random.default_rng(99)
    ts_mut.loc[cut:, "Price"] = ts["Price"].iloc[cut] * np.exp(
        np.cumsum(rng2.normal(0.01, 0.05, n - cut)))   # wildly different future
    ts_mut.loc[cut:, "OversoldBreadth"] = rng2.uniform(0, 100, n - cut)
    ts_mut.loc[cut:, "OverboughtBreadth"] = rng2.uniform(0, 100, n - cut)
    df_mut = analog_prediction_series(ts_mut, "Gold", H, mom_window=MOM)

    # Predictions at as-of dates whose ENTIRE input (state features look back
    # ~3*mom_window for Hurst; realized outcomes end at t) predates the
    # mutation must be identical. Conservative safety margin: dates at or
    # before cut - 3*MOM - 1.
    safe = pd.Timestamp(ts["Date"].iloc[cut - 3 * MOM - 1])
    a = df[df["Date"] <= safe].set_index("Date")["Predicted"]
    b = df_mut[df_mut["Date"] <= safe].set_index("Date")["Predicted"]
    assert len(a) >= 5, "mutation test needs enough pre-cut as-of dates"
    assert a.index.equals(b.index), "as-of grid changed before the mutation point"
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=0.0), \
        "future mutation changed past predictions — CAUSALITY VIOLATION"
    checks += 1

    # 5. Warm-up exclusion: no as-of date inside the ValidRow=False region.
    warmup_last = pd.Timestamp(ts["Date"].iloc[119])
    assert (df["Date"] > warmup_last).all(), "as-of grid includes warm-up rows"
    checks += 1

    # 6. Stride: consecutive completed as-of positions are >= H apart
    #    (non-overlapping outcome windows), except the appended final point.
    pos = [date_to_pos[pd.Timestamp(d)] for d in df["Date"]]
    gaps = np.diff(pos[:-1])
    assert (gaps >= H).all(), f"stride violated: min gap {gaps.min()}"
    checks += 1

    print(f"analog_prediction_series: ALL {checks} CHECK GROUPS PASSED "
          f"({len(df)} as-of dates, causality mutation test included)")


if __name__ == "__main__":
    run()
