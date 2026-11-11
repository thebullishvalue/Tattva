"""Swayam: the view bank, its skill weights and its aggregate never look ahead."""
import numpy as np
import pandas as pd

from engines.swayam import aggregate_views, build_swayam_frames, default_swayam_members
from engines.swayam.kernel import calculate_mmr, view_skill_weights

RNG = np.random.default_rng(21)
T = 900
IDX = pd.bdate_range("2019-01-01", periods=T)


def _data():
    f = RNG.normal(0, 0.01, (T, 2))
    close = 100 * np.exp(np.cumsum(f @ np.array([0.7, -0.4]) + RNG.normal(0, 0.006, T)))
    spread = np.abs(RNG.normal(0, 0.004, T)) * close
    ohlcv = pd.DataFrame({"Open": close * (1 + RNG.normal(0, 0.002, T)), "High": close + spread,
                          "Low": close - spread, "Close": close,
                          "Volume": RNG.integers(1_000, 5_000, T).astype(float)}, index=IDX)
    load = RNG.normal(0, 1, (12, 2))
    macro = pd.DataFrame(50 * np.exp(np.cumsum(f @ load.T + RNG.normal(0, 0.004, (T, 12)), axis=0)),
                         index=IDX, columns=[f"M{j}" for j in range(12)])
    macro.iloc[:300, -1] = np.nan                       # a driver that lists late
    return ohlcv, macro


def _run(ohlcv, macro):
    members = default_swayam_members((8, 14, 22))
    views = build_swayam_frames(ohlcv, macro, list(macro.columns), members=members,
                                regime_sensitivity=8.0, base_weight=0.0, num_vars=4)
    w = view_skill_weights(views, horizon=10)
    return views, w, aggregate_views(views, weights=w)


def test_swayam_never_rewrites_history():
    ohlcv, macro = _data()
    views, w, daily = _run(ohlcv, macro)
    cut = 200
    views_p, w_p, daily_p = _run(ohlcv.iloc[:-cut], macro.iloc[:-cut])
    assert len(views) == len(views_p) > 5
    for name, part in views_p.items():
        full = views[name].loc[part.index]
        for col in ("Unified_Osc", "MSF_Osc", "MMR_Osc"):
            assert np.array_equal(full[col].to_numpy(), part[col].to_numpy(), equal_nan=True), (name, col)
        assert (full["Condition"] == part["Condition"]).all(), name
    assert np.array_equal(w.loc[w_p.index].to_numpy(), w_p.to_numpy())
    for col in ("Oversold_Pct", "Overbought_Pct", "Avg_Signal"):
        assert np.allclose(daily.loc[daily_p.index, col].to_numpy(), daily_p[col].to_numpy(),
                           rtol=0, atol=1e-12), col


def test_views_carry_only_their_own_columns():
    ohlcv, macro = _data()
    views, _w, _d = _run(ohlcv, macro)
    for name, df in views.items():
        assert not set(macro.columns) & set(df.columns), name
        assert df.shape[1] < 40


def test_mmr_reads_a_run_beyond_its_drivers():
    """A target that outruns what its drivers imply reads stretched up, and a copy of
    the target among the candidates is never used to explain it."""
    ohlcv, macro = _data()
    df = ohlcv.join(macro)
    df["Copy"] = df["Close"] * 3.0
    bump = np.ones(T)
    bump[700:715] = np.exp(np.linspace(0, 0.12, 15))
    bump[715:] = np.exp(0.12)
    df["Close"] = df["Close"] * bump                    # an idiosyncratic 12% run
    sig, drivers, _q = calculate_mmr(df, 20, 4, list(macro.columns) + ["Copy"])
    assert sig.iloc[706:716].mean() > 0.3
    sig2, drivers2, _ = calculate_mmr(ohlcv.join(macro).assign(Copy=lambda d: d["Close"] * 3.0),
                                      20, 4, list(macro.columns) + ["Copy"])
    assert "Copy" not in {d["Symbol"] for d in drivers2}
    assert sig2.abs().mean() > 0.05                     # the copy did not silence the read
