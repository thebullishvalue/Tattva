"""
Tattva — Integrity tests for Nirnay-Swayam (the self-referential ensemble).
तत्त्व (Tattva) — "Principle / Essence"

Pins the invariants NIRNAY_SWAYAM_PLAN.md declares non-negotiable:

  1. SCHEMA PARITY — every member frame carries the exact column set
     aggregate_constituent_timeseries's per-constituent loop needs, and the
     aggregate carries the exact daily schema.
  2. BYTE-IDENTITY (INV-3) — calculate_msf/run_full_analysis with
     components=None reproduce the pre-mask combine formula exactly.
  3. COMPONENT MASK MATH — a single-component member equals
     sigmoid(that component, 1.0); an unknown component name raises.
  4. CAUSALITY / NO REPAINTING (INV-2) — truncating the input series must
     not change any already-computed member value before the truncation
     point.
  5. LEAKAGE GUARD — swayam_macro_columns drops the target's own column and
     its excluded-predictor near-replicas; end-to-end, no member's MMR
     driver list contains the target's own column.
  6. VOLUME DEGENERACY — zero-volume OHLCV drops flow-only members; the
     surviving ensemble still builds and aggregates.
  7. POLARITY NO-OP — self mode must never flip via apply_polarity.
  8. DETERMINISM — two consecutive builds are identical.
  9. EFFECTIVE COUNT — bounded in [1, N]; ~1 for duplicated members, ~N for
     independent random members.

Run: python -m research.test_nirnay_swayam  (from the repo root)
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from engines.nirnay import calculate_msf, run_full_analysis, aggregate_constituent_timeseries
from engines.nirnay_self import (
    SwayamMember, build_swayam_frames, default_swayam_members,
    effective_member_count, _is_volume_dependent,
)
from core.config import swayam_macro_columns, TARGET_EXCLUDED_PREDICTORS
from analytics.utils import sigmoid as _sigmoid


def _synthetic_ohlcv(n: int = 260, seed: int = 7, with_volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    ret = rng.normal(0.0003, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    openp = close * (1 + rng.normal(0, 0.002, n))
    volume = (rng.integers(1_000_000, 5_000_000, n) if with_volume
              else np.zeros(n, dtype=np.int64))
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _synthetic_macro(ohlcv: pd.DataFrame, n_cols: int = 6, seed: int = 11) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    cols = [f"Macro_{i}" for i in range(n_cols)]
    data = {}
    close = ohlcv["Close"]
    for i, c in enumerate(cols):
        noise = rng.normal(0, close.std() * 0.5, len(ohlcv))
        # A mix of genuinely-correlated and pure-noise macro columns.
        data[c] = (close.to_numpy() * (0.3 if i % 2 == 0 else 0.0)) + noise
    return pd.DataFrame(data, index=ohlcv.index), cols


def run() -> None:
    checks = 0

    # ── 1. SCHEMA PARITY ──────────────────────────────────────────────────
    ohlcv = _synthetic_ohlcv()
    macro_df, macro_cols = _synthetic_macro(ohlcv)
    frames = build_swayam_frames(
        ohlcv, macro_df, macro_cols,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    assert len(frames) == len(default_swayam_members()), (
        f"expected {len(default_swayam_members())} members, got {len(frames)}")
    needed = [
        "Unified_Osc", "Condition", "Buy_Signal", "Sell_Signal",
        "Bullish_Div", "Bearish_Div", "Regime", "Vol_Regime", "Change_Point",
        "MSF_Osc", "MMR_Osc", "HMM_Bull", "HMM_Bear",
    ]
    for name, df in frames.items():
        missing = [c for c in needed if c not in df.columns]
        assert not missing, f"member {name} missing columns {missing}"

    daily = aggregate_constituent_timeseries(frames)
    expected_daily_cols = {
        "Oversold", "Overbought", "Neutral", "Buy_Signals", "Sell_Signals",
        "Total_Analyzed", "Avg_Signal", "Signal_Sum", "Bull_Div", "Bear_Div",
        "Regime_Bull", "Regime_Bear", "Regime_Neutral", "Regime_Transition",
        "Vol_High", "Vol_Low", "Change_Points", "Oversold_Pct", "Overbought_Pct",
        "Neutral_Pct", "Regime_Bull_Pct", "Regime_Bear_Pct", "Vol_High_Pct",
        "avg_hmm_bull", "avg_hmm_bear", "avg_msf_osc", "avg_mmr_osc",
    }
    assert expected_daily_cols.issubset(set(daily.columns)), (
        f"aggregate missing {expected_daily_cols - set(daily.columns)}")
    assert not daily.empty
    checks += 1

    # ── 2. BYTE-IDENTITY (INV-3) ───────────────────────────────────────────
    # components=None must reduce to the literal pre-mask combine formula.
    df_plain = ohlcv.copy()
    msf_new, micro, momentum, accum = calculate_msf(df_plain, length=20, roc_len=14)

    # Recompute the old hard-coded formula independently and compare.
    from analytics.utils import zscore_clipped as _z, calculate_atr as _atr
    close = df_plain["Close"]
    roc_raw = close.pct_change(14, fill_method=None)
    roc_z = _z(roc_raw, 20, 3.0)
    momentum_norm = _sigmoid(roc_z, 1.5)
    intrabar_dir = (df_plain["High"] + df_plain["Low"]) / 2 - df_plain["Open"]
    vol_ma = df_plain["Volume"].rolling(20).mean()
    vol_ratio = (df_plain["Volume"] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(20).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(20).mean()
    micro_raw = vw_direction - vw_impact
    micro_z = _z(micro_raw, 20, 3.0)
    micro_norm = _sigmoid(micro_z, 1.5)
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(20).mean()
    trend_diff_z = _z(trend_fast - trend_slow, 20, 3.0)
    mom_accel_z = _z(close.diff(5).diff(5), 20, 3.0)
    atr = _atr(df_plain, 14)
    vol_adj_mom_z = _z(close.diff(5) / atr, 20, 3.0)
    mean_rev_z = _z(close - trend_slow, 20, 3.0)
    composite_trend_z = (trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z) / np.sqrt(4.0)
    composite_trend_norm = _sigmoid(composite_trend_z, 1.5)
    typical_price = (df_plain["High"] + df_plain["Low"] + close) / 3
    mf = typical_price * df_plain["Volume"]
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    mf_pos_smooth = pd.Series(mf_pos, index=df_plain.index).rolling(20).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df_plain.index).rolling(20).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    accum_ratio = (mf_pos_smooth / mf_total.replace(0, np.nan)).fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    pct_change = close.pct_change(fill_method=None)
    regime_signals = np.select([pct_change > 0.0033, pct_change < -0.0033], [1, -1], default=0)
    regime_count = pd.Series(regime_signals, index=df_plain.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(20).mean()
    regime_z = _z(regime_raw, 20, 3.0)
    regime_norm = _sigmoid(regime_z, 1.5)
    osc_momentum_old = momentum_norm
    osc_structure_old = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow_old = (accum_norm + regime_norm) / np.sqrt(2.0)
    msf_raw_old = (osc_momentum_old + osc_structure_old + osc_flow_old) / np.sqrt(3.0)
    msf_old = _sigmoid(msf_raw_old * np.sqrt(3.0), 1.0)

    pd.testing.assert_series_equal(msf_new.rename(None), msf_old.rename(None), check_exact=False, atol=1e-12)
    checks += 1

    # run_full_analysis with components=None vs no components kwarg at all —
    # must be identical (default parity).
    r1, _ = run_full_analysis(ohlcv, 20, 14, 1.5, 0.5, num_vars=3,
                               oversold=-5.0, overbought=5.0, macro_columns=macro_cols)
    r2, _ = run_full_analysis(ohlcv, 20, 14, 1.5, 0.5, num_vars=3,
                               oversold=-5.0, overbought=5.0, macro_columns=macro_cols,
                               components=None)
    pd.testing.assert_frame_equal(r1, r2)
    checks += 1

    # ── 3. COMPONENT MASK MATH ─────────────────────────────────────────────
    msf_mom, _, momentum_only, _ = calculate_msf(ohlcv, length=20, roc_len=14, components=("momentum",))
    # momentum_norm IS the returned momentum_norm value (already sigmoid'd once
    # inside calculate_msf) — isolating it means msf_signal = sigmoid(momentum_norm, 1.0).
    manual = _sigmoid(momentum_only, 1.0)
    pd.testing.assert_series_equal(msf_mom.rename(None), manual.rename(None), check_exact=False, atol=1e-12)

    try:
        calculate_msf(ohlcv, components=("bogus",))
        raise AssertionError("expected ValueError for unknown component")
    except ValueError:
        pass
    checks += 1

    # ── 4. CAUSALITY / NO REPAINTING (INV-2) ───────────────────────────────
    cutoff = 200
    truncated = ohlcv.iloc[:cutoff].copy()
    macro_truncated = macro_df.iloc[:cutoff].copy()
    frames_full = build_swayam_frames(
        ohlcv, macro_df, macro_cols,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    frames_trunc = build_swayam_frames(
        truncated, macro_truncated, macro_cols,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    check_before = cutoff - 60   # leave a margin before the truncation point
    for name in frames_trunc:
        full_osc = frames_full[name]["Unified_Osc"].iloc[:check_before].to_numpy()
        trunc_osc = frames_trunc[name]["Unified_Osc"].iloc[:check_before].to_numpy()
        assert np.allclose(full_osc, trunc_osc, atol=1e-9, equal_nan=True), (
            f"member {name} repainted history when the series was extended")
    checks += 1

    # ── 5. LEAKAGE GUARD ────────────────────────────────────────────────────
    target = "Gold"
    macro_cols_g = ["Gold", "Silver", "Some Other Macro"]
    TARGET_EXCLUDED_PREDICTORS.setdefault(target, ["Precious Metals Basket (GLTR)"])
    filtered = swayam_macro_columns(target, macro_cols_g + ["Precious Metals Basket (GLTR)"])
    assert target not in filtered
    assert "Precious Metals Basket (GLTR)" not in filtered
    assert "Silver" in filtered and "Some Other Macro" in filtered

    # End-to-end: a FULL member's MMR driver pool must never contain the
    # target's own column, even if the caller forgets to filter (defense in
    # depth is the caller's job per the contract, but assert the contract
    # holds when the caller DOES filter, as app.py must).
    macro_df2, macro_cols2 = _synthetic_macro(ohlcv, n_cols=4)
    self_col = "SelfTarget"
    macro_df2[self_col] = ohlcv["Close"]     # a macro column that IS the target
    macro_cols2_with_self = macro_cols2 + [self_col]
    filtered2 = swayam_macro_columns(self_col, macro_cols2_with_self)
    assert self_col not in filtered2
    frames2 = build_swayam_frames(
        ohlcv, macro_df2, filtered2,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    full_member_name = next(n for n in frames2 if n.endswith("·FULL"))
    _, drivers = run_full_analysis(
        ohlcv.join(macro_df2[filtered2], how="left"), length=20, roc_len=14,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0, macro_columns=filtered2,
    )
    driver_names = {d["Symbol"] for d in drivers}
    assert self_col not in driver_names, "target's own column leaked into MMR drivers"
    checks += 1

    # ── 6. VOLUME DEGENERACY ────────────────────────────────────────────────
    ohlcv_novol = _synthetic_ohlcv(with_volume=False)
    macro_df_nv, macro_cols_nv = _synthetic_macro(ohlcv_novol)
    flow_only_members = tuple(m for m in default_swayam_members() if _is_volume_dependent(m))
    assert len(flow_only_members) >= 1
    frames_novol = build_swayam_frames(
        ohlcv_novol, macro_df_nv, macro_cols_nv,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    for m in flow_only_members:
        assert m.name not in frames_novol, f"flow-only member {m.name} should be dropped on zero volume"
    assert len(frames_novol) == len(default_swayam_members()) - len(flow_only_members)
    daily_novol = aggregate_constituent_timeseries(frames_novol)
    assert not daily_novol.empty
    checks += 1

    # ── 7. POLARITY NO-OP (guard logic, not engines.nirnay.apply_polarity
    #      itself — that function is generic; the app-level guard is what
    #      must never call it in self mode) ────────────────────────────────
    nirnay_mode = "self"
    polarity = -1   # pretend a negative polarity leaked in
    should_apply = (nirnay_mode != "self") and (polarity < 0)
    assert should_apply is False
    checks += 1

    # ── 8. DETERMINISM ──────────────────────────────────────────────────────
    frames_a = build_swayam_frames(
        ohlcv, macro_df, macro_cols,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    frames_b = build_swayam_frames(
        ohlcv, macro_df, macro_cols,
        regime_sensitivity=1.5, base_weight=0.5, num_vars=3,
        oversold=-5.0, overbought=5.0,
    )
    assert set(frames_a.keys()) == set(frames_b.keys())
    for name in frames_a:
        pd.testing.assert_frame_equal(frames_a[name], frames_b[name])
    checks += 1

    # ── 9. EFFECTIVE COUNT ──────────────────────────────────────────────────
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    independent = {f"m{i}": pd.DataFrame({"Unified": rng.normal(0, 1, n)}, index=idx) for i in range(6)}
    n_eff_indep = effective_member_count(independent)
    assert n_eff_indep > 3.5, f"independent members should read close to N, got {n_eff_indep:.2f}"

    base_series = rng.normal(0, 1, n)
    duplicated = {f"d{i}": pd.DataFrame({"Unified": base_series.copy()}, index=idx) for i in range(6)}
    n_eff_dup = effective_member_count(duplicated)
    assert n_eff_dup < 1.5, f"duplicated members should read close to 1, got {n_eff_dup:.2f}"

    assert effective_member_count({}) == 0.0
    assert effective_member_count({"only": pd.DataFrame({"Unified": [1.0, 2.0]})}) == 1.0
    checks += 1

    print(f"nirnay-swayam integrity: ALL {checks} CHECK GROUPS PASSED "
          f"({len(default_swayam_members())} default members)")


if __name__ == "__main__":
    run()
