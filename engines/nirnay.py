"""
Tattva v2.0.0 — Nirnay Engine: Per-instrument MSF + MMR with regime intelligence.
तत्त्व (Tattva) — "Principle / Essence"

NIRNAY — Per-constituent MSF + MMR analysis with HMM/GARCH/CUSUM regime intelligence aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.regime import (
    AdaptiveKalmanFilter,
    AdaptiveHMM,
    GARCHDetector,
    CUSUMDetector,
)


# ─── Utility functions ───────────────────────────────────────────────────────


def _sigmoid(x: np.ndarray | float, scale: float = 1.0) -> np.ndarray | float:
    """Sigmoid transformation bounding to [-1, 1].

    Formula: ``2 / (1 + exp(-x/scale)) - 1`` (original Nirnay formula).
    """
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0


def _zscore_clipped(series: pd.Series, window: int, clip: float = 3.0) -> pd.Series:
    """Rolling causal z-score with outlier clipping. Uses shift(1) to prevent today's outlier from biasing the denominator."""
    series_filled = series.ffill().fillna(0)
    roll_mean = series_filled.rolling(window=window, min_periods=1).mean().shift(1).bfill()
    roll_std = series_filled.rolling(window=window, min_periods=1).std().shift(1).bfill()
    z = (series_filled - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)


def _calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Exponential moving average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ─── Market Strength Factor ──────────────────────────────────────────────────


def calculate_msf(
    df: pd.DataFrame,
    length: int = 20,
    roc_len: int = 14,
    clip: float = 3.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Market Strength Factor from OHLCV data.

    Combines four orthogonal components:
    - **Momentum**: Rate of change z-score
    - **Microstructure**: Volume-weighted direction vs impact
    - **Trend**: Multi-timeframe composite (trend diff + momentum accel
      + volume-adjusted momentum + mean reversion)
    - **Flow**: Accumulation/distribution ratio + regime counting

    Returns
    -------
    msf_signal, micro_norm, momentum_norm, accum_norm
    """
    close = df["Close"]

    # Momentum
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = _zscore_clipped(roc_raw, length, clip)
    momentum_norm = _sigmoid(roc_z, 1.5)

    # Microstructure
    intrabar_dir = (df["High"] + df["Low"]) / 2 - df["Open"]
    vol_ma = df["Volume"].rolling(length).mean()
    vol_ratio = (df["Volume"] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    micro_raw = vw_direction - vw_impact
    micro_z = _zscore_clipped(micro_raw, length, clip)
    micro_norm = _sigmoid(micro_z, 1.5)

    # Trend
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = _zscore_clipped(trend_fast - trend_slow, length, clip)
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = _zscore_clipped(mom_accel_raw, length, clip)
    atr = _calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = _zscore_clipped(vol_adj_mom_raw, length, clip)
    mean_rev_z = _zscore_clipped(close - trend_slow, length, clip)
    composite_trend_z = (
        trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z
    ) / np.sqrt(4.0)
    composite_trend_norm = _sigmoid(composite_trend_z, 1.5)

    # Flow
    typical_price = (df["High"] + df["Low"] + close) / 3
    mf = typical_price * df["Volume"]
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    pct_change = close.pct_change(fill_method=None)
    regime_signals = np.select(
        [pct_change > 0.0033, pct_change < -0.0033], [1, -1], default=0
    )
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = _zscore_clipped(regime_raw, length, clip)
    regime_norm = _sigmoid(regime_z, 1.5)

    # Combine
    osc_momentum = momentum_norm
    osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
    osc_flow = (accum_norm + regime_norm) / np.sqrt(2.0)
    msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
    msf_signal = _sigmoid(msf_raw * np.sqrt(3.0), 1.0)

    return msf_signal, micro_norm, momentum_norm, accum_norm


# ─── Macro-Micro Regime ──────────────────────────────────────────────────────


def calculate_mmr(
    df: pd.DataFrame,
    length: int = 20,
    num_vars: int = 5,
    macro_columns: list[str] | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
    """Calculate Macro-Micro Regime via rolling R²-weighted regression.

    Finds the top ``num_vars`` macro indicators most correlated with price,
    builds a weighted composite prediction, and measures the deviation of
    actual price from that prediction.

    Returns
    -------
    mmr_signal, driver_details, mmr_quality
    """
    if macro_columns is None:
        macro_columns = []
    available_macros = [v for v in macro_columns if v in df.columns]
    target = df["Close"]

    if len(df) < length + 10 or not available_macros:
        return (pd.Series(0.0, index=df.index), [], pd.Series(0.0, index=df.index))

    y_mean = target.rolling(length, min_periods=1).mean().shift(1).bfill()
    y_std = target.rolling(length, min_periods=1).std().shift(1).bfill()

    preds_list = []
    r2_list = []
    
    # Vectorized causal rolling computations
    for ticker in available_macros:
        x = df[ticker].ffill().fillna(0)
        x_mean = x.rolling(length, min_periods=1).mean().shift(1).bfill()
        x_std = x.rolling(length, min_periods=1).std().shift(1).bfill()
        
        # Pearson correlation shifted (only prior data used to estimate relationship)
        roll_corr = x.rolling(length, min_periods=length).corr(target).shift(1).bfill().fillna(0)
        slope = roll_corr * (y_std / x_std.replace(0, np.nan)).fillna(0)
        intercept = y_mean - (slope * x_mean)
        
        pred = (slope * x) + intercept
        r2 = roll_corr**2
        
        preds_list.append(pred)
        r2_list.append(r2)

    all_preds = pd.concat(preds_list, axis=1)
    all_r2 = pd.concat(r2_list, axis=1)
    
    # Causally select top `num_vars` drivers per row!
    all_preds_arr = all_preds.values
    all_r2_arr = all_r2.values
    
    n_rows = len(df)
    y_predicted = np.empty(n_rows, dtype=np.float64)
    model_r2_arr = np.empty(n_rows, dtype=np.float64)
    
    for i in range(n_rows):
        row_r2 = all_r2_arr[i]
        valid_mask = ~np.isnan(row_r2)
        if np.sum(valid_mask) < num_vars:
            y_predicted[i] = y_mean.iloc[i]
            model_r2_arr[i] = 0.0
            continue
            
        top_indices = np.argsort(row_r2[valid_mask])[-num_vars:]
        top_real_indices = np.where(valid_mask)[0][top_indices]
        
        r2_sel = row_r2[top_real_indices]
        preds_sel = all_preds_arr[i, top_real_indices]
        
        r2_sum = np.sum(r2_sel)
        if r2_sum > 1e-6:
            y_predicted[i] = np.sum(preds_sel * r2_sel) / r2_sum
            model_r2_arr[i] = np.sum(r2_sel**2) / r2_sum
        else:
            y_predicted[i] = y_mean.iloc[i]
            model_r2_arr[i] = 0.0

    deviation = target - pd.Series(y_predicted, index=df.index)
    mmr_z = _zscore_clipped(deviation, length, 3.0)
    mmr_signal = _sigmoid(mmr_z, 1.5)
    mmr_quality = pd.Series(np.sqrt(model_r2_arr), index=df.index).fillna(0)

    # For display purposes (not trading logic), get the trailing global top drivers
    driver_details = []
    if len(df) > length:
        trailing_corr = df[available_macros].iloc[-length:].corrwith(target.iloc[-length:]).abs().sort_values(ascending=False)
        for ticker in trailing_corr.head(num_vars).index:
            driver_details.append({
                "Symbol": ticker,
                "Correlation": round(float(trailing_corr[ticker]), 4),
            })

    return mmr_signal, driver_details, mmr_quality


# ─── Full Analysis Pipeline ──────────────────────────────────────────────────


def run_full_analysis(
    df: pd.DataFrame,
    length: int,
    roc_len: int,
    regime_sensitivity: float,
    base_weight: float,
    macro_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the complete Nirnay pipeline on a single stock DataFrame.

    Steps
    -----
    1. Calculate MSF and MMR signals
    2. Compute adaptive clarity-based weights
    3. Blend signals with agreement multiplier
    4. Classify conditions (Oversold/Overbought/Neutral)
    5. Run regime intelligence loop (Kalman → GARCH → HMM → CUSUM)
    """
    if macro_columns is None:
        macro_columns = []

    # Compute MSF + MMR as locals, then attach in a single concat. Inserting
    # the six columns one-by-one into the (wide, 100+ macro) frame triggers
    # pandas' "highly fragmented DataFrame" PerformanceWarning on every stock.
    msf, micro, momentum, flow = calculate_msf(df, length, roc_len)
    mmr, drivers, mmr_quality = calculate_mmr(
        df, length, num_vars=5, macro_columns=macro_columns
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "MSF": msf, "Micro": micro, "Momentum": momentum, "Flow": flow,
                    "MMR": mmr, "MMR_Quality": mmr_quality,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Adaptive weighting based on signal clarity
    msf_clarity = df["MSF"].abs()
    mmr_clarity = df["MMR"].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df["MMR_Quality"]).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001

    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum

    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum

    unified_signal = (msf_w_norm * df["MSF"]) + (mmr_w_norm * df["MMR"])

    # Agreement multiplier amplifies aligned signals, dampens conflicts
    agreement = df["MSF"] * df["MMR"]
    agree_strength = agreement.abs()
    multiplier = np.where(
        agreement > 0,
        1.0 + 0.2 * agree_strength,
        1.0 - 0.1 * agree_strength,
    )

    # Compute all derived columns as local arrays first, then join in ONE
    # block-build via pd.concat. Note: df.assign() also triggers the
    # PerformanceWarning on newer pandas because internally it does a per-kwarg
    # column insert loop. pd.concat with a fresh inner DataFrame builds the
    # twelve new columns as a single block and merges them in one operation.
    unified = np.asarray((unified_signal * multiplier).clip(-1.0, 1.0))
    unified_osc = unified * 10.0
    msf_osc = df["MSF"].to_numpy() * 10.0
    mmr_osc = df["MMR"].to_numpy() * 10.0
    close_arr = df["Close"].to_numpy()

    agreement_arr = agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement)
    strong_agreement = agreement_arr > 0.3
    buy_signal = strong_agreement & (unified_osc < -5)
    sell_signal = strong_agreement & (unified_osc > 5)

    # Divergence detection (shift(1) ↔ prepend NaN, drop last)
    prev_unified_osc = np.concatenate(([np.nan], unified_osc[:-1]))
    prev_close = np.concatenate(([np.nan], close_arr[:-1]))
    with np.errstate(invalid="ignore"):  # NaN comparisons → False, silently
        osc_rising = unified_osc > prev_unified_osc
        price_falling = close_arr < prev_close
        osc_falling = unified_osc < prev_unified_osc
        price_rising = close_arr > prev_close
    bullish_div = osc_rising & price_falling & (unified_osc < -5)
    bearish_div = osc_falling & price_rising & (unified_osc > 5)

    condition = np.where(
        unified_osc < -5,
        "Oversold",
        np.where(unified_osc > 5, "Overbought", "Neutral"),
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Unified": unified,
                    "Unified_Osc": unified_osc,
                    "MSF_Osc": msf_osc,
                    "MMR_Osc": mmr_osc,
                    "MSF_Weight": msf_w_norm,
                    "MMR_Weight": mmr_w_norm,
                    "Agreement": agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement),
                    "Buy_Signal": buy_signal,
                    "Sell_Signal": sell_signal,
                    "Bullish_Div": bullish_div,
                    "Bearish_Div": bearish_div,
                    "Condition": condition,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Regime intelligence loop
    hmm = AdaptiveHMM()
    garch = GARCHDetector()
    cusum = CUSUMDetector()
    kalman = AdaptiveKalmanFilter()

    regimes: list[str] = []
    hmm_bulls: list[float] = []
    hmm_bears: list[float] = []
    vol_regimes: list[str] = []
    change_points: list[bool] = []
    confidences: list[float] = []
    signal_history: list[float] = []

    unified_vals = df["Unified"].values

    for i in range(len(df)):
        sig = unified_vals[i] if not np.isnan(unified_vals[i]) else 0.0

        # Kalman smoothing
        filtered = kalman.update(sig)

        # GARCH volatility regime
        shock = sig - signal_history[-1] if signal_history else 0.0
        garch.update(shock)
        vol_regime, _ = garch.get_regime()

        # HMM state estimation
        hmm_probs = hmm.update(filtered)
        change = cusum.update(filtered)

        bull_p = hmm_probs["BULL"]
        bear_p = hmm_probs["BEAR"]

        if change:
            regime = "TRANSITION"
        elif bull_p > 0.6:
            regime = "BULL"
        elif bear_p > 0.6:
            regime = "BEAR"
        elif bull_p > 0.4:
            regime = "WEAK_BULL"
        elif bear_p > 0.4:
            regime = "WEAK_BEAR"
        else:
            regime = "NEUTRAL"

        regimes.append(regime)
        hmm_bulls.append(bull_p)
        hmm_bears.append(bear_p)
        vol_regimes.append(vol_regime)
        change_points.append(change)
        confidences.append(max(bull_p, bear_p, hmm_probs["NEUTRAL"]))
        signal_history.append(sig)

    # Join the six regime-intelligence columns as ONE block via pd.concat.
    # df.assign() also fragments under newer pandas because it inserts kwargs
    # one-by-one internally; pd.concat with a pre-built inner DataFrame avoids
    # that entirely (single block-build, single merge).
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Regime": regimes,
                    "HMM_Bull": hmm_bulls,
                    "HMM_Bear": hmm_bears,
                    "Vol_Regime": vol_regimes,
                    "Change_Point": change_points,
                    "Confidence": confidences,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    return df, drivers


# ─── Constituent Aggregation ─────────────────────────────────────────────────


def aggregate_constituent_timeseries(
    constituent_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-constituent Nirnay results into daily statistics.

    Produces the same column schema as the original Nirnay system:
    Oversold/Overbought counts and percentages, signal counts, regime
    distributions, and average oscillator values.
    """
    all_dates: set[pd.Timestamp] = set()
    for sym, df in constituent_results.items():
        all_dates.update(df.index)

    sorted_dates = sorted(all_dates)
    rows: list[dict[str, Any]] = []

    for date in sorted_dates:
        day_stats: dict[str, Any] = {
            "Date": date.date() if hasattr(date, "date") else date,
            "Oversold": 0,
            "Overbought": 0,
            "Neutral": 0,
            "Buy_Signals": 0,
            "Sell_Signals": 0,
            "Total_Analyzed": 0,
            "Avg_Signal": 0.0,
            "Signal_Sum": 0.0,
            "Bull_Div": 0,
            "Bear_Div": 0,
            "Regime_Bull": 0,
            "Regime_Bear": 0,
            "Regime_Neutral": 0,
            "Regime_Transition": 0,
            "Vol_High": 0,
            "Vol_Low": 0,
            "Change_Points": 0,
        }
        oscs: list[float] = []
        msf_oscs: list[float] = []
        mmr_oscs: list[float] = []
        hmm_bulls: list[float] = []
        hmm_bears: list[float] = []

        for sym, df in constituent_results.items():
            if date not in df.index:
                continue
            try:
                row = df.loc[date]
                day_stats["Total_Analyzed"] += 1
                day_stats["Signal_Sum"] += row.get("Unified_Osc", 0.0)

                cond = row.get("Condition", "Neutral")
                if cond == "Oversold":
                    day_stats["Oversold"] += 1
                elif cond == "Overbought":
                    day_stats["Overbought"] += 1
                else:
                    day_stats["Neutral"] += 1

                if row.get("Buy_Signal", False):
                    day_stats["Buy_Signals"] += 1
                if row.get("Sell_Signal", False):
                    day_stats["Sell_Signals"] += 1
                if row.get("Bullish_Div", False):
                    day_stats["Bull_Div"] += 1
                if row.get("Bearish_Div", False):
                    day_stats["Bear_Div"] += 1

                regime = row.get("Regime", "NEUTRAL")
                if "BULL" in regime:
                    day_stats["Regime_Bull"] += 1
                elif "BEAR" in regime:
                    day_stats["Regime_Bear"] += 1
                elif regime == "TRANSITION":
                    day_stats["Regime_Transition"] += 1
                else:
                    day_stats["Regime_Neutral"] += 1

                vol_regime = row.get("Vol_Regime", "NORMAL")
                if vol_regime in ("HIGH", "EXTREME"):
                    day_stats["Vol_High"] += 1
                elif vol_regime == "LOW":
                    day_stats["Vol_Low"] += 1

                if row.get("Change_Point", False):
                    day_stats["Change_Points"] += 1

                oscs.append(float(row.get("Unified_Osc", 0)))
                msf_oscs.append(float(row.get("MSF_Osc", 0)))
                mmr_oscs.append(float(row.get("MMR_Osc", 0)))
                hmm_bulls.append(float(row.get("HMM_Bull", 0.33)))
                hmm_bears.append(float(row.get("HMM_Bear", 0.33)))
            except Exception:
                pass

        n = day_stats["Total_Analyzed"]
        if n > 0:
            day_stats["Avg_Signal"] = day_stats["Signal_Sum"] / n
            day_stats["Oversold_Pct"] = day_stats["Oversold"] / n * 100
            day_stats["Overbought_Pct"] = day_stats["Overbought"] / n * 100
            day_stats["Neutral_Pct"] = day_stats["Neutral"] / n * 100
            day_stats["Regime_Bull_Pct"] = day_stats["Regime_Bull"] / n * 100
            day_stats["Regime_Bear_Pct"] = day_stats["Regime_Bear"] / n * 100
            day_stats["Vol_High_Pct"] = day_stats["Vol_High"] / n * 100

        day_stats["avg_hmm_bull"] = float(np.mean(hmm_bulls)) if hmm_bulls else 0.33
        day_stats["avg_hmm_bear"] = float(np.mean(hmm_bears)) if hmm_bears else 0.33
        day_stats["avg_msf_osc"] = float(np.mean(msf_oscs)) if msf_oscs else 0.0
        day_stats["avg_mmr_osc"] = float(np.mean(mmr_oscs)) if mmr_oscs else 0.0

        rows.append(day_stats)

    return pd.DataFrame(rows).set_index("Date")
