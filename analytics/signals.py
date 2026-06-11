"""
Tattva v2.0.0 — Signal generators: Market Strength Factor (MSF) and Macro-Micro Regime (MMR).
तत्त्व (Tattva) — "Principle / Essence"

ANALYTICS — Per-constituent signal computation with ATR normalization and z-score clipping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.utils import sigmoid, zscore_clipped, calculate_atr


class MSFCalculator:
    """Market Strength Factor — multi-component signal generator.

    Combines four orthogonal components:
    - **Momentum**: Rate of change z-score
    - **Microstructure**: Volume-weighted direction vs impact
    - **Trend**: Multi-timeframe composite
    - **Flow**: Accumulation/distribution + regime counting

    Parameters
    ----------
    length : int
        Rolling window size.
    roc_len : int
        Rate of change period.
    """

    def __init__(self, length: int = 20, roc_len: int = 14) -> None:
        self.length = length
        self.roc_len = roc_len

    def calculate(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate MSF components.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        msf_signal : pd.Series
            Combined MSF signal [-1, 1].
        micro_norm : pd.Series
            Microstructure component.
        momentum_norm : pd.Series
            Momentum component.
        flow_norm : pd.Series
            Flow component.
        """
        close = df["Close"] if "Close" in df.columns else df["close"]
        high = df["High"] if "High" in df.columns else df["high"]
        low = df["Low"] if "Low" in df.columns else df["low"]
        open_price = df["Open"] if "Open" in df.columns else df["open"]
        volume = df["Volume"] if "Volume" in df.columns else df["volume"]

        # Momentum
        roc_raw = close.pct_change(self.roc_len, fill_method=None)
        roc_z = zscore_clipped(roc_raw, self.length, 3.0)
        momentum_norm = sigmoid(roc_z, 1.5)

        # Microstructure
        intrabar_dir = (high + low) / 2 - open_price
        vol_ma = volume.rolling(self.length).mean()
        vol_ratio = (volume / vol_ma).fillna(1.0)
        vw_direction = (intrabar_dir * vol_ratio).rolling(self.length).mean()
        price_change_imp = close.diff(5)
        vw_impact = (price_change_imp * vol_ratio).rolling(self.length).mean()
        micro_raw = vw_direction - vw_impact
        micro_z = zscore_clipped(micro_raw, self.length, 3.0)
        micro_norm = sigmoid(micro_z, 1.5)

        # Trend
        trend_fast = close.rolling(5).mean()
        trend_slow = close.rolling(self.length).mean()
        trend_diff_z = zscore_clipped(trend_fast - trend_slow, self.length, 3.0)
        mom_accel_raw = close.diff(5).diff(5)
        mom_accel_z = zscore_clipped(mom_accel_raw, self.length, 3.0)
        atr = calculate_atr(df, 14)
        vol_adj_mom_raw = close.diff(5) / atr
        vol_adj_mom_z = zscore_clipped(vol_adj_mom_raw, self.length, 3.0)
        mean_rev_z = zscore_clipped(close - trend_slow, self.length, 3.0)
        composite_trend_z = (
            trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z
        ) / np.sqrt(4.0)
        composite_trend_norm = sigmoid(composite_trend_z, 1.5)

        # Flow
        typical_price = (high + low + close) / 3
        mf = typical_price * volume
        mf_pos = np.where(close > close.shift(1), mf, 0)
        mf_neg = np.where(close < close.shift(1), mf, 0)
        mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(self.length).mean()
        mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(self.length).mean()
        mf_total = mf_pos_smooth + mf_neg_smooth
        accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
        accum_ratio = accum_ratio.fillna(0.5)
        accum_norm = 2.0 * (accum_ratio - 0.5)
        pct_change = close.pct_change(fill_method=None)
        regime_signals = np.select(
            [pct_change > 0.0033, pct_change < -0.0033], [1, -1], default=0
        )
        regime_count = pd.Series(regime_signals, index=df.index).cumsum()
        regime_raw = regime_count - regime_count.rolling(self.length).mean()
        regime_z = zscore_clipped(regime_raw, self.length, 3.0)
        regime_norm = sigmoid(regime_z, 1.5)
        flow_norm = (accum_norm + regime_norm) / np.sqrt(2.0)

        # Combine
        osc_momentum = momentum_norm
        osc_structure = (micro_norm + composite_trend_norm) / np.sqrt(2.0)
        osc_flow = flow_norm
        msf_raw = (osc_momentum + osc_structure + osc_flow) / np.sqrt(3.0)
        msf_signal = sigmoid(msf_raw * np.sqrt(3.0), 1.0)

        return msf_signal, micro_norm, momentum_norm, flow_norm


class MMRCalculator:
    """Macro-Micro Regime — rolling R²-weighted macro regression.

    Finds the top correlated macro indicators, builds a weighted
    composite prediction, and measures price deviation from it.

    Parameters
    ----------
    length : int
        Rolling window size.
    num_vars : int
        Number of top macro drivers to use.
    """

    def __init__(self, length: int = 20, num_vars: int = 5) -> None:
        self.length = length
        self.num_vars = num_vars

    def calculate(
        self, df: pd.DataFrame, macro_columns: list[str]
    ) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
        """Calculate MMR signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Close and macro columns.
        macro_columns : list[str]
            Names of macro indicator columns.

        Returns
        -------
        mmr_signal : pd.Series
            MMR signal [-1, 1].
        driver_details : list[dict]
            Top macro drivers with correlation values.
        mmr_quality : pd.Series
            Model quality (√R²) per observation.
        """
        close = df["Close"] if "Close" in df.columns else df["close"]
        available_macros = [m for m in macro_columns if m in df.columns]

        if len(df) < self.length + 10 or not available_macros:
            return (
                pd.Series(0.0, index=df.index),
                [],
                pd.Series(0.0, index=df.index),
            )

        correlations = df[available_macros].corrwith(close).abs().sort_values(
            ascending=False
        )
        top_drivers = correlations.head(self.num_vars).index.tolist()

        preds: list[pd.Series] = []
        r2_sum: float | pd.Series = 0
        r2_sq_sum: float | pd.Series = 0
        y_mean = close.rolling(self.length).mean()
        y_std = close.rolling(self.length).std()
        driver_details: list[dict[str, Any]] = []

        for ticker in top_drivers:
            x = df[ticker]
            x_mean = x.rolling(self.length).mean()
            x_std = x.rolling(self.length).std()
            roll_corr = x.rolling(self.length).corr(close)
            slope = roll_corr * (y_std / x_std)
            intercept = y_mean - (slope * x_mean)
            pred = (slope * x) + intercept
            r2 = roll_corr**2
            preds.append(pred * r2)
            r2_sum += r2
            r2_sq_sum += r2**2
            driver_details.append({
                "Symbol": ticker,
                "Correlation": round(float(df[ticker].corr(close)), 4),
            })

        r2_sum = r2_sum.replace(0, np.nan)
        y_predicted = sum(preds) / r2_sum if preds else y_mean
        deviation = close - y_predicted
        mmr_z = zscore_clipped(deviation, self.length, 3.0)
        mmr_signal = sigmoid(mmr_z, 1.5)
        model_r2 = r2_sq_sum / r2_sum
        mmr_quality = np.sqrt(model_r2.fillna(0))

        return mmr_signal, driver_details, mmr_quality
