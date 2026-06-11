"""
Tattva v2.0.0 — Data schema contracts: dataclasses defining unified data structures.
तत्त्व (Tattva) — "Principle / Essence"

DATA — Only the dataclasses actually consumed by the application are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UnifiedDataset:
    """Unified dataset combining Aarambh and Nirnay data sources.

    This is the primary data structure passed through the convergence
    pipeline. It holds both the Aarambh fair-value time-series and
    the Nirnay constituent-level results, aligned on a common date index.

    Attributes
    ----------
    date_index : pd.DatetimeIndex
        Calendar dates from the Aarambh Google Sheet.
    nifty50_pe : np.ndarray
        Nifty 50 PE ratio series — the Aarambh target variable.
    aarambh_predictors : pd.DataFrame
        Macro and breadth predictors from Google Sheets.
    constituent_ohlcv : dict[str, pd.DataFrame]
        Per-constituent OHLCV data from yfinance, keyed by symbol.
    macro_df : pd.DataFrame
        Combined macro indicators (Yahoo Finance + bond yields) for
        Nirnay MMR regression.
    trading_days : list[pd.Timestamp]
        Trading day timestamps from the date index.
    """

    date_index: pd.DatetimeIndex
    nifty50_pe: np.ndarray
    aarambh_predictors: pd.DataFrame
    constituent_ohlcv: dict[str, pd.DataFrame]
    macro_df: pd.DataFrame
    trading_days: list[pd.Timestamp]
