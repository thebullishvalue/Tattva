"""
Tattva v2.0.0 — Data tab: Merged data table + CSV export with search and filtering.
तत्त्व (Tattva) — "Principle / Essence"

UI — Raw data inspection: unified dataset viewer with export capability.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime

from ui.components import render_section_header


def render_data_tab(ts_filtered, ts, active_target):
    """Data Table: Time series data with search, column visibility, and export."""

    render_section_header(
        f"Time Series Data ({len(ts_filtered)} observations)",
        icon="database",
        accent="cyan",
    )

    display_cols = [
        "Date", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
        "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "Regime",
        "BullishDiv", "BearishDiv",
    ]
    display_cols = [c for c in display_cols if c in ts_filtered.columns]
    display_df = ts_filtered[display_cols].copy()

    rounding = {
        "AvgZ": 3, "ModelSpread": 3, "FairValue": 2,
        "Residual": 1, "ConvictionScore": 1, "OversoldBreadth": 1, "OverboughtBreadth": 1,
    }
    for col, decimals in rounding.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].round(decimals)

    if "BullishDiv" in display_df.columns:
        display_df["BullishDiv"] = display_df["BullishDiv"].apply(lambda x: "●" if x else "○")
    if "BearishDiv" in display_df.columns:
        display_df["BearishDiv"] = display_df["BearishDiv"].apply(lambda x: "●" if x else "○")

    # ── Search/filter ───────────────────────────────────────────────────
    search_col1, search_col2 = st.columns([1, 4], gap="small")
    with search_col1:
        date_range_option = st.selectbox("Date Range", ["All", "Last 30", "Last 90", "Last 180", "Last 365"], key="data_date_range")
    with search_col2:
        search_term = st.text_input("Search", placeholder="Filter by regime or value...", key="data_search")

    # Apply date range filter
    filtered_df = display_df.copy()
    if date_range_option != "All" and "Date" in filtered_df.columns:
        try:
            from pandas import DateOffset
            max_date = pd.to_datetime(filtered_df["Date"]).max()
            offsets = {"Last 30": 30, "Last 90": 90, "Last 180": 180, "Last 365": 365}
            cutoff = max_date - pd.Timedelta(days=offsets[date_range_option])
            filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])
            filtered_df = filtered_df[filtered_df["Date"] >= cutoff]
        except Exception:
            pass

    # Apply text search
    if search_term:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False),
        ).any(axis=1)
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, width='stretch', height=520)

    # ── Export section ──────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:var(--sp-4);"></div>',
        unsafe_allow_html=True,
    )
    csv_data = ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "\u2913  Download Full CSV",
        csv_data,
        f"tattva_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
