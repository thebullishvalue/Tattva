"""
Tattva — Data tab: every published series, one row per session, exportable.
तत्त्व (Tattva) — "Principle / Essence"

UI — the record behind every page, in the units every page shows it in: the
mispricing as a percentage and in σ (never a log gap rounded to one decimal,
which printed 0.0 on every row), model spread in basis points, the valuation
state word, Swayam's net breadth and the convergence readings joined on the
same date. The export is the same joined frame at full precision, so "the
evidence, exportable" is literally true rather than true of one engine.

There is one date filter: the chart window, in the table's own header, as on
every other page. The second "Date Range" selector that used to sit beside it
filtered the already-windowed frame again, in calendar days of its own.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (render_section_header, render_table_panel,
                           render_note, provisional_note)


def _by_date(frame: pd.DataFrame | None) -> pd.DataFrame:
    """A frame re-indexed on normalized dates (empty when absent)."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    f = frame[~frame.index.duplicated(keep="last")].copy()
    f.index = pd.to_datetime(f.index, errors="coerce").normalize()
    return f[~f.index.isna()]


def build_record(ts: pd.DataFrame) -> pd.DataFrame:
    """The joined, full-precision record: Mūla + Swayam + Convergence per session."""
    if ts is None or ts.empty:
        return pd.DataFrame()
    m = ts.copy()
    if "Date" in m.columns:
        m.index = pd.to_datetime(m["Date"], errors="coerce").dt.normalize()
    cols = {
        "Price": "Price", "FairValue": "Fair value", "FairValueLo": "Fair value lo (95%)",
        "FairValueHi": "Fair value hi (95%)", "PctMispricing": "Gap %", "FVO": "Oscillator σ",
        "Regime": "Valuation", "ModelSpread": "Model spread bps", "XSConsistency": "Cross-section",
        "MRProb": "P(price reverts)", "MulaKappa": "κ̂ (price-side)",
        "MulaDriftPct": "Expected move %", "MulaSdPct": "Expected move SD %",
        "OversoldBreadth": "Windows cheap %", "OverboughtBreadth": "Windows rich %",
        "ConvictionBounded": "Window agreement", "Provisional": "Provisional", "Valid": "Valued",
    }
    out = pd.DataFrame(index=m.index)
    for src, name in cols.items():
        if src in m.columns:
            out[name] = m[src]
    if "Gap %" in out.columns:
        out["Gap %"] = pd.to_numeric(out["Gap %"], errors="coerce") * 100.0
    if "Model spread bps" in out.columns:
        out["Model spread bps"] = pd.to_numeric(out["Model spread bps"], errors="coerce") * 1e4

    sw = _by_date(st.session_state.get("swayam_daily"))
    if not sw.empty:
        if {"Oversold_Pct", "Overbought_Pct"} <= set(sw.columns):
            out["Swayam net breadth"] = (sw["Oversold_Pct"] - sw["Overbought_Pct"]).reindex(out.index)
            out["Swayam oversold %"] = sw["Oversold_Pct"].reindex(out.index)
            out["Swayam overbought %"] = sw["Overbought_Pct"].reindex(out.index)
        if "Avg_Signal" in sw.columns:
            out["Swayam avg signal"] = sw["Avg_Signal"].reindex(out.index)

    cv = _by_date(st.session_state.get("convergence_df"))
    if not cv.empty:
        for src, name in (("agreement_ratio", "Agreement"),
                          ("convergence_score", "Weighted composite")):
            if src in cv.columns:
                out[name] = cv[src].reindex(out.index)
    cons = st.session_state.get("hero_series")
    if cons is not None and len(cons):
        c = cons.copy()
        c.index = pd.to_datetime(c.index, errors="coerce").normalize()
        out["Consensus"] = c[~c.index.duplicated(keep="last")].reindex(out.index)
    out.index.name = "Date"
    return out


def render_data_tab(ts_filtered, ts, active_target):
    """The record: windowed on screen, full in the export."""
    record = build_record(ts)
    if record.empty:
        render_note("No published record for this run.")
        return
    shown = record
    if ts_filtered is not None and not ts_filtered.empty and "Date" in ts_filtered.columns:
        first = pd.to_datetime(ts_filtered["Date"], errors="coerce").min()
        if pd.notna(first):
            shown = record[record.index >= first.normalize()]

    render_section_header(
        "The Record",
        f"Every published series for {active_target}, one row per session, in the units the "
        f"pages show them in. Readings keep their native sign — positive = stretched up (gap, "
        f"oscillator, Swayam signal, consensus, composite) — except Swayam net breadth, which "
        f"is oversold minus overbought and so reads positive when the views lean bullish.",
        icon="database", accent="cyan",
    )
    _pn = provisional_note(st.session_state.get("fvo_ts"))
    if _pn:
        render_note(_pn)

    with st.container(key="filterbar"):
        search_term = st.text_input("Search", placeholder="Filter by valuation word or value…",
                                    key="data_search")
    view = shown.iloc[::-1].copy()                 # newest first
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(4 if col in ("κ̂ (price-side)",) else 2)
    if search_term:
        mask = view.astype(str).apply(
            lambda c: c.str.contains(search_term, case=False, na=False)).any(axis=1)
        view = view[mask]

    # HTML tables render every row into the DOM, so the screen is capped; the
    # export below carries every row at full precision.
    _CAP = 300
    render_table_panel(
        view, "data-record", index_label="Date",
        units=f"{len(view):,} sessions × {len(view.columns)} series",
        max_rows=_CAP, max_height=560, window=True,
        footer=(f"Showing the newest {_CAP:,} of {len(view):,} — the export has all of them."
                if len(view) > _CAP else ""),
        sign_color_cols={"Gap %", "Oscillator σ", "Expected move %", "Swayam net breadth"},
    )

    csv_data = record.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download the record (CSV)", csv_data,
        f"tattva_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )
    render_note(f"Full history · {len(record):,} sessions · Mūla, Swayam and Convergence "
                f"joined on date at full precision.")
