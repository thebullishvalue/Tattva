"""
Tattva — Data tab: every published series, one row per session, exportable.
तत्त्व (Tattva) — "Principle / Essence"

UI — the record behind every page, in the units every page shows it in: the
mispricing as a percentage and in σ (never a log gap rounded to one decimal,
which printed 0.0 on every row), model spread in basis points, the valuation
state word and Swayam's published lean (as net breadth, in points) joined on
the same date — and the call itself at every session (``call_ts``: what was
issued, why, and every number it was composed from). The export is the same
joined frame at full precision, so "the evidence, exportable" is literally
true rather than true of one engine.

Every signed column declares its orientation (``SIGN_ORIENTATION``): a
READING keeps its native sign — positive = stretched up, rich — and is tinted
by the lean it implies, so a rich gap reads in the short colour; a LEAN is
bullish-positive and tinted as it reads; EVIDENCE keeps its native sign in the
neutral ink — the ECM's gap-implied move, as on the Mūla page, because a lean
that is evidence carries its grade, not a direction's colour.

There is one date filter: the chart window, in the table's own header, as on
every other page. The second "Date Range" selector that used to sit beside it
filtered the already-windowed frame again, in calendar days of its own.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.semantics import valuation_state
from ui.components import (render_section_header, render_table_panel,
                           render_note, provisional_note)


def _by_date(frame: pd.DataFrame | None) -> pd.DataFrame:
    """A frame re-indexed on normalized dates (empty when absent)."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    f = frame[~frame.index.duplicated(keep="last")].copy()
    f.index = pd.to_datetime(f.index, errors="coerce").normalize()
    return f[~f.index.isna()]


#: Header prefix of the call's own columns (call_ts) in the record.
CALL_PREFIX = "Call · "

#: Mūla's published columns in the record, with the header each is shown under.
MULA_COLUMNS: dict[str, str] = {
    "Price": "Price", "FairValue": "Fair value", "FairValueLo": "Fair value lo (95%)",
    "FairValueHi": "Fair value hi (95%)", "PctMispricing": "Gap %", "FVO": "Oscillator σ",
    "GapScale": "Gap scale %", "ModelSpread": "Model spread bps", "GapPhi": "Gap persistence φ",
    "GapHalfLife": "Gap half-life (d)", "MRProb": "P(price-side κ > 0)",
    "MulaKappa": "κ̂ (price-side)", "MulaHalfLife": "Price-side half-life (d)",
    "MulaDriftPct": "Gap-implied move % (excl. drift)",
    "MulaSdPct": "Gap-implied move SD %", "KFactors": "Latent factors k",
    "Provisional": "Provisional", "Valid": "Valued",
}

#: Every signed column of the record and the orientation it is declared in
#: (U41, U44): ``'reading'`` keeps its native sign, positive = stretched up, and
#: is tinted by the lean it implies (positive in the short colour); ``'lean'`` is
#: bullish-positive and tinted as it reads; ``'evidence'`` is bullish-positive
#: and untinted — EVIDENCE→sizer:reversion, whose grade
#: (edges['evidence:ecm_drift']) has not earned a direction's colour (design §7).
#: No other column is sign-coloured.
SIGN_ORIENTATION: dict[str, str] = {
    "Gap %": "reading", "Oscillator σ": "reading", "Swayam net breadth": "lean",
    "Gap-implied move % (excl. drift)": "evidence",
}


def build_record(ts: pd.DataFrame) -> pd.DataFrame:
    """The joined, full-precision record: Mūla + Swayam + the call, per session."""
    if ts is None or ts.empty:
        return pd.DataFrame()
    m = ts.copy()
    if "Date" in m.columns:
        m.index = pd.to_datetime(m["Date"], errors="coerce").dt.normalize()
    out = pd.DataFrame(index=m.index)
    for src, name in MULA_COLUMNS.items():
        out[name] = m[src]
    # The state word, derived where it is shown (core.semantics), beside the σ.
    out.insert(out.columns.get_loc("Oscillator σ") + 1, "Valuation",
               [valuation_state(v) for v in pd.to_numeric(m["FVO"], errors="coerce")])
    for name, scale in (("Gap %", 100.0), ("Gap scale %", 100.0), ("Model spread bps", 1e4)):
        out[name] = pd.to_numeric(out[name], errors="coerce") * scale

    # Swayam's one published lean (engines.swayam.aggregate_views' Lean), in
    # points of net breadth, and the shares it is made of — never re-derived.
    sw = _by_date(st.session_state.get("swayam_daily"))
    if not sw.empty:
        out["Swayam net breadth"] = (sw["Lean"] * 100.0).reindex(out.index)
        out["Swayam oversold %"] = sw["Oversold_Pct"].reindex(out.index)
        out["Swayam overbought %"] = sw["Overbought_Pct"].reindex(out.index)
    # The call at every session, every column of call_ts (core.call.CALL_COLUMNS).
    ct = _by_date(st.session_state.get("call_ts"))
    for c in ct.columns:
        out[CALL_PREFIX + c] = ct[c].reindex(out.index)
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
        f"oscillator) — and their tint follows the lean they imply, so a rich gap reads in the "
        f"short colour. Swayam net breadth (its published lean in points, oversold minus "
        f"overbought) reads bullish-positive and is tinted as it reads. The gap-implied move "
        f"(excludes drift) is evidence: bullish-positive, untinted, graded on the Mūla page.",
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
            view[col] = view[col].round(4 if col in ("κ̂ (price-side)", "Gap persistence φ") else 2)
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
        col_labels={c: c for c in view.columns},     # the record's headers are already named
        sign_color_cols={c for c, o in SIGN_ORIENTATION.items() if o == "lean"},
        contrarian_cols={c for c, o in SIGN_ORIENTATION.items() if o == "reading"},
    )

    csv_data = record.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download the record (CSV)", csv_data,
        f"tattva_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )
    render_note(f"Full history · {len(record):,} sessions · Mūla, Swayam and the call joined "
                f"on date at full precision.")
