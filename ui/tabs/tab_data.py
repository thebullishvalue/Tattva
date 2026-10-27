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

Every column is registered to a card (``COLUMN_CARDS`` and
:func:`column_card`, cards of ``core.roles.CARDS``), and its header carries
that card's role tag — what the column is FOR: the claim, a precondition, a
sizer, a grade, evidence. Every signed column declares its orientation
(``SIGN_ORIENTATION``): a READING of the claim's state keeps its native sign —
positive = stretched up, rich — and is tinted by the lean it implies, so a
rich gap reads in the short colour; the call's own signed weight and
direction are LEANS, bullish-positive and tinted as they read; EVIDENCE and
the sizers' readings keep their native sign in the neutral ink, because
directional colour is the claim's and the call's alone.

There is one date filter: the chart window, in the table's own header, as on
every other page. The second "Date Range" selector that used to sit beside it
filtered the already-windowed frame again, in calendar days of its own.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.roles import role_tag
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

#: Swayam's columns in the record: its published lean, in points of net
#: breadth, and the shares of views it is made of.
SWAYAM_COLUMNS: dict[str, str] = {
    "Lean": "Swayam net breadth (pts)", "Oversold_Pct": "Swayam oversold (% of views)",
    "Overbought_Pct": "Swayam overbought (% of views)",
}

#: Every signed column of the record and the orientation it is declared in
#: (U41, U44): ``'reading'`` is the claim's state, keeps its native sign,
#: positive = stretched up, and is tinted by the lean it implies (positive in
#: the short colour); ``'lean'`` is the call's own (its signed weight and its
#: direction), bullish-positive and tinted as it reads; ``'evidence'`` keeps
#: its native sign untinted — a sizer's reading or EVIDENCE, which has not
#: earned a direction's colour (design §11). No other column is sign-coloured.
SIGN_ORIENTATION: dict[str, str] = {
    "Gap %": "reading", "Oscillator σ": "reading",
    CALL_PREFIX + "w": "lean", CALL_PREFIX + "d": "lean",
    "Swayam net breadth (pts)": "evidence", "Gap-implied move % (excl. drift)": "evidence",
}

#: The card every Mūla and Swayam column of the record is registered to.
COLUMN_CARDS: dict[str, str] = {
    "Price": "data:price", "Fair value": "data:fair_value", "Fair value lo (95%)": "data:fair_value",
    "Fair value hi (95%)": "data:fair_value", "Gap %": "data:claim", "Oscillator σ": "data:claim",
    "Valuation": "data:claim", "Gap scale %": "data:claim", "Model spread bps": "data:model_fit",
    "Gap persistence φ": "data:gap", "Gap half-life (d)": "data:gap",
    "P(price-side κ > 0)": "data:reversion", "κ̂ (price-side)": "data:reversion",
    "Price-side half-life (d)": "data:gap", "Gap-implied move % (excl. drift)": "data:ecm_move",
    "Gap-implied move SD %": "data:ecm_move", "Latent factors k": "data:model_fit",
    "Provisional": "data:published", "Valued": "data:published",
    "Swayam net breadth (pts)": "data:swayam", "Swayam oversold (% of views)": "data:swayam_shares",
    "Swayam overbought (% of views)": "data:swayam_shares",
}

#: The card of each call_ts column (core.call.CALL_COLUMNS) that no prefix rule
#: in :func:`column_card` covers.
_CALL_CARDS: dict[str, str] = {
    "published": "data:published", "valid": "data:published", "provisional": "data:published",
    "withheld_reason": "data:published", "printed": "data:printed", "behind": "data:printed",
    "precondition": "data:precondition", "fvo": "data:claim", "pct": "data:claim",
    "d": "data:claim", "trust": "data:trust", "mr_prob": "data:reversion",
    "kappa": "data:reversion", "size": "data:size", "w": "data:call", "action": "data:call",
    "reason": "data:call", "binding": "data:call", "grid_date": "data:claim_grade",
}


def column_card(column: str) -> str:
    """The registered card of a record column: what the column is FOR.

    Mūla's and Swayam's columns are listed in ``COLUMN_CARDS``; a call_ts
    column (``CALL_PREFIX`` + name) is the claim's grade (``claim_*``), a
    sizer's block (``swayam_*``, ``reversion_*``) or listed in
    ``_CALL_CARDS``. An unregistered column raises: the record shows nothing
    whose role is undeclared.
    """
    if column in COLUMN_CARDS:
        return COLUMN_CARDS[column]
    if column.startswith(CALL_PREFIX):
        c = column[len(CALL_PREFIX):]
        if c in _CALL_CARDS:
            return _CALL_CARDS[c]
        for prefix, card in (("claim_", "data:claim_grade"), ("swayam_", "data:swayam"),
                             ("reversion_", "data:reversion")):
            if c.startswith(prefix):
                return card
    raise KeyError(f"record column {column!r} is registered to no card")


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
        for src, name in SWAYAM_COLUMNS.items():
            out[name] = sw[src].reindex(out.index) * (100.0 if src == "Lean" else 1.0)
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
        f"pages show them in, each header tagged with the role its column plays. The claim's "
        f"readings keep their native sign — positive = stretched up (gap, oscillator) — and "
        f"their tint follows the lean they imply, so a rich gap reads in the short colour; "
        f"call_ts's signed weight w and direction d are tinted as they read. Everything else — Swayam's "
        f"net breadth (its published lean in points), the gap-implied move (excludes drift) — "
        f"keeps its native sign in the neutral ink.",
        icon="database", accent="cyan", card="data:record",
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
        col_tags={c: role_tag(column_card(c)) for c in view.columns},
        sign_color_cols={c for c, o in SIGN_ORIENTATION.items() if o == "lean"},
        contrarian_cols={c for c, o in SIGN_ORIENTATION.items() if o == "reading"},
    )

    csv_data = record.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download the record (CSV)", csv_data,
        f"tattva_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )
    render_note(f"Full history · {len(record):,} sessions · Mūla, Swayam and call_ts joined "
                f"on date at full precision.")
