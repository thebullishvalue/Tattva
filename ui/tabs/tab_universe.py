"""
Tattva — Universe: every catalogue target's valuation, side by side.

One Mūla fit per target (no Swayam, no precedent), so the whole catalogue is a
few minutes. The page ranks targets by how far they sit from fair value and
shows each one's own record beside it. It prints states, never calls: whether
a state may be called is the class record's decision, shown as a column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.config import CATALOGUE_TARGETS
from core.logger_config import console
from pipeline.scan import scan
from ui.components import (render_empty_state, render_kpi_strip, render_note,
                           render_section_header, render_table_panel)
from ui.theme import progress_bar

_CLASS_NAMES = {"commodity": "Commodity", "fx": "FX", "crypto": "Crypto",
                "india_index": "India", "us_index": "US"}


def _display(res: pd.DataFrame) -> pd.DataFrame:
    ok = res[res["error"].isna()].copy()
    ok = ok.sort_values("fvo")
    out = pd.DataFrame({
        "Target": ok["target"],
        "Class": ok["class"].map(_CLASS_NAMES).fillna("—"),
        "State": ok["state"].str.title(),
        "Gap (typical sizes)": ok["fvo"],
        "Gap %": ok["pct_gap"],
        "Percentile of own history": ok["gap_percentile"],
        "Gap half-life (d)": ok["half_life"],
        "P(price reverts)": ok["mr_prob"],
        "Cross-section": ok["xs_consistency"],
        "Own hit %": ok["hit"],
        "Any-day %": ok["base"],
        "Claims": ok["claims"],
        "Class licensed": np.where(ok["licensed"], "yes", "no"),
        "As of": ok["as_of"],
    })
    return out.reset_index(drop=True)


def render_universe_tab(df: pd.DataFrame, active_target: str) -> None:
    render_section_header(
        "Universe — Where Everything Stands",
        "Mūla's valuation of every catalogue target against the same macro cross-section, "
        "cheapest first. The gap is in units of its own typical size; the hit rates are each "
        "target's own claims on the side it is on today, against what that direction earned "
        "on any day.",
        icon="grid", accent="accent",
    )
    fp = (df.attrs or {}).get("panel_fingerprint", "")
    cached = st.session_state.get("universe_scan")
    slot = st.empty()
    label = "Scan catalogue" if not cached or cached.get("fp") != fp else "Rescan"
    if st.button(label, type="primary", key="universe_scan_btn"):
        targets = list(CATALOGUE_TARGETS)
        was = console.enabled
        console.enabled = False                 # one trace per target would bury the log
        try:
            res = scan(df, targets, progress=lambda i, n, t: progress_bar(
                slot, int(100 * i / max(n, 1)), "Scanning the catalogue",
                f"{t} · {i}/{n}" if t else "done"))
        finally:
            console.enabled = was
        slot.empty()
        cached = {"fp": fp, "result": res}
        st.session_state["universe_scan"] = cached

    if not cached or cached.get("fp") != fp:
        render_empty_state(
            "Nothing scanned yet",
            f"A scan fits the valuation engine on all {len(CATALOGUE_TARGETS)} catalogue targets "
            f"using the data already in this session — about ten seconds each.",
            eyebrow="Universe",
        )
        return

    res: pd.DataFrame = cached["result"]
    ok = res[res["error"].isna()]
    cheap = int((ok["fvo"] <= -0.5).sum())
    rich = int((ok["fvo"] >= 0.5).sum())
    render_kpi_strip([
        {"label": "Valued", "value": f"{len(ok)}", "subtext": f"of {len(res)} catalogue targets",
         "color_class": "neutral"},
        {"label": "Cheap", "value": f"{cheap}", "subtext": "half a typical gap below, or more",
         "color_class": "success"},
        {"label": "Rich", "value": f"{rich}", "subtext": "half a typical gap above, or more",
         "color_class": "danger"},
        {"label": "Class-licensed", "value": f"{int(ok['licensed'].sum())}",
         "subtext": "states the record allows calling", "color_class": "accent"},
    ], max_cols=4, key="universe-kpis")

    view = _display(res)
    render_table_panel(
        view, "universe-table", context=f"{len(view)} targets · cheapest first",
        label_col="Target", max_height=720,
        col_precision={"Gap (typical sizes)": 2, "Gap %": 1, "Percentile of own history": 0,
                       "Gap half-life (d)": 0, "P(price reverts)": 2, "Cross-section": 2,
                       "Own hit %": 1, "Any-day %": 1, "Claims": 0},
    )
    bad = res[res["error"].notna()]
    if len(bad):
        render_note("Not valued: " + "; ".join(f"{t} ({e})" for t, e in zip(bad["target"], bad["error"])) + ".")

    names = list(view["Target"])
    c1, c2 = st.columns([3, 1], gap="small")
    with c1:
        pick = st.selectbox("Open a target", names, key="universe_pick",
                            index=names.index(active_target) if active_target in names else 0)
    with c2:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if st.button("Analyse", key="universe_open", width="stretch") and pick != active_target:
            st.session_state["active_target"] = pick
            st.session_state.pop("engine", None)
            st.session_state.pop("engine_cache", None)
            for k in ("target_category", "target_select"):
                st.session_state.pop(k, None)
            st.rerun()
    render_note("A positive gap means stretched up (rich). "
                "The scan reads the valuation engine only — open a target for Swayam, precedent "
                "and the full evidence.")
