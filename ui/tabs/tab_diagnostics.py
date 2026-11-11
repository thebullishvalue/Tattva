"""
Tattva — System tab: does the machinery behind the call actually hold up?

Everything here answers a question the engine pages assume the answer to. Each
engine page grades its own lean; this page puts the grades side by side, shows
what the convergence learner concluded from them, the statistics of the gap
Mūla's call rests on, and whether the data layer feeding all of it is healthy.

Reading order — the house convention every analysis tab follows:

  1 ANSWER    does any of it work?               Walk-forward edge, every engine
  2 TRUST     what did the learner conclude?     Learned vs prior dimension weights
  3 STATE     is the gap a valid spread?         Gap statistics (OU · ADF · KPSS)
  4 DETAIL    is the data sound?                 Data layer health

(Driver importance moved to the Mūla page, beside the valuation it explains;
the regime chart that duplicated Swayam's went; and the "Signal Performance"
table — a BUY/SELL event study of Mūla's window stretch, titled "walk-forward
hit rates" though it was neither the hero's call nor walk-forward — went with
the rival signal it graded.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import (chart_layout, style_axes,
                      chart_color, grid_rgba)
from ui.components import (render_metric_card, render_section_header, render_chip,
                           render_empty_state, render_chart_panel, render_table_panel,
                           render_note, edge_tier, edge_periods)
from core.config import UI_CHART_HEIGHT_MEDIUM
from data.cache import all_caches
from data.circuit_breaker import all_circuits, CircuitState

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "ou_half_life": (
        "Full-history Ornstein-Uhlenbeck fit on the mispricing gap: days for a shock to the "
        "gap to halve. It measures the gap's persistence however it closes — mostly through "
        "fair value moving, on most instruments — not how fast price converges."
    ),
    "adf_pvalue": (
        "Tests whether the gap has a unit root (drifts away from fair value for good). "
        "p < 0.05 rejects the unit root."
    ),
    "kpss_pvalue": (
        "Corroborating test with the opposite null: is the gap stationary? p > 0.05 fails "
        "to reject stationarity — a second, independent confirmation."
    ),
}

#: The engines graded on the System page, in reading order: key in the
#: ``edges`` session dict → (display name, what is graded).
_EDGE_ROWS = (
    ("mula", "Mūla", "the valuation call (oscillator lean)"),
    ("swayam", "Swayam", "net breadth (oversold − overbought)"),
)


def _render_edges(edges: dict, horizon: int) -> None:
    """Every engine's walk-forward edge, side by side, graded one way."""
    rows, bars = [], []
    for key, name, what in _EDGE_ROWS:
        e = (edges or {}).get(key) or {}
        ic = e.get("ic")
        tier, chip = edge_tier(ic)
        pos, nb = edge_periods(e)
        rows.append({
            "Engine": name, "Graded": what,
            "IC": f"{ic:+.3f}" if ic is not None and np.isfinite(ic) else "—",
            "± SE": f"{e['se']:.3f}" if e.get("se") is not None and np.isfinite(e.get("se")) else "—",
            "Calls": int(e.get("n", 0) or 0),
            "Periods +": f"{pos}/{nb}" if nb else "—",
            "Read": chip,
        })
        for b in e.get("blocks") or []:
            if b.get("ic") is not None and np.isfinite(b.get("ic")):
                bars.append((name, f"{str(b.get('start'))[:7]}", b.get("ic")))
    if not any(r["IC"] != "—" for r in rows):
        render_empty_state(
            "Nothing graded yet",
            "The walk-forward edges need roughly a year of published history past each "
            "engine's warm-up before a non-overlapping read exists.",
            eyebrow="System",
        )
        return
    render_table_panel(pd.DataFrame(rows), "system-edges", units=f"non-overlapping {horizon}d IC", windowed=False,
                       label_col="Engine", max_height=200)
    if bars:
        fig = go.Figure()
        colours = {"Mūla": "violet", "Swayam": "cyan"}
        for name in dict.fromkeys(n for n, _p, _v in bars):
            pts = [(p, v) for n, p, v in bars if n == name]
            fig.add_trace(go.Bar(x=[p for p, _ in pts], y=[v for _, v in pts], name=name,
                                 marker=dict(color=chart_color(colours.get(name, "slate")))))
        fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
        fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_MEDIUM), barmode="group")
        style_axes(fig, y_title="IC per period")
        render_chart_panel(fig, "system-edge-blocks", units="IC per period, by engine", windowed=False)
    render_note("A durable edge is positive in most periods, not large in one. With ~100–150 "
                "non-overlapping calls the 1-SE band is roughly ±0.08–0.10 — read the IC "
                "against it.")


def _render_call_record(edges: dict, target: str, horizon: int) -> None:
    """The call as issued: this target, and its asset class pooled."""
    from core.trust import class_record, is_licensed
    render_section_header(
        "The Call's Record",
        f"Claims as they are issued — entered at the next close, held {horizon} sessions — "
        f"against what the same direction earned on any day. The class row is what licenses "
        f"a call; one target's own rows cannot.",
        icon="target", accent="accent",
    )
    own, cls = edges.get("call") or {}, class_record(target) or {}
    rows = []
    for scope, rec in (("This target", own), (f"Asset class ({cls.get('asset_class', '—')})", cls)):
        for side, name in (("long", "Buy claims"), ("short", "Sell claims")):
            r = (rec or {}).get(side) or {}
            if not r.get("n_claims"):
                continue
            rows.append({
                "Scope": scope, "Side": name, "Claims": int(r["n_claims"]),
                "Hit %": r.get("hit"), "Any-day %": r.get("base"),
                "Excess bps": r.get("excess_bps"), "t": r.get("t"),
                "Licensed": ("yes" if is_licensed(r) else "no") if scope != "This target" else "—",
            })
    if not rows:
        render_empty_state("No graded claims yet",
                           "A claim is graded once its holding window has closed.", eyebrow="System")
        return
    render_table_panel(pd.DataFrame(rows), "system-call-record", windowed=False, label_col="Scope",
                       units="as issued, in excess of always holding the direction", max_height=220,
                       col_precision={"Hit %": 1, "Any-day %": 1, "Excess bps": 0, "t": 2, "Claims": 0})
    render_note("A side is licensed when its pooled excess is at least one standard error above "
                "zero and its hit rate beats the any-day base (core/trust.py). The class record "
                "is regenerated by studies/pooled_record.py.")


def render_diagnostics_tab(engine, signal, horizon: int = 10):
    """System — edges, learner, gap statistics, data layer."""
    render_section_header(
        "Walk-Forward Edge — Every Engine",
        f"Each engine's lean against the {horizon}-day forward return, on non-overlapping "
        f"windows: the same grading every engine page shows for itself.",
        icon="trending", accent="accent",
    )
    _render_edges(st.session_state.get("edges") or {}, horizon)
    _render_call_record(st.session_state.get("edges") or {},
                        st.session_state.get("active_target", ""), horizon)

    render_section_header(
        "Gap Statistics",
        "Is Mūla's mispricing gap a valid mean-reverting spread at all? Necessary for the "
        "valuation to mean anything; the price-side reversion read on the Mūla page is what "
        "makes it a trade.",
        icon="crosshair", accent="violet",
    )
    adf = float(signal.get("adf_pvalue", np.nan) if signal.get("adf_pvalue") is not None else np.nan)
    kpss = float(signal.get("kpss_pvalue", np.nan) if signal.get("kpss_pvalue") is not None else np.nan)
    stationary = bool(np.isfinite(adf) and np.isfinite(kpss) and adf < 0.05 and kpss > 0.05)
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        hl = signal.get("ou_half_life")
        render_metric_card("OU Half-Life (full history)",
                           f"{float(hl):.0f}d" if hl is not None and np.isfinite(hl) else "—",
                           "Persistence of the gap itself", "info", tooltip=TOOLTIPS["ou_half_life"])
    with c2:
        render_metric_card("ADF p-value", f"{adf:.3f}" if np.isfinite(adf) else "—",
                           "Rejects a unit root if p < 0.05",
                           "success" if np.isfinite(adf) and adf < 0.05 else "danger",
                           tooltip=TOOLTIPS["adf_pvalue"])
    with c3:
        render_metric_card("KPSS p-value", f"{kpss:.3f}" if np.isfinite(kpss) else "—",
                           "Consistent with stationarity if p > 0.05",
                           "success" if np.isfinite(kpss) and kpss > 0.05 else "danger",
                           tooltip=TOOLTIPS["kpss_pvalue"])
    # Tones by EQUALITY. The chip used to test `"Stationary" in label`, which is
    # also true of "Non-Stationary", so it was green whatever the tests said.
    st.markdown(
        f'<div class="chip-row">'
        f'<span class="cr-item">Stationarity'
        f'{render_chip("Stationary" if stationary else "Non-stationary", "success" if stationary else "warning", as_html=True)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DATA LAYER HEALTH — cache hit rate + circuit breaker state per source
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Data Layer Health",
        "Two-tier cache statistics and circuit-breaker state for each external service.",
        icon="database",
        accent="cyan",
    )

    # ── Caches ────────────────────────────────────────────────────────────
    cache_cols = st.columns(len(all_caches()))
    for col, cache in zip(cache_cols, all_caches()):
        s = cache.stats()
        hit_pct = s["hit_rate"] * 100.0
        total = s["hits"] + s["misses"]
        # Color: green ≥70% hit rate, amber 30-70%, rose <30% (or no data)
        if total == 0:
            color_cls = "neutral"
        elif hit_pct >= 70:
            color_cls = "success"
        elif hit_pct >= 30:
            color_cls = "warning"
        else:
            color_cls = "danger"
        last_fetch = s["last_fetch_time"]
        if last_fetch:
            mins = (pd.Timestamp.now().timestamp() - last_fetch) / 60.0
            sub = f"{s['disk_entries']} disk · last fetch {mins:.0f}m ago"
        else:
            sub = f"{s['disk_entries']} disk · no fetch this run"
        with col:
            render_metric_card(
                f"CACHE · {s['namespace'].upper()}",
                f"{hit_pct:.0f}%" if total else "—",
                sub,
                color_cls,
                tooltip=(
                    f"{s['hits']} hits / {s['misses']} misses · "
                    f"{s['stale_hits']} stale-fallback · "
                    f"{s['writes']} writes · TTL {s['ttl_seconds']}s · version {s['version']}"
                ),
            )

    # ── Circuit Breakers ──────────────────────────────────────────────────
    circ_cols = st.columns(len(all_circuits()))
    for col, cb in zip(circ_cols, all_circuits()):
        st_dict = cb.get_state()
        state = st_dict["state"]
        if state == CircuitState.CLOSED.value:
            color_cls = "success"
            label_val = "CLOSED"
        elif state == CircuitState.HALF_OPEN.value:
            color_cls = "warning"
            label_val = "HALF-OPEN"
        else:
            color_cls = "danger"
            label_val = "OPEN"
        last_fail = st_dict["last_failure"]
        if last_fail:
            mins = (pd.Timestamp.now().timestamp() - last_fail) / 60.0
            sub = f"{st_dict['failure_count']} fails · last {mins:.0f}m ago"
        else:
            sub = f"{st_dict['success_count']} successful calls"
        with col:
            render_metric_card(
                f"CIRCUIT · {st_dict['name'].upper()}",
                label_val,
                sub,
                color_cls,
                tooltip=(
                    f"Threshold: {st_dict['failure_threshold']} failures · "
                    f"Recovery: {st_dict['recovery_timeout']:.0f}s · "
                    f"OPEN blocks calls; HALF-OPEN allows 1 test call after recovery timeout."
                ),
            )
