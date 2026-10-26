"""
Tattva — System tab: does the machinery behind the call actually hold up?

Everything here answers a question the engine pages assume the answer to: the
statistics of the gap Mūla's claim rests on, and whether the data layer
feeding all of it is healthy. Every grade lives on the Grades page.

Reading order — the house convention every analysis tab follows:

  1 STATE     is the gap a valid spread?         Gap statistics (OU · ADF · KPSS)
  2 DETAIL    is the data sound?                 Data layer health

(Driver importance moved to the Mūla page, beside the valuation it explains;
the regime chart that duplicated Swayam's went; the "Signal Performance"
table — a BUY/SELL event study of Mūla's window stretch, titled "walk-forward
hit rates" though it was neither the hero's call nor walk-forward — went with
the rival signal it graded; and the engine-edge table and the learned-weight
charts went with the parallel opinions and the learner they described. The
grades moved to the Grades page, where every one of them reads the issued
call's sample.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import render_metric_card, render_section_header, render_chip
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


def render_diagnostics_tab(engine, signal):
    """System — gap statistics, data layer."""
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
    theta_stable = bool(signal.get("theta_stable", True))
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
        f'<span class="cr-item">\u03b8 Stability'
        f'{render_chip("Stable" if theta_stable else "Unstable", "success" if theta_stable else "warning", as_html=True)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DATA LAYER HEALTH — cache hit rate + circuit breaker state per source
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

