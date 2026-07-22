"""
Tattva — Convergence tab: Unified signal with timeframe filtering.
तत्त्व (Tattva) — "Principle / Essence"

UI — Cross-system convergence visualization: conviction scores with DDM filtering.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import chart_layout, style_axes
from ui.components import (render_metric_card, render_section_header, section_gap,
                           render_info_box, render_data_table)
from convergence.normalization import (
    align_aarambh_nirnay,
    causal_normalize,
    classify_normalized_signal,
    DEFAULT_THRESHOLDS,
)
from core.config import (
    rgba,  # centralized chart palette (single source: config._PALETTE_RGB)
    get_instrument_config, InstrumentConfig,  # per-instrument marker/tier anchors
    # Marker/tier constants are NOT imported here — they are resolved per-instrument
    # off get_instrument_config(active_target) at render time (see below).
    COLOR_GREEN,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_MUTED,
    UI_CHART_HEIGHT_STACKED,
)

# ── Alias colors for tab-local use ────────────────────────────────────────
EMERALD = COLOR_GREEN
ROSE = COLOR_RED
AMBER = COLOR_AMBER
CYAN = COLOR_CYAN
SLATE = COLOR_MUTED

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "nishkarsh_conviction": (
        "Composite score combining Aarambh (top-down) and Nirnay (bottom-up) into a single "
        "signal. Near 0 = both systems uncertain — avoid new positions. Large absolute values "
        "= high-conviction opportunities."
    ),
    "aarambh_conviction": (
        "Aarambh's fair-value breadth: how many lookback windows see the market as overbought "
        "vs. oversold. Below -20 = most stocks cheap (bullish); above +20 = most expensive (bearish)."
    ),
    "nirnay_avg": (
        "Average technical signal across the Nirnay bottom-up units — basket constituents, or "
        "self-ensemble views of the instrument's own price (Swayam self mode). Negative = net "
        "bullish; positive = net bearish. Moves slowly and confirms (or contradicts) Aarambh's "
        "top-down view."
    ),
    "agreement": (
        "How often Aarambh and Nirnay point in the same direction. Above 70% = both systems "
        "agree — trust the signal. Below 50% = they disagree — stay flat until alignment improves."
    ),
}


def _dynamic_range(vals, padding=0.15):
    """Compute a padded y-axis range from a list of values."""
    valid = [v for v in vals if v is not None and not np.isnan(v)]
    if not valid:
        return (-1, 1)
    mn, mx = min(valid), max(valid)
    span = mx - mn if mx != mn else 1.0
    pad = span * padding
    return (round(mn - pad, 2), round(mx + pad, 2))


def render_convergence_tab(ts_filtered=None):
    """Render the convergence dashboard tab with amber-gold system identity."""
    convergence_df = st.session_state.get("convergence_df")
    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    aarambh_ts = st.session_state.get("aarambh_ts")
    nirnay_daily = st.session_state.get("nirnay_daily")
    # Nirnay's bottom-up units are basket CONSTITUENTS in basket mode and
    # self-ensemble VIEWS in Swayam self mode — keep copy accurate for both.
    _self_mode = st.session_state.get("nirnay_mode") == "self"
    _units = "views" if _self_mode else "constituents"

    # ── Per-instrument marker / tier anchors ────────────────────────────────
    # This target's own classification tiers (defaults == the pooled house
    # convention; a _PER_INSTRUMENT_OVERRIDES entry retunes how THIS target's
    # already-computed signal is MARKED, not how it's computed). Shadow the
    # module-global names with per-instrument values for the rest of this render.
    try:
        _icfg = get_instrument_config(st.session_state.get("active_target", ""))
    except KeyError:
        _icfg = InstrumentConfig()
    UI_CONSENSUS_STRONG = _icfg.ui_consensus_strong
    UI_CONSENSUS_MODERATE = _icfg.ui_consensus_moderate
    UI_CONVRAW_STRONG = _icfg.ui_convraw_strong
    UI_CONVRAW_MODERATE = _icfg.ui_convraw_moderate
    UI_NIRNAY_AVG_THRESHOLD = _icfg.ui_nirnay_avg_threshold
    UI_AGREEMENT_STRONG = _icfg.ui_agreement_strong
    UI_AGREEMENT_MODERATE = _icfg.ui_agreement_moderate
    CONVICTION_MODERATE = _icfg.conviction_moderate
    UI_NIRNAY_BULLISH = _icfg.ui_nirnay_bullish
    UI_NIRNAY_BEARISH = _icfg.ui_nirnay_bearish

    if convergence_df is None or convergence_df.empty:
        st.info("No convergence data available. Run the analysis first.")
        return

    # ── SINGLE SOURCE OF TRUTH ───────────────────────────────────────────────
    # Align Aarambh + Nirnay ONCE, here, before anything renders. The metric cards
    # AND the 3-row plot below both read these exact arrays, so a card can never
    # disagree with the plot point it mirrors (the old bug: card read the raw last
    # ts row, the plot read the Nirnay-aligned last row → drift on calendar gaps).
    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    aligned_dates, aligned_aarambh_raw, aligned_nirnay_raw = align_aarambh_nirnay(
        aarambh_ts, nirnay_daily, filter_dates=filtered_dates,
    )
    has_overlap = bool(aligned_dates)

    norm_a = norm_n = norm_avg = np.array([], dtype=np.float64)
    aligned_conv_raw: list = []
    if has_overlap:
        # Key by the full engine config (target + features + horizon + date range) so
        # switching predictor sets with the same target never reuses stale z-scores.
        # Also fold in content (row count + latest raw Aarambh/Nirnay reading): a
        # "Refresh Data" that updates the LAST bar's value without changing the
        # date-range fingerprint that engine_cache is built from would otherwise
        # keep this key unchanged and silently reuse pre-refresh z-scores against
        # the post-refresh raw series (audit finding C1).
        _last_a = aligned_aarambh_raw[-1] if aligned_aarambh_raw else 0.0
        _last_n = aligned_nirnay_raw[-1] if aligned_nirnay_raw else 0.0
        _np_key = (
            f"conv_norm_causal::{st.session_state.get('engine_cache', st.session_state.get('active_target', ''))}"
            f"|{len(aligned_dates)}|{_last_a:.6g}|{_last_n:.6g}"
        )
        if _np_key not in st.session_state:
            # Compute per-date CAUSAL expanding-window z-scores over the FULL aligned
            # series.  Applying terminal-point μ/σ to a historical slice is look-ahead
            # bias: earlier bars appear less extreme than they were at the time because
            # σ is estimated from data that didn't yet exist.
            # causal_normalize is the SAME transform convergence.normalization's
            # compute_normalized_convergence uses (audit finding F16) — a
            # hand-duplicated copy here previously had to be kept in sync by
            # inspection for this plot to match the Convergence-tab cards.
            _full_dates, full_a, full_n = align_aarambh_nirnay(aarambh_ts, nirnay_daily)
            fa = np.array(full_a, dtype=np.float64)
            fn = np.array(full_n, dtype=np.float64)
            na_full = causal_normalize(fa)
            nn_full = causal_normalize(fn)
            def _dk(d):
                return str(d.date()) if hasattr(d, "date") else str(d)
            _p = {
                "a": {_dk(d): v for d, v in zip(_full_dates, na_full)},
                "n": {_dk(d): v for d, v in zip(_full_dates, nn_full)},
                "_n": len(full_a),
            }
            st.session_state[_np_key] = _p
        params = st.session_state[_np_key]
        def _dk(d):
            return str(d.date()) if hasattr(d, "date") else str(d)
        norm_a = np.array([params["a"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_n = np.array([params["n"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_avg = (norm_a + norm_n) / 2.0
        at_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")] if aarambh_ts is not None else None
        for d in aligned_dates:
            d_str = str(d.date()) if hasattr(d, "date") else str(d)
            val = None
            if at_dedup is not None:
                if d in at_dedup.index:
                    val = float(at_dedup.loc[d]["ConvictionRaw"])
                elif "Date" in at_dedup.columns:
                    mask = at_dedup["Date"].astype(str).str.contains(d_str)
                    if mask.any():
                        val = float(at_dedup.loc[mask, "ConvictionRaw"].iloc[0])
            aligned_conv_raw.append(val)

    # System identity background
    st.markdown(
        '<div class="tab-bg convergence"></div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER + METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Convergence Analysis",
        "Aarambh top-down vs Nirnay bottom-up. Agreement = reliable signal. Divergence = stand aside.",
        icon="target",
    )

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        # Mirrors Row 1 of the Unified Signal plot: average of normalized Aarambh
        # + Nirnay z-scores, in [-1, +1].
        if nishkarsh_norm:
            score = nishkarsh_norm["value"]
            sig = nishkarsh_norm["signal"]
            color = "success" if "BUY" in sig else "danger" if "SELL" in sig else "neutral"
            render_metric_card("TATTVA CONVICTION", f"{score:+.2f}", sig, color, tooltip=TOOLTIPS["nishkarsh_conviction"])
        else:
            render_metric_card("TATTVA CONVICTION", "N/A", "Not computed", "neutral")

    with col2:
        # Mirrors Row 2 of the plot — reads the SAME aligned ConvictionRaw last point
        # (falls back to the raw last ts row only when there is no Nirnay overlap).
        a_conv = None
        if has_overlap and aligned_conv_raw and aligned_conv_raw[-1] is not None:
            a_conv = aligned_conv_raw[-1]
        elif aarambh_ts is not None and "ConvictionRaw" in aarambh_ts.columns:
            a_conv = float(aarambh_ts["ConvictionRaw"].iloc[-1])
        if a_conv is not None:
            render_metric_card("AARAMBH CONVICTION", f"{a_conv:+.2f}", "Market breadth: oversold vs overbought",
                               "success" if a_conv < -CONVICTION_MODERATE else "danger" if a_conv > CONVICTION_MODERATE else "neutral",
                               tooltip=TOOLTIPS["aarambh_conviction"])
        else:
            render_metric_card("AARAMBH CONVICTION", "N/A", "", "neutral")

    with col3:
        # Mirrors Row 3 of the plot — reads the SAME aligned Nirnay Avg Signal point.
        if has_overlap and len(aligned_nirnay_raw):
            n_avg = float(aligned_nirnay_raw[-1])
            render_metric_card("NIRNAY AVG SIGNAL", f"{n_avg:.2f}", f"Bottom-up {_units[:-1]} momentum",
                               "success" if n_avg < UI_NIRNAY_BULLISH else "danger" if n_avg > UI_NIRNAY_BEARISH else "neutral",
                               tooltip=TOOLTIPS["nirnay_avg"])
        else:
            render_metric_card("NIRNAY AVG SIGNAL", "N/A", f"No {_units[:-1]} data", "neutral")

    with col4:
        agreement = convergence_df["agreement_ratio"].iloc[-1]
        render_metric_card("AGREEMENT", f"{agreement:.0%}", "Aarambh and Nirnay alignment",
                           "success" if agreement > UI_AGREEMENT_STRONG else "warning" if agreement > UI_AGREEMENT_MODERATE else "neutral",
                           tooltip=TOOLTIPS["agreement"])

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # HERO SIGNAL — HISTORY: the exact headline object over time.
    # Trace = the NORMALIZED CONSENSUS ([-1,+1], negative = bullish) — the
    # same series as Row 1 of the Unified Signal chart below, and the series
    # whose LAST point is the hero card's score (consensus-headline product
    # decision). Each point is classified with the hero's factory
    # DEFAULT_THRESHOLDS, so marker colors are literally "what the hero card would
    # have said on that day" (Row 1 below shows the same series with
    # EXTREMENESS markers instead — different lens on one object). Dashed
    # overlay = the DDM-smoothed trend the TREND evidence row compares
    # today's print against.
    # ═══════════════════════════════════════════════════════════════════════
    _hero_series = st.session_state.get("hero_series")
    _hero_smoothed = st.session_state.get("hero_smoothed")
    if _hero_series is not None and len(_hero_series):
        render_section_header(
            "Hero Signal — History",
            "The hero card's headline signal over time (normalized consensus, "
            "negative = bullish — the same series as Row 1 below). Marker colors = "
            "the hero's classification on each day; dashed line = the DDM-smoothed "
            "trend behind the TREND evidence row.",
            icon="target",
            accent="amber",
        )
        _hs = _hero_series
        _hm = _hero_smoothed
        if filtered_dates is not None:
            _mask = [str(d.date()) in filtered_dates for d in _hs.index]
            _hs = _hs[_mask]
            if _hm is not None and len(_hm) == len(_hero_series):
                _hm = _hm[_mask]
        # Compact display labels for the plot (hover + band annotations): the
        # classifier's "STRONG BUY"/"STRONG SELL" are abbreviated to "S. Buy"/
        # "S. Sell" and the tier casing is normalised (Buy/Hold/Sell) so the
        # markers and the left-edge band labels read cleanly at small sizes.
        _HERO_LABEL_DISPLAY = {
            "STRONG BUY": "S. Buy", "BUY": "Buy", "HOLD": "Hold",
            "SELL": "Sell", "STRONG SELL": "S. Sell",
        }
        if len(_hs):
            _hero_colors, _hero_sizes = [], []
            _hero_labels = []
            for _v in _hs.to_numpy():
                # Consensus classifier with the consensus's OWN factory
                # DEFAULT_THRESHOLDS — the exact pairing the hero card's
                # headline label uses (compute_normalized_convergence).
                _lbl = classify_normalized_signal(float(_v))
                _hero_labels.append(_HERO_LABEL_DISPLAY.get(_lbl, _lbl))
                if _lbl == "STRONG BUY":
                    _hero_colors.append(EMERALD); _hero_sizes.append(8)
                elif _lbl == "BUY":
                    _hero_colors.append(rgba("emerald", 0.85)); _hero_sizes.append(6)
                elif _lbl == "STRONG SELL":
                    _hero_colors.append(ROSE); _hero_sizes.append(8)
                elif _lbl == "SELL":
                    _hero_colors.append(rgba("rose", 0.85)); _hero_sizes.append(6)
                else:
                    _hero_colors.append(rgba("slate", 0.75)); _hero_sizes.append(4)

            _fig_hero = go.Figure()
            _fig_hero.add_trace(go.Scatter(
                x=_hs.index, y=np.clip(_hs.to_numpy(), 0, None),
                fill="tozeroy", fillcolor=rgba("rose", 0.05),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            _fig_hero.add_trace(go.Scatter(
                x=_hs.index, y=np.clip(_hs.to_numpy(), None, 0),
                fill="tozeroy", fillcolor=rgba("emerald", 0.05),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            if _hm is not None and len(_hm):
                _fig_hero.add_trace(go.Scatter(
                    x=_hm.index, y=_hm.to_numpy(), mode="lines",
                    name="Smoothed trend (DDM)",
                    line=dict(color=AMBER, width=1.3, dash="dash"),
                ))
            _fig_hero.add_trace(go.Scatter(
                x=_hs.index, y=_hs.to_numpy(), mode="lines+markers",
                name="Hero signal (consensus)",
                line=dict(color=SLATE, width=1.6),
                marker=dict(size=_hero_sizes, color=_hero_colors),
                text=_hero_labels,
            ))
            # Factory classification bands — the hero's actual cut-points
            # (the consensus's own p75/p90-anchored factory set).
            _t = DEFAULT_THRESHOLDS
            _fig_hero.add_hline(y=_t["buy_strong"], line_dash="dot",
                                line_color=rgba("emerald", 0.30), line_width=0.7,
                                annotation_text="S. Buy", annotation_position="left",
                                annotation_font_size=9)
            _fig_hero.add_hline(y=_t["buy_moderate"], line_dash="dot",
                                line_color=rgba("emerald", 0.18), line_width=0.5,
                                annotation_text="Buy", annotation_position="left",
                                annotation_font_size=9)
            _fig_hero.add_hline(y=_t["sell_moderate"], line_dash="dot",
                                line_color=rgba("rose", 0.18), line_width=0.5,
                                annotation_text="Sell", annotation_position="left",
                                annotation_font_size=9)
            _fig_hero.add_hline(y=_t["sell_strong"], line_dash="dot",
                                line_color=rgba("rose", 0.30), line_width=0.7,
                                annotation_text="S. Sell", annotation_position="left",
                                annotation_font_size=9)
            _fig_hero.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5)
            _fig_hero.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED // 2))
            # Dynamic y-range (a fixed ±1.05 would waste vertical space on
            # the consensus's typical scale); include the strong bands so the
            # cut-points stay visible even in quiet stretches.
            style_axes(_fig_hero, y_title="Hero signal",
                       y_range=_dynamic_range(list(_hs.to_numpy())
                                              + [_t["buy_strong"], _t["sell_strong"]]))
            st.plotly_chart(_fig_hero, width='stretch', key="hero_signal_history")
            st.caption("Negative = bullish (system-wide sign convention). The last point is "
                       "the score on the hero card above.")
        section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # UNIFIED NORMALIZED SIGNAL — 3-row stacked chart
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Unified Signal — Normalized Convergence",
        "Z-scored to [−1, 1]. Combined signal (top) decomposed into constituent inputs (below).",
        icon="layers",
        accent="cyan",
    )

    # Aligned series already computed once at the top (single source of truth with
    # the metric cards). Aarambh-only targets (no Nirnay basket) have no overlap →
    # the cards above still rendered; the plot just can't be drawn.
    if not has_overlap:
        st.warning("No overlapping dates between Aarambh and Nirnay data sources "
                   "(this target runs Aarambh-only — see the cards above).")
        return

    # Short-history guard: z-scoring needs a stable σ. When the FULL Aarambh∩Nirnay
    # overlap is tiny (brand-new sheet target, freshly-listed basket constituents),
    # σ collapses to its 1e-10 floor and the whole normalized plot flat-lines at 0 —
    # which misreads as a confident "neutral". The cards above already show the raw
    # latest reads honestly; here we suppress the misleading plot and say why.
    MIN_CONV_NORM_POINTS = 10
    _n_full = int(params.get("_n", len(aligned_dates)))
    if _n_full < MIN_CONV_NORM_POINTS:
        render_info_box(
            "Building convergence history",
            f"Only {_n_full} overlapping session{'s' if _n_full != 1 else ''} between Aarambh and Nirnay "
            f"so far — too few to z-score into a stable convergence view (the plot would flat-line at zero "
            f"and misread as neutral). The cards above reflect the latest raw reads; this view populates "
            f"once {MIN_CONV_NORM_POINTS}+ shared sessions accrue.",
            color="cyan",
        )
        return

    # Honesty for the carry-forward: when the bottom-up source's native data ends
    # before the latest plotted session (its market(s) closed / haven't posted),
    # say so — those trailing breadth points are carried forward, provisional.
    # In self mode the "source" is the instrument's own OHLCV (self-ensemble
    # views), not a constituent basket — keep the copy accurate.
    _nn_last = st.session_state.get("nirnay_native_last")
    _plot_last = aligned_dates[-1] if aligned_dates else None
    try:
        if _nn_last is not None and _plot_last is not None \
                and pd.Timestamp(_nn_last).normalize() < pd.Timestamp(_plot_last).normalize():
            _src = ("The instrument's own price data" if _self_mode
                    else "The constituent basket's data")
            _why = ("the instrument's market is closed or hasn't posted yet" if _self_mode
                    else "the constituents' markets are closed or haven't posted yet")
            render_info_box(
                "Breadth carried forward",
                f"{_src} ends {pd.Timestamp(_nn_last):%d %b %Y}; later sessions "
                f"(through {pd.Timestamp(_plot_last):%d %b %Y}) carry its last reads forward — {_why}, "
                f"so bottom-up breadth on those bars is provisional.",
                color="amber",
            )
    except Exception:
        pass

    # Compute ranges
    unified_y = _dynamic_range(norm_avg)
    conv_y = _dynamic_range(aligned_conv_raw)
    nirnay_y = _dynamic_range(aligned_nirnay_raw)

    # ── Build 3-row chart ───────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # Convergence color mapping
    avg_colors, avg_sizes = [], []
    for v in norm_avg:
        if v < -UI_CONSENSUS_STRONG:
            avg_colors.append(EMERALD); avg_sizes.append(8)
        elif v <= -UI_CONSENSUS_MODERATE:
            avg_colors.append(rgba("emerald", 1.0)); avg_sizes.append(6)
        elif v > UI_CONSENSUS_STRONG:
            avg_colors.append(ROSE); avg_sizes.append(8)
        elif v >= UI_CONSENSUS_MODERATE:
            avg_colors.append(rgba("rose", 1.0)); avg_sizes.append(6)
        else:
            avg_colors.append(rgba("slate", 0.95)); avg_sizes.append(5)

    # ── Row 1: Unified normalized ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, 0, None),
        fill="tozeroy", fillcolor=rgba("rose", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, None, 0),
        fill="tozeroy", fillcolor=rgba("emerald", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_a, mode="lines", name="Aarambh",
        line=dict(color=rgba("slate", 0.25), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_n, mode="lines", name="Nirnay",
        line=dict(color=rgba("cyan", 0.2), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_avg, mode="lines+markers", name="Consensus (50/50)",
        line=dict(color=SLATE, width=2),
        marker=dict(size=avg_sizes, color=avg_colors),
    ), row=1, col=1)
    # Calibrated-model overlay — the DDM-filtered smoothed trend of the SAME
    # calibrated product signal the hero card headlines (audit findings
    # F1/F2/F6: calibrated_conv_series was computed every run but never
    # plotted). Aligned onto aligned_dates (same reindex the consensus/raw
    # traces use) so it's directly comparable to the diagnostic consensus
    # line, not a second unrelated read.
    _calib_series = st.session_state.get("calibrated_conv_series")
    if _calib_series is not None and len(_calib_series):
        _calib_aligned = _calib_series.reindex(pd.DatetimeIndex(aligned_dates)).to_numpy()
        fig.add_trace(go.Scatter(
            x=aligned_dates, y=_calib_aligned, mode="lines", name="Calibrated Model",
            line=dict(color=AMBER, width=1.5, dash="dash"),
            connectgaps=True,
        ), row=1, col=1)
    fig.add_hline(y=UI_CONSENSUS_STRONG, line_dash="dot", line_color=rgba("rose", 0.15), line_width=0.5, row=1, col=1)
    fig.add_hline(y=-UI_CONSENSUS_STRONG, line_dash="dot", line_color=rgba("emerald", 0.15), line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=1, col=1)

    # ── Row 2: Base Conviction ────────────────────────────────────────
    conv_vals = [float(v) if v is not None else np.nan for v in aligned_conv_raw]
    conv_colors, conv_sizes = [], []
    for v in aligned_conv_raw:
        if v is None:
            conv_colors.append(rgba("slate", 0.90)); conv_sizes.append(5)
        elif v > UI_CONVRAW_STRONG:
            conv_colors.append(ROSE); conv_sizes.append(7)
        elif v >= UI_CONVRAW_MODERATE:
            conv_colors.append(rgba("rose", 1.0)); conv_sizes.append(6)
        elif v < -UI_CONVRAW_STRONG:
            conv_colors.append(EMERALD); conv_sizes.append(7)
        elif v <= -UI_CONVRAW_MODERATE:
            conv_colors.append(rgba("emerald", 1.0)); conv_sizes.append(6)
        else:
            conv_colors.append(rgba("slate", 0.95)); conv_sizes.append(5)

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, 0, None),
        fill="tozeroy", fillcolor=rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, None, 0),
        fill="tozeroy", fillcolor=rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=conv_vals, mode="lines+markers", name="Base Conviction",
        line=dict(color=SLATE, width=1.5),
        marker=dict(size=conv_sizes, color=conv_colors),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=2, col=1)
    fig.add_hline(y=UI_CONVRAW_STRONG, line_dash="dot", line_color=rgba("rose", 0.12), line_width=0.5, row=2, col=1)
    fig.add_hline(y=-UI_CONVRAW_STRONG, line_dash="dot", line_color=rgba("emerald", 0.12), line_width=0.5, row=2, col=1)

    # ── Row 3: Nirnay Avg Signal ──────────────────────────────────────
    nirnay_colors = [EMERALD if v < -UI_NIRNAY_AVG_THRESHOLD else ROSE if v > UI_NIRNAY_AVG_THRESHOLD else rgba("slate", 0.95) for v in aligned_nirnay_raw]

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_nirnay_raw, 0, None),
        fill="tozeroy", fillcolor=rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_nirnay_raw, None, 0),
        fill="tozeroy", fillcolor=rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=aligned_nirnay_raw, mode="lines+markers", name="Avg Signal",
        line=dict(color=SLATE, width=1.2),
        marker=dict(size=5, color=nirnay_colors),
    ), row=3, col=1)
    fig.add_hline(y=UI_NIRNAY_AVG_THRESHOLD, line_dash="dot", line_color=rgba("rose", 0.15), line_width=0.5, row=3, col=1)
    fig.add_hline(y=-UI_NIRNAY_AVG_THRESHOLD, line_dash="dot", line_color=rgba("emerald", 0.15), line_width=0.5, row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.06)", line_width=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Normalized", y_range=unified_y, row=1, col=1)
    style_axes(fig, y_title="Conviction", y_range=conv_y, row=2, col=1)
    style_axes(fig, y_title="Avg Signal", y_range=nirnay_y, row=3, col=1)

    st.plotly_chart(fig, width='stretch', key="convergence_overlay")
    st.caption(f"{len(aligned_dates)} overlapping trading days")

    # ═══════════════════════════════════════════════════════════════════════
    # RECENT DIVERGENCES — the section the hero card's RISK row points at.
    # The hero previously said "see the Convergence tab" while NO tab actually
    # rendered the divergence events (a pointer to nowhere — hero-rigor pass).
    # Shows the most recent events with dates so the reader can judge whether
    # the flagged disagreement is current or already resolved; the hero's own
    # count is windowed to ~DIV_LOOKBACK trading days (audit finding F7).
    # ═══════════════════════════════════════════════════════════════════════
    div_events = st.session_state.get("divergence_events")
    if div_events is not None and hasattr(div_events, "empty") and not div_events.empty:
        section_gap()
        render_section_header(
            "Recent Divergences",
            "Latest cross-system disagreement events (Aarambh vs Nirnay), most recent first. "
            "The hero card's RISK row counts only the last ~20 trading days of these.",
            icon="zap",
            accent="rose",
        )
        _recent = div_events.tail(10).iloc[::-1].copy()
        _cols = [c for c in ("divergence_type", "aarambh_signal", "nirnay_signal",
                             "severity", "description") if c in _recent.columns]
        render_data_table(_recent[_cols] if _cols else _recent,
                          index_label="Date", max_height=360)
        st.caption(f"{len(div_events)} divergence events across the full history — "
                   f"showing the {len(_recent)} most recent.")
