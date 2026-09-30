"""
Tattva — Convergence tab: where the two engines agree, and where they do not.

The headline object of the whole app: the normalized consensus of Mūla's
valuation conviction and Swayam's breadth, plus the divergences worth knowing
about when they disagree.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Agreement ratio + coverage
  2 ANCHOR    what is the underlying claim?      Convergence analysis cards
  3 SIGNAL    what does it say to do?            Hero signal history
  4 STATE     how does that sit historically?    Unified Signal plot
  5 DETAIL    the evidence behind it             Recent divergences
"""

from __future__ import annotations

import streamlit as st

from analytics.adaptive import tier_now, adaptive_tiers
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba, panel_bg)
from ui.components import (render_metric_card, render_section_header,                            render_info_box, render_empty_state,
                           render_chart_panel, render_note)
from convergence.normalization import (
    align_fvo_swayam,
    causal_normalize,
    classify_normalized_signal,
)
from core.config import (
    get_instrument_config, InstrumentConfig,  # per-instrument marker/tier anchors
    # Marker/tier constants are NOT imported here — they are resolved per-instrument
    # off get_instrument_config(active_target) at render time (see below).
    UI_CHART_HEIGHT_STACKED,
)

# (Tab-local colour aliases stood here as module-level constants. They were
# evaluated ONCE at import, when there is no session to read a theme from,
# so every chart drawn through them was frozen to whichever theme happened
# to be active at first import — the same import-time binding that made the
# original COLOR_* constants unable to follow Paper mode. Colours are
# resolved at the call site now, per render.

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "nishkarsh_conviction": (
        "Composite score combining Mūla (top-down) and Swayam (bottom-up) into a single "
        "signal. Near 0 = both systems uncertain — avoid new positions. Large absolute values "
        "= high-conviction opportunities."
    ),
    "fvo_conviction": (
        "Mūla's fair-value breadth: how many lookback windows see the market as overbought "
        "vs. oversold. Below -20 = most stocks cheap (bullish); above +20 = most expensive (bearish)."
    ),
    "swayam_avg": (
        "Average technical signal across the Swayam bottom-up units — basket constituents, or "
        "self-ensemble views of the instrument's own price (Swayam self mode). Negative = net "
        "bullish; positive = net bearish. Moves slowly and confirms (or contradicts) Mūla's "
        "top-down view."
    ),
    "agreement": (
        "How often Mūla and Swayam point in the same direction. THE BASELINE IS NOT 50%: "
        "both engines are causal transforms of the same target price, and two such "
        "transforms of even a pure random walk agree ~60% of the time by construction "
        "(measured, research/test_audit_invariants.py). So ~60% is neutral, not agreement. "
        "Above 80% is where a reading starts to clear that floor; above 89% is strong. "
        "Below ~60% is genuine disagreement — stand aside."
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


def _settled(vals, dates=None):
    """Last SETTLED reading in a series, and its date if it is not the latest.

    The panel-completeness gate publishes NaN for any session fitted on a
    fraction of the cross-section: a live half-open row typically carries ~0.39
    of the admitted panel, so `ConvictionRaw.iloc[-1]` is NaN for most of the
    trading day. Every card here reads `[-1]`, and every one of those guards
    tested `is not None` — which NaN passes, so the card rendered the string
    "+nan" rather than falling back.

    Three outcomes, and the distinction between the last two is the point:
    a settled current reading, a settled EARLIER reading (returned with its
    date, so the caller can say which session it belongs to), or nothing.
    Showing the stale number unlabelled would defeat the gate that produced
    the blank in the first place.
    """
    arr = np.asarray([np.nan if v is None else v for v in np.atleast_1d(vals)],
                     dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(arr))
    if not len(finite):
        return None, None
    i = int(finite[-1])
    if i == len(arr) - 1:
        return float(arr[i]), None            # current session is settled
    stamp = None
    if dates is not None and i < len(dates):
        d = dates[i]
        stamp = str(d.date()) if hasattr(d, "date") else str(d)
    return float(arr[i]), (stamp or "earlier session")


def _forming(fvo_ts) -> str | None:
    """How much of the cross-section has printed, when today is not yet trusted.

    Returns e.g. "105/240 printed", or None when the newest session is settled.
    The engine now PUBLISHES the forming session (flagged `Provisional`) rather
    than withholding it, so this is what tells the reader that the number on the
    card is today's-so-far and still moving — not a final reading.

    Reads `Provisional` where present and falls back to `~Valid`, so a cached
    frame from before that column existed still labels correctly.
    """
    if fvo_ts is None:
        return None
    cols = set(getattr(fvo_ts, "columns", ()))
    if not {"NAvailable", "NAdmitted"} <= cols:
        return None
    try:
        if "Provisional" in cols:
            unsettled = bool(fvo_ts["Provisional"].iloc[-1])
        elif "Valid" in cols:
            unsettled = not bool(fvo_ts["Valid"].iloc[-1])
        else:
            return None
        if not unsettled:
            return None
        av = float(pd.to_numeric(fvo_ts["NAvailable"], errors="coerce").iloc[-1])
        ad = float(pd.to_numeric(fvo_ts["NAdmitted"], errors="coerce").iloc[-1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(av) and np.isfinite(ad)) or ad <= 0:
        return None
    return f"{int(av)}/{int(ad)} printed"


def _asof(subtext: str, stamp: str | None, forming: str | None = None) -> str:
    """Card subtitle, qualified by which session the number on display belongs to.

    Three states, and the middle one is the common case during a live session:

      settled today          -> the descriptive subtext, unchanged
      today, still forming    -> "Provisional · 105/240 printed"
      an earlier session      -> "As of 2026-08-14 · today 105/240 printed"

    REPLACES rather than appends: the descriptive subtext ("Market breadth:
    oversold vs overbought") is the less useful half once the number on display
    is not a final one, and appending overflows the card at narrow widths.
    """
    if stamp:
        return f"As of {stamp} · today {forming}" if forming else f"As of {stamp}"
    if forming:
        return f"Provisional · {forming}"
    return subtext


def render_convergence_tab(ts_filtered=None):
    """Render the convergence dashboard tab with amber-gold system identity."""
    # System identity background
    convergence_df = st.session_state.get("convergence_df")
    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    fvo_ts = st.session_state.get("fvo_ts")
    swayam_daily = st.session_state.get("swayam_daily")
    # Swayam's bottom-up units are basket CONSTITUENTS in basket mode and
    # self-ensemble VIEWS in Swayam self mode — keep copy accurate for both.
    _self_mode = st.session_state.get("swayam_mode") == "self"
    _units = "views" if _self_mode else "constituents"

    # ── Per-instrument marker / tier anchors ────────────────────────────────
    # Marker tiers for THIS render. The config values are warm-up priors; once
    # a series has a year of its own history the tier is its causal p90/p75 —
    # so the "extreme" line on each row means extreme *for this instrument*
    # rather than extreme for the pooled universe the constants were anchored
    # to. Shadow the module-global names for the rest of this render.
    try:
        _icfg = get_instrument_config(st.session_state.get("active_target", ""))
    except KeyError:
        _icfg = InstrumentConfig()

    def _tier(series, prior, q):
        """Causal tier for one marker row, or the prior while history is short."""
        if series is None or not len(series):
            return float(prior)
        return tier_now(np.asarray(series, dtype=float), float(prior), q=q)

    _hero = st.session_state.get("hero_series")
    _cons_hist = _hero.to_numpy(dtype=float) if _hero is not None and len(_hero) else None
    _fvo_ts = st.session_state.get("fvo_ts")
    _conv_raw = (_fvo_ts["ConvictionRaw"].to_numpy(dtype=float)
                 if _fvo_ts is not None and "ConvictionRaw" in _fvo_ts.columns else None)
    _swayam_d = st.session_state.get("swayam_daily")
    _swayam_avg = (_swayam_d["Avg_Signal"].to_numpy(dtype=float)
                   if _swayam_d is not None and "Avg_Signal" in getattr(_swayam_d, "columns", ())
                   else None)
    _agree = st.session_state.get("convergence_df")
    _agree = (_agree["agreement_ratio"].to_numpy(dtype=float)
              if _agree is not None and "agreement_ratio" in getattr(_agree, "columns", ()) else None)

    UI_CONSENSUS_STRONG = _tier(_cons_hist, _icfg.ui_consensus_strong, 0.90)
    UI_CONSENSUS_MODERATE = _tier(_cons_hist, _icfg.ui_consensus_moderate, 0.75)
    UI_CONVRAW_STRONG = _tier(_conv_raw, _icfg.ui_convraw_strong, 0.90)
    UI_CONVRAW_MODERATE = _tier(_conv_raw, _icfg.ui_convraw_moderate, 0.75)
    UI_SWAYAM_AVG_THRESHOLD = _tier(_swayam_avg, _icfg.ui_swayam_avg_threshold, 0.75)
    # Agreement is a [0,1] ratio, not a signed oscillator — its tiers are
    # quantiles of the ratio itself, so |x| would be a no-op and the p-levels
    # read directly.
    UI_AGREEMENT_STRONG = _tier(_agree, _icfg.ui_agreement_strong, 0.90)
    UI_AGREEMENT_MODERATE = _tier(_agree, _icfg.ui_agreement_moderate, 0.75)
    CONVICTION_MODERATE = _tier(_conv_raw, _icfg.conviction_moderate, 0.75)
    UI_SWAYAM_BULLISH = -UI_SWAYAM_AVG_THRESHOLD
    UI_SWAYAM_BEARISH = UI_SWAYAM_AVG_THRESHOLD

    if convergence_df is None or convergence_df.empty:
        render_empty_state(
            "No convergence data available",
            "Convergence fuses Mūla's valuation read with Swayam's breadth — run an "
            "analysis first so both engines have something to agree or disagree over.",
            eyebrow="Convergence",
            action_label="Run analysis in the sidebar, then return to this page.",
        )
        return

    # ── SINGLE SOURCE OF TRUTH ───────────────────────────────────────────────
    # Align Mūla + Swayam ONCE, here, before anything renders. The metric cards
    # AND the 3-row plot below both read these exact arrays, so a card can never
    # disagree with the plot point it mirrors (the old bug: card read the raw last
    # ts row, the plot read the Swayam-aligned last row → drift on calendar gaps).
    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    aligned_dates, aligned_fvo_raw, aligned_swayam_raw = align_fvo_swayam(
        fvo_ts, swayam_daily, filter_dates=filtered_dates,
    )
    has_overlap = bool(aligned_dates)

    norm_a = norm_n = norm_avg = np.array([], dtype=np.float64)
    aligned_conv_raw: list = []
    #: The causal-normalization cache, hoisted so the per-date tier levels
    #: below can read the FULL normalized series rather than the visible
    #: window — a tier built from only what the window shows would move every
    #: time the window changed, which is the same repaint by another route.
    params: dict = {}
    if has_overlap:
        # Key by the full engine config (target + features + horizon + date range) so
        # switching predictor sets with the same target never reuses stale z-scores.
        # Also fold in content (row count + latest raw Mūla/Swayam reading): a
        # "Refresh Data" that updates the LAST bar's value without changing the
        # date-range fingerprint that engine_cache is built from would otherwise
        # keep this key unchanged and silently reuse pre-refresh z-scores against
        # the post-refresh raw series (audit finding C1).
        _last_a = aligned_fvo_raw[-1] if aligned_fvo_raw else 0.0
        _last_n = aligned_swayam_raw[-1] if aligned_swayam_raw else 0.0
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
            _full_dates, full_a, full_n = align_fvo_swayam(fvo_ts, swayam_daily)
            fa = np.array(full_a, dtype=np.float64)
            fn = np.array(full_n, dtype=np.float64)
            na_full = causal_normalize(fa)
            nn_full = causal_normalize(fn)
            def _dk(d):
                return str(d.date()) if hasattr(d, "date") else str(d)
            _p = {
                "a": {_dk(d): v for d, v in zip(_full_dates, na_full)},
                "n": {_dk(d): v for d, v in zip(_full_dates, nn_full)},
                # The consensus the markers actually plot, kept whole so its
                # tiers can be built from its own past.
                "navg_vals": ((na_full + nn_full) / 2.0),
                "navg_dates": list(_full_dates),
                "_n": len(full_a),
            }
            st.session_state[_np_key] = _p
        params = st.session_state[_np_key]
        def _dk(d):
            return str(d.date()) if hasattr(d, "date") else str(d)
        norm_a = np.array([params["a"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_n = np.array([params["n"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_avg = (norm_a + norm_n) / 2.0
        at_dedup = fvo_ts[~fvo_ts.index.duplicated(keep="last")] if fvo_ts is not None else None
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

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER + METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Convergence Analysis",
        "Mūla top-down vs Swayam bottom-up. Agreement = reliable signal. Divergence = stand aside.",
        icon="target",
    )

    col1, col2, col3, col4 = st.columns(4, gap="small")

    # Shared across the four cards: if the newest session is being withheld, this
    # is why. Computed once so every card gives the same reason.
    _fill = _forming(fvo_ts)

    with col1:
        # Mirrors Row 1 of the Unified Signal plot: average of normalized Mūla
        # + Swayam z-scores, in [-1, +1].
        score, stale = (nishkarsh_norm["value"], None) if nishkarsh_norm else (None, None)
        if score is not None and not np.isfinite(score):
            # The stored headline belongs to an unsettled session. Recover the
            # last settled point from the same normalized average the plot draws.
            score, stale = _settled(norm_avg, aligned_dates) if has_overlap else (None, None)
        if score is not None:
            # When falling back to an earlier session the stored label belongs to
            # the wrong value, so relabel through the same classifier that
            # produced it rather than inventing a threshold here.
            sig = nishkarsh_norm["signal"] if (nishkarsh_norm and not stale) \
                else classify_normalized_signal(score)
            color = "success" if "BUY" in sig else "danger" if "SELL" in sig else "neutral"
            render_metric_card("TATTVA CONVICTION", f"{score:+.2f}", _asof(sig, stale, _fill), color,
                               tooltip=TOOLTIPS["nishkarsh_conviction"])
        else:
            render_metric_card("TATTVA CONVICTION", "N/A", "Not computed", "neutral")

    with col2:
        # Mirrors Row 2 of the plot — reads the SAME aligned ConvictionRaw series
        # (falls back to the raw ts column only when there is no Swayam overlap).
        if has_overlap and aligned_conv_raw:
            a_conv, stale = _settled(aligned_conv_raw, aligned_dates)
        elif fvo_ts is not None and "ConvictionRaw" in fvo_ts.columns:
            a_conv, stale = _settled(fvo_ts["ConvictionRaw"].to_numpy(), fvo_ts.index)
        else:
            a_conv, stale = None, None
        if a_conv is not None:
            render_metric_card("Mūla CONVICTION", f"{a_conv:+.2f}",
                               _asof("Market breadth: oversold vs overbought", stale, _fill),
                               "success" if a_conv < -CONVICTION_MODERATE else "danger" if a_conv > CONVICTION_MODERATE else "neutral",
                               tooltip=TOOLTIPS["fvo_conviction"])
        else:
            render_metric_card("Mūla CONVICTION", "N/A", "Session incomplete", "neutral")

    with col3:
        # Mirrors Row 3 of the plot — reads the SAME aligned Swayam Avg Signal series.
        n_avg, stale = _settled(aligned_swayam_raw, aligned_dates) if (
            has_overlap and len(aligned_swayam_raw)) else (None, None)
        if n_avg is not None:
            render_metric_card("SWAYAM AVG SIGNAL", f"{n_avg:.2f}",
                               _asof(f"Bottom-up {_units[:-1]} momentum", stale, _fill),
                               "success" if n_avg < UI_SWAYAM_BULLISH else "danger" if n_avg > UI_SWAYAM_BEARISH else "neutral",
                               tooltip=TOOLTIPS["swayam_avg"])
        else:
            render_metric_card("SWAYAM AVG SIGNAL", "N/A", f"No {_units[:-1]} data", "neutral")

    with col4:
        agreement, stale = _settled(convergence_df["agreement_ratio"].to_numpy(),
                                    convergence_df.index)
        if agreement is not None:
            render_metric_card("AGREEMENT", f"{agreement:.0%}",
                               _asof("Mūla and Swayam alignment", stale, _fill),
                               "success" if agreement > UI_AGREEMENT_STRONG else "warning" if agreement > UI_AGREEMENT_MODERATE else "neutral",
                               tooltip=TOOLTIPS["agreement"])
        else:
            render_metric_card("AGREEMENT", "N/A", "Session incomplete", "neutral")


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
    # the metric cards). Mūla-only targets (no Swayam basket) have no overlap →
    # the cards above still rendered; the plot just can't be drawn.
    if not has_overlap:
        render_empty_state(
            "No engine overlap",
            "Mūla and Swayam share no dates for this target, so the consensus overlay "
            "cannot be drawn. The metric cards above are Mūla-only reads and remain valid.",
            eyebrow="Convergence",
            action_label="Pick a target with a resolvable Swayam view bank.",
        )
        return

    # Short-history guard: z-scoring needs a stable σ. When the FULL Mūla∩Swayam
    # overlap is tiny (brand-new sheet target, freshly-listed basket constituents),
    # σ collapses to its 1e-10 floor and the whole normalized plot flat-lines at 0 —
    # which misreads as a confident "neutral". The cards above already show the raw
    # latest reads honestly; here we suppress the misleading plot and say why.
    MIN_CONV_NORM_POINTS = 10
    _n_full = int(params.get("_n", len(aligned_dates)))
    if _n_full < MIN_CONV_NORM_POINTS:
        render_info_box(
            "Building convergence history",
            f"Only {_n_full} overlapping session{'s' if _n_full != 1 else ''} between Mūla and Swayam "
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
    _nn_last = st.session_state.get("swayam_native_last")
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
    swayam_y = _dynamic_range(aligned_swayam_raw)

    # ── Build 3-row chart ───────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # ── Signal marker tiers ───────────────────────────────────────────
    # One definition for all three rows. SIZE carries strength, COLOUR carries
    # direction, and opacity now only separates "no signal" from "signal".
    #
    # The moderate tier used to be drawn at 55% alpha, which composited to
    # 2.31:1 against the panel in Terminal and 2.46:1 in Paper — the reader
    # could not see the dots, and they are the ones a moderate signal depends
    # on. At 0.90 the same dot measures 4.30:1 and 4.62:1. Strength stopped
    # being encoded in opacity because a half-transparent 5px dot is not a
    # weaker signal, it is an invisible one.
    _SZ_STRONG, _SZ_MODERATE, _SZ_FLAT = 9, 6, 4
    _A_MODERATE, _A_FLAT = 0.90, 0.55

    def _marker(sizes, colors):
        """Marker styling shared by every row of this figure.

        The 1px outline is drawn in the panel's own colour so that dots in a
        dense run read as separate marks instead of a thick segment, and so a
        dot sitting on the connecting line still detaches from it.
        """
        return dict(size=sizes, color=colors,
                    line=dict(width=1, color=panel_bg()))

    def _dkey(d):
        return str(d.date()) if hasattr(d, "date") else str(d)

    # ── Per-date tier levels ──────────────────────────────────────────
    # A marker's tier must come from the threshold knowable AT ITS OWN DATE.
    # These used to be classified against `tier_now(...)` — one scalar, the
    # p90/p75 of the whole history "as of the last row", which `tier_now`'s
    # own docstring reserves for "display code that colours a SINGLE CURRENT
    # reading". Applied to a series it re-labels the past: each new day moves
    # the quantile a little, every earlier point is re-tested against the new
    # level, and any dot near a boundary changes size and colour. The values
    # never moved — `causal_normalize` above guarantees that — but the chart
    # said something different about a past date than it had the day before,
    # which is repainting as far as a reader is concerned.
    #
    # `adaptive_tiers` is the column form of the same statistic and is what
    # the Mūla engine already publishes with: each row's level is built from
    # strictly earlier rows, so re-running on more data cannot move a label
    # that was already drawn.
    def _tier_at(tiermap, dates, name, fallback):
        """Per-date tier levels for `dates`, falling back while history is short."""
        m = tiermap.get(name, {})
        return np.array([m.get(_dkey(d), fallback) for d in dates], dtype=float)

    def _tier_map(values, dates, priors, quantiles):
        if values is None or not len(values) or dates is None or not len(dates):
            return {}
        n = min(len(values), len(dates))
        t = adaptive_tiers(np.asarray(values[:n], dtype=float), priors, quantiles)
        return {k: {_dkey(d): v for d, v in zip(list(dates)[:n], arr)}
                for k, arr in t.items()}

    _QS = {"strong": 0.90, "moderate": 0.75}

    # Row 1 — classified on the same normalized consensus the markers plot.
    _navg_map = _tier_map(
        params.get("navg_vals"), params.get("navg_dates"),
        {"strong": _icfg.ui_consensus_strong, "moderate": _icfg.ui_consensus_moderate}, _QS)
    _cs = _tier_at(_navg_map, aligned_dates, "strong", UI_CONSENSUS_STRONG)
    _cm = _tier_at(_navg_map, aligned_dates, "moderate", UI_CONSENSUS_MODERATE)

    # Row 2 — the raw conviction series, on its own dates.
    _conv_map = _tier_map(
        _conv_raw, (fvo_ts.index if fvo_ts is not None else None),
        {"strong": _icfg.ui_convraw_strong, "moderate": _icfg.ui_convraw_moderate}, _QS)
    _vs = _tier_at(_conv_map, aligned_dates, "strong", UI_CONVRAW_STRONG)
    _vm = _tier_at(_conv_map, aligned_dates, "moderate", UI_CONVRAW_MODERATE)

    # Row 3 — Swayam's average signal, one threshold (it has a single band).
    _sw_map = _tier_map(
        _swayam_avg, (_swayam_d.index if _swayam_d is not None else None),
        {"moderate": _icfg.ui_swayam_avg_threshold}, _QS)
    _ss = _tier_at(_sw_map, aligned_dates, "moderate", UI_SWAYAM_AVG_THRESHOLD)

    # Convergence color mapping
    avg_colors, avg_sizes = [], []
    for v, _st, _mo in zip(norm_avg, _cs, _cm):
        if v < -_st:
            avg_colors.append(chart_color("emerald")); avg_sizes.append(_SZ_STRONG)
        elif v <= -_mo:
            avg_colors.append(chart_rgba("emerald", _A_MODERATE)); avg_sizes.append(_SZ_MODERATE)
        elif v > _st:
            avg_colors.append(chart_color("rose")); avg_sizes.append(_SZ_STRONG)
        elif v >= _mo:
            avg_colors.append(chart_rgba("rose", _A_MODERATE)); avg_sizes.append(_SZ_MODERATE)
        else:
            avg_colors.append(chart_rgba("slate", _A_FLAT)); avg_sizes.append(_SZ_FLAT)

    # ── Row 1: Unified normalized ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_a, mode="lines", name="Mūla",
        line=dict(color=chart_rgba("slate", 0.25), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_n, mode="lines", name="Swayam",
        line=dict(color=chart_rgba("cyan", 0.2), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_avg, mode="lines+markers", name="Consensus (50/50)",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=_marker(avg_sizes, avg_colors),
    ), row=1, col=1)
    # (A dashed accent 'Calibrated Model' overlay was drawn here. It plotted
    # a second smoothed read of the same signal on top of the consensus
    # line, in the ONE colour this system reserves for interaction —
    # two lines making the same claim, the louder of which meant nothing.)
    # The threshold a marker was judged against, drawn AS A STEP so the line and
    # the dots always tell the same story. A flat `add_hline` at today's level
    # was the last moving element on this figure: the markers stopped being
    # re-labelled, but the bar they were measured against still slid with each
    # run, so a dot could sit the wrong side of its own threshold.
    _step = dict(mode="lines", line=dict(width=0.5, dash="dot"),
                 hoverinfo="skip", showlegend=False)
    fig.add_trace(go.Scatter(x=aligned_dates, y=_cs,
                             line_color=chart_rgba("rose", 0.15), **_step), row=1, col=1)
    fig.add_trace(go.Scatter(x=aligned_dates, y=-_cs,
                             line_color=chart_rgba("emerald", 0.15), **_step), row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=1, col=1)

    # ── Row 2: Base Conviction ────────────────────────────────────────
    conv_vals = [float(v) if v is not None else np.nan for v in aligned_conv_raw]
    conv_colors, conv_sizes = [], []
    for v, _st, _mo in zip(aligned_conv_raw, _vs, _vm):
        if v is None:
            conv_colors.append(chart_rgba("slate", _A_FLAT)); conv_sizes.append(_SZ_FLAT)
        elif v > _st:
            conv_colors.append(chart_color("rose")); conv_sizes.append(_SZ_STRONG)
        elif v >= _mo:
            conv_colors.append(chart_rgba("rose", _A_MODERATE)); conv_sizes.append(_SZ_MODERATE)
        elif v < -_st:
            conv_colors.append(chart_color("emerald")); conv_sizes.append(_SZ_STRONG)
        elif v <= -_mo:
            conv_colors.append(chart_rgba("emerald", _A_MODERATE)); conv_sizes.append(_SZ_MODERATE)
        else:
            conv_colors.append(chart_rgba("slate", _A_FLAT)); conv_sizes.append(_SZ_FLAT)

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=conv_vals, mode="lines+markers", name="Base Conviction",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=_marker(conv_sizes, conv_colors),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=2, col=1)
    fig.add_trace(go.Scatter(x=aligned_dates, y=_vs,
                             line_color=chart_rgba("rose", 0.12), **_step), row=2, col=1)
    fig.add_trace(go.Scatter(x=aligned_dates, y=-_vs,
                             line_color=chart_rgba("emerald", 0.12), **_step), row=2, col=1)

    # ── Row 3: Swayam Avg Signal ──────────────────────────────────────
    # TWO tiers here, not three: Swayam has a single threshold, so there is no
    # moderate band to draw. (The comment that used to sit here claimed three,
    # which the code never implemented.)
    swayam_colors = [chart_color("emerald") if v < -_t
                     else chart_color("rose") if v > _t
                     else chart_rgba("slate", _A_FLAT)
                     for v, _t in zip(aligned_swayam_raw, _ss)]
    swayam_sizes = [_SZ_STRONG if abs(v) > _t else _SZ_FLAT
                    for v, _t in zip(aligned_swayam_raw, _ss)]

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_swayam_raw, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_swayam_raw, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=aligned_swayam_raw, mode="lines+markers", name="Avg Signal",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=_marker(swayam_sizes, swayam_colors),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=aligned_dates, y=_ss,
                             line_color=chart_rgba("rose", 0.15), **_step), row=3, col=1)
    fig.add_trace(go.Scatter(x=aligned_dates, y=-_ss,
                             line_color=chart_rgba("emerald", 0.15), **_step), row=3, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Normalized", y_range=unified_y, row=1, col=1)
    style_axes(fig, y_title="Conviction", y_range=conv_y, row=2, col=1)
    style_axes(fig, y_title="Avg Signal", y_range=swayam_y, row=3, col=1)

    render_chart_panel(fig, "convergence_overlay", units="normalized · conviction · signal", window=True)
    render_note(f"{len(aligned_dates)} overlapping trading days")
