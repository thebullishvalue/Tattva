"""
Tattva — Reusable UI components: metric cards, signal badges, headers, section headers.
तत्त्व (Tattva) — "Principle / Essence"

UI — "Graphite" institutional terminal design language (see ui/theme.py).

Every function here emits markup for classes defined in ui/theme.css and
nothing else: no inline colours, no inline type. If a component needs a new
look, the rule belongs in the stylesheet, so the light theme keeps working
as a token swap rather than needing a parallel set of Python branches.
"""

from __future__ import annotations

import datetime as _dt
import html as html_mod
from contextlib import contextmanager as _contextmanager

import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as _components_html


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "sun":        '<svg aria-label="Light mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon":       '<svg aria-label="Dark mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "rocket":     '<svg aria-label="Strong Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4.5c1.62-1.63 5-2.5 5-2.5"/><path d="M12 15v5s3.03-.55 4.5-2c1.63-1.62 2.5-5 2.5-5"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up-right": '<svg aria-label="Weak Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>',
    "arrow-down-right": '<svg aria-label="Weak Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Crisis icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
}


#: The app's single icon drawing style. Every icon is normalised to these by
#: ``get_icon`` regardless of what its own SVG literal declares — the set was
#: pasted in over time and carries three different stroke weights and a mix of
#: butt/round terminals, which is exactly how an icon set stops reading as a
#: set. One weight, round terminals, no fills.
ICON_STROKE = 1.6
_ICON_LINECAP = "round"


def get_icon(name: str, size: int = 18, stroke_width: float | None = None) -> str:
    """Return an SVG icon normalised to the app's one icon style.

    ``stroke_width`` is accepted for call sites that predate ``ICON_STROKE``
    but is deliberately clamped: the two callers that passed 1.8 and 2 were
    drawing the same icons as everything else, one and two notches heavier,
    inside components that sat side by side.
    """
    import re
    base_svg = ICONS.get(name, ICONS["chart"])

    # Strip whatever the literal declared, so the result cannot inherit a
    # per-icon weight or terminal style.
    for attr in ("width", "height", "stroke-width", "stroke-linecap", "stroke-linejoin"):
        base_svg = re.sub(rf'\s+{attr}="[^"]*"', "", base_svg)

    sw = ICON_STROKE if stroke_width is None else min(float(stroke_width), 1.75)
    return base_svg.replace(
        "<svg",
        f'<svg width="{size}" height="{size}" stroke-width="{sw}" '
        f'stroke-linecap="{_ICON_LINECAP}" stroke-linejoin="{_ICON_LINECAP}"',
    )


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a styled section header with icon, title, and optional description.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
    """
    svg = get_icon(icon, size=16)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    # `.desc` is a DIRECT child of the header, not nested inside `.text`. The
    # header is a two-row grid — icon and title on row 1, description on row 2
    # under the title — and a nested description is not a grid item, so it
    # could not be placed and fell back to flowing under the title with its
    # own margin. That is the gap that made the subtitle read as a detached
    # paragraph.
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text"><h3>{html_mod.escape(title)}</h3></div>'
        f'{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sub_header(title: str) -> None:
    """A labelled division INSIDE a section — one tier below a section header.

    Mono micro-label, no icon, no rule. Used where a section has two named
    parts (Diagnostics' "Importance Over Time" above its history table), which
    the tab files had been hand-rolling as inline-styled divs at three
    different sizes.
    """
    st.markdown(f'<div class="sub-head">{html_mod.escape(title)}</div>',
                unsafe_allow_html=True)


def render_control_hint(text: str) -> None:
    """Render the canonical terse helper caption beneath a control.

    This is the single source of truth for the "sub-control hint" tier — the
    uppercase micro-caption used by e.g. the "Swayam basket · producer
    cross-section" and Signal-Horizon hints. Use it instead of ``st.caption``
    for control helper text so the sidebar/tab fine-print stays one coherent
    visual hierarchy. Keep the text terse and ``·``-separated.
    """
    st.markdown(
        f'<div class="control-hint">{html_mod.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    """The one caption tier — a note under a chart, table or control.

    Replaces bare ``st.caption`` everywhere. Streamlit's caption renders in
    its own sans face at its own size with its own margin, so eight of them
    scattered across four tab files read as eight different kinds of aside.
    This is the same object as ``render_control_hint`` (identical styling on
    purpose) named for its other use, so a reader of the tab code does not
    have to know that "control hint" also means "chart footnote".
    """
    st.markdown(
        f'<div class="control-hint">{text}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL SYSTEM — one anatomy for every framed thing in the app
# ═══════════════════════════════════════════════════════════════════════
#
# A panel is: header (title / context, meta and chip right) · body · footer.
# Charts, tables and embedded iframes all use it, so a screen mixing them
# reads as one grid instead of as several products sharing a page.
#
# It is a real ``st.container`` rather than an HTML string because the body
# holds WIDGETS — a Plotly figure, a components.v1 iframe — which no amount
# of markdown can wrap. The container carries `key="panel-<id>"`, and
# theme.css styles `[class*="st-key-panel-"]`.

def render_panel_header(
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
) -> None:
    """Render a panel header.

    ``title`` — what the panel shows. Omit it when the section header
    directly above already names the panel; a panel header that restates the
    section header is a second title, not a header.
    ``context`` — the panel's own metadata (instrument, window, units).
    ``meta``/``chip`` — right-aligned status: as-of, source, freshness.
    """
    if not (title or context or meta or chip):
        return
    left = ""
    if title:
        left += f'<span class="ph-title">{html_mod.escape(title)}</span>'
    if context:
        left += f'<span class="ph-context">{html_mod.escape(context)}</span>'
    right = ""
    if meta:
        right += f'<span>{html_mod.escape(meta)}</span>'
    if chip:
        right += render_chip(chip[0], chip[1], as_html=True) or ""
    st.markdown(
        f'<div class="panel-hdr"><div class="ph-left">{left}</div>'
        f'<div class="ph-right">{right}</div></div>',
        unsafe_allow_html=True,
    )


@_contextmanager
def panel(
    key: str,
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
):
    """Context manager wrapping any content in the shared panel chrome.

    ``with panel("fvo-fairvalue", context="GOLD · 6M"): st.plotly_chart(...)``

    Use it directly for anything that is neither a chart nor a table (an
    embedded widget, a bespoke layout) so that thing still belongs to the
    system rather than sitting on the page unframed.
    """
    with st.container(key=f"panel-{key}"):
        if window:
            # A widget cannot live inside the header's HTML string, so the
            # header and the control are emitted as two siblings of one
            # container, and `.st-key-panelrow-*` turns that container's
            # vertical block into a centred row. Columns were the obvious
            # choice here and the wrong one: `stColumn` computes to zero
            # height, so the header hung 8px off the control's centre line and
            # no amount of override CSS pulled it back. Siblings in a single
            # flex row centre against each other by construction.
            with st.container(key=f"panelrow-{key}"):
                render_panel_header(title, context=context, meta=meta, chip=chip)
                render_window_control(key)
        else:
            render_panel_header(title, context=context, meta=meta, chip=chip)
        yield
        if footer:
            st.markdown(f'<div class="panel-foot">{footer}</div>', unsafe_allow_html=True)


def default_chart_context(units: str = "", windowed: bool = True) -> str:
    """The context line every chart panel gets for free: instrument · window.

    Read from session state rather than threaded through every call site,
    which is both less plumbing and strictly more correct — a context built
    from the same keys the command bar reads cannot disagree with it.

    ``windowed=False`` for panels the chart window does not frame (an edge
    graded over the whole history, today's driver snapshot): they used to
    print "6M" in their header like every other panel, a claim about the
    data on them that was false.
    """
    parts = [
        str(st.session_state.get("active_target", "") or "").upper(),
        str(st.session_state.get("tf_selected", "") or "") if windowed else "full history",
    ]
    if units:
        parts.append(units)
    return " · ".join(p for p in parts if p)


def render_chart_panel(
    fig,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
    windowed: bool = True,
) -> None:
    """Render a Plotly figure inside the shared panel chrome.

    The ONE way a chart reaches the screen in this app. Every call passes the
    same ``PLOTLY_CONFIG``, which is what removes the stock Plotly toolbar and
    its logo — previously eighteen of the app's charts shipped Plotly's own
    chrome, complete with a link out to plotly.com.

    ``title`` is normally EMPTY. Every chart in Tattva already sits under a
    ``render_section_header`` that names it and explains how to read it; a
    panel title would be that title again, four pixels lower. What the section
    header cannot say is which instrument and window the plot is actually
    drawn on, so that is what the panel header carries — supplied
    automatically from session state, with ``units`` appended.
    """
    from ui.theme import PLOTLY_CONFIG   # local: avoids a circular import
    ctx = default_chart_context(units, windowed) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer,
               window=window):
        st.plotly_chart(fig, width="stretch", key=f"chart-{key}", config=PLOTLY_CONFIG)


#: Calendar span of each chart window. Charts cut on calendar time (a 6M
#: window is six calendar months whatever the instrument's session count);
#: TIMEFRAME_TRADING_DAYS is the same windows in sessions, for the one path
#: that has no dates. Keyed identically — asserted — so the two cannot drift.
_WINDOW_MONTHS = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24}


def window_cutoff(last, tf: str | None):
    """First date shown under chart window ``tf`` ending at ``last``; None = all."""
    from core.config import TIMEFRAME_TRADING_DAYS
    assert set(_WINDOW_MONTHS) == set(TIMEFRAME_TRADING_DAYS), "window keys drifted"
    if not tf or tf not in _WINDOW_MONTHS or last is None or pd.isna(last):
        return None
    return pd.Timestamp(last) - pd.DateOffset(months=_WINDOW_MONTHS[tf])


def render_window_control(key: str = "window") -> None:   # noqa: ARG001
    """The chart-window selector, rendered inside a panel header.

    It used to sit in a toolbar strip docked under the command bar — page
    chrome, physically distant from the thing it changes. A control that
    reframes a chart belongs ON that chart, so it now renders in the panel
    header, right-aligned opposite the panel's context line.

    All charts on a page share one window, so exactly one panel per page
    carries the control (``render_chart_panel(..., window=True)``). It writes
    the shared ``tf_selected`` key that the page's series filtering reads.
    """
    from core.config import TIMEFRAME_TRADING_DAYS
    options = list(TIMEFRAME_TRADING_DAYS) + ["ALL"]
    if st.session_state.get("tf_selected") not in options:
        st.session_state["tf_selected"] = "6M"
    st.segmented_control(
        "Window", options, key="tf_selected", label_visibility="collapsed",
        help="Chart window. Applies to every plot on this page; the engines "
             "always fit on the full history.",
    )


def render_table_panel(
    df,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
    windowed: bool = True,
    **table_kwargs,
) -> None:
    """Render a DataFrame inside the shared panel chrome.

    Same header anatomy — and the same ``units``/``context`` contract — as
    ``render_chart_panel``, deliberately, so a table and a chart sitting side
    by side are visibly the same kind of object.

    ``units`` MUST be declared here rather than left to ``**table_kwargs``:
    without it, every ``units=`` at a call site fell through to
    ``render_data_table``, which has no such parameter, and the page died with
    ``render_data_table() got an unexpected keyword argument 'units'``.
    Remaining kwargs still pass through to the table renderer.
    """
    ctx = default_chart_context(units, windowed) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer,
               window=window):
        render_data_table(df, **table_kwargs)


def render_chip(label: str, tone: str = "neutral", *, as_html: bool = False) -> str | None:
    """Render (or return) a status chip — one badge system for the whole app.

    ``tone``: ``accent`` / ``success`` / ``danger`` / ``warning`` / ``info`` /
    ``neutral``. Used for analog-tier badges, session/freshness state, and
    any other "state in a word" reading (previously several one-off HTML
    blocks with their own colour logic). Pass ``as_html=True`` to get the raw
    ``<span>`` back for composition inside a larger ``st.markdown`` call
    (e.g. ``render_top_bar``) instead of rendering it standalone.
    """
    html = f'<span class="chip chip-{html_mod.escape(tone)}">{html_mod.escape(label)}</span>'
    if as_html:
        return html
    st.markdown(html, unsafe_allow_html=True)
    return None


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a terminal-styled metric card with optional tooltip.

    Args:
        label: Card label (rendered uppercase).
        value: Primary metric value.
        subtext: Optional secondary description below value.
        color_class: Semantic color — "neutral", "success", "danger", "warning", "info", "violet".
        tooltip: Optional hover explanation text.
        icon: Optional ICONS key — small icon inlined before the label.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )

    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f'<span class="label">{icon_html}{html_mod.escape(label)}</span>'
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpi_strip(items: list[dict], *, max_cols: int = 5, key: str = "kpi-strip") -> None:
    """Lay out ``render_metric_card`` items in rows of at most ``max_cols``.

    Each item is the keyword set ``render_metric_card`` takes: ``label``,
    ``value``, and optionally ``subtext``/``color_class``/``tooltip``/
    ``icon``. Used for the Overview page's KPI row — caps row width so cards
    never squeeze past legibility (wraps to a second row instead of the
    6-wide layouts elsewhere in the app that get tight on narrow viewports).
    """
    if not items:
        return
    with st.container(key=key):
        for i in range(0, len(items), max_cols):
            row = items[i:i + max_cols]
            cols = st.columns(len(row), gap="small")
            for c, item in zip(cols, row):
                with c:
                    render_metric_card(
                        label=item.get("label", ""),
                        value=item.get("value", ""),
                        subtext=item.get("subtext", ""),
                        color_class=item.get("color_class", "neutral"),
                        tooltip=item.get("tooltip", ""),
                        icon=item.get("icon", ""),
                    )


def render_header(title: str, tagline: str) -> None:
    """Render the cold-start masthead.

    Stacked, not inline: the mark reads at display size on its own line and
    the tagline sits under it as a rule-delimited subtitle. Inline (the
    previous arrangement) the two competed for the same optical line and the
    mark ended up the same size as a section heading — which is what a
    masthead must not be, since it is the only thing on a cold-start screen
    that says what the application is.
    """
    head, tail = (title[:3], title[3:]) if len(title) > 3 else (title, "")
    st.markdown(
        f'<div class="premium-header">'
        f'<div class="title">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_nav_brand(title: str = "TATTVA", tagline: str = "तत्त्व · Unified Convergence") -> None:
    """Render the control rail's brand block.

    The mark is split so the second half carries the accent — a product mark
    that is one flat colour reads as a heading, not as a mark. Left-aligned
    (not centred) because everything below it in the rail is left-aligned,
    and a centred mark over a left-aligned column is the single most common
    tell of a template.
    """
    head, tail = (title[:3], title[3:]) if len(title) > 3 else (title, "")
    st.markdown(
        f'<div class="nav-brand">'
        f'<div class="mark">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_top_bar(
    *,
    target: str = "",
    price: float | None = None,
    change_pct: float | None = None,
    status_label: str = "",
    status_tone: str = "neutral",
    meta: str = "",
    meta_items: "list[tuple[str, str]] | None" = None,
    open_strip: bool = False,
) -> None:
    """Render the command bar — the first element on every page.

    Reading order left to right is *identity → value → trust*: which
    instrument, what it is worth, and whether the data behind that is
    current. Nothing renders above this bar; data-quality notices that used
    to sit on top of the page now hang below it in the notice rail, so the
    instrument is always the first thing on screen.

    ``target``/``price``/``change_pct`` describe the active instrument (omit
    price/change when unresolved, e.g. before the target column has loaded).
    ``status_label``/``status_tone`` render as one chip summarising freshness
    at a glance — the full explanation lives in the notice rail below.
    ``meta_items`` are key/value pairs shown right-aligned (as-of date,
    horizon, spine size); ``meta`` is the legacy single-caption form and is
    folded into them. ``open_strip=True`` leaves the bottom corners square
    for a ``toolbar_strip`` to dock flush underneath.
    """
    instrument_html = ""
    if target:
        instrument_html = (
            f'<div class="cb-instrument">'
            f'<span class="eyebrow">Instrument</span>'
            f'<span class="sym">{html_mod.escape(target)}</span>'
            f'</div>'
        )
    quote_html = ""
    if price is not None:
        # `change_pct` is in PERCENT POINTS (-0.42 == -0.42%), the same unit it
        # is printed in. It used to arrive as a fraction and be formatted with
        # "%.2f%%", so every sub-1% session — i.e. most of them — printed as
        # "0.00%". Flat band is a half basis point.
        chg = change_pct if change_pct is not None else 0.0
        chg_cls = "up" if chg > 0.005 else "down" if chg < -0.005 else "flat"
        arrow = "▲" if chg_cls == "up" else "▼" if chg_cls == "down" else "▬"
        # Arrow carries the sign, so the number is unsigned — "▼ 0.42%", not
        # "▼ -0.42%". Direction is stated twice (glyph + colour) and never by
        # colour alone, for red/green deficiency.
        chg_html = (
            f'<span class="chg {chg_cls}">{arrow} {abs(chg):.2f}%</span>'
            if change_pct is not None else ""
        )
        quote_html = (
            f'<div class="cb-quote"><span class="px">{price:,.2f}</span>{chg_html}</div>'
        )

    items = list(meta_items or [])
    if meta:
        items.append(("As of", meta.replace("As of ", "")))
    meta_html = "".join(
        f'<div class="cb-meta"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v">{html_mod.escape(str(v))}</span></div>'
        for k, v in items if v
    )
    chip_html = render_chip(status_label, status_tone, as_html=True) if status_label else ""
    open_cls = " open" if open_strip else ""
    st.markdown(
        f'<div class="command-bar{open_cls}">'
        f'<div class="cb-left">'
        f'<div class="cb-brand"><span class="mark">TAT<span class="accent-ink">TVA</span></span>'
        f'<span class="sub">तत्त्व</span></div>'
        f'{instrument_html}{quote_html}'
        f'</div>'
        f'<div class="cb-right">{meta_html}{chip_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_notice_rail(notices: "list[dict] | None") -> None:
    """Render queued data-quality notices as a compact rail under the chrome.

    Each notice is ``{"kind": "warning"|"info", "title": str, "body": str}``.
    These used to render as full-width boxes at the very top of the page —
    three of them (stale source, partial session, carried-forward
    predictors) pushed the instrument itself below the fold, so on exactly
    the days the data most needed scrutiny the interface led with an apology
    instead of a price. One row each, severity on the left rule, title and
    explanation on one line: same information, a third of the vertical cost,
    and now BELOW the thing it qualifies rather than above it.
    """
    if not notices:
        return
    rows = "".join(
        f'<div class="notice {html_mod.escape(n.get("kind", "info"))}">'
        f'<div class="n-title">{html_mod.escape(n.get("title", ""))}</div>'
        f'<div class="n-body">{n.get("body", "")}</div>'
        f'</div>'
        for n in notices
    )
    st.markdown(f'<div class="notice-rail">{rows}</div>', unsafe_allow_html=True)


def render_rail_readout(rows: "list[tuple[str, str, str]]") -> None:
    """Render the rail's session readout — ``(label, value, tone)`` rows.

    ``tone`` is "" / "accent" / "long" / "short" / "caution". Values are
    right-aligned and tabular so a changed number is seen rather than read.
    """
    if not rows:
        return
    body = "".join(
        f'<div class="row"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v {html_mod.escape(tone)}">{html_mod.escape(str(v))}</span></div>'
        for k, v, tone in rows
    )
    st.markdown(f'<div class="rail-readout">{body}</div>', unsafe_allow_html=True)


#: Instruments on the tape, in reading order: the broad risk complex, then
#: rates, then the dollar, then the commodity and volatility poles. Ordered
#: by what a desk checks first rather than alphabetically — a tape is scanned
#: peripherally, and a familiar running order is what makes that possible.
TICKER_INSTRUMENTS: tuple[str, ...] = (
    "US Large Cap (S&P 500)", "US Nasdaq 100", "US Small Cap (Russell 2000)",
    "Global Equity (ACWI)", "Japan Equity (EWJ)", "Eurozone Equity (EZU)",
    "Emerging Markets Equity", "India Equity",
    "US 10-Year Treasury Yield", "US Treasury Long (20Y+)",
    "US Corporate Investment Grade", "US High Yield Corporate",
    "Dollar Index", "EUR/USD", "USD/JPY", "USD/INR",
    "Gold", "Silver", "Copper", "Crude Oil", "Brent Crude", "Natural Gas",
    "Broad Commodity Index (DBC)", "Equity Volatility (VIX)",
)


#: Display-name overrides for the tape. Used only where the yfinance ticker
#: is not what a desk would call the instrument (a futures root with an "=F"
#: suffix, an index with a caret) — everywhere else the real ticker is the
#: right label, because that is what a tape shows.
_TAPE_ALIAS: dict[str, str] = {
    "GC=F": "GOLD", "SI=F": "SILVER", "HG=F": "COPPER", "PL=F": "PLAT",
    "CL=F": "WTI", "BZ=F": "BRENT", "NG=F": "NATGAS",
    "^VIX": "VIX", "^MOVE": "MOVE", "^TNX": "US10Y", "^TYX": "US30Y",
    "^FVX": "US5Y", "^IRX": "US3M",
    "DX-Y.NYB": "DXY", "EURUSD=X": "EURUSD", "JPY=X": "USDJPY",
    "INR=X": "USDINR", "CNY=X": "USDCNY",
}


def _tape_symbol(column: str) -> str:
    """Ticker for a macro column, as a tape would print it.

    Resolves the display name back through the config maps to its real symbol
    — "US Nasdaq 100" prints as QQQ, not as a name truncated mid-word. Falls
    back to a word-boundary-safe abbreviation for anything unmapped (a
    Google-Sheet series, a user column), because cutting "US Nasdaq 100" to
    "US Nasdaq 10" is worse than showing fewer words.
    """
    try:
        from core.config import (COMMODITY_TARGETS, GLOBAL_MACRO_MAP,
                                 MACRO_SYMBOLS_YF)
        lookup = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **COMMODITY_TARGETS}
    except Exception:
        lookup = {}
    tkr = lookup.get(column)
    if tkr:
        if tkr in _TAPE_ALIAS:
            return _TAPE_ALIAS[tkr]
        # Strip exchange suffixes (.NS/.L/.TO) and index carets for display.
        clean = tkr.lstrip("^").split(".")[0].replace("=X", "").replace("=F", "")
        if clean:
            return clean.upper()[:10]
    # Unmapped: prefer a trailing parenthetical, else whole words up to 12 chars.
    if "(" in column and column.rstrip().endswith(")"):
        return column.split("(")[-1].rstrip(")")[:10].upper()
    out = ""
    for word in column.split():
        if len(out) + len(word) + 1 > 12:
            break
        out = f"{out} {word}".strip()
    return (out or column[:12]).upper()


def render_ticker(frame, instruments: tuple[str, ...] = TICKER_INSTRUMENTS,
                  seconds_per_item: float = 3.6) -> None:
    """Render the running tape from the already-fetched macro panel.

    No additional network call: the panel behind the valuation engine already
    holds every one of these instruments, so the tape is a view of the data the
    run is using rather than a second, possibly disagreeing, source.

    The track is emitted TWICE and animated to -50%, which is what makes the
    loop seamless — at the moment the first copy leaves the viewport the second
    is exactly where the first began. Duration scales with item count so the
    scroll speed stays constant no matter how many instruments are listed;
    a tape that accelerates as you add symbols is unreadable.

    Direction is carried by an arrow glyph as well as by colour. Roughly 8% of
    men have red/green colour deficiency, and the sign of a move is the one
    reading here that must never be ambiguous.
    """
    if frame is None or not len(frame):
        return
    cols = [c for c in instruments if c in getattr(frame, "columns", ())]
    if not cols:
        return

    tail = frame[cols].tail(2)
    if len(tail) < 2:
        return
    prev, last = tail.iloc[0], tail.iloc[1]

    items: list[str] = []
    for c in cols:
        try:
            p1, p0 = float(last[c]), float(prev[c])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(p1) and np.isfinite(p0)) or p0 == 0:
            continue
        chg = (p1 / p0 - 1.0) * 100.0
        cls, arrow = (("up", "▲") if chg > 0.005 else
                      ("down", "▼") if chg < -0.005 else ("flat", "•"))
        sym = _tape_symbol(c)
        px = f"{p1:,.2f}" if abs(p1) < 10000 else f"{p1:,.0f}"
        items.append(
            f'<span class="tt-item">'
            f'<span class="tt-sym">{html_mod.escape(sym)}</span>'
            f'<span class="tt-px">{px}</span>'
            f'<span class="tt-chg {cls}" data-arrow="{arrow}">{abs(chg):.2f}%</span>'
            f'</span><span class="tt-sep">|</span>'
        )
    if not items:
        return

    run = "".join(items)
    duration = max(40.0, len(items) * float(seconds_per_item))
    st.markdown(
        f'<div class="ticker" role="marquee" aria-label="Live macro tape">'
        f'<div class="tt-track" style="--tt-duration:{duration:.0f}s">{run}{run}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    body: str,
    *,
    eyebrow: str = "",
    action_label: str = "",
) -> None:
    """Render a professional, icon-free empty/degraded state.

    One system for every "nothing to show yet" moment in the app — cold
    start (no data loaded), "no weights yet", "no usable precedent", short-
    history guards — rather than each tab hand-rolling its own notice.
    ``body`` may carry simple inline HTML (``<strong>``, line breaks), same
    convention as ``render_interpretation_card``. ``action_label`` is a short
    hint at what to do next (e.g. "Pick a target in the sidebar, then Run
    Analysis"), not a real button — the actual control lives wherever it
    already does (sidebar, etc.); this just points at it.
    """
    eyebrow_html = f'<div class="es-eyebrow">{html_mod.escape(eyebrow)}</div>' if eyebrow else ""
    action_html = f'<div class="es-action">{html_mod.escape(action_label)}</div>' if action_label else ""
    st.markdown(
        f'<div class="empty-state">'
        f'{eyebrow_html}'
        f'<div class="es-title">{html_mod.escape(title)}</div>'
        f'<div class="es-body">{body}</div>'
        f'{action_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """Render an info box. ``color`` is applied as a modifier class (cyan / amber /
    emerald / rose / violet) so callers can theme it; was previously ignored."""
    st.markdown(
        f'<div class="info-box {html_mod.escape(color)}">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_interpretation_card(
    title: str,
    body: str,
    color: str = "neutral",
) -> None:
    """Render a state-aware interpretation card — terminal readout style.

    Args:
        title: Short state label (e.g. "NEUTRAL", "STRONG OVERSOLD").
        body: One-paragraph explanation (raw HTML allowed — caller is trusted).
        color: Semantic color — "neutral", "success", "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Data-table styling tokens ──────────────────────────────────────────
# render_data_table renders into an isolated components.v1.html iframe, which
# does NOT inherit the app's CSS variables — so the theme values it needs are
# mirrored here as literals, in BOTH a dark and light set. Any change to the
# corresponding --token in theme.css/ui.theme.LIGHT_TOKENS has to be made
# here too; there is no way around that while the table lives in an iframe,
# and a stale colour here is the visible symptom. Header rule/hover use the
# primary ACCENT (not amber — amber is caution/warning only in this system).
_TABLE_TOKENS_DARK = {
    "ink_primary":   "#E6EAF1",   # --ink
    "ink_tertiary":  "#8B95A6",   # --ink-tertiary
    "border":        "rgba(255, 255, 255, 0.07)",   # --line
    "border_subtle": "rgba(255, 255, 255, 0.035)",  # --line-faint
    "accent":        "#4C7DF0",   # --accent
    "emerald":       "#2CA36B",   # --long   (positive numeric cells)
    "rose":          "#DD5A5A",   # --short  (negative numeric cells)
    "accent_border": "rgba(76, 125, 240, 0.34)",
    "accent_hover":  "rgba(76, 125, 240, 0.10)",
    "row_odd":       "rgba(255, 255, 255, 0.015)",
    "row_even":      "transparent",
    "surface_a":     "#0F1217",   # --surface-1
    "surface_b":     "#0F1217",
    "header_a":      "#151920",   # --surface-2
    "header_b":      "#151920",
}
_TABLE_TOKENS_LIGHT = {
    "ink_primary":   "#141920",
    "ink_tertiary":  "#5E6979",
    "border":        "rgba(15, 23, 42, 0.10)",
    "border_subtle": "rgba(15, 23, 42, 0.05)",
    "accent":        "#2B5FD9",
    "emerald":       "#0F7A54",
    "rose":          "#C0392F",
    "accent_border": "rgba(43, 95, 217, 0.32)",
    "accent_hover":  "rgba(43, 95, 217, 0.07)",
    "row_odd":       "rgba(15, 23, 42, 0.022)",
    "row_even":      "transparent",
    "surface_a":     "#FFFFFF",
    "surface_b":     "#FFFFFF",
    "header_a":      "#EEF1F5",
    "header_b":      "#EEF1F5",
}


def _table_tokens() -> dict:
    """Active-theme token set for the iframe-isolated data table."""
    return _TABLE_TOKENS_LIGHT if st.session_state.get("theme") == "light" else _TABLE_TOKENS_DARK

#: Webfont the iframe must import for itself, for the same isolation reason.
_TABLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=JetBrains+Mono:wght@400;500;600;700&display=swap")


# ═══════════════════════════════════════════════════════════════════════
#  THE CALL — the hero, its grade and its chain
# ═══════════════════════════════════════════════════════════════════════
#
# Everything here formats ONE object, the verdict (core.call.verdict_from_row
# on call_ts at d*): nothing is computed, thresholded or re-derived, so the
# card, the command bar and the rail cannot disagree. The card reads top-down
# as the decision itself: the action and the sentence that explains it, then
# what limits it; how the call has done on the sessions it was issued; then
# the chain that produced it, in decision order, ending in the size's
# arithmetic. The size is a unitless index and is never printed as a '%'.

from core.call import edge_periods, format_size  # noqa: E402
from core.knobs import knob as _knob  # noqa: E402
from core.semantics import edge_tier as _t_tier  # noqa: E402


def edge_tier(edge) -> tuple[str, str]:
    """The label of a grade — one scale, from ``core.semantics.edge_tier``.

    Given a grade dict (``series_edge``'s), its ``(label, tone)`` on the
    grade's own t: |t| < 1 is never an edge, and a grade measured against a
    lean leaves it neutral. Given a bare IC (a float, or None) — the System
    page's engine table, until it is rebuilt — the older ``(tier, chip)`` on
    the IC's own scale.
    """
    if isinstance(edge, dict):
        return _t_tier(edge.get("t"))
    ic = edge
    if ic is None or not np.isfinite(ic):
        return "uncalibrated", "NO READ"
    if ic >= 0.10:
        return "solid", "SOLID EDGE"
    if ic >= 0.05:
        return "modest", "MODEST EDGE"
    if ic > 0:
        return "marginal", "MARGINAL"
    return "no_edge", "NO EDGE"


#: Card class per action: only an issued call is coloured as a direction.
_ACTION_CLASS = {"BUY": "buy", "SELL": "sell"}


def _render_call_grade(cg: dict, horizon: int) -> None:
    """The GRADE→call strip: the call graded on the sessions it was issued."""
    def pct(v):
        return f"{v * 100:+.2f}%" if v is not None else "—"

    ip, hp = cg.get("issued_payoff"), cg.get("held_payoff")
    warm = _knob("GRADE_WARMUP_WIN")
    prospective = cg.get("prospective") or {}
    pooled = cg.get("pooled") or "no pooled class grade"
    if cg.get("graded") and None not in (cg.get("ic"), cg.get("se"), cg.get("t")):
        value, sub = f"{cg['ic']:+.3f}", f"± {cg['se']:.3f} · t {cg['t']:.2f} · {cg['label']}"
    elif cg.get("n_win", 0) < warm:
        value = "not yet graded"
        sub = f"{cg.get('n_win', 0)} of {warm} independent issued windows · {pooled}"
    else:                                            # warmed up, but nothing to rank
        value, sub = "ungraded", f"{cg.get('status', 'ungraded')} · {pooled}"
    render_kpi_strip([
        {"label": "Call grade · as issued", "value": value, "subtext": sub,
         "color_class": "neutral"},
        {"label": "Coverage", "value": f"{cg.get('issued_n', 0)} of {cg.get('claim_n', 0)}",
         "subtext": "claim sessions on which a call was issued", "color_class": "neutral"},
        {"label": "Hit vs base",
         "value": (f"{cg['hit']:.0f}% vs {cg['base']:.0f}%"
                   if cg.get("hit") is not None and cg.get("base") is not None else "—"),
         "subtext": "issued calls, against the side-matched no-skill rate", "color_class": "neutral"},
        {"label": "Issued vs held",
         "value": f"{pct((ip or {}).get('mean'))} / {pct((hp or {}).get('mean'))}",
         "subtext": (f"drift-neutral {horizon}D claim payoff · n/h "
                     f"{(ip or {}).get('n_win', 0)} / {(hp or {}).get('n_win', 0)}"),
         "color_class": "neutral"},
        {"label": "Prospective record", "value": f"{prospective.get('n_win', 0)} windows",
         "subtext": prospective.get("status", ""), "color_class": "neutral"},
    ], max_cols=5, key="call-grade")


def render_hero_card(verdict: dict) -> None:
    """The call, its grade and its chain — Overview's first three blocks.

    1. The CALL: the action word, the size beside it (when a call is issued),
       the headline, what limits it, and any caveat on d*.
    2. GRADE→call: the call graded on the sessions it was issued.
    3. The chain: one row per role in decision order — claim, preconditions,
       trust, each sizer, size — each with its reading, its grade and its
       effect, ending in the size's arithmetic.
    """
    esc = html_mod.escape
    act, h = verdict["action"], int(verdict["horizon_days"])
    cls = _ACTION_CLASS.get(act, "hold")
    size_txt = f"size {format_size(verdict['size'])}"
    formula = verdict["size_formula"] if verdict["direction"] else "no claim to size"
    asof = verdict.get("asof")
    when = f"as of {asof:%d %b %Y}" if asof is not None else "no published session"
    caveat = (f'<div class="hero-risk">{esc(verdict["caveat"])}</div>'
              if verdict.get("caveat") else "")
    st.markdown(
        f"""<div class="hero-card {cls}">
  <div class="hero-top">
    <div class="hero-signal-block">
      <div class="hero-eyebrow">Tattva &bull; {h}-session call from the next close</div>
      <div class="hero-signal">{esc(verdict["word"])}</div>
    </div>
    <div class="hero-conviction-block">
      <span class="hero-chip">{esc(when)}</span>
      <div class="hero-conviction">{esc(size_txt)}</div>
      <div class="hero-conviction-sub">{esc(formula)}</div>
    </div>
  </div>
  <div class="hero-headline">{esc(verdict["headline"])}</div>
  <div class="hero-limit">{esc(verdict["limit"])}</div>
  {caveat}
</div>
""",
        unsafe_allow_html=True,
    )

    _render_call_grade(verdict["call_grade"], h)

    binding = f"sizer:{verdict['binding']}" if verdict.get("binding") else None
    rows = "".join(
        f'<div class="hero-gate{" binding" if r["key"] == binding else ""}">'
        f'<span class="hero-gate-name">{esc(r["role"])}</span>'
        f'<span></span>'
        f'<span class="hero-gate-label">{esc(r["title"])} &middot; {esc(r["reading"])}</span>'
        f'<span class="hero-gate-detail">{esc(r["effect"])}'
        + (f' &middot; {esc(r["grade"])}' if r["grade"] else "")
        + f' &middot; {esc(r["detail"])}</span>'
        f'</div>'
        for r in verdict["chain"]
    )
    st.markdown(
        f'<div class="hero-card {cls}"><div class="hero-eyebrow">The chain &bull; in decision '
        f'order</div><div class="hero-gates">{rows}</div>'
        f'<div class="hero-foot">{esc(verdict["size_formula"])} &mdash; trust is the claim&rsquo;s '
        f'earned base size; a sizer is &times;1.00 until its own grade earns a weight.</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  EDGE — one grading, used by every engine
# ═══════════════════════════════════════════════════════════════════════
#
# Each engine page leads with the same question asked the same way: when this
# engine leaned one way, did the instrument move that way over the horizon?
# The number is the rank IC of the engine's bullish-positive lean against the
# h-day forward return, sampled every h sessions so no two scored windows
# overlap (core.grade.series_edge). One function computes it and
# one component draws it, so "edge" cannot mean four things on four pages.

def edge_summary_text(edge: dict | None, what: str) -> str:
    """One sentence for an edge read — the footer under its chart."""
    if not edge or edge.get("ic") is None or not np.isfinite(edge.get("ic", np.nan)):
        return f"Not enough scored history to grade {what} yet."
    se = edge.get("se")
    pos, nb = edge_periods(edge)
    return (f"{what[0].upper() + what[1:]}: non-overlapping {edge.get('horizon', '?')}-day IC "
            f"{edge['ic']:+.3f}" + (f" ± {se:.3f}" if se is not None and np.isfinite(se) else "")
            + f" over {int(edge.get('n', 0) or 0)} calls"
            + (f"; {pos} of {nb} periods positive." if nb else "."))


def render_edge_panel(edge: dict | None, key: str, *, what: str, show_chart: bool = True) -> None:
    """Edge cards (+ per-period bars) for one engine's lean.

    ``what`` names the thing graded ("Mūla's call", "Swayam's net breadth").
    Renders an explicit empty state rather than nothing when the series is
    too short to grade — "not measured" is information too.
    """
    from ui.theme import chart_layout, style_axes, chart_color, grid_rgba
    ic = (edge or {}).get("ic")
    if ic is None or not np.isfinite(ic):
        render_empty_state(
            "Not yet graded",
            f"{what[0].upper() + what[1:]} has too little scored history for a "
            f"non-overlapping walk-forward read.",
            eyebrow="Edge",
        )
        return
    label, tone = edge_tier(edge)
    se = edge.get("se")
    # Only scored periods are drawn: an unscored one has no IC, and drawing it
    # as an empty bar in the negative colour read as a bad period.
    blocks = [b for b in (edge.get("blocks") or [])
              if b.get("ic") is not None and np.isfinite(b.get("ic"))]
    pos, n_scored = edge_periods(edge)
    render_kpi_strip([
        {"label": "Walk-Forward IC", "value": f"{ic:+.3f}",
         "subtext": (f"{label} · ± {se:.3f} (1 SE)" if se is not None and np.isfinite(se)
                     else label),
         "color_class": tone,
         "tooltip": ("Rank correlation of the lean with the forward return over the "
                     "horizon, scored on non-overlapping windows. 0 = no skill; the SE "
                     "says how much of the number is sampling noise.")},
        {"label": "Positive Periods", "value": f"{pos}/{n_scored}" if n_scored else "—",
         "subtext": "Durability across regimes — one good period is not an edge.",
         "color_class": ("success" if n_scored and pos > n_scored * 0.6
                         else "warning" if n_scored and pos >= n_scored * 0.5 else "danger")},
        {"label": "Calls Scored", "value": f"{int(edge.get('n', 0) or 0)}",
         "subtext": f"one per {edge.get('horizon', '?')} sessions, none overlapping",
         "color_class": "neutral"},
    ], max_cols=3, key=f"edge-{key}")
    if show_chart and blocks:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[f"{str(b.get('start'))[:7]} → {str(b.get('end'))[:7]}" for b in blocks],
            y=[b.get("ic") for b in blocks],
            marker=dict(color=[chart_color("emerald") if (b.get("ic") or 0) > 0
                               else chart_color("rose") for b in blocks]),
            hovertemplate="%{x}<br>IC %{y:+.3f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=grid_rgba(0.10), line_width=0.6)
        fig.update_layout(**chart_layout(height=240, show_legend=False))
        style_axes(fig, y_title="IC")
        render_chart_panel(fig, f"edge-{key}", units="IC per period", windowed=False,
                           footer=html_mod.escape(edge_summary_text(edge, what)))


def _fmt_cell(value, precision: int) -> str:
    """Format one cell value for display (NaN → em dash; floats to `precision`).

    Dates render date-only: Tattva is a DAILY system, so a Timestamp's
    ``00:00:00`` time component is noise — never shown.
    """
    if value is None:
        return "—"
    # Date-only for any datetime-like (pd.Timestamp subclasses datetime.date).
    if isinstance(value, (pd.Timestamp, _dt.date)):
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if value != value:            # NaN
            return "—"
        return f"{value:,.{precision}f}"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return html_mod.escape(str(value))


# Column-name tokens that must stay UPPER-CASE when a raw column name is
# prettified into a professional header ("MSF_Osc" → "MSF Osc", not "Msf Osc").
# (Deliberately NOT including "OSC" — an oscillator column reads more
# professionally as "Osc" than "OSC", matching the source design.)
_HEADER_ACRONYMS = {
    "RSI", "MA", "MSF", "MMR", "VAP", "IC", "HR", "HMM", "GARCH", "CUSUM",
    "ADF", "KPSS", "DDM", "OU", "PCA", "US", "FX", "ID", "N", "T", "Z", "R2",
    "OHLC", "OHLCV", "ATR", "MACD", "EMA", "SMA",
}


def _prettify_header(name: str) -> str:
    """Turn a raw column/field name into a professional table header.

    ``divergence_type`` → ``Divergence Type``; ``MSF_Osc`` → ``MSF Osc``;
    ``Change_Point`` → ``Change Point``; ``val_ic`` → ``Val IC``. Already-clean
    headers ("Buy Avg Δ", "Period") pass through with only per-word acronym
    casing applied.
    """
    raw = str(name).replace("_", " ").strip()
    if not raw:
        return ""
    out = []
    for word in raw.split():
        up = word.upper()
        if up in _HEADER_ACRONYMS:
            out.append(up)
        elif word.isupper() and len(word) <= 4:   # keep short all-caps as-is
            out.append(word)
        elif any(ch.isdigit() for ch in word) and word.isupper():
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
        # Preserve non-alphanumeric tokens verbatim (Δ, %, etc.)
        if not word[:1].isalnum():
            out[-1] = word
    return " ".join(out)


def render_data_table(
    df: "pd.DataFrame",
    *,
    index_label: str | None = None,
    show_index: bool | None = None,
    max_rows: int | None = None,
    precision: int = 2,
    col_precision: dict[str, int] | None = None,
    sign_color_cols: "set[str] | None" = None,
    label_col: str | None = None,
    col_labels: dict[str, str] | None = None,
    max_height: int = 520,
    row_height: int = 27,
) -> None:
    """Render a DataFrame as the app's one institutional table.

    The only table primitive in Tattva — there is no bare ``st.dataframe``
    anywhere, because Streamlit's grid brings its own typeface, row height,
    header treatment and hover, none of which can be reached from the app's
    stylesheet. Sticky muted header, hairline row rules, right-aligned tabular
    numerics, a bolder "label" column, and horizontal/vertical scroll under a
    fixed ``max_height`` — safe on both the 10-row divergence table and the
    full dataset viewer.

    Rows are 27px (was 42): the old height came from 0.6rem cell padding at a
    0.75rem font, which is a comfortable READING density, not a scanning one.
    A table a trader scans should fit twice as many rows in the same panel.

    Wrap it in ``render_table_panel`` rather than calling it directly, so the
    table gets the same header anatomy as every chart.

    Parameters
    ----------
    index_label : shown as the first column header when the index is rendered;
        also forces the index to render.
    show_index : override index rendering (default: auto — shown when the index
        is not a plain 0..N RangeIndex, i.e. it carries dates/labels).
    max_rows : cap to the LAST ``max_rows`` rows (tables are newest-relevant).
    precision / col_precision : default and per-column float precision.
    sign_color_cols : numeric columns whose values are tinted emerald/rose by
        sign (the "signal" colouring from Pragyam's per-signal columns).
    label_col : the column to style as the bold Space-Grotesk label (default:
        the index if shown, else the first column).
    col_labels : explicit header overrides ``{raw_name: display}``; any column
        not listed is auto-prettified (``MSF_Osc`` → ``MSF Osc``).
    """
    if df is None or getattr(df, "empty", True):
        st.markdown('<div class="panel-state">No rows to display.</div>',
                    unsafe_allow_html=True)
        return

    view = df.tail(max_rows).copy() if max_rows else df.copy()
    if isinstance(view.columns, pd.MultiIndex):
        view.columns = [" · ".join(str(x) for x in c) for c in view.columns]

    if show_index is None:
        show_index = index_label is not None or not isinstance(view.index, pd.RangeIndex)
    idx_header = (index_label or _prettify_header(view.index.name or "")) if show_index else ""
    col_labels = col_labels or {}

    def _header(c: str) -> str:
        return col_labels.get(c) or _prettify_header(c)

    cols = list(view.columns)
    numeric_cols = {c for c in cols if pd.api.types.is_numeric_dtype(view[c])}
    sign_cols = (sign_color_cols or set()) & numeric_cols
    col_precision = col_precision or {}
    # The label column: explicit, else the index (when shown), else first column.
    if label_col is None:
        label_col = "__index__" if show_index else (cols[0] if cols else None)

    t = _table_tokens()

    def _header_cells() -> str:
        cells = []
        if show_index:
            cells.append(f'<th class="lbl">{html_mod.escape(str(idx_header))}</th>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            cells.append(f'<th class="{cls}">{html_mod.escape(_header(c))}</th>')
        return "".join(cells)

    def _value_html(c: str, val) -> str:
        p = col_precision.get(c, precision)
        text = _fmt_cell(val, p)
        if c in sign_cols and text != "—":
            try:
                fv = float(val)
                color = (t["emerald"] if fv > 1e-12 else t["rose"] if fv < -1e-12
                         else t["ink_tertiary"])
                return f'<span style="color:{color};font-weight:600;">{text}</span>'
            except (TypeError, ValueError):
                pass
        return text

    body_rows = []
    for idx, row in view.iterrows():
        tds = []
        if show_index:
            tds.append(f'<td class="lbl">{_fmt_cell(idx, precision)}</td>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            tds.append(f'<td class="{cls}">{_value_html(c, row[c])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    n_rows = len(view)
    _HEADER_H = 30                      # sticky header row, matches the CSS above
    content_h = _HEADER_H + n_rows * row_height + 4
    iframe_h = min(content_h, max_height)

    # The iframe cannot see the app's stylesheet, so the design tokens it needs
    # are restated here as literals (see _TABLE_TOKENS_*). Values below mirror
    # theme.css exactly: --fs-2xs header, --fs-xs body, s2/s3 cell padding,
    # hairline rules. Nothing here is a one-off number.
    table_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    @import url('{_TABLE_FONTS}');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    /* JetBrains Mono — the app's data face, and the one _TABLE_FONTS actually
       imports. This said 'IBM Plex Mono' while importing JetBrains, so the
       declared family was never loaded and every table in the app fell through
       to the system default (Menlo/Courier). Tables were the one surface
       rendering in a typeface the rest of the UI does not use. */
    body {{ font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
            background:transparent; color:{t['ink_primary']};
            font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1,"zero" 1; }}
    /* A hair of top padding so the sticky header cannot sit flush against
       the panel header rendered directly above this iframe — the two read as
       one doubled, overlapping header row without it. */
    .tt-scroll {{ padding-top:2px; max-height:{max_height}px; overflow:auto;
                  scrollbar-width:thin; scrollbar-color:{t['ink_tertiary']} transparent; }}
    .tt-scroll::-webkit-scrollbar {{ width:9px; height:9px; }}
    .tt-scroll::-webkit-scrollbar-track {{ background:transparent; }}
    .tt-scroll::-webkit-scrollbar-thumb {{ background:{t['border']}; border-radius:100px; }}
    .tt-scroll::-webkit-scrollbar-thumb:hover {{ background:{t['ink_tertiary']}; }}
    .tt-scroll::-webkit-scrollbar-corner {{ background:transparent; }}
    table {{ width:100%; border-collapse:collapse; }}
    /* Header: muted, uppercase, hairline rule. It was a 2px accent-coloured
       rule with an accent-coloured first cell — the heaviest horizontal line
       in the app, sitting under its quietest content. A header is a label for
       a column, not a claim about it. */
    thead th {{ position:sticky; top:0; z-index:2;
        background:{t['header_a']};
        color:{t['ink_tertiary']}; font-size:0.625rem; font-weight:600;
        text-transform:uppercase; letter-spacing:0.12em; padding:0.5rem 0.75rem;
        border-bottom:1px solid {t['border']}; text-align:left; white-space:nowrap; }}
    thead th.num {{ text-align:right; }}
    /* Row separation is a hairline OR a tint, never both — the two together
       are what made this read as a spreadsheet export. */
    tbody tr {{ border-bottom:1px solid {t['border_subtle']};
                transition:background 120ms cubic-bezier(0.2,0,0,1); }}
    tbody tr:last-child {{ border-bottom:none; }}
    tbody tr:hover {{ background:{t['accent_hover']}; }}
    tbody td {{ padding:0.4rem 0.75rem; color:{t['ink_primary']}; font-size:0.6875rem;
                line-height:1.5; vertical-align:middle; white-space:nowrap; }}
    tbody td.num {{ text-align:right; }}
    tbody td.lbl {{ font-weight:600; color:{t['ink_primary']}; }}
    tbody td.txt {{ color:{t['ink_tertiary']}; }}
    </style></head><body>
    <div class="tt-scroll"><table>
    <thead><tr>{_header_cells()}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    </table></div></body></html>"""

    _components_html(table_html, height=iframe_h, scrolling=False)


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def provisional_note(fvo_ts) -> str:
    """One sentence naming the newest session as unsettled, or "" when it is not.

    The engine separates `publish` (the value is emitted) from `Valid` (the value
    may be trusted), and marks the difference `Provisional`. That distinction was
    introduced for the convergence cards and initially surfaced only there — so
    the valuation chart drew today's forming fair value as an ordinary point and the
    Data tab listed it unlabelled, which is the same misreading the cards were
    fixed to prevent. This is the shared accessor, so every surface that shows a
    number can say whether it has settled.

    Empty string when the newest session is settled, so callers can treat it as
    falsy and render nothing.
    """
    try:
        import pandas as _pd
        import numpy as _np
        if fvo_ts is None:
            return ""
        cols = set(getattr(fvo_ts, "columns", ()))
        if "Provisional" in cols:
            unsettled = bool(fvo_ts["Provisional"].iloc[-1])
        elif "Valid" in cols:
            unsettled = not bool(fvo_ts["Valid"].iloc[-1])
        else:
            return ""
        if not unsettled:
            return ""
        if {"NAvailable", "NScheduled"} <= cols:
            av = float(_pd.to_numeric(fvo_ts["NAvailable"], errors="coerce").iloc[-1])
            sc = float(_pd.to_numeric(fvo_ts["NScheduled"], errors="coerce").iloc[-1])
            if _np.isfinite(av) and _np.isfinite(sc) and sc > 0:
                return (f"Latest session is still forming — {int(av)} of {int(sc)} "
                        f"scheduled instruments have printed. Its reading will move "
                        f"until the session settles; earlier dates are final.")
        return ("Latest session is still forming. Its reading will move until the "
                "session settles; earlier dates are final.")
    except (IndexError, KeyError, TypeError, ValueError):
        return ""
