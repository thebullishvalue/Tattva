"""
Tattva — Reusable UI components: metric cards, signal badges, headers, section headers.
तत्त्व (Tattva) — "Principle / Essence"

UI — Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import datetime as _dt
import html as html_mod

import pandas as pd
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


def get_icon(name: str, size: int = 18, stroke_width: float = 1.5) -> str:
    """Return an SVG icon string with custom size and stroke width."""
    import re
    base_svg = ICONS.get(name, ICONS["chart"])

    # Clean existing attributes to avoid duplicates or stale values
    base_svg = re.sub(r'\s+width="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+height="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+stroke-width="[^"]*"', '', base_svg)

    # Inject standardized attributes
    return base_svg.replace('<svg', f'<svg width="{size}" height="{size}" stroke-width="{stroke_width}"')


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
    svg = get_icon(icon, size=16, stroke_width=1.8)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f'{desc_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_gap() -> None:
    """Insert vertical spacing between major sections."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def render_control_hint(text: str) -> None:
    """Render the canonical terse helper caption beneath a control.

    This is the single source of truth for the "sub-control hint" tier — the
    uppercase micro-caption used by e.g. the "Nirnay basket · producer
    cross-section" and Signal-Horizon hints. Use it instead of ``st.caption``
    for control helper text so the sidebar/tab fine-print stays one coherent
    visual hierarchy. Keep the text terse and ``·``-separated.
    """
    st.markdown(
        f'<div class="control-hint">{html_mod.escape(text)}</div>',
        unsafe_allow_html=True,
    )


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
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12, stroke_width=2)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{icon_html}{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )




def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead."""
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
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


def render_nishkarsh_signal_card(
    signal: str,
    conviction: float,
    agreement: float,
    explanation: str,
    val_ic: float | None = None,
    wf_pos: float | None = None,
    source: str | None = None,
) -> None:
    """DEPRECATED shim — superseded by build_hero_verdict + render_hero_card.

    Kept only so an out-of-tree caller doesn't break; the live app no longer
    calls this. Renders a minimal legacy card.
    """
    verdict = build_hero_verdict(
        calib_conviction=None,
        calib_signal=None,
        has_profile=val_ic is not None,
        consensus={"value": conviction, "signal": signal},
        aarambh_signal={},
        agreement=agreement,
        val_ic=val_ic,
        wf_pos=wf_pos,
        precedent=None,
        n_divergences=0,
        horizon_days=10,
    )
    render_hero_card(verdict)


# ── Hero verdict: pure interpretation logic (no Streamlit) ──────────────────

# Trust tiers for the NON-OVERLAPPING Val IC (see convergence.intelligence.
# _score_frame_nonoverlap): its effective sample size is ~n_val / stride, so at
# a realistic holdout (~300-500 days) and the longest scoring stride (10),
# n_eff ~ 30-50 → SE(IC) ~ 1/sqrt(n_eff-3) ~ 0.15-0.19. The tier cut-points
# below (0.10 / 0.20) sit at roughly 1 SE (MODEST) and 2 SE (SOLID) for that
# worst-case n_eff so the chip is not overconfident.
def _trust_tier(
    val_ic: float | None,
    wf_pos: float | None,
    wf_n: int | None = None,
) -> dict:
    """Map a non-overlapping Val IC to a (tier, chip label, prose) bundle.

    ``wf_n``: total walk-forward windows behind ``wf_pos`` — a bare ratio
    ("67% of windows positive") hides that it might be 2/3 windows, which is
    not durability evidence; with the count the reader can weigh it.
    """
    if val_ic is None:
        return {"tier": "uncalibrated", "chip": "UNCALIBRATED",
                "val_ic": None, "wf_pos": wf_pos, "wf_n": wf_n,
                "prose": "Edge not yet calibrated (run Intelligence Mode for a Val IC)."}
    if val_ic <= 0:
        tier, chip = "no_edge", "NO EDGE"
        prose = f"No validated edge (Val IC {val_ic:+.3f}) — treat the direction as noise."
    elif val_ic < 0.10:
        tier, chip = "marginal", "MARGINAL"
        prose = f"Marginal edge (Val IC {val_ic:+.3f}) — within ~1 SE of zero at this sample size."
    elif val_ic < 0.20:
        tier, chip = "modest", "MODEST EDGE"
        prose = f"Modest validated edge (Val IC {val_ic:+.3f})."
    else:
        tier, chip = "solid", "SOLID EDGE"
        prose = f"Solid validated edge (Val IC {val_ic:+.3f})."
    if wf_pos is not None:
        if wf_n:
            prose += f" Walk-forward: {round(wf_pos * wf_n)}/{wf_n} windows positive."
        else:
            prose += f" Walk-forward: {wf_pos:.0%} of windows positive."
    return {"tier": tier, "chip": chip, "val_ic": float(val_ic),
            "wf_pos": wf_pos, "wf_n": wf_n, "prose": prose}

# Minimum number of DISTINCT analogs (post-Theiler-exclusion) before the
# precedent's direction is treated as probative. Below this, "% positive" is
# a handful of coin flips — the hero must not claim agreement/divergence on it.
_PRECEDENT_MIN_N = 5

# Precedent hit-rate band around 50% inside which the base rate is called
# SPLIT (no lean). 15pp on n>=5 analogs: even at n=10, 65% positive is 6-7 of
# 10 — barely one analog away from a coin flip, so anything inside the band
# must not be sold as a lean.
_PRECEDENT_SPLIT_BAND = 15.0

# Flat band on the [-1, +1] signal scale: below this magnitude a reading is
# treated as "flat" rather than a lean. Used for (a) the neutral headline's
# lean wording and (b) the TREND row's noise gate — a raw-vs-smoothed sign
# disagreement where both magnitudes are inside this band is indistinguishable
# from noise and must not be escalated to a "trend contradiction".
# On the post-continuous-consensus composite this sits at ≈ p42 of pooled
# |composite| (4 real targets, 2026-07-11 measurement: p25=0.024, p50=0.062)
# — i.e. the quietest ~40% of days read as "flat", which is the intent.
_FLAT_BAND = 0.05

_NEUTRAL_LABELS = {"HOLD", "NEUTRAL", "N/A", ""}

# ── Decision synthesis: weigh ALL evidence rows into an action tier ─────────
# The HEADLINE stays the normalized consensus (a reconciliation invariant:
# hero = Unified Signal plot top row = TATTVA CONVICTION card), but the ACTION
# the card recommends must NOT be read off that raw value alone — before this
# layer existed, a CALIBRATED row could say "stand aside" in small print while
# the headline still shouted BUY. Each evidence row carries a signed weight;
# their sum (on top of a trust-tier base) maps to a decision tier. Weights are
# ordinal judgments, documented so they can be challenged, not fitted numbers:
#
#   MODEL       base: solid +2 / modest +1 / marginal · uncalibrated 0;
#               no_edge is a HARD GATE (validated no edge → direction is
#               noise, no amount of soft evidence overrides that).
#   CALIBRATED  confirm +1 / not-confirmed (neutral) -1 / conflict -2 —
#               same two engines under learned weights; the strongest single
#               second opinion, and the ONLY row whose variant actually
#               earned the Val IC.
#   TREND       confirm +1 / conflict -1 — one-day print vs its own DDM
#               trend; a contradiction may be an early turn, so it warns
#               but never dominates.
#   PRECEDENT   confirm +1 / conflict -2 — hero_study.py found the analog
#               base rate historically the STRONGER directional read, so a
#               coherent divergence outweighs a trend wobble.
#   INTERNALS   confirm +1 / conflict -2 — engines split means the consensus
#               mean straddles zero: the headline's own construction is
#               undermined, not merely contradicted.
#   RISK        -1 — recent flagged divergence events.
#
# Tier map: net >= +3 HIGH · +1..+2 MODERATE · 0..-1 LOW · <= -2 STAND ASIDE.
# Cap: without a validated edge (uncalibrated / marginal / no wf evidence)
# the tier is capped at MODERATE — soft confirmations can never promote an
# unvalidated signal to full-size conviction.
_ACTION_WEIGHTS: dict[tuple[str, str], int] = {
    ("CALIBRATED", "confirm"): +1, ("CALIBRATED", "neutral"): -1,
    ("CALIBRATED", "conflict"): -2,
    ("TREND", "confirm"): +1, ("TREND", "conflict"): -1,
    ("PRECEDENT", "confirm"): +1, ("PRECEDENT", "conflict"): -2,
    ("INTERNALS", "confirm"): +1, ("INTERNALS", "conflict"): -2,
    ("RISK", "conflict"): -1,
}

_MODEL_BASE = {"solid": 2, "modest": 1, "marginal": 0, "uncalibrated": 0}

_ACTION_TIERS = {
    "high": ("HIGH CONVICTION", "evidence stack supports acting at plan size."),
    "moderate": ("MODERATE CONVICTION", "act at reduced size."),
    "low": ("LOW CONVICTION", "small size or paper-trade only."),
    "stand_aside": ("STAND ASIDE", "opposing evidence outweighs the signal — do not act on the headline."),
    "none": ("NO ACTION", "headline is HOLD — nothing to size."),
}


def _synthesize_action(direction: str, trust: dict, evidence: list[dict]) -> dict:
    """Fold the trust tier and every evidence row into one decision tier.

    Pure and deterministic — the returned ``drivers`` string itemises each
    contribution so the tier is auditable from the card itself. Returns
    ``{level, label, prose, score, drivers}``; ``score`` is the net evidence
    sum (None for the ``none`` / hard-gated paths where no sum was taken).
    """
    if direction == "neutral":
        label, prose = _ACTION_TIERS["none"]
        return {"level": "none", "label": label, "prose": prose,
                "score": None, "drivers": ""}

    tier = trust["tier"]
    if tier == "no_edge":
        label, _ = _ACTION_TIERS["stand_aside"]
        return {"level": "stand_aside", "label": label,
                "prose": ("validated NO EDGE — the direction is noise "
                          "regardless of the other evidence rows."),
                "score": None, "drivers": f"no-edge gate (Val IC {trust['val_ic']:+.3f})"}

    score = _MODEL_BASE.get(tier, 0)
    drivers = [f"{tier} edge {score:+d}"] if score else [f"{tier} edge +0"]
    for row in evidence:
        if row["tag"] == "MODEL":
            continue
        w = _ACTION_WEIGHTS.get((row["tag"], row["state"]))
        if w:
            score += w
            drivers.append(f"{row['tag'].lower()} {row['state']} {w:+d}")

    if score >= 3:
        level = "high"
    elif score >= 1:
        level = "moderate"
    elif score >= 0:
        level = "low"
    else:
        level = "stand_aside"

    capped = False
    if level == "high" and tier in ("uncalibrated", "marginal"):
        level, capped = "moderate", True

    label, prose = _ACTION_TIERS[level]
    drivers_str = " · ".join(drivers) + f" → net {score:+d}"
    if capped:
        drivers_str += " (capped: edge not validated)"
    return {"level": level, "label": label, "prose": prose,
            "score": score, "drivers": drivers_str}


def build_hero_verdict(
    *,
    calib_conviction: float | None,
    calib_signal: str | None,
    has_profile: bool,
    consensus: dict | None,
    aarambh_signal: dict,
    agreement: float,
    val_ic: float | None,
    wf_pos: float | None,
    precedent: dict | None,
    n_divergences: int,
    horizon_days: int,
    agreement_strong: float = 0.7,
    agreement_moderate: float = 0.5,
    smoothed: float | None = None,
    wf_n: int | None = None,
    div_window: int | None = None,
) -> dict:
    """Assemble the hero card's verdict — ALL interpretation logic, pure data in/out.

    Separated from rendering so the decision rules (label normalisation, trust
    tiering, the precedent gates, agreement tiers, trend comparison) are
    unit-testable without Streamlit (see research/test_hero_verdict.py).
    Returns a dict the renderer consumes verbatim:

    ``signal`` (display label), ``signal_class`` (css), ``score`` ([-1,+1]),
    ``source``, ``direction`` (bullish/bearish/neutral), ``headline`` (one
    sentence), ``trust`` ({tier, chip, val_ic, wf_pos, wf_n}),
    ``consensus_score`` (float|None), ``evidence`` (ordered rows of
    {tag, state, text}; state ∈ confirm|conflict|neutral|info), and
    ``action`` ({level, label, prose, score, drivers}) — the DECISION
    synthesis that folds the trust tier and every evidence row into the tier
    the card actually recommends (see _synthesize_action). The headline names
    the direction; the action names what to do about it.

    HEADLINE = THE NORMALIZED CONSENSUS — a product decision: the headline is
    the causal expanding-z average of Aarambh's ConvictionRaw and Nirnay's
    Avg_Signal (``consensus`` dict, from
    convergence.normalization.compute_normalized_convergence), classified
    with its OWN factory p75/p90-anchored thresholds (DEFAULT_THRESHOLDS). This is the SAME object as the
    Unified Signal plot's top row and the TATTVA CONVICTION card, so hero,
    card and plot reconcile identically by construction — no fitted layer
    between the engines and the verdict, and no cross-object reconciliation
    gap. The Optuna-calibrated composite (a DIFFERENT construction of the
    same two engines) is the CALIBRATED evidence row (agrees / disagrees /
    not-confirmed), and the trust chip's Val IC — which was EARNED by that
    calibrated composite, not by the consensus — is explicitly attributed to
    it in the MODEL row. The raw factory-weight composite is no longer
    surfaced here at all; it remains the research baseline in
    research/calibration_lift_study.py.

    ``consensus``: pass None when the convergence is DEGENERATE (no
    Aarambh∩Nirnay overlap) — the chain then falls through to the honest
    "Aarambh only" source instead of claiming a two-engine convergence that
    structurally does not exist.

    ``smoothed``: the DDM-filtered value of the SAME consensus series
    ([-1,+1]) — the trend behind today's print, so the TREND row can say
    whether the print extends, softens, or contradicts it instead of leaving
    that to the hero-history plot.
    """
    # ── 1. Headline object: NORMALIZED CONSENSUS → Aarambh-only ───────────
    if consensus:
        score = float(consensus.get("value", 0.0))
        raw_label = str(consensus.get("signal", "HOLD"))
        source = "Convergence consensus (normalized)"
        headline_is_convergence = True
    else:
        score = float(aarambh_signal.get("conviction_score", 0)) / 100.0
        raw_label = str(aarambh_signal.get("signal", "HOLD"))
        source = "Aarambh only (no bottom-up convergence)"
        headline_is_convergence = False

    # ── 2. Label normalisation (the DDM classifier says NEUTRAL, the
    # normalized-consensus classifier says HOLD — one display vocabulary) ───
    label_up = raw_label.upper()
    if label_up in _NEUTRAL_LABELS:
        signal, direction = "HOLD", "neutral"
    elif "BUY" in label_up:
        signal, direction = raw_label, "bullish"
    elif "SELL" in label_up:
        signal, direction = raw_label, "bearish"
    else:
        signal, direction = "HOLD", "neutral"

    # Sign-coherence guard: under the system-wide convention (negative score =
    # bullish; every classifier is constructed with buy thresholds < 0 < sell
    # thresholds), a bullish label with a positive score (or vice versa) is
    # impossible from any legitimate caller — it means a contract violation
    # upstream (e.g. a future classifier change that flips convention without
    # updating this card). Refusing to display the contradiction beats
    # propagating it: neutralise the verdict and say why, loudly.
    if (direction == "bullish" and score > 0) or (direction == "bearish" and score < 0):
        signal, direction = "HOLD", "neutral"
        source += " · sign-convention mismatch"

    signal_class = ("undervalued" if direction == "bullish"
                    else "overvalued" if direction == "bearish" else "fair")

    # ── 3. Trust tier ───────────────────────────────────────────────────────
    trust = _trust_tier(val_ic, wf_pos, wf_n=wf_n)

    # ── 4. Headline sentence ────────────────────────────────────────────────
    # A neutral label does NOT mean the score is flat — it means the score is
    # inside the calibrated HOLD band. Saying "no directional edge" at e.g.
    # +0.28 (just under a +0.30 SELL threshold) overstates flatness; say what
    # is actually true: inside the band, with the lean named when it exists.
    if direction == "neutral":
        if abs(score) < _FLAT_BAND:
            headline = (f"{source}: {score:+.2f} — flat, no directional lean over "
                        f"the next ~{horizon_days} trading days right now.")
        else:
            lean = "bullish" if score < 0 else "bearish"
            headline = (f"{source}: {score:+.2f} — inside the HOLD band over the "
                        f"next ~{horizon_days} trading days (leaning {lean}, but below the "
                        f"action threshold).")
    else:
        headline = (f"{source} reads {direction} ({signal}, {score:+.2f}) over "
                    f"the next ~{horizon_days} trading days.")

    # ── 5. Evidence rows ────────────────────────────────────────────────────
    evidence: list[dict] = []

    # Will a CALIBRATED evidence row exist? Gated on has_profile: with no
    # profile, "calibrated" was classified with factory thresholds anyway —
    # a near-duplicate of the headline that would add noise, not evidence.
    _calib_row = (has_profile and calib_conviction is not None
                  and calib_signal is not None)

    # MODEL — validated edge. (The Val IC / walk-forward numbers were earned
    # by the CALIBRATED composite — the nearest available validation evidence
    # for the consensus headline; the explicit attribution note was dropped
    # from the card copy as noise.)
    model_state = ("confirm" if trust["tier"] in ("modest", "solid")
                   else "conflict" if trust["tier"] == "no_edge" else "neutral")
    evidence.append({"tag": "MODEL", "state": model_state, "text": trust["prose"]})

    # CALIBRATED — the Optuna-calibrated composite (a DIFFERENT construction
    # of the same two engines), as a second opinion on the consensus
    # headline. Same label-normalisation rules as the headline so the
    # comparison is direction-vs-direction, not string-vs-string.
    if _calib_row:
        c_score = float(calib_conviction) / 100.0
        c_up = str(calib_signal).upper()
        if c_up in _NEUTRAL_LABELS or not ("BUY" in c_up or "SELL" in c_up):
            c_dir = "neutral"
        elif "BUY" in c_up:
            c_dir = "bullish"
        else:
            c_dir = "bearish"
        _c_stub = f"{str(calib_signal)} ({c_score:+.2f}) under learned weights/thresholds"
        if c_dir == direction and c_dir != "neutral":
            evidence.append({"tag": "CALIBRATED", "state": "confirm",
                             "text": f"Calibrated variant agrees — {_c_stub}."})
        elif c_dir == "neutral" and direction != "neutral":
            evidence.append({"tag": "CALIBRATED", "state": "neutral",
                             "text": (f"Calibrated variant reads {_c_stub} — the consensus "
                                      f"{direction} read is NOT confirmed by calibration; "
                                      f"treat it as lower-conviction.")})
        elif c_dir != "neutral" and direction == "neutral":
            evidence.append({"tag": "CALIBRATED", "state": "info",
                             "text": (f"Calibrated variant leans {c_dir} — {_c_stub} — "
                                      f"while the consensus read is neutral.")})
        elif c_dir == "neutral" and direction == "neutral":
            evidence.append({"tag": "CALIBRATED", "state": "neutral",
                             "text": f"Calibrated variant also neutral — {_c_stub}."})
        else:
            evidence.append({"tag": "CALIBRATED", "state": "conflict",
                             "text": (f"Calibrated variant DISAGREES — {_c_stub}; consensus and "
                                      f"calibrated composite point opposite ways. Stand aside "
                                      f"or size down until they re-align.")})

    # TREND — today's print vs the DDM-smoothed trend of the SAME consensus
    # series. Only meaningful on the convergence path (the smoothed series is
    # the DDM of the consensus; comparing it against an Aarambh fallback
    # score would cross two different objects).
    if headline_is_convergence and smoothed is not None:
        sm = float(smoothed)
        raw_flat, sm_flat = abs(score) < _FLAT_BAND, abs(sm) < _FLAT_BAND
        if raw_flat and sm_flat:
            pass  # both flat — nothing to interpret, no row
        elif sm_flat:
            evidence.append({"tag": "TREND", "state": "neutral",
                             "text": (f"No established trend (smoothed {sm:+.2f}) — today's "
                                      f"{score:+.2f} is fresh evidence, not yet a trend.")})
        elif raw_flat:
            evidence.append({"tag": "TREND", "state": "neutral",
                             "text": (f"Today's print ({score:+.2f}) has gone flat against a "
                                      f"{'bullish' if sm < 0 else 'bearish'} smoothed trend "
                                      f"({sm:+.2f}) — possible stall.")})
        elif (score < 0) == (sm < 0):
            if abs(score) >= abs(sm):
                evidence.append({"tag": "TREND", "state": "confirm",
                                 "text": (f"Today's print ({score:+.2f}) extends the smoothed "
                                          f"trend ({sm:+.2f}) — signal strengthening.")})
            else:
                evidence.append({"tag": "TREND", "state": "neutral",
                                 "text": (f"Today's print ({score:+.2f}) is softer than the "
                                          f"smoothed trend ({sm:+.2f}) — same direction, "
                                          f"fading intensity.")})
        else:
            evidence.append({"tag": "TREND", "state": "conflict",
                             "text": (f"Today's print ({score:+.2f}) contradicts the smoothed "
                                      f"trend ({sm:+.2f}) — an early turn or one-day noise; "
                                      f"wait for confirmation.")})

    # PRECEDENT — non-parametric base rate, gated on a real sample AND on
    # internal coherence: the lean is only probative when the median return
    # and the hit-rate majority point the same way. A median of -0.4% with
    # 65% positive (few large losers, many small winners) is a skewed-outcome
    # distribution, not a directional lean — calling it "bearish" and then
    # flagging a false divergence against a bullish headline would be worse
    # than saying nothing.
    if precedent and int(precedent.get("n") or 0) >= 1:
        p_n = int(precedent["n"])
        p_med = float(precedent["median"])
        p_pos = float(precedent["positive_pct"])
        p_h = int(precedent["horizon"])
        p_dir = int(precedent.get("dir") or 0)
        p_majority = 1 if p_pos > 50 else -1 if p_pos < 50 else 0
        p_word = "bullish" if p_dir > 0 else "bearish" if p_dir < 0 else "flat"
        hero_sign = 1 if direction == "bullish" else -1 if direction == "bearish" else 0
        stub = (f"{p_n} distinct analogs at +{p_h}d: {p_med:+.1f}% median, "
                f"{p_pos:.0f}% positive")
        if p_n < _PRECEDENT_MIN_N:
            evidence.append({"tag": "PRECEDENT", "state": "neutral",
                             "text": f"Thin sample — only {stub}. Not probative; ignore the lean."})
        elif abs(p_pos - 50) < _PRECEDENT_SPLIT_BAND:
            evidence.append({"tag": "PRECEDENT", "state": "neutral",
                             "text": f"Split — {stub}. No directional lean either way."})
        elif p_dir != 0 and p_majority != 0 and p_dir != p_majority:
            evidence.append({"tag": "PRECEDENT", "state": "neutral",
                             "text": (f"Mixed — {stub}. Median and hit-rate point opposite "
                                      f"ways (skewed outcomes), so the base rate has no "
                                      f"robust lean; don't read direction into it.")})
        elif hero_sign == 0:
            evidence.append({"tag": "PRECEDENT", "state": "info",
                             "text": f"Leans {p_word} — {stub}. The model itself reads no edge."})
        elif p_dir == hero_sign:
            evidence.append({"tag": "PRECEDENT", "state": "confirm",
                             "text": f"Agrees — {stub}, confirming the {direction} read."})
        else:
            evidence.append({"tag": "PRECEDENT", "state": "conflict",
                             "text": (f"Diverges ({p_word}) — {stub}. The analog base rate has "
                                      f"historically been the stronger directional read; treat "
                                      f"the {direction} signal with caution.")})

    # INTERNALS — Aarambh vs Nirnay alignment (the headline's own two
    # contributions — the consensus IS their 50/50 mean, so no separate
    # reconciliation sentence is needed anymore: headline and consensus are
    # the same object by construction).
    consensus_score: float | None = None
    a_norm = consensus.get("aarambh_norm") if consensus else None
    n_norm = consensus.get("nirnay_norm") if consensus else None
    if a_norm is not None and n_norm is not None:
        consensus_score = float(consensus.get("value", 0.0))
        aligned = (a_norm < 0) == (n_norm < 0)
        agree_tier = ("strong" if agreement > agreement_strong
                      else "moderate" if agreement > agreement_moderate else "weak")
        state = "confirm" if (aligned and agreement > agreement_strong) \
            else "conflict" if not aligned else "neutral"
        evidence.append({
            "tag": "INTERNALS", "state": state,
            "text": (f"Aarambh {a_norm:+.2f} / Nirnay {n_norm:+.2f} — "
                     f"{agree_tier} agreement ({agreement:.0%}), "
                     + ("engines aligned." if aligned else "engines split.")),
        })

    # RISK — recent flagged divergence events (the caller windows the count to
    # ~div_window trading days; a bare all-history count reads as a permanent
    # alarm — audit finding F7). Points at the Convergence tab's "Recent
    # Divergences" section, which renders the actual events.
    if n_divergences:
        _win = f" in the last ~{div_window} trading days" if div_window else ""
        evidence.append({"tag": "RISK", "state": "conflict",
                         "text": (f"{n_divergences} divergence event"
                                  f"{'s' if n_divergences != 1 else ''}{_win} — "
                                  f"see Recent Divergences on the Convergence tab "
                                  f"before acting.")})

    # ── 6. Decision synthesis — the tier the card RECOMMENDS ───────────────
    # The headline above is the raw consensus read; this folds MODEL,
    # CALIBRATED, TREND, PRECEDENT, INTERNALS and RISK into the action tier,
    # so the card's bottom line is an integrated judgment, not the raw value.
    action = _synthesize_action(direction, trust, evidence)

    return {
        "signal": signal, "signal_class": signal_class, "score": score,
        "source": source, "direction": direction, "headline": headline,
        "trust": trust, "consensus_score": consensus_score, "evidence": evidence,
        "action": action,
    }


def render_hero_card(verdict: dict) -> None:
    """Render the hero verdict as a structured signal card.

    Layout (top → bottom): eyebrow label · source · signal + score ·
    trust chip · headline · evidence rows. Every text field is escaped here
    (plain text in, HTML out) — the old card passed markdown ``**bold**``
    through ``html.escape`` and showed literal asterisks.
    """
    trust = verdict["trust"]
    tier = trust["tier"]
    chip_style = {
        "uncalibrated": ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"),
        "no_edge":      ("#FB7185", "rgba(251,113,133,0.12)"),
        "marginal":     ("#D4A853", "rgba(212,168,83,0.12)"),
        "modest":       ("#34D399", "rgba(52,211,153,0.12)"),
        "solid":        ("#34D399", "rgba(52,211,153,0.18)"),
    }.get(tier, ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"))
    ic_text = (f"Val IC {trust['val_ic']:+.3f}" if trust.get("val_ic") is not None
               else "Val IC —")
    # Prefer explicit window counts over a bare percentage ("WF 4/6+" beats
    # "WF 67%+"): with only a handful of walk-forward windows the ratio alone
    # overstates the durability evidence (see _trust_tier's wf_n docstring).
    if trust.get("wf_pos") is not None and trust.get("wf_n"):
        wf_text = f" &bull; WF {round(trust['wf_pos'] * trust['wf_n'])}/{trust['wf_n']}+"
    elif trust.get("wf_pos") is not None:
        wf_text = f" &bull; WF {trust['wf_pos']:.0%}+"
    else:
        wf_text = ""
    chip = (
        f'<span class="hero-chip" style="background:{chip_style[1]};color:{chip_style[0]};">'
        f'{html_mod.escape(trust["chip"])} &bull; {ic_text}{wf_text}</span>'
    )

    rows_html = "".join(
        f'<div class="hero-evidence-row {html_mod.escape(r["state"])}">'
        f'<span class="hero-tag"><span class="dot"></span>{html_mod.escape(r["tag"])}</span>'
        f'<span class="hero-text">{html_mod.escape(r["text"])}</span>'
        f'</div>'
        for r in verdict["evidence"]
    )

    # DECISION line — the integrated tier from _synthesize_action, rendered
    # BELOW the evidence it is derived from. ``drivers`` itemises every
    # contribution so the tier is auditable from the card itself.
    action = verdict.get("action")
    action_html = ""
    if action:
        _drivers = (f'<span class="hero-action-drivers">'
                    f'{html_mod.escape(action["drivers"])}</span>'
                    if action.get("drivers") else "")
        action_html = (
            f'<div class="hero-action {html_mod.escape(action["level"])}">'
            f'<span class="hero-action-label">{html_mod.escape(action["label"])}</span>'
            f'<span class="hero-action-prose">{html_mod.escape(action["prose"])}</span>'
            f'{_drivers}</div>'
        )

    st.markdown(
        f"""
        <div class="signal-card {html_mod.escape(verdict["signal_class"])}">
            <div class="label">TATTVA CONVERGENCE SIGNAL &#40;&#x0924;&#x0924;&#x094D;&#x0924;&#x094D;&#x0935;&#41;</div>
            <div class="hero-source">{html_mod.escape(verdict["source"])}</div>
            <div class="value">{html_mod.escape(verdict["signal"])}
                <span class="hero-score">{verdict["score"]:+.2f}</span>
            </div>
            <div class="hero-chip-row">{chip}</div>
            <div class="hero-headline">{html_mod.escape(verdict["headline"])}</div>
            <div class="hero-evidence">{rows_html}</div>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Signal data table (Obsidian Quant — ported from Pragyam's Position Guide) ──
# Rendered via components.html (an IFRAME), which CANNOT see the app's theme.css
# :root variables — so every token below is the RESOLVED value from theme.css,
# inlined into the iframe's own <style>. Keep these in sync with theme.css if the
# palette ever changes (they are the same Obsidian-Quant tokens: amber #D4A853,
# IBM Plex Mono / Space Grotesk, glass surfaces).
_TABLE_TOKENS = {
    "ink_primary":   "#F1F5F9",
    "ink_secondary": "#94A3B8",
    "ink_tertiary":  "#5B6675",   # mirrors theme.css --ink-tertiary (3.31:1 muted tier)
    "amber":         "#D4A853",
    "emerald":       "#2DD4A8",
    "rose":          "#E8555A",
    "orange":        "#F59E0B",
    "border":        "rgba(255,255,255,0.05)",
    "border_subtle": "rgba(255,255,255,0.03)",
    "amber_border":  "rgba(212,168,83,0.30)",
    "amber_hover":   "rgba(212,168,83,0.05)",
    "row_odd":       "rgba(255,255,255,0.01)",
    "row_even":      "rgba(255,255,255,0.005)",
}

_TABLE_FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;"
                "500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap")


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
    row_height: int = 42,
) -> None:
    """Render a DataFrame as an Obsidian-Quant signal table (Pragyam design).

    A theme-faithful replacement for ``st.dataframe`` across Tattva: a rounded
    glass card, uppercase amber-ruled header (sticky on scroll), zebra rows with
    an amber hover, right-aligned tabular numerics, and a bolder first "label"
    column. Wide tables scroll horizontally; long tables scroll vertically under
    a fixed ``max_height`` — so it is safe on both the 10-row divergence table
    and the full dataset viewer.

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
        st.caption("No rows to display.")
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

    t = _TABLE_TOKENS

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
    content_h = 44 + n_rows * row_height + 28
    iframe_h = min(content_h, max_height + 28)

    table_html = f"""<!DOCTYPE html><html><head><style>
    @import url('{_TABLE_FONTS}');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'IBM Plex Mono',monospace; background:transparent;
            color:{t['ink_primary']}; padding:2px; }}
    .tt-wrap {{ border-radius:10px; overflow:hidden; border:1px solid {t['border']};
                background:linear-gradient(145deg,rgba(17,24,39,0.45) 0%,rgba(17,24,39,0.40) 100%); }}
    /* Scrollbar matched to theme.css's global scroller (5px · ink-tertiary thumb ·
       transparent track · 3px radius) so the table scrolls like the plots do. */
    .tt-scroll {{ max-height:{max_height}px; overflow:auto;
                  scrollbar-width:thin; scrollbar-color:{t['ink_tertiary']} transparent; }}
    .tt-scroll::-webkit-scrollbar {{ width:5px; height:5px; }}
    .tt-scroll::-webkit-scrollbar-track {{ background:transparent; }}
    .tt-scroll::-webkit-scrollbar-thumb {{ background:{t['ink_tertiary']}; border-radius:3px; }}
    .tt-scroll::-webkit-scrollbar-corner {{ background:transparent; }}
    table {{ width:100%; border-collapse:collapse; }}
    thead th {{ position:sticky; top:0; z-index:2;
        background:linear-gradient(180deg,rgba(10,14,23,0.98) 0%,rgba(10,14,23,0.92) 100%);
        color:{t['ink_tertiary']}; font-size:0.62rem; font-weight:600;
        text-transform:uppercase; letter-spacing:0.1em; padding:0.7rem 0.75rem;
        border-bottom:2px solid {t['amber_border']}; text-align:left; white-space:nowrap; }}
    thead th.num {{ text-align:right; }}
    tbody tr {{ border-bottom:1px solid {t['border_subtle']}; transition:background 0.15s ease; }}
    tbody tr:nth-child(odd) {{ background:{t['row_odd']}; }}
    tbody tr:nth-child(even) {{ background:{t['row_even']}; }}
    tbody tr:hover {{ background:{t['amber_hover']}; }}
    tbody td {{ padding:0.6rem 0.75rem; color:{t['ink_primary']}; font-size:0.75rem;
                vertical-align:middle; white-space:nowrap; }}
    tbody td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    tbody td.lbl {{ font-family:'Space Grotesk',sans-serif; font-weight:600;
                    font-size:0.76rem; letter-spacing:0.02em; color:{t['ink_primary']}; }}
    thead th.lbl {{ color:{t['amber']}; }}
    </style></head><body>
    <div class="tt-wrap"><div class="tt-scroll"><table>
    <thead><tr>{_header_cells()}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    </table></div></div></body></html>"""

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


