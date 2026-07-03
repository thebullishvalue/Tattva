"""
Tattva — Reusable UI components: metric cards, signal badges, headers, section headers.
तत्त्व (Tattva) — "Principle / Essence"

UI — Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import html as html_mod

import streamlit as st


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
        calib_conviction=conviction * 100.0,
        calib_signal=signal,
        has_profile=val_ic is not None,
        consensus=None,
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
# a realistic holdout (~300-500 days) and the Positional lens (stride=10),
# n_eff ~ 30-50 → SE(IC) ~ 1/sqrt(n_eff-3) ~ 0.15-0.19. The tier cut-points
# below (0.10 / 0.20) sit at roughly 1 SE (MODEST) and 2 SE (SOLID) for that
# worst-case n_eff so the chip is not overconfident.
def _trust_tier(val_ic: float | None, wf_pos: float | None) -> dict:
    """Map a non-overlapping Val IC to a (tier, chip label, prose) bundle."""
    if val_ic is None:
        return {"tier": "uncalibrated", "chip": "UNCALIBRATED",
                "val_ic": None, "wf_pos": wf_pos,
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
        prose += f" Walk-forward: {wf_pos:.0%} of windows positive."
    return {"tier": tier, "chip": chip, "val_ic": float(val_ic),
            "wf_pos": wf_pos, "prose": prose}

# Minimum number of DISTINCT analogs (post-Theiler-exclusion) before the
# precedent's direction is treated as probative. Below this, "% positive" is
# a handful of coin flips — the hero must not claim agreement/divergence on it.
_PRECEDENT_MIN_N = 5

_NEUTRAL_LABELS = {"HOLD", "NEUTRAL", "N/A", ""}


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
) -> dict:
    """Assemble the hero card's verdict — ALL interpretation logic, pure data in/out.

    Separated from rendering so the decision rules (label normalisation, trust
    tiering, the precedent minimum-n gate, agreement tiers) are unit-testable
    without Streamlit. Returns a dict the renderer consumes verbatim:

    ``signal`` (display label), ``signal_class`` (css), ``score`` ([-1,+1]),
    ``source``, ``direction`` (bullish/bearish/neutral), ``headline`` (one
    sentence), ``trust`` ({tier, chip, val_ic, wf_pos}), ``consensus_score``
    (float|None — shown so the hero's number and the Convergence tab's
    consensus card reconcile explicitly instead of looking contradictory),
    ``evidence`` (ordered rows of {tag, state, text};
    state ∈ confirm|conflict|neutral|info).
    """
    # ── 1. Headline object: calibrated model → consensus → Aarambh-only ────
    if calib_conviction is not None and calib_signal is not None:
        score = float(calib_conviction) / 100.0          # DDM ±100 → [-1, +1]
        raw_label = str(calib_signal)
        source = "Calibrated model" if has_profile else "Convergence model (uncalibrated)"
        headline_is_calibrated = True
    elif consensus:
        score = float(consensus.get("value", 0.0))
        raw_label = str(consensus.get("signal", "HOLD"))
        source = "System consensus (uncalibrated)"
        headline_is_calibrated = False
    else:
        score = float(aarambh_signal.get("conviction_score", 0)) / 100.0
        raw_label = str(aarambh_signal.get("signal", "HOLD"))
        source = "Aarambh only (no basket convergence)"
        headline_is_calibrated = False

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
    signal_class = ("undervalued" if direction == "bullish"
                    else "overvalued" if direction == "bearish" else "fair")

    # ── 3. Trust tier ───────────────────────────────────────────────────────
    trust = _trust_tier(val_ic, wf_pos)

    # ── 4. Headline sentence ────────────────────────────────────────────────
    if direction == "neutral":
        headline = (f"{source}: {score:+.2f} — no directional edge at the "
                    f"{horizon_days}d lens right now.")
    else:
        headline = (f"{source} reads {direction} ({signal}, {score:+.2f}) over "
                    f"the next ~{horizon_days} trading days.")

    # ── 5. Evidence rows ────────────────────────────────────────────────────
    evidence: list[dict] = []

    # MODEL — validated edge behind the headline number.
    model_state = ("confirm" if trust["tier"] in ("modest", "solid")
                   else "conflict" if trust["tier"] == "no_edge" else "neutral")
    evidence.append({"tag": "MODEL", "state": model_state, "text": trust["prose"]})

    # PRECEDENT — non-parametric base rate, gated on a real sample.
    if precedent and int(precedent.get("n") or 0) >= 1:
        p_n = int(precedent["n"])
        p_med = float(precedent["median"])
        p_pos = float(precedent["positive_pct"])
        p_h = int(precedent["horizon"])
        p_dir = int(precedent.get("dir") or 0)
        p_word = "bullish" if p_dir > 0 else "bearish" if p_dir < 0 else "flat"
        hero_sign = 1 if direction == "bullish" else -1 if direction == "bearish" else 0
        stub = (f"{p_n} distinct analogs at +{p_h}d: {p_med:+.1f}% median, "
                f"{p_pos:.0f}% positive")
        if p_n < _PRECEDENT_MIN_N:
            evidence.append({"tag": "PRECEDENT", "state": "neutral",
                             "text": f"Thin sample — only {stub}. Not probative; ignore the lean."})
        elif abs(p_pos - 50) < 15:
            evidence.append({"tag": "PRECEDENT", "state": "neutral",
                             "text": f"Split — {stub}. No directional lean either way."})
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

    # INTERNALS — Aarambh vs Nirnay alignment + explicit consensus reconciliation.
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
        recon = (f" Consensus reads {consensus_score:+.2f} (Convergence tab)."
                 if headline_is_calibrated else "")
        evidence.append({
            "tag": "INTERNALS", "state": state,
            "text": (f"Aarambh {a_norm:+.2f} / Nirnay {n_norm:+.2f} — "
                     f"{agree_tier} agreement ({agreement:.0%}), "
                     + ("engines aligned." if aligned else "engines split.") + recon),
        })

    # RISK — flagged divergence events.
    if n_divergences:
        evidence.append({"tag": "RISK", "state": "conflict",
                         "text": (f"{n_divergences} divergence event"
                                  f"{'s' if n_divergences != 1 else ''} flagged — "
                                  f"see the Convergence tab before acting.")})

    return {
        "signal": signal, "signal_class": signal_class, "score": score,
        "source": source, "direction": direction, "headline": headline,
        "trust": trust, "consensus_score": consensus_score, "evidence": evidence,
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
    wf_text = (f" &bull; WF {trust['wf_pos']:.0%}+" if trust.get("wf_pos") is not None
               else "")
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
        </div>
        """,
        unsafe_allow_html=True,
    )


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


