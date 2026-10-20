"""
Tattva — one vocabulary for direction, shared by every layer.
तत्त्व (Tattva) — "Principle / Essence"

Every engine publishes in its own native units, and those units share one
orientation: a POSITIVE reading means the instrument is stretched UP.

    Mūla    FVO > 0          price above the level the cross-section implies (rich)
    Swayam  oscillator > 0   the instrument's own views read it overbought
    Convergence score > 0    the two agree it is stretched up

The directional call is CONTRARIAN on all of them — stretched up is bearish,
stretched down is bullish — so a reading and the call it implies have opposite
signs. That inversion used to be re-derived, inline and slightly differently,
at every consumer; three learners ended up scoring the wrong sign because of
it. This module is the one place it is written down:

* a **reading** keeps the native orientation (positive = rich / overbought),
  and is what charts and cards display;
* a **lean** is the call that reading implies, BULLISH-POSITIVE, in [-1, +1],
  and is what anything that scores direction against a forward return uses.

The words are defined once here too: the valuation states (Cheap / Fair /
Rich), the one Swayam word (confirms / split / contradicts) and the one grade
label (``edge_tier``), so no two pages can word the same reading differently.
Every cut they use is a registered knob (``core.knobs``); none is a literal.
"""

from __future__ import annotations

import math

from core.knobs import knob

#: |FVO| below this is inside the normal day-to-day noise of price around fair
#: value: no directional claim is made.
CLAIM_SIGMA: float = knob("CLAIM_SIGMA")
#: |FVO| at or beyond this is a deep mispricing — the state reads VERY CHEAP /
#: VERY RICH and the valuation lean saturates.
DEEP_SIGMA: float = knob("DEEP_SIGMA")
#: Net Swayam breadth, aligned with the claim, inside ±this is a split read.
SWAYAM_DEAD: float = knob("SWAYAM_DEAD")

_T0: float = knob("STRENGTH_T0")
_T1: float = knob("STRENGTH_T1")

#: Mūla valuation states, cheapest first.
VALUATION_STATES: tuple[str, ...] = ("VERY CHEAP", "CHEAP", "FAIR", "RICH", "VERY RICH")


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def valuation_state(fvo) -> str:
    """The Mūla state word for an oscillator reading (σ). NaN → ``"FAIR"``
    is deliberately NOT done: a missing valuation is ``"NO VALUATION"``."""
    v = _finite(fvo)
    if v is None:
        return "NO VALUATION"
    if v <= -DEEP_SIGMA:
        return "VERY CHEAP"
    if v <= -CLAIM_SIGMA:
        return "CHEAP"
    if v < CLAIM_SIGMA:
        return "FAIR"
    if v < DEEP_SIGMA:
        return "RICH"
    return "VERY RICH"


def claim_direction(fvo) -> int:
    """+1 bullish (cheap), -1 bearish (rich), 0 no claim (inside the noise)."""
    v = _finite(fvo)
    if v is None or abs(v) < CLAIM_SIGMA:
        return 0
    return 1 if v < 0 else -1


def valuation_lean(fvo) -> float:
    """Bullish-positive lean of a Mūla oscillator reading, in [-1, +1].

    Linear in σ and saturating at ``DEEP_SIGMA``: a 2σ-cheap reading is a full
    bullish lean, a 1σ one half of it.
    """
    v = _finite(fvo)
    if v is None:
        return 0.0
    return max(-1.0, min(1.0, -v / DEEP_SIGMA))


def breadth_lean(oversold_pct, overbought_pct) -> float:
    """Bullish-positive lean of a breadth split (percent points, 0-100 each)."""
    os_, ob_ = _finite(oversold_pct), _finite(overbought_pct)
    if os_ is None or ob_ is None:
        return 0.0
    return max(-1.0, min(1.0, (os_ - ob_) / 100.0))


def oscillator_lean(osc, scale: float = 10.0) -> float:
    """Bullish-positive lean of a Swayam oscillator reading (±``scale``)."""
    v = _finite(osc)
    if v is None or scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, -v / scale))


def swayam_regime_lean(bull_pct, bear_pct) -> float:
    """Bullish-positive lean of Swayam's HMM regime split.

    Swayam's "bull" state is the high-oscillator state — the views reading the
    instrument as persistently stretched UP. Under the system's contrarian
    reading that is a bearish lean, so the lean is ``(bear - bull) / 100``.
    """
    b, r = _finite(bull_pct), _finite(bear_pct)
    if b is None or r is None:
        return 0.0
    return max(-1.0, min(1.0, (r - b) / 100.0))


def state_lean(state: str) -> int:
    """Sign of a valuation-state word: +1 for the cheap family, -1 for rich."""
    s = str(state or "").upper()
    if "CHEAP" in s:
        return 1
    if "RICH" in s:
        return -1
    return 0


def swayam_word(aligned) -> str:
    """The one Swayam word for a claim-aligned net breadth (``d · Lean``).

    ``'confirms'`` beyond +SWAYAM_DEAD, ``'contradicts'`` beyond −SWAYAM_DEAD,
    ``'split'`` inside the band. A missing reading carries no information and
    reads ``'split'``, the no-information word — never a confirmation.
    """
    v = _finite(aligned)
    if v is None or abs(v) <= SWAYAM_DEAD:
        return "split"
    return "confirms" if v > 0 else "contradicts"


def edge_tier(t) -> tuple[str, str]:
    """``(label, tone)`` for a grade's t — the one label scale every page uses.

    Applied to the posterior t of a role object (claim, sizer, call) and to the
    own t of evidence. |t| < 1 is never called an edge, and a grade measured
    AGAINST the lean leaves it neutral: nothing is ever inverted on its grade.
    An undefined t (no SE yet) is ``'ungraded'``.
    """
    v = _finite(t)
    if v is None:
        return "ungraded", "neutral"
    if v <= -_T0:
        return "measured against: neutral, never inverted", "danger"
    if v <= -1:
        return "negative, unproven", "warning"
    if v < 1:
        return "no measured edge", "neutral"
    if v < _T0:
        return "unproven", "warning"
    if v < _T1:
        return "edge", "success"
    return "solid edge", "success"
