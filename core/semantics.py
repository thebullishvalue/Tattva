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

The valuation state words (Cheap / Fair / Rich) are also defined once, here,
so the hero, the Mūla page and the Precedent page cannot classify the same
oscillator reading three different ways.
"""

from __future__ import annotations

import math

#: |FVO| below this is inside the normal day-to-day noise of price around fair
#: value: no directional claim is made. The hero's mispricing gate starts to
#: open here.
CLAIM_SIGMA = 0.5
#: |FVO| at or beyond this is a deep mispricing — the gate is fully open and
#: the state reads VERY CHEAP / VERY RICH.
DEEP_SIGMA = 2.0

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


def lean_word(lean: float, dead: float = 0.1) -> str:
    """'bullish' / 'bearish' / 'neutral' for a lean, with a dead band."""
    v = _finite(lean)
    if v is None or abs(v) < dead:
        return "neutral"
    return "bullish" if v > 0 else "bearish"
