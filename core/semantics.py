"""
Tattva — one vocabulary for direction, shared by every layer.
तत्त्व (Tattva) — "Principle / Essence"

Every engine publishes in its own native units, and those units share one
orientation: a POSITIVE reading means the instrument is stretched UP.

    Mūla    FVO > 0          price above the level the cross-section implies (rich)
    Swayam  oscillator > 0   its views read it overbought

The directional call is CONTRARIAN on both — stretched up is bearish,
stretched down is bullish — so a reading and the call it implies have opposite
signs. That inversion used to be re-derived, inline and slightly differently,
at every consumer; three learners ended up scoring the wrong sign because of
it. This module is the one place it is written down:

* a **reading** keeps the native orientation (positive = rich / overbought),
  and is what charts and cards display;
* a **lean** is the call that reading implies, BULLISH-POSITIVE, in [-1, +1],
  and is what anything that scores direction against a forward return uses.

The words are defined once here too: the valuation states (Cheap / Fair /
Rich), the one Swayam word (confirms / split / contradicts), the one grade
label (``edge_tier``) and the one gap-stationarity word
(``gap_stationarity``), so no two pages can word the same reading
differently. Every cut they use is a registered knob (``core.knobs``); none is
a literal. :func:`claim_oscillator` is the claim's one candidate alternative
unit, dormant until the pooled-prior study's Decision A and an owner-signed
change.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.knobs import knob

#: |FVO| below this is inside the normal day-to-day noise of price around fair
#: value: no directional claim is made.
CLAIM_SIGMA: float = knob("CLAIM_SIGMA")
#: |FVO| at or beyond this is a deep mispricing — the state reads VERY CHEAP /
#: VERY RICH and the valuation lean saturates.
DEEP_SIGMA: float = knob("DEEP_SIGMA")
#: Net Swayam breadth, aligned with the claim, inside ±this is a split read.
SWAYAM_DEAD: float = knob("SWAYAM_DEAD")
#: The size both stationarity tests on Mūla's gap are read at.
STATIONARITY_ALPHA: float = knob("STATIONARITY_ALPHA")
#: Published sessions Decision A's candidate unit centres and scales FVO on.
CLAIM_CENTRING_WINDOW: int = knob("CLAIM_CENTRING_WINDOW")

_T0: float = knob("STRENGTH_T0")
_T1: float = knob("STRENGTH_T1")

#: The three gap-stationarity words (:func:`gap_stationarity`), strongest first.
GAP_STATES: tuple[str, ...] = (
    "stationary", "reverts around a drifting level: unproven (see claim unit)",
    "unit root not rejected")

#: Decision A's candidate unit (:func:`claim_oscillator`); ``'none'`` is FVO itself.
CLAIM_UNITS: tuple[str, ...] = ("none", f"robust_z_{CLAIM_CENTRING_WINDOW}")


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
    """Bullish-positive lean of a breadth split (percent points, 0-100 each).

    Swayam's one lean: ``engines.swayam.kernel.aggregate_views`` publishes it
    as ``Lean`` on every breadth row, and everything downstream reads that
    column. An oversold majority is the bullish reading (contrarian).
    """
    os_, ob_ = _finite(oversold_pct), _finite(overbought_pct)
    if os_ is None or ob_ is None:
        return 0.0
    return max(-1.0, min(1.0, (os_ - ob_) / 100.0))


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


def gap_stationarity(adf_p: float, kpss_p: float) -> str:
    """The one word for whether Mūla's gap is stationary, from its two tests.

    ``adf_p`` is the ADF p-value (null: a unit root), ``kpss_p`` the KPSS
    p-value (null: stationary about a constant), both read at
    STATIONARITY_ALPHA:

      * ``'stationary'`` — ADF rejects the unit root and KPSS does not
        reject stationarity: both tests agree;
      * ``'reverts around a drifting level: unproven (see claim unit)'`` — ADF
        rejects the unit root but KPSS rejects a constant level too (or did
        not run): the gap reverts, but around a level that moves, which is
        the claim's unit question (C12), not a settled stationarity;
      * ``'unit root not rejected'`` — ADF does not reject (or did not run).

    Evidence about the claim, labelled a full-history diagnostic wherever it
    is shown; it is no precondition and moves nothing. It replaces two
    definitions that contradicted each other on three of four instruments
    (ADF alone on the Mūla page, ADF and KPSS on System).
    """
    adf, kp = _finite(adf_p), _finite(kpss_p)
    if adf is None or adf >= STATIONARITY_ALPHA:
        return GAP_STATES[2]
    if kp is not None and kp > STATIONARITY_ALPHA:
        return GAP_STATES[0]
    return GAP_STATES[1]


def claim_oscillator(fvo: pd.Series, mode: str) -> pd.Series:
    """The claim's oscillator in ``mode``'s unit — DORMANT: nothing applies it.

    ``'none'`` is Mūla's FVO itself (a copy). ``'robust_z_252'`` is Decision
    A's pre-registered candidate: on the PUBLISHED sessions (FVO finite), each
    session's FVO against the median and the MAD of the CLAIM_CENTRING_WINDOW
    published sessions strictly before it,

        z_t = (FVO_t − median(FVO_{t−W..t−1})) / (1.4826 · MAD(FVO_{t−W..t−1})),

    so it is causal and lagged: row t reads nothing dated t or later besides
    FVO_t itself. NaN until W published sessions precede it, on every
    unpublished session, and where the window's MAD is 0.

    The pooled-prior study evaluates it on held-out members and records the
    decision in ``KNOBS['CLAIM_CENTRING']``; adopting it is an owner-signed
    change that publishes it in place as FVO. Until then the claim, every
    grade and every page read raw FVO.
    """
    s = pd.to_numeric(pd.Series(fvo), errors="coerce").astype(float)
    if mode == CLAIM_UNITS[0]:
        return s.copy()
    if mode != CLAIM_UNITS[1]:
        raise ValueError(f"claim unit {mode!r} is not one of {CLAIM_UNITS}")
    w = int(CLAIM_CENTRING_WINDOW)
    out = pd.Series(np.nan, index=s.index)
    pos = np.flatnonzero(s.notna().to_numpy())
    x = s.to_numpy()[pos]
    if len(x) > w:
        win = np.lib.stride_tricks.sliding_window_view(x[:-1], w)    # x[t−W..t−1], t = W…
        med = np.median(win, axis=1)
        mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(mad > 0, (x[w:] - med) / mad, np.nan)
        out.iloc[pos[w:]] = z
    return out
