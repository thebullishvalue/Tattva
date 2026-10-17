"""
Tattva — CrossSystemDivergenceDetector: when Mūla and Swayam disagree.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — the dates on which the two engines' calls part company.

Each engine's STANCE is read through ``core.semantics``, so a divergence is a
disagreement between the same calls the rest of the system makes:

  Mūla    the valuation claim — CHEAP / RICH past the claim threshold
          (|FVO| ≥ 0.5σ), EXTREME past the deep-mispricing line (≥ 2σ); the
          same thresholds the hero's mispricing gate opens between.
  Swayam  the instrument's own views — BULLISH when its net view breadth AND
          its average oscillator both lean stretched-down, BEARISH mirrored.

Divergence types
----------------
- ``MULA_LEADS``: a deep mispricing the instrument's own price has not
  confirmed — Swayam neutral. The valuation is ahead of the tape.
- ``SWAYAM_LEADS``: the instrument's views have turned but Mūla makes no claim
  — a move that has not (yet) shown up as a mispricing.
- ``CONTRADICTION``: the two calls point opposite ways.

(The Mūla stance used to be read off ``ConvictionBounded`` — the DDM of the
multi-window agreement score — while the hero read the oscillator, so a
"Mūla bullish" divergence and a "fairly valued" hero could sit on the same
date. The Swayam stance used oversold share alone. Both now read the leans
everything else reads.)

A date with no Swayam row is not a divergence: there is nothing for Mūla to
disagree with. It used to be compared against a 50/50 placeholder, so every
Mūla extreme on such a date registered as Mūla "leading" a Swayam that had
said nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.config import DIV_LOOKBACK, DIV_PERSISTENCE_THRESHOLD
from core.semantics import (
    CLAIM_SIGMA,
    DEEP_SIGMA,
    breadth_lean,
    oscillator_lean,
)

#: Swayam's stance needs BOTH of its reads to lean the same way: net view
#: breadth of at least half the bank (e.g. 60% oversold vs 10% overbought) and
#: an average oscillator past ±3 on its ±10 scale. Declared, not estimated —
#: they define what "the views have turned" means, and they are the levels the
#: stance has used since it was introduced (oversold share > 60%, |osc| > 3),
#: restated on the net split.
SWAYAM_STANCE_BREADTH = 0.5
SWAYAM_STANCE_OSC = 0.3


@dataclass
class DivergenceEvent:
    """A single cross-system divergence event.

    Attributes
    ----------
    date : str
        Date of the divergence (``YYYY-MM-DD``).
    divergence_type : str
        ``MULA_LEADS``, ``SWAYAM_LEADS`` or ``CONTRADICTION``.
    mula_stance / swayam_stance : str
        ``EXTREME_BULLISH`` / ``BULLISH`` / ``NEUTRAL`` / ``BEARISH`` /
        ``EXTREME_BEARISH`` (Swayam has no EXTREME grade).
    severity : float
        Severity in [0, 1] — how strong the leading or contradicting read is.
    persistent : bool
        At least ``persistence_threshold`` events of this type within the
        lookback window ending on this date.
    description : str
        Human-readable explanation.
    """

    date: str
    divergence_type: str
    mula_stance: str
    swayam_stance: str
    severity: float
    persistent: bool
    description: str


def mula_stance(fvo) -> str:
    """Mūla's stance from its oscillator (σ): the claim and its depth."""
    try:
        v = float(fvo)
    except (TypeError, ValueError):
        return "NEUTRAL"
    if v != v:
        return "NEUTRAL"
    if v <= -DEEP_SIGMA:
        return "EXTREME_BULLISH"
    if v <= -CLAIM_SIGMA:
        return "BULLISH"
    if v >= DEEP_SIGMA:
        return "EXTREME_BEARISH"
    if v >= CLAIM_SIGMA:
        return "BEARISH"
    return "NEUTRAL"


def swayam_stance(oversold_pct, overbought_pct, avg_osc) -> str:
    """Swayam's stance: both net breadth and the average view must lean."""
    b = breadth_lean(oversold_pct, overbought_pct)
    o = oscillator_lean(avg_osc)
    if b >= SWAYAM_STANCE_BREADTH and o >= SWAYAM_STANCE_OSC:
        return "BULLISH"
    if b <= -SWAYAM_STANCE_BREADTH and o <= -SWAYAM_STANCE_OSC:
        return "BEARISH"
    return "NEUTRAL"


def _side(stance: str) -> int:
    return 1 if "BULLISH" in stance else -1 if "BEARISH" in stance else 0


class CrossSystemDivergenceDetector:
    """Detects and classifies cross-system divergence events.

    Parameters
    ----------
    lookback : int
        Window (sessions, applied as ~1.5x calendar days) for persistence.
    persistence_threshold : int
        Events of one type within ``lookback`` needed to flag it persistent.
    """

    def __init__(
        self,
        lookback: int = DIV_LOOKBACK,
        persistence_threshold: int = DIV_PERSISTENCE_THRESHOLD,
    ) -> None:
        self.lookback = lookback
        self.persistence_threshold = persistence_threshold
        self.events: list[DivergenceEvent] = []
        # (date, type) pairs, pruned by CALENDAR distance from each new event
        # rather than by count — detect() only records dates on which some
        # divergence fires, so a count window over a sparse run would span
        # months and call it "persistent".
        self._recent: list[tuple[str, str]] = []

    def detect(
        self,
        mula: dict[str, object],
        swayam: dict[str, object] | None,
        date: str,
    ) -> DivergenceEvent | None:
        """Detect a divergence on one date, or return ``None``.

        ``mula`` needs ``"fvo"``; ``swayam`` the aggregate's
        ``"oversold_pct"``, ``"overbought_pct"`` and ``"avg_unified_osc"``, or
        ``None`` when Swayam has no row for the date (never a divergence).
        """
        if swayam is None:
            return None
        fvo = mula.get("fvo")
        ms = mula_stance(fvo)
        ss = swayam_stance(swayam.get("oversold_pct"), swayam.get("overbought_pct"),
                           swayam.get("avg_unified_osc"))
        m_side, s_side = _side(ms), _side(ss)
        if m_side == s_side:
            return None

        try:
            fv = float(fvo)
        except (TypeError, ValueError):
            fv = float("nan")
        os_ = float(swayam.get("oversold_pct", 0.0) or 0.0)
        ob_ = float(swayam.get("overbought_pct", 0.0) or 0.0)
        osc = float(swayam.get("avg_unified_osc", 0.0) or 0.0)
        m_word = "cheap" if m_side > 0 else "rich"
        s_word = "bullish" if s_side > 0 else "bearish"

        if m_side != 0 and s_side != 0:
            div_type = "CONTRADICTION"
            severity = min(1.0, (min(abs(fv) / DEEP_SIGMA, 1.0)
                                 + abs(oscillator_lean(osc))) / 2.0)
            description = (
                f"Mūla reads the target {m_word} ({fv:+.2f}σ) while Swayam's views "
                f"read it {s_word} ({os_:.0f}% oversold vs {ob_:.0f}% overbought, "
                f"average {osc:+.1f}). The two calls point opposite ways."
            )
        elif "EXTREME" in ms:
            div_type = "MULA_LEADS"
            severity = min(1.0, abs(fv) / (2.0 * DEEP_SIGMA))
            description = (
                f"Mūla reads a deep mispricing ({fv:+.2f}σ, {m_word}) that the "
                f"target's own views have not confirmed ({os_:.0f}% oversold vs "
                f"{ob_:.0f}% overbought). The valuation is ahead of the tape."
            )
        elif s_side != 0 and m_side == 0:
            div_type = "SWAYAM_LEADS"
            severity = min(1.0, abs(oscillator_lean(osc)))
            description = (
                f"Swayam's views have turned {s_word} ({os_:.0f}% oversold vs "
                f"{ob_:.0f}% overbought, average {osc:+.1f}) but Mūla makes no claim "
                f"({fv:+.2f}σ, inside the noise). A move not yet seen as a mispricing."
            )
        else:
            # Mūla makes an ordinary (non-extreme) claim and Swayam is merely
            # neutral: unconfirmed, which the hero's convergence gate already
            # discounts, but not a divergence.
            return None

        self._recent.append((date, div_type))
        try:
            _cutoff = pd.Timestamp(date) - pd.Timedelta(days=int(self.lookback * 1.5))
            self._recent = [(d, t) for d, t in self._recent if pd.Timestamp(d) >= _cutoff]
        except (ValueError, TypeError):
            # Non-date sentinel (an integer row label from a research caller).
            if len(self._recent) > self.lookback:
                self._recent = self._recent[-self.lookback:]
        persistent = sum(1 for _, t in self._recent if t == div_type) >= self.persistence_threshold

        event = DivergenceEvent(
            date=date, divergence_type=div_type, mula_stance=ms, swayam_stance=ss,
            severity=round(float(severity), 3), persistent=bool(persistent),
            description=description,
        )
        self.events.append(event)
        return event

    def get_events(self) -> pd.DataFrame:
        """All events, indexed by date string. Columns: ``divergence_type``,
        ``mula_stance``, ``swayam_stance``, ``severity``, ``persistent``,
        ``description``. Empty frame when nothing fired."""
        if not self.events:
            return pd.DataFrame(columns=["divergence_type", "mula_stance", "swayam_stance",
                                         "severity", "persistent", "description"])
        return pd.DataFrame([
            {"date": e.date, "divergence_type": e.divergence_type,
             "mula_stance": e.mula_stance, "swayam_stance": e.swayam_stance,
             "severity": e.severity, "persistent": e.persistent,
             "description": e.description}
            for e in self.events
        ]).set_index("date")
