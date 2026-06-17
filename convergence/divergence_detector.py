"""
Tattva — CrossSystemDivergenceDetector: Detects when Aarambh and Nirnay disagree.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.

Divergence types
----------------
- ``AARAMBH_LEADS``: Valuation extreme but constituents haven't turned
  (early warning).
- ``NIRNAY_LEADS``: Constituents turning but valuation not yet extreme
  (momentum-first move).
- ``CONTRADICTION``: Persistent disagreement (uncertain environment).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.config import DIV_LOOKBACK, DIV_PERSISTENCE_THRESHOLD


@dataclass
class DivergenceEvent:
    """A single cross-system divergence event.

    Attributes
    ----------
    date : str
        Date of the divergence.
    divergence_type : str
        One of ``AARAMBH_LEADS``, ``NIRNAY_LEADS``, ``CONTRADICTION``.
    aarambh_signal : str
        Aarambh stance classification at this date.
    nirnay_signal : str
        Nirnay stance classification at this date.
    severity : float
        Severity score [0, 1].
    description : str
        Human-readable explanation.
    """

    date: str
    divergence_type: str
    aarambh_signal: str
    nirnay_signal: str
    severity: float
    description: str


class CrossSystemDivergenceDetector:
    """Detects and classifies cross-system divergence events.

    Parameters
    ----------
    lookback : int
        Window for persistence tracking.
    persistence_threshold : int
        Minimum occurrences within ``lookback`` to flag as persistent.
    """

    def __init__(
        self,
        lookback: int = DIV_LOOKBACK,
        persistence_threshold: int = DIV_PERSISTENCE_THRESHOLD,
    ) -> None:
        self.lookback = lookback
        self.persistence_threshold = persistence_threshold
        self.events: list[DivergenceEvent] = []
        self._recent_types: list[str] = []

    def detect(
        self,
        aarambh_signal: dict[str, object],
        nirnay_day_stats: dict[str, object],
        date: str,
    ) -> DivergenceEvent | None:
        """Detect divergence for a single date.

        Parameters
        ----------
        aarambh_signal : dict
            Output from ``FairValueEngine.get_current_signal()``.
        nirnay_day_stats : dict
            Aggregated Nirnay stats for the date.
        date : str
            Date string.

        Returns
        -------
        DivergenceEvent | None
            Event if a divergence is detected, ``None`` otherwise.
        """
        conviction = float(aarambh_signal.get("conviction_score", 0))
        oversold_breadth = float(aarambh_signal.get("oversold_breadth", 50))
        nirnay_os_pct = float(nirnay_day_stats.get("oversold_pct", 50))
        nirnay_ob_pct = float(nirnay_day_stats.get("overbought_pct", 50))
        nirnay_avg_osc = float(nirnay_day_stats.get("avg_unified_osc", 0))

        aarambh_stance = self._classify_aarambh_stance(conviction, oversold_breadth)
        nirnay_stance = self._classify_nirnay_stance(nirnay_os_pct, nirnay_ob_pct, nirnay_avg_osc)

        div_type: str | None = None
        severity = 0.0
        description = ""

        if aarambh_stance == "EXTREME_BULLISH" and nirnay_stance != "BULLISH":
            div_type = "AARAMBH_LEADS"
            severity = min(1.0, abs(conviction) / 100.0)
            description = (
                f"Aarambh shows extreme oversold (conviction={conviction:.0f}) "
                f"but Nirnay constituents haven't turned ({nirnay_os_pct:.0f}% oversold). "
                f"Early warning: valuation dislocation not yet reflected in price structure."
            )
        elif aarambh_stance == "EXTREME_BEARISH" and nirnay_stance != "BEARISH":
            div_type = "AARAMBH_LEADS"
            severity = min(1.0, abs(conviction) / 100.0)
            description = (
                f"Aarambh shows extreme overbought (conviction={conviction:.0f}) "
                f"but Nirnay constituents haven't turned. "
                f"Early warning: valuation risk not yet reflected in price structure."
            )
        elif nirnay_stance == "BULLISH" and aarambh_stance not in ("BULLISH", "EXTREME_BULLISH"):
            div_type = "NIRNAY_LEADS"
            severity = min(1.0, abs(nirnay_avg_osc) / 10.0)
            description = (
                f"Nirnay constituents turning bullish ({nirnay_os_pct:.0f}% oversold, "
                f"avg osc={nirnay_avg_osc:.1f}) but Aarambh valuation not yet extreme "
                f"(conviction={conviction:.0f}). Momentum-first move."
            )
        elif nirnay_stance == "BEARISH" and aarambh_stance not in ("BEARISH", "EXTREME_BEARISH"):
            div_type = "NIRNAY_LEADS"
            severity = min(1.0, abs(nirnay_avg_osc) / 10.0)
            description = (
                f"Nirnay constituents turning bearish but Aarambh valuation not yet "
                f"extreme. Momentum-first move to the downside."
            )
        elif aarambh_stance == "BULLISH" and nirnay_stance == "BEARISH":
            div_type = "CONTRADICTION"
            severity = 0.7
            description = (
                f"Aarambh says bullish (conviction={conviction:.0f}) but Nirnay says "
                f"bearish ({nirnay_ob_pct:.0f}% overbought). Contradictory signals — "
                f"uncertain environment."
            )
        elif aarambh_stance == "BEARISH" and nirnay_stance == "BULLISH":
            div_type = "CONTRADICTION"
            severity = 0.7
            description = (
                f"Aarambh says bearish (conviction={conviction:.0f}) but Nirnay says "
                f"bullish ({nirnay_os_pct:.0f}% oversold). Contradictory signals."
            )

        if div_type is None:
            return None

        self._recent_types.append(div_type)
        if len(self._recent_types) > self.lookback:
            self._recent_types.pop(0)

        persistent = self._recent_types.count(div_type) >= self.persistence_threshold

        event = DivergenceEvent(
            date=date,
            divergence_type=div_type,
            aarambh_signal=aarambh_stance,
            nirnay_signal=nirnay_stance,
            severity=round(severity, 3),
            description=description + (" [PERSISTENT]" if persistent else ""),
        )
        self.events.append(event)
        return event

    def get_events(self) -> pd.DataFrame:
        """Return all detected divergence events as a DataFrame."""
        if not self.events:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for e in self.events:
            rows.append(
                {
                    "date": e.date,
                    "divergence_type": e.divergence_type,
                    "aarambh_signal": e.aarambh_signal,
                    "nirnay_signal": e.nirnay_signal,
                    "severity": e.severity,
                    "description": e.description,
                }
            )
        return pd.DataFrame(rows).set_index("date")

    @staticmethod
    def _classify_aarambh_stance(conviction: float, oversold_breadth: float) -> str:
        """Classify Aarambh's directional stance."""
        if conviction < -60 and oversold_breadth > 70:
            return "EXTREME_BULLISH"
        if conviction < -20 or oversold_breadth > 50:
            return "BULLISH"
        if conviction > 60 and oversold_breadth < 30:
            return "EXTREME_BEARISH"
        if conviction > 20 or oversold_breadth < 50:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _classify_nirnay_stance(os_pct: float, ob_pct: float, avg_osc: float) -> str:
        """Classify Nirnay's directional stance."""
        if os_pct > 60 and avg_osc < -3:
            return "BULLISH"
        if ob_pct > 60 and avg_osc > 3:
            return "BEARISH"
        return "NEUTRAL"
