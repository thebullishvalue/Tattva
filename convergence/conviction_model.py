"""
Tattva — CompositeDDM: the drift-diffusion trend of the weighted composite.
तत्त्व (Tattva) — "Principle / Essence"

CONVERGENCE — the same leaky drift-diffusion primitive the Mūla engine runs on
its window-agreement score, applied to the cross-system ``convergence_score``
once its dimension weights have been learned. The result is the composite's
TREND: a bounded, smoothed series with a heuristic uncertainty band, drawn on
the Convergence page beside the raw composite it filters.

It is evidence, not a verdict. The class used to be ``UnifiedConvictionModel``
and its output ``nishkarsh_conviction`` (a name inherited from a sibling
fork), classified STRONG BUY … STRONG SELL — a third BUY/SELL source next to
the hero's action and the consensus card, and it was fitted twice per run
(before and after the weights were learned) to produce numbers no page showed.
It now runs once, on the learned composite, and its words are leans
(``convergence.normalization.lean_label``) at the smoothed series' own causal
tiers — "conviction" and BUY/SELL belong to the hero's chain alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.ddm_filter import drift_diffusion_filter
from analytics.utils import _apply_conviction_bounds
from convergence.normalization import COMPOSITE_PRIOR, causal_tiers, lean_label
from core.config import (
    CONV_DDM_DRIFT_SCALE,
    CONV_DDM_LEAK_RATE,
    CONV_DDM_LONG_RUN_VAR,
)


class CompositeDDM:
    """Drift-diffusion filter over a convergence-score series.

    Parameters
    ----------
    leak_rate : float
        DDM memory decay rate. Higher = faster forgetting.
    drift_scale : float
        Weight on each new observation (steady-state gain = drift / leak).
    long_run_var : float
        Long-run variance the band mean-reverts toward.
    """

    def __init__(
        self,
        leak_rate: float = CONV_DDM_LEAK_RATE,
        drift_scale: float = CONV_DDM_DRIFT_SCALE,
        long_run_var: float = CONV_DDM_LONG_RUN_VAR,
    ) -> None:
        self.leak_rate = leak_rate
        self.drift_scale = drift_scale
        self.long_run_var = long_run_var

    def fit_series(self, scores: pd.Series) -> pd.DataFrame:
        """Filter ``scores`` (±100, reading-oriented: negative = bullish).

        Returns a frame on the same index with ``smoothed`` (tanh-bounded to
        ±100), ``upper`` / ``lower`` (±1.96 filter SD — a heuristic band, not a
        calibrated interval; see ``drift_diffusion_filter``), and ``label`` —
        the lean word at that row's own causal p90 / p75 of ``|smoothed|``,
        built from strictly earlier rows. Empty frame for empty input.
        """
        cols = ["smoothed", "upper", "lower", "label"]
        if scores is None or len(scores) == 0:
            return pd.DataFrame(columns=cols)
        x = pd.to_numeric(pd.Series(scores), errors="coerce").ffill().fillna(0.0)
        filtered, _gains, variances = drift_diffusion_filter(
            x.to_numpy(dtype=np.float64),
            leak_rate=self.leak_rate,
            drift_scale=self.drift_scale,
            long_run_var=self.long_run_var,
        )
        sd = np.sqrt(np.maximum(np.asarray(variances, dtype=np.float64), 0.0))
        smoothed = _apply_conviction_bounds(filtered)
        out = pd.DataFrame({
            "smoothed": np.round(smoothed, 2),
            "upper": np.round(_apply_conviction_bounds(filtered + 1.96 * sd), 2),
            "lower": np.round(_apply_conviction_bounds(filtered - 1.96 * sd), 2),
        }, index=x.index)
        # The smoothed series has its own scale (the filter's gain is
        # drift/leak, not 1), so it is classified on its OWN distribution —
        # the composite's priors only warm it up.
        strong, moderate = causal_tiers(out["smoothed"].to_numpy(), COMPOSITE_PRIOR, scale=100.0)
        out["label"] = [lean_label(v, s, m) for v, s, m in zip(out["smoothed"], strong, moderate)]
        return out
