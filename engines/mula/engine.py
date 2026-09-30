"""
Tattva — MŪLA engine: FairValueEngine drop-in carrying the ECM layer.
तत्त्व (Tattva) — "Principle / Essence"

Subclasses the production ``FairValueEngine`` so the ENTIRE proven pipeline
(panel admission, factor estimation, conformal bands, DDM conviction, OU /
Hurst diagnostics, pivots, divergences) runs unchanged, then appends the
MŪLA layer's columns to ``ts_data`` and republishes the valuation-
informativeness gate on learned evidence:

  MRProb      := WValuation + WFull — the pooled predictive weight on every
                 design that lets the gap close the price. This REPLACES the
                 incumbent ADF-style gate as the published ``mr_prob``; the
                 old value is preserved in ``GapRevProb`` for comparison.
  GapRevProb  the incumbent's online AR(1) mean-reversion evidence
  MulaKappa / MulaDriftPct / MulaSdPct / WValuation / WMomentum / WFull
              see ``engines.mula.core``

``get_current_signal`` extends its result with ``expected_drift_pct``,
``w_valuation`` and ``mula_kappa`` so the hero card and the Precedent state
vector can consume the decomposition without reaching into ``ts_data``.

Failure posture: if anything in the layer raises, the engine logs a warning
and returns the incumbent fit untouched — MŪLA may fail, the signal must not.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import FairValueEngine
from .ecm import MulaLayer, WARMUP
from core.config import FORECAST_HORIZON

log = logging.getLogger(__name__)


class MulaEngine(FairValueEngine):
    """FairValueEngine + causal error-correction / expert-pooling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.mula_summary: dict = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, target_px: pd.Series, expl_px: pd.DataFrame,
            printed: pd.DataFrame | None = None,
            feature_names: list[str] | None = None,
            progress_callback=None, config=None) -> "MulaEngine":
        """Run the incumbent pipeline, then the MŪLA layer on top."""
        super().fit(target_px, expl_px, printed=printed,
                    feature_names=feature_names,
                    progress_callback=progress_callback, config=config)
        try:
            self._apply_mula_layer(int(getattr(
                config, "forecast_horizon", FORECAST_HORIZON)
                if config is not None else FORECAST_HORIZON))
        except Exception:                                    # pragma: no cover
            log.exception("MŪLA layer failed; publishing incumbent output")
        return self

    # ── Layer application ─────────────────────────────────────────────────

    def _apply_mula_layer(self, horizon: int) -> None:
        series = self.mve.get("series")
        if series is None or series.empty or len(series) != len(self.y):
            return
        y = np.asarray(self.y, dtype=np.float64)
        z = pd.to_numeric(series["gap"], errors="coerce").to_numpy(dtype=np.float64)

        dy = np.diff(y, prepend=np.nan)          # Δp_t
        r_lag = np.roll(dy, 1); r_lag[0] = np.nan
        dz = np.diff(z, prepend=np.nan)          # Δz_t
        dz_lag = np.roll(dz, 1); dz_lag[0] = np.nan
        z_lag = np.roll(z, 1); z_lag[0] = np.nan

        layer = MulaLayer(horizon=max(int(horizon), 1))
        for t in range(len(y)):
            layer.step(z_lag[t], dz_lag[t], r_lag[t], dy[t])

        mula = layer.frame(pd.RangeIndex(len(y)))
        for col in mula.columns:
            vals = mula[col].to_numpy(dtype=np.float64).copy()
            vals[:min(WARMUP, len(vals))] = np.nan   # publish only post-warmup
            self.ts_data[col] = vals

        # Republish the informativeness gate on pooled-learned evidence.
        wv = mula["WValuation"].to_numpy(dtype=np.float64)
        wf = mula["WFull"].to_numpy(dtype=np.float64)
        pooled = wv + wf
        pooled[:min(WARMUP, len(pooled))] = np.nan
        if "MRProb" in self.ts_data.columns:
            self.ts_data["GapRevProb"] = self.ts_data["MRProb"]
        self.ts_data["MRProb"] = pooled

        last = -1
        self.mula_summary = {
            "kappa": float(mula["MulaKappa"].iloc[last]),
            "expected_drift_pct": float(mula["MulaDriftPct"].iloc[last]),
            "drift_sd_pct": float(mula["MulaSdPct"].iloc[last]),
            "w_valuation": float(wv[last]),
            "w_momentum": float(mula["WMomentum"].iloc[last]),
            "w_full": float(wf[last]),
        }

    def get_current_signal(self) -> dict:
        """Incumbent signal + MŪLA's expected-return decomposition."""
        sig = super().get_current_signal()
        if self.mula_summary:
            s = self.mula_summary
            sig["mula_kappa"] = s.get("kappa", 0.0)
            sig["expected_drift_pct"] = s.get("expected_drift_pct", 0.0)
            sig["drift_sd_pct"] = s.get("drift_sd_pct", 0.0)
            sig["w_valuation"] = s.get("w_valuation", 0.0)
            sig["gap_rev_prob_legacy"] = None
            cur = self.ts_data.iloc[-1] if not self.ts_data.empty else None
            if cur is not None and "GapRevProb" in self.ts_data.columns:
                try:
                    v = float(cur["GapRevProb"])
                    sig["gap_rev_prob_legacy"] = v if np.isfinite(v) else None
                except (TypeError, ValueError):
                    pass
        return sig

