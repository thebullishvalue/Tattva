"""The engine never looks ahead, and neither does the call built on it — asserted.

Every published value is a function of data available at its own date, so
fitting on data[:T] and on data[:T−k] must agree EXACTLY on every shared date.
A component that consulted the future cannot pass this; a component that
quietly stopped producing anything cannot either (the all-NaN check).

The call (call_ts: the composer at every session, with its as-of grades) is
held to the same rule on every session whose outcome had resolved by T.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from core.call import CALL_COLUMNS
from core.config import InstrumentConfig
from core.instruments import instrument
from core.knobs import knob
from core.semantics import breadth_lean
from engines.mula import MulaEngine
from pipeline.assemble import assemble

COLS = ("FairValue", "FVO", "ModelSpread", "MRProb", "Confidence", "XSConsistency",
        "MulaKappa", "MulaDriftPct", "MulaSdPct", "WValuation", "ConvictionBounded",
        "OversoldBreadth")


def _panel(T=1100, n=48, k=3, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    f = np.cumsum(rng.normal(0, 0.01, (T, k)), axis=0)           # integrated factors
    load = rng.normal(0, 1, (n, k))
    x = f @ load.T + np.cumsum(rng.normal(0, 0.004, (T, n)), axis=0) * 0.3
    expl = pd.DataFrame(100 * np.exp(x), index=idx, columns=[f"I{j:02d}" for j in range(n)])
    gap = np.zeros(T)
    for t in range(1, T):
        gap[t] = 0.93 * gap[t - 1] + 0.008 * rng.standard_normal()
    tgt = pd.Series(100 * np.exp(f @ np.array([0.6, -0.3, 0.2]) + gap), index=idx, name="T")
    return tgt, expl


@lru_cache(maxsize=None)
def _fitted(cut: int) -> MulaEngine:
    """The engine fitted on the synthetic panel less its last ``cut`` sessions."""
    tgt, expl = _panel()
    if cut:
        tgt, expl = tgt.iloc[:-cut], expl.iloc[:-cut]
    eng = MulaEngine()
    eng.fit(tgt, expl, feature_names=list(expl.columns), config=InstrumentConfig())
    return eng


def test_truncation_never_rewrites_history():
    full = _fitted(0).ts_data
    part = _fitted(200).ts_data
    n = len(part)
    for col in COLS:
        a = full[col].to_numpy(dtype=float)[:n]
        b = part[col].to_numpy(dtype=float)
        assert np.isfinite(b).any(), f"{col} is all-NaN"
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all(), f"{col}: publication pattern moved"
        assert np.array_equal(a[both], b[both]), f"{col}: max diff {np.max(np.abs(a[both] - b[both]))}"


def test_latest_reading_is_never_a_coerced_zero():
    sig = _fitted(0).get_current_signal()
    assert np.isfinite(sig["fvo"])
    assert sig["valuation_state"] in ("VERY CHEAP", "CHEAP", "FAIR", "RICH", "VERY RICH")
    assert sig["regime"] == sig["valuation_state"]


def _call_ts(cut: int) -> pd.DataFrame:
    """call_ts on the engine fitted without the last ``cut`` sessions, with a
    synthetic Swayam read (some sessions carried) on the same dates."""
    tgt, _ = _panel()
    dates = tgt.index[:len(tgt) - cut]
    rng = np.random.default_rng(5)
    lean = np.clip(rng.normal(0, 0.4, len(tgt)), -1, 1)
    os_, ob_ = np.clip(100 * lean, 0, None), np.clip(-100 * lean, 0, None)
    sw = pd.DataFrame({"Oversold_Pct": os_, "Overbought_Pct": ob_,
                       "Lean": [breadth_lean(a, b) for a, b in zip(os_, ob_)],
                       "_Native": rng.uniform(size=len(tgt)) > 0.05}, index=tgt.index).loc[dates]
    eng = _fitted(cut)
    fvo_ts = eng.ts_data.copy()
    fvo_ts.index = dates
    return assemble(fvo_ts, sw, instrument=instrument("Gold"), cfg=InstrumentConfig(),
                    asof=eng.asof, today=dates[-1], priors=None).call_ts


def test_the_call_never_rewrites_a_resolved_session():
    full, part = _call_ts(0), _call_ts(200)
    h, lag = InstrumentConfig().forecast_horizon, knob("EXECUTION_LAG")
    resolved = part.index[:len(part) - (lag + h)]            # outcome known by the cut
    assert (part.loc[resolved, "d"] != 0).any() and np.isfinite(part["claim_t"]).any()
    for col in CALL_COLUMNS:
        a, b = full.loc[resolved, col], part.loc[resolved, col]
        if pd.api.types.is_float_dtype(a):
            assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), col
        else:
            assert all(x == y or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)), col
