"""The engine never looks ahead, and it is asserted.

Every published value is a function of data available at its own date, so
fitting on data[:T] and on data[:T−k] must agree EXACTLY on every shared date.
A component that consulted the future cannot pass this; a component that
quietly stopped producing anything cannot either (the all-NaN check).
"""
import numpy as np
import pandas as pd

from core.config import InstrumentConfig
from engines.mula import MulaEngine

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


def _fit(tgt, expl):
    eng = MulaEngine()
    eng.fit(tgt, expl, feature_names=list(expl.columns), config=InstrumentConfig())
    return eng.ts_data


def test_truncation_never_rewrites_history():
    tgt, expl = _panel()
    full = _fit(tgt, expl)
    cut = 200
    part = _fit(tgt.iloc[:-cut], expl.iloc[:-cut])
    n = len(part)
    for col in COLS:
        a = full[col].to_numpy(dtype=float)[:n]
        b = part[col].to_numpy(dtype=float)
        assert np.isfinite(b).any(), f"{col} is all-NaN"
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all(), f"{col}: publication pattern moved"
        assert np.array_equal(a[both], b[both]), f"{col}: max diff {np.max(np.abs(a[both] - b[both]))}"


def test_latest_reading_is_never_a_coerced_zero():
    tgt, expl = _panel()
    eng = MulaEngine()
    eng.fit(tgt, expl, feature_names=list(expl.columns), config=InstrumentConfig())
    sig = eng.get_current_signal()
    assert np.isfinite(sig["fvo"])
    assert sig["valuation_state"] in ("VERY CHEAP", "CHEAP", "FAIR", "RICH", "VERY RICH")
    assert sig["regime"] == sig["valuation_state"]
