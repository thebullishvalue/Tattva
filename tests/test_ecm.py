"""The error-correction layer: sign, horizon and the price-side question.

Two synthetic worlds with the same stationary gap z:
  A — PRICE closes it:  p = fv + z, fv a random walk, z AR(1) with φ.
      Then b = φ − 1, κ = 1 − φ > 0 and E[p(t+h) − p(t) | z] = z(φʰ − 1).
  B — FAIR VALUE closes it: p a random walk, the gap reverts on its own.
      Price ignores the gap: κ ≈ 0, P(κ > 0) ≈ ½.
The published drift used to have the opposite sign to the model's own expected
move and roughly one day's size (corr −0.89 with the truth in world A).
"""
import numpy as np
import pandas as pd

from analytics.causal import OnlineAR1
from engines.mula.ecm import MulaLayer


def _run(p, z, h=10):
    layer, ar = MulaLayer(horizon=h), OnlineAR1(halflife=504.0)
    last = prev = None
    last_dp = np.nan
    for t in range(len(p)):
        ar.update(z[t])
        phi = ar.solve()[0]
        if last is None:
            layer.skip()
        else:
            dp = p[t] - p[last]
            dz = (z[last] - z[prev]) if prev is not None else np.nan
            layer.step(z[last], dz, last_dp, dp, z[t], phi)
            last_dp = dp
        prev, last = last, t
    return layer.frame(pd.RangeIndex(len(p)))


def _worlds(seed=0, T=3000, phi=0.95):
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    fv = np.zeros(T)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + 0.01 * rng.standard_normal()
        fv[t] = fv[t - 1] + 0.005 * rng.standard_normal()
    zb = np.zeros(T)
    for t in range(1, T):
        zb[t] = phi * zb[t - 1] + 0.01 * rng.standard_normal()
    return fv + z, z, np.cumsum(0.01 * rng.standard_normal(T)), zb


def test_price_closes_the_gap():
    pa, za, _, _ = _worlds()
    phi, h = 0.95, 10
    f = _run(pa, za, h)
    ok = f["MulaDriftPct"].notna().to_numpy()
    assert ok.sum() > 2000
    true = za * (phi ** h - 1) * 100
    # Sign and size: the published h-day drift tracks the model's own expected move.
    assert np.corrcoef(f["MulaDriftPct"][ok], true[ok])[0, 1] > 0.8
    assert 0.6 < np.median(f["MulaDriftPct"][ok] / true[ok]) < 1.4
    # κ̂ is published positive when price reverts, near its true value.
    assert 0.02 < f["MulaKappa"].median() < 0.08
    assert f["MulaRevProb"].median() > 0.9


def test_fair_value_closes_the_gap():
    _, _, pb, zb = _worlds()
    g = _run(pb, zb)
    assert abs(g["MulaKappa"].median()) < 0.01
    assert 0.3 < g["MulaRevProb"].median() < 0.75


def test_h_day_sd_is_an_h_day_sd():
    pa, za, _, _ = _worlds()
    h = 10
    f = _run(pa, za, h)
    fwd = (pd.Series(pa).shift(-h) - pd.Series(pa)).to_numpy() * 100
    realised = np.nanstd(fwd)
    assert 0.7 * realised < f["MulaSdPct"].median() < 1.4 * realised
