"""Every learner scores the sign the system reads; the edge grades what it says."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analytics.adaptive import OnlineSkillWeights
from convergence.cross_validator import CrossValidator
from convergence.divergence_detector import CrossSystemDivergenceDetector
from convergence.intelligence import DIMENSIONS, online_dimension_weights, series_edge
from convergence.normalization import compute_normalized_convergence
from engines.swayam.kernel import view_skill_weights

RNG = np.random.default_rng(7)
IDX = pd.bdate_range("2017-01-02", periods=2000)
PRICE = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0, 0.01, len(IDX)))), index=IDX)


def test_series_edge_recovers_a_planted_ic():
    h = 10
    fwd = np.log(PRICE).shift(-h) - np.log(PRICE)
    z = (fwd - fwd.mean()) / fwd.std()
    for rho in (0.0, 0.15, 0.30):
        lean = rho * z + np.sqrt(1 - rho ** 2) * pd.Series(RNG.normal(0, 1, len(IDX)), index=IDX)
        planted = np.mean([spearmanr(lean.iloc[o:-h:h], fwd.iloc[o:-h:h]).correlation for o in range(h)])
        e = series_edge(lean, PRICE, h)
        assert abs(e["ic"] - planted) < 1e-9
        assert e["n"] == e["n_obs"] // h              # non-overlapping count
        assert abs(e["ic"] - rho) < 3 * e["se"]


def test_series_edge_cannot_see_the_future():
    h = 10
    fwd = np.log(PRICE).shift(-h) - np.log(PRICE)
    e = series_edge(fwd.shift(h), PRICE, h)           # a lean built from the past only
    assert abs(e["ic"]) < 3 * e["se"]


def test_dimension_learner_scores_the_signed_contribution():
    """In an up-trend, a dimension that is merely always > 0.5 must not win."""
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n)
    fwd = RNG.normal(0.004, 0.01, n)
    cd = RNG.choice([-0.5, 0.5], n)
    right = np.sign(cd) == np.sign(fwd)
    frame = pd.DataFrame({
        "dim_direction": np.where(right, 0.9, 0.1),
        "dim_breadth": np.full(n, 0.8),
        "dim_magnitude": RNG.uniform(0, 1, n),
        "dim_regime": RNG.uniform(0, 1, n),
        "consensus_direction": cd,
        "Ret_10b": fwd,
    }, index=idx)
    w = online_dimension_weights(frame, horizon=10).iloc[-1]
    assert w["direction"] > 0.5 and w["breadth"] < 0.2
    # The old, unsigned criterion collapsed onto the constant dimension.
    old = OnlineSkillWeights(list(DIMENSIONS), floor=0.05)
    D = frame[[f"dim_{d}" for d in DIMENSIONS]].to_numpy()
    for t in range(10, n):
        old.observe(2 * D[t - 10] - 1, fwd[t - 10])
    assert old.weights()[1] > 0.5


def test_swayam_skill_rewards_the_contrarian_correct_view():
    f10 = (PRICE.shift(-10) / PRICE - 1).fillna(0)
    zf = (f10 - f10.mean()) / f10.std()
    frames = {
        "contrarian": pd.DataFrame({"Close": PRICE, "Unified_Osc": (-3 * zf + RNG.normal(0, 2, len(IDX))).clip(-10, 10)}),
        "momentum": pd.DataFrame({"Close": PRICE, "Unified_Osc": (3 * zf + RNG.normal(0, 2, len(IDX))).clip(-10, 10)}),
    }
    w = view_skill_weights(frames, horizon=10, shrink=1.0).iloc[-1]
    assert w["contrarian"] > w["momentum"]


def test_view_weights_are_causal():
    frames = {k: pd.DataFrame({"Close": PRICE, "Unified_Osc": RNG.normal(0, 3, len(IDX))}) for k in "abc"}
    full = view_skill_weights(frames, horizon=10)
    part = view_skill_weights({k: v.iloc[:-250] for k, v in frames.items()}, horizon=10)
    assert np.nanmax(np.abs(full.loc[part.index].to_numpy() - part.to_numpy())) == 0.0


def test_consensus_is_zero_when_both_engines_read_zero():
    m = 400
    idx = pd.bdate_range("2020-01-01", periods=m)
    fvo = np.abs(RNG.normal(1.0, 0.5, m)); fvo[-1] = 0.0
    sw = np.abs(RNG.normal(3.0, 1.0, m)); sw[-1] = 0.0
    c = compute_normalized_convergence(pd.DataFrame({"FVO": fvo}, index=idx),
                                       pd.DataFrame({"Avg_Signal": sw}, index=idx))
    assert abs(c["value"]) < 1e-12 and c["label"] == "NEUTRAL"


def test_no_swayam_row_is_skipped_not_faked():
    cv = CrossValidator()
    m = {"fvo": -1.2, "oversold_breadth": 66, "overbought_breadth": 0, "regime": "CHEAP"}
    assert cv.compute_convergence(m, None, "2024-01-02") is None
    r = cv.compute_convergence(m, {"oversold_pct": 60, "overbought_pct": 0, "avg_unified_osc": -4,
                                   "regime_bull_pct": 0, "regime_bear_pct": 60}, "2024-01-03")
    assert r.convergence_score < 0 and len(cv.history) == 1
    dd = CrossSystemDivergenceDetector()
    assert dd.detect({"fvo": -3.0}, None, "2024-01-02") is None
    ev = dd.detect({"fvo": -1.0}, {"oversold_pct": 0, "overbought_pct": 70, "avg_unified_osc": 5}, "2024-01-03")
    assert ev is not None and ev.divergence_type == "CONTRADICTION"


def test_no_divergence_on_a_session_mula_did_not_value():
    dd = CrossSystemDivergenceDetector()
    sw = {"oversold_pct": 70, "overbought_pct": 0, "avg_unified_osc": -6}
    assert dd.detect({"fvo": float("nan")}, sw, "2024-01-02") is None
    assert dd.detect({"fvo": None}, sw, "2024-01-03") is None
    assert dd.get_events().empty
