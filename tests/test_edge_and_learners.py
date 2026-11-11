"""Every learner scores the sign the system reads; the edge grades what it says."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from convergence.divergence_detector import CrossSystemDivergenceDetector
from convergence.intelligence import call_record, series_edge
from core.config import EXECUTION_LAG
from convergence.normalization import compute_normalized_convergence
from engines.swayam.kernel import view_skill_weights

RNG = np.random.default_rng(7)
IDX = pd.bdate_range("2017-01-02", periods=2000)
PRICE = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0, 0.01, len(IDX)))), index=IDX)


def _fwd(h, lag=EXECUTION_LAG):
    lp = np.log(PRICE)
    return lp.shift(-(h + lag)) - lp.shift(-lag)


def test_series_edge_recovers_a_planted_ic():
    h = 10
    fwd = _fwd(h)
    z = (fwd - fwd.mean()) / fwd.std()
    for rho in (0.0, 0.15, 0.30):
        lean = rho * z + np.sqrt(1 - rho ** 2) * pd.Series(RNG.normal(0, 1, len(IDX)), index=IDX)
        ok = fwd.notna()
        planted = np.mean([spearmanr(lean[ok].iloc[o::h], fwd[ok].iloc[o::h]).correlation for o in range(h)])
        e = series_edge(lean, PRICE, h)
        assert abs(e["ic"] - planted) < 1e-9
        assert e["n"] == e["n_obs"] // h              # non-overlapping count
        assert abs(e["ic"] - rho) < 3 * e["se"]


def test_series_edge_cannot_see_the_future():
    h = 10
    e = series_edge(_fwd(h).shift(h + EXECUTION_LAG), PRICE, h)   # built from the past only
    assert abs(e["ic"]) < 3 * e["se"]


def test_the_grade_is_entered_at_the_next_close():
    """A lean that is the session's own return is perfect at lag 0 and worthless at lag 1."""
    r1 = np.log(PRICE).shift(-1) - np.log(PRICE)          # the move from t to t+1
    assert series_edge(r1, PRICE, 1, lag=0)["ic"] > 0.99
    e = series_edge(r1, PRICE, 1, lag=1)
    assert abs(e["ic"]) < 3 * e["se"]


def test_call_record_recovers_a_planted_hit_rate():
    h = 10
    fwd = _fwd(h)
    right = pd.Series(RNG.random(len(IDX)) < 0.60, index=IDX)
    # cheap (negative FVO) is the buy claim: make it right 60% of the time
    fvo = pd.Series(np.where(right, -np.sign(fwd), np.sign(fwd)) * 1.0, index=IDX).fillna(0.0)
    rec = call_record(fvo, PRICE, h)
    assert 57 < rec["hit"] < 63 and rec["t"] > 2
    assert 45 < rec["base"] < 55                    # what any day would have hit
    assert rec["n_independent"] == rec["n_claims"] // h
    assert rec["long"]["n_claims"] + rec["short"]["n_claims"] == rec["n_claims"]

def test_call_record_does_not_credit_the_drift():
    """Buying at random in a rising market wins often and earns nothing in excess."""
    idx = pd.bdate_range("2012-01-02", periods=4000)
    up = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0.0015, 0.01, len(idx)))), index=idx)
    fvo = pd.Series(np.where(RNG.random(len(idx)) < 0.5, -1.0, 0.0), index=idx)   # random buy claims
    rec = call_record(fvo, up, 10)
    assert rec["hit"] > 55                           # the drift pays
    assert abs(rec["hit"] - rec["base"]) < 4         # ... exactly as much as it pays anyone
    assert abs(rec["t"]) < 2.5 and abs(rec["excess_bps"]) < abs(rec["mean_bps"]) / 3


def test_no_claim_means_nothing_to_grade():
    h = 10
    fwd = _fwd(h)
    fvo = pd.Series(-np.sign(fwd), index=IDX).fillna(0.0)
    assert call_record(fvo * 0.3, PRICE, h)["n_claims"] == 0


def test_swayam_skill_rewards_the_contrarian_correct_view():
    f10 = (PRICE.shift(-10 - EXECUTION_LAG) / PRICE.shift(-EXECUTION_LAG) - 1).fillna(0)
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


def test_no_swayam_row_is_never_a_divergence():
    dd = CrossSystemDivergenceDetector()
    assert dd.detect({"fvo": -3.0}, None, "2024-01-02") is None
    ev = dd.detect({"fvo": -1.0}, {"oversold_pct": 0, "overbought_pct": 70, "avg_unified_osc": 5}, "2024-01-03")
    assert ev is not None and ev.divergence_type == "CONTRADICTION"


def test_convergence_frame_is_the_two_leans_joined():
    from pipeline.prep import PreparedPanel
    from pipeline.run import run_convergence
    from core.config import InstrumentConfig
    idx = pd.bdate_range("2024-01-01", periods=5)
    fvo_ts = pd.DataFrame({"FVO": [-1.2, 0.1, 1.5, np.nan, -0.8]}, index=idx)
    sw = pd.DataFrame({"Oversold_Pct": [60, 50, 40, 0, 0], "Overbought_Pct": [0, 0, 0, 0, 55],
                       "Avg_Signal": [-4, -3, -2, 0, 4], "_Native": [True, True, True, True, False]},
                      index=idx).drop(idx[1])                       # Swayam has no row on day 2
    prep = PreparedPanel("T", idx, pd.Series(1.0, index=idx), pd.DataFrame(index=idx), [], InstrumentConfig())
    cdf = run_convergence(prep, fvo_ts, sw)["convergence_df"]
    assert list(cdf.index) == [idx[0], idx[2], idx[4]]              # no Swayam row / no valuation: absent
    assert cdf["agree"].tolist() == [1.0, 0.0, 0.0]                 # cheap+oversold agree; the rest oppose
    assert cdf["swayam_native"].tolist() == [True, True, False]


def test_no_divergence_on_a_session_mula_did_not_value():
    dd = CrossSystemDivergenceDetector()
    sw = {"oversold_pct": 70, "overbought_pct": 0, "avg_unified_osc": -6}
    assert dd.detect({"fvo": float("nan")}, sw, "2024-01-02") is None
    assert dd.detect({"fvo": None}, sw, "2024-01-03") is None
    assert dd.get_events().empty
