"""Trust (core/trust.py): the GLS posterior, the class-prior table and the one strength map."""
import json
import math

import pandas as pd
import pytest

from core.trust import Prior, PriorTable, load_priors, posterior, strength


def _prior(mu=0.05, var_mu=0.02 ** 2, tau2=0.03 ** 2, kappa=0.0, se_floor=0.0, scope="class-LOO"):
    return Prior(mu=mu, var_mu=var_mu, tau2=tau2, kappa=kappa, se_floor=se_floor, k=7, scope=scope,
                 vintage=pd.Timestamp("2024-03-29"), study="studies/pooled_priors.py",
                 run_date="2026-10-19")


def test_no_prior_is_the_own_grade():
    p = posterior(0.07, 0.05, None)
    assert (p.ic, p.se, p.t, p.w_own, p.scope) == (0.07, 0.05, 0.07 / 0.05, 1.0, "own")
    assert math.isnan(posterior(float("nan"), 0.05, None).t)
    assert posterior(float("nan"), 0.05, None).scope == "ungraded"


def test_an_independent_prior_is_more_precise_than_either_measurement():
    for se, var_mu, tau2 in ((0.05, 0.02 ** 2, 0.03 ** 2), (0.02, 0.05 ** 2, 0.0), (0.04, 0.0, 0.04 ** 2)):
        pr = _prior(var_mu=var_mu, tau2=tau2, kappa=0.0)
        v_a, v_b = se ** 2, tau2 + var_mu
        p = posterior(0.07, se, pr)
        assert p.se ** 2 <= min(v_a, v_b) + 1e-15
        assert 0 < p.w_own < 1 and min(0.07, pr.mu) <= p.ic <= max(0.07, pr.mu)


def test_a_perfectly_correlated_prior_adds_no_precision():
    se = 0.05
    # The same measurement twice: no gain at all.
    twin = _prior(mu=0.07, var_mu=se ** 2, tau2=0.0, kappa=se ** 2)
    assert posterior(0.07, se, twin).se == pytest.approx(se, abs=1e-15)
    # A noisier copy of the same error (ρ = 1): the posterior is the own grade.
    v_b = 0.08 ** 2
    copy = _prior(mu=0.02, var_mu=v_b, tau2=0.0, kappa=math.sqrt(se ** 2 * v_b))
    p = posterior(0.07, se, copy)
    assert p.se == pytest.approx(se, abs=1e-15) and p.w_own == 1.0 and p.ic == 0.07


def test_the_se_floor_is_honoured():
    pr = _prior(var_mu=0.005 ** 2, tau2=0.0, se_floor=0.03)
    p = posterior(0.07, 0.01, pr)
    assert p.se == 0.03 and p.t == pytest.approx(p.ic / 0.03)


def test_no_own_grade_reads_the_prior_predictive():
    pr = _prior(mu=0.06, var_mu=0.02 ** 2, tau2=0.03 ** 2, se_floor=0.01)
    for ic, se in ((float("nan"), 0.05), (0.07, float("nan")), (None, None)):
        p = posterior(ic, se, pr)
        assert p.ic == 0.06 and p.se == pytest.approx(math.sqrt(0.02 ** 2 + 0.03 ** 2))
        assert p.w_own == 0.0 and p.scope == "class-LOO"


def test_equal_ic_at_different_se_gives_different_strength():
    precise, noisy = posterior(0.12, 0.03, None), posterior(0.12, 0.13, None)
    assert strength(precise.t) == 1.0 and strength(noisy.t) == 0.0


def test_the_strength_map():
    assert strength(1.645) == 0.0 and strength(2.5) == 1.0
    assert strength((1.645 + 2.5) / 2) == pytest.approx(0.5)
    assert strength(-4.0) == 0.0 and strength(9.0) == 1.0
    assert strength(float("nan")) == 0.0 and strength(None) == 0.0


def _priors_file(tmp_path, h=10, lag=1):
    entry = {"mu": 0.04, "var_mu": 0.0004, "tau2": 0.0009, "kappa": 0.0001, "se_floor": 0.02,
             "k": 7, "scope": "class-LOO"}
    data = {"study": "studies/pooled_priors.py", "run_date": "2026-10-19", "h": h, "exec_lag": lag,
            "priors": {"claim": {
                "Gold": {"2022-03-31": entry, "2023-03-31": dict(entry, mu=0.05)},
                "__class__:commodity": {"2022-03-31": dict(entry, scope="class-all")},
            }}}
    p = tmp_path / f"pooled_priors_h{h}_L{lag}.json"
    p.write_text(json.dumps(data))
    return p


def test_lookup_reads_the_latest_vintage_on_or_before_t(tmp_path):
    table = load_priors(10, 1, path=str(_priors_file(tmp_path)))
    assert isinstance(table, PriorTable) and table.meta["run_date"] == "2026-10-19"
    assert table.lookup("claim", "Gold", "commodity", "2022-03-30") is None
    assert table.lookup("claim", "Gold", "commodity", "2022-03-31").mu == 0.04
    assert table.lookup("claim", "Gold", "commodity", "2023-02-01").mu == 0.04
    late = table.lookup("claim", "Gold", "commodity", "2026-01-02")
    assert late.mu == 0.05 and late.vintage == pd.Timestamp("2023-03-31") and late.scope == "class-LOO"
    # A non-member reads its class's all-member entry; an unknown object reads nothing.
    assert table.lookup("claim", "Silver", "commodity", "2024-01-02").scope == "class-all"
    assert table.lookup("claim", "Bitcoin", "crypto", "2024-01-02") is None
    assert table.lookup("sizer:swayam", "Gold", "commodity", "2024-01-02") is None


def test_a_file_for_another_horizon_or_lag_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_priors(10, 1, path=str(_priors_file(tmp_path, h=20, lag=1)))
    with pytest.raises(ValueError):
        load_priors(10, 1, path=str(_priors_file(tmp_path, h=10, lag=0)))
    assert load_priors(10, 1, path=str(tmp_path / "missing.json")) is None
