"""THE grade (core/grade.py): it recovers skill, cannot see the future, and its SE is calibrated."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import spearmanr

from core.grade import (_offset_averaged_ic, causal_drift, expanding_edge, forward_return,
                        series_edge, shift_null_se)
from core.knobs import knob
from core.semantics import edge_tier
from core.trust import posterior, strength

RNG = np.random.default_rng(7)
IDX = pd.bdate_range("2017-01-02", periods=2000)
PRICE = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0, 0.01, len(IDX)))), index=IDX)
H = 10
L = knob("EXECUTION_LAG")


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


# ── helpers ─────────────────────────────────────────────────────────────


def _walk(rng, n, drift=0.0, vol=0.01):
    idx = pd.bdate_range("2017-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


def _ar1(rng, n, phi):
    """Unit-variance AR(1)."""
    return lfilter([np.sqrt(1 - phi * phi)], [1, -phi], rng.normal(size=n))


def _planted(rng, target: pd.Series, ic: float) -> pd.Series:
    """A lean whose population Spearman IC with ``target`` is ``ic``."""
    z = (target - target.mean()) / target.std()
    rho = 2 * np.sin(np.pi * ic / 6)                  # Pearson ρ giving Spearman ``ic``
    return rho * z + np.sqrt(1 - rho ** 2) * pd.Series(rng.normal(size=len(z)), index=z.index)


# ── (a) recovery, (b) no foresight ──────────────────────────────────────


def test_a_planted_ic_of_030_is_recovered():
    rng = np.random.default_rng(11)
    P = _walk(rng, 2000)
    for drift, lag in (("none", 0), ("causal", L)):
        lean = _planted(rng, forward_return(P, H, lag), 0.30)
        e = series_edge(lean, P, H, drift=drift, exec_lag=lag)
        assert abs(e["ic"] - 0.30) < 0.05, (drift, lag, e["ic"])
        assert e["t"] > knob("STRENGTH_T1") and e["exec_lag"] == lag and e["drift"] == drift


def test_b_a_forward_shifted_lean_has_no_edge():
    rng = np.random.default_rng(12)
    P = _walk(rng, 2000)
    lean = _planted(rng, forward_return(P, H, L), 0.30)
    for k in (H + L, 3 * H):                          # the same lean, published k sessions late
        e = series_edge(lean.shift(k), P, H, drift="causal", exec_lag=L)
        assert abs(e["ic"]) < 2 * e["se"], (k, e["ic"], e["se"])


# ── (c) the SE is calibrated ────────────────────────────────────────────


def _calibration(make_lean, n_seeds=60, leans_per_seed=9, n=1500):
    """mean shift-null SE / empirical SD of the IC, under no skill.

    Each seed draws one price path and several independent leans against it:
    under the null their ICs are uncorrelated, and 60 single draws would leave
    the empirical SD itself ±9% uncertain against a ±12% band.
    """
    ics, ses = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y = forward_return(_walk(rng, n), H).to_numpy()
        ok = np.isfinite(y)
        for _ in range(leans_per_seed):
            x = make_lean(rng, n)[ok]
            ics.append(_offset_averaged_ic(x, y[ok], H))
            ses.append(shift_null_se(x, y[ok], H))
    return np.mean(ses) / np.std(ics, ddof=1)


def test_c_shift_null_se_is_calibrated():
    cases = {f"AR(1) φ={phi}": (lambda rng, n, phi=phi: _ar1(rng, n, phi)) for phi in (0.0, 0.9, 0.98)}
    # Near-constant and persistent, like the ECM's MRProb: a logistic AR(1) in [0.4, 0.8].
    cases["MRProb-like"] = lambda rng, n: 0.4 + 0.4 / (1 + np.exp(-2 * _ar1(rng, n, 0.995)))
    ratios = {k: _calibration(f) for k, f in cases.items()}
    assert all(0.88 <= r <= 1.12 for r in ratios.values()), ratios


def test_c_signed_grade_se_is_calibrated():
    """A sizer is graded as d·z against d·Y. On a no-skill claim whose side runs
    ~13 rows (the audited four: 14), each rotated product must keep its row's
    side, or the null understates the SE (0.73 of the truth, se_calibration)."""
    def signed_pair(rng, y):
        d = np.where(np.cumsum(rng.random(len(y)) < 1 / 13) % 2 == 0, 1.0, -1.0)
        z = np.clip(_ar1(rng, len(y), 0.9) / 3, -1, 1)
        return d * z, d * y, d

    ics, ses = [], []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        e = forward_return(_walk(rng, 1500), H).to_numpy()
        e = e[np.isfinite(e)]
        for _ in range(9):
            x, y, d = signed_pair(rng, e)
            ics.append(_offset_averaged_ic(x, y, H))
            ses.append(shift_null_se(x, y, H, sign=d))
    assert 0.88 <= np.mean(ses) / np.std(ics, ddof=1) <= 1.12


def test_series_edge_reports_the_shift_null_and_discloses_the_naive_se():
    rng = np.random.default_rng(13)
    P = _walk(rng, 1500)
    lean = pd.Series(_ar1(rng, 1500, 0.9), index=P.index)
    e = series_edge(lean, P, H)
    y = forward_return(P, H).reindex(lean.index)
    ok = y.notna()
    assert e["se_kind"] == "shift-null"
    assert e["se"] == shift_null_se(lean[ok].to_numpy(), y[ok].to_numpy(), H)
    assert e["se_naive"] == 1 / np.sqrt(e["n_win"] - 3) and e["n"] == e["n_win"]
    # Every 2.8.0 key survives, with the new ones beside them.
    assert {"ic", "se", "t", "n", "n_obs", "horizon", "hit", "base", "positive_blocks", "scored_blocks",
            "n_blocks", "blocks", "n_win", "se_kind", "se_naive", "exec_lag", "drift", "base_majority",
            "payoff", "span"} <= set(e)
    assert e["span"] == (lean.index[0], y[ok].index[-1])


# ── (d) the null opens rarely, per look ─────────────────────────────────


def test_d_planted_null_false_open_per_look():
    """200 no-skill instruments graded as of every grid date, as trust reads them."""
    looks = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        P = _walk(rng, 1500)
        phi = (0.0, 0.9, 0.98)[seed % 3]
        g = expanding_edge(pd.Series(_ar1(rng, 1500, phi), index=P.index), P, H,
                           drift="causal", exec_lag=L)
        g = g.drop_duplicates("grid_date").dropna(subset=["t"])
        looks += [strength(posterior(a, b, None).t) for a, b in zip(g["ic"], g["se"])]
    looks = np.array(looks)
    assert len(looks) > 10_000
    assert np.mean(looks > 0) <= 0.065 and looks.mean() <= 0.03, (np.mean(looks > 0), looks.mean())


# ── (e) drift earns nothing ─────────────────────────────────────────────


def _drift_only_sizer(seed):
    """A +20%/yr trend, a persistent no-skill claim d, and a sizer that is
    merely higher on BUY claims: const + 0.05·1[d=+1]."""
    rng = np.random.default_rng(seed)
    P = _walk(rng, 1600, drift=np.log(1.2) / 252)
    e = _ar1(rng, 1600, 0.97)
    d = pd.Series(np.where(e > 0.3, -1.0, np.where(e < -0.3, 1.0, 0.0)), index=P.index)
    sizer = (0.5 + 0.05 * (d > 0)).where(d != 0) + rng.normal(0, 1e-6, len(d))
    return {drift: series_edge(sizer, P, H, sign=d, drift=drift, exec_lag=L)
            for drift in ("causal", "none")}


def test_e_planted_drift_is_not_skill():
    grades = [_drift_only_sizer(seed) for seed in range(100)]
    opens = {k: np.mean([strength(posterior(g[k]["ic"], g[k]["se"], None).t) > 0 for g in grades])
             for k in ("causal", "none")}
    ic = {k: np.mean([g[k]["ic"] for g in grades]) for k in ("causal", "none")}
    # Graded against the drift-neutral target, the factor has no IC on average and
    # opens at about the null's rate (a calibrated t is inside ±1.5 only 87% of
    # the time, so that is not asked) ...
    assert abs(ic["causal"]) < 0.02 and opens["causal"] <= 0.10, (ic, opens)
    # ... while the raw target credits it with the trend.
    assert ic["none"] > 0.1 and opens["none"] > 0.5, (ic, opens)


# ── (f) the execution lag ───────────────────────────────────────────────


def test_f_exec_lag_one_is_the_price_one_row_later():
    rng = np.random.default_rng(14)
    P = _walk(rng, 1500)
    lean = pd.Series(_ar1(rng, 1500, 0.9), index=P.index)
    lagged = series_edge(lean, P, H, exec_lag=1)
    shifted = series_edge(lean, P.shift(-1), H)
    for k in ("ic", "se", "t", "n", "n_win", "n_obs", "hit", "base_majority", "payoff", "span",
              "positive_blocks", "scored_blocks", "blocks"):
        assert lagged[k] == shifted[k], k
    assert lagged["exec_lag"] == 1 and lagged["ic"] != series_edge(lean, P, H)["ic"]


# ── (g) the as-of grade cannot see the future ───────────────────────────


def test_g_expanding_edge_is_invariant_to_truncation():
    rng = np.random.default_rng(15)
    P = _walk(rng, 1800, drift=0.0004)
    d = pd.Series(np.sign(_ar1(rng, 1800, 0.95)), index=P.index)
    lean = _planted(rng, forward_return(P, H, L), 0.10) * d
    kw = dict(sign=d, mask=d != 0, drift="causal", exec_lag=L)
    T = P.index[1300]
    short = expanding_edge(lean.loc[:T], P.loc[:T], H, **kw)
    later = expanding_edge(lean.iloc[:1301 + 250], P.iloc[:1301 + 250], H, **kw).loc[:T]
    full = expanding_edge(lean, P, H, **kw).loc[:T]
    pd.testing.assert_frame_equal(short, later)
    pd.testing.assert_frame_equal(short, full)
    assert short["t"].notna().sum() > 0 and short.index.equals(P.loc[:T].index)


def test_the_as_of_grade_at_a_grid_date_is_the_grade_of_the_data_to_that_date():
    rng = np.random.default_rng(16)
    P = _walk(rng, 1500)
    lean = _planted(rng, forward_return(P, H, L), 0.10)
    g = expanding_edge(lean, P, H, drift="causal", exec_lag=L)
    grid = P.index[::H]
    assert g["grid_date"].isin(grid).all() and (g["grid_date"] <= g.index).all()
    for gd in (grid[60], grid[-1]):
        e = series_edge(lean.loc[:gd], P.loc[:gd], H, drift="causal", exec_lag=L)
        row = g.loc[gd]
        assert row["n_win"] == e["n_win"] and row["ic"] == e["ic"] and row["se"] == e["se"]
    # Before the warm-up the as-of grade is NaN, but its window count is still reported.
    early = g[g["n_win"] < knob("GRADE_WARMUP_WIN")]
    assert len(early) and early["ic"].isna().all() and early["t"].isna().all()


def test_causal_drift_reads_only_resolved_returns():
    rng = np.random.default_rng(17)
    Y = forward_return(_walk(rng, 600, drift=0.001), H, L)
    mu, up = causal_drift(Y, H, L, knob("DRIFT_MIN_RESOLVED"))
    s = 400
    bumped = Y.copy()
    bumped.iloc[s - H - L + 1:] += 1.0                # rewrite every return not yet resolved at s
    mu2, up2 = causal_drift(bumped, H, L, knob("DRIFT_MIN_RESOLVED"))
    assert mu.iloc[s] == mu2.iloc[s] and up.iloc[s] == up2.iloc[s]
    assert mu.iloc[s] == Y.iloc[: s - H - L + 1].mean()
    assert mu.iloc[: knob("DRIFT_MIN_RESOLVED") + H + L - 1].isna().all()


def test_a_no_skill_always_buy_lean_hits_its_side_matched_base():
    """In an up-trend a constant BUY wins more than half the time; the side-matched
    causal base says so, so hit − base reads no skill."""
    rng = np.random.default_rng(18)
    P = _walk(rng, 2000, drift=np.log(1.25) / 252)
    e = series_edge(pd.Series(1.0, index=P.index), P, H, drift="causal", exec_lag=L)
    assert e["hit"] > 52 and abs(e["hit"] - e["base"]) < 3


# ── (h) a sub-SE IC is not an edge ──────────────────────────────────────


def test_h_ic_0078_at_se_0089_is_no_edge():
    post = posterior(0.078, 0.089, None)
    assert strength(post.t) == 0.0
    assert edge_tier(post.t)[0] == "no measured edge"
