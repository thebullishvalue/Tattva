"""The pooled-prior study (studies/pooled_priors.py, studies/_pool.py) on synthetic members.

tests/fixtures/priors_synthetic.json holds the study universe's class
structure with explicit as-of claim grades on a quarterly grid, and the specs
these tests draw payoff products, planted catalogues and pipeline-shaped
members from. Asserted:

  (a) leave-one-out: planting IC 0.9 on a member leaves its own prior unchanged
      at every vintage (and moves everyone else's);
  (b) causality: a vintage appended after t, with the data behind it, never
      changes the lookup at t, and there is nothing before the first vintage;
  (c) the scope rule: class-LOO with at least POOL_MIN_MEMBERS other graded
      members of the class, global-LOO otherwise — on the universe's classes,
      Gold is class-LOO with k = 7 and USD/INR global-LOO;
  (d) twins at ρ = 0.95: κ approaches se_i·se_j and the posterior gains under
      10% precision;
  (e) a planted catalogue (k = 7, ρ = 0.3, global null): one look opens at
      most 6.5% of the time, with mean strength at most 0.03;
  (f) an R4 or an R6 failure writes no file and exits non-zero; with both
      passing the study writes, and twice on the same inputs writes the same
      bytes.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import studies.pooled_priors as study
from core.config import InstrumentConfig
from core.instruments import instrument
from core.knobs import knob
from core.trust import Prior, PriorTable, posterior
from pipeline.assemble import _terciles, gather_inputs, grade_series
from studies._pool import class_all_prior, dl_tau2, loo_prior, payoff_rho, simulate_looks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "priors_synthetic.json"), encoding="utf-8"))
H = int(FIX["h"])
LAG = knob("EXECUTION_LAG")
MIN_OTHERS = knob("POOL_MIN_MEMBERS")
WARMUP = knob("GRADE_WARMUP_WIN")


# ─── The synthetic catalogue ─────────────────────────────────────────────────

def _universe() -> tuple[list[str], list[str]]:
    names, classes = [], []
    for c, members in FIX["classes"].items():
        names += members
        classes += [c] * len(members)
    return names, classes


def _grades(plant: dict | None = None, upto=None) -> dict[str, pd.DataFrame]:
    """The fixture's as-of claim grades, by grid date; ``plant`` overrides ICs."""
    grid = pd.DatetimeIndex(FIX["grid"])
    out = {}
    for m, rows in FIX["grades"]["claim"].items():
        g = pd.DataFrame(rows, index=grid, columns=["ic", "se", "n_win"], dtype=float)
        if plant and m in plant:
            g["ic"] = plant[m]
        out[m] = g if upto is None else g.loc[:upto]
    return out


def _payoff_pairs(names, classes, upto=None) -> dict[str, pd.DataFrame]:
    """Graded pairs whose products u correlate at ``class_rho`` within a class and
    at the twins' ρ between the twins (x, y independent: corr(u) = corr(x)·corr(y))."""
    spec = FIX["payoff"]
    rng = np.random.default_rng(spec["seed"])
    n = spec["n_sessions"]
    dates = pd.bdate_range(spec["start"], periods=n)
    known = dates[LAG + H:]
    fac = {c: rng.standard_normal((2, n)) for c in dict.fromkeys(classes)}
    twin_of = {a: (b, r) for a, b, r in spec["twins"]} | {b: (a, r) for a, b, r in spec["twins"]}
    twin_fac = {frozenset((a, b)): rng.standard_normal((2, n)) for a, b, _ in spec["twins"]}
    out = {}
    for m, c in zip(names, classes):
        if m in twin_of:
            other, r = twin_of[m]
            common, load = twin_fac[frozenset((m, other))], math.sqrt(r)
        else:
            common, load = fac[c], math.sqrt(spec["class_rho"])
        own = rng.standard_normal((2, n))
        xy = math.sqrt(load) * common + math.sqrt(1 - load) * own
        p = pd.DataFrame({"x": xy[0][:len(known)], "y": xy[1][:len(known)], "s": 1.0,
                          "known": known}, index=dates[:len(known)])
        out[m] = p if upto is None else p[p["known"] <= upto]
    return out


def _windows() -> np.ndarray:
    spec = FIX["payoff"]
    return pd.bdate_range(spec["start"], periods=spec["n_sessions"])[::H].to_numpy(dtype="datetime64[ns]")


def _pool(grades, pairs, vints):
    names, classes = _universe()
    return study.pool_object(names, classes, grades, pairs, [pd.Timestamp(v) for v in vints], _windows())


# ─── (a) leave-one-out ───────────────────────────────────────────────────────

def test_a_planted_skill_never_raises_its_own_prior():
    names, classes = _universe()
    pairs = _payoff_pairs(names, classes)
    base, _ = _pool(_grades(), pairs, FIX["grid"])
    planted, _ = _pool(_grades(plant={"Silver": 0.9}), pairs, FIX["grid"])
    assert base["Silver"] and base["Silver"] == planted["Silver"]
    # The plant is visible to every other commodity member once Silver is graded.
    moved = [v for v in base["Copper"] if base["Copper"][v]["mu"] != planted["Copper"][v]["mu"]]
    assert moved and all(planted["Copper"][v]["mu"] > base["Copper"][v]["mu"] for v in moved)


def test_dl_tau2_and_the_loo_prior_by_hand():
    ic = np.array([0.10, 0.02, -0.04, 0.05, 0.01])
    se = np.array([0.05, 0.06, 0.05, 0.07, 0.04])
    cls = np.array(["a", "a", "a", "b", "b"], dtype=object)
    w = 1 / se ** 2
    q = df = d = 0.0
    for c in ("a", "b"):
        m = cls == c
        mu = (w[m] @ ic[m]) / w[m].sum()
        q += w[m] @ (ic[m] - mu) ** 2
        d += w[m].sum() - (w[m] @ w[m]) / w[m].sum()
    df = len(ic) - 2
    assert dl_tau2(ic, se, cls) == pytest.approx(max(0.0, (q - df) / d))
    rho = np.eye(5)
    e = loo_prior(0, ic, se, cls, rho, min_members=2)
    others = np.arange(1, 5)
    t2 = dl_tau2(ic[others], se[others], cls[others])
    pool = np.array([1, 2])
    ws = 1 / (se[pool] ** 2 + t2)
    assert e["scope"] == "class-LOO" and e["k"] == 2 and e["tau2"] == pytest.approx(t2)
    assert e["mu"] == pytest.approx((ws @ ic[pool]) / ws.sum())
    assert e["var_mu"] == pytest.approx((ws ** 2 @ (se[pool] ** 2 + t2)) / ws.sum() ** 2)
    assert e["kappa"] == 0.0
    # The class's all-member prior for an instrument outside the study.
    allc = class_all_prior("a", ic, se, cls, rho, kappa=np.zeros(5), min_members=2)
    assert allc["scope"] == "class-all" and allc["k"] == 3 and allc["se_floor"] == pytest.approx(
        math.sqrt(allc["var_mu"]))


# ─── (b) causality ───────────────────────────────────────────────────────────

def test_b_an_appended_vintage_never_changes_an_earlier_lookup():
    names, classes = _universe()
    grid = FIX["grid"]
    cut = pd.Timestamp(grid[-5])
    early, _ = _pool(_grades(upto=cut), _payoff_pairs(names, classes, upto=cut), grid[:-4])
    late, _ = _pool(_grades(), _payoff_pairs(names, classes), grid)
    meta = {"study": "studies/pooled_priors.py", "run_date": "2025-01-01"}
    t_early, t_late = PriorTable({"claim": early}, meta), PriorTable({"claim": late}, meta)
    first = pd.Timestamp(grid[0])
    for t in pd.date_range(first - pd.Timedelta(days=60), pd.Timestamp(grid[-4]) - pd.Timedelta(days=1), freq="5D"):
        for m, c in zip(names, classes):
            a, b = t_early.lookup("claim", m, c, t), t_late.lookup("claim", m, c, t)
            assert a == b, (m, t)
            if t < first:
                assert a is None
    assert t_late.lookup("claim", "Gold", "commodity", grid[-1]) is not None


# ─── (c) the scope rule ──────────────────────────────────────────────────────

def test_c_a_class_with_fewer_than_three_others_falls_back_to_global_loo():
    names, classes = _universe()
    entries, _ = _pool(_grades(), _payoff_pairs(names, classes), FIX["grid"])
    g = _grades()
    seen = set()
    for v in FIX["grid"]:
        graded = {m for m in names if g[m].at[pd.Timestamp(v), "n_win"] >= WARMUP}
        for m, c in zip(names, classes):
            e = entries.get(m, {}).get(v)
            if e is None:
                continue
            others = [o for o, oc in zip(names, classes) if oc == c and o != m and o in graded]
            want = "class-LOO" if len(others) >= MIN_OTHERS else "global-LOO"
            assert e["scope"] == want, (m, v)
            assert e["k"] == (len(others) if want == "class-LOO" else len(graded - {m}))
            seen.add(want)
    assert seen == {"class-LOO", "global-LOO"}
    last = FIX["grid"][-1]
    assert (entries["Gold"][last]["scope"], entries["Gold"][last]["k"]) == ("class-LOO", 7)
    assert entries["USD/INR"][last]["scope"] == "global-LOO"
    assert entries["S&P 500"][last]["scope"] == "global-LOO"      # us_index: 2 others
    assert entries["__class__:commodity"][last]["scope"] == "class-all"


# ─── (d) twins ───────────────────────────────────────────────────────────────

def test_d_twins_add_almost_no_precision():
    a, b, r = FIX["payoff"]["twins"][0]
    names, classes = _universe()
    pairs = _payoff_pairs(names, classes)
    u = {m: study._u(pairs[m], None, _windows()) for m in (a, b, "Gold")}
    rho = payoff_rho(u)
    assert rho.at[a, b] == pytest.approx(r, abs=0.05) and abs(rho.at[a, "Gold"]) < 0.25
    last = _grades()
    ic = np.array([last[a]["ic"].iloc[-1], last[b]["ic"].iloc[-1]])
    se = np.array([last[a]["se"].iloc[-1], last[b]["se"].iloc[-1]])
    cls = np.array(["india_index"] * 2, dtype=object)

    def gain(rho2):
        e = loo_prior(0, ic, se, cls, rho2)
        post = posterior(ic[0], se[0], Prior(vintage=pd.NaT, study="", run_date="", **e))
        return e, (se[0] / post.se) ** 2 - 1

    twin = rho.loc[[a, b], [a, b]].to_numpy()
    e, g = gain(twin)
    assert e["kappa"] / (se[0] * se[1]) >= 0.9 and g < 0.10
    # The same second measurement, independent: a real gain.
    assert gain(np.eye(2))[1] > 0.5


# ─── (e) false opens on a planted null catalogue ─────────────────────────────

def test_e_a_null_catalogue_rarely_opens():
    p = FIX["planted"]
    rng = np.random.default_rng(p["seed"])
    cls = np.repeat([f"c{j}" for j in range(p["n_classes"])], p["k"]).astype(object)
    se = rng.uniform(*p["se"], len(cls))
    rho = np.where(cls[:, None] == cls[None, :], p["rho"], 0.0)
    np.fill_diagonal(rho, 1.0)
    out = simulate_looks(se, cls, rho, theta=0.0, tau=0.0, n_sim=p["n_sim"], seed=p["seed"])
    assert set(out["scope"]) == {"class-LOO"}
    assert out["open"].mean() <= p["max_false_open"]
    assert out["strength"].mean() <= p["max_strength"]


# ─── (f) the whole study on pipeline-shaped synthetic members ────────────────

def _member(name: str, rng, common: dict, spec: dict) -> dict:
    """One member as ``run_member`` records it, from a synthetic Mūla frame and
    Swayam read run through the app's own loop pieces (gather_inputs, grade_series)."""
    n, burn = spec["n_sessions"], spec["burn_in"]
    idx = pd.bdate_range(spec["start"], periods=n)
    c = common[instrument(name).asset_class]
    ret = 0.9 * c["ret"] + math.sqrt(1 - 0.81) * rng.normal(0, 0.01, n)
    fvo = 0.9 * c["fvo"] + math.sqrt(1 - 0.81) * rng.normal(0, 1.0, n)
    fvo[:burn] = np.nan
    price = 100 * np.exp(np.cumsum(ret))
    mr = 0.5 + 0.3 * np.tanh(np.cumsum(rng.normal(0, 0.05, n)))
    lean = np.clip(rng.normal(0, 0.3, n), -1, 1)
    fvo_ts = pd.DataFrame({"FVO": fvo, "Price": price, "PctMispricing": 0.02 * fvo,
                           "Valid": np.isfinite(fvo), "Provisional": False, "WithheldReason": "",
                           "MRProb": mr, "MulaKappa": 0.001}, index=idx)
    sw = pd.DataFrame({"Lean": lean, "_Native": rng.uniform(size=n) > 0.05}, index=idx)
    inst = instrument(name)
    inputs = gather_inputs(fvo_ts, sw, instrument=inst, h=H, today=idx[-1])
    grades = grade_series(inputs, inputs["price"], h=H, exec_lag=LAG, priors=None, instrument=inst)
    return {"target": name, "asset_class": inst.asset_class, "ticker": inst.ticker, "h": H,
            "exec_lag": LAG, "asof": idx[-1], "start": idx[0], "fingerprint": f"fp-{name}",
            "fingerprint_upto": idx[-2], "inputs": inputs,
            "grades": {o: g[["ic", "se", "n_win", "grid_date"]] for o, g in grades.items()},
            "gap_split": None}


@pytest.fixture(scope="module")
def records():
    spec = FIX["pipeline"]
    rng = np.random.default_rng(spec["seed"])
    n = spec["n_sessions"]
    common = {}
    for c in dict.fromkeys(spec["members"].values()):
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.98 * f[t - 1] + 0.2 * rng.standard_normal()
        common[c] = {"fvo": f / f.std(), "ret": rng.normal(0.0002, 0.01, n) - 0.0005 * np.roll(f, 1)}
    return {m: _member(m, rng, common, spec) for m in spec["members"]}


@pytest.fixture
def quick(monkeypatch):
    """Fewer draws, shifts and simulations: the same code, sooner."""
    for name, value in (("BOOT_DRAWS", 30), ("R4_SHIFTS", 40), ("R5_SIMS", 40), ("R6_DRAWS", 40)):
        monkeypatch.setattr(study, name, value)


def _run(records, out, previous=None):
    run_date = max(r["inputs"].index[-1] for r in records.values())
    names = list(records)
    result = study.analyse(records, universe_names=names, run_date=run_date, previous=previous)
    code = study.finish(result, records, out, universe_names=names, failures={}, run_date=run_date,
                        git={"sha": "test", "dirty": []}, mode="test", complete=False)
    return result, code


def test_f_an_r4_failure_writes_nothing(records, quick, monkeypatch, tmp_path):
    real = study.brute_rotation_sd
    monkeypatch.setattr(study, "brute_rotation_sd", lambda *a, **k: 0.5 * real(*a, **k))
    out = tmp_path / "priors.json"
    result, code = _run(records, out)
    assert code == 2 and not out.exists()
    assert not result["checks"]["R4"]["claim"]["pass"] and result["abort"][0].startswith("R4")


def test_f_an_r6_failure_writes_nothing(records, quick, monkeypatch, tmp_path):
    # The members' errors are strongly correlated within a class; a closed form
    # told they are independent (and that true ICs do not differ) understates
    # the class mean's variance, which the joint bootstrap sees.
    monkeypatch.setattr(study, "payoff_rho", lambda u, m=60: pd.DataFrame(
        np.eye(len(u)), index=list(u), columns=list(u)))
    monkeypatch.setattr(study, "dl_tau2", lambda *a: 0.0)
    out = tmp_path / "priors.json"
    result, code = _run(records, out)
    assert code == 2 and not out.exists()
    fails = [a for a in result["abort"] if a.startswith("R6")]
    assert fails, result["abort"]


def test_f_the_study_writes_its_schema_and_reproduces_byte_for_byte(records, quick, tmp_path):
    from test_priors import check_schema

    one = tmp_path / "one.json"
    result, code = _run(records, one)
    assert code == 0, result["abort"]
    data = json.loads(one.read_text(encoding="utf-8"))
    check_schema(data)
    # Every member is held out (none is burned or correlated with one), so the
    # decisions are made on all of them, and each is a recorded value.
    assert data["held_out"]["members"] == list(records)
    assert data["decisions"]["A"]["value"] in ("none", "robust_z_252")
    assert isinstance(data["decisions"]["B"]["value"], bool)
    assert data["decisions"]["C"]["value"] in ("none", "causal_tercile")
    assert data["decisions"]["B"]["delta_ic"]["n_members"] == len(records)
    assert data["arms"]["F"]["status"].startswith("first run")
    # Against an earlier study (this one, dated 2021-06-30): arm F grades the calls after it.
    later = json.loads(one.read_text())
    later["run_date"] = "2021-06-30"
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps(later))
    assert _run(records, tmp_path / "three.json", previous=earlier)[1] == 0
    f = json.loads((tmp_path / "three.json").read_text())["arms"]["F"]
    assert f["since"] == "2021-06-30" and set(f["per_member"]) == set(records)
    # Rerun on the same inputs — its own output beside it, which is no previous
    # study — the study writes the same bytes.
    assert _run(records, tmp_path / "again.json", previous=one)[1] == 0
    assert (tmp_path / "again.json").read_bytes() == one.read_bytes()


def test_the_tercile_rule_is_the_pipelines():
    rng = np.random.default_rng(0)
    size = pd.Series(rng.uniform(0.1, 1.5, 300))
    y = pd.Series(rng.normal(0, 0.01, 300))
    lab = study._tercile_labels(size.to_numpy())
    mine = {f"T{k}": float(y[lab == k].mean()) for k in (1, 2, 3)}
    theirs = _terciles(size, y, H)
    assert all(mine[t] == pytest.approx(theirs[t]["mean"]) for t in mine)


@pytest.fixture
def registry(monkeypatch):
    """The study registers its ten stocks as the app registers a typed symbol;
    keep that out of every other test's catalogue."""
    import core.config as config
    import core.instruments as instruments

    for mod, name in ((config, "ALL_TARGETS"), (config, "STOCK_TARGET_MARKETS"),
                      (config, "TARGET_EXCLUDED_PREDICTORS"), (config, "INSTRUMENT_CONFIGS"),
                      (instruments, "INSTRUMENTS")):
        monkeypatch.setattr(mod, name, dict(getattr(mod, name)))


def test_the_cli_runs_offline_and_resumes(registry, quick, tmp_path, capsys):
    out, wd = tmp_path / "out.json", tmp_path / "work"
    argv = ["--out", str(out), "--offline-fixture", "--members", "Gold", "--workdir", str(wd),
            "--no-arm-p", "--allow-dirty"]
    assert study.main(argv) == 0
    first = out.read_bytes()
    data = json.loads(first)
    assert (data["mode"], data["complete"], data["members"]) == ("offline-fixture", False, ["Gold"])
    assert data["fingerprint_upto"]["Gold"] < data["run_date"] == "2026-10-16"
    assert json.loads((wd / "timings.json").read_text())["Gold"] > 0
    # A rerun resumes from the cached member and writes the same bytes.
    capsys.readouterr()
    assert study.main(argv) == 0
    assert "Gold: cached" in capsys.readouterr().out and out.read_bytes() == first
    assert study.main(["--out", str(out), "--members", "Nowhere"]) == 1


def test_the_universe_is_the_pre_registered_one(registry):
    names = study.universe()
    assert len(names) == 38 and "Nifty 50 - PE" not in names
    counts = pd.Series([instrument(m).asset_class for m in names]).value_counts().to_dict()
    assert counts == {"commodity": 8, "fx": 2, "crypto": 7, "india_index": 8, "us_index": 3,
                      "stock_us": 5, "stock_india": 5}
    assert study.p_members() == ["Gold", "USD/INR", "Bitcoin", "Nifty 50", "S&P 500",
                                 "AAPL (US)", "RELIANCE (NSE)"]
    assert InstrumentConfig().forecast_horizon == H
