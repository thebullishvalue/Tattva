"""The committed pooled-prior study output, and the app with and without it.

``data/priors/pooled_priors_h10_L1.json`` is written by ``studies/pooled_priors.py``
on the network and committed after the script that wrote it. Against the
committed file (skipped, saying why, until it exists):

  * the schema holds, and no prior can contain its member or the future;
  * ``script_sha256`` is the committed script's SHA-256, so the numbers cannot
    drift from the code that produced them;
  * its h and L are the knobs', and KNOBS registers its results as measured,
    naming the study and its run date;
  * the call at every session still never rewrites a resolved session with
    the real priors loaded.

Whether or not it exists:

  * with no file the app runs and labels 'no pooled prior: per-instrument
    grade only';
  * a run compared with the study hashes its history up to the session the
    study hashed it to, so its fingerprint matches until that history changes;
  * the knobs a study file registers, read from a file of the same shape.
"""
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import core.knobs as knobs
import core.trust as trust
from core.call import CALL_COLUMNS
from core.config import FORECAST_HORIZON, InstrumentConfig
from core.instruments import instrument
from core.knobs import KNOBS, PRIORS_DIR, knob, priors_file_name
from core.semantics import breadth_lean
from core.trust import PriorTable, load_priors
from data.prep import history_fingerprint
from pipeline.assemble import NO_PRIOR_LABEL, assemble

ROOT = PRIORS_DIR.parent.parent
PATH = PRIORS_DIR / priors_file_name(FORECAST_HORIZON, knob("EXECUTION_LAG"))
STUDY = "studies/pooled_priors.py"
OBJECTS = ("claim", "sizer:swayam", "sizer:reversion", "call")
ENTRY = {"mu", "var_mu", "tau2", "kappa", "se_floor", "k", "scope"}
SCOPES = {"class-LOO", "global-LOO", "class-all", "global-all"}
TOP = {"study", "git_sha", "git_dirty", "script_sha256", "run_date", "epoch", "h", "exec_lag",
       "knobs_hash", "code_version", "mode", "complete", "universe", "members", "failures",
       "held_out", "classes", "fingerprints", "fingerprint_upto", "vintages", "priors", "summary",
       "decisions", "arms", "checks"}

needs_file = pytest.mark.skipif(
    not PATH.is_file(),
    reason=f"{PATH.relative_to(ROOT)} is not committed yet: studies/pooled_priors.py has not "
           f"been run on the network (the orchestrator runs it after this package)")


def check_schema(data: dict) -> None:
    """The study output's shape (also asserted on synthetic output by test_study.py)."""
    assert TOP <= set(data), sorted(TOP - set(data))
    assert data["study"] == STUDY
    members = data["members"]
    assert set(data["fingerprints"]) == set(members) == set(data["fingerprint_upto"])
    assert set(members) <= set(data["universe"]) and not set(members) & set(data["failures"])
    run_date = data["run_date"]
    assert data["vintages"] == sorted(data["vintages"]) and all(v <= run_date for v in data["vintages"])
    assert all(pd.Timestamp(v).is_quarter_end for v in data["vintages"])
    assert all(u <= run_date for u in data["fingerprint_upto"].values())
    cls_of = {m: data["universe"][m]["class"] for m in members}
    sizes = pd.Series(cls_of).value_counts().to_dict()
    assert set(data["priors"]) == set(OBJECTS)
    for obj, by_who in data["priors"].items():
        for who, by_v in by_who.items():
            if who.startswith("__class__:"):
                assert who.removeprefix("__class__:") in data["classes"]
            else:
                assert who in members, who
            for v, e in by_v.items():
                assert v in data["vintages"] and set(e) == ENTRY and e["scope"] in SCOPES, (obj, who, v)
                assert all(isinstance(e[f], (int, float)) and math.isfinite(e[f]) for f in ENTRY - {"scope"})
                assert e["var_mu"] >= 0 and e["tau2"] >= 0 and e["se_floor"] >= 0 and e["k"] >= 1
                # Leave-one-out: a member's pool never holds the member itself.
                if e["scope"] == "class-LOO":
                    assert e["k"] <= sizes[cls_of[who]] - 1
                elif e["scope"] == "global-LOO":
                    assert e["k"] <= len(members) - 1
    for cls, by_obj in data["summary"].items():
        assert cls in data["classes"]
        for obj, row in by_obj.items():
            assert obj in OBJECTS and set(row) == {"mu", "se_mu", "tau", "k", "asof"}
    for key, values in (("A", {"none", "robust_z_252"}), ("B", {True, False}),
                        ("C", {"none", "causal_tercile"})):
        assert data["decisions"][key]["value"] in values
    assert set(data["arms"]) == {"E", "F", "P", "G"} and set(data["checks"]) == {"R4", "R5", "R6"}
    assert all(r["pass"] for r in data["checks"]["R4"].values())
    assert set(data["checks"]["R6"]) == set(OBJECTS) and all(r["pass"] for r in data["checks"]["R6"].values())


def _committed() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


# ─── The committed file ──────────────────────────────────────────────────────

@needs_file
def test_the_schema_holds():
    data = _committed()
    check_schema(data)
    assert data["mode"] == "network" and data["git_dirty"] == []
    table = load_priors(data["h"], data["exec_lag"])
    first = pd.Timestamp(data["vintages"][0])
    for m in data["members"]:
        assert table.lookup("claim", m, data["universe"][m]["class"], first - pd.Timedelta(days=1)) is None


@needs_file
def test_the_numbers_are_the_committed_scripts():
    assert _committed()["script_sha256"] == hashlib.sha256((ROOT / STUDY).read_bytes()).hexdigest()


@needs_file
def test_h_and_l_are_the_knobs():
    data = _committed()
    assert data["h"] == FORECAST_HORIZON == InstrumentConfig().forecast_horizon
    assert data["exec_lag"] == knob("EXECUTION_LAG")
    # The study registers its STUDY_STOCKS the way the app registers a typed
    # symbol; do the same here so the check does not depend on test order.
    from studies.pooled_priors import universe
    universe()
    assert all(instrument(m).asset_class == data["universe"][m]["class"] for m in data["members"])


@needs_file
def test_knobs_registers_the_study_as_measured():
    data = _committed()
    assert KNOBS["POOLED_PRIOR"].value == PATH.relative_to(ROOT).as_posix()
    for name, key in (("CLAIM_CENTRING", "A"), ("SIZERS_ENABLED", "B"), ("TIER_RULE", "C")):
        assert KNOBS[name].value == data["decisions"][key]["value"], name
    want = {f"PRIOR:{obj}:{cls}" for cls, by_obj in data["summary"].items() for obj in by_obj}
    assert set(knobs.STUDY_GENERATED) == want
    for name in ("POOLED_PRIOR", "CLAIM_CENTRING", "SIZERS_ENABLED", "TIER_RULE", *want):
        k = KNOBS[name]
        assert (k.kind, k.study, k.date) == ("measured", data["study"], data["run_date"]), name
    for cls, by_obj in data["summary"].items():
        for obj, row in by_obj.items():
            assert KNOBS[f"PRIOR:{obj}:{cls}"].value == row


def _call_ts(cut: int, priors) -> pd.DataFrame:
    """test_causality's synthetic call_ts, with ``priors`` loaded."""
    from test_causality import _fitted, _panel

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
                    asof=eng.asof, today=dates[-1], priors=priors).call_ts


@needs_file
def test_the_call_never_rewrites_a_resolved_session_with_the_real_priors():
    priors = load_priors(FORECAST_HORIZON, knob("EXECUTION_LAG"))
    full, part = _call_ts(0, priors), _call_ts(200, priors)
    h, lag = InstrumentConfig().forecast_horizon, knob("EXECUTION_LAG")
    resolved = part.index[:len(part) - (lag + h)]
    assert (part.loc[resolved, "claim_prior_scope"] != "none").any()
    for col in CALL_COLUMNS:
        a, b = full.loc[resolved, col], part.loc[resolved, col]
        if pd.api.types.is_float_dtype(a):
            assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), col
        else:
            assert all(x == y or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)), col


# ─── With no file ────────────────────────────────────────────────────────────

def test_without_the_file_the_app_runs_per_instrument_only(request, monkeypatch, tmp_path):
    if not PATH.is_file():
        at = request.getfixturevalue("gold_app")     # the app as it stands: there is no file
    else:
        from conftest import install_offline_fetch, run_app_offline

        # The file removed: nothing to load, and the registry as import leaves
        # it with no file — the placeholders, no measured entry.
        monkeypatch.setattr(trust, "PRIORS_DIR", tmp_path)
        monkeypatch.setattr(knobs, "KNOBS", {**{n: k for n, k in KNOBS.items() if k.kind != "measured"},
                                             **knobs.PLACEHOLDERS})
        install_offline_fetch(monkeypatch)
        at = run_app_offline("Gold", "Commodities")
    assert not at.exception, [e.value for e in at.exception]
    ss = at.session_state
    assert ss["priors_meta"]["label"] == NO_PRIOR_LABEL == "no pooled prior: per-instrument grade only"
    ct = ss["call_ts"]
    assert set(ct["claim_prior_scope"]) <= {"none", "ungraded"} and ct.attrs["priors_study"] is None
    held = ct["action"] == "HOLD"
    assert ct.loc[held, "reason"].str.contains("no pooled prior").all()
    for page in ss["_smoke_pages"]:
        at.session_state["_smoke_page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])


# ─── The history fingerprint is hashed to the study's session ────────────────

def test_a_run_compared_with_the_study_hashes_its_history_to_the_studied_session(offline_fetch):
    from pipeline.run import fetch_panel, prepare, target_bars

    panel = fetch_panel("Gold")
    bars, _ = target_bars("Gold")
    own = prepare(panel, "Gold", bars=bars, priors=None)
    studied = own.data.index[-60]
    table = PriorTable({}, {"study": STUDY, "run_date": "2026-01-01",
                            "fingerprint_upto": {"Gold": studied.strftime("%Y-%m-%d")}})
    cols = [own.data.columns[0], *own.features]
    then = prepare(panel, "Gold", bars=bars, priors=table)
    assert then.fingerprint == history_fingerprint(own.data, cols, studied) != own.fingerprint
    # Newer sessions do not move it; a revised value on or before the studied session does.
    later = prepare(panel.iloc[:-20], "Gold", bars=bars, priors=table)
    assert later.fingerprint == then.fingerprint
    revised = panel.copy()
    row = revised.index[pd.to_datetime(revised["DATE"]).dt.normalize() == own.data.index[-100]][0]
    revised.loc[row, "Gold"] = revised.loc[row, "Gold"] * 1.01
    assert prepare(revised, "Gold", bars=bars, priors=table).fingerprint != then.fingerprint
    # An instrument the study did not run is hashed to its newest settled session.
    assert table.fingerprint_upto("Silver") is None


# ─── What a study file registers ─────────────────────────────────────────────

def _study_file(tmp_path, *, h=FORECAST_HORIZON, lag=None):
    data = {"study": STUDY, "run_date": "2026-10-26", "h": h,
            "exec_lag": knob("EXECUTION_LAG") if lag is None else lag,
            "decisions": {"A": {"value": "none"}, "B": {"value": False}, "C": {"value": "none"}},
            "summary": {"commodity": {"claim": {"mu": 0.031, "se_mu": 0.02, "tau": 0.01, "k": 8,
                                                "asof": "2026-09-30"}},
                        "fx": {"call": {"mu": -0.01, "se_mu": 0.05, "tau": 0.0, "k": 2,
                                        "asof": "2026-09-30"}}}}
    p = tmp_path / priors_file_name(h, data["exec_lag"])
    p.write_text(json.dumps(data))
    return p


def _mini_repo(monkeypatch, tmp_path):
    """The registry as import leaves it with no file, rooted at ``tmp_path`` (which
    holds a stand-in study script: a measured knob must name a study file)."""
    (tmp_path / "studies").mkdir()
    (tmp_path / STUDY).write_text("")
    monkeypatch.setattr(knobs, "KNOBS", {**{n: k for n, k in KNOBS.items() if k.kind != "measured"},
                                         **knobs.PLACEHOLDERS})
    monkeypatch.setattr(knobs, "STUDY_GENERATED", [])
    monkeypatch.setattr(knobs, "_ROOT", tmp_path)


def test_a_study_file_registers_measured_knobs(monkeypatch, tmp_path):
    _mini_repo(monkeypatch, tmp_path)
    before = knobs.knobs_hash()
    knobs._register_study(_study_file(tmp_path))
    k = knobs.KNOBS
    assert k["SIZERS_ENABLED"].value is False and k["TIER_RULE"].value == "none"
    assert k["CLAIM_CENTRING"].value == "none"
    assert k["POOLED_PRIOR"].value == priors_file_name(FORECAST_HORIZON, knob("EXECUTION_LAG"))
    assert knobs.STUDY_GENERATED == ["PRIOR:claim:commodity", "PRIOR:call:fx"]
    assert k["PRIOR:claim:commodity"].value == {"mu": 0.031, "se_mu": 0.02, "tau": 0.01, "k": 8,
                                                "asof": "2026-09-30"}
    for name in ("POOLED_PRIOR", "CLAIM_CENTRING", "SIZERS_ENABLED", "TIER_RULE", *knobs.STUDY_GENERATED):
        assert (k[name].kind, k[name].study, k[name].date) == ("measured", STUDY, "2026-10-26")
    # What the study's numbers assume does not move when its results are registered.
    assert knobs.knobs_hash() == before


def test_a_missing_or_mismatched_file_leaves_the_placeholders(monkeypatch, tmp_path):
    _mini_repo(monkeypatch, tmp_path)
    knobs._register_study(tmp_path / "missing.json")
    knobs._register_study(_study_file(tmp_path, h=FORECAST_HORIZON + 10))
    knobs._register_study(_study_file(tmp_path, lag=1 - knob("EXECUTION_LAG")))
    assert all(knobs.KNOBS[n] is knobs.PLACEHOLDERS[n] for n in knobs.PLACEHOLDERS)
    assert knobs.STUDY_GENERATED == []
