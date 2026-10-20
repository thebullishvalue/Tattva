"""The KNOBS registry (core/knobs.py): every decision constant is declared, and none is a stray literal."""
import ast
import os

import pytest

import core.config as config
import core.knobs as knobs
from core.knobs import KNOBS, Knob, knob, register_measured

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Numbers the grade, trust and the semantic words may write as literals: the
#: arithmetic identities, percent, the half and a division guard. Every other
#: number they use is a registered knob.
ALLOWED = {0, 1, 2, 3, 100, 0.5, 1e-12}
#: Unit constants allowed at one named site: Swayam's oscillators are published
#: on a ±10 scale (engines/swayam/kernel.py), and oscillator_lean leaves with
#: its only consumers, the Convergence package.
UNIT_SITES = {("core/semantics.py", "oscillator_lean"): {10.0}}
LINTED = ("core/grade.py", "core/trust.py", "core/semantics.py")


def test_every_knob_declares_kind_consumer_coupling_and_reason():
    assert KNOBS
    for name, k in KNOBS.items():
        assert isinstance(k, Knob) and k.name == name
        assert k.kind in ("structural", "structural-inherited", "measured")
        assert k.consumer.strip() and k.horizon_coupling.strip() and k.reason.strip(), name
        if k.ref:
            mod, attr = k.ref.rsplit(".", 1)
            assert mod == "core.config" and getattr(config, attr) == k.value, name


def test_the_named_knobs_are_registered_with_their_values():
    assert knob("EXECUTION_LAG") == 1
    assert (knob("STRENGTH_T0"), knob("STRENGTH_T1")) == (1.645, 2.5)
    assert (knob("CLAIM_SIGMA"), knob("DEEP_SIGMA"), knob("SWAYAM_DEAD")) == (0.5, 2.0, 0.10)
    assert (knob("GRADE_WARMUP_WIN"), knob("DRIFT_MIN_RESOLVED"), knob("GRADE_N_BLOCKS")) == (30, 60, 6)
    for name in ("SWAYAM_OVERSOLD", "SWAYAM_OVERBOUGHT", "SWAYAM_LENGTHS", "SWAYAM_ROC_FRAC",
                 "SWAYAM_MMR_NUM_VARS", "SWAYAM_REGIME_SENSITIVITY", "SWAYAM_BASE_WEIGHT"):
        assert KNOBS[name].kind == "structural-inherited" and "not re-validated" in KNOBS[name].reason
    for name in ("SIZERS_ENABLED", "TIER_RULE", "CLAIM_CENTRING", "POOLED_PRIOR"):
        assert KNOBS[name].reason == "set by studies/pooled_priors.py"
    with pytest.raises(KeyError):
        knob("NO_SUCH_KNOB")


def test_a_measured_knob_names_a_study_file_that_exists(monkeypatch):
    monkeypatch.setattr(knobs, "KNOBS", dict(KNOBS))
    register_measured("SIZERS_ENABLED", False, consumer="composer", study="studies/se_calibration.py",
                      date="2026-10-19")
    for name, k in knobs.KNOBS.items():
        if k.kind == "measured":
            assert k.study and k.date, name
            assert os.path.isfile(os.path.join(ROOT, k.study)), (name, k.study)
    assert knobs.knob("SIZERS_ENABLED") is False
    # The writer refuses a study that is not in the repository, and never
    # overwrites a structural knob.
    with pytest.raises(ValueError):
        register_measured("TIER_RULE", "causal_tercile", consumer="composer",
                          study="studies/not_committed.py", date="2026-10-19")
    with pytest.raises(ValueError):
        register_measured("EXECUTION_LAG", 0, consumer="estimand", study="studies/se_calibration.py",
                          date="2026-10-19")


def _numeric_literals(path):
    """(value, enclosing function) for every number written in ``path``, outside f-strings."""
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    out = []

    def walk(node, func):
        if isinstance(node, ast.JoinedStr):           # f-string formatting
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            out.append((node.value, func, node.lineno))
        for child in ast.iter_child_nodes(node):
            walk(child, func)

    walk(tree, None)
    return out


def test_no_unregistered_literal_in_the_grade_trust_or_words():
    stray = [(path, line, value) for path in LINTED for value, func, line in _numeric_literals(path)
             if value not in ALLOWED and value not in UNIT_SITES.get((path, func), set())]
    assert not stray, f"register these in core.knobs: {stray}"
