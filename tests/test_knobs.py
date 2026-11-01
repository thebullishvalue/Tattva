"""The KNOBS registry (core/knobs.py): every decision constant is declared, and none is a stray literal.

Two lints hold the registry to the code:

  * the grade, trust and the semantic words write no number but an arithmetic
    identity or a knob (``LINTED``);
  * no decision-path module defines a numeric module-level constant that no
    knob refers to (``DECISION_PATH``) — the engines' publication gate, the
    factor bank, the error-correction layer and Swayam's screen included —
    unless it is declared below as evidence-only (it moves nothing a decision
    reads) or a numerical guard. Numbers written inside an engine's function
    bodies are not linted: they are the estimators' arithmetic.
"""
import ast
import importlib
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
#: Unit constants allowed at one named site: the MAD-to-SD factor of a
#: normal (1/Φ⁻¹(3/4)) in Decision A's candidate claim unit.
UNIT_SITES: dict[tuple[str, str], set] = {("core/semantics.py", "claim_oscillator"): {1.4826}}
LINTED = ("core/grade.py", "core/trust.py", "core/semantics.py")


def test_every_knob_declares_kind_consumer_coupling_and_reason():
    assert KNOBS
    for name, k in KNOBS.items():
        assert isinstance(k, Knob) and k.name == name
        assert k.kind in ("structural", "structural-inherited", "measured")
        assert k.consumer.strip() and k.horizon_coupling.strip() and k.reason.strip(), name
        if k.ref:
            mod, attr = k.ref.rsplit(".", 1)
            assert getattr(importlib.import_module(mod), attr) == k.value, name


def test_the_named_knobs_are_registered_with_their_values():
    assert knob("EXECUTION_LAG") == 1
    assert (knob("STRENGTH_T0"), knob("STRENGTH_T1")) == (1.645, 2.5)
    assert (knob("CLAIM_SIGMA"), knob("DEEP_SIGMA"), knob("SWAYAM_DEAD")) == (0.5, 2.0, 0.10)
    assert (knob("GRADE_WARMUP_WIN"), knob("DRIFT_MIN_RESOLVED"), knob("GRADE_N_BLOCKS")) == (30, 60, 6)
    for name in ("SWAYAM_OVERSOLD", "SWAYAM_OVERBOUGHT", "SWAYAM_LENGTHS", "SWAYAM_ROC_FRAC",
                 "SWAYAM_MMR_NUM_VARS", "SWAYAM_REGIME_SENSITIVITY", "SWAYAM_BASE_WEIGHT"):
        assert KNOBS[name].kind == "structural-inherited" and "not re-validated" in KNOBS[name].reason
    # The study's placeholders: their defaults, or — once its output is
    # committed — the measured values it registered (tests/test_priors.py).
    for name in ("SIZERS_ENABLED", "TIER_RULE", "CLAIM_CENTRING", "POOLED_PRIOR"):
        assert knobs.PLACEHOLDERS[name].reason == "set by studies/pooled_priors.py"
        assert KNOBS[name] is knobs.PLACEHOLDERS[name] or (
            KNOBS[name].kind == "measured" and KNOBS[name].study == "studies/pooled_priors.py"), name
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


#: The decision-path modules: everything between the fetched panel and the call.
DECISION_PATH = ("engines/mula/base.py", "engines/mula/valuation.py", "engines/mula/factors.py",
                 "engines/mula/ecm.py", "engines/swayam/kernel.py", "engines/swayam/ensemble.py",
                 "analytics/causal.py", "data/prep.py", "pipeline/run.py", "pipeline/assemble.py",
                 "core/call.py", "core/grade.py", "core/trust.py", "core/semantics.py")
#: Module-level numbers a decision-path module may define without a knob, and why.
UNREGISTERED_OK: dict[str, str] = {
    "engines/mula/valuation.py:CI_LEVEL": "evidence:fair_value: the 95% band's coverage target",
    "engines/mula/valuation.py:CI_MIN_OBS": "evidence:fair_value: residuals before the band "
                                            "goes empirical",
    "engines/mula/valuation.py:CI_ADAPT_GAMMA": "evidence:fair_value: the band's adaptive step",
    "engines/mula/valuation.py:CI_ALPHA_MIN": "evidence:fair_value: the band's widest level",
    "engines/mula/valuation.py:CI_ALPHA_MAX": "evidence:fair_value: the band's narrowest level",
    "analytics/causal.py:EPS": "numerical guard: a division floor (1e-12)",
    "analytics/causal.py:VAR_FLOOR": "numerical guard: a variance floor (1e-10)",
}


def _module_numbers(path):
    """``name`` for each module-level ``NAME = <number or tuple of numbers>`` in ``path``."""
    def numeric(v):
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
            v = v.operand
        if isinstance(v, (ast.Tuple, ast.List)):
            return bool(v.elts) and all(numeric(e) for e in v.elts)
        return (isinstance(v, ast.Constant) and isinstance(v.value, (int, float))
                and not isinstance(v.value, bool))

    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    out = []
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
        if targets and numeric(node.value):
            out += [t.id for t in targets if isinstance(t, ast.Name)]
    return out


def test_no_decision_path_module_defines_an_unregistered_constant():
    refs = {k.ref for k in KNOBS.values() if k.ref}
    stray = []
    for path in DECISION_PATH:
        module = path[:-3].replace("/", ".")
        for name in _module_numbers(path):
            if f"{module}.{name}" not in refs and f"{path}:{name}" not in UNREGISTERED_OK:
                stray.append(f"{path}:{name}")
    assert not stray, f"register these in core.knobs (by ref) or declare why not: {stray}"
    # The declared exceptions still exist (else drop them).
    for key in UNREGISTERED_OK:
        path, name = key.split(":")
        assert name in _module_numbers(path), key


def test_the_lint_sees_a_stray_constant(tmp_path, monkeypatch):
    mod = tmp_path / "stray.py"
    mod.write_text("X = 0.25\nY: int = 3\nZ = (1, 2.0)\nW = -4\nNAME = 'x'\nF = 2 * 3\n")
    monkeypatch.setattr(__import__(__name__), "ROOT", str(tmp_path))
    assert _module_numbers("stray.py") == ["X", "Y", "Z", "W"]


def test_no_unregistered_literal_in_the_grade_trust_or_words():
    stray = [(path, line, value) for path in LINTED for value, func, line in _numeric_literals(path)
             if value not in ALLOWED and value not in UNIT_SITES.get((path, func), set())]
    assert not stray, f"register these in core.knobs: {stray}"
