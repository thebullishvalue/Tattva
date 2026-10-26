"""C16: nothing is published without a reader — a static consumer scan.

Every name Tattva publishes is listed from the objects that publish it: Mūla's
frame (``engines.mula.base.PUBLISHED_COLUMNS``), Swayam's daily breadth, the
call at every session (``core.call.CALL_COLUMNS``), the ``edges`` grades, the
engine's signal, the session bundle (``app._BUNDLE_KEYS``), the
``InstrumentConfig`` fields and the ``KNOBS`` registry. A name is READ when
production code outside the scope that produces it uses it: as a string
literal for a column or key (an f-string such as ``f"{s.key}_beta"`` matches
every name it can spell), as an attribute for a config field, as a literal or
its ``core.config`` constant for a knob. Docstrings, bare strings and the
targets of assignments (``df["X"] = …``) are not reads. This is the charter's
quoted-name scan (C16), made exact on the syntax tree.

The Data record (``ui/tabs/tab_data.py``) exports what it shows. A name whose
ONLY reader is that export must be listed, with its reason, in
``EXPORT_ALLOWLIST``, and every listed name must be published and exported. A
knob consumed only by the pooled-prior study, which is not committed yet, is
listed in ``STUDY_PENDING``.
"""
import ast
import dataclasses
import os
import re

import pandas as pd

import app
from core.call import CALL_COLUMNS
from core.config import InstrumentConfig
from core.knobs import KNOBS
from engines.mula.base import PUBLISHED_COLUMNS
from engines.swayam import aggregate_views

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Where readers live: every production module (the tests are not readers).
READER_DIRS = ("core", "pipeline", "engines", "ui", "data", "analytics", "studies")
#: The record's export: a reader only through EXPORT_ALLOWLIST.
EXPORT = "ui/tabs/tab_data.py"

#: Published names whose one reader is the Data record, and why they are kept.
EXPORT_ALLOWLIST: dict[str, str] = {
    "GapPhi": "the gap's per-session AR(1) coefficient the ECM projects with; the pages show "
              "it as GapHalfLife, the record keeps the coefficient itself",
}
#: Knobs whose consumer is the pooled-prior study (studies/pooled_priors.py, WP6).
STUDY_PENDING: dict[str, str] = {
    "POOL_MIN_MEMBERS": "the study's class-LOO vs global-LOO scope rule",
}


# ─── The scan ────────────────────────────────────────────────────────────────

def _files() -> list[str]:
    out = ["app.py"]
    for d in READER_DIRS:
        for base, _dirs, names in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            out += [os.path.relpath(os.path.join(base, n), ROOT) for n in names if n.endswith(".py")]
    return sorted(out)


class _Uses(ast.NodeVisitor):
    """Every (scope, kind, value) a module uses: string literals, f-string
    patterns, attribute names and bare names in a Load context."""

    def __init__(self):
        self.scope: list[str] = []
        self.uses: list[tuple[str, str, object]] = []
        self._skip: set[int] = set()

    def _here(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _named(self, node, name):
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self._named(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._named(node, node.name)

    def visit_Assign(self, node):
        if not self.scope:                           # module level: scope by its target
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names:
                self.scope.append(f"<module>:{names[0]}")
                self.generic_visit(node)
                self.scope.pop()
                return
        self.generic_visit(node)

    visit_AnnAssign = visit_Assign

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return                                   # a docstring or a bare string
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.slice, ast.Constant):
            self._skip.add(id(node.slice))           # df["X"] = … writes X
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        # An f-string that can spell a name (f"{s.key}_beta" → swayam_beta,
        # reversion_beta) reads every name it matches. One with no name-like
        # literal of its own (f"{a}_{b}") would match anything, and is no read.
        lit = "".join(str(v.value) for v in node.values if isinstance(v, ast.Constant))
        if (any(isinstance(v, ast.FormattedValue) for v in node.values)
                and sum(ch.isalnum() for ch in lit) >= 2):
            pat = "".join(re.escape(str(v.value)) if isinstance(v, ast.Constant) else r"\w+"
                          for v in node.values)
            self.uses.append((self._here(), "pattern", re.compile(pat)))
        for v in node.values:                        # names inside the {…} are uses
            if isinstance(v, ast.FormattedValue):
                self.visit(v.value)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self._skip:
            self.uses.append((self._here(), "str", node.value))

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.uses.append((self._here(), "attr", node.attr))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.uses.append((self._here(), "name", node.id))


def _scan() -> dict[str, list[tuple[str, str, object]]]:
    out = {}
    for f in _files():
        v = _Uses()
        v.visit(ast.parse(open(os.path.join(ROOT, f), encoding="utf-8").read()))
        out[f] = v.uses
    return out


USES = _scan()


def _in(scope_spec, path: str, scope: str) -> bool:
    """Is (path, scope) inside one of ``scope_spec``'s producer scopes?

    A spec is a path prefix (a whole file or directory) or ``path::scope``,
    which covers that scope and everything nested in it.
    """
    for spec in scope_spec:
        if "::" in spec:
            p, s = spec.split("::", 1)
            if path == p and (scope == s or scope.startswith(s + ".")):
                return True
        elif path == spec or path.startswith(spec.rstrip("/") + "/"):
            return True
    return False


def _readers(name: str, producers, kinds=("str",)) -> set[str]:
    """The files that read ``name`` outside ``producers``."""
    found = set()
    for path, uses in USES.items():
        for scope, kind, value in uses:
            if kind not in kinds and kind != "pattern":
                continue
            if _in(producers, path, scope):
                continue
            if (kind == "pattern" and "str" in kinds and value.fullmatch(name)) or \
                    (kind != "pattern" and value == name):
                found.add(path)
    return found


def _unread(names, producers, kinds=("str",)) -> dict[str, set[str]]:
    """``{name: readers}`` for the names with no reader but the export."""
    out = {}
    for n in names:
        r = _readers(n, producers, kinds)
        if not r - {EXPORT}:
            out[n] = r
    return out


def _check(names, producers, kinds=("str",), allow=()) -> None:
    orphans = {n: r for n, r in _unread(names, producers, kinds).items() if n not in allow}
    assert not orphans, f"published with no reader outside its producer: {sorted(orphans)}"


# ─── The published names ─────────────────────────────────────────────────────

def _swayam_columns() -> list[str]:
    idx = pd.bdate_range("2024-01-01", periods=3)
    view = pd.DataFrame({"Condition": ["Oversold", "Neutral", "Overbought"],
                         "Can_Vote": [True, True, False]}, index=idx)
    # _Native is added by pipeline.run.build_swayam when the breadth is carried
    # onto the model spine.
    return list(aggregate_views({"a": view, "b": view}).columns) + ["_Native"]


def test_every_mula_column_has_a_reader():
    _check(PUBLISHED_COLUMNS, ("engines/mula",), allow=EXPORT_ALLOWLIST)


def test_every_swayam_column_has_a_reader():
    _check(_swayam_columns(), ("engines/swayam", "pipeline/run.py::build_swayam"))


def test_every_call_column_has_a_reader():
    _check(CALL_COLUMNS, ("core/call.py::compose_series", "core/call.py::<module>:CALL_COLUMNS",
                          "core/call.py::<module>:_SIZER_READINGS",
                          "core/call.py::<module>:_CLAIM_GRADE"), allow=EXPORT_ALLOWLIST)


def test_every_edge_has_a_reader(gold_app):
    _check(gold_app.session_state["edges"], ("pipeline/assemble.py::grade_all",))


def test_every_signal_key_has_a_reader(gold_app):
    _check(gold_app.session_state["engine"].get_current_signal(), ("engines/mula/base.py",))


def test_every_bundle_key_has_a_reader():
    _check(app._BUNDLE_KEYS, ("app.py::<module>:_BUNDLE_KEYS", "pipeline/run.py::run_target"))


def test_every_config_field_has_a_reader():
    fields = [f.name for f in dataclasses.fields(InstrumentConfig)]
    _check(fields, ("core/config.py",), kinds=("attr", "str"))


def test_every_knob_has_a_reader():
    plain = [n for n, k in KNOBS.items() if not k.ref]
    by_ref = [n for n, k in KNOBS.items() if k.ref]
    _check(plain, ("core/knobs.py",), allow=STUDY_PENDING)
    # A by-reference knob is read through its core.config constant (or the
    # InstrumentConfig field it defaults).
    _check(by_ref, ("core/knobs.py",), kinds=("str", "name", "attr"), allow=STUDY_PENDING)


def test_the_allowlists_are_current():
    published = set(PUBLISHED_COLUMNS) | set(CALL_COLUMNS) | set(_swayam_columns())
    for name in EXPORT_ALLOWLIST:
        assert name in published, f"{name} is no longer published"
        assert EXPORT in _readers(name, ()), f"{name} is not in the Data record"
    # Each allowlisted name really has no reader but the export (else drop it).
    assert set(_unread(EXPORT_ALLOWLIST, ("engines/mula", "core/call.py::compose_series"))) \
        == set(EXPORT_ALLOWLIST)
    for name in STUDY_PENDING:
        assert name in KNOBS and not _readers(name, ("core/knobs.py",)), name


def test_the_scan_sees_a_reader_and_an_orphan():
    # A real reader is found, an f-string reader is matched, a write is not a read.
    assert "pipeline/assemble.py" in _readers("MRProb", ("engines/mula",))
    assert "core/call.py" in _readers("reversion_factor", ("core/call.py::compose_series",))
    assert _unread(["NO_SUCH_COLUMN_ANYWHERE"], ())
