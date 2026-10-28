"""The copy says only what is measured, and the call's words stay on the call path.

  * Banned phrases — copy that asserts what the design grades instead
    (reversion, independence, the retired gate count and labels) — are absent
    from every file under ui/ and from app.py, comments included, and from
    every rendered page, tables and rail included: copy that reaches a page
    from outside ui/ (a grade's status, a knob's reason, a role's consumer)
    is held to the same list.
  * The reserved words (call, BUY, SELL, STRONG, conviction, size) appear in
    the display strings of ui/ and app.py only inside the call-path code: the
    hero and its GRADE→call strip, the chain rows, the call's history and
    its size terciles, the command bar and the rail's call readout, and the
    record's call_ts columns. Everywhere else a reserved word reaches a page
    only from the registries that define it — core.call (the verdict's
    words) and core.roles (role titles and chips). Docstrings, identifier-
    like keys (``"call"``, ``"call_ts"``, ``"size"``) and the terminal log
    (``console.*``) are not page copy.
  * No '%' follows a size, anywhere a page draws.
  * No tier word (STRONG, LEAN) is drawn while KNOBS['TIER_RULE'] is 'none'.
  * Breadth is printed only in points.

The rendered checks read the offline fixture app and every synthetic state of
tests/test_render.py, every page.
"""
import ast
import os
import re

import pytest

from core.call import format_size
from core.knobs import knob
from test_render import all_renders, page_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Copy the design retired: case-insensitive, whole words where a word.
BANNED = ("price reverts", "mispriced → will close", "will close the gap", "independent",
          "six", "own internals", "fifteen angles", "walk-forward edge", "walk-forward ic",
          "unified convergence", "exact decomposition")

#: The call-path code, by file: the functions (and the one module-level name)
#: whose display strings may word the call.
CALL_PATH: dict[str, set[str]] = {
    "app.py": {"_top_bar", "_command_meta", "_render_call_readout", "_call_readout_rows"},
    "ui/components.py": {"render_hero_card", "_render_call_grade", "render_gate_rows",
                         "render_top_bar", "render_call_history", "render_size_terciles"},
    "ui/tabs/tab_data.py": {"<module>:CALL_PREFIX"},
}
RESERVED_CI = re.compile(r"(?<![\w-])(calls?|conviction|sizes?)(?![\w-])", re.I)
RESERVED_CS = re.compile(r"(?<![\w-])(BUY|SELL|STRONG)(?![\w-])")
#: A string that is a key or an identifier, not copy.
KEY_LIKE = re.compile(r"[a-z0-9_:.\-]+")


def _banned_hits(text: str) -> list[str]:
    """The banned phrases in ``text``, case-insensitive, whole words where a word."""
    low, hits = text.lower(), []
    for phrase in BANNED:
        pat = re.escape(phrase)
        if re.fullmatch(r"\w+", phrase):
            pat = rf"(?<![\w-]){pat}(?![\w-])"
        hits += [f"{phrase}: …{low[max(0, m.start() - 40):m.end() + 40]}…"
                 for m in re.finditer(pat, low)]
    return hits


def _sources(exts=(".py",)) -> list[str]:
    out = ["app.py"]
    for base, _dirs, names in os.walk(os.path.join(ROOT, "ui")):
        if "__pycache__" not in base:
            out += [os.path.relpath(os.path.join(base, n), ROOT) for n in names if n.endswith(exts)]
    return sorted(out)


def test_the_banned_phrases_are_absent_from_ui_and_app():
    hits = []
    for rel in _sources((".py", ".css")):
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read().lower()
        for phrase in BANNED:
            pat = re.escape(phrase)
            if re.fullmatch(r"\w+", phrase):
                pat = rf"(?<![\w-]){pat}(?![\w-])"
            for m in re.finditer(pat, text):
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {phrase}")
    assert hits == [], hits


def test_no_banned_phrase_reaches_a_rendered_page(renders):
    hits = []
    for where, r in renders.items():
        frames = [page_text([re.sub(r"<script.*?</script>", "", f, flags=re.S)]) for f in r.iframes]
        for part, text in (("page", r.text), ("rail", r.rail), *(("table", f) for f in frames)):
            hits += [f"{where} [{part}] {h}" for h in _banned_hits(text)]
    assert hits == [], hits[:10]


class _Copy(ast.NodeVisitor):
    """Every display string of a module, with the scope it is in."""

    def __init__(self):
        self.scope: list[str] = []
        self.strings: list[tuple[str, int, str]] = []
        self.functions: set[str] = set()
        self._skip: set[int] = set()

    def _docstring(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            self._skip.add(id(body[0].value))

    def visit_Module(self, node):
        self._docstring(node)
        self.generic_visit(node)

    def _named(self, node, name):
        self._docstring(node)
        self.functions.add(name)
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self._named(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._named(node, node.name)

    def visit_Assign(self, node):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not self.scope and names:
            self.functions.add(f"<module>:{names[0]}")
            self.scope.append(f"<module>:{names[0]}")
            self.generic_visit(node)
            self.scope.pop()
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "console":
            return                                        # the terminal log, not a page
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self._skip \
                and not KEY_LIKE.fullmatch(node.value):
            self.strings.append((".".join(self.scope), node.lineno, node.value))


def test_the_reserved_words_appear_only_in_the_call_path_code():
    hits = []
    for rel in _sources():
        v = _Copy()
        v.visit(ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read()))
        allowed = CALL_PATH.get(rel, set())
        assert allowed <= v.functions, (rel, sorted(allowed - v.functions))   # no stale entry
        for scope, line, s in v.strings:
            if set(scope.split(".")) & allowed:
                continue
            for m in (*RESERVED_CI.finditer(s), *RESERVED_CS.finditer(s)):
                hits.append(f"{rel}:{line} [{scope or '<module>'}] {m.group(0)!r} in {s[:60]!r}")
    assert hits == [], hits


@pytest.fixture(scope="module")
def renders(gold_app):
    return all_renders(gold_app)


def test_no_percent_follows_a_size(renders):
    assert "%" not in format_size(0.31) and format_size(None) == "—"
    pat = re.compile(r"\bsizes?\b\s*[:=·]?\s*[-+]?\d+(?:\.\d+)?\s*%", re.I)
    for where, r in renders.items():
        for text in (r.text, r.rail):
            assert not pat.search(text), (where, pat.search(text).group(0))


def test_no_tier_word_while_the_tier_rule_is_none(renders):
    if knob("TIER_RULE") != "none":
        pytest.skip(f"TIER_RULE is {knob('TIER_RULE')!r}: Decision C has released tier words")
    for where, r in renders.items():
        for text in (r.text, r.rail):
            assert not re.search(r"\b(STRONG|LEAN)\b", text), (where, re.search(r"\b(STRONG|LEAN)\b", text))


def test_breadth_is_printed_only_in_points(renders):
    pat = re.compile(r"breadth\W{0,3}([-+]?\d+(?:\.\d+)?)\s*(\S+)", re.I)
    seen = 0
    for where, r in renders.items():
        for text in (r.text, r.rail):
            for m in pat.finditer(text):
                seen += 1
                assert m.group(2).startswith("pts"), (where, m.group(0))
    assert seen, "no breadth was printed: the check is vacuous"
