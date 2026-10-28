"""The docs say what the shipped system is — and nothing the audit refuted.

  * README.md, ARCHITECTURE.md and CHANGELOG.md carry none of the refuted
    2.8.0 claims ('one graded call' — unless the call's grade is in fact
    'graded' — 'six independent', 'independent gates', 'the edge grades the
    call', 'price reverts', 'read conviction against', and no table with a
    "Mūla's call" IC column). The CHANGELOG may quote them only inside a note
    marked 'corrected in 2.9.0'. README.md, ARCHITECTURE.md and the 2.9.0
    entry also keep the design's copy rules (tests/test_copy.py's list).
  * The README's edge table is byte-identical to
    ``studies.render_docs_table.render_edge_table`` on the committed study
    file, and the 2.9.0 entry carries the same block while that study is the
    one it names.
  * ARCHITECTURE.md lists every ROLES key and every KNOBS name of kind
    structural or measured (the study-generated ``PRIOR:<object>:<class>``
    family by its pattern), with each knob's kind and value as registered, and
    its contracts table names, clause by clause, the test files
    tests/test_charter.py indexes.
  * The README's pages are the app's st.Page titles, in order, and the docs
    name the version core.config ships.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from core.config import VERSION
from core.knobs import KNOBS, STUDY_GENERATED
from core.roles import ROLES
from studies.render_docs_table import BEGIN, END, committed_path, readme_block, render_edge_table
from test_charter import CLAUSES
from test_copy import BANNED as COPY_BANNED

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "ARCHITECTURE.md", "CHANGELOG.md")

#: The refuted 2.8.0 claims (case-insensitive). 'one graded call' is allowed only
#: while the call's own grade reads 'graded'.
REFUTED = ("six independent", "independent gates", "the edge grades the call", "price reverts",
           "read conviction against")
GRADED_ONLY = "one graded call"
#: A CHANGELOG note that quotes a refuted claim to correct it carries this marker.
CORRECTED = "corrected in 2.9.0"


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _blocks(text: str) -> list[str]:
    """Markdown blocks: a heading, a list item with its continuation lines, a table or
    a paragraph; a blank line ends one."""
    blocks: list[list[str]] = []
    for line in text.splitlines():
        starts = re.match(r"\s*(#|[-*] |\d+\. )", line)
        if not line.strip():
            blocks.append([])
        elif starts or not blocks or not blocks[-1]:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return ["\n".join(b) for b in blocks if b]


def _outside_corrections(name: str) -> str:
    """The doc's text, without the CHANGELOG's marked corrections."""
    blocks = _blocks(_read(name))
    if name == "CHANGELOG.md":
        blocks = [b for b in blocks if CORRECTED not in b.lower()]
    return "\n\n".join(blocks)


def _entry(version: str) -> str:
    """One CHANGELOG entry, from its heading to the next."""
    text = _read("CHANGELOG.md")
    start = text.index(f"## [{version}]")
    nxt = text.find("\n## [", start + 1)
    return text[start:nxt if nxt != -1 else None]


def _table_headers(text: str) -> list[list[str]]:
    lines = text.splitlines()
    return [[c.strip() for c in line.strip().strip("|").split("|")]
            for line, below in zip(lines, lines[1:])
            if line.lstrip().startswith("|") and re.match(r"\s*\|\s*:?-{3,}", below)]


def _phrase_hits(text: str, phrases) -> list[str]:
    low, hits = text.lower(), []
    for phrase in phrases:
        pat = re.escape(phrase.lower())
        if re.fullmatch(r"\w+", phrase):
            pat = rf"(?<![\w-]){pat}(?![\w-])"
        hits += [f"{phrase}: …{low[max(0, m.start() - 50):m.end() + 50]}…"
                 for m in re.finditer(pat, low)]
    return hits


def _knob_value(v) -> str:
    """A knob's value as the ARCHITECTURE table prints it."""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, tuple):
        return ", ".join(str(x) for x in v)
    return str(v)


# ─── No refuted claim ────────────────────────────────────────────────────────

def test_the_docs_carry_no_refuted_claim(gold_app):
    graded = bool((gold_app.session_state["edges"].get("call") or {}).get("graded"))
    hits = []
    for name in DOCS:
        text = _outside_corrections(name)
        phrases = REFUTED if graded else (*REFUTED, GRADED_ONLY)
        hits += [f"{name}: {h}" for h in _phrase_hits(text, phrases)]
        for header in _table_headers(text):
            if any(c.lower() == "mūla's call" for c in header):
                hits.append(f"{name}: a 'Mūla's call' IC column in {header}")
    assert hits == [], hits


def test_the_new_docs_keep_the_copy_rules():
    hits = []
    for name, text in (("README.md", _read("README.md")),
                       ("ARCHITECTURE.md", _read("ARCHITECTURE.md")),
                       ("CHANGELOG.md [2.9.0]", "\n\n".join(
                           b for b in _blocks(_entry("2.9.0")) if CORRECTED not in b.lower()))):
        hits += [f"{name}: {h}" for h in _phrase_hits(text, COPY_BANNED)]
    assert hits == [], hits


def test_the_corrected_notes_are_where_the_refuted_claims_are():
    """Every refuted 2.8.0 claim the CHANGELOG still quotes sits in a marked note."""
    marked = [b for b in _blocks(_read("CHANGELOG.md")) if CORRECTED in b.lower()]
    for phrase in ("one graded call", "six independent gates", "the edge grades the call",
                   "reversion means price reverts", "where the edge is not"):
        assert any(phrase in b.lower() for b in marked), phrase


# ─── Generated numbers ───────────────────────────────────────────────────────

def test_the_readme_edge_table_is_the_committed_studys():
    path = committed_path()
    if not path.is_file():
        pytest.skip("no committed pooled-prior study: the README's table has nothing to render")
    readme = _read("README.md")
    assert readme.count(BEGIN) == 1 and readme.count(END) == 1
    assert readme_block(readme) == render_edge_table(str(path))


def test_the_changelog_carries_the_study_it_names():
    path = committed_path()
    if not path.is_file():
        pytest.skip("no committed pooled-prior study")
    import json
    run_date = json.loads(path.read_text(encoding="utf-8"))["run_date"]
    entry = _entry(VERSION)
    if f"run {run_date}" not in entry:
        pytest.skip(f"the {VERSION} entry reports another study than the one committed")
    assert render_edge_table(str(path)) in entry


# ─── ARCHITECTURE.md is the shipped system ───────────────────────────────────

def test_architecture_lists_every_role_and_every_structural_or_measured_knob():
    text = _read("ARCHITECTURE.md")
    missing = [k for k in ROLES if f"`{k}`" not in text]
    for name, k in KNOBS.items():
        if k.kind not in ("structural", "measured"):
            continue
        if name in STUDY_GENERATED:                  # PRIOR:<object>:<class>, from the study
            obj, cls = name.removeprefix("PRIOR:").rsplit(":", 1)
            if "PRIOR:<object>:<class>" not in text or f"`{obj}`" not in text or f"`{cls}`" not in text:
                missing.append(name)
        elif f"`{name}`" not in text:
            missing.append(name)
    assert missing == [], missing


def test_architectures_knob_table_is_the_registry():
    rows = re.findall(r"^\| `([A-Z][A-Z0-9_]+)` \| ([a-z-]+) \| `([^`]*)` \|", _read("ARCHITECTURE.md"), re.M)
    table = {name: (kind, value) for name, kind, value in rows}
    expected = {n: (k.kind, _knob_value(k.value)) for n, k in KNOBS.items() if n not in STUDY_GENERATED}
    assert table == expected


def test_architectures_contracts_table_is_the_charter_index():
    lines = _read("ARCHITECTURE.md").splitlines()
    for clause, (_contract, specs) in CLAUSES.items():
        row = [line for line in lines if line.startswith(f"| {clause} |")]
        assert len(row) == 1, f"no single {clause} row in ARCHITECTURE.md's contracts table"
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", row[0])[1:-1]]
        listed = {f.strip() for f in cells[-1].split(",")}
        assert listed == {f"tests/{s.split('::')[0]}" for s in specs}, (clause, listed)


# ─── The pages and the version ───────────────────────────────────────────────

def test_the_readme_names_the_apps_pages():
    import app

    readme = _read("README.md")
    section = readme[readme.index("## The pages"):]
    section = section[:section.index("\n## ", 1)]
    names = re.findall(r"^- \*\*([^*]+)\*\*", section, re.M)
    assert names == [p.title for p in app.PAGES]


def test_the_docs_name_the_shipped_version():
    assert VERSION == "2.9.0"
    assert f"v{VERSION}" in _read("README.md").split("\n\n")[1]
    assert re.search(r"^## \[([^\]]+)\]", _read("CHANGELOG.md"), re.M).group(1) == VERSION
    assert f"Tattva {VERSION}" in _read("ARCHITECTURE.md")
