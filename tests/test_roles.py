"""C15: every card is registered to a role, a lean carries its grade, and the
pages read in decision order.

``core.roles.CARDS`` registers every card a page draws — each KPI item, each
section, each Data column — to ``(role key, target)``, and its chip is the
role kind's fixed text. Checked on what the pages actually draw: the offline
fixture app and every synthetic state of tests/test_render.py, every page.

  * every rendered KPI and section carries a registered card whose role and
    target are ROLES keys, and a section's chip is its card's;
  * every Data column is registered, and its header carries the card's tag;
  * no page or table draws an unformatted registry template ('{target}');
  * a KPI whose reading is a lean word sits in an item that carries its own
    grade (IC ± SE (t, n/h));
  * Overview's first blocks are the CALL, GRADE→call and the chain, then
    EVIDENCE→claim and the call's history;
  * the Mūla and Swayam pages print Overview's chain values verbatim;
  * the rail holds the call's readout and nothing else;
  * no estimator runs in ui/.
"""
import html
import os
import re
from types import SimpleNamespace

import pytest

from core.roles import CARDS, CHAIN, CHIPS, ROLES, card_role, chip, role_tag
from test_render import all_renders, page_text, rendered_app, rendered_states, synthetic_bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: A card's opening tag: a KPI item or a section header, and its attributes.
_OPEN = re.compile(r'<div class="(?P<cls>metric-card|section-hdr)\b[^"]*"(?P<attrs>[^>]*)>')
_CARD = re.compile(r'data-card="([^"]*)"')
#: A lean's grade as render_kpi_strip prints it.
EDGE = re.compile(r"IC (?:[+-]\d+\.\d{3} ± \d+\.\d{3} \(t [+-]?\d+\.\d{2}, n/h \d+\)"
                  r"|not yet graded \(n/h \d+\))")
#: Words that state a direction: a KPI reading one of them is a lean.
LEAN_WORDS = re.compile(r"\b(bullish|bearish|cheap|rich|confirms|contradicts)\b", re.I)
RAIL = ["Call", "Size", "Claim t_post", "Call grade", "As of"]


@pytest.fixture(scope="module")
def renders(gold_app):
    return all_renders(gold_app)


def test_the_registry_is_closed_and_its_chips_are_fixed():
    assert set(CHIPS) == {r.kind for r in ROLES.values()}
    for card, (role_key, target) in CARDS.items():
        assert role_key in ROLES and target in ROLES, card
        role, t = card_role(card)
        assert chip(card) == CHIPS[role.kind].format(target=t)
        assert chip(card).startswith(role_tag(card))
        if role.kind == "EVIDENCE":
            assert chip(card) == f"EVIDENCE→{target} · does not move the call"
    with pytest.raises(KeyError, match="unregistered card"):
        chip("overview:nope")


def test_every_rendered_kpi_and_section_names_a_registered_card(renders):
    seen = set()
    for where, r in renders.items():
        for value in r.markdown:
            for m in _OPEN.finditer(value):
                card = _CARD.search(m.group("attrs"))
                assert card, (where, m.group(0))
                card = card.group(1)
                assert card in CARDS, (where, card)
                role_key, target = CARDS[card]
                assert role_key in ROLES and target in ROLES, (where, card)
                seen.add(card)
                if m.group("cls") == "section-hdr":
                    assert f">{html.escape(chip(card))}</span>" in value, (where, card)
        seen.update(c for v in r.markdown for c in _CARD.findall(v))
    # Every registered card of a page that drew is drawn somewhere.
    unused = {c for c in CARDS if not c.startswith(("landing:", "data:"))} - seen
    assert not unused, sorted(unused)


def _th_tags(frame_html: str) -> list[tuple[str, str]]:
    return [(html.unescape(a), html.unescape(b)) for a, b in
            re.findall(r'<th class="[^"]*">([^<]*)<span class="tag">([^<]*)</span></th>', frame_html)]


def test_no_page_or_table_draws_an_unformatted_placeholder(renders):
    """A registry's template ('CLAIM→{target}') is always formatted before it is drawn."""
    placeholder = re.compile(r"\{[A-Za-z_]\w*\}")
    for where, r in renders.items():
        for text in (r.text, r.rail, *map(html.unescape, r.iframes)):
            assert not placeholder.search(text), (where, placeholder.search(text).group(0))
    system = next(r for w, r in renders.items() if w.endswith("/System"))
    roles = next(f for f in system.iframes if "INFRASTRUCTURE→call" in f)
    for chip_text in ("CLAIM→call", "GRADE→claim", "SIZER→call", "EVIDENCE→claim",
                      "PRECONDITION→claim", "INFRASTRUCTURE→trust"):
        assert chip_text in roles, chip_text


def test_every_data_column_names_a_registered_card(renders, gold_app, monkeypatch):
    import ui.tabs.tab_data as tab_data

    for where, r in renders.items():
        if not where.endswith("/Data"):
            continue
        frame = next(f for f in r.iframes if 'class="tag"' in f)
        heads = re.findall(r"<th class=\"[^\"]*\">(.*?)</th>", frame)
        tags = _th_tags(frame)
        assert len(tags) == len(heads) - 1, where            # every column but the date index
        for name, tag in tags:
            card = tab_data.column_card(name)
            assert card in CARDS and CARDS[card][1] in ROLES, (where, name)
            assert tag == role_tag(card), (where, name, tag)
    # The full record (the export), not just the window on screen.
    ss = gold_app.session_state
    monkeypatch.setattr(tab_data, "st", SimpleNamespace(
        session_state={k: ss[k] for k in ("swayam_daily", "call_ts")}))
    ts = ss["engine"].ts_data.copy()
    ts["Date"] = ss["fvo_ts"].index
    for c in tab_data.build_record(ts).columns:
        assert tab_data.column_card(c) in CARDS, c
    with pytest.raises(KeyError):
        tab_data.column_card("Some new column")


def test_every_lean_word_sits_in_an_item_that_carries_an_edge(renders):
    leans = 0
    for where, r in renders.items():
        for value in r.markdown:
            m = _OPEN.search(value)
            if not m or m.group("cls") != "metric-card":
                continue
            reading = page_text([re.search(r"<h2>(.*?)</h2>", value, re.S).group(1)])
            sub = re.search(r'<div class="sub-metric">(.*?)</div>', value, re.S)
            sub = html.unescape(sub.group(1)) if sub else ""
            lean = 'data-lean="1"' in m.group("attrs")
            if LEAN_WORDS.search(reading):
                assert lean, (where, reading)
            if lean:
                leans += 1
                assert EDGE.search(sub), (where, reading, sub)
    assert leans, "no lean was drawn: the check is vacuous"


def _blocks(markdown: list[str]) -> list[str]:
    """Overview's blocks in order, by role: the shell dropped, repeats collapsed."""
    out = []
    for v in markdown:
        if re.search(r"<style|command-bar|notice-rail", v):
            continue
        if "hero-chain" in v:
            label = "chain"
        else:
            card = _CARD.search(v)
            if not card:
                continue                                     # a note, a chart panel's header
            label = role_tag(card.group(1))
        if not out or out[-1] != label:
            out.append(label)
    return out


def test_overview_opens_with_the_call_its_grade_and_the_chain(renders):
    want = ["CALL", "GRADE→call", "chain", "EVIDENCE→claim", "CALL", "GRADE→call"]
    for where, r in renders.items():
        if where.endswith("/Overview"):
            assert _blocks(r.markdown)[:6] == want, (where, _blocks(r.markdown))
            first = next(v for v in r.markdown if not re.search(r"<style|command-bar|notice-rail", v))
            assert 'data-card="overview:call"' in first and "hero-card" in first, where


def _verdicts(gold_app):
    yield "fixture", gold_app.session_state["verdict"], rendered_app(gold_app)
    for (state, page), _r in rendered_states().items():
        if page == "Overview":
            yield state, synthetic_bundle(state).verdict, {
                p: rendered_states()[(state, p)] for p in ("Mūla", "Swayam")}


def test_the_mula_and_swayam_pages_print_overviews_chain_values(gold_app):
    for where, verdict, pages in _verdicts(gold_app):
        chain = {r["key"]: r for r in verdict["chain"]}
        assert list(chain) == list(CHAIN), where
        for page, keys in (("Mūla", ("claim", "precondition", "trust", "sizer:reversion")),
                           ("Swayam", ("sizer:swayam",))):
            text = pages[page].text
            for key in keys:
                for field in ("title", "reading", "effect", "grade", "detail"):
                    assert chain[key][field] in text, (where, page, key, field, chain[key][field])


def test_the_rail_holds_the_calls_readout_only(renders):
    for where, r in renders.items():
        rows = [html.unescape(k) for v in r.sidebar for k in
                re.findall(r'<span class="k">([^<]*)</span>', v) if "rail-readout" in v]
        assert rows == RAIL, (where, rows)
        assert "weight" not in r.rail.lower(), where


def test_no_estimator_runs_in_ui():
    hits = []
    for base, _dirs, names in os.walk(os.path.join(ROOT, "ui")):
        if "__pycache__" in base:
            continue
        for n in names:
            if n.endswith(".py"):
                path = os.path.join(base, n)
                for i, line in enumerate(open(path, encoding="utf-8"), 1):
                    if re.search(r"series_edge|expanding_edge|spearmanr|rankdata", line):
                        hits.append(f"{os.path.relpath(path, ROOT)}:{i}")
    assert hits == [], hits
