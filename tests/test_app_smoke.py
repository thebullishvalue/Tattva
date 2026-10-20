"""The app runs, offline, and every page renders — and the call leads.

AppTest drives app.py on the committed fixtures (``offline_fetch`` via the
session-scoped ``gold_app`` run): cold start, select Gold, Run Analysis, then
every st.Page the app defines — read from the app itself, so a page added or
removed later is covered without editing this file. A page that raises is
caught by the app's own guard and drawn as "… could not be drawn", so that
text is a failure here too. Every later package keeps this green.

The bundle carries the closed loop (call_ts, its grades, the evidence and the
verdict at d*); Overview opens with the call, its grade as issued and the
chain, and prints no tier word and no size as a percentage; the rail reads the
same verdict; the Data record exports every call_ts column.
"""
import re
from types import SimpleNamespace

from core.call import CALL_COLUMNS, verdict_from_row
from pipeline.assemble import NO_PRIOR_LABEL

EDGE_KEYS = {"claim", "claim:buy", "claim:sell", "sizer:swayam", "sizer:reversion",
             "call", "call:prospective"}


def _page_text(at) -> str:
    return "\n".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)


def test_the_analysis_runs_and_every_page_renders(gold_app):
    at = gold_app
    assert not at.exception, [e.value for e in at.exception]
    for key in ("engine", "fvo_ts", "edges", "asof", "horizon", "epoch", "fingerprint",
                "call_ts", "grades_ts", "evidence", "verdict", "exec_lag", "priors_meta"):
        assert key in at.session_state, key
    pages = at.session_state["_smoke_pages"]
    assert len(pages) >= 2
    for page in pages:
        at.session_state["_smoke_page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
        assert "could not be drawn" not in _page_text(at), page


def test_the_bundle_carries_the_closed_loop(gold_app):
    ss = gold_app.session_state
    edges = ss["edges"]
    assert EDGE_KEYS <= set(edges) and "mula" not in edges
    assert ss["verdict"] == verdict_from_row(ss["call_ts"], edges, asof=ss["asof"],
                                             horizon=ss["horizon"], exec_lag=ss["exec_lag"],
                                             target="Gold")
    assert ss["call_ts"].attrs["h"] == ss["horizon"] and ss["call_ts"].attrs["fingerprint"] == ss["fingerprint"]
    assert ss["priors_meta"]["label"] == NO_PRIOR_LABEL


def test_overview_leads_with_the_call_its_grade_and_the_chain(gold_app):
    at = gold_app
    at.session_state["_smoke_page"] = "Overview"
    at.run()
    md = [m.value for m in at.markdown]
    first = next(i for i, v in enumerate(md) if 'class="hero-card' in v)
    # Only the page shell (style, tape, command bar, notices) precedes the call.
    for v in md[:first]:
        assert re.search(r"<style|ticker|command-bar|notice-rail", v), v[:120]
    grade = next(i for i, v in enumerate(md) if "Call grade · as issued" in v)
    chain = next(i for i, v in enumerate(md) if "The chain" in v)
    assert first < grade < chain
    verdict = at.session_state["verdict"]
    assert verdict["word"] in md[first] and verdict["size_formula"] in re.sub(r"<[^>]+>", "", md[chain])

    text = _page_text(at)
    assert not re.search(r"\bsize\s*[:=]?\s*[-+]?\d+(?:\.\d+)?\s*%", text)
    for word in ("STRONG", "LEAN", "Conviction", "Engine Agreement"):
        assert word not in text, word
    rail = "\n".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.sidebar.markdown)
    assert "Claim t_post" in rail and verdict["action"] in rail and "weight" not in rail


def test_the_data_record_exports_every_call_column(gold_app, monkeypatch):
    import ui.tabs.tab_data as tab_data

    ss = gold_app.session_state
    state = {k: ss[k] for k in ("swayam_daily", "convergence_df", "hero_series", "call_ts")}
    monkeypatch.setattr(tab_data, "st", SimpleNamespace(session_state=state))
    ts = ss["engine"].ts_data.copy()
    ts["Date"] = ss["fvo_ts"].index
    record = tab_data.build_record(ts)
    for c in CALL_COLUMNS:
        assert tab_data.CALL_PREFIX + c in record.columns, c
    assert record.loc[ss["asof"], tab_data.CALL_PREFIX + "action"] == ss["verdict"]["action"]
