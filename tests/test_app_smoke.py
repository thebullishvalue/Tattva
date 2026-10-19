"""The app runs, offline, and every page renders.

AppTest drives app.py on the committed fixtures (``offline_fetch`` via the
session-scoped ``gold_app`` run): cold start, select Gold, Run Analysis, then
every st.Page the app defines — read from the app itself, so a page added or
removed later is covered without editing this file. A page that raises is
caught by the app's own guard and drawn as "… could not be drawn", so that
text is a failure here too. Every later package keeps this green.
"""
import re


def _page_text(at) -> str:
    return "\n".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)


def test_the_analysis_runs_and_every_page_renders(gold_app):
    at = gold_app
    assert not at.exception, [e.value for e in at.exception]
    for key in ("engine", "fvo_ts", "edges", "asof", "horizon", "epoch", "fingerprint"):
        assert key in at.session_state, key
    pages = at.session_state["_smoke_pages"]
    assert len(pages) >= 2
    for page in pages:
        at.session_state["_smoke_page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
        assert "could not be drawn" not in _page_text(at), page
