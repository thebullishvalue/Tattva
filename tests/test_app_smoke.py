"""The app runs, offline, and every page renders — and the call leads.

AppTest drives app.py on the committed fixtures (``offline_fetch`` via the
session-scoped ``gold_app`` run): cold start, select Gold, Run Analysis, then
every st.Page the app defines — read from the app itself, so a page added or
removed later is covered without editing this file. A page that raises is
caught by the app's own guard and drawn as "… could not be drawn", so that
text is a failure here too. Every later package keeps this green.

The pages are the role groups — Overview, Mūla, Swayam, Grades, System and
Data (Convergence and Precedent are gone). The bundle carries the closed loop
(call_ts, its grades, the evidence and the verdict at d*) and none of the
parallel opinions' keys, and its priors label is the committed study's when
one is loaded, else 'no pooled prior: per-instrument grade only'; Overview
opens with the call, its grade as issued and the chain, and prints no tier
word and no size as a percentage; the rail reads the same verdict; the Data
record exports every call_ts column. The Mūla page opens with Overview's own
chain rows for the claim, its preconditions, its trust and the reversion
sizer, labels the ECM's move as the gap-implied move (drift excluded), carries
none of the retired copy, and the one gap-stationarity word appears once
across Mūla and System.
"""
import html
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.call import CALL_COLUMNS, verdict_from_row
from core.knobs import knob
from core.semantics import GAP_STATES
from core.trust import load_priors
from pipeline.assemble import NO_PRIOR_LABEL, describe_priors

ROOT = Path(__file__).resolve().parents[1]
EDGE_KEYS = {"claim", "claim:buy", "claim:sell", "sizer:swayam", "sizer:reversion",
             "call", "call:prospective", "evidence:swayam_lean", "evidence:ecm_drift"}
#: Copy the Mūla page must not carry (the retired cards and their advice).
MULA_BANNED = ("price reverts", "will close", "treat", "with caution", "window agreement",
               "r² vs trailing mean")
PAGES = ["Overview", "Mūla", "Swayam", "Grades", "System", "Data"]
#: Keys of the parallel opinions and learners WP4 deleted: none may come back.
GONE_KEYS = ("convergence_df", "divergence_events", "hero_series", "intelligence_active_weights",
             "intelligence_active_profile", "swayam_view_weights", "_prec_key",
             "_precedent_analogs_cache", "_analog_skill_cache")


def _page_text(at) -> str:
    return "\n".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)


def test_the_analysis_runs_and_every_page_renders(gold_app):
    at = gold_app
    assert not at.exception, [e.value for e in at.exception]
    for key in ("engine", "fvo_ts", "edges", "asof", "horizon", "epoch", "fingerprint",
                "call_ts", "grades_ts", "evidence", "verdict", "exec_lag", "priors_meta"):
        assert key in at.session_state, key
    pages = at.session_state["_smoke_pages"]
    assert pages == PAGES, pages
    for page in pages:
        at.session_state["_smoke_page"] = page
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.error, (page, [e.value for e in at.error])
        assert "could not be drawn" not in _page_text(at), page


def test_the_bundle_carries_the_closed_loop(gold_app):
    ss = gold_app.session_state
    edges = ss["edges"]
    assert EDGE_KEYS <= set(edges) and not {"mula", "swayam", "composite"} & set(edges)
    assert not [k for k in GONE_KEYS if k in ss], [k for k in GONE_KEYS if k in ss]
    assert set(edges["sizer:swayam"]["buckets"]) == {"contradicts", "split", "confirms"}
    assert ss["verdict"] == verdict_from_row(ss["call_ts"], edges, asof=ss["asof"],
                                             horizon=ss["horizon"], exec_lag=ss["exec_lag"],
                                             target="Gold")
    assert ss["call_ts"].attrs["h"] == ss["horizon"] and ss["call_ts"].attrs["fingerprint"] == ss["fingerprint"]
    # The run's priors are the committed study's file for its h and L, or none.
    table = load_priors(ss["horizon"], knob("EXECUTION_LAG"))
    meta = ss["priors_meta"]
    assert meta["label"] == describe_priors(table)["label"]
    if table is None:
        assert meta["label"] == NO_PRIOR_LABEL and meta["status"] == "none"
    else:
        assert meta["status"] == "loaded" and meta["label"] != NO_PRIOR_LABEL
        assert ss["call_ts"].attrs["priors_study"] == table.meta["study"]


def test_overview_leads_with_the_call_its_grade_and_the_chain(gold_app):
    at = gold_app
    at.session_state["_smoke_page"] = "Overview"
    at.run()
    md = [m.value for m in at.markdown]
    first = next(i for i, v in enumerate(md) if 'class="hero-card' in v)
    # Only the page shell (style, command bar, notices) precedes the call.
    for v in md[:first]:
        assert re.search(r"<style|command-bar|notice-rail", v), v[:120]
    grade = next(i for i, v in enumerate(md) if "Call grade · as issued" in v)
    chain = next(i for i, v in enumerate(md) if 'class="hero-card hero-chain' in v)
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
    state = {k: ss[k] for k in ("swayam_daily", "call_ts")}
    monkeypatch.setattr(tab_data, "st", SimpleNamespace(session_state=state))
    ts = ss["engine"].ts_data.copy()
    ts["Date"] = ss["fvo_ts"].index
    record = tab_data.build_record(ts)
    for c in CALL_COLUMNS:
        assert tab_data.CALL_PREFIX + c in record.columns, c
    assert record.loc[ss["asof"], tab_data.CALL_PREFIX + "action"] == ss["verdict"]["action"]
    # Swayam's net breadth is its published Lean in points, not a re-derivation.
    lean = tab_data._by_date(ss["swayam_daily"])["Lean"].reindex(record.index) * 100.0
    assert np.array_equal(record["Swayam net breadth (pts)"].to_numpy(dtype=float),
                          lean.to_numpy(dtype=float), equal_nan=True)


def test_the_mula_page_reads_the_verdict_and_carries_no_retired_copy(gold_app):
    at = gold_app
    text = {}
    for page in ("Mūla", "System"):
        at.session_state["_smoke_page"] = page
        at.run()
        assert not at.exception, page
        text[page] = html.unescape("\n".join(
            re.sub(r"<[^>]+>", " ", re.sub(r"<style.*?</style>", "", m.value, flags=re.S))
            for m in at.markdown))
    mula = text["Mūla"]
    # Overview's formatted claim, trust and reversion values, verbatim.
    chain = {r["key"]: r for r in at.session_state["verdict"]["chain"]}
    for key in ("claim", "precondition", "trust", "sizer:reversion"):
        for field in ("reading", "effect", "grade", "detail"):
            assert chain[key][field] in mula, (key, field, chain[key][field])
    assert "gap-implied 10D move (excludes drift)" in mula
    low = mula.lower()
    for phrase in MULA_BANNED:
        assert phrase not in low, phrase
    # One definition of the gap's stationarity, shown once.
    both = text["Mūla"] + text["System"]
    assert sum(both.count(label) for label in GAP_STATES) == 1
    assert at.session_state["evidence"]["gap_stationarity"]["state"] in mula


def test_new_data_never_restores_a_stale_cached_run(offline_fetch):
    """The run key hashes every prepared row: a value revised after the session a
    pooled-prior study hashed the history to changes the key, although the
    (study-comparable) fingerprint does not."""
    import app
    from core.trust import PriorTable
    from pipeline.run import fetch_panel, prepare, target_bars

    panel = fetch_panel("Gold")
    bars, _ = target_bars("Gold")
    own = prepare(panel, "Gold", bars=bars, priors=None)
    studied = own.data.index[-60]
    table = PriorTable({}, {"study": "studies/pooled_priors.py", "run_date": "2026-01-01",
                            "fingerprint_upto": {"Gold": studied.strftime("%Y-%m-%d")}})
    then = prepare(panel, "Gold", bars=bars, priors=table)
    assert app._run_key(10, "Gold", then) == app._run_key(10, "Gold", prepare(panel, "Gold", bars=bars,
                                                                             priors=table))
    revised = panel.copy()
    row = revised.index[pd.to_datetime(revised["DATE"]).dt.normalize() == own.data.index[-10]][0]
    revised.loc[row, "Gold"] = revised.loc[row, "Gold"] * 1.01
    later = prepare(revised, "Gold", bars=bars, priors=table)
    assert later.fingerprint == then.fingerprint
    assert app._run_key(10, "Gold", later) != app._run_key(10, "Gold", then)


# ─── A run kept open across sessions; a typed stock's price ──────────────────

def _until_run(at):
    for _ in range(4):
        if "verdict" in at.session_state and "engine" in at.session_state:
            break
        at.run()
    return at


def test_a_run_left_open_across_sessions_is_read_against_the_current_one(monkeypatch):
    """M5: the cached run is keyed on its data, so a tab left open for days restores it —
    and it must be read against the session that has closed NOW: behind, P2, the
    elapsed horizon and the chip are one number, re-derived at render with nothing
    refitted, and equal to what a fresh assemble on the same data and day says."""
    from conftest import install_offline_fetch, run_app_offline
    from core.config import get_instrument_config
    from core.instruments import instrument
    from pipeline.assemble import assemble

    install_offline_fetch(monkeypatch)
    day = {"d": pd.Timestamp("2026-10-19")}
    monkeypatch.setattr("data.calendars.session_closed_through", lambda exchange, now=None: day["d"])
    at = _until_run(run_app_offline("USD/INR", "Currency (FX)"))
    ss = at.session_state
    d_star, key = ss["asof"], ss["engine_cache"]
    v = ss["verdict"]
    assert (v["behind"], v["elapsed"]) == (1, 0) and v["action"] != "NO CALL"
    assert "1 session closed since d*, 0 of 10 of the horizon elapsed" in \
        {r["key"]: r for r in v["chain"]}["precondition"]["detail"]

    day["d"] = pd.Timestamp("2026-11-04")                # 13 sessions later, same tab
    at.session_state["_smoke_page"] = "Overview"
    at.run()
    ss = at.session_state
    assert ss["engine_cache"] == key                      # nothing refitted
    v = ss["verdict"]
    assert (v["behind"], v["elapsed"], v["action"]) == (13, 10, "NO CALL")
    assert "the 10-session horizon from the next close has run" in v["headline"]
    pre = {r["key"]: r for r in v["chain"]}["precondition"]
    assert "13 sessions closed since d*, 10 of 10 of the horizon elapsed" in pre["detail"]
    text = _page_text(at)
    assert "STALE" in text and "13 trading days behind" in text and "1 of 10" not in text
    fresh = assemble(ss["fvo_ts"], ss["swayam_daily"], instrument=instrument("USD/INR"),
                     cfg=get_instrument_config("USD/INR"), asof=d_star, today=day["d"],
                     priors=load_priors(10, knob("EXECUTION_LAG")))
    for c in CALL_COLUMNS:
        a, b = ss["call_ts"].at[d_star, c], fresh.call_ts.at[d_star, c]
        assert a == b or (pd.isna(a) and pd.isna(b)), c
    for k in ("action", "headline", "limit", "chain", "behind", "elapsed", "call_grade"):
        assert v[k] == fresh.verdict[k], k


def test_a_typed_stocks_price_is_fetched_once_and_refresh_runs_once(monkeypatch, request):
    """M6: a typed stock's price is fetched once with the panel and kept, so a page click
    after the fetch cache has expired can neither move d* nor recompute; Refresh fetches
    it forced and the pipeline runs once on the new print."""
    import core.config as config
    import core.instruments as instruments
    import pipeline.run as pipeline_run
    from conftest import _APP_WRAPPER, install_offline_fetch
    from streamlit.testing.v1 import AppTest
    from studies.make_fixtures import load_panel

    # The app registers the typed symbol in the shared registries (other modules hold
    # them by reference): what this test adds is removed when it ends.
    regs = (config.ALL_TARGETS, config.STOCK_TARGET_MARKETS, config.TARGET_EXCLUDED_PREDICTORS,
            config.INSTRUMENT_CONFIGS, instruments.INSTRUMENTS)
    before = [set(d) for d in regs]
    request.addfinalizer(lambda: [d.pop(k, None) for d, b in zip(regs, before) for k in set(d) - b])
    install_offline_fetch(monkeypatch)
    panel = load_panel()
    dates = pd.DatetimeIndex(pd.to_datetime(panel["DATE"])).normalize()
    rng = np.random.default_rng(7)
    base = pd.Series(panel["Nifty 50"].to_numpy(float), index=dates)
    close = 2500 * np.exp(np.cumsum(np.log(base).diff().fillna(0) + rng.normal(0, 0.01, len(base))))
    v1 = close[close.index < dates[-1]]
    v2 = close.copy()
    v2.iloc[-1] = v1.iloc[-1] * 1.04
    state = {"live": v1, "calls": [], "fits": 0}

    def series(ticker, start, end, force=False):
        state["calls"].append("force" if force else "cached")
        return state["live"].copy()

    real_fit = pipeline_run.fit_mula

    def fit(*a, **k):
        state["fits"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr("data.fetcher.fetch_stock_target_series", series)
    monkeypatch.setattr("data.universe.resolve_stock_symbol",
                        lambda raw, market: ("RELIANCE.NS", "NSE"))
    monkeypatch.setattr("pipeline.run.fit_mula", fit)
    at = AppTest.from_string(_APP_WRAPPER.replace("APP_PATH", repr(str(ROOT / "app.py"))),
                             default_timeout=1200)
    at.run()
    at.selectbox(key="target_category").set_value("India Stocks").run()
    at.text_input(key="stock_symbol_india").input("RELIANCE").run()
    [b for b in at.button if b.label == "Run Analysis"][0].click().run()
    _until_run(at)
    ss = at.session_state
    d1, key1 = ss["asof"], ss["engine_cache"]
    assert d1 == v1.index[-1] and state["fits"] == 1 and state["calls"] == ["cached"]

    # The source posts the next session and the one-hour cache entry expires: a plain
    # page click reads the price kept with the run, not a new one.
    state["live"] = v2
    at.session_state["_smoke_page"] = "Grades"
    at.run()
    ss = at.session_state
    assert (ss["asof"], ss["engine_cache"], state["fits"], state["calls"]) == (d1, key1, 1, ["cached"])

    # Refresh: the price is fetched forced, with the panel, and the pipeline runs once.
    [b for b in at.button if b.label == "Refresh data"][0].click().run()
    _until_run(at)
    at.session_state["_smoke_page"] = "Overview"
    at.run()
    ss = at.session_state
    assert state["calls"] == ["cached", "force"] and state["fits"] == 2
    assert ss["asof"] == v2.index[-1] and ss["engine_cache"] != key1
    assert ss["fvo_ts"]["Price"].loc[ss["asof"]] == v2.iloc[-1]
