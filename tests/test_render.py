"""Every page renders in every state the verdict can be in — and the grammar holds.

Two sources of pages:

  * the app itself, under AppTest on the offline fixture (``gold_app``, no
    pooled priors) and once more with a pooled-prior study file loaded
    (``tests/fixtures/priors_offline_h10_L1.json``, the study's own output on
    the fixture: ``python -m studies.pooled_priors --offline-fixture
    --allow-dirty --out tests/fixtures/priors_offline_h10_L1.json``), its
    knobs registered as the app registers a committed file;
  * a synthetic bundle builder (:func:`synthetic_bundle`): synthetic Mūla and
    Swayam frames run through the real closed loop (``pipeline.assemble``),
    one per state — HOLD (with no prior, and shrunk toward a class-LOO and a
    global-LOO pool), BUY (a strong class prior opens trust and the Swayam
    sizer), NO CLAIM, NO CALL on a non-tradeable target, a withheld newest
    session, a provisional d*, and Jeera, which has no OHLCV and so no Swayam
    reading — each drawn through ``app.render_page`` for every page in
    ``app.PAGES``, with the command bar and the rail.

A page must draw without an exception, a Streamlit error or the app's own
"could not be drawn" guard. A KPI or section with no registered card raises.
:func:`rendered_states` is shared with tests/test_roles.py and
tests/test_copy.py, which read the same renders.
"""
from __future__ import annotations

import html
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.knobs as knobs
import core.trust as trust
from core.call import NOT_TRADEABLE
from core.config import get_instrument_config
from core.instruments import ASSET_CLASS_NAMES, asset_class_name, instrument
from core.knobs import KNOBS, knob, priors_file_name
from core.trust import PriorTable
from pipeline.assemble import NO_PRIOR_LABEL, assemble, describe_priors

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PRIORS = ROOT / "tests" / "fixtures" / "priors_offline_h10_L1.json"
STUDY = "studies/pooled_priors.py"

#: Each synthetic state: (target, how d*'s FVO is set, its pooled priors — none, or a
#: kind of :func:`_priors` — and the action it must read).
STATES: dict[str, tuple[str, str, str | None, str]] = {
    "hold": ("Gold", "rich", None, "HOLD"),
    "hold_class": ("Gold", "rich", "class-LOO", "HOLD"),
    "hold_global": ("Gold", "rich", "global-LOO", "HOLD"),
    "buy": ("Gold", "cheap", "strong", "BUY"),
    "no_claim": ("Gold", "fair", None, "NO CLAIM"),
    "no_call": ("Nifty 50 - PE", "rich", None, "NO CALL"),
    "withheld": ("Gold", "withheld", None, "HOLD"),
    "provisional": ("Gold", "provisional", None, "HOLD"),
    "jeera": ("Jeera", "rich", None, "HOLD"),
}

N_SESSIONS, BURN_IN = 1100, 250


# ─── The synthetic bundle ────────────────────────────────────────────────────

class _Engine:
    """The slice of MulaEngine the pages read: its frame, signal, stats and drivers."""

    def __init__(self, fvo_ts: pd.DataFrame, signal: dict):
        self.ts_data = fvo_ts.reset_index(drop=True)
        self.asof = signal["asof"]
        self.panel_absent: list[str] = []
        self.latest_feature_impacts = {"Equity (US)": 41.0, "Rates": 33.0, "FX": 26.0}
        self._signal, self._hist = signal, pd.DataFrame(
            [[41.0, 33.0, 26.0]] * 3, columns=list(self.latest_feature_impacts),
            index=pd.DatetimeIndex(fvo_ts.index[-63::21], name="Date"))

    def get_current_signal(self) -> dict:
        return self._signal

    def get_model_stats(self) -> dict:
        return {"r2_oos": 0.91, "n_features": 150, "n_blocks": 8}

    def get_feature_impact_history(self) -> pd.DataFrame:
        return self._hist


def _frames(how: str, seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Synthetic Mūla frame (every published column), Swayam frame and view frames."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=N_SESSIONS, name="Date")
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, N_SESSIONS)))
    fvo = np.zeros(N_SESSIONS)
    for i in range(1, N_SESSIONS):
        fvo[i] = 0.97 * fvo[i - 1] + rng.normal(0.0, 0.25)
    fvo[-2:] = {"rich": 1.2, "cheap": -1.2, "fair": 0.2, "withheld": 1.2,
                "provisional": 1.2}[how]
    scale = 0.02
    gap = fvo * scale
    fv = price * np.exp(-gap)
    valid = np.arange(N_SESSIONS) >= BURN_IN
    reason = np.where(valid, "", "burn-in").astype(object)
    f = pd.DataFrame({
        "Price": price, "FairValue": fv, "FairValueLo": fv * 0.96, "FairValueHi": fv * 1.04,
        "FVO": np.where(valid, fvo, np.nan), "PctMispricing": np.where(valid, np.expm1(gap), np.nan),
        "GapScale": scale, "ModelSpread": 0.0012, "GapPhi": 0.97, "Residual": gap,
        "GapHalfLife": 23.0, "MRProb": 0.5 + 0.2 * np.sin(np.arange(N_SESSIONS) / 40.0),
        "MulaKappa": 0.004, "MulaHalfLife": 170.0, "MulaDriftPct": -0.3 * fvo, "MulaSdPct": 2.1,
        "KFactors": 5.0, "NAvailable": 150.0, "NAdmitted": 150.0, "NScheduled": 152.0,
        "Valid": valid, "Provisional": False, "WithheldReason": reason,
    }, index=idx)
    if how == "withheld":
        f.loc[idx[-1], ["FVO", "PctMispricing"]] = np.nan
        f.loc[idx[-1], ["Valid", "WithheldReason"]] = [False, "incomplete fetch"]
    if how == "provisional":
        f.loc[idx[-1], ["Valid", "Provisional"]] = [False, True]
    lean = np.clip(rng.normal(0.0, 0.25, N_SESSIONS), -1, 1)
    lean[-1] = -0.3                                   # contradicts a BUY: the binding sizer
    over = np.clip(50 + 50 * lean, 0, 100)
    sw = pd.DataFrame({"Lean": lean, "Oversold_Pct": over, "Overbought_Pct": 100 - over,
                       "N_Voting": 5, "Total_Analyzed": 12, "_Native": True}, index=idx.rename(None))
    sw.loc[idx[-3:-1], "_Native"] = False              # two carried sessions before d*
    views = {"L22·FULL": pd.DataFrame({
        "Close": price[-60:], "Unified_Osc": rng.normal(0, 3, 60), "MSF_Osc": rng.normal(0, 3, 60),
        "MMR_Osc": rng.normal(0, 3, 60), "Condition": "Neutral"}, index=idx[-60:])}
    return f, sw, views


def _priors(kind: str) -> PriorTable:
    """A study for Gold, with the fixture study's decisions, checks, arms and summary
    as its provenance.

    ``'strong'`` opens the claim and the Swayam sizer: a class-LOO claim prior
    and a global-LOO Swayam prior at IC 0.30. ``'class-LOO'`` / ``'global-LOO'``
    shrink the claim toward a pool with no skill (IC 0), so the claim holds and
    its reason names that pool.
    """
    meta = {k: v for k, v in json.loads(FIXTURE_PRIORS.read_text()).items() if k != "priors"}
    entry = {"var_mu": 1e-4, "tau2": 0.0, "kappa": 0.0, "se_floor": 0.01, "k": 6}
    if kind == "strong":
        return PriorTable({
            "claim": {"Gold": {"2019-01-01": {**entry, "mu": 0.30, "scope": "class-LOO"}}},
            "sizer:swayam": {"Gold": {"2019-01-01": {**entry, "mu": 0.30, "scope": "global-LOO"}}},
        }, meta)
    return PriorTable({"claim": {"Gold": {"2019-01-01": {**entry, "mu": 0.0, "scope": kind,
                                                         "k": 6 if kind == "class-LOO" else 36}}}},
                      meta)


class Bundle:
    """One synthetic run: the session state the pages read and the page context."""

    def __init__(self, state: str):
        import app

        target, how, priors, self.expected = STATES[state]
        cfg, inst = get_instrument_config(target), instrument(target)
        fvo_ts, sw, views = _frames(how)
        if not inst.has_ohlcv:
            sw, views = pd.DataFrame(), {}
        table = _priors(priors) if priors else None
        pub = fvo_ts.index[fvo_ts["FVO"].notna().to_numpy()]
        d_star = pub[-1]
        asm = assemble(fvo_ts, sw, instrument=inst, cfg=cfg, asof=d_star, today=fvo_ts.index[-1],
                       priors=table, epoch=fvo_ts.index[0], fingerprint="synthetic")
        last = fvo_ts.iloc[-1]
        withheld = bool(last["WithheldReason"])
        signal = {"fvo": float(fvo_ts.loc[d_star, "FVO"]),
                  "pct_mispricing": float(fvo_ts.loc[d_star, "PctMispricing"]),
                  "n_available": 150, "asof_row_withheld": withheld,
                  "withheld_reason": str(last["WithheldReason"]) if withheld else "",
                  "published_row": int(fvo_ts.index.get_loc(d_star)), "asof": d_star,
                  "provisional": bool(fvo_ts.loc[d_star, "Provisional"])}
        engine = _Engine(fvo_ts, signal)
        ts = engine.ts_data.copy()
        ts["Date"] = fvo_ts.index.to_numpy()
        ts_filtered = ts[ts["Date"] >= ts["Date"].max() - pd.DateOffset(months=6)]
        self.target, self.verdict, self.horizon = target, asm.verdict, int(cfg.forecast_horizon)
        self.priors_meta = describe_priors(table)
        self.session = {
            "active_target": target, "theme": "dark", "tf_selected": "6M", "fvo_ts": fvo_ts,
            "swayam_daily": sw, "swayam_view_dfs": views, "swayam_n_eff": 2.9 if views else None,
            "swayam_native_last": sw.index[sw["_Native"]].max() if len(sw) else None,
            "call_ts": asm.call_ts, "grades_ts": asm.grades_ts, "edges": asm.edges,
            "evidence": asm.evidence, "verdict": asm.verdict, "asof": d_star,
            "horizon": self.horizon, "exec_lag": knob("EXECUTION_LAG"), "epoch": fvo_ts.index[0],
            "fingerprint": "synthetic", "priors_meta": self.priors_meta,
        }
        self.asof_items = ([("Valued as of", f"{d_star:%d %b %Y}"),
                            ("Newest", f"{fvo_ts.index[-1]:%d %b} · {signal['withheld_reason']}")]
                           if withheld else [("As of", f"{d_star:%d %b %Y}")])
        self.ctx = app.PageContext(
            target=target, asset_class=inst.asset_class, cfg=cfg, horizon=self.horizon,
            verdict=asm.verdict, edges=asm.edges, evidence=asm.evidence, call_ts=asm.call_ts,
            grades_ts=asm.grades_ts, priors_meta=self.priors_meta, fvo_ts=fvo_ts, engine=engine,
            signal=signal, model_stats=engine.get_model_stats(), ts=ts, ts_filtered=ts_filtered,
            x_axis=ts_filtered["Date"], selected_tf="6M")


@lru_cache(maxsize=None)
def synthetic_bundle(state: str) -> Bundle:
    """The synthetic run for ``state`` (built once per process)."""
    return Bundle(state)


# ─── Drawing a page, and reading what was drawn ──────────────────────────────

def _page_script(state: str, page: str) -> None:
    """An AppTest script: one page of one synthetic state, with its command bar and rail."""
    import streamlit as st

    import app
    import test_render
    from ui.components import render_notice_rail, render_top_bar
    from ui.theme import inject_css

    b = test_render.synthetic_bundle(state)
    for k, v in b.session.items():
        st.session_state[k] = v
    inject_css(theme="dark")
    spec = next(p for p in app.PAGES if p.title == page)
    with st.sidebar:
        app._render_call_readout(b.verdict)
    render_top_bar(target=b.target, price=100.0, change_pct=0.1, status_label="CURRENT",
                   status_tone="success",
                   meta_items=app._command_meta(b.verdict, b.horizon, b.asof_items,
                                                "6M" if spec.windowed else None))
    render_notice_rail([])
    app.render_page(page, b.ctx)


def page_text(markdown_values) -> str:
    """The visible text of rendered markdown: styles dropped, tags stripped, unescaped."""
    return html.unescape("\n".join(
        re.sub(r"<[^>]+>", " ", re.sub(r"<style.*?</style>", "", v, flags=re.S))
        for v in markdown_values))


class Render:
    """One drawn page: its main and rail markdown, iframes, exceptions and errors."""

    def __init__(self, at):
        self.markdown = [m.value for m in at.markdown]
        self.sidebar = [m.value for m in at.sidebar.markdown]
        self.iframes = [getattr(e.proto, "srcdoc", "") for e in at.get("iframe")]
        self.exceptions = [e.value for e in at.exception]
        self.errors = [e.value for e in at.error]

    @property
    def text(self) -> str:
        return page_text(self.markdown)

    @property
    def rail(self) -> str:
        return page_text(self.sidebar)


@lru_cache(maxsize=None)
def rendered_states() -> dict[tuple[str, str], Render]:
    """Every page of every synthetic state, drawn once per process: ``{(state, page): Render}``."""
    from streamlit.testing.v1 import AppTest

    import app

    out = {}
    for state in STATES:
        synthetic_bundle(state)
        for spec in app.PAGES:
            at = AppTest.from_function(_page_script, args=(state, spec.title), default_timeout=300)
            at.run()
            out[(state, spec.title)] = Render(at)
    return out


_APP_RENDERS: dict[int, dict[str, Render]] = {}


def rendered_app(at) -> dict[str, Render]:
    """Every page of a running AppTest (e.g. ``gold_app``), drawn once: ``{page: Render}``."""
    key = id(at)
    if key not in _APP_RENDERS:
        pages = {}
        for page in at.session_state["_smoke_pages"]:
            at.session_state["_smoke_page"] = page
            at.run()
            pages[page] = Render(at)
        _APP_RENDERS[key] = pages
    return _APP_RENDERS[key]


def all_renders(gold_app) -> dict[str, Render]:
    """The fixture app's pages and every synthetic state's, keyed 'state/page'."""
    out = {f"fixture/{p}": r for p, r in rendered_app(gold_app).items()}
    out.update({f"{s}/{p}": r for (s, p), r in rendered_states().items()})
    return out


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_the_synthetic_states_are_the_states_they_name():
    for state, (_target, _how, kind, action) in STATES.items():
        b = synthetic_bundle(state)
        v = b.verdict
        assert v["action"] == action, (state, v["action"], v["headline"])
        if state == "buy":
            assert v["size"] > 0 and v["direction"] == 1
            # The Swayam row contradicts the BUY: with the sizers released it binds.
            assert v["binding"] == ("swayam" if knob("SIZERS_ENABLED") else "none")
            assert set(b.session["call_ts"]["claim_prior_scope"]) >= {"class-LOO"}
            assert b.priors_meta["status"] == "loaded"
        elif kind is None:
            assert b.priors_meta["label"] == NO_PRIOR_LABEL
        else:
            assert b.session["call_ts"].loc[b.verdict["asof"], "claim_prior_scope"] == kind
        if state == "no_call":
            assert v["limit"] == NOT_TRADEABLE
        if state == "withheld":
            assert v["caveat"] and "was not valued (incomplete fetch)" in v["caveat"]
        if state == "provisional":
            assert v["provisional"] and v["caveat"].startswith("PROVISIONAL")
        if state == "jeera":
            assert b.session["swayam_daily"].empty
            chain = {r["key"]: r for r in v["chain"]}
            assert chain["sizer:swayam"]["reading"] == "no Swayam read"


def test_a_hold_names_the_class_pool_and_the_study_it_was_shrunk_toward():
    """The verdict names the pool behind a HOLD — the instrument's class, by its page
    name, for a class pool, and the study's date — in its headline, its limit and the
    trust row; call_ts's own reason keeps the short form (it is graded data, unchanged)."""
    run_date = json.loads(FIXTURE_PRIORS.read_text())["run_date"]
    for state, pool, k in (("hold_class", "Commodities class-LOO prior +0.000", 6),
                           ("hold_global", "global-LOO prior +0.000", 36)):
        b = synthetic_bundle(state)
        v = b.verdict
        trust = {r["key"]: r for r in v["chain"]}["trust"]["grade"]
        for text in (v["headline"], v["limit"], trust):
            assert f"{pool} ± 0.010, k {k}, study {run_date}" in text, (state, text)
        reason = b.session["call_ts"].loc[v["asof"], "reason"]
        assert "study" not in reason and "Commodities" not in reason and "commodity" not in reason
        assert reason.endswith(f"{pool.split(' prior')[0].split()[-1]} prior +0.000 ± 0.010, k {k})")
    assert "Commodities" not in synthetic_bundle("hold_global").verdict["headline"]
    # With no prior: 'no pooled prior', and the limit is call_ts's reason itself.
    b = synthetic_bundle("hold")
    v = b.verdict
    trust = {r["key"]: r for r in v["chain"]}["trust"]["grade"]
    for text in (v["headline"], v["limit"], trust):
        assert "no pooled prior" in text and "study" not in text, text
    assert v["limit"] == f"Limited by the claim's grade: {b.session['call_ts'].loc[v['asof'], 'reason']}"
    assert "pooled" not in v["call_grade"]


@pytest.mark.parametrize("state", list(STATES))
def test_every_page_renders_in_every_synthetic_state(state):
    import app

    for spec in app.PAGES:
        r = rendered_states()[(state, spec.title)]
        assert not r.exceptions, (state, spec.title, r.exceptions)
        assert not r.errors, (state, spec.title, r.errors)
        assert "could not be drawn" not in r.text, (state, spec.title)
        assert synthetic_bundle(state).verdict["action"] in r.rail, (state, spec.title)


def test_the_jeera_swayam_page_opens_with_its_neutral_row_and_says_why():
    text = rendered_states()[("jeera", "Swayam")].text
    assert "no Swayam read" in text and "β 0.00 → ×1.00" in text
    assert "No Swayam breadth for this target" in text


def test_the_buy_state_draws_its_size_formula_its_binding_sizer_and_the_study():
    b = synthetic_bundle("buy")
    overview = rendered_states()[("buy", "Overview")]
    text = overview.text
    assert b.verdict["size_formula"] in text and b.verdict["limit"] in text
    grades = rendered_states()[("buy", "Grades")].text
    for label in ("Decision A · claim centring", "Decision B · SIZERS_ENABLED",
                  "Decision C · TIER_RULE", "catalogue · R4", "catalogue · R5", "catalogue · arm E",
                  "catalogue · arm P", "Arm F:"):
        assert label in grades, label
    assert "study not yet run" not in grades
    assert "study not yet run" in rendered_states()[("hold", "Grades")].text


def test_no_page_prints_an_asset_class_key(gold_app):
    """A class is printed by its page name (core.instruments.ASSET_CLASS_NAMES), never
    by its registry key; the fixture app reads the committed study, whose Grades
    tables list every class."""
    keys = [k for k in ASSET_CLASS_NAMES if "_" in k]
    pat = re.compile(r"(?<![\w:])(" + "|".join(keys) + r")(?![\w:])")
    seen = set()
    for where, r in all_renders(gold_app).items():
        frames = [page_text([f]) for f in r.iframes]
        for text in (r.text, r.rail, *frames):
            assert not pat.search(text), (where, pat.search(text).group(0))
            seen |= {n for n in ASSET_CLASS_NAMES.values() if n in text}
    if _committed_priors():
        assert {"Commodities", "India Indices", "US Indices"} <= seen, seen


def test_a_missing_or_unregistered_card_raises():
    from ui import components as c

    with pytest.raises(ValueError, match="names no card"):
        c.render_kpi_strip([{"label": "Coverage", "value": "1"}])
    with pytest.raises(KeyError, match="unregistered card"):
        c.render_kpi_strip([{"label": "Coverage", "value": "1", "card": "overview:nope"}])
    with pytest.raises(ValueError, match="carries no grade"):
        c.render_kpi_strip([{"label": "Valuation", "value": "RICH", "card": "mula:valuation",
                             "lean": True}])
    with pytest.raises(TypeError):
        c.render_section_header("Who Closes the Gap?", "")     # card is required
    with pytest.raises(KeyError, match="unregistered card"):
        c.render_section_header("Who Closes the Gap?", "", card="overview:nope")
    with pytest.raises(KeyError, match="unregistered card"):
        c.render_role_chip("nope")
    with pytest.raises(TypeError):
        c.render_metric_card("Coverage", "1")                  # card is required
    with pytest.raises(KeyError, match="not chain rows"):
        c.render_gate_rows(synthetic_bundle("hold").verdict, ["claim", "conviction"])


# ─── The app with and without a pooled-prior study file ──────────────────────

@lru_cache(maxsize=None)
def _committed_priors() -> bool:
    """Is a pooled-prior study file committed for h = 10 and this L? Read once, before
    any fixture points the loader elsewhere."""
    return trust.load_priors(10, knob("EXECUTION_LAG")) is not None


@pytest.fixture(scope="module")
def other_priors_app(tmp_path_factory):
    """The fixture app in the priors state the repository is NOT in.

    With a committed study file, the file removed (the loader pointed at an
    empty directory, the registry as import leaves it with no file). Without
    one, the offline study's file loaded from a mini repository — the file
    where the app looks for it (``data/priors``) and a stand-in for the study
    script, since a measured knob must name its study — with its knobs
    registered as import registers a committed file. Nothing is written to
    the repository either way.
    """
    from conftest import install_offline_fetch, run_app_offline

    committed = _committed_priors()
    mini = tmp_path_factory.mktemp("repo")
    pdir = mini / "data" / "priors"
    pdir.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trust, "PRIORS_DIR", pdir)
        mp.setattr(knobs, "KNOBS", {**{n: k for n, k in KNOBS.items() if k.kind != "measured"},
                                    **knobs.PLACEHOLDERS})
        mp.setattr(knobs, "STUDY_GENERATED", [])
        if not committed:
            (mini / "studies").mkdir()
            (mini / STUDY).write_text("")
            path = pdir / priors_file_name(10, knob("EXECUTION_LAG"))
            shutil.copy(FIXTURE_PRIORS, path)
            mp.setattr(knobs, "_ROOT", mini)
            knobs._register_study(path)
            assert knobs.KNOBS["POOLED_PRIOR"].kind == "measured"
        install_offline_fetch(mp)
        at = run_app_offline("Gold", "Commodities")
        yield at, rendered_app(at)


@pytest.fixture(scope="module")
def priors_states(gold_app, other_priors_app):
    """``{'loaded': (at, pages), 'none': (at, pages)}`` — the app in both priors states."""
    here = (gold_app, rendered_app(gold_app))
    return ({"loaded": here, "none": other_priors_app} if _committed_priors()
            else {"loaded": other_priors_app, "none": here})


def test_every_page_renders_with_and_without_the_pooled_priors(priors_states):
    import app

    for state, (at, pages) in priors_states.items():
        assert not at.exception, (state, [e.value for e in at.exception])
        assert set(pages) == {p.title for p in app.PAGES}, state
        for page, r in pages.items():
            assert not r.exceptions and not r.errors, (state, page, r.exceptions, r.errors)
            assert "could not be drawn" not in r.text, (state, page)


def test_with_the_priors_loaded_every_surface_reads_the_study(priors_states):
    at, pages = priors_states["loaded"]
    ss = at.session_state
    meta = ss["priors_meta"]
    assert meta["status"] == "loaded" and meta["label"] == f"pooled prior: {STUDY} ({meta['run_date']})"
    ct = ss["call_ts"]
    assert ct.attrs["priors_study"] == STUDY
    scopes = set(ct["claim_prior_scope"]) - {"none", "ungraded"}
    assert scopes and scopes <= {"class-LOO", "global-LOO", "class-all", "global-all"}, scopes
    held = ct["action"] == "HOLD"
    if held.any():                                   # a HOLD names the pool it was shrunk toward
        named = ct.loc[held & (ct["claim_prior_scope"].isin(scopes)), "reason"]
        assert named.str.contains("|".join(f"{s} prior" for s in scopes)).all()
    # The verdict names the pool with its class and the study's date.
    scope = str(ct.loc[ss["asof"], "claim_prior_scope"])
    if scope in scopes:
        trust = {r["key"]: r for r in ss["verdict"]["chain"]}["trust"]["grade"]
        assert f"study {meta['run_date']}" in trust, trust
        assert (f"{asset_class_name(ct.attrs['asset_class'])} {scope} prior"
                if scope.startswith("class") else f"{scope} prior") in trust, trust
        assert trust in pages["Overview"].text and trust in pages["Mūla"].text
    grades = pages["Grades"].text
    for label in ("Decision A · claim centring", "Decision B · SIZERS_ENABLED",
                  "Decision C · TIER_RULE", "catalogue · R4", "catalogue · R5", "catalogue · arm E",
                  "catalogue · arm P", "Arm F:", meta["label"]):
        assert label in grades, label
    assert "study not yet run" not in grades
    assert meta["label"] in pages["System"].text and "Studied at" in pages["System"].text
    # The KNOBS table the System page lists carries the study's measured entries.
    assert any("measured" in f and "POOLED_PRIOR" in f for f in pages["System"].iframes)
    # The Grades table's priors are the scopes call_ts read.
    assert any(any(f"{s} · k " in f for s in scopes) for f in pages["Grades"].iframes)


def test_without_the_priors_every_surface_says_so(priors_states):
    at, pages = priors_states["none"]
    ss = at.session_state
    assert ss["priors_meta"]["label"] == NO_PRIOR_LABEL and ss["priors_meta"]["status"] == "none"
    assert set(ss["call_ts"]["claim_prior_scope"]) <= {"none", "ungraded"}
    assert "study not yet run" in pages["Grades"].text
    assert "study not yet run" in pages["System"].text
    assert NO_PRIOR_LABEL in pages["Grades"].text


def test_the_landing_page_renders_with_registered_cards():
    from streamlit.testing.v1 import AppTest

    from conftest import _APP_WRAPPER
    from core.roles import CARDS

    at = AppTest.from_string(_APP_WRAPPER.replace("APP_PATH", repr(str(ROOT / "app.py"))),
                             default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cards = [c for m in at.markdown for c in re.findall(r'data-card="([^"]*)"', m.value)]
    assert {"landing:coverage", "landing:systems", "landing:returns"} <= set(cards) <= set(CARDS)
    assert any(b.label == "Run Analysis" for b in at.button)
