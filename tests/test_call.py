"""THE call (core.call) and the closed loop (pipeline.assemble): pure, causal, graded as issued.

A synthetic panel with a PLANTED claim (FVO leans against the forward return,
so cheap precedes a rise) and a planted Swayam lean drives the whole loop:
inputs read by date → as-of grades → the call at every session → the call
graded as issued → the verdict at d*. Hand-built inputs pin the arithmetic.

The retired tests' intents are asserted here: from test_hero_verdict (the
chain in decision order, the action word follows the size, no claim inside the
noise, absent evidence is neutral and an unvalidated claim is not acted on,
κ̂ reaches the card, the binding factor is the smallest, no claim has nothing
to confirm, periods count scored periods only) and from test_edge_and_learners
(a carried Swayam row is ×1.00 and never fabricated; nothing is issued on an
unvalued session).
"""
import math

import numpy as np
import pandas as pd
import pytest

import pipeline.assemble as asm_mod
from core.call import (ACTIONS, CALL_COLUMNS, NOT_TRADEABLE, SIZERS, compose_series,
                       edge_periods, format_size, verdict_from_row)
from core.config import InstrumentConfig
from core.instruments import instrument
from core.knobs import knob
from core.roles import CHAIN, ROLES
from core.semantics import breadth_lean
from pipeline.assemble import assemble, gather_inputs

H = 10
L = knob("EXECUTION_LAG")
T0 = knob("STRENGTH_T0")
GOLD = instrument("Gold")
EDGE_KEYS = {"claim", "claim:buy", "claim:sell", "sizer:swayam", "sizer:reversion",
             "call", "call:prospective", "evidence:swayam_lean"}


# ─── The synthetic panel ─────────────────────────────────────────────────────

def _panel(n=1500, seed=3, burn=120):
    """``(fvo_ts, swayam_daily)`` with a planted claim and a planted Swayam lean."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, name="Date")
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=idx)
    lp = np.log(price)
    Y = (lp.shift(-(H + L)) - lp.shift(-L)).fillna(0.0)
    z = (Y - Y.mean()) / Y.std()
    fvo = -(0.5 * z + 1.0 * rng.normal(size=n))                 # cheap (FVO < 0) precedes a rise
    lean = (0.4 * z + rng.normal(size=n)).clip(-1, 1) * 0.6     # oversold precedes a rise
    valid = np.arange(n) >= burn
    fvo_ts = pd.DataFrame({
        "Price": price,
        "FVO": np.where(valid, fvo, np.nan),
        "PctMispricing": np.where(valid, fvo * 0.01, np.nan),
        "FairValue": price * np.exp(-np.where(valid, fvo * 0.01, 0.0)),
        "Valid": valid, "Provisional": False,
        "WithheldReason": np.where(valid, "", "burn-in"),
        "MRProb": np.clip(0.55 + 0.1 * rng.normal(size=n), 0.01, 0.99),
        "MulaKappa": 0.01 * rng.normal(size=n),
    }, index=idx)
    native = np.ones(n, dtype=bool)
    native[::17] = False                                          # some carried rows
    sw = _breadth((100 * lean).clip(lower=0), (-100 * lean).clip(lower=0), native)
    return fvo_ts, sw


def _breadth(oversold, overbought, native) -> pd.DataFrame:
    """A Swayam frame as aggregate_views publishes it: the shares and their one Lean."""
    return pd.DataFrame({"Oversold_Pct": oversold, "Overbought_Pct": overbought,
                         "Lean": [breadth_lean(a, b) for a, b in zip(oversold, overbought)],
                         "_Native": native}, index=oversold.index)


def _asof(fvo_ts):
    f = pd.to_numeric(fvo_ts["FVO"], errors="coerce")
    ok = f.notna() & (fvo_ts["WithheldReason"] == "")
    return fvo_ts.index[ok.to_numpy()][-1]


def _assemble(fvo_ts, sw, cfg=None, **kw):
    return assemble(fvo_ts, sw, instrument=kw.pop("instrument", GOLD), cfg=cfg or InstrumentConfig(),
                    asof=_asof(fvo_ts), today=fvo_ts.index[-1], priors=None, **kw)


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def loop(panel):
    return _assemble(*panel)


# ─── Hand-built inputs for the arithmetic ───────────────────────────────────

def _inputs(n, *, fvo=-1.0, z_sw=0.0, z_rev=0.0, **cols):
    idx = pd.bdate_range("2024-01-01", periods=n)
    f = pd.Series(fvo, index=idx, dtype=float)
    out = pd.DataFrame({
        "price": 100.0, "fvo": f, "pct": f, "valid": True, "provisional": False,
        "published": f.notna(), "withheld_reason": "", "printed": True, "behind": 0,
        "claim": f.map(lambda v: 0 if not np.isfinite(v) or abs(v) < 0.5 else (1 if v < 0 else -1)),
        "swayam_lean": 0.2, "swayam_native": True, "z_swayam": z_sw,
        "mr_prob": 0.6, "kappa": 0.012, "z_reversion": z_rev,
    }, index=idx)
    for k, v in cols.items():
        out[k] = v
    out.attrs = {"h": H, "target": "Synthetic", "asset_class": "commodity", "asof": idx[-1]}
    return out


def _grade(idx, t):
    """An as-of grade frame with posterior t ``t`` on every date."""
    return pd.DataFrame({"ic": 0.1, "se": 0.1, "n_win": 100, "grid_date": idx, "prior_mu": np.nan,
                         "prior_sd": np.nan, "prior_scope": "none", "prior_k": 0, "ic_post": 0.1,
                         "se_post": 0.1, "t_post": t, "w_own": 1.0, "strength": 0.0}, index=idx)


def _grades(idx, claim_t, sizer_t):
    g = {"claim": _grade(idx, claim_t)}
    for s in SIZERS:
        g[f"sizer:{s.key}"] = _grade(idx, sizer_t)
    return g


def _t_for(strength):
    return T0 + strength * (knob("STRENGTH_T1") - T0)


# ─── (a) Truncation ──────────────────────────────────────────────────────────

def _same(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
        return np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
    return all((x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b))


@pytest.mark.parametrize("cut", [1, 60, 300])
def test_a_resolved_rows_never_change_when_the_panel_is_cut(panel, loop, cut):
    fvo_ts, sw = panel
    T = fvo_ts.index[-1 - cut]
    part = _assemble(fvo_ts.loc[:T], sw.loc[:T]).call_ts
    resolved = part.index[: len(part) - (L + H)]                  # Y resolved inside the cut
    full = loop.call_ts.loc[resolved]
    assert list(part.columns) == list(full.columns) == list(CALL_COLUMNS)
    assert (part.loc[resolved, "w"] != 0).any(), "the planted claim must issue calls to test"
    for col in CALL_COLUMNS:
        assert _same(part.loc[resolved, col], full[col]), col


# ─── (b) β = 0 is exactly ×1.00; absent or carried Swayam is ×1.00, never filled ─

def test_b_unskilled_sizer_is_exactly_one_for_any_reading():
    zs = [-1.0, -0.3, 0.0, 0.7, 1.0, np.nan]
    inp = _inputs(len(zs), z_sw=zs, z_rev=zs)
    for sizer_t, enabled in ((1.0, True), (np.nan, True), (3.0, False)):
        ct = compose_series(inp, _grades(inp.index, 3.0, sizer_t), tradeable=True,
                            sizers_enabled=enabled)
        for s in SIZERS:
            assert (ct[f"{s.key}_beta"] == 0.0).all()
            assert (ct[f"{s.key}_factor"] == 1.0).all(), ct[f"{s.key}_factor"].tolist()
        assert (ct["size"] == ct["trust"]).all() and (ct["trust"] == 1.0).all()


def test_b_absent_or_carried_swayam_is_neutral_and_never_fabricated(panel):
    fvo_ts, sw = panel
    sw = sw.copy()
    absent, carried = fvo_ts.index[-40], fvo_ts.index[-30]
    sw = sw.drop(index=absent)
    sw.loc[carried, "_Native"] = False
    inp = gather_inputs(fvo_ts, sw, instrument=GOLD, h=H, today=fvo_ts.index[-1])
    assert np.isnan(inp.loc[absent, "swayam_lean"])                 # not 0, not 50/50
    assert not inp.loc[absent, "swayam_native"] and not inp.loc[carried, "swayam_native"]
    assert np.isnan(inp.loc[absent, "z_swayam"]) and np.isnan(inp.loc[carried, "z_swayam"])
    # A fully skilled Swayam sizer (β = 1) still leaves both sessions at ×1.00.
    ct = compose_series(inp, _grades(inp.index, 3.0, 3.0), tradeable=True, sizers_enabled=True)
    for when in (absent, carried):
        assert ct.loc[when, "swayam_z"] == 0.0 and ct.loc[when, "swayam_factor"] == 1.0
    assert (ct["swayam_beta"] == 1.0).all()
    native = inp["swayam_native"] & (ct["d"] != 0)
    assert (ct.loc[native, "swayam_factor"] != 1.0).any()          # a native reading does act


# ─── (c) Sizer inputs never move the direction (C3) ─────────────────────────

def test_c_perturbing_every_sizer_input_never_changes_d(panel, loop):
    fvo_ts, sw = panel
    rng = np.random.default_rng(11)
    f2, s2 = fvo_ts.copy(), sw.copy()
    f2["MRProb"] = rng.uniform(0, 1, len(f2))
    f2["MulaKappa"] = rng.normal(0, 0.05, len(f2))
    s2 = _breadth(pd.Series(rng.uniform(0, 100, len(s2)), index=s2.index),
                  pd.Series(rng.uniform(0, 100, len(s2)), index=s2.index),
                  rng.uniform(size=len(s2)) > 0.3)
    other = _assemble(f2, s2).call_ts
    assert other["d"].equals(loop.call_ts["d"])
    for col in ("claim_t", "trust", "precondition"):
        assert _same(other[col], loop.call_ts[col]), col


# ─── (d) Trust 0 is a HOLD that names its grade ─────────────────────────────

def test_d_trust_zero_is_hold_naming_t_post_and_the_bar():
    inp = _inputs(3, fvo=[-1.2, 0.9, -0.6])
    ct = compose_series(inp, _grades(inp.index, 1.0, 3.0), tradeable=True, sizers_enabled=True)
    assert (ct["action"] == "HOLD").all() and (ct["trust"] == 0).all() and (ct["size"] == 0).all()
    assert list(ct["d"]) == [1, -1, 1]                               # the claim is kept, not acted on
    for r in ct["reason"]:
        assert "t_post 1.00" in r and "1.645" in r


# ─── (e) A failed grade: the claim is ungraded (trust 0, d kept); a sizer gets β = 0 ─

def _raising(when):
    real = asm_mod.expanding_edge

    def fake(lean, price, horizon, *, sign=None, **kw):
        if when(sign):
            raise RuntimeError("forced")
        return real(lean, price, horizon, sign=sign, **kw)
    return fake


def test_e_a_failed_claim_grade_is_ungraded_and_keeps_d(panel, loop, monkeypatch):
    monkeypatch.setattr(asm_mod, "expanding_edge", _raising(lambda sign: sign is None))
    a = _assemble(*panel)
    ct = a.call_ts
    assert "claim" not in a.grades_ts
    assert (ct["trust"] == 0).all() and (ct["w"] == 0).all()
    assert (ct["claim_prior_scope"] == "ungraded").all()
    assert ct["d"].equals(loop.call_ts["d"])
    held = ct["action"] == "HOLD"
    assert held.any() and ct.loc[held, "reason"].str.contains("ungraded").all()


def test_e_a_failed_sizer_grade_gives_beta_zero(panel, loop, monkeypatch):
    monkeypatch.setattr(asm_mod, "expanding_edge", _raising(lambda sign: sign is not None))
    ct = _assemble(*panel).call_ts
    for s in SIZERS:
        assert (ct[f"{s.key}_beta"] == 0).all() and (ct[f"{s.key}_factor"] == 1.0).all()
    assert ct["trust"].equals(loop.call_ts["trust"])


# ─── (f) Preconditions ──────────────────────────────────────────────────────

def test_f_a_withheld_session_is_no_call_and_nothing_is_issued_on_it(panel):
    fvo_ts, sw = panel
    f = fvo_ts.copy()
    last = f.index[-1]
    f.loc[last, ["FVO", "PctMispricing"]] = np.nan
    f.loc[last, "Valid"] = False
    f.loc[last, "WithheldReason"] = "incomplete fetch"
    a = _assemble(f, sw)
    ct = a.call_ts
    assert ct.loc[last, "action"] == "NO CALL" and ct.loc[last, "reason"] == "incomplete fetch"
    unvalued = ~ct["published"]
    assert unvalued.sum() > 100                                    # the burn-in too
    assert (ct.loc[unvalued, "action"] == "NO CALL").all()
    assert (ct.loc[unvalued, ["d", "size", "w"]] == 0).all().all()
    assert a.verdict["asof"] == f.index[-2] and "incomplete fetch" in a.verdict["caveat"]


def test_f_untradeable_is_no_call_and_inside_the_noise_is_no_claim():
    inp = _inputs(4, fvo=[-1.0, 0.8, 0.3, -0.49])
    ct = compose_series(inp, _grades(inp.index, 3.0, 3.0), tradeable=False, sizers_enabled=True)
    assert list(ct["action"]) == ["NO CALL"] * 4
    assert (ct["reason"] == NOT_TRADEABLE).all() and (ct["w"] == 0).all()
    ct = compose_series(inp, _grades(inp.index, 3.0, 3.0), tradeable=True, sizers_enabled=True)
    assert list(ct["action"]) == ["BUY", "SELL", "NO CLAIM", "NO CLAIM"]
    assert list(ct["precondition"]) == ["", "", "claim_sigma", "claim_sigma"]
    assert "inside the ±0.5σ noise" in ct["reason"].iloc[2]
    assert (ct["binding"].iloc[2:] == "").all()                     # no claim, nothing binds


def test_f_a_target_behind_by_the_horizon_is_no_call():
    inp = _inputs(2, behind=[0, H])
    ct = compose_series(inp, _grades(inp.index, 3.0, 1.0), tradeable=True, sizers_enabled=True)
    assert list(ct["action"]) == ["BUY", "NO CALL"]
    assert ct["reason"].iloc[1] == f"{H} sessions behind: the {H}-session horizon has run"


# ─── (g) Binding is the smallest factor below 1, never trust ────────────────

def test_g_binding_is_the_smallest_factor_below_one_never_trust():
    inp = _inputs(4, z_sw=[-0.5, 0.3, 0.4, -0.1], z_rev=[-0.2, -0.4, 0.1, 0.0])
    # Trust 0.3 is smaller than every factor; it still never binds.
    ct = compose_series(inp, _grades(inp.index, _t_for(0.3), 3.0), tradeable=True,
                        sizers_enabled=True)
    assert np.allclose(ct["trust"], 0.3)
    assert list(ct["binding"]) == ["swayam", "reversion", "none", "swayam"]
    f = ct[["swayam_factor", "reversion_factor"]].to_numpy()
    assert np.allclose(ct["size"], 0.3 * f.prod(axis=1))           # size = trust × ∏ factor
    assert set(ct["binding"]) <= {s.key for s in SIZERS} | {"none", ""}


# ─── (h) The verdict is call_ts at d* ───────────────────────────────────────

def test_h_the_verdict_is_call_ts_at_d_star(loop, panel):
    d_star = _asof(panel[0])
    v = loop.verdict
    again = verdict_from_row(loop.call_ts, loop.edges, asof=d_star, horizon=H, exec_lag=L,
                             target=GOLD.name)
    assert v == again
    row = loop.call_ts.loc[d_star]
    assert v["asof"] == d_star and v["action"] == row["action"] and v["size"] == row["size"]
    assert v["trust"] == row["trust"] and v["direction"] == row["d"]
    assert v["headline"].startswith(v["action"])


# ─── (i) The edges, one horizon ─────────────────────────────────────────────

def test_i_every_role_is_graded_at_the_one_horizon(loop):
    assert EDGE_KEYS <= set(loop.edges)
    for k, e in loop.edges.items():
        assert e["horizon"] == H and e["exec_lag"] == L and e["attrs"]["h"] == H, k
    assert loop.call_ts.attrs["h"] == H == loop.verdict["horizon_days"]
    assert loop.edges["call"]["issued_n"] == int((loop.call_ts["w"] != 0).sum())
    assert loop.edges["call"]["claim_n"] == int((loop.call_ts["d"] != 0).sum())


def test_j_a_twenty_session_config_reaches_every_object(panel):
    a = _assemble(*panel, cfg=InstrumentConfig(forecast_horizon=20))
    assert a.call_ts.attrs["h"] == 20 == a.verdict["horizon_days"]
    for k, e in a.edges.items():
        assert e["horizon"] == 20 and e["attrs"]["h"] == 20, k
    for obj, g in a.grades_ts.items():
        grid = pd.DatetimeIndex(g["grid_date"].unique())
        assert len(grid) == math.ceil(len(g) / 20), obj               # a grid every 20 rows


# ─── (k) The chain, in decision order ───────────────────────────────────────

def test_k_the_chain_is_in_decision_order(loop):
    assert CHAIN == ("claim", "precondition", "trust", "sizer:swayam", "sizer:reversion", "size")
    assert all(k in ROLES for k in CHAIN)
    chain = loop.verdict["chain"]
    assert [r["key"] for r in chain] == list(CHAIN)
    assert [r["role"] for r in chain] == ["CLAIM", "PRECONDITION", "GRADE", "SIZER", "SIZER", "CALL"]
    for r in chain:
        assert set(r) == {"role", "key", "title", "reading", "grade", "effect", "detail"}


# ─── (l) κ̂ reaches the reversion row ────────────────────────────────────────

def test_l_the_reversion_row_carries_mrprob_and_kappa():
    inp = _inputs(1)
    ct = compose_series(inp, _grades(inp.index, 1.0, 1.0), tradeable=True, sizers_enabled=True)
    v = verdict_from_row(ct, {}, asof=inp.index[-1], horizon=H, exec_lag=L, target="Synthetic")
    row = next(r for r in v["chain"] if r["key"] == "sizer:reversion")
    assert "MRProb 0.60" in row["detail"] and "κ̂ +0.0120/d" in row["detail"]


# ─── (m) Periods count scored periods only ──────────────────────────────────

def test_m_periods_count_scored_periods_only():
    e = dict(ic=0.08, se=0.05, t=1.6, positive_blocks=3, scored_blocks=4,
             blocks=[{"ic": 0.1}, {"ic": float("nan")}, {"ic": -0.1}, {"ic": 0.2}, {"ic": 0.3},
                     {"ic": float("nan")}])
    assert edge_periods(e) == (3, 4)
    inp = _inputs(1)
    ct = compose_series(inp, _grades(inp.index, 1.0, 1.0), tradeable=True, sizers_enabled=True)
    v = verdict_from_row(ct, {"claim": e}, asof=inp.index[-1], horizon=H, exec_lag=L,
                         target="Synthetic")
    trust = next(r for r in v["chain"] if r["key"] == "trust")
    assert "3 of 4 periods positive" in trust["grade"]


# ─── The retired hero tests, re-expressed ───────────────────────────────────

def test_the_action_word_follows_the_size():
    """A claim the grade does not support is a HOLD — never BUY beside 'not shown to pay'."""
    inp = _inputs(2, fvo=[-1.4, 1.4])
    held = compose_series(inp, _grades(inp.index, 1.2, 1.0), tradeable=True, sizers_enabled=True)
    acted = compose_series(inp, _grades(inp.index, _t_for(0.5), 1.0), tradeable=True,
                           sizers_enabled=True)
    assert list(held["action"]) == ["HOLD", "HOLD"] and list(acted["action"]) == ["BUY", "SELL"]
    assert np.allclose(acted["size"], 0.5) and list(acted["w"]) == pytest.approx([0.5, -0.5])
    for ct, word in ((held, "HOLD"), (acted, "SELL")):
        v = verdict_from_row(ct, {}, asof=inp.index[-1], horizon=H, exec_lag=L, target="X")
        assert v["action"] == v["word"] == word and v["headline"].startswith(word)
    assert set(ACTIONS) == {"NO CALL", "NO CLAIM", "HOLD", "BUY", "SELL"}


def test_no_claim_inside_the_noise_has_nothing_to_size_or_confirm():
    inp = _inputs(1, fvo=0.1)
    ct = compose_series(inp, _grades(inp.index, 3.0, 3.0), tradeable=True, sizers_enabled=True)
    v = verdict_from_row(ct, {}, asof=inp.index[-1], horizon=H, exec_lag=L, target="X")
    assert v["action"] == "NO CLAIM" and v["size"] == 0.0 and v["binding"] is None
    assert v["size_formula"].startswith("no claim to size")
    sw = next(r for r in v["chain"] if r["key"] == "sizer:swayam")
    assert "no claim to confirm" in sw["reading"]


def test_absent_evidence_is_neutral_and_an_ungraded_claim_is_not_acted_on():
    inp = _inputs(2, z_sw=[np.nan, 0.5], z_rev=[np.nan, -0.5])
    ct = compose_series(inp, {}, tradeable=True, sizers_enabled=True)
    assert (ct["action"] == "HOLD").all() and (ct["trust"] == 0).all()
    assert (ct["claim_prior_scope"] == "ungraded").all()
    assert ct["reason"].str.startswith("claim ungraded").all()
    for s in SIZERS:
        assert (ct[f"{s.key}_factor"] == 1.0).all()


def test_identical_issued_calls_are_ungraded_and_the_rail_still_reads():
    """Every issued call the same (one side, size 1.00): nothing to rank, so not 'graded'.

    Trust saturated and both sizers at β = 0 give w ≡ +1 on every row, past
    the warm-up. The call's IC and SE are then undefined; the grade must say
    so rather than read 'graded', and the rail readout drawn on every page
    must build from the verdict without formatting a missing IC.
    """
    import app

    n = 1200
    inp = _inputs(n)
    grades = _grades(inp.index, 3.0, 0.0)
    ct = compose_series(inp, grades, tradeable=True, sizers_enabled=True)
    assert (ct["w"] == 1.0).all() and (ct["size"] == 1.0).all()
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=inp.index)
    edges = asm_mod.grade_all(ct, inp, price, {}, h=H, exec_lag=L, grades_ts=grades,
                              priors_meta={})
    e = edges["call"]
    assert e["n_win"] >= knob("GRADE_WARMUP_WIN") and not np.isfinite(e["ic"])
    assert e["graded"] is False and e["status"] != "graded"
    assert "nothing to rank" in e["status"]
    v = verdict_from_row(ct, edges, asof=inp.index[-1], horizon=H, exec_lag=L, target="X")
    assert v["call_grade"]["graded"] is False and v["call_grade"]["ic"] is None
    assert dict((k, val) for k, val, _ in app._call_readout_rows(v))["Call grade"] == "ungraded"


def test_the_size_is_an_index_never_a_percentage():
    assert format_size(0.3149) == "0.31" and format_size(0) == "0.00"
    assert format_size(float("nan")) == "—" and format_size(None) == "—"
    assert "%" not in format_size(1.7)
