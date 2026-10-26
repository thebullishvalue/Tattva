"""Swayam's one lean: equal-weight breadth, published once, read the same everywhere.

The view bank runs on the committed Gold fixture bars with a small macro
driver pool from the fixture panel (enough for the macro-anchored views to
vote; the full ~220-column pool would only make the test slower). Its
reduction (``aggregate_views``) publishes the shares, the view counts and ONE
lean, ``Lean = core.semantics.breadth_lean(Oversold_Pct, Overbought_Pct)``,
which the composer, the sizer's grade, every page and the export read.

The retired learner tests' intents are asserted here: the contrarian sign
convention (an oversold majority is the bullish lean) and causality (now of
the equal-weight breadth, under OHLCV truncation). A carried Swayam row is
×1.00 in the composer; the bank's members that cannot reach the ±vote cut are
flagged, and the voting count is exactly the members that can.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.call import SIZERS, compose_series
from core.config import InstrumentConfig, swayam_macro_columns
from core.instruments import instrument
from core.semantics import breadth_lean
from engines.swayam import (aggregate_views, build_swayam_frames, default_swayam_members,
                            member_can_vote, voting_views)
from engines.swayam.kernel import can_vote, osc_reach
from pipeline.assemble import gather_inputs
from pipeline.run import swayam_drivers
from studies.make_fixtures import load_ohlcv, load_panel

CFG = InstrumentConfig()
H = CFG.forecast_horizon
#: A small, liquid driver pool from the fixture panel (none a replica of Gold).
DRIVERS = ("Silver", "Copper", "Crude Oil", "Dollar Index", "US Large Cap (S&P 500)",
           "US Treasury Long (20Y+)", "EUR/USD", "Emerging Markets Equity")
BREADTH = ["Oversold_Pct", "Overbought_Pct", "Total_Analyzed", "N_Voting", "Lean"]


def _frames(bars: pd.DataFrame, drivers: pd.DataFrame, cfg=CFG) -> dict:
    cols = swayam_macro_columns("Gold", list(drivers.columns))
    return build_swayam_frames(
        bars, drivers, cols, members=default_swayam_members(cfg.swayam_lengths, cfg.swayam_roc_frac),
        regime_sensitivity=cfg.swayam_regime_sensitivity, base_weight=cfg.swayam_base_weight,
        num_vars=cfg.swayam_mmr_num_vars, oversold=cfg.swayam_oversold,
        overbought=cfg.swayam_overbought)


@pytest.fixture(scope="module")
def inputs():
    bars = load_ohlcv("Gold")
    drivers = swayam_drivers(load_panel(), "Gold")
    return bars, drivers[[c for c in DRIVERS if c in drivers.columns]]


@pytest.fixture(scope="module")
def bank(inputs):
    frames = _frames(*inputs)
    return frames, aggregate_views(frames)


# ─── The one lean ────────────────────────────────────────────────────────────

def test_lean_is_breadth_lean_of_the_published_shares(bank):
    _frames_, daily = bank
    assert list(daily.columns) == BREADTH
    want = [breadth_lean(a, b) for a, b in zip(daily["Oversold_Pct"], daily["Overbought_Pct"])]
    assert np.array_equal(daily["Lean"].to_numpy(), np.asarray(want))
    assert daily["Lean"].abs().max() > 0, "the bank must vote on the fixture to test anything"


def test_the_contrarian_convention_oversold_is_a_positive_lean(bank):
    # Hand-built: two of three views oversold on day 1, two overbought on day 2.
    idx = pd.bdate_range("2024-01-01", periods=2)
    views = {k: pd.DataFrame({"Condition": c, "Can_Vote": True}, index=idx)
             for k, c in (("a", ["Oversold", "Overbought"]), ("b", ["Oversold", "Overbought"]),
                          ("c", ["Neutral", "Oversold"]))}
    d = aggregate_views(views)
    assert d["Lean"].iloc[0] > 0 > d["Lean"].iloc[1]
    assert d["Lean"].iloc[0] == pytest.approx(2 / 3) and d["Lean"].iloc[1] == pytest.approx(-1 / 3)
    # And on the bank: an oversold majority is a bullish lean, an overbought one bearish.
    _f, daily = bank
    net = daily["Oversold_Pct"] - daily["Overbought_Pct"]
    assert (np.sign(daily["Lean"]) == np.sign(net)).all()
    assert (daily.loc[net > 0, "Lean"] > 0).any() and (daily.loc[net < 0, "Lean"] < 0).any()


# ─── Who can vote ────────────────────────────────────────────────────────────

def test_a_bank_member_that_cannot_reach_the_cut_is_flagged(bank):
    frames, _d = bank
    members = default_swayam_members(CFG.swayam_lengths, CFG.swayam_roc_frac)
    for m in members:
        flagged = not member_can_vote(m, CFG)
        assert flagged == (not m.use_macro), m.name      # at base weight 0, price-only views are mute
        if m.name in frames:
            assert bool(frames[m.name]["Can_Vote"].iloc[0]) == (not flagged), m.name
        if flagged and m.name in frames:
            # The flag is the truth: a flagged view never left Neutral.
            osc = frames[m.name]["Unified_Osc"].abs()
            assert osc.max() < osc_reach(False, CFG.swayam_base_weight) <= CFG.swayam_overbought
            assert (frames[m.name]["Condition"] == "Neutral").all(), m.name
    assert voting_views(CFG) == (sum(m.use_macro for m in members), len(members))
    # A larger MSF share lifts a price-only view's reach past the cut: the flag follows it.
    wider = replace(CFG, swayam_base_weight=0.2)
    assert can_vote(False, 0.2, wider.swayam_oversold, wider.swayam_overbought)
    assert voting_views(wider) == (len(members), len(members))


def test_n_voting_counts_the_members_that_can_reach_the_cut(bank):
    frames, daily = bank
    can = pd.concat([f["Can_Vote"].astype(int).rename(k) for k, f in frames.items()], axis=1)
    can.index = [d.date() for d in can.index]
    reported = can.notna().sum(axis=1)
    assert (daily["Total_Analyzed"] == reported.reindex(daily.index)).all()
    assert (daily["N_Voting"] == can.sum(axis=1).reindex(daily.index)).all()
    n_vote, n_bank = voting_views(CFG)
    assert daily["N_Voting"].max() == n_vote < n_bank == daily["Total_Analyzed"].max()
    # The voters are the only views that ever cast a vote.
    voted = [k for k, f in frames.items() if (f["Condition"] != "Neutral").any()]
    assert voted and all(bool(frames[k]["Can_Vote"].iloc[0]) for k in voted)


# ─── Causality: equal weights, nothing learned, nothing revised ──────────────

def test_equal_weight_breadth_is_unchanged_under_ohlcv_truncation(inputs, bank):
    bars, drivers = inputs
    _f, full = bank
    T = bars.index[-251]
    part = aggregate_views(_frames(bars.loc[:T], drivers.loc[:T]))
    assert len(part) and part.index[-1] == T.date()
    assert np.array_equal(part.to_numpy(dtype=float), full.loc[part.index].to_numpy(dtype=float))


# ─── The composer reads a carried row as ×1.00 ──────────────────────────────

def test_a_carried_row_is_x1_in_the_composer():
    n = 80
    idx = pd.bdate_range("2024-01-01", periods=n, name="Date")
    fvo_ts = pd.DataFrame({"Price": 100.0, "FVO": -1.5, "PctMispricing": -0.02, "Valid": True,
                           "Provisional": False, "WithheldReason": "", "MRProb": 0.5,
                           "MulaKappa": 0.0}, index=idx)
    os_ = pd.Series(80.0, index=idx)                  # a strong oversold read every session
    ob_ = pd.Series(0.0, index=idx)
    native = np.ones(n, dtype=bool)
    native[-1] = native[-5] = False                   # two sessions carried from the day before
    sw = pd.DataFrame({"Oversold_Pct": os_, "Overbought_Pct": ob_,
                       "Lean": [breadth_lean(a, b) for a, b in zip(os_, ob_)],
                       "_Native": native}, index=idx)
    inp = gather_inputs(fvo_ts, sw, instrument=instrument("Gold"), h=H, today=idx[-1])
    grade = pd.DataFrame({"ic": 0.2, "se": 0.05, "n_win": 100, "grid_date": idx, "prior_mu": np.nan,
                          "prior_sd": np.nan, "prior_scope": "none", "prior_k": 0, "ic_post": 0.2,
                          "se_post": 0.05, "t_post": 4.0, "w_own": 1.0, "strength": 1.0}, index=idx)
    grades = {"claim": grade, **{f"sizer:{s.key}": grade for s in SIZERS}}
    ct = compose_series(inp, grades, tradeable=True, sizers_enabled=True)
    assert (ct["swayam_beta"] == 1.0).all() and (ct["d"] == 1).all()   # skilled, and a claim
    carried = ~native
    assert (ct.loc[carried, "swayam_z"] == 0.0).all() and (ct.loc[carried, "swayam_factor"] == 1.0).all()
    # The carried session keeps its reading on the record — it is just not acted on.
    assert (ct.loc[carried, "swayam_lean"] == 0.8).all() and not ct.loc[carried, "swayam_native"].any()
    assert (ct.loc[~carried, "swayam_factor"] == np.exp(0.8)).all()     # a native read does act
