"""C1 one horizon, C2 one as-of session — and the spine they are counted on.

C1: h is resolved once, from the instrument's config, and every consumer
carries that same h: the engine, the run's bundle, every grade. A config that
cannot carry it is refused at construction.

C2: every verdict input is read at d*, Mūla's as-of session. When the newest
row's valuation is withheld but Swayam has a native row on it, the command bar
and the verdict (call_ts at d*) still read d* — not the frame's last row.

The spine: every Valid Gold row is a session GC=F printed; the CME-open US
holidays with no Gold bar are gone. A Provisional row is read from the ECM's
state at the last Valid row, never absorbed into it (U82).
"""
import numpy as np
import pandas as pd
import pytest

import app
from core.config import InstrumentConfig, get_instrument_config
from core.instruments import instrument
from core.semantics import breadth_lean
from data.calendars import session_mask
from engines.mula import MulaEngine
from pipeline.assemble import assemble
from studies.make_fixtures import load_ohlcv, load_panel


def _panel(T=700, n=40, k=3, seed=5):
    """A cointegrated synthetic panel: n instruments on k integrated factors."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    f = np.cumsum(rng.normal(0, 0.01, (T, k)), axis=0)
    x = f @ rng.normal(0, 1, (n, k)).T + np.cumsum(rng.normal(0, 0.004, (T, n)), axis=0) * 0.3
    expl = pd.DataFrame(100 * np.exp(x), index=idx, columns=[f"I{j:02d}" for j in range(n)])
    gap = np.zeros(T)
    for t in range(1, T):
        gap[t] = 0.93 * gap[t - 1] + 0.008 * rng.standard_normal()
    tgt = pd.Series(100 * np.exp(f @ np.array([0.6, -0.3, 0.2]) + gap), index=idx, name="T")
    return tgt, expl


def _fit(tgt, expl):
    eng = MulaEngine()
    eng.fit(tgt, expl, feature_names=list(expl.columns), config=InstrumentConfig())
    return eng


# ─── C1 ──────────────────────────────────────────────────────────────────────

def test_one_horizon_from_config_to_bundle(gold_app):
    ss = gold_app.session_state
    h = get_instrument_config("Gold").forecast_horizon
    assert ss["engine"].horizon == h == ss["horizon"]
    assert all(e["horizon"] == h for e in ss["edges"].values() if e)


def test_a_horizon_the_config_cannot_carry_is_refused():
    # Any whole number of sessions >= 1 is a horizon (no second horizon list
    # to belong to since the Precedent term structure went); nothing else is.
    assert InstrumentConfig(forecast_horizon=7).forecast_horizon == 7
    for bad in (10.0, 0, -5, True):
        with pytest.raises(AssertionError):
            InstrumentConfig(forecast_horizon=bad)


# ─── C2 ──────────────────────────────────────────────────────────────────────

def test_a_withheld_newest_row_reads_every_input_at_d_star():
    tgt, expl = _panel()
    expl.iloc[-1] = expl.iloc[-2]          # nothing printed: the gate withholds the newest row
    eng = _fit(tgt, expl)
    sig = eng.get_current_signal()
    dates = tgt.index
    assert np.isnan(eng.ts_data["FVO"].iloc[-1]) and sig["asof_row_withheld"]
    d_star = sig["asof"]
    assert d_star == eng.asof == dates[-2]

    # Swayam carries a NATIVE row after d*, with values that differ from
    # d*'s — a positional read would take it.
    os_, ob_ = np.linspace(10, 60, len(dates)), np.linspace(50, 5, len(dates))
    sw = pd.DataFrame({"Oversold_Pct": os_, "Overbought_Pct": ob_,
                       "Lean": [breadth_lean(a, b) for a, b in zip(os_, ob_)],
                       "_Native": True}, index=dates)
    fvo_ts = eng.ts_data.copy()
    fvo_ts.index = dates
    asm = assemble(fvo_ts, sw, instrument=instrument("Gold"), cfg=InstrumentConfig(),
                   asof=d_star, today=dates[-1], priors=None)
    verdict = asm.verdict
    lean = sw["Lean"]
    assert verdict["asof"] == d_star
    assert asm.call_ts.loc[d_star, "swayam_lean"] == lean.loc[d_star] != lean.iloc[-1]
    assert verdict["action"] == asm.call_ts.loc[d_star, "action"]
    assert asm.call_ts.loc[dates[-1], "action"] == "NO CALL"      # the withheld row issues nothing

    bar = app._asof_status(sig, "GC=F", dates[-1] + pd.Timedelta(days=3))
    assert bar["as_of"] == d_star == bar["valued_as_of"]


def test_the_bar_reads_d_star_on_the_live_run(gold_app):
    ss = gold_app.session_state
    fvo = ss["fvo_ts"]
    published = fvo.index[np.isfinite(fvo["FVO"].to_numpy(dtype=float))]
    assert ss["asof"] == published.max() == ss["engine"].get_current_signal()["asof"]
    bar = [m.value for m in gold_app.markdown if 'class="command-bar' in m.value]
    assert bar and f"{ss['asof']:%d %b %Y}" in bar[-1]


# ─── The spine and the provisional read ─────────────────────────────────────

def test_every_valid_gold_row_has_a_target_print(gold_app):
    fvo = gold_app.session_state["fvo_ts"]
    bars = load_ohlcv("Gold")
    printed = pd.DatetimeIndex(bars.index[bars["Close"].notna()])
    assert fvo.index[fvo["Valid"].astype(bool)].isin(printed).all()

    d = pd.DatetimeIndex(pd.to_datetime(load_panel()["DATE"]))
    on_cme = d[session_mask("GC=F", d)]
    closed = on_cme[~on_cme.isin(printed)]            # CME open, no GC=F bar
    assert not fvo.index.isin(closed).any()
    # 2.8.0's 2020-06-01 … 2026-10-16 spine carried 41 CME-open US holidays
    # with no Gold bar (35 of them Valid, a carried price valued as a session).
    # All 41 are gone. Six genuine unchanged closes — a bar, with volume, at
    # the previous close — are prints and stay.
    assert len(closed[(closed >= "2020-06-01") & (closed <= "2026-10-16")]) == 41
    assert fvo.index.isin(pd.DatetimeIndex(["2022-10-06", "2025-12-05"])).sum() == 2


def test_a_provisional_row_is_read_not_absorbed(monkeypatch):
    tgt, expl = _panel()
    expl.iloc[-1] = expl.iloc[-2]          # the newest session is still filling …
    monkeypatch.setattr("data.calendars.session_today", lambda exchange: tgt.index[-1])

    def fit_with_close(px):
        t = tgt.copy()
        t.iloc[-1] = px
        return _fit(t, expl).ts_data

    a = fit_with_close(tgt.iloc[-2] * 1.01)
    b = fit_with_close(tgt.iloc[-2] * 0.97)
    assert a["Provisional"].iloc[-1] and not a["Valid"].iloc[-1]   # … and is published
    # Nothing the read learns can depend on the provisional close.
    for col in ("MulaKappa", "MulaRevProb", "MulaHalfLife", "MulaSdPct",
                "WValuation", "WMomentum", "WFull"):
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        assert np.isfinite(x[-1]) or col == "MulaHalfLife", col
        assert np.array_equal(x, y, equal_nan=True), col
    # Its own gap still sets where the h-session projection starts.
    assert a["MulaDriftPct"].iloc[-1] != b["MulaDriftPct"].iloc[-1]
