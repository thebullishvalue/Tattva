"""The prepared panel does not depend on the date it is prepared (data.prep).

The prep this replaced started the history at the youngest surviving
feature's listing date, so the start — and with it every published value of a
permanent-memory engine — moved with the run date. Here: run as of T and as
of T+k, the rows dated ≤ T are identical; the start is max(epoch, the target's
first print); a young feature cannot move it; and a session the target did
not print is not on the spine.
"""
import numpy as np
import pandas as pd
import pytest

from core.config import ANALYSIS_EPOCH, MIN_DATA_POINTS
from core.instruments import instrument, predictor_universe
from data.calendars import session_bars, session_mask
from data.prep import PrepError, history_fingerprint, prepare_panel
from studies.make_fixtures import load_ohlcv, load_panel

GOLD = "Gold"


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def bars():
    return session_bars(load_ohlcv(GOLD), instrument(GOLD).ticker)


def _feats(panel, target=GOLD):
    return predictor_universe(target, panel.select_dtypes(include=[np.number]).columns.tolist())


def _prep_asof(panel, bars, asof, *, epoch=ANALYSIS_EPOCH, target=GOLD, feats=None,
               min_rows=MIN_DATA_POINTS):
    """Prepare as a run on ``asof`` would have: only data dated ≤ asof exists."""
    raw = panel[pd.to_datetime(panel["DATE"]) <= asof].reset_index(drop=True)
    b = bars[bars.index <= asof] if bars is not None else None
    return prepare_panel(raw, target, feats if feats is not None else _feats(panel, target),
                         "DATE", epoch=epoch, min_target_rows=min_rows, bars=b)


@pytest.mark.parametrize("k", [1, 20, 250])
def test_rows_through_T_do_not_depend_on_the_run_date(panel, bars, k):
    dates = pd.DatetimeIndex(pd.to_datetime(panel["DATE"]))
    T = dates[-1 - 260]
    T_k = dates[dates.get_loc(T) + k]
    early, late = _prep_asof(panel, bars, T), _prep_asof(panel, bars, T_k)
    assert early.start == late.start
    assert early.features == late.features
    pd.testing.assert_frame_equal(early.data, late.data.loc[:T], check_exact=True)
    cols = [GOLD, *early.features]
    assert history_fingerprint(early.data, cols, T) == history_fingerprint(late.data, cols, T)
    assert len(late.data.loc[late.data.index > T]) > 0


def test_start_is_the_later_of_the_epoch_and_the_first_target_print(panel, bars):
    d = pd.to_datetime(panel["DATE"])
    # Gold printed from the first fixture row, after the epoch: start = first print.
    first_gold = d[panel[GOLD].notna()].min()
    assert ANALYSIS_EPOCH < first_gold
    assert _prep_asof(panel, bars, d.max()).start == first_gold
    # An epoch after the first print: start = the first printed session from it.
    late_epoch = pd.Timestamp("2019-03-16")                     # a Saturday
    p = _prep_asof(panel, bars, d.max(), epoch=late_epoch)
    assert p.start == pd.Timestamp("2019-03-18")
    assert p.data.index.min() >= late_epoch
    # A target first printed after the epoch starts at its own first print.
    sol = _prep_asof(panel, None, d.max(), target="Solana")
    assert sol.start == d[panel["Solana"].notna()].min() == pd.Timestamp("2020-04-10")


def test_a_young_feature_does_not_move_the_start(panel, bars):
    d = pd.to_datetime(panel["DATE"])
    T = d.iloc[-1]
    young = pd.Series(np.nan, index=panel.index)
    listed = d.index[-200]
    young.loc[listed:] = np.linspace(100.0, 120.0, len(young.loc[listed:]))
    with_young = panel.assign(**{"A Young ETF": young})
    feats = _feats(panel) + ["A Young ETF"]
    base = _prep_asof(panel, bars, T)
    p = _prep_asof(with_young, bars, T, feats=feats)
    assert p.start == base.start
    assert "A Young ETF" in p.features
    pd.testing.assert_frame_equal(p.data[base.data.columns], base.data, check_exact=True)
    # Its NaN head is left for the engine's print floor, never back-filled.
    assert p.data["A Young ETF"].loc[:d.iloc[listed - 1]].isna().all()


def test_a_session_the_target_did_not_print_is_not_on_the_spine(panel, bars):
    d = pd.DatetimeIndex(pd.to_datetime(panel["DATE"]))
    p = _prep_asof(panel, bars, d.max())
    on_cme = d[session_mask(instrument(GOLD).ticker, d)]
    closed = on_cme[~on_cme.isin(bars.index)]          # CME open, GC=F has no bar
    assert pd.Timestamp("2021-07-05") in closed          # US Independence Day (observed)
    assert len(closed) == p.report["target_not_printed"] > 0
    assert not p.data.index.isin(closed).any()
    assert p.data.index.isin(bars.index).all()


def test_too_little_target_history_fails_loudly(panel, bars):
    with pytest.raises(PrepError):
        _prep_asof(panel, bars, pd.to_datetime(panel["DATE"]).max(),
                   epoch=pd.Timestamp("2025-01-01"))
