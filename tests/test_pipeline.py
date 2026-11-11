"""The pipeline runs headless, end to end, and refuses data it cannot use."""
import numpy as np
import pandas as pd
import pytest

from core import config
from core.config import InstrumentConfig
from pipeline.prep import PrepError, prepare
from pipeline.run import run_target

from test_causality import _panel

TARGET = "Synthetic Target"


@pytest.fixture()
def registered(monkeypatch):
    monkeypatch.setitem(config.ALL_TARGETS, TARGET, None)
    monkeypatch.setitem(config.INSTRUMENT_CONFIGS, TARGET, InstrumentConfig())


def _frame(T=1700, **kw):
    tgt, expl = _panel(T=T, **kw)
    df = expl.copy()
    df[TARGET] = tgt
    df.insert(0, "DATE", df.index)
    return df.reset_index(drop=True)


def test_run_target_end_to_end(registered):
    res = run_target(_frame(), TARGET, today=pd.Timestamp("2030-01-01"))
    assert np.isfinite(res.signal["fvo"])
    assert res.fvo_ts["FVO"].notna().sum() > 1000
    assert res.edges["mula"]["n"] > 50
    assert res.swayam_daily.empty            # no OHLCV was supplied


def test_prepare_refuses_short_history(registered):
    with pytest.raises(PrepError):
        prepare(_frame(T=900), TARGET)


def test_prepare_refuses_non_positive_target(registered):
    df = _frame()
    df.loc[df.index[-5], TARGET] = -1.0
    with pytest.raises(PrepError):
        prepare(df, TARGET)


def test_prepare_refuses_missing_target(registered):
    with pytest.raises(PrepError):
        prepare(_frame().drop(columns=[TARGET]), TARGET)
