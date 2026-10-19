"""The instrument registry (core.instruments): one definition of what each target is.

Twin pairs are HARD-CODED here, not read back from the registry: a test that
recomputed them from the code under test could not catch the code dropping one.
"""
import numpy as np
import pytest

import core.config as config
import core.instruments as instruments
from core.config import CATALOGUE_TARGETS, TARGET_CATEGORIES, get_instrument_config
from core.instruments import (INSTRUMENTS, PREDICTOR_UNIVERSE, SHEET_SERIES, instrument,
                              predictor_universe, ticker_class_map)
from engines.mula.blocks import classify
from studies.make_fixtures import load_panel

#: (USD-listed member, local-currency member) — together they rebuild the FX rate.
HARD_TWIN_PAIRS = {
    "USD/INR": [("India Equity", "Nifty 50"), ("India Gov Bonds (LSE Proxy)", "Nifty 50"),
                ("India Equity", "Nifty 500"), ("India Gov Bonds (LSE Proxy)", "Nifty 500")],
    "Dollar Index": [("Eurozone Equity (EZU)", "Euro Stoxx 50"),
                     ("Japan Equity (EWJ)", "Nikkei 225"),
                     ("UK Equity (EWU)", "FTSE 100 (UK)")],
}


@pytest.fixture(scope="module")
def panel_cols():
    p = load_panel()
    return p.select_dtypes(include=[np.number]).columns.tolist()


def test_no_fx_target_holds_both_members_of_a_twin_pair(panel_cols):
    fx = [t for t, i in INSTRUMENTS.items() if i.asset_class == "fx"]
    assert set(HARD_TWIN_PAIRS) <= set(fx)
    for target, pairs in HARD_TWIN_PAIRS.items():
        feats = set(predictor_universe(target, panel_cols))
        for usd, local in pairs:
            assert usd in panel_cols and local in panel_cols
            assert not (usd in feats and local in feats), f"{target} holds {usd} + {local}"
            # The dropped side is the USD fund; the local index stays a flow driver.
            assert usd not in feats and local in feats


def test_iind_is_india_equity_not_rates(panel_cols):
    assert ticker_class_map()["IIND.L"] == "Equity"
    assert classify("India Gov Bonds (LSE Proxy)") == "Equity"
    assert "Rates" != classify("India Gov Bonds (LSE Proxy)")
    # Excluded wherever INDA is: from every India index's cross-section.
    for target, inst in INSTRUMENTS.items():
        if inst.asset_class == "india_index":
            feats = predictor_universe(target, panel_cols)
            assert "India Equity" not in feats and "India Gov Bonds (LSE Proxy)" not in feats


def test_no_sheet_series_is_a_predictor(panel_cols):
    assert SHEET_SERIES == {"Jeera", "Nifty 50 - PE"}
    cols = panel_cols + sorted(SHEET_SERIES)
    for target in INSTRUMENTS:
        assert not SHEET_SERIES & set(predictor_universe(target, cols)), target
    assert not SHEET_SERIES & set(PREDICTOR_UNIVERSE)


def test_every_catalogue_target_has_an_instrument():
    listed = {t for names in TARGET_CATEGORIES.values() for t in names}
    for target in set(CATALOGUE_TARGETS) | listed:
        inst = instrument(target)
        assert inst.name == target and inst.ticker == CATALOGUE_TARGETS[target]
        get_instrument_config(target)
    pe, jeera = instrument("Nifty 50 - PE"), instrument("Jeera")
    assert pe.asset_class == "sheet" and not pe.tradeable
    assert jeera.asset_class == "commodity" and not jeera.has_ohlcv and jeera.tradeable
    assert instrument("USD/INR").currency == "INR"


def test_an_unregistered_target_raises(panel_cols):
    with pytest.raises(KeyError):
        instrument("NOT A TARGET")
    with pytest.raises(KeyError):
        predictor_universe("NOT A TARGET", panel_cols)
    with pytest.raises(KeyError):
        get_instrument_config("NOT A TARGET")


def test_a_registered_stock_gets_an_instrument(monkeypatch):
    for mod, name in ((config, "ALL_TARGETS"), (config, "STOCK_TARGET_MARKETS"),
                      (config, "TARGET_EXCLUDED_PREDICTORS"), (config, "INSTRUMENT_CONFIGS"),
                      (instruments, "INSTRUMENTS")):
        monkeypatch.setattr(mod, name, dict(getattr(mod, name)))
    config.register_stock_target("ZZTEST (US)", "ZZTEST", "us")
    inst = instruments.instrument("ZZTEST (US)")
    assert (inst.asset_class, inst.currency, inst.tradeable, inst.has_ohlcv) == (
        "stock_us", "USD", True, True)
    assert config.get_instrument_config("ZZTEST (US)").forecast_horizon >= 1
