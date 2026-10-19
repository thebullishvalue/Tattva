"""Tattva test suite — shared setup.

Run from the repository root:  python -m pytest tests -q

Every test here runs offline. Most run on synthetic data: the properties they
assert (causality, sign conventions, the ECM's horizon arithmetic, the edge
scorer's recovery of a planted signal, the conviction chain's rules) are
properties of the code, not of any market. The rest read the committed
fixtures under tests/fixtures (built by ``python -m studies.make_fixtures``):
the 2.8.0 shared panel and the Gold / USD/INR OHLCV, served to the app through
``offline_fetch`` in place of every network fetch.
"""
import os
import sys
import warnings

# One BLAS thread, as app.py runs the engine — set before numpy loads, so the
# golden comparison (tests/test_golden_mula.py) does not depend on core count.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "tattva", "numba"))
warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the tests marked slow (extra full-panel fits)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: an extra full-panel fit; runs only with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def install_offline_fetch(mp) -> None:
    """Point every network fetch in ``data.fetcher`` at the committed fixtures.

    ``mp`` is a ``pytest.MonkeyPatch``. The app imports these names from
    ``data.fetcher`` on every script run, so patching the module is enough.
    Each stub honours its (start, end) window, as the real fetch does.
    """
    from studies.make_fixtures import TARGETS, load_ohlcv, load_panel
    from core.instruments import instrument

    panel = load_panel()
    ohlcv = {instrument(t).ticker: load_ohlcv(t) for t in TARGETS}

    def _window(idx, start, end):
        idx = pd.DatetimeIndex(idx)
        return (idx >= pd.Timestamp(start).normalize()) & (idx <= pd.Timestamp(end))

    def fetch_commodity_dataset(start_date, end_date, force=False):
        df = panel[_window(panel["DATE"], start_date, end_date)].reset_index(drop=True)
        df.attrs = dict(panel.attrs)
        return df, None

    def fetch_ohlcv(symbols, start_date, end_date, force=False):
        return {s: o[_window(o.index, start_date, end_date)]
                for s, o in ohlcv.items() if s in symbols}

    def fetch_stock_target_series(ticker, start_date, end_date, force=False):
        return None

    def fetch_sheet_series(name, start_date=None, end_date=None, force=False):
        return None

    for name, fn in (("fetch_commodity_dataset", fetch_commodity_dataset),
                     ("fetch_ohlcv", fetch_ohlcv),
                     ("fetch_stock_target_series", fetch_stock_target_series),
                     ("fetch_sheet_series", fetch_sheet_series)):
        mp.setattr(f"data.fetcher.{name}", fn)


@pytest.fixture
def offline_fetch(monkeypatch):
    """Serve data.fetcher's network fetches from tests/fixtures for one test."""
    install_offline_fetch(monkeypatch)


# ─── The app, run offline under AppTest ──────────────────────────────────────
# st.navigation is replaced by a stand-in that records every page the app
# defines and runs the one named in session_state["_smoke_page"], so a test
# can visit each page without knowing the list in advance.
_APP_WRAPPER = '''
import runpy
import streamlit as st


class _Nav:
    def __init__(self, pages):
        self.pages = [p for section in pages.values() for p in section]

    def run(self):
        st.session_state["_smoke_pages"] = [p.title for p in self.pages]
        want = st.session_state.get("_smoke_page") or self.pages[0].title
        for p in self.pages:
            if p.title == want:
                return p._page()
        raise RuntimeError(f"no page {want!r}")


st.navigation = lambda pages, position="sidebar": _Nav(pages)
runpy.run_path(APP_PATH, run_name="__main__")
'''


def run_app_offline(target: str, category: str):
    """Cold start → select ``target`` → Run Analysis, on the fixtures.

    Returns the AppTest once the pipeline has stored its bundle. The caller
    must have installed the offline fetch.
    """
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1 import element_tree as _et

    # AppTest reads a single-select segmented control's value as a bare
    # string where its own widget-state code expects a list.
    if not getattr(_et.ButtonGroup, "_tattva_patched", False):
        _orig = _et.ButtonGroup.value.fget

        def _value(self):
            v = _orig(self)
            return [v] if isinstance(v, str) else ([] if v is None else v)
        _et.ButtonGroup.value = property(_value)
        _et.ButtonGroup._tattva_patched = True

    script = _APP_WRAPPER.replace("APP_PATH", repr(os.path.join(ROOT, "app.py")))
    at = AppTest.from_string(script, default_timeout=1200)
    at.run()
    at.selectbox(key="target_category").set_value(category).run()
    at.selectbox(key="target_select").set_value(target).run()
    [b for b in at.button if b.label == "Run Analysis"][0].click().run()
    for _ in range(4):
        if "engine" in at.session_state and "edges" in at.session_state:
            break
        at.run()
    return at


@pytest.fixture(scope="session")
def gold_app():
    """One offline run of the whole app on Gold, shared by the tests that read it."""
    with pytest.MonkeyPatch.context() as mp:
        install_offline_fetch(mp)
        yield run_app_offline("Gold", "Commodities")
