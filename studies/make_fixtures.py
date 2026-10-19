"""
Tattva — build the offline test fixtures and the golden Mūla baseline.
तत्त्व (Tattva) — "Principle / Essence"

    python -m studies.make_fixtures --state-dir DIR   # panel + OHLCV fixtures
    python -m studies.make_fixtures --golden          # golden_mula_*.csv.gz

The first form extracts, from pickled 2.8.0 session states (one per target,
each a dict of ``st.session_state``), the shared raw panel (key ``data``,
2017-10-19 … 2026-10-16) and the Gold and USD/INR OHLCV (the Swayam fetch
cache, or one yfinance fetch at build time when a state lacks it). The four
audited states share one panel, so the first one found is used.

The second form fits ``MulaEngine`` on the PREPARED Gold and USD/INR fixture
panels — exactly as the app prepares them (registered cross-section, epoch,
target-printed spine) — and writes the published columns the engine surgery
of later packages must not move. ``tests/test_golden_mula.py`` refits and
compares at |Δ| < 1e-12.

Everything is written as gzip CSV with Python's shortest round-trip float
repr, so a fixture reads back bit-identical.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# One BLAS thread, as app.py runs the engine (set before numpy loads): the
# golden baseline is compared at 1e-12, and a reduction split across a
# machine-dependent number of threads need not round the same way twice.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
PANEL_FILE = "panel.csv.gz"
#: target → (OHLCV fixture, golden baseline, pickled-state file stem)
TARGETS: dict[str, tuple[str, str, str]] = {
    "Gold": ("ohlcv_gold.csv.gz", "golden_mula_gold.csv.gz", "gold"),
    "USD/INR": ("ohlcv_usdinr.csv.gz", "golden_mula_usdinr.csv.gz", "usdinr"),
}
#: The published Mūla columns the golden baseline pins.
GOLDEN_COLUMNS: tuple[str, ...] = (
    "FVO", "FairValue", "FairValueLo", "FairValueHi", "PctMispricing", "MRProb",
    "MulaKappa", "MulaHalfLife", "MulaDriftPct", "MulaSdPct", "GapHalfLife",
    "Valid", "Provisional",
)
_BOOL_COLUMNS = ("Valid", "Provisional")


# ─── Readers (shared with tests/conftest.py and the tests) ──────────────────

def load_panel() -> pd.DataFrame:
    """The fixture panel exactly as ``data.fetcher.fetch_commodity_dataset``
    shapes it: a ``DATE`` column, one float column per series, a RangeIndex,
    and the ``panel_fingerprint`` attr recomputed on the content."""
    from data.fetcher import _panel_fingerprint

    df = pd.read_csv(FIXTURES / PANEL_FILE, parse_dates=["DATE"], float_precision="round_trip")
    by_date = df.set_index("DATE")
    df.attrs = {"source": "fixture", "stale_backfills": {},
                "panel_columns": sorted(by_date.columns),
                "panel_fingerprint": _panel_fingerprint(by_date)}
    return df


def load_ohlcv(target: str) -> pd.DataFrame:
    """The target's raw fixture OHLCV (date-indexed, yfinance columns)."""
    ohlcv = pd.read_csv(FIXTURES / TARGETS[target][0], index_col=0, parse_dates=True,
                        float_precision="round_trip")
    ohlcv.index.name = "Date"
    return ohlcv


def load_golden(target: str) -> pd.DataFrame:
    g = pd.read_csv(FIXTURES / TARGETS[target][1], index_col=0, parse_dates=True,
                    float_precision="round_trip")
    for c in _BOOL_COLUMNS:
        g[c] = g[c].astype(bool)
    return g


def fit_fixture(target: str, panel: pd.DataFrame | None = None):
    """Prepare the fixture panel for ``target`` the way app.py does and fit Mūla.

    Returns ``(engine, prep)``.
    """
    from core.config import ANALYSIS_EPOCH, MIN_DATA_POINTS, get_instrument_config
    from core.instruments import instrument, predictor_universe
    from data.calendars import resolve_exchange, session_bars
    from data.prep import prepare_panel
    from engines.mula import MulaEngine

    panel = load_panel() if panel is None else panel
    cfg, inst = get_instrument_config(target), instrument(target)
    numeric = panel.select_dtypes(include=[np.number]).columns.tolist()
    bars = session_bars(load_ohlcv(target), inst.ticker)
    prep = prepare_panel(panel, target, predictor_universe(target, numeric), "DATE",
                         epoch=ANALYSIS_EPOCH, min_target_rows=MIN_DATA_POINTS, bars=bars)
    engine = MulaEngine()
    engine.fit(prep.data[target].astype(float), prep.data[prep.features].astype(float),
               feature_names=prep.features, config=cfg,
               exchange=resolve_exchange(inst.ticker), epoch=ANALYSIS_EPOCH,
               fingerprint=prep.fingerprint)
    return engine, prep


def golden_frame(engine, prep) -> pd.DataFrame:
    """The golden columns of a fit, indexed by session date."""
    out = engine.ts_data[list(GOLDEN_COLUMNS)].copy()
    out.index = pd.DatetimeIndex(prep.data.index, name="Date")
    for c in _BOOL_COLUMNS:
        out[c] = out[c].astype(bool)
    return out


# ─── Builders ────────────────────────────────────────────────────────────────

def _write_csv(frame: pd.DataFrame, name: str, **kw) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    # float_format=None → repr, the shortest string that round-trips exactly.
    frame.to_csv(path, compression={"method": "gzip", "mtime": 0}, **kw)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.2f} MB)")


def build_fixtures(state_dir: Path) -> None:
    states = {stem: pickle.load(open(state_dir / f"{stem}.pkl", "rb"))
              for stem in sorted({t[2] for t in TARGETS.values()})
              if (state_dir / f"{stem}.pkl").exists()}
    if not states:
        raise SystemExit(f"no pickled states under {state_dir}")
    panel = next(iter(states.values()))["data"]
    _write_csv(panel, PANEL_FILE, index=False)

    for target, (ohlcv_file, _golden, stem) in TARGETS.items():
        cache = (states.get(stem) or {}).get("_swayam_fetch_cache") or {}
        ohlcv = cache.get("target_ohlcv")
        if ohlcv is None or ohlcv.empty:
            from core.instruments import instrument
            from data.fetcher import fetch_ohlcv
            ticker = instrument(target).ticker
            dates = pd.to_datetime(panel["DATE"])
            ohlcv = fetch_ohlcv([ticker], dates.min(), dates.max()).get(ticker)
            if ohlcv is None or ohlcv.empty:
                raise SystemExit(f"no OHLCV for {target} in the state or from yfinance")
            print(f"{target}: OHLCV fetched from yfinance ({len(ohlcv)} bars)")
        ohlcv = ohlcv.copy()
        ohlcv.index = pd.DatetimeIndex(ohlcv.index, name="Date")
        _write_csv(ohlcv, ohlcv_file)


def build_golden() -> None:
    panel = load_panel()
    for target, (_ohlcv, golden_file, _stem) in TARGETS.items():
        engine, prep = fit_fixture(target, panel)
        g = golden_frame(engine, prep)
        print(f"{target}: {len(g)} rows from {prep.start:%Y-%m-%d}, "
              f"{int(g['Valid'].sum())} Valid, fingerprint {prep.fingerprint}")
        _write_csv(g, golden_file)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-dir", type=Path,
                    help="directory of pickled 2.8.0 session states (gold.pkl, usdinr.pkl, …)")
    ap.add_argument("--golden", action="store_true",
                    help="fit Mūla on the fixture panels and write the golden baselines")
    args = ap.parse_args(argv)
    if not args.state_dir and not args.golden:
        ap.error("give --state-dir, --golden, or both")
    if args.state_dir:
        build_fixtures(args.state_dir)
    if args.golden:
        build_golden()


if __name__ == "__main__":
    main()
