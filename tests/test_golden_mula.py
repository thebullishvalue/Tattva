"""The golden Mūla baseline: refitting the fixture panels reproduces it exactly.

tests/fixtures/golden_mula_*.csv.gz were written by
``python -m studies.make_fixtures --golden`` once the history was pinned
(epoch, registered cross-section, target-printed spine) and before any later
engine surgery. Every later change to the engine must leave the published
fair value, its gap, its band and the error-correction read where they are,
to |Δ| < 1e-12 on every settled (non-Provisional) row — or say, in its own
commit, that it moved them and why.
"""
import numpy as np
import pytest

from studies.make_fixtures import GOLDEN_COLUMNS, fit_fixture, golden_frame, load_golden


def _assert_reproduces(target: str) -> None:
    engine, prep = fit_fixture(target)
    got, want = golden_frame(engine, prep), load_golden(target)
    assert got.index.equals(want.index), "the prepared spine moved"
    settled = ~want["Provisional"].to_numpy(dtype=bool)
    for col in GOLDEN_COLUMNS:
        a = got[col].to_numpy(dtype=float)[settled]
        b = want[col].to_numpy(dtype=float)[settled]
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col}: publication pattern moved"
        both = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[both] - b[both]), initial=0.0))
        assert diff < 1e-12, f"{col}: max |Δ| {diff:.3g}"


def test_gold_reproduces_the_golden_baseline():
    _assert_reproduces("Gold")


@pytest.mark.slow
def test_usdinr_reproduces_the_golden_baseline():
    _assert_reproduces("USD/INR")
