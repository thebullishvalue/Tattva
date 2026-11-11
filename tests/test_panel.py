"""The panel the engine reads: ragged, alive, and free of duplicate instruments."""
import numpy as np
import pandas as pd
import pytest

from core import config
from core.config import InstrumentConfig
from engines.mula import MulaEngine
from pipeline.prep import PrepError, prepare

from test_causality import COLS, _panel
from test_pipeline import TARGET, _frame, registered  # noqa: F401 - fixture

TODAY = pd.Timestamp("2030-01-01")


def _fit(tgt, expl):
    eng = MulaEngine()
    eng.fit(tgt, expl, feature_names=list(expl.columns), config=InstrumentConfig(), today=TODAY)
    return eng.ts_data


def test_late_predictor_does_not_shorten_history(registered):  # noqa: F811
    df = _frame()
    late = [c for c in df.columns if c not in ("DATE", TARGET)][-1]
    df.loc[: 599, late] = np.nan
    prep = prepare(df, TARGET)
    assert len(prep.price) == len(df)                 # the target keeps every row
    assert late in prep.features                      # and the late predictor stays in
    ts = _fit(prep.price, prep.panel)
    first = int(np.flatnonzero(ts["FVO"].notna().to_numpy())[0])
    assert first < 400                                # published long before row 600


def test_late_predictor_never_moves_earlier_rows():
    tgt, expl = _panel()
    late = expl.columns[-1]
    ragged = expl.copy()
    ragged.loc[ragged.index[:900], late] = np.nan
    with_late = _fit(tgt, ragged)
    without = _fit(tgt, expl.drop(columns=[late]))
    # The late instrument is admitted only after its own print floor, so until
    # it first prints the two fits read the same cross-section.
    for col in ("FairValue", "FVO", "XSConsistency"):
        a = with_late[col].to_numpy(dtype=float)[:900]
        b = without[col].to_numpy(dtype=float)[:900]
        assert (np.isfinite(a) == np.isfinite(b)).all(), col
        ok = np.isfinite(a)
        assert np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-9), col


def test_truncation_holds_on_a_ragged_panel():
    tgt, expl = _panel()
    expl = expl.copy()
    expl.loc[expl.index[:500], expl.columns[-3:]] = np.nan
    full = _fit(tgt, expl)
    part = _fit(tgt.iloc[:-200], expl.iloc[:-200])
    n = len(part)
    for col in COLS:
        a = full[col].to_numpy(dtype=float)[:n]
        b = part[col].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all(), col
        assert np.array_equal(a[both], b[both]), col


def test_dead_target_is_refused(registered):  # noqa: F811
    df = _frame()
    df.loc[df.index[-400]:, TARGET] = float(df[TARGET].iloc[-400])
    with pytest.raises(PrepError, match="not live"):
        prepare(df, TARGET)


def test_print_mask_round_trips():
    from data.fetcher import pack_mask, unpack_mask
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=37)
    m = pd.DataFrame(rng.random((37, 5)) > 0.4, index=idx, columns=list("abcde"))
    assert unpack_mask(pack_mask(m), idx).equals(m)
    assert unpack_mask(pack_mask(m), idx[:-1]) is None      # misaligned -> infer instead


INVERSE_PAIRS = (("INR=X", "INRUSD=X"), ("JPY=X", "JPYUSD=X"), ("KRW=X", "KRWUSD=X"),
                 ("CNY=X", "CNYUSD=X"), ("UUP", "UDN"))


def test_no_inverse_pairs_in_the_universe():
    tickers = set(config.GLOBAL_MACRO_MAP.values()) | set(config.MACRO_SYMBOLS_YF.values())
    for a, b in INVERSE_PAIRS:
        assert not ({a, b} <= tickers), f"{a} and {b} are the same instrument"
    inr_legs = [t for t in tickers if t.endswith("INR=X") and t != "INR=X"]
    assert not inr_legs, f"INR crosses duplicate USD/INR: {inr_legs}"


def test_factor_levels_keep_their_meaning_when_the_memory_switches():
    """Switch to a memory whose basis is reordered and sign-flipped: slot 0 must
    go on accumulating the same factor."""
    from engines.mula.factors import OnlineFactorModel
    rng = np.random.default_rng(3)
    n, T = 30, 900
    load = np.zeros((n, 2))
    load[:15, 0] = 1.0
    load[15:, 1] = 1.0
    f = rng.normal(0, 1, (T, 2)) * np.array([2.0, 1.2])
    z = f @ load.T + rng.normal(0, 0.5, (T, n))
    avail = np.ones(n, dtype=bool)
    fm = OnlineFactorModel(n)
    scores = np.zeros((T, 2))
    for t in range(T):
        if t == 600:                       # scramble the memory about to become active
            m = fm.mem[0]
            m.V[:, [0, 1]] = -m.V[:, [1, 0]]
            m.lam_eig[[0, 1]] = m.lam_eig[[1, 0]]
            m.last_recompute = fm.t        # hold that basis for the check
            m.C_ref = m.correlation()
        fm._logw = (np.array([-1e9, -1e9, 0.0]) if t < 600 else np.array([0.0, -1e9, -1e9]))
        out = fm.update(z[t].copy(), avail)
        scores[t] = out["scores"][:2]
    before = np.corrcoef(scores[300:600, 0], f[300:600, 0])[0, 1]
    after = np.corrcoef(scores[600:640, 0], f[600:640, 0])[0, 1]
    assert abs(before) > 0.8
    assert np.sign(after) == np.sign(before) and abs(after) > 0.8
