"""C8: the sizers' inputs are disjoint from the claim's — and the gate a new sizer must pass.

Tattva's call is one claim (Mūla's FVO) scaled by sizers. A sizer that is the
claim under another name re-counts it: the retired window stack read ρ ≈ 0.98
with FVO, Precedent matched on five of the claim's own eight features. So a
quantity becomes a SIZER only through this file (design §2.3):

  1. its inputs are published columns disjoint from the claim's and from every
     other sizer's (``LINEAGE``), and the composer's z reads them and nothing
     of the claim's level;
  2. on claim days its z has |ρ| < RHO_MAX (Spearman) with the claim's input,
     FVO and |FVO|, and with every other sizer's z;
  3. it then enters the pooled study as a pre-registered candidate and earns β
     through the one strength map like any sizer (×1.00 until it does).

The two shipped sizers pass on the fixture. Measured on the four 2.8.0 pickles
the largest |ρ| is 0.20 / 0.12 / 0.22 / 0.16 (Gold, Nifty 50, S&P 500,
USD/INR), the reversion z against |FVO| in each but Nifty.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from core.call import SIZERS, compose_series
from core.instruments import instrument
from core.roles import ROLES
from core.semantics import CLAIM_SIGMA, claim_direction
from pipeline.assemble import gather_inputs

#: The largest |ρ| a sizer's z may have with the claim's input or another sizer's.
RHO_MAX = 0.5
#: The published columns the claim is read from (core.call: d = claim_direction(FVO)).
CLAIM_INPUTS = frozenset({"FVO", "PctMispricing"})
#: Every sizer and the published columns its reading comes from: Swayam's breadth
#: (engines.swayam.aggregate_views, carried onto the spine by pipeline.run) and
#: the error-correction layer's read (engines.mula.ecm).
LINEAGE: dict[str, frozenset] = {
    "swayam": frozenset({"Lean", "_Native"}),
    "reversion": frozenset({"MRProb", "MulaKappa"}),
}


def promotion_gate(z: dict[str, pd.Series], fvo: pd.Series, rows: pd.Series) -> dict[str, float]:
    """The largest |ρ| of each sizer's z on claim days ``rows``, against FVO,
    |FVO| and every other sizer's z (Spearman, on the days both are defined)."""
    def rho(a, b):
        m = rows & a.notna() & b.notna()
        return abs(float(a[m].rank().corr(b[m].rank()))) if m.sum() > 2 else 0.0

    out = {k: max(rho(v, fvo), rho(v, fvo.abs())) for k, v in z.items()}
    for a, b in combinations(z, 2):
        r = rho(z[a], z[b])
        out[a], out[b] = max(out[a], r), max(out[b], r)
    return out


def _zs(call_ts: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """The sizers' raw readings as z on claim days, from the issued record."""
    fvo = call_ts["fvo"].astype(float)
    claim = fvo.map(claim_direction).astype(float)
    rows = call_ts["valid"].astype(bool) & call_ts["printed"].astype(bool) & (claim != 0)
    native = call_ts["swayam_native"].astype(bool)
    z = {"swayam": (claim * call_ts["swayam_lean"].astype(float)).where(native),
         "reversion": 2.0 * call_ts["mr_prob"].astype(float) - 1.0}
    return z, fvo, rows


# ─── 1. lineage ──────────────────────────────────────────────────────────────

def test_every_shipped_sizer_is_declared_with_disjoint_inputs():
    # A sizer ships only through this table: a new one fails here until declared.
    assert {s.key for s in SIZERS} == set(LINEAGE)
    for s in SIZERS:
        assert ROLES[f"sizer:{s.key}"].kind == "SIZER" and ROLES[f"sizer:{s.key}"].grade_key
    for a, b in combinations(LINEAGE, 2):
        assert not LINEAGE[a] & LINEAGE[b], (a, b)
    for k, cols in LINEAGE.items():
        assert not cols & CLAIM_INPUTS, k


def _frames(n=400, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    fvo = pd.Series(rng.normal(0, 1.4, n), index=idx)
    fvo_ts = pd.DataFrame({
        "Price": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), "FVO": fvo,
        "PctMispricing": fvo * 0.02, "Valid": True, "Provisional": False, "WithheldReason": "",
        "MRProb": rng.uniform(0.2, 0.9, n), "MulaKappa": rng.normal(0, 0.01, n)}, index=idx)
    lean = np.clip(rng.normal(0, 0.3, n), -1, 1)
    sw = pd.DataFrame({"Lean": lean, "_Native": rng.uniform(size=n) > 0.05}, index=idx)
    return fvo_ts, sw


def _inputs(fvo_ts, sw):
    return gather_inputs(fvo_ts, sw, instrument=instrument("Gold"), h=10,
                         today=fvo_ts.index[-1])


def test_the_composer_reads_each_sizer_only_from_its_own_lineage():
    fvo_ts, sw = _frames()
    base = _inputs(fvo_ts, sw)
    # Move every claim's level but none of its signs (|FVO| stays past the cut
    # on claim days, inside it elsewhere): neither sizer's reading moves.
    moved = fvo_ts.copy()
    f = moved["FVO"]
    moved["FVO"] = np.where(f.abs() >= CLAIM_SIGMA, f * 1.7, f * 0.3)
    alt = _inputs(moved, sw)
    assert (alt["claim"] == base["claim"]).all()
    for c in ("swayam_lean", "z_swayam", "mr_prob", "kappa", "z_reversion"):
        assert np.array_equal(alt[c].to_numpy(), base[c].to_numpy(), equal_nan=True), c
    # Each sizer's own input moves only its own z, never the claim or the other's.
    sw2 = sw.assign(Lean=-sw["Lean"])
    a2 = _inputs(fvo_ts, sw2)
    assert (a2["claim"] == base["claim"]).all() and a2["z_reversion"].equals(base["z_reversion"])
    f3 = fvo_ts.assign(MRProb=1.0 - fvo_ts["MRProb"])
    a3 = _inputs(f3, sw)
    assert (a3["claim"] == base["claim"]).all() and a3["z_swayam"].equals(base["z_swayam"])
    # And the composer's direction is the claim's whatever the sizers read.
    ct = compose_series(a2, {}, tradeable=True, sizers_enabled=True)
    assert (ct["d"] == compose_series(base, {}, tradeable=True, sizers_enabled=True)["d"]).all()


# ─── 2. |ρ| < 0.5 on the fixture's claim days ────────────────────────────────

def test_the_sizers_are_uncorrelated_with_the_claim_on_claim_days(gold_app):
    z, fvo, rows = _zs(gold_app.session_state["call_ts"])
    assert rows.sum() > 500
    rho = promotion_gate(z, fvo, rows)
    assert max(rho.values()) < RHO_MAX, rho


@pytest.mark.slow
def test_the_sizers_are_uncorrelated_with_the_claim_on_usdinr(offline_fetch):
    from pipeline.run import run_target
    z, fvo, rows = _zs(run_target("USD/INR")["call_ts"])
    rho = promotion_gate(z, fvo, rows)
    assert max(rho.values()) < RHO_MAX, rho


# ─── 3. the gate ─────────────────────────────────────────────────────────────

def test_the_gate_refuses_the_claim_under_another_name(gold_app):
    z, fvo, rows = _zs(gold_app.session_state["call_ts"])
    # The retired window stack's kind of candidate: a smooth function of FVO.
    stack = (-fvo.rolling(5, min_periods=1).mean() / 3).clip(-1, 1)
    assert promotion_gate({**z, "window_stack": stack}, fvo, rows)["window_stack"] > RHO_MAX
    # Passing the gate earns nothing by itself: β comes only from the sizer's own
    # grade, and wherever β is 0 the sizer is exactly ×1.00.
    ct = gold_app.session_state["call_ts"]
    for s in SIZERS:
        unearned = ct[f"{s.key}_beta"] == 0
        assert unearned.any() and (ct.loc[unearned, f"{s.key}_factor"] == 1.0).all(), s.key
