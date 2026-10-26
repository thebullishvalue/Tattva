"""C12: the claim's unit holds — the charter's contract, expected to fail today.

The cuts the claim is read at (±CLAIM_SIGMA no claim, ±DEEP_SIGMA deep) assume
FVO is a centred, unit-scale oscillator: |mean| < 0.15, SD in [0.8, 1.2], RICH
and CHEAP claims within [0.67, 1.5] of each other. On the audited instruments
it is not (U04): FVO sits rich and drifts with the trend, so most claims are
SELL. No transform ships in this release (design §8) — the owner kept Mūla's
claim, the rank IC that sets trust is drift-invariant, and the four audited
instruments are burned for the decision. The candidate fix (Decision A,
``core.semantics.claim_oscillator``) is pre-registered with the pooled-prior
study and is applied only by an owner-signed change.

So the contract is asserted on the live fixture as a STRICT xfail: the day it
passes, this test fails and the marker must come off. The contract's own
arithmetic is asserted for real on synthetic oscillators.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.evidence import claim_unit

_C12 = ("C12: FVO is biased rich and drifts with the trend (U04); no claim-unit transform "
        "ships until the pooled-prior study's Decision A and an owner-signed change (design §8)")


def _fvo_ts(fvo: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(fvo))
    return pd.DataFrame({"FVO": fvo, "Valid": True}, index=idx)


def test_the_contract_holds_on_a_unit_oscillator_and_fails_on_a_rich_one():
    rng = np.random.default_rng(0)
    unit = claim_unit(_fvo_ts(rng.normal(0, 1, 3000)))
    assert unit["passes"] and all(unit["contract"].values())
    rich = claim_unit(_fvo_ts(rng.normal(0.5, 1.3, 3000)))
    assert not rich["passes"]
    assert rich["contract"] == {"mean": False, "sd": False, "rich_cheap": False}


@pytest.mark.xfail(strict=True, reason=_C12)
def test_the_claims_unit_holds_on_gold(gold_app):
    cu = gold_app.session_state["evidence"]["claim_unit"]
    assert cu["passes"], cu
