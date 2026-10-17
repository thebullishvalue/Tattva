"""The conviction chain (ui.components.build_hero_verdict) — pure data in/out."""
import math

from ui.components import build_hero_verdict, format_conviction

MULA = dict(fvo=-1.4, pct_mispricing=-0.031, xs_consistency=0.82, mr_prob=0.9,
            w_valuation=0.5, w_full=0.2, mula_kappa=0.012, mula_half_life=58,
            expected_drift_pct=0.9, drift_sd_pct=3.1, valuation_state="CHEAP")
EDGE = dict(ic=0.08, se=0.05, n=130, positive_blocks=5, blocks=[{}] * 6)
PREC = dict(n_eff=14.2, positive_pct=68, horizon=10, median=1.2, usable=True)


def _v(**kw):
    args = dict(mula=MULA, swayam_breadth={"oversold_pct": 60, "overbought_pct": 10},
                agreement=0.7, edge=EDGE, precedent=PREC, n_divergences=0,
                horizon_days=10, div_window=20)
    args.update(kw)
    return build_hero_verdict(**args)


def test_six_independent_gates_in_order():
    v = _v()
    assert [g["name"] for g in v["gates"]] == [
        "mispricing", "reversion", "cross_section", "convergence", "edge", "precedent"]
    assert math.isclose(v["conviction"], math.prod(g["value"] for g in v["gates"]))


def test_signal_word_follows_the_tier():
    """A claim the gates do not support is a HOLD — never BUY beside STAND ASIDE."""
    weak = _v(swayam_breadth={"oversold_pct": 5, "overbought_pct": 50})
    assert weak["action"]["level"] == "standaside"
    assert weak["signal"] == "HOLD" and weak["signal_class"] == "hold"
    assert "does not support acting" in weak["headline"]
    strong = _v()
    assert strong["signal"].endswith("BUY") and strong["signal_class"] == "buy"
    rich = _v(mula=dict(MULA, fvo=1.4, pct_mispricing=0.031),
              swayam_breadth={"oversold_pct": 10, "overbought_pct": 60},
              precedent=dict(PREC, positive_pct=32))
    assert rich["signal"].endswith("SELL") or rich["signal"] == "HOLD"


def test_no_claim_inside_the_noise():
    v = _v(mula=dict(MULA, fvo=0.1, pct_mispricing=0.002))
    assert v["signal"] == "HOLD" and v["action"]["label"] == "NO CLAIM"
    assert v["conviction"] is None and format_conviction(v) == "—"
    assert v["binding"] is None


def test_missing_evidence_is_neutral_and_unvalidated_edge_is_discounted():
    v = _v(swayam_breadth=None, precedent=None, edge=None)
    g = {x["name"]: x["value"] for x in v["gates"]}
    assert g["convergence"] == 0.5 and g["precedent"] == 0.5
    assert g["edge"] < 0.5


def test_unusable_precedent_does_not_vote():
    v = _v(precedent=dict(PREC, usable=False, note="unlike anything in the record"))
    gate = [x for x in v["gates"] if x["name"] == "precedent"][0]
    assert gate["value"] == 0.5 and "unlike anything" in gate["detail"]


def test_kappa_reaches_the_card():
    """The ECM detail printed κ̂=0.000 for months: the reader used the wrong key."""
    gate = [x for x in _v()["gates"] if x["name"] == "reversion"][0]
    assert "+0.0120/d" in gate["detail"]


def test_binding_constraint_is_the_smallest_gate():
    v = _v(edge=dict(EDGE, ic=-0.05))
    smallest = min(v["gates"], key=lambda g: g["value"])
    assert v["binding"] == smallest["name"]
    assert v["limit"].startswith("Capped by")
