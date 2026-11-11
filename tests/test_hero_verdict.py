"""The verdict (ui.components.build_hero_verdict) — pure data in/out."""
from ui.components import build_hero_verdict

MULA = dict(fvo=-1.4, pct_mispricing=-0.031, xs_consistency=0.82, mr_prob=0.9,
            w_valuation=0.5, w_full=0.2, mula_kappa=0.012, mula_half_life=58,
            expected_drift_pct=0.9, drift_sd_pct=3.1, valuation_state="CHEAP")
GOOD = dict(n_claims=4000, hit=56.0, base=52.0, excess_bps=40.0, t=2.4)
BAD = dict(n_claims=4000, hit=47.0, base=52.0, excess_bps=-30.0, t=-1.1)
RECORD = dict(asset_class="commodity", n_claims=8000, hit=51.0, base=50.0, t=0.6,
              long=GOOD, short=BAD)
CALL = dict(n_claims=300, hit=55.0, base=52.0, excess_bps=20.0, t=0.5, long=GOOD, short=BAD)
PREC = dict(n_eff=14.2, positive_pct=68, horizon=10, median=1.2, usable=True)
SKILL = dict(edge=4.0, n=60, hit=58.0, base=54.0)


def _v(**kw):
    args = dict(mula=MULA, swayam_breadth={"oversold_pct": 60, "overbought_pct": 10},
                call=CALL, class_record=RECORD, precedent=PREC,
                precedent_skill=SKILL, n_divergences=0, horizon_days=10, div_window=20)
    args.update(kw)
    return build_hero_verdict(**args)


def test_six_evidence_rows_in_order_and_no_score():
    v = _v()
    assert [g["name"] for g in v["evidence"]] == [
        "mispricing", "reversion", "cross_section", "convergence", "edge", "precedent"]
    assert "conviction" not in v and "gates" not in v and "binding" not in v
    assert all(g["tone"] in ("supports", "neutral", "against") for g in v["evidence"])


def test_a_licensed_side_is_called():
    v = _v()                                      # cheap, and the long side is licensed
    assert v["licensed"] and v["signal"] == "BUY" and v["signal_class"] == "buy"
    assert v["action"]["label"] == "GRADED CALL"


def test_an_unlicensed_side_is_a_state_not_a_call():
    rich = _v(mula=dict(MULA, fvo=1.4, pct_mispricing=0.031, valuation_state="RICH"))
    assert not rich["licensed"]
    assert rich["signal"] == "RICH" and rich["signal_class"] == "hold"
    assert rich["action"]["label"] == "NO GRADED CALL"
    assert "47.0%" in rich["limit"] and "52.0%" in rich["limit"]     # the record is printed
    assert "No sell call is issued" in rich["headline"]


def test_the_licence_needs_both_the_t_and_the_hit_rate():
    lucky = dict(RECORD, long=dict(GOOD, hit=51.0))                  # t fine, hit below base
    assert _v(class_record=lucky)["signal"] == "CHEAP"
    thin = dict(RECORD, long=dict(GOOD, t=0.6))                      # hit fine, t too small
    assert _v(class_record=thin)["signal"] == "CHEAP"


def test_no_record_means_no_call():
    v = _v(class_record=None)
    assert not v["licensed"] and v["signal"] == "CHEAP"
    assert v["trust"]["chip"] == "NO RECORD"
    assert _v(class_record=None, call=None)["signal_class"] == "hold"


def test_one_targets_own_record_cannot_license_a_call():
    v = _v(class_record=dict(RECORD, long=BAD), call=dict(CALL, long=dict(GOOD, t=5.0)))
    assert v["signal"] == "CHEAP" and not v["licensed"]


def test_no_claim_inside_the_noise():
    v = _v(mula=dict(MULA, fvo=0.1, pct_mispricing=0.002, valuation_state="FAIR"))
    assert v["signal"] == "FAIR" and v["action"]["label"] == "NO CALL"
    assert not v["licensed"] and v["signal_class"] == "hold"


def test_missing_valuation_is_said():
    v = _v(mula={})
    assert v["signal"] == "NO VALUATION" and v["signal_class"] == "hold"


def test_missing_evidence_reads_neutral():
    v = _v(swayam_breadth=None, precedent=None)
    rows = {g["name"]: g for g in v["evidence"]}
    assert rows["convergence"]["tone"] == "neutral" and rows["precedent"]["tone"] == "neutral"


def test_precedent_counts_only_when_the_matcher_has_skill():
    rows = {g["name"]: g for g in _v()["evidence"]}
    assert rows["precedent"]["label"] == "precedent agrees"
    unproven = {g["name"]: g for g in _v(precedent_skill=dict(edge=-3.0, n=60))["evidence"]}
    assert unproven["precedent"]["label"] == "matcher unproven"
    assert unproven["precedent"]["tone"] == "neutral"
    assert {g["name"]: g for g in _v(precedent_skill=None)["evidence"]}["precedent"]["label"] == "matcher unproven"


def test_withheld_session_and_divergences_are_flagged():
    v = _v(mula=dict(MULA, asof_row_withheld=True, withheld_reason="market closed"), n_divergences=2)
    assert "market closed" in v["caveat"] and "2 Mūla/Swayam" in v["risk"]
