"""
Decision-table test for ui.components.build_hero_verdict — the hero card's
ENTIRE interpretation logic (headline chain, label normalisation, trust
tiering, CALIBRATED second-opinion row, precedent gates, trend comparison,
internals alignment, risk row).

build_hero_verdict is deliberately a pure function (no Streamlit) so that
every branch of the verdict a user acts on can be pinned by tests instead of
verified by eyeballing rendered cards. Any change to the hero's decision
rules should extend this table.

HEADLINE = THE NORMALIZED CONSENSUS (product decision): the causal
expanding-z average of the two engines' readings — the same object as the
Unified Signal plot's top row and the TATTVA CONVICTION card, classified with
its own factory p75/p90-anchored thresholds (DEFAULT_THRESHOLDS). The Optuna-calibrated composite is the
CALIBRATED evidence row; the raw factory composite is research-only
(research/calibration_lift_study.py).

Run: python -m research.test_hero_verdict  (from the repo root)
"""
from __future__ import annotations

from ui.components import build_hero_verdict, _trust_tier


def _cons(value: float, signal: str, a_norm: float | None = None,
          n_norm: float | None = None) -> dict:
    """Consensus dict as compute_normalized_convergence emits it."""
    d = {"value": value, "signal": signal}
    if a_norm is not None:
        d["aarambh_norm"] = a_norm
    if n_norm is not None:
        d["nirnay_norm"] = n_norm
    return d


def _v(**overrides) -> dict:
    """Baseline kwargs: consensus bullish headline, calibrated agrees, decent trust."""
    base = dict(
        consensus=_cons(-0.42, "BUY", a_norm=-0.50, n_norm=-0.34),
        calib_conviction=-15.0,      # calibrated composite (±100; its scale is smaller)
        calib_signal="BUY",
        has_profile=True,
        aarambh_signal={"conviction_score": -30.0, "signal": "BUY"},
        agreement=0.75,
        val_ic=0.15,
        wf_pos=4 / 6,
        precedent=None,
        n_divergences=0,
        horizon_days=10,
    )
    base.update(overrides)
    return base


def _row(verdict: dict, tag: str) -> dict | None:
    return next((r for r in verdict["evidence"] if r["tag"] == tag), None)


def run() -> None:
    checks = 0

    # ── 1. Headline chain: CONSENSUS → Aarambh-only ──────────────────────
    v = build_hero_verdict(**_v())
    assert v["source"] == "Convergence consensus (normalized)" and v["score"] == -0.42
    assert v["direction"] == "bullish" and v["signal"] == "BUY"
    checks += 1

    # Degenerate gate is the CALLER's job (consensus passed as None) — chain
    # must fall to the honest Aarambh-only source.
    v = build_hero_verdict(**_v(consensus=None))
    assert v["source"] == "Aarambh only (no bottom-up convergence)" and v["score"] == -0.30
    checks += 1

    # ── 2. Label normalisation ───────────────────────────────────────────
    for raw, want_sig, want_dir in [
        ("NEUTRAL", "HOLD", "neutral"), ("HOLD", "HOLD", "neutral"),
        ("", "HOLD", "neutral"), ("garbage", "HOLD", "neutral"),
        ("WEAK BUY", "WEAK BUY", "bullish"), ("STRONG SELL", "STRONG SELL", "bearish"),
    ]:
        val = -0.35 if "BUY" in raw.upper() else 0.35 if "SELL" in raw.upper() else 0.0
        v = build_hero_verdict(**_v(consensus=_cons(val, raw)))
        assert (v["signal"], v["direction"]) == (want_sig, want_dir), (raw, v["signal"], v["direction"])
    checks += 1

    # signal_class css mapping
    assert build_hero_verdict(**_v())["signal_class"] == "undervalued"
    assert build_hero_verdict(**_v(consensus=_cons(0.45, "SELL")))["signal_class"] == "overvalued"
    assert build_hero_verdict(**_v(consensus=_cons(0.0, "HOLD")))["signal_class"] == "fair"
    checks += 1

    # ── 3. Sign-coherence guard (contract violation → neutralised, flagged) ──
    v = build_hero_verdict(**_v(consensus=_cons(+0.45, "BUY")))
    assert v["signal"] == "HOLD" and v["direction"] == "neutral"
    assert "sign-convention mismatch" in v["source"]
    checks += 1
    # Boundary: value exactly 0 with a BUY label is NOT a contradiction (0 is
    # sign-ambiguous), guard must not fire.
    v = build_hero_verdict(**_v(consensus=_cons(0.0, "BUY")))
    assert v["direction"] == "bullish"
    checks += 1

    # ── 4. Trust tiers ───────────────────────────────────────────────────
    assert _trust_tier(None, None)["tier"] == "uncalibrated"
    assert _trust_tier(-0.02, None)["tier"] == "no_edge"
    assert _trust_tier(0.0, None)["tier"] == "no_edge"      # boundary: <= 0
    assert _trust_tier(0.09, None)["tier"] == "marginal"
    assert _trust_tier(0.10, None)["tier"] == "modest"      # boundary
    assert _trust_tier(0.19, None)["tier"] == "modest"
    assert _trust_tier(0.20, None)["tier"] == "solid"       # boundary
    checks += 1
    # WF counts preferred over ratio.
    t = _trust_tier(0.15, 4 / 6, wf_n=6)
    assert "4/6 windows positive" in t["prose"]
    assert t["wf_n"] == 6
    t = _trust_tier(0.15, 0.67)         # no count available → ratio fallback
    assert "67% of windows" in t["prose"]
    checks += 1

    # ── 5. MODEL row: trust prose only — no attribution note, no timestamp
    # (both dropped from the card copy as noise).
    r = _row(build_hero_verdict(**_v()), "MODEL")
    assert "Modest validated edge" in r["text"]
    assert "calibrated variant" not in r["text"] and "Calibrated 20" not in r["text"]
    checks += 1

    # ── 6. CALIBRATED second-opinion row ──────────────────────────────────
    r = _row(build_hero_verdict(**_v()), "CALIBRATED")
    assert r["state"] == "confirm" and "agrees" in r["text"]
    # Calibrated disagrees (opposite direction; coherent negative=bullish scores)
    r = _row(build_hero_verdict(**_v(calib_conviction=+14.0, calib_signal="SELL")), "CALIBRATED")
    assert r["state"] == "conflict" and "DISAGREES" in r["text"]
    # Calibrated neutral while headline directional → not confirmed
    r = _row(build_hero_verdict(**_v(calib_conviction=-5.0, calib_signal="HOLD")), "CALIBRATED")
    assert r["state"] == "neutral" and "NOT confirmed" in r["text"]
    # Headline neutral while calibrated directional → info lean
    r = _row(build_hero_verdict(**_v(consensus=_cons(-0.02, "HOLD"))), "CALIBRATED")
    assert r["state"] == "info" and "leans bullish" in r["text"]
    # Both neutral
    r = _row(build_hero_verdict(**_v(consensus=_cons(-0.02, "HOLD"),
                                     calib_conviction=-5.0, calib_signal="HOLD")), "CALIBRATED")
    assert r["state"] == "neutral" and "also neutral" in r["text"]
    # No profile → no CALIBRATED row (uncalibrated variant adds no evidence)
    assert _row(build_hero_verdict(**_v(has_profile=False)), "CALIBRATED") is None
    # Calibrated values absent → no row
    assert _row(build_hero_verdict(**_v(calib_conviction=None, calib_signal=None)),
                "CALIBRATED") is None
    checks += 1

    # ── 7. Neutral headline: flat vs inside-the-HOLD-band lean ──────────
    v = build_hero_verdict(**_v(consensus=_cons(0.01, "HOLD")))
    assert "flat" in v["headline"]
    v = build_hero_verdict(**_v(consensus=_cons(-0.28, "HOLD")))
    assert "inside the HOLD band" in v["headline"] and "leaning bullish" in v["headline"]
    v = build_hero_verdict(**_v(consensus=_cons(+0.28, "HOLD")))
    assert "leaning bearish" in v["headline"]
    checks += 1

    # ── 8. TREND row (convergence path only; smoothed = DDM of the consensus) ──
    def _trend(value, smooth, **kw):
        sig = "BUY" if value < 0 else "SELL" if value > 0 else "HOLD"
        return _row(build_hero_verdict(**_v(consensus=_cons(value, sig),
                                            smoothed=smooth, **kw)), "TREND")

    assert _trend(-0.45, -0.30)["state"] == "confirm"        # extends trend
    assert "extends" in _trend(-0.45, -0.30)["text"]
    r = _trend(-0.20, -0.50)
    assert r["state"] == "neutral" and "softer" in r["text"]  # fading intensity
    r = _trend(-0.45, +0.30)
    assert r["state"] == "conflict" and "contradicts" in r["text"]  # trend break
    r = _trend(-0.45, +0.02)
    assert r["state"] == "neutral" and "not yet a trend" in r["text"]  # no trend yet
    # headline flat vs established trend → stall
    r = _row(build_hero_verdict(**_v(consensus=_cons(-0.02, "HOLD"),
                                     smoothed=-0.40)), "TREND")
    assert r["state"] == "neutral" and "stall" in r["text"]
    # both flat → no row at all (nothing to interpret)
    assert _row(build_hero_verdict(**_v(consensus=_cons(-0.02, "HOLD"),
                                        smoothed=0.01)), "TREND") is None
    # non-convergence path → never a TREND row (different objects)
    assert _row(build_hero_verdict(**_v(consensus=None, smoothed=-0.40)), "TREND") is None
    checks += 1

    # ── 9. PRECEDENT gates (order: thin → split → mixed → info/agree/diverge) ──
    def _prec(**p):
        base = {"horizon": 10, "median": 1.2, "positive_pct": 80.0, "n": 8, "dir": 1}
        base.update(p)
        return base

    r = _row(build_hero_verdict(**_v(precedent=_prec(n=3))), "PRECEDENT")
    assert r["state"] == "neutral" and "Thin sample" in r["text"]
    r = _row(build_hero_verdict(**_v(precedent=_prec(positive_pct=60.0))), "PRECEDENT")
    assert r["state"] == "neutral" and "Split" in r["text"]       # |60-50| < 15
    # MIXED: median bearish but 70% positive → skewed outcomes, no robust lean
    r = _row(build_hero_verdict(**_v(precedent=_prec(median=-0.4, dir=-1,
                                                     positive_pct=70.0))), "PRECEDENT")
    assert r["state"] == "neutral" and "Mixed" in r["text"]
    # AGREE: bullish precedent vs bullish headline
    r = _row(build_hero_verdict(**_v(precedent=_prec())), "PRECEDENT")
    assert r["state"] == "confirm" and "Agrees" in r["text"]
    # DIVERGE: bearish precedent (coherent: median<0 AND <50% positive) vs bullish headline
    r = _row(build_hero_verdict(**_v(precedent=_prec(median=-1.0, dir=-1,
                                                     positive_pct=25.0))), "PRECEDENT")
    assert r["state"] == "conflict" and "Diverges" in r["text"]
    # INFO: coherent lean but hero itself neutral
    r = _row(build_hero_verdict(**_v(consensus=_cons(0.0, "HOLD"),
                                     precedent=_prec())), "PRECEDENT")
    assert r["state"] == "info" and "no edge" in r["text"]
    # No precedent / n=0 → no row
    assert _row(build_hero_verdict(**_v(precedent=None)), "PRECEDENT") is None
    assert _row(build_hero_verdict(**_v(precedent=_prec(n=0))), "PRECEDENT") is None
    checks += 1

    # ── 10. INTERNALS row ─────────────────────────────────────────────────
    r = _row(build_hero_verdict(**_v()), "INTERNALS")     # aligned + strong (0.75 > 0.7)
    assert r["state"] == "confirm" and "aligned" in r["text"]
    # NO reconciliation sentence anymore — the headline IS the consensus.
    assert "Consensus reads" not in r["text"]
    r = _row(build_hero_verdict(**_v(agreement=0.60)), "INTERNALS")   # aligned + moderate
    assert r["state"] == "neutral"
    r = _row(build_hero_verdict(**_v(consensus=_cons(-0.05, "HOLD",
                                                     a_norm=-0.35, n_norm=+0.25))), "INTERNALS")
    assert r["state"] == "conflict" and "split" in r["text"]
    # No consensus → no INTERNALS row
    assert _row(build_hero_verdict(**_v(consensus=None)), "INTERNALS") is None
    checks += 1

    # ── 11. RISK row ──────────────────────────────────────────────────────
    assert _row(build_hero_verdict(**_v(n_divergences=0)), "RISK") is None
    r = _row(build_hero_verdict(**_v(n_divergences=1, div_window=20)), "RISK")
    assert "1 divergence event in the last ~20 trading days" in r["text"]
    assert "Recent Divergences" in r["text"]
    r = _row(build_hero_verdict(**_v(n_divergences=3)), "RISK")
    assert "3 divergence events" in r["text"]              # plural, no window copy
    checks += 1

    # ── 12. Row ordering: MODEL, CALIBRATED, TREND, PRECEDENT, INTERNALS, RISK ──
    v = build_hero_verdict(**_v(smoothed=-0.30, precedent={"horizon": 10, "median": 1.2,
                                                           "positive_pct": 80.0, "n": 8,
                                                           "dir": 1},
                                n_divergences=2))
    tags = [r["tag"] for r in v["evidence"]]
    assert tags == ["MODEL", "CALIBRATED", "TREND", "PRECEDENT", "INTERNALS", "RISK"], tags
    checks += 1

    # ── 13. ACTION synthesis: the decision must weigh ALL evidence, not just
    # the raw consensus value. Same headline (BUY -0.42) throughout — only the
    # surrounding evidence changes, and the recommended tier must move with it.
    def _act(**kw):
        return build_hero_verdict(**_v(**kw))["action"]

    # Baseline: modest edge +1, calibrated agrees +1, internals confirm +1 → HIGH
    a = _act()
    assert a["level"] == "high" and a["score"] == 3, a
    assert "net +3" in a["drivers"]
    # Calibrated flips to DISAGREE (-2 instead of +1) → same headline, LOW
    a = _act(calib_conviction=+14.0, calib_signal="SELL")
    assert a["level"] == "low" and a["score"] == 0, a
    # Precedent coherently diverges too (-2) → STAND ASIDE despite BUY headline
    a = _act(calib_conviction=+14.0, calib_signal="SELL",
             precedent={"horizon": 10, "median": -1.0, "positive_pct": 25.0,
                        "n": 8, "dir": -1})
    assert a["level"] == "stand_aside" and a["score"] == -2, a
    # Validated no-edge is a HARD GATE: everything else confirming, still stand aside
    a = _act(val_ic=-0.02)
    assert a["level"] == "stand_aside" and "no-edge gate" in a["drivers"], a
    # Unvalidated edge is capped at MODERATE even with a full confirm stack
    a = _act(val_ic=None, has_profile=False,
             smoothed=-0.30,
             precedent={"horizon": 10, "median": 1.2, "positive_pct": 80.0,
                        "n": 8, "dir": 1})
    assert a["level"] == "moderate" and "capped" in a["drivers"], a
    # Neutral headline → NO ACTION (nothing to size)
    a = _act(consensus=_cons(0.01, "HOLD"))
    assert a["level"] == "none" and a["score"] is None, a
    # Calibrated "NOT confirmed" (neutral on a directional headline) weighs -1
    a = _act(calib_conviction=-5.0, calib_signal="HOLD")
    assert a["score"] == 1 and "calibrated neutral -1" in a["drivers"], a
    # Recent divergences (RISK) drag the tier down
    a = _act(n_divergences=2)
    assert a["score"] == 2 and a["level"] == "moderate", a
    checks += 1

    print(f"build_hero_verdict: ALL {checks} CHECK GROUPS PASSED")


if __name__ == "__main__":
    run()
