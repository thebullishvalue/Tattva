"""The integration charter, clause by clause: each of C1–C16 indexed to the tests that enforce it.

:data:`CLAUSES` is the index — one entry per charter clause, its contract in
one line and the tests that hold the shipped system to it (``file::test``).
ARCHITECTURE.md's contracts table is the same index in prose
(``tests/test_docs.py`` holds the two together). One test per clause asserts
that each of its enforcing tests exists (a module-level ``test_`` function,
not skipped, not marked slow, not expected to fail) and PASSED: in this
session when it ran here (``conftest.py`` records every outcome and runs this
module last), else in one fresh pytest run of the tests this session did not
select (not collected, or deselected by -k, -m, --deselect or --lf).

C12 — the claim's unit holds — is the charter's contract that is expected to
fail today. The cuts the claim is read at (±CLAIM_SIGMA no claim, ±DEEP_SIGMA
deep) assume FVO is a centred, unit-scale oscillator: |mean| < 0.15, SD in
[0.8, 1.2], RICH and CHEAP claims within [0.67, 1.5] of each other. On the
audited instruments it is not (U04): FVO sits rich and drifts with the trend,
so most claims are SELL. No transform ships in this release (design §8) — the
owner kept Mūla's claim, the rank IC that sets trust is drift-invariant, and
the four audited instruments are burned for the decision. The candidate fix
(Decision A, ``core.semantics.claim_oscillator``) was pre-registered with the
pooled-prior study, which recorded 'none', and it is applied only by an
owner-signed change. So the contract is asserted on the live fixture as a
STRICT xfail: the day it passes, that test fails and the marker must come
off. The contract's own arithmetic is asserted for real on synthetic
oscillators.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import conftest
from pipeline.evidence import claim_unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_C12 = ("C12: FVO is biased rich and drifts with the trend (U04); no claim-unit transform "
        "ships until the pooled-prior study's Decision A and an owner-signed change (design §8)")


# ─── C12: the claim's unit ───────────────────────────────────────────────────

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


# ─── The index ───────────────────────────────────────────────────────────────

#: Each clause: ``(contract, enforcing tests)``; a test is ``'<file>::<function>'``
#: under tests/ (a parametrised test counts once, every case must pass).
CLAUSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "C1": ("One estimand, one horizon: h only from InstrumentConfig.forecast_horizon and L only "
           "from EXECUTION_LAG, the same in every grade, the verdict and the bundle", (
        "test_contracts_asof_horizon.py::test_one_horizon_from_config_to_bundle",
        "test_contracts_asof_horizon.py::test_a_horizon_the_config_cannot_carry_is_refused",
        "test_call.py::test_i_every_role_is_graded_at_the_one_horizon",
        "test_call.py::test_j_a_twenty_session_config_reaches_every_object",
        "test_grade.py::test_f_exec_lag_one_is_the_price_one_row_later",
        "test_trust.py::test_a_file_for_another_horizon_or_lag_is_refused",
    )),
    "C2": ("One as-of session: every verdict input is read at d*, Mūla's newest published "
           "session", (
        "test_contracts_asof_horizon.py::test_a_withheld_newest_row_reads_every_input_at_d_star",
        "test_contracts_asof_horizon.py::test_the_bar_reads_d_star_on_the_live_run",
        "test_contracts_asof_horizon.py::test_a_provisional_row_is_read_not_absorbed",
        "test_call.py::test_h_the_verdict_is_call_ts_at_d_star",
        "test_causality.py::test_latest_reading_is_never_a_coerced_zero",
    )),
    "C3": ("One claim, one sign vocabulary: only claim_direction(FVO) sets d, nothing flips it, "
           "and every lean keeps one orientation", (
        "test_call.py::test_c_perturbing_every_sizer_input_never_changes_d",
        "test_semantics.py::test_readings_and_leans_have_opposite_signs",
        "test_semantics.py::test_claim_direction_dead_band_and_saturation",
        "test_semantics.py::test_the_one_swayam_word",
        "test_swayam.py::test_lean_is_breadth_lean_of_the_published_shares",
        "test_swayam.py::test_the_contrarian_convention_oversold_is_a_positive_lean",
    )),
    "C4": ("One grader, in the pipeline: core.grade is the only estimator, it cannot see the "
           "future, and no page runs one", (
        "test_roles.py::test_no_estimator_runs_in_ui",
        "test_app_smoke.py::test_the_bundle_carries_the_closed_loop",
        "test_grade.py::test_series_edge_recovers_a_planted_ic",
        "test_grade.py::test_series_edge_cannot_see_the_future",
    )),
    "C5": ("The call is graded as issued: call_ts at every session, resolved rows never "
           "rewritten, its grade heads Overview", (
        "test_causality.py::test_the_call_never_rewrites_a_resolved_session",
        "test_call.py::test_a_resolved_rows_never_change_when_the_panel_is_cut",
        "test_call.py::test_identical_issued_calls_are_ungraded_and_the_rail_still_reads",
        "test_app_smoke.py::test_overview_leads_with_the_call_its_grade_and_the_chain",
    )),
    "C6": ("Every sizer earns its strength, and no unregistered knob: β = 0 and ×1.00 while "
           "unskilled, every decision constant in KNOBS", (
        "test_call.py::test_b_unskilled_sizer_is_exactly_one_for_any_reading",
        "test_call.py::test_e_a_failed_sizer_grade_gives_beta_zero",
        "test_call.py::test_g_binding_is_the_smallest_factor_below_one_never_trust",
        "test_knobs.py::test_every_knob_declares_kind_consumer_coupling_and_reason",
        "test_knobs.py::test_a_measured_knob_names_a_study_file_that_exists",
        "test_knobs.py::test_no_unregistered_literal_in_the_grade_trust_or_words",
    )),
    "C7": ("Claim trust is causal and precision-weighted: the as-of grade, its calibrated SE, "
           "shrunk to a leave-one-out prior, through the one map", (
        "test_grade.py::test_h_ic_0078_at_se_0089_is_no_edge",
        "test_grade.py::test_g_expanding_edge_is_invariant_to_truncation",
        "test_grade.py::test_c_shift_null_se_is_calibrated",
        "test_trust.py::test_equal_ic_at_different_se_gives_different_strength",
        "test_trust.py::test_a_study_member_never_reads_its_classs_all_member_pool",
        "test_study.py::test_a_planted_skill_never_raises_its_own_prior",
        "test_study.py::test_b_an_appended_vintage_never_changes_an_earlier_lookup",
        "test_call.py::test_d_trust_zero_is_hold_naming_t_post_and_the_bar",
    )),
    "C8": ("No double count: sizer inputs disjoint from the claim, |ρ| < 0.5 on claim days, "
           "twins dropped and pooled at their true precision", (
        "test_lineage.py::test_every_shipped_sizer_is_declared_with_disjoint_inputs",
        "test_lineage.py::test_the_composer_reads_each_sizer_only_from_its_own_lineage",
        "test_lineage.py::test_the_sizers_are_uncorrelated_with_the_claim_on_claim_days",
        "test_lineage.py::test_the_gate_refuses_the_claim_under_another_name",
        "test_instruments.py::test_no_fx_target_holds_both_members_of_a_twin_pair",
        "test_study.py::test_d_twins_add_almost_no_precision",
    )),
    "C9": ("Drift-adjusted base rates: every grade on the drift-neutral target, every hit "
           "beside the side-matched causal base", (
        "test_grade.py::test_e_planted_drift_is_not_skill",
        "test_grade.py::test_a_no_skill_always_buy_lean_hits_its_side_matched_base",
        "test_grade.py::test_causal_drift_reads_only_resolved_returns",
    )),
    "C10": ("Live value = graded value: the app's call is run_target's, and a page prints the "
            "chain's own values", (
        "test_pipeline_equivalence.py::test_run_target_is_the_app_pipeline_byte_for_byte",
        "test_call.py::test_b_absent_or_carried_swayam_is_neutral_and_never_fabricated",
        "test_swayam.py::test_a_carried_row_is_x1_in_the_composer",
        "test_roles.py::test_the_mula_and_swayam_pages_print_overviews_chain_values",
    )),
    "C11": ("Tiers and units calibrated: the size is a unitless index, no '%' and no tier word "
            "until Decision C passes, breadth in points", (
        "test_copy.py::test_no_tier_word_while_the_tier_rule_is_none",
        "test_copy.py::test_no_percent_follows_a_size",
        "test_copy.py::test_breadth_is_printed_only_in_points",
        "test_call.py::test_the_size_is_an_index_never_a_percentage",
        "test_study.py::test_the_tercile_rule_is_the_pipelines",
    )),
    "C12": ("The claim's unit holds: |mean FVO| < 0.15, SD in [0.8, 1.2], RICH:CHEAP in "
            "[0.67, 1.5] — reported as evidence, a strict xfail until Decision A is applied", (
        "test_charter.py::test_the_contract_holds_on_a_unit_oscillator_and_fails_on_a_rich_one",
    )),
    "C13": ("One definition per concept, one copy per quantity: one stationarity word, no "
            "retired or banned copy on a page or in the docs, the docs' numbers generated", (
        "test_semantics.py::test_gap_stationarity_has_three_states",
        "test_app_smoke.py::test_the_mula_page_reads_the_verdict_and_carries_no_retired_copy",
        "test_copy.py::test_the_banned_phrases_are_absent_from_ui_and_app",
        "test_copy.py::test_no_banned_phrase_reaches_a_rendered_page",
        "test_copy.py::test_the_reserved_words_appear_only_in_the_call_path_code",
        "test_docs.py::test_the_docs_carry_no_refuted_claim",
        "test_docs.py::test_the_readme_edge_table_is_the_committed_studys",
    )),
    "C14": ("Learners and smoothers answer to the grade: none ships on the decision path, "
            "Swayam's views count equally and causally, drift is not skill", (
        "test_app_smoke.py::test_the_bundle_carries_the_closed_loop",
        "test_swayam.py::test_equal_weight_breadth_is_unchanged_under_ohlcv_truncation",
        "test_grade.py::test_e_planted_drift_is_not_skill",
    )),
    "C15": ("Role registry, grades on lean words, page grammar: every card registered to a "
            "role, every lean beside its grade, Overview and the rail on the call path", (
        "test_roles.py::test_the_registry_is_closed_and_its_chips_are_fixed",
        "test_roles.py::test_every_rendered_kpi_and_section_names_a_registered_card",
        "test_roles.py::test_every_data_column_names_a_registered_card",
        "test_roles.py::test_every_lean_word_sits_in_an_item_that_carries_an_edge",
        "test_roles.py::test_overview_opens_with_the_call_its_grade_and_the_chain",
        "test_roles.py::test_the_rail_holds_the_calls_readout_only",
        "test_render.py::test_a_missing_or_unregistered_card_raises",
        "test_render.py::test_every_page_renders_in_every_synthetic_state",
    )),
    "C16": ("No orphans: every published column, edge, signal key, bundle key, config field "
            "and knob has a reader", (
        "test_consumers.py::test_every_mula_column_has_a_reader",
        "test_consumers.py::test_every_swayam_column_has_a_reader",
        "test_consumers.py::test_every_call_column_has_a_reader",
        "test_consumers.py::test_every_edge_has_a_reader",
        "test_consumers.py::test_every_signal_key_has_a_reader",
        "test_consumers.py::test_every_bundle_key_has_a_reader",
        "test_consumers.py::test_every_config_field_has_a_reader",
        "test_consumers.py::test_every_knob_has_a_reader",
        "test_consumers.py::test_the_allowlists_are_current",
        "test_consumers.py::test_the_scan_sees_a_reader_and_an_orphan",
    )),
}

#: C12's live contract: indexed as a strict xfail, not as a pass.
C12_XFAIL = "test_charter.py::test_the_claims_unit_holds_on_gold"


def _node(spec: str) -> str:
    return os.path.join("tests", spec)


def _decorators(spec: str) -> list[str]:
    """The source of each decorator on the module-level test ``spec`` names; raises if
    there is no such function."""
    file, name = spec.split("::")
    path = os.path.join(ROOT, "tests", file)
    assert os.path.isfile(path), f"{spec}: no file tests/{file}"
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, f"{spec}: no module-level test {name} in tests/{file}"
    return [ast.unparse(d) for d in fn.decorator_list]


def _session_outcomes(spec: str) -> list[str] | None:
    """The outcome of every case of ``spec`` in this session, or None if it was not selected."""
    cases = [n for n in conftest.COLLECTED if n == spec or n.startswith(spec + "[")]
    every = [n for n in conftest.EVERY if n == spec or n.startswith(spec + "[")]
    if not cases or len(cases) < len(every):        # not selected, or only some of its cases
        return None
    return [conftest.OUTCOMES.get(n, "not run") for n in cases]


_INDEX = "test_charter.py::test_each_clause_is_enforced_by_tests_that_exist_and_pass"
_C12_INDEX = "test_charter.py::test_c12_stays_a_strict_xfail_with_its_reason"


def _wanted() -> set[str]:
    """The indexed tests this session's index tests read: those of the selected
    clauses (every clause when no index test is selected, e.g. a direct call)."""
    clauses = [c for c in CLAUSES if f"{_INDEX}[{c}]" in conftest.COLLECTED]
    wanted = {s for c in (clauses or CLAUSES) for s in CLAUSES[c][1]}
    if _C12_INDEX in conftest.COLLECTED or not clauses:
        wanted.add(C12_XFAIL)
    return wanted


@lru_cache(maxsize=None)
def _fresh_run() -> dict[str, list[str]]:
    """Every indexed test this session's index reads but did not select, run once in a
    fresh pytest: ``{spec: [outcome per case]}``."""
    missing = sorted(s for s in _wanted() if _session_outcomes(s) is None)
    if not missing:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        xml = os.path.join(tmp, "index.xml")
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        f"--junitxml={xml}", *(_node(s) for s in missing)],
                       cwd=ROOT, capture_output=True, text=True, check=False)
        cases = list(ET.parse(xml).getroot().iter("testcase"))
    out: dict[str, list[str]] = {}
    for case in cases:
        file = case.get("classname", "").split(".")[-1] + ".py"
        name = case.get("name", "").split("[")[0]
        kids = {c.tag for c in case}
        skipped = next((c for c in case if c.tag == "skipped"), None)
        outcome = ("failed" if kids & {"failure", "error"}
                   else "xfailed" if skipped is not None and skipped.get("type") == "pytest.xfail"
                   else "skipped" if skipped is not None else "passed")
        out.setdefault(f"{file}::{name}", []).append(outcome)
    return out


def _outcomes(spec: str) -> list[str]:
    here = _session_outcomes(spec)
    return here if here is not None else _fresh_run().get(spec, ["not run"])


def test_the_index_covers_every_clause_once():
    assert list(CLAUSES) == [f"C{i}" for i in range(1, 17)]
    for clause, (contract, specs) in CLAUSES.items():
        assert contract.strip() and specs, clause
        assert len(set(specs)) == len(specs), clause


@pytest.mark.parametrize("clause", list(CLAUSES))
def test_each_clause_is_enforced_by_tests_that_exist_and_pass(clause):
    _contract, specs = CLAUSES[clause]
    for spec in specs:
        marks = _decorators(spec)
        assert not any(m.startswith(("pytest.mark.skip", "pytest.mark.xfail", "pytest.mark.slow"))
                       for m in marks), (spec, marks)
    failed = {spec: o for spec in specs if (o := _outcomes(spec)) and set(o) != {"passed"}}
    assert not failed, f"{clause}: enforcing tests that did not pass: {failed}"


def test_c12_stays_a_strict_xfail_with_its_reason():
    marks = _decorators(C12_XFAIL)
    assert marks == ["pytest.mark.xfail(strict=True, reason=_C12)"], marks
    assert "Decision A" in _C12 and "owner-signed" in _C12
    assert _outcomes(C12_XFAIL) == ["xfailed"], _outcomes(C12_XFAIL)
