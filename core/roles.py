"""
Tattva — the role registry: what every part of the decision is FOR, declared once.
तत्त्व (Tattva) — "Principle / Essence"

Tattva issues one signed call, ``w_t = d_t · size_t`` on the h-session return
from the next close. Every part that can be read on a page has exactly one of
these roles in producing, grading or auditing it:

  * ``CALL``           — the call itself and its size;
  * ``CLAIM``          — Mūla's direction, ``d = claim_direction(FVO)``: the
                         only source of sign;
  * ``PRECONDITION``   — a binary condition that can void the claim and never
                         sizes it;
  * ``SIZER``          — a factor ``exp(β·z)`` whose β is earned by its own
                         grade (×1.00 while unskilled or absent);
  * ``GRADE``          — a measured skill number (core.grade) and what it moves
                         (trust is the claim's grade mapped once);
  * ``EVIDENCE``       — shown to audit one role-holder; never moves the call;
  * ``INFRASTRUCTURE`` — what guarantees one as-of row and causality.

``target`` is the role-holder a role serves or audits, ``grade_key`` the
``edges`` entry that grades it, ``consumer`` what reads it. ``CHAIN`` is the
decision order the verdict's chain rows follow, on every page.

``CARDS`` registers every card a page draws — each KPI item, section and Data
column — to ``(role key, target)``: the role it shows and the role-holder it
serves or audits, both keys of ``ROLES``. A card's chip is fixed text per role
kind (``CHIPS``), so an EVIDENCE card always reads 'EVIDENCE→<target> · does
not move the call' and no page words its own role. ``ui.components`` refuses
to draw a KPI or a section whose card is not registered here.

No Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoleKind = Literal["CALL", "GRADE", "CLAIM", "PRECONDITION", "SIZER", "EVIDENCE", "INFRASTRUCTURE"]


@dataclass(frozen=True)
class Role:
    """One registered role. See the module docstring for the fields."""

    key: str
    kind: RoleKind
    target: str | None
    title: str
    grade_key: str | None
    consumer: str


ROLES: dict[str, Role] = {r.key: r for r in (
    Role("call", "CALL", None, "The call", "call",
         "every page: the command bar, the sidebar and Overview read verdict_from_row(call_ts at d*)"),
    Role("claim", "CLAIM", "call", "Mūla claim", "claim",
         "core.call: d = claim_direction(FVO) when every precondition holds, else 0"),
    Role("precondition", "PRECONDITION", "claim", "Preconditions", None,
         "core.call: the chain row that reports P1–P4"),
    Role("precondition:published", "PRECONDITION", "claim", "Published", None,
         "core.call P1: FVO finite and the session not withheld; else NO CALL with the reason"),
    Role("precondition:printed", "PRECONDITION", "claim", "Target printed", None,
         "core.call P2: the target printed and, at d*, is fewer than h sessions behind"),
    Role("precondition:tradeable", "PRECONDITION", "claim", "Tradeable", None,
         "core.call P3: core.instruments tradeable; a valuation ratio is NO CALL"),
    Role("precondition:claim_sigma", "PRECONDITION", "claim", "Outside the noise", None,
         "core.call P4: |FVO| ≥ CLAIM_SIGMA; else NO CLAIM"),
    Role("trust", "GRADE", "claim", "Trust", "claim",
         "core.call: trust = f(t_post of the claim), the claim's earned base size; never binding"),
    Role("sizer:swayam", "SIZER", "call", "Swayam confirmation", "sizer:swayam",
         "core.call: ×exp(β·z), z = d·Lean on a native Swayam row, else 0"),
    Role("sizer:reversion", "SIZER", "call", "Reversion (ECM)", "sizer:reversion",
         "core.call: ×exp(β·z), z = 2·MRProb − 1"),
    Role("size", "CALL", "call", "Size", "call",
         "core.call: size = trust × ∏ factor, a unitless index (1.0 = a fully trusted "
         "claim with neutral sizers)"),
    Role("grade:call", "GRADE", "call", "Call grade (as issued)", "call",
         "Overview's GRADE→call strip and the sidebar: the call graded on the sessions it was issued"),
    Role("grade:call:prospective", "GRADE", "call", "Call grade (prospective)", "call:prospective",
         "Grades: the call graded only on the sessions after the priors file's run date, the "
         "one clean out-of-sample record"),
    Role("grade:claim", "GRADE", "claim", "Claim grade", "claim",
         "trust (as of each session, shrunk) and Mūla's claim-grade card (full history)"),
    Role("grade:sizer:swayam", "GRADE", "sizer:swayam", "Swayam sizer grade", "sizer:swayam",
         "β of the Swayam sizer: its z against d·Y on native claim rows; the Swayam page's "
         "contradicts / split / confirms payoff table"),
    Role("grade:sizer:reversion", "GRADE", "sizer:reversion", "Reversion sizer grade", "sizer:reversion",
         "β of the reversion sizer: its z against d·Y on claim rows"),
    Role("evidence:gap_split", "EVIDENCE", "claim", "Who closes the gap?", None,
         "Overview and Mūla: the measured split of gap closure between price and fair value"),
    Role("evidence:claim_unit", "EVIDENCE", "claim", "The claim's unit", None,
         "Mūla: FVO's causal mean, SD and RICH:CHEAP balance against the C12 contract"),
    Role("evidence:gap_stationarity", "EVIDENCE", "claim", "Gap stationarity", None,
         "Mūla: ADF and KPSS on the published gap through d*, one word "
         "(core.semantics.gap_stationarity), a full-history diagnostic"),
    Role("evidence:claim_side", "EVIDENCE", "claim", "The claim by side", None,
         "Overview's per-side line, Mūla and the Grades table: hit against the side-matched "
         "base, read off edges claim:buy and claim:sell"),
    Role("evidence:fair_value", "EVIDENCE", "claim", "Price and fair value", None,
         "Mūla: the fitted level, its 95% band and the oscillator the claim is read off"),
    Role("evidence:model_fit", "EVIDENCE", "claim", "Model fit", None,
         "Mūla: OOS R², the cross-section's size and the fair-value level's own spread"),
    Role("evidence:drivers", "EVIDENCE", "claim", "What drives fair value", None,
         "Mūla: each asset-class block's share of the fitted level, read off the engine"),
    Role("evidence:swayam_lean", "EVIDENCE", "sizer:swayam", "Swayam's lean, standalone",
         "evidence:swayam_lean",
         "the Swayam page: the published Lean over time, and graded on its own on the valued "
         "sessions with a native Swayam row"),
    Role("evidence:view_bank", "EVIDENCE", "sizer:swayam", "The view bank", None,
         "the Swayam page: which views can vote, how they are counted, how many they are "
         "worth, and each view's own oscillators"),
    Role("evidence:ecm_drift", "EVIDENCE", "sizer:reversion", "Gap-implied move",
         "evidence:ecm_drift",
         "the Mūla page: the ECM's gap-implied h-session move (MulaDriftPct, the drift "
         "excluded), graded on its own on the valued sessions"),
    Role("grades", "GRADE", "call", "Every grade", None,
         "the Grades page: one table of every grade, the trust and β paths, the issued-size "
         "terciles and the pooled-prior study's results"),
    Role("infra:asof", "INFRASTRUCTURE", None, "As of d*", None,
         "every verdict input is read at d*, Mūla's latest published session"),
    Role("infra:data", "INFRASTRUCTURE", None, "The data", None,
         "the fetch from the epoch, the prepared target-printed spine, the engine's completeness "
         "gate and the caches and circuits behind them"),
    Role("infra:priors", "INFRASTRUCTURE", "trust", "Pooled priors", None,
         "core.trust.load_priors: the committed pooled-prior study read as data — the class "
         "priors every posterior shrinks toward, its decisions and its checks"),
    Role("infra:registry", "INFRASTRUCTURE", None, "The registries", None,
         "core.knobs, core.roles and core.instruments: every constant, role and instrument, "
         "declared once"),
    Role("infra:record", "INFRASTRUCTURE", None, "The record", None,
         "the Data page: every published series and call_ts, joined by date, exported"),
)}

#: The chip a card carries, fixed per role kind (``{target}`` is the card's
#: target). No page writes its own: :func:`chip` is the only reader.
CHIPS: dict[str, str] = {
    "CALL": "CALL",
    "CLAIM": "CLAIM→{target}",
    "PRECONDITION": "PRECONDITION→{target} · voids the claim, never sizes it",
    "SIZER": "SIZER→{target} · ×1.00 until its own grade earns a weight",
    "GRADE": "GRADE→{target}",
    "EVIDENCE": "EVIDENCE→{target} · does not move the call",
    "INFRASTRUCTURE": "INFRASTRUCTURE→{target}",
}

#: Every card a page draws: ``card id → (role key, target)``, both ROLES keys.
#: An id is ``<page>:<block>``; one id may label several KPI items of a strip
#: that show the same role. Infrastructure serves the call, so its target is
#: ``call`` unless it serves one role-holder (the priors serve trust).
CARDS: dict[str, tuple[str, str]] = {
    # ── The landing page ────────────────────────────────────────────────
    "landing:coverage": ("infra:data", "call"),
    "landing:systems": ("infra:registry", "call"),
    "landing:returns": ("call", "call"),
    # ── Overview, in decision order ─────────────────────────────────────
    "overview:call": ("call", "call"),
    "overview:call_grade": ("grade:call", "call"),
    "overview:gap_split": ("evidence:gap_split", "claim"),
    "overview:history": ("call", "call"),
    "overview:terciles": ("grade:call", "call"),
    # ── Mūla: the claim, its grade, its evidence, the reversion sizer ───
    "mula:claim": ("claim", "call"),
    "mula:valuation": ("claim", "call"),
    "mula:grade": ("grade:claim", "claim"),
    "mula:sides": ("evidence:claim_side", "claim"),
    "mula:gap_split": ("evidence:gap_split", "claim"),
    "mula:fair_value": ("evidence:fair_value", "claim"),
    "mula:claim_unit": ("evidence:claim_unit", "claim"),
    "mula:gap_stationarity": ("evidence:gap_stationarity", "claim"),
    "mula:model_fit": ("evidence:model_fit", "claim"),
    "mula:drivers": ("evidence:drivers", "claim"),
    "mula:reversion": ("sizer:reversion", "call"),
    "mula:reversion_grade": ("grade:sizer:reversion", "sizer:reversion"),
    "mula:ecm_move": ("evidence:ecm_drift", "sizer:reversion"),
    # ── Swayam: the confirmation sizer, its grade, its evidence ─────────
    "swayam:sizer": ("sizer:swayam", "call"),
    "swayam:sizer_grade": ("grade:sizer:swayam", "sizer:swayam"),
    "swayam:buckets": ("grade:sizer:swayam", "sizer:swayam"),
    "swayam:breadth": ("evidence:swayam_lean", "sizer:swayam"),
    "swayam:bank": ("evidence:view_bank", "sizer:swayam"),
    "swayam:lean": ("evidence:swayam_lean", "sizer:swayam"),
    "swayam:views": ("evidence:view_bank", "sizer:swayam"),
    # ── Grades ──────────────────────────────────────────────────────────
    "grades:table": ("grades", "call"),
    "grades:paths": ("grades", "call"),
    "grades:terciles": ("grade:call", "call"),
    "grades:study": ("infra:priors", "trust"),
    # ── System ──────────────────────────────────────────────────────────
    "system:run": ("infra:asof", "call"),
    "system:priors": ("infra:priors", "trust"),
    "system:health": ("infra:data", "call"),
    "system:knobs": ("infra:registry", "call"),
    "system:roles": ("infra:registry", "call"),
    # ── Data: the record, and every column of it ────────────────────────
    "data:record": ("infra:record", "call"),
    "data:price": ("infra:data", "call"),
    "data:claim": ("claim", "call"),
    "data:fair_value": ("evidence:fair_value", "claim"),
    "data:gap": ("evidence:gap_split", "claim"),
    "data:model_fit": ("evidence:model_fit", "claim"),
    "data:published": ("precondition:published", "claim"),
    "data:printed": ("precondition:printed", "claim"),
    "data:precondition": ("precondition", "claim"),
    "data:claim_grade": ("grade:claim", "claim"),
    "data:trust": ("trust", "claim"),
    "data:swayam": ("sizer:swayam", "call"),
    "data:swayam_shares": ("evidence:swayam_lean", "sizer:swayam"),
    "data:reversion": ("sizer:reversion", "call"),
    "data:ecm_move": ("evidence:ecm_drift", "sizer:reversion"),
    "data:size": ("size", "call"),
    "data:call": ("call", "call"),
}

#: The verdict's chain rows, in decision order: the claim, what can void it,
#: what it has earned, what scales it, and the size that results.
CHAIN: tuple[str, ...] = ("claim", "precondition", "trust", "sizer:swayam", "sizer:reversion", "size")


def chain_roles(row: str) -> tuple[Role, ...]:
    """The roles one chain row reports: the row's own, or each ``row:<part>``."""
    parts = tuple(r for k, r in ROLES.items() if k.startswith(row + ":"))
    return parts if parts else (ROLES[row],)


def card_role(card: str) -> tuple[Role, str]:
    """``(role, target)`` of a registered card. An unregistered id raises."""
    try:
        role_key, target = CARDS[card]
    except KeyError:
        raise KeyError(f"unregistered card {card!r}: every card a page draws is declared in "
                       f"core.roles.CARDS") from None
    return ROLES[role_key], target


def role_tag(card: str) -> str:
    """A card's role in two words: 'CALL', or '<KIND>→<target>'."""
    role, target = card_role(card)
    return CHIPS[role.kind].split(" · ")[0].format(target=target)


def chip(card: str) -> str:
    """A card's chip: its role kind's fixed text, at its target."""
    role, target = card_role(card)
    return CHIPS[role.kind].format(target=target)
