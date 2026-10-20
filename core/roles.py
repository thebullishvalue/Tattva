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
    Role("grade:claim", "GRADE", "claim", "Claim grade", "claim",
         "trust (as of each session, shrunk) and Mūla's claim-grade card (full history)"),
    Role("grade:sizer:swayam", "GRADE", "sizer:swayam", "Swayam sizer grade", "sizer:swayam",
         "β of the Swayam sizer: its z against d·Y on native claim rows"),
    Role("grade:sizer:reversion", "GRADE", "sizer:reversion", "Reversion sizer grade", "sizer:reversion",
         "β of the reversion sizer: its z against d·Y on claim rows"),
    Role("evidence:gap_split", "EVIDENCE", "claim", "Who closes the gap?", None,
         "Overview: the measured split of gap closure between price and fair value"),
    Role("evidence:claim_unit", "EVIDENCE", "claim", "The claim's unit", None,
         "Overview: FVO's causal mean, SD and RICH:CHEAP balance against the C12 contract"),
    Role("evidence:claim_side", "EVIDENCE", "claim", "The claim by side", None,
         "Overview: hit against the side-matched base, read off edges claim:buy and claim:sell"),
    Role("infra:asof", "INFRASTRUCTURE", None, "As of d*", None,
         "every verdict input is read at d*, Mūla's latest published session"),
)}

#: The verdict's chain rows, in decision order: the claim, what can void it,
#: what it has earned, what scales it, and the size that results.
CHAIN: tuple[str, ...] = ("claim", "precondition", "trust", "sizer:swayam", "sizer:reversion", "size")


def chain_roles(row: str) -> tuple[Role, ...]:
    """The roles one chain row reports: the row's own, or each ``row:<part>``."""
    parts = tuple(r for k, r in ROLES.items() if k.startswith(row + ":"))
    return parts if parts else (ROLES[row],)
