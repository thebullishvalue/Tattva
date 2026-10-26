"""
Tattva — Swayam breadth engine.
तत्त्व (Tattva) — "Principle / Essence"

    kernel.py    per-series MSF + MMR (the analysis kernel; formerly the Swayam
                 engine, minus its basket orchestration), and the equal-weight
                 reduction of member votes into breadth and its one lean
    ensemble.py  the self-referential view bank built on that kernel, and which
                 of its views can reach the vote at all

Swayam is the system's only breadth formulation. The basket read it replaced
required hand-curated proxy constituents per target; see ensemble.py's
docstring for why that had to go.
"""

from .ensemble import (SwayamMember, build_swayam_frames, default_swayam_members,
                       effective_member_count, member_can_vote, voting_views)
from .kernel import aggregate_views, calculate_mmr, calculate_msf, run_full_analysis

__all__ = [
    "SwayamMember",
    "build_swayam_frames",
    "default_swayam_members",
    "effective_member_count",
    "member_can_vote",
    "voting_views",
    "aggregate_views",
    "calculate_msf",
    "calculate_mmr",
    "run_full_analysis",
]
