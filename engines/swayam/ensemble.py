"""
Tattva — Swayam: the self-referential breadth ensemble.
तत्त्व (Tattva) — "Principle / Essence"

SWAYAM (स्वयम् — "self"). Breadth read on the TARGET'S OWN price, as a
deterministic ensemble of causal views spanning three diversity axes —

  • timescale       — MSF/MMR at log-spaced lookback windows
  • information set — macro-anchored (MSF+MMR) vs pure price-action (MSF only)
  • mechanism       — MSF's own components (momentum at the fastest
                      timescale, flow at the slowest) as standalone views

This is now the system's ONLY breadth formulation. It replaced a basket read
(run the same kernel across an index's constituents, or a curated proxy basket
for a commodity/FX target) that has since been removed. The reason is not
performance: a basket is a *proxy*, and every proxy needed hand-curation —
which names, capped at how many, co-directional or inverse (there was a
polarity flag to flip the ones that ran the other way). Those are judgement
calls the data never got to overrule, they went stale as constituents changed,
and they cost a 503-name fetch on a large index. Asking the same question of
the target's own price makes "breadth" mean agreement among views of the
thing itself, which needs no curation and no proxy assumption.

What it is, stated plainly: a SHORT-WINDOW MACRO-RESIDUAL breadth. The views
that can vote are the macro-anchored ones, and their vote is driven by the
MMR deviation of the target from a rolling regression on the same macro panel
Mūla values against; the price-only views cannot reach the ±5 cut at the
shipped blend weight (``kernel.can_vote``; :func:`voting_views` counts them).
It is not a read of the instrument's own internals, and its views are not
independent of each other: views of one price are correlated by construction
(``effective_member_count``).

Each member runs through the UNCHANGED per-series kernel
(``swayam.kernel.run_full_analysis``), producing the identical frame schema,
and ``aggregate_views`` reduces member votes, every view counted once, into
the breadth the Swayam confirmation sizer reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from engines.swayam.kernel import can_vote, run_full_analysis


# ─── Member specification ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SwayamMember:
    """One causal view in the self-ensemble.

    ``name`` keys the view everywhere it appears (the frame dict and the Swayam
    page's drill-down selector) — so it must be unique within a member set.
    """

    name: str
    length: int
    roc_len: int
    use_macro: bool = True
    components: tuple[str, ...] | None = None
    # Members whose ONLY active components are volume-dependent (flow) are
    # dropped by build_swayam_frames on near-zero-volume instruments (index
    # levels, some FX/commodity futures) — see _is_volume_dependent below.


def _is_volume_dependent(member: SwayamMember) -> bool:
    """True if the member's signal would run ~neutral with no real volume.

    ``flow`` (accumulation/distribution + regime counting) is the only MSF
    component that depends on Volume; ``momentum`` and ``structure`` are
    price/OHLC-only (structure's microstructure sub-term also uses Volume,
    but only inside a composite alongside volume-free trend terms, so it
    degrades gracefully rather than going fully neutral — matching the
    documented USD/INR ``=X``-cross behaviour: no special-case needed
    there). Only a member whose components are *exclusively* ``flow``
    needs to be dropped outright.
    """
    return member.components == ("flow",)


def _default_roc_len(length: int, frac: float = 0.7) -> int:
    return max(5, round(frac * length))


def default_swayam_members(
    lengths: tuple[int, ...] = (10, 14, 20, 28, 40),
    roc_frac: float = 0.7,
) -> tuple[SwayamMember, ...]:
    """Build the default 12-member Swayam grid.

    Grid (for the default ``lengths``):
      • 5 timescale × macro-anchored  members: L{10,14,20,28,40}·FULL
      • 5 timescale × price-only      members: L{10,14,20,28,40}·PRICE
      • 2 horizon-extreme component members (price-only): fastest·MOM,
        slowest·FLOW

    (Three macro-anchored component-isolation members at the middle length,
    L{mid}·MOM / ·STRUCT / ·FLOW, were removed without retuning: they shared
    L{mid}·FULL's MMR exactly (correlation 1.000), and the MMR is what casts a
    macro-anchored view's vote, so they counted the middle window four times.)
    """
    if not lengths:
        raise ValueError("lengths must be non-empty")
    sorted_lengths = tuple(sorted(lengths))
    fastest = sorted_lengths[0]
    slowest = sorted_lengths[-1]

    members: list[SwayamMember] = []
    for length in sorted_lengths:
        roc_len = _default_roc_len(length, roc_frac)
        members.append(SwayamMember(f"L{length}·FULL", length, roc_len, use_macro=True))
    for length in sorted_lengths:
        roc_len = _default_roc_len(length, roc_frac)
        members.append(SwayamMember(f"L{length}·PRICE", length, roc_len, use_macro=False))

    fastest_roc = _default_roc_len(fastest, roc_frac)
    slowest_roc = _default_roc_len(slowest, roc_frac)
    members.append(SwayamMember(f"L{fastest}·MOM", fastest, fastest_roc, use_macro=False, components=("momentum",)))
    members.append(SwayamMember(f"L{slowest}·FLOW", slowest, slowest_roc, use_macro=False, components=("flow",)))

    return tuple(members)


# Default grid instance — config wires SWAYAM_LENGTHS/ROC_FRAC through
# this factory, so a different grid is a config change, never an edit here.
# The frames come back in grid order, which is the order the Swayam page's
# view selector lists them in.
SWAYAM_MEMBERS: tuple[SwayamMember, ...] = default_swayam_members()


def member_can_vote(member: SwayamMember, cfg) -> bool:
    """Whether ``member`` can reach the ±vote cut under ``cfg`` (``kernel.can_vote``).

    A member that cannot is flagged here rather than silently diluting the
    breadth: it sits in ``Total_Analyzed`` but never in ``N_Voting``.
    """
    return can_vote(member.use_macro, cfg.swayam_base_weight, cfg.swayam_oversold,
                    cfg.swayam_overbought)


def voting_views(cfg) -> tuple[int, int]:
    """``(N, M)``: of the M views ``cfg``'s bank holds, the N that can reach the ±vote cut.

    The configured bank's count (the run console's line; the Swayam page reads
    the run's own published ``N_Voting`` / ``Total_Analyzed``). ``cfg`` is an
    ``InstrumentConfig``; the bank is ``default_swayam_members`` on its
    lengths. A run on an instrument with no volume drops its flow-only views
    (:func:`build_swayam_frames`), which can never vote at the shipped blend
    weight, so N is unchanged and the run's own ``Total_Analyzed`` can be
    smaller than M.
    """
    members = default_swayam_members(cfg.swayam_lengths, cfg.swayam_roc_frac)
    return sum(member_can_vote(m, cfg) for m in members), len(members)


# ─── Ensemble builder ────────────────────────────────────────────────────────


def build_swayam_frames(
    target_ohlcv: pd.DataFrame,
    macro_df: pd.DataFrame | None,
    macro_cols: list[str],
    members: tuple[SwayamMember, ...] = SWAYAM_MEMBERS,
    *,
    regime_sensitivity: float,
    base_weight: float,
    num_vars: int = 5,
    oversold: float = -5.0,
    overbought: float = 5.0,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run run_full_analysis once per member on the SAME target OHLCV frame.

    Returns ``{member.name: result_df}``, ready for
    ``swayam.aggregate_views``.

    ``macro_cols`` MUST already be leakage-filtered by the caller (drop the
    target's own column + its excluded-predictor near-replicas — see
    ``core.config.swayam_macro_columns``); this function does not
    re-filter.
    """
    if target_ohlcv is None or target_ohlcv.empty:
        return {}

    # Join the macro driver pool once (ffilled onto the target's calendar);
    # every macro-anchored member reads the same joined frame.
    base = target_ohlcv.copy()
    has_macro_source = macro_df is not None and not macro_df.empty and bool(macro_cols)
    if has_macro_source:
        base = base.join(macro_df, how="left")
        base[macro_cols] = base[macro_cols].ffill()

    # Volume-degeneracy guard: drop flow-only members on near-zero-volume
    # instruments (e.g. index levels with no yfinance volume) — the same
    # documented degradation as USD/INR's `=X` crosses, applied here as an
    # explicit drop instead of a silent near-neutral vote.
    vol = target_ohlcv["Volume"] if "Volume" in target_ohlcv.columns else None
    vol_ok = bool((vol.fillna(0) > 0).mean() >= 0.5) if vol is not None else False

    active_members = [m for m in members if vol_ok or not _is_volume_dependent(m)]

    frames: dict[str, pd.DataFrame] = {}
    total = len(active_members)
    for i, member in enumerate(active_members):
        try:
            member_macro_cols = macro_cols if (member.use_macro and has_macro_source) else []
            result_df = run_full_analysis(
                base,
                length=member.length,
                roc_len=member.roc_len,
                regime_sensitivity=regime_sensitivity,
                base_weight=base_weight,
                num_vars=num_vars,
                oversold=oversold,
                overbought=overbought,
                macro_columns=member_macro_cols,
                components=member.components,
            )
            frames[member.name] = result_df
        except Exception:
            # A single bad member must not kill the ensemble.
            continue
        if progress_cb is not None:
            progress_cb(i + 1, total, member.name)

    return frames


# ─── Effective-breadth diagnostic (display-only) ────────────────────────────


def effective_member_count(frames: dict[str, pd.DataFrame]) -> float:
    """Effective number of independent votes among ensemble members.

    N_eff = (sum of eigenvalues)^2 / (sum of squared eigenvalues) of the
    correlation matrix of member ``Unified`` oscillators — full-sample,
    DIAGNOSTIC ONLY (a full-history statistic, so say "over the full history"
    wherever it is shown). Never fed into the causal signal path; it exists
    to disclose that views of ONE price are correlated by construction, so a
    dozen views are worth far fewer than a dozen votes.

    Returns ``0.0`` for fewer than 2 usable members.
    """
    series = []
    for df in frames.values():
        if df is not None and "Unified" in df.columns:
            series.append(df["Unified"])
    if len(series) < 2:
        return float(len(series))

    wide = pd.concat(series, axis=1, join="inner").dropna()
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        return float(wide.shape[1])

    corr = wide.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = float(eigvals.sum())
    s2 = float((eigvals ** 2).sum())
    if s2 <= 0:
        return float(len(series))
    return (s1 * s1) / s2
