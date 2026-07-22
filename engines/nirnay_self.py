"""
Tattva — Nirnay-Swayam: self-referential ensemble for basket-free targets.
तत्त्व (Tattva) — "Principle / Essence"

NIRNAY-SWAYAM (स्वयम् — "self") — for targets routed to "self" mode by
data.constituents.get_nirnay_mode (commodities and individual stocks — the ones
better read on their own price than through a constituent/producer basket), the
Nirnay breadth read is formulated on the TARGET'S OWN OHLCV: a deterministic
ensemble of causal views spanning three diversity axes —

  • timescale     — MSF/MMR at 5 log-spaced lookback windows
  • information set — macro-anchored (MSF+MMR) vs pure price-action (MSF only)
  • mechanism     — MSF's own orthogonal components (momentum / structure /
                    flow) promoted to standalone voters

Each member is run through the UNCHANGED per-instrument pipeline
(engines.nirnay.run_full_analysis), producing the identical per-instrument
frame schema. engines.nirnay.aggregate_constituent_timeseries then turns
member votes into the identical breadth schema — nothing downstream
(polarity, calendar reindex, cross-validator, calibration, precedent, UI)
needs to know the "constituents" are views of one instrument.

See NIRNAY_SWAYAM_PLAN.md §1-§3 for the full design rationale, including the
three purpose-preservation invariants (independence from Aarambh, causality/
no repainting, basket-mode byte-identity) this module must not violate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from engines.nirnay import run_full_analysis


# ─── Member specification ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SwayamMember:
    """One causal view in the self-ensemble.

    ``name`` is used everywhere a constituent symbol is used today (dict
    key, drill-down selector) — so it must be unique within a member set.
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
    """Build the default 15-member Swayam grid (NIRNAY_SWAYAM_PLAN.md §3.2).

    Grid (for the default ``lengths``):
      • 5 timescale × macro-anchored  members: L{10,14,20,28,40}·FULL
      • 5 timescale × price-only      members: L{10,14,20,28,40}·PRICE
      • 3 component-isolation members (canonical length = middle of
        ``lengths``, macro-anchored): L{mid}·MOM, L{mid}·STRUCT, L{mid}·FLOW
      • 2 horizon-extreme component members (price-only): fastest·MOM,
        slowest·FLOW
    """
    if not lengths:
        raise ValueError("lengths must be non-empty")
    sorted_lengths = tuple(sorted(lengths))
    mid = sorted_lengths[len(sorted_lengths) // 2]
    fastest = sorted_lengths[0]
    slowest = sorted_lengths[-1]

    members: list[SwayamMember] = []
    for length in sorted_lengths:
        roc_len = _default_roc_len(length, roc_frac)
        members.append(SwayamMember(f"L{length}·FULL", length, roc_len, use_macro=True))
    for length in sorted_lengths:
        roc_len = _default_roc_len(length, roc_frac)
        members.append(SwayamMember(f"L{length}·PRICE", length, roc_len, use_macro=False))

    mid_roc = _default_roc_len(mid, roc_frac)
    members.append(SwayamMember(f"L{mid}·MOM", mid, mid_roc, use_macro=True, components=("momentum",)))
    members.append(SwayamMember(f"L{mid}·STRUCT", mid, mid_roc, use_macro=True, components=("structure",)))
    members.append(SwayamMember(f"L{mid}·FLOW", mid, mid_roc, use_macro=True, components=("flow",)))

    fastest_roc = _default_roc_len(fastest, roc_frac)
    slowest_roc = _default_roc_len(slowest, roc_frac)
    members.append(SwayamMember(f"L{fastest}·MOM", fastest, fastest_roc, use_macro=False, components=("momentum",)))
    members.append(SwayamMember(f"L{slowest}·FLOW", slowest, slowest_roc, use_macro=False, components=("flow",)))

    return tuple(members)


# Default grid instance — config wires NIRNAY_SWAYAM_LENGTHS/ROC_FRAC through
# this factory so the research sweep (research/nirnay_swayam_study.py) can
# vary the grid without touching this module.
SWAYAM_MEMBERS: tuple[SwayamMember, ...] = default_swayam_members()

# Canonical display order for the Nirnay tab's "Select View" drop-down
# (member-grid order, not alphabetical — alphabetical would separate
# L10·FULL and L10·PRICE from their timescale siblings).
SWAYAM_MEMBER_ORDER: tuple[str, ...] = tuple(m.name for m in SWAYAM_MEMBERS)


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

    Returns ``{member.name: result_df}`` — the exact shape Phase 3 builds
    for basket constituents, ready for
    ``engines.nirnay.aggregate_constituent_timeseries`` unchanged.

    ``macro_cols`` MUST already be leakage-filtered by the caller (drop the
    target's own column + its excluded-predictor near-replicas — see
    ``data.constituents.swayam_macro_columns``); this function does not
    re-filter.
    """
    if target_ohlcv is None or target_ohlcv.empty:
        return {}

    # Join macro data once, exactly as the basket-mode Phase-3 loop does
    # (ffill macro cols onto the target's calendar).
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
            result_df, _drivers = run_full_analysis(
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
            # A single bad member must not kill the ensemble — mirrors the
            # basket-mode per-constituent try/except in app.py.
            continue
        if progress_cb is not None:
            progress_cb(i + 1, total, member.name)

    return frames


# ─── Effective-breadth diagnostic (display-only) ────────────────────────────


def effective_member_count(frames: dict[str, pd.DataFrame]) -> float:
    """Effective number of independent votes among ensemble members.

    N_eff = (sum of eigenvalues)^2 / (sum of squared eigenvalues) of the
    correlation matrix of member ``Unified`` oscillators — full-sample,
    DIAGNOSTIC ONLY. Never fed into the causal signal path, calibration, or
    ``expected_constituents``; it exists purely to disclose to the user that
    self-ensemble members are correlated by construction (NIRNAY_SWAYAM_PLAN.md
    §1.4) — far more than an independent-name basket.

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
