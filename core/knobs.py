"""
Tattva — the KNOBS registry: every number that can move the call, declared once.
तत्त्व (Tattva) — "Principle / Essence"

A knob is any constant whose value can change what Tattva issues: a cut that
voids a claim, the bar a grade must clear, the window an estimator needs, the
lag between a call and the close it can trade at. Each is registered here with

  * ``kind``             — ``structural`` (fixed by an argument, never tuned),
                           ``structural-inherited`` (an older value kept
                           without a study behind it: it is NOT evidence), or
                           ``measured`` (set by a committed study, named with
                           its run date);
  * ``consumer``         — the role it moves (``estimand``, ``claim``,
                           ``grade``, ``trust``, ``sizer:<name>``, ...);
  * ``horizon_coupling`` — whether it is expressed in units of the outcome
                           horizon ``h`` or is horizon-free, and why (U54);
  * ``reason``           — the argument for its value, or the study and date.

``knob(name)`` is how code reads one. ``register_measured`` is the ONLY writer
of a measured entry: the pooled-prior study's output is read as data and
registered through it, so no measured value is ever typed by hand.

Engine and data constants that are already named and documented where they
live are registered BY REFERENCE (``ref``) rather than copied, so the registry
cannot drift from the value the code runs on.

No Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core import config as _cfg

KnobKind = Literal["structural", "structural-inherited", "measured"]
_KINDS: frozenset[str] = frozenset({"structural", "structural-inherited", "measured"})

#: Repository root: a measured knob's study is a tracked path relative to it.
_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Knob:
    """One registered constant. See the module docstring for the fields."""

    name: str
    value: Any
    kind: KnobKind
    consumer: str
    horizon_coupling: str
    reason: str
    study: str | None = None
    date: str | None = None
    ref: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"knob {self.name}: unknown kind {self.kind!r}")
        for field in ("consumer", "horizon_coupling", "reason"):
            if not str(getattr(self, field) or "").strip():
                raise ValueError(f"knob {self.name}: {field} is required")
        if self.kind == "measured" and not (self.study and self.date):
            raise ValueError(f"knob {self.name}: a measured knob names its study and date")


KNOBS: dict[str, Knob] = {}

#: Names a study may write: the placeholders below. Any other name that is
#: already registered is structural and can never be overwritten by a result.
_STUDY_SET: set[str] = set()

_SET_BY_STUDY = "set by studies/pooled_priors.py"


def knob(name: str) -> Any:
    """The value of a registered knob. An unregistered name raises."""
    try:
        return KNOBS[name].value
    except KeyError:
        raise KeyError(f"unregistered knob {name!r}: every decision constant is "
                       f"declared in core.knobs") from None


def _register(name: str, value: Any, kind: KnobKind, *, consumer: str,
              horizon_coupling: str, reason: str, ref: str | None = None) -> None:
    if name in KNOBS:
        raise ValueError(f"knob {name} registered twice")
    KNOBS[name] = Knob(name, value, kind, consumer, horizon_coupling, reason, ref=ref)


def _by_ref(name: str, *, consumer: str, horizon_coupling: str, reason: str) -> None:
    """Register a constant that lives in ``core.config`` without copying it."""
    _register(name, getattr(_cfg, name), "structural", consumer=consumer,
              horizon_coupling=horizon_coupling, reason=reason, ref=f"core.config.{name}")


def _placeholder(name: str, value: Any, *, consumer: str, horizon_coupling: str) -> None:
    """A default the pooled-prior study replaces with a measured value."""
    _register(name, value, "structural", consumer=consumer,
              horizon_coupling=horizon_coupling, reason=_SET_BY_STUDY)
    _STUDY_SET.add(name)


def register_measured(name: str, value: Any, *, consumer: str, study: str, date: str) -> None:
    """Record a value set by a committed study — the only writer of measured knobs.

    ``study`` is the study script's path relative to the repository root (or
    absolute) and must exist: a measured value whose code is not in the repo is
    not reproducible, and is refused. ``name`` must be new or one of the
    placeholders the study owns; a structural knob is never overwritten.
    """
    old = KNOBS.get(name)
    if old is not None and old.kind != "measured" and name not in _STUDY_SET:
        raise ValueError(f"knob {name} is {old.kind}; a study cannot overwrite it")
    path = Path(study)
    if not (path if path.is_absolute() else _ROOT / path).is_file():
        raise ValueError(f"knob {name}: study {study!r} is not a file in the repository")
    coupling = (old.horizon_coupling if old is not None
                else "measured at the study's h and L: one priors file per (h, L)")
    KNOBS[name] = Knob(name, value, "measured", consumer, coupling,
                       f"measured by {study} on {date}", study=str(study), date=str(date))


# ════════════════════════════════════════════════════════════════════════
# The estimand
# ════════════════════════════════════════════════════════════════════════

_register(
    "EXECUTION_LAG", 1, "structural",
    consumer="estimand: Y_t = log P(t+L+h) − log P(t+L), in every grade",
    horizon_coupling="horizon-free: the sessions between publishing after the close of t "
                     "and the first close the call can trade at, whatever h",
    reason="a call computed after the close of t cannot trade at P(t); pending owner C1 "
           "amendment; 0 on veto (and the study re-run: the priors file records L)")

# ════════════════════════════════════════════════════════════════════════
# The grade (core.grade)
# ════════════════════════════════════════════════════════════════════════

_register(
    "DRIFT_MIN_RESOLVED", 60, "structural", consumer="grade: the causal drift μ_s and up-rate p_s",
    horizon_coupling="horizon-free: counted in resolved rows, not in windows of h",
    reason="the drift-neutral target and the side-matched base are undefined until 60 "
           "forward returns have resolved; those rows are not graded")
_register(
    "GRADE_SHIFT_MARGIN_H", 5, "structural", consumer="grade: the shift-null SE",
    horizon_coupling="in units of h: shifts within 5h of the true alignment are excluded",
    reason="a shift shorter than a few horizons still overlaps the true pairing through the "
           "persistence of the lean and of overlapping h-day returns, so it is not null")
_register(
    "GRADE_MIN_PAIRS_H", 12, "structural", consumer="grade: the shift-null SE",
    horizon_coupling="in units of h: 12h graded pairs",
    reason="below 12h pairs the admissible shifts [5h, N − 5h] span under 2h, too few to "
           "estimate the null's spread; the SE (and t) are NaN")
_register(
    "GRADE_MIN_WIN", 5, "structural", consumer="grade: the offset-averaged rank IC",
    horizon_coupling="in windows of h: each phase offset needs this many pairs (5h rows)",
    reason="a rank correlation of fewer than five pairs is not a number; 2.8.0's floor, kept")
_register(
    "GRADE_WARMUP_WIN", 30, "structural", consumer="grade: the as-of grade (expanding_edge)",
    horizon_coupling="in windows of h: n_win = graded rows // h",
    reason="at 30 independent windows the shift-null SE is within about 10–19% of the "
           "truth even for persistence φ ≥ 0.98 (simulated); before it the grade is NaN")
_register(
    "GRADE_N_BLOCKS", 6, "structural", consumer="grade: the durability blocks",
    horizon_coupling="horizon-free: a count of consecutive stretches of history",
    reason="one good period is not an edge; six periods of a decade are about a "
           "market cycle each (2.8.0's value)")
_register(
    "MIN_BLOCK_OBS", 8, "structural", consumer="grade: the durability blocks",
    horizon_coupling="in windows of h: non-overlapping windows inside one block",
    reason="a block with fewer windows reports no IC: eight is already an SE of about 0.4")

# ════════════════════════════════════════════════════════════════════════
# Trust and the one strength map (core.trust)
# ════════════════════════════════════════════════════════════════════════

_register(
    "STRENGTH_T0", 1.645, "structural", consumer="trust and every sizer β: f(t) opens here",
    horizon_coupling="horizon-free: a t-statistic",
    reason="a one-sided 5% false-open per look under the null (simulated 0.050); T0 = 1 "
           "would open 13–20% of null instruments")
_register(
    "STRENGTH_T1", 2.5, "structural", consumer="trust and every sizer β: f(t) is full here",
    horizon_coupling="horizon-free: a t-statistic",
    reason="full strength at the one-sided 0.6% point; the calibrated SE leaves no "
           "conservatism to buy back with a higher bar")
_register(
    "POOL_MIN_MEMBERS", 3, "structural", consumer="trust: the prior's scope",
    horizon_coupling="horizon-free: a member count",
    reason="a class-LOO prior needs at least 3 other members with n_win ≥ "
           "GRADE_WARMUP_WIN; with fewer, the prior is global-LOO")
_register(
    "SIZER_BETA_MAX", 1.0, "structural", consumer="sizer:*: β = SIZER_BETA_MAX · f(t_post)",
    horizon_coupling="horizon-free: an exponent on exp(β·z)",
    reason="at β = 1 the realistic Swayam z range gives about ×0.6–×1.7: a fully skilled "
           "sizer can scale the claim, never veto or invert it")
_register(
    "SIZER_Z_CLIP", 1.0, "structural", consumer="sizer:*: z clipped to ±SIZER_Z_CLIP",
    horizon_coupling="horizon-free: z is a share of views or 2·P − 1",
    reason="both sizer inputs live in [−1, 1] by construction; the clip only guards the range")
_register(
    "TIER_MIN_WIN", 20, "structural", consumer="composer: Decision C (tier words)",
    horizon_coupling="in windows of h: n/h per causal size tercile",
    reason="a tercile's payoff on fewer than 20 independent windows cannot show monotonicity")

# ════════════════════════════════════════════════════════════════════════
# The semantic cuts (core.semantics)
# ════════════════════════════════════════════════════════════════════════

_register(
    "CLAIM_SIGMA", 0.5, "structural", consumer="precondition:claim_sigma (P4) and the claim",
    horizon_coupling="horizon-free: σ units of Mūla's oscillator",
    reason="|FVO| below it is inside the normal day-to-day noise of price around fair "
           "value: no directional claim is made")
_register(
    "DEEP_SIGMA", 2.0, "structural", consumer="claim: the VERY CHEAP / VERY RICH state words",
    horizon_coupling="horizon-free: σ units of Mūla's oscillator",
    reason="|FVO| at or beyond it is a deep mispricing; valuation_lean saturates there")
_register(
    "SWAYAM_DEAD", 0.10, "structural", consumer="sizer:swayam: the confirms / split / contradicts words",
    horizon_coupling="horizon-free: a share of Swayam's views (net breadth)",
    reason="a net breadth inside ±10 points is under one view in ten either way: split; "
           "the one Swayam dead band (U39), formerly hard-coded four times")

# ════════════════════════════════════════════════════════════════════════
# Swayam's kernel (engines.swayam; values live in core.config)
# ════════════════════════════════════════════════════════════════════════

_INHERITED = "2.7 |IC| study not retained, not re-validated, differences < 1 SE"
for _name, _consumer, _coupling, _extra in (
    ("SWAYAM_OVERSOLD", "sizer:swayam input: a view's oversold condition (Unified_Osc ≤ it)",
     "horizon-free: oscillator units on the ±10 scale", ""),
    ("SWAYAM_OVERBOUGHT", "sizer:swayam input: a view's overbought condition (Unified_Osc ≥ it)",
     "horizon-free: oscillator units on the ±10 scale", ""),
    ("SWAYAM_LENGTHS", "sizer:swayam input: the view bank's timescale axis",
     "horizon-free by declaration: indicator lookbacks in sessions, not rescaled with h", ""),
    ("SWAYAM_ROC_FRAC", "sizer:swayam input: each view's ROC length as a share of its length",
     "horizon-free: a fraction of each view's own length", ""),
    ("SWAYAM_MMR_NUM_VARS", "sizer:swayam input: macro drivers in each view's MMR",
     "horizon-free: a count of drivers", ""),
    ("SWAYAM_REGIME_SENSITIVITY", "sizer:swayam input: the MSF/MMR adaptive blend's gain",
     "horizon-free: a sigmoid gain", ""),
    ("SWAYAM_BASE_WEIGHT", "sizer:swayam input: MSF share of the fixed half of the MSF/MMR blend",
     "horizon-free: a blend weight", "; consequence: price-only members never vote"),
):
    _register(_name, getattr(_cfg, _name), "structural-inherited", consumer=_consumer,
              horizon_coupling=_coupling, reason=_INHERITED + _extra, ref=f"core.config.{_name}")

# ════════════════════════════════════════════════════════════════════════
# Mūla's valuation and the target-history precondition (by reference)
# ════════════════════════════════════════════════════════════════════════

_by_ref("FVO_BURN_IN", consumer="claim: sessions absorbed before a valuation is published",
        horizon_coupling="horizon-free: one year of covariance memory",
        reason="the weight an EW correlation matrix over ~200 instruments needs before the "
               "Marchenko–Pastur edge means anything")
_by_ref("FVO_MIN_PRINTS", consumer="claim: prints a predictor needs before it enters the cross-section",
        horizon_coupling="horizon-free: a second moment's sample size",
        reason="a second moment is not estimable from less; applied forward in time")
_by_ref("FVO_VALUATION_DELTAS", consumer="claim: the valuation regression's discount grid",
        horizon_coupling="horizon-free: coefficient memories of years",
        reason="a modelling commitment to slow (long-run) relations; likelihood-scoring a "
               "level regression's discount is degenerate (it picks fair value = price)")
_by_ref("MIN_DATA_POINTS", consumer="infrastructure: the target-history precondition (data.prep)",
        horizon_coupling="horizon-free: sessions of the target's own prints",
        reason="a run needs this much target history; prepare_panel raises below it")

# ════════════════════════════════════════════════════════════════════════
# Placeholders the pooled-prior study replaces (register_measured)
# ════════════════════════════════════════════════════════════════════════

_STUDY_COUPLING = "measured at the study's h and L: one priors file per (h, L)"
_placeholder("SIZERS_ENABLED", True,
             consumer="composer: β_i = SIZER_BETA_MAX · f(t_post,i) · SIZERS_ENABLED (Decision B)",
             horizon_coupling=_STUDY_COUPLING)
_placeholder("TIER_RULE", "none", consumer="composer: tier words on the size (Decision C)",
             horizon_coupling=_STUDY_COUPLING)
_placeholder("CLAIM_CENTRING", "none",
             consumer="claim: Decision A, recorded by the study, never applied without an owner-signed change",
             horizon_coupling=_STUDY_COUPLING)
_placeholder("POOLED_PRIOR", None, consumer="trust: the class prior file behind every posterior",
             horizon_coupling=_STUDY_COUPLING)

del _name, _consumer, _coupling, _extra
