"""
Tattva — study: the pooled class priors every posterior shrinks toward, and the
three decisions the catalogue makes (owner decision 2).
तत्त्व (Tattva) — "Principle / Essence"

    python -m studies.pooled_priors --out data/priors/pooled_priors_h10_L1.json
    python -m studies.pooled_priors --out OUT.json --members Gold,Silver --workdir DIR
    python -m studies.pooled_priors --out OUT.json --offline-fixture --allow-dirty   # plumbing, no network

PRE-REGISTRATION. Everything below was fixed, and committed, before the study
was first run on the network; ``git log`` on this file proves the order. Its
SHA-256 is written into the output as ``script_sha256`` and
``tests/test_priors.py`` holds the committed output to the committed script,
so a result cannot drift from the code that produced it.

UNIVERSE (38 members). Every tradeable catalogue target in ``core.instruments``
(28; 'Nifty 50 - PE' is a valuation ratio and is not called), in catalogue
order, plus STUDY_STOCKS, chosen by free-float size at design time with no
grade in view: AAPL, MSFT, JPM, XOM, JNJ (stock_us) and RELIANCE, TCS,
HDFCBANK, INFY, ITC (stock_india), registered under the names the app gives a
typed symbol ('AAPL (US)', 'RELIANCE (NSE)'). Classes: commodity 8, fx 2,
crypto 7, india_index 8, us_index 3, stock_us 5, stock_india 5. Jeera has no
OHLCV, so it has no Swayam reading: it enters the claim, the reversion sizer
and the call only.

HELD-OUT SET (HO), for every universe-wide decision: the universe minus the
four instruments the design was audited on (Gold, Nifty 50, S&P 500, USD/INR)
minus every member whose daily log-return correlation with one of them is
≥ 0.8 over [epoch, run date] — Pearson, on the sessions both printed, prices
only, so the screen is label-free. The excluded members and their
correlations are recorded. The priors themselves pool every member, leave one
out.

PER MEMBER. ``pipeline.run.run_target(member, end=run date, priors=None)`` —
the app's own pipeline, with no prior, on the committed code (its git SHA is
recorded). Kept: the composer's inputs, the as-of own grades of the claim and
both sizers (``grades_ts``), their full-history grades, the gap split and the
claim's unit. A member that raises is logged, listed under ``failures`` and
left out. Runs are cached per member under ``--workdir`` and a rerun resumes.

OBJECTS. Each is graded as the app grades it: ``core.grade`` on the
drift-neutral forward return Y − μ at h = 10 and L = EXECUTION_LAG, on the
claim rows (Valid, target-printed sessions with |FVO| ≥ CLAIM_SIGMA).

  * ``claim``           the claim's sign d;
  * ``sizer:swayam``    z = d·Lean on the native claim rows, against d·(Y − μ);
  * ``sizer:reversion`` z = 2·MRProb − 1, against d·(Y − μ);
  * ``call``            w = d·trust·∏exp(β·z) composed by ``core.call`` with
                        THIS study's leave-one-out priors and Decision B's
                        SIZERS_ENABLED, graded on the sessions it was issued
                        (w ≠ 0). Its pooled grade is the class's pooled call
                        grade shown beside a call not yet graded.

POOLING, per object and vintage. Vintages are the calendar quarter-ends from
epoch + 2 years to the run date. At vintage v:

  1. a member's own grade is its as-of grade at the latest grid date ≤ v, used
     only once n_win ≥ GRADE_WARMUP_WIN;
  2. its error correlations: u_s = r̃(lean)·r̃(target) on its graded pairs
     whose outcome had resolved by v (ranks standardised over those pairs),
     averaged within windows of h weekdays counted from the epoch (one grid
     for every member); ρ_jk over their common windows, 0 below
     MIN_COMMON_WINDOWS (``studies._pool.payoff_rho``);
  3. member i's prior is ``studies._pool.loo_prior``: τ² by DerSimonian–Laird
     with class fixed effects over every other graded member; the pool is
     i's class when it holds ≥ POOL_MIN_MEMBERS of them ('class-LOO'), else
     every other member ('global-LOO'); μ, var_μ and κ from that pool;
     se_floor the SE of the pool with i added. Nothing in it reads i's IC;
  4. an instrument outside the study reads its class's
     ``studies._pool.class_all_prior`` ('__class__:<class>').

``summary[class][object]`` is the class's own graded members pooled at the
last vintage (any number of them): {mu, se_mu, tau, k, asof}.

DECISIONS, on HO members only. Each is a pooled paired estimate — the equal-
weight mean over the HO members of each member's own paired difference —
with a joint moving-block bootstrap SE: B = 200 draws of blocks of 5h
consecutive weekdays between the epoch and the run date, one draw shared by
every member and both sides of every pair, fixed seed.

  A  claim centring. Candidate ``core.semantics.claim_oscillator(FVO,
     'robust_z_252')``. Adopt iff (a) among the HO members whose raw FVO fails
     the C12 contract (``pipeline.evidence.claim_unit``) the candidate passes
     it on at least 2/3 (and at least one fails), and (b) ΔIC = IC(the claim
     sign under the candidate, on its own claim rows) − IC(the raw claim
     sign) ≥ −1 SE. Recorded as CLAIM_CENTRING ('robust_z_252' or 'none');
     never applied — adopting it is a separate owner-signed change.
  B  sizer release. Each member's call composed with the leave-one-out
     priors, sizers enabled; Δ = IC(d·∏factor) − IC(d) on the sessions with
     a claim (trust ≡ 1: the sizers alone). SIZERS_ENABLED = Δ ≥ −1 SE. The
     issued call against the bare claim, IC(w) − IC(d), is reported as
     information only: the bare claim assumes the skill trust must earn.
  C  tier words. The issued calls of the final call (w ≠ 0), each in its
     causal expanding size tercile (the rule of ``pipeline.assemble``'s
     tercile payoff); pooled drift-neutral payoff d·(Y − μ) per tercile.
     TIER_RULE = 'causal_tercile' iff T1 < T2 < T3 and every tercile has
     n/h ≥ TIER_MIN_WIN, else 'none'.

A decision with nothing to decide on (no HO member with a defined difference,
no issued call) takes its conservative value: 'none', False, 'none'.

ARMS (disclosures; they move nothing).

  E  lag. Every object over its full history at L = 0 and L = 1, per member
     and pooled per class: what EXECUTION_LAG removes.
  F  prospective record. The previous study's output (``--previous``,
     default the existing ``--out`` file, if dated before this run): each
     member's call composed with that file's priors and SIZERS_ENABLED,
     graded on the issued sessions after its run date. Empty on the first
     run.
  P  path sensitivity, for the first listed member of each catalogue class
     in TARGET_CATEGORIES and for AAPL and RELIANCE: the full-history claim
     IC with the history started 0, 1, 2, 5, 10 and 21 sessions later, and
     a panel jackknife — 5 draws, each dropping 10 of the member's predictors
     at random (seed fixed per member). Ranges recorded.
  G  who closes the gap (``pipeline.evidence.gap_split``) per member, and
     its mean and spread per class.

CHECKS. On R4 or R6 failure the study prints its report, exits 2 and writes
nothing.

  R4 calibration (abort). For every member and object, the shift-null SE's
     rotation part against a 200-shift brute-force null of the offset-
     averaged IC (``studies.se_calibration.brute_rotation_sd``) on the full-
     history pairs. The median ratio per object must lie in [0.85, 1.15]. An
     object no member can check aborts — except the call, which has no SE
     where no call was issued.
  R6 correlation (abort). At the last vintage, for every object and every
     pool a prior is drawn from — each class with ≥ POOL_MIN_MEMBERS graded
     members, and the whole catalogue (global-LOO, global-all) — the ratio of
     the variance of the pool's mean over R6_DRAWS = 1000 joint bootstrap
     draws (each member's IC recomputed on its resampled pairs, the weights
     held) to the closed-form var_μ of that pool. Per object, the median ratio
     over its pools must be ≤ 1.5, as R4 takes the median over members: one
     pool's ratio is too noisy to abort on (under no skill a single member's
     bootstrap-to-null variance ratio spans about 0.4–1.7 from the 10th to the
     90th percentile, and a class of near-copies is one member). Every pool's
     ratio is recorded and one above 1.5 is flagged. A smaller class pool feeds
     no prior and is recorded only. The closed form's estimation part,
     Σ w*_j w*_k ρ_jk se_j se_k / (Σ w*)² — all that a resampling of dates
     can see, τ² being spread between instruments — is recorded beside it.
  R5 power (recorded). Per object and class, at the last vintage's graded
     SEs, ρ and τ: one look's false-open rate and mean strength under the
     global null, and P(open) with the class planted at IC 0.05 and 0.08
     (``studies._pool.simulate_looks``, 500 draws, fixed seed).

OUTPUT ``pooled_priors_h{h}_L{L}.json``: study, git_sha, git_dirty,
script_sha256, run_date, epoch, h, exec_lag, knobs_hash, code_version, mode,
complete, universe, members, failures, held_out, classes, fingerprints (each
member's history fingerprint, hashed up to ``fingerprint_upto``, its newest
settled session on or before the run date), vintages, priors[object][member
| '__class__:<class>'][vintage] = {mu, var_mu, tau2, kappa, se_floor, k,
scope}, summary, decisions {A, B, C}, arms {E, F, P, G}, checks {R4, R5, R6}.
Floats to 10 significant digits, NaN as null, keys sorted: re-run on the same
histories, code and run date it is byte for byte the same file. Wall times
go to the work directory's log, not into the file.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

# One BLAS thread, as app.py runs the engine (set before numpy loads), so a
# member's run does not depend on the machine's core count.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.call import SIZERS, compose_series  # noqa: E402
from core.config import (ANALYSIS_EPOCH, CATALOGUE_TARGETS, TARGET_CATEGORIES,  # noqa: E402
                         VERSION, get_instrument_config, register_stock_target)
from core.grade import (_graded_frame, _null_parts, _offset_averaged_ic, _rank,  # noqa: E402
                        expanding_edge, series_edge)
from core.instruments import INSTRUMENTS, instrument  # noqa: E402
from core.knobs import knob, knobs_hash  # noqa: E402
from core.semantics import claim_direction, claim_oscillator  # noqa: E402
from core.trust import PriorTable, load_priors  # noqa: E402
from pipeline import run as pipeline_run  # noqa: E402
from pipeline.assemble import _claim_rows, _payoff, _shrunk, gather_inputs  # noqa: E402
from pipeline.evidence import claim_unit  # noqa: E402
from studies._pool import (POOL_MIN_MEMBERS, class_all_prior, dl_tau2, loo_prior,  # noqa: E402
                           payoff_rho, simulate_looks)
from studies.se_calibration import brute_rotation_sd  # noqa: E402

STUDY = "studies/pooled_priors.py"
EXECUTION_LAG: int = knob("EXECUTION_LAG")
GRADE_WARMUP_WIN: int = knob("GRADE_WARMUP_WIN")
TIER_MIN_WIN: int = knob("TIER_MIN_WIN")

# ════════════════════════════════════════════════════════════════════════
# The pre-registered constants
# ════════════════════════════════════════════════════════════════════════

#: The ten stocks, by free-float size at design time: app name → (ticker, market).
STUDY_STOCKS: dict[str, tuple[str, str]] = {
    "AAPL (US)": ("AAPL", "us"), "MSFT (US)": ("MSFT", "us"), "JPM (US)": ("JPM", "us"),
    "XOM (US)": ("XOM", "us"), "JNJ (US)": ("JNJ", "us"),
    "RELIANCE (NSE)": ("RELIANCE.NS", "india"), "TCS (NSE)": ("TCS.NS", "india"),
    "HDFCBANK (NSE)": ("HDFCBANK.NS", "india"), "INFY (NSE)": ("INFY.NS", "india"),
    "ITC (NSE)": ("ITC.NS", "india"),
}
#: The four instruments the design was audited on: never held out.
BURNED: tuple[str, ...] = ("Gold", "Nifty 50", "S&P 500", "USD/INR")
#: A member this correlated with a burned instrument is not held out either.
HO_MAX_CORR = 0.8
#: The objects pooled, in order.
OBJECTS: tuple[str, ...] = ("claim", *(f"sizer:{s.key}" for s in SIZERS), "call")
#: Vintages start this many years after the epoch.
VINTAGE_START_YEARS = 2
#: Common windows below which two members' error correlation is taken as 0.
MIN_COMMON_WINDOWS = 60
#: The joint moving-block bootstrap: block length in horizons, draws, seed.
BOOT_BLOCK_H, BOOT_DRAWS, SEED = 5, 200, 0
#: Decision A's candidate unit and its label-free bar.
A_CANDIDATE, A_SHARE = "robust_z_252", 2 / 3
#: R4: brute-force shifts and the band the median ratio must lie in.
R4_SHIFTS, R4_BAND = 200, (0.85, 1.15)
#: R6: the most the bootstrap variance may exceed the closed form by, and its
#: draws (more than the decisions': the check is a property of the data, not of
#: the seed).
R6_MAX_RATIO, R6_DRAWS = 1.5, 1000
#: R5: draws and the class ICs planted for power.
R5_SIMS, R5_THETAS = 500, (0.05, 0.08)
#: Arm P: start shifts (sessions) and the jackknife (draws, columns dropped).
P_SHIFTS: tuple[int, ...] = (0, 1, 2, 5, 10, 21)
P_JACKKNIFE: tuple[int, int] = (5, 10)
#: Arm P's two stock members.
P_STOCKS: tuple[str, ...] = ("AAPL (US)", "RELIANCE (NSE)")

#: The offline fixtures' targets (tests/fixtures), for --offline-fixture.
FIXTURE_MEMBERS: tuple[str, ...] = ("Gold", "USD/INR")
#: Bumped when a cached member record's layout changes.
RECORD_VERSION = 1


# ════════════════════════════════════════════════════════════════════════
# The universe
# ════════════════════════════════════════════════════════════════════════

def universe() -> list[str]:
    """The 38 members, in order: the tradeable catalogue, then STUDY_STOCKS
    (registered as the app registers a typed symbol)."""
    for name, (ticker, market) in STUDY_STOCKS.items():
        register_stock_target(name, ticker, market)
    catalogue = [n for n in CATALOGUE_TARGETS if INSTRUMENTS[n].tradeable]
    return catalogue + list(STUDY_STOCKS)


def p_members() -> list[str]:
    """Arm P's members: the first listed of each catalogue class, AAPL and RELIANCE."""
    firsts = [names[0] for names in TARGET_CATEGORIES.values()
              if names and INSTRUMENTS[names[0]].tradeable]
    return firsts + list(P_STOCKS)


def vintages(run_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Quarter-ends from epoch + VINTAGE_START_YEARS to the run date."""
    start = pd.Timestamp(ANALYSIS_EPOCH) + pd.DateOffset(years=VINTAGE_START_YEARS)
    return list(pd.date_range(start, run_date, freq="QE"))


# ════════════════════════════════════════════════════════════════════════
# One member: the app's pipeline, with no prior
# ════════════════════════════════════════════════════════════════════════

def _claim_ic(target: str, fvo_ts: pd.DataFrame, h: int, run_date) -> float:
    """The full-history claim grade's IC on a Mūla frame, as grade_all grades it."""
    inputs = gather_inputs(fvo_ts, None, instrument=instrument(target), h=h, today=run_date)
    claim = inputs["claim"].astype(float)
    return float(series_edge(claim, inputs["price"], h, mask=_claim_rows(inputs), drift="causal",
                             exec_lag=EXECUTION_LAG)["ic"])


def path_sensitivity(target: str, *, run_date, fetch, base_ic: float) -> dict:
    """Arm P for one member: the claim IC under start shifts and a panel jackknife."""
    cfg = get_instrument_config(target)
    h = int(cfg.forecast_horizon)
    panel = pipeline_run.fetch_panel(target, epoch=ANALYSIS_EPOCH, end=run_date, fetch=fetch)
    bars, _ = pipeline_run.target_bars(target, epoch=ANALYSIS_EPOCH, end=run_date, fetch=fetch)
    base = pipeline_run.prepare(panel, target, epoch=ANALYSIS_EPOCH, bars=bars, priors=None)

    def ic_of(prep) -> float:
        engine = pipeline_run.fit_mula(prep, cfg)
        return _claim_ic(target, pipeline_run.fvo_frame(engine, prep), h, run_date)

    shifts = {"0": base_ic}
    for k in P_SHIFTS[1:]:
        start = base.data.index[k]
        shifts[str(k)] = ic_of(pipeline_run.prepare(panel, target, epoch=start, bars=bars, priors=None))
    rng = np.random.default_rng(SEED + zlib.crc32(target.encode("utf-8")))
    n_draws, n_drop = P_JACKKNIFE
    feats = sorted(base.features)
    jack = []
    for _ in range(n_draws):
        dropped = sorted(rng.choice(feats, size=min(n_drop, len(feats)), replace=False).tolist())
        prep = pipeline_run.prepare(panel.drop(columns=dropped), target, epoch=ANALYSIS_EPOCH,
                                    bars=bars, priors=None)
        jack.append({"dropped": dropped, "ic": ic_of(prep)})
    values = [v for v in [*shifts.values(), *(j["ic"] for j in jack)] if np.isfinite(v)]
    return {"base_ic": base_ic, "shifts": shifts, "jackknife": jack,
            "range": [min(values), max(values)] if values else None}


def run_member(target: str, *, run_date: pd.Timestamp, fetch, arm_p: bool) -> dict:
    """One member's record: the app's run with no prior, reduced to what the study reads."""
    inst = instrument(target)
    h = int(get_instrument_config(target).forecast_horizon)
    run = pipeline_run.run_target(target, end=run_date, priors=None, fetch=fetch)
    fvo_ts = run["fvo_ts"]
    inputs = gather_inputs(fvo_ts, run["swayam_daily"], instrument=inst, h=h, today=run_date)
    settled = fvo_ts.index[-2] if len(fvo_ts) > 1 else fvo_ts.index[0] - pd.Timedelta(days=1)
    rec = {
        "version": RECORD_VERSION, "target": target, "asset_class": inst.asset_class,
        "ticker": inst.ticker, "h": h, "exec_lag": int(run["exec_lag"]), "asof": run["asof"],
        "start": fvo_ts.index[0], "fingerprint": run["fingerprint"],
        "fingerprint_upto": min(pd.Timestamp(settled), run_date),
        "inputs": inputs,
        "grades": {obj: g[["ic", "se", "n_win", "grid_date"]] for obj, g in run["grades_ts"].items()},
        "edges": {k: {f: run["edges"][k].get(f) for f in ("ic", "se", "t", "n_win")}
                  for k in ("claim", *(f"sizer:{s.key}" for s in SIZERS)) if k in run["edges"]},
        "gap_split": run["evidence"].get("gap_split"),
        "claim_unit": run["evidence"].get("claim_unit"),
    }
    if arm_p:
        rec["arm_p"] = path_sensitivity(target, run_date=run_date, fetch=fetch,
                                        base_ic=float(rec["edges"]["claim"]["ic"]))
    return rec


# ════════════════════════════════════════════════════════════════════════
# Graded pairs, the window grid and the joint bootstrap
# ════════════════════════════════════════════════════════════════════════

def _pairs(lean: pd.Series, price: pd.Series, h: int, lag: int, *, mask, sign=None) -> pd.DataFrame:
    """A grade's pairs ``x`` (lean), ``y`` (signed drift-neutral target) and ``s``
    (sign), dated by session, with ``known`` the session the outcome resolved."""
    df = _graded_frame(lean, price, h, mask=mask, sign=sign, drift="causal", exec_lag=lag)
    known = price.index[df["pos"].to_numpy(dtype=np.int64) + lag + h]
    return pd.DataFrame({"x": df["x"], "y": df["y"], "s": df["s"], "known": known}, index=df.index)


def _specs(inputs: pd.DataFrame, call_ts: pd.DataFrame | None = None) -> dict:
    """Each object's ``(lean, mask, sign)``, as ``pipeline.assemble`` grades it."""
    rows, claim = _claim_rows(inputs), inputs["claim"].astype(float)
    out = {"claim": (claim, rows, None),
           **{f"sizer:{s.key}": (inputs[s.z_col], rows, claim) for s in SIZERS}}
    if call_ts is not None:
        w = call_ts["w"].astype(float)
        out["call"] = (w, w != 0, None)
    return out


def _std_rank(a: np.ndarray) -> np.ndarray | None:
    r = _rank(a)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 1e-12 else None


def _u(pairs: pd.DataFrame, v, windows: np.ndarray) -> pd.Series:
    """The payoff products resolved by ``v``, averaged per window of the grid."""
    p = pairs[pairs["known"] <= v] if v is not None else pairs
    if len(p) < 3:
        return pd.Series(dtype=float)
    rx, ry = _std_rank(p["x"].to_numpy(dtype=np.float64)), _std_rank(p["y"].to_numpy(dtype=np.float64))
    if rx is None or ry is None:
        return pd.Series(dtype=float)
    lab = windows[np.searchsorted(windows, p.index.to_numpy(dtype="datetime64[ns]"), side="right") - 1]
    return pd.Series(rx * ry, index=lab).groupby(level=0).mean()


class JointBootstrap:
    """Moving blocks of BOOT_BLOCK_H·h weekdays between the epoch and the run
    date; draw b is the same set of blocks for every member and every side."""

    def __init__(self, run_date, h: int, n_draws: int = BOOT_DRAWS, seed: int = SEED):
        self.cal = pd.bdate_range(ANALYSIS_EPOCH, run_date).to_numpy(dtype="datetime64[ns]")
        self.block = BOOT_BLOCK_H * int(h)
        n = len(self.cal)
        rng = np.random.default_rng(seed)
        n_blk = -(-n // self.block)
        self.starts = [rng.integers(0, max(1, n - self.block + 1), size=n_blk) for _ in range(n_draws)]

    def rows(self, dates: pd.DatetimeIndex, draw: int) -> np.ndarray:
        """Positions of the rows (``dates`` sorted) that fall in draw ``draw``'s blocks."""
        pos = np.searchsorted(self.cal, dates.to_numpy(dtype="datetime64[ns]"))
        s = self.starts[draw]
        lo, hi = np.searchsorted(pos, s), np.searchsorted(pos, s + self.block)
        return np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(s) else np.zeros(0, int)


def _ic(p: pd.DataFrame, idx: np.ndarray | None, h: int) -> float:
    x, y = p["x"].to_numpy(dtype=np.float64), p["y"].to_numpy(dtype=np.float64)
    if idx is not None:
        x, y = x[idx], y[idx]
    return _offset_averaged_ic(x, y, h)


def paired(sides: dict[str, tuple[pd.DataFrame, pd.DataFrame]], boot: JointBootstrap, h: int) -> dict:
    """Pooled paired ΔIC = mean over members of IC(b) − IC(a), with its joint bootstrap SE."""
    per = {m: _ic(b, None, h) - _ic(a, None, h) for m, (a, b) in sides.items()}
    used = [m for m, d in per.items() if np.isfinite(d)]
    if not used:
        return {"delta": None, "se": None, "n_members": 0, "per_member": per}
    draws = []
    for k in range(len(boot.starts)):
        vals = [_ic(sides[m][1], boot.rows(sides[m][1].index, k), h)
                - _ic(sides[m][0], boot.rows(sides[m][0].index, k), h) for m in used]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            draws.append(float(np.mean(vals)))
    se = float(np.std(draws, ddof=1)) if len(draws) > 1 else math.nan
    return {"delta": float(np.mean([per[m] for m in used])), "se": se, "n_members": len(used),
            "per_member": per}


# ════════════════════════════════════════════════════════════════════════
# Pooling
# ════════════════════════════════════════════════════════════════════════

def pool_object(names: list[str], classes: list[str], grades: dict[str, pd.DataFrame],
                pairs: dict[str, pd.DataFrame], vints: list[pd.Timestamp],
                windows: np.ndarray) -> tuple[dict, dict]:
    """One object's priors at every vintage, and its state at the last vintage.

    ``grades[m]`` is member m's as-of grade (``ic, se, n_win`` by date),
    ``pairs[m]`` its graded pairs (``_pairs``). Returns ``(entries, last)``:
    ``entries[member | '__class__:<c>'][vintage] = prior``, and ``last`` the
    arrays the last vintage pooled (for the summary, R5 and R6).
    """
    cls = np.array(classes, dtype=object)
    vix = pd.DatetimeIndex(vints)
    at = {m: (grades[m][["ic", "se", "n_win"]].reindex(vix, method="ffill") if m in grades
              else pd.DataFrame(np.nan, index=vix, columns=["ic", "se", "n_win"])) for m in names}
    entries: dict[str, dict] = {}
    last: dict = {}
    for v in vix:
        ic = np.array([at[m].at[v, "ic"] for m in names], dtype=np.float64)
        se = np.array([at[m].at[v, "se"] for m in names], dtype=np.float64)
        n_win = np.array([at[m].at[v, "n_win"] for m in names], dtype=np.float64)
        graded = np.isfinite(ic) & np.isfinite(se) & (se > 0) & (n_win >= GRADE_WARMUP_WIN)
        ic, se = np.where(graded, ic, np.nan), np.where(graded, se, np.nan)
        rho = payoff_rho({m: _u(pairs[m], v, windows) if m in pairs else pd.Series(dtype=float)
                          for m in names}, MIN_COMMON_WINDOWS).to_numpy()
        key = v.strftime("%Y-%m-%d")
        kappa = np.full(len(names), np.nan)
        for i, m in enumerate(names):
            e = loo_prior(i, ic, se, cls, rho, min_members=POOL_MIN_MEMBERS)
            if e is not None:
                entries.setdefault(m, {})[key] = e
                if graded[i]:
                    kappa[i] = e["kappa"]
        for c in dict.fromkeys(classes):
            e = class_all_prior(c, ic, se, cls, rho, kappa=kappa, min_members=POOL_MIN_MEMBERS)
            if e is not None:
                entries.setdefault(f"__class__:{c}", {})[key] = e
        last = {"vintage": v, "ic": ic, "se": se, "n_win": n_win, "graded": graded, "rho": rho,
                "cls": cls, "names": names, "kappa": kappa}
    return entries, last


def summarise(last: dict) -> dict:
    """Each class's own graded members pooled at the last vintage: {mu, se_mu, tau, k, asof}."""
    out = {}
    if not last:
        return out
    for c in dict.fromkeys(last["cls"]):
        e = class_all_prior(c, last["ic"], last["se"], last["cls"], last["rho"],
                            kappa=last["kappa"], min_members=1)
        if e is not None and e["scope"] == "class-all":
            out[c] = {"mu": e["mu"], "se_mu": math.sqrt(e["var_mu"]), "tau": math.sqrt(e["tau2"]),
                      "k": e["k"], "asof": last["vintage"].strftime("%Y-%m-%d")}
    return out


# ════════════════════════════════════════════════════════════════════════
# The calls the study's own priors compose
# ════════════════════════════════════════════════════════════════════════

def compose(rec: dict, table: PriorTable | None, sizers_enabled: bool) -> pd.DataFrame:
    """The member's call at every session, its grades shrunk to ``table`` — the
    app's composer on the app's own as-of grades."""
    inst = instrument(rec["target"])
    shrunk = {obj: _shrunk(g, obj, table, inst) for obj, g in rec["grades"].items()}
    return compose_series(rec["inputs"], shrunk, tradeable=inst.tradeable,
                          sizers_enabled=bool(sizers_enabled))


def _tercile_labels(size: np.ndarray) -> np.ndarray:
    """Each issued call's causal expanding size tercile — ``pipeline.assemble``'s
    rule: the share of issued sizes so far at or below its own."""
    return np.array([min(3, max(1, math.ceil(3 * float(np.mean(size[:i + 1] <= v)))))
                     for i, v in enumerate(size)], dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════
# The held-out set and the three decisions
# ════════════════════════════════════════════════════════════════════════

def held_out(records: dict[str, dict], names: list[str]) -> dict:
    """HO: not burned, and correlated below HO_MAX_CORR with every burned member."""
    ret = {m: np.log(r["inputs"]["price"].where(lambda p: p > 0)).diff() for m, r in records.items()}
    excluded, missing = {}, [b for b in BURNED if b not in records]
    for m in names:
        if m in BURNED or m not in records:
            continue
        for b in BURNED:
            if b not in ret:
                continue
            both = pd.concat([ret[m], ret[b]], axis=1, join="inner").dropna()
            rho = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 2 else math.nan
            if np.isfinite(rho) and rho >= HO_MAX_CORR and \
                    rho > excluded.get(m, {}).get("rho", -np.inf):
                excluded[m] = {"with": b, "rho": rho}
    members = [m for m in names if m in records and m not in BURNED and m not in excluded]
    return {"members": members, "burned": list(BURNED), "excluded_corr": excluded,
            "burned_not_run": missing,
            "rule": f"universe − {', '.join(BURNED)} − daily log-return ρ ≥ {HO_MAX_CORR:g} "
                    f"with any of them over [epoch, run date]"}


def decision_a(records: dict, ho: list[str], boot: JointBootstrap) -> dict:
    """Claim centring: recorded, never applied."""
    raw_fail, cand_pass, sides = [], [], {}
    for m in ho:
        r = records[m]
        inputs, h = r["inputs"], r["h"]
        fvo = inputs["fvo"].where(inputs["published"])
        osc = claim_oscillator(fvo, A_CANDIDATE)
        raw = claim_unit(pd.DataFrame({"FVO": inputs["fvo"], "Valid": inputs["valid"]}))
        cand = claim_unit(pd.DataFrame({"FVO": osc, "Valid": inputs["valid"]}))
        if not raw["passes"]:
            raw_fail.append(m)
            if cand["passes"]:
                cand_pass.append(m)
        d_c = osc.map(lambda x: claim_direction(x) if np.isfinite(x) else 0).astype(float)
        rows_c = inputs["valid"] & inputs["printed"] & (d_c != 0)
        sides[m] = (_pairs(inputs["claim"].astype(float), inputs["price"], h, EXECUTION_LAG,
                           mask=_claim_rows(inputs)),
                    _pairs(d_c, inputs["price"], h, EXECUTION_LAG, mask=rows_c))
    est = paired(sides, boot, _h(records))
    a_ok = bool(raw_fail) and len(cand_pass) >= A_SHARE * len(raw_fail)
    b_ok = est["delta"] is not None and np.isfinite(est["se"]) and est["delta"] >= -est["se"]
    return {"value": A_CANDIDATE if (a_ok and b_ok) else "none", "candidate": A_CANDIDATE,
            "raw_fails": raw_fail, "candidate_passes": cand_pass, "a_label_free": bool(a_ok),
            "b_delta_ic": est, "b_ok": bool(b_ok), "applied": False,
            "rule": "adopt iff the candidate passes C12 on ≥ 2/3 of the HO members where raw FVO "
                    "fails, and ΔIC ≥ −1 SE; recorded only, never applied"}


def decision_b(records: dict, ho: list[str], table: PriorTable, boot: JointBootstrap) -> dict:
    """Sizer release: do the sizers, on their own causal β, cost the claim's IC?"""
    sides, info = {}, {}
    for m in ho:
        r = records[m]
        ct = compose(r, table, True)
        price, h = r["inputs"]["price"], r["h"]
        d = ct["d"].astype(float)
        sized = d * np.prod([ct[f"{s.key}_factor"] for s in SIZERS], axis=0)
        mask = d != 0
        bare = _pairs(d, price, h, EXECUTION_LAG, mask=mask)
        sides[m] = (bare, _pairs(sized, price, h, EXECUTION_LAG, mask=mask))
        info[m] = (bare, _pairs(ct["w"].astype(float), price, h, EXECUTION_LAG, mask=mask))
    est = paired(sides, boot, _h(records))
    ok = est["delta"] is not None and np.isfinite(est["se"]) and est["delta"] >= -est["se"]
    return {"value": bool(ok), "delta_ic": est,
            "call_vs_bare_claim": paired(info, boot, _h(records)),
            "rule": "SIZERS_ENABLED iff IC(d·∏factor) − IC(d) ≥ −1 SE on HO claim rows, trust ≡ 1, "
                    "causal β from the leave-one-out priors; call vs bare claim is information"}


def decision_c(records: dict, ho: list[str], finals: dict[str, pd.DataFrame],
               boot: JointBootstrap) -> dict:
    """Tier words: does a larger causal size tercile pay more?"""
    h = _h(records)
    rows = []
    for m in ho:
        ct, price = finals[m], records[m]["inputs"]["price"]
        issued = ct["w"] != 0
        if not issued.any():
            continue
        y = _payoff(ct["d"], price, h=h, exec_lag=EXECUTION_LAG)[issued]
        lab = pd.Series(_tercile_labels(ct["size"][issued].to_numpy(dtype=np.float64)), index=y.index)
        ok = y.notna()
        rows.append((y[ok], lab[ok]))

    def pooled(sel) -> dict:
        out = {}
        for k in (1, 2, 3):
            ys = np.concatenate([y.to_numpy()[i][(lab.to_numpy()[i] == k)] for (y, lab), i in sel]) \
                if sel else np.zeros(0)
            out[f"T{k}"] = (float(ys.mean()) if len(ys) else math.nan, len(ys))
        return out

    full = pooled([(r, slice(None)) for r in rows])
    draws = {f"T{k}": [] for k in (1, 2, 3)}
    for b in range(len(boot.starts)):
        sel = [((y, lab), boot.rows(y.index, b)) for y, lab in rows]
        for t, (mean, _n) in pooled(sel).items():
            if np.isfinite(mean):
                draws[t].append(mean)
    terc = {t: {"mean": mean, "n": n, "n_win": n // h,
                "se": float(np.std(draws[t], ddof=1)) if len(draws[t]) > 1 else None}
            for t, (mean, n) in full.items()}
    means = [terc[f"T{k}"]["mean"] for k in (1, 2, 3)]
    monotone = all(np.isfinite(means)) and means[0] < means[1] < means[2]
    enough = all(terc[t]["n_win"] >= TIER_MIN_WIN for t in terc)
    return {"value": "causal_tercile" if (monotone and enough) else "none", "terciles": terc,
            "monotone": bool(monotone), "enough_windows": bool(enough),
            "members_with_issued_calls": len(rows),
            "rule": f"causal_tercile iff pooled payoff T1 < T2 < T3 among HO issued calls with "
                    f"n/h ≥ {TIER_MIN_WIN} per tercile"}


def _h(records: dict) -> int:
    hs = {int(r["h"]) for r in records.values()}
    if len(hs) != 1:
        raise ValueError(f"members at more than one horizon: {sorted(hs)}")
    return hs.pop()


# ════════════════════════════════════════════════════════════════════════
# Checks
# ════════════════════════════════════════════════════════════════════════

def check_r4(pairs: dict[str, dict[str, pd.DataFrame]], h: int) -> dict:
    """R4: the closed-form shift-null rotation against a brute-force null, per object."""
    out = {}
    for obj, by_m in pairs.items():
        ratios = {}
        for m, p in by_m.items():
            x, y = p["x"].to_numpy(dtype=np.float64), p["y"].to_numpy(dtype=np.float64)
            s = p["s"].to_numpy(dtype=np.float64) if obj.startswith("sizer:") else None
            parts = _null_parts(x, y, h, s)
            if parts is None:
                continue
            brute = brute_rotation_sd(x, y, s, h, R4_SHIFTS, SEED)
            if np.isfinite(brute) and brute > 0:
                ratios[m] = parts[0] / brute
        med = float(np.median(list(ratios.values()))) if ratios else math.nan
        evaluable = bool(ratios)
        ok = (R4_BAND[0] <= med <= R4_BAND[1]) if evaluable else obj == "call"
        out[obj] = {"median": med, "n": len(ratios), "band": list(R4_BAND), "pass": bool(ok),
                    "per_member": ratios}
    return out


def check_r6(last: dict, pairs: dict[str, pd.DataFrame], boot: JointBootstrap, h: int) -> dict:
    """R6: for every pool a prior is drawn from, the bootstrap variance of its pooled
    mean against its closed-form var_μ; the object passes when the median ratio
    over those pools is at most R6_MAX_RATIO.

    The pools are each class with at least POOL_MIN_MEMBERS graded members
    (its class-LOO and class-all priors) and the whole catalogue (``'global'``,
    the global-LOO and global-all priors); a smaller class pool is recorded,
    not checked. Each draw recomputes every graded member's IC on its
    resampled pairs once; a pool's mean holds its closed-form weights.
    """
    out = {}
    if not last or not last["graded"].any():
        return {"pools": out, "median": None, "max_ratio": R6_MAX_RATIO, "pass": True,
                "status": "nothing graded at the last vintage"}
    names, cls, graded = last["names"], last["cls"], last["graded"]
    se, rho, v = last["se"], last["rho"], last["vintage"]
    tau2 = dl_tau2(last["ic"][graded], se[graded], cls[graded])
    idx = np.array([i for i in np.flatnonzero(graded) if names[i] in pairs], dtype=np.int64)
    resolved = {i: pairs[names[i]][pairs[names[i]]["known"] <= v] for i in idx}
    draws = np.array([[_ic(resolved[i], boot.rows(resolved[i].index, b), h) for i in idx]
                      for b in range(len(boot.starts))])
    pools = {c: idx[cls[idx] == c] for c in dict.fromkeys(cls[idx])}
    pools["global"] = idx
    for c, pool in pools.items():
        s = se[pool]
        w = 1.0 / (s * s + tau2)
        est = float(w @ (rho[np.ix_(pool, pool)] * np.outer(s, s)) @ w) / float(w.sum()) ** 2
        full = est + tau2 * float(w @ w) / float(w.sum()) ** 2
        cols = np.searchsorted(idx, pool)
        ok = np.isfinite(draws[:, cols]).all(axis=1)
        means = draws[ok][:, cols] @ w / float(w.sum())
        var = float(np.var(means, ddof=1)) if len(means) > 1 else math.nan
        ratio = var / full if full > 0 and np.isfinite(var) else math.nan
        checked = len(pool) >= (2 if c == "global" else POOL_MIN_MEMBERS)
        out[c] = {"k": int(len(pool)), "boot_var": var, "var_mu": full, "estimation_var": est,
                  "ratio": ratio, "ratio_estimation": var / est if est > 0 else math.nan,
                  "draws": int(ok.sum()), "checked": bool(checked),
                  "flag": bool(checked and not (np.isfinite(ratio) and ratio <= R6_MAX_RATIO))}
    checked = [r["ratio"] for r in out.values() if r["checked"]]
    if not checked:
        return {"pools": out, "median": None, "max_ratio": R6_MAX_RATIO, "pass": True,
                "status": "no pool of two or more graded members"}
    med = float(np.median(checked)) if all(np.isfinite(checked)) else math.nan
    return {"pools": out, "median": med, "max_ratio": R6_MAX_RATIO,
            "pass": bool(np.isfinite(med) and med <= R6_MAX_RATIO)}


def check_r5(last: dict) -> dict:
    """R5: one look's false-open rate and power per class, at the observed se, ρ and τ."""
    out = {}
    if not last or not last["graded"].any():
        return out
    g = last["graded"]
    se, cls, rho = last["se"][g], last["cls"][g], last["rho"][np.ix_(g, g)]
    nwin = last["n_win"][g]
    tau = math.sqrt(dl_tau2(last["ic"][g], se, cls))
    null = simulate_looks(se, cls, rho, theta=0.0, tau=0.0, n_sim=R5_SIMS, seed=SEED)
    for c in dict.fromkeys(cls):
        idx = np.flatnonzero(cls == c)
        sub = rho[np.ix_(idx, idx)]
        off = sub[~np.eye(len(idx), dtype=bool)]
        row = {"k": int(len(idx)), "n_win_median": float(np.median(nwin[idx])),
               "rho_mean": float(off.mean()) if len(off) else None, "tau": tau,
               "false_open": float(null["open"][idx].mean()),
               "e_strength_null": float(null["strength"][idx].mean()),
               "scope": null["scope"][idx[0]]}
        for th in R5_THETAS:
            p = simulate_looks(se, cls, rho, theta={c: th}, tau=tau, n_sim=R5_SIMS, seed=SEED,
                               targets=idx)
            row[f"power_{th:g}"] = float(p["open"].mean())
        out[c] = row
    return out


# ════════════════════════════════════════════════════════════════════════
# Arms
# ════════════════════════════════════════════════════════════════════════

def arm_e(records: dict, finals: dict, names: list[str], windows: np.ndarray, h: int) -> dict:
    """Every object at L = 0 and L = 1 over its full history, per member and class."""
    per: dict = {}
    pooled: dict = {}
    cls = np.array([records[m]["asset_class"] for m in names], dtype=object)
    for lag in (0, 1):
        for obj in OBJECTS:
            ic, se, u = [], [], {}
            for m in names:
                r = records[m]
                lean, mask, sign = _specs(r["inputs"], finals.get(m))[obj]
                e = series_edge(lean, r["inputs"]["price"], h, mask=mask, sign=sign, drift="causal",
                                exec_lag=lag)
                per.setdefault(m, {}).setdefault(obj, {})[f"L{lag}"] = \
                    {"ic": e["ic"], "se": e["se"], "n_win": e["n_win"]}
                ok = e["n_win"] >= GRADE_WARMUP_WIN and np.isfinite(e["ic"]) and np.isfinite(e["se"])
                ic.append(e["ic"] if ok else np.nan)
                se.append(e["se"] if ok else np.nan)
                u[m] = _u(_pairs(lean, r["inputs"]["price"], h, lag, mask=mask, sign=sign), None, windows)
            rho = payoff_rho(u, MIN_COMMON_WINDOWS).to_numpy()
            for c in dict.fromkeys(cls):
                e = class_all_prior(c, ic, se, cls, rho, kappa=np.zeros(len(names)), min_members=1)
                if e is not None and e["scope"] == "class-all":
                    pooled.setdefault(c, {}).setdefault(obj, {})[f"L{lag}"] = \
                        {"mu": e["mu"], "se_mu": math.sqrt(e["var_mu"]), "k": e["k"]}
    return {"per_member": per, "per_class": pooled}


def arm_f(records: dict, previous: Path | None, h: int, run_date: pd.Timestamp) -> dict:
    """The prospective record: calls composed with the previous study, after its run date.

    A file dated on or after this run date is not a previous study (a rerun
    replacing its own output), so a rerun writes the same record.
    """
    prev = load_priors(h, EXECUTION_LAG, path=str(previous)) if previous is not None else None
    if prev is None or pd.Timestamp(prev.meta["run_date"]) >= run_date:
        return {"status": "first run: no previous study", "per_member": {}}
    since = pd.Timestamp(prev.meta["run_date"])
    enabled = bool(prev.meta["decisions"]["B"]["value"])
    per = {}
    for m, r in records.items():
        ct = compose(r, prev, enabled)
        mask = (ct["w"] != 0) & (ct.index > since)
        e = series_edge(ct["w"].astype(float), r["inputs"]["price"], h, mask=mask, drift="causal",
                        exec_lag=EXECUTION_LAG)
        per[m] = {"issued_n": int(mask.sum()), "n_win": e["n_win"], "ic": e["ic"], "se": e["se"],
                  "hit": e["hit"], "base": e["base"]}
    return {"status": f"calls after {since:%Y-%m-%d}, composed with that study", "since": since,
            "previous_sha": prev.meta.get("script_sha256"), "per_member": per}


def arm_g(records: dict) -> dict:
    """Who closes the gap, per member and per class."""
    per = {m: {k: r["gap_split"].get(k) for k in ("b_price", "b_fv", "price_share", "se_share", "n_win")}
           for m, r in records.items() if r.get("gap_split") and "error" not in r["gap_split"]}
    by_cls: dict = {}
    for m, g in per.items():
        by_cls.setdefault(records[m]["asset_class"], []).append(g)
    cls_out = {}
    for c, rows in by_cls.items():
        row = {"k": len(rows)}
        for f in ("b_price", "b_fv", "price_share"):
            vals = np.array([r[f] for r in rows if r[f] is not None and np.isfinite(r[f])])
            row[f] = {"mean": float(vals.mean()) if len(vals) else None,
                      "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None}
        cls_out[c] = row
    return {"per_member": per, "per_class": cls_out}


# ════════════════════════════════════════════════════════════════════════
# The study
# ════════════════════════════════════════════════════════════════════════

def analyse(records: dict[str, dict], *, universe_names: list[str], run_date: pd.Timestamp,
            previous: Path | None) -> dict:
    """Everything after the member runs: priors, decisions, arms and checks.

    ``records`` holds the members that ran, by name. Returns the output's
    body (without provenance) and ``abort`` — the checks that failed.
    """
    names = [m for m in universe_names if m in records]
    h = _h(records)
    vints = vintages(run_date)
    windows = pd.bdate_range(ANALYSIS_EPOCH, run_date)[::h].to_numpy(dtype="datetime64[ns]")
    classes = [records[m]["asset_class"] for m in names]
    boot = JointBootstrap(run_date, h, BOOT_DRAWS, SEED)
    ho = held_out(records, names)

    def obj_pairs(obj, finals=None):
        out = {}
        for m in names:
            lean, mask, sign = _specs(records[m]["inputs"], (finals or {}).get(m))[obj]
            out[m] = _pairs(lean, records[m]["inputs"]["price"], h, EXECUTION_LAG, mask=mask, sign=sign)
        return out

    priors, lasts, pairs = {}, {}, {}
    for obj in OBJECTS[:-1]:
        pairs[obj] = obj_pairs(obj)
        grades = {m: records[m]["grades"][obj] for m in names if obj in records[m]["grades"]}
        priors[obj], lasts[obj] = pool_object(names, classes, grades, pairs[obj], vints, windows)
    meta = {"study": STUDY, "run_date": run_date.strftime("%Y-%m-%d")}
    table = PriorTable(priors, meta)

    dec_b = decision_b(records, ho["members"], table, boot)
    finals = {m: compose(records[m], table, dec_b["value"]) for m in names}
    call_grades = {}
    for m in names:
        w = finals[m]["w"].astype(float)
        call_grades[m] = expanding_edge(w, records[m]["inputs"]["price"], h, mask=w != 0,
                                        drift="causal", exec_lag=EXECUTION_LAG,
                                        warmup_win=GRADE_WARMUP_WIN)
    pairs["call"] = obj_pairs("call", finals)
    priors["call"], lasts["call"] = pool_object(names, classes, call_grades, pairs["call"], vints, windows)

    decisions = {"A": decision_a(records, ho["members"], boot), "B": dec_b,
                 "C": decision_c(records, ho["members"], finals, boot)}
    summary: dict = {}
    for obj in OBJECTS:
        for c, row in summarise(lasts[obj]).items():
            summary.setdefault(c, {})[obj] = row
    full_pairs = {obj: {m: p[p["known"] <= run_date] for m, p in by_m.items()} for obj, by_m in pairs.items()}
    checks = {"R4": check_r4(full_pairs, h),
              "R6": {obj: check_r6(lasts[obj], pairs[obj], JointBootstrap(run_date, h, R6_DRAWS, SEED), h)
                     for obj in OBJECTS},
              "R5": {obj: check_r5(lasts[obj]) for obj in OBJECTS}}
    arms = {"E": arm_e(records, finals, names, windows, h), "F": arm_f(records, previous, h, run_date),
            "P": {m: records[m]["arm_p"] for m in names if "arm_p" in records[m]},
            "G": arm_g(records)}
    abort = [f"R4 {obj}: median {_f(r['median'], '.3f')} (n {r['n']})" for obj, r in checks["R4"].items()
             if not r["pass"]]
    abort += [f"R6 {obj}: median ratio {_f(r['median'], '.3f')} over its pools"
              for obj, r in checks["R6"].items() if not r["pass"]]
    return {"names": names, "h": h, "vintages": [v.strftime("%Y-%m-%d") for v in vints],
            "held_out": ho, "priors": priors, "summary": summary, "decisions": decisions,
            "arms": arms, "checks": checks, "abort": abort}


def _clean(x):
    """JSON-safe: numpy scalars to Python, floats to 10 significant digits, NaN to null,
    dates to ISO strings, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return float(f"{v:.10g}") if math.isfinite(v) else None
    if x is pd.NaT:
        return None
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return x


def build_output(result: dict, records: dict, *, universe_names: list[str], failures: dict,
                 run_date: pd.Timestamp, git: dict, mode: str, complete: bool) -> dict:
    """The output file's content: provenance, then the study's results."""
    names = result["names"]
    classes: dict[str, list[str]] = {}
    for m in names:
        classes.setdefault(records[m]["asset_class"], []).append(m)
    return _clean({
        "study": STUDY, "git_sha": git["sha"], "git_dirty": git["dirty"],
        "script_sha256": hashlib.sha256((ROOT / STUDY).read_bytes()).hexdigest(),
        "run_date": run_date.strftime("%Y-%m-%d"), "epoch": pd.Timestamp(ANALYSIS_EPOCH),
        "h": result["h"], "exec_lag": EXECUTION_LAG, "knobs_hash": knobs_hash(),
        "code_version": VERSION, "mode": mode, "complete": bool(complete),
        "universe": {m: {"class": instrument(m).asset_class, "ticker": instrument(m).ticker}
                     for m in universe_names},
        "members": names, "failures": failures, "held_out": result["held_out"], "classes": classes,
        "fingerprints": {m: records[m]["fingerprint"] for m in names},
        "fingerprint_upto": {m: records[m]["fingerprint_upto"] for m in names},
        "vintages": result["vintages"], "priors": result["priors"], "summary": result["summary"],
        "decisions": result["decisions"], "arms": result["arms"], "checks": result["checks"],
    })


def write_output(data: dict, out: Path) -> None:
    """Write atomically: a reader never sees half a file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)


def finish(result: dict, records: dict, out: Path, **provenance) -> int:
    """Write the output unless a check failed: 0 written, 2 aborted (nothing written)."""
    if result["abort"]:
        return 2
    write_output(build_output(result, records, **provenance), out)
    return 0


# ════════════════════════════════════════════════════════════════════════
# The report
# ════════════════════════════════════════════════════════════════════════

def _f(x, fmt="+.3f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, fmt)


def report(result: dict, timings: dict[str, float]) -> str:
    """The study's printed report: every number the CHANGELOG copies."""
    c, d, a = result["checks"], result["decisions"], result["arms"]
    lines = ["POOLED-PRIOR STUDY", "",
             f"members {len(result['names'])} · held out {len(result['held_out']['members'])}: "
             f"{', '.join(result['held_out']['members']) or 'none'}",
             f"excluded by correlation: {result['held_out']['excluded_corr'] or 'none'}", "",
             "R4  FFT / brute-force null, median per object"]
    for obj, r in c["R4"].items():
        lines.append(f"    {obj:<16} {_f(r['median'], '.3f')}  (n {r['n']})  {'pass' if r['pass'] else 'FAIL'}")
    lines += ["", "R6  bootstrap / closed-form var_μ: median over pools; per pool (estimation part; "
                  "* recorded only, ! above the bar)"]
    for obj, r6 in c["R6"].items():
        lines.append(f"    {obj:<16} median {_f(r6['median'], '.2f')} {'pass' if r6['pass'] else 'FAIL'} · " + "  ".join(
            f"{k} {_f(r['ratio'], '.2f')} ({_f(r['ratio_estimation'], '.2f')})"
            f"{'' if r['checked'] else '*'}{'!' if r['flag'] else ''}" for k, r in r6["pools"].items()))
    lines += ["", "Class priors at the last vintage: μ ± se_μ, τ, k"]
    for cls, by_obj in result["summary"].items():
        for obj, r in by_obj.items():
            lines.append(f"    {cls:<12} {obj:<16} {_f(r['mu'])} ± {_f(r['se_mu'], '.3f')}  "
                         f"τ {_f(r['tau'], '.3f')}  k {r['k']}")
    da, db, dc = d["A"], d["B"], d["C"]
    lines += ["", f"Decision A (claim centring): {da['value']} — raw fails on {len(da['raw_fails'])}, "
                  f"candidate passes on {len(da['candidate_passes'])}; ΔIC "
                  f"{_f(da['b_delta_ic']['delta'])} ± {_f(da['b_delta_ic']['se'], '.3f')}",
              f"Decision B (SIZERS_ENABLED): {db['value']} — ΔIC(d·∏factor − d) "
              f"{_f(db['delta_ic']['delta'])} ± {_f(db['delta_ic']['se'], '.3f')}; call vs bare claim "
              f"{_f(db['call_vs_bare_claim']['delta'])} ± {_f(db['call_vs_bare_claim']['se'], '.3f')} (information)",
              f"Decision C (TIER_RULE): {dc['value']} — " + "  ".join(
                  f"{t} {_f(r['mean'], '+.5f')} (n/h {r['n_win']})" for t, r in dc["terciles"].items())]
    lines += ["", "R5  per class (claim): false-open · E[f] · power 0.05 / 0.08"]
    for cls, r in (c["R5"].get("claim") or {}).items():
        lines.append(f"    {cls:<12} k {r['k']}  {r['false_open']:.3f} · {r['e_strength_null']:.3f} · "
                     f"{r['power_0.05']:.2f} / {r['power_0.08']:.2f}  ({r['scope']})")
    lines += ["", "Arm E  lag 0 → lag 1, class μ"]
    for cls, by_obj in a["E"]["per_class"].items():
        lines.append(f"    {cls:<12} " + "  ".join(
            f"{obj} {_f(r.get('L0', {}).get('mu'))}→{_f(r.get('L1', {}).get('mu'))}" for obj, r in by_obj.items()))
    lines += ["", "Arm P  claim IC range (start shifts, jackknife)"]
    for m, r in a["P"].items():
        rng = r.get("range") or [None, None]
        lines.append(f"    {m:<16} base {_f(r['base_ic'])}  range {_f(rng[0])} … {_f(rng[1])}")
    if timings:
        t = np.array(list(timings.values()))
        lines += ["", f"Wall time per member: median {np.median(t):.0f} s, max {t.max():.0f} s, "
                      f"total {t.sum() / 60:.1f} min ({len(t)} runs this session)"]
    if result["abort"]:
        lines += ["", "ABORT — nothing written: " + "; ".join(result["abort"])]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Running it
# ════════════════════════════════════════════════════════════════════════

def _offline_fetch() -> SimpleNamespace:
    """The committed fixtures served in place of the four network fetches (the
    stubs tests/conftest.py installs)."""
    from studies.make_fixtures import TARGETS, load_ohlcv, load_panel

    panel = load_panel()
    ohlcv = {instrument(t).ticker: load_ohlcv(t) for t in TARGETS}

    def _window(idx, start, end):
        idx = pd.DatetimeIndex(idx)
        return (idx >= pd.Timestamp(start).normalize()) & (idx <= pd.Timestamp(end))

    def fetch_commodity_dataset(start_date, end_date, force=False):
        df = panel[_window(panel["DATE"], start_date, end_date)].reset_index(drop=True)
        df.attrs = dict(panel.attrs)
        return df, None

    def fetch_ohlcv(symbols, start_date, end_date, force=False):
        return {s: o[_window(o.index, start_date, end_date)] for s, o in ohlcv.items() if s in symbols}

    return SimpleNamespace(fetch_commodity_dataset=fetch_commodity_dataset, fetch_ohlcv=fetch_ohlcv,
                           fetch_stock_target_series=lambda *a, **k: None,
                           fetch_sheet_series=lambda *a, **k: None,
                           last_date=pd.Timestamp(panel["DATE"].max()).normalize())


def _git_state() -> dict:
    """HEAD and the tracked or new source files that differ from it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    sha = git("rev-parse", "HEAD").strip() or "unknown"
    status = git("status", "--porcelain", "--", "core", "data", "engines", "pipeline",
                 "analytics", "studies")
    dirty = sorted(line[3:] for line in status.splitlines()
                   if line[3:].endswith(".py") and not line[3:].startswith("data/priors/"))
    return {"sha": sha, "dirty": dirty}


def _code_key() -> str:
    """The member stage's code — the pipeline and this script's member functions
    and constants — so a cached record from other code is re-run, while a
    change to the pooling alone re-pools the cached members."""
    hsh = hashlib.sha256(repr((RECORD_VERSION, STUDY_STOCKS, P_SHIFTS, P_JACKKNIFE, SEED)).encode())
    for d in ("core", "data", "engines", "pipeline", "analytics"):
        for p in sorted((ROOT / d).rglob("*.py")):
            hsh.update(p.read_bytes())
    for fn in (_claim_ic, path_sensitivity, run_member):
        hsh.update(inspect.getsource(fn).encode("utf-8"))
    return hsh.hexdigest()[:16]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def run_members(members: list[str], *, run_date: pd.Timestamp, workdir: Path, fetch,
                arm_p: bool, log) -> tuple[dict, dict, dict]:
    """Run (or resume) each member: ``(records, failures, timings)``. A member
    cached from this code and run date is reused; a failed one is retried."""
    key = _code_key()
    cache = workdir / "members"
    cache.mkdir(parents=True, exist_ok=True)
    p_set = set(p_members()) if arm_p else set()
    records, failures, timings = {}, {}, {}
    for n, m in enumerate(members, 1):
        path = cache / f"{_slug(m)}.pkl"
        if path.is_file():
            with open(path, "rb") as fh:
                got = pickle.load(fh)
            if got.get("code") == key and got.get("run_date") == run_date and not got.get("error") \
                    and (m not in p_set or "arm_p" in got["record"]):
                records[m] = got["record"]
                log(f"[{n}/{len(members)}] {m}: cached ({got.get('wall_s', 0):.0f} s when run)")
                continue
        t0 = time.perf_counter()
        try:
            rec, err = run_member(m, run_date=run_date, fetch=fetch, arm_p=m in p_set), None
        except Exception as e:                       # noqa: BLE001 - one member, not the study
            rec, err = None, f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        timings[m] = wall
        with open(path, "wb") as fh:
            pickle.dump({"code": key, "run_date": run_date, "record": rec, "error": err,
                         "wall_s": wall}, fh)
        if err:
            failures[m] = err
            log(f"[{n}/{len(members)}] {m}: FAILED in {wall:.0f} s — {err}")
        else:
            records[m] = rec
            log(f"[{n}/{len(members)}] {m}: ok in {wall:.0f} s")
    return records, failures, timings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The pooled-prior study (pre-registered: see the module docstring).")
    ap.add_argument("--out", type=Path, required=True, help="the output JSON")
    ap.add_argument("--members", help="comma-separated subset of the universe (default: all)")
    ap.add_argument("--workdir", type=Path, help="per-member cache and log (resumable)")
    ap.add_argument("--run-date", help="YYYY-MM-DD (default: the workdir's, else today)")
    ap.add_argument("--previous", type=Path, help="the previous study's output, for arm F "
                                                  "(default: the --out file, if it exists)")
    ap.add_argument("--offline-fixture", action="store_true",
                    help="run on the committed offline fixtures (Gold, USD/INR), no network")
    ap.add_argument("--no-arm-p", action="store_true", help="skip arm P (a plumbing run: not complete)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run on uncommitted source (recorded in git_dirty)")
    args = ap.parse_args(argv)

    names = universe()
    fetch = _offline_fetch() if args.offline_fixture else None
    mode = "offline-fixture" if args.offline_fixture else "network"
    members = ([m.strip() for m in args.members.split(",") if m.strip()] if args.members
               else list(FIXTURE_MEMBERS) if args.offline_fixture else names)
    unknown = [m for m in members if m not in names]
    if unknown:
        print(f"not in the study universe: {unknown}", file=sys.stderr)
        return 1
    git = _git_state()
    if git["dirty"] and not args.allow_dirty:
        print(f"uncommitted source: {git['dirty']} — commit first, or pass --allow-dirty",
              file=sys.stderr)
        return 1

    meta_dir = args.workdir
    recorded = None
    if meta_dir is not None and (meta_dir / "study.json").is_file():
        recorded = json.loads((meta_dir / "study.json").read_text())["run_date"]
    run_date = pd.Timestamp(args.run_date or recorded or
                            (fetch.last_date if fetch is not None else pd.Timestamp.today())).normalize()
    workdir = meta_dir or Path.home() / ".cache" / "tattva" / "pooled_priors" / f"{mode}-{run_date:%Y-%m-%d}"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "study.json").write_text(json.dumps({"run_date": run_date.strftime("%Y-%m-%d"),
                                                    "mode": mode}))
    log_path = workdir / "study.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"study {mode} · run date {run_date:%Y-%m-%d} · {len(members)} members · workdir {workdir}")
    records, failures, timings = run_members(members, run_date=run_date, workdir=workdir, fetch=fetch,
                                             arm_p=not args.no_arm_p, log=log)
    timing_file = workdir / "timings.json"
    old = json.loads(timing_file.read_text()) if timing_file.is_file() else {}
    timing_file.write_text(json.dumps({**old, **{m: round(t, 1) for m, t in timings.items()}}, indent=1))
    if not records:
        log("no member ran: nothing to pool")
        return 1
    previous = args.previous or (args.out if args.out.is_file() else None)
    t0 = time.perf_counter()
    result = analyse(records, universe_names=names, run_date=run_date, previous=previous)
    log(f"analysis in {time.perf_counter() - t0:.0f} s")
    text = report(result, {**old, **timings})
    print(text)
    (workdir / "report.txt").write_text(text + "\n", encoding="utf-8")
    complete = set(records) | set(failures) == set(names) and not args.no_arm_p
    code = finish(result, records, args.out, universe_names=names, failures=failures,
                  run_date=run_date, git=git, mode=mode, complete=complete)
    log(f"wrote {args.out}" if code == 0 else f"aborted (exit {code}): nothing written")
    return code


if __name__ == "__main__":
    sys.exit(main())
