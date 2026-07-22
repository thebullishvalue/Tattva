"""
Tattva — Intelligent Tuning Orchestrator (interactive + segment-aware).

ONE entrypoint that runs the research/tuning suite and produces a single
consolidated report + a "current vs study-validated" reference for every tuned
constant. Intended as a PERIODIC re-tune (e.g. after a data refresh), not a hot
path.

Design choice — it REPORTS, it does not auto-apply. Every tuning decision needs
human judgement (interactions, regime shifts, redundancy). Auto-writing config
from a re-run invites overfitting/regime-chasing. So: run → read the report →
apply by hand, guided by the reference table this prints.

The suite is organised into SEGMENTS (tiers). You can run everything end-to-end,
a single segment, a hand-picked set of studies, or just the fast correctness
tests — interactively (a menu when you run with no flags on a terminal) or via
flags for scripting.

Usage:
    python3 research/run_tuning.py                      # interactive menu (on a TTY)
    python3 research/run_tuning.py --list               # show the suite + ETAs
    python3 research/run_tuning.py --all                # run EVERY tuning study
    python3 research/run_tuning.py --all --fresh        # …end-to-end FROM SCRATCH (no resume)
    python3 research/run_tuning.py --segment engines    # run one segment (tier)
    python3 research/run_tuning.py --tests              # run only the correctness tests
    python3 research/run_tuning.py --only analog,nirnay # run a hand-picked subset
    python3 research/run_tuning.py --skip aarambh_full  # everything except the long sweep

--fresh clears the persistent study-result cache (research/.tune_cache/, the
aarambh resume CSV) so nothing is carried over from a previous report — every
result is recomputed. It does NOT clear the raw market-data cache (~/.cache/tattva);
that is shared and expensive to refetch, and the preflight re-warms it anyway.

Reports land in research/reports/tuning_<timestamp>.txt.

Reflects the current system: single fixed forecast horizon (no lens selector),
commodities + individual stocks in Nirnay-Swayam self mode, per-instrument /
per-asset InstrumentConfig, and the fixed 1/3/5/10/20/60d precedent term
structure. See core/config.py (InstrumentConfig / CLASS_CONFIG_DEFAULTS).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# The suite prints Unicode (box-drawing, spinner, ✓/✗) throughout. On Windows the
# console defaults to a legacy codepage (cp1252) that can't encode these, so a bare
# print() crashes with UnicodeEncodeError. Force this process's stdout/stderr to
# UTF-8 (errors="replace" so it degrades instead of crashing on any exotic glyph).
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover — very old/patched streams
        pass

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE / "reports"

# Preflight sufficiency floors. Every TUNING study pulls the SAME 9-year universe
# via fetch_commodity_dataset (cache→live), so one thin/rate-limited fetch would
# silently corrupt the whole multi-hour run. We warm + validate it ONCE up front.
_PREFLIGHT_MIN_COLS = 50
_PREFLIGHT_MIN_TARGET_FRAC = 0.6

# ── The suite ────────────────────────────────────────────────────────────────
# key → (script, title, rough ETA minutes, is_test).
# Ordered bottom-up through the architecture: each tier is tuned before the
# layers that consume it, so the report reads top-to-bottom with every later
# result resting on decisions already settled above it.
SUITE = [
    # ── T1 · Engines (primitives) ────────────────────────────────────────────
    ("aarambh_full",   "aarambh_tuning_study.py",     "[T1 engine] Aarambh defaults — REFIT/ENSEMBLE/MAX/MIN/PCA (post-purge)", 130, False),
    ("aarambh_maxmin", "confirm_max_sweep.py",        "[T1 engine] Aarambh MAX_TRAIN × live-MIN interaction confirm",          50, False),
    ("nirnay",         "nirnay_tuning_study.py",       "[T1 engine] Nirnay structural knobs (basket-mode: FX + Jeera + indices)", 85, False),
    ("nirnay_index",   "nirnay_index_check.py",        "[T1 engine] Nirnay MSF_LENGTH generalization on equity indices",        22, False),
    ("nirnay_swayam",  "nirnay_swayam_study.py",       "[T1 engine] Nirnay-Swayam A/B efficacy — self-ensemble vs basket breadth IC", 30, False),
    ("swayam",         "swayam_tuning_study.py",       "[T1 engine] Swayam GRID tune — swayam_lengths spans + roc_frac (self mode)", 45, False),
    ("ddm",            "ddm_smoothing_study.py",       "[T1 engine] DDM leak sweep at constant gain (consensus + engine filters)", 16, False),
    # ── T2 · Horizon scope (frames the analog + the fixed forecast horizon) ──
    ("precedent_univ", "precedent_universe_sweep.py",  "[T2 scope] Analog horizon scope across the universe → forecast + precedent horizons", 28, False),
    # ── T3 · Analog engine (within the scope, on Aarambh ts_data) ────────────
    ("analog",         "analog_tuning_study.py",       "[T3 analog] blend/TOP_N/recency/features/aggregation",                  12, False),
    ("analog_confirm", "analog_confirm.py",            "[T3 analog] combined config confirm (maha-only + drop-AvgZ)",           6, False),
    # ── T4 · Cross-engine validation (post-purge model vs tuned analog) ──────
    ("precedent_model","precedent_vs_model_sweep.py",  "[T4 validate] purged model vs analog by asset class",                   40, False),
    ("precedent_horiz","precedent_study.py",           "[T4 validate] model vs analog at 2..120d on one target",                5, False),
    # ── T5 · Interpretation (apex layers that consume everything above) ──────
    ("markers",        "markers_study.py",             "[T5 interp] Unified-Signal plot marker thresholds (quantile-anchored)", 6, False),
    ("hero_thresholds","hero_threshold_study.py",      "[T5 interp] hero BUY/SELL/STRONG classification cut-points",            10, False),
    ("hero",           "hero_study.py",                "[T5 interp] hero interpretation — convergence vs +markers vs +precedent", 4, False),
    ("calibration_lift","calibration_lift_study.py",   "[T5 interp] hero headline arms — consensus vs raw vs CALIBRATED (paired OOS)", 45, False),
    ("conv_weights",   "conv_weights_study.py",        "[T5 interp] factory dim weights (CONV_WEIGHT_*) — unfitted vector sweep", 20, False),
    ("ui_anchors",     "ui_anchors_study.py",          "[T5 interp] remaining UI/tier constants — distribution anchors + occupancy", 12, False),
    # ── T6 · Per-asset / per-instrument configs (InstrumentConfig registry) ──
    ("per_asset",      "per_asset_config_study.py",    "[T6 configs] Per-asset configs for the UNCOVERED classes — US indices · ETF · India stocks=Nifty100 · US stocks=Nasdaq100", 55, False),
    # ── Correctness tests (assertion-based; no live fetch, no tuning) ────────
    ("t_configs",      "test_instrument_configs.py",   "[test] per-instrument config registry integrity",                       1, True),
    ("t_aarambh",      "test_aarambh_config.py",        "[test] aarambh per-instrument config threading into the engine",       1, True),
    ("t_swayam",       "test_nirnay_swayam.py",         "[test] Nirnay-Swayam ensemble integrity",                              1, True),
    ("t_stocks",       "test_stock_targets.py",         "[test] individual-stock target fetch + symbol resolution",             1, True),
    ("t_convergence",  "test_convergence_integrity.py", "[test] convergence signal-chain integrity",                            1, True),
    ("t_analog",       "test_analog_series.py",         "[test] analog prediction series (causality)",                         1, True),
    ("t_regime",       "test_regime_equivalence.py",    "[test] regime kernel vs reference equivalence",                        1, True),
    ("t_polarity",     "test_polarity.py",              "[test] apply_polarity inverse-basket path",                           1, True),
    ("t_hero",         "test_hero_verdict.py",          "[test] hero-verdict decision table",                                  1, True),
    ("t_calendars",    "test_calendars.py",             "[test] per-exchange freshness calendars",                            1, True),
]
_BY_KEY = {k: (s, t, e, is_test) for k, s, t, e, is_test in SUITE}

# Studies that keep a PERSISTENT on-disk result cache and understand `--fresh`
# (recompute from scratch instead of resuming). Only aarambh's long sweep does —
# every other study recomputes in-process each run, so a fresh run is automatic.
_FRESH_CAPABLE = {"aarambh_full"}
# Persistent study-result caches wiped by a "from scratch" run. NOT the raw
# market-data cache (~/.cache/tattva) — that is shared and costly to refetch.
_TUNE_CACHE_DIR = _HERE / ".tune_cache"

# Segments (tiers) → ordered study keys. The interactive menu and --segment
# select these. "all" is every TUNING study (tests excluded); "tests" is the
# fast correctness segment.
SEGMENTS: dict[str, tuple[str, list[str]]] = {
    "engines":    ("T1 · Engine primitives (Aarambh · Nirnay · Swayam · DDM)",
                   ["aarambh_full", "aarambh_maxmin", "nirnay", "nirnay_index", "nirnay_swayam", "swayam", "ddm"]),
    "scope":      ("T2 · Horizon scope (forecast + precedent term structure)",
                   ["precedent_univ"]),
    "analog":     ("T3 · Analog engine (blend / features)",
                   ["analog", "analog_confirm"]),
    "validation": ("T4 · Cross-engine validation (model vs analog)",
                   ["precedent_model", "precedent_horiz"]),
    "interp":     ("T5 · Interpretation (markers · hero · thresholds · weights · UI)",
                   ["markers", "hero_thresholds", "hero", "calibration_lift", "conv_weights", "ui_anchors"]),
    "per_asset":  ("T6 · Per-asset configs for classes no other tier covers (US idx · ETF · stocks)",
                   ["per_asset"]),
    "tests":      ("Correctness tests (fast, assertion-based — no tuning)",
                   [k for k, *_ in SUITE if _BY_KEY[k][3]]),
}
_TUNING_KEYS = [k for k, *_ in SUITE if not _BY_KEY[k][3]]   # everything except tests

# Tuned constants → validating study key(s). Printed at the end with the LIVE
# current value so you can eyeball drift after a re-run. These are now the
# DEFAULT values of core.config.InstrumentConfig (each instrument's registry
# entry may override); the studies validate the defaults / per-asset baselines.
CONFIG_REF = [
    ("MIN_TRAIN_SIZE",        "core.config",            ["aarambh_full", "aarambh_maxmin"]),
    ("MAX_TRAIN_SIZE",        "core.config",            ["aarambh_full", "aarambh_maxmin"]),
    ("REFIT_INTERVAL",        "core.config",            ["aarambh_full"]),
    ("ENSEMBLE_MODELS",       "core.config",            ["aarambh_full"]),
    ("HUBER_EPSILON",         "core.config",            ["aarambh_full"]),
    ("LOOKBACK_WINDOWS",      "core.config",            ["aarambh_full"]),
    ("DDM_LEAK_RATE",         "core.config",            ["ddm"]),
    ("CONV_DDM_LEAK_RATE",    "core.config",            ["ddm"]),
    ("CONV_WEIGHT_DIRECTION", "core.config",            ["conv_weights"]),
    ("CONVICTION_STRONG",     "core.config",            ["ui_anchors"]),
    ("UI_AGREEMENT_STRONG",   "core.config",            ["ui_anchors"]),
    ("UI_MODEL_SPREAD_HIGH",  "core.config",            ["ui_anchors"]),
    ("NIRNAY_OVERSOLD",       "core.config",            ["ui_anchors"]),
    # Nirnay structural knobs — InstrumentConfig defaults. `nirnay` validates
    # the basket-mode classes; `swayam`/`per_asset` validate the self-mode ones.
    ("NIRNAY_MSF_LENGTH",     "core.config",            ["nirnay", "nirnay_index", "per_asset"]),
    ("NIRNAY_ROC_LEN",        "core.config",            ["nirnay", "per_asset"]),
    ("NIRNAY_REGIME_SENSITIVITY", "core.config",        ["nirnay", "per_asset"]),
    ("NIRNAY_BASE_WEIGHT",    "core.config",            ["nirnay", "per_asset"]),
    ("NIRNAY_MMR_NUM_VARS",   "core.config",            ["nirnay", "per_asset"]),
    # Swayam grid — self-ensemble tunables.
    ("NIRNAY_SWAYAM_LENGTHS", "core.config",            ["swayam", "per_asset"]),
    ("NIRNAY_SWAYAM_ROC_FRAC","core.config",            ["swayam", "per_asset"]),
    # Forecast + precedent horizons (single fixed horizon; no lens selector).
    ("FORECAST_HORIZON",      "core.config",            ["precedent_univ", "precedent_model"]),
    ("FORECAST_MOMENTUM",     "core.config",            ["precedent_univ"]),
    ("PRECEDENT_HORIZONS",    "core.config",            ["precedent_univ"]),
    ("ANALOG_W_MAHA",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("ANALOG_W_TRAJ",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("ANALOG_W_RECV",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("UI_CONSENSUS_STRONG",   "core.config",            ["markers"]),
    ("UI_CONVRAW_STRONG",     "core.config",            ["markers"]),
    ("UI_NIRNAY_AVG_THRESHOLD", "core.config",          ["markers"]),
    ("DEFAULT_THRESHOLDS",    "convergence.normalization", ["hero_thresholds"]),
    ("COMPOSITE_THRESHOLDS",  "convergence.normalization", ["hero_thresholds"]),
]

_NOISE = ("Numba", "numba", "UserWarning", "FutureWarning", "HTTP Error",
          "Failed download", "delisted", "possibly delisted", "YFTz", "unavailable after backfill")


def _preflight(warn_only: bool = False) -> bool:
    """Fetch + validate the shared universe ONCE before the tuning suite runs."""
    import numpy as np
    import pandas as pd
    from core.config import MIN_DATA_POINTS, ALL_TARGETS
    from data.fetcher import fetch_commodity_dataset

    print("Preflight — fetching/validating the shared universe (this also warms the "
          "cache for every study)…", flush=True)
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * 9)
    t0 = time.time()
    try:
        df, err = fetch_commodity_dataset(start, end)
    except Exception as e:  # noqa: BLE001
        df, err = None, f"fetch raised: {e}"

    problems: list[str] = []
    n_rows = n_cols = n_tgt = 0
    span = ""
    if df is None or df.empty:
        problems.append(f"fetch returned no data ({err})")
    else:
        n_rows = len(df)
        n_cols = int(df.select_dtypes(include=[np.number]).shape[1])
        n_tgt = sum(1 for t in ALL_TARGETS if t in df.columns)
        if "DATE" in df.columns:
            span = f"{pd.to_datetime(df['DATE']).min():%Y-%m-%d} … {pd.to_datetime(df['DATE']).max():%Y-%m-%d}"
        if n_rows < MIN_DATA_POINTS:
            problems.append(f"only {n_rows} rows (< MIN_DATA_POINTS={MIN_DATA_POINTS}) — walk-forward can't run")
        if n_cols < _PREFLIGHT_MIN_COLS:
            problems.append(f"only {n_cols} numeric columns (universe looks partial; a healthy pull is ~150+)")
        need_tgt = int(len(ALL_TARGETS) * _PREFLIGHT_MIN_TARGET_FRAC)
        if n_tgt < need_tgt:
            problems.append(f"only {n_tgt}/{len(ALL_TARGETS)} targets present (need ≥ {need_tgt})")

    dt = time.time() - t0
    if problems:
        head = "PREFLIGHT FAILED" if not warn_only else "PREFLIGHT WARNING"
        print("\n" + "!" * 90)
        print(f"  {head} (checked in {dt:.0f}s) — got {n_rows} rows · {n_cols} cols · "
              f"{n_tgt} targets {('· ' + span) if span else ''}")
        for p in problems:
            print(f"    - {p}")
        print("\n  Almost always a yfinance rate-limit / partial pull. Wait a few minutes and "
              "retry, or\n  run the app's 'Refresh Data' once to repopulate ~/.cache/tattva. "
              "Every study shares\n  this fetch, so a thin pull would corrupt the entire run.")
        print("  Bypass with --skip-preflight (or --preflight-warn to proceed anyway).")
        print("!" * 90 + "\n", flush=True)
        return bool(warn_only)

    print(f"Preflight OK ({dt:.0f}s) — {n_rows} rows · {n_cols} numeric cols · "
          f"{n_tgt}/{len(ALL_TARGETS)} targets · {span}\n", flush=True)
    return True


def _list():
    tuning_eta = sum(_BY_KEY[k][2] for k in _TUNING_KEYS)
    print(f"\nTattva research suite — {len(_TUNING_KEYS)} tuning studies "
          f"(~{tuning_eta} min / ~{tuning_eta/60:.1f}h full) + "
          f"{len(SEGMENTS['tests'][1])} correctness tests\n")
    for seg, (desc, keys) in SEGMENTS.items():
        seg_eta = sum(_BY_KEY[k][2] for k in keys)
        print(f"  ── {seg}  ({desc})  ~{seg_eta} min")
        for k in keys:
            _, title, eta, _is_test = _BY_KEY[k]
            print(f"       {k:<16} {eta:>4}m  {title}")
    print("\n  Run: python3 research/run_tuning.py [--all | --segment <name> | --tests |")
    print("                                       --only k1,k2 | --skip k1,k2]")


def _clear_study_caches() -> None:
    """Wipe the PERSISTENT study-result cache so a run recomputes FROM SCRATCH,
    carrying nothing from a previous report. Deliberately does NOT touch the raw
    market-data cache (~/.cache/tattva) — that is shared and expensive to refetch,
    and the preflight re-warms it."""
    import shutil
    if _TUNE_CACHE_DIR.exists():
        n = sum(1 for p in _TUNE_CACHE_DIR.rglob("*") if p.is_file())
        shutil.rmtree(_TUNE_CACHE_DIR, ignore_errors=True)
        print(f"From scratch — cleared {n} cached study-result file(s) in {_TUNE_CACHE_DIR} "
              f"(raw market-data cache preserved).", flush=True)
    else:
        print("From scratch — no persistent study-result cache to clear (already clean).", flush=True)


_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"   # braille spinner frames


def _fmt_dur(s: float) -> str:
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"


def _stream_output(proc, key, eta, report, t0):
    """Stream the subprocess output line-by-line to the console + report, with a
    live HEARTBEAT during silent stretches so a long-running study never looks
    frozen (some chunks run 10+ min between prints). The heartbeat is console-only
    (never written to the report) and shown only on an interactive TTY — piped /
    redirected output keeps the plain line stream. Uses select() so it stays
    single-threaded; falls back to a blocking read where select isn't usable."""
    live = _sys.stdout.isatty() and _sys.platform != "win32"
    fd = proc.stdout
    last_out = time.time()
    hb_shown = False
    spin_i = 0
    try:
        import select
    except Exception:
        live = False
    while True:
        if live:
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                idle = time.time() - last_out
                if idle >= 2.0:   # only after a real silence, not between prints
                    spin_i = (spin_i + 1) % len(_SPIN)
                    _sys.stdout.write(
                        f"\r\033[K  {_SPIN[spin_i]} {key} · {_fmt_dur(time.time()-t0)} elapsed "
                        f"(~{eta}m est) · quiet {_fmt_dur(idle)}")
                    _sys.stdout.flush()
                    hb_shown = True
                continue
        line = fd.readline()
        if not line:
            break
        if hb_shown:                       # wipe the heartbeat before real output
            _sys.stdout.write("\r\033[K"); hb_shown = False
        report.write(line)
        if not any(n in line for n in _NOISE):
            _sys.stdout.write(line); _sys.stdout.flush()
        last_out = time.time()
    if hb_shown:
        _sys.stdout.write("\r\033[K"); _sys.stdout.flush()


def _run_one(key, report, fresh: bool = False):
    script, title, eta, _is_test = _BY_KEY[key]
    banner = f"\n{'='*90}\n### {key} · {title}\n### script: {script} · ~{eta} min · started {datetime.now():%H:%M:%S}\n{'='*90}\n"
    print(banner, flush=True); report.write(banner)
    t0 = time.time()
    # Put the repo root on PYTHONPATH so `from core...`/`from ui...` resolve for
    # EVERY script — the tuning studies carry their own path shim, but the
    # assertion tests were written to run via `-m research.<name>` / PYTHONPATH=.,
    # so a bare-file invocation needs the root injected here.
    _root = str(_HERE.parent)
    # Force child studies to emit UTF-8 too: they print the same Unicode glyphs, and
    # a child writing to its pipe under a legacy Windows codepage would crash before
    # the parent ever decodes it. PYTHONUTF8=1 + PYTHONIOENCODING pin the child's I/O
    # to UTF-8, matching this process's Popen(encoding="utf-8") on the read side.
    _env = {**_os.environ, "PYTHONPATH": _root + _os.pathsep + _os.environ.get("PYTHONPATH", ""),
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    _cmd = [_sys.executable, "-u", str(_HERE / script)]
    if fresh and key in _FRESH_CAPABLE:   # tell the study to ignore any resume cache
        _cmd.append("--fresh")
    proc = subprocess.Popen(_cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=_env)
    _stream_output(proc, key, eta, report, t0)
    proc.wait()
    foot = f"\n[{key}] exit {proc.returncode} in {time.time()-t0:.0f}s\n"
    print(foot, flush=True); report.write(foot)
    return proc.returncode


def _config_reference(report=None):
    # Mirror every line to BOTH the console and the report file so the consolidated
    # report actually ENDS with this reference (as the README promises). It used to
    # print()-only, so the report .txt stopped at the last study and the documented
    # "current-vs-validated reference" was console-only.
    def emit(line=""):
        print(line)
        if report is not None:
            report.write(line + "\n")
    emit("\n" + "=" * 90)
    emit("  TUNED-CONFIG REFERENCE — live current value · validating study(ies)")
    emit("  (these are InstrumentConfig DEFAULTS; per-instrument entries may override)")
    emit("=" * 90)
    import importlib
    cache = {}
    def _get(mod, name):
        if mod not in cache:
            try:
                cache[mod] = importlib.import_module(mod)
            except Exception:
                cache[mod] = None
        m = cache[mod]
        return getattr(m, name, "?") if m else "?"
    emit(f"  {'constant':<26} {'current':<24} validated by")
    emit("  " + "-" * 84)
    for name, mod, keys in CONFIG_REF:
        val = _get(mod, name)
        emit(f"  {name:<26} {str(val):<24} {', '.join(keys)}")
    emit("\n  Review the report above against these — apply changes by hand (see CHANGELOG /"
         "\n  the rationale comments in core/config.py). The orchestrator never auto-edits config.")


def _resolve_keys(args) -> tuple[list[str], bool]:
    """Resolve the CLI flags to an ordered key list. Returns (keys, is_tests_only)."""
    if args.tests:
        return SEGMENTS["tests"][1], True
    if args.segment:
        seg = args.segment.strip()
        if seg not in SEGMENTS:
            print(f"Unknown segment {seg!r}. Choices: {', '.join(SEGMENTS)}")
            return [], False
        return SEGMENTS[seg][1], seg == "tests"
    # Default population: all TUNING studies (tests excluded unless asked).
    keys = list(_TUNING_KEYS)
    if args.all:
        keys = list(_TUNING_KEYS)
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        keys = [k for k, *_ in SUITE if k in want]
    if args.skip:
        drop = {x.strip() for x in args.skip.split(",")}
        keys = [k for k in keys if k not in drop]
    return keys, all(_BY_KEY[k][3] for k in keys) if keys else False


def _interactive() -> tuple[list[str], bool, bool] | None:
    """Terminal menu. Returns (keys, is_tests_only, fresh) or None to quit."""
    seg_names = list(SEGMENTS)
    while True:
        tuning_eta = sum(_BY_KEY[k][2] for k in _TUNING_KEYS)
        print("\n" + "═" * 74)
        print("  Tattva — Research & Tuning Suite")
        print("═" * 74)
        print("  Segments:")
        for i, seg in enumerate(seg_names, 1):
            desc, keys = SEGMENTS[seg]
            seg_eta = sum(_BY_KEY[k][2] for k in keys)
            print(f"    {i}) {seg:<11} ~{seg_eta:>4}m  {desc}")
        print("  Actions:")
        print(f"    a) Run EVERYTHING end-to-end            (~{tuning_eta/60:.1f}h, {len(_TUNING_KEYS)} tuning studies)")
        print(f"    f) Run EVERYTHING end-to-end FROM SCRATCH (~{tuning_eta/60:.1f}h; wipes cached study")
        print( "                                             results first — nothing carried from a prior report)")
        print("    s) Pick specific studies (comma-separated keys)")
        print("    l) List every study")
        print("    q) Quit")
        choice = input("  Select [1-%d / a / f / s / l / q]: " % len(seg_names)).strip().lower()

        if choice in ("q", ""):
            return None
        if choice == "l":
            _list(); continue
        if choice == "a":
            return list(_TUNING_KEYS), False, False
        if choice == "f":
            return list(_TUNING_KEYS), False, True
        if choice == "s":
            raw = input("  Study keys (comma-separated; 'l' to list): ").strip()
            if raw.lower() == "l":
                _list(); continue
            want = {x.strip() for x in raw.split(",") if x.strip()}
            keys = [k for k, *_ in SUITE if k in want]
            unknown = want - set(_BY_KEY)
            if unknown:
                print(f"  Unknown keys ignored: {', '.join(sorted(unknown))}")
            if not keys:
                print("  Nothing recognised — try again."); continue
            fresh = False
            if any(k in _FRESH_CAPABLE for k in keys):
                fresh = input("  From scratch? wipe cached study results first [y/N]: ").strip().lower() == "y"
            return keys, all(_BY_KEY[k][3] for k in keys), fresh
        if choice.isdigit() and 1 <= int(choice) <= len(seg_names):
            seg = seg_names[int(choice) - 1]
            keys = SEGMENTS[seg][1]
            fresh = False
            if any(k in _FRESH_CAPABLE for k in keys):
                fresh = input("  From scratch? wipe cached study results first [y/N]: ").strip().lower() == "y"
            return keys, seg == "tests", fresh
        print("  Unrecognised choice — try again.")


def main():
    ap = argparse.ArgumentParser(description="Tattva intelligent tuning orchestrator")
    ap.add_argument("--list", action="store_true", help="show the suite and exit")
    ap.add_argument("--all", action="store_true", help="run every TUNING study (non-interactive)")
    ap.add_argument("--segment", default="", help=f"run one segment: {', '.join(SEGMENTS)}")
    ap.add_argument("--tests", action="store_true", help="run ONLY the correctness tests")
    ap.add_argument("--only", default="", help="comma-separated study keys to run")
    ap.add_argument("--skip", default="", help="comma-separated study keys to skip")
    ap.add_argument("--skip-preflight", action="store_true", help="skip the up-front data sufficiency check")
    ap.add_argument("--preflight-warn", action="store_true", help="run preflight but only WARN (don't abort)")
    ap.add_argument("--fresh", action="store_true",
                    help="from scratch: wipe the persistent study-result cache (aarambh resume CSV) "
                         "so nothing carries from a previous report (raw data cache preserved)")
    args = ap.parse_args()

    if args.list:
        _list(); return

    # Decide the run set. Interactive menu only when no selection flags were
    # given AND we're on a real terminal (so piped/CI invocation never hangs).
    _selection_flags = args.all or args.segment or args.tests or args.only or args.skip
    if not _selection_flags and _sys.stdin.isatty():
        picked = _interactive()
        if picked is None:
            print("Nothing selected. Bye."); return
        keys, tests_only, fresh = picked
    else:
        keys, tests_only = _resolve_keys(args)
        fresh = args.fresh
        if not _selection_flags and not _sys.stdin.isatty():
            # Non-interactive with no flags → safest default is the fast tests,
            # not a silent multi-hour run.
            print("No flags and no TTY — defaulting to the correctness tests only. "
                  "Use --all for the full tuning suite.")
            keys, tests_only = SEGMENTS["tests"][1], True

    if not keys:
        print("No studies selected. Use --list to see keys."); return

    # "From scratch" only matters for tuning studies with a persistent cache.
    if fresh and tests_only:
        fresh = False
    if fresh and not any(k in _FRESH_CAPABLE for k in keys):
        print("Note: --fresh has no effect on this selection (no study here keeps a "
              "persistent cache); every one already recomputes each run.")

    # Tests need no live fetch; tuning studies do → preflight only for those.
    if not tests_only and not args.skip_preflight:
        if not _preflight(warn_only=args.preflight_warn):
            print("Aborted before running any study (preflight). Nothing was tuned.")
            return

    if fresh:
        _clear_study_caches()

    eta = sum(_BY_KEY[k][2] for k in keys)
    _REPORTS.mkdir(exist_ok=True)
    rpath = _REPORTS / f"tuning_{datetime.now():%Y%m%d_%H%M%S}.txt"
    label = "tests" if tests_only else "studies"
    _mode = "  ·  FROM SCRATCH (no resume)" if fresh else ""
    print(f"\nRunning {len(keys)} {label} (~{eta} min / ~{eta/60:.1f}h){_mode}: {', '.join(keys)}")
    print(f"Report → {rpath}\n")

    t0 = time.time()
    results = {}
    with open(rpath, "w", encoding="utf-8") as report:
        report.write(f"Tattva tuning run · {datetime.now():%Y-%m-%d %H:%M} · {label}={keys}"
                     f"{' · FROM SCRATCH' if fresh else ''}\n")
        for k in keys:
            results[k] = _run_one(k, report, fresh=fresh)
        if not tests_only:
            _config_reference(report)

    print("\n" + "=" * 90)
    print(f"  DONE — {len(keys)} {label} in {(time.time()-t0)/60:.1f} min")
    for k in keys:
        print(f"    {k:<16} {'OK' if results[k] == 0 else 'FAILED (exit %d)' % results[k]}")
    print(f"  Full report: {rpath}")


if __name__ == "__main__":
    main()
