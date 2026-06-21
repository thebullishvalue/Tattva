"""
Tattva — Intelligent Tuning Orchestrator.

ONE command that re-runs the whole research/tuning suite we built and produces a
single consolidated report + a "current vs study-validated" reference for every
tuned constant. Intended as a PERIODIC re-tune (e.g. after a data refresh), not a
hot path.

Design choice — it REPORTS, it does not auto-apply. Every tuning decision we made
needed human judgement (interactions, regime shifts, redundancy). Auto-writing
config from a re-run invites overfitting/regime-chasing. So: run → read the report
→ apply by hand (as we did), guided by the reference table this prints.

Usage:
    python3 research/run_tuning.py --list                # show the suite + ETAs
    python3 research/run_tuning.py                       # run ALL (multi-hour)
    python3 research/run_tuning.py --only analog,nirnay  # run a subset
    python3 research/run_tuning.py --skip aarambh_full   # everything except the 2h sweep

Reports land in research/reports/tuning_<timestamp>.txt.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE / "reports"

# key → (script, title, rough ETA minutes).
# ORDER IS DELIBERATE — it walks the system bottom-up, each tier tuned before the
# layers that consume it (so when you read the report top-to-bottom, every later
# result rests on decisions already made above it):
#
#   T1 Engines      → the primitives (Aarambh forecast, Nirnay breadth) that every
#                     downstream layer is built from.
#   T2 Horizon scope→ which forward horizons the analog actually works at; this sets
#                     SIGNAL_HORIZONS and therefore FRAMES the analog tuning below.
#   T3 Analog engine→ best blend/features WITHIN that horizon scope, on Aarambh ts_data.
#   T4 Validation   → purged model vs the (now-tuned) analog — confirms which signal
#                     leads, given T1+T3 are settled.
#   T5 Interpretation→ the apex layers (convergence markers, hero) that consume all
#                     of the above.
SUITE = [
    # ── T1 · Engines (primitives) ────────────────────────────────────────────
    ("aarambh_full",   "aarambh_tuning_study.py",      "[T1 engine] Aarambh defaults — REFIT/ENSEMBLE/MAX/MIN/PCA (post-purge, 33)", 140),
    ("aarambh_maxmin", "confirm_max_sweep.py",          "[T1 engine] Aarambh MAX_TRAIN × MIN=750 interaction confirm",               25),
    ("nirnay",         "nirnay_tuning_study.py",        "[T1 engine] Nirnay structural knobs (MSF/ROC/sensitivity/base/num_vars)",   14),
    ("nirnay_index",   "nirnay_index_check.py",         "[T1 engine] Nirnay MSF_LENGTH generalization on equity indices",             6),
    # ── T2 · Horizon scope (frames the analog + the lenses) ──────────────────
    ("precedent_univ", "precedent_universe_sweep.py",   "[T2 scope] Analog horizon scope across the universe → SIGNAL_HORIZONS",      16),
    # ── T3 · Analog engine (within the scope, on Aarambh ts_data) ────────────
    ("analog",         "analog_tuning_study.py",        "[T3 analog] blend/TOP_N/recency/features/aggregation (1/10/20d)",            12),
    ("analog_confirm", "analog_confirm.py",             "[T3 analog] combined config confirm (maha-only + drop-AvgZ)",                6),
    # ── T4 · Cross-engine validation (post-purge model vs tuned analog) ──────
    ("precedent_model","precedent_vs_model_sweep.py",   "[T4 validate] purged model vs analog by asset class (10/20d)",               16),
    ("precedent_horiz","precedent_study.py",            "[T4 validate] model vs analog at 5/10/20/90d on one target",                 2),
    # ── T5 · Interpretation (apex layers that consume everything above) ──────
    ("markers",        "markers_study.py",              "[T5 interp] Unified-Signal plot marker thresholds (quantile-anchored)",      6),
    ("hero",           "hero_study.py",                 "[T5 interp] hero interpretation — convergence vs +markers vs +precedent",    4),
]
_BY_KEY = {k: (s, t, e) for k, s, t, e in SUITE}

# Tuned constants → (where defined, the study key(s) that validate it). Printed at
# the end with the LIVE current value so you can eyeball drift after a re-run.
CONFIG_REF = [
    ("MIN_TRAIN_SIZE",        "core.config",            ["aarambh_full", "aarambh_maxmin"]),
    ("MAX_TRAIN_SIZE",        "core.config",            ["aarambh_full", "aarambh_maxmin"]),
    ("REFIT_INTERVAL",        "core.config",            ["aarambh_full"]),
    ("ENSEMBLE_MODELS",       "core.config",            ["aarambh_full"]),
    ("NIRNAY_MSF_LENGTH",     "core.config",            ["nirnay", "nirnay_index"]),
    ("NIRNAY_ROC_LEN",        "core.config",            ["nirnay"]),
    ("NIRNAY_REGIME_SENSITIVITY", "core.config",        ["nirnay"]),
    ("NIRNAY_BASE_WEIGHT",    "core.config",            ["nirnay"]),
    ("NIRNAY_MMR_NUM_VARS",   "core.config",            ["nirnay"]),
    ("ANALOG_W_MAHA",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("ANALOG_W_TRAJ",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("ANALOG_W_RECV",         "analytics.analogs",      ["analog", "analog_confirm"]),
    ("UI_CONSENSUS_STRONG",   "core.config",            ["markers"]),
    ("UI_CONVRAW_STRONG",     "core.config",            ["markers"]),
    ("UI_NIRNAY_AVG_THRESHOLD", "core.config",          ["markers"]),
]

_NOISE = ("Numba", "numba", "UserWarning", "FutureWarning", "HTTP Error",
          "Failed download", "delisted", "possibly delisted", "YFTz", "unavailable after backfill")


def _list():
    total = sum(e for _, _, _, e in SUITE)
    print(f"\nTattva tuning suite — {len(SUITE)} studies (~{total} min / ~{total/60:.1f}h total)\n")
    print(f"  {'key':<16} {'~min':>5}  what it informs")
    print("  " + "-" * 84)
    for k, s, t, e in SUITE:
        print(f"  {k:<16} {e:>5}  {t}")
    print("\n  Run: python3 research/run_tuning.py [--only k1,k2] [--skip k1,k2]")


def _run_one(key, report):
    script, title, eta = _BY_KEY[key]
    banner = f"\n{'='*90}\n### {key} · {title}\n### script: {script} · ~{eta} min · started {datetime.now():%H:%M:%S}\n{'='*90}\n"
    print(banner, flush=True); report.write(banner)
    t0 = time.time()
    proc = subprocess.Popen([_sys.executable, "-u", str(_HERE / script)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        report.write(line)
        if not any(n in line for n in _NOISE):
            _sys.stdout.write(line); _sys.stdout.flush()
    proc.wait()
    foot = f"\n[{key}] exit {proc.returncode} in {time.time()-t0:.0f}s\n"
    print(foot, flush=True); report.write(foot)
    return proc.returncode


def _config_reference():
    print("\n" + "=" * 90)
    print("  TUNED-CONFIG REFERENCE — live current value · validating study(ies)")
    print("=" * 90)
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
    print(f"  {'constant':<26} {'current':<22} validated by")
    print("  " + "-" * 84)
    for name, mod, keys in CONFIG_REF:
        val = _get(mod, name)
        print(f"  {name:<26} {str(val):<22} {', '.join(keys)}")
    print("\n  Review the report above against these — apply changes by hand (see CHANGELOG /"
          "\n  the rationale comments in core/config.py). The orchestrator never auto-edits config.")


def main():
    ap = argparse.ArgumentParser(description="Tattva intelligent tuning orchestrator")
    ap.add_argument("--list", action="store_true", help="show the suite and exit")
    ap.add_argument("--only", default="", help="comma-separated study keys to run")
    ap.add_argument("--skip", default="", help="comma-separated study keys to skip")
    args = ap.parse_args()

    if args.list:
        _list(); return

    keys = [k for k, *_ in SUITE]
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        keys = [k for k in keys if k in want]
    if args.skip:
        drop = {x.strip() for x in args.skip.split(",")}
        keys = [k for k in keys if k not in drop]
    if not keys:
        print("No studies selected. Use --list to see keys."); return

    eta = sum(_BY_KEY[k][2] for k in keys)
    _REPORTS.mkdir(exist_ok=True)
    rpath = _REPORTS / f"tuning_{datetime.now():%Y%m%d_%H%M%S}.txt"
    print(f"\nRunning {len(keys)} studies (~{eta} min / ~{eta/60:.1f}h): {', '.join(keys)}")
    print(f"Report → {rpath}\n")

    t0 = time.time()
    results = {}
    with open(rpath, "w") as report:
        report.write(f"Tattva tuning run · {datetime.now():%Y-%m-%d %H:%M} · studies={keys}\n")
        for k in keys:
            results[k] = _run_one(k, report)
        _config_reference()

    print("\n" + "=" * 90)
    print(f"  DONE — {len(keys)} studies in {(time.time()-t0)/60:.1f} min")
    for k in keys:
        print(f"    {k:<16} {'OK' if results[k] == 0 else 'FAILED (exit %d)' % results[k]}")
    print(f"  Full report: {rpath}")


if __name__ == "__main__":
    main()
