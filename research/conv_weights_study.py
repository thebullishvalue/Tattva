"""
Tattva — FACTORY DIM-WEIGHT study: are CONV_WEIGHT_* (0.30/0.25/0.25/0.20)
the right RAW composite weights?

The calibration-lift study proved the Optuna-FITTED weights add zero OOS lift
over the factory weights — but the factory values themselves were hand-set and
never swept. This sweeps a structured grid of unfitted weight vectors over the
four agreement dims (direction/breadth/magnitude/regime), scoring each with the
same non-overlapping IC machinery on the live calibration frames. Unfitted
configs compared on the full frame is an honest RANKING (nothing is trained —
the same OFAT logic as every engine-lever study).

Also reports each dim SOLO (1/0/0/0 corners) — the dim-level attribution that
tells you which agreement dimensions actually carry the composite's signal.

Run: python -u research/conv_weights_study.py   (script mode from repo root)
"""
from __future__ import annotations
import warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import os as _os, sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from convergence.intelligence import (
    DEFAULT_WEIGHTS, _score_frame_nonoverlap,
)
from convergence.normalization import COMPOSITE_THRESHOLDS
from calibration_lift_study import _convergence_frame, TARGETS, HOLD

# (label, (w_direction, w_breadth, w_magnitude, w_regime)) — normalized below.
GRID = [
    ("current .30/.25/.25/.20", (0.30, 0.25, 0.25, 0.20)),
    ("equal .25×4",             (0.25, 0.25, 0.25, 0.25)),
    ("direction only",          (1.00, 0.00, 0.00, 0.00)),
    ("breadth only",            (0.00, 1.00, 0.00, 0.00)),
    ("magnitude only",          (0.00, 0.00, 1.00, 0.00)),
    ("regime only",             (0.00, 0.00, 0.00, 1.00)),
    ("drop direction",          (0.00, 0.33, 0.33, 0.34)),
    ("drop breadth",            (0.40, 0.00, 0.30, 0.30)),
    ("drop magnitude",          (0.40, 0.30, 0.00, 0.30)),
    ("drop regime",             (0.375, 0.3125, 0.3125, 0.0)),
    ("direction-heavy .5",      (0.50, 0.20, 0.20, 0.10)),
    ("direction-heavy .4",      (0.40, 0.25, 0.20, 0.15)),
    ("breadth-heavy",           (0.20, 0.45, 0.20, 0.15)),
    ("magnitude-heavy",         (0.20, 0.20, 0.45, 0.15)),
    ("regime-heavy",            (0.20, 0.20, 0.15, 0.45)),
    ("dir+breadth 50/50",       (0.50, 0.50, 0.00, 0.00)),
    ("dir+magnitude 50/50",     (0.50, 0.00, 0.50, 0.00)),
    # 2026-07-13: extra pairwise corners + heavier single-dim tilts + the two
    # remaining drop-pairs, so every face of the 4-simplex is probed.
    ("dir+regime 50/50",        (0.50, 0.00, 0.00, 0.50)),
    ("breadth+magnitude 50/50", (0.00, 0.50, 0.50, 0.00)),
    ("breadth+regime 50/50",    (0.00, 0.50, 0.00, 0.50)),
    ("magnitude+regime 50/50",  (0.00, 0.00, 0.50, 0.50)),
    ("direction-dominant .7",   (0.70, 0.10, 0.10, 0.10)),
    ("breadth-dominant .7",     (0.10, 0.70, 0.10, 0.10)),
    ("magnitude-dominant .7",   (0.10, 0.10, 0.70, 0.10)),
    ("regime-dominant .7",      (0.10, 0.10, 0.10, 0.70)),
    ("drop dir+breadth",        (0.00, 0.00, 0.50, 0.50)),
    ("drop dir+magnitude",      (0.00, 0.50, 0.00, 0.50)),
    ("dir-light .15",           (0.15, 0.30, 0.30, 0.25)),
    ("near-current +dir",       (0.40, 0.20, 0.25, 0.15)),
    ("near-current +regime",    (0.25, 0.25, 0.20, 0.30)),
]


def _w(vec) -> dict[str, float]:
    s = sum(vec) or 1.0
    d, b, m, r = (x / s for x in vec)
    return {"w_direction": d, "w_breadth": b, "w_magnitude": m, "w_regime": r}


def main() -> None:
    from markers_study import _load
    _load()
    print(f"Tattva — factory dim-weight study · {len(TARGETS)} targets · "
          f"hold {HOLD} · thresholds = COMPOSITE factory", flush=True)

    frames = {}
    t0 = time.time()
    for k, tgt in enumerate(TARGETS, 1):
        try:
            f = _convergence_frame(tgt)
        except Exception as e:
            f = None
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
        if f is None or len(f) < 250:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True)
            continue
        frames[tgt] = f
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(f)} frame rows", flush=True)
    if not frames:
        raise SystemExit("no frames")

    print(f"\n  frames in {time.time()-t0:.0f}s")
    print("\n" + "=" * 74)
    print("  FACTORY DIM WEIGHTS — mean non-overlapping IC across targets")
    print("=" * 74)
    print(f"  {'config':<26} {'mean IC':>8} {'IC>0':>6}   per-target")
    print("  " + "-" * 70)
    best = (None, -9)
    PT: dict = {}       # {target: {label: ic}} for the per-instrument reco
    for label, vec in GRID:
        w = _w(vec)
        ics = {}
        for tgt, f in frames.items():
            ic, _ = _score_frame_nonoverlap(f, w, dict(COMPOSITE_THRESHOLDS), HOLD)
            if np.isfinite(ic):
                ics[tgt] = ic
                PT.setdefault(tgt, {})[label] = ic
        if not ics:
            continue
        arr = np.array(list(ics.values()))
        mark = "  ←current" if label.startswith("current") else ""
        detail = " ".join(f"{t.split()[0][:4]}={v:+.2f}" for t, v in ics.items())
        print(f"  {label:<26} {arr.mean():>+8.3f} {int((arr > 0).sum()):>3}/{len(arr):<2}"
              f"   {detail}{mark}")
        if arr.mean() > best[1]:
            best = (label, float(arr.mean()))
    print("  " + "-" * 70)
    print(f"  best: {best[0]} (mean IC {best[1]:+.3f})")
    print("\n  DECISION RULE: adopt a new weight vector only if it beats the current")
    print("  one by a margin that survives the per-target spread (not one target");
    print("  carrying the mean) — otherwise the factory vector stands as validated.")

    # ── PER-INSTRUMENT dim weights (gated: best vector beats this target's OWN
    #    factory IC beyond noise) ──────────────────────────────────────────
    from core.config import InstrumentConfig as _IC
    from research._per_instrument import IC_FLOOR, IC_MARGIN, print_overrides_snippet
    LABEL_VEC = dict(GRID)
    factory_w = _IC().weights_seed()
    print("\n" + "=" * 74)
    print(f"  PER-INSTRUMENT DIM WEIGHTS (best vector vs factory, gate |IC|≥{IC_FLOOR} "
          f"& beat factory by ≥{IC_MARGIN})")
    print("=" * 74)
    overrides: dict = {}
    for tgt, f in frames.items():
        fac_ic, _ = _score_frame_nonoverlap(f, factory_w, dict(COMPOSITE_THRESHOLDS), HOLD)
        cand = [(ic, lbl) for lbl, ic in PT.get(tgt, {}).items() if np.isfinite(ic)]
        if not cand:
            continue
        bic, blbl = max(cand)
        adopt = (bic >= IC_FLOOR and (not np.isfinite(fac_ic) or bic - fac_ic >= IC_MARGIN))
        flag = "  ADOPT" if adopt else "  (noise → factory)"
        print(f"    {tgt:<22} best={blbl:<22} |IC|={bic:+.3f}  factory|IC|={fac_ic:+.3f}{flag}")
        if adopt:
            w = _w(LABEL_VEC[blbl])
            overrides[tgt] = {"conv_weight_direction": round(w["w_direction"], 3),
                              "conv_weight_breadth": round(w["w_breadth"], 3),
                              "conv_weight_magnitude": round(w["w_magnitude"], 3),
                              "conv_weight_regime": round(w["w_regime"], 3)}
    print_overrides_snippet(overrides)


if __name__ == "__main__":
    main()
