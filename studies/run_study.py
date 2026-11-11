"""
Tattva — run the pre-registered candidate study (studies/PREREGISTRATION.md).

    python studies/run_study.py --panel p.pkl            # development sample
    python studies/run_study.py --panel p.pkl --final C3 # ONE holdout read for a cleared candidate

The development run never reads the final HOLDOUT published sessions of any
target. Results go to studies/results/<date>.json and .md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timezone

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "studies"))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

from candidates import CANDIDATES, FAST_DELTAS, cross_sectional_claims   # noqa: E402
from convergence.intelligence import claim_outcomes                       # noqa: E402
from core.config import EXECUTION_LAG, asset_class                        # noqa: E402
from core.logger_config import console                                    # noqa: E402
from pipeline.prep import PrepError, prepare                              # noqa: E402
from pipeline.run import run_mula                                         # noqa: E402
from pooled_record import _clean, _json, graded_targets, load_panel, pool  # noqa: E402

HOLDOUT = 252
BAR_T = 2.0
N_BLOCKS = 6


def fits(df: pd.DataFrame, cache: str | None) -> dict:
    """``{fit: {target: frame(FVO, GapPercentile, Price)}}`` for the base and fast fits."""
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    today = pd.Timestamp(df["DATE"].max())
    out: dict = {"base": {}, "fast": {}}
    for t in graded_targets():
        try:
            prep = prepare(df, t)
        except PrepError as e:
            print(f"{t}: skipped — {e.title}")
            continue
        for name, p in (("base", prep),
                        ("fast", replace(prep, config=replace(prep.config,
                                                              fvo_valuation_deltas=FAST_DELTAS)))):
            eng = run_mula(p, today=today)
            ts = eng.ts_data[["FVO", "GapPercentile", "Price"]].copy()
            ts.index = prep.calendar
            out[name][t] = ts
        print(f"fitted {t}", flush=True)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


def split(ts: pd.DataFrame, final: bool) -> tuple[pd.DataFrame, pd.Series]:
    """``(frame the rule reads, mask of sessions that may be graded)``.

    Development: the frame simply ends before the holdout. Final: the rule
    reads everything (it is causal) and only holdout sessions are graded.
    """
    pub = np.flatnonzero(ts["FVO"].notna().to_numpy())
    if len(pub) <= HOLDOUT:
        return ts.iloc[:0], pd.Series(dtype=bool)
    cut = pub[-HOLDOUT]
    if not final:
        dev = ts.iloc[:cut]
        return dev, pd.Series(True, index=dev.index)
    mask = pd.Series(False, index=ts.index)
    mask.iloc[cut:] = True
    return ts, mask


def evaluate(cand, all_fits: dict, final: bool) -> dict:
    frames = {t: split(ts, final) for t, ts in all_fits[cand.fit].items()}
    frames = {t: v for t, v in frames.items() if len(v[0])}
    if cand.cross_sectional:
        claims: dict = {}
        by_cls: dict = {}
        for t, (ts, _m) in frames.items():
            by_cls.setdefault(asset_class(t), {})[t] = ts["FVO"]
        for group in by_cls.values():
            claims.update(cross_sectional_claims(group))
    else:
        claims = {t: cand.rule(ts) for t, (ts, _m) in frames.items()}

    outcomes, by_class = {}, {}
    for t, (ts, mask) in frames.items():
        d = claims[t].reindex(ts.index).fillna(0.0)
        pub = ts["FVO"].notna()
        o = claim_outcomes(d, pub, ts["Price"], cand.horizon)
        # The benchmark (base, mean) is taken over every published session of
        # the frame; only the masked sessions are counted as claims.
        o = o[mask.reindex(o.index, fill_value=False).to_numpy()]
        lp = np.log(ts["Price"])
        o["sd"] = float((lp.shift(-(EXECUTION_LAG + cand.horizon)) - lp.shift(-EXECUTION_LAG)).std())
        outcomes[t] = o
        by_class.setdefault(asset_class(t), {})[t] = o
    rec = pool(outcomes, cand.horizon)
    rec.pop("targets", None)

    allc = pd.concat([o for o in outcomes.values() if len(o)]).sort_index() if outcomes else pd.DataFrame()
    blocks = []
    if len(allc):
        for part in np.array_split(np.arange(len(allc)), N_BLOCKS):
            if len(part):
                b = allc.iloc[part]
                blocks.append({"start": b.index[0], "end": b.index[-1], "n": int(len(b)),
                               "excess_sd": float((b["ex"] / b["sd"]).mean())})
    rec["blocks"] = blocks
    rec["positive_blocks"] = int(sum(1 for b in blocks if b["excess_sd"] > 0))
    rec["classes"] = {c: {k: v for k, v in pool(o, cand.horizon).items()
                          if k in ("n_claims", "hit", "base", "excess_bps", "t")}
                      for c, o in by_class.items()}
    hit_ok = rec.get("hit", 0) > rec.get("base", 100)
    if final:
        rec["passes"] = bool(rec.get("excess_sd", -1) > 0 and hit_ok)
    else:
        rec["passes"] = bool(rec.get("t", -9) >= BAR_T and hit_ok and rec["positive_blocks"] >= 4)
    return rec


def table(results: dict, final: bool) -> str:
    head = ("| ID | Reading | h | Claims | Hit | Base | Excess (bps) | t | Blocks + | "
            + ("Holdout pass" if final else "Clears bar") + " |\n|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for cid, r in results.items():
        rows.append(f"| {cid} | {r['name']} | {r['horizon']} | {r.get('n_claims', 0):,} | "
                    f"{r.get('hit', float('nan')):.1f}% | {r.get('base', float('nan')):.1f}% | "
                    f"{r.get('excess_bps', float('nan')):+.0f} | {r.get('t', float('nan')):+.2f} | "
                    f"{r['positive_blocks']}/{len(r['blocks'])} | {'**yes**' if r['passes'] else 'no'} |")
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--fits", help="pickle cache of the Mūla fits")
    ap.add_argument("--final", help="candidate id: read the holdout once for it")
    a = ap.parse_args()
    console.enabled = False
    df = load_panel(a.panel)
    all_fits = fits(df, a.fits)
    cands = [c for c in CANDIDATES if not a.final or c.id == a.final]
    if a.final and not cands:
        raise SystemExit(f"unknown candidate {a.final}")
    results = {}
    for c in cands:
        r = evaluate(c, all_fits, bool(a.final))
        results[c.id] = {"name": c.name, "horizon": c.horizon, **r}
        print(f"{c.id} {c.name:24s} claims {r.get('n_claims', 0):6d} hit {r.get('hit', float('nan')):5.1f}% "
              f"base {r.get('base', float('nan')):5.1f}% excess {r.get('excess_bps', float('nan')):+5.0f}bps "
              f"t {r.get('t', float('nan')):+.2f} blocks {r['positive_blocks']}/{len(r['blocks'])} "
              f"{'PASS' if r['passes'] else '-'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    kind = f"final_{a.final}" if a.final else "development"
    out = {"generated": stamp, "sample": kind, "holdout_sessions": HOLDOUT, "bar_t": BAR_T,
           "lag": EXECUTION_LAG, "panel_fingerprint": (df.attrs or {}).get("panel_fingerprint"),
           "panel_through": str(pd.Timestamp(df["DATE"].max()).date()), "candidates": results}
    d = os.path.join(ROOT, "studies", "results")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stamp}_{kind}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(_clean(json.loads(json.dumps(out, default=_json))), f, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"# Candidate study — {kind} sample, {stamp}\n\n"
                f"Panel through {out['panel_through']} (fingerprint {out['panel_fingerprint']}). "
                f"Final {HOLDOUT} published sessions of every target "
                f"{'are the only ones graded' if a.final else 'were not read'}. "
                f"Entered at the next close; excess is over always holding the direction.\n\n"
                + table(results, bool(a.final)))
    print(f"\nwrote {base}.json / .md")


if __name__ == "__main__":
    main()
