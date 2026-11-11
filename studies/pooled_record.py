"""
Tattva — the pooled call record.

Runs Mūla on every catalogue target, grades the call exactly as it is issued
(``convergence.intelligence.call_record``) and pools the outcomes within each
asset class. Writes ``data/priors/call_record.json``, which the verdict reads.

    python studies/pooled_record.py                 # fetch live, write the record
    python studies/pooled_record.py --panel p.pkl   # reuse a pickled panel
    python studies/pooled_record.py --holdout 252   # leave the last 252 sessions unread

POOLING. Each claim is scored in excess of always holding its direction on that
target (``call_outcomes``' ``ex``), divided by the target's own forward-return
SD so that Bitcoin and USD/INR count alike, and the class mean is taken over
all claims. The standard error counts one independent observation per h claims per
target, then divides that count by ``BREADTH_DISCOUNT``: targets in one class
move together (eight Nifty indices are not eight markets), so their claims are
not independent across targets either. Two is a declared conservatism, not an
estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

from convergence.intelligence import call_outcomes, summarise_outcomes   # noqa: E402
from core.config import (CATALOGUE_TARGETS, EXECUTION_LAG, SHEET_TARGETS,  # noqa: E402
                         asset_class, fetch_window)
from core.trust import RECORD_PATH                                        # noqa: E402
from pipeline.prep import PrepError, prepare                              # noqa: E402
from pipeline.run import run_mula                                         # noqa: E402

BREADTH_DISCOUNT = 2.0


def load_panel(path: str | None) -> pd.DataFrame:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    from data.fetcher import fetch_commodity_dataset
    df, err = fetch_commodity_dataset(*fetch_window())
    if df is None:
        raise SystemExit(f"fetch failed: {err}")
    if path:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    return df


def graded_targets() -> list[str]:
    """Catalogue targets that are prices. A valuation ratio is not a tradeable call."""
    return [t for t in CATALOGUE_TARGETS if t not in SHEET_TARGETS]


def fit_fvo(df: pd.DataFrame, target: str, today: pd.Timestamp) -> tuple[pd.Series, pd.Series, int]:
    prep = prepare(df, target)
    eng = run_mula(prep, today=today)
    fvo = pd.Series(eng.ts_data["FVO"].to_numpy(dtype=float), index=prep.calendar)
    return fvo, prep.price, int(prep.config.forecast_horizon)


def _json(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def pool(outcomes: dict[str, pd.DataFrame], horizon: int) -> dict:
    """Pool per-target claim outcomes (``call_outcomes`` frames with an ``sd`` column)."""
    def side(sel) -> dict:
        parts = [o[sel(o)] for o in outcomes.values()]
        parts = [s for s in parts if len(s)]
        if not parts:
            return {"n_claims": 0, "n_independent": 0}
        n_ind = sum(len(s) // horizon for s in parts)
        allc = pd.concat(parts)
        ex = (allc["ex"] / allc["sd"]).to_numpy()
        n_eff = max(n_ind / BREADTH_DISCOUNT, 1.0)
        se = float(np.std(ex, ddof=1) / np.sqrt(n_eff))
        return {"n_claims": int(len(allc)), "n_independent": int(n_ind), "n_effective": float(n_eff),
                "hit": float((allc["y"] > 0).mean() * 100), "base": float(allc["base"].mean() * 100),
                "mean_bps": float(allc["y"].mean() * 1e4), "excess_bps": float(allc["ex"].mean() * 1e4),
                "excess_sd": float(np.mean(ex)), "se_sd": se,
                "t": float(np.mean(ex) / se) if se > 0 else float("nan")}
    rec = side(lambda o: o["direction"] != 0)
    rec["long"] = side(lambda o: o["direction"] > 0)
    rec["short"] = side(lambda o: o["direction"] < 0)
    rec["targets"] = sorted(outcomes)
    return rec


def build(df: pd.DataFrame, holdout: int = 0, log=print) -> dict:
    today = pd.Timestamp(df["DATE"].max())
    per_target, by_class, horizon = {}, {}, None
    for t in graded_targets():
        try:
            fvo, price, h = fit_fvo(df, t, today)
        except PrepError as e:
            log(f"{t}: skipped — {e.title}")
            continue
        horizon = h
        if holdout:
            fvo, price = fvo.iloc[:-holdout], price.iloc[:-holdout]
        out = call_outcomes(fvo, price, h)
        lp = np.log(price)
        sd = float((lp.shift(-(EXECUTION_LAG + h)) - lp.shift(-EXECUTION_LAG)).std())
        out["sd"] = sd
        rec = summarise_outcomes(out, h)
        per_target[t] = rec
        by_class.setdefault(asset_class(t), {})[t] = out
        log(f"{t:18s} claims {rec['n_claims']:5d}  hit {rec['hit']:5.1f}%  base {rec['base']:5.1f}%  "
            f"excess {rec['excess_bps']:+6.0f} bps  t {rec['t']:+.2f}")
    classes = {c: pool(o, horizon) for c, o in by_class.items()}
    everything = {t: o for cls in by_class.values() for t, o in cls.items()}
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "panel_fingerprint": (df.attrs or {}).get("panel_fingerprint"),
        "panel_through": today.strftime("%Y-%m-%d"),
        "horizon": horizon, "lag": EXECUTION_LAG, "holdout_sessions": int(holdout),
        "breadth_discount": BREADTH_DISCOUNT,
        "all": pool(everything, horizon), "classes": classes, "targets": per_target,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", help="pickled panel to reuse (written if absent)")
    ap.add_argument("--holdout", type=int, default=0, help="sessions at the end left unread")
    ap.add_argument("--out", default=str(RECORD_PATH))
    a = ap.parse_args()
    from core.logger_config import console
    console.enabled = False
    rec = build(load_panel(a.panel), a.holdout)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(_clean(json.loads(json.dumps(rec, default=_json))), f, indent=1)
    print(f"\nwrote {a.out}")
    for c, r in {"ALL": rec["all"], **rec["classes"]}.items():
        print(f"{c:12s} claims {r['n_claims']:6d} hit {r['hit']:5.1f}% base {r['base']:5.1f}% excess {r['excess_bps']:+5.0f}bps t {r['t']:+.2f}"
              f" | long hit {r['long'].get('hit', float('nan')):5.1f}% base {r['long'].get('base', float('nan')):5.1f}% t {r['long'].get('t', float('nan')):+.2f}"
              f" | short hit {r['short'].get('hit', float('nan')):5.1f}% base {r['short'].get('base', float('nan')):5.1f}% t {r['short'].get('t', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
