"""
Tattva — study: is the grade's standard error calibrated?
तत्त्व (Tattva) — "Principle / Essence"

    python -m studies.se_calibration                          # synthetic cases only
    python -m studies.se_calibration STATE.pkl [...] --check  # plus the pickles' 4 cases each

Every grade's SE (``core.grade.series_edge``'s ``se``) comes from
:func:`core.grade.shift_null_se`. Its main part is a closed form: the spread,
over every admissible circular shift, of the FULL-SAMPLE rank correlation of
the rotated pairs, by FFT, while the IC it is the SE of is the OFFSET-AVERAGED
one. A signed grade (a sizer against d·Y) adds an analytic side-mean part,
which no rotation can see. This study checks, per case:

  * ``fft/brute`` — the closed-form rotation part against ``brute``, the SD of
    the offset-averaged IC itself over ``--shifts`` random admissible shifts
    (``[5h, N − 5h]``), rotated exactly as the grade's null rotates them (the
    design's R4 check);
  * ``se`` — the SE the grade reports, with its ``side_mean`` part, and
    ``legacy/se`` — 2.8.0's ``1/√(n_win − 3)`` against it.

The synthetic section measures the TRUE sampling SD of the IC over
``--seeds`` independent no-skill draws (``mc_sd``), which no null can see from
one history, on a +20%/yr trend graded drift-neutral: unsigned AR(1) leans, an
MRProb-like lean, and signed sizers on a claim whose side flips about every 13
rows (as on the audited instruments) — one carrying the side (d·z), one
ignoring it, one leaning to BUY, and one that only says the side. ``plain`` is
the rotation of the signed pair as is, without the grade's side handling. (A
side-only factor's SE is all side-mean part; its rotation part, and so its
``fft/brute``, is a ratio of two near-zero numbers.)

A pickle is a pickled session state or bundle holding ``fvo_ts`` (FVO, Price,
MRProb) and ``swayam_daily`` (Oversold_Pct, Overbought_Pct, _Native). Its four
cases are the design's: ``claim`` (d on claim rows against drift-neutral Y),
``sizer:swayam`` (d·Lean on native claim rows, sign d), ``sizer:reversion``
(MRProb on claim rows, sign d) and ``lean`` (valuation_lean(FVO) on valued
rows against drift-neutral Y). ``--check`` exits non-zero when any pickle
case's ``fft/brute`` is outside [0.9, 1.1].
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.signal import lfilter  # noqa: E402

from core.grade import (GRADE_SHIFT_MARGIN_H, _datetime_index, _graded_frame,  # noqa: E402
                        _null_parts, _offset_averaged_ic, _rank, shift_null_se)
from core.knobs import knob  # noqa: E402
from core.semantics import breadth_lean, claim_direction, valuation_lean  # noqa: E402

CHECK_BAND = (0.9, 1.1)


def brute_rotation_sd(x: np.ndarray, y: np.ndarray, s: np.ndarray | None, h: int,
                      n_shifts: int, seed: int = 0) -> float:
    """SD of the offset-averaged IC over random admissible circular shifts.

    The rotation is the grade's: the centred rank series of the lean, times
    each row's side (``s``; none for an unsigned grade), is rotated and
    re-signed by the side of the row it lands on, so every rotated product
    keeps the side of its own row. Only the statistic differs from the closed
    form — the offset-averaged IC here, the full-sample rank correlation there.
    """
    rng = np.random.default_rng(seed)
    n, margin = len(x), GRADE_SHIFT_MARGIN_H * h
    side = np.ones(n) if s is None else s
    a = side * (_rank(x) - (n + 1) / 2)
    a = a - a.mean()
    ics = [_offset_averaged_ic(side * np.roll(a, int(k)), y, h)
           for k in rng.integers(margin, n - margin + 1, n_shifts)]
    return float(np.nanstd(ics))


def _case_row(case: str, df: pd.DataFrame, signed: bool, h: int, n_shifts: int) -> dict:
    x, y = df["x"].to_numpy(dtype=np.float64), df["y"].to_numpy(dtype=np.float64)
    s = df["s"].to_numpy(dtype=np.float64) if signed else None
    rotation, side_mean = _null_parts(x, y, h, s)
    se = shift_null_se(x, y, h, sign=s)
    n_win = len(df) // h
    return {"case": case, "n_win": n_win, "ic": _offset_averaged_ic(x, y, h), "se": se,
            "side_mean": side_mean,
            "fft/brute": rotation / brute_rotation_sd(x, y, s, h, n_shifts),
            "legacy/se": (1 / np.sqrt(max(n_win - 3, 1))) / se}


# ════════════════════════════════════════════════════════════════════════
# The pickles
# ════════════════════════════════════════════════════════════════════════


def pickle_cases(path: Path, h: int, lag: int, n_shifts: int) -> list[dict]:
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    f = _datetime_index(state["fvo_ts"])
    price = _datetime_index(pd.to_numeric(f["Price"], errors="coerce"))
    fvo = pd.to_numeric(f["FVO"], errors="coerce")
    d = fvo.map(claim_direction).where(fvo.notna())
    claim = d.fillna(0) != 0
    sw = _datetime_index(state["swayam_daily"])
    sw = sw[sw["_Native"].astype(bool)]
    lean_sw = pd.Series([breadth_lean(a, b) for a, b in zip(sw["Oversold_Pct"], sw["Overbought_Pct"])],
                        index=sw.index).reindex(f.index)
    cases = {
        "claim": (d.where(claim), None),
        "sizer:swayam": ((d * lean_sw).where(claim), d),
        "sizer:reversion": (pd.to_numeric(f["MRProb"], errors="coerce").where(claim), d),
        "lean": (fvo.map(valuation_lean).where(fvo.notna()), None),
    }
    rows = []
    for case, (lean, sign) in cases.items():
        df = _graded_frame(lean, price, h, mask=None, sign=sign, drift="causal", exec_lag=lag)
        rows.append({"target": path.stem, **_case_row(case, df, sign is not None, h, n_shifts)})
    return rows


# ════════════════════════════════════════════════════════════════════════
# Synthetic no-skill cases
# ════════════════════════════════════════════════════════════════════════


def _ar1(rng, n: int, phi: float) -> np.ndarray:
    return lfilter([np.sqrt(1 - phi * phi)], [1, -phi], rng.normal(size=n))


def _mrprob(rng, n: int) -> np.ndarray:
    """Near-constant and persistent, like the ECM's MRProb: a logistic AR(1) in [0.4, 0.8]."""
    return 0.4 + 0.4 / (1 + np.exp(-2 * _ar1(rng, n, 0.995)))


def _claim_side(rng, n: int) -> np.ndarray:
    """A no-skill claim side that flips about every 13 rows (the audited four: 14)."""
    return np.where(np.cumsum(rng.random(n) < 1 / 13) % 2 == 0, 1.0, -1.0)


def _unsigned(make):
    return lambda rng, n: (make(rng, n), None)


def _signed(make):
    def draw(rng, n):
        d = _claim_side(rng, n)
        return make(rng, n, d), d
    return draw


SYNTHETIC = {
    **{f"AR(1) φ={phi}": _unsigned(lambda rng, n, phi=phi: _ar1(rng, n, phi)) for phi in (0.0, 0.9, 0.98)},
    "MRProb-like": _unsigned(_mrprob),
    "sizer d·z": _signed(lambda rng, n, d: d * np.clip(_ar1(rng, n, 0.9) / 3, -1, 1)),
    "sizer MRProb": _signed(lambda rng, n, d: _mrprob(rng, n)),
    "sizer MRProb + 0.05·BUY": _signed(lambda rng, n, d: _mrprob(rng, n) + 0.05 * (d > 0)),
    "side only: 0.05·BUY": _signed(lambda rng, n, d: 0.5 + 0.05 * (d > 0) + rng.normal(0, 1e-6, n)),
}


def synthetic(h: int, n: int, seeds: int, n_shifts: int, lag: int) -> pd.DataFrame:
    """Per case: fft/brute on the first 10 seeds, and the SEs against the true SD over all."""
    idx = pd.bdate_range("2017-01-02", periods=n)
    out = []
    for case, draw in SYNTHETIC.items():
        ics, ses, plains, ratios = [], [], [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            price = pd.Series(100 * np.exp(np.cumsum(rng.normal(np.log(1.2) / 252, 0.01, n))), index=idx)
            lean, sign = draw(rng, n)
            df = _graded_frame(pd.Series(lean, index=idx), price, h, mask=None,
                               sign=None if sign is None else pd.Series(sign, index=idx),
                               drift="causal", exec_lag=lag)
            x, y = df["x"].to_numpy(dtype=np.float64), df["y"].to_numpy(dtype=np.float64)
            s = None if sign is None else df["s"].to_numpy(dtype=np.float64)
            ics.append(_offset_averaged_ic(x, y, h))
            ses.append(shift_null_se(x, y, h, sign=s))
            plains.append(shift_null_se(x, y, h))
            if seed < 10:
                ratios.append(_null_parts(x, y, h, s)[0] / brute_rotation_sd(x, y, s, h, n_shifts, seed))
        mc_sd = float(np.std(ics, ddof=1))
        out.append({"case": case, "fft/brute (median)": float(np.median(ratios)), "mc_sd": mc_sd,
                    "se/mc_sd": float(np.mean(ses)) / mc_sd, "plain/mc_sd": float(np.mean(plains)) / mc_sd})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pickles", nargs="*", type=Path, help="pickled session states or bundles")
    ap.add_argument("--h", type=int, default=10, help="forecast horizon (default 10)")
    ap.add_argument("--lag", type=int, default=knob("EXECUTION_LAG"), help="execution lag L")
    ap.add_argument("--shifts", type=int, default=200, help="brute-force shifts per case")
    ap.add_argument("--seeds", type=int, default=200, help="synthetic draws per case")
    ap.add_argument("--n", type=int, default=1500, help="synthetic history length")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if a pickle case's fft/brute is outside {CHECK_BAND}")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 160)

    if not args.no_synthetic:
        print(f"Synthetic no-skill cases (h={args.h}, L={args.lag}, N={args.n}, {args.seeds} seeds, "
              f"{args.shifts} shifts on the first 10; +20%/yr trend, drift-neutral target)")
        print(synthetic(args.h, args.n, args.seeds, args.shifts, args.lag).round(3).to_string(index=False))
    if not args.pickles:
        return 0
    table = pd.DataFrame([r for p in args.pickles
                          for r in pickle_cases(p, args.h, args.lag, args.shifts)])
    print(f"\nPickles (h={args.h}, L={args.lag}, {args.shifts} shifts, drift-neutral target)")
    print(table.round(3).to_string(index=False))
    ratio = table["fft/brute"]
    print(f"fft/brute {ratio.min():.3f}–{ratio.max():.3f} (median {ratio.median():.3f}) over "
          f"{len(table)} cases; legacy/se {table['legacy/se'].min():.2f}–{table['legacy/se'].max():.2f}")
    if args.check and not ratio.between(*CHECK_BAND).all():
        print(f"FAIL: fft/brute outside {CHECK_BAND}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
