"""
TATTVA — Samanvaya, the value engine inside pragati.pine (section 4c)

Ported verbatim from Sanket's samanvaya.py. In Tattva it is the value half of Swayam's
self-read of the target (its RV leg against ~20 macro drivers plus a price-only breadth leg);
Mūla's 200-instrument cointegrating fair value stays the system's other, independent value read.
══════════════════════════════════════════════════════════════════════════════

Carried from Pragyam's port (pragyam/samanvaya.py) and extended for a screener.
Pragati's VALUE ingredient: where price stands, rich (+) or cheap (−), against
what the macro drivers explain. pragati.py blends it with conviction into the
trace; its ladder mean is the MTF value tape; cvgrid.py reads that tape as the
grid's column.

The engine, as the Pine runs it on its defaults
───────────────────────────────────────────────
RV LEG     The name's per-bar log return regressed on up to 3 macro factors,
           chosen by stepwise partial correlation over a 250-period window read
           12 periods in arrears (so the fit applied to any bar ended 12
           periods earlier), admitted past a Fisher / Šidák floor, solved by
           ridge Gram-Schmidt. The hedge is applied only in proportion to its
           own out-of-sample skill over the last 104 periods — "the hedge weighs
           itself". The running sum of the hedged residual is the spread; its z
           over five timescales (8 … 55), each high-passed by a 400-bar EMA,
           averaged and rescaled by 0.887, is the RV z.
BREADTH    Seven views of the name's own price through the Market Strength
           Factor (momentum, structure, flow) at five timescales.
BLEND      0.5 / 0.5 in z-space, variance restored with the measured leg
           correlation, soft-bounded once: the chart's value reading.
TAPE       Ladder up. The parent rung is the whole composite on the frame
           above the chart — the RV ensemble on the spread sampled at the
           parent's closes, finished with the current spread as the forming
           parent bar, and breadth at the parent's last CLOSED bar. The tape is
           the ladder mean of z, NOT variance-restored: frames that disagree
           pull it toward zero.

The two charts Sanket runs
──────────────────────────
    chart   anchor (model clock)     value ladder     driver lateness measured in
    D       the daily bar itself     W · D            a 24-hour bar
    W       the weekly bar itself    M · W            a 168-hour bar

On a weekly chart the Pine's daily model clock changes on every weekly bar, so
each weekly bar is its own anchor period and the model runs on weekly returns
— exactly what this does. The M rung's breadth needs 100 closed months, which
no fetch here carries, so on Weekly the M rung is its RV leg alone — the
Pine's own f_tpBlend rule for a rung whose breadth is not yet calibrated.

What changed from Pragyam's port, and why
─────────────────────────────────────────
· SELECTION AND REGRESSION ARE VECTORISED ACROSS TIME. Only the three
  hysteresis paths — which are genuinely path-dependent — run as scalar loops.
  Pragyam ran every bar's partial correlations in a Python loop over ~30 ETFs;
  Sanket screens up to ~500 names and measures ~15 years in its edge study, so
  the same loop would dominate the run. Same numbers.
· THE BASKET'S ROTATION IS KEPT. Pragyam dropped the basket-warm gate because
  it reads no signal; Sanket fires the Pine's ▲▼ TURN, which may not open a
  window while the basket settles (21 bars after a rotation), so the full
  selection path — all three indices and the model order — is returned.
· A WEEKLY CHART. The parent key, the driver resampling and the lateness span
  follow the chart.
· WEEKS ARE MONDAY-TO-SUNDAY (W-SUN), the grouping Sanket's weekly resample
  uses, so crypto's weekend bars fall in the week they trade in.

The macro basket — what yfinance can carry (unchanged from Pragyam)
───────────────────────────────────────────────────────────────────
yfinance carries the US yields, the INR crosses, the dollar index and the
commodity futures directly. Every other 10-year yield is proxied by that
market's government-bond ETF, converted to a yield move with its approximate
modified duration (Δy ≈ −100 · Δln(P) / D). Non-US 2-year yields have no proxy
and drop out of their pools, as the Pine's pooling allows. The EXPANDED basket
(default) adds Brent, copper and the name's home equity index (Nifty / S&P
500), so value means rich or cheap AFTER what the market and the macro explain
— Pragyam's pre-registered test kept it because it raised median out-of-sample
hedge skill in every universe it ran on (ETF 0.01 → 0.45, Nifty 50 0.005 →
0.23, Dow 30 0.01 → 0.20).

DRIVER TIMING, as the Pine: a driver whose bar closes more than a third of a
chart bar after the name's is read at its PREVIOUS close. US drivers against an
NSE name are 10½ hours late — lagged on Daily, kept on Weekly (a sixteenth of a
week).

Author: @thebullishvalue
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _swv

# ── Study-fixed constants (pragati.pine section 4c, each swept across 24 targets) ─
CORR_LEN = 250          # estimation sample, anchor bars
HOLD_OUT = 12           # selection hold-out, anchor bars
HYST_MARG = 0.10        # selection hysteresis
RIDGE_LAM = 0.05        # ridge shrinkage (scaled by 200/n below 200 obs)
MIN_RHO = 0.0           # economic floor; Fisher does the work
MAX_VARS = 3            # maximum model order
MIN_OBS = 30            # enoughData — the basket is tradable past this
SKILL_WIN = 104         # anchor bars over which hedge skill is measured
SKILL_MIN = 20          # skill needs this many periods before it is trusted
MODEL_MIN = 10          # the model is "built" at this many observations
GAIN = 0.5              # display scaling; monotone
DRIFT_LEN = 400         # high-pass EMA on each ensemble member
Z_WINDOWS = (8, 13, 21, 34, 55)
ENS_SCALE = 0.887       # sd of the mean of five members at measured rho 0.734
B_WINDOWS = (10, 14, 20, 28, 40)
ROC_FRAC = 0.70
REGIME_SENS = 8.0
BASE_WEIGHT = 0.0
CLIP_Z = 3.0
VAR_CORR_LEN = 200      # window for the leg correlation
LEG_MIX = 0.5           # weight on the RV leg — Samanvaya's measured best
THETA = 1.5             # entry threshold θ, in |z|
DRIVER_LATE = 1.0 / 3.0
SELF_RHO = 0.99         # self-containment screen
SELF_WIN = 200
BASKET_SETTLE = Z_WINDOWS[2]                    # 21 bars after a rotation
PARENT_BREADTH_MIN = 2 * B_WINDOWS[-1] + 20     # parent bars before its breadth is trusted

# The two charts Sanket runs: the bar's length in hours (driver lateness is
# measured in chart bars) and the parent frame the value ladder climbs to.
CHARTS = {"D": {"bar_hours": 24.0, "parent": "W"},
          "W": {"bar_hours": 168.0, "parent": "M"}}


def softbound(x):
    """The Pine's f_softBound: x / (1 + |x|)."""
    return x / (1.0 + np.abs(x))


# θ on the ±100 scale: the knee where value becomes RICH or CHEAP, the trace's
# signal rail, and the height of a 3σ histogram column.
THETA_OSC = float(100.0 * softbound(THETA * GAIN))          # 42.857…

# ── Drivers: yfinance ticker, unit, modified duration (proxies), close (UTC h) ──
# `yield` is in percentage points; `price` becomes a log; `bond` is a bond-ETF
# PRICE converted to a yield move with its approximate modified duration.
DRIVERS: Dict[str, Tuple[str, str, Optional[float], float]] = {
    "US10Y":  ("^TNX",          "yield", None, 20.5),
    "US30Y":  ("^TYX",          "yield", None, 20.5),
    "US02Y":  ("ZT=F",          "bond",  1.9,  21.0),   # 2-year T-note futures
    "IN10Y":  ("SETF10GILT.NS", "bond",  6.8,  10.0),   # 10-year gilt ETF
    "JP10Y":  ("1482.T",        "bond",  9.5,  6.0),    # Japan govt bond ETF
    "CN10Y":  ("CBON",          "bond",  5.0,  20.5),   # China bond ETF (USD-listed)
    "EU10Y":  ("EXX6.DE",       "bond",  6.0,  15.5),   # German govt bond ETF
    "GB10Y":  ("IGLT.L",        "bond",  11.0, 15.5),   # UK gilt ETF
    "DXY":    ("DX-Y.NYB",      "price", None, 21.0),
    "USOIL":  ("CL=F",          "price", None, 21.0),
    "BRENT":  ("BZ=F",          "price", None, 21.0),
    "GOLD":   ("GC=F",          "price", None, 21.0),
    "SILVER": ("SI=F",          "price", None, 21.0),
    "COPPER": ("HG=F",          "price", None, 21.0),
    "USDINR": ("USDINR=X",      "price", None, 22.0),
    "EURINR": ("EURINR=X",      "price", None, 22.0),
    "GBPINR": ("GBPINR=X",      "price", None, 22.0),
    "JPYINR": ("JPYINR=X",      "price", None, 22.0),
    "NIFTY":  ("^NSEI",         "price", None, 10.0),
    "SPX":    ("^GSPC",         "price", None, 20.5),
}
DRIVER_TICKERS: List[str] = sorted({v[0] for v in DRIVERS.values()})

# Factor definitions: (name, kind, spec). "pool" averages its constituents'
# returns na-safely; "sub" is the difference of two; "pool_sub" pools
# differences; "home" is the name's home equity index.
_SAMANVAYA = [
    ("Global 10Y",   "pool",     ["US10Y", "IN10Y", "JP10Y", "CN10Y", "EU10Y", "GB10Y"]),
    ("Global curve", "pool_sub", [("US10Y", "US02Y")]),
    ("US rates",     "pool",     ["US02Y", "US10Y", "US30Y"]),
    ("US 30s2s",     "sub",      ("US30Y", "US02Y")),
    ("Dollar",       "pool",     ["DXY"]),
    ("Energy",       "pool",     ["USOIL"]),
    ("Precious",     "pool",     ["GOLD", "SILVER"]),
    ("INR basket",   "pool",     ["USDINR", "EURINR", "GBPINR", "JPYINR"]),
]
_EXPANDED = [
    f if f[0] != "Energy" else ("Energy", "pool", ["USOIL", "BRENT"]) for f in _SAMANVAYA
] + [
    ("Industrial metals", "pool", ["COPPER"]),
    ("Home market",       "home", None),
]
BASKETS = {"samanvaya": _SAMANVAYA, "expanded": _EXPANDED}
DEFAULT_BASKET = "expanded"

VALUE_COLUMNS = (
    "value_z", "value", "rv_z", "breadth_z", "legs_split",
    "model_built", "enough", "n_obs", "hedge", "drivers",
    "bars_since_rot", "basket_warm",
    "parent_z", "tape", "tape_ready", "tape_rungs",
)


def _close_utc(symbol: str) -> float:
    """Approximate daily close of the name's own market, in UTC hours."""
    s = str(symbol).upper()
    if s.endswith(".NS") or s.endswith(".BO") or s in ("^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX") \
            or s.startswith("^CNX") or s.startswith("^NSE"):
        return 10.0
    if s.endswith("=X"):
        return 22.0
    if s.endswith("-USD"):
        return 24.0
    if s.endswith("=F"):
        return 21.0
    return 20.5


def _home_driver(symbol: str) -> str:
    return "NIFTY" if _close_utc(symbol) == 10.0 else "SPX"


def _sidak_z(k: int) -> float:
    """Two-sided critical z at 5% family-wise over the k candidates offered.

    The Pine hardcodes 2.7655 for its nine; the basket here offers ten, so it is
    computed. The standard library's normal quantile — no scipy needed.
    """
    from statistics import NormalDist
    a = 1.0 - (1.0 - 0.05) ** (1.0 / max(k, 1))
    return float(NormalDist().inv_cdf(1.0 - a / 2.0))


def period_key(index: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    """The parent bar each chart bar falls in. W = Monday-to-Sunday weeks."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.to_period("W-SUN" if freq == "W" else freq)


def _weekly_last(s: pd.Series) -> pd.Series:
    """A daily series on Sanket's weekly calendar: each Monday-labelled week's last value.

    Mirrors sanket.resample_to_weekly (W-MON, closed/label left), so a driver
    lands on exactly the index a weekly chart carries.
    """
    return s.resample("W-MON", closed="left", label="left").last().dropna()


def prepare_drivers(closes: Optional[pd.DataFrame], chart: str = "D") -> Optional[pd.DataFrame]:
    """The driver closes on the chart's calendar, prepared ONCE per universe.

    On a weekly chart every name would otherwise re-resample all twenty drivers.
    The result is tagged, so driver_returns does not resample it twice.
    """
    if closes is None or closes.empty:
        return closes
    c = closes.copy()
    c.index = pd.DatetimeIndex(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.DatetimeIndex(c.index)
    if chart == "W":
        c = c.apply(lambda s: _weekly_last(s.dropna()))
    c.attrs["chart"] = chart
    return c


# ══════════════════════════════════════════════════════════════════════════════
#  1-3 · DATA, NATURAL UNITS, FACTORS
# ══════════════════════════════════════════════════════════════════════════════
def driver_returns(closes: Optional[pd.DataFrame], index: pd.DatetimeIndex,
                   symbol: str, chart: str = "D") -> Dict[str, pd.Series]:
    """Each driver's return on the name's own calendar, in natural units.

    Aligned by date with the Pine's DRIVER TIMING rule, then differenced once.
    Yields move in percentage points, prices in log returns, bond-ETF proxies
    in yield points via their duration.
    """
    out: Dict[str, pd.Series] = {}
    if closes is None or closes.empty:
        return out
    bar_h = CHARTS[chart]["bar_hours"]
    tgt_close = _close_utc(symbol)
    for key, (tk, kind, dur, close_utc) in DRIVERS.items():
        if tk not in closes.columns:
            continue
        s = pd.to_numeric(closes[tk], errors="coerce").dropna()
        if s.empty:
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) else s.index
        if kind != "yield":
            s = s[s > 0]
        if chart == "W" and closes.attrs.get("chart") != "W":
            s = _weekly_last(s)
        # Late by more than a third of a chart bar: read at the PREVIOUS close,
        # on the driver's own calendar — close[1] of the bar matched to this date.
        if max(0.0, close_utc - tgt_close) / bar_h > DRIVER_LATE:
            s = s.shift(1).dropna()
        s = s.reindex(s.index.union(index)).ffill().reindex(index)
        lvl = s if kind == "yield" else np.log(s)
        v = lvl.diff()
        if kind == "bond":
            v = -100.0 * v / float(dur)
        out[key] = v
    return out


def _self_screen(y: pd.Series, v: pd.Series, symbol: str, driver_ticker: str) -> pd.Series:
    """The Pine's SELF-CONTAINMENT screen on one constituent, as a latch.

    True from the first bar the constituent is identified as the target itself —
    by name, or by |ρ| > 0.99 of returns over 200 bars — and forever after.
    """
    if str(symbol).upper() == str(driver_ticker).upper():
        return pd.Series(True, index=v.index)
    c = y.rolling(SELF_WIN, min_periods=SELF_WIN).corr(v)
    return (c.abs() > SELF_RHO).cummax().fillna(False).astype(bool)


def build_factors(y: pd.Series, drivers: Dict[str, pd.Series], symbol: str,
                  basket: str = DEFAULT_BASKET) -> Tuple[pd.DataFrame, List[str]]:
    """Factor returns for one name: the basket's pools, after the self-screen."""
    spec = BASKETS[basket]
    kept: Dict[str, pd.Series] = {}
    for key, v in drivers.items():
        tk = DRIVERS[key][0]
        kept[key] = v.where(~_self_screen(y, v, symbol, tk))
    home = _home_driver(symbol)
    empty = pd.Series(np.nan, index=y.index)
    cols, names = {}, []
    for name, kind, parts in spec:
        if kind == "pool":
            ser = [kept[p] for p in parts if p in kept]
            f = pd.concat(ser, axis=1).mean(axis=1) if ser else empty
        elif kind == "sub":
            a, b = parts
            f = (kept[a] - kept[b]) if a in kept and b in kept else empty
        elif kind == "pool_sub":
            ser = [kept[a] - kept[b] for a, b in parts if a in kept and b in kept]
            f = pd.concat(ser, axis=1).mean(axis=1) if ser else empty
        else:   # home
            f = kept.get(home, empty)
        cols[name] = f
        names.append(name if kind != "home" else f"Home market ({'Nifty' if home == 'NIFTY' else 'S&P 500'})")
    return pd.DataFrame(cols, index=y.index), names


# ══════════════════════════════════════════════════════════════════════════════
#  4-10 · ANCHOR SAMPLE, SELECTION, REGRESSION, HEDGE SKILL
# ══════════════════════════════════════════════════════════════════════════════
def _partial(rxy, rxz, ryz):
    """Partial correlation of x and y controlling for z, from pairwise ρ."""
    d = np.sqrt(np.maximum(1e-12, (1.0 - rxz * rxz) * (1.0 - ryz * ryz)))
    return np.clip((rxy - rxz * ryz) / d, -1.0, 1.0)


def _masked_argmax(A: np.ndarray, *exclude: np.ndarray) -> np.ndarray:
    """Per-row argmax of |score|, excluded columns skipped — the Pine's f_argmax.

    Ties go to the lowest index (Pine iterates upward and keeps a strict >).
    """
    B = A.copy()
    rows = np.arange(len(B))
    for ex in exclude:
        B[rows, ex] = -1.0
    return np.argmax(B, axis=1)


def _hold_path(raw: np.ndarray, A: np.ndarray, *exclude: np.ndarray) -> np.ndarray:
    """The Pine's f_hold along time: HYSTERESIS on the selection.

    A challenger replaces the incumbent only when its |score| beats the
    incumbent's by more than 10%; candidates inside the margin are a statistical
    tie, and swapping between them on noise would rotate the basket for nothing.
    Path-dependent, so it is the one part that must run bar by bar — on scalars.
    """
    T = len(raw)
    out = np.empty(T, dtype=int)
    ex = [e.tolist() for e in exclude]
    raw_l = raw.tolist()
    prev = -1
    for t in range(T):
        r = raw_l[t]
        if prev >= 0 and prev != r and all(prev != e[t] for e in ex):
            prev = r if A[t, r] > A[t, prev] * (1.0 + HYST_MARG) else prev
        else:
            prev = r
        out[t] = prev
    return out


def fit_residual(y: pd.Series, F: pd.DataFrame) -> Dict[str, object]:
    """Sections 4-10 on the chart's own anchor: the out-of-sample residual path.

    Returns a dict of aligned arrays / Series:
        resid    the residual as actually hedged (ε_used)
        hedge    the hedge weight applied — measured skill, clipped to 0 … 1
        n_obs    observations in the fit (Pine's nObs)
        i1 i2 i3 the three selected factor indices, every bar (hysteresis path)
        k_eff    the model order admitted past the significance floor
    Missing factor data is per-column, never per-row: a missing return is a
    zero, and a factor that never exists stays constant and is never chosen.
    """
    T, K = len(y), F.shape[1]
    Y = y.to_numpy(dtype=float)
    Fv = F.to_numpy(dtype=float)
    valid = np.isfinite(Y)
    O = np.column_stack([np.where(valid, Y, 0.0), np.nan_to_num(Fv, nan=0.0)])
    O[~valid] = 0.0
    M = K + 1

    # Observation sequence and cumulative moments over it. At bar t the buffer
    # holds the observations pushed before t, minus the 12 still in the FIFO.
    seq = np.flatnonzero(valid)
    Os = O[seq]
    CS = np.vstack([np.zeros((1, M)), np.cumsum(Os, axis=0)])
    CP = np.concatenate([np.zeros((1, M, M)),
                         np.cumsum(Os[:, :, None] * Os[:, None, :], axis=0)])
    pushed = np.concatenate([[0], np.cumsum(valid)[:-1]]).astype(int)
    hi = np.clip(pushed - HOLD_OUT, 0, len(seq))
    lo = np.clip(hi - CORR_LEN, 0, None)
    n = (hi - lo).astype(float)
    S = CS[hi] - CS[lo]
    SP = CP[hi] - CP[lo]
    diag = np.einsum("tii->ti", SP)
    nn = np.where(n > 0, n, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n[:, None, None] * SP - S[:, :, None] * S[:, None, :]
        dvar = np.maximum(n[:, None] * diag - S * S, 0.0)
        den = np.sqrt(dvar[:, :, None] * dvar[:, None, :])
        corr = np.where((n[:, None, None] > 2) & (den > 1e-12),
                        np.clip(num / np.where(den > 0, den, 1.0), -1.0, 1.0), 0.0)
        var = np.where(n[:, None] > 1, (diag - S * S / nn[:, None]) / nn[:, None], np.nan)
        sd = np.sqrt(np.maximum(var, 0.0))
        mean = np.where(n[:, None] > 0, S / nn[:, None], np.nan)
    del SP, CP, num, den

    tt = np.arange(T)
    kk = np.arange(1, M)

    # ── Section 6 · stepwise selection, with hysteresis ──
    S1 = corr[:, 0, 1:]
    A1 = np.abs(S1)
    i1 = _hold_path(_masked_argmax(A1), A1)
    ry1 = S1[tt, i1]
    rk1 = corr[tt[:, None], kk[None, :], (i1 + 1)[:, None]]
    S2 = _partial(S1, ry1[:, None], rk1)
    S2[tt, i1] = 0.0
    A2 = np.abs(S2)
    i2 = _hold_path(_masked_argmax(A2, i1), A2, i1)

    r12 = corr[tt, i1 + 1, i2 + 1]
    ry2 = S1[tt, i2]
    ry21 = _partial(ry2, ry1, r12)
    rk2 = corr[tt[:, None], kk[None, :], (i2 + 1)[:, None]]
    rk21 = _partial(rk2, rk1, r12[:, None])
    ryk1 = _partial(S1, ry1[:, None], rk1)
    S3 = _partial(ryk1, ry21[:, None], rk21)
    S3[tt, i1] = 0.0
    S3[tt, i2] = 0.0
    A3 = np.abs(S3)
    i3 = _hold_path(_masked_argmax(A3, i1, i2), A3, i1, i2)

    # ── Admission: Fisher / Šidák floor sized to the sample actually available ──
    z_crit = _sidak_z(K)
    q2 = A2[tt, i2]
    q3 = A3[tt, i3]
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = np.where(n - 4.0 > 0, np.tanh(z_crit / np.sqrt(np.maximum(n - 4.0, 1e-9))), 1.0)
        f3 = np.where(n - 5.0 > 0, np.tanh(z_crit / np.sqrt(np.maximum(n - 5.0, 1e-9))), 1.0)
    f2 = np.maximum(MIN_RHO, f2)
    f3 = np.maximum(MIN_RHO, f3)
    adm2 = (MAX_VARS >= 2) & (q2 >= f2)
    adm3 = adm2 & (MAX_VARS >= 3) & (q3 >= f3)
    k_eff = 1 + adm2.astype(int) + adm3.astype(int)

    # ── Section 8 · Gram-Schmidt in correlation space, with ridge ──
    lam = RIDGE_LAM * np.maximum(1.0, 200.0 / np.maximum(n, 1.0))
    r13 = corr[tt, i1 + 1, i3 + 1]
    r23 = corr[tt, i2 + 1, i3 + 1]
    ry3 = S1[tt, i3]
    vu2 = np.maximum(0.0, 1.0 - r12 * r12)
    with np.errstate(invalid="ignore", divide="ignore"):
        a32 = np.where(vu2 > 1e-10, (r23 - r13 * r12) / np.where(vu2 > 1e-10, vu2, 1.0), 0.0)
    vu3 = np.maximum(0.0, 1.0 - r13 * r13 - a32 * a32 * vu2)
    c1 = ry1
    c2 = ry2 - r12 * ry1
    c3 = ry3 - r13 * ry1 - a32 * c2
    g1 = c1 / (1.0 + lam)
    g2 = np.where(k_eff >= 2, c2 / (vu2 + lam), 0.0)
    g3 = np.where(k_eff >= 3, c3 / (vu3 + lam), 0.0)
    w3 = g3
    w2 = g2 - a32 * g3
    w1 = g1 - r12 * g2 - (r13 - r12 * a32) * g3

    sdy = sd[:, 0]
    y_hat = mean[:, 0].copy()
    for w_k, i_k in ((w1, i1), (w2, i2), (w3, i3)):
        s_k = sd[tt, i_k + 1]
        ok = np.isfinite(s_k) & (s_k > 1e-12) & np.isfinite(sdy)
        with np.errstate(invalid="ignore", divide="ignore"):
            h_k = np.where(ok, w_k * sdy / np.where(ok, s_k, 1.0), 0.0)
        # ŷ = icept + Σ h·f, icept = m_Y − Σ h·m_X — Section 9: on the chart's own
        # anchor the chart bar IS the period, so the intercept needs no rescale.
        f_k = np.nan_to_num(Fv[tt, i_k], nan=0.0)
        y_hat = y_hat + h_k * (f_k - np.nan_to_num(mean[tt, i_k + 1], nan=0.0))
    y_hat[n <= 0] = np.nan
    resid_full = Y - y_hat

    # ── Section 10 · THE HEDGE WEIGHS ITSELF ──
    # Skill over the last 104 periods, each residual measured with the
    # coefficients in force at the time, read through the PREVIOUS bar.
    ok = np.isfinite(resid_full) & np.isfinite(Y)
    e2 = pd.Series(np.where(ok, resid_full ** 2, np.nan), index=y.index).dropna()
    y2 = pd.Series(np.where(ok, Y ** 2, np.nan), index=y.index).dropna()
    se = e2.rolling(SKILL_WIN, min_periods=SKILL_MIN).sum()
    sy = y2.rolling(SKILL_WIN, min_periods=SKILL_MIN).sum()
    skill = (1.0 - se / sy).where(sy > 1e-18)
    skill = skill.reindex(y.index).ffill().shift(1)
    hedge = skill.clip(lower=0.0, upper=1.0).fillna(0.0)

    hv = hedge.to_numpy()
    resid = pd.Series(np.where(np.isfinite(resid_full), Y - hv * (Y - resid_full), Y),
                      index=y.index).where(np.isfinite(Y))
    return {"resid": resid, "hedge": hedge, "n_obs": pd.Series(n, index=y.index),
            "i1": i1, "i2": i2, "i3": i3, "k_eff": k_eff}


def bars_since_rotation(i1: np.ndarray, i2: np.ndarray, i3: np.ndarray,
                        k_eff: np.ndarray) -> np.ndarray:
    """Section 7 · ROTATION DETECTION: bars since the basket last changed.

    Set membership, not order: swapping ranks 2 and 3 spans the same subspace,
    so the fit is unchanged and it is not a rotation. A change in the admitted
    model order is. The Pine's ▲▼ are only tradable once the basket has been
    stable for a representative window (21 bars).
    """
    T = len(i1)
    out = np.zeros(T, dtype=int)
    if T == 0:
        return out
    p1, p2, p3 = np.roll(i1, 1), np.roll(i2, 1), np.roll(i3, 1)

    def _in(v):
        return (v == p1) | (v == p2) | (v == p3)

    rot = ~(_in(i1) & _in(i2) & _in(i3)) | (k_eff != np.roll(k_eff, 1))
    rot[0] = False
    run = 0
    for t, r in enumerate(rot.tolist()):
        run = 0 if r else run + 1
        out[t] = run
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  11-12 · THE SPREAD AND THE RV ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def rv_z(spread: pd.Series) -> pd.Series:
    """Five timescale members, each high-passed, averaged, rescaled by 0.887."""
    acc = 0.0
    for zl in Z_WINDOWS:
        sd = spread.rolling(zl).std(ddof=0)
        m = spread.rolling(zl).mean()
        z = ((spread - m) / sd).where(sd > 1e-14, 0.0).fillna(0.0)
        acc = acc + (z - _ema(z, DRIFT_LEN))
    return acc / len(Z_WINDOWS) / ENS_SCALE


# ══════════════════════════════════════════════════════════════════════════════
#  LEG 2 · BREADTH — Tattva's Swayam price-action half, seven MSF members
# ══════════════════════════════════════════════════════════════════════════════
def _sig(x, s):
    return 2.0 / (1.0 + np.exp(-x / s)) - 1.0


# Breadth runs on plain arrays: seven members × a dozen rolling windows each,
# on the chart and again on the parent frame, is thousands of tiny pandas
# operations per name, and a screener runs it on hundreds of names. Windowed
# means and deviations are taken over explicit sliding windows, so a window
# holding a NaN yields NaN — pandas' rolling(w) semantics, which the Pine's
# na propagation matches.
def _rmean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = _swv(x, w).mean(axis=1)
    return out


def _rstd(x: np.ndarray, w: int, ddof: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = _swv(x, w).std(axis=1, ddof=ddof)
    return out


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def _zc(src: np.ndarray, win: int) -> np.ndarray:
    """Tattva's zscore_clipped: z against the PREVIOUS window (shift 1, ddof 1)."""
    m = np.nan_to_num(_lag(_rmean(src, win), 1), nan=0.0)
    sd = _lag(_rstd(src, win, 1), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(sd) & (sd != 0), (src - m) / np.where(sd != 0, sd, 1.0), np.nan)
    return np.clip(np.nan_to_num(z, nan=0.0), -CLIP_Z, CLIP_Z)


def _msf(a: Dict[str, np.ndarray], win: int, roc: int, mask: int) -> np.ndarray:
    o, h, l, c, v = a["open"], a["high"], a["low"], a["close"], a["volume"]
    c5, c10, cr = _lag(c, 5), _lag(c, 10), _lag(c, roc)
    with np.errstate(invalid="ignore", divide="ignore"):
        momentum = _sig(_zc((c - cr) / cr, win), 1.5)
        if mask == 1:
            return _sig(momentum, 1.0)
        vol_ma = _rmean(v, win)
        vol_ratio = np.where(np.isfinite(vol_ma) & (vol_ma != 0), v / np.where(vol_ma != 0, vol_ma, 1.0), 1.0)
        trend_slow = _rmean(c, win)
        structure = flow = 0.0
        if mask in (0, 2):
            vw_dir = _rmean(((h + l) / 2.0 - o) * vol_ratio, win)
            vw_imp = _rmean((c - c5) * vol_ratio, win)
            micro = _sig(_zc(vw_dir - vw_imp, win), 1.5)
            ctz = (_zc(_rmean(c, 5) - trend_slow, win) + _zc(c - 2.0 * c5 + c10, win)
                   + _zc((c - c5) / a["atr"], win) + _zc(c - trend_slow, win)) / 2.0
            structure = (micro + _sig(ctz, 1.5)) / np.sqrt(2.0)
        if mask in (0, 3):
            mf = (h + l + c) / 3.0 * v
            c1 = _lag(c, 1)
            pos = _rmean(np.where(c > c1, mf, 0.0), win)
            neg = _rmean(np.where(c < c1, mf, 0.0), win)
            tot = pos + neg
            accum = 2.0 * (np.where(np.isfinite(tot) & (tot != 0), pos / np.where(tot != 0, tot, 1.0), 0.5) - 0.5)
            pct = c / c1 - 1.0
            cnt = np.cumsum(np.select([pct > 0.0033, pct < -0.0033], [1.0, -1.0], 0.0))
            regime = _sig(_zc(cnt - _rmean(cnt, win), win), 1.5)
            flow = (accum + regime) / np.sqrt(2.0)
    msum = {0: momentum + structure + flow, 2: structure, 3: flow}[mask]
    return _sig(msum, 1.0)


def _unified_msf(msf: np.ndarray) -> np.ndarray:
    cl = np.abs(msf) ** REGIME_SENS
    w_ad = cl / (cl + 0.001)
    w_fin = 0.5 * BASE_WEIGHT + 0.5 * w_ad
    mmr = 0.5 * (1.0 - BASE_WEIGHT)
    return np.clip(w_fin / (w_fin + mmr) * msf, -1.0, 1.0)


def breadth_z(df: pd.DataFrame) -> pd.Series:
    """The breadth leg: seven MSF members, mean × 10 — already unit variance.

    VOLUME DEGENERACY: flow is the only component that needs real volume, so the
    flow-only member is dropped on instruments that do not report it.
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = (df["volume"].to_numpy(dtype=float) if "volume" in df.columns
         else np.full(len(df), np.nan))
    pc = _lag(c, 1)
    tr = np.where(np.isfinite(pc), np.fmax(h, pc) - np.fmin(l, pc), h - l)
    a = {"open": df["open"].to_numpy(dtype=float), "high": h, "low": l, "close": c,
         "volume": v, "tr": tr,
         "atr": pd.Series(tr).ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy()}
    rb = [int(max(5.0, np.floor(ROC_FRAC * b + 0.5))) for b in B_WINDOWS]
    mem = [_unified_msf(_msf(a, b, r, 0)) for b, r in zip(B_WINDOWS, rb)]
    mem.append(_unified_msf(_msf(a, B_WINDOWS[0], rb[0], 1)))        # fastest · momentum
    flow = _unified_msf(_msf(a, B_WINDOWS[-1], rb[-1], 3))           # slowest · flow
    has_vol = np.cumsum(np.isfinite(v) & (v != 0)) / np.arange(1, len(c) + 1) >= 0.5
    out = (sum(mem) + np.where(has_vol, flow, 0.0)) * 10.0 / np.where(has_vol, 7.0, 6.0)
    return pd.Series(out, index=df.index, name="breadth_z")


def blend(rz: pd.Series, bz: pd.Series, rho: pd.Series) -> pd.Series:
    """Blend in z-space with the measured leg ρ, variance restored — once.

    The Pine's f_tpBlend: a rung missing one leg falls back to the other.
    """
    w_rv, w_br = LEG_MIX, 1.0 - LEG_MIX
    rc = rho.fillna(0.71).clip(-0.90, 0.99)
    mv = w_rv * w_rv + w_br * w_br + 2.0 * w_rv * w_br * rc
    both = (w_rv * rz + w_br * bz) * np.where(mv > 1e-9, 1.0 / np.sqrt(mv), 1.0)
    out = both.where(rz.notna() & bz.notna())
    if w_rv > 0:
        out = out.fillna(rz.where(bz.isna()))
    if w_br > 0:
        out = out.fillna(bz.where(rz.isna()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MTF LADDER · the parent rung, reconstructed on the chart's bars
# ══════════════════════════════════════════════════════════════════════════════
def parent_rv(spread: pd.Series, built: pd.Series, key: pd.PeriodIndex) -> pd.Series:
    """The RV ensemble on the parent frame, finished with the current spread.

    Settled samples are the spread at each closed parent bar's last chart bar
    (where the model was built); each member's drift EMA advances once per
    closed parent bar. The forming parent bar's reading uses the current spread
    as its close — so on the parent's final chart bar it lands exactly on the
    value that parent bar then settles on (f_tpStep / f_tpLive).
    """
    k = pd.Series(key, index=spread.index)
    last = k != k.shift(-1)
    settle_mask = (last & built).to_numpy()
    # A parent bar settles on the first chart bar of the NEXT one, so the final
    # parent bar in the data is still forming.
    if len(settle_mask):
        settle_mask[-1] = False
    smp = spread.to_numpy(dtype=float)[settle_mask]
    smp_key = k.to_numpy()[settle_mask]
    # How many settled samples each chart bar can see: every settled parent bar
    # before its own.
    avail = np.searchsorted(smp_key, k.to_numpy(), side="left")
    cs1 = np.concatenate([[0.0], np.cumsum(smp)])
    cs2 = np.concatenate([[0.0], np.cumsum(smp * smp)])
    al = 2.0 / (DRIFT_LEN + 1.0)
    cur = spread.to_numpy(dtype=float)
    acc = np.zeros(len(cur))
    okk = np.ones(len(cur), dtype=bool)
    j = np.arange(len(smp))
    for zl in Z_WINDOWS:
        w = zl - 1
        # settled z of each sample j, window = samples j-zl+1 … j
        jj = j + 1
        lo = jj - zl
        valid_j = lo >= 0
        a1 = np.where(valid_j, cs1[jj] - cs1[np.clip(lo, 0, None)], np.nan)
        a2 = np.where(valid_j, cs2[jj] - cs2[np.clip(lo, 0, None)], np.nan)
        mu = a1 / zl
        sdv = np.sqrt(np.maximum(a2 / zl - mu * mu, 0.0))
        zs = np.where(sdv > 1e-14, (smp - mu) / np.where(sdv > 0, sdv, 1.0), 0.0)
        zs = np.where(valid_j, zs, np.nan)
        ema = pd.Series(zs, dtype=float).ewm(alpha=al, adjust=False, ignore_na=True).mean().to_numpy()
        ema = np.where(np.isfinite(pd.Series(zs, dtype=float).ffill().to_numpy()), ema, np.nan)
        # live z at each chart bar: the last w settled samples + the current spread
        p = avail
        ok = p >= w
        s1 = np.where(ok, cs1[p] - cs1[np.clip(p - w, 0, None)], np.nan) + cur
        s2 = np.where(ok, cs2[p] - cs2[np.clip(p - w, 0, None)], np.nan) + cur * cur
        mu_l = s1 / zl
        sd_l = np.sqrt(np.maximum(s2 / zl - mu_l * mu_l, 0.0))
        zl_live = np.where(sd_l > 1e-14, (cur - mu_l) / np.where(sd_l > 0, sd_l, 1.0), 0.0)
        e_prev = (ema[np.clip(p - 1, 0, None)] if len(ema) else np.full(len(p), np.nan))
        e_prev = np.where(p > 0, e_prev, np.nan)
        ek = np.where(np.isfinite(e_prev), al * zl_live + (1.0 - al) * e_prev, zl_live)
        good = ok & np.isfinite(zl_live)
        okk &= good
        acc = acc + np.where(good, zl_live - ek, 0.0)
    out = np.where(okk & built.to_numpy() & np.isfinite(cur), acc / len(Z_WINDOWS) / ENS_SCALE, np.nan)
    return pd.Series(out, index=spread.index)


def parent_breadth(df: pd.DataFrame, key: pd.PeriodIndex) -> pd.Series:
    """Breadth on the parent frame, as of its last CLOSED bar, per chart bar."""
    g = df.groupby(key)
    pb = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                       "low": g["low"].min(), "close": g["close"].last(),
                       "volume": g["volume"].sum(min_count=1) if "volume" in df.columns else np.nan})
    bz = breadth_z(pb)
    bz = bz.where(np.arange(len(bz)) > PARENT_BREADTH_MIN)
    prev = bz.shift(1).reindex(key)
    return pd.Series(prev.to_numpy(dtype=float), index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
#  THE VALUE READINGS FOR ONE NAME
# ══════════════════════════════════════════════════════════════════════════════
def _driver_labels(names: List[str], i1, i2, i3, k_eff, built) -> pd.Series:
    """The drivers in use per bar, as text — built once per distinct basket."""
    code = (np.asarray(k_eff) * 1_000_000 + np.asarray(i1) * 10_000
            + np.asarray(i2) * 100 + np.asarray(i3))
    lab = {}
    for c in np.unique(code):
        k, r = divmod(int(c), 1_000_000)
        a, r = divmod(r, 10_000)
        b, d = divmod(r, 100)
        lab[int(c)] = " · ".join(names[i] for i in (a, b, d)[:k])
    out = pd.Series([lab[int(c)] for c in code], dtype=object)
    return out.where(np.asarray(built, dtype=bool), "")


def compute_value(df: pd.DataFrame, driver_closes: Optional[pd.DataFrame], symbol: str,
                  chart: str = "D", basket: Optional[str] = None) -> pd.DataFrame:
    """Every value reading for one name's OHLCV (lower-case columns), per bar.

    Columns (VALUE_COLUMNS):
        value_z         Samanvaya's unified z — the trace's value ingredient
        value           the same on its ±100 scale (+ rich, − cheap)
        rv_z breadth_z  the two legs; legs_split when they disagree past ½σ
        model_built     nObs ≥ 10 — the Pine's modelBuilt
        enough          nObs ≥ 30 — enoughData
        n_obs hedge drivers
        bars_since_rot  bars since the driver basket last rotated
        basket_warm     ≥ 21 of them — a TURN window may open
        parent_z        the parent rung's blended z (W on Daily, M on Weekly)
        tape            the MTF value tape, ±100, + rich
        tape_ready      model built and the parent rung calibrated
        tape_rungs      rungs in the ladder mean, the chart included

    `driver_closes` carries the macro drivers (fetched once per universe).
    Without them the engine still runs: every factor is empty, the hedge never
    earns skill, and the RV leg is the name's own path — the Pine's "Macro
    hedge: Off", which is where Auto lands anyway when the drivers explain
    nothing.
    """
    cols = list(VALUE_COLUMNS)
    if df is None or df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
        return pd.DataFrame(index=getattr(df, "index", None), columns=cols)
    df = df.sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    if len(df) < max(Z_WINDOWS) + HOLD_OUT + MODEL_MIN + 2:
        return pd.DataFrame(index=df.index, columns=cols)

    y = np.log(df["close"]).diff()
    F, names = build_factors(y, driver_returns(driver_closes, df.index, symbol, chart), symbol,
                             basket or DEFAULT_BASKET)
    fit = fit_residual(y, F)
    resid, hedge, n_obs = fit["resid"], fit["hedge"], fit["n_obs"]

    built = n_obs >= MODEL_MIN
    ready = built & resid.notna()
    spread = resid.where(ready, 0.0).fillna(0.0).cumsum()

    rz = rv_z(spread)
    bz = breadth_z(df)
    rho = rz.rolling(VAR_CORR_LEN, min_periods=VAR_CORR_LEN).corr(bz)
    w_rv, w_br = LEG_MIX, 1.0 - LEG_MIX
    rc = rho.fillna(0.71).clip(-0.90, 0.99)
    mv = w_rv * w_rv + w_br * w_br + 2.0 * w_rv * w_br * rc
    unified = (w_rv * rz + w_br * bz) * np.where(mv > 1e-9, 1.0 / np.sqrt(mv), 1.0)
    value = (100.0 * softbound(unified * GAIN)).clip(-100.0, 100.0)
    legs_split = ((rz > 0) != (bz > 0)) & (rz.abs() > 0.5) & (bz.abs() > 0.5)

    # ── the parent rung and the tape ──
    key = period_key(df.index, CHARTS[chart]["parent"])
    p_rv = parent_rv(spread, built, key)
    p_bz = parent_breadth(df, key)
    p_rho = p_rv.rolling(VAR_CORR_LEN, min_periods=VAR_CORR_LEN).corr(p_bz)
    p_u = blend(p_rv, p_bz, p_rho)
    rungs = 1 + p_u.notna().astype(int)
    tape_z = (unified + p_u.fillna(0.0)) / rungs
    tape = (100.0 * softbound(tape_z * GAIN)).clip(-100.0, 100.0)
    tape_ready = built & p_u.notna()

    bsr = bars_since_rotation(fit["i1"], fit["i2"], fit["i3"], fit["k_eff"])
    drv = _driver_labels(names, fit["i1"], fit["i2"], fit["i3"], fit["k_eff"], built.to_numpy())
    drv.index = df.index

    return pd.DataFrame({
        "value_z": unified.astype(float),
        "value": value.astype(float),
        "rv_z": rz.astype(float),
        "breadth_z": bz.astype(float),
        "legs_split": legs_split.astype(bool),
        "model_built": built.astype(bool),
        "enough": (n_obs >= MIN_OBS).astype(bool),
        "n_obs": n_obs.astype(float),
        "hedge": hedge.where(built).astype(float),
        "drivers": drv,
        "bars_since_rot": bsr,
        "basket_warm": bsr >= BASKET_SETTLE,
        "parent_z": p_u.astype(float),
        "tape": tape.astype(float),
        "tape_ready": tape_ready.astype(bool),
        "tape_rungs": rungs.where(built, 0).astype(int),
    }, index=df.index)[cols]


__all__ = [
    "BASKETS", "CHARTS", "DEFAULT_BASKET", "DRIVERS", "DRIVER_TICKERS",
    "THETA", "THETA_OSC", "VALUE_COLUMNS",
    "bars_since_rotation", "blend", "breadth_z", "build_factors", "compute_value",
    "driver_returns", "fit_residual", "parent_breadth", "parent_rv", "period_key",
    "prepare_drivers",
    "rv_z", "softbound",
]
