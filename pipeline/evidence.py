"""
Tattva — admitted evidence: who closes the gap, the claim's unit, the gap's
stationarity and the reversion sizer's gap-implied move.
तत्त्व (Tattva) — "Principle / Essence"

Evidence audits a role-holder from that role-holder's own published outputs,
with no model of its own, and never moves the call. Three pieces audit the
claim here and one the reversion sizer:

  * :func:`gap_split` — when Mūla's gap was open, how much of its closing
    over the call's horizon did PRICE do, and how much did FAIR VALUE do by
    moving toward price? The claim is a price-side claim, so Tattva does not
    assume the answer: it measures it and shows it. (On the audited
    instruments fair value does most of the closing.)
  * :func:`claim_unit` — is the oscillator in the unit its cuts assume (the
    charter's C12 contract: centred, unit SD, RICH and CHEAP balanced)?
  * :func:`gap_tests` — ADF and KPSS on the published gap through d*, read
    into the one stationarity word (``core.semantics.gap_stationarity``): a
    full-history diagnostic, no precondition;
  * :func:`ecm_drift` — the error-correction layer's gap-implied h-session
    move (``MulaDriftPct``, the drift excluded), a lean of the reversion
    sizer's own layer that ``pipeline.assemble.grade_all`` grades as
    ``edges['evidence:ecm_drift']``.

No Streamlit dependency.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from core.knobs import knob
from core.semantics import claim_direction, gap_stationarity

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover — optional dependency
    _HAS_STATSMODELS = False

#: Pairs per phase offset below which an offset's slope is not computed.
_MIN_PAIRS: int = knob("GRADE_MIN_WIN")

#: The C12 contract on the claim's unit (charter clause C12): what a centred,
#: unit-scale oscillator reads. Evidence thresholds; they never move the call.
UNIT_MAX_ABS_MEAN = 0.15
UNIT_SD_RANGE = (0.8, 1.2)
UNIT_RICH_CHEAP_RANGE = (0.67, 1.5)

#: Published gaps the stationarity tests need: with this many or fewer
#: neither runs, and the word is 'unit root not rejected'.
TEST_MIN_OBS = 30


def _frame(fvo_ts: pd.DataFrame) -> pd.DataFrame:
    f = fvo_ts.copy()
    f.index = pd.DatetimeIndex(pd.to_datetime(f.index)).normalize()
    return f[~f.index.duplicated(keep="last")].sort_index()


def _slopes(gap: np.ndarray, dp: np.ndarray, dfv: np.ndarray, h: int) -> tuple[float, float]:
    """Offset-averaged stride-h OLS slopes of the price and fair-value moves on the gap."""
    bp, bf = [], []
    for o in range(h):
        g, a, b = gap[o::h], dp[o::h], dfv[o::h]
        if len(g) < _MIN_PAIRS:
            continue
        gc = g - g.mean()
        v = float(gc @ gc)
        if not v > 0:
            continue
        bp.append(float(gc @ (a - a.mean())) / v)
        bf.append(float(gc @ (b - b.mean())) / v)
    return (float(np.mean(bp)), float(np.mean(bf))) if bp else (math.nan, math.nan)


def _share(b_price: float, b_fv: float) -> float:
    """Price's share of the gap closed: −b_price / (b_fv − b_price)."""
    closed = b_fv - b_price
    return -b_price / closed if np.isfinite(closed) and closed != 0 else math.nan


def gap_split(fvo_ts: pd.DataFrame, h: int, exec_lag: int, *, block_h: int = 5,
              n_boot: int = 200, seed: int = 0) -> dict:
    """Who closes Mūla's gap over the call's horizon: price or fair value.

    On Valid sessions s, with ``gap_s = log P_s − log FV_s`` and the moves
    over the call's window ``Δ_h X_s = X(s+L+h) − X(s+L)``, regress ``Δ_h log P``
    and ``Δ_h log FV`` on ``gap_s`` — stride h, so no two windows overlap,
    averaged over the h phase offsets. A gap closed entirely by price gives
    ``b_price = −1, b_fv = 0``; one closed entirely by fair value moving to
    price gives ``b_price = 0, b_fv = +1``. ``price_share`` is price's part of
    what closed, ``−b_price / (b_fv − b_price)``.

    The SEs are a moving-block bootstrap of the session rows (blocks of
    ``block_h · h`` sessions, ``n_boot`` draws, fixed ``seed``). This is
    evidence, not a grade: it moves nothing.

    Returns ``{b_price, b_fv, price_share, closed, se_b_price, se_b_fv,
    se_share, n_win, n_obs, h, exec_lag, span}``.
    """
    h, lag = max(1, int(h)), int(exec_lag)
    f = _frame(fvo_ts)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(pd.to_numeric(f["Price"], errors="coerce").where(lambda s: s > 0))
        lfv = np.log(pd.to_numeric(f["FairValue"], errors="coerce").where(lambda s: s > 0))
    gap = lp - lfv
    dp = lp.shift(-(h + lag)) - lp.shift(-lag)
    dfv = lfv.shift(-(h + lag)) - lfv.shift(-lag)
    rows = pd.DataFrame({"gap": gap, "dp": dp, "dfv": dfv})[f["Valid"].astype(bool)].dropna()
    g, a, b = (rows[c].to_numpy(dtype=np.float64) for c in ("gap", "dp", "dfv"))
    n = len(rows)
    b_price, b_fv = _slopes(g, a, b, h)

    se_p = se_f = se_s = math.nan
    block = max(1, int(block_h) * h)
    if n >= 2 * block and np.isfinite(b_price):
        rng = np.random.default_rng(seed)
        n_blk = -(-n // block)
        draws = []
        for _ in range(int(n_boot)):
            starts = rng.integers(0, n - block + 1, size=n_blk)
            ix = (starts[:, None] + np.arange(block)).ravel()[:n]
            bp, bf = _slopes(g[ix], a[ix], b[ix], h)
            draws.append((bp, bf, _share(bp, bf)))
        d = np.asarray(draws)
        se_p, se_f, se_s = (float(np.nanstd(d[:, j], ddof=1)) for j in range(3))

    return {"b_price": b_price, "b_fv": b_fv, "price_share": _share(b_price, b_fv),
            "closed": b_fv - b_price, "se_b_price": se_p, "se_b_fv": se_f, "se_share": se_s,
            "n_win": n // h, "n_obs": n, "h": h, "exec_lag": lag,
            "span": (rows.index[0], rows.index[-1]) if n else None}


def claim_unit(fvo_ts: pd.DataFrame) -> dict:
    """The claim's unit as of d*, against the C12 contract.

    On the Valid sessions up to the newest one (the causal expanding value at
    d*): FVO's mean and SD, the RICH:CHEAP ratio of claim sessions (|FVO| ≥
    CLAIM_SIGMA, by ``claim_direction``) and the SELL share of claims. A
    centred, unit-scale oscillator reads |mean| < 0.15, SD in [0.8, 1.2] and
    RICH:CHEAP in [0.67, 1.5]; ``contract`` says which of these hold.
    """
    f = _frame(fvo_ts)
    fvo = pd.to_numeric(f["FVO"], errors="coerce")[f["Valid"].astype(bool)].dropna()
    claims = fvo.map(claim_direction)
    rich, cheap = int((claims < 0).sum()), int((claims > 0).sum())
    mean = float(fvo.mean()) if len(fvo) else math.nan
    sd = float(fvo.std(ddof=1)) if len(fvo) > 1 else math.nan
    ratio = rich / cheap if cheap else math.nan
    contract = {
        "mean": bool(abs(mean) < UNIT_MAX_ABS_MEAN),
        "sd": bool(UNIT_SD_RANGE[0] <= sd <= UNIT_SD_RANGE[1]),
        "rich_cheap": bool(UNIT_RICH_CHEAP_RANGE[0] <= ratio <= UNIT_RICH_CHEAP_RANGE[1]),
    }
    return {"mean": mean, "sd": sd, "rich_cheap": ratio,
            "sell_share": rich / (rich + cheap) if rich + cheap else math.nan,
            "n": int(len(fvo)), "asof": fvo.index[-1] if len(fvo) else None,
            "contract": contract, "passes": all(contract.values())}


def gap_tests(fvo_ts: pd.DataFrame) -> dict:
    """The gap's stationarity as of d*: ADF and KPSS, and the one word for them.

    Both tests run on the published gap (``Residual``) of the Valid sessions,
    the history through d* the claim is graded on: ADF with its lag order by
    AIC (null: a unit root), KPSS about a constant with automatic lags (null:
    stationary; statsmodels reports its p-value only inside [0.01, 0.10]).
    ``state`` is ``core.semantics.gap_stationarity(adf_pvalue, kpss_pvalue)``,
    labelled a full-history diagnostic wherever it is shown: one test of the
    whole record, not an as-of series, and no precondition of the call.

    Returns ``{state, adf_pvalue, kpss_pvalue, n, asof, scope}``.
    """
    f = _frame(fvo_ts)
    gap = pd.to_numeric(f["Residual"], errors="coerce")[f["Valid"].astype(bool)].dropna()
    g = gap.to_numpy(dtype=np.float64)
    adf_p = kpss_p = math.nan
    if _HAS_STATSMODELS and len(g) > TEST_MIN_OBS:
        with warnings.catch_warnings():
            # statsmodels announces a return-type change (the p-value is
            # element [1] under both) and that KPSS's table is truncated.
            warnings.simplefilter("ignore")
            adf_p = float(adfuller(g, autolag="AIC")[1])
            kpss_p = float(kpss(g, regression="c", nlags="auto")[1])
    return {"state": gap_stationarity(adf_p, kpss_p), "adf_pvalue": adf_p, "kpss_pvalue": kpss_p,
            "n": int(len(g)), "asof": gap.index[-1] if len(g) else None,
            "scope": "full-history diagnostic"}


def ecm_drift(fvo_ts: pd.DataFrame, h: int) -> dict:
    """The reversion sizer's layer, read as a lean: the gap-implied h-session move.

    ``MulaDriftPct`` is the error-correction layer's projection of today's
    gap over the horizon, b·z·Σφᵏ in percent: the move PRICE would make if it
    responded to the gap as the layer estimates, with the regression's
    intercept — the drift — left out. It is the reversion sizer's mechanism in
    price units, so it is EVIDENCE→sizer:reversion: bullish-positive, graded
    as ``edges['evidence:ecm_drift']`` (drift-neutral, on the valued
    sessions), shown neutral-toned beside that grade, and never the expected
    move.

    Returns ``{lean, label, value, sd, asof, h}``: the series by date, the
    words it is shown under, and its reading and h-session SD (``MulaSdPct``)
    at d*.
    """
    f = _frame(fvo_ts)
    lean = pd.to_numeric(f["MulaDriftPct"], errors="coerce").astype(float)
    sd = pd.to_numeric(f["MulaSdPct"], errors="coerce").astype(float)
    pub = f.index[pd.to_numeric(f["FVO"], errors="coerce").notna().to_numpy()]
    d_star = pub[-1] if len(pub) else None

    def _at(x: pd.Series) -> float:
        v = x.get(d_star) if d_star is not None else None
        return float(v) if v is not None and np.isfinite(v) else math.nan

    return {"lean": lean, "label": f"gap-implied {int(h)}D move (excludes drift)",
            "value": _at(lean), "sd": _at(sd), "asof": d_star, "h": int(h)}


def build_evidence(fvo_ts: pd.DataFrame, *, h: int, exec_lag: int) -> dict:
    """The admitted evidence: ``{'gap_split', 'claim_unit', 'gap_stationarity',
    'ecm_drift'}``.

    Each piece is computed on its own, and one that fails reads
    ``{'error': ...}`` without touching the others or the call.
    """
    out: dict = {}
    for name, fn in (("gap_split", lambda: gap_split(fvo_ts, h, exec_lag)),
                     ("claim_unit", lambda: claim_unit(fvo_ts)),
                     ("gap_stationarity", lambda: gap_tests(fvo_ts)),
                     ("ecm_drift", lambda: ecm_drift(fvo_ts, h))):
        try:
            out[name] = fn()
        except Exception as e:                       # noqa: BLE001 - evidence, never the call
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out
