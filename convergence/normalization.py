"""
Tattva — the normalized CONSENSUS of Mūla and Swayam, and the tier words for every composite.
तत्त्व (Tattva) — "Principle / Essence"

The consensus is the plain 50/50 average of the two engines' readings, each put
on a common scale: Mūla's valuation oscillator (FVO, σ) and Swayam's average
view oscillator (±10). It is EVIDENCE — the Convergence page draws it and the
hero's convergence gate reads its sign — not the system's call. The call is the
conviction chain (``ui.components.build_hero_verdict``): one claim from Mūla,
gates that discount it. This module used to describe the consensus as "the
hero card's headline value"; it stopped being that when the chain replaced it,
and a docstring claiming otherwise is how a diagnostic ends up presented as a
verdict ("TATTVA CONVICTION").

ORIENTATION. Like every engine reading (``core.semantics``), the consensus is
positive when the engines see the instrument stretched UP — rich and
overbought — and negative when stretched DOWN. Negative = bullish.

THE SCALE IS ZERO-ANCHORED. Each input is divided by its own causal RMS — the
root-mean-square of strictly EARLIER readings — and nothing is subtracted.
The previous transform subtracted the expanding MEAN, so it measured each
engine relative to its own historical average rather than its reading: an
asset that had mostly been rich and trending printed a "STRONG BUY" consensus
on a day both engines read exactly zero. Both inputs have a meaningful zero
(fair value; a neutral oscillator), and that zero is what the sign has to
mean. Scaling by strictly earlier rows also stops today's value from being
judged against a spread that already contains it.

THE WORDS ARE LEANS, NOT ORDERS. A consensus or a composite is classified
STRONGLY BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY BEARISH; BUY and SELL
are reserved for the hero's action. The tier levels are the instrument's own
causal p90 / p75 of the series' magnitude (``analytics.adaptive``), with the
pooled constants below as warm-up priors only — and the card, the plot markers
and anything else that labels these series must classify through
:func:`lean_label` with the per-date tiers from :func:`causal_tiers`, so the
label on a card and the marker on the plot beside it are the same decision.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from analytics.adaptive import adaptive_tiers

log = logging.getLogger(__name__)


# ── Warm-up priors for the tiers (|value| on the ±1 scale) ─────────────────
# Used until a series has a year of its own history; analytics.adaptive then
# re-derives the same p90 / p75 from the instrument's own past. The consensus
# pair was anchored on the old demeaned construction; on the zero-anchored one
# it is only a starting point, and the causal estimate takes over after 252
# rows either way.
CONSENSUS_PRIOR: dict[str, float] = {"strong": 0.404, "moderate": 0.279}
COMPOSITE_PRIOR: dict[str, float] = {"strong": 0.159, "moderate": 0.092}

#: Rows of an input's own history before its scale is trusted. Before this the
#: normalized value is NaN rather than a ratio to a two-point RMS.
MIN_SCALE_OBS = 20

LEAN_LABELS: tuple[str, ...] = (
    "STRONGLY BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY BEARISH")


def lean_label(reading: float, strong: float, moderate: float) -> str:
    """Tier word for a READING-oriented value (negative = bullish).

    ``strong`` / ``moderate`` are magnitudes on the reading's own scale.
    NaN reads NEUTRAL only when there is genuinely nothing to classify; callers
    that can be missing should say so themselves rather than show a word.
    """
    try:
        v = float(reading)
    except (TypeError, ValueError):
        return "NEUTRAL"
    if not np.isfinite(v):
        return "NEUTRAL"
    s, m = abs(float(strong)), abs(float(moderate))
    if v <= -s:
        return "STRONGLY BULLISH"
    if v <= -m:
        return "BULLISH"
    if v >= s:
        return "STRONGLY BEARISH"
    if v >= m:
        return "BEARISH"
    return "NEUTRAL"


def causal_tiers(values, prior: dict[str, float], scale: float = 1.0,
                 min_obs: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(strong, moderate)`` tier levels from each row's own past.

    p90 / p75 of ``|values|`` over strictly earlier rows, falling back to
    ``prior × scale`` while history is short. ``scale`` lets a ±100 series use
    the ±1 priors (pass 100).
    """
    arr = np.asarray(values, dtype=np.float64)
    tiers = adaptive_tiers(
        np.where(np.isfinite(arr), arr, np.nan),
        {"strong": float(prior["strong"]) * scale,
         "moderate": float(prior["moderate"]) * scale},
        quantiles={"strong": 0.90, "moderate": 0.75}, min_obs=min_obs)
    return tiers["strong"], tiers["moderate"]


def classify_convergence_score(score_pm100: float, strong: float | None = None,
                               moderate: float | None = None) -> str:
    """Tier word for the weighted composite ``convergence_score`` (±100).

    ``strong`` / ``moderate`` on the ±100 scale; omitted, the composite's own
    warm-up priors. Prefer passing the per-date levels from
    :func:`causal_tiers` (``scale=100``) — that is what makes a label agree
    with the marker drawn for the same date.
    """
    s = COMPOSITE_PRIOR["strong"] * 100.0 if strong is None else strong
    m = COMPOSITE_PRIOR["moderate"] * 100.0 if moderate is None else moderate
    return lean_label(score_pm100, s, m)


def _swayam_signal_column(df: pd.DataFrame) -> str | None:
    """Return the first available Swayam average-signal column, or None."""
    for c in ("avg_unified_osc", "Avg_Signal", "avg_signal"):
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("avg_signal", "avg_unified_osc"):
            return c
    return None


def align_fvo_swayam(
    fvo_ts: pd.DataFrame | None,
    swayam_daily: pd.DataFrame | None,
    filter_dates: Iterable[str] | None = None,
) -> tuple[list, list[float], list[float]]:
    """Align Mūla's oscillator (``FVO``) and Swayam's average signal by date.

    Only dates on which Mūla PUBLISHED a valuation (finite FVO — burn-in and
    withheld sessions are NaN) and Swayam has a row are returned. The Mūla
    input is the oscillator itself, the same number the hero's claim is made
    from; it used to be ``ConvictionRaw``, the window-agreement score, which is
    a different construct, so the consensus and the claim could read different
    Mūlas.

    Args:
        fvo_ts: Mūla frame with an ``FVO`` column, date-indexed (app.py sets
            the index to ``Date``) or carrying a ``Date`` column.
        swayam_daily: Swayam aggregate indexed by date, with ``Avg_Signal``.
        filter_dates: optional ``YYYY-MM-DD`` strings to restrict to (the
            plot's visible window).

    Returns ``(dates, fvo_raw, swayam_raw)`` — parallel lists, empty when
    either input is missing or nothing overlaps.
    """
    if fvo_ts is None or "FVO" not in fvo_ts.columns:
        return [], [], []
    if swayam_daily is None or swayam_daily.empty:
        return [], [], []
    avg_col = _swayam_signal_column(swayam_daily)
    if avg_col is None:
        return [], [], []

    sw = pd.to_numeric(swayam_daily[avg_col], errors="coerce")
    sw.index = pd.to_datetime(sw.index, errors="coerce").normalize()
    sw = sw[~sw.index.isna()]
    sw = sw[~sw.index.duplicated(keep="last")]

    a = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
    if "Date" in a.columns:
        d_index = pd.to_datetime(a["Date"], errors="coerce", dayfirst=True)
    else:
        d_index = pd.to_datetime(pd.Series(a.index), errors="coerce")
    fvo = pd.Series(pd.to_numeric(a["FVO"], errors="coerce").to_numpy(dtype=np.float64),
                    index=pd.DatetimeIndex(d_index).normalize())
    fvo = fvo[~fvo.index.isna()]
    fvo = fvo[~fvo.index.duplicated(keep="last")]

    joined = pd.DataFrame({"a": fvo, "n": sw.reindex(fvo.index)}).dropna()
    if filter_dates is not None:
        keep = set(filter_dates)
        joined = joined[[d.strftime("%Y-%m-%d") in keep for d in joined.index]]
    return (list(joined.index), joined["a"].astype(float).tolist(),
            joined["n"].astype(float).tolist())


def causal_normalize(arr: np.ndarray, min_obs: int = MIN_SCALE_OBS) -> np.ndarray:
    """Zero-anchored causal scaling: ``x_t / RMS(x_0..x_{t-1}) / 3``, clipped to ±1.

    No recentring — the sign stays the engine's own (see the module
    docstring) — and the scale at *t* is built from rows strictly before *t*.
    The first ``min_obs`` rows return NaN: there is no spread yet to measure a
    reading against. A 3-RMS reading maps to ±1.
    """
    x = np.asarray(arr, dtype=np.float64)
    if len(x) == 0:
        return x
    sq = pd.Series(np.where(np.isfinite(x), x * x, np.nan))
    rms = sq.expanding(min_periods=min_obs).mean().shift(1).pow(0.5).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = x / np.where(rms > 1e-12, rms, np.nan) / 3.0
    return np.clip(z, -1.0, 1.0)


def consensus_series(
    fvo_ts: pd.DataFrame | None,
    swayam_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """Full normalized-consensus history, with its own causal tiers and labels.

    Columns, indexed by date:
      ``RawA`` / ``RawN`` — Mūla FVO (σ) and Swayam Avg_Signal (±10);
      ``NormA`` / ``NormN`` — each zero-anchored-scaled to ±1;
      ``Consensus`` — their mean (negative = bullish);
      ``Strong`` / ``Moderate`` — that row's tier levels, from strictly
      earlier rows; ``Label`` — :func:`lean_label` of the row at those tiers.
    Rows where either input is still inside its scale warm-up are dropped.
    Empty when there is no overlap. Single source for the Convergence page's
    card, its plot markers and the hero's convergence gate.
    """
    cols = ["RawA", "RawN", "NormA", "NormN", "Consensus", "Strong", "Moderate", "Label"]
    dates, raw_a, raw_n = align_fvo_swayam(fvo_ts, swayam_daily)
    if not raw_a:
        return pd.DataFrame(columns=cols)
    arr_a = np.array(raw_a, dtype=np.float64)
    arr_n = np.array(raw_n, dtype=np.float64)
    norm_a = causal_normalize(arr_a)
    norm_n = causal_normalize(arr_n)
    df = pd.DataFrame({"RawA": arr_a, "RawN": arr_n, "NormA": norm_a, "NormN": norm_n},
                      index=pd.DatetimeIndex(dates))
    df["Consensus"] = (df["NormA"] + df["NormN"]) / 2.0
    df = df.dropna(subset=["Consensus"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    strong, moderate = causal_tiers(df["Consensus"].to_numpy(), CONSENSUS_PRIOR)
    df["Strong"], df["Moderate"] = strong, moderate
    df["Label"] = [lean_label(v, s, m) for v, s, m in zip(df["Consensus"], strong, moderate)]
    return df[cols]


def compute_normalized_convergence(
    fvo_ts: pd.DataFrame | None,
    swayam_daily: pd.DataFrame | None,
) -> dict | None:
    """The latest point of :func:`consensus_series`, or ``None``.

    ``{"value", "label", "strong", "moderate", "fvo_norm", "swayam_norm",
    "fvo_raw", "swayam_raw", "date"}`` — ``label`` is classified at the tiers
    built from everything before this date, i.e. exactly the marker the plot
    draws for the same point.
    """
    ser = consensus_series(fvo_ts, swayam_daily)
    if ser.empty:
        return None
    last = ser.iloc[-1]
    return {
        "value": float(last["Consensus"]),
        "label": str(last["Label"]),
        "strong": float(last["Strong"]),
        "moderate": float(last["Moderate"]),
        "fvo_norm": float(last["NormA"]),
        "swayam_norm": float(last["NormN"]),
        "fvo_raw": float(last["RawA"]),
        "swayam_raw": float(last["RawN"]),
        "date": ser.index[-1],
    }
