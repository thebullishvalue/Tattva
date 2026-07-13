"""
Tattva — Shared normalization math for the Unified Convergence Signal.
तत्त्व (Tattva) — "Principle / Essence"

Single source of truth for the math behind the Convergence Analysis cards and
the Unified Signal — Normalized Convergence plot. Both call into here, so the
card values are guaranteed to match the plot.

Pipeline:
  align(aarambh_ts, nirnay_daily)   →  dates, raw_a[], raw_n[]
  causal_normalize(arr)             →  causal expanding-z, /3, clipped to [-1, +1]
  classify_normalized_signal(v)     →  STRONG BUY / BUY / HOLD / SELL / STRONG SELL

TWO DISTINCT SIGNALS, TWO DISTINCT CLASSIFIERS (audit finding F1) ─────────────
This module computes the NORMALIZED CONSENSUS — a causal expanding-z average
of raw Aarambh/Nirnay readings. It is a DIAGNOSTIC (shown on the Convergence
tab, reconciled explicitly in the hero evidence row), not what Intelligence
Mode calibrates. ``convergence.intelligence.ConvergenceTuner`` learns its
(weights, thresholds) against the DIRECTIONAL COMPOSITE
(``-consensus_direction * (agreement+1)/2``, computed from the calibrated
dim_* weights, with consensus_direction the CONTINUOUS mean of the engines'
signed strengths — see cross_validator's orientation block) — a
differently-constructed distribution than this module's expanding-z
consensus of raw engine readings. Applying one set of learned cut-points to
both would classify a series they were never validated on.
So: ``classify_normalized_signal`` here always uses the FACTORY thresholds
(the consensus is never calibrated). The calibrated thresholds instead
classify ``convergence_score`` (the composite, ±100 scale, AFTER
``intelligence.apply_calibrated_weights`` has re-weighted it) via
``classify_convergence_score`` below — that pairing is what
``app.py``'s hero card and Convergence-tab headline read as the product
signal (see ``convergence.intelligence`` module docstring for the full
calibration story).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ── Signal classification thresholds (factory defaults) ─────────────────────
# These match the plot marker thresholds and are also the fallback when
# Intelligence Mode is not active. The calibrated thresholds in a saved
# profile may be asymmetric (`buy_strong != -sell_strong`).
#
# TWO SEPARATE FACTORY SETS — one per distribution (the F1 principle:
# thresholds are only valid for the distribution they were anchored on).
# BOTH are anchored at the pooled p75 (moderate) / p90 (strong) of their own
# |signal| distribution — the markers-study percentile convention — so a
# "STRONG" label means the same extremeness on the hero card, the Unified
# Signal plot markers, and the hero-history bands (one vocabulary).
#   • DEFAULT_THRESHOLDS   — for the NORMALIZED CONSENSUS (expanding-z avg).
#   • COMPOSITE_THRESHOLDS — for the DIRECTIONAL COMPOSITE (raw/calibrated
#     product signal).
# Study: `hero_thresholds` (research/hero_threshold_study.py) — its
# threshold-separation sweep finds no pair with a believable forward-return
# spread, so BOTH sets carry the occupancy-convention anchors PRINTED by the
# latest suite run (per its decision rule); measurements live in
# research/TUNING_COVERAGE.md and the CHANGELOG.
_STRONG = 0.41
_MODERATE = 0.26

DEFAULT_THRESHOLDS: dict[str, float] = {
    "buy_strong":     -_STRONG,
    "buy_moderate":   -_MODERATE,
    "sell_moderate":  +_MODERATE,
    "sell_strong":    +_STRONG,
}

COMPOSITE_THRESHOLDS: dict[str, float] = {
    "buy_strong":     -0.16,
    "buy_moderate":   -0.11,
    "sell_moderate":  +0.11,
    "sell_strong":    +0.16,
}


def classify_normalized_signal(
    v: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map a normalized-CONSENSUS value (in ``[-1, +1]``) to a signal label.

    Args:
        v: the normalized consensus value (see ``compute_normalized_convergence``).
        thresholds: optional override. When ``None``, uses the symmetric factory
            defaults (``DEFAULT_THRESHOLDS`` — p75/p90 occupancy-anchored).

    NOTE (audit finding F1): this classifies the normalized CONSENSUS, a
    different distribution than the one Intelligence Mode calibrates (see
    module docstring). Do not pass a saved profile's learned thresholds here —
    they were fit against ``classify_convergence_score``'s input, not this
    function's. ``compute_normalized_convergence`` (below) always calls this
    with the factory defaults for that reason.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    if v <= t["buy_strong"]:
        return "STRONG BUY"
    if v <= t["buy_moderate"]:
        return "BUY"
    if v >= t["sell_strong"]:
        return "STRONG SELL"
    if v >= t["sell_moderate"]:
        return "SELL"
    return "HOLD"


def classify_convergence_score(
    score_pm100: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map ``convergence_score`` (the directional composite, ``±100`` scale) to
    a signal label (thresholds on the ``±1`` scale — rescale by /100).

    This is the pairing Intelligence Mode actually calibrates:
    ``convergence.intelligence.ConvergenceTuner`` learns ``(weights,
    thresholds)`` by scoring ``_composite_signal`` (which — after
    ``apply_calibrated_weights`` re-weights ``convergence_df`` — IS
    ``convergence_score / 100``) against forward returns and binning it with
    these exact thresholds. Use this function (not
    ``classify_normalized_signal``) wherever the raw or calibrated product
    signal is classified — the hero card and the Convergence tab's headline
    (audit findings F1/F2). ``thresholds=None`` falls back to
    ``COMPOSITE_THRESHOLDS`` — the composite's OWN data-anchored factory
    cut-points, NOT the consensus's DEFAULT_THRESHOLDS (which sit far into the
    composite's tail and would label almost every day HOLD).
    """
    return classify_normalized_signal(float(score_pm100) / 100.0,
                                      thresholds or COMPOSITE_THRESHOLDS)


def _nirnay_signal_column(df: pd.DataFrame) -> str | None:
    """Return the first available Nirnay average-signal column, or None."""
    for c in ("avg_unified_osc", "Avg_Signal", "avg_signal"):
        if c in df.columns:
            return c
    # Case-insensitive fallback (Avg-Signal, AVG_SIGNAL, etc.)
    for c in df.columns:
        cl = c.lower().replace("-", "_")
        if cl in ("avg_signal", "avg_unified_osc"):
            return c
    return None


def align_aarambh_nirnay(
    aarambh_ts: pd.DataFrame | None,
    nirnay_daily: pd.DataFrame | None,
    filter_dates: Iterable[str] | None = None,
) -> tuple[list, list[float], list[float]]:
    """Align Aarambh ``ConvictionRaw`` and Nirnay average signal on overlapping dates.

    Args:
        aarambh_ts: DataFrame with a ``ConvictionRaw`` column and either a
            ``Date`` column or a DatetimeIndex.
        nirnay_daily: DataFrame indexed by date with a Nirnay avg-signal column
            (``avg_unified_osc`` / ``Avg_Signal`` / ``avg_signal``).
        filter_dates: Optional iterable of date-strings (``YYYY-MM-DD``); rows
            whose date is not in this set are skipped (used by the plot to
            honour the user's lookback selection).

    Returns:
        ``(dates, aarambh_raw, nirnay_raw)`` — three parallel lists. Empty
        lists if either input is missing or there are no overlapping dates.
    """
    if aarambh_ts is None or "ConvictionRaw" not in aarambh_ts.columns:
        return [], [], []
    if nirnay_daily is None or nirnay_daily.empty:
        return [], [], []

    df_n = nirnay_daily[~nirnay_daily.index.duplicated(keep="last")].copy()
    avg_col = _nirnay_signal_column(df_n)
    if avg_col is None:
        return [], [], []

    nirnay_lookup: dict[str, float] = {}
    for idx in df_n.index:
        key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
        nirnay_lookup[key] = float(df_n.loc[idx][avg_col])

    a_dedup = aarambh_ts[~aarambh_ts.index.duplicated(keep="last")]
    date_series = a_dedup["Date"] if "Date" in a_dedup.columns else a_dedup.index

    filter_set = set(filter_dates) if filter_dates is not None else None

    dates: list = []
    raw_a: list[float] = []
    raw_n: list[float] = []
    for d_val in date_series:
        ts_key = str(d_val.date()) if hasattr(d_val, "date") else str(pd.Timestamp(d_val).date())
        if filter_set is not None and ts_key not in filter_set:
            continue
        if ts_key not in nirnay_lookup:
            continue
        try:
            raw_a.append(float(a_dedup.loc[d_val, "ConvictionRaw"]))
            raw_n.append(nirnay_lookup[ts_key])
            dates.append(d_val if hasattr(d_val, "date") else pd.Timestamp(ts_key))
        except Exception:
            pass
    return dates, raw_a, raw_n


def causal_normalize(arr: np.ndarray) -> np.ndarray:
    """Causal expanding-window z-score, ``/3`` and clipped to ``[-1, +1]``.

    Each point is normalised using only the history available up to that
    date (an expanding, not rolling, window) — no future data leakage.
    SINGLE SOURCE OF TRUTH for this transform: it previously had two
    hand-written copies (here and in ``ui/tabs/tab_convergence.py``'s
    per-config cache-building block) that had to be kept in exact sync by
    inspection for the tab's plot to match this module's card values (audit
    finding F16). Both now call this helper.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return arr
    s = pd.Series(arr)
    mu = s.expanding().mean().to_numpy()
    sigma = s.expanding().std().clip(lower=1e-10).fillna(1.0).to_numpy()
    return np.clip((arr - mu) / sigma / 3.0, -1.0, 1.0)


def consensus_series(
    aarambh_ts: pd.DataFrame | None,
    nirnay_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """FULL normalized-consensus history: the exact series the Unified Signal
    plot's top row draws and — since the consensus-headline product decision —
    the series whose last point IS the hero card's headline value.

    Columns: ``NormA`` (causal-z Aarambh ConvictionRaw), ``NormN`` (causal-z
    Nirnay Avg_Signal), ``Consensus`` (their 50/50 mean, in [-1, +1], negative
    = bullish), indexed by DatetimeIndex. Empty frame when there is no
    Aarambh∩Nirnay overlap. Single source of truth — the latest-point dict
    (``compute_normalized_convergence``) and the hero-history plot both
    derive from this construction, so card, plot, and hero can never drift.
    """
    dates, raw_a, raw_n = align_aarambh_nirnay(aarambh_ts, nirnay_daily)
    if not raw_a:
        return pd.DataFrame(columns=["NormA", "NormN", "Consensus", "RawA", "RawN"])
    arr_a = np.array(raw_a, dtype=np.float64)
    arr_n = np.array(raw_n, dtype=np.float64)
    norm_a = causal_normalize(arr_a)
    norm_n = causal_normalize(arr_n)
    idx = pd.to_datetime(pd.Index(dates), errors="coerce")
    return pd.DataFrame(
        {"NormA": norm_a, "NormN": norm_n, "Consensus": (norm_a + norm_n) / 2.0,
         "RawA": arr_a, "RawN": arr_n},
        index=idx,
    )


def compute_normalized_convergence(
    aarambh_ts: pd.DataFrame | None,
    nirnay_daily: pd.DataFrame | None,
) -> dict | None:
    """Latest normalized-CONSENSUS value + per-system contributions.

    The last point of :func:`consensus_series` — what the Unified Signal
    plot's top row displays at its right edge, the TATTVA CONVICTION card
    shows, and (per the consensus-headline product decision) the hero card
    headlines. Returns ``None`` if alignment yields no rows.

    Its ``signal`` label always uses the symmetric FACTORY thresholds (no
    ``thresholds`` parameter: a previous revision accepted the Intelligence
    Mode calibrated thresholds here, applying cut-points learned against a
    differently-shaped distribution — audit finding F1). The calibrated
    composite is classified separately via ``classify_convergence_score``.
    """
    ser = consensus_series(aarambh_ts, nirnay_daily)
    if ser.empty:
        return None
    last = ser.iloc[-1]
    latest = float(last["Consensus"])
    return {
        "value": latest,
        "signal": classify_normalized_signal(latest),
        "aarambh_norm": float(last["NormA"]),
        "nirnay_norm": float(last["NormN"]),
        "aarambh_raw": float(last["RawA"]),
        "nirnay_raw": float(last["RawN"]),
    }
