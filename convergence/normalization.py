"""
Tattva — Shared normalization math for the Unified Convergence Signal.
तत्त्व (Tattva) — "Principle / Essence"

Single source of truth for the math behind the Convergence Analysis cards and
the Unified Signal — Normalized Convergence plot. Both call into here, so the
card values are guaranteed to match the plot.

Pipeline:
  align(fvo_ts, swayam_daily)   →  dates, raw_a[], raw_n[]
  causal_normalize(arr)             →  causal expanding-z, /3, clipped to [-1, +1]
  classify_normalized_signal(v)     →  STRONG BUY / BUY / HOLD / SELL / STRONG SELL

TWO DISTINCT SIGNALS, TWO DISTINCT CLASSIFIERS (audit finding F1) ─────────────
This module computes the NORMALIZED CONSENSUS — a causal expanding-z average
of raw Mūla/Swayam readings. It is a DIAGNOSTIC (shown on the Convergence
tab, reconciled explicitly in the hero evidence row), and it is NOT the object
the online learner weights. ``convergence.intelligence`` learns its dimension
weights against the DIRECTIONAL COMPOSITE
(``-consensus_direction * (agreement+1)/2``, computed from the learned dim_*
weights, with consensus_direction the CONTINUOUS mean of the engines' signed
strengths — see cross_validator's orientation block) — a
differently-constructed distribution than this module's expanding-z
consensus of raw engine readings. Applying one set of learned cut-points to
both would classify a series they were never validated on.
So: ``classify_normalized_signal`` here always uses the FACTORY thresholds
(the consensus is never re-weighted). The composite thresholds instead
classify ``convergence_score`` (±100 scale, AFTER
``intelligence.apply_online_weights`` has re-weighted it) via
``classify_convergence_score`` below — that pairing is what
``app.py``'s hero card and Convergence-tab headline read as the product
signal (see ``convergence.intelligence`` module docstring for the full
calibration story).
"""

from __future__ import annotations

from typing import Iterable

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Signal classification thresholds (warm-up priors) ──────────────────────
# These are the fallback used until a signal has enough of its own history for
# analytics.adaptive to estimate the cut-point from its causal empirical
# distribution. They are symmetric by construction; the asymmetric variants a
# persisted calibration profile could once carry are gone with that profile.
#
# TWO SEPARATE FACTORY SETS — one per distribution (the F1 principle:
# thresholds are only valid for the distribution they were anchored on).
# BOTH are anchored at the pooled p75 (moderate) / p90 (strong) of their own
# |signal| distribution — the same p75/p90 convention analytics.adaptive
# re-derives per instrument. So "STRONG" means the same extremeness whether it
# came from the prior or the estimate, and on the hero card, the Unified Signal
# plot markers and the hero-history bands alike (one vocabulary).
#   • DEFAULT_THRESHOLDS   — for the NORMALIZED CONSENSUS (expanding-z avg).
#   • COMPOSITE_THRESHOLDS — for the DIRECTIONAL COMPOSITE (raw/calibrated
#     product signal).
# Study: `hero_thresholds` (research/hero_threshold_study.py) — its
# threshold-separation sweep finds no pair with a believable forward-return
# spread, so BOTH sets carry the occupancy-convention anchors PRINTED by the
# latest suite run (per its decision rule); measurements live in
# research/TUNING_COVERAGE.md and the CHANGELOG.
# hero_thresholds 2026-07-20 CONSENSUS occupancy anchor p75/p90 = ±0.284/±0.428.
_STRONG = 0.404
_MODERATE = 0.279

DEFAULT_THRESHOLDS: dict[str, float] = {
    "buy_strong":     -_STRONG,
    "buy_moderate":   -_MODERATE,
    "sell_moderate":  +_MODERATE,
    "sell_strong":    +_STRONG,
}

# Re-anchored to p75/p90 (±0.19/±0.33) from the latest `hero_thresholds` run:
# the composite distribution shifted once commodities moved to Swayam self mode
# (self-breadth changed the composite for the commodity targets in the pool), and
# the old ±0.11/±0.16 had drifted to ~p58/p69 — moderate firing on 42% of days.
# p75/p90 restores the house occupancy convention (consensus / convergence_score
# / markers all use it). Cascades to intelligence DEFAULT_THRESHOLDS + the
# conviction-model tiers (both derived from this dict).
COMPOSITE_THRESHOLDS: dict[str, float] = {
    "buy_strong":     -0.159,
    "buy_moderate":   -0.092,
    "sell_moderate":  +0.092,
    "sell_strong":    +0.159,
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
    different distribution than the weighted composite's (see
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

    This is the pairing the online learner actually weights:
    ``convergence.intelligence.online_dimension_weights`` learns the dimension
    weights by scoring the signed lean of each dimension against realised
    forward returns, and ``apply_online_weights`` then rebuilds
    ``convergence_score`` with them — so the quantity classified here IS
    ``convergence_score / 100``. Use this function (not
    ``classify_normalized_signal``) wherever the weighted product signal is
    classified — the hero card and the Convergence tab's headline (audit
    findings F1/F2). ``thresholds=None`` falls back to
    ``COMPOSITE_THRESHOLDS`` — the composite's OWN warm-up priors, NOT the
    consensus's DEFAULT_THRESHOLDS (which sit far into the composite's tail and
    would label almost every day HOLD).
    """
    return classify_normalized_signal(float(score_pm100) / 100.0,
                                      thresholds or COMPOSITE_THRESHOLDS)


def _swayam_signal_column(df: pd.DataFrame) -> str | None:
    """Return the first available Swayam average-signal column, or None."""
    for c in ("avg_unified_osc", "Avg_Signal", "avg_signal"):
        if c in df.columns:
            return c
    # Case-insensitive fallback (Avg-Signal, AVG_SIGNAL, etc.)
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
    """Align Mūla ``ConvictionRaw`` and Swayam average signal on overlapping dates.

    Args:
        fvo_ts: DataFrame with a ``ConvictionRaw`` column and either a
            ``Date`` column or a DatetimeIndex.
        swayam_daily: DataFrame indexed by date with a Swayam avg-signal column
            (``avg_unified_osc`` / ``Avg_Signal`` / ``avg_signal``).
        filter_dates: Optional iterable of date-strings (``YYYY-MM-DD``); rows
            whose date is not in this set are skipped (used by the plot to
            honour the user's lookback selection).

    Returns:
        ``(dates, fvo_raw, swayam_raw)`` — three parallel lists. Empty
        lists if either input is missing or there are no overlapping dates.
    """
    if fvo_ts is None or "ConvictionRaw" not in fvo_ts.columns:
        return [], [], []
    if swayam_daily is None or swayam_daily.empty:
        return [], [], []

    df_n = swayam_daily[~swayam_daily.index.duplicated(keep="last")].copy()
    avg_col = _swayam_signal_column(df_n)
    if avg_col is None:
        return [], [], []

    swayam_lookup: dict[str, float] = {}
    for idx in df_n.index:
        key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
        swayam_lookup[key] = float(df_n.loc[idx][avg_col])

    a_dedup = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
    date_series = a_dedup["Date"] if "Date" in a_dedup.columns else a_dedup.index

    filter_set = set(filter_dates) if filter_dates is not None else None

    dates: list = []
    raw_a: list[float] = []
    raw_n: list[float] = []
    n_failed = 0
    for d_val in date_series:
        ts_key = str(d_val.date()) if hasattr(d_val, "date") else str(pd.Timestamp(d_val).date())
        if filter_set is not None and ts_key not in filter_set:
            continue
        if ts_key not in swayam_lookup:
            continue
        try:
            raw_a.append(float(a_dedup.loc[d_val, "ConvictionRaw"]))
            raw_n.append(swayam_lookup[ts_key])
            dates.append(d_val if hasattr(d_val, "date") else pd.Timestamp(ts_key))
        except Exception:
            n_failed += 1

    # A TOTAL failure must not read as "no overlap".
    #
    # `.loc[d_val]` indexes by the frame's INDEX while `d_val` comes from the
    # Date COLUMN when one exists. Those agree in production (app.py does
    # `set_index("Date")`, leaving no column) but not for a caller that keeps a
    # RangeIndex and adds a Date column — then every lookup raises, the bare
    # except swallowed all of them, and the function returned empty lists that
    # are indistinguishable from two series with genuinely no shared dates.
    # The hero card, the TATTVA CONVICTION card and the Unified Signal plot all
    # read this, so a shape mismatch would silently blank the app's headline
    # with no diagnostic anywhere. Warn instead: an empty result is now either
    # honestly empty, or loud.
    if n_failed and not dates:
        log.warning(
            "align_fvo_swayam: every one of %d candidate rows failed lookup — "
            "fvo_ts has a 'Date' column but its index is %s, so .loc(date) "
            "cannot resolve. Pass a date-indexed frame (app.py uses "
            "set_index('Date')). Returning empty, which callers render as "
            "'no overlap'.",
            n_failed, type(a_dedup.index).__name__,
        )
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
    fvo_ts: pd.DataFrame | None,
    swayam_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    """FULL normalized-consensus history: the exact series the Unified Signal
    plot's top row draws and — since the consensus-headline product decision —
    the series whose last point IS the hero card's headline value.

    Columns: ``NormA`` (causal-z Mūla ConvictionRaw), ``NormN`` (causal-z
    Swayam Avg_Signal), ``Consensus`` (their 50/50 mean, in [-1, +1], negative
    = bullish), indexed by DatetimeIndex. Empty frame when there is no
    Mūla∩Swayam overlap. Single source of truth — the latest-point dict
    (``compute_normalized_convergence``) and the hero-history plot both
    derive from this construction, so card, plot, and hero can never drift.
    """
    dates, raw_a, raw_n = align_fvo_swayam(fvo_ts, swayam_daily)
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
    fvo_ts: pd.DataFrame | None,
    swayam_daily: pd.DataFrame | None,
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
    ser = consensus_series(fvo_ts, swayam_daily)
    if ser.empty:
        return None
    last = ser.iloc[-1]
    latest = float(last["Consensus"])
    return {
        "value": latest,
        "signal": classify_normalized_signal(latest),
        "fvo_norm": float(last["NormA"]),
        "swayam_norm": float(last["NormN"]),
        "fvo_raw": float(last["RawA"]),
        "swayam_raw": float(last["RawN"]),
    }
