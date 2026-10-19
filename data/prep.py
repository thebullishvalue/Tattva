"""
Tattva — the prepared panel: one pure function from the fetched panel to the
rows Mūla values.
तत्त्व (Tattva) — "Principle / Essence"

Every published value is a function of the rows this module hands the engine,
so the function must not depend on the date it is run. The prep it replaces
did: it kept only features with no NaN in the last MIN_DATA_POINTS sessions
and then dropped every row any feature was missing on, so the history began
at the listing date of the youngest surviving feature — 2017-11-09 as of
2023-10-16, 2019-07-24 as of 2025-10-16, 2020-06-01 (SGOV's listing) today.
Each new run started from a different place, the engine's permanent-memory
coefficients carried that start forward, and the ±0.5σ claim differed on
32-60% of sessions between the history as issued and the history as re-run.

Now the start is ``max(epoch, the target's first print)``, fixed for every run
(``core.config.ANALYSIS_EPOCH``), and nothing about a FEATURE can move it:
features are forward-filled (never back-filled) and a young one enters the
cross-section causally, once it has ``FVO_MIN_PRINTS`` prints. The one
precondition left is about the target: ``min_target_rows`` sessions of its own
prints, or :class:`PrepError`.

Invariant (tests/test_prep_asof.py): run as of T and as of T+k, the prepared
rows dated ≤ T are identical.

No Streamlit dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.instruments import instrument
from data.calendars import session_mask, target_printed


class PrepError(ValueError):
    """The target cannot support a run (too little of its own history)."""


@dataclass(frozen=True)
class PreparedPanel:
    """The rows Mūla values: target + features on the target-printed spine.

    ``data`` is indexed by session date (named after the source date column)
    with the target as its first column. ``start`` is the first session;
    ``fingerprint`` hashes the settled rows (all but the newest, which may
    still tick); ``report`` is the prep trace for the run console.
    """

    data: pd.DataFrame
    features: list[str]
    start: pd.Timestamp
    fingerprint: str
    report: dict = field(default_factory=dict)


def history_fingerprint(data: pd.DataFrame, cols, upto) -> str:
    """Content hash of the settled rows of ``data`` dated ≤ ``upto``.

    Hashes the column names, the dates and the float64 values in a fixed column
    order, so a revised settled value, a dropped column or a moved start all
    change it, while a newer row does not. ``data`` is date-indexed (a
    :class:`PreparedPanel`'s ``data``).
    """
    cols = [str(c) for c in cols]
    rows = data.loc[data.index <= pd.Timestamp(upto)]
    h = hashlib.sha256()
    h.update("\n".join(cols).encode("utf-8"))
    h.update(np.ascontiguousarray(pd.DatetimeIndex(rows.index).asi8).tobytes())
    for c in cols:
        v = (pd.to_numeric(rows[c], errors="coerce").to_numpy(dtype="float64")
             if c in rows.columns else np.full(len(rows), np.nan))
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()[:12]


def prepare_panel(
    raw: pd.DataFrame,
    target: str,
    features,
    date_col: str,
    *,
    epoch,
    min_target_rows: int,
    bars: pd.DataFrame | None = None,
) -> PreparedPanel:
    """The fetched panel → the rows the engine values, in this order:

    1. parse ``date_col`` (unparseable dates dropped, one row per date);
    2. keep rows dated ≥ ``epoch``;
    3. keep the target's own sessions that the target actually printed on —
       its exchange calendar (``session_mask``) ∩ ``target_printed`` (``bars``
       is the target's OHLCV; without it the repeat rule decides);
    4. coerce to numeric;
    5. forward-fill the features — causal, never back-filled;
    6. drop rows with no target value (the target only: a feature's NaN head
       is left for the engine's print floor to keep out);
    7. raise :class:`PrepError` with fewer than ``min_target_rows`` rows.

    ``features`` absent from ``raw`` are left out and reported. The feature
    list does not depend on the run date, so the as-of invariant holds.
    """
    ticker = instrument(target).ticker
    epoch = pd.Timestamp(epoch)
    features = list(features)
    report: dict = {"target": target, "epoch": epoch, "min_required": int(min_target_rows),
                    "feats_requested": len(features)}
    feats = [f for f in dict.fromkeys(features) if f in raw.columns and f != target]
    report["feats_absent"] = [f for f in features if f not in raw.columns]
    if target not in raw.columns:
        raise PrepError(f"'{target}' is not a column of the panel.")

    data = raw[[date_col, target, *feats]].copy()
    report["rows_initial"] = len(data)
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data = data.dropna(subset=[date_col]).sort_values(date_col, kind="mergesort")
    data = data.drop_duplicates(subset=[date_col], keep="last")
    data = data[data[date_col] >= epoch]
    report["rows_epoch"] = len(data)

    dates = data[date_col]
    in_session = session_mask(ticker, dates)
    printed = target_printed(data[target].to_numpy(), bars, dates).to_numpy(dtype=bool)
    report["sessions_dropped"] = int((~in_session).sum())
    report["target_not_printed"] = int((in_session & ~printed).sum())
    report["printed_source"] = ("target OHLCV" if bars is not None and len(bars)
                                else "repeat rule (no OHLCV)")
    data = data[in_session & printed]
    report["rows_session"] = len(data)

    for c in (target, *feats):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[feats] = data[feats].ffill()
    data = data.dropna(subset=[target])
    data = data.set_index(date_col)
    data.index = pd.DatetimeIndex(data.index, name=date_col)
    report["rows_final"] = len(data)
    report["feats_kept"] = len(feats)

    if len(data) < int(min_target_rows):
        raise PrepError(
            f"'{target}' has {len(data)} sessions of its own prints since "
            f"{epoch:%Y-%m-%d}; a run needs at least {int(min_target_rows)}.")

    start = pd.Timestamp(data.index[0])
    report["start"] = start
    settled_upto = data.index[-2] if len(data) > 1 else data.index[0] - pd.Timedelta(days=1)
    fingerprint = history_fingerprint(data, [target, *feats], settled_upto)
    return PreparedPanel(data=data, features=feats, start=start,
                         fingerprint=fingerprint, report=report)
