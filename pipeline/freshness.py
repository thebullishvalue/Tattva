"""
Tattva — data freshness: how current the target's last session is, and what to say about it.

Pure: reads the shared model matrix and returns the status chip and the queued
notices the page shell renders under the command bar.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from core.config import (ALL_TARGETS, GLOBAL_MACRO_MAP, INDEX_TARGETS_MAP, MACRO_SYMBOLS_YF,
                         SESSION_FRESH_FLOOR, STALENESS_DAYS)
from core.logger_config import console
from data.calendars import is_session, session_mask, trading_days_behind

from .prep import DATE_COL

#: Friendly column name → ticker, for resolving each column to its exchange.
_COLUMN_TICKERS = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **INDEX_TARGETS_MAP}


def freshness(df: pd.DataFrame, target: str, today: date | None = None) -> tuple[dict, list[dict]]:
    """``(status, notices)`` for ``target`` on the fetched frame ``df``.

    ``status`` is ``{"as_of", "behind", "label", "tone"}``. Freshness is counted
    in trading days on the target's own exchange calendar; "today" is the
    earlier of UTC and machine-local so a time-zone skew never overstates it.
    """
    _notices: list[dict] = []

    def _notice(kind: str, title: str, body: str) -> None:
        _notices.append({"kind": kind, "title": title, "body": body})

    _COL = {**_COLUMN_TICKERS, **ALL_TARGETS}
    _tgt_ticker = ALL_TARGETS.get(target)
    if today is None:
        today = min(datetime.now(timezone.utc).date(), datetime.now().date())
    _fresh = {"as_of": None, "behind": None, "label": "", "tone": "neutral"}
    if DATE_COL in df.columns:
        try:
            _all_dates = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
            _mask = _all_dates.notna()
            if target in df.columns:
                _mask &= pd.to_numeric(df[target], errors="coerce").notna()
            _cand = _all_dates[_mask]
            _sm = session_mask(_tgt_ticker, _cand) if len(_cand) else np.zeros(0, dtype=bool)
            _dates = _cand[_sm] if _sm.any() else _cand
            if len(_dates):
                latest_date = _dates.max().to_pydatetime().replace(tzinfo=None)
                behind = trading_days_behind(_tgt_ticker, latest_date.date(), today)
                _fresh.update(as_of=latest_date, behind=behind)
                if behind >= STALENESS_DAYS:
                    _fresh.update(label="STALE", tone="danger")
                elif behind >= 1:
                    _fresh.update(label=f"{behind}D BEHIND", tone="warning")
                else:
                    # "Current", not "live": this is end-of-day data that can be
                    # hours old; it is the newest session the source has.
                    _fresh.update(label="CURRENT", tone="success")
                ds = latest_date.strftime("%d %b %Y")
                if behind >= STALENESS_DAYS:
                    _notice(
                        "warning", "Latest data unavailable",
                        (f"Newest data is {ds} — {behind} trading days behind. The price source "
                         f"(yfinance) hasn't published more recent data, so every signal below "
                         f"reflects {ds}, not today. Use Refresh data in the rail once the "
                         f"source updates."),
                    )
                elif behind >= 1:
                    _notice(
                        "info", "Data freshness",
                        f"Signals are as of {ds} ({behind} trading day"
                        f"{'s' if behind > 1 else ''} behind — today's bar may not be published yet).",
                    )

                # Predictors refilled from a cached snapshot (data.fetcher's
                # rate-limit recovery). Read off the frame itself, so a frame
                # served from cache carries its own warning — it used to live in
                # a per-session registry that a cached frame arrived without.
                _stale_backfills = (df.attrs or {}).get("stale_backfills") or {}
                if _stale_backfills:
                    _sb_items = sorted(_stale_backfills.items(), key=lambda kv: kv[1])
                    _sb_preview = ", ".join(f"{k} (as of {v})" for k, v in _sb_items[:5])
                    _sb_more = f" +{len(_sb_items) - 5} more" if len(_sb_items) > 5 else ""
                    _notice(
                        # WARNING, not info: a snapshot-reconstructed column is
                        # reindexed onto this run's calendar and ffilled, so it is
                        # a DIFFERENT SERIES from a live pull, not a staler one —
                        # and every published date can move with it.
                        "warning", "Predictors carried from snapshot — history differs",
                        f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                        f"from a prior cached snapshot: {_sb_preview}{_sb_more}. A rebuilt column "
                        f"is not merely staler than a live one — it is a different series, so "
                        f"this run's published history may differ from a run that fetched "
                        f"cleanly. Compare the panel fingerprint before treating past values as "
                        f"final; refresh once the source is healthy for an authoritative record.",
                    )

                # Session completeness (exchange-aware): of the inputs whose market
                # was OPEN on the panel's newest date, how many actually posted a
                # fresh value rather than a carried-forward one? Markets that were
                # closed that day are legitimately carried and excluded, so a
                # foreign holiday does not trip a false "partial session".
                num = df.select_dtypes(include=[np.number])
                _panel_last = _all_dates.max()
                if len(num) >= 2 and pd.notna(_panel_last):
                    cols = list(num.columns)
                    last_r = num.iloc[-1].to_numpy(dtype=np.float64)
                    prev_r = num.iloc[-2].to_numpy(dtype=np.float64)
                    finite = np.isfinite(last_r) & np.isfinite(prev_r)
                    should_post = np.array(
                        [is_session(_COL.get(c), _panel_last.date()) for c in cols])
                    judged = finite & should_post
                    denom = int(judged.sum())
                    if denom >= 3:
                        fresh_frac = float(((last_r != prev_r) & judged).sum() / denom)
                        # A session still in progress is expected to be mostly
                        # carried forward — that is not a data problem.
                        if fresh_frac < SESSION_FRESH_FLOOR and _panel_last.date() < today:
                            _notice(
                                "warning", "Partial latest session",
                                (f"Only {fresh_frac:.0%} of the markets open on "
                                 f"{_panel_last:%d %b %Y} have posted — the rest are forward-filled "
                                 f"from the prior session, so the inputs behind the latest reading "
                                 f"are stale. Treat it as provisional; refresh once those markets post."),
                            )

                # The target's own last genuine print. Its column is forward-filled
                # across days it did not post, and continuous prices do not repeat
                # exactly, so a run of identical closes at the tail is a carried
                # value. One rule for every source — the sheet targets used to be
                # re-fetched here, on every rerun, to answer the same question.
                if target in df.columns:
                    tv = pd.to_numeric(df[target], errors="coerce").to_numpy()
                    j = len(tv) - 1
                    while j > 0 and np.isfinite(tv[j]) and tv[j] == tv[j - 1]:
                        j -= 1
                    t_last = (pd.Timestamp(_all_dates.iloc[j]).to_pydatetime()
                              if j < len(tv) - 1 and pd.notna(_all_dates.iloc[j]) else None)
                    if t_last is not None:
                        t_behind = trading_days_behind(_tgt_ticker, t_last.date(), today)
                        if t_behind >= 1:
                            if is_session(_tgt_ticker, today) and t_behind == 1:
                                _notice(
                                    "info", f"{target} price not yet updated",
                                    (f"Today's bar is carried forward from {t_last:%d %b %Y} — the "
                                     f"market is open but the last fetch did not return a new price "
                                     f"for it. Refresh data in the rail to pull the latest."),
                                )
                            else:
                                _notice(
                                    "warning", f"{target} data is lagging",
                                    (f"This target last updated {t_last:%d %b %Y} ({t_behind} trading "
                                     f"day{'s' if t_behind > 1 else ''} behind the macro universe) — "
                                     f"later rows carry that value forward, so its latest reading may "
                                     f"be stale."),
                                )
        except Exception as _fe:                      # noqa: BLE001 - never break a run to report
            console.warning(f"Freshness check skipped: {type(_fe).__name__}: {_fe}")
    return _fresh, _notices
