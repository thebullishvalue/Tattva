"""
Tattva — data preparation: from the shared model matrix to one target's panel.

Pure pandas. ``prepare`` turns the fetched frame into what the engines read
(the target's price on its own session calendar and the explanatory panel
beside it) or raises :class:`PrepError` saying which stage failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import (ALL_TARGETS, EPOCH, MIN_DATA_POINTS, fetch_window, RAW_YIELD_PREDICTORS,
                         STOCK_TARGET_MARKETS, InstrumentConfig, get_instrument_config)
from core.logger_config import console
from data.calendars import resolve_exchange, session_mask

__all__ = ["EPOCH", "fetch_window", "PrepError", "PreparedPanel", "prepare", "config_for", "log_report", "DATE_COL"]

DATE_COL = "DATE"

#: A target must have changed price on at least this share of its last
#: LIVENESS_WINDOW sessions to be valued.
LIVENESS_WINDOW = 252
LIVENESS_FLOOR = 0.50


class PrepError(Exception):
    """Data preparation could not produce a usable panel.

    ``user_message`` is what the page shows; ``report`` is the stage-by-stage
    row accounting gathered up to the failure.
    """

    def __init__(self, title: str, detail: str, user_message: str, report: dict | None = None):
        super().__init__(f"{title}: {detail}")
        self.title = title
        self.detail = detail
        self.user_message = user_message
        self.report = report or {}


@dataclass
class PreparedPanel:
    target: str
    calendar: pd.DatetimeIndex
    price: pd.Series                 # target price level, indexed by calendar
    panel: pd.DataFrame              # explanatory price levels, same index
    features: list[str]
    config: InstrumentConfig
    report: dict = field(default_factory=dict)
    printed: pd.DataFrame | None = None   # true print mask, when the fetch carried one

    @property
    def date_range(self) -> str:
        if len(self.calendar):
            return f"{self.calendar[0].date()}_{self.calendar[-1].date()}"
        return "n0"

    @property
    def cache_key(self) -> str:
        return (f"fvo{self.config.forecast_horizon}|{self.target}|"
                f"{'|'.join(sorted(self.features))}|{self.date_range}")


def config_for(target: str) -> InstrumentConfig:
    try:
        return get_instrument_config(target)
    except KeyError:
        return InstrumentConfig()


def prepare(df: pd.DataFrame, target: str) -> PreparedPanel:
    """Build ``target``'s panel from the shared model matrix ``df``."""
    cfg = config_for(target)
    ticker = ALL_TARGETS.get(target)
    exch = resolve_exchange(ticker) or "weekday"
    report: dict = {"target": target, "ticker": ticker, "exchange": exch,
                    "min_required": MIN_DATA_POINTS, "fwd_h": cfg.forecast_horizon,
                    "burn_in": int(cfg.fvo_burn_in),
                    "fingerprint": (df.attrs or {}).get("panel_fingerprint")}

    if target not in df.columns:
        raise PrepError(
            "Target column missing", f"'{target}' not in fetched dataset — its source fetch failed.",
            f"'{target}' data is currently unavailable (its source fetch failed). "
            f"Pick another target, or re-run once the source is back online.", report)
    if DATE_COL not in df.columns:
        raise PrepError("No date column", "the dataset has no DATE column.",
                        "The dataset has no DATE column.", report)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {target, *cfg.excluded_predictors}
    features = [c for c in numeric if c not in excluded and c not in STOCK_TARGET_MARKETS]
    report["feats_requested"] = len(features)

    data = df[[target] + features + [DATE_COL]].copy()
    report["rows_initial"] = len(data)
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce", dayfirst=True)
    data = data.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # The fetched matrix is a Mon–Fri spine; a row on the target's own market
    # holiday carries its last close forward. Keep the target's real sessions.
    rows_pre = len(data)
    if len(data):
        mask = session_mask(ticker, data[DATE_COL])
        if mask.any():
            data = data[mask].reset_index(drop=True)
    report["rows_session"] = len(data)
    report["sessions_dropped"] = max(0, rows_pre - len(data))

    data[[target] + features] = data[[target] + features].ffill()
    data = data.dropna(subset=[target]).reset_index(drop=True)
    # The panel is ragged on purpose. A predictor that lists later keeps its
    # leading NaNs and joins the cross-section when its own print count allows
    # (the engine admits per instrument), so no predictor's listing date can
    # shorten the target's history. Only one that could never be admitted is
    # dropped here.
    before = list(features)
    features = [f for f in features if int(data[f].notna().sum()) >= cfg.fvo_min_prints]
    report["feats_dropped"] = [f for f in before if f not in features]
    report["rows_final"] = len(data)
    report["feats_kept"] = len(features)

    # A target whose own price has stopped updating cannot be valued: every
    # row is a carried quote.
    tail = data[target].tail(LIVENESS_WINDOW)
    live = float((tail.diff().abs() > 0).mean()) if len(tail) > 1 else 0.0
    report["live_share"] = live
    if live < LIVENESS_FLOOR:
        raise PrepError(
            "Target feed not live",
            f"{target}: {tail.nunique()} distinct prices in the last {len(tail)} sessions "
            f"({live:.0%} of sessions changed).",
            f"'{target}' price feed is not live: {tail.nunique()} distinct prices in the last "
            f"{len(tail)} sessions. It cannot be valued.", report)

    if len(data) < MIN_DATA_POINTS:
        raise PrepError(
            "Insufficient data spine",
            f"{target}: {len(data)} usable {exch} sessions after cleaning, need ≥{MIN_DATA_POINTS}. "
            f"Fetched {report['rows_initial']} rows → {report['rows_session']} after session spine → "
            f"{len(data)} with a target price.",
            f"Need {MIN_DATA_POINTS}+ data points — '{target}' yielded only {len(data)} usable "
            f"{exch} trading sessions after cleaning. Try a target with more history.", report)
    if not features:
        raise PrepError("No valid features", f"{target}: every predictor was dropped for short history.",
                        "No valid features found after data cleaning.", report)
    if (data[target] <= 0).any():
        raise PrepError(
            "Non-positive target", f"{target}: contains values ≤ 0; the engine models log price.",
            f"'{target}' has non-positive values — the valuation engine models log price and "
            f"needs a strictly positive series.", report)

    # Raw yield LEVELS are rate series, not prices: they print at or below zero
    # and are not instruments the target can be valued against. The Treasury
    # ETF complex already carries the curve.
    yields = [f for f in features if f in RAW_YIELD_PREDICTORS]
    if yields:
        features = [f for f in features if f not in RAW_YIELD_PREDICTORS]
        report["yield_cols_dropped"] = len(yields)
        report["feats_kept"] = len(features)
    if not features:
        raise PrepError("Empty valuation panel", f"{target}: no price-level predictors survived.",
                        "No price-level predictors remain — the valuation engine needs a "
                        "cross-section of prices.", report)

    cal = pd.DatetimeIndex(pd.to_datetime(data[DATE_COL].values))
    price = pd.Series(data[target].to_numpy(dtype=np.float64), index=cal, name=target)
    panel = data[features].astype(float)
    panel.index = cal
    return PreparedPanel(target=target, calendar=cal, price=price, panel=panel,
                         features=features, config=cfg, report=report,
                         printed=_print_mask(df, cal, features))


def _print_mask(df: pd.DataFrame, cal: pd.DatetimeIndex, features: list[str]) -> pd.DataFrame | None:
    """The fetch's own print mask on this panel, or None to let the engine infer it."""
    from data.fetcher import unpack_mask
    dates = pd.DatetimeIndex(pd.to_datetime(df[DATE_COL], errors="coerce"))
    mask = unpack_mask((df.attrs or {}).get("printed"), dates)
    if mask is None or not set(features).issubset(mask.columns):
        return None
    mask = mask[~mask.index.isna()]
    mask = mask[~mask.index.duplicated(keep="last")]
    return mask.reindex(index=cal, columns=features, fill_value=False)


def log_report(report: dict, stage: str = "complete") -> None:
    """Print the data-preparation trace to the run console."""
    dropped = report.get("feats_dropped", [])
    console.section("DATA PREPARATION")
    console.item("Target", f"{report.get('target')}  (ticker={report.get('ticker') or 'n/a'}, "
                           f"exch={report.get('exchange')})")
    console.item("Scoring Horizon", f"{report.get('fwd_h', '?')}d")
    console.item("Rows · fetched", report.get("rows_initial", "?"))
    console.item("Rows · after session spine",
                 f"{report.get('rows_session', '?')}  ({report.get('sessions_dropped', '?')} non-session rows removed)")
    console.item("Instruments · requested", report.get("feats_requested", "?"))
    console.item("Instruments · dropped (too few prints to ever join)",
                 f"{len(dropped)}" + (f" → {', '.join(dropped[:6])}{'…' if len(dropped) > 6 else ''}"
                                      if dropped else ""))
    if report.get("yield_cols_dropped"):
        console.item("Instruments · dropped (raw yield levels)", report["yield_cols_dropped"])
    console.item("Instruments · in valuation panel", report.get("feats_kept", "?"))
    if report.get("fingerprint"):
        console.item("Panel fingerprint", report["fingerprint"])
    console.item("Rows · final spine", report.get("rows_final", "?"))
    if stage == "complete":
        console.checkpoint(f"Data spine ≥ {report.get('min_required')}", "OK")
