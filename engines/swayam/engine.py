"""
Tattva — SWAYAM engine: the target read by itself, through the Pragati indicator.
तत्त्व (Tattva) — "Principle / Essence"

SWAYAM (स्वयम् — "self") asks what the target's OWN price and volume say, independently of
Mūla's cross-sectional valuation. It is the Pragati indicator (pragati.pine v9.3, the engine
Sanket screens with and Pragyam sizes from) run on the target:

    CONVICTION   who controls — the participation-weighted share of each bar's effort that
                 became displacement (Nishchaya v3), read on a LADDER of timeframes: the daily
                 chart and the weekly rung reconstructed from it (Ladder up; see
                 engines/pragati/core.py for why Tattva never reads Ladder down)
    VALUE        where price stands — Samanvaya: an RV leg hedged against ~20 macro drivers
                 plus a price-only breadth leg, judged against θ
    GRID         the two tapes meet in a 3 × 3 state (sellers / faint / buyers × cheap / fair /
                 rich), rows moved only by a confirmed push of conviction's own histogram
    EVENTS       ▲ CAPITULATION TURN — sellers in control of a cheap price with value
                 momentum already reverting, outside conviction's quiet regime;
                 ▼ DISTRIBUTION — sellers taking control of a rich price

WHY THIS REPLACED THE MSF / MMR VIEW BANK (measured, research/swayam_v3_study.md): on Tattva's
own 25 targets at its 10-day horizon the old bank's oscillator anti-predicted in two of three
periods (IC −0.081 / −0.044 / +0.030, 2019-21 / 2022-23 / 2024-26); its 15 views were 15
readings of one series (N_eff ≈ 8-10) whose skill weights did not persist. The conviction tape
read as Tattva reads Swayam (sellers in control = oversold = bullish) scored +0.284 / +0.041 /
+0.027, and the ▲ was followed by +0.08 / +0.19 / +0.25σ excess returns — +0.25 / +0.14σ on
commodities and FX and +0.39 / +0.64σ on indices in the last two periods. Crypto is the
stated exception (≈ 0), as it was on 380 instruments in Sanket's audit.

THE CONTRACT. Convergence, Intelligence, Precedent and the hero read Swayam through a fixed
per-date schema, kept exactly:

    Oversold_Pct / Overbought_Pct   share of the conviction ladder's RUNGS (daily, weekly) with
                                    sellers / buyers past the inner zone (±30) — the views
    Avg_Signal                      the conviction tape on Swayam's ±10 scale (negative =
                                    sellers = oversold, the side Tattva reads as bullish)
    Regime_Bull_Pct / _Bear_Pct /   the grid's measured lean: capitulation and washout BULL,
    Regime_Neutral                  distribution BEAR, every other cell NEUTRAL
    Total_Analyzed                  rungs reporting (1-2)
    Buy_Signals / Sell_Signals      a ▲ / ▼ on this bar

Every value is a function of bars that exist at its own date: the ladder is built from daily
bars, the value model from closed anchor periods, the grid's rows from the histogram's own
past. A row is published only once the conviction tape has calibrated; before that Swayam is
silent rather than neutral-by-default. No Streamlit dependency.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engines.pragati import core as pg
from engines.pragati import grid as cg
from engines.pragati import samanvaya as sv

log = logging.getLogger(__name__)

#: The grid's measured lean per cell (0-8; 9 = unread): capitulation and washout lean bullish,
#: distribution bearish, the rest carry no measured edge (Sanket v9 audit; Tattva study).
CELL_LEAN = (1.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RUNG_NAMES = ("Daily", "Weekly")
CONTRACT_COLUMNS = ("Oversold_Pct", "Overbought_Pct", "Avg_Signal", "Regime_Bull_Pct",
                    "Regime_Bear_Pct", "Regime_Neutral", "Total_Analyzed", "Buy_Signals",
                    "Sell_Signals")


def _lower_bars(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Title-case OHLCV (Tattva's fetcher) → the lower-case bars the Pragati core reads."""
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    lo = pd.DataFrame({c.lower(): pd.to_numeric(df[c], errors="coerce")
                       for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns},
                      index=df.index)
    if "volume" not in lo:
        lo["volume"] = np.nan
    return lo.dropna(subset=["open", "high", "low", "close"])


class SwayamEngine:
    """The Pragati indicator on the target: conviction ladder × value → grid → events."""

    def __init__(self, params: pg.Params | None = None) -> None:
        self.params = params or pg.DEFAULT
        self.frame = pd.DataFrame()          # the full per-bar read (tapes, grid, events)
        self.daily = pd.DataFrame()          # the Convergence contract, published rows only
        self.rungs = pd.DataFrame()          # each ladder rung on ±100 — Swayam's views
        self.drivers_used: list[str] = []

    # ── fit ──────────────────────────────────────────────────────────────────────
    def fit(self, ohlcv: pd.DataFrame, drivers: pd.DataFrame | None, symbol: str) -> "SwayamEngine":
        p = self.params
        lo = _lower_bars(ohlcv)
        if len(lo) < p.norm + p.length + p.signal:
            log.warning("Swayam: %d bars — too short to calibrate (needs %d)", len(lo),
                        p.norm + p.length + p.signal)
            return self
        drv = None
        if drivers is not None and not drivers.empty:
            try:
                drv = sv.prepare_drivers(drivers, "D")
            except Exception as e:                       # noqa: BLE001 — value runs unhedged
                log.warning("Swayam: drivers unusable (%s) — value leg runs unhedged", e)
        val = sv.compute_value(lo, drv, symbol, chart="D")
        out = pg.compute(lo, val, p, chart="D")
        ch = pg.chart_conviction(lo, p)
        cv_ready = np.cumsum(ch["sd_ok"].to_numpy(bool)) > p.norm + p.smooth + p.signal
        grid = cg.classify(out, ch["osc"], ch["raw_sd"], cv_ready, p)
        out = pg.signals(out, grid, p)
        extra = val[["rv_z", "breadth_z", "legs_split", "hedge", "drivers", "basket_warm"]]
        self.frame = pd.concat([lo[["close"]], out, grid, extra], axis=1)
        self.drivers_used = list(drv.columns) if drv is not None else []

        # the views: each rung of the conviction ladder on the ±100 scale
        sd_ok = pd.Series(ch["sd_ok"].to_numpy(bool), index=lo.index)
        d_rung = (100.0 * np.tanh(ch["z"])).where(sd_ok)
        w_z, _ = pg.parent_rung(lo, "W", p.parent_norm, p)
        w_rung = (100.0 * np.tanh(w_z)).where(sd_ok)
        self.rungs = pd.DataFrame({"Daily": d_rung, "Weekly": w_rung}, index=lo.index)
        self.daily = self._contract()
        return self

    # ── the Convergence contract ──────────────────────────────────────────────────
    def _contract(self) -> pd.DataFrame:
        f, R, z1 = self.frame, self.rungs, float(self.params.z1)
        ready = f["c_ready"].fillna(False).astype(bool)
        n = R.notna().sum(axis=1)
        denom = n.where(n > 0)
        os_pct = 100.0 * (R <= -z1).sum(axis=1) / denom
        ob_pct = 100.0 * (R >= z1).sum(axis=1) / denom
        cell = f["cvg_cell"].fillna(9).astype(int).clip(0, 9).to_numpy()
        lean = np.asarray(CELL_LEAN)[cell]
        bull = np.where(lean > 0, 100.0, 0.0)
        bear = np.where(lean < 0, 100.0, 0.0)
        d = pd.DataFrame({
            "Oversold_Pct": os_pct, "Overbought_Pct": ob_pct,
            "Avg_Signal": pd.to_numeric(f["c_tape"], errors="coerce") / 10.0,
            "Regime_Bull_Pct": bull, "Regime_Bear_Pct": bear,
            "Regime_Neutral": 100.0 - bull - bear,
            "Total_Analyzed": n.astype(int),
            "Buy_Signals": f["turn_buy"].fillna(False).astype(bool).astype(int),
            "Sell_Signals": f["turn_sell"].fillna(False).astype(bool).astype(int),
        }, index=f.index)
        return d[ready & (n > 0)]

    # ── the current read, for the hero, the tab and the run log ──────────────────
    def current(self) -> dict:
        if self.frame.empty or self.daily.empty:
            return {}
        t = self.daily.index[-1]
        f = self.frame.loc[t]
        cell = int(f["cvg_cell"]) if pd.notna(f["cvg_cell"]) else cg.UNREAD
        units = float(f["cvg_units"]) if pd.notna(f.get("cvg_units")) else float("nan")
        last_up = self.frame.index[self.frame["turn_buy"].fillna(False).astype(bool)]
        last_dn = self.frame.index[self.frame["turn_sell"].fillna(False).astype(bool)]
        return {
            "date": t,
            "conviction": float(f["c_tape"]),
            "chart_conviction": float(f["conv"]) if pd.notna(f["conv"]) else float("nan"),
            "value": float(f["v_tape"]) if pd.notna(f["v_tape"]) else float("nan"),
            "trace": float(f["trace"]) if pd.notna(f["trace"]) else float("nan"),
            "push_tier": str(f.get("push_tier", "")),
            "quiet": bool(f.get("cvg_quiet", False)),
            "cell": cell,
            "state": cg.NAMES[cell],
            "family": cg.FAMILY[cell],
            "units": units,
            "lean": CELL_LEAN[cell],
            "bars_in_state": (int(self.frame.index.get_loc(t) - int(f["cvg_since"]))
                              if pd.notna(f.get("cvg_since")) else None),
            "last_capitulation": last_up[-1] if len(last_up) else None,
            "last_distribution": last_dn[-1] if len(last_dn) else None,
            "rungs": {k: (float(v) if pd.notna(v) else None) for k, v in self.rungs.loc[t].items()},
            "drivers": len(self.drivers_used),
        }


__all__ = ["CELL_LEAN", "CONTRACT_COLUMNS", "RUNG_NAMES", "SwayamEngine"]
