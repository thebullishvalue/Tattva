"""
TATTVA — Pragati · प्रगति, the conviction × value oscillator (pragati.pine v9.3)

Ported verbatim from Sanket's pragati.py (the bar-for-bar port of the Pine) so both products
read the same indicator. ONE deliberate difference: Tattva reads the conviction ladder UP
(W · D, the parent rung reconstructed on the chart). Tattva guarantees that a published value
never changes on a later run, and yfinance's intraday history rolls off (1h reaches ~730 days
back), so a Ladder-down reading of a past bar would be rewritten once its intrabars expire.
Ladder up is built from daily bars that exist forever. The Ladder-down code is kept only so
the two copies stay diffable; it refuses to run here.
══════════════════════════════════════════════════════════════════════════════

"Is the push paid for — and at what price?"

The screener's engine: a Python port of pragati.pine, the same indicator
Pragyam's Conviction-Value Grid reads, carried the rest of the way — the trace,
its histogram and the signal set — because a screener's job is the events.

ONE TRACE, ITS PUSH, AND ITS TWO INGREDIENTS ACROSS HORIZONS
    conviction tape   the flow ingredient across the ladder (who controls)
    trace             conviction × value on this chart — how far the move is
                      STRETCHED, in one-sided effort and in price vs fair value
    histogram         the trace's own push — is the stretch building or releasing
    value tape        the value ingredient across the ladder (rich or cheap)

WHAT IT MEASURES (the Pine's own header)
    conviction      c = (C − C[1]) / TR                   bounded −1 … +1
    participation   w = min(V / EMA(V), cap)              relative TR if no volume
    raw share       AGREEMENT  100 · Σ(c·w) / Σ(|c|·w)    (default; the measured one)
                    EFFORT     100 · Σ(c·w) / Σw
    conviction      100 · tanh(raw / 3σ), EMA(3)          σ over the normalization window
    value           Samanvaya's unified z (samanvaya.py)
    trace           100 · softbound(0.5 · (w_c·z_c + w_v·z_v) / √(w_c² + w_v² + 2·w_c·w_v·ρ))
    histogram       trace − EMA(trace, 9)

ONE SIGNAL SET, READ FROM THE GRID (signals(), once cvgrid has read the bar)
    ▲    CAPITULATION   the first bar the Conviction-Value Grid stands in DOWN · cheap
                        (sellers in control across the ladder, value cheap past θ) with
                        value momentum REVERTING — the fast end has turned toward fair.
    ▼    DISTRIBUTION   the first bar sellers hold control of a price rich past θ.
    ◆    RESUME         a trend resuming (off by default). The histogram dipped to the
                        wrong side inside 6 bars and now crosses k·σ; chart conviction
                        and the conviction tape past the inner zone on the ◆'s side;
                        the value tape short of θ; effort not absorbed on the bar.
Cooldown 10 bars per direction (▲ and long ◆ share one clock). A ▲▼ takes precedence
over a ◆ on the same bar. A ▲▼ is a DECLARATION that stands until the opposite one;
it has no exit.

DIVERGENCE IS EVIDENCE, NOT A SIGNAL, found on conviction's own pivots — where it was
measured — never on the drawn trace: zone-gated, at a price value called stretched.

THE LADDER ON A DAILY-BAR FEED — LADDER DOWN (since v9.1), from yfinance's intraday history
    chart   conviction ladder                                 value ladder
    D       1m·3m·5m·15m·30m·1h·4h inside · D  (Ladder down)   W · D  (Ladder up)
            ↺ W · D (Ladder up) on bars older than the intraday history
    W       1h·4h·D inside · W  (Ladder down)                 M · W  (Ladder up)
The Pine's default conviction ladder (v9.1) is Ladder DOWN. intraday.py supplies the lower frames
yfinance carries — 1m for 7 days, 5m / 15m / 30m for 60, 1h for ~730; 3m and 4h built from them —
and each joins where its own history has calibrated, exactly as the Pine's rungs join when their
intrabars reach back far enough. A daily bar older than every intraday frame has nothing inside it
to read; there the tape reads Ladder up (W · D, the parent reconstructed on the chart) and is
marked ↺ — the Pine's FALLBACK rule, extended from "no frames on this chart" to "no intrabars on
this bar". The daily tape switches to Ladder down once a lower rung has held for the Pine's own
warm-up (ladReady: norm + smooth bars). On Weekly the daily bars inside each week are always
there, so Weekly is Ladder down throughout, with 1h and 4h joining where they exist.

    · The daily chart's W conviction rung normalises over 52 weeks, not 200: at
      200 it would need four years of weekly history (Pragyam's adaptation).
    · The quiet-regime test ranks rawSd over up to 4 × norm bars as the Pine
      does, but over whatever history exists once there are norm of them — the
      screener's panel cannot supply 800.

REPAINT POSTURE. Every column is computed from bars that exist: the parent rung
is the parent's SETTLED state plus the forming parent bar aggregated from the
chart, the child rung only the daily bars inside the week. Signals are
committed on the bar's close, as the Pine commits them on barstate.isconfirmed.
A live, unclosed session is provisional until it closes.

NAMING. Siddhi → Nishchaya → Dhṛti → Pragati. Sanket carried Siddhi (v7); it
now carries Pragati, the line's current head.

Author: @thebullishvalue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _swv

from engines.pragati import samanvaya as sv


@dataclass(frozen=True)
class Params:
    """The Pine's inputs, at its defaults. None is fitted — "Do not tune this to a backtest."

    The indicator's own finding across 900 randomised parameter sets: fitted
    edge and unseen edge correlate at about zero; lookback ≥ 20 and inner zone
    ≤ 40 were the only preferences that held.
    """
    # 1 · conviction engine
    length: int = 20            # inpLen   — bars of effort the reading accounts for
    smooth: int = 3             # inpSm    — final EMA on conviction and its tape
    norm: int = 200             # inpNorm  — sample behind the scaling σ
    participation: str = "Auto"  # Auto / Volume / True range / Off
    denominator: str = "Agreement"  # Agreement / Effort
    vol_n: int = 20             # inpVN    — participation baseline
    cap: float = 3.0            # inpCap   — ceiling on the participation weight
    z1: float = 30.0            # inner zone
    z2: float = 60.0            # outer zone
    # 3 · trace
    mix: float = 0.5            # value's weight in the trace (0.5 = conviction × value)
    signal: int = 9             # inpSig   — the trace's signal EMA
    # 4 · signals
    turn: bool = True           # ▲▼ (inpState) — read from the grid: ▲ the capitulation
                                #   turn, ▼ distribution (signals(), once the grid is read)
    # v9.1: the conviction ladder reads DOWN — the frames inside the chart bar (1m … 4h on a
    # daily chart, from intraday.py); where a bar has no intraday history the tape falls
    # back to Ladder up (W · D) and is marked ↺. "up" reads W · D everywhere (v8 / v9).
    ladder: str = "up"            # Tattva: always UP (W · D) — see the module note
    resume: bool = False        # ◆ RESUME — OFF by default in v8: negative in both eras
                                #   of the audit outside crypto (studies/pine_audit.md)
    k: float = 0.5              # impulse threshold, σ of the histogram
    pull: int = 6               # pullback window
    effort: bool = True         # ◆ only: the push is producing result, not absorbed
    cool: int = 10              # cooldown per direction
    # 5 · divergence evidence
    pl: int = 5                 # pivot left
    pr: int = 5                 # pivot right
    gap_min: int = 5
    gap_max: int = 60
    zone_gate: bool = True
    # 7 · display — the quiet regime also withholds a push from the grid
    quiet: bool = True
    # ADAPTED — the daily chart's W conviction rung (see module docstring)
    parent_norm: int = 52

    @property
    def theta(self) -> float:
        """θ on the trace's ±100 scale — Samanvaya's 1.5σ, ±42.9."""
        return sv.THETA_OSC


DEFAULT = Params()
RAIL_OUT = 70.0          # the trace's outer rail — Samanvaya's ±70
EFF_ABSORBED = 20.0      # effort → result in the bottom fifth of its history
QUIET_PCT = 20.0         # rawSd in the bottom fifth of its history
SLOPE_DEADBAND = 0.10    # a trace move inside 0.1σ of its bar-to-bar change is flat
RECENT = 20              # bars an absorption or a divergence stays "recent" evidence (display)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMITIVES — Pine's ta.* on pandas / numpy, numerically faithful
#  ta.ema seeds on its first value with alpha = 2/(n+1); ta.stdev is the
#  POPULATION deviation; ta.tr(true) includes the gap.
# ══════════════════════════════════════════════════════════════════════════════
def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=max(int(n), 1), adjust=False).mean()


def _tanh(x):
    """The Pine's f_tanh: clamped to ±10 before exponentiating."""
    return np.tanh(np.clip(x, -10.0, 10.0))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high, pc], axis=1).max(axis=1) - pd.concat([low, pc], axis=1).min(axis=1)
    return tr.where(pc.notna(), high - low)


def _weights(tr: pd.Series, volume: pd.Series, p: Params):
    """Section 1 · participation. Returns (w, vol_last, vol_avg, tr_avg).

    A hollow bar — holiday, half session, no print — carries the last good
    volume into the baseline rather than feeding it NaN; measured in the Pine,
    without the carry volume weighting was live on only 73% of bars.
    """
    vol_ok = volume.notna() & (volume > 0)
    vol_last = volume.where(vol_ok).ffill()
    vol_avg = _ema(vol_last, p.vol_n)
    tr_avg = _ema(tr, p.vol_n)
    w_vol = (volume / vol_avg).where(vol_ok & vol_avg.notna() & (vol_avg > 1e-12))
    w_rng = (tr / tr_avg).where(tr_avg.notna() & (tr_avg > 1e-12), 1.0)
    mode = p.participation
    if mode == "Off":
        w_raw = pd.Series(1.0, index=tr.index)
    elif mode == "True range":
        w_raw = w_rng
    elif mode == "Volume":
        w_raw = w_vol.fillna(1.0)
    else:
        w_raw = w_vol.fillna(w_rng)
    w = w_raw.fillna(1.0).clip(lower=0.0, upper=p.cap)
    return w, vol_last, vol_avg, tr_avg


def _percentrank(x: np.ndarray, length: int, min_len: Optional[int] = None) -> np.ndarray:
    """ta.percentrank: % of the previous `length` values ≤ the current one.

    With `min_len` unset this is the Pine exactly: a full window, and a na
    compares false. With `min_len` set it is ADAPTED: non-finite prior values
    are left out and the rank is taken over whatever of the window exists, once
    at least `min_len` values do.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    out = np.full(T, np.nan)
    if T < 2:
        return out
    pad = np.concatenate([np.full(length, np.nan), x[:-1]])
    W = _swv(pad, length)[:T]
    cur = x[:, None]
    le = (W <= cur).sum(axis=1)
    if min_len is None:
        out = 100.0 * le / length
        out[np.arange(T) < length] = np.nan
    else:
        fin = np.isfinite(W).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(fin >= min_len, 100.0 * le / np.maximum(fin, 1), np.nan)
    out[~np.isfinite(x)] = np.nan
    return out


def _pivots(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """ta.pivothigh / ta.pivotlow: the pivot's value on its CONFIRMATION bar, else NaN.

    A pivot is strictly beyond every one of `left` bars before it and `right`
    bars after it; a window holding na confirms nothing. Ties on a smoothed
    oscillator are vanishingly rare, so strictness on both sides is the
    conservative reading.
    """
    T = len(x)
    out = np.full(T, np.nan)
    n = left + right + 1
    if T < n:
        return out
    W = _swv(x, n)                              # window ending at t = i + n - 1
    c = W[:, left]
    others = np.delete(W, left, axis=1)
    ok = np.isfinite(W).all(axis=1)
    hit = (c[:, None] > others).all(axis=1) if high else (c[:, None] < others).all(axis=1)
    idx = np.flatnonzero(ok & hit) + n - 1
    out[idx] = x[idx - right]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  1-2 · THE CHART'S OWN CONVICTION — Nishchaya v3 exactly, the flow ingredient
# ══════════════════════════════════════════════════════════════════════════════
def chart_conviction(df: pd.DataFrame, p: Params = DEFAULT) -> pd.DataFrame:
    """The chart's conviction, its warm-up gates and the effort → result reading.

    Columns: c (this bar's ΔC/TR), w, raw, eff_share, eff_pct, raw_sd, z (the
    tanh argument), osc (the smoothed ±100 conviction), sd_ok (rawSd's window is
    free of warm-up zeros), conv_ready (calibrated and smoothed).
    """
    high, low, close = df["high"], df["low"], df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    tr = true_range(high, low, close)
    disp = close - close.shift(1).fillna(close)
    c = (disp / tr).where(tr > 1e-12, 0.0).fillna(0.0)
    w, _, _, _ = _weights(tr, volume, p)

    num = (c * w).rolling(p.length).mean()
    den_a = (c.abs() * w).rolling(p.length).mean()        # agreement: Σ|c|·w
    den_e = w.rolling(p.length).mean()                     # effort:    Σw
    den = den_e if p.denominator == "Effort" else den_a
    # Ratio of sums, pinned to 0 while the window fills — a dead bar contributes
    # nothing to either side rather than an undefined ratio.
    raw = (100.0 * num / den).where(den > 1e-12, 0.0).fillna(0.0)

    # EFFORT → RESULT: the share of participation that became directional
    # displacement at all, and where it sits in its own history.
    eff_share = (den_a / den_e).where(den_e > 1e-12)
    eff_pct = _percentrank(eff_share.to_numpy(), p.norm)

    raw_sd = raw.rolling(p.norm).std(ddof=0)
    z = (raw / (3.0 * raw_sd)).where(raw_sd.notna() & (raw_sd >= 1e-9), 0.0)
    scaled = 100.0 * _tanh(z)
    osc = _ema(scaled, p.smooth) if p.smooth > 1 else scaled

    bar = np.arange(len(df))
    n_raw = np.maximum(bar - (p.length + p.vol_n) + 1, 0)
    sd_ok = n_raw > p.norm
    n_osc = np.cumsum(sd_ok)
    conv_ready = n_osc > p.norm + p.smooth
    return pd.DataFrame({"c": c, "w": w, "raw": raw, "eff_share": eff_share, "eff_pct": eff_pct,
                         "raw_sd": raw_sd, "z": z, "osc": osc, "sd_ok": sd_ok,
                         "conv_ready": conv_ready}, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
#  4 · THE PARENT, RECONSTRUCTED ON THE CHART                 (Ladder up)
#
#  The parent supplies only what it knows for certain: its state as of its last
#  CLOSED bar. Every sum is over one bar fewer than its window, so the chart can
#  complete it with the parent bar now forming. It lands exactly on the parent's
#  settled value at the parent's close, and sees nothing of the rest of it.
# ══════════════════════════════════════════════════════════════════════════════
def _parent_settled(pb: pd.DataFrame, norm: int, p: Params) -> pd.DataFrame:
    tr = true_range(pb["high"], pb["low"], pb["close"])
    c = ((pb["close"] - pb["close"].shift(1).fillna(pb["close"])) / tr).where(tr > 1e-12, 0.0)
    w, vol_last, vol_avg, tr_avg = _weights(tr, pb["volume"], p)
    num = (c * w).rolling(p.length).mean()
    den_a = (c.abs() * w).rolling(p.length).mean()
    den_e = w.rolling(p.length).mean()
    den = den_e if p.denominator == "Effort" else den_a
    raw = (100.0 * num / den).where(den > 1e-12, 0.0).fillna(0.0)
    sd = raw.rolling(norm).std(ddof=0)
    z_settled = (raw / (3.0 * sd)).where(sd.notna() & (sd >= 1e-9), 0.0)
    n_bars = pd.Series(np.maximum(np.arange(len(pb)) - (p.length + p.vol_n) + 1, 0), index=pb.index)
    s_num = (c * w).rolling(p.length - 1).sum()
    s_r = raw.rolling(norm - 1).sum()
    return pd.DataFrame({
        "c_prev": pb["close"], "v_last": vol_last, "v_avg": vol_avg, "tr_avg": tr_avg,
        "s_num": s_num, "s_da": (c.abs() * w).rolling(p.length - 1).sum(),
        "s_de": w.rolling(p.length - 1).sum(),
        "s_r": s_r, "s_r2": (raw * raw).rolling(norm - 1).sum(),
        "z_settled": z_settled,
        # The Pine also demands a volume baseline here (not na(pVa)), which a symbol that
        # prints no volume — index spot, FX — never has, so its parent rung never
        # calibrates and every signal pauses on "conviction warming" for good. The develop
        # step already falls back to true range when there is no volume, exactly as the
        # chart's own participation does, so only the true-range baseline is required.
        "rdy": (n_bars > norm) & s_num.notna() & s_r.notna() & tr_avg.notna(),
    })


def parent_rung(df: pd.DataFrame, freq: str, norm: int, p: Params = DEFAULT):
    """Section 4 · a higher frame's conviction z, RECONSTRUCTED on the chart bars.

    Returns (z, recon_err): the rung's live z per chart bar (NaN until the parent
    can calibrate) and, on each parent bar's final chart bar, the gap between
    the live reading and the value the parent settled on, in points.
    """
    key = sv.period_key(df.index, freq)
    g = df.groupby(key)
    vol = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    pb = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                       "volume": vol.groupby(key).sum(min_count=1)})
    st = _parent_settled(pb, norm, p)
    prev = st.shift(1).reindex(key)
    prev.index = df.index

    # f_agg — the forming parent bar, from the chart bars since it opened.
    a_h = df["high"].groupby(key).cummax()
    a_l = df["low"].groupby(key).cummin()
    a_v = vol.fillna(0.0).groupby(key).cumsum()
    a_c = df["close"]

    # f_develop — the parent's arithmetic, finished with the forming bar.
    cp = prev["c_prev"]
    tr_d = np.maximum(a_h, cp) - np.minimum(a_l, cp)
    c_d = ((a_c - cp) / tr_d).where(tr_d > 1e-12, 0.0)
    v_ok = a_v > 0
    alpha = 2.0 / (p.vol_n + 1)
    vl = a_v.where(v_ok, prev["v_last"])
    v_a = (alpha * vl + (1.0 - alpha) * prev["v_avg"]).where(vl.notna(), prev["v_avg"])
    t_a = alpha * tr_d + (1.0 - alpha) * prev["tr_avg"]
    w_v = (a_v / v_a).where(v_ok & v_a.notna() & (v_a > 1e-12))
    w_r = (tr_d / t_a).where(t_a.notna() & (t_a > 1e-12), 1.0)
    if p.participation == "Off":
        w_x = pd.Series(1.0, index=df.index)
    elif p.participation == "True range":
        w_x = w_r
    elif p.participation == "Volume":
        w_x = w_v.fillna(1.0)
    else:
        w_x = w_v.fillna(w_r)
    w_d = w_x.fillna(1.0).clip(lower=0.0, upper=p.cap)
    nm = (prev["s_num"] + c_d * w_d) / p.length
    d_a = (prev["s_da"] + c_d.abs() * w_d) / p.length
    d_e = (prev["s_de"] + w_d) / p.length
    dn = d_e if p.denominator == "Effort" else d_a
    rw = (100.0 * nm / dn).where(dn > 1e-12, 0.0)
    mu = (prev["s_r"] + rw) / norm
    var = (prev["s_r2"] + rw * rw) / norm - mu * mu
    sd = np.sqrt(var.clip(lower=0.0))
    z = (rw / (3.0 * sd)).where(sd >= 1e-9, 0.0)
    z = z.where(prev["rdy"].astype("boolean").fillna(False).astype(bool))

    # RECONSTRUCTION CHECK — on the first chart bar of each new parent bar, the
    # last live reading of the bar that just closed (the forming bar then WAS the
    # settled bar) is compared with the value the parent itself settled on, in
    # points, and carried until the next close — as the Pine records it, so the
    # check never needs to know a bar is the last of its parent in advance.
    k = pd.Series(key, index=df.index)
    first = (k != k.shift(1)) & k.shift(1).notna()
    settled_prev = st["z_settled"].reindex(pd.PeriodIndex(k.shift(1).where(first).dropna()))
    settled_prev.index = k.index[first.to_numpy()]
    z_prev = z.shift(1)
    err = (100.0 * (_tanh(z_prev) - _tanh(settled_prev.reindex(df.index)))).abs()
    err = err.where(first & z_prev.notna()).ffill()
    return z, err


# ══════════════════════════════════════════════════════════════════════════════
#  4b · A CHILD, READ FROM INSIDE THE CHART BAR               (Ladder down)
#
#  Sampling a lower frame at the chart bar's close would describe only its last
#  bar. So the lower frame runs the full engine on its own continuous history,
#  and the chart bar takes the participation-weighted AVERAGE of that frame's z
#  across all of its bars inside the chart bar: the bar seen from inside.
# ══════════════════════════════════════════════════════════════════════════════
def week_label(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Monday a daily bar's week is labelled by — sanket.resample_to_weekly's bins."""
    idx = pd.DatetimeIndex(index).normalize()
    return idx - pd.to_timedelta(idx.dayofweek, unit="D")


def inside_rung(frame: pd.DataFrame, labels: pd.DatetimeIndex, chart_index: pd.DatetimeIndex,
                p: Params = DEFAULT) -> pd.Series:
    """A lower frame's z, participation-weighted inside each chart bar it belongs to (f_inside).

    ``labels`` gives, per bar of ``frame``, the chart bar it sits inside (NaT: none). The
    frame runs the full engine on its own history; its z is na until its own normalization
    window is clean (f_child), so a rung joins only where it has calibrated."""
    ch = chart_conviction(frame, p)
    z = ch["z"].where(ch["sd_ok"])
    w = ch["w"].fillna(1.0).clip(lower=1e-6)
    lab = pd.DatetimeIndex(labels)
    keep = ~lab.isna()
    z, w, lab = z[keep], w[keep], lab[keep]
    num = (z * w).groupby(lab).sum(min_count=1)
    den = w.where(z.notna()).groupby(lab).sum(min_count=1)
    y = (num / den).where(den > 0)
    return y.reindex(pd.DatetimeIndex(chart_index).normalize()).set_axis(chart_index)


def ladder_down_rungs(intraday: dict | None, chart_index: pd.DatetimeIndex, chart: str,
                      p: Params = DEFAULT) -> pd.DataFrame:
    """The intraday rungs of a Ladder-down conviction tape, one column per frame present."""
    raise RuntimeError("Tattva reads the conviction ladder UP (W · D): intraday history rolls "
                       "off yfinance, so a Ladder-down past would be rewritten on a later run.")
    import intraday as _idm  # noqa: F401  (unreachable in Tattva)
    cols = {}
    if not intraday:
        return pd.DataFrame(index=chart_index)
    idx = pd.DatetimeIndex(chart_index)
    days = idx.normalize()
    for f, bars in intraday.items():
        if bars is None or len(bars) == 0:
            continue
        lab = _idm.session_days(bars.index, days if chart == "D" else pd.date_range(days.min(), days.max() + pd.Timedelta(days=6), freq="D"))
        if chart == "W":
            lab = week_label(lab)
        cols[f] = inside_rung(bars, lab, idx, p)
    return pd.DataFrame(cols, index=chart_index)


def child_rung(daily: pd.DataFrame, chart_index: pd.DatetimeIndex, p: Params = DEFAULT) -> pd.Series:
    """The daily frame's z, participation-weighted inside each weekly chart bar (f_inside)."""
    ch = chart_conviction(daily, p)
    z = ch["z"].where(ch["sd_ok"])               # f_child: na until its own window is clean
    w = ch["w"].fillna(1.0).clip(lower=1e-6)
    lab = week_label(daily.index)
    num = (z * w).groupby(lab).sum(min_count=1)
    den = w.where(z.notna()).groupby(lab).sum(min_count=1)
    y = (num / den).where(den > 0)
    return y.reindex(week_label(chart_index)).set_axis(chart_index)


# ══════════════════════════════════════════════════════════════════════════════
#  THE WHOLE INDICATOR, BAR BY BAR
# ══════════════════════════════════════════════════════════════════════════════
COLUMNS = (
    # conviction — the flow ingredient
    "c", "w", "raw", "raw_sd", "eff_share", "eff_pct", "eff_abs", "conv", "conv_z", "conv_ready",
    # value — the position ingredient
    "value", "value_z", "value_built",
    # the trace and its push
    "trace", "trace_ok", "hist", "hist_sd", "hist_z", "thr", "hist_ready",
    "push", "push_tier", "quiet", "split", "settling",
    # the tapes
    "c_tape", "c_ready", "c_rungs", "c_ladder", "v_tape", "v_ready", "v_rungs",
    # the stack
    "stack_ok", "stack_why",
    # evidence
    "bull_div", "bear_div", "div_x1", "div_p1", "div_x2", "div_p2",
    "abs_seen", "bull_div_seen", "bear_div_seen",
    # the ◆'s condition, before the cooldown — signals() adds the signal set
    "con_cand_l", "con_cand_s",
    # reconstruction
    "rec_err",
)


def compute(df: pd.DataFrame, value: pd.DataFrame, p: Params = DEFAULT, chart: str = "D",
            daily: Optional[pd.DataFrame] = None, intraday: Optional[dict] = None) -> pd.DataFrame:
    """Every reading, piece of evidence and signal pragati.pine draws, per bar.

    df      the chart's OHLCV, lower-case columns, ascending
    value   samanvaya.compute_value(df, …) on the same index
    chart   "D" (conviction Ladder down from ``intraday``, ↺ W · D before it) or
            "W" (D inside · W, needs `daily`; 1h / 4h from ``intraday`` where present)
    daily   the daily bars behind a weekly chart, for its Ladder-down rung
    """
    T = len(df)
    idx = df.index
    th = p.theta
    wV = float(p.mix)
    wC = 1.0 - wV

    # ── conviction ──
    ch = chart_conviction(df, p)
    sd_ok = ch["sd_ok"].to_numpy()
    osc = ch["osc"]

    # ── the conviction ladder ──
    rec_err = pd.Series(np.nan, index=idx)
    z_chart = ch["z"]
    sdok_s = pd.Series(sd_ok, index=idx)
    if chart == "W":
        # Weekly: Ladder up has no frames above it, so the Pine reads DOWN — the daily bars
        # inside the week (always available) and, where yfinance has them, 1h and 4h.
        rung = child_rung(daily, idx, p) if daily is not None and len(daily) else pd.Series(np.nan, index=idx)
        extra = ladder_down_rungs(intraday, idx, "W", p) if p.ladder == "down" else pd.DataFrame(index=idx)
        R = pd.concat([rung.rename("D"), extra], axis=1)
        multi = bool(R.notna().to_numpy().any()) or (daily is not None and len(daily) > 0)
        n_child = R.notna().sum(axis=1)
        z_cnt = (1 + n_child).where(sdok_s, 0)
        z_lad = ((z_chart + R.sum(axis=1, min_count=1).fillna(0.0)) / z_cnt.where(z_cnt > 0, 1)).where(sdok_s, 0.0)
        lad = 100.0 * _tanh(z_lad)
        c_tape = _ema(lad, p.smooth) if p.smooth > 1 else lad
        n_lad = np.cumsum(multi & sd_ok & (z_cnt.to_numpy() > 1))
        c_ready = n_lad > p.norm + p.smooth
        c_ladder = pd.Series(np.where(c_ready, "down", ""), index=idx)
    else:
        # Ladder up — W · D, the parent reconstructed on the chart. Always computed: it is the
        # tape itself on "up", and the ↺ fallback on "down" wherever no intrabar exists.
        rung, rec_err = parent_rung(df, "W", p.parent_norm, p)
        z_cnt = (1 + rung.notna().astype(int)).where(sdok_s, 0)
        z_lad = ((z_chart + rung.fillna(0.0)) / z_cnt.where(z_cnt > 0, 1)).where(sdok_s, 0.0)
        lad = 100.0 * _tanh(z_lad)
        c_tape = _ema(lad, p.smooth) if p.smooth > 1 else lad
        n_lad = np.cumsum(sd_ok & (z_cnt.to_numpy() > 1))
        c_ready = n_lad > p.norm + p.smooth
        c_ladder = pd.Series(np.where(c_ready, "up", ""), index=idx)
        if p.ladder == "down":
            # Ladder DOWN — every lower frame yfinance carries, each averaged inside the day.
            # It takes over once it has held a lower rung for the Pine's own warm-up
            # (ladReady: norm + smooth bars); before that the tape reads Ladder up, marked ↺.
            R = ladder_down_rungs(intraday, idx, "D", p)
            n_child = R.notna().sum(axis=1) if R.shape[1] else pd.Series(0, index=idx)
            cnt_d = (1 + n_child).where(sdok_s, 0)
            z_dn = ((z_chart + (R.sum(axis=1, min_count=1).fillna(0.0) if R.shape[1] else 0.0))
                    / cnt_d.where(cnt_d > 0, 1)).where(sdok_s, 0.0)
            lad_dn = 100.0 * _tanh(z_dn)
            tape_dn = _ema(lad_dn, p.smooth) if p.smooth > 1 else lad_dn
            use_dn = np.cumsum(sd_ok & (n_child.to_numpy() > 0)) > p.norm + p.smooth
            c_tape = pd.Series(np.where(use_dn, tape_dn, c_tape), index=idx)
            z_cnt = pd.Series(np.where(use_dn, cnt_d, z_cnt), index=idx)
            c_ready = np.where(use_dn, True, c_ready)
            c_ladder = pd.Series(np.where(use_dn, "down", np.where(c_ready, "up↺", "")), index=idx)
        multi = True

    # ── value ──
    built = value["model_built"].astype(bool).to_numpy()
    u_z = value["value_z"].astype(float)
    u_osc = value["value"].astype(float)
    v_tape = value["tape"].astype(float).where(built, 0.0)
    v_ready = value["tape_ready"].astype(bool).to_numpy()
    basket_ok = (value["basket_warm"].astype(bool) & value["enough"].astype(bool)).to_numpy()

    # ── THE TRACE · conviction × value on this chart ──
    # Both ingredients in unit σ — conviction as raw/σ, smoothed as the engine
    # smooths it; value as Samanvaya's unified z — blended in z-space with their
    # measured correlation, variance restored, bounded ONCE on Samanvaya's scale.
    z_conv_u = 3.0 * z_chart
    z_conv = _ema(z_conv_u, p.smooth) if p.smooth > 1 else z_conv_u
    z_val = u_z.where(built, 0.0)
    # v8 (the Pine): ρ = ta.correlation(zcS, unified_z, 200), nz → 0 until it exists
    rho = z_conv.rolling(sv.VAR_CORR_LEN, min_periods=sv.VAR_CORR_LEN).corr(u_z)
    rho_c = rho.fillna(0.0).clip(-0.90, 0.99)
    cv_var = wC * wC + wV * wV + 2.0 * wC * wV * rho_c
    cv_corr = np.where(cv_var > 1e-9, 1.0 / np.sqrt(cv_var), 1.0)
    trace_ok = ((wC <= 0.0) | sd_ok) & ((wV <= 0.0) | built)
    trace_z = (wC * z_conv + wV * z_val) * cv_corr
    trace = (100.0 * sv.softbound(trace_z * sv.GAIN)).clip(-100.0, 100.0)
    split = ((wC > 0.0) & (wV > 0.0) & built & ((z_conv > 0.0) != (u_z > 0.0))
             & (z_conv.abs() > 0.5) & (u_z.abs() > 0.5))

    # ── THE HISTOGRAM · the trace's own push ──
    sig = _ema(trace, p.signal)
    hist = trace - sig
    n_tr = np.cumsum(trace_ok)
    # the Pine's warm-up: conviction's smoothing, the blend's ρ window, then the
    # histogram's own σ window
    ready = n_tr > ((p.smooth if wC > 0.0 else 0) + (sv.VAR_CORR_LEN if 0.0 < wV < 1.0 else 0)
                    + p.norm + p.signal)
    hist_sd = hist.rolling(p.norm).std(ddof=0)
    thr = p.k * hist_sd.fillna(0.0)
    h_prev = hist.shift(1)
    thr_prev = thr.shift(1)
    imp_up = ((hist > thr) & (h_prev <= thr_prev)).to_numpy()
    imp_dn = ((hist < -thr) & (h_prev >= -thr_prev)).to_numpy()

    # ── the push, in five levels, read from the column as drawn ──
    d_tr = trace - trace.shift(1).fillna(trace)
    slope_db = SLOPE_DEADBAND * d_tr.rolling(p.norm).std(ddof=0).fillna(0.0)
    h_mag = hist.abs()
    above = hist >= 0.0
    expand = h_prev.notna() & (h_mag > h_prev.abs())
    with_tr = (above & (d_tr > slope_db)) | (~above & (d_tr < -slope_db))
    h_imp = h_mag >= thr
    raw_pr = _percentrank(ch["raw_sd"].to_numpy(), 4 * p.norm, min_len=p.norm)
    quiet = p.quiet & (wC > 0.0) & sd_ok & np.isfinite(raw_pr) & (np.nan_to_num(raw_pr, nan=100.0) <= QUIET_PCT)
    live = ready & (expand | with_tr).to_numpy() & ~quiet
    strong = (expand & h_imp).to_numpy()
    ab = above.to_numpy()
    push = np.where(~live, 0, np.where(ab, np.where(strong, 2, 1), np.where(strong, -2, -1)))
    tier = np.where(~ready, "warming",
                    np.where(expand, np.where(h_imp, "impulse", "pressing"),
                             np.where(with_tr, "decelerating", "turning")))
    tier = np.where(quiet & ready, np.char.add(tier.astype(str), " · quiet"), tier)

    # ── the stack ──
    m_c = c_tape.to_numpy()
    m_v = v_tape.to_numpy()
    stack_ok = ready & c_ready & v_ready
    eff_pct = ch["eff_pct"].to_numpy()
    eff_abs = np.isfinite(eff_pct) & (np.nan_to_num(eff_pct, nan=100.0) <= EFF_ABSORBED)

    # ── DIVERGENCE EVIDENCE, on conviction's own pivots ──
    oscv = osc.to_numpy()
    hi_p = df["high"].to_numpy(dtype=float)
    lo_p = df["low"].to_numpy(dtype=float)
    uo = u_osc.to_numpy()
    conv_ready = ch["conv_ready"].to_numpy()
    pv_h = _pivots(oscv, p.pl, p.pr, True)
    pv_l = _pivots(oscv, p.pl, p.pr, False)
    span = p.pl + p.pr + 1
    bull_div = np.zeros(T, dtype=bool)
    bear_div = np.zeros(T, dtype=bool)
    div_x1 = np.full(T, np.nan)
    div_p1 = np.full(T, np.nan)
    div_x2 = np.full(T, np.nan)
    div_p2 = np.full(T, np.nan)
    for is_high, pv, prices, out in ((True, pv_h, hi_p, bear_div), (False, pv_l, lo_p, bull_div)):
        last_v = last_b = last_p = last_x = None
        for t in np.flatnonzero(np.isfinite(pv) & conv_ready):
            b = t - p.pr
            win = prices[max(0, t - span + 1):t + 1]
            off = int(np.argmax(win) if is_high else np.argmin(win))
            px = max(0, t - span + 1) + off
            pp = float(prices[px])
            gap = 0 if last_b is None else b - last_b
            if last_v is not None and p.gap_min <= gap <= p.gap_max:
                t_piv = uo[t - p.pr] if t - p.pr >= 0 else np.nan
                if is_high:
                    ok = (pv[t] < last_v and pp > last_p and (not p.zone_gate or last_v >= p.z1)
                          and built[t] and np.isfinite(t_piv) and t_piv > 0.0)
                else:
                    ok = (pv[t] > last_v and pp < last_p and (not p.zone_gate or last_v <= -p.z1)
                          and built[t] and np.isfinite(t_piv) and t_piv < 0.0)
                if ok:
                    out[t] = True
                    div_x1[t], div_p1[t], div_x2[t], div_p2[t] = last_x, last_p, px, pp
            last_v, last_b, last_p, last_x = float(pv[t]), b, pp, px

    # ── recent evidence, for the screen's Evidence column: context, read by no signal ──
    def _recent(ev: np.ndarray) -> np.ndarray:
        return pd.Series(ev.astype(float), index=idx).rolling(RECENT, min_periods=1).max().to_numpy() > 0

    abs_seen, bull_seen, bear_seen = _recent(eff_abs), _recent(bull_div), _recent(bear_div)

    # ── the ◆'s condition · the push resumes inside conviction, with value room ──
    tradable = (wV <= 0.0) | basket_ok
    hv = hist.to_numpy()
    hs = pd.Series(hv, index=idx)
    pulled_up = (hs.rolling(p.pull).min() < 0.0).to_numpy()
    pulled_dn = (hs.rolling(p.pull).max() > 0.0).to_numpy()
    not_abs = (not p.effort) | ~eff_abs
    # chart conviction on the ◆'s side, the ladder in control, value room (v5 / v8)
    oscv_ = osc.to_numpy(dtype=float)
    t_con_l = (m_c >= p.z1) & (m_v < th) & (oscv_ > 0.0) & not_abs
    t_con_s = (m_c <= -p.z1) & (m_v > -th) & (oscv_ < 0.0) & not_abs
    # before the cooldown and the ▲▼'s precedence — signals() runs the per-direction
    # clock with the grid's ▲▼, as the Pine's section 8b does
    con_cand_l = np.asarray(p.resume & stack_ok & pulled_up & imp_up & t_con_l, dtype=bool)
    con_cand_s = np.asarray(p.resume & stack_ok & pulled_dn & imp_dn & t_con_s, dtype=bool)

    # ── why the stack cannot judge, when it cannot ──
    need = sv.HOLD_OUT + sv.MODEL_MIN
    why = np.full(T, "", dtype=object)
    for t in np.flatnonzero(~stack_ok):
        if not ready[t]:
            # each chart bar closes one anchor period (the model clock is the chart's own bar)
            why[t] = (f"value model building · {min(t, need)}/{need} periods"
                      if wV > 0.0 and not built[t] else "chart warming")
        elif not c_ready[t]:
            why[t] = "conviction tape warming" if multi else "no conviction frames on this chart"
        elif not v_ready[t]:
            why[t] = "value tape warming"

    hsd = hist_sd.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        hist_z = np.where(np.isfinite(hsd) & (hsd > 1e-12), hv / np.where(hsd > 0, hsd, 1.0), np.nan)

    return pd.DataFrame({
        "c": ch["c"], "w": ch["w"], "raw": ch["raw"], "raw_sd": ch["raw_sd"],
        "eff_share": ch["eff_share"], "eff_pct": eff_pct, "eff_abs": eff_abs,
        "conv": osc, "conv_z": z_conv, "conv_ready": conv_ready,
        "value": u_osc, "value_z": u_z, "value_built": built,
        "trace": trace.where(trace_ok), "trace_ok": trace_ok,
        "hist": hist, "hist_sd": hist_sd, "hist_z": hist_z, "thr": thr, "hist_ready": ready,
        "push": push, "push_tier": tier, "quiet": quiet, "split": split, "settling": ~tradable,
        "c_tape": c_tape, "c_ready": c_ready, "c_rungs": z_cnt, "c_ladder": c_ladder,
        "v_tape": v_tape, "v_ready": v_ready, "v_rungs": value["tape_rungs"],
        "stack_ok": stack_ok, "stack_why": why,
        "bull_div": bull_div, "bear_div": bear_div,
        "div_x1": div_x1, "div_p1": div_p1, "div_x2": div_x2, "div_p2": div_p2,
        "abs_seen": abs_seen, "bull_div_seen": bull_seen, "bear_div_seen": bear_seen,
        "con_cand_l": con_cand_l, "con_cand_s": con_cand_s,
        "rec_err": rec_err,
    }, index=idx)[list(COLUMNS)]


def signals(out: pd.DataFrame, grid: pd.DataFrame, p: Params = DEFAULT) -> pd.DataFrame:
    """The Pine's section 8b: the ▲▼ read from the grid, on ``compute``'s output.

    ▲ CAPITULATION TURN — the first bar the grid stands in DOWN · cheap (cell 0) with
    value momentum reverting (the 5 × 5 value phase +1), not in conviction's quiet regime
    (v9.3). ▼ DISTRIBUTION — the first bar
    in DOWN · rich (cell 2). Both need the stack able to judge, and the per-direction
    cooldown, which a ◆ shares. Adds turn_buy / turn_sell, resume_long / resume_short
    (the ◆ yields to a ▲▼ on the same bar), decl / decl_since, and armed / armed_age —
    'armed' is a name in capitulation whose value is still cheapening: the ▲ comes when
    it turns.
    """
    out = out.copy()
    T = len(out)
    cell = grid["cvg_cell"].to_numpy(dtype=int)
    vph = grid["cvg_vph"].to_numpy(dtype=int)
    cap = (cell == 0) & (vph == 1)
    dist = cell == 2
    # v9.3: not in conviction's quiet regime on that bar (the Pine's gQuiet)
    quiet = grid["cvg_quiet"].to_numpy(bool) if "cvg_quiet" in grid else np.zeros(T, bool)
    cap_b = cap & ~np.r_[False, cap[:-1]] & ~quiet
    dist_s = dist & ~np.r_[False, dist[:-1]]
    ok = out["stack_ok"].fillna(False).to_numpy(bool)
    cl, cs = out["con_cand_l"].to_numpy(bool), out["con_cand_s"].to_numpy(bool)
    tb, ts = np.zeros(T, bool), np.zeros(T, bool)
    rl, rs = np.zeros(T, bool), np.zeros(T, bool)
    decl = np.zeros(T, dtype=int)
    decl_since = np.full(T, -1, dtype=int)
    armed = np.zeros(T, dtype=int)
    armed_age = np.zeros(T, dtype=int)
    last_l = last_s = None
    d_now, d_bar, in_cap = 0, -1, 0
    for t in range(T):
        cool_l = last_l is None or t - last_l >= p.cool
        cool_s = last_s is None or t - last_s >= p.cool
        buy = p.turn and ok[t] and cool_l and cap_b[t]
        sell = p.turn and ok[t] and cool_s and dist_s[t]
        con_l = cl[t] and not buy and cool_l
        con_s = cs[t] and not sell and cool_s
        if buy or con_l:
            last_l = t
        if sell or con_s:
            last_s = t
        if buy:
            d_now, d_bar = 1, t
        if sell:
            d_now, d_bar = -1, t
        tb[t], ts[t], rl[t], rs[t] = buy, sell, con_l, con_s
        decl[t], decl_since[t] = d_now, d_bar
        in_cap = in_cap + 1 if cell[t] == 0 else 0
        if cell[t] == 0 and vph[t] != 1 and ok[t]:
            armed[t], armed_age[t] = 1, in_cap
    out["turn_buy"], out["turn_sell"] = tb, ts
    out["resume_long"], out["resume_short"] = rl, rs
    out["decl"], out["decl_since"] = decl, decl_since
    out["armed"], out["armed_age"] = armed, armed_age
    return out


def warmup_bars(p: Params = DEFAULT) -> int:
    """Bars before the histogram's σ window is clean — the Pine's `ready`.

    Conviction calibrates at length + vol_n + norm (sdOK); the trace counts from
    there, and the histogram is ready norm + signal bars later. The value model
    builds long before that (22 anchor periods), so it does not bind on Daily.
    The TAPES bind separately: the parent value rung needs ~55 parent bars of
    spread, and the stack waits for them (`stack_why` says so).
    """
    return int(p.length + p.vol_n + 2 * p.norm + p.signal + 1)


__all__ = ["COLUMNS", "DEFAULT", "Params", "RAIL_OUT", "chart_conviction", "child_rung",
           "compute", "parent_rung", "signals", "true_range", "warmup_bars", "week_label"]
