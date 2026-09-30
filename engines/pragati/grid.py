"""
TATTVA — the Conviction-Value Grid, 3 × 3 · Pragati v8 (pragati.pine, the "CVG" block)

Ported verbatim from Sanket's cvgrid.py (units, names, the state engine, the phases).
══════════════════════════════════════════════════════════════════════════════

Pragyam's grid, read for one chart — the grid pragati.pine v8 draws and Pragyam sizes
its book from. A state, not a signal: where a name stands BETWEEN signals, as something
to DO about it, at a weight.

THE AXES are the two ladder tapes:
    rows     UP (buyers in control) past +30 · FAINT between · DOWN (sellers) past −30
    columns  cheap past −θ · fair between · rich past +θ

CONVICTION'S OWN HISTOGRAM RUNS THE ROWS. A row moves to its tape only while
conviction's histogram confirms a push that way — on the move's side, not TURNING, not
QUIET. Otherwise the row is HELD against its tape.

THE 5 × 5 PHASES. Each momentum tape confirms its own instrument's edge: conviction's
row edge while the faster view RUNS on the row's side, value's column edge once the fast
end is REVERTING toward fair. An unconfirmed edge sits half-way to the middle.

GRADED UNITS. The name's units are the nine cells' units read at its shaded position on
the map (the Pine's own tape ramps; a held row keeps between half and all of its cell
by how firmly the push holding it is drawn) — Pragyam's graded map, bit for bit.

THE UNITS — v8, measured (studies/pine_audit.md, research in Pragyam):

                   CHEAP                FAIR                  RICH
    UP  buyers     Buy · turned 3       Hold · building 1½    Trim · paid ¾
    FAINT          Accumulate · basing 1½  Wait · idle 1      Trim · stalling ¾
    DOWN sellers   Buy · capitulation 4 Accumulate · washout 1½  Exit · distribution ¼

v5 / Pragyam's seed had DOWN·cheap Watch 1, DOWN·fair Reduce ½, UP·fair Add 3 and
UP·rich Hold 1½. Chosen on data before 2018 across 380 instruments in six classes and
confirmed after it; in Pragyam's own allocator (monthly, every name held) the same four
moves beat the seed units in both eras on Nifty 50 and Dow 30, at lower turnover.
Crypto trends and is the stated exception.

v9.1: Buy · capitulation 3 → 4. The one cell positive in every era of the v9 audit
(look-ahead-free scoring); at 4 units the grid read as a position improved or tied in all
three eras on both scorers, and in Pragyam's allocator it beat 3 in all three eras on
Nifty 50 and Dow 30 (every-name and top-30 books, net of costs).

Author: @thebullishvalue
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engines.pragati import core as pg

N_ROWS, N_COLS = 3, 3
UNREAD = N_ROWS * N_COLS          # 9

# Cell = row · 3 + column. Rows: 0 DOWN (sellers), 1 FAINT, 2 UP (buyers).
# Columns: 0 cheap, 1 fair, 2 rich.
NAMES = (
    "Buy · capitulation",  "Accumulate · washout", "Exit · distribution",
    "Accumulate · basing", "Wait · idle",          "Trim · stalling",
    "Buy · turned",        "Hold · building",      "Trim · paid",
    "Unread",
)
#: Pragyam's name for each cell — the family its book sizes from.
FAMILY = ("DISLOCATED", "FADING", "DISTRIBUTION", "BASING", "IDLE", "STALLING",
          "TURNED", "BUILDING", "PAID", "UNREAD")
UNITS = (
    4.00, 1.50, 0.25,
    1.50, 1.00, 0.75,
    3.00, 1.50, 0.75,
    1.00,
)
#: The same units keyed as the state engine reads them: (row −1 DOWN / 0 FAINT / +1 UP, column).
UNITS_RC = {(r - 1, c): UNITS[r * 3 + c] for r in range(3) for c in range(3)}
# The side each action works: +1 builds (Buy, Accumulate), −1 cuts (Trim, Exit), 0 neither.
SIDES = (
    1,  1, -1,
    1,  0, -1,
    1,  0, -1,
    0,
)
MEANING = (
    "sellers in control at a cheap price - capitulation; measured, followed by gains in both eras on every asset class but crypto",
    "sellers in control at a fair price - a washout; measured, followed by gains in both eras outside crypto",
    "sellers in control of a rich price - distribution",
    "cheap, control not yet decided - the base is forming",
    "fair price, control not yet decided - nothing to do",
    "rich, and control has faded - the move has stalled",
    "cheap, and buyers now in control - a dislocation that turned",
    "buyers in control at a fair price - hold it; measured, adding here earned nothing in either era",
    "buyers in control of a price already rich - paid for; take some off",
    "a tape not yet calibrated, or switched off",
)
ROW_LABELS = ("sellers in control", "control undecided", "buyers in control")
COL_LABELS = ("cheap", "fair", "rich")

# ── TONES — one colour per meaning, everywhere the grid appears (Pragyam's CVG_TONE) ──
#   emerald  build — capitulation, a turn        cyan   accumulate — washout, base
#   amber    hold or trim — building, rich        slate  wait / unread
#   rose     exit — distribution
TONES = (
    "emerald", "cyan",  "rose",
    "cyan",    "slate", "amber",
    "emerald", "amber", "amber",
    "slate",
)
TONE_CHIP = {"emerald": "success", "amber": "warning", "cyan": "info",
             "slate": "neutral", "rose": "danger"}
ACTIONS = ("Buy", "Accumulate", "Hold", "Wait", "Trim", "Exit")
ACTION_TONE = {"Buy": "emerald", "Accumulate": "cyan", "Hold": "amber", "Wait": "slate",
               "Trim": "amber", "Exit": "rose", "Unread": "slate"}
ACTION_UNITS = {"Buy": 4.0, "Accumulate": 1.5, "Hold": 1.5, "Wait": 1.0, "Trim": 0.75,
                "Exit": 0.25, "Unread": 1.0}

PUSH_TEXT = {1: "push ↑", 0: "no push", -1: "push ↓"}
PUSH_GLYPH = {1: "↑", 0: "·", -1: "↓"}
READ_THE_PUSH = ("Conviction's histogram runs the rows: a row moves only on a push that way, "
                 "and is HELD otherwise. Measured: inside the capitulation cells the push's "
                 "direction made no consistent difference - the state carried the edge, not "
                 "the timing - so Buy · capitulation and Accumulate · washout do not wait for "
                 "a push ↑.")

COLUMNS = ("cvg_cell", "cvg_units", "cvg_held", "cvg_since", "cvg_from", "cvg_chart",
           "cvg_lead", "cvg_cph", "cvg_vph", "cvg_push")


def action(cell: int) -> str:
    return NAMES[int(cell)].split(" · ")[0]


def reason(cell: int) -> str:
    parts = NAMES[int(cell)].split(" · ")
    return parts[1] if len(parts) > 1 else "needs both tapes"


def row_of(c: float, z1: float) -> int:
    """The conviction tape's row: 2 UP, 1 FAINT, 0 DOWN."""
    return 2 if c >= z1 else 0 if c <= -z1 else 1


def col_of(v: float, theta: float) -> int:
    """The value tape's column: 0 cheap, 1 fair, 2 rich."""
    return 0 if v <= -theta else 2 if v >= theta else 1


# ════════════════════════════════════════════════════════════════════════════════════════
# THE STATE ENGINE · the Pine's grid block, ported bar for bar
# ════════════════════════════════════════════════════════════════════════════════════════
def _stdev(x: pd.Series, n: int) -> pd.Series:
    """ta.stdev (population), na until a full window of values exists."""
    return x.rolling(n, min_periods=n).std(ddof=0)


def _tape_ink(x, knee, solid):
    a = np.abs(x)
    bright = 0.65 + 0.35 * np.clip((a - knee) / max(solid - knee, 1e-9), 0.0, 1.0)
    faint = 0.12 + 0.33 * np.clip(a / max(knee, 1e-9), 0.0, 1.0)
    return np.where(a >= knee, bright, faint)


def _g_ink(m, lo, hi, t0, t1):
    rng = hi - lo
    f = np.where(rng > 1e-12, np.clip((m - lo) / np.where(rng > 1e-12, rng, 1.0), 0.0, 1.0), 0.0)
    return 1.0 - (t0 + (t1 - t0) * f) / 100.0


def _g_at(rp, cp, units: dict):
    """f_gAt: the nine units read at a position — bilinear between the cells."""
    r = min(1.0, max(-1.0, rp))
    c = min(1.0, max(-1.0, cp))
    r0 = -1 if r < 0 else 0
    c0 = -1 if c < 0 else 0
    wr, wc = r - r0, c - c0
    U = lambda rr, cc: units[(rr, cc)]                     # noqa: E731
    lo_ = (1.0 - wc) * U(r0, c0 + 1) + wc * U(r0, c0 + 2)
    hi_ = (1.0 - wc) * U(r0 + 1, c0 + 1) + wc * U(r0 + 1, c0 + 2)
    return (1.0 - wr) * lo_ + wr * hi_


def _engine(out: pd.DataFrame, cv: pd.Series, raw_sd: pd.Series, cv_ready: np.ndarray,
            p, units: dict, ladder_down=False) -> pd.DataFrame:
    """The Pine's grid state engine, bar for bar: row memory gated by conviction's own
    histogram, the two momentum sides and 5 × 5 phases, and the graded units read at the
    name's shaded position (f_gAt)."""
    T = len(out)
    th = float(p.theta)
    # conviction's own histogram and its tiers — the gate Pragyam reads
    cv_hist = cv - pg._ema(cv, p.signal)
    d_cv = cv - cv.shift(1).fillna(cv)
    cv_db = 0.1 * _stdev(d_cv, p.norm).fillna(0.0)
    cv_hsd = _stdev(cv_hist, p.norm)
    cv_thr = p.k * cv_hsd.fillna(0.0)
    cv_mag = cv_hist.abs()
    above = (cv_hist >= 0).to_numpy()
    h1 = cv_hist.shift(1).fillna(0.0)
    expand = np.where(above, cv_hist > h1, cv_hist < h1)
    with_ = np.where(above, d_cv > cv_db, d_cv < -cv_db)
    imp = (cv_mag >= cv_thr).to_numpy()
    cv_hi = np.maximum.reduce([2.0 * cv_hsd.fillna(0.0).to_numpy(), 1.5 * cv_thr.to_numpy(),
                               np.full(T, 1e-9)])
    raw_pr = pg._percentrank(raw_sd.to_numpy(dtype=float), min(4 * p.norm, 4999))
    quiet = cv_ready & np.isfinite(raw_pr) & (np.nan_to_num(raw_pr, nan=100.0) < 20.0)
    tier = np.where(expand & imp, 0, np.where(expand, 1, np.where(with_, 2, 3)))
    gate = np.where(~cv_ready, np.nan,
                    np.where((tier != 3) & ~quiet, np.where(above, 1.0, -1.0), 0.0))
    m = cv_mag.to_numpy()
    thr_ = cv_thr.to_numpy()
    ink = np.select([tier == 0, tier == 1, tier == 2],
                    [_g_ink(m, thr_, cv_hi, 18.0, 0.0),
                     _g_ink(m, np.zeros(T), np.maximum(thr_, 1e-9), 60.0, 38.0),
                     _g_ink(m, np.zeros(T), cv_hi, 70.0, 50.0)],
                    _g_ink(m, np.zeros(T), cv_hi, 88.0, 72.0))
    g_push = np.where(~cv_ready, np.nan, np.where(above, 1.0, -1.0) * ink * np.where(quiet, 0.45, 1.0))

    # the momentum tapes: faster − slower. Ladder up (default): chart − tape; a conviction
    # ladder looking DOWN is the faster view, so there it is tape − chart (the Pine's ladMom).
    m_c = out["c_tape"]
    m_v = out["v_tape"]
    c_ready = out["c_ready"].fillna(False).to_numpy(bool)
    v_ready = out["v_ready"].fillna(False).to_numpy(bool)
    ld = np.broadcast_to(np.asarray(ladder_down, dtype=bool), (T,))     # per bar: ↺ bars read up
    lad_mom = pd.Series(np.where(ld, (m_c - cv).to_numpy(dtype=float), (cv - m_c).to_numpy(dtype=float)),
                        index=cv.index)
    lm_ready = c_ready & cv_ready
    lm_sd = lad_mom.where(lm_ready).rolling(p.norm, min_periods=p.norm).std(ddof=0)
    lm_thr = (p.k * lm_sd.fillna(0.0)).to_numpy()
    tp_mom = out["value"] - m_v
    tp_thr = (0.5 * _stdev(tp_mom, 200).fillna(0.0)).to_numpy()
    lmv = lad_mom.to_numpy(dtype=float)
    tpv = tp_mom.to_numpy(dtype=float)
    mc = m_c.to_numpy(dtype=float)
    mv = m_v.to_numpy(dtype=float)
    read = c_ready & v_ready

    def side(prev, x, knee):
        if not np.isfinite(x):
            return 0.0
        if not np.isfinite(knee) or knee <= 0:
            return float(np.sign(x))
        if x >= knee:
            return 1.0
        if x <= -knee:
            return -1.0
        return prev if prev != 0.0 else float(np.sign(x))

    row = np.full(T, np.nan); col = np.full(T, np.nan); held = np.zeros(T, bool)
    u = np.full(T, np.nan); cell_flat = np.full(T, np.nan)
    c_ph = np.zeros(T, int); v_ph = np.zeros(T, int)
    g_row = None
    sc = sv_ = 0.0
    ink_c = _tape_ink(mc, p.z1, p.z2)
    ink_v = _tape_ink(mv, th, 70.0)
    for t in range(T):
        if not read[t]:
            g_row, sc, sv_ = None, 0.0, 0.0
            continue
        tgt = 1 if mc[t] >= p.z1 else (-1 if mc[t] <= -p.z1 else 0)
        gc = 0 if mv[t] <= -th else (2 if mv[t] >= th else 1)
        gg = gate[t]
        if g_row is None or not np.isfinite(gg):
            g_row = tgt
        elif tgt > g_row and gg > 0:
            g_row = tgt
        elif tgt < g_row and gg < 0:
            g_row = tgt
        sc = side(sc, lmv[t] if lm_ready[t] else np.nan, lm_thr[t])
        sv_ = side(sv_, tpv[t] if v_ready[t] else np.nan, tp_thr[t])
        hd = g_row != tgt
        cph = 0 if (g_row == 0 or sc == 0.0) else (1 if sc == g_row else -1)
        vph = 0 if (gc == 1 or sv_ == 0.0) else (1 if sv_ == (1.0 if gc == 0 else -1.0) else -1)
        if hd:
            hs = 1.0 if g_row > tgt else -1.0
            hh = 0.0 if not np.isfinite(g_push[t]) else min(1.0, max(0.0, g_push[t] * hs))
            nb = g_row + (1 if tgt > g_row else -1)
            pos_r = g_row * (0.5 + 0.5 * hh) + nb * (0.5 - 0.5 * hh)
        elif g_row != 0:
            pos_r = g_row * ink_c[t]
        else:
            pos_r = (1.0 if mc[t] > 0 else -1.0) * (ink_c[t] - 0.12)
        pos_c = (gc - 1) * ink_v[t] if gc != 1 else (1.0 if mv[t] > 0 else -1.0) * (ink_v[t] - 0.12)
        if g_row != 0 and cph == -1:
            pos_r *= 0.5
        if gc != 1 and vph == -1:
            pos_c *= 0.5
        row[t], col[t], held[t], c_ph[t], v_ph[t] = g_row, gc, hd, cph, vph
        u[t] = _g_at(pos_r, pos_c, units)
        cell_flat[t] = units[(g_row, gc)]
    return pd.DataFrame({"row": row, "col": col, "held": held, "units": u,
                         "cell_units": cell_flat, "cph": c_ph, "vph": v_ph,
                         "push": g_push, "gate": gate, "quiet": quiet}, index=out.index)


def classify(out: pd.DataFrame, cv: pd.Series, raw_sd: pd.Series, cv_ready: np.ndarray,
             p) -> pd.DataFrame:
    """The grid, bar by bar, on pragati.compute's output.

    ``cv`` is chart conviction (the Nishchaya v3 oscillator), ``raw_sd`` its raw share's σ,
    ``cv_ready`` its calibration gate. Columns: cvg_cell (0-8, 9 unread), cvg_units (the
    graded units), cvg_held, cvg_since, cvg_from, cvg_chart (where this chart alone would
    place the name), cvg_lead (+1 the chart cell carries more units, −1 fewer), cvg_cph /
    cvg_vph (the 5 × 5 phases: +1 confirmed, −1 not, 0 no edge), cvg_push (the drawn
    push that runs the rows, −1 … +1) and cvg_quiet (conviction's quiet regime — the Pine's
    gQuiet, which also gates the ▲).
    """
    ld = (out["c_ladder"] == "down").to_numpy() if "c_ladder" in out.columns else False
    g = _engine(out, cv, raw_sd, cv_ready, p, UNITS_RC, ladder_down=ld)
    T = len(g)
    row = g["row"].to_numpy(dtype=float)
    col = g["col"].to_numpy(dtype=float)
    ok = np.isfinite(row) & np.isfinite(col)
    cell = np.where(ok, (np.nan_to_num(row) + 1) * 3 + np.nan_to_num(col), UNREAD).astype(int)
    since = np.zeros(T, dtype=int)
    frm = np.full(T, UNREAD, dtype=int)
    cur, prev, st = UNREAD, UNREAD, 0
    for t in range(T):
        if cell[t] != cur:
            prev, cur, st = cur, cell[t], t
        since[t], frm[t] = st, prev
    th = float(p.theta)
    cc = cv.to_numpy(dtype=float)
    vv = out["value"].to_numpy(dtype=float)
    cok = cv_ready & out["value_built"].fillna(False).to_numpy(bool) & np.isfinite(cc) & np.isfinite(vv)
    chart = np.where(cok, np.array([row_of(a, p.z1) * 3 + col_of(b, th) if k else UNREAD
                                    for a, b, k in zip(cc, vv, cok)]), UNREAD).astype(int)
    u = np.asarray(UNITS)
    lead = np.where((cell != UNREAD) & (chart != UNREAD), np.sign(u[chart] - u[cell]), 0).astype(int)
    return pd.DataFrame({"cvg_cell": cell, "cvg_units": g["units"].to_numpy(),
                         "cvg_held": g["held"].to_numpy(bool), "cvg_since": since,
                         "cvg_from": frm, "cvg_chart": chart, "cvg_lead": lead,
                         "cvg_cph": g["cph"].to_numpy(), "cvg_vph": g["vph"].to_numpy(),
                         "cvg_push": g["push"].to_numpy(),
                         "cvg_quiet": g["quiet"].to_numpy(bool)}, index=out.index)


def tooltip(cell: int, units: float, bars: int, frm: int, chart: int, lead: int,
            held: bool) -> str:
    """The Pine's gTip: what the state means, its units, and how to read the push."""
    cell, frm, chart = int(cell), int(frm), int(chart)
    if cell == UNREAD:
        return MEANING[UNREAD]
    s = (f"{NAMES[cell]} ({FAMILY[cell]}) - {MEANING[cell]}.\n\n{units:.3g} units, graded · "
         f"{UNITS[cell]:g} flat · {bars} bars here"
         + (f" · came from {NAMES[frm]}" if frm not in (UNREAD, cell) else ""))
    if chart != UNREAD:
        s += (f"\n\nCHART CELL: {NAMES[chart]} - where this chart's own conviction and value "
              "place it" + (". The chart is already in a cell carrying more units: it LEADS ↑."
                            if lead > 0 else
                            ". The chart is already in a cell carrying fewer units: it LEADS ↓."
                            if lead < 0 else "."))
    s += "\n\n" + READ_THE_PUSH
    if held:
        s += ("\n\nHELD: the conviction tape has moved to another row, but conviction's "
              "histogram has not confirmed a push that way, so the row stands.")
    return s


__all__ = [
    "ACTIONS", "ACTION_TONE", "ACTION_UNITS", "COLUMNS", "COL_LABELS", "FAMILY", "MEANING",
    "NAMES", "N_COLS", "N_ROWS", "PUSH_GLYPH", "PUSH_TEXT", "READ_THE_PUSH", "ROW_LABELS",
    "SIDES", "TONES", "TONE_CHIP", "UNITS", "UNITS_RC", "UNREAD", "action", "classify",
    "col_of", "reason", "row_of", "tooltip",
]
