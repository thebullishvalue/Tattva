"""
TATTVA — तत्त्व. One 10-day call per target, from the convergence of the world (Mūla) and the
self (Swayam), with its own track record beside it.

    streamlit run app.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from tattva import VERSION, charts, engine
from tattva import evaluate as ev
from tattva.data import fetch_symbol, fetch_universe, resolve_symbol
from tattva.features import HORIZON
from tattva.universe import CATEGORIES, FACTORS, STOCK_CATEGORIES, TARGETS_BY_NAME

warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="Tattva", page_icon="◈", layout="wide")


# ── data & compute (cached) ──────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=6 * 3600, show_spinner="Fetching market data (≈45 series since 2008)…")
def universe(stamp: int) -> dict[str, pd.DataFrame]:
    return fetch_universe(force=stamp > 0)


@st.cache_resource(ttl=6 * 3600, show_spinner="Scoring every catalogue target's history…")
def catalogue(stamp: int):
    runs = engine.run_catalogue(universe(stamp))
    pools = {n: TARGETS_BY_NAME[n].pool for n in runs}
    return runs, ev.table(runs, pools)


@st.cache_resource(ttl=6 * 3600, show_spinner="Reading the stock…")
def stock_run(stamp: int, ticker: str, label: str, category: str):
    bars = fetch_symbol(ticker)
    if bars is None or len(bars) < 300:
        return None
    return engine.analyze(label, ticker, category, bars, engine.world_close(universe(stamp)))


def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.{nd}f}%"


# ── sidebar ──────────────────────────────────────────────────────────────────────────────────
stamp = st.session_state.setdefault("stamp", 0)
with st.sidebar:
    st.markdown("### TATTVA · तत्त्व")
    st.caption("One 10-day call per target: the world and the self, converged.")
    category = st.selectbox("Asset class", list(CATEGORIES))
    run = None
    if category in STOCK_CATEGORIES:
        market = STOCK_CATEGORIES[category]
        raw = st.text_input("Symbol", placeholder="RELIANCE" if market == "india" else "AAPL")
        if raw:
            ticker, info = resolve_symbol(raw, market)
            if ticker is None:
                st.error(info)
            else:
                run = stock_run(stamp, ticker, f"{raw.strip().upper()} ({info})", category)
                if run is None:
                    st.error(f"{ticker}: not enough history (needs 300+ daily bars).")
    else:
        name = st.selectbox("Target", CATEGORIES[category])
        runs, _ = catalogue(stamp)
        run = runs.get(name)
        if run is None:
            st.warning(f"{name}: no data from its source right now "
                       f"({TARGETS_BY_NAME[name].ticker}). Try again later.")
    st.divider()
    if st.button("Refresh data", width="stretch"):
        st.session_state["stamp"] = stamp + 1
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"v{VERSION} · yfinance, unadjusted daily bars · @thebullishvalue")

# ── header ───────────────────────────────────────────────────────────────────────────────────
if run is None:
    st.title("TATTVA · तत्त्व")
    st.write("Pick a target in the sidebar. Tattva makes one directional call for the next "
             f"{HORIZON} trading days — **BUY**, **SELL** or **NO CALL** — by converging two "
             "independent reads, and shows how its past calls on that target turned out.")
    st.stop()

now = run.current()
st.title(run.name)
if not now:
    st.warning("Not enough history yet to read this target (Tattva needs about a year of bars).")
    st.stop()
st.caption(f"Last close {now['close']:,.4g} on {now['date']:%a %d %b %Y} · ticker {run.ticker} · "
           f"call horizon: next open → {HORIZON} sessions later")

tab_call, tab_mula, tab_swayam, tab_record, tab_about = st.tabs(
    ["Call", "Mūla · world", "Swayam · self", "Record", "How it works"])

# ── Call ─────────────────────────────────────────────────────────────────────────────────────
with tab_call:
    call = now["call"]
    lean = "up" if now["Ey"] > 0 else "down"
    if call == 1:
        verdict, tone = "▲ BUY", "green"
    elif call == -1:
        verdict, tone = "▼ SELL", "red"
    else:
        verdict, tone = "● NO CALL", "gray"
    tdir = "up" if now["z"] > 0 else "down"
    if call and np.sign(now["z"]) == call:
        reason = f"Mūla and Swayam agree, strongly, and lean {tdir} with this target's usual drift."
    elif call:
        reason = (f"Mūla and Swayam lean {tdir} together, but by less than this target's usual "
                  f"{'up' if now['mu'] > 0 else 'down'}-drift: the expected move keeps the drift's "
                  "direction, smaller than usual.")
    elif not now["agree"]:
        reason = "Mūla and Swayam disagree."
    else:
        reason = f"Timing is weak (|z| {abs(now['z']):.2f} < {engine.CALL_Z})."
    with st.container(border=True):
        c1, c2 = st.columns([1.2, 2])
        with c1:
            st.markdown(f"## :{tone}[{verdict}]")
            st.markdown(f"{reason}  \nExpected lean: **{lean}**, {pct(now['exp_move'])} over "
                        f"{HORIZON} sessions (typical 10-day swing ±{100 * now['sigma10']:.1f}%).")
        with c2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected 10-day move", pct(now["exp_move"]),
                      help="Drift + timing: the target's own average 10-day move plus the engines' "
                           "timing, in its own volatility units, converted to percent.")
            m2.metric("Conviction z", f"{now['z']:+.2f}",
                      help=f"Timing in units of its own typical size; a call needs |z| ≥ {engine.CALL_Z}.")
            m3.metric("Mūla", f"{now['M']:+.2f}", help="World-relative: + = lagged its world → leans up.")
            m4.metric("Swayam", f"{now['S']:+.2f}", help="Self: + = stretched down → leans up.")
    st.plotly_chart(charts.price_with_calls(run), width="stretch", theme="streamlit")
    a, b = st.columns(2)
    a.plotly_chart(charts.engines(run), width="stretch", theme="streamlit")
    b.plotly_chart(charts.conviction(run, engine.CALL_Z), width="stretch", theme="streamlit")

    T1 = ev.table({run.name: run})
    if not T1.empty:
        tot = T1[["tattva_calls", "tattva_hits", "tattva_pnl", "drift_calls", "drift_hits",
                  "long_calls", "long_hits"]].sum()
        st.subheader("This target's record")
        r1, r2, r3 = st.columns(3)
        r1.metric("Calls scored", int(tot["tattva_calls"]),
                  help="Non-overlapping: every 10th session since 2012.")
        hit = tot["tattva_hits"] / tot["tattva_calls"] if tot["tattva_calls"] else np.nan
        base = tot["drift_hits"] / tot["drift_calls"] if tot["drift_calls"] else np.nan
        r2.metric("Hit rate", pct(hit, 0).lstrip("+") if np.isfinite(hit) else "—",
                  delta=(f"{100 * (hit - base):+.1f} pts vs drift alone" if np.isfinite(hit) and np.isfinite(base) else None))
        r3.metric("Mean 10-day return per call", pct(tot["tattva_pnl"] / tot["tattva_calls"], 2)
                  if tot["tattva_calls"] else "—")
        led = ev.target_record(run).head(12)
        if not led.empty:
            show = pd.DataFrame({
                "Date": led.index.strftime("%d %b %Y"),
                "Call": np.where(led["call"] > 0, "▲ BUY", "▼ SELL"),
                "Realised 10-day": (100 * led["ret"]).map("{:+.2f}%".format),
                "Outcome": np.where(led["hit"], "✓ right", "✗ wrong")})
            with st.expander("Latest scored calls"):
                st.dataframe(show, hide_index=True, width="stretch")

# ── Mūla ─────────────────────────────────────────────────────────────────────────────────────
with tab_mula:
    st.markdown("**Mūla (मूल, root) reads the target against its world.** Every day it "
                "re-estimates how the target moves with ~25 global markets, then asks how far "
                "the target has run *beyond or short of* what those markets implied over the "
                "last 60 days. Overshooting the world tends to be given back.")
    c1, c2, c3 = st.columns(3)
    c1.metric("60-day gap vs world", f"{now['gap60']:+.2f} σ",
              help="Sum of the last 60 days' surprises, in units of their own typical size. "
                   "+ = ran ahead of its world.")
    c2.metric("Mūla signal", f"{now['M']:+.2f}", help="−gap, scaled: + leans up.")
    fv = now["fair_value"]
    c3.metric("World-implied price", f"{fv:,.4g}" if fv else "—",
              delta=(f"{100 * (fv / now['close'] - 1):+.1f}% vs close" if fv else None))
    st.plotly_chart(charts.fair_value(run), width="stretch", theme="streamlit")
    a, b = st.columns([3, 2])
    a.plotly_chart(charts.contributions(run), width="stretch", theme="streamlit")
    with b:
        u = run.mula["usable"].iloc[-1]
        off = [n for n, ok in u.items() if not ok]
        st.markdown("**Factors in the world**")
        st.caption(", ".join(n for n in FACTORS if n in u.index and u[n]) or "—")
        if off:
            st.markdown("**Left out for this target**")
            st.caption(", ".join(off) + " — each is the target itself or moves with it almost "
                       "one-for-one (|ρ| > 0.95 over the last year), so it would explain the "
                       "target with itself.")

# ── Swayam ───────────────────────────────────────────────────────────────────────────────────
with tab_swayam:
    st.markdown("**Swayam (स्वयम्, self) reads the target's own market, nothing else.** It "
                "asks how stretched the price is — its 60- and 250-day trend and its place in "
                "the 60-day range, each against its own typical size. Over ten days, stretch "
                "tends to be given back.")
    c1, c2 = st.columns([1, 2])
    c1.metric("Swayam signal", f"{now['S']:+.2f}", help="+ = stretched down → leans up.")
    c2.plotly_chart(charts.swayam_parts(now["parts"]), width="stretch", theme="streamlit")
    st.plotly_chart(charts.range_band(run), width="stretch", theme="streamlit")

# ── Record ───────────────────────────────────────────────────────────────────────────────────
with tab_record:
    _, T = catalogue(stamp)
    S = ev.summary(T)
    st.markdown("Every catalogue target, every period, scored on **non-overlapping** 10-day "
                "outcomes (one every 10 sessions). IC is the rank correlation of a forecast with "
                "what happened; 0 is no skill. The periods are P0 2012–mid-2019, P1 mid-2019–2021, "
                "P2 2022–23 and P3 2024 onward.")
    a, b = st.columns(2)
    a.plotly_chart(charts.ic_by_period(S), width="stretch", theme="streamlit")
    b.plotly_chart(charts.hit_rates(S), width="stretch", theme="streamlit")
    tbl = S.rename(columns={"M": "Mūla IC", "S": "Swayam IC", "F": "Tattva IC",
                            "tattva_hit": "Hit rate", "tattva_ret%": "Return/call %",
                            "drift_hit": "Drift-only hit", "long_hit": "Always-long hit",
                            "long_ret%": "Always-long return %", "calls": "Calls", "targets": "Targets"})
    cols = ["Targets", "Mūla IC", "Swayam IC", "Tattva IC", "Calls", "Hit rate", "Return/call %",
            "Drift-only hit", "Always-long hit", "Always-long return %"]
    st.dataframe(tbl[cols].style.format({c: "{:+.3f}" for c in ("Mūla IC", "Swayam IC", "Tattva IC")}
                                        | {c: "{:.1%}" for c in ("Hit rate", "Drift-only hit", "Always-long hit")}
                                        | {c: "{:+.2f}" for c in ("Return/call %", "Always-long return %")}),
                 width="stretch")
    st.markdown("**By asset pool**")
    P = ev.summary(T, "pool").rename(columns={"M": "Mūla IC", "S": "Swayam IC", "F": "Tattva IC",
                                              "tattva_hit": "Hit rate", "calls": "Calls"})
    st.dataframe(P[["Mūla IC", "Swayam IC", "Tattva IC", "Calls", "Hit rate"]]
                 .style.format({"Mūla IC": "{:+.3f}", "Swayam IC": "{:+.3f}", "Tattva IC": "{:+.3f}",
                                "Hit rate": "{:.1%}"}), width="stretch")
    st.info("Where the edge is and isn't: timing skill is modest (IC ≈ 0.01–0.06) and strongest on "
            "equity indices. Crypto shows none. Calls beat drift alone in every period, but in the "
            "2012–2021 bull runs simply holding long had a higher hit rate. The full study, "
            "including the pre-registered design that failed, is in "
            "research/tattva_v3_study.md.", icon="ℹ️")

# ── How it works ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(f"""
**What Tattva answers.** For one target: over the next {HORIZON} sessions, up or down, and is
there enough evidence to say so?

**Two independent reads.**
- **Mūla (world).** The target's 60-day run beyond or short of what ~25 global markets implied,
  from an exposure model re-estimated every day on past data only.
- **Swayam (self).** The target's own stretch: its 60- and 250-day trend and its place in its
  60-day range.

**Convergence.** Timing = the average of the two. The call adds timing to the target's own
average 10-day move, because a BUY or SELL is judged on the actual move:
expected move = drift + {engine.K_TIMING} × z (volatility units). Tattva calls only when
|z| ≥ {engine.CALL_Z} **and** the two reads agree; otherwise it says NO CALL.

**Guarantees.**
- Every number on a date uses only bars up to that date.
- Prices are unadjusted, so published history never moves. `tests/test_causality.py` re-runs on
  truncated data and requires exact agreement.

**Limits.**
- Modest, uneven edge. None on crypto.
- A research tool, not investment advice.
""")
