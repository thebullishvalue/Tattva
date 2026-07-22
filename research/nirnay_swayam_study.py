"""
Tattva — Nirnay-Swayam A/B efficacy study (NIRNAY_SWAYAM_PLAN.md §7).

Question: on targets where BOTH modes are runnable (a basket exists AND the
target has its own tradeable OHLCV), does the self-referential Swayam
ensemble retain a comparable fraction of the basket-mode breadth signal's
predictive content — or does dropping cross-sectional independence for
multi-view agreement lose most of the edge?

Method: for each target, build BOTH nirnay_daily aggregates (existing basket
path vs engines.nirnay_self.build_swayam_frames on the target's own OHLCV,
using the SAME production knobs — NIRNAY_REGIME_SENSITIVITY etc. — as
app.py), then score each at H ∈ {10, 20} with a NON-OVERLAPPING (stride=H)
Spearman rank IC vs the target's forward log-return, on three read
dimensions: breadth spread, Avg_Signal, and regime spread. Cross-target
means are compared against the §7.2 acceptance gates:

  G1 — mean |IC_breadth|(Swayam) >= 0.75 * mean |IC_breadth|(basket) at
       BOTH H=10 and H=20.
  G2 — sign of the cross-target mean SIGNED IC agrees between modes at
       both horizons (Swayam must not be anti-signal).
  G3 — on >=half the targets, Swayam Avg_Signal correlates >=0.4 with
       basket Avg_Signal (same-regime sanity, not equality).
  G4 — structural smoke test on >=2 stock targets: the Swayam ensemble
       builds cleanly on real OHLCV with a non-degenerate breadth read.
       (The FULL intelligence-calibration Val IC check needs the app.py
       orchestration — Aarambh walk-forward + PCA + purge — which isn't
       factored into an importable API; confirm that leg by running
       `streamlit run app.py` on a stock target and checking the
       Diagnostics -> Intelligence Center Val IC / walk-forward chart.)

This is a REPORT-ONLY study (repo convention — see README's "Re-tuning"
section): it prints the gate verdicts; shipping NIRNAY_SWAYAM_FALLBACK=True
or promoting Swayam beyond individual-stock targets is a config change
applied by hand after review, never by this script.

Run: python3 -u research/nirnay_swayam_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from data.fetcher import fetch_commodity_dataset, fetch_macro_live, fetch_constituent_ohlcv
from data.constituents import get_commodity_basket
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries
from engines.nirnay_self import build_swayam_frames, default_swayam_members
from core.config import (
    ALL_TARGETS, register_stock_target, swayam_macro_columns,
    NIRNAY_MSF_LENGTH, NIRNAY_ROC_LEN, NIRNAY_REGIME_SENSITIVITY,
    NIRNAY_BASE_WEIGHT, NIRNAY_MMR_NUM_VARS, NIRNAY_OVERSOLD, NIRNAY_OVERBOUGHT,
)

# Targets where BOTH a basket AND the target's own tradeable OHLCV exist —
# the only population G1-G3 can be measured on (individual stocks have no
# basket to compare against; that's the whole point of Swayam).
AB_TARGETS = ["Gold", "Silver", "Copper", "Cotton", "USD/INR", "Brent Crude", "Jeera",
              "Nifty 50", "Nifty Bank"]
HORIZONS = (10, 20)
G1_RATIO = 0.75
G3_MIN_CORR = 0.4
G3_MIN_FRACTION = 0.5


_DATA = {}
def _load():
    if "df" not in _DATA:
        end = pd.Timestamp.today(); start = end - pd.Timedelta(days=365 * 9)
        df, err = fetch_commodity_dataset(start, end)
        if df is None:
            raise SystemExit(err)
        macro = fetch_macro_live(start, end)
        _DATA["df"] = df
        _DATA["macro"] = macro if macro is not None else pd.DataFrame()
        _DATA["macro_cols"] = list(_DATA["macro"].columns)
        _DATA["start"], _DATA["end"] = start, end
    return _DATA


def _target_price(target: str) -> pd.Series:
    d = _load()
    tgt = d["df"][["DATE", target]].dropna().copy()
    tgt["DATE"] = pd.to_datetime(tgt["DATE"])
    return tgt.set_index("DATE")[target].astype(float)


def _basket_daily(target: str) -> pd.DataFrame:
    """Basket-mode nirnay_daily — the existing production path, unchanged."""
    d = _load()
    cons, _src = get_commodity_basket(target)
    if not cons:
        return pd.DataFrame()
    ohlcv = fetch_constituent_ohlcv(cons, d["start"], d["end"]) or {}
    frames = {}
    for sym, odf in ohlcv.items():
        merged = odf.copy()
        if not d["macro"].empty:
            merged = merged.join(d["macro"], how="left")
            merged[d["macro_cols"]] = merged[d["macro_cols"]].ffill()
        try:
            res, _ = run_full_analysis(
                merged, length=NIRNAY_MSF_LENGTH, roc_len=NIRNAY_ROC_LEN,
                regime_sensitivity=NIRNAY_REGIME_SENSITIVITY, base_weight=NIRNAY_BASE_WEIGHT,
                num_vars=NIRNAY_MMR_NUM_VARS, oversold=NIRNAY_OVERSOLD, overbought=NIRNAY_OVERBOUGHT,
                macro_columns=d["macro_cols"],
            )
            frames[sym] = res
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return aggregate_constituent_timeseries(frames)


def _swayam_daily(target: str) -> pd.DataFrame:
    """Self-mode nirnay_daily — the target's own OHLCV through the ensemble."""
    d = _load()
    ticker = ALL_TARGETS.get(target)
    if not ticker:
        return pd.DataFrame()
    ohlcv_map = fetch_constituent_ohlcv([ticker], d["start"], d["end"]) or {}
    target_ohlcv = ohlcv_map.get(ticker)
    if target_ohlcv is None or target_ohlcv.empty:
        return pd.DataFrame()
    swayam_cols = swayam_macro_columns(target, d["macro_cols"])
    frames = build_swayam_frames(
        target_ohlcv, d["macro"], swayam_cols,
        regime_sensitivity=NIRNAY_REGIME_SENSITIVITY, base_weight=NIRNAY_BASE_WEIGHT,
        num_vars=NIRNAY_MMR_NUM_VARS, oversold=NIRNAY_OVERSOLD, overbought=NIRNAY_OVERBOUGHT,
    )
    if not frames:
        return pd.DataFrame()
    return aggregate_constituent_timeseries(frames)


def _non_overlapping_ic(daily: pd.DataFrame, col: str, price: pd.Series, h: int) -> float:
    if daily.empty or col not in daily.columns:
        return np.nan
    s = pd.to_numeric(daily[col], errors="coerce")
    s.index = pd.to_datetime(daily.index)
    s = s.reindex(price.index, method="ffill")
    pr, val = price.to_numpy(), s.to_numpy()
    n = len(pr)
    p, r = [], []
    for t in range(60, n - h, h):    # skip warmup; non-overlapping stride
        if pr[t] > 0 and np.isfinite(val[t]):
            p.append(val[t]); r.append((pr[t + h] / pr[t] - 1) * 100)
    p, r = np.array(p), np.array(r)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 12:
        return np.nan
    return float(spearmanr(p[m], r[m])[0])


def _breadth_spread(daily: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(daily.get("Oversold_Pct", 0), errors="coerce") - \
           pd.to_numeric(daily.get("Overbought_Pct", 0), errors="coerce")


def _regime_spread(daily: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(daily.get("Regime_Bull_Pct", 0), errors="coerce") - \
           pd.to_numeric(daily.get("Regime_Bear_Pct", 0), errors="coerce")


def _score_target(target: str) -> dict:
    price = _target_price(target)
    basket = _basket_daily(target)
    swayam = _swayam_daily(target)
    out = {"target": target, "basket_n": len(basket), "swayam_n": len(swayam)}
    if basket.empty or swayam.empty:
        return out

    for h in HORIZONS:
        out[f"basket_ic_breadth_{h}"] = _non_overlapping_ic(
            pd.DataFrame({"spread": _breadth_spread(basket)}, index=basket.index), "spread", price, h)
        out[f"swayam_ic_breadth_{h}"] = _non_overlapping_ic(
            pd.DataFrame({"spread": _breadth_spread(swayam)}, index=swayam.index), "spread", price, h)
        out[f"basket_ic_avg_{h}"] = _non_overlapping_ic(basket, "Avg_Signal", price, h)
        out[f"swayam_ic_avg_{h}"] = _non_overlapping_ic(swayam, "Avg_Signal", price, h)
        out[f"basket_ic_regime_{h}"] = _non_overlapping_ic(
            pd.DataFrame({"spread": _regime_spread(basket)}, index=basket.index), "spread", price, h)
        out[f"swayam_ic_regime_{h}"] = _non_overlapping_ic(
            pd.DataFrame({"spread": _regime_spread(swayam)}, index=swayam.index), "spread", price, h)

    # G3 same-regime sanity: correlate Avg_Signal series on the overlapping calendar.
    b_avg = pd.to_numeric(basket.get("Avg_Signal"), errors="coerce")
    s_avg = pd.to_numeric(swayam.get("Avg_Signal"), errors="coerce")
    b_avg.index = pd.to_datetime(basket.index)
    s_avg.index = pd.to_datetime(swayam.index)
    joined = pd.concat([b_avg.rename("b"), s_avg.rename("s")], axis=1).dropna()
    out["avg_signal_corr"] = float(joined["b"].corr(joined["s"])) if len(joined) > 20 else np.nan
    return out


def _run_ab_study() -> list[dict]:
    print(f"Nirnay-Swayam A/B study · {len(AB_TARGETS)} targets · "
          f"basket vs self-ensemble · non-overlapping IC at H={HORIZONS}", flush=True)
    rows = []
    t0 = time.time()
    for target in AB_TARGETS:
        try:
            row = _score_target(target)
        except Exception as e:
            row = {"target": target, "error": str(e)}
        rows.append(row)
        print(f"  [{target}] basket_n={row.get('basket_n', 0)} swayam_n={row.get('swayam_n', 0)}"
              + (f"  ERROR: {row['error']}" if "error" in row else ""), flush=True)
    print(f"  scored {len(rows)} targets in {time.time()-t0:.0f}s\n", flush=True)
    return rows


def _gate_report(rows: list[dict]) -> None:
    print("=" * 78)
    print("  NIRNAY-SWAYAM ACCEPTANCE GATES (NIRNAY_SWAYAM_PLAN.md §7.2)")
    print("=" * 78)

    all_g1_g2_pass = True
    for h in HORIZONS:
        basket_abs = [abs(r[f"basket_ic_breadth_{h}"]) for r in rows
                      if np.isfinite(r.get(f"basket_ic_breadth_{h}", np.nan))]
        swayam_abs = [abs(r[f"swayam_ic_breadth_{h}"]) for r in rows
                      if np.isfinite(r.get(f"swayam_ic_breadth_{h}", np.nan))]
        basket_signed = [r[f"basket_ic_breadth_{h}"] for r in rows
                         if np.isfinite(r.get(f"basket_ic_breadth_{h}", np.nan))]
        swayam_signed = [r[f"swayam_ic_breadth_{h}"] for r in rows
                         if np.isfinite(r.get(f"swayam_ic_breadth_{h}", np.nan))]
        if not basket_abs or not swayam_abs:
            print(f"  H={h}: insufficient data (basket={len(basket_abs)}, swayam={len(swayam_abs)})")
            all_g1_g2_pass = False
            continue
        m_basket, m_swayam = np.mean(basket_abs), np.mean(swayam_abs)
        g1 = m_swayam >= G1_RATIO * m_basket
        g2 = np.sign(np.mean(basket_signed)) == np.sign(np.mean(swayam_signed))
        all_g1_g2_pass &= bool(g1) and bool(g2)
        print(f"  H={h}:  mean|IC_breadth| basket={m_basket:.3f}  swayam={m_swayam:.3f}  "
              f"ratio={m_swayam / m_basket if m_basket else float('nan'):.2f}  "
              f"G1={'PASS' if g1 else 'FAIL'}  G2={'PASS' if g2 else 'FAIL'}")

    corrs = [r["avg_signal_corr"] for r in rows if np.isfinite(r.get("avg_signal_corr", np.nan))]
    n_ok = sum(1 for c in corrs if c >= G3_MIN_CORR)
    g3 = len(corrs) > 0 and (n_ok / len(corrs)) >= G3_MIN_FRACTION
    print(f"\n  G3: Avg_Signal corr >= {G3_MIN_CORR} on {n_ok}/{len(corrs)} targets "
          f"(need >= {G3_MIN_FRACTION:.0%})  {'PASS' if g3 else 'FAIL'}")
    for r in rows:
        if np.isfinite(r.get("avg_signal_corr", np.nan)):
            print(f"      {r['target']:<14} corr={r['avg_signal_corr']:+.2f}")

    print(f"\n  VERDICT: {'SHIP' if (all_g1_g2_pass and g3) else 'DO NOT SHIP'} "
          f"(G1&G2 across horizons: {'PASS' if all_g1_g2_pass else 'FAIL'}, G3: {'PASS' if g3 else 'FAIL'})")
    if not (all_g1_g2_pass and g3):
        print("  If close: sweep the member grid (§7.3 / run this script's --sweep) before concluding.")


def _g4_smoke_test() -> None:
    print("\n" + "=" * 78)
    print("  G4 — STRUCTURAL SMOKE TEST on stock targets")
    print("=" * 78)
    # Individual stocks are free-form (resolve_stock_symbol), not a static
    # registry — probe two representative liquid names via the SAME
    # resolution path the sidebar uses, so this smoke test tracks reality.
    from data.universe import resolve_stock_symbol
    probes = [("RELIANCE", "india"), ("AAPL", "us")]
    stock_sample = []
    for raw, market in probes:
        ticker, exch_or_err = resolve_stock_symbol(raw, market)
        if ticker is None:
            print(f"  [{raw}] resolution FAILED — {exch_or_err}")
            continue
        base = ticker.rsplit(".", 1)[0] if market == "india" else raw.upper()
        display_name = f"{base} ({exch_or_err})"
        register_stock_target(display_name, ticker, market)
        stock_sample.append(display_name)
    for target in stock_sample:
        try:
            daily = _swayam_daily(target)
            if daily.empty:
                print(f"  [{target}] FAIL — empty aggregate")
                continue
            spread = _breadth_spread(daily)
            non_degenerate = spread.std() > 0.5   # not a flat/constant breadth read
            print(f"  [{target}] rows={len(daily)}  breadth_std={spread.std():.2f}  "
                  f"{'PASS' if non_degenerate else 'FAIL (degenerate breadth)'}")
        except Exception as e:
            print(f"  [{target}] FAIL — {e}")
    print("\n  NOTE: full Val-IC / walk-forward-chart confirmation needs the app.py\n"
          "  orchestration (Aarambh walk-forward + PCA + purge). Confirm by running\n"
          "  `streamlit run app.py`, selecting a stock target, and checking\n"
          "  Diagnostics -> Intelligence Center.")


def main() -> None:
    rows = _run_ab_study()
    _gate_report(rows)
    _g4_smoke_test()


if __name__ == "__main__":
    main()
