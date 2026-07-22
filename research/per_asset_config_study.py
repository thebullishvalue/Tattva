"""
Tattva — Per-asset-class config tuning (the InstrumentConfig registry).

Every instrument carries its own InstrumentConfig (core/config.py). This study
covers the classes NO OTHER study covers, at the RIGHT granularity:

  ASSET-LEVEL (pooled, one config per market — free-form symbols can't be pre-tuned):
  • stock_india — universe = the NIFTY 100 constituents (100 India stocks)
  • stock_us    — universe = the NASDAQ 100 constituents (US stocks)
      → sweep the Swayam grid (swayam_lengths span + swayam_roc_frac) on each
        stock's OWN OHLCV, so STOCK_CONFIGS["india"/"us"] is computed from a broad,
        representative universe, not a few names.

  PER-INSTRUMENT (each named target gets its own nirnay_msf_length):
  • us_index — S&P 500 / Nasdaq 100 / Dow Jones, each scored on its own basket
  • etf      — the India sector-ETF target
      → per member, emit a gated _PER_INSTRUMENT_OVERRIDES snippet.

The OTHER per-instrument classes are owned by a focused study (see COVERAGE_MAP
below) and NOT recomputed here:
  commodity (self)     → `swayam`      (per-commodity Swayam grid)
  india_index (basket) → `nirnay_index`(per-index MSF sweep)
  fx / Jeera (basket)  → `nirnay`      (basket-mode knob OFAT incl. USD/INR, Jeera)

Objective: NON-OVERLAPPING OOS rank IC of breadth spread (Oversold% − Overbought%)
vs the member's forward FORECAST_HORIZON-day return, averaged across the class's
members (mean |IC|). The winning knob value per class is the recommended
CLASS_CONFIG_DEFAULTS / STOCK_CONFIGS setting.

REPORT-ONLY periodic re-tune; HEAVY (a Swayam ensemble per stock across the
Nifty/Nasdaq 100). Each universe is batch-fetched in ONE yfinance call up front
(gentle on the rate-limiter); a thin prefetch is flagged UNRELIABLE rather than
silently trusted. For a fast smoke pass, cap the universe via the env var:
  STOCK_UNIVERSE_CAP=20 python3 -u research/per_asset_config_study.py  (0 = full)

Run: python3 -u research/per_asset_config_study.py
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: repo root on path so `from core...` resolves
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from data.fetcher import fetch_commodity_dataset, fetch_macro_live, fetch_constituent_ohlcv
from data.universe import resolve_index_constituents
from engines.nirnay import run_full_analysis, aggregate_constituent_timeseries
from engines.nirnay_self import build_swayam_frames, default_swayam_members
from core.config import (
    ALL_TARGETS, swayam_macro_columns, TARGET_EXCLUDED_PREDICTORS,
    NIRNAY_SWAYAM_LENGTHS, NIRNAY_SWAYAM_ROC_FRAC, NIRNAY_MSF_LENGTH,
    CLASS_CONFIG_DEFAULTS,
)
from research._per_instrument import (
    per_instrument_reco, merge_overrides, print_overrides_snippet,
)

H = 10   # forward horizon for the breadth-IC objective (fixed FORECAST_HORIZON)
STOCK_UNIVERSE_CAP = int(_os.environ.get("STOCK_UNIVERSE_CAP", "0"))  # 0 = full universe; env-overridable for a fast smoke pass
MIN_UNIVERSE_COVERAGE = 0.75  # if a batched prefetch returns < this fraction of the resolved universe, warn (thin snapshot → unreliable)

# Candidate knob values (compact — this study is per-class, and heavy).
SELF_LENGTH_SPANS = {
    "default-5": (10, 14, 20, 28, 40),   # current default
    "fast-5":    (5, 8, 12, 18, 28),
    "wide-7":    (8, 12, 18, 26, 36, 50, 70),
}
SELF_ROC_FRACS = [0.55, 0.7, 0.85]
BASKET_MSF_LENGTHS = [8, 12, 18, 26, 40]   # around the tuned default (18)

# Classes OWNED by this study (no other study covers them) → (mode, description,
# member resolver). Stock classes resolve a live index universe.
CLASS_SPEC = {
    "us_index":    ("basket", "US indices (constituents)",          ["S&P 500", "Nasdaq 100", "Dow Jones"]),
    "etf":         ("basket", "India sector-ETF universe",          ["India Sector ETFs"]),
    "stock_india": ("self",   "India stocks = NIFTY 100 universe",  "__NIFTY100__"),
    "stock_us":    ("self",   "US stocks = NASDAQ 100 universe",    "__NASDAQ100__"),
}

# Classes owned by a focused study — printed as a coverage map so "every class is
# tuned somewhere" is visible, without recomputing them here (no redundancy).
COVERAGE_MAP = {
    "commodity":   "swayam       (deep Swayam grid on the commodity futures)",
    "india_index": "nirnay_index (MSF sweep on the India indices)",
    "fx":          "nirnay       (basket-mode OFAT incl. USD/INR)",
    # Jeera rides the fx/basket knobs via `nirnay` (it's in that study's TARGETS).
}

_DATA: dict = {}


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


def _price_from_close(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _non_overlap_ic(breadth: pd.Series, price: pd.Series) -> float:
    breadth = breadth.reindex(price.index, method="ffill")
    pr, br = price.to_numpy(), breadth.to_numpy()
    n = len(pr)
    p, r = [], []
    for t in range(60, n - H, H):
        if pr[t] > 0 and np.isfinite(br[t]):
            p.append(br[t]); r.append((pr[t + H] / pr[t] - 1) * 100)
    p, r = np.array(p), np.array(r)
    m = np.isfinite(p) & np.isfinite(r)
    return float(spearmanr(p[m], r[m])[0]) if m.sum() >= 12 else np.nan


def _spread(daily: pd.DataFrame) -> pd.Series:
    s = (pd.to_numeric(daily.get("Oversold_Pct", 0), errors="coerce")
         - pd.to_numeric(daily.get("Overbought_Pct", 0), errors="coerce"))
    s.index = pd.to_datetime(daily.index)
    return s


# ── SELF-mode member scoring (own OHLCV → Swayam breadth) ────────────────────
_OHLCV: dict = {}
def _prefetch_ohlcv(members) -> float:
    """Batch-fetch a whole self-mode universe in ONE call, then read per-ticker
    from cache. Fetching 100+ names one-at-a-time (the old path) issued 100+
    yfinance round-trips and reliably tripped the rate-limiter / circuit breaker,
    collapsing the universe to a thin snapshot. One batched download is far
    gentler. Returns the coverage fraction so a thin result can be flagged rather
    than silently trusted."""
    d = _load()
    tickers = [ALL_TARGETS.get(m, m) for m in members]
    want = [t for t in tickers if t not in _OHLCV]
    if want:
        got = fetch_constituent_ohlcv(want, d["start"], d["end"]) or {}
        for t in want:
            _OHLCV[t] = got.get(t)   # None ⇒ fetch failed for this name
    have = sum(1 for t in tickers if _OHLCV.get(t) is not None and not _OHLCV[t].empty)
    frac = have / len(tickers) if tickers else 0.0
    tag = "" if frac >= MIN_UNIVERSE_COVERAGE else "  ⚠ THIN — likely rate-limited; re-run on healthy data"
    print(f"    prefetched OHLCV: {have}/{len(tickers)} names ({frac:.0%}){tag}", flush=True)
    return frac


def _fetch_ohlcv(ticker):
    """Cache reader. Populated in bulk by _prefetch_ohlcv; falls back to a single
    fetch only for the small hand-listed self classes that skip prefetch."""
    if ticker not in _OHLCV:
        d = _load()
        m = fetch_constituent_ohlcv([ticker], d["start"], d["end"]) or {}
        _OHLCV[ticker] = m.get(ticker)
    return _OHLCV[ticker]


def _self_member_ic(name_or_ticker, lengths, roc_frac, excluded) -> float:
    d = _load()
    ticker = ALL_TARGETS.get(name_or_ticker, name_or_ticker)
    ohlcv = _fetch_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        return np.nan
    members = default_swayam_members(tuple(lengths), float(roc_frac))
    drop = {name_or_ticker, *excluded}
    cols = [c for c in d["macro_cols"] if c not in drop]
    try:
        frames = build_swayam_frames(ohlcv, d["macro"], cols, members=members,
                                     regime_sensitivity=6.0, base_weight=0.0, num_vars=4,
                                     oversold=-5.0, overbought=5.0)
        daily = aggregate_constituent_timeseries(frames)
    except Exception:
        return np.nan
    if daily.empty:
        return np.nan
    return _non_overlap_ic(_spread(daily), _price_from_close(ohlcv["Close"]))


# ── BASKET-mode member scoring (constituents → breadth) ──────────────────────
_BASKET: dict = {}
def _basket_ohlcv(target):
    if target not in _BASKET:
        d = _load()
        syms, _ = resolve_index_constituents(target) if target not in ("USD/INR",) else ([], "")
        if target == "USD/INR":
            from data.constituents import get_commodity_basket
            syms, _ = get_commodity_basket(target)
        _BASKET[target] = fetch_constituent_ohlcv(syms, d["start"], d["end"]) or {}
    return _BASKET[target]


def _basket_member_ic(target, msf_length) -> float:
    d = _load()
    ohlcv = _basket_ohlcv(target)
    if not ohlcv:
        return np.nan
    excl = set(TARGET_EXCLUDED_PREDICTORS.get(target, []))
    macro_cols = [c for c in d["macro_cols"] if c not in excl and c != target]
    frames = {}
    for sym, odf in ohlcv.items():
        merged = odf.copy()
        if not d["macro"].empty:
            merged = merged.join(d["macro"], how="left")
            merged[d["macro_cols"]] = merged[d["macro_cols"]].ffill()
        try:
            res, _ = run_full_analysis(merged, length=int(msf_length), roc_len=2,
                                       regime_sensitivity=6.0, base_weight=0.0, num_vars=4,
                                       oversold=-5.0, overbought=5.0, macro_columns=macro_cols)
            frames[sym] = res
        except Exception:
            continue
    if not frames:
        return np.nan
    daily = aggregate_constituent_timeseries(frames)
    if daily.empty:
        return np.nan
    price = _price_from_close(d["df"].set_index(pd.to_datetime(d["df"]["DATE"]))[target]) \
        if target in d["df"].columns else None
    if price is None or price.empty:
        return np.nan
    return _non_overlap_ic(_spread(daily), price)


def _stock_universe(kind: str) -> list[str]:
    index_name = "Nifty 100" if kind == "__NIFTY100__" else "Nasdaq 100"
    syms, src = resolve_index_constituents(index_name)
    if STOCK_UNIVERSE_CAP and len(syms) > STOCK_UNIVERSE_CAP:
        syms = syms[:STOCK_UNIVERSE_CAP]
    print(f"    {index_name}: {len(syms)} constituents ({src})", flush=True)
    return syms


def _score_self_class(members, excluded):
    """Sweep swayam length span + roc_frac; return (best_len_label, best_roc)."""
    # length span (roc = default)
    print(f"    {'length span':<12} {'mean|IC|':>9} {'mean IC':>9}", flush=True)
    best_len = (None, -9.0)
    for label, lengths in SELF_LENGTH_SPANS.items():
        ics = [_self_member_ic(m, lengths, NIRNAY_SWAYAM_ROC_FRAC, excluded) for m in members]
        arr = np.array([x for x in ics if np.isfinite(x)])
        if not len(arr):
            continue
        mabs = float(np.mean(np.abs(arr)))
        print(f"    {label:<12} {mabs:>9.3f} {float(np.mean(arr)):>+9.3f}  (n={len(arr)})", flush=True)
        if mabs > best_len[1]:
            best_len = (label, mabs)
    # roc frac (lengths = default)
    print(f"    {'roc_frac':<12} {'mean|IC|':>9} {'mean IC':>9}", flush=True)
    best_roc = (None, -9.0)
    for roc in SELF_ROC_FRACS:
        ics = [_self_member_ic(m, NIRNAY_SWAYAM_LENGTHS, roc, excluded) for m in members]
        arr = np.array([x for x in ics if np.isfinite(x)])
        if not len(arr):
            continue
        mabs = float(np.mean(np.abs(arr)))
        print(f"    {str(roc):<12} {mabs:>9.3f} {float(np.mean(arr)):>+9.3f}  (n={len(arr)})", flush=True)
        if mabs > best_roc[1]:
            best_roc = (roc, mabs)
    return best_len, best_roc


def _score_basket_class(members):
    """Sweep MSF length PER MEMBER; return (class-best, {L: {member: ic}})."""
    print(f"    {'msf_length':<12} {'mean|IC|':>9} {'mean IC':>9}", flush=True)
    best = (None, -9.0)
    table: dict = {}       # {msf_value: {member: ic}} for the per-instrument reco
    for L in BASKET_MSF_LENGTHS:
        ics = {m: _basket_member_ic(m, L) for m in members}
        table[L] = ics
        arr = np.array([x for x in ics.values() if np.isfinite(x)])
        if not len(arr):
            continue
        mabs = float(np.mean(np.abs(arr)))
        print(f"    {str(L):<12} {mabs:>9.3f} {float(np.mean(arr)):>+9.3f}  (n={len(arr)})", flush=True)
        if mabs > best[1]:
            best = (L, mabs)
    return best, table


def main():
    _load()
    print("Per-asset-class config study · objective: breadth-spread IC vs "
          f"+{H}d return (non-overlapping)", flush=True)
    print(f"Defaults: swayam_lengths={NIRNAY_SWAYAM_LENGTHS} roc_frac={NIRNAY_SWAYAM_ROC_FRAC} "
          f"nirnay_msf_length={NIRNAY_MSF_LENGTH}", flush=True)
    t0 = time.time()
    recs: dict[str, str] = {}
    overrides: dict = {}       # per-INSTRUMENT overrides for the basket classes (us_index/etf)

    for cls, (mode, desc, spec) in CLASS_SPEC.items():
        print(f"\n### {cls}  [{mode}]  {desc}", flush=True)
        if mode == "self":
            # SELF here == the STOCK classes → tuned at ASSET-CLASS level (pooled
            # universe), never per instrument (free-form symbols can't be pre-tuned).
            if isinstance(spec, str):        # stock universe sentinel
                members = _stock_universe(spec)
                excluded = list(TARGET_EXCLUDED_PREDICTORS.get(
                    "Nifty 100" if spec == "__NIFTY100__" else "Nasdaq 100", []))
            else:
                members = spec
                excluded = []
            if not members:
                print("    (no members resolved — skipped)", flush=True)
                continue
            frac = _prefetch_ohlcv(members) if isinstance(spec, str) else 1.0
            best_len, best_roc = _score_self_class(members, excluded)
            warn = "" if frac >= MIN_UNIVERSE_COVERAGE else "  [UNRELIABLE: thin universe]"
            recs[cls] = (f"swayam_lengths={SELF_LENGTH_SPANS.get(best_len[0], best_len[0])} "
                         f"roc_frac={best_roc[0]}{warn}")
        else:
            # BASKET here == us_index / etf → PER-INSTRUMENT class: score each member
            # and emit its own nirnay_msf_length override (gated).
            best, table = _score_basket_class(spec)
            recs[cls] = f"nirnay_msf_length={best[0]} (class-level best)"
            merge_overrides(overrides, per_instrument_reco(
                "nirnay_msf_length", table, NIRNAY_MSF_LENGTH, set(spec)))

    print(f"\n  total {(time.time()-t0)/60:.1f} min", flush=True)
    print("\n" + "=" * 72)
    print("  RECOMMENDATIONS")
    print("=" * 72)
    print("  ASSET-LEVEL classes (stocks — pooled universe, one config per market):")
    for cls in ("stock_india", "stock_us"):
        print(f"    {cls:<14} {recs.get(cls, '(no result)')}")
    print("\n  PER-INSTRUMENT classes owned here (us_index / etf members):")
    for cls in ("us_index", "etf"):
        print(f"    {cls:<14} class-level: {recs.get(cls, '(no result)')}")
    print_overrides_snippet(overrides)
    print("\n  PER-INSTRUMENT classes owned ELSEWHERE (per target, not recomputed here):")
    for cls, where in COVERAGE_MAP.items():
        print(f"    {cls:<14} → {where}")
    print("\n  Coverage: stocks tuned at ASSET level (STOCK_CONFIGS); commodity/fx/"
          "india_index/\n  us_index/etf tuned PER INSTRUMENT (core.config._PER_INSTRUMENT_OVERRIDES via\n"
          "  the snippet above + the mapped studies). The orchestrator never auto-edits config.")
    # Sanity: every class this study + the coverage map name must be a real class.
    _known = set(CLASS_CONFIG_DEFAULTS)
    _missing = [c for c in (set(CLASS_SPEC) | set(COVERAGE_MAP)) if c not in _known]
    if _missing:
        print(f"  WARNING: classes not in CLASS_CONFIG_DEFAULTS: {_missing}")


if __name__ == "__main__":
    main()
