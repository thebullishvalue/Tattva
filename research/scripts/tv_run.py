"""Headless Tattva baseline: the app's data prep, the CURRENT Mūla and Swayam, saved per target.

Replicates app.py's preparation (target + active features, session spine, ffill, the
MIN_DATA_POINTS history guard, raw-yield drop) and runs MulaEngine and the Swayam bank
exactly as Phases 1-3 do. Saves: the Mūla ts_data (dated), the aggregated Swayam breadth,
and the target's own OHLCV (for the new engines and for scoring).

usage: python tv_run.py "Gold" "Silver" ...     (or no args: the default target list)
"""
import os, sys, pickle, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/home/user/tattva"); os.chdir("/home/user/tattva")
import numpy as np, pandas as pd
from core.config import (ALL_TARGETS, TARGET_EXCLUDED_PREDICTORS, RAW_YIELD_PREDICTORS, MIN_DATA_POINTS,
                         get_instrument_config, swayam_macro_columns)
from data.fetcher import fetch_commodity_dataset, fetch_macro_live, fetch_constituent_ohlcv
from data.calendars import session_mask
from engines.mula import MulaEngine
from engines.swayam import aggregate_views, build_swayam_frames, default_swayam_members
from engines.swayam.kernel import view_skill_weights

OUT = "/tmp/claude-0/-home-user/33a407fc-2a93-57b3-a635-d4a00f7757de/scratchpad/tv/base"
os.makedirs(OUT, exist_ok=True)
END = pd.Timestamp("2026-09-29"); START = END - pd.DateOffset(days=365 * 9)
_DS = {}


def dataset():
    if "df" not in _DS:
        df, err = fetch_commodity_dataset(START, END)
        assert df is not None, err
        _DS["df"] = df
        _DS["macro"] = fetch_macro_live(START, END)
    return _DS["df"], _DS["macro"]


def prep(target):
    df, _ = dataset()
    date_col = "DATE" if "DATE" in df.columns else [c for c in df.columns if "date" in c.lower()][0]
    numeric = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in numeric if c != target]
    excl = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    feats = [f for f in feats if f not in excl]
    data = df[[target] + feats + [date_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", dayfirst=True)
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    for c in [target] + feats:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    m = session_mask(ALL_TARGETS.get(target), data[date_col])
    if m.any():
        data = data[m].reset_index(drop=True)
    data[[target] + feats] = data[[target] + feats].ffill()
    win = min(MIN_DATA_POINTS, len(data))
    feats = [f for f in feats if data[f].tail(win).notna().all()]
    data = data.dropna(subset=[target] + feats).reset_index(drop=True)
    feats = [f for f in feats if f not in RAW_YIELD_PREDICTORS]
    return data, feats, date_col


def run(target):
    t0 = time.time()
    cfg = get_instrument_config(target)
    data, feats, dc = prep(target)
    cal = pd.to_datetime(data[dc].values)
    px = pd.Series(data[target].to_numpy(float), index=cal, name=target)
    X = data[feats].astype(float); X.index = cal
    eng = MulaEngine()
    eng.fit(px, X, feature_names=feats, config=cfg)
    ts = eng.ts_data.copy(); ts["Price"] = px.to_numpy(); ts.index = cal
    # Swayam — the target's own OHLCV and the macro frame, as Phase 1/3
    _, macro = dataset()
    tk = ALL_TARGETS[target]
    ohlcv = fetch_constituent_ohlcv([tk], START, END).get(tk)
    sw = pd.DataFrame(); nviews = 0
    if ohlcv is not None and not ohlcv.empty:
        cols = swayam_macro_columns(target, list(macro.columns))
        mem = default_swayam_members(cfg.swayam_lengths, cfg.swayam_roc_frac)
        views = build_swayam_frames(ohlcv, macro, cols, members=mem,
                                    regime_sensitivity=cfg.swayam_regime_sensitivity,
                                    base_weight=cfg.swayam_base_weight, num_vars=cfg.swayam_mmr_num_vars,
                                    oversold=cfg.swayam_oversold, overbought=cfg.swayam_overbought)
        w = view_skill_weights(views, horizon=cfg.forecast_horizon)
        sw = aggregate_views(views, weights=w); nviews = len(views)
    pickle.dump({"target": target, "ticker": tk, "mula": ts, "swayam": sw, "ohlcv": ohlcv,
                 "n_feats": len(feats), "n_views": nviews, "h": cfg.forecast_horizon},
                open(f"{OUT}/{target.replace('/', '_').replace(' ', '_')}.pkl", "wb"))
    print(f"{target:18s} rows {len(ts):5d} valid {int(ts['Valid'].sum()):5d} feats {len(feats):3d} "
          f"views {nviews:2d} swayam {len(sw):5d}  [{time.time() - t0:.0f}s]", flush=True)


DEFAULT = ["Gold", "Silver", "Copper", "Brent Crude", "Cotton", "USD/INR", "Dollar Index", "Aluminium", "Zinc",
           "Bitcoin", "Ethereum"]
if __name__ == "__main__":
    tg = sys.argv[1:] or DEFAULT
    for t in tg:
        try:
            run(t)
        except Exception as e:
            print(f"{t:18s} FAILED {type(e).__name__}: {e}", flush=True)
