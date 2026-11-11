"""Headless replica of app.py's data prep + Mula fit, for audit experiments."""
import os, sys, time, pickle, warnings
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from core.config import ALL_TARGETS, MIN_DATA_POINTS, RAW_YIELD_PREDICTORS, get_instrument_config, STOCK_TARGET_MARKETS
from data.calendars import session_mask
from engines.mula import MulaEngine
SP = os.path.dirname(os.path.abspath(__file__))

def load_panel():
    p = os.path.join(SP, "panel.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from data.fetcher import fetch_commodity_dataset
    end = pd.Timestamp.today(); start = end - pd.DateOffset(days=365*9)
    df, err = fetch_commodity_dataset(start, end)
    assert df is not None, err
    pickle.dump(df, open(p, "wb"))
    return df

def prep(df, target, drop_head=0):
    cfg = get_instrument_config(target)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    excl = {target, *cfg.excluded_predictors}
    feats = [c for c in num if c not in excl and c not in STOCK_TARGET_MARKETS]
    data = df[[target] + feats + ["DATE"]].copy()
    data["DATE"] = pd.to_datetime(data["DATE"]); data = data.dropna(subset=["DATE"]).sort_values("DATE")
    if drop_head: data = data.iloc[drop_head:]
    sm = session_mask(ALL_TARGETS.get(target), data["DATE"])
    if sm.any(): data = data[sm].reset_index(drop=True)
    data[[target] + feats] = data[[target] + feats].ffill()
    win = min(MIN_DATA_POINTS, len(data))
    feats = [f for f in feats if data[f].tail(win).notna().all()]
    data = data.dropna(subset=[target] + feats).reset_index(drop=True)
    feats = [f for f in feats if f not in RAW_YIELD_PREDICTORS]
    return data, feats, cfg

def fit(df, target, drop_head=0):
    data, feats, cfg = prep(df, target, drop_head)
    cal = pd.to_datetime(data["DATE"].values)
    tgt = pd.Series(data[target].to_numpy(float), index=cal, name=target)
    X = data[feats].astype(float); X.index = cal
    t0 = time.time()
    eng = MulaEngine(); eng.fit(tgt, X, feature_names=feats, config=cfg)
    ts = eng.ts_data.copy(); ts["Price"] = tgt.values; ts.index = cal
    return eng, ts, time.time() - t0, len(feats)
