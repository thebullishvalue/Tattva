"""
Tattva — one target's full analysis, start to finish, with no Streamlit in it.

``run_prepared`` takes a :class:`pipeline.prep.PreparedPanel` and returns a
:class:`RunResult`: Mūla's valuation, Swayam's breadth, the convergence
evidence, every grade, and the precedent base rate. The app, the studies and
the tests all call this one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from analytics.analogs import find_similar_periods, summarize_forward
from convergence import intelligence
from convergence.conviction_model import CompositeDDM
from convergence.cross_validator import CrossValidator
from convergence.divergence_detector import CrossSystemDivergenceDetector
from convergence.normalization import (classify_convergence_score,
                                       compute_normalized_convergence, consensus_series)
from core.config import MIN_DATA_POINTS, STOCK_TARGET_MARKETS, swayam_macro_columns
from core.logger_config import console
from core.semantics import breadth_lean, valuation_lean
from engines.mula import MulaEngine
from engines.mula.blocks import block_membership
from engines.swayam import (aggregate_views, build_swayam_frames,
                            default_swayam_members, effective_member_count)
from engines.swayam.kernel import view_skill_weights

from .prep import DATE_COL, PreparedPanel, prepare

Progress = Callable[[int, str, str], None]

#: Per-view columns the Swayam page's drill-down reads; the rest is dropped
#: once aggregation has used it.
SWAYAM_DRILLDOWN_COLS = (
    "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
    "Regime", "Vol_Regime", "Change_Point", "Confidence",
)


@dataclass
class RunResult:
    prepared: PreparedPanel
    engine: MulaEngine
    signal: dict
    model_stats: dict
    fvo_ts: pd.DataFrame
    swayam_daily: pd.DataFrame
    swayam_view_dfs: dict
    swayam_view_weights: dict | None
    swayam_n_eff: float | None
    swayam_native_last: pd.Timestamp | None
    convergence_df: pd.DataFrame
    divergence_events: pd.DataFrame
    hero_series: pd.Series | None
    intelligence_active_weights: dict
    intelligence_active_profile: dict
    edges: dict
    precedent_summary: dict | None
    analogs: list = field(default_factory=list)


def _fmt(v, spec, none="n/a"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return none
    return format(v, spec) if np.isfinite(v) else none


def _slim(dfs: dict) -> dict:
    out = {}
    for name, df in (dfs or {}).items():
        cols = [c for c in SWAYAM_DRILLDOWN_COLS if c in df.columns]
        out[name] = df[cols] if cols else df.iloc[:, :0]
    return out


def swayam_driver_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The panel Mūla reads, date-indexed, as Swayam's MMR driver pool."""
    if DATE_COL not in df.columns:
        return pd.DataFrame()
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in STOCK_TARGET_MARKETS or c == target]
    out = df[[DATE_COL] + cols].copy()
    out.index = pd.to_datetime(out.pop(DATE_COL), errors="coerce").dt.normalize()
    out = out[~out.index.isna()]
    return out[~out.index.duplicated(keep="last")]


# ── Phase: Mūla ─────────────────────────────────────────────────────────────

def run_mula(prep: PreparedPanel, progress: Progress | None = None,
             today: pd.Timestamp | None = None) -> MulaEngine:
    cfg = prep.config
    blk_names, blk_map = block_membership(prep.features)
    console.section("Engine Configuration")
    console.item("Target", prep.target)
    console.item("Cross-section", f"{len(prep.features)} macro instruments → {len(blk_names)} asset-class blocks")
    console.item("Blocks", ", ".join(f"{b}({sum(1 for v in blk_map.values() if v == b)})" for b in blk_names))
    console.item("Observations", f"{len(prep.price)} rows")
    console.item("Burn-in", f"{cfg.fvo_burn_in} rows before first publication")
    console.item("Print Floor", f"{cfg.fvo_min_prints} prints before an instrument may join")
    console.item("Coefficient Memory", f"deltas {cfg.fvo_valuation_deltas}")
    console.item("Min Data Points", MIN_DATA_POINTS)

    engine = MulaEngine()
    cb = None
    if progress is not None:
        cb = lambda pct, msg: progress(int(20 + pct * 20), "Running Mūla Engine", msg)  # noqa: E731
    engine.fit(prep.price, prep.panel, printed=prep.printed, feature_names=prep.features,
               config=cfg, progress_callback=cb, today=today)
    # The analog matcher and the graders key off a column named "Price".
    engine.ts_data["Price"] = prep.price.to_numpy(dtype=np.float64)
    return engine


def _log_mula(engine: MulaEngine, prep: PreparedPanel) -> dict:
    sig = engine.get_current_signal()
    stats = engine.get_model_stats()
    h = prep.config.forecast_horizon
    console.section("Engine Results")
    console.item("Valuation", f"{sig.get('valuation_state', '?')} · oscillator "
                 f"{_fmt(sig.get('fvo'), '+.2f')}σ ({_fmt((sig.get('pct_mispricing') or 0) * 100, '+.2f')}% vs fair value)")
    console.item("Fair Value", f"{_fmt(sig.get('fair_value'), ',.2f')} vs price {_fmt(sig.get('actual'), ',.2f')}")
    if sig.get("asof_row_withheld"):
        console.warning(f"Latest session not valued ({sig.get('withheld_reason')}) — reading is the last valued session")
    console.item("R² vs Trailing Mean", f"{_fmt(stats.get('r2_vs_anchor'), '+.3f')} (edge over a 252d anchor)")
    console.item("Model Spread", f"{_fmt((sig.get('model_spread') or 0) * 10000, '.1f')} bps (SD of the fair-value level)")
    console.item("Cross-Section", f"{_fmt(sig.get('xs_consistency'), '.2f')} (leave-one-block-out sign agreement)")
    console.item("Price Reverts", f"P={_fmt(sig.get('mr_prob'), '.2f')} · κ̂ {_fmt(sig.get('mula_kappa'), '+.4f')}/d · "
                 f"expected {h}d move {_fmt(sig.get('expected_drift_pct'), '+.2f')}% "
                 f"± {_fmt(sig.get('drift_sd_pct'), '.2f')}%")
    console.item("Gap Half-Life", f"{_fmt(sig.get('gap_half_life'), '.0f')}d (online AR1) · "
                 f"OU {_fmt(sig.get('ou_half_life'), '.0f')}d (full history)")
    wr = engine.ts_data.get("WithheldReason")
    if wr is not None:
        wr = wr.astype(str)
        counts = wr[wr != ""].value_counts()
        if len(counts):
            console.item("Gaps in the trace", " · ".join(f"{k}: {int(v)}" for k, v in counts.items()))
        bad = int(counts.get("incomplete fetch", 0))
        if bad:
            dates = prep.calendar[np.asarray(wr == "incomplete fetch")]
            console.warning(
                f"{bad} session(s) withheld as INCOMPLETE FETCH "
                f"({', '.join(f'{d:%Y-%m-%d}' for d in list(dates)[-3:])}). The calendar says those "
                f"exchanges were open, so this is a data gap in this run — refresh to fill it.")
    console.item("Factors", f"k={sig.get('k_factors')} above the MP edge · "
                            f"{sig.get('n_available')} instruments printed on the latest row")
    console.item("Market Regime", f"{sig.get('market_regime')} (stress pct {_fmt(sig.get('stress'), '.2f')})")
    console.success(f"Mūla engine complete | {len(engine.ts_data)} output rows "
                    f"({int(engine.ts_data['Valid'].sum())} valued, {engine.min_train_size} burn-in)")
    return sig


# ── Phase: Swayam ───────────────────────────────────────────────────────────

def run_swayam(prep: PreparedPanel, ohlcv: pd.DataFrame | None, drivers: pd.DataFrame,
               progress: Progress | None = None) -> dict:
    """The view bank and its skill-weighted aggregate, on Swayam's own calendar."""
    cfg, h = prep.config, prep.config.forecast_horizon
    out = {"view_dfs": {}, "daily": pd.DataFrame(), "n_eff": None, "view_weights": None}
    if ohlcv is None or ohlcv.empty:
        return out
    console.section("Self-Referential View Bank")
    pool = list(drivers.columns)
    cols = swayam_macro_columns(prep.target, pool)
    console.item("Leakage guard", f"{len(pool) - len(cols)} drivers dropped (the target and its near-replicas)")
    members = default_swayam_members(cfg.swayam_lengths, cfg.swayam_roc_frac)
    console.item("Views (bank)", f"{len(members)} · timescale × information-set × mechanism")

    def _cb(done, total, name):
        if progress is not None:
            progress(int(45 + done / max(total, 1) * 30), f"View {name}", f"{done}/{total} views")

    views = build_swayam_frames(
        ohlcv, drivers, cols, members=members,
        regime_sensitivity=cfg.swayam_regime_sensitivity, base_weight=cfg.swayam_base_weight,
        num_vars=cfg.swayam_mmr_num_vars, oversold=cfg.swayam_oversold,
        overbought=cfg.swayam_overbought, progress_cb=_cb,
    )
    if not views:
        return out
    out["n_eff"] = effective_member_count(views)
    console.success(f"View bank: {len(views)} views · ~{out['n_eff']:.1f} effective")
    weights = view_skill_weights(views, horizon=h)
    out["daily"] = aggregate_views(views, weights=weights)
    if not weights.empty:
        last = weights.iloc[-1].sort_values(ascending=False)
        out["view_weights"] = {k: float(v) for k, v in weights.iloc[-1].items()}
        console.item("Top Views", " · ".join(f"{k} {v:.2f}" for k, v in last.head(4).items()))
    out["view_dfs"] = _slim(views)
    return out


def _onto_calendar(daily: pd.DataFrame, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Carry Swayam's aggregate onto the model calendar, marking native rows."""
    if daily.empty:
        return daily, None
    native_last = pd.Timestamp(daily.index.max())
    nd = daily.copy()
    nd.index = pd.to_datetime(nd.index).normalize()
    nd = nd[~nd.index.duplicated(keep="last")].sort_index()
    spine = pd.DatetimeIndex(sorted(cal.normalize().unique()))
    native = set(nd.index)
    out = nd.reindex(spine, method="ffill").dropna(how="all")
    out["_Native"] = out.index.isin(native)
    return out, native_last


# ── Phase: convergence, grades ──────────────────────────────────────────────

def run_convergence(prep: PreparedPanel, fvo_ts: pd.DataFrame, swayam_daily: pd.DataFrame,
                    progress: Progress | None = None) -> dict:
    cfg, h = prep.config, prep.config.forecast_horizon
    validator = CrossValidator(active_weights=cfg.weights_seed())
    detector = CrossSystemDivergenceDetector()

    sw_by_date = {}
    if not swayam_daily.empty:
        for idx, row in swayam_daily[~swayam_daily.index.duplicated(keep="last")].iterrows():
            sw_by_date[str(pd.Timestamp(idx).date())] = row

    overlap = native_overlap = 0
    total = len(fvo_ts.index)
    for i, ts_idx in enumerate(fvo_ts.index):
        row_a = fvo_ts.loc[ts_idx]
        date_str = str(pd.Timestamp(ts_idx).date())
        mula_in = {"fvo": row_a.get("FVO", np.nan),
                   "oversold_breadth": row_a.get("OversoldBreadth", 0.0),
                   "overbought_breadth": row_a.get("OverboughtBreadth", 0.0),
                   "regime": str(row_a.get("Regime", ""))}
        sw_in = None
        row_n = sw_by_date.get(date_str)
        if row_n is not None:
            sw_in = {"oversold_pct": float(row_n.get("Oversold_Pct", 0.0)),
                     "overbought_pct": float(row_n.get("Overbought_Pct", 0.0)),
                     "avg_unified_osc": float(row_n.get("Avg_Signal", 0.0)),
                     "regime_bull_pct": float(row_n.get("Regime_Bull_Pct", 0.0)),
                     "regime_bear_pct": float(row_n.get("Regime_Bear_Pct", 0.0)),
                     "native": bool(row_n.get("_Native", True))}
        if validator.compute_convergence(mula_in, sw_in, date_str) is not None:
            overlap += 1
            native_overlap += int(sw_in["native"])
        detector.detect(mula_in, sw_in, date_str)
        if progress is not None and ((i + 1) % 200 == 0 or i == total - 1):
            progress(int(76 + (i + 1) / total * 6), "Computing Convergence", f"{i + 1}/{total} sessions")
    cdf = validator.get_convergence_series()
    console.item("Scored sessions", f"{overlap} ({native_overlap} with a native Swayam row)")

    weight_hist = pd.DataFrame()
    if native_overlap >= intelligence.LEARNER_WARMUP and not cdf.empty:
        try:
            cdf, weight_hist = intelligence.apply_online_weights(
                cdf, fvo_ts, horizon=h, target_col="Price", prior=cfg.weights_seed())
        except Exception as e:                       # noqa: BLE001 - prior weights stand
            console.warning(f"Online weighting failed: {type(e).__name__}: {e} — prior weights stand")
    prior_w = {k.removeprefix("w_"): float(v) for k, v in cfg.weights_seed().items()}
    tot = sum(prior_w.values()) or 1.0
    prior_w = {k: v / tot for k, v in prior_w.items()}
    active_w = ({k: float(v) for k, v in weight_hist.iloc[-1].items()}
                if not weight_hist.empty else prior_w)

    if not cdf.empty:
        ddm = CompositeDDM(leak_rate=cfg.ddm_leak, drift_scale=cfg.ddm_drift,
                           long_run_var=cfg.ddm_lrv).fit_series(cdf["convergence_score"])
        for c in ("smoothed", "upper", "lower", "label"):
            cdf[f"ddm_{c}"] = ddm[c].to_numpy()
        last = cdf.iloc[-1]
        console.item("Weighted composite", f"{last['convergence_score']:+.1f} "
                     f"({classify_convergence_score(last['convergence_score'])})")

    consensus = consensus_series(fvo_ts, swayam_daily)
    now = compute_normalized_convergence(fvo_ts, swayam_daily)
    if now:
        console.item("Consensus", f"{now['value']:+.2f} ({now['label']})")
    events = detector.get_events()
    console.item("Divergence events", len(events))
    return {"convergence_df": cdf, "events": events, "active_weights": active_w,
            "learned": not weight_hist.empty,
            "hero_series": (consensus["Consensus"].rename("Consensus")
                            if not consensus.empty else None)}


def grade(prep: PreparedPanel, fvo_ts: pd.DataFrame, swayam_daily: pd.DataFrame,
          convergence_df: pd.DataFrame) -> dict:
    """Every engine's lean against the h-session forward return."""
    h = prep.config.forecast_horizon
    price = pd.to_numeric(fvo_ts["Price"], errors="coerce")
    edges: dict = {}
    try:
        fvo = pd.to_numeric(fvo_ts["FVO"], errors="coerce")
        edges["mula"] = intelligence.series_edge(fvo.map(valuation_lean).where(fvo.notna()), price, h)
        if not swayam_daily.empty:
            sw = (swayam_daily[swayam_daily["_Native"].astype(bool)]
                  if "_Native" in swayam_daily.columns else swayam_daily)
            edges["swayam"] = intelligence.series_edge(
                pd.Series([breadth_lean(a, b) for a, b in zip(sw["Oversold_Pct"], sw["Overbought_Pct"])],
                          index=sw.index), price, h)
        if not convergence_df.empty:
            edges["composite"] = intelligence.series_edge(
                pd.Series(-convergence_df["convergence_score"].to_numpy(dtype=float) / 100.0,
                          index=pd.to_datetime(convergence_df.index, errors="coerce")), price, h)
        for k, e in edges.items():
            console.item(f"{k} edge", (f"IC {e['ic']:+.3f} ± {e['se']:.3f} · {e['n']} calls · "
                                       f"{e['positive_blocks']}/{e['scored_blocks']} periods +")
                         if e and e.get("ic") is not None and np.isfinite(e["ic"]) else "not graded")
    except Exception as e:                           # noqa: BLE001 - a diagnostic, not the run
        console.warning(f"Edge grading skipped: {type(e).__name__}: {e}")
    return edges


def precedent(prep: PreparedPanel, engine: MulaEngine) -> tuple[dict | None, list]:
    """The analog base rate at the forecast horizon, and the analogs behind it."""
    cfg = prep.config
    ts = engine.ts_data.copy()
    ts["Date"] = prep.calendar
    summary, analogs = None, []
    try:
        hold = tuple(cfg.precedent_horizons)
        analogs = find_similar_periods(ts, prep.target, hold_horizons=hold,
                                       mom_window=cfg.analog_mom_window) or []
        ps = summarize_forward(analogs, hold) if analogs else {}
        hp = int(cfg.forecast_horizon)
        if hp not in ps:
            cands = [x for x in hold if x <= hp]
            hp = max(cands) if cands else min(hold)
        row = ps.get(int(hp))
        if row:
            summary = {"horizon": int(hp), "median": float(row["median"]),
                       "positive_pct": float(row["positive_pct"]),
                       "n": int(row["n"]), "n_eff": float(row.get("n_eff", row["n"])),
                       "p25": float(row.get("p25", float("nan"))),
                       "p75": float(row.get("p75", float("nan"))),
                       "usable": bool(row.get("usable", True)),
                       "note": str(row.get("note", "") or "")}
    except Exception as e:                           # noqa: BLE001 - evidence, not the run
        console.warning(f"Precedent read skipped: {type(e).__name__}: {e}")
    return summary, analogs


# ── The whole run ───────────────────────────────────────────────────────────

def run_prepared(prep: PreparedPanel, df: pd.DataFrame | None = None, *,
                 ohlcv: pd.DataFrame | None = None, progress: Progress | None = None,
                 today: pd.Timestamp | None = None) -> RunResult:
    """Run every phase for an already-prepared target.

    ``df`` is the shared model matrix (Swayam's driver pool is read from it);
    ``ohlcv`` the target's own OHLCV, or ``None`` for a target without one —
    Swayam is then skipped and its evidence reads as absent.
    """
    def _p(pct, title, detail=""):
        if progress is not None:
            progress(int(pct), title, detail)

    console.start_phase("MŪLA ENGINE", 1, 4)
    _p(20, "Running Mūla Engine", f"Valuation · {len(prep.features)} instruments · {len(prep.price)} rows")
    engine = run_mula(prep, progress, today)
    sig = _log_mula(engine, prep)
    console.end_phase("MŪLA ENGINE")
    _p(40, "Mūla Engine Complete", f"{sig.get('valuation_state', '')} · {_fmt(sig.get('fvo'), '+.2f')}σ")

    console.start_phase("SWAYAM ENGINE", 2, 4)
    _p(42, "Running Swayam Engine", "self-referential view bank")
    drivers = swayam_driver_frame(df, prep.target) if df is not None else pd.DataFrame()
    sw = run_swayam(prep, ohlcv, drivers, progress)
    swayam_daily, native_last = _onto_calendar(sw["daily"], prep.calendar)
    console.end_phase("SWAYAM ENGINE")
    _p(75, "Swayam Engine Complete", f"{len(sw['view_dfs'])} views · {len(swayam_daily)} trading days")

    console.start_phase("CONVERGENCE", 3, 4)
    _p(76, "Computing Convergence", "per session")
    fvo_ts = engine.ts_data.copy()
    fvo_ts["Date"] = prep.calendar
    fvo_ts = fvo_ts.set_index("Date")
    fvo_ts = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
    conv = run_convergence(prep, fvo_ts, swayam_daily, progress)
    _p(93, "Grading Every Engine", f"Non-overlapping {prep.config.forecast_horizon}d")
    edges = grade(prep, fvo_ts, swayam_daily, conv["convergence_df"])
    console.end_phase("CONVERGENCE")

    console.start_phase("PRECEDENT", 4, 4)
    _p(96, "Reading Precedent", "analog base rate")
    summary, analogs = precedent(prep, engine)
    console.end_phase("PRECEDENT")
    _p(100, "Analysis Complete", prep.target)

    return RunResult(
        prepared=prep, engine=engine, signal=engine.get_current_signal(),
        model_stats=engine.get_model_stats(), fvo_ts=fvo_ts,
        swayam_daily=swayam_daily, swayam_view_dfs=sw["view_dfs"],
        swayam_view_weights=sw["view_weights"], swayam_n_eff=sw["n_eff"],
        swayam_native_last=native_last,
        convergence_df=conv["convergence_df"], divergence_events=conv["events"],
        hero_series=conv["hero_series"],
        intelligence_active_weights=conv["active_weights"],
        intelligence_active_profile={"weights": conv["active_weights"], "learned": conv["learned"]},
        edges=edges, precedent_summary=summary, analogs=analogs,
    )


def run_target(df: pd.DataFrame, target: str, *, ohlcv: pd.DataFrame | None = None,
               progress: Progress | None = None, today: pd.Timestamp | None = None) -> RunResult:
    """``prepare`` then :func:`run_prepared`. Raises ``PrepError`` on unusable data."""
    return run_prepared(prepare(df, target), df, ohlcv=ohlcv, progress=progress, today=today)
