"""
Tattva — Swayam kernel: per-series MSF + MMR with regime intelligence.
तत्त्व (Tattva) — "Principle / Essence"

The single-series analysis kernel underneath the Swayam self-ensemble. It
computes, for ONE price series: the Market Strength Factor (momentum /
structure / flow), the Macro-Music Regression deviation oscillator against a
macro driver pool, HMM/GARCH/CUSUM regime state, and their unification into
``Unified_Osc`` + ``Condition``.

This was the Nirnay engine. Nirnay ran this kernel across a *basket* — an
index's constituents, or a curated proxy basket for a commodity/FX target —
and aggregated the results into cross-sectional breadth. That orchestration is
gone: a basket is a proxy for the target, and a proxy that has to be curated by
hand (which names, co-directional or inverse, capped at how many) is a set of
judgement calls the data never gets to overrule. Swayam asks the same breadth
question of the target's OWN price, across a bank of timescales × information
sets × mechanisms, so "breadth" means agreement among views of the thing
itself rather than among hand-picked cousins of it.

What survived is exactly the per-series math, which was never basket-specific;
``aggregate_views`` (formerly ``aggregate_constituent_timeseries``) survived
too, because the ensemble reduces its member views the same way the basket
reduced its constituents — only now each view is weighted by its own realised
skill rather than counted equally (see ``view_skill_weights``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.regime import run_regime_loop
# NOTE (audit finding F12): analytics.regime also exposes object-oriented
# reference implementations (AdaptiveKalmanFilter, AdaptiveHMM, GARCHDetector,
# CUSUMDetector) that mirror run_regime_loop's njit kernel step-for-step. This
# module only ever called the kernel — the object classes were imported but
# dead code, and (being a hand-maintained duplicate of the kernel's logic)
# a maintenance hazard: the HMM label-switching fix had to be applied to BOTH
# independently. Kept as the readable reference (not deleted) but no longer
# imported here; research/regime_equivalence_check.py asserts the two stay
# numerically identical so a future edit to either side fails loudly instead
# of silently drifting.

# ─── Utility functions ───────────────────────────────────────────────────────
# sigmoid / zscore_clipped / calculate_atr moved to analytics.utils (audit
# finding F11) — this module's private copies were the CANONICAL semantics
# (analytics.utils previously carried a second, non-equivalent copy that no
# caller here ever used); import them so there is exactly one implementation.
from analytics.utils import sigmoid as _sigmoid, zscore_clipped as _zscore_clipped, calculate_atr as _calculate_atr
from analytics.adaptive import OnlineSkillWeights


# ─── Market Strength Factor ──────────────────────────────────────────────────


_MSF_COMPONENT_NAMES = ("momentum", "structure", "flow")


def calculate_msf(
    df: pd.DataFrame,
    length: int = 20,
    roc_len: int = 14,
    clip: float = 3.0,
    components: tuple[str, ...] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Market Strength Factor from OHLCV data.

    Combines four orthogonal components:
    - **Momentum**: Rate of change z-score
    - **Microstructure**: Volume-weighted direction vs impact
    - **Trend**: Multi-timeframe composite (trend diff + momentum accel
      + volume-adjusted momentum + mean reversion)
    - **Flow**: Accumulation/distribution ratio + regime counting

    ``components`` optionally restricts the final combine to a subset of
    ``("momentum", "structure", "flow")`` — used by the Swayam
    self-ensemble (engines/swayam/ensemble.py) to promote a single mechanism to
    a standalone voter. ``None`` (default) combines all three and is
    byte-identical to the pre-mask behaviour.

    Returns
    -------
    msf_signal, micro_norm, momentum_norm, accum_norm
    """
    close = df["Close"]

    # Momentum
    roc_raw = close.pct_change(roc_len, fill_method=None)
    roc_z = _zscore_clipped(roc_raw, length, clip)
    momentum_norm = _sigmoid(roc_z, 1.5)

    # Microstructure
    intrabar_dir = (df["High"] + df["Low"]) / 2 - df["Open"]
    vol_ma = df["Volume"].rolling(length).mean()
    vol_ratio = (df["Volume"] / vol_ma).fillna(1.0)
    vw_direction = (intrabar_dir * vol_ratio).rolling(length).mean()
    price_change_imp = close.diff(5)
    vw_impact = (price_change_imp * vol_ratio).rolling(length).mean()
    micro_raw = vw_direction - vw_impact
    micro_z = _zscore_clipped(micro_raw, length, clip)
    micro_norm = _sigmoid(micro_z, 1.5)

    # Trend
    trend_fast = close.rolling(5).mean()
    trend_slow = close.rolling(length).mean()
    trend_diff_z = _zscore_clipped(trend_fast - trend_slow, length, clip)
    mom_accel_raw = close.diff(5).diff(5)
    mom_accel_z = _zscore_clipped(mom_accel_raw, length, clip)
    atr = _calculate_atr(df, 14)
    vol_adj_mom_raw = close.diff(5) / atr
    vol_adj_mom_z = _zscore_clipped(vol_adj_mom_raw, length, clip)
    mean_rev_z = _zscore_clipped(close - trend_slow, length, clip)
    composite_trend_z = (
        trend_diff_z + mom_accel_z + vol_adj_mom_z + mean_rev_z
    ) / np.sqrt(4.0)
    composite_trend_norm = _sigmoid(composite_trend_z, 1.5)

    # Flow
    typical_price = (df["High"] + df["Low"] + close) / 3
    mf = typical_price * df["Volume"]
    mf_pos = np.where(close > close.shift(1), mf, 0)
    mf_neg = np.where(close < close.shift(1), mf, 0)
    mf_pos_smooth = pd.Series(mf_pos, index=df.index).rolling(length).mean()
    mf_neg_smooth = pd.Series(mf_neg, index=df.index).rolling(length).mean()
    mf_total = mf_pos_smooth + mf_neg_smooth
    accum_ratio = mf_pos_smooth / mf_total.replace(0, np.nan)
    accum_ratio = accum_ratio.fillna(0.5)
    accum_norm = 2.0 * (accum_ratio - 0.5)
    pct_change = close.pct_change(fill_method=None)
    regime_signals = np.select(
        [pct_change > 0.0033, pct_change < -0.0033], [1, -1], default=0
    )
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length).mean()
    regime_z = _zscore_clipped(regime_raw, length, clip)
    regime_norm = _sigmoid(regime_z, 1.5)

    # Combine — optionally restricted to a component subset (Swayam
    # mechanism-isolated members). ``components=None`` uses all three, and
    # the √3/√3 scaling below reduces to exactly the pre-mask formula.
    osc_parts = {
        "momentum": momentum_norm,
        "structure": (micro_norm + composite_trend_norm) / np.sqrt(2.0),
        "flow": (accum_norm + regime_norm) / np.sqrt(2.0),
    }
    active_names = components if components is not None else _MSF_COMPONENT_NAMES
    unknown = set(active_names) - set(_MSF_COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown MSF component(s): {sorted(unknown)}")
    active = [osc_parts[name] for name in active_names]
    n_active = float(len(active))
    msf_raw = sum(active) / np.sqrt(n_active)
    msf_signal = _sigmoid(msf_raw * np.sqrt(n_active), 1.0)

    return msf_signal, micro_norm, momentum_norm, accum_norm


# ─── Macro-Micro Regime ──────────────────────────────────────────────────────


#: |log-log correlation| above which a macro candidate is treated as a COPY of
#: the target rather than a driver of it, and dropped from the MMR pool. Guards
#: the silent-death mode described in calculate_mmr. Set high deliberately: the
#: point is to catch replicas (r ~ 1.0), not to thin genuinely correlated macro.
MMR_REPLICA_MAX_CORR: float = 0.98


def calculate_mmr(
    df: pd.DataFrame,
    length: int = 20,
    num_vars: int = 5,
    macro_columns: list[str] | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], pd.Series]:
    """Calculate Macro-Micro Regime via rolling R²-weighted regression.

    Finds the top ``num_vars`` macro indicators most correlated with price,
    builds a weighted composite prediction, and measures the deviation of
    actual price from that prediction.

    CAVEAT — ``mmr_quality`` is a max-statistic and reads high by construction:
    it is computed over the per-row TOP-``num_vars`` R² values selected from
    ~100 candidate macros on a ~``length``-day window. Under a pure null
    (no genuine relationship anywhere) the expected maximum |correlation|
    among ~100 candidates at ~20 effective observations is ~0.5-0.6 (order
    statistics of r; cf. the multiple-testing magnitude in Harvey, Liu & Zhu
    2016, RFS 29(1)), so a "high quality" reading is NOT evidence the macro
    fit is real. The selection stays causal (shift(1) correlations), so the
    deviation oscillator itself is usable, and since the inflation applies
    ~uniformly across constituents the cross-sectional breadth signal is
    mostly unharmed — but do not repurpose ``mmr_quality`` as an absolute
    confidence measure without benchmarking it against a same-row max-R² of
    permuted candidates.

    Returns
    -------
    mmr_signal, driver_details, mmr_quality
    """
    if macro_columns is None:
        macro_columns = []
    available_macros = [v for v in macro_columns if v in df.columns]
    target = df["Close"]

    # REPLICA SCREEN — structural, because the name-based one cannot see this.
    #
    # core.config.swayam_macro_columns drops the target's own column and its
    # TARGET_EXCLUDED_PREDICTORS near-replicas, which handles everything it can
    # NAME. A series that happens to track the target ~1.0 under an unrelated
    # name is invisible to it, and that is the failure this function is most
    # exposed to: driver selection locks onto the replica, predicted tracks
    # actual, the deviation collapses to ~0, and the MMR half of every
    # macro-anchored member dies while mmr_quality reads perfect. Silent, and
    # indistinguishable downstream from "no macro relationship today".
    #
    # Screened here rather than in config because only this function has the
    # DATA. Correlation is computed on the FULL sample and used solely to
    # exclude — it never selects, weights or scales anything, so it cannot leak
    # a forward-looking preference into the signal; the per-bar driver ranking
    # below stays causal (shift(1)) exactly as before. On the live 240-column
    # pool this removes nothing (measured 2026-08-17: 0 survivors above 0.98 on
    # Gold/Copper/Silver), so it is a guard against a future addition, not a
    # change to today's output.
    if available_macros and len(df) > 200:
        _t = np.log(target.clip(lower=1e-12))
        _keep = []
        for _c in available_macros:
            _v = pd.to_numeric(df[_c], errors="coerce")
            _m = _v.notna() & _t.notna() & (_v > 0)
            if _m.sum() < 100:
                _keep.append(_c)
                continue
            _r = np.corrcoef(_t[_m], np.log(_v[_m]))[0, 1]
            if np.isfinite(_r) and abs(_r) > MMR_REPLICA_MAX_CORR:
                continue                      # a copy of the target, not a driver
            _keep.append(_c)
        available_macros = _keep

    if len(df) < length + 10 or not available_macros:
        return (pd.Series(0.0, index=df.index), [], pd.Series(0.0, index=df.index))

    y_mean = target.rolling(length, min_periods=1).mean().shift(1).fillna(float(target.iloc[0]))
    y_std = target.rolling(length, min_periods=1).std().shift(1).fillna(1.0)

    preds_list = []
    r2_list = []
    last_corr: dict[str, float] = {}   # signed r at the last bar, per ticker

    # Vectorized causal rolling computations
    for ticker in available_macros:
        x = df[ticker].ffill().fillna(0)
        x_mean = x.rolling(length, min_periods=1).mean().shift(1).fillna(0)
        x_std = x.rolling(length, min_periods=1).std().shift(1).fillna(1.0)

        # Pearson correlation shifted (only prior data used to estimate relationship).
        # rolling.corr emits ±inf on near-zero-variance windows (pegged/constant
        # series, ff-filled holiday runs), and catastrophic cancellation can also
        # yield finite |r| > 1 — both are numerically meaningless for a Pearson r
        # and, squared, would hijack the top-N driver selection below (±inf sorts
        # last in argsort, so a constant column would always win a "top driver"
        # slot). Sanitize: non-finite -> NaN (excluded downstream), then clip to
        # the valid Pearson range.
        roll_corr = (
            x.rolling(length, min_periods=length).corr(target)
             .replace([np.inf, -np.inf], np.nan)
             .clip(-1.0, 1.0)
             .shift(1)
             .fillna(0)
        )
        slope = roll_corr * (y_std / x_std.replace(0, np.nan)).fillna(0)
        intercept = y_mean - (slope * x_mean)

        pred = (slope * x) + intercept
        r2 = roll_corr**2

        preds_list.append(pred)
        r2_list.append(r2)
        if len(roll_corr):
            last_corr[ticker] = float(roll_corr.iloc[-1])

    all_preds = pd.concat(preds_list, axis=1)
    all_r2 = pd.concat(r2_list, axis=1)
    
    # Causally select top `num_vars` drivers per row!
    all_preds_arr = all_preds.values
    all_r2_arr = all_r2.values
    
    n_rows = len(df)
    y_predicted = np.empty(n_rows, dtype=np.float64)
    model_r2_arr = np.empty(n_rows, dtype=np.float64)
    
    for i in range(n_rows):
        row_r2 = all_r2_arr[i]
        valid_mask = np.isfinite(row_r2)
        if np.sum(valid_mask) < num_vars:
            y_predicted[i] = y_mean.iloc[i]
            model_r2_arr[i] = 0.0
            continue
            
        top_indices = np.argsort(row_r2[valid_mask])[-num_vars:]
        top_real_indices = np.where(valid_mask)[0][top_indices]
        
        r2_sel = row_r2[top_real_indices]
        preds_sel = all_preds_arr[i, top_real_indices]
        
        r2_sum = np.sum(r2_sel)
        if np.isfinite(r2_sum) and r2_sum > 1e-6:
            y_predicted[i] = np.sum(preds_sel * r2_sel) / r2_sum
            model_r2_arr[i] = np.sum(r2_sel**2) / r2_sum
        else:
            y_predicted[i] = y_mean.iloc[i]
            model_r2_arr[i] = 0.0

    deviation = target - pd.Series(y_predicted, index=df.index)
    mmr_z = _zscore_clipped(deviation, length, 3.0)
    mmr_signal = _sigmoid(mmr_z, 1.5)
    mmr_quality = pd.Series(np.sqrt(model_r2_arr), index=df.index).fillna(0)

    # Top drivers from the causal per-row R² weights at the last bar (same
    # weights that actually drove the MMR signal, not a trailing static corr).
    # Two fixes here:
    #   • Symbol — rolling().corr() drops the Series name, so all_r2's columns
    #     are INTEGER positions; the old code emitted those integers as the
    #     "Symbol". Map position → available_macros[position] (the loop order).
    #   • Correlation — the SIGNED r; sqrt(r²) would silently report |r| and
    #     display an inversely-related driver (r = -0.8) as +0.8.
    driver_details = []
    if len(all_r2) > 0:
        last_r2 = all_r2.iloc[-1].dropna()
        if len(last_r2) > 0:
            top_r2 = last_r2.nlargest(num_vars)
            for col in top_r2.index:
                if isinstance(col, (int, np.integer)) and 0 <= int(col) < len(available_macros):
                    name = available_macros[int(col)]
                else:
                    name = str(col)
                signed_r = last_corr.get(name)
                if signed_r is None:
                    signed_r = float(np.sqrt(max(0.0, float(top_r2[col]))))
                driver_details.append({
                    "Symbol": name,
                    "Correlation": round(float(signed_r), 4),
                })

    return mmr_signal, driver_details, mmr_quality


# ─── Full Analysis Pipeline ──────────────────────────────────────────────────


def run_full_analysis(
    df: pd.DataFrame,
    length: int,
    roc_len: int,
    regime_sensitivity: float,
    base_weight: float,
    num_vars: int = 5,
    oversold: float = -5.0,
    overbought: float = 5.0,
    macro_columns: list[str] | None = None,
    components: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the complete Swayam pipeline on a single stock DataFrame.

    Steps
    -----
    1. Calculate MSF and MMR signals
    2. Compute adaptive clarity-based weights
    3. Blend signals with agreement multiplier
    4. Classify conditions (Oversold/Overbought/Neutral)
    5. Run regime intelligence loop (Kalman → GARCH → HMM → CUSUM)

    ``components`` is forwarded to :func:`calculate_msf` (``None`` — the
    default — is byte-identical to the pre-mask MSF combine).
    """
    if macro_columns is None:
        macro_columns = []

    # Compute MSF + MMR as locals, then attach in a single concat. Inserting
    # the six columns one-by-one into the (wide, 100+ macro) frame triggers
    # pandas' "highly fragmented DataFrame" PerformanceWarning on every stock.
    msf, micro, momentum, flow = calculate_msf(df, length, roc_len, components=components)
    mmr, drivers, mmr_quality = calculate_mmr(
        df, length, num_vars=num_vars, macro_columns=macro_columns
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "MSF": msf, "Micro": micro, "Momentum": momentum, "Flow": flow,
                    "MMR": mmr, "MMR_Quality": mmr_quality,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Adaptive weighting based on signal clarity
    msf_clarity = df["MSF"].abs()
    mmr_clarity = df["MMR"].abs()
    msf_clarity_scaled = msf_clarity.pow(regime_sensitivity)
    mmr_clarity_scaled = (mmr_clarity * df["MMR_Quality"]).pow(regime_sensitivity)
    clarity_sum = msf_clarity_scaled + mmr_clarity_scaled + 0.001

    msf_w_adaptive = msf_clarity_scaled / clarity_sum
    mmr_w_adaptive = mmr_clarity_scaled / clarity_sum

    msf_w_final = 0.5 * base_weight + 0.5 * msf_w_adaptive
    mmr_w_final = 0.5 * (1.0 - base_weight) + 0.5 * mmr_w_adaptive
    w_sum = msf_w_final + mmr_w_final
    msf_w_norm = msf_w_final / w_sum
    mmr_w_norm = mmr_w_final / w_sum

    unified_signal = (msf_w_norm * df["MSF"]) + (mmr_w_norm * df["MMR"])

    # Agreement multiplier amplifies aligned signals, dampens conflicts
    agreement = df["MSF"] * df["MMR"]
    agree_strength = agreement.abs()
    multiplier = np.where(
        agreement > 0,
        1.0 + 0.2 * agree_strength,
        1.0 - 0.1 * agree_strength,
    )

    # Compute all derived columns as local arrays first, then join in ONE
    # block-build via pd.concat. Note: df.assign() also triggers the
    # PerformanceWarning on newer pandas because internally it does a per-kwarg
    # column insert loop. pd.concat with a fresh inner DataFrame builds the
    # twelve new columns as a single block and merges them in one operation.
    unified = np.asarray((unified_signal * multiplier).clip(-1.0, 1.0))
    unified_osc = unified * 10.0
    msf_osc = df["MSF"].to_numpy() * 10.0
    mmr_osc = df["MMR"].to_numpy() * 10.0
    close_arr = df["Close"].to_numpy()

    agreement_arr = agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement)
    strong_agreement = agreement_arr > 0.3
    buy_signal = strong_agreement & (unified_osc < oversold)
    sell_signal = strong_agreement & (unified_osc > overbought)

    # Divergence detection (shift(1) ↔ prepend NaN, drop last)
    prev_unified_osc = np.concatenate(([np.nan], unified_osc[:-1]))
    prev_close = np.concatenate(([np.nan], close_arr[:-1]))
    with np.errstate(invalid="ignore"):  # NaN comparisons → False, silently
        osc_rising = unified_osc > prev_unified_osc
        price_falling = close_arr < prev_close
        osc_falling = unified_osc < prev_unified_osc
        price_rising = close_arr > prev_close
    bullish_div = osc_rising & price_falling & (unified_osc < oversold)
    bearish_div = osc_falling & price_rising & (unified_osc > overbought)

    condition = np.where(
        unified_osc < oversold,
        "Oversold",
        np.where(unified_osc > overbought, "Overbought", "Neutral"),
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Unified": unified,
                    "Unified_Osc": unified_osc,
                    "MSF_Osc": msf_osc,
                    "MMR_Osc": mmr_osc,
                    "MSF_Weight": msf_w_norm,
                    "MMR_Weight": mmr_w_norm,
                    "Agreement": agreement.to_numpy() if hasattr(agreement, "to_numpy") else np.asarray(agreement),
                    "Buy_Signal": buy_signal,
                    "Sell_Signal": sell_signal,
                    "Bullish_Div": bullish_div,
                    "Bearish_Div": bearish_div,
                    "Condition": condition,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Regime intelligence loop — single-pass Numba kernel (faithful port of the
    # Kalman → GARCH → HMM → CUSUM sequential filters; output is identical to
    # the per-step object implementation but ~15× faster: the old Python loop
    # spent its time in per-step NumPy dispatch over tiny windows).
    regimes, hmm_bulls, hmm_bears, vol_regimes, change_points, confidences = (
        run_regime_loop(df["Unified"].values)
    )

    # Join the six regime-intelligence columns as ONE block via pd.concat.
    # df.assign() also fragments under newer pandas because it inserts kwargs
    # one-by-one internally; pd.concat with a pre-built inner DataFrame avoids
    # that entirely (single block-build, single merge).
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Regime": regimes,
                    "HMM_Bull": hmm_bulls,
                    "HMM_Bear": hmm_bears,
                    "Vol_Regime": vol_regimes,
                    "Change_Point": change_points,
                    "Confidence": confidences,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    return df, drivers


# ─── Constituent Aggregation ─────────────────────────────────────────────────


#: How much of the raw skill differential to keep, 0.0 = equal weight,
#: 1.0 = the unshrunk estimator. Set from measurement, not taste: split-half
#: rank correlation of the weights is -0.379 / -0.007 / -0.054 on Gold /
#: Copper / Silver, i.e. no evidence the ranking persists, while raw spreads
#: reach 9x. Halving keeps the differentiation where it is real (Bitcoin
#: measures +0.744 persistence) and halves the cost where it is not.
#: See the note in view_skill_weights and research/test_swayam_invariants.py.
SKILL_WEIGHT_SHRINK: float = 0.5


def view_skill_weights(
    view_results: dict[str, pd.DataFrame],
    horizon: int = 10,
    halflife: float = 252.0,
    floor: float = 0.02,
) -> pd.DataFrame:
    """Causal per-date weight for each view, from its own realised skill.

    Returns a ``(dates x views)`` frame whose rows sum to the view count, so it
    drops straight into :func:`aggregate_views` without changing the scale of
    any count or percentage it produces.

    The ordering is the whole point. At date *t* a view's weight reflects only
    outcomes that had already resolved by *t*: the call made at ``s`` is scored
    against the move over ``(s, s+h]``, which is not knowable until ``s+h``, so
    the loop below absorbs ``s = t - h`` on step ``t`` and never earlier. A
    weight published on *t* therefore never changes when more data arrives —
    the same discipline the FVO engine applies to its two valuation views, and
    the reason breadth here can be recomputed years later and match.

    All views read the SAME underlying price (that is what makes the ensemble
    self-referential), so the outcome series is shared and taken from whichever
    member carries a usable Close.
    """
    if not view_results:
        return pd.DataFrame()

    names = list(view_results.keys())
    frames = [view_results[k] for k in names]
    idx = None
    for df in frames:
        if df is not None and not df.empty:
            idx = df.index if idx is None else idx.union(df.index)
    if idx is None or len(idx) == 0:
        return pd.DataFrame()
    idx = idx.sort_values()

    # Signed call per view per date.
    sig = pd.DataFrame(0.0, index=idx, columns=names)
    for k, df in zip(names, frames):
        if df is None or df.empty or "Unified_Osc" not in df.columns:
            continue
        sig[k] = pd.to_numeric(df["Unified_Osc"], errors="coerce").reindex(idx)
    sig = sig.fillna(0.0)

    # Shared outcome: the target's own forward return over the horizon.
    close = None
    for df in frames:
        if df is not None and not df.empty and "Close" in df.columns:
            c = pd.to_numeric(df["Close"], errors="coerce").reindex(idx)
            if c.notna().sum() > horizon:
                close = c
                break
    if close is None:
        return pd.DataFrame(1.0, index=idx, columns=names)

    h = max(1, int(horizon))
    fwd = (close.shift(-h) / close - 1.0).to_numpy(dtype=np.float64)
    S = sig.to_numpy(dtype=np.float64)

    skill = OnlineSkillWeights(names, halflife=halflife, floor=floor)
    n = len(names)
    W = np.empty((len(idx), n), dtype=np.float64)
    for t in range(len(idx)):
        s = t - h
        if s >= 0:
            skill.observe(S[s], fwd[s])   # resolved at t; never before
        W[t] = skill.weights() * n        # rows sum to n, preserving count scale

    # SHRINK TOWARD EQUAL WEIGHT.
    #
    # The weights above are causal and never revised — that part is sound. What
    # was never checked is whether the skill they measure PERSISTS, and measured
    # on the app path (with macro, so MMR live) it does not:
    #
    #   split-half rank corr of view weights   Gold -0.379 · Copper -0.007 · Silver -0.054
    #   raw spread between best and worst view Gold  9.0x  · Copper  4.4x  · Silver  4.9x
    #
    # Negative persistence means a view that scored well in the first half tends
    # to score WORSE in the second, so a 9x weight differential is being assigned
    # on evidence that does not survive out of sample. Comparing the aggregate
    # against a plain equal-weighted one at h=10 confirms the cost:
    #
    #   Gold   skill IC -0.0104 (0.1σ)  vs equal -0.0324 (0.5σ)   equal 3x better
    #   Copper skill IC -0.0380 (0.5σ)  vs equal -0.0493 (0.7σ)   equal better
    #   Silver skill IC -0.0121 (0.2σ)  vs equal -0.0107 (0.2σ)   tie
    #
    # Neither IC is individually significant, so that comparison alone would be
    # weak — but it agrees in direction with the independent persistence
    # measurement, and two independent readings pointing the same way is what
    # justifies acting.
    #
    # Shrinkage rather than deletion, because skill weighting is NOT uniformly
    # useless: Bitcoin measured +0.744 persistence on the same test. A fixed
    # pull toward 1.0 caps the damage where the ranking is noise while keeping
    # half the differentiation where it is real. SKILL_WEIGHT_SHRINK = 1.0
    # restores the previous behaviour exactly.
    if SKILL_WEIGHT_SHRINK < 1.0:
        W = 1.0 + SKILL_WEIGHT_SHRINK * (W - 1.0)
        # Re-normalise so rows still sum to n; aggregate_views relies on that
        # scale for its counts and percentages.
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W * n, rs, out=np.full_like(W, 1.0), where=rs > 0)
    return pd.DataFrame(W, index=idx, columns=names)


def aggregate_views(
    view_results: dict[str, pd.DataFrame],
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate per-view Swayam results into daily statistics.

    Column schema is unchanged: Oversold/Overbought counts and percentages,
    signal counts, regime distributions, and average oscillator values.

    ``weights`` is an optional ``(dates x views)`` frame from
    :func:`view_skill_weights`, normalised so each row sums to the view count.
    Passing it makes every count and mean a WEIGHTED one — a view that has
    stopped predicting this instrument contributes proportionally less to
    breadth — while leaving the scale of every published column identical, so
    ``Oversold_Pct`` still reads 0-100 and ``Total_Analyzed`` still reports how
    many views actually reported (coverage, which is a count of participation,
    not of skill, and so stays unweighted).

    ``weights=None`` gives the equal-weighted reduction the basket read used.
    """
    if not view_results:
        return pd.DataFrame()

    # ── Vectorized aggregation ────────────────────────────────────────────
    # The previous implementation looped over every (date × constituent) pair
    # with a per-cell ``df.loc[date]`` lookup — O(D·C) Python with a Series
    # materialization on each step (~12s for an 18-name basket over ~2k days).
    # We instead stack every constituent's needed columns into one long frame
    # and let a single ``groupby(Date)`` do all the reductions in C. The output
    # schema, column order, and values are identical (validated to 1e-6).
    needed = [
        "Unified_Osc", "Condition", "Buy_Signal", "Sell_Signal",
        "Bullish_Div", "Bearish_Div", "Regime", "Vol_Regime", "Change_Point",
        "MSF_Osc", "MMR_Osc", "HMM_Bull", "HMM_Bear",
    ]
    defaults = {
        "Unified_Osc": 0.0, "Condition": "Neutral", "Buy_Signal": False,
        "Sell_Signal": False, "Bullish_Div": False, "Bearish_Div": False,
        "Regime": "NEUTRAL", "Vol_Regime": "NORMAL", "Change_Point": False,
        "MSF_Osc": 0.0, "MMR_Osc": 0.0, "HMM_Bull": 0.33, "HMM_Bear": 0.33,
    }

    parts: list[pd.DataFrame] = []
    for sym, df in view_results.items():
        if df is None or df.empty:
            continue
        sub = pd.DataFrame(index=df.index)
        for col in needed:
            sub[col] = df[col] if col in df.columns else defaults[col]
        sub["Date"] = [d.date() if hasattr(d, "date") else d for d in df.index]
        # Per-view weight, aligned to this view's own rows. Missing (a view
        # absent from the weight frame, or dates it does not cover) means an
        # unweighted vote, which is the equal-weight behaviour.
        if weights is not None and sym in weights.columns:
            sub["_w"] = pd.to_numeric(weights[sym], errors="coerce").reindex(df.index).fillna(1.0)
        else:
            sub["_w"] = 1.0
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    big = pd.concat(parts, ignore_index=True)

    # Per-row indicator columns (mirror the original branch logic exactly).
    cond = big["Condition"].astype(str)
    regime = big["Regime"].astype(str)
    vol = big["Vol_Regime"].astype(str)
    osc = pd.to_numeric(big["Unified_Osc"], errors="coerce")
    is_bull = regime.str.contains("BULL", regex=False)
    is_bear = regime.str.contains("BEAR", regex=False) & ~is_bull

    w = pd.to_numeric(big["_w"], errors="coerce").fillna(1.0).clip(lower=0.0)

    ind = pd.DataFrame({
        "Date": big["Date"],
        "_w": w,
        "Oversold": (cond == "Oversold").astype(float) * w,
        "Overbought": (cond == "Overbought").astype(float) * w,
        "Neutral": (~cond.isin(["Oversold", "Overbought"])).astype(float) * w,
        "Buy_Signals": big["Buy_Signal"].fillna(False).astype(bool).astype(float) * w,
        "Sell_Signals": big["Sell_Signal"].fillna(False).astype(bool).astype(float) * w,
        "Total_Analyzed": 1,
        "Signal_Sum": osc * w,
        "Bull_Div": big["Bullish_Div"].fillna(False).astype(bool).astype(float) * w,
        "Bear_Div": big["Bearish_Div"].fillna(False).astype(bool).astype(float) * w,
        "Regime_Bull": is_bull.astype(float) * w,
        "Regime_Bear": is_bear.astype(float) * w,
        "Regime_Transition": ((regime == "TRANSITION") & ~is_bull & ~is_bear).astype(float) * w,
        "Vol_High": vol.isin(["HIGH", "EXTREME"]).astype(float) * w,
        "Vol_Low": (vol == "LOW").astype(float) * w,
        "Change_Points": big["Change_Point"].fillna(False).astype(bool).astype(float) * w,
        "_msf": pd.to_numeric(big["MSF_Osc"], errors="coerce") * w,
        "_mmr": pd.to_numeric(big["MMR_Osc"], errors="coerce") * w,
        "_hb": pd.to_numeric(big["HMM_Bull"], errors="coerce") * w,
        "_hbe": pd.to_numeric(big["HMM_Bear"], errors="coerce") * w,
    })
    # Regime_Neutral is the original "else" branch: not bull/bear/transition.
    # Residual of the weighted parts (was "1 - ..."; with weights the row's
    # total vote mass is w, not 1).
    ind["Regime_Neutral"] = (
        ind["_w"] - ind["Regime_Bull"] - ind["Regime_Bear"] - ind["Regime_Transition"]
    ).clip(lower=0)

    g = ind.groupby("Date", sort=True)
    sums = g[[
        "Oversold", "Overbought", "Neutral", "Buy_Signals", "Sell_Signals",
        "Total_Analyzed", "Signal_Sum", "Bull_Div", "Bear_Div",
        "Regime_Bull", "Regime_Bear", "Regime_Neutral", "Regime_Transition",
        "Vol_High", "Vol_Low", "Change_Points",
    ]].sum()
    # The "_*" columns are already weight-multiplied, so a plain .mean() would
    # divide by the row COUNT and leave the weighting in the numerator only.
    # Divide by the summed weight instead — a proper weighted mean.
    wsum = g["_w"].sum().replace(0.0, np.nan)
    means = g[["Signal_Sum", "_msf", "_mmr", "_hb", "_hbe"]].sum().div(wsum, axis=0)

    n = sums["Total_Analyzed"]
    out = pd.DataFrame(index=sums.index)
    out["Oversold"] = sums["Oversold"]
    out["Overbought"] = sums["Overbought"]
    out["Neutral"] = sums["Neutral"]
    out["Buy_Signals"] = sums["Buy_Signals"]
    out["Sell_Signals"] = sums["Sell_Signals"]
    out["Total_Analyzed"] = sums["Total_Analyzed"]
    out["Avg_Signal"] = means["Signal_Sum"]
    out["Signal_Sum"] = sums["Signal_Sum"]
    out["Bull_Div"] = sums["Bull_Div"]
    out["Bear_Div"] = sums["Bear_Div"]
    out["Regime_Bull"] = sums["Regime_Bull"]
    out["Regime_Bear"] = sums["Regime_Bear"]
    out["Regime_Neutral"] = sums["Regime_Neutral"]
    out["Regime_Transition"] = sums["Regime_Transition"]
    out["Vol_High"] = sums["Vol_High"]
    out["Vol_Low"] = sums["Vol_Low"]
    out["Change_Points"] = sums["Change_Points"]
    out["Oversold_Pct"] = sums["Oversold"] / n * 100
    out["Overbought_Pct"] = sums["Overbought"] / n * 100
    out["Neutral_Pct"] = sums["Neutral"] / n * 100
    out["Regime_Bull_Pct"] = sums["Regime_Bull"] / n * 100
    out["Regime_Bear_Pct"] = sums["Regime_Bear"] / n * 100
    out["Vol_High_Pct"] = sums["Vol_High"] / n * 100
    out["avg_hmm_bull"] = means["_hb"]
    out["avg_hmm_bear"] = means["_hbe"]
    out["avg_msf_osc"] = means["_msf"]
    out["avg_mmr_osc"] = means["_mmr"]
    out.index.name = "Date"
    return out
