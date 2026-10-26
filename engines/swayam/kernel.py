"""
Tattva — Swayam kernel: per-series MSF + MMR, and the views' one breadth.
तत्त्व (Tattva) — "Principle / Essence"

The single-series analysis kernel underneath the Swayam self-ensemble. It
computes, for ONE price series: the Market Strength Factor (momentum /
structure / flow), the Macro-Micro Regime deviation oscillator against a
macro driver pool (a short-window residual of the target on the same panel
Mūla values against), and their unification into ``Unified_Osc`` +
``Condition``. (A per-view Kalman → GARCH → HMM → CUSUM regime loop used to
run here, fifteen times per target; nothing on the call read it, and it went.)

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
reduced its constituents: every view counted once. (Views were briefly
weighted by a learner of their own realised skill; it added no measurable IC
over equal weights, its loss rewarded the drift side, and it was removed.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Utility functions ───────────────────────────────────────────────────────
# sigmoid / zscore_clipped / calculate_atr moved to analytics.utils (audit
# finding F11) — this module's private copies were the CANONICAL semantics
# (analytics.utils previously carried a second, non-equivalent copy that no
# caller here ever used); import them so there is exactly one implementation.
from analytics.utils import sigmoid as _sigmoid, zscore_clipped as _zscore_clipped, calculate_atr as _calculate_atr
from core.semantics import breadth_lean


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
) -> tuple[pd.Series, pd.Series]:
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
    ~uniformly across views the breadth signal is
    mostly unharmed — but do not repurpose ``mmr_quality`` as an absolute
    confidence measure without benchmarking it against a same-row max-R² of
    permuted candidates.

    Returns
    -------
    mmr_signal, mmr_quality
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
        return (pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index))

    y_mean = target.rolling(length, min_periods=1).mean().shift(1).fillna(float(target.iloc[0]))
    y_std = target.rolling(length, min_periods=1).std().shift(1).fillna(1.0)

    preds_list = []
    r2_list = []

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
    return mmr_signal, mmr_quality


# ─── Full Analysis Pipeline ──────────────────────────────────────────────────


def osc_reach(use_macro: bool, base_weight: float) -> float:
    """The least upper bound of |Unified_Osc| a view can print, on the ±10 scale.

    A macro-anchored view blends MSF with a live MMR under the agreement
    multiplier (up to ×1.2) and clips the blend to ±1, so it reaches ±10. A
    price-only view has MMR ≡ 0: its MSF share of the blend stays below
    (1 + base_weight) / 2 and |MSF| < 1, so |Unified_Osc| stays strictly below
    10 · (1 + base_weight) / 2 — at base_weight 0, never past ±5.
    """
    return 10.0 if use_macro else 10.0 * (1.0 + float(base_weight)) / 2.0


def can_vote(use_macro: bool, base_weight: float, oversold: float, overbought: float) -> bool:
    """Whether a view can reach the Oversold or the Overbought condition at all.

    A view whose reach does not pass the nearer cut never votes: it is counted
    in ``Total_Analyzed``, so it dilutes both shares, but it can never move
    ``Oversold_Pct`` or ``Overbought_Pct``.
    """
    return osc_reach(use_macro, base_weight) > min(-float(oversold), float(overbought))


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
) -> pd.DataFrame:
    """Run the complete Swayam pipeline on a single stock DataFrame.

    Steps
    -----
    1. Calculate MSF and MMR signals
    2. Compute adaptive clarity-based weights
    3. Blend signals with agreement multiplier
    4. Classify conditions (Oversold/Overbought/Neutral)

    ``Can_Vote`` (constant per view, :func:`can_vote`) says whether the view
    can reach a condition at all. ``components`` is forwarded to
    :func:`calculate_msf` (``None`` — the default — is byte-identical to the
    pre-mask MSF combine).
    """
    if macro_columns is None:
        macro_columns = []

    # Compute MSF + MMR as locals, then attach in a single concat. Inserting
    # the six columns one-by-one into the (wide, 100+ macro) frame triggers
    # pandas' "highly fragmented DataFrame" PerformanceWarning on every stock.
    msf, micro, momentum, flow = calculate_msf(df, length, roc_len, components=components)
    mmr, mmr_quality = calculate_mmr(
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
    # new columns as a single block and merges them in one operation.
    unified = np.asarray((unified_signal * multiplier).clip(-1.0, 1.0))
    unified_osc = unified * 10.0
    msf_osc = df["MSF"].to_numpy() * 10.0
    mmr_osc = df["MMR"].to_numpy() * 10.0

    condition = np.where(
        unified_osc < oversold,
        "Oversold",
        np.where(unified_osc > overbought, "Overbought", "Neutral"),
    )
    use_macro = any(c in df.columns for c in macro_columns)

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
                    "Condition": condition,
                    "Can_Vote": can_vote(use_macro, base_weight, oversold, overbought),
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

    return df


# ─── View aggregation ────────────────────────────────────────────────────────


def aggregate_views(view_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reduce the views' votes into Swayam's daily breadth, every view counted once.

    One row per date, over the views that reported on it:

    * ``Oversold_Pct`` / ``Overbought_Pct`` — the share (0–100) of them whose
      ``Condition`` is Oversold / Overbought;
    * ``Total_Analyzed`` — how many views reported;
    * ``N_Voting`` — how many of those can reach a condition at all
      (``Can_Vote``, :func:`can_vote`); the others sit in the shares'
      denominator but can never move them;
    * ``Lean`` — ``core.semantics.breadth_lean(Oversold_Pct, Overbought_Pct)``,
      THE published Swayam lean, bullish-positive (oversold views are the
      bullish reading under Tattva's contrarian convention). The composer, the
      sizer's grade, every page and the export read this column; none
      re-derives it.

    Equal weights: a view's vote is not scaled by anything learned. The
    reduction is one stacked frame and one ``groupby(Date)`` (the earlier
    per-cell ``df.loc[date]`` loop was ~12 s for 18 series over ~2k days).
    """
    parts: list[pd.DataFrame] = []
    for df in (view_results or {}).values():
        if df is None or df.empty:
            continue
        cond = df["Condition"].astype(str)
        parts.append(pd.DataFrame({
            "Date": [d.date() if hasattr(d, "date") else d for d in df.index],
            "Oversold": (cond == "Oversold").astype(float).to_numpy(),
            "Overbought": (cond == "Overbought").astype(float).to_numpy(),
            "Total_Analyzed": 1,
            "N_Voting": df["Can_Vote"].astype(bool).astype(int).to_numpy(),
        }))
    if not parts:
        return pd.DataFrame()

    sums = pd.concat(parts, ignore_index=True).groupby("Date", sort=True)[
        ["Oversold", "Overbought", "Total_Analyzed", "N_Voting"]].sum()
    n = sums["Total_Analyzed"]
    out = pd.DataFrame(index=sums.index)
    out["Oversold_Pct"] = sums["Oversold"] / n * 100
    out["Overbought_Pct"] = sums["Overbought"] / n * 100
    out["Total_Analyzed"] = n
    out["N_Voting"] = sums["N_Voting"]
    out["Lean"] = [breadth_lean(a, b) for a, b in zip(out["Oversold_Pct"], out["Overbought_Pct"])]
    out.index.name = "Date"
    return out
