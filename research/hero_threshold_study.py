"""
Tattva — HERO CLASSIFICATION THRESHOLD study (BUY / SELL / STRONG tiers).

The hero card's headline label comes from classify_normalized_signal: the
NORMALIZED CONSENSUS classified at DEFAULT_THRESHOLDS. The CALIBRATED evidence
row's factory fallback classifies the DIRECTIONAL COMPOSITE at
COMPOSITE_THRESHOLDS. The markers study anchors the PLOT guides; this study
owns these ACTION cut-points (both sets are read LIVE, so re-runs always sweep
against whatever currently ships).

HISTORY: first run 2026-07-12 (reports/tuning_20260712_141326.txt) — the then-
hand-set consensus ±0.3/±0.5 (p82/p93) found no separation-sweep winner and was
re-anchored to the occupancy convention ±0.26/±0.39 (p75/p90), unifying the
hero classifier with the plot-marker tiers; the composite ±0.11/±0.18 matched
its 8-target p75/p90 (0.107/0.174) within rounding and was kept.

Question: where should the moderate/strong cut-points sit so that
  (a) the tiers mean a consistent, data-anchored extremeness (occupancy), and
  (b) the tiers actually ORDER forward returns (S.SELL < SELL < HOLD < BUY <
      S.BUY separation) as well as this weak signal allows?

Method (honest, causal):
  • CONSENSUS: per target, the exact live construction — causal_normalize of
    Aarambh ConvictionRaw and Nirnay Avg_Signal, averaged (markers_study
    machinery) — vs +10d/+20d forward returns, NON-OVERLAPPING (stride = h).
  • COMPOSITE: the live calibration frame (calibration_lift_study's
    _convergence_frame: Aarambh walk-forward → Nirnay basket → CrossValidator
    loop), composite = _composite_signal(frame, DEFAULT_WEIGHTS), vs the
    frame's Ret_5b/Ret_10b, stride-sampled.
  • Candidate (moderate, strong) pairs are PERCENTILE-ANCHORED on the pooled
    |signal| distribution (p60–p85 moderate × p85–p97.5 strong) so the same
    grid adapts to each distribution's scale; the CURRENT pair is scored
    alongside for comparison.
  • Score per pair = mean over horizons of the buy-vs-sell forward-return
    spread at the moderate tier plus at the strong tier, with occupancy
    floors (each strong side ≥ 25 pooled non-overlapping obs) so a pair
    can't win by firing never. Sign convention: NEGATIVE signal = bullish,
    so buy-side = v ≤ −m and its forward return should be HIGHER.

Honest notes:
  • The consensus/composite carry ≈ |IC| 0.02–0.05 — tier separations will be
    small and noisy. A rough Welch t on the moderate spread is printed; if NO
    pair separates believably, the honest anchor is the OCCUPANCY convention
    (p75 moderate / p90 strong — the markers convention), not the sweep max.
  • 8-target set (commodity/FX + small India baskets — the markers set);
    large-basket indices are skipped for runtime, same caveat as markers.

Run: python -u hero_threshold_study.py    (from research/)
"""
from __future__ import annotations
import warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # research/: put repo root on path so `from core...` resolves
# Windows consoles default to cp1252 which can't encode ← → · and other glyphs
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from convergence.normalization import (
    align_aarambh_nirnay, causal_normalize, DEFAULT_THRESHOLDS, COMPOSITE_THRESHOLDS,
)
from convergence.intelligence import _composite_signal, DEFAULT_WEIGHTS
from markers_study import _aarambh_ts, _nirnay_daily, _load, TARGETS

# Percentile-anchored candidate grid (moderate × strong). Percentiles adapt the
# same grid to each signal's own scale — the F1 principle (thresholds are only
# valid for the distribution they were anchored on) made executable.
# Widened 2026-07-13: moderate anchor 50→87.5, strong 85→99 (finer), so the
# (moderate, strong) pair search covers looser and tighter tier splits.
P_MOD = (50, 55, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5)
P_STR = (85, 87.5, 90, 92.5, 95, 96.5, 97.5, 99)
_MIN_STRONG_N = 25          # occupancy floor: pooled non-overlap obs per strong side


def _consensus_obs(hs=(10, 20)):
    """Pooled non-overlapping (signal, fwd%) per horizon for the live consensus.
    Returns (pooled_obs, per_target_signal) — the latter for the per-instrument reco."""
    pooled = {h: ([], []) for h in hs}
    per_tgt: dict = {}
    for k, tgt in enumerate(TARGETS, 1):
        try:
            ts = _aarambh_ts(tgt)
            nd = _nirnay_daily(tgt)
        except Exception as e:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
            continue
        if ts is None or nd is None or nd.empty:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped", flush=True)
            continue
        dates, raw_a, raw_n = align_aarambh_nirnay(ts, nd)
        if len(dates) < 100:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped ({len(dates)} aligned)", flush=True)
            continue
        v = (causal_normalize(np.array(raw_a, float))
             + causal_normalize(np.array(raw_n, float))) / 2.0
        per_tgt[tgt] = v
        pr = pd.to_numeric(ts["Price"], errors="coerce").to_numpy(float)
        pidx = {d: i for i, d in enumerate(ts.index)}
        for h in hs:
            V, R = pooled[h]
            for j in range(0, len(dates), h):          # stride = h → non-overlapping
                i = pidx.get(dates[j])
                if i is None or i + h >= len(pr) or not (pr[i] > 0) or not np.isfinite(v[j]):
                    continue
                V.append(v[j]); R.append((pr[i + h] / pr[i] - 1) * 100)
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(dates)} aligned days", flush=True)
    return {h: (np.array(V), np.array(R)) for h, (V, R) in pooled.items()}, per_tgt


def _composite_obs(hs=(5, 10)):
    """Pooled non-overlapping (composite, Ret_hb%) per horizon from the live frame.
    Returns (pooled_obs, per_target_signal) — the latter for the per-instrument reco."""
    from calibration_lift_study import _convergence_frame
    pooled = {h: ([], []) for h in hs}
    per_tgt: dict = {}
    for k, tgt in enumerate(TARGETS, 1):
        try:
            frame = _convergence_frame(tgt)
        except Exception as e:
            frame = None
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} ERR {e}", flush=True)
        if frame is None or len(frame) < 250:
            print(f"  [{k}/{len(TARGETS)}] {tgt:<12} skipped "
                  f"({0 if frame is None else len(frame)} frame rows)", flush=True)
            continue
        comp = _composite_signal(frame, DEFAULT_WEIGHTS.copy())
        per_tgt[tgt] = np.asarray(comp, dtype=float)
        for h in hs:
            col = f"Ret_{h}b"
            if col not in frame.columns:
                continue
            r = pd.to_numeric(frame[col], errors="coerce").to_numpy(float) * 100.0
            V, R = pooled[h]
            for j in range(0, len(comp), h):           # stride = h → non-overlapping
                if np.isfinite(comp[j]) and np.isfinite(r[j]):
                    V.append(comp[j]); R.append(r[j])
        print(f"  [{k}/{len(TARGETS)}] {tgt:<12} {len(frame)} frame rows", flush=True)
    return {h: (np.array(V), np.array(R)) for h, (V, R) in pooled.items()}, per_tgt


def _spread_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b); rough believability gauge, not a verdict."""
    if len(a) < 8 or len(b) < 8:
        return np.nan
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(max(va + vb, 1e-12)))


def _eval_pair(obs: dict, m: float, s: float):
    """Score a (moderate, strong) pair across horizons. Negative signal = bullish."""
    rows, score, ok = [], 0.0, True
    for h, (V, R) in obs.items():
        if not len(V):
            continue
        sb = R[V <= -s]; b = R[(V <= -m) & (V > -s)]
        ss = R[V >= +s]; sl = R[(V >= +m) & (V < +s)]
        hold = R[(V > -m) & (V < +m)]
        buy_all, sell_all = R[V <= -m], R[V >= +m]
        sp_mod = (buy_all.mean() - sell_all.mean()) if len(buy_all) > 3 and len(sell_all) > 3 else np.nan
        sp_str = (sb.mean() - ss.mean()) if len(sb) > 3 and len(ss) > 3 else np.nan
        if len(sb) < _MIN_STRONG_N or len(ss) < _MIN_STRONG_N:
            ok = False
        t_mod = _spread_t(buy_all, sell_all)
        rows.append((h, sb, b, hold, sl, ss, sp_mod, sp_str, t_mod))
        score += (np.nan_to_num(sp_mod) + np.nan_to_num(sp_str)) / (2 * len(obs))
    return score, ok, rows


def _sweep(name: str, obs: dict, cur: tuple[float, float]):
    hs = [h for h in obs if len(obs[h][0])]
    allv = np.abs(np.concatenate([obs[h][0] for h in hs]))
    n_tot = {h: len(obs[h][0]) for h in hs}
    print(f"\n{'=' * 78}\n  {name} — pooled non-overlap obs: "
          + " · ".join(f"+{h}d n={n_tot[h]}" for h in hs))
    # Derive the quantile set from the actual sweep grids (+ the p75/p90
    # occupancy anchor) so a widened P_MOD/P_STR can never reference a missing
    # key. sorted(set(...)) keeps it de-duplicated.
    qs = {p: float(np.quantile(allv, p / 100))
          for p in sorted(set(P_MOD) | set(P_STR) | {50, 75, 90})}
    print("  |signal| percentiles: "
          + "  ".join(f"p{p:g}={qs[p]:.3f}" for p in (50, 75, 90)
                      if p in qs))
    print(f"  current pair: moderate ±{cur[0]:g} (≈p{float(np.mean(allv <= cur[0]) * 100):.0f})"
          f" · strong ±{cur[1]:g} (≈p{float(np.mean(allv <= cur[1]) * 100):.0f})")

    # Candidates: percentile grid + the current pair.
    cands: list[tuple[str, float, float]] = [(f"p{pm:g}/p{ps:g}", qs[pm], qs[ps])
                                             for pm in P_MOD for ps in P_STR if qs[ps] > qs[pm]]
    cands.append(("current", cur[0], cur[1]))

    results = []
    for label, m, s in cands:
        score, ok, rows = _eval_pair(obs, m, s)
        results.append((score, ok, label, m, s, rows))
    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\n  {'pair':<14} {'m':>6} {'s':>6}  {'score':>7}  per-horizon: "
          f"spread_mod / spread_strong / t_mod  ·  occupancy S.BUY|BUY|HOLD|SELL|S.SELL")
    for score, ok, label, m, s, rows in results[:12] + \
            [r for r in results if r[2] == "current" and r not in results[:12]]:
        cells = []
        for h, sb, b, hold, sl, ss, sp_mod, sp_str, t_mod in rows:
            n = len(sb) + len(b) + len(hold) + len(sl) + len(ss)
            occ = "|".join(f"{len(x) / max(n, 1) * 100:.0f}" for x in (sb, b, hold, sl, ss))
            cells.append(f"+{h}d {sp_mod:+.2f}/{sp_str:+.2f}/t{t_mod:+.1f} [{occ}]")
        flag = "" if ok else "  ⚠thin-strong"
        mark = "  ←current" if label == "current" else ""
        print(f"  {label:<14} {m:>6.3f} {s:>6.3f}  {score:>+7.3f}  " + "  ".join(cells) + flag + mark)

    best = next((r for r in results if r[1]), None)
    conv = ("p75/p90", qs[75], qs[90])
    print(f"\n  best clean pair: {best[2]} (m={best[3]:.3f}, s={best[4]:.3f}, score {best[0]:+.3f})"
          if best else "\n  no pair met the occupancy floor")
    print(f"  occupancy-convention anchor (markers convention): p75/p90 = "
          f"±{conv[1]:.3f} / ±{conv[2]:.3f}")
    print("  DECISION RULE: adopt the sweep max ONLY if its moderate-spread t is"
          " believable (|t|≳2 on BOTH horizons);\n  otherwise anchor at p75/p90 —"
          " occupancy consistency beats a noise-picked spread.")
    return qs, results


def main():
    _load()
    print(f"Tattva — hero classification threshold study · {len(TARGETS)} targets", flush=True)
    t0 = time.time()

    print("\n### building CONSENSUS series (live causal construction)…", flush=True)
    cons, cons_pt = _consensus_obs((10, 20))
    _sweep("CONSENSUS (hero headline · DEFAULT_THRESHOLDS)", cons,
           (abs(DEFAULT_THRESHOLDS["buy_moderate"]), abs(DEFAULT_THRESHOLDS["buy_strong"])))

    print("\n### building COMPOSITE frames (live CrossValidator pipeline)…", flush=True)
    comp, comp_pt = _composite_obs((5, 10))
    _sweep("COMPOSITE (calibrated-row factory · COMPOSITE_THRESHOLDS)", comp,
           (abs(COMPOSITE_THRESHOLDS["buy_moderate"]), abs(COMPOSITE_THRESHOLDS["buy_strong"])))

    # ── PER-INSTRUMENT thresholds (gated: target |signal| p75/p90 vs the pooled
    #    house convention) ──────────────────────────────────────────────────
    from research._per_instrument import (per_instrument_anchor_reco, merge_overrides,
                                           print_overrides_snippet)
    print("\n" + "=" * 78)
    print("  PER-INSTRUMENT CLASSIFICATION THRESHOLDS (target p75/p90 vs pooled)")
    print("=" * 78)
    def _q(arr, q):
        a = np.abs(np.asarray(arr, float)); a = a[np.isfinite(a)]
        return (float(np.quantile(a, q)), int(len(a))) if len(a) else (float("nan"), 0)
    own = set(TARGETS)
    overrides: dict = {}
    for field, pt, pooled in (
        ("consensus_moderate", cons_pt, abs(DEFAULT_THRESHOLDS["buy_moderate"])),
        ("consensus_strong",   cons_pt, abs(DEFAULT_THRESHOLDS["buy_strong"])),
        ("composite_moderate", comp_pt, abs(COMPOSITE_THRESHOLDS["buy_moderate"])),
        ("composite_strong",   comp_pt, abs(COMPOSITE_THRESHOLDS["buy_strong"])),
    ):
        q = 0.75 if field.endswith("moderate") else 0.90
        merge_overrides(overrides, per_instrument_anchor_reco(
            field, {t: _q(pt[t], q) for t in pt}, pooled, own))
    print_overrides_snippet(overrides)

    print(f"\n  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
