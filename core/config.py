"""
Tattva — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "2.5.0"
PRODUCT_NAME = "Tattva"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
# ── Walk-forward windowing ───────────────────────────────────────────────────
# RE-TUNED post-purge (2026-06-19). The walk-forward now PURGES forward-label
# overlap (FairValueEngine.fit(purge=h) drops training rows within h of the forecast
# point — their labels span (t, t+h] and otherwise leak into the forecast window).
# That removed a large future-leak the OLD study was unknowingly scored on, so the
# defaults were re-chosen honestly: 33 targets, BOTH lenses (10d & 20d), metric =
# NON-OVERLAPPING OOS rank IC of forecast vs realized return. Repro:
# research/aarambh_tuning_study.py (+ research/confirm_max_sweep.py for the MAX×MIN interaction).
# Reality check: post-purge directional IC is modest everywhere (combined ≈ 0;
# ~+0.02–0.04 at 10d, ~0/negative at 20d, US equities negative) — no setting unlocks
# a strong edge, so the analog/Precedent base rate carries more directional signal
# than the model at these horizons. These are the honest optima, not big wins.
#   • MIN_TRAIN_SIZE = 750 — the ONE real gain. Starting OOS later (better-
#     conditioned models) lifted combined IC −0.004 → +0.019 vs MIN=500 (both
#     horizons up; India-Eq +0.007 → +0.033), monotone over 300→500→750. Still
#     leaves ~1500 OOS rows, so Intelligence calibration is not starved. (The old
#     "MIN=500 is best" was a leakage artifact.)
#   • MAX_TRAIN_SIZE = 750 — confirmed at MIN=750: 750/1000/1500 tie (~+0.020,
#     within noise), so a ~3y window wins on cost + adaptivity. The one hard rule —
#     never cap BELOW MIN: MAX=500 at MIN=750 collapses to −0.035 (throws away the
#     well-conditioned window). MAX (750) ≥ MIN (750) satisfies it.
#   • REFIT_INTERVAL = 10 — the OLD "more refit = monotonically more skill"
#     (10→0.072 … 3→0.259) was PURE LEAKAGE (a fresher training tail overlapped the
#     forecast window more). Post-purge it VANISHES: 5 and 10 tie best (combined
#     −0.004), 3 and 7 are worse. So 10 is chosen — identical skill at ~2× LESS
#     walk-forward cost (notably cheaper on Streamlit Cloud). Cost ∝ 1/REFIT.
MIN_TRAIN_SIZE = 750
MAX_TRAIN_SIZE = 750
REFIT_INTERVAL = 10
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500

# ── Walk-forward ensemble members ────────────────────────────────────────────
# Which models the FairValueEngine fits per walk-forward window and averages.
# Chosen by backtest on the full real macro universe across all 5 targets, scored
# by rank IC of the forecast vs realized forward returns (the system's skill
# metric; forecast R² is ~0 by design and not used). Findings:
#   • "elasticnet" — DROPPED. ~0 IC standalone, NEGATIVE on Silver/Cotton/
#     USD-INR. L1 sparsity on already-orthogonal PCA components discards signal,
#     and it dragged the old 4-model ensemble down. Never re-add it.
#   • "ridge"      — DROPPED from the default. On PCA(20) it is ~identical to OLS
#     (corr ≈ 0.99; L2 barely matters with 20 orthogonal components), so it only
#     double-counts. (ols, huber) scored HIGHER than (ridge, ols, huber) in both
#     backtests. Still selectable as a regularization safety net.
#   • "ols"        — anchor. Strong IC, ~free, and REQUIRED for the feature-
#     impact attribution, so it is always fit regardless of this tuple.
#   • "huber"      — robust (down-weights fat-tail/shock days) and the top
#     individual model; pairs with OLS for the best ensemble. It is the dominant
#     cost (~80% of the walk-forward, up to 12s on USD/INR).
# Two sensible baskets:
#   • ("ols", "huber")  — DEFAULT. Highest measured OOS rank-IC and the most
#     robust worst-target, plus fat-tail robustness. Cost: Huber dominates the
#     walk-forward (up to ~12s on USD/INR alone).
#   • ("ridge", "ols")  — SPEED basket. ~8× faster walk-forward (USD/INR engine
#     fit ~3.5s vs ~16s with Huber), but lowest skill of every basket tested.
#     Switch to this if walk-forward latency matters more than skill.
# (ols is always fit internally regardless — it powers feature-impact attribution.)
#
# 2026-06-17 re-study (offline, cached 9y/129-ticker macro snapshot, 6 targets:
# Cotton/USD-INR/Nifty 50/Gold/Silver/Copper; metric = OOS Spearman IC of the
# 10d-forward forecast, n≈825/target). Mean IC | worst-target IC:
#     ols+huber           0.202 | 0.097   ← best on both → new DEFAULT
#     ridge+ols+huber+enet 0.199 | 0.094
#     elasticnet+ols      0.198 | 0.092
#     ridge+ols+huber     0.197 | 0.094
#     ols (baseline)      0.196 | 0.091
#     ridge+ols           0.192 | 0.088   ← prior default, ranked last
# Spread is within ~1 SE (≈0.035), so the win is consistent-direction, not large;
# elasticnet stays out (no lift over the simpler baskets). Reproduce: ensemble_study.py.
#
# 2026-06-19 POST-PURGE re-check (33 targets, both lenses, non-overlapping OOS IC;
# research/aarambh_tuning_study.py). With the leak removed the ABSOLUTE ICs collapse (the
# 0.202 above was leak-inflated) but the RANKING holds: ols+huber best (combined
# −0.001), ols −0.003, ridge+ols −0.004, ridge+ols+huber −0.002, and
# ols+huber+elasticnet WORST (−0.007). Conclusion unchanged → keep ("ols","huber");
# elasticnet stays out. (PCA components, set in app.py's engine.fit, re-confirmed at
# 20 — PCA=30 overfits hard: combined −0.059. Do not raise it.)
ENSEMBLE_MODELS = ("ols", "huber")
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 1500

# ── Signal horizon (forecast lens) ───────────────────────────────────────────
# The trader-facing "how far ahead am I reading?" selector. ALL data stays DAILY
# (yfinance 1d) — we do NOT resample to weekly bars, which would starve the
# walk-forward / conformal / Optuna machinery (9y ≈ 470 weekly rows < the 500/750
# train windows and the 1500 MIN_DATA_POINTS floor). Instead we lengthen the
# FORECAST horizon on daily bars, which keeps every day-denominated constant valid
# and the full ~2250-row sample intact.
#
# TWO lenses only — finalized from a universe-wide computational study (33 targets,
# honest non-overlapping walk-forward of the analog/precedent engine; see
# research/precedent_universe_sweep.py). The analog rank-IC across the universe is:
#     +5d 0.089 (33/33 +ve) · +10d 0.127 (30/33) · +20d 0.162 (28/33, PEAK)
#     +40d 0.058 (1/33 sig) · +60d 0.048 (0/33 sig)  ← collapses past 20d
# Recent-half IC is positive only at 5d (+0.022) and 10d (+0.034); 20d+ has decayed
# to ~0/negative. So the precedent edge lives in the 5–20d band and is DEAD beyond.
# We therefore keep a SHORT (Tactical 10d) and a LONG (Positional 20d) lens — 20d
# being the longest horizon the analog actually supports — and drop 40/60/90d.
#
# Each preset maps to:
#   • horizon  — FWD_HORIZON: the forward log-return the engine forecasts (t→t+h).
#   • momentum — FWD_MOM_K: trailing predictor-momentum window (~2× horizon).
#   • hold     — forward-return horizons the Precedent tab AND the Intelligence
#     Val-IC/walk-forward score over. Trimmed to ONLY the computationally-validated
#     analog horizons (5/10/20d): Tactical reads 5d+10d (the current-regime-reliable
#     pair), Positional reads 10d+20d (10d anchor + the 20d full-sample peak).
#   • ddm_leak / ddm_drift / ddm_lrv — DDM smoothing; leak scales ~(10/horizon) so a
#     longer lens turns over slower (Tactical 0.10 → Positional 0.05). drift/lrv held
#     at the convergence defaults (CONV_DDM_DRIFT_SCALE / CONV_DDM_LONG_RUN_VAR).
# The engine cache key includes (horizon, momentum), so both lenses coexist in one
# session — position on the long lens, hedge on the short one.
SIGNAL_HORIZONS = {
    "Tactical (10d)":    {"horizon": 10, "momentum": 20,
                          "hold": (5, 10),
                          "ddm_leak": 0.10, "ddm_drift": 0.12, "ddm_lrv": 50.0,
                          "blurb": "≈2 weeks · hedging & short-term trades"},
    "Positional (20d)":  {"horizon": 20, "momentum": 40,
                          "hold": (10, 20),
                          "ddm_leak": 0.05, "ddm_drift": 0.12, "ddm_lrv": 50.0,
                          "blurb": "≈1 month · positioning (analog ceiling)"},
}
DEFAULT_SIGNAL_HORIZON = "Tactical (10d)"

# Honorary +1d tile on the Precedent tab — DISPLAY ONLY. The analog has no edge at
# 1d (research/analog_tuning_study.py: full IC ≈ +0.04, recent ≈ 0 → noise), so it is shown
# for reference/curiosity with a caveat and is deliberately NOT in any lens `hold`
# grid (kept out of the Intelligence Val-IC / walk-forward calibration so it can't
# dilute it). Set to None to hide the tile entirely.
PRECEDENT_HONORARY_HORIZON = 1

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

# Z-score zone boundaries
Z_EXTREME = 2.0
Z_THRESHOLD = 1.0

# Staleness (in TRADING days behind — weekends ignored).
STALENESS_DAYS = 3
# Session completeness floor: the latest row is a "real" session only if at least
# this fraction of inputs posted NATIVELY (detected as "changed vs the prior row" —
# continuous prices move every session; forward-filled columns don't). Below this,
# the row is a PARTIAL session (e.g. non-US markets in, US not yet posted on a
# timezone/publish lag) that the ff-fill makes *look* complete. Full sessions run
# ~0.95+; weekend/partial rows ~0.03–0.3, so 0.6 separates them cleanly.
SESSION_FRESH_FLOOR = 0.6

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

# DDM parameters (calibrated for daily conviction series)
DDM_LEAK_RATE = 0.08
DDM_DRIFT_SCALE = 0.15
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────
# These are now the SINGLE SOURCE OF TRUTH for the Nirnay engine — app.py reads
# them and passes them into engines.nirnay.run_full_analysis (they were previously
# dead: the engine ran on hardcoded literals in app.py / nirnay.py and these
# constants were referenced nowhere). Not in the Optuna search, so they are
# hand-set — but a 2026-06-20 structural sweep (research/nirnay_tuning_study.py +
# research/nirnay_index_check.py: breadth-oscillator OOS IC vs forward return) CONFIRMS the
# current values as the best global compromise. Findings: breadth is a weak
# dimension everywhere (|IC| ≈ 0.02–0.06, no knob unlocks more); REGIME_SENSITIVITY
# is INERT (1.0/1.5/2.0 identical) and MMR_NUM_VARS ~flat; MSF_LENGTH=10 beats 20 on
# commodities (|IC| 0.057 vs 0.036) but LOSES on equity indices (0.025 vs 0.055) —
# and indices are 26 of 33 targets, so 20 stays as the cross-universe optimum.
NIRNAY_MSF_LENGTH = 20            # MSF oscillator rolling-window length
NIRNAY_ROC_LEN = 14              # rate-of-change lookback inside MSF
NIRNAY_REGIME_SENSITIVITY = 1.5  # clarity-weight exponent (corrected from a stale
                                 # 1.0 here that disagreed with the 1.5 the engine
                                 # actually ran — 1.5 preserves prior behaviour)
NIRNAY_BASE_WEIGHT = 0.6         # MSF vs MMR base blend (0.6 → 60% MSF)
NIRNAY_MMR_NUM_VARS = 5          # top-N macro drivers selected per row in MMR

# Nirnay condition thresholds (unified oscillator scale: -10 to +10). Classify the
# per-instrument signal into Oversold / Overbought / Neutral and gate buy/sell +
# divergence flags. (A ±7 "strong" tier was defined here but had NO code path, so
# it was removed rather than left as dead config.)
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5

# ─── Convergence Layer Defaults ──────────────────────────────────────────────

# Adaptive weighting base allocation
CONV_WEIGHT_DIRECTION = 0.30
CONV_WEIGHT_BREADTH = 0.25
CONV_WEIGHT_MAGNITUDE = 0.25
CONV_WEIGHT_REGIME = 0.20

# Adaptive shift limits (±10% based on clarity ratios)
CONV_ADAPTIVE_SHIFT_MAX = 0.10

# Convergence zone thresholds
CONV_STRONG_BULLISH = -60
CONV_MODERATE_BULLISH = -30
CONV_WEAK_BULLISH = -10
CONV_WEAK_BEARISH = 10
CONV_MODERATE_BEARISH = 30
CONV_STRONG_BEARISH = 60

# DDM for convergence score
CONV_DDM_LEAK_RATE = 0.10
CONV_DDM_DRIFT_SCALE = 0.12
CONV_DDM_LONG_RUN_VAR = 50.0

# Divergence detection
DIV_LOOKBACK = 20
DIV_PERSISTENCE_THRESHOLD = 5

# ─── Column Normalization ────────────────────────────────────────────────────

# ─── Global Macro Bond ETF Universe ──────────────────────────────────────────
# Adapted from Sanket — proxy for global yield dynamics via yfinance-available
# bond ETFs. Replaces the (now-broken) Stooq direct yield endpoints.
# Yields the same macro signal Stooq did, but via a stable yfinance source.

GLOBAL_MACRO_MAP = {
    # ── US Treasuries (Full Curve) ─────────────────────────────────────────
    "US Treasury 1-3 Month":             "BIL",
    "US Treasury Ultra-Short (0-1Y)":    "SHV",
    "US Treasury 0-3 Month (SGOV)":      "SGOV",
    "US Treasury Short (1-3Y)":          "SHY",
    "US Treasury Short (1-3Y) Vanguard": "VGSH",
    "US Treasury Intermediate (3-7Y)":   "IEI",
    "US Treasury Intermediate (7-10Y)":  "IEF",
    "US Treasury Intermediate Vanguard": "VGIT",
    "US Treasury Long (10-20Y)":         "TLH",
    "US Treasury Long (20Y+)":           "TLT",
    "US Treasury Long Vanguard":         "VGLT",
    "US Treasury Total Market":          "GOVT",
    # ── Direct Yield Indices (Raw %) ───────────────────────────────────────
    "US 13-Week T-Bill Yield":           "^IRX",
    "US 5-Year Treasury Yield":          "^FVX",
    "US 10-Year Treasury Yield":         "^TNX",
    "US 30-Year Treasury Yield":         "^TYX",
    # ── Inflation-Protected (TIPS) ─────────────────────────────────────────
    "US TIPS Broad Market":              "TIP",
    "US TIPS Short-Term":                "VTIP",
    "International Govt Inflation-Linked": "WIP",
    # ── Aggregate / Multi-Sector ───────────────────────────────────────────
    "US Core Aggregate Bond":            "AGG",
    "US Total Bond Market":              "BND",
    "US Floating Rate Notes":            "FLOT",
    "Global Aggregate Bond (Hedged)":    "BNDW",
    "Total International Bond (ex-US)":  "BNDX",
    # ── US Corporate: Investment Grade ─────────────────────────────────────
    "US Corporate Investment Grade":     "LQD",
    "US Corporate Short-Term (1-5Y)":    "VCSH",
    "US Corporate Intermediate":         "VCIT",
    "US Corporate Long-Term":            "VCLT",
    # ── High Yield & Alternative Credit ────────────────────────────────────
    "US High Yield Corporate":           "HYG",
    "US High Yield Corporate SPDR":      "JNK",
    "Global High Yield Bond":            "GHYG",
    "Global Green Bond":                 "BGRN",
    "Preferred Stock (Hybrid)":          "PFF",
    "Convertible Bonds":                 "CWB",
    "Fallen Angels (Recent HY)":         "FALN",
    # ── Structured & Asset-Backed ──────────────────────────────────────────
    "US Mortgage-Backed Securities":     "MBB",
    "US Mortgage-Backed Vanguard":       "VMBS",
    "US Senior Loan (Floating Rate)":    "BKLN",
    # ── Municipal Bonds ────────────────────────────────────────────────────
    "US Municipal National":             "MUB",
    "US Municipal Tax-Exempt Vanguard":  "VTEB",
    # ── Developed Markets Sovereign (Europe) ───────────────────────────────
    "International Treasury (ex-US)":    "IGOV",
    "International Treasury SPDR":       "BWX",
    "International Corporate Bonds":     "IBND",
    "Eurozone Government Bond":          "IEGA.L",
    "Eurozone Corporate Bond (IG)":      "IEAC.L",
    "Germany Govt Bonds (Bunds/Long)":   "IBGL.L",
    "Germany Short-Term (Schatz)":       "SDEU.L",
    "UK Gilts":                          "IGLT.L",
    "UK Gilts (Inflation-Linked)":       "INXG.L",
    "UK Corporate Bonds":                "SLXX.L",
    # ── Developed Markets Sovereign (Asia-Pacific) ─────────────────────────
    # (No reliable free JGB ETF on yfinance — JGBL.L returned ~13% coverage and was
    # silently dropped by the ≥20% filter, so it's omitted rather than feigned.)
    "Australia Government Bonds":        "VGB.AX",
    "Canada Broad Aggregate Bond":       "XBB.TO",
    # ── Asia-Pacific Equity Benchmarks ─────────────────────────────────────
    "Nikkei 225":                        "^N225",
    # (^TPX / TOPIX returns no data on yfinance — dropped; ^N225 covers Japan equity.)
    "KOSPI":                             "^KS11",
    "KOSDAQ":                            "^KQ11",
    # ── India Fixed Income ─────────────────────────────────────────────────
    "India Gov Bonds (LSE Proxy)":       "IIND.L",
    "India 8-13Y G-Sec":                 "LTGILTBEES.NS",
    "India 5Y G-Sec":                    "GILT5YBEES.NS",
    "India AAA PSU Bond (Bharat 2030)":  "EBBETF0430.NS",
    "India Overnight Rate (Liquid)":     "LIQUIDBEES.NS",
    # ── Emerging Markets ───────────────────────────────────────────────────
    "EM Sovereign Debt (USD)":           "EMB",
    "EM Sovereign Debt USD Invesco":     "PCY",
    "EM Sovereign (Local Currency)":     "EMLC",
    "EM High Yield Corporate":           "EMHY",
    "China Government Bonds":            "CBON",
    "China CNY Local Bonds":             "CNYB.L",
    # ── Broad Duration Proxies ─────────────────────────────────────────────
    "Short-Term Broad Bond":             "BSV",
    "Long-Term Broad Bond":              "BLV",
    # ── Equity Benchmarks (Risk-On Proxies) ────────────────────────────────
    "US Large Cap (S&P 500)":            "SPY",
    "US Nasdaq 100":                     "QQQ",
    "US Small Cap (Russell 2000)":       "IWM",
    "Global Equity (ACWI)":              "ACWI",
    "Developed ex-US Equity":            "EFA",
    # ── Volatility & Risk ──────────────────────────────────────────────────
    "Equity Volatility (VIX)":           "^VIX",
    "Mid-Term VIX Futures":              "VIXM",
    # ── China / EM / Cyclical Growth ───────────────────────────────────────
    "China Large Cap (FXI)":             "FXI",
    "China Broad (MCHI)":                "MCHI",
    "China Shanghai Composite":          "000001.SS",
    "China Shenzhen Component":          "399001.SZ",
    "Emerging Markets Equity":           "EEM",
    "Brazil Equity (Commodity)":         "EWZ",
    "Australia Equity (Commodity)":      "EWA",
    "India Equity":                      "INDA",
    # ── Sectors: Materials / Energy / Industrials / Financials ─────────────
    "US Materials Sector":               "XLB",
    "Metals & Mining (XME)":             "XME",
    "Global Miners (PICK)":              "PICK",
    "US Energy Sector":                  "XLE",
    "US Industrials Sector":             "XLI",
    "US Financials Sector":              "XLF",
    "US Regional Banks (Credit Stress)": "KRE",
    # ── Broad Commodity & Real-Asset Indices ───────────────────────────────
    "Broad Commodity Index (DBC)":       "DBC",
    "Commodity Index (GSG)":             "GSG",
    "Base Metals (DBB)":                 "DBB",
    "Agriculture (DBA)":                 "DBA",
    "Precious Metals Basket (GLTR)":     "GLTR",
    "Palladium (PALL)":                  "PALL",
    # ── Thematic / Strategic Metals ────────────────────────────────────────
    "Lithium & Battery (LIT)":           "LIT",
    "Uranium (URA)":                     "URA",
    "Steel (SLX)":                       "SLX",
    "Rare Earth / Strategic Metals":     "REMX",
    # ── Dollar & Inflation ─────────────────────────────────────────────────
    "US Dollar Bullish (UUP)":           "UUP",
    "TIPS (Inflation-Protected, SCHP)":  "SCHP",
    # ── FX Complex (currency factor — beyond UUP/DXY) ──────────────────────
    "US Dollar Bearish (UDN)":           "UDN",
    "USD Bullish Broad (USDU)":          "USDU",
    "Euro (FXE)":                        "FXE",
    "Japanese Yen (FXY)":                "FXY",
    "British Pound (FXB)":               "FXB",
    "Swiss Franc (FXF)":                 "FXF",
    "Australian Dollar (FXA)":           "FXA",
    "Canadian Dollar (FXC)":             "FXC",
    "EM Currencies (CEW)":               "CEW",
    # ── Real Estate / REITs (rate-sensitive real asset) ────────────────────
    "US REITs (VNQ)":                    "VNQ",
    "International REITs (VNQI)":         "VNQI",
    "Global REITs (REET)":               "REET",
    # ── Inflation Expectations (tradeable breakeven proxy) ─────────────────
    "Inflation Expectations (RINF)":     "RINF",
    # ── Equity Sectors (defensive/cyclical rotation — completes GICS) ──────
    "US Utilities (XLU)":                "XLU",
    "US Consumer Staples (XLP)":         "XLP",
    "US Consumer Discretionary (XLY)":   "XLY",
    "US Technology (XLK)":               "XLK",
    "US Health Care (XLV)":              "XLV",
    "US Real Estate Sector (XLRE)":      "XLRE",
    "US Communication Services (XLC)":   "XLC",
    "US Homebuilders (XHB)":             "XHB",
    "US Transports (IYT)":               "IYT",
    "Semiconductors (SMH)":              "SMH",
    # ── Equity Style Factors (risk-appetite rotation) ──────────────────────
    "US Value (VTV)":                    "VTV",
    "US Growth (VUG)":                   "VUG",
    "US Momentum (MTUM)":                "MTUM",
    "US Low Volatility (USMV)":          "USMV",
    "US High Beta (SPHB)":               "SPHB",
    "US High Dividend (VYM)":            "VYM",
    # ── Regional Equity Breadth (single-country) ───────────────────────────
    "Japan Equity (EWJ)":                "EWJ",
    "Eurozone Equity (EZU)":             "EZU",
    "South Korea Equity (EWY)":          "EWY",
    "Mexico Equity (EWW)":               "EWW",
    "Taiwan Equity (EWT)":               "EWT",
    "UK Equity (EWU)":                   "EWU",
    # ── Europe Equity Benchmarks ────────────────────────────────────────────
    "DAX (Germany)":                     "^GDAXI",
    "CAC 40 (France)":                   "^FCHI",
    "Euro Stoxx 50":                     "^STOXX50E",
    "FTSE 100 (UK)":                     "^FTSE",
    "IBEX 35 (Spain)":                   "^IBEX",
    "AEX (Netherlands)":                 "^AEX",
    "SMI (Switzerland)":                 "^SSMI",
    # ── Real Assets / Thematic ─────────────────────────────────────────────
    "Timber & Forestry (WOOD)":          "WOOD",
    "Global Infrastructure (IGF)":       "IGF",
}

# Yahoo Finance macro symbols — commodities and FX, fetched alongside Global Macro.
MACRO_SYMBOLS_YF = {
    # Major FX
    "Dollar Index": "DX-Y.NYB",
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "JPY/INR": "JPYINR=X",
    "AUD/INR": "AUDINR=X",
    "NZD/INR": "NZDINR=X",
    "CAD/INR": "CADINR=X",
    "CHF/INR": "CHFINR=X",
    "CNY/INR": "CNYINR=X",
    "SGD/INR": "SGDINR=X",
    "HKD/INR": "HKDINR=X",
    "INR/USD": "INRUSD=X",
    "USD/BDT": "BDT=X",
    "USD/CNY": "CNY=X",
    "USD/CNH": "CNH=X",
    "CNY/USD": "CNYUSD=X",
    "USD/JPY": "JPY=X",
    "JPY/USD": "JPYUSD=X",
    "USD/KRW": "KRW=X",
    "KRW/USD": "KRWUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/SEK": "USDSEK=X",
    "USD/NOK": "USDNOK=X",
    "USD/CHF": "USDCHF=X",
    "USD/VND": "USDVND=X",
    "USD/PHP": "USDPHP=X",
    "USD/IDR": "USDIDR=X",
    "USD/SGD": "USDSGD=X",
    "USD/TRY": "USDTRY=X",
    # Asia-Pacific EM Equities
    "Vietnam Equity (VNM)":             "VNM",
    "Philippines Equity (EPHE)":        "EPHE",
    "Indonesia Equity (EIDO)":          "EIDO",
    "Singapore Equity (EWS)":           "EWS",
    # Middle East Equities
    "UAE Equity (UAE)":                 "UAE",
    # Commodities - Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    # Commodities - Energy
    "Crude Oil": "CL=F",
    "Brent Crude": "BZ=F",        # Brent crude (front-month)
    "Natural Gas": "NG=F",
    # Commodities - Agriculture
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Cotton": "CT=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
}

# ─── Commodity Targets & Baskets ─────────────────────────────────────────────
# User-selectable Aarambh targets. Each maps to a yfinance front-month future
# (already present in MACRO_SYMBOLS_YF). The Aarambh predictor pool is the rest
# of MACRO_SYMBOLS_YF (commodities + FX) with the selected target excluded.

COMMODITY_TARGETS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Cotton": "CT=F",
    "Brent Crude": "BZ=F",
    "USD/INR": "INR=X",
    # Jeera (NCDEX cumin) is NOT a yfinance symbol — its daily price is pulled
    # from a published Google Sheet (data/sheets.py) and injected as a column in
    # the Aarambh matrix. The value here is a non-yfinance sentinel ticker: it
    # documents the source and is deliberately kept OUT of MACRO_SYMBOLS_YF /
    # GLOBAL_MACRO_MAP so it is never sent to yf.download.
    "Jeera": "JEERA.NCDEX",
}

# Per-target basket of related instruments fed to the Nirnay regime engine in
# place of Nifty 50 constituents. Nirnay measures *cross-sectional* breadth and
# regime, so the basket must be a set of INDEPENDENT bottom-up opinions.
#
# Curation principles:
#   • Pure single-name miners/streamers only — NO composite/index ETFs (GDX,
#     SIL, COPX) whose holdings are the other basket members (double counting),
#     and NO spot proxies (GLD/SLV/CPER) that duplicate the Aarambh target.
#   • ~15-20 names so breadth has fine granularity (not 12.5% steps).
#   • Each name a distinct company → genuine cross-sectional dispersion.
COMMODITY_BASKETS = {
    "Gold": [
        "NEM", "GOLD", "AEM", "KGC", "AU", "GFI", "HMY", "BTG", "IAG",
        "EGO", "AGI", "SSRM", "EQX",            # producers (NGD removed — unfetchable on yfinance)
        "OR", "WPM", "FNV", "RGLD",             # royalty / streamers
    ],
    "Silver": [
        "PAAS", "HL", "AG", "CDE", "FSM",
        "SVM", "EXK", "ASM",                    # silver producers (MAG removed — unfetchable on yfinance)
        "WPM",                                  # streamer
    ],
    "Copper": [
        "FCX", "SCCO", "TECK", "ERO", "HBM",
        "IVN.TO", "CS.TO", "TGB", "NEXA",       # copper-pure miners
    ],
    # Cotton has no pure-play single-name equities (the only cotton-pure
    # instrument, BAL, is a spot proxy that duplicates the Aarambh target).
    # So the basket is a HYBRID cross-section of the ag economy: agribusiness
    # processors / traders / input suppliers (bottom-up company "votes") plus a
    # few sibling softs/grains futures for complex breadth. Reframe: Nirnay here
    # reads the agricultural complex's regime, not "cotton miners".
    "Cotton": [
        "ADM", "BG", "NTR", "MOS", "CF",        # grain traders / fertilizer
        "CTVA", "FMC", "DE", "AGCO", "ANDE",    # seeds/chem / equipment / grain
        "ZC=F", "ZS=F", "SB=F",                 # corn / soy / sugar (ag complex)
    ],
    # USD/INR is FX — no producer equities exist. The basket is a CO-DIRECTIONAL
    # dollar-strength complex (rises when the rupee weakens, like USD/INR):
    # volume-bearing long-USD ETFs + a spread of USD/Asia crosses. polarity = +1.
    #
    # DATA-BACKED CURATION (11y daily+weekly return-correlation study vs USD/INR;
    # see CHANGELOG). This co-directional design beats the alternative on every
    # axis: the equal-weight basket tracks USD/INR at daily r ≈ +0.354 / weekly
    # +0.404 with LOW intra-basket redundancy (0.21 — genuinely independent
    # votes). USD/Asia crosses are the strongest members (SGD/KRW/IDR/THB/PHP
    # daily r +0.20..+0.36); USDU/UUP/DXY are weaker daily but strong weekly
    # (+0.29..+0.31) AND volume-bearing, so Nirnay's MSF microstructure runs on
    # real flow (the =X crosses carry no yfinance volume → those components run
    # ~neutral, unaffecting the price-based momentum/trend). CNY=X adds the
    # China/EM-Asia anchor INR co-moves with.
    #
    # REJECTED ALTERNATIVE — an INVERSE India/EM-equity basket (INDA/EPI/INDY/
    # IBN/HDB/INFY…, polarity -1): daily signal is broken by US-calendar async
    # (r ≈ +0.05), members are far more redundant (0.48), and its weekly signal
    # (-0.42) is mostly just Nifty beta (USD/INR vs Nifty weekly = -0.32), i.e. it
    # reads the India equity regime, not the currency. Co-directional wins.
    "USD/INR": [
        "UUP", "USDU", "DX-Y.NYB",              # long-USD anchors (UUP/USDU volume-bearing) + Dollar Index
        "SGD=X", "KRW=X", "IDR=X",              # USD/SGD, USD/KRW, USD/IDR (strongest co-directional)
        "THB=X", "PHP=X", "TWD=X",              # USD/THB, USD/PHP, USD/TWD
        "CNY=X", "MYR=X",                       # USD/CNY (China anchor), USD/MYR
    ],
    # Brent Crude (BZ=F).  Co-directional producer cross-section: integrated
    # majors + E&P + oilfield services. NO energy-sector ETFs (XLE) or oil-price
    # proxies (USO) — those double-count or duplicate the target.
    "Brent Crude": [
        "XOM", "CVX", "COP", "BP", "SHEL", "TTE", "EQNR",   # integrated majors
        "EOG", "OXY", "DVN", "FANG", "CTRA",                # E&P producers (HES removed — delisted, Chevron acquisition)
        "SLB", "HAL", "BKR",                                # oilfield services
    ],
    # Jeera (NCDEX cumin) has NO listed pure-play producers — same problem as
    # Cotton, handled the same way: a HYBRID cross-section of the *Indian* agri
    # economy (independent bottom-up "votes"), so Nirnay reads the Indian agri-
    # complex regime, not "jeera miners". All names are NSE (.NS), which trade
    # the same Indian calendar as the NCDEX target (clean alignment).
    #
    # DATA-BACKED CURATION (11y daily return-correlation study vs NCDEX jeera;
    # see CHANGELOG). Jeera is an idiosyncratic, domestic, supply-driven (monsoon
    # /sowing/rabi-harvest) market, so ALL single-name linkages are modest
    # (max daily r ~0.08) — but the equal-weight basket aggregates to r≈+0.087
    # daily / +0.082 weekly (vs Nifty's +0.076 daily that DECAYS to +0.010 weekly),
    # i.e. a genuine agri-regime signal, not market beta. Members are the highest-
    # linkage names within each fundamentally-aligned subsector, chosen for
    # cross-sectional dispersion (distinct companies, no double-counting).
    #
    # KEY FINDING — NO global ag-soft futures (unlike Cotton's ZC/ZS/SB): CT=F,
    # ZW=F, ZC=F, CC=F etc. are empirically DECOUPLED from jeera (daily r ≈ 0 to
    # negative; Cotton CT=F 3y r = -0.10). Jeera trades the domestic Indian
    # complex, not CBOT/ICE. Domestic "soft commodity" exposure is instead
    # captured via sugar equities (strongest weekly/3y linkage). True sibling
    # spices (coriander/turmeric/guar) WOULD fit but are NCDEX-only → would need
    # their own sheets via data/sheets.py (planned enhancement).
    #
    # ALSO TESTED & EXCLUDED — international spice/ingredient majors (McCormick,
    # Olam/ofi, IFF, Symrise, Givaudan, Sensient, Kerry, ADM, Bunge, Nestle,
    # Unilever): all flat-to-NEGATIVE vs jeera (Olam weekly r = -0.10, McCormick
    # 3y r = -0.08). Fundamentally correct — these are cumin BUYERS, so a price
    # spike is a cost/margin headwind (inverse, not co-directional), and they
    # trade async non-Indian calendars. Co-directional jeera exposure is almost
    # entirely a domestic-Indian-agri phenomenon.
    "Jeera": [
        # agri-inputs / agrochem / fertilizer — jeera's core monsoon & sowing driver
        "COROMANDEL.NS", "UPL.NS", "ZUARIIND.NS", "RALLIS.NS", "DHANUKA.NS",
        # sugar — domestic monsoon-levered soft commodity (best weekly/3y linkage)
        "DALMIASUG.NS", "DHAMPURSUG.NS", "EIDPARRY.NS",
        # FMCG / packaged foods — spice & staple demand (Tata Sampann packs jeera)
        "HINDUNILVR.NS", "MARICO.NS", "HERITGFOOD.NS", "TATACONSUM.NS",
        # spice-direct ethnic foods — closest listed exposure to cumin itself
        "ADFFOODS.NS",
        # farm equipment — rural-income / monsoon levered (Cotton's DE/AGCO analog)
        "ESCORTS.NS", "VSTTILLERS.NS",
        # staple grain processor + seed cycle (sowing/acreage link)
        "LTFOODS.NS", "KSCL.NS",
    ],
}

# ─── Target metadata: polarity + archetype ───────────────────────────────────
# Nirnay assumes its basket is POSITIVELY co-directional with the target (miners
# rise when the metal rises). For inverse baskets (e.g. India-equity proxies vs
# USD/INR) set polarity = -1 and the aggregate breadth is flipped to the
# target's orientation before convergence (see engines/nirnay.apply_polarity).
# Default (missing key) = +1. All current targets use co-directional baskets.
TARGET_POLARITY = {
    "Gold": +1,
    "Silver": +1,
    "Copper": +1,
    "Cotton": +1,
    "Brent Crude": +1,     # oil producers are co-directional with crude
    "USD/INR": +1,   # dollar-strength complex is co-directional with USD/INR
    "Jeera": +1,     # Indian agri complex is (loosely) co-directional with jeera
}

# Basket archetype — documentation / UI labeling only (no computational effect):
#   producer = single-name equities levered to the target (metals)
#   hybrid   = agribusiness equities + sibling futures (ag commodities)
#   proxy    = cross-asset ETFs / FX pairs expressing the same macro driver (FX)
TARGET_ARCHETYPE = {
    "Gold": "producer",
    "Silver": "producer",
    "Copper": "producer",
    "Cotton": "hybrid",
    "Brent Crude": "producer",
    "USD/INR": "proxy",
    "Jeera": "hybrid",   # Indian agribusiness/FMCG cross-section (no producers)
}

# Predictors that quasi-replicate a target and must be excluded from Aarambh
# to avoid contaminating its fair-value residual (the spread the whole system
# trades). GLTR is a precious-metals basket holding gold + silver, so it lets
# the regression "explain" the metal with itself.
TARGET_EXCLUDED_PREDICTORS = {
    "Gold":   ["Precious Metals Basket (GLTR)"],
    "Silver": ["Precious Metals Basket (GLTR)"],
    # DBA (Agriculture ETF) holds cotton + softs/grains → it would let the
    # regression "explain" cotton with a basket containing cotton.
    "Cotton": ["Agriculture (DBA)"],
    # DBB (DB Base Metals) is ~⅓ copper → a copper-bearing basket the regression
    # could explain copper with. (The broad commodity indices DBC/GSG hold only a
    # few % copper — legitimate macro drivers, so they are kept; cf. Brent, which
    # excludes them because crude DOMINATES those indices.)
    "Copper": ["Base Metals (DBB)"],
    # EVERY INR-leg cross is a replica of USD/INR: INR/USD is its exact reciprocal,
    # and X/INR = X/USD × USD/INR all carry the target's own currency leg. Excluding
    # the whole set (computed so future additions are covered automatically) keeps
    # the fair-value residual honest. Dollar Index is kept — a driver, not a replica.
    "USD/INR": [n for n in MACRO_SYMBOLS_YF
                if (n.endswith("/INR") or n == "INR/USD") and n != "USD/INR"],
    # WTI is ~the same barrel as Brent; the broad commodity indices + energy
    # sector ETF are crude-dominated → all would let crude "explain" itself.
    "Brent Crude": ["Crude Oil", "Broad Commodity Index (DBC)",
              "Commodity Index (GSG)", "US Energy Sector"],
}

# ─── Index targets (equity indices: India sectoral/broad, US, sector-ETF) ─────
# The Aarambh target is the index price; the Nirnay basket is the index's own
# constituents (resolved live + cached in data/universe.py). Their price tickers
# are merged into the fetched universe so the index level is an available column.
from data.universe import INDEX_TARGETS, INDEX_TARGETS_MAP  # noqa: E402

# Equity-index ETFs already in the macro pool that would replicate an index
# target (so they are excluded from that target's predictor set).
_US_INDEX_ETFS = ["US Large Cap (S&P 500)", "US Nasdaq 100",
                  "US Small Cap (Russell 2000)", "Global Equity (ACWI)"]
_INDIA_INDEX_ETFS = ["India Equity"]

_INDEX_NAMES = list(INDEX_TARGETS.keys())
for _name, _meta in INDEX_TARGETS.items():
    TARGET_POLARITY.setdefault(_name, +1)
    TARGET_ARCHETYPE.setdefault(_name, "index")
    # An index must not be "explained" by sibling equity indices → exclude every
    # other index column, plus the same-market broad ETFs, from its predictors.
    _excl = [n for n in _INDEX_NAMES if n != _name]
    if _meta["kind"] in ("india", "etf"):
        _excl = _excl + _INDIA_INDEX_ETFS
    elif _meta["kind"] == "us":
        _excl = _excl + _US_INDEX_ETFS
    TARGET_EXCLUDED_PREDICTORS[_name] = _excl

# Full target catalogue (commodities/FX + indices) → friendly name : yf ticker.
ALL_TARGETS = {**COMMODITY_TARGETS, **INDEX_TARGETS_MAP}

# Sidebar grouping — ordered category → target names.
TARGET_CATEGORIES: dict[str, list[str]] = {
    "Commodities": ["Gold", "Silver", "Copper", "Brent Crude", "Cotton", "Jeera"],
    "Currency (FX)": ["USD/INR"],
}
for _name, _meta in INDEX_TARGETS.items():
    TARGET_CATEGORIES.setdefault(_meta["category"], []).append(_name)

# ─── Sheet-sourced targets (non-yfinance; injected via fetcher exogenous path) ─
# Daily series pulled from a published Google Sheet (data/sheets.py SHEET_SOURCES,
# keyed by the same name) and injected into the model matrix exactly like Jeera.
# Registered here so they appear in the sidebar under their chosen category, with a
# sentinel ticker kept OUT of the yfinance maps. A sheet target may BORROW another
# index's constituents for its Nirnay basket via NIRNAY_BASKET_ALIAS below (e.g.
# Nifty 50 PE → the Nifty 50 stocks). (Jeera predates this registry and stays in
# COMMODITY_TARGETS.)
SHEET_TARGETS: dict[str, dict] = {
    "Nifty 50 - PE": {"ticker": "NIFTY50_PE.SHEET", "category": "India Indices",
                      "polarity": +1, "archetype": "index"},
}
for _sname, _smeta in SHEET_TARGETS.items():
    ALL_TARGETS[_sname] = _smeta["ticker"]
    TARGET_CATEGORIES.setdefault(_smeta["category"], []).append(_sname)
    TARGET_POLARITY.setdefault(_sname, _smeta.get("polarity", +1))
    TARGET_ARCHETYPE.setdefault(_sname, _smeta.get("archetype", "index"))
    # Don't let sibling India equity indices / the India ETF "explain" the PE.
    TARGET_EXCLUDED_PREDICTORS.setdefault(_sname, list(_INDEX_NAMES) + _INDIA_INDEX_ETFS)

# Targets that borrow another index's live constituents for their Nirnay basket
# (the target itself isn't a yfinance index, so it has no constituents of its own).
# Nifty 50 PE co-moves with the Nifty 50 complex → use the Nifty 50 stocks (the PE
# rises with constituent strength, so polarity stays +1). Resolved in
# data.constituents.get_commodity_basket.
NIRNAY_BASKET_ALIAS: dict[str, str] = {
    "Nifty 50 - PE": "Nifty 50",
}

# ─── Chart Theme ─────────────────────────────────────────────────────────────

CHART_BG = "rgba(0,0,0,0)"
CHART_GRID = "rgba(255,255,255,0.03)"
CHART_ZEROLINE = "rgba(255,255,255,0.08)"
CHART_FONT_COLOR = "#94A3B8"

# Signal colors - Obsidian Quant
COLOR_GREEN = "#34D399"  # EMERALD
COLOR_RED = "#FB7185"    # ROSE
COLOR_GOLD = "#D4A853"   # AMBER GOLD
COLOR_CYAN = "#22D3EE"   # CYAN
COLOR_AMBER = "#D4A853"  # AMBER
COLOR_PURPLE = "#A78BFA"  # VIOLET (matches CSS --violet)
COLOR_MUTED = "rgba(148,163,184,0.4)"  # SLATE

# ─── UI Thresholds (centralized magic numbers) ──────────────────────────────

# Conviction score thresholds for signal classification
UI_CONVICTION_STRONG = 60
UI_CONVICTION_MODERATE = 40
UI_CONVICTION_WEAK = 20

# Z-score thresholds for extreme values
UI_Z_EXTREME = 2.0
UI_Z_THRESHOLD = 1.0

# Breadth percentage thresholds
UI_BREADTH_HIGH = 60  # % threshold for high breadth alert

# Agreement ratio thresholds
UI_AGREEMENT_STRONG = 0.7
UI_AGREEMENT_MODERATE = 0.5

# Nirnay avg signal thresholds
UI_NIRNAY_BULLISH = -2
UI_NIRNAY_BEARISH = 2

# ── Unified-Signal plot marker thresholds (data-anchored) ────────────────────
# The 3-row Unified Signal plot's reference lines + marker-color tiers. Set to the
# p90 (strong) / p75 (moderate) quantiles of each signal's own distribution, pooled
# across 8 targets / 17.6k days (research/markers_study.py), so "strong/moderate" means the
# same extremeness on every row. This CORRECTED hand-set values that were badly
# mis-scaled: the old Row-1 ±0.5 fired only 3% of days (too tight), while Row-2 ±20
# and Row-3 ±2 fired 51% / 41% of days (too loose). The conviction rows are mean-
# reverting (high extension → lower forward return, monotone), Nirnay-avg is flat
# (interpretive guide only) — so these are EXTREMENESS markers, not actionable edges.
UI_CONSENSUS_STRONG = 0.40      # Row 1 · norm_avg (consensus, [-1,1])
UI_CONSENSUS_MODERATE = 0.25
UI_CONVRAW_STRONG = 60          # Row 2 · ConvictionRaw (Aarambh, ~[-100,100])
UI_CONVRAW_MODERATE = 40
UI_NIRNAY_AVG_THRESHOLD = 2.5   # Row 3 · Avg_Signal (Nirnay, [-10,10]) — single tier

# Model spread thresholds
UI_MODEL_SPREAD_LOW = 20.0
UI_MODEL_SPREAD_HIGH = 50.0

# OOS R² thresholds
UI_R2_STRONG = 0.7
UI_R2_ACCEPTABLE = 0.4

# Band width interpretation
UI_BAND_NARROW = 30
UI_BAND_WIDE = 60

# HMM probability threshold
UI_HMM_CONFIDENT = 0.5

# ADF/KPSS p-value thresholds
UI_ADF_SIGNIFICANT = 0.05
UI_KPSS_NOT_SIGNIFICANT = 0.05

# Chart height defaults
UI_CHART_HEIGHT_SMALL = 280
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540
UI_CHART_HEIGHT_STACKED = 680

# Data table defaults
UI_TABLE_HEIGHT = 520
UI_TABLE_HISTORY_ROWS = 10
