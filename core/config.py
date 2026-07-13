"""
Tattva — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "2.6.0"
PRODUCT_NAME = "Tattva"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────
# TUNED-CONSTANT CONVENTION: every tuned/anchored value in this file is
# study-validated. Comments state the constant's ROLE, its validating study
# key, and any hard guard that must survive a re-tune — measurement numbers,
# run dates and report files live in research/TUNING_COVERAGE.md and the
# CHANGELOG, not here.

# Z-score band lengths driving the Z_lb/AvgZ/breadth STATE features (not the
# forecast regression). Set per the `aarambh_full` LOOKBACK_WINDOWS lever
# recommendation of the latest suite run ("ultra-short(3-10)"). NOTE: 3 bands
# quantize the breadth columns to 33% steps (was 20% with 5 bands).
LOOKBACK_WINDOWS = (3, 5, 10)
# ── Walk-forward windowing ───────────────────────────────────────────────────
# Expanding-window walk-forward bounds + refit cadence for FairValueEngine.
#   • MIN_TRAIN_SIZE — OOS forecasting starts here; also the floor on the training
#     window (kept large enough that Intelligence calibration is not starved).
#   • MAX_TRAIN_SIZE — cap on the training window. Invariant: never cap BELOW MIN —
#     capping under the floor starves the fit.
#   • REFIT_INTERVAL — re-fit the ensemble every N walk-forward rows.
# The walk-forward PURGES forward-label overlap: FairValueEngine.fit(purge=h) drops
# training rows within h of the forecast point (their labels span (t, t+h] and would
# otherwise leak into the forecast window).
# Set per the `aarambh_full` RECOMMENDED output of the latest suite run
# (best non-leaked, non-degenerate row per lever; see
# research/TUNING_COVERAGE.md for the report). Re-tune via
# research/aarambh_tuning_study.py (+ confirm_max_sweep.py for MAX×MIN).
MIN_TRAIN_SIZE = 150
MAX_TRAIN_SIZE = 350
REFIT_INTERVAL = 63
RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
# Set per the `aarambh_full` HUBER_EPSILON lever recommendation.
HUBER_EPSILON = 1.1
HUBER_MAX_ITER = 500

# ── Walk-forward ensemble members ────────────────────────────────────────────
# Which models FairValueEngine fits per walk-forward window and averages, scored by
# rank IC of the forecast vs realized forward returns (forecast R² is ~0 by design).
# Members: ols, huber, ridge, elasticnet.
#   • "ols"   — always fit internally regardless of this tuple: it powers the
#               feature-impact attribution. Anchor member.
#   • "huber" — robust (down-weights fat-tail/shock days); the dominant cost of
#               the walk-forward.
#   • "ridge" / "elasticnet" — regularized alternatives, selectable.
# ("ols","huber") — the skill default. The latest suite run recommended a
# single-member "huber" basket on IC alone, but that is deliberately NOT taken:
# a one-member ensemble collapses the Model Spread indicator (dispersion ACROSS
# members) to zero, and the IC gap between the two is within noise. ols also
# powers the feature-impact attribution regardless of this tuple.
ENSEMBLE_MODELS = ("huber")
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
# TWO lenses only — a SHORT (Tactical 10d) and a LONG (Positional 20d); 40/60/90d
# were dropped. Study-tuned: `precedent_univ` + `precedent_model`. GUARD — the
# post-purge picture is that the analog carries no horizon on its own; the short
# band is held up by the purged MODEL forecast (edge at 1–10d, fading by 15–20d).
# Tactical 10d sits inside that validated band; Positional 20d is a slow product
# lens (turnover/positioning), NOT an edge claim — the UI must not suggest one.
#
# Each preset maps to:
#   • horizon  — FWD_HORIZON: the forward log-return the engine forecasts (t→t+h).
#   • momentum — FWD_MOM_K: trailing predictor-momentum window (~2× horizon).
#   • hold     — forward-return horizons the Precedent tab AND the Intelligence
#     Val-IC/walk-forward score over. Trimmed to ONLY the computationally-validated
#     analog horizons (5/10/20d): Tactical reads 5d+10d (the current-regime-reliable
#     pair), Positional reads 10d+20d (10d anchor + the 20d full-sample peak).
#   • ddm_leak / ddm_drift / ddm_lrv — DDM smoothing; leak scales ~(10/horizon) so a
#     longer lens turns over slower (Tactical carries the CONV_DDM_LEAK_RATE value,
#     Positional half of it). drift/lrv held at the convergence defaults
#     (CONV_DDM_DRIFT_SCALE / CONV_DDM_LONG_RUN_VAR).
#     INVARIANT — ddm_drift/ddm_leak (the DDM's steady-state GAIN — see
#     analytics.ddm_filter's state equation) must be held EQUAL across lenses.
#     Leak controls MEMORY (how fast the filter forgets); drift controls GAIN
#     (how strongly it responds to new evidence) — they are independent knobs,
#     and only leak should change between lenses. A previous revision here held
#     drift fixed at 0.12 while halving leak (0.10 -> 0.05) for Positional,
#     which silently DOUBLED gain (0.12/0.05 = 2.4x vs 0.12/0.10 = 1.2x) instead
#     of just slowing turnover — the identical convergence history read twice as
#     strong through the Positional lens before the tanh bound, so conviction
#     magnitudes and STRONG/MODERATE tiers were not comparable Tactical vs
#     Positional, and Positional routinely saturated toward +-100 (audit
#     finding F3). drift is now co-scaled with leak so gain = 1.2x on both.
# Tactical ddm values = the `ddm` study's consensus-filter best row
# (CONV_DDM_LEAK_RATE/DRIFT_SCALE); Positional = half leak at the same 1.2×
# gain per the invariant above.
SIGNAL_HORIZONS = {
    "Tactical (10d)":    {"horizon": 10, "momentum": 20,
                          "hold": (5, 10),
                          "ddm_leak": 0.15, "ddm_drift": 0.18, "ddm_lrv": 50.0,
                          "blurb": "≈2 weeks · hedging & short-term trades"},
    "Positional (20d)":  {"horizon": 20, "momentum": 40,
                          "hold": (10, 20),
                          "ddm_leak": 0.075, "ddm_drift": 0.09, "ddm_lrv": 50.0,
                          "blurb": "≈1 month · positioning (analog ceiling)"},
}
DEFAULT_SIGNAL_HORIZON = "Tactical (10d)"

# Predictors that are RAW YIELD LEVELS (percent, e.g. "US 10-Year Treasury Yield"
# = ^TNX quoted as 4.25), not prices. Two consequences:
#   1. Momentum must be an arithmetic level-DIFFERENCE, not a log-return: yields
#      can print at/near/below zero (2020-21 zero-rate era; ^IRX printed ≤0 on
#      several sessions), and log() of a non-positive value is undefined. A
#      basis-point-scale diff is also the economically correct "momentum" for a
#      rate series (a rate MOVE, not a rate RETURN).
#   2. Because app.py's predictor-momentum matrix requires EVERY feature's
#      window to be finite on a given row (a walk-forward training row with any
#      NaN feature is unusable), a single log(non-positive) NaN from one yield
#      ticker used to poison that row for the WHOLE feature set — silently
#      deleting rows for every target, invisible in the prep trace, worst around
#      the 2020 zero-rate stretch (see audit finding F4). Computing these
#      columns' momentum as a level-diff instead removes the log(<=0) failure
#      mode entirely, so they no longer NaN out on a sub-zero/zero print.
RAW_YIELD_PREDICTORS = frozenset({
    "US 13-Week T-Bill Yield", "US 5-Year Treasury Yield",
    "US 10-Year Treasury Yield", "US 30-Year Treasury Yield",
})

# Honorary +1d tile on the Precedent tab — DISPLAY ONLY. The analog has no edge
# at 1d (study: `analog` — noise), so it is shown for reference/curiosity with a
# caveat and is deliberately NOT in any lens `hold` grid (kept out of the
# Intelligence Val-IC / walk-forward calibration so it can't dilute it). Set to
# None to hide the tile entirely.
PRECEDENT_HONORARY_HORIZON = 1

# Signal thresholds (ENGINE ConvictionBounded → signal mapping). Single source
# of truth for the engine (engines/aarambh.py) and the UI tabs that display the
# engine's conviction (ui/tabs/tab_aarambh.py, tab_convergence.py) — a duplicate
# UI_CONVICTION_* triple previously lived alongside this one (identical values,
# audit finding F15); consolidated here.
# Data-anchored to |ConvictionBounded|'s own pooled distribution at the
# p50/p75/p90 convention (study: `ui_anchors`). GUARD (F1 pairing):
# convergence/conviction_model.py does NOT use these — its series is the
# DDM-smoothed COMPOSITE, binned on the composite's own anchors
# (COMPOSITE_THRESHOLDS × 100; see _TIER_* there).
CONVICTION_STRONG = 27    # = p90
CONVICTION_MODERATE = 17  # = p75
CONVICTION_WEAK = 9       # = p50 ("any lean at all" floor)

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

# DDM parameters (daily conviction series). Set per the `ddm` study's
# best-mean-IC row for the ENGINE filter (leak swept at constant gain per the
# F3 invariant — drift is the co-scaled value from that same row).
# GUARD: sweep leak WITH drift co-scaled, never leak alone.
DDM_LEAK_RATE = 0.65
DDM_DRIFT_SCALE = 1.219
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH for the Nirnay engine — app.py reads them and passes them
# into engines.nirnay.run_full_analysis. Not in the Optuna search, so hand-set and
# swept structurally (research/nirnay_tuning_study.py + research/nirnay_index_check.py:
# breadth-oscillator OOS IC vs forward return).
# Set per the latest suite run's `nirnay` per-knob winners, with MSF_LENGTH
# resolved CROSS-UNIVERSE from the two result tables (`nirnay` 7 commodity
# targets + `nirnay_index` 5 equity indices, weighted by universe share
# 7:27): weighted mean |IC| peaks at MSF=18 — above both the commodity-slice
# winner (10) and the previous value (20). The other four knobs have no
# equity-side sweep, so their commodity winners apply as the only result.
NIRNAY_MSF_LENGTH = 18            # MSF oscillator rolling-window length —
                                 # universe-weighted result winner
NIRNAY_ROC_LEN = 2               # rate-of-change lookback inside MSF
NIRNAY_REGIME_SENSITIVITY = 6.0  # clarity-weight exponent
NIRNAY_BASE_WEIGHT = 0.0         # MSF share of the FIXED half of the MSF/MMR
                                 # blend (engines/nirnay: 0.5*bw + 0.5*adaptive)
                                 # — 0.0 sends the fixed half fully to MMR; MSF
                                 # still enters via the adaptive clarity half
NIRNAY_MMR_NUM_VARS = 4          # top-N macro drivers per row in MMR

# Nirnay condition thresholds (unified oscillator scale: -10 to +10). Classify the
# per-instrument signal into Oversold / Overbought / Neutral and gate buy/sell +
# divergence flags. (A ±7 "strong" tier was defined here but had NO code path, so
# it was removed rather than left as dead config.)
# Data-anchored: ±5 sits inside the p75–p85 occupancy band a per-instrument
# condition tier should occupy (study: `ui_anchors`).
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5

# ─── Convergence Layer Defaults ──────────────────────────────────────────────

# Optuna TPE trial budget for Intelligence Mode auto-calibration (app.py Phase
# 4b). Was previously read from a session-state key ("intel_n_trials") that no
# UI control ever wrote — a phantom knob permanently defaulting to 50 with no
# way to change it short of a debugger. Promoted to a named constant; raise it
# if calibration quality (Val IC stability across reruns) warrants the extra
# walk-forward-validation cost.
INTEL_N_TRIALS = 50

# Adaptive weighting base allocation. Set per the `conv_weights` study's best
# vector of the latest suite run ("direction-heavy .5").
CONV_WEIGHT_DIRECTION = 0.50
CONV_WEIGHT_BREADTH = 0.20
CONV_WEIGHT_MAGNITUDE = 0.20
CONV_WEIGHT_REGIME = 0.10

# Adaptive shift limits (±10% based on clarity ratios)
CONV_ADAPTIVE_SHIFT_MAX = 0.10

# Convergence-score label tiers (CrossValidator's signal string, ±100 scale).
# Data-anchored at p75/p90/p97.5 of |convergence_score| — the latest
# `ui_anchors` run measures the current values sitting on those occupancy
# targets, so they stand per that study's own output.
CONV_STRONG_BULLISH = -27
CONV_MODERATE_BULLISH = -18
CONV_WEAK_BULLISH = -11
CONV_WEAK_BEARISH = 11
CONV_MODERATE_BEARISH = 18
CONV_STRONG_BEARISH = 27

# DDM for convergence score. Set per the `ddm` study's best-mean-IC row for
# the CONSENSUS filter (drift = the co-scaled value from that row).
# GUARD: sweep leak WITH drift co-scaled (F3 invariant).
CONV_DDM_LEAK_RATE = 0.15
CONV_DDM_DRIFT_SCALE = 0.18
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
    # ── Direct Yield Indices (Raw %) — see RAW_YIELD_PREDICTORS below ───────
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
    # Bond volatility — the fixed-income VIX complement (rates-vol regime the
    # equity VIX misses); coverage-verified on yfinance.
    "US Bond Volatility (MOVE)":         "^MOVE",
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
    # EM FX legs — LatAm/Africa coverage (CEW only carries the basket level)
    # plus the USD/Asia crosses the USD/INR Nirnay basket already uses.
    "USD/MXN": "MXN=X",
    "USD/BRL": "BRL=X",
    "USD/ZAR": "ZAR=X",
    "USD/THB": "THB=X",
    "USD/TWD": "TWD=X",
    "USD/MYR": "MYR=X",
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
    # Refined products — the crack/product-demand factor. GUARD: both are
    # crude-plus-crack, so they are EXCLUDED from the Brent target's predictor
    # set (TARGET_EXCLUDED_PREDICTORS, same-barrel logic as WTI) while
    # remaining valid macro predictors everywhere else.
    "RBOB Gasoline": "RB=F",
    "Heating Oil": "HO=F",
    # Commodities - Agriculture
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Cotton": "CT=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    # Cocoa completes the softs; soybean oil carries the edible-oil import
    # complex (India inflation/agri).
    "Cocoa": "CC=F",
    "Soybean Oil": "ZL=F",
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
        "AYA.TO", "USAS",                       # correlation-validated adds — pure-play scarcity
                                                # relief (GATO/SILV delisted via M&A; scarcity is
                                                # structural)
        "WPM",                                  # streamer
    ],
    "Copper": [
        "FCX", "SCCO", "TECK", "ERO", "HBM",
        "IVN.TO", "CS.TO", "TGB", "NEXA",       # copper-pure miners
        "FM.TO", "LUN.TO",                      # First Quantum + Lundin — correlation-validated
                                                # adds (major pure-plays; TSX calendar matches
                                                # existing IVN.TO/CS.TO members)
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
        "ZW=F",                                 # wheat — completes the corn/soy acreage-
                                                # competition triangle (correlation in line with
                                                # the sibling futures)
    ],
    # USD/INR is FX — no producer equities exist. The basket is a CO-DIRECTIONAL
    # dollar-strength complex (rises when the rupee weakens, like USD/INR):
    # volume-bearing long-USD ETFs + a spread of USD/Asia crosses. polarity = +1.
    # Data-backed curation (11y return-correlation study — measurements in the
    # CHANGELOG): USD/Asia crosses are the strongest co-directional members;
    # UUP/USDU/DXY are volume-bearing so Nirnay's MSF microstructure runs on
    # real flow (the =X crosses carry no yfinance volume → those components run
    # ~neutral); CNY=X adds the China/EM-Asia anchor INR co-moves with.
    # GUARD — REJECTED ALTERNATIVE: an INVERSE India/EM-equity basket
    # (INDA/IBN/HDB/…, polarity -1) reads the India EQUITY regime, not the
    # currency — its daily signal is broken by US-calendar async and its weekly
    # signal is mostly Nifty beta. Co-directional wins; don't revisit without
    # re-running the correlation study.
    "USD/INR": [
        "UUP", "USDU", "DX-Y.NYB",              # long-USD anchors (UUP/USDU volume-bearing) + Dollar Index
        "SGD=X", "KRW=X", "IDR=X",              # USD/SGD, USD/KRW, USD/IDR (strongest co-directional)
        "THB=X", "PHP=X", "TWD=X",              # USD/THB, USD/PHP, USD/TWD
        "CNY=X", "MYR=X",                       # USD/CNY (China anchor), USD/MYR
    ],
    # Brent Crude (BZ=F).  Co-directional producer cross-section: integrated
    # majors + E&P + oilfield services. NO energy-sector ETFs (XLE) or oil-price
    # proxies (USO) — those double-count or duplicate the target. REFINERS
    # (VLO/MPC/PSX) are deliberately excluded: they are crack-spread businesses —
    # a crude rally can COMPRESS refining margins, so they are not cleanly
    # co-directional.
    "Brent Crude": [
        "XOM", "CVX", "COP", "BP", "SHEL", "TTE", "EQNR",   # integrated majors
        "SU", "CNQ",                                        # NYSE-listed Canadian majors
                                                            # (correlation-validated adds)
        "EOG", "OXY", "DVN", "FANG", "CTRA",                # E&P producers (HES removed — delisted, Chevron acquisition)
        "SLB", "HAL", "BKR",                                # oilfield services
    ],
    # Jeera (NCDEX cumin) has NO listed pure-play producers — same problem as
    # Cotton, handled the same way: a HYBRID cross-section of the *Indian* agri
    # economy (independent bottom-up "votes"), so Nirnay reads the Indian agri-
    # complex regime, not "jeera miners". All names are NSE (.NS), which trade
    # the same Indian calendar as the NCDEX target (clean alignment).
    # Data-backed curation (11y return-correlation study — measurements in the
    # CHANGELOG): members are the highest-linkage names within each
    # fundamentally-aligned subsector, chosen for cross-sectional dispersion.
    # GUARDS from that study:
    #   • NO global ag-soft futures (unlike Cotton's ZC/ZS/SB) — jeera trades
    #     the domestic Indian complex, not CBOT/ICE; empirically decoupled.
    #     Domestic soft-commodity exposure comes via the sugar equities instead.
    #   • NO international spice/ingredient majors (McCormick, Olam, ADM, …) —
    #     they are cumin BUYERS (a price spike is a cost headwind → inverse,
    #     not co-directional) and trade async non-Indian calendars.
    #   • True sibling spices (coriander/turmeric/guar) WOULD fit but are
    #     NCDEX-only → need their own sheets via data/sheets.py (planned).
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
    # RBOB/heating oil are refined FROM that barrel (crude + crack margin) —
    # same-barrel logic.
    "Brent Crude": ["Crude Oil", "Broad Commodity Index (DBC)",
              "Commodity Index (GSG)", "US Energy Sector",
              "RBOB Gasoline", "Heating Oil"],
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

# ── Chart palette — SINGLE SOURCE OF TRUTH (Obsidian Quant) ──────────────────
# Every chart color derives from these RGB triples: the COLOR_* constants below
# AND the rgba(...) fills/markers used inline across ui/tabs/* (via the rgba()
# helper). Change a chart hue in exactly ONE place here. Values are UNCHANGED
# from the historical hard-coded set — this is a centralization, not a recolor.
#
# KNOWN DIVERGENCE from the CSS token palette (ui/theme.css :root): the charts
# use the brighter Tailwind-400 family; the CSS surfaces (cards, chips, tables)
# use a -500/custom family (--emerald #2DD4A8 · --rose #E8555A · --cyan #06B6D4
# · --violet #8B5CF6). Only amber-gold #D4A853 is shared. To reconcile the app
# to one palette, edit the triples here to match :root — a one-line-per-color
# change, deferred because it visibly shifts rendered chart hues.
_PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "emerald": (52, 211, 153),   # #34D399 — bullish
    "rose":    (251, 113, 133),  # #FB7185 — bearish
    "cyan":    (34, 211, 238),   # #22D3EE — system / Nirnay
    "amber":   (212, 168, 83),   # #D4A853 — primary accent (shared with CSS --amber)
    "violet":  (167, 139, 250),  # #A78BFA
    "slate":   (148, 163, 184),  # #94A3B8 — muted line/marker
}


def _palette_hex(name: str) -> str:
    r, g, b = _PALETTE_RGB[name]
    return f"#{r:02X}{g:02X}{b:02X}"


def rgba(name: str, alpha) -> str:
    """Semantic chart color → ``rgba()`` string. The ONE way inline Plotly
    fills/markers should reference the palette (never a raw numeric triple), so
    the chart palette stays single-sourced in ``_PALETTE_RGB``."""
    r, g, b = _PALETTE_RGB[name]
    return f"rgba({r},{g},{b},{alpha})"


COLOR_GREEN = _palette_hex("emerald")   # #34D399
COLOR_RED = _palette_hex("rose")        # #FB7185
COLOR_GOLD = _palette_hex("amber")      # #D4A853
COLOR_CYAN = _palette_hex("cyan")       # #22D3EE
COLOR_AMBER = _palette_hex("amber")     # #D4A853
COLOR_PURPLE = _palette_hex("violet")   # #A78BFA (NOT the CSS --violet #8B5CF6 — see divergence note)
COLOR_MUTED = rgba("slate", 0.4)        # rgba(148,163,184,0.4)

# ─── UI Thresholds (centralized magic numbers) ──────────────────────────────
# NOTE (audit finding F15): UI_CONVICTION_* / UI_Z_* previously duplicated
# CONVICTION_* (above) / a dead Z_EXTREME-Z_THRESHOLD pair with identical
# values and no independent tuning need. UI callers now import CONVICTION_*
# directly; Z_EXTREME/Z_THRESHOLD had zero consumers anywhere and were
# removed rather than consolidated.

# Breadth percentage thresholds — high-breadth ALERT tier (fires on ~p96 of
# pooled breadth obs; the distribution is quantized in 20% steps by the 5
# lookback bands). Study: `ui_anchors`.
UI_BREADTH_HIGH = 60

# Agreement ratio tiers (hero INTERNALS row + convergence metric card).
# Data-anchored at p75/p90 of the pooled agreement_ratio distribution
# (study: `ui_anchors`) — "strong" must mean strong.
UI_AGREEMENT_STRONG = 0.91    # = p90
UI_AGREEMENT_MODERATE = 0.82  # = p75

# Nirnay avg-signal lean tier (metric-card coloring). Data-anchored at p75 of
# pooled |Avg_Signal|, matching UI_NIRNAY_AVG_THRESHOLD — one anchor for the
# same series everywhere (study: `ui_anchors`).
UI_NIRNAY_BULLISH = -2.9
UI_NIRNAY_BEARISH = 2.9

# ── Unified-Signal plot marker thresholds (data-anchored) ────────────────────
# The 3-row Unified Signal plot's reference lines + marker-color tiers, set to
# the p90 (strong) / p75 (moderate) quantiles of each signal's OWN pooled
# distribution (study: `markers`) so "strong/moderate" means the same
# extremeness on every row. EXTREMENESS markers, not actionable edges.
UI_CONSENSUS_STRONG = 0.39      # Row 1 · norm_avg (consensus, [-1,1]) = p90
UI_CONSENSUS_MODERATE = 0.26    #                                       = p75
UI_CONVRAW_STRONG = 50          # Row 2 · ConvictionRaw (Aarambh, ~[-100,100]) = p90
UI_CONVRAW_MODERATE = 20        #                                              = p75
UI_NIRNAY_AVG_THRESHOLD = 2.9   # Row 3 · Avg_Signal (Nirnay, [-10,10]) —
                                # single tier at p75, matching the other rows'
                                # moderate tier

# Model spread tiers — BASIS POINTS (tab_aarambh converts the raw
# log-return-std column ×1e4 before comparing). Data-anchored at ~p75/p90.
# GUARD: anchor these only from the LIVE ols+huber basket — the fast
# ridge+ols research basket's spread is ~2× tighter and would mis-anchor.
UI_MODEL_SPREAD_LOW = 20.0
UI_MODEL_SPREAD_HIGH = 35.0

# OOS R² thresholds
UI_R2_STRONG = 0.7
UI_R2_ACCEPTABLE = 0.4

# (UI_BAND_NARROW/WIDE were removed: the CI band width is pinned by the DDM's
# mean-reverting variance — measured degenerate, the tiers could never fire.
# The band itself is still drawn on the conviction chart.)

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
