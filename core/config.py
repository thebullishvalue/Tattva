"""
Tattva — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.

The engine-tuning constants below are the DEFAULTS of the per-instrument config
registry: `InstrumentConfig` (routing + every Aarambh / Nirnay / Swayam / DDM /
convergence knob) → `CLASS_CONFIG_DEFAULTS` (per asset class) → `INSTRUMENT_CONFIGS`
(one explicit entry per catalogue target). The five catalogue classes (commodity,
fx, india_index, us_index, etf) are tuned PER INSTRUMENT via `PER_INSTRUMENT_TUNING`
/ `_PER_INSTRUMENT_OVERRIDES`; India/US stocks are tuned at ASSET level via
`STOCK_CONFIGS`. `get_instrument_config(target)` is the single read path (no silent
global fallback). See the InstrumentConfig section lower in this file.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "2.7.0"
PRODUCT_NAME = "Tattva"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────
# Tuned/anchored values are study-validated; measurements, run dates and report
# files live in research/TUNING_COVERAGE.md + CHANGELOG, not here. Study keys in
# the value comments (e.g. `aarambh_full`) name the validating research script.

# Z-score band lengths for the Z_lb/AvgZ/breadth STATE features (not the forecast).
LOOKBACK_WINDOWS = (3, 5, 10)          # aarambh_full: "ultra-short(3-10)"
# Expanding walk-forward for FairValueEngine: OOS starts at MIN_TRAIN; window floors
# at MIN, caps at MAX; refit every REFIT_INTERVAL rows. fit(purge=h) drops training
# rows within h of the forecast point (forward-label overlap). GUARD: MAX must stay
# ≥ MIN — the engine uses max(MAX,MIN), so MAX<MIN silently makes MAX inert.
MIN_TRAIN_SIZE = 100     # aarambh_full 2026-07-21 MIN surface is NOISE (oscillates ±0.08 with no trend: 200=+0.054,
                         # 252=+0.075, 350=-0.008, 500=-0.029, 625=+0.048 …). Rather than the 252 noise-spike argmax
                         # (which would reverse the deliberate coverage choice and ~halve OOS points for short-history
                         # stocks), pinned to the study's own coherence floor _MIN_SANE_WINDOW=100 — the smallest window
                         # that fits the PCA ensemble. Honours coverage + coherence without chasing a noise peak.
MAX_TRAIN_SIZE = 350     # aarambh_full 2026-07-21: MAX curve is flat noise ≥350 (+0.004); a stable value ≥ MIN on that
                         # plateau (≈1.4y of trading rows). MAX≥MIN guard satisfied (engine uses max(MAX,MIN)).
REFIT_INTERVAL = 63      # aarambh_full 2026-07-21 (near-noise IC; a slower refit cadence, harmless either way)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)   # aarambh_full: "ultra-narrow(0.1..10)"
HUBER_EPSILON = 4.0      # aarambh_full 2026-07-20
HUBER_MAX_ITER = 500

# Ensemble members FairValueEngine fits per window and averages (rank IC; forecast
# R²≈0 by design). "ols" is always fit (it powers feature-impact attribution).
# aarambh_full 2026-07-21 → ("ridge", "ols", "elasticnet") (best combined IC +0.007;
# a multi-member basket also restores the Model Spread dispersion indicator).
ENSEMBLE_MODELS = ("ridge", "ols", "elasticnet")
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 1500

# Single fixed forecast horizon — the trader-facing Signal-Horizon (Tactical/
# Positional) selector was removed (the edge lives at 1–10d and fades by 15–20d;
# precedent_univ/precedent_model). Data stays DAILY: weekly bars (9y ≈ 470 rows)
# would starve the walk-forward below MIN_DATA_POINTS.
FORECAST_HORIZON = 10       # FWD_HORIZON: forward log-return the engine forecasts (t→t+h)
FORECAST_MOMENTUM = 20      # FWD_MOM_K: trailing predictor-momentum window (~2× horizon)
HOLD_HORIZONS = (5, 10)     # Intelligence Val-IC / calibration grid (analog 5d+10d pair)

# Predictors that are RAW YIELD LEVELS (e.g. ^TNX at 4.25), not prices. Their
# momentum is an arithmetic level-DIFF, not a log-return: yields can print ≤0
# (zero-rate era) and log(≤0) is NaN, which would poison the whole feature row
# (audit F4). A basis-point move is also the correct "momentum" for a rate.
RAW_YIELD_PREDICTORS = frozenset({
    "US 13-Week T-Bill Yield", "US 5-Year Treasury Yield",
    "US 10-Year Treasury Yield", "US 30-Year Treasury Yield",
})

# Precedent-tab analog term-structure horizons (trading days), FIXED and decoupled
# from HOLD_HORIZONS. 1d is a normal member (weak/noisy edge, disclosed by the
# Analog-Skill chart's per-horizon IC); 60d is the long/regime end (edge fades past
# ~20d). The hero's precedent second-opinion reads at FORECAST_HORIZON (10d).
PRECEDENT_HORIZONS = (1, 3, 5, 10, 20, 60)

# ENGINE ConvictionBounded → signal mapping (engines/aarambh.py + the tabs that
# display it). Data-anchored to |ConvictionBounded| p50/p75/p90 (study: ui_anchors).
# NOT used by conviction_model.py (that bins the DDM-smoothed COMPOSITE on
# COMPOSITE_THRESHOLDS). Stood pending a confirming run (last ui_anchors saw an
# unexplained shift on this un-retuned metric).
CONVICTION_STRONG = 15.13
CONVICTION_MODERATE = 10.89
CONVICTION_WEAK = 6.56       # "any lean at all" floor

# Staleness (in TRADING days behind — weekends ignored).
STALENESS_DAYS = 3
# Session completeness floor: the latest row is a "real" session only if ≥ this
# fraction of inputs posted NATIVELY (changed vs the prior row). Full sessions run
# ~0.95+, partial/weekend rows ~0.03–0.3, so 0.6 separates them.
SESSION_FRESH_FLOOR = 0.6

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

# ENGINE DDM smoothing (daily conviction series). ddm 2026-07-20: leak 0.03.
# GUARD: sweep leak WITH drift co-scaled (drift = leak × gain 1.88), never alone.
DDM_LEAK_RATE = 0.03
DDM_DRIFT_SCALE = 0.056
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────
# Defaults for BASKET-mode Nirnay (run_full_analysis) — the InstrumentConfig
# nirnay_* fields inherit them. Self-mode targets (commodities/stocks) ignore
# nirnay_msf_length: the Swayam members carry their own lengths (swayam_lengths).
# Swept structurally by `nirnay` (FX/Jeera + commodity baskets) + `nirnay_index`
# (India indices): breadth-oscillator OOS IC vs +10d return.
#
# These BASKET-mode class defaults applied from the 2026-07-21 suite. India
# indices and US indices additionally carry their OWN per-instrument MSF via
# _PER_INSTRUMENT_OVERRIDES (which supersede this default for those targets).
# NIRNAY_MSF_LENGTH is HELD at 20: the report's three validating universes give
# contradictory class winners (nirnay 3, nirnay_index 40, per_asset us_index 18 /
# etf 12) — a flat, sign-flipping surface with no single literal recommendation —
# and short (=3) actively HURTS the 12 un-overridden India indices (nirnay_index
# table: |IC| 0.043@3 vs 0.070@20). 20 is the standing cross-universe middle that
# the per-instrument overrides then specialise. (Revisit if a reconciliation rule
# is chosen.)
NIRNAY_MSF_LENGTH = 20           # HELD (cross-universe contradiction — see note above)
NIRNAY_ROC_LEN = 45              # nirnay 2026-07-21 class-level best (|IC| 0.080 @45 vs 0.068 @60)
NIRNAY_REGIME_SENSITIVITY = 8.0  # nirnay 2026-07-21 (|IC| 0.073 @8.0; unchanged)
NIRNAY_BASE_WEIGHT = 0.0        # MSF share of the FIXED half of the MSF/MMR blend
                                # (engines/nirnay: 0.5*bw + 0.5*adaptive). nirnay
                                # 2026-07-21 best = 0.0 (unchanged).
NIRNAY_MMR_NUM_VARS = 4           # nirnay 2026-07-21 class-level best (|IC| 0.076 @4 vs 0.074 @15)

# Nirnay condition thresholds (Unified_Osc ±10): classify Oversold/Overbought/
# Neutral and gate buy/sell + divergence. ±5 = p75–p85 occupancy (study: ui_anchors).
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5

# ─── Convergence Layer Defaults ──────────────────────────────────────────────

# Optuna TPE trial budget for Intelligence Mode auto-calibration (app.py Phase 4b).
INTEL_N_TRIALS = 50

# Adaptive weighting base allocation (conv_weights: "direction-heavy .5").
CONV_WEIGHT_DIRECTION = 0.50
CONV_WEIGHT_BREADTH = 0.20
CONV_WEIGHT_MAGNITUDE = 0.20
CONV_WEIGHT_REGIME = 0.10

# Adaptive shift limits (±10% based on clarity ratios)
CONV_ADAPTIVE_SHIFT_MAX = 0.10

# Convergence-score label tiers (CrossValidator signal string, ±100). Data-anchored
# at p75/p90/p97.5 of |convergence_score| (study: ui_anchors).
CONV_STRONG_BULLISH = -15.61
CONV_MODERATE_BULLISH = -9.18
CONV_WEAK_BULLISH = -4.46
CONV_WEAK_BEARISH = 4.46
CONV_MODERATE_BEARISH = 9.18
CONV_STRONG_BEARISH = 15.61

# CONSENSUS DDM smoothing (hero trend / conviction model). ddm 2026-07-20: leak 0.01.
# GUARD: sweep leak WITH drift co-scaled (drift = leak × gain 1.20), never alone.
CONV_DDM_LEAK_RATE = 0.01
CONV_DDM_DRIFT_SCALE = 0.012
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

# Per-target basket of related instruments for the Nirnay regime engine — a set of
# INDEPENDENT bottom-up opinions (cross-sectional breadth). The metals/energy/cotton
# baskets (Gold/Silver/Copper/Cotton/Brent, now archetype "self") are no longer
# fetched live but are RETAINED for the research A/B harness (nirnay_swayam_study) —
# don't delete without updating that study. USD/INR + Jeera stay live basket targets.
# Curation: pure single-name miners only — NO composite ETFs (GDX/SIL/COPX) or spot
# proxies (GLD/SLV/CPER); ~15-20 names for breadth granularity.
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
    # Cotton has no pure-play equities → HYBRID ag-complex basket (processors /
    # traders / input suppliers + sibling softs futures); reads the ag regime.
    "Cotton": [
        "ADM", "BG", "NTR", "MOS", "CF",        # grain traders / fertilizer
        "CTVA", "FMC", "DE", "AGCO", "ANDE",    # seeds/chem / equipment / grain
        "ZC=F", "ZS=F", "SB=F",                 # corn / soy / sugar (ag complex)
        "ZW=F",                                 # wheat — completes the corn/soy acreage-
                                                # competition triangle (correlation in line with
                                                # the sibling futures)
    ],
    # USD/INR is FX — a CO-DIRECTIONAL dollar-strength complex (long-USD ETFs +
    # USD/Asia crosses; polarity +1), curated by an 11y correlation study (CHANGELOG).
    # GUARD — rejected: an inverse India-equity basket reads the equity regime, not
    # the currency; co-directional wins.
    "USD/INR": [
        "UUP", "USDU", "DX-Y.NYB",              # long-USD anchors (UUP/USDU volume-bearing) + Dollar Index
        "SGD=X", "KRW=X", "IDR=X",              # USD/SGD, USD/KRW, USD/IDR (strongest co-directional)
        "THB=X", "PHP=X", "TWD=X",              # USD/THB, USD/PHP, USD/TWD
        "CNY=X", "MYR=X",                       # USD/CNY (China anchor), USD/MYR
    ],
    # Brent Crude (BZ=F): co-directional producer cross-section (majors + E&P +
    # services). GUARD: NO energy ETFs (XLE) / oil proxies (USO) — double-count;
    # NO refiners (VLO/MPC/PSX) — crack-spread businesses, not cleanly co-directional.
    "Brent Crude": [
        "XOM", "CVX", "COP", "BP", "SHEL", "TTE", "EQNR",   # integrated majors
        "SU", "CNQ",                                        # NYSE-listed Canadian majors
                                                            # (correlation-validated adds)
        "EOG", "OXY", "DVN", "FANG", "CTRA",                # E&P producers (HES removed — delisted, Chevron acquisition)
        "SLB", "HAL", "BKR",                                # oilfield services
    ],
    # Jeera (NCDEX cumin): no listed pure-plays → HYBRID Indian agri-complex basket,
    # all NSE (.NS) for calendar alignment. GUARDS (11y correlation study, CHANGELOG):
    # NO global ag-soft futures (jeera trades the domestic complex, decoupled from
    # CBOT/ICE); NO spice/ingredient majors (McCormick/Olam/ADM — cumin BUYERS, so
    # inverse + async calendars).
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

# Target archetype. Mostly UI labeling, EXCEPT ``self`` which is COMPUTATIONAL:
# get_nirnay_mode returns "self" (Nirnay-Swayam on the target's OWN OHLCV) iff
# archetype=="self", else "basket". Vocabulary: self (own OHLCV, needs volume) ·
# producer (single-name equities) · hybrid (agri equities+futures) · proxy
# (cross-asset ETFs/FX) · index (own constituents) — the last four are basket mode.
# Commodity FUTURES run self (real volume); Jeera stays hybrid (NCDEX has no
# yfinance OHLCV/volume) and USD/INR stays proxy (FX, volume-less) by NECESSITY.
TARGET_ARCHETYPE = {
    "Gold": "self",
    "Silver": "self",
    "Copper": "self",
    "Cotton": "self",
    "Brent Crude": "self",
    "USD/INR": "proxy",
    "Jeera": "hybrid",   # Indian agribusiness/FMCG cross-section (no yfinance OHLCV → basket)
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

# ─── Stock targets (individual equities; Nirnay runs in SWAYAM self mode) ────
# The Aarambh target is the stock's own price; there is no constituent basket —
# Nirnay-Swayam formulates breadth on the stock's own OHLCV
# (engines/nirnay_self.py) instead.
#
# STOCK_TARGETS stays EMPTY — individual stocks are entered as free-form symbols
# (India/US Stocks asset classes), resolved live via resolve_stock_symbol (NSE .NS
# then BSE .BO for India; bare for US) and registered at runtime by
# register_stock_target(). The dict stays for any future pinned default stock.
STOCK_TARGETS: dict[str, dict] = {}

# category label → market key. The Asset Class selector renders these as a
# free-form "Symbol" text input (app.py) instead of the usual Target
# drop-down — they must still exist in TARGET_CATEGORIES (even with zero
# static members) so the Asset Class selector lists them at all.
FREEFORM_STOCK_CATEGORIES: dict[str, str] = {
    "India Stocks": "india",
    "US Stocks": "us",
}
for _cat in FREEFORM_STOCK_CATEGORIES:
    TARGET_CATEGORIES.setdefault(_cat, [])


def register_stock_target(display_name: str, ticker: str, market: str) -> None:
    """Register an individual-stock target (archetype 'self') at runtime.

    Idempotent — safe to call on every Streamlit rerun (module-level config
    dicts survive reruns within a process but the registration must be
    replayed from st.session_state on each one; see app.py). Applies the
    same wiring the old static STOCK_TARGETS loop used: ALL_TARGETS,
    polarity +1, archetype 'self', and the market-based Aarambh predictor
    exclusions (own-market index targets + broad ETFs — the same guard that
    feeds the Nirnay-Swayam MMR leakage filter via TARGET_EXCLUDED_PREDICTORS,
    see swayam_macro_columns above). Also installs the instrument's own
    InstrumentConfig, cloned from the market's STOCK_CONFIGS asset-class config
    with its market-based exclusions. Does NOT append to TARGET_CATEGORIES —
    freeform categories render a text input, not a list.
    """
    ALL_TARGETS[display_name] = ticker
    TARGET_POLARITY.setdefault(display_name, +1)
    TARGET_ARCHETYPE.setdefault(display_name, "self")
    excl = list(_INDEX_NAMES)
    excl += _INDIA_INDEX_ETFS if market == "india" else _US_INDEX_ETFS
    TARGET_EXCLUDED_PREDICTORS.setdefault(display_name, excl)
    # Per-instrument config from the asset-class stock config (India / US).
    INSTRUMENT_CONFIGS.setdefault(display_name, _dc_replace(
        STOCK_CONFIGS.get(market, CLASS_CONFIG_DEFAULTS["stock_us"]),
        archetype="self", polarity=+1, excluded_predictors=tuple(excl),
    ))

# ─── Nirnay-Swayam (self-referential ensemble) ───────────────────────────────
# Timescale axis (log-spaced around the tuned NIRNAY_MSF_LENGTH=20) + the ROC
# fraction used to derive each member's roc_len — see engines/nirnay_self.py
# (default_swayam_members) for how these build the 15-member grid.
NIRNAY_SWAYAM_LENGTHS = (8, 14, 22, 34, 52)   # swayam 2026-07-21 class-level best (|IC| 0.096 vs default-5)
NIRNAY_SWAYAM_ROC_FRAC = 0.85                  # swayam 2026-07-21 class-level best (|IC| 0.094 vs 0.7)

# When True, a target whose basket resolves EMPTY runs Swayam instead of
# falling back to Aarambh-only. Ship False; flip only after the acceptance
# gates in NIRNAY_SWAYAM_PLAN.md §7.2 have passed on real data.
NIRNAY_SWAYAM_FALLBACK = False


def swayam_macro_columns(target: str, macro_cols: list[str]) -> list[str]:
    """Macro candidates for self-mode MMR: drop the target's own column and
    its TARGET_EXCLUDED_PREDICTORS near-replicas.

    In basket mode a constituent correlating with the target's own macro
    column is harmless (|r|<1, a different instrument). In self mode it is
    fatal: the member's Close correlates ~1.0 with the target's own macro
    column, MMR's top-N driver selection locks onto it, predicted ≈ actual,
    deviation ≈ 0, and the MMR half of every macro-anchored member dies
    silently while mmr_quality reads perfect. This reuses the same
    self-explanation guard TARGET_EXCLUDED_PREDICTORS already applies to
    Aarambh, applied here to the MMR driver pool instead.
    """
    drop = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    return [c for c in macro_cols if c not in drop]


# ═══════════════════════════════════════════════════════════════════════════
# Per-instrument configuration registry
# ═══════════════════════════════════════════════════════════════════════════
# Every named target has its OWN full config — routing (mode/basket/polarity/
# excluded) + all tunable engine knobs. app.py reads get_instrument_config(target),
# so any instrument retunes in isolation. EVERY catalogue target has an explicit
# INSTRUMENT_CONFIGS entry (no silent fallback — get_instrument_config raises for an
# unregistered target); free-form stocks are configured per ASSET CLASS
# (STOCK_CONFIGS), registered at resolution time. Field defaults equal the former
# global constants, so the registry is behaviour-preserving until a config diverges.
from dataclasses import dataclass, replace as _dc_replace, fields as _dc_fields  # noqa: E402


@dataclass(frozen=True)
class InstrumentConfig:
    """Full per-instrument configuration (routing + all engine tuning knobs)."""

    # ── Routing / identity ──────────────────────────────────────────────────
    archetype: str = "index"                       # self | producer | hybrid | proxy | index
    polarity: int = 1                              # +1 co-directional basket, -1 inverse
    basket: tuple[str, ...] = ()                   # explicit basket (commodity/FX); () = none
    basket_alias: str | None = None                # borrow another target's basket
    excluded_predictors: tuple[str, ...] = ()      # Aarambh + Swayam-MMR leakage guard

    # ── Nirnay engine ───────────────────────────────────────────────────────
    nirnay_msf_length: int = NIRNAY_MSF_LENGTH
    nirnay_roc_len: int = NIRNAY_ROC_LEN
    nirnay_regime_sensitivity: float = NIRNAY_REGIME_SENSITIVITY
    nirnay_base_weight: float = NIRNAY_BASE_WEIGHT
    nirnay_mmr_num_vars: int = NIRNAY_MMR_NUM_VARS
    nirnay_oversold: float = NIRNAY_OVERSOLD
    nirnay_overbought: float = NIRNAY_OVERBOUGHT

    # ── Nirnay-Swayam ensemble grid ─────────────────────────────────────────
    swayam_lengths: tuple[int, ...] = NIRNAY_SWAYAM_LENGTHS
    swayam_roc_frac: float = NIRNAY_SWAYAM_ROC_FRAC

    # ── Aarambh forecast ────────────────────────────────────────────────────
    forecast_horizon: int = FORECAST_HORIZON
    forecast_momentum: int = FORECAST_MOMENTUM
    hold_horizons: tuple[int, ...] = HOLD_HORIZONS
    pca_components: int = 20                         # aarambh_full 2026-07-21 PCA surface is NOISE (whole column within
                                                     # ±0.03 of 0; argmax flipped 20→100 across runs while its IC fell
                                                     # 0.053→0.023). Kept at the DENOISING default 20 — the causal-PCA
                                                     # layer exists to reduce ~112 collinear inputs to ~20 orthogonal
                                                     # components; pca=100 barely denoises AND exceeds the MIN=100 window.

    # ── Aarambh training loop (per-instrument tunable, like nirnay/swayam) ──
    # Default to the global constants; a per-instrument / asset-class override
    # (via _PER_INSTRUMENT_OVERRIDES / STOCK_CONFIGS) retunes them for one target.
    aarambh_refit_interval: int = REFIT_INTERVAL
    aarambh_min_train_size: int = MIN_TRAIN_SIZE
    aarambh_max_train_size: int = MAX_TRAIN_SIZE
    aarambh_ensemble_models: tuple[str, ...] = ENSEMBLE_MODELS
    aarambh_ridge_alphas: tuple[float, ...] = RIDGE_ALPHAS
    aarambh_huber_epsilon: float = HUBER_EPSILON
    aarambh_lookback_windows: tuple[int, ...] = LOOKBACK_WINDOWS

    # ── Convergence DDM (consensus filter) ──────────────────────────────────
    ddm_leak: float = CONV_DDM_LEAK_RATE
    ddm_drift: float = CONV_DDM_DRIFT_SCALE
    ddm_lrv: float = CONV_DDM_LONG_RUN_VAR

    # ── Convergence dimension weights (calibration SEED / no-profile fallback) ─
    conv_weight_direction: float = CONV_WEIGHT_DIRECTION
    conv_weight_breadth: float = CONV_WEIGHT_BREADTH
    conv_weight_magnitude: float = CONV_WEIGHT_MAGNITUDE
    conv_weight_regime: float = CONV_WEIGHT_REGIME

    # ── Precedent analog term structure ─────────────────────────────────────
    precedent_horizons: tuple[int, ...] = PRECEDENT_HORIZONS

    # ── Analog matcher blend (analytics.analogs) ────────────────────────────
    analog_w_maha: float = 1.0     # Mahalanobis (covariance-aware state match)
    analog_w_traj: float = 0.0     # trajectory cosine (dropped default)
    analog_w_recv: float = 0.0     # recency decay (dropped default)

    # ── Interpretation / display tiers (data-anchored, per-instrument) ──────
    # Defaults mirror the pooled house-convention module globals (below). An
    # override retunes how THIS target's signal is CLASSIFIED/marked, not computed.
    # Structural tiers (R²/ADF/KPSS/HMM/dims) stay global. Thresholds are consumed
    # as dicts via consensus_thresholds()/composite_thresholds().
    consensus_strong: float = 0.404      # normalized-consensus [-1,1] p90
    consensus_moderate: float = 0.279    # p75
    composite_strong: float = 0.159       # directional composite [-1,1] p90
    composite_moderate: float = 0.092     # p75
    # Convergence-score display tiers (magnitudes; ×100 scale) + conviction tiers.
    conv_display_strong: float = 15.61
    conv_display_moderate: float = 9.18
    conv_display_weak: float = 4.46
    conviction_strong: float = 15.13
    conviction_moderate: float = 10.89
    conviction_weak: float = 6.56
    # Unified-Signal plot markers (per row: consensus / ConvictionRaw / Nirnay avg).
    ui_consensus_strong: float = 0.41
    ui_consensus_moderate: float = 0.28
    ui_convraw_strong: float = 66.67
    ui_convraw_moderate: float = 33.33
    ui_nirnay_avg_threshold: float = 2.87
    # Other UI display tiers.
    ui_agreement_strong: float = 0.89
    ui_agreement_moderate: float = 0.799
    ui_breadth_high: float = 60.0
    ui_model_spread_low: float = 15.82
    ui_model_spread_high: float = 29.92
    ui_nirnay_bullish: float = -2.9
    ui_nirnay_bearish: float = 2.9

    def weights_seed(self) -> dict[str, float]:
        """Convergence dimension weights as the CrossValidator/Intelligence seed."""
        return {
            "w_direction": self.conv_weight_direction,
            "w_breadth": self.conv_weight_breadth,
            "w_magnitude": self.conv_weight_magnitude,
            "w_regime": self.conv_weight_regime,
        }

    def consensus_thresholds(self) -> dict[str, float]:
        """Normalized-CONSENSUS classification cut-points (classify_normalized_signal
        `thresholds=` seed). Symmetric ±strong / ±moderate."""
        return {
            "buy_strong": -self.consensus_strong, "buy_moderate": -self.consensus_moderate,
            "sell_moderate": self.consensus_moderate, "sell_strong": self.consensus_strong,
        }

    def composite_thresholds(self) -> dict[str, float]:
        """Directional-COMPOSITE classification cut-points (classify_convergence_score
        `thresholds=` seed / Intelligence calibration seed)."""
        return {
            "buy_strong": -self.composite_strong, "buy_moderate": -self.composite_moderate,
            "sell_moderate": self.composite_moderate, "sell_strong": self.composite_strong,
        }


# Per-asset-class DEFAULT tuning. Each class is a NAMED constant so an entire class
# can be retuned in one place (e.g. give all commodities a different Swayam grid)
# without editing every member. The India-index default IS the Nifty 50 baseline —
# the other India indices copy it (per spec). Values below are the `per_asset`
# 2026-07-21 class-level bests for the classes it owns (us_index/etf MSF, stock Swayam
# grids). commodity/fx inherit the global defaults; the self-mode STOCK grids are
# PINNED to per_asset's stock recommendation so they do NOT drift with the global
# NIRNAY_SWAYAM_* (which the `swayam` study tunes on self-mode COMMODITIES).
CLASS_CONFIG_DEFAULTS: dict[str, InstrumentConfig] = {
    "commodity":   InstrumentConfig(),
    "fx":          InstrumentConfig(),
    "india_index": InstrumentConfig(),   # == Nifty 50 baseline tuning
    "us_index":    InstrumentConfig(),   # per_asset us_index MSF (18) was n=3 targets vs a NaN default — not
    "etf":         InstrumentConfig(),   # credible; etf (12) was n=1. Both inert (members carry their own MSF), so
                                         # kept at the global default rather than pinning a degenerate class-level best.
    # per_asset 2026-07-21 (asset-level, pooled Nifty100 / Nasdaq100 universes):
    "stock_india": InstrumentConfig(archetype="self", swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.7),
    "stock_us":    InstrumentConfig(archetype="self", swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.55),
}

# Free-form stock ASSET-CLASS configs — one per market, applied to any symbol
# entered under India Stocks / US Stocks (register_stock_target clones the
# right one per resolved symbol, filling in its market-based exclusions).
STOCK_CONFIGS: dict[str, InstrumentConfig] = {
    "india": CLASS_CONFIG_DEFAULTS["stock_india"],
    "us":    CLASS_CONFIG_DEFAULTS["stock_us"],
}

_CATEGORY_TO_CLASS: dict[str, str] = {
    "Commodities":   "commodity",
    "Currency (FX)": "fx",
    "India Indices": "india_index",
    "US Indices":    "us_index",
    "ETF Universe":  "etf",
}

# ── PER-INSTRUMENT vs ASSET-LEVEL tuning ─────────────────────────────────────
# The 5 catalogue classes are tuned PER INSTRUMENT (each target carries its own
# knobs on its class default); the India/US STOCK classes stay ASSET-LEVEL
# (STOCK_CONFIGS) since free-form symbols can't be pre-tuned. Invariant: the
# per-instrument classes are exactly the catalogue (non-stock) classes.
PER_INSTRUMENT_CLASSES: tuple[str, ...] = tuple(dict.fromkeys(_CATEGORY_TO_CLASS.values()))
ASSET_LEVEL_CLASSES: tuple[str, ...] = ("stock_india", "stock_us")
assert set(PER_INSTRUMENT_CLASSES).isdisjoint(ASSET_LEVEL_CLASSES)

# Fields that MAY be set per instrument = every InstrumentConfig knob EXCEPT the
# routing/identity fields (those come from the routing maps, not the tuner).
_ROUTING_FIELDS: frozenset[str] = frozenset(
    {"archetype", "polarity", "basket", "basket_alias", "excluded_predictors"})
_TUNABLE_FIELDS: frozenset[str] = frozenset(f.name for f in _dc_fields(InstrumentConfig)) - _ROUTING_FIELDS

# Explicit per-instrument tuning SLOT per catalogue target (auto-seeded, so it
# can't drift). Empty dict = inherits the class default; the slot marks the wiring target.
PER_INSTRUMENT_TUNING: dict[str, dict] = {
    _nm: {}
    for _cat, _cls in _CATEGORY_TO_CLASS.items()
    if _cls in PER_INSTRUMENT_CLASSES
    for _nm in TARGET_CATEGORIES.get(_cat, [])
}

# Per-instrument overrides, PRUNED to only those that clear a REAL statistical bar
# (2026-07-21 suite). The studies' own gates (aarambh joint Δ>=0.02; breadth margin>=0.03)
# are a fraction of one IC standard error (SE ~= 1/sqrt(n-3) ~= 0.09 at n~130), so they
# rubber-stamp noise. Re-gated here at ~1 SE, everything dropped inheriting the (coherent)
# class default:
#   - aarambh config: kept iff joint IC>0 AND (joint-default) >= 1/sqrt(n-3) AND n>=50
#     (10 of 19 survive; the rest revert to the class-default forecast).
#   - nirnay_msf_length: kept iff best|IC| >= 0.10 AND (best-default)|IC| >= 0.06
#     (nan-default us_index/etf: |IC| >= 0.14). 7 survive.
#   - swayam_lengths: both candidate spans beat their default by only ~0.03 (< bar) -> revert
#     to the class Swayam grid (the target keeps its breadth signal, just not a bespoke span).
#   - analog_w_*: dropped (the analog study's own verdict is that the class default 1/0/0 stands).
#   - ui_nirnay_avg_threshold: KEPT (Gold/Jeera) — a data-anchored DISPLAY calibration
#     (the target's own p75, gated >=25% divergence + n>=250), not an edge claim.
# Result: 16 targets (was 25).
_PER_INSTRUMENT_OVERRIDES: dict[str, dict] = {
    # -- Commodities --
    'Gold': {'ui_nirnay_avg_threshold': 3.6887},
    'Jeera': {'nirnay_msf_length': 5, 'ui_nirnay_avg_threshold': 2.1131},
    # -- Currency (FX) --
    'USD/INR': {'aarambh_refit_interval': 30, 'aarambh_max_train_size': 1250, 'aarambh_min_train_size': 1000, 'pca_components': 15, 'aarambh_ensemble_models': ('elasticnet',), 'aarambh_ridge_alphas': (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0), 'nirnay_msf_length': 10},
    # -- India Indices --
    'Nifty Next 50': {'aarambh_refit_interval': 30, 'aarambh_max_train_size': 750, 'aarambh_min_train_size': 625, 'pca_components': 100, 'aarambh_ensemble_models': ('ridge', 'huber'), 'aarambh_ridge_alphas': (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)},
    'Nifty IT': {'aarambh_refit_interval': 8, 'aarambh_max_train_size': 252, 'aarambh_min_train_size': 100, 'pca_components': 8, 'aarambh_huber_epsilon': 1.35, 'aarambh_ensemble_models': ('elasticnet',), 'aarambh_ridge_alphas': (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)},
    'Nifty Pharma': {'nirnay_msf_length': 10},
    'Nifty 100': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 500, 'aarambh_min_train_size': 252, 'pca_components': 75, 'aarambh_ensemble_models': ('ridge', 'elasticnet'), 'aarambh_ridge_alphas': (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)},
    'Nifty Midcap 50': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 625, 'aarambh_min_train_size': 252, 'pca_components': 8, 'aarambh_huber_epsilon': 1.5, 'aarambh_ensemble_models': ('ridge',), 'aarambh_ridge_alphas': (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)},
    'Nifty Pvt Bank': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 500, 'aarambh_min_train_size': 252, 'aarambh_ensemble_models': ('ols',), 'aarambh_ridge_alphas': (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)},
    'Nifty PSU Bank': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 750, 'aarambh_min_train_size': 625, 'pca_components': 60, 'aarambh_huber_epsilon': 2.5, 'aarambh_ensemble_models': ('elasticnet',), 'nirnay_msf_length': 3},
    'Nifty Realty': {'nirnay_msf_length': 10},
    'Nifty Media': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 252, 'aarambh_min_train_size': 200, 'pca_components': 8, 'aarambh_huber_epsilon': 1.35, 'aarambh_ensemble_models': ('ridge',)},
    'Nifty PSE': {'aarambh_refit_interval': 18, 'aarambh_max_train_size': 750, 'aarambh_min_train_size': 625, 'pca_components': 8, 'aarambh_huber_epsilon': 3.0, 'aarambh_ensemble_models': ('ridge', 'elasticnet'), 'aarambh_ridge_alphas': (0.01, 0.1, 1.0, 10.0, 100.0)},
    # -- US Indices --
    'S&P 500': {'nirnay_msf_length': 40},
    'Nasdaq 100': {'nirnay_msf_length': 18},
    # -- ETF Universe --
    'India Sector ETFs': {'aarambh_refit_interval': 63, 'aarambh_max_train_size': 350, 'aarambh_min_train_size': 252, 'pca_components': 5, 'aarambh_ensemble_models': ('ridge', 'elasticnet')},
}
_bad_fields = {k for _ov in _PER_INSTRUMENT_OVERRIDES.values() for k in _ov if k not in _TUNABLE_FIELDS}
assert not _bad_fields, (
    f"_PER_INSTRUMENT_OVERRIDES sets non-tunable/unknown fields {sorted(_bad_fields)} "
    f"(routing fields come from the routing maps; valid tuning fields: {sorted(_TUNABLE_FIELDS)})")
_bad_names = [n for n in _PER_INSTRUMENT_OVERRIDES if n not in PER_INSTRUMENT_TUNING]
assert not _bad_names, (
    f"_PER_INSTRUMENT_OVERRIDES targets {_bad_names} are not per-instrument-class "
    "catalogue targets (stocks are tuned at asset-class level via STOCK_CONFIGS)")
for _nm, _ov in _PER_INSTRUMENT_OVERRIDES.items():
    PER_INSTRUMENT_TUNING[_nm].update(_ov)

# Build one explicit InstrumentConfig per named target: class-default tuning +
# that instrument's own routing (archetype/polarity/basket/alias/excluded from
# the maps above) + its per-instrument tuning overrides (empty until wired).
# Every India index gets its own entry copying the Nifty 50 baseline; Nifty 50
# and Nifty 50 - PE differ only where their routing/tuning differ.
INSTRUMENT_CONFIGS: dict[str, InstrumentConfig] = {}
for _cat, _names in TARGET_CATEGORIES.items():
    _cls = _CATEGORY_TO_CLASS.get(_cat)
    if _cls is None:
        continue   # free-form stock categories — configured per-symbol at runtime
    _base = CLASS_CONFIG_DEFAULTS[_cls]
    for _nm in _names:
        INSTRUMENT_CONFIGS[_nm] = _dc_replace(
            _base,
            archetype=TARGET_ARCHETYPE.get(_nm, _base.archetype),
            polarity=TARGET_POLARITY.get(_nm, _base.polarity),
            basket=tuple(COMMODITY_BASKETS.get(_nm, ())),
            basket_alias=NIRNAY_BASKET_ALIAS.get(_nm),
            excluded_predictors=tuple(TARGET_EXCLUDED_PREDICTORS.get(_nm, ())),
            **PER_INSTRUMENT_TUNING.get(_nm, {}),   # per-instrument knob overrides
        )

# Completeness guard ("defining them is a must"): every non-stock catalogue
# target must have resolved to an explicit config at import time.
_missing_cfg = [t for t in ALL_TARGETS if t not in INSTRUMENT_CONFIGS]
assert not _missing_cfg, f"targets without an InstrumentConfig: {_missing_cfg}"


def get_instrument_config(target: str) -> "InstrumentConfig":
    """Return the explicit per-instrument config, or raise if unregistered.

    No silent fallback — a target reaching the pipeline without a config is a
    registration bug (a free-form stock must be registered via
    register_stock_target before analysis; every catalogue target is registered
    at import). Callers that need a tolerant default can catch KeyError.
    """
    cfg = INSTRUMENT_CONFIGS.get(target)
    if cfg is None:
        raise KeyError(
            f"No InstrumentConfig registered for target {target!r}. Every "
            "instrument must have an explicit config (see INSTRUMENT_CONFIGS / "
            "register_stock_target)."
        )
    return cfg

# ─── Chart Theme ─────────────────────────────────────────────────────────────

CHART_BG = "rgba(0,0,0,0)"
CHART_GRID = "rgba(255,255,255,0.03)"
CHART_ZEROLINE = "rgba(255,255,255,0.08)"
CHART_FONT_COLOR = "#94A3B8"

# ── Chart palette — SINGLE SOURCE OF TRUTH (Obsidian Quant) ──────────────────
# Every chart color derives from these RGB triples (COLOR_* + inline rgba() via the
# rgba() helper) — change a chart hue in ONE place here. NOTE: charts use the
# brighter Tailwind-400 family; CSS surfaces (theme.css :root) use a -500 family
# (only amber-gold #D4A853 is shared); reconciling to one palette shifts chart hues.
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
UI_AGREEMENT_STRONG = 0.89    # = p90
UI_AGREEMENT_MODERATE = 0.799  # = p75

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
UI_CONSENSUS_STRONG = 0.41      # Row 1 · norm_avg (consensus, [-1,1]) = p90 (markers 2026-07-20)
UI_CONSENSUS_MODERATE = 0.28    #                                       = p75 (markers 2026-07-20)
UI_CONVRAW_STRONG = 66.67       # Row 2 · ConvictionRaw (Aarambh, ~[-100,100]) = p90 (markers 2026-07-20)
UI_CONVRAW_MODERATE = 33.33     #                                              = p75 (markers 2026-07-20)
UI_NIRNAY_AVG_THRESHOLD = 2.87   # Row 3 · Avg_Signal (Nirnay, [-10,10]) —
                                # single tier at p75, matching the other rows'
                                # moderate tier

# Model spread tiers — BASIS POINTS (tab_aarambh converts the raw
# log-return-std column ×1e4 before comparing). Data-anchored at ~p75/p90.
# GUARD: anchor these only from the LIVE ols+huber basket — the fast
# ridge+ols research basket's spread is ~2× tighter and would mis-anchor.
UI_MODEL_SPREAD_LOW = 15.82
UI_MODEL_SPREAD_HIGH = 29.92

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
