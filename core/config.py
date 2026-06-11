"""
Tattva v2.0.0 — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

VERSION = "2.0.0"
PRODUCT_NAME = "TATTVA"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
MIN_TRAIN_SIZE = 50
MAX_TRAIN_SIZE = 504
REFIT_INTERVAL = 5
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 1500

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

# Z-score zone boundaries
Z_EXTREME = 2.0
Z_THRESHOLD = 1.0

# Staleness
STALENESS_DAYS = 3

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

# Default predictors for NIFTY50 use case
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)

# Google Sheets URL (should be set via secrets or environment variable)
# This is only a placeholder for type hints
DEFAULT_SHEET_URL = ""

# DDM parameters (calibrated for daily conviction series)
DDM_LEAK_RATE = 0.08
DDM_DRIFT_SCALE = 0.15
DDM_LONG_RUN_VAR = 100.0

# ─── Nirnay Engine Defaults ──────────────────────────────────────────────────

NIRNAY_MSF_LENGTH = 20
NIRNAY_ROC_LEN = 14
NIRNAY_REGIME_SENSITIVITY = 1.0
NIRNAY_BASE_WEIGHT = 0.6
NIRNAY_MMR_NUM_VARS = 5

# Nirnay signal thresholds (oscillator scale: -10 to +10)
NIRNAY_OVERSOLD = -5
NIRNAY_OVERBOUGHT = 5
NIRNAY_STRONG_BUY = -7
NIRNAY_STRONG_SELL = 7

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
    "Germany Govt Bonds (Bunds/Long)":   "BUNL.L",
    "Germany Short-Term (Schatz)":       "SDEU.L",
    "UK Gilts":                          "IGLT.L",
    "UK Gilts (Inflation-Linked)":       "INXG.L",
    "UK Corporate Bonds":                "SLXX.L",
    # ── Developed Markets Sovereign (Asia-Pacific) ─────────────────────────
    "Japan Government Bonds (Broad)":    "JGBL.L",
    "Australia Government Bonds":        "VGB.AX",
    "Canada Broad Aggregate Bond":       "XBB.TO",
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
}

# Yahoo Finance macro symbols — commodities and FX, fetched alongside Global Macro.
MACRO_SYMBOLS_YF = {
    # Major FX
    "Dollar Index": "DX-Y.NYB",
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "JPY/INR": "JPYINR=X",
    # Commodities - Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    # Commodities - Energy
    "Crude Oil": "CL=F",
    "Brent Crude": "BZ=F",
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
        "EGO", "AGI", "SSRM", "EQX", "NGD",     # producers
        "OR", "WPM", "FNV", "RGLD",             # royalty / streamers
    ],
    "Silver": [
        "PAAS", "HL", "AG", "CDE", "FSM",
        "MAG", "SVM", "EXK", "ASM",             # silver producers
        "WPM",                                  # streamer
    ],
    "Copper": [
        "FCX", "SCCO", "TECK", "ERO", "HBM",
        "IVN.TO", "CS.TO", "TGB", "NEXA",       # copper-pure miners
    ],
}

# Predictors that quasi-replicate a target and must be excluded from Aarambh
# to avoid contaminating its fair-value residual (the spread the whole system
# trades). GLTR is a precious-metals basket holding gold + silver, so it lets
# the regression "explain" the metal with itself.
TARGET_EXCLUDED_PREDICTORS = {
    "Gold":   ["Precious Metals Basket (GLTR)"],
    "Silver": ["Precious Metals Basket (GLTR)"],
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
