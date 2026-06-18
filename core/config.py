"""
Tattva — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — Merged from both Aarambh (correl.py) and Nirnay (nirnay_core.py) monoliths.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "2.1.0"
PRODUCT_NAME = "Tattva"
COMPANY = "@thebullishvalue"

# ─── Aarambh Engine Defaults ─────────────────────────────────────────────────

LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
# ── Walk-forward windowing ───────────────────────────────────────────────────
# Chosen by backtest on the real macro universe, all 5 targets, scored by rank IC
# of the forecast vs realized forward return (holdout = fixed last 504 pts so the
# comparison is apples-to-apples; full-OOS shown too). The OLD (1500/2000/10) was
# the WEAKEST point tested.
#   • MIN_TRAIN_SIZE — where OOS forecasting begins. Late-forecast skill is
#     independent of it (the window is MAX-capped regardless), so a large value
#     only wastes history: MIN=1500 left just 776 OOS rows — starving the
#     Intelligence calibration + walk-forward IC — for NO recent-skill gain.
#     MIN=500 yields ~1786 OOS rows and the best full-OOS IC; first fit stays
#     well-conditioned (»20 PCA components).
#   • MAX_TRAIN_SIZE — training-window cap. 750 matched/beat 2000 on IC across
#     targets (regimes shift → a ~3y window is more adaptive) and is cheaper per
#     fit. Large windows buy nothing with ~9y of data.
#   • REFIT_INTERVAL — refit cadence; the dominant skill lever (forecast horizon
#     is 10d, so a stale model decays as its 20d-momentum signal turns over).
#     Measured avg holdout IC (ridge+ols): 10→0.072, 7→0.147, 5→0.194, 3→0.259,
#     monotone & consistent across all 5 targets. Cost ∝ 1/REFIT (more chunks).
#     5 = chosen sweet spot (~2.7× current IC at ~2× walk-forward cost); raise to
#     7/10 to favour speed, drop to 3 for max skill (~2× the cost of 5).
MIN_TRAIN_SIZE = 500
MAX_TRAIN_SIZE = 750
REFIT_INTERVAL = 5
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
ENSEMBLE_MODELS = ("ols", "huber")
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
    # Commodities - Metals
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    # Commodities - Energy
    "Crude Oil": "CL=F",
    "UKOIL": "BZ=F",        # Brent crude (front-month) — "UKOIL" per the target naming
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
    "UKOIL": "BZ=F",
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
    # UKOIL = Brent crude. Co-directional producer cross-section: integrated
    # majors + E&P + oilfield services. NO energy-sector ETFs (XLE) or oil-price
    # proxies (USO) — those double-count or duplicate the target.
    "UKOIL": [
        "XOM", "CVX", "COP", "BP", "SHEL", "TTE", "EQNR",   # integrated majors
        "EOG", "OXY", "DVN", "FANG", "HES", "CTRA",         # E&P producers
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
    "UKOIL": +1,     # oil producers are co-directional with crude
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
    "UKOIL": "producer",
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
    # The other INR crosses are quasi-replicas of USD/INR (all priced in INR).
    # Dollar Index is kept — it is a legitimate driver, not a replica.
    "USD/INR": ["EUR/INR", "GBP/INR", "JPY/INR"],
    # WTI is ~the same barrel as Brent; the broad commodity indices + energy
    # sector ETF are crude-dominated → all would let crude "explain" itself.
    "UKOIL": ["Crude Oil", "Broad Commodity Index (DBC)",
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
    "Commodities": ["Gold", "Silver", "Copper", "UKOIL", "Cotton", "Jeera"],
    "Currency (FX)": ["USD/INR"],
}
for _name, _meta in INDEX_TARGETS.items():
    TARGET_CATEGORIES.setdefault(_meta["category"], []).append(_name)

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
