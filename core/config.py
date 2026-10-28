"""
Tattva — Configuration constants, thresholds, column mappings, and shared defaults.
तत्त्व (Tattva) — "Principle / Essence"

CORE — macro universe, target catalogue and structural knobs.

The engine-tuning constants below are the DEFAULTS of the per-instrument config
registry: `InstrumentConfig` (every Mūla / Swayam knob and the horizon) →
`CLASS_CONFIG_DEFAULTS` (per asset class) → `INSTRUMENT_CONFIGS` (one explicit
entry per catalogue target, its class's config). India/US stocks are configured
at ASSET level via `STOCK_CONFIGS`. `get_instrument_config(target)` is the single
read path (no silent global fallback). See the InstrumentConfig section lower in
this file. What each target IS (ticker, asset class, currency, tradeable, twins,
leakage exclusions) lives in `core.instruments`, which is built from the maps
below.
"""

import pandas as pd

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "2.9.0"
PRODUCT_NAME = "Tattva"
COMPANY = "@thebullishvalue"

# ─── Mūla Engine Defaults ─────────────────────────────────────────────────────
# The Mūla (मूल) engine replaced Aarambh's walk-forward
# ensemble regression. It is a RECURSIVE dynamic cointegrating regression of
# log price on the integrated factors of the macro cross-section — there is no
# training window, no refit cadence and no ensemble, so the former
# MIN/MAX_TRAIN_SIZE, REFIT_INTERVAL, RIDGE_ALPHAS, HUBER_* and
# ENSEMBLE_MODELS knobs (and every per-instrument tuning of them) are gone
# rather than retained as inert settings.
#
# Where a value below was set by a measurement, the measurement is quoted next
# to it; the CHANGELOG carries the history of each change.

# STRUCTURAL — the window of Decision A's candidate claim unit,
# core.semantics.claim_oscillator(FVO, 'robust_z_252'): FVO against the median
# and MAD of the 252 published sessions before it (a year). Pre-registered for
# the pooled-prior study, which records the decision; nothing applies it
# without an owner-signed change. (The 21/63/252-session stack of robust
# z-scores of the gap whose one-year window this keeps read ρ 0.96–0.99 with
# FVO itself, and went.)
CLAIM_CENTRING_WINDOW = 252

# Observations absorbed before a valuation is published. One year is where an
# exponentially weighted correlation matrix over ~200 instruments has enough
# weight for the Marchenko-Pastur edge to be meaningful; publishing earlier
# would be publishing the prior.
FVO_BURN_IN = 252
# Prints an instrument must have accumulated BY TIME t before it may enter the
# cross-section at t. A second moment is not estimable from less, and the gate
# is applied forward in time so admission never retroactively changes.
FVO_MIN_PRINTS = 250
# Discount grid for the valuation regression — implied coefficient memories of
# ~4y, ~8y, ~40y and permanent. The restriction to the slow end is a modelling
# commitment, not a tuned choice: scoring discount factors by one-step
# predictive likelihood is degenerate for a LEVEL regression (the model that
# tracks price most closely always wins, and its limit is the useless "fair
# value = price"). A long-run relation that re-estimates itself in months is
# not a long-run relation. Within the slow family the data still selects.
FVO_VALUATION_DELTAS = (0.999, 0.9995, 0.9999, 1.0)

# A TARGET-history precondition, nothing more: a run needs this many sessions
# of the target's own prints on the prepared spine (data.prep.prepare_panel
# raises below it). It no longer decides which predictors survive — young
# instruments enter the cross-section through FVO_MIN_PRINTS instead.
MIN_DATA_POINTS = 1500

# STRUCTURAL — the first session of the analysis history, fixed for every
# target, every fetch and every run. It is the first weekday of 2017, the
# calendar year in which 2.8.0's nine-year window (run date − 9y) began; it was
# fixed before any IC was computed on it and is never tuned. The published
# history starts at max(ANALYSIS_EPOCH, the target's first print), so it does
# not roll forward with the run date. Changing it rewrites every published
# value: that is a versioned history rewrite, never a tweak.
ANALYSIS_EPOCH = pd.Timestamp("2017-01-02")

# Evaluation horizons. FORECAST_HORIZON is no longer a label horizon the engine
# is trained on (nothing is): it is THE horizon h of the call's estimand — the
# holding period every grade scores against and the window the UI projects the
# current mispricing over. Data stays DAILY.
FORECAST_HORIZON = 10       # scoring / display horizon (trading days)

# Predictors that are RAW YIELD LEVELS (e.g. ^TNX at 4.25), not prices. The Mūla
# cross-section is a panel of PRICES: it takes logs and first differences, and
# a rate series can print ≤0 (zero-rate era), where log(≤0) is NaN. They are
# excluded from the valuation panel rather than transformed — a yield level is
# not an instrument whose price the target can be valued against.
RAW_YIELD_PREDICTORS = frozenset({
    "US 13-Week T-Bill Yield", "US 5-Year Treasury Yield",
    "US 10-Year Treasury Yield", "US 30-Year Treasury Yield"})

# Staleness (in TRADING days behind — weekends ignored).
STALENESS_DAYS = 3

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

# ─── Swayam Breadth Defaults ─────────────────────────────────────────────────
# The per-series kernel's structural knobs. These are shared by every member of
# the Swayam view bank; the member's own timescale comes from swayam_lengths.
#
# SWAYAM_MSF_LENGTH / SWAYAM_ROC_LEN are gone: they parameterised the single
# basket-mode read, and the Swayam members each carry their own length. Their
# tuning history is why no single length is chosen: three validating universes
# returned contradictory class winners (3, 40, 18, 12) on a flat sign-flipping
# surface. The bank keeps several lengths and counts every view once (equal
# weights); no learner re-weights them, and the breadth is graded as the
# Swayam-confirmation sizer instead.
SWAYAM_REGIME_SENSITIVITY = 8.0  # swayam 2026-07-21 (|IC| 0.073 @8.0)
SWAYAM_BASE_WEIGHT = 0.0         # MSF share of the FIXED half of the MSF/MMR
                                 # blend (kernel: 0.5*bw + 0.5*adaptive).
SWAYAM_MMR_NUM_VARS = 4          # swayam 2026-07-21 class-level best

# Condition thresholds on Unified_Osc (±10 scale): a view votes Oversold below
# SWAYAM_OVERSOLD and Overbought above SWAYAM_OVERBOUGHT, else Neutral. On the
# views that can vote, ±5 sits at about p82-83 of |osc|; its originating study
# is not retained (core.knobs: structural-inherited). A price-only view cannot
# reach it at the shipped base weight (engines.swayam.kernel.can_vote).
SWAYAM_OVERSOLD = -5
SWAYAM_OVERBOUGHT = 5

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
    # An MSCI India EQUITY ETF listed in London (USD), not a bond fund: 16-17%
    # annualised vol, 10d corr 0.87 with INDA and 0.83 with Nifty 50, 0.18 with
    # India 8-13Y G-Sec. The display name is a historical mislabel kept as the
    # panel's column key; core.instruments files it under Equity and it is
    # excluded wherever INDA is (_INDIA_INDEX_ETFS, the USD/INR twins).
    "India Gov Bonds (LSE Proxy)":       "IIND.L",
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
    "Global Infrastructure (IGF)":       "IGF"}

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
    # plus the USD/Asia crosses the USD/INR Swayam basket already uses.
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
    "Soybean Oil": "ZL=F"}

# ─── Commodity, FX and crypto targets ────────────────────────────────────────
# User-selectable targets outside the index catalogue, each with its yfinance
# ticker (a sentinel for the sheet-sourced Jeera). The Mūla predictor pool is
# NOT built from this map: it is the registered universe of the whole shared
# panel (core.instruments.predictor_universe) minus RAW_YIELD_PREDICTORS, the
# sheet series, the target, its TARGET_EXCLUDED_PREDICTORS and its twins.

COMMODITY_TARGETS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Cotton": "CT=F",
    "Brent Crude": "BZ=F",
    "USD/INR": "INR=X",
    # Same ticker as the "Dollar Index" PREDICTOR column in MACRO_SYMBOLS_YF.
    # That is intentional and safe only because TARGET_EXCLUDED_PREDICTORS
    # drops the column (and its replicas) when DXY is the target — see there.
    "Dollar Index": "DX-Y.NYB",
    # Base metals. Both are LME contracts quoted in USD/tonne and print on the
    # same weekday calendar as the rest of the commodity complex.
    "Aluminium": "ALI=F",
    "Zinc": "ZNC=F",
    # Crypto trades 7 days a week — 366 daily bars a year against ~252 for a
    # future — while every macro predictor prints on weekdays only. The shared
    # matrix is a weekday spine (data/fetcher.py drops weekend rows when it
    # builds it), so weekend bars never reach the model rather than being
    # valued against stale macro. A Saturday move is real, but nothing in the
    # cross-section was open to explain it. (calendars.py treats crypto as
    # 24×7, and counts "days behind" on weekdays to match this spine.)
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "BNB": "BNB-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    # Jeera (NCDEX cumin) is NOT a yfinance symbol — its daily price is pulled
    # from a published Google Sheet (data/sheets.py) and joined as the TARGET
    # column only when Jeera is the target; it is never a predictor. The value
    # here is a non-yfinance sentinel ticker: it documents the source and is
    # deliberately kept OUT of MACRO_SYMBOLS_YF / GLOBAL_MACRO_MAP so it is
    # never sent to yf.download.
    "Jeera": "JEERA.NCDEX"}

# ─── Target metadata ─────────────────────────────────────────────────────────
# What used to live here: TARGET_POLARITY (was the target's proxy basket
# co-directional with it, or did breadth need flipping?) and TARGET_ARCHETYPE
# (self / producer / hybrid / proxy / index — which routed each target to
# Swayam self-mode or to a constituent basket). Both are gone with the basket
# engine. Every target now reads breadth off its own price through Swayam, so
# there is no proxy whose orientation could disagree and no routing decision
# left to encode. See engines/swayam/ensemble.py for why the proxy read went.

# Predictors that quasi-replicate a target and must be excluded from Mūla
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
    # DBB is roughly equal thirds aluminium, zinc and copper, so it would let
    # the regression explain either metal with a basket containing it — the
    # same reason it is excluded for copper.
    "Aluminium": ["Base Metals (DBB)"],
    "Zinc": ["Base Metals (DBB)"],
    # EVERY INR-leg cross is a replica of USD/INR: INR/USD is its exact reciprocal,
    # and X/INR = X/USD × USD/INR all carry the target's own currency leg. Excluding
    # the whole set (computed so future additions are covered automatically) keeps
    # the fair-value residual honest. Dollar Index is kept — a driver, not a replica.
    "USD/INR": [n for n in MACRO_SYMBOLS_YF
                if (n.endswith("/INR") or n == "INR/USD") and n != "USD/INR"],
    # The Dollar Index IS a fixed basket of six crosses, so most of the FX
    # complex reconstructs it arithmetically rather than explaining it:
    # EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%. Left in,
    # a handful of predictors rebuild the target to ~96% by weight and the
    # "mispricing" that remains is rounding — an OOS R2 near 1.0 that means
    # nothing. Excluded here: the index column itself, the three USD-index
    # ETFs that track it (UUP long, UDN its exact inverse, USDU a broad USD
    # basket), and the component legs. Non-component FX stays — AUD, the INR
    # legs and the EM crosses are drivers of the dollar, not pieces of it.
    "Dollar Index": [
        "Dollar Index",
        "US Dollar Bullish (UUP)", "US Dollar Bearish (UDN)",
        "USD Bullish Broad (USDU)",
        "Euro (FXE)", "EUR/USD",
        "Japanese Yen (FXY)", "USD/JPY", "JPY/USD",
        "British Pound (FXB)", "GBP/USD",
        "Canadian Dollar (FXC)",
        "Swiss Franc (FXF)", "USD/CHF",
        "USD/SEK",
    ],
    # WTI is ~the same barrel as Brent; the broad commodity indices + energy
    # sector ETF are crude-dominated → all would let crude "explain" itself.
    # RBOB/heating oil are refined FROM that barrel (crude + crack margin) —
    # same-barrel logic.
    "Brent Crude": ["Crude Oil", "Broad Commodity Index (DBC)",
              "Commodity Index (GSG)", "US Energy Sector",
              "RBOB Gasoline", "Heating Oil"]}

# ─── Index targets (equity indices: India broad/sectoral, US benchmarks) ──────
# The Mūla target is the index price level, and Swayam reads breadth off that
# same series (its own OHLCV) — there is no constituent basket. The index
# tickers are merged into the fetched universe so each level is a column.
from data.universe import INDEX_TARGETS, INDEX_TARGETS_MAP  # noqa: E402

# Equity-index ETFs already in the macro pool that would replicate an index
# target (so they are excluded from that target's predictor set). IIND.L is the
# London-listed twin of INDA (see GLOBAL_MACRO_MAP), so it goes wherever INDA does.
_US_INDEX_ETFS = ["US Large Cap (S&P 500)", "US Nasdaq 100",
                  "US Small Cap (Russell 2000)", "Global Equity (ACWI)"]
_INDIA_INDEX_ETFS = ["India Equity", "India Gov Bonds (LSE Proxy)"]

# The US equity complex: every US-listed equity ETF in the panel — broad,
# sector, style and thematic. A US INDEX is a weighted sum of exactly these
# pieces, so leaving them in rebuilds the target instead of valuing it:
# measured on 9y of daily returns, VTV + VUG alone explain 99.5% of the S&P
# 500 (XLK + XLC + XLY: 97.8% of the Nasdaq 100). That is the Dollar Index
# pathology (see TARGET_EXCLUDED_PREDICTORS above) — an OOS R² near 1.0 and a
# "mispricing" that is rounding. Excluded, the index is valued against rates,
# credit, the dollar, commodities, volatility and foreign equity, which is the
# question Mūla asks. Individual US STOCKS keep these as predictors: a sector
# ETF is a driver of one of its members, not a replica of it.
_US_EQUITY_COMPLEX = _US_INDEX_ETFS + [
    "US Materials Sector", "US Energy Sector", "US Industrials Sector",
    "US Financials Sector", "US Regional Banks (Credit Stress)",
    "US Utilities (XLU)", "US Consumer Staples (XLP)",
    "US Consumer Discretionary (XLY)", "US Technology (XLK)",
    "US Health Care (XLV)", "US Real Estate Sector (XLRE)",
    "US Communication Services (XLC)", "US Homebuilders (XHB)",
    "US Transports (IYT)", "Semiconductors (SMH)", "Metals & Mining (XME)",
    "US Value (VTV)", "US Growth (VUG)", "US Momentum (MTUM)",
    "US Low Volatility (USMV)", "US High Beta (SPHB)", "US High Dividend (VYM)",
    "US REITs (VNQ)",
]

_INDEX_NAMES = list(INDEX_TARGETS.keys())
for _name, _meta in INDEX_TARGETS.items():
    # An index must not be "explained" by sibling equity indices → exclude every
    # other index column, plus the same-market broad ETFs, from its predictors.
    _excl = [n for n in _INDEX_NAMES if n != _name]
    if _meta["kind"] == "india":
        _excl = _excl + _INDIA_INDEX_ETFS
    elif _meta["kind"] == "us":
        _excl = _excl + _US_EQUITY_COMPLEX
    TARGET_EXCLUDED_PREDICTORS[_name] = _excl

# Full target catalogue (commodities/FX + indices) → friendly name : yf ticker.
ALL_TARGETS = {**COMMODITY_TARGETS, **INDEX_TARGETS_MAP}

# Sidebar grouping — ordered category → target names.
TARGET_CATEGORIES: dict[str, list[str]] = {
    "Commodities": ["Gold", "Silver", "Copper", "Aluminium", "Zinc",
                    "Brent Crude", "Cotton", "Jeera"],
    "Currency (FX)": ["USD/INR", "Dollar Index"],
    "Crypto": ["Bitcoin", "Ethereum", "Solana", "XRP", "BNB", "Cardano",
               "Dogecoin"]}
for _name, _meta in INDEX_TARGETS.items():
    TARGET_CATEGORIES.setdefault(_meta["category"], []).append(_name)

# ─── Sheet-sourced targets (non-yfinance; joined as the target column only) ───
# Daily series pulled from a published Google Sheet (data/sheets.py SHEET_SOURCES,
# keyed by the same name) and joined to a run's private copy of the panel when
# selected, exactly like Jeera — never injected into the shared matrix, so a
# Sheets outage cannot change any other target's panel. Registered here so they
# appear in the sidebar under their chosen category, with a sentinel ticker kept
# OUT of the yfinance maps. (Jeera predates this registry and stays in
# COMMODITY_TARGETS.)
SHEET_TARGETS: dict[str, dict] = {
    "Nifty 50 - PE": {"ticker": "NIFTY50_PE.SHEET", "category": "India Indices"}}
for _sname, _smeta in SHEET_TARGETS.items():
    ALL_TARGETS[_sname] = _smeta["ticker"]
    TARGET_CATEGORIES.setdefault(_smeta["category"], []).append(_sname)
    # Don't let sibling India equity indices / the India ETF "explain" the PE.
    TARGET_EXCLUDED_PREDICTORS.setdefault(_sname, list(_INDEX_NAMES) + _INDIA_INDEX_ETFS)

#: The catalogue, frozen at import: every selectable catalogue target. The
#: yfinance-sourced ones are columns of the shared model matrix; the sheet
#: series are joined per run as the target only. ALL_TARGETS itself grows at
#: runtime (register_stock_target adds
#: each free-form symbol a user types), and anything that iterates it to decide
#: what to FETCH would then pull one user's symbol into every later user's
#: panel — where it became a predictor of Gold, moved every published fair
#: value, and changed the panel fingerprint. What the shared matrix contains is
#: decided here, once, and never by a session.
CATALOGUE_TARGETS: dict[str, str] = dict(ALL_TARGETS)

# ─── Stock targets (individual equities) ─────────────────────────────────────
# The Mūla target is the stock's own price; breadth is Swayam on that stock's
# own OHLCV (engines/swayam/). Individual stocks are entered as free-form
# symbols (India/US Stocks asset classes), resolved live via
# resolve_stock_symbol (NSE .NS then BSE .BO for India; bare for US) and
# registered at runtime by register_stock_target().

# category label → market key. The Asset Class selector renders these as a
# free-form "Symbol" text input (app.py) instead of the usual Target
# drop-down — they must still exist in TARGET_CATEGORIES (even with zero
# static members) so the Asset Class selector lists them at all.
FREEFORM_STOCK_CATEGORIES: dict[str, str] = {
    "India Stocks": "india",
    "US Stocks": "us"}
for _cat in FREEFORM_STOCK_CATEGORIES:
    TARGET_CATEGORIES.setdefault(_cat, [])


#: Free-form stock targets registered at runtime, display name -> market.
#: This is a FACT about where a target's price comes from, not a routing label:
#: these symbols are deliberately absent from the macro batch fetch (a
#: per-target ticker set would break that batch's (start, end)-keyed cache), so
#: their price column is injected separately. It replaced TARGET_ARCHETYPE,
#: which encoded the same fact but bundled it with the breadth-mode routing
#: that no longer exists.
STOCK_TARGET_MARKETS: dict[str, str] = {}


def is_stock_target(name: str) -> bool:
    """Is this target a runtime-registered free-form stock symbol?"""
    return name in STOCK_TARGET_MARKETS


def register_stock_target(display_name: str, ticker: str, market: str) -> None:
    """Register an individual-stock target at runtime.

    Idempotent — safe to call on every Streamlit rerun (module-level config
    dicts survive reruns within a process but the registration must be
    replayed from st.session_state on each one; see app.py). Wires
    ALL_TARGETS and the market-based Mūla predictor exclusions (own-market
    index targets + broad ETFs — the same guard that feeds the Swayam MMR
    leakage filter via TARGET_EXCLUDED_PREDICTORS, see swayam_macro_columns
    below). Also installs the instrument's InstrumentConfig (the market's
    STOCK_CONFIGS asset-class config) and its core.instruments entry. Does NOT append to
    TARGET_CATEGORIES — freeform categories render a text input, not a list.
    """
    ALL_TARGETS[display_name] = ticker
    STOCK_TARGET_MARKETS[display_name] = market
    excl = list(_INDEX_NAMES)
    excl += _INDIA_INDEX_ETFS if market == "india" else _US_INDEX_ETFS
    TARGET_EXCLUDED_PREDICTORS.setdefault(display_name, excl)
    # The instrument's config is its market's asset-class stock config.
    INSTRUMENT_CONFIGS.setdefault(display_name,
                                  STOCK_CONFIGS.get(market, CLASS_CONFIG_DEFAULTS["stock_us"]))
    # Local import: core.instruments is built FROM this module.
    from core.instruments import register_stock_instrument
    register_stock_instrument(display_name, ticker, market)

# ─── Swayam (self-referential ensemble) ───────────────────────────────
# Timescale axis (log-spaced) + the ROC fraction that derives each member's
# roc_len — see engines/swayam/ensemble.py
# (default_swayam_members) for how these build the 12-member grid.
SWAYAM_LENGTHS = (8, 14, 22, 34, 52)   # swayam 2026-07-21 class-level best (|IC| 0.096 vs default-5)
SWAYAM_ROC_FRAC = 0.85                  # swayam 2026-07-21 class-level best (|IC| 0.094 vs 0.7)

# (The empty-basket fallback flag lived here. With no baskets, there is no
# empty-basket case to fall back FROM.)


def swayam_macro_columns(target: str, macro_cols: list[str]) -> list[str]:
    """Macro candidates for Swayam's MMR: drop the target's own column and
    its TARGET_EXCLUDED_PREDICTORS near-replicas.

    Every Swayam view reads the target's own price, so a driver that IS the
    target is fatal: the view's Close correlates ~1.0 with the target's own
    macro column, MMR's top-N driver selection locks onto it, predicted ≈
    actual, deviation ≈ 0, and the MMR half of every macro-anchored view dies
    silently while mmr_quality reads perfect. This reuses the same
    self-explanation guard TARGET_EXCLUDED_PREDICTORS already applies to
    Mūla, applied here to the MMR driver pool instead.
    """
    drop = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    return [c for c in macro_cols if c not in drop]


# ═══════════════════════════════════════════════════════════════════════════
# Per-instrument configuration registry
# ═══════════════════════════════════════════════════════════════════════════
# Every named target has its OWN config entry — structure (view bank, horizon)
# and estimability floors. app.py reads get_instrument_config(target). EVERY
# catalogue target has an explicit INSTRUMENT_CONFIGS entry (no silent fallback —
# get_instrument_config raises for an unregistered target); free-form stocks are
# configured per ASSET CLASS (STOCK_CONFIGS), registered at resolution time. Field
# defaults equal the global constants above. Every field has a reader on the
# claim, sizer or grade path (tests/test_consumers.py).
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class InstrumentConfig:
    """Per-instrument configuration.

    Read the two sections below as two different KINDS of thing, because
    they are:

    **Structure** — what question to ask. The horizon (what you intend to
    hold), the Swayam view bank and the Mūla discount grid (the hypothesis
    space to average over). These are declared because no amount of data tells
    you what you are trying to do.

    **Estimability floors** — how much evidence before publishing anything.
    The Mūla burn-in and print floor. Argued from first principles, not swept.

    What is NO LONGER here: routing (``archetype`` / ``polarity`` / ``basket``
    / ``basket_alias``) died with the basket breadth engine, and the Swayam
    basket knobs (``swayam_msf_length`` / ``swayam_roc_len``) with it. The
    leakage exclusions are read from ``core.instruments`` (built from
    ``TARGET_EXCLUDED_PREDICTORS``); the copy this config carried had no
    reader. The display warm-up priors (``conviction_*``, ``ui_*``) went with
    the tiers they seeded, and the second horizon list (``hold_horizons``)
    with the calibration frame that read it: one h, ``forecast_horizon``.
    """

    # ── Structure: Swayam breadth ───────────────────────────────────────────
    # `swayam_lengths` is a SPAN of timescales, not a length to pick — every
    # member is counted once (engines.swayam.kernel.aggregate_views).
    swayam_lengths: tuple[int, ...] = SWAYAM_LENGTHS
    swayam_roc_frac: float = SWAYAM_ROC_FRAC
    swayam_regime_sensitivity: float = SWAYAM_REGIME_SENSITIVITY
    swayam_base_weight: float = SWAYAM_BASE_WEIGHT
    swayam_mmr_num_vars: int = SWAYAM_MMR_NUM_VARS
    swayam_oversold: float = SWAYAM_OVERSOLD
    swayam_overbought: float = SWAYAM_OVERBOUGHT

    # ── Structure: the horizon ──────────────────────────────────────────────
    # THE horizon h of the estimand: resolved once per run from this field and
    # passed explicitly everywhere (no default, no fallback downstream).
    forecast_horizon: int = FORECAST_HORIZON

    # ── Estimability floors: Mūla valuation engine ──────────────────────────
    # Default to the global constants; an asset-class config (CLASS_CONFIG_
    # DEFAULTS / STOCK_CONFIGS) can retune them. Note what is NOT here: the
    # engine is recursive, so it has no training window, refit cadence,
    # ensemble roster or regularisation path to tune. What remains are
    # estimability floors (how much history before publishing, how many
    # prints before an instrument may join the cross-section) and the
    # coefficient-memory grid.
    fvo_burn_in: int = FVO_BURN_IN
    fvo_min_prints: int = FVO_MIN_PRINTS
    fvo_valuation_deltas: tuple[float, ...] = FVO_VALUATION_DELTAS

    # (The Convergence layer's knobs — its drift-diffusion filter, its prior
    # dimension weights and its agreement tiers — and the Precedent matcher's
    # term structure and state window left with the parallel opinions they
    # tuned: none moved the call.)

    def __post_init__(self) -> None:
        """The horizon is THE estimand's h, so a config that cannot carry it
        fails at construction — the class and stock configs included — rather
        than somewhere downstream that would substitute another."""
        h = self.forecast_horizon
        assert isinstance(h, int) and not isinstance(h, bool), (
            f"forecast_horizon must be an int, got {h!r}")
        assert h >= 1, f"forecast_horizon must be >= 1, got {h}"



# Per-asset-class DEFAULT tuning. Each class is a NAMED constant so an entire class
# can be retuned in one place (e.g. give all commodities a different Swayam grid)
# without editing every member. The India-index default IS the Nifty 50 baseline —
# the other India indices copy it (per spec). The only non-default values below are
# the `per_asset` 2026-07-21 STOCK Swayam grids; commodity / fx / crypto / index
# classes inherit the global defaults. The stock grids are PINNED to per_asset's
# stock recommendation so they do NOT drift with the global SWAYAM_* globals (which
# the `swayam` study tunes on commodities).
CLASS_CONFIG_DEFAULTS: dict[str, InstrumentConfig] = {
    "commodity":   InstrumentConfig(),
    "fx":          InstrumentConfig(),
    "india_index": InstrumentConfig(),   # == Nifty 50 baseline tuning
    "us_index":    InstrumentConfig(),
    # Crypto starts on the global default: it has not been through a tuning
    # sweep, and a hand-picked knob would be a guess wearing a number. A swept
    # value belongs in the per-instrument registry once one exists.
    "crypto":      InstrumentConfig(),
    # per_asset 2026-07-21 (asset-level, pooled Nifty100 / Nasdaq100 universes):
    "stock_india": InstrumentConfig(swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.7),
    "stock_us":    InstrumentConfig(swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.55)}

# Free-form stock ASSET-CLASS configs — one per market, applied to any symbol
# entered under India Stocks / US Stocks (register_stock_target installs the
# right one per resolved symbol; its market-based exclusions go to
# TARGET_EXCLUDED_PREDICTORS).
STOCK_CONFIGS: dict[str, InstrumentConfig] = {
    "india": CLASS_CONFIG_DEFAULTS["stock_india"],
    "us":    CLASS_CONFIG_DEFAULTS["stock_us"]}

_CATEGORY_TO_CLASS: dict[str, str] = {
    "Commodities":   "commodity",
    "Currency (FX)": "fx",
    "Crypto":        "crypto",
    "India Indices": "india_index",
    "US Indices":    "us_index"}

# One explicit InstrumentConfig per named target: its class's config. (Each
# catalogue target used to carry a per-instrument override slot on top of its
# class default. The last overrides — a display prior on Gold and Jeera — left
# with the Swayam average signal they coloured, and the empty slots went with
# them: a retune now lands in CLASS_CONFIG_DEFAULTS or as a new entry here.)
INSTRUMENT_CONFIGS: dict[str, InstrumentConfig] = {
    _nm: CLASS_CONFIG_DEFAULTS[_cls]
    for _cat, _cls in _CATEGORY_TO_CLASS.items()
    for _nm in TARGET_CATEGORIES.get(_cat, [])
}

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

# ── Chart palette — SINGLE SOURCE OF TRUTH ───────────────────────────────────
# Every chart colour derives from these RGB triples (COLOR_* + inline rgba() via
# the rgba() helper), and each one now EQUALS its CSS token in ui/theme.css.
#
# They used to differ: charts ran a brighter Tailwind-400 family while the
# chrome ran -500, sharing only the amber. The consequence was subtle and
# constant — a green line in a plot was not the same green as the value in the
# card beside it, so the eye read them as two different kinds of "positive"
# when they were the same claim about the same number. A design system that
# stops at the edge of the chart is not a design system.
#
# Every hue below clears WCAG AA (>= 4.5:1) on the chart surface, which matters
# more here than in most apps: these are thin 1.5px lines, not filled shapes.
_PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    "emerald": (44, 163, 107),   # #2CA36B - Long / Bullish
    "rose":    (221, 90, 90),    # #DD5A5A - Short / Bearish
    "accent":  (76, 125, 240),   # #4C7DF0 - Primary / Interactive (brand, active nav, primary CTA)
    "cyan":    (78, 159, 196),   # #4E9FC4 - Info (distinct from Accent — informational tone only)
    "amber":   (215, 154, 60),   # #D79A3C - Caution / Warning (warning semantics ONLY, not brand)
    "violet":  (155, 143, 212),  # #9B8FD4 - Secondary / Attribution
    "slate":   (126, 135, 151),  # #7E8797 - Neutral / Muted
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


COLOR_GREEN = _palette_hex("emerald")   # #2CA36B
COLOR_RED = _palette_hex("rose")        # #DD5A5A
COLOR_CYAN = _palette_hex("cyan")       # #4E9FC4
COLOR_AMBER = _palette_hex("amber")     # #D79A3C
COLOR_ACCENT = _palette_hex("accent")   # #4C7DF0 — primary/interactive; use for featured
                                         # series & emphasis, never for warning semantics.
COLOR_PURPLE = _palette_hex("violet")   # #9B8FD4

# ─── UI Layout ───────────────────────────────────────────────────────────────
# (The display tiers that used to live here and on InstrumentConfig — R² and
# model-spread colour cuts, window-agreement and Swayam-signal tiers — went with
# the advisory colour they set: evidence is neutral-toned and carries its own
# grade instead.)

# Chart height defaults
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540
