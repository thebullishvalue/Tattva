"""
Tattva — configuration: the macro universe, the target catalogue, structural
constants and warm-up priors.

`InstrumentConfig` holds every per-target setting. Catalogue targets take their
asset class's default (`CLASS_CONFIG_DEFAULTS`) plus the few values in
`INSTRUMENT_OVERRIDES`; free-form stocks take their market's (`STOCK_CONFIGS`).
`get_instrument_config(target)` is the single read path.
"""

# ─── Version / Product ───────────────────────────────────────────────────────

# Single source of truth for the app version — ui/theme.py imports these (do not
# redefine elsewhere; past drift between config and theme is why this is centralized).
VERSION = "3.1.7"
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

# Window lengths for the Z_lb / AvgZ / window-breadth STATE features — each a
# robust z-score of the mispricing gap against its own trailing window.
#
# ~1 month, ~1 quarter, ~1 year. These used to be (3, 5, 10) — an "ultra-short"
# grid that made every reading built on it (the engine's zone labels, its
# window breadth, ConvictionRaw/Bounded) a measure of how far the gap had moved
# over the last one-to-two weeks: corr 0.85 with the 5-day CHANGE in the gap,
# 0.3 with its level. That is a timing read, while every page labels Mūla as
# "is this cheap or dear?" — so the engine pointed two ways at once (Gold on
# 2026-10-16: zone stack "OVERSOLD / BUY", oscillator +0.01σ, fair). At these
# memories the gap (half-life ~1-2 weeks) completes several cycles inside every
# window, so each Z_lb reads the LEVEL of the mispricing against its own
# history and the stack describes the same thing the oscillator does. The 10d
# IC of neither grid was distinguishable from zero on the evidence, so
# coherence, not a measured edge, decided it. (The 3-day window was also never
# populated: its min_periods of 5 could not fit in 3 rows.)
LOOKBACK_WINDOWS = (21, 63, 252)

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

#: Reciprocals and re-denominations of an instrument already in the panel are
#: not listed (INR/USD beside USD/INR, the X/INR crosses, UDN beside UUP): a
#: duplicated instrument is counted twice by the factor model.
MIN_DATA_POINTS = 1500

#: First date of every fetch. Fixed, so the estimators' origin never moves
#: from one day's run to the next.
EPOCH = "2015-01-01"


def fetch_window(today=None):
    """``(EPOCH, today)`` — the one window every fetch uses."""
    import pandas as _pd
    return _pd.Timestamp(EPOCH), (_pd.Timestamp.today() if today is None else _pd.Timestamp(today))



# The one horizon every grade is scored at and the UI projects over (sessions).
# The engine is fitted to no label, so this is a scoring choice only.
FORECAST_HORIZON = 10       # scoring / display horizon (trading days)
#: A call dated t is acted on at the close of t+1. The panel row for t holds
#: closes from markets that shut after the target did (New York after Mumbai),
#: so the target's own close on t is not a price the call could have traded.
EXECUTION_LAG = 1

# Trailing window for the PRECEDENT analog matcher's state features (momentum,
# realized vol, and — at 3x this — the rolling Hurst) in analytics.analogs.
# Validated by the `precedent_univ` study, which sweeps it against the horizon.
# This was formerly FORECAST_MOMENTUM and did double duty as the removed
# forecast engine's predictor-momentum window; the two uses were never the same
# quantity, they merely shared a value, so it is named for the one that remains.
ANALOG_MOM_WINDOW = 20

# Predictors that are RAW YIELD LEVELS (e.g. ^TNX at 4.25), not prices. The Mūla
# cross-section is a panel of PRICES: it takes logs and first differences, and
# a rate series can print ≤0 (zero-rate era), where log(≤0) is NaN. They are
# excluded from the valuation panel rather than transformed — a yield level is
# not an instrument whose price the target can be valued against.
RAW_YIELD_PREDICTORS = frozenset({
    "US 13-Week T-Bill Yield", "US 5-Year Treasury Yield",
    "US 10-Year Treasury Yield", "US 30-Year Treasury Yield"})

# Precedent-tab analog term-structure horizons (trading days), fixed. 1d is a
# normal member (weak/noisy edge, disclosed by the
# Analog-Skill chart's per-horizon IC); 60d is the long/regime end (edge fades past
# ~20d). The hero's precedent second-opinion reads at FORECAST_HORIZON (10d).
PRECEDENT_HORIZONS = (1, 3, 5, 10, 20, 60)

# Warm-up priors for the tiers of Mūla's WINDOW AGREEMENT (ConvictionBounded,
# ±100) — the engine's zone labels and the Mūla page's "broadly / leans" read.
# p90 / p75 of |ConvictionBounded| on the 21/63/252-session windows, measured
# 2026-10-17 on Gold / S&P 500 / USD-INR (70.2/57.3, 74.1/59.6, 69.8/54.4).
# Each instrument's own causal quantiles replace them after a year.
CONVICTION_STRONG = 70.0
CONVICTION_MODERATE = 57.0

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

# ─── Swayam Breadth Defaults ─────────────────────────────────────────────────
# The per-series kernel's structural knobs. These are shared by every member of
# the Swayam view bank; the member's own timescale comes from swayam_lengths.
SWAYAM_REGIME_SENSITIVITY = 8.0  # swayam 2026-07-21 (|IC| 0.073 @8.0)
SWAYAM_BASE_WEIGHT = 0.0         # MSF share of the FIXED half of the MSF/MMR
                                 # blend (kernel: 0.5*bw + 0.5*adaptive).
SWAYAM_MMR_NUM_VARS = 4          # swayam 2026-07-21 class-level best

# Condition thresholds on Unified_Osc (±10 scale): classify Oversold/Overbought/
# Neutral and gate buy/sell + divergence. ±5 = p75-p85 occupancy (ui_anchors).
SWAYAM_OVERSOLD = -5
SWAYAM_OVERBOUGHT = 5

DIV_LOOKBACK = 20
DIV_PERSISTENCE_THRESHOLD = 5

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
    "USD/BDT": "BDT=X",
    "USD/CNY": "CNY=X",
    "USD/CNH": "CNH=X",
    "USD/JPY": "JPY=X",
    "USD/KRW": "KRW=X",
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

# ─── Commodity Targets & Baskets ─────────────────────────────────────────────
# User-selectable Mūla targets. Each maps to a yfinance front-month future
# (already present in MACRO_SYMBOLS_YF). The Mūla predictor pool is the rest
# of MACRO_SYMBOLS_YF (commodities + FX) with the selected target excluded.

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
    # from a published Google Sheet (data/sheets.py) and injected as a column in
    # the Mūla matrix. The value here is a non-yfinance sentinel ticker: it
    # documents the source and is deliberately kept OUT of MACRO_SYMBOLS_YF /
    # GLOBAL_MACRO_MAP so it is never sent to yf.download.
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
    # The Dollar Index IS a fixed basket of six crosses, so most of the FX
    # complex reconstructs it arithmetically rather than explaining it:
    # EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%. Left in,
    # a handful of predictors rebuild the target to ~96% by weight and the
    # "mispricing" that remains is rounding — an OOS R2 near 1.0 that means
    # nothing. Excluded here: the index column itself, the three USD-index
    # ETFs that track it (UUP, and USDU, a broad USD
    # basket), and the component legs. Non-component FX stays — AUD, the INR
    # legs and the EM crosses are drivers of the dollar, not pieces of it.
    "Dollar Index": [
        "Dollar Index",
        "US Dollar Bullish (UUP)",
        "USD Bullish Broad (USDU)",
        "Euro (FXE)", "EUR/USD",
        "Japanese Yen (FXY)", "USD/JPY",
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

# ─── Index targets (equity indices: India sectoral/broad, US, sector-ETF) ─────
# The Mūla target is the index price; the Swayam basket is the index's own
# constituents (resolved live + cached in data/universe.py). Their price tickers
# are merged into the fetched universe so the index level is an available column.
from data.universe import INDEX_TARGETS, INDEX_TARGETS_MAP  # noqa: E402

# Equity-index ETFs already in the macro pool that would replicate an index
# target (so they are excluded from that target's predictor set).
_US_INDEX_ETFS = ["US Large Cap (S&P 500)", "US Nasdaq 100",
                  "US Small Cap (Russell 2000)", "Global Equity (ACWI)"]
_INDIA_INDEX_ETFS = ["India Equity"]

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
    "Commodities": ["Gold", "Silver", "Copper", "Aluminium",
                    "Brent Crude", "Cotton", "Jeera"],
    "Currency (FX)": ["USD/INR", "Dollar Index"],
    "Crypto": ["Bitcoin", "Ethereum", "Solana", "XRP", "BNB", "Cardano",
               "Dogecoin"]}
for _name, _meta in INDEX_TARGETS.items():
    TARGET_CATEGORIES.setdefault(_meta["category"], []).append(_name)

# ─── Sheet-sourced targets (non-yfinance; injected via fetcher exogenous path) ─
# Daily series pulled from a published Google Sheet (data/sheets.py SHEET_SOURCES,
# keyed by the same name) and injected into the model matrix exactly like Jeera.
# Registered here so they appear in the sidebar under their chosen category, with a
# sentinel ticker kept OUT of the yfinance maps.
# (Jeera predates this registry and stays in COMMODITY_TARGETS.)
SHEET_TARGETS: dict[str, dict] = {
    "Nifty 50 - PE": {"ticker": "NIFTY50_PE.SHEET", "category": "India Indices"}}
for _sname, _smeta in SHEET_TARGETS.items():
    ALL_TARGETS[_sname] = _smeta["ticker"]
    TARGET_CATEGORIES.setdefault(_smeta["category"], []).append(_sname)
    # Don't let sibling India equity indices / the India ETF "explain" the PE.
    TARGET_EXCLUDED_PREDICTORS.setdefault(_sname, list(_INDEX_NAMES) + _INDIA_INDEX_ETFS)

#: The catalogue, frozen at import: every target that is a column of the shared
#: model matrix. ALL_TARGETS itself grows at runtime (register_stock_target adds
#: each free-form symbol a user types), and anything that iterates it to decide
#: what to FETCH would then pull one user's symbol into every later user's
#: panel — where it became a predictor of Gold, moved every published fair
#: value, and changed the panel fingerprint. What the shared matrix contains is
#: decided here, once, and never by a session.
CATALOGUE_TARGETS: dict[str, str] = dict(ALL_TARGETS)

# ─── Stock targets (individual equities) ─────────────────────────────────────
# The Mūla target is the stock's own price; breadth is Swayam on that price —
# Swayam formulates breadth on the stock's own OHLCV
# (engines/swayam/) instead.
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
    replayed from st.session_state on each one; see app.py). Applies the
    same wiring the old static STOCK_TARGETS loop used: ALL_TARGETS,
    and the market-based Mūla predictor
    exclusions (own-market index targets + broad ETFs — the same guard that
    feeds the Swayam MMR leakage filter via TARGET_EXCLUDED_PREDICTORS,
    see swayam_macro_columns above). Also installs the instrument's own
    InstrumentConfig, cloned from the market's STOCK_CONFIGS asset-class config
    with its market-based exclusions. Does NOT append to TARGET_CATEGORIES —
    freeform categories render a text input, not a list.
    """
    ALL_TARGETS[display_name] = ticker
    STOCK_TARGET_MARKETS[display_name] = market
    excl = list(_INDEX_NAMES)
    excl += _INDIA_INDEX_ETFS if market == "india" else _US_INDEX_ETFS
    TARGET_EXCLUDED_PREDICTORS.setdefault(display_name, excl)
    # Per-instrument config from the asset-class stock config (India / US).
    INSTRUMENT_CONFIGS.setdefault(display_name, _dc_replace(
        STOCK_CONFIGS.get(market, CLASS_CONFIG_DEFAULTS["stock_us"]),
        excluded_predictors=tuple(excl),
    ))

# ─── Swayam (self-referential ensemble) ───────────────────────────────
# Timescale axis (log-spaced) + the ROC fraction that derives each member's
# roc_len — see engines/swayam/ensemble.py
# (default_swayam_members) for how these build the 15-member grid.
SWAYAM_LENGTHS = (8, 14, 22, 34, 52)   # swayam 2026-07-21 class-level best (|IC| 0.096 vs default-5)
SWAYAM_ROC_FRAC = 0.85                  # swayam 2026-07-21 class-level best (|IC| 0.094 vs 0.7)

# (The empty-basket fallback flag lived here. With no baskets, there is no
# empty-basket case to fall back FROM.)


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
    Mūla, applied here to the MMR driver pool instead.
    """
    drop = {target, *TARGET_EXCLUDED_PREDICTORS.get(target, [])}
    return [c for c in macro_cols if c not in drop]


# ═══════════════════════════════════════════════════════════════════════════
# Per-instrument configuration registry
# ═══════════════════════════════════════════════════════════════════════════
# Every named target has its OWN full config — structure (leakage exclusions,
# view bank, horizons), estimability floors, and warm-up priors. app.py reads get_instrument_config(target),
# so any instrument retunes in isolation. EVERY catalogue target has an explicit
# INSTRUMENT_CONFIGS entry (no silent fallback — get_instrument_config raises for an
# unregistered target); free-form stocks are configured per ASSET CLASS
# (STOCK_CONFIGS), registered at resolution time. Field defaults equal the former
# global constants, so the registry is behaviour-preserving until a config diverges.
from dataclasses import dataclass, replace as _dc_replace  # noqa: E402


@dataclass(frozen=True)
class InstrumentConfig:
    """Per-instrument configuration.

    Read the three sections below as three different KINDS of thing, because
    they are:

    **Structure** — what question to ask. Horizons (what you intend to hold),
    the Swayam view bank and the Mūla discount grid (the hypothesis space to
    average over), the leakage exclusions (which predictors would let a target
    explain itself). These are declared because no amount of data tells you
    what you are trying to do.

    **Estimability floors** — how much evidence before publishing anything.
    The Mūla burn-in and print floor. Argued from first principles, not swept.

    **Warm-up priors** — the ``*_strong`` / ``*_moderate`` / ``ui_*`` numbers.
    These used to BE the thresholds, anchored by full-history studies. They are
    now only the value used until an instrument has accumulated enough of its
    own history for :mod:`analytics.adaptive` to estimate the cut-point from
    the causal empirical distribution of the signal itself. After that the
    prior is superseded. Keeping them means an instrument's first year behaves
    exactly as it always did rather than flapping on a quantile of forty
    points; it does not mean they are still tuned.

    What is NO LONGER here: routing (``archetype`` / ``polarity`` / ``basket``
    / ``basket_alias``) died with the basket breadth engine, and the Swayam
    basket knobs (``swayam_msf_length`` / ``swayam_roc_len``) with it. The
    Swayam kernel knobs kept their values but are named for the engine that
    now owns them.
    """

    # ── Structure: leakage guard ────────────────────────────────────────────
    excluded_predictors: tuple[str, ...] = ()      # Mūla + Swayam-MMR leakage guard

    # ── Structure: Swayam breadth ───────────────────────────────────────────
    # `swayam_lengths` is a SPAN to weight, not a length to pick — members are
    # weighted by their own online skill (analytics.adaptive.OnlineSkillWeights).
    swayam_lengths: tuple[int, ...] = SWAYAM_LENGTHS
    swayam_roc_frac: float = SWAYAM_ROC_FRAC
    swayam_regime_sensitivity: float = SWAYAM_REGIME_SENSITIVITY
    swayam_base_weight: float = SWAYAM_BASE_WEIGHT
    swayam_mmr_num_vars: int = SWAYAM_MMR_NUM_VARS
    swayam_oversold: float = SWAYAM_OVERSOLD
    swayam_overbought: float = SWAYAM_OVERBOUGHT

    # ── Scoring / display horizons ──────────────────────────────────────────
    forecast_horizon: int = FORECAST_HORIZON
    analog_mom_window: int = ANALOG_MOM_WINDOW   # precedent state-feature window

    # ── Estimability floors: Mūla valuation engine ──────────────────────────
    # The engine is recursive, so it has no training window, refit cadence or
    # regularisation path. What remains are estimability floors (history before
    # publishing, prints before an instrument may join) and the memory grid.
    fvo_burn_in: int = FVO_BURN_IN
    fvo_min_prints: int = FVO_MIN_PRINTS
    fvo_valuation_deltas: tuple[float, ...] = FVO_VALUATION_DELTAS
    fvo_lookback_windows: tuple[int, ...] = LOOKBACK_WINDOWS

    # ── Precedent analog term structure ─────────────────────────────────────
    precedent_horizons: tuple[int, ...] = PRECEDENT_HORIZONS

    # ── Warm-up priors for the adaptive thresholds ─────────────────────────
    # Each is the p90 ("strong") / p75 ("moderate") / p50 ("weak") of its own
    # signal's distribution, measured once by a pooled study. analytics.adaptive
    # now re-derives exactly those quantiles from the signal's OWN causal past,
    # per instrument; these values hold only until it has enough history.
    # Structural cut-points (R²/ADF/KPSS/HMM) stay fixed — they are properties
    # of a statistical test, not of an instrument's distribution.
    # Mūla window agreement (ConvictionBounded, ±100) — p90 / p75.
    conviction_strong: float = CONVICTION_STRONG
    conviction_moderate: float = CONVICTION_MODERATE
    # Swayam average signal (±10) — p75, the one cut-point both the Swayam and
    # the Convergence page colour it with.
    ui_swayam_avg_threshold: float = 2.87
    # Model spread (SD of the fair-value level), basis points — p10 / p90. The
    # spread now measures the fair value's own uncertainty, not the residual
    # noise around it, and runs ~17-77 bps across instruments (2026-10-17), so
    # the prior is wide and each instrument's own history takes over.
    ui_model_spread_low: float = 20.0
    ui_model_spread_high: float = 80.0
    # (The consensus and composite tiers are no longer per-instrument fields:
    # convergence.normalization classifies both at their own causal p90 / p75,
    # with one shared warm-up prior each. Seven other display fields went with
    # the cards and markers that read them.)


CLASS_CONFIG_DEFAULTS: dict[str, InstrumentConfig] = {
    "commodity":   InstrumentConfig(),
    "fx":          InstrumentConfig(),
    "india_index": InstrumentConfig(),   # == Nifty 50 baseline tuning
    "us_index":    InstrumentConfig(),
    # per_asset 2026-07-21 (asset-level, pooled Nifty100 / Nasdaq100 universes):
    # Crypto starts on the global default: it has not been through a tuning
    # sweep, and a hand-picked knob would be a guess wearing a number. A swept
    # value belongs in the per-instrument registry once one exists.
    "crypto":      InstrumentConfig(),
    # per_asset 2026-07-21 (asset-level, pooled Nifty100 / Nasdaq100 universes):
    "stock_india": InstrumentConfig(swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.7),
    "stock_us":    InstrumentConfig(swayam_lengths=(10, 14, 20, 28, 40), swayam_roc_frac=0.55)}

# Free-form stock ASSET-CLASS configs — one per market, applied to any symbol
# entered under India Stocks / US Stocks (register_stock_target clones the
# right one per resolved symbol, filling in its market-based exclusions).
STOCK_CONFIGS: dict[str, InstrumentConfig] = {
    "india": CLASS_CONFIG_DEFAULTS["stock_india"],
    "us":    CLASS_CONFIG_DEFAULTS["stock_us"]}

_CATEGORY_TO_CLASS: dict[str, str] = {
    "Commodities":   "commodity",
    "Currency (FX)": "fx",
    "Crypto":        "crypto",
    "India Indices": "india_index",
    "US Indices":    "us_index"}

def asset_class(target: str) -> str | None:
    """The asset class a target is graded with. A stock borrows its market's
    index class — one stock's history cannot license a call on its own."""
    if target in STOCK_TARGET_MARKETS:
        return "india_index" if STOCK_TARGET_MARKETS[target] == "india" else "us_index"
    for cat, names in TARGET_CATEGORIES.items():
        if target in names:
            return _CATEGORY_TO_CLASS.get(cat)
    return None


#: The few values that are genuinely one instrument's own. Everything else
#: is its asset class's default.
INSTRUMENT_OVERRIDES: dict[str, dict] = {
    # Display tier for Swayam's average signal: the target's own p75, used
    # until its causal estimate takes over.
    "Gold": {"ui_swayam_avg_threshold": 3.6887},
    "Jeera": {"ui_swayam_avg_threshold": 2.1131},
}

# One explicit InstrumentConfig per catalogue target: its class default, its
# own leakage exclusions and its overrides.
INSTRUMENT_CONFIGS: dict[str, InstrumentConfig] = {}
for _cat, _names in TARGET_CATEGORIES.items():
    _cls = _CATEGORY_TO_CLASS.get(_cat)
    if _cls is None:
        continue   # free-form stock categories — configured per symbol at runtime
    for _nm in _names:
        INSTRUMENT_CONFIGS[_nm] = _dc_replace(
            CLASS_CONFIG_DEFAULTS[_cls],
            excluded_predictors=tuple(TARGET_EXCLUDED_PREDICTORS.get(_nm, ())),
            **INSTRUMENT_OVERRIDES.get(_nm, {}),
        )
assert set(INSTRUMENT_OVERRIDES) <= set(INSTRUMENT_CONFIGS), "override for an unknown target"

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

# ─── UI Thresholds ───────────────────────────────────────────────────────────
# Display tiers that belong to an INSTRUMENT (agreement, Swayam signal, model
# spread, window agreement) live on InstrumentConfig as warm-up priors for that
# instrument's own causal quantiles. What stays here are properties of a
# statistic or of the layout, the same for every instrument.


# (UI_BAND_NARROW/WIDE were removed: the CI band width is pinned by the DDM's
# mean-reverting variance — measured degenerate, the tiers could never fire.
# The band itself is still drawn on the conviction chart.)

# Chart height defaults
UI_CHART_HEIGHT_MEDIUM = 340
UI_CHART_HEIGHT_LARGE = 380
UI_CHART_HEIGHT_XLARGE = 540
UI_CHART_HEIGHT_STACKED = 680
