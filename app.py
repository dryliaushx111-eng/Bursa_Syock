"""
Bursa Pro Analyzer (Watchlist & Favorite Dashboard Edition)
Quantitative Equity, Sector Intelligence, Watchlist Dashboard, Position Sizing & KLSE Workbench
Target Runtime: Python 3.11+
"""

from __future__ import annotations

import math
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: BURSA MALAYSIA CONSTANTS & SECTOR INTELLIGENCE LIBRARY
# ==============================================================================

MYT = ZoneInfo("Asia/Kuala_Lumpur")
BOARD_LOT_SIZE = 100
CURRENCY_SYMBOL = "RM"

KLCI_COMPONENTS: Dict[str, str] = {
    "1155.KL": "Malayan Banking Bhd (Maybank)",
    "1295.KL": "Public Bank Bhd",
    "1023.KL": "CIMB Group Holdings Bhd",
    "5347.KL": "Tenaga Nasional Bhd",
    "5225.KL": "IHH Healthcare Bhd",
    "5183.KL": "PETRONAS Chemicals Group Bhd",
    "6947.KL": "CelcomDigi Bhd",
    "8869.KL": "Press Metal Aluminium Holdings Bhd",
    "5819.KL": "Hong Leong Bank Bhd",
    "1066.KL": "RHB Bank Bhd",
    "5681.KL": "PETRONAS Dagangan Bhd",
    "6033.KL": "PETRONAS Gas Bhd",
    "6012.KL": "Maxis Bhd",
    "4863.KL": "Telekom Malaysia Bhd",
    "3816.KL": "MISC Bhd",
    "2445.KL": "Kuala Lumpur Kepong Bhd",
    "5285.KL": "SD Guthrie Bhd",
    "4197.KL": "Sime Darby Bhd",
    "1961.KL": "IOI Corporation Bhd",
    "4707.KL": "Nestle (Malaysia) Bhd",
    "7084.KL": "QL Resources Bhd",
    "6742.KL": "YTL Power International Bhd",
    "4677.KL": "YTL Corporation Bhd",
    "5296.KL": "MR D.I.Y. Group (M) Bhd",
    "1082.KL": "Hong Leong Financial Group Bhd",
    "3182.KL": "Genting Bhd",
    "4715.KL": "Genting Malaysia Bhd",
    "4065.KL": "PPB Group Bhd",
    "5398.KL": "Gamuda Bhd",
    "5211.KL": "Sunway Bhd",
    "0166.KL": "Inari Amertron Bhd (Tech Mid-Cap)",
    "3867.KL": "Malaysian Pacific Industries Bhd (Tech)",
    "7277.KL": "Dialog Group Bhd (Energy)",
    "3336.KL": "IJM Corporation Bhd (Construction)",
    "7113.KL": "Top Glove Corporation Bhd (Healthcare)",
    "^KLSE": "FBM KLCI Index",
}

PERIOD_MAP: Dict[str, str] = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "Max": "max",
}

POSITIVE_KEYWORDS: set[str] = {
    "surge", "surges", "jump", "jumps", "gain", "gains", "profit", "profits",
    "growth", "dividend", "dividends", "upgrade", "upgraded", "buy", "outperform",
    "bullish", "record", "higher", "strong", "beat", "beats", "expansion",
    "contract", "award", "awarded", "rally", "rallies", "positive", "recovery",
}

NEGATIVE_KEYWORDS: set[str] = {
    "drop", "drops", "fall", "falls", "loss", "losses", "decline", "declines",
    "downgrade", "downgraded", "sell", "underperform", "bearish", "lower",
    "weak", "miss", "misses", "slump", "slumps", "plunge", "plunges", "cut",
    "cuts", "lawsuit", "fine", "penalty", "negative", "concern", "risk", "debt",
}


@dataclass(frozen=True)
class SectorProfile:
    """Configuration profile for a specific Bursa Malaysia sector."""

    sector_name: str
    regime_style: str
    primary_indicators: List[str]
    macro_proxies: Dict[str, str]
    indicator_weights: Dict[str, float]
    rss_search_terms: List[str]
    catalyst_keywords: List[str]
    reference_portals: List[str]


BURSA_SECTOR_PROFILES: Dict[str, SectorProfile] = {
    "Financial Services": SectorProfile(
        sector_name="Financial Services",
        regime_style="Mean-Reverting & Institutional Accumulation",
        primary_indicators=["Bollinger Bands (20,2)", "RSI (14)", "SMA (200)", "Chaikin Money Flow (20)"],
        macro_proxies={"MYR=X": "USD/MYR Spot Exchange Rate", "^KLSE": "FBM KLCI Index", "^TNX": "US 10Y Treasury Yield"},
        indicator_weights={
            "SMA (20)": 8.0, "SMA (50)": 12.0, "SMA (200)": 16.0, "EMA (12/26)": 6.0,
            "Bollinger Bands (20,2)": 15.0, "RSI (14)": 15.0, "MACD (12,26,9)": 8.0,
            "Stochastic (14,3,3)": 8.0, "OBV": 3.0, "Chaikin Money Flow (20)": 6.0, "Money Flow Index (14)": 3.0,
        },
        rss_search_terms=["Bank Negara Malaysia OPR banking", "Bursa Malaysia banking NIM loan growth"],
        catalyst_keywords=["opr", "nim", "dividend", "loan", "bnm", "casa", "provision", "capital"],
        reference_portals=[
            "Bank Negara Malaysia (bnm.gov.my) — Monetary Policy Statements (OPR)",
            "The Edge Malaysia — Banking & Finance Desk",
        ],
    ),
    "Plantation": SectorProfile(
        sector_name="Plantation",
        regime_style="Commodity-Cyclical (FCPO & Weather Driven)",
        primary_indicators=["MACD (12,26,9)", "EMA (12/26)", "OBV", "RSI (14)"],
        macro_proxies={"ZL=F": "Soybean Oil Futures (FCPO Proxy)", "MYR=X": "USD/MYR Spot Exchange Rate"},
        indicator_weights={
            "SMA (20)": 10.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 14.0,
            "Bollinger Bands (20,2)": 6.0, "RSI (14)": 12.0, "MACD (12,26,9)": 16.0,
            "Stochastic (14,3,3)": 6.0, "OBV": 6.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0,
        },
        rss_search_terms=["MPOB palm oil stockpile export Malaysia", "crude palm oil FCPO Bursa Malaysia"],
        catalyst_keywords=["fcpo", "mpob", "cpo", "stockpile", "el nino", "la nina", "export", "biodiesel", "ffb"],
        reference_portals=[
            "Malaysian Palm Oil Board (mpob.gov.my) — Monthly 10th Supply & Demand Report",
            "Intertek (ITS) & Amspec Malaysia — 5-Day Export Cargo Surveys",
        ],
    ),
    "Energy & Petrochemicals": SectorProfile(
        sector_name="Energy & Petrochemicals",
        regime_style="High-Beta Trend & Crude Oil / PETRONAS Capex Correlated",
        primary_indicators=["EMA (12/26)", "MACD (12,26,9)", "Stochastic (14,3,3)", "OBV"],
        macro_proxies={"BZ=F": "Brent Crude Oil Futures", "CL=F": "WTI Crude Oil Futures", "MYR=X": "USD/MYR Spot"},
        indicator_weights={
            "SMA (20)": 10.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 15.0,
            "Bollinger Bands (20,2)": 5.0, "RSI (14)": 10.0, "MACD (12,26,9)": 16.0,
            "Stochastic (14,3,3)": 8.0, "OBV": 6.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0,
        },
        rss_search_terms=["PETRONAS contract oil gas Bursa Malaysia", "Brent crude oil OPEC Malaysia energy"],
        catalyst_keywords=["petronas", "fpso", "brent", "opec", "charter", "capex", "upstream", "rig"],
        reference_portals=[
            "PETRONAS Activity Outlook (PAO) Annual Report",
            "Bursa Malaysia Energy Index Company Announcements",
        ],
    ),
    "Technology & EMS": SectorProfile(
        sector_name="Technology & EMS",
        regime_style="High-Momentum Breakout (SOX & Nasdaq Correlated)",
        primary_indicators=["EMA (12/26)", "MACD (12,26,9)", "RSI (14)", "Money Flow Index (14)"],
        macro_proxies={"^SOX": "PHLX Semiconductor Index", "^IXIC": "Nasdaq Composite", "MYR=X": "USD/MYR Spot"},
        indicator_weights={
            "SMA (20)": 12.0, "SMA (50)": 10.0, "SMA (200)": 8.0, "EMA (12/26)": 16.0,
            "Bollinger Bands (20,2)": 6.0, "RSI (14)": 14.0, "MACD (12,26,9)": 16.0,
            "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0,
        },
        rss_search_terms=["Malaysia semiconductor OSAT Penang technology Bursa", "MIDA NIMP semiconductor AI server Malaysia"],
        catalyst_keywords=["osat", "wafer", "semiconductor", "apple", "broadcom", "utilization", "penang", "kulim"],
        reference_portals=[
            "Semiconductor Industry Association (semiconductors.org) — Monthly Global Billings",
            "MIDA (mida.gov.my) — National Semiconductor Strategy (NSS) Updates",
        ],
    ),
    "Construction, Property & Infrastructure": SectorProfile(
        sector_name="Construction, Property & Infrastructure",
        regime_style="Orderbook Catalyst & Volume Breakout",
        primary_indicators=["OBV", "Money Flow Index (14)", "SMA (20/50)", "MACD (12,26,9)"],
        macro_proxies={"^KLSE": "FBM KLCI Index", "MYR=X": "USD/MYR Spot Exchange Rate"},
        indicator_weights={
            "SMA (20)": 12.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 10.0,
            "Bollinger Bands (20,2)": 6.0, "RSI (14)": 10.0, "MACD (12,26,9)": 12.0,
            "Stochastic (14,3,3)": 6.0, "OBV": 10.0, "Chaikin Money Flow (20)": 6.0, "Money Flow Index (14)": 6.0,
        },
        rss_search_terms=["Bursa Malaysia construction contract award data centre", "MRT3 ECRL Johor Singapore SEZ property Malaysia"],
        catalyst_keywords=["letter of award", "order book", "tender", "mrt3", "ecrl", "data centre", "sez", "gdv"],
        reference_portals=[
            "Bursa Malaysia Company Announcements — Contracts Awarded",
            "Ministry of Works (KKR) & Ministry of Transport (MOT) Infrastructure Briefings",
        ],
    ),
    "Utilities & Telecommunications": SectorProfile(
        sector_name="Utilities & Telecommunications",
        regime_style="Defensive Yield + Data Centre Power Demand",
        primary_indicators=["SMA (50/200)", "Bollinger Bands (20,2)", "Chaikin Money Flow (20)", "RSI (14)"],
        macro_proxies={"^TNX": "US 10Y Treasury Yield", "^KLSE": "FBM KLCI Index", "MYR=X": "USD/MYR Spot"},
        indicator_weights={
            "SMA (20)": 10.0, "SMA (50)": 14.0, "SMA (200)": 16.0, "EMA (12/26)": 8.0,
            "Bollinger Bands (20,2)": 12.0, "RSI (14)": 12.0, "MACD (12,26,9)": 10.0,
            "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 5.0, "Money Flow Index (14)": 3.0,
        },
        rss_search_terms=["Tenaga YTL Power data centre grid tariff Malaysia", "MCMC 5G spectrum telecom Bursa Malaysia"],
        catalyst_keywords=["icpt", "tariff", "grid", "renewable", "solar", "lss", "5g", "spectrum", "hyperscale"],
        reference_portals=[
            "Suruhanjaya Tenaga / Energy Commission (st.gov.my) — ICPT & LSS Tariff Reviews",
            "MCMC (mcmc.gov.my) — 5G & Spectrum Policy Announcements",
        ],
    ),
    "Industrial & Materials": SectorProfile(
        sector_name="Industrial & Materials",
        regime_style="Global Manufacturing & LME Metal Cyclical",
        primary_indicators=["MACD (12,26,9)", "SMA (50)", "EMA (12/26)", "RSI (14)"],
        macro_proxies={"ALI=F": "Aluminium Futures (LME Proxy)", "HG=F": "Copper Futures", "MYR=X": "USD/MYR Spot"},
        indicator_weights={
            "SMA (20)": 10.0, "SMA (50)": 14.0, "SMA (200)": 12.0, "EMA (12/26)": 12.0,
            "Bollinger Bands (20,2)": 8.0, "RSI (14)": 12.0, "MACD (12,26,9)": 14.0,
            "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0,
        },
        rss_search_terms=["Press Metal aluminium LME industrial Bursa Malaysia", "Malaysia manufacturing PMI export industrial"],
        catalyst_keywords=["lme", "aluminium", "smelter", "alumina", "pmi", "freight", "tariff"],
        reference_portals=[
            "London Metal Exchange (lme.com) — Aluminium & Base Metal Spot Prices",
            "S&P Global Malaysia Manufacturing PMI (1st business day of month)",
        ],
    ),
    "Consumer & Healthcare": SectorProfile(
        sector_name="Consumer & Healthcare",
        regime_style="Domestic Consumption & Defensive Cashflow",
        primary_indicators=["SMA (20/50)", "RSI (14)", "Stochastic (14,3,3)", "Money Flow Index (14)"],
        macro_proxies={"MYR=X": "USD/MYR (Import Cost Proxy)", "^KLSE": "FBM KLCI Index"},
        indicator_weights={
            "SMA (20)": 12.0, "SMA (50)": 12.0, "SMA (200)": 12.0, "EMA (12/26)": 8.0,
            "Bollinger Bands (20,2)": 10.0, "RSI (14)": 14.0, "MACD (12,26,9)": 10.0,
            "Stochastic (14,3,3)": 10.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0,
        },
        rss_search_terms=["Malaysia retail consumer F&B healthcare Bursa", "IHH Healthcare Nestle MR DIY QL Resources Malaysia"],
        catalyst_keywords=["subsidy", "sara", "str", "epf", "wage", "footfall", "asp", "hospital", "glove"],
        reference_portals=[
            "Department of Statistics Malaysia (dosm.gov.my) — Monthly CPI & Retail Trade",
            "Ministry of Health (MOH) & Domestic Trade (KPDN) Policy Briefings",
        ],
    ),
}

TICKER_TO_SECTOR: Dict[str, str] = {
    "1155.KL": "Financial Services", "1295.KL": "Financial Services", "1023.KL": "Financial Services",
    "5819.KL": "Financial Services", "1066.KL": "Financial Services", "1082.KL": "Financial Services",
    "1015.KL": "Financial Services", "5258.KL": "Financial Services", "5285.KL": "Plantation",
    "1961.KL": "Plantation", "2445.KL": "Plantation", "2291.KL": "Plantation",
    "5183.KL": "Energy & Petrochemicals", "5681.KL": "Energy & Petrochemicals", "3816.KL": "Energy & Petrochemicals",
    "7277.KL": "Energy & Petrochemicals", "5210.KL": "Energy & Petrochemicals", "7293.KL": "Energy & Petrochemicals",
    "5347.KL": "Utilities & Telecommunications", "6742.KL": "Utilities & Telecommunications", "4677.KL": "Utilities & Telecommunications",
    "6033.KL": "Utilities & Telecommunications", "6947.KL": "Utilities & Telecommunications", "6012.KL": "Utilities & Telecommunications",
    "4863.KL": "Utilities & Telecommunications", "5398.KL": "Construction, Property & Infrastructure", "5211.KL": "Construction, Property & Infrastructure",
    "3336.KL": "Construction, Property & Infrastructure", "8664.KL": "Construction, Property & Infrastructure", "5148.KL": "Construction, Property & Infrastructure",
    "8869.KL": "Industrial & Materials", "4197.KL": "Industrial & Materials", "0166.KL": "Technology & EMS",
    "3867.KL": "Technology & EMS", "0138.KL": "Technology & EMS", "0128.KL": "Technology & EMS",
    "5005.KL": "Technology & EMS", "5225.KL": "Consumer & Healthcare", "4707.KL": "Consumer & Healthcare",
    "7084.KL": "Consumer & Healthcare", "5296.KL": "Consumer & Healthcare", "3182.KL": "Consumer & Healthcare",
    "4715.KL": "Consumer & Healthcare", "4065.KL": "Consumer & Healthcare", "7113.KL": "Consumer & Healthcare",
    "5168.KL": "Consumer & Healthcare", "^KLSE": "Financial Services",
}


@dataclass
class IndicatorSignal:
    category: str
    indicator: str
    reading: str
    signal: str
    raw_score: int
    weight: float
    rationale: str


def get_sector_profile(symbol: str, manual_sector: str = "Auto-Detect") -> SectorProfile:
    if manual_sector != "Auto-Detect" and manual_sector in BURSA_SECTOR_PROFILES:
        return BURSA_SECTOR_PROFILES[manual_sector]
    sector_key = TICKER_TO_SECTOR.get(symbol.upper(), "Financial Services")
    return BURSA_SECTOR_PROFILES[sector_key]


# ==============================================================================
# SECTION 2: MARKET HOURS, POSITION SIZING & BURSA FEE CALCULATOR
# ==============================================================================


def get_bursa_market_status(now_myt: Optional[datetime] = None) -> str:
    if now_myt is None:
        now_myt = datetime.now(MYT)
    if now_myt.weekday() >= 5:
        return "CLOSED (Weekend)"
    current_t = now_myt.time()
    if time(9, 0) <= current_t < time(12, 30):
        return "OPEN (Morning Session 09:00–12:30 MYT)"
    if time(12, 30) <= current_t < time(14, 30):
        return "MIDDAY BREAK (12:30–14:30 MYT)"
    if time(14, 30) <= current_t < time(17, 0):
        return "OPEN (Afternoon Session 14:30–17:00 MYT)"
    return "CLOSED (Outside 09:00–17:00 MYT Hours)"


def normalize_bursa_ticker(raw_symbol: str) -> str:
    cleaned = raw_symbol.strip().upper()
    if not cleaned:
        return "1155.KL"
    if cleaned == "^KLSE" or cleaned.endswith(".KL"):
        return cleaned
    if cleaned.isdigit():
        return f"{cleaned}.KL"
    return f"{cleaned}.KL"


def validate_period_interval(period_label: str, interval: str) -> Tuple[str, Optional[str]]:
    yf_period = PERIOD_MAP.get(period_label, "1y")
    if interval == "5m" and yf_period not in ("1mo",):
        return ("1mo", "Yahoo Finance restricts 5m intraday history to 60 days. Period adjusted to 1M.")
    if interval == "1h" and yf_period in ("2y", "5y", "max"):
        return ("1y", "Yahoo Finance restricts 1h intraday history to 730 days. Period adjusted to 1Y.")
    return yf_period, None


def calculate_bursa_leg_fees(
    contract_value: float,
    brokerage_pct: float = 0.10,
    min_brokerage_rm: float = 8.00,
    apply_sst_on_brokerage: bool = True,
    sst_rate_pct: float = 8.0,
    clearing_fee_pct: float = 0.03,
    clearing_cap_rm: float = 1000.00,
    stamp_duty_per_1k: float = 1.00,
    stamp_duty_cap_rm: float = 1000.00,
) -> Dict[str, float]:
    if contract_value <= 0.0:
        return {"contract_value": 0.0, "brokerage": 0.0, "sst": 0.0, "clearing_fee": 0.0, "stamp_duty": 0.0, "total_fee": 0.0, "effective_fee_pct": 0.0}
    brokerage = round(max(min_brokerage_rm, contract_value * (brokerage_pct / 100.0)), 2)
    sst = round(brokerage * (sst_rate_pct / 100.0), 2) if apply_sst_on_brokerage else 0.0
    clearing_fee = round(min(clearing_cap_rm, contract_value * (clearing_fee_pct / 100.0)), 2)
    stamp_duty = float(min(stamp_duty_cap_rm, math.ceil(contract_value / 1000.0) * stamp_duty_per_1k))
    total_fee = round(brokerage + sst + clearing_fee + stamp_duty, 2)
    effective_pct = (total_fee / contract_value) * 100.0
    return {"contract_value": round(contract_value, 2), "brokerage": brokerage, "sst": sst, "clearing_fee": clearing_fee, "stamp_duty": stamp_duty, "total_fee": total_fee, "effective_fee_pct": effective_pct}


def compute_bursa_position_size(
    account_capital: float, max_risk_pct: float, entry_price: float, stop_loss: float, take_profit: float,
    brokerage_pct: float = 0.10, min_brokerage_rm: float = 8.00, apply_sst: bool = True, stamp_duty_per_1k: float = 1.00, manual_lots_override: Optional[int] = None
) -> Dict[str, Any]:
    risk_budget_rm = account_capital * (max_risk_pct / 100.0)
    per_share_stop_dist = entry_price - stop_loss
    if entry_price <= 0 or per_share_stop_dist <= 0:
        return {"ok": False, "error": "Entry Price must be strictly greater than Stop-Loss Price."}

    max_lots_by_capital = int(account_capital // (entry_price * BOARD_LOT_SIZE))
    max_lots_by_gross_risk = int(risk_budget_rm // (per_share_stop_dist * BOARD_LOT_SIZE))
    search_ceiling = max(0, min(max_lots_by_capital, max_lots_by_gross_risk))

    optimal_lots = 0
    for candidate_lots in range(search_ceiling, 0, -1):
        shares = candidate_lots * BOARD_LOT_SIZE
        buy_val = shares * entry_price
        stop_val = shares * stop_loss
        buy_fees = calculate_bursa_leg_fees(buy_val, brokerage_pct, min_brokerage_rm, apply_sst, stamp_duty_per_1k=stamp_duty_per_1k)
        stop_fees = calculate_bursa_leg_fees(stop_val, brokerage_pct, min_brokerage_rm, apply_sst, stamp_duty_per_1k=stamp_duty_per_1k)
        total_outlay = buy_val + buy_fees["total_fee"]
        net_loss_rm = (buy_val - stop_val) + buy_fees["total_fee"] + stop_fees["total_fee"]
        if total_outlay <= account_capital and net_loss_rm <= risk_budget_rm:
            optimal_lots = candidate_lots
            break

    active_lots = manual_lots_override if (manual_lots_override is not None and manual_lots_override > 0) else optimal_lots
    shares = active_lots * BOARD_LOT_SIZE

    if active_lots <= 0:
        return {"ok": False, "error": "Risk budget or capital is too small for even 1 Board Lot (100 shares)."}

    buy_val = shares * entry_price
    stop_val = shares * stop_loss
    tp_val = shares * take_profit
    buy_fees = calculate_bursa_leg_fees(buy_val, brokerage_pct, min_brokerage_rm, apply_sst, stamp_duty_per_1k=stamp_duty_per_1k)
    stop_fees = calculate_bursa_leg_fees(stop_val, brokerage_pct, min_brokerage_rm, apply_sst, stamp_duty_per_1k=stamp_duty_per_1k)
    tp_fees = calculate_bursa_leg_fees(tp_val, brokerage_pct, min_brokerage_rm, apply_sst, stamp_duty_per_1k=stamp_duty_per_1k)

    total_buy_outlay = buy_val + buy_fees["total_fee"]
    round_trip_fees_stop = buy_fees["total_fee"] + stop_fees["total_fee"]
    round_trip_fees_tp = buy_fees["total_fee"] + tp_fees["total_fee"]
    net_loss_stop = (buy_val - stop_val) + round_trip_fees_stop
    net_profit_tp = (tp_val - buy_val) - round_trip_fees_tp
    net_rr = (net_profit_tp / net_loss_stop) if net_loss_stop > 0 else 0.0
    breakeven_price = (total_buy_outlay + stop_fees["total_fee"]) / shares

    return {
        "ok": True, "optimal_lots": optimal_lots, "active_lots": active_lots, "shares": shares,
        "risk_budget_rm": risk_budget_rm, "gross_buy_val": buy_val, "total_buy_outlay": total_buy_outlay,
        "capital_utilization_pct": (total_buy_outlay / account_capital) * 100.0, "buy_fees": buy_fees,
        "stop_fees": stop_fees, "tp_fees": tp_fees, "round_trip_fees_tp": round_trip_fees_tp,
        "breakeven_price": breakeven_price, "net_loss_stop": net_loss_stop,
        "net_risk_pct_of_capital": (net_loss_stop / account_capital) * 100.0, "net_profit_tp": net_profit_tp, "net_rr": net_rr,
    }


# ==============================================================================
# SECTION 3: DATA FETCHING & CACHING (YFINANCE)
# ==============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(symbol: str, yf_period: str, interval: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=yf_period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(), f"No historical data returned for '{symbol}'."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return pd.DataFrame(), f"Missing columns for {symbol}: {missing}"
        df = df[required_cols].copy()
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df["Volume"] = df["Volume"].fillna(0).astype(float)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(MYT)
        else:
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if len(df) < 5:
            return pd.DataFrame(), f"Insufficient historical bars ({len(df)}) for '{symbol}'."
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"Error fetching historical data for '{symbol}': {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(symbol: str) -> Dict[str, Any]:
    fetched_at = datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S MYT (UTC+8)")
    try:
        ticker_obj = yf.Ticker(symbol)
        recent = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
        if recent is None or recent.empty:
            return {"ok": False, "error": f"Unable to retrieve latest quote for {symbol}.", "fetched_at": fetched_at}
        if isinstance(recent.columns, pd.MultiIndex):
            recent.columns = [c[0] for c in recent.columns]
        recent = recent.dropna(subset=["Close"])
        if recent.empty:
            return {"ok": False, "error": f"Latest quote contains no valid closing prices for {symbol}.", "fetched_at": fetched_at}
        last_row = recent.iloc[-1]
        prev_close = float(recent.iloc[-2]["Close"]) if len(recent) >= 2 else float(last_row["Open"])
        last_price = float(last_row["Close"])
        change = last_price - prev_close
        pct_change = (change / prev_close * 100.0) if prev_close != 0 else 0.0
        return {
            "ok": True, "symbol": symbol, "last_price": last_price, "prev_close": prev_close,
            "change": change, "pct_change": pct_change, "day_high": float(last_row["High"]),
            "day_low": float(last_row["Low"]), "volume": float(last_row.get("Volume", 0.0)),
            "lot_cost": last_price * BOARD_LOT_SIZE, "fetched_at": fetched_at,
        }
    except Exception as exc:
        return {"ok": False, "error": f"Live quote fetch error for {symbol}: {exc}", "fetched_at": fetched_at}


# ==============================================================================
# SECTION 4: TECHNICAL INDICATORS & SUPPORT/RESISTANCE
# ==============================================================================


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    out["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    out["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    out["SMA_200"] = close.rolling(window=200, min_periods=1).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    bb_std = close.rolling(window=20, min_periods=2).std(ddof=0)
    out["BB_Mid"] = out["SMA_20"]
    out["BB_Upper"] = out["BB_Mid"] + 2.0 * bb_std
    out["BB_Lower"] = out["BB_Mid"] - 2.0 * bb_std

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    out["RSI_14"] = out["RSI_14"].fillna(50.0)

    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    denom_stoch = (high_14 - low_14).replace(0.0, np.nan)
    stoch_k_raw = 100.0 * (close - low_14) / denom_stoch
    out["Stoch_K"] = stoch_k_raw.rolling(window=3, min_periods=1).mean().fillna(50.0)
    out["Stoch_D"] = out["Stoch_K"].rolling(window=3, min_periods=1).mean().fillna(50.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR_14"] = true_range.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()

    direction = np.sign(close.diff()).fillna(0.0)
    out["OBV"] = (direction * volume).cumsum()
    out["OBV_SMA_20"] = out["OBV"].rolling(window=20, min_periods=1).mean()

    hl_range = (high - low).replace(0.0, np.nan)
    mfm = (((close - low) - (high - close)) / hl_range).fillna(0.0)
    mfv = mfm * volume
    vol_sum_20 = volume.rolling(window=20, min_periods=1).sum().replace(0.0, np.nan)
    out["CMF_20"] = (mfv.rolling(window=20, min_periods=1).sum() / vol_sum_20).fillna(0.0)

    tp = (high + low + close) / 3.0
    rmf = tp * volume
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, rmf, 0.0), index=out.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, rmf, 0.0), index=out.index)
    pos_mf_14 = pos_mf.rolling(window=14, min_periods=1).sum()
    neg_mf_14 = neg_mf.rolling(window=14, min_periods=1).sum().replace(0.0, np.nan)
    mfr = pos_mf_14 / neg_mf_14
    out["MFI_14"] = (100.0 - (100.0 / (1.0 + mfr))).fillna(50.0)
    out["Vol_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()
    return out


def compute_support_resistance(df: pd.DataFrame, window: int = 5) -> Tuple[float, float, List[float], List[float]]:
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    if n >= (2 * window + 1):
        for i in range(window, n - window):
            if highs[i] == np.max(highs[i - window : i + window + 1]):
                swing_highs.append(float(highs[i]))
            if lows[i] == np.min(lows[i - window : i + window + 1]):
                swing_lows.append(float(lows[i]))
    last_close = float(df["Close"].iloc[-1])
    recent_slice = df.tail(min(60, len(df)))
    below_pivots = [lvl for lvl in swing_lows[-10:] if lvl < last_close]
    above_pivots = [lvl for lvl in swing_highs[-10:] if lvl > last_close]
    support = float(max(below_pivots)) if below_pivots else float(recent_slice["Low"].min())
    resistance = float(min(above_pivots)) if above_pivots else float(recent_slice["High"].max())
    return support, resistance, swing_lows[-5:], swing_highs[-5:]


# ==============================================================================
# SECTION 5: SIGNAL SCORING & SCREENER
# ==============================================================================


def evaluate_signals(df: pd.DataFrame, sector_profile: SectorProfile) -> Tuple[pd.DataFrame, float, List[IndicatorSignal]]:
    last = df.iloc[-1]
    close = float(last["Close"])
    has_volume = float(df["Volume"].tail(20).sum()) > 0
    w = sector_profile.indicator_weights
    signals: List[IndicatorSignal] = []

    sma20 = float(last["SMA_20"])
    s_sma20 = ("Bullish", 1, f"Close > SMA20") if close > sma20 * 1.002 else ("Bearish", -1, f"Close < SMA20") if close < sma20 * 0.998 else ("Neutral", 0, "Near SMA20")
    signals.append(IndicatorSignal("Trend", "SMA (20)", f"RM {sma20:.3f}", s_sma20[0], s_sma20[1], w["SMA (20)"], s_sma20[2]))

    sma50 = float(last["SMA_50"])
    s_sma50 = ("Bullish", 1, f"Close > SMA50") if close > sma50 * 1.003 else ("Bearish", -1, f"Close < SMA50") if close < sma50 * 0.997 else ("Neutral", 0, "Near SMA50")
    signals.append(IndicatorSignal("Trend", "SMA (50)", f"RM {sma50:.3f}", s_sma50[0], s_sma50[1], w["SMA (50)"], s_sma50[2]))

    sma200 = float(last["SMA_200"])
    s_sma200 = ("Bullish", 1, f"Above SMA200") if close > sma200 * 1.005 else ("Bearish", -1, f"Below SMA200") if close < sma200 * 0.995 else ("Neutral", 0, "Testing SMA200")
    signals.append(IndicatorSignal("Trend", "SMA (200)", f"RM {sma200:.3f}", s_sma200[0], s_sma200[1], w["SMA (200)"], s_sma200[2]))

    ema12, ema26 = float(last["EMA_12"]), float(last["EMA_26"])
    s_ema = ("Bullish", 1, "EMA(12)>EMA(26)") if ema12 > ema26 else ("Bearish", -1, "EMA(12)<EMA(26)") if ema12 < ema26 else ("Neutral", 0, "Flat")
    signals.append(IndicatorSignal("Trend", "EMA (12/26)", f"{ema12:.3f}/{ema26:.3f}", s_ema[0], s_ema[1], w["EMA (12/26)"], s_ema[2]))

    bb_u, bb_l = float(last["BB_Upper"]), float(last["BB_Lower"])
    s_bb = ("Bullish", 1, "Oversold at Lower Band") if close <= bb_l else ("Bearish", -1, "Overbought at Upper Band") if close >= bb_u else ("Neutral", 0, "Inside BB")
    signals.append(IndicatorSignal("Volatility", "Bollinger Bands (20,2)", f"[{bb_l:.3f}, {bb_u:.3f}]", s_bb[0], s_bb[1], w["Bollinger Bands (20,2)"], s_bb[2]))

    rsi = float(last["RSI_14"])
    s_rsi = ("Bullish", 1, "Oversold (<30)") if rsi < 30 else ("Bearish", -1, "Overbought (>70)") if rsi > 70 else ("Bullish", 1, "Constructive (55-70)") if rsi >= 55 else ("Bearish", -1, "Weak (30-45)") if rsi <= 45 else ("Neutral", 0, "Neutral")
    signals.append(IndicatorSignal("Momentum", "RSI (14)", f"{rsi:.1f}", s_rsi[0], s_rsi[1], w["RSI (14)"], s_rsi[2]))

    macd_val, macd_sig, macd_h = float(last["MACD"]), float(last["MACD_Signal"]), float(last["MACD_Hist"])
    s_macd = ("Bullish", 1, "MACD > Signal") if macd_val > macd_sig else ("Bearish", -1, "MACD < Signal") if macd_val < macd_sig else ("Neutral", 0, "Flat")
    signals.append(IndicatorSignal("Momentum", "MACD (12,26,9)", f"Hist:{macd_h:.4f}", s_macd[0], s_macd[1], w["MACD (12,26,9)"], s_macd[2]))

    st_k, st_d = float(last["Stoch_K"]), float(last["Stoch_D"])
    s_stoch = ("Bullish", 1, "Stoch bullish turn") if st_k < 20 and st_k >= st_d else ("Bearish", -1, "Stoch bearish turn") if st_k > 80 and st_k <= st_d else ("Bullish", 1, "%K>%D") if st_k > st_d else ("Bearish", -1, "%K<%D") if st_k < st_d else ("Neutral", 0, "Neutral")
    signals.append(IndicatorSignal("Momentum", "Stochastic (14,3,3)", f"%K:{st_k:.1f}/%D:{st_d:.1f}", s_stoch[0], s_stoch[1], w["Stochastic (14,3,3)"], s_stoch[2]))

    obv_val, obv_sma = float(last["OBV"]), float(last["OBV_SMA_20"])
    s_obv = ("Neutral", 0, "No Volume") if not has_volume else ("Bullish", 1, "OBV > 20-SMA") if obv_val > obv_sma else ("Bearish", -1, "OBV < 20-SMA") if obv_val < obv_sma else ("Neutral", 0, "Flat")
    signals.append(IndicatorSignal("Money Flow", "OBV", f"{obv_val:,.0f}", s_obv[0], s_obv[1], w["OBV"], s_obv[2]))

    cmf_val = float(last["CMF_20"])
    s_cmf = ("Neutral", 0, "No Volume") if not has_volume else ("Bullish", 1, "CMF > +0.05") if cmf_val > 0.05 else ("Bearish", -1, "CMF < -0.05") if cmf_val < -0.05 else ("Neutral", 0, "Neutral")
    signals.append(IndicatorSignal("Money Flow", "Chaikin Money Flow (20)", f"{cmf_val:.3f}", s_cmf[0], s_cmf[1], w["Chaikin Money Flow (20)"], s_cmf[2]))

    mfi_val = float(last["MFI_14"])
    s_mfi = ("Neutral", 0, "No Volume") if not has_volume else ("Bullish", 1, "MFI Oversold") if mfi_val < 20 else ("Bearish", -1, "MFI Overbought") if mfi_val > 80 else ("Bullish", 1, "Strong Inflow") if mfi_val > 55 else ("Bearish", -1, "Weak Flow") if mfi_val < 45 else ("Neutral", 0, "Neutral")
    signals.append(IndicatorSignal("Money Flow", "Money Flow Index (14)", f"{mfi_val:.1f}", s_mfi[0], s_mfi[1], w["Money Flow Index (14)"], s_mfi[2]))

    composite_score = float(sum(item.raw_score * item.weight for item in signals))
    composite_score = max(-100.0, min(100.0, composite_score))

    summary_rows = [
        {
            "Category": item.category, "Indicator": item.indicator, "Reading": item.reading,
            "Signal": item.signal, f"Sector Weight ({sector_profile.sector_name}) %": item.weight,
            "Weighted Contribution": item.raw_score * item.weight, "Interpretation": item.rationale,
        }
        for item in signals
    ]
    return pd.DataFrame(summary_rows), composite_score, signals


def interpret_money_flow(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    vol_now = float(last["Volume"])
    vol_avg20 = float(last["Vol_SMA_20"])
    vol_ratio = (vol_now / vol_avg20) if vol_avg20 > 0 else 0.0
    cmf = float(last["CMF_20"])
    mfi = float(last["MFI_14"])
    obv = float(last["OBV"])
    obv_sma = float(last["OBV_SMA_20"])

    if vol_avg20 == 0:
        return {"obv": obv, "cmf": cmf, "mfi": mfi, "vol_now": vol_now, "vol_avg20": vol_avg20, "vol_ratio": 0.0, "state": "Neutral", "summary": "Neutral — No volume reported."}

    score = 0
    if obv > obv_sma: score += 1
    elif obv < obv_sma: score -= 1
    if cmf > 0.05: score += 1
    elif cmf < -0.05: score -= 1
    if mfi >= 55: score += 1
    elif mfi <= 45: score -= 1

    if score >= 2:
        state, summary = "Accumulation", f"Accumulation — Institutional buying pressure detected (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={vol_ratio:.2f}x)."
    elif score <= -2:
        state, summary = "Distribution", f"Distribution — Net selling pressure dominates (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={vol_ratio:.2f}x)."
    else:
        state, summary = "Neutral", f"Neutral — Mixed money flow signals (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={vol_ratio:.2f}x)."

    return {"obv": obv, "cmf": cmf, "mfi": mfi, "vol_now": vol_now, "vol_avg20": vol_avg20, "vol_ratio": vol_ratio, "state": state, "summary": summary}


@st.cache_data(ttl=900, show_spinner=False)
def run_bursa_sector_screener(symbols: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        if sym == "^KLSE": continue
        raw_df, err = fetch_historical_data(sym, "6mo", "1d")
        if err or raw_df.empty or len(raw_df) < 20: continue
        ind_df = compute_technical_indicators(raw_df)
        sec_prof = get_sector_profile(sym, "Auto-Detect")
        _, comp_score, _ = evaluate_signals(ind_df, sec_prof)
        mf = interpret_money_flow(ind_df)
        sup_lvl, res_lvl, _, _ = compute_support_resistance(ind_df)
        last = ind_df.iloc[-1]
        prev_close = float(ind_df.iloc[-2]["Close"]) if len(ind_df) >= 2 else float(last["Open"])
        close_now = float(last["Close"])
        chg_pct = ((close_now - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
        atr_pct = (float(last["ATR_14"]) / close_now * 100.0) if close_now > 0 else 0.0
        verdict = "Bullish" if comp_score >= 25 else "Bearish" if comp_score <= -25 else "Neutral"
        rows.append({
            "Ticker": sym, "Company": KLCI_COMPONENTS.get(sym, sym), "Sector": sec_prof.sector_name,
            "Close (RM)": round(close_now, 3), "1D Chg (%)": round(chg_pct, 2), "1 Lot (RM)": round(close_now * BOARD_LOT_SIZE, 2),
            "Sector Score": round(comp_score, 1), "Verdict": verdict, "RSI (14)": round(float(last["RSI_14"]), 1),
            "CMF (20)": round(float(last["CMF_20"]), 3), "MFI (14)": round(float(last["MFI_14"]), 1),
            "Vol / 20d Avg": round(mf["vol_ratio"], 2), "Money Flow": mf["state"], "ATR %": round(atr_pct, 2),
            "Support (RM)": round(sup_lvl, 3), "Resistance (RM)": round(res_lvl, 3),
        })
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.sort_values(by=["Sector Score", "Vol / 20d Avg"], ascending=[False, False], inplace=True)
        out_df.reset_index(drop=True, inplace=True)
    return out_df


# ==============================================================================
# SECTION 6: NEWS & FORECASTING
# ==============================================================================


def score_headline_sentiment(title: str, catalyst_keywords: List[str]) -> Tuple[str, int, List[str]]:
    lower_title = title.lower()
    tokens = {tok.strip(".,:;!?()[]\"'").lower() for tok in title.split() if tok.strip(".,:;!?()[]\"'")}
    pos_hits = len(tokens.intersection(POSITIVE_KEYWORDS))
    neg_hits = len(tokens.intersection(NEGATIVE_KEYWORDS))
    net = pos_hits - neg_hits
    matched = [kw.upper() for kw in catalyst_keywords if kw.lower() in lower_title]
    if net > 0: return "Positive", net, matched
    if net < 0: return "Negative", net, matched
    return "Neutral", 0, matched


def parse_rss_url(url: str, source_fallback: str, catalyst_keywords: List[str], timeout: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BursaProAnalyzer/3.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
        parsed = feedparser.parse(raw_bytes)
        items = []
        for entry in parsed.entries[:12]:
            title = str(entry.get("title", "")).strip()
            link = str(entry.get("link", "")).strip()
            if not title or not link: continue
            source_title = entry.get("source", {}).get("title", source_fallback) if isinstance(entry.get("source"), dict) else source_fallback
            pub_str = str(entry.get("published", entry.get("updated", "")))
            pub_myt = "Unknown time"
            if pub_str:
                try:
                    dt = parsedate_to_datetime(pub_str)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    pub_myt = dt.astimezone(MYT).strftime("%Y-%m-%d %H:%M MYT")
                except Exception:
                    pub_myt = pub_str[:25]
            sentiment, score, catalysts = score_headline_sentiment(title, catalyst_keywords)
            items.append({"title": title, "source": source_title, "published_myt": pub_myt, "link": link, "sentiment": sentiment, "score": score, "catalysts": ", ".join(catalysts) if catalysts else "—"})
        return items, None
    except Exception as exc:
        return [], f"{source_fallback} feed unreachable ({exc})"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_bursa_news(company_name: str, symbol: str, sector_rss_query: str, catalyst_keywords: Tuple[str, ...]) -> Tuple[List[Dict[str, Any]], List[str]]:
    clean_name = company_name.split("(")[0].replace("Bhd", "").strip()
    q_comp = urllib.parse.quote(f"{clean_name} Bursa Malaysia")
    q_sec = urllib.parse.quote(sector_rss_query)
    feeds = [
        (f"https://news.google.com/rss/search?q={q_comp}&hl=en-MY&gl=MY&ceid=MY:en", "Google News (Company)"),
        (f"https://news.google.com/rss/search?q={q_sec}&hl=en-MY&gl=MY&ceid=MY:en", "Google News (Sector)"),
        ("https://www.thestar.com.my/rss/Business/Business-News", "The Star Business"),
        ("https://theedgemalaysia.com/rss", "The Edge Malaysia"),
    ]
    all_articles, feed_notices, seen = [], [], set()
    kw_list = list(catalyst_keywords)
    for rss_url, label in feeds:
        articles, err = parse_rss_url(rss_url, label, kw_list)
        if err:
            feed_notices.append(err)
            continue
        if "The Star" in label or "The Edge" in label:
            keywords = {w.lower() for w in clean_name.split() if len(w) > 2}
            keywords.update({symbol.replace(".KL", "").lower(), "bursa", "klci", "ringgit"})
            articles = [a for a in articles if any(k in a["title"].lower() for k in keywords)][:5]
        for art in articles:
            nt = art["title"].lower()
            if nt not in seen:
                seen.add(nt)
                all_articles.append(art)
    return all_articles[:16], feed_notices


def generate_forecast_and_backtest(df: pd.DataFrame, horizon: int, model_type: str = "ARIMA(1,1,1)") -> Dict[str, Any]:
    close_series = df["Close"].astype(float).dropna()
    if len(close_series) < 35:
        return {"ok": False, "error": "At least 35 historical bars required for forecasting."}
    values = close_series.values
    dates = close_series.index
    median_delta = pd.Series(dates).diff().median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0): median_delta = pd.Timedelta(days=1)

    future_dates, curr_dt = [], dates[-1]
    while len(future_dates) < horizon:
        curr_dt += median_delta
        if median_delta >= pd.Timedelta(days=1) and curr_dt.weekday() >= 5: continue
        future_dates.append(curr_dt)

    try:
        if model_type.startswith("ARIMA"):
            model = ARIMA(values, order=(1, 1, 1)).fit()
            fc_res = model.get_forecast(steps=horizon)
            mean_fc = np.asarray(fc_res.predicted_mean, dtype=float)
            ci_80 = np.asarray(fc_res.conf_int(alpha=0.20), dtype=float)
            ci_95 = np.asarray(fc_res.conf_int(alpha=0.05), dtype=float)
            lower_80, upper_80 = ci_80[:, 0], ci_80[:, 1]
            lower_95, upper_95 = ci_95[:, 0], ci_95[:, 1]
        else:
            hw = ExponentialSmoothing(values, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated").fit(optimized=True)
            mean_fc = np.asarray(hw.forecast(steps=horizon), dtype=float)
            sigma = float(np.std(values - hw.fittedvalues, ddof=1)) if len(values) > 2 else float(np.std(values) * 0.02)
            step_scale = np.sqrt(np.arange(1, horizon + 1, dtype=float))
            lower_80, upper_80 = mean_fc - 1.2816 * sigma * step_scale, mean_fc + 1.2816 * sigma * step_scale
            lower_95, upper_95 = mean_fc - 1.9600 * sigma * step_scale, mean_fc + 1.9600 * sigma * step_scale

        bt_steps = min(15, max(5, len(values) // 6))
        train_start = len(values) - bt_steps
        actuals, preds, actual_dirs, pred_dirs = [], [], [], []
        for idx in range(train_start, len(values)):
            hist = values[:idx]
            true_v, prev_v = float(values[idx]), float(hist[-1])
            try:
                if model_type.startswith("ARIMA"):
                    p_v = float(ARIMA(hist, order=(1, 1, 1)).fit().forecast(steps=1)[0])
                else:
                    p_v = float(ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal=None).fit(optimized=True).forecast(steps=1)[0])
            except Exception:
                p_v = prev_v
            actuals.append(true_v)
            preds.append(p_v)
            actual_dirs.append(1 if true_v >= prev_v else -1)
            pred_dirs.append(1 if p_v >= prev_v else -1)

        mape = float(mean_absolute_percentage_error(actuals, preds) * 100.0)
        dir_acc = (sum(1 for a, p in zip(actual_dirs, pred_dirs) if a == p) / len(actual_dirs)) * 100.0 if actual_dirs else 0.0
        fc_df = pd.DataFrame({"Date": future_dates, "Forecast": mean_fc, "Lower_80": lower_80, "Upper_80": upper_80, "Lower_95": lower_95, "Upper_95": upper_95})
        return {
            "ok": True, "model_type": model_type, "horizon": horizon, "forecast_df": fc_df,
            "last_close": float(values[-1]), "target_price": float(mean_fc[-1]),
            "expected_return_pct": float((mean_fc[-1] - values[-1]) / values[-1] * 100.0),
            "mape_pct": mape, "directional_accuracy_pct": dir_acc, "bt_steps": bt_steps, "bt_dates": dates[train_start:], "bt_preds": preds,
        }
    except Exception as exc:
        return {"ok": False, "error": f"Forecasting failed: {exc}"}


# ==============================================================================
# SECTION 7: PLOTLY CHARTS
# ==============================================================================


def build_main_technical_chart(df: pd.DataFrame, symbol: str, show_sma: bool, show_ema: bool, show_bb: bool, show_sr: bool, panel4_mode: str, support_lvl: float, resistance_lvl: float) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=(f"{symbol} Price ({CURRENCY_SYMBOL}) & Volume", "RSI (14)", "MACD (12, 26, 9)", f"Panel 4: {panel4_mode}"),
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", increasing_line_color="#00E676", decreasing_line_color="#FF5252"), row=1, col=1, secondary_y=False)
    vol_colors = np.where(df["Close"] >= df["Open"], "rgba(0, 230, 118, 0.28)", "rgba(255, 82, 82, 0.28)")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.45), row=1, col=1, secondary_y=True)

    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="#FFD54F", width=1.3)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="#4FC3F7", width=1.4)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200", line=dict(color="#BA68C8", width=1.6)), row=1, col=1, secondary_y=False)
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12", line=dict(color="#26A69A", width=1.2, dash="dot")), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_26"], name="EMA 26", line=dict(color="#FF8A65", width=1.2, dash="dot")), row=1, col=1, secondary_y=False)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="rgba(176,190,197,0.55)", width=1, dash="dash")), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="rgba(176,190,197,0.55)", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(176,190,197,0.07)"), row=1, col=1, secondary_y=False)
    if show_sr:
        fig.add_hline(y=support_lvl, line_dash="dot", line_color="#00E676", annotation_text=f"Support RM{support_lvl:.3f}", annotation_position="bottom right", row=1, col=1)
        fig.add_hline(y=resistance_lvl, line_dash="dot", line_color="#FF5252", annotation_text=f"Resistance RM{resistance_lvl:.3f}", annotation_position="top right", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI (14)", line=dict(color="#FFA726", width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    hist_colors = np.where(df["MACD_Hist"] >= 0, "#26A69A", "#EF5350")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist", marker_color=hist_colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#42A5F5", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="MACD Signal", line=dict(color="#FFCA28", width=1.4)), row=3, col=1)

    if panel4_mode.startswith("All"):
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K", line=dict(color="#29B6F6", width=1.4)), row=4, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D", line=dict(color="#FF7043", width=1.3, dash="dot")), row=4, col=1, secondary_y=False)
        atr_s = df["ATR_14"]
        atr_norm = ((atr_s - atr_s.min()) / (atr_s.max() - atr_s.min()) * 100.0) if (atr_s.max() - atr_s.min()) > 0 else pd.Series(50, index=df.index)
        fig.add_trace(go.Scatter(x=df.index, y=atr_norm, customdata=df["ATR_14"], name="ATR [Scaled]", line=dict(color="#AB47BC", width=1.4)), row=4, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="#66BB6A", width=1.5)), row=4, col=1, secondary_y=True)
    elif "Stochastic" in panel4_mode:
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K", line=dict(color="#29B6F6", width=1.5)), row=4, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D", line=dict(color="#FF7043", width=1.4)), row=4, col=1, secondary_y=False)
    elif "ATR" in panel4_mode:
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR_14"], name="ATR RM", line=dict(color="#AB47BC", width=1.7)), row=4, col=1, secondary_y=False)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="#66BB6A", width=1.7)), row=4, col=1, secondary_y=False)

    fig.update_yaxes(title_text=f"Price ({CURRENCY_SYMBOL})", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_layout(template="plotly_dark", height=920, margin=dict(l=45, r=45, t=45, b=35), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.01, x=0), hovermode="x unified")
    return fig


def build_macro_proxy_chart(stock_df: pd.DataFrame, macro_df: pd.DataFrame, symbol: str, macro_symbol: str, macro_label: str) -> Tuple[Optional[go.Figure], Optional[float]]:
    s_close, m_close = stock_df["Close"].copy(), macro_df["Close"].copy()
    s_close.index, m_close.index = pd.to_datetime(s_close.index).tz_localize(None).normalize(), pd.to_datetime(m_close.index).tz_localize(None).normalize()
    aligned = pd.DataFrame({"Stock": s_close.groupby(s_close.index).last(), "Macro": m_close.groupby(m_close.index).last()}).dropna()
    if len(aligned) < 15: return None, None
    aligned["Rolling_Corr_20"] = aligned["Stock"].pct_change().rolling(window=20, min_periods=10).corr(aligned["Macro"].pct_change()).fillna(0.0)
    latest_corr = float(aligned["Rolling_Corr_20"].iloc[-1])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.68, 0.32], specs=[[{"secondary_y": True}], [{"secondary_y": False}]], subplot_titles=(f"{symbol} vs {macro_label} ({macro_symbol})", "20-Bar Rolling Return Correlation"))
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Stock"], name=f"{symbol}", line=dict(color="#00E676", width=2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Macro"], name=f"{macro_label}", line=dict(color="#FFD54F", width=1.8, dash="dot")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Rolling_Corr_20"], name="Rolling Corr", line=dict(color="#4FC3F7", width=1.6), fill="tozeroy", fillcolor="rgba(79,195,247,0.14)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=40, t=45, b=35), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    return fig, latest_corr


def build_foreign_flow_chart(df: pd.DataFrame, flow_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{symbol} Close", line=dict(color="#4FC3F7", width=2)), secondary_y=False)
    bar_colors = np.where(flow_df["Net_Flow"] >= 0, "#00E676", "#FF5252")
    fig.add_trace(go.Bar(x=flow_df["Date"], y=flow_df["Net_Flow"], name="Foreign Net Flow (RM mil) [User-supplied]", marker_color=bar_colors, opacity=0.65), secondary_y=True)
    fig.update_layout(template="plotly_dark", title=f"{symbol} Price vs Foreign Net Flow", height=400, margin=dict(l=40, r=40, t=50, b=35), hovermode="x unified")
    return fig


def build_forecast_chart(df: pd.DataFrame, fc_result: Dict[str, Any], symbol: str) -> go.Figure:
    recent_df = df.tail(120)
    fc_df = fc_result["forecast_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["Close"], name="Historical Close", line=dict(color="#4FC3F7", width=2)))
    fig.add_trace(go.Scatter(x=fc_result["bt_dates"], y=fc_result["bt_preds"], name="Walk-Forward Pred", line=dict(color="#FFD54F", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]), y=pd.concat([fc_df["Upper_95"], fc_df["Lower_95"][::-1]]), fill="toself", fillcolor="rgba(255,138,101,0.14)", line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig.add_trace(go.Scatter(x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]), y=pd.concat([fc_df["Upper_80"], fc_df["Lower_80"][::-1]]), fill="toself", fillcolor="rgba(255,138,101,0.28)", line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], name=f"{fc_result['model_type']} Forecast", line=dict(color="#FF7043", width=2.2, dash="dash")))
    fig.update_layout(template="plotly_dark", title=f"{symbol} Statistical Price Projection", height=460, margin=dict(l=40, r=40, t=50, b=35), hovermode="x unified")
    return fig


def build_screener_scatter_chart(screener_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    color_map = {"Bullish": "#00E676", "Neutral": "#FFD54F", "Bearish": "#FF5252"}
    for verdict_label, group in screener_df.groupby("Verdict"):
        fig.add_trace(go.Scatter(
            x=group["RSI (14)"], y=group["Sector Score"], mode="markers+text",
            text=group["Ticker"].str.replace(".KL", "", regex=False), textposition="top center",
            name=verdict_label, marker=dict(size=np.clip(group["Vol / 20d Avg"] * 14, 10, 42), color=color_map.get(verdict_label, "#4FC3F7"), opacity=0.85, line=dict(width=1, color="#fff")),
        ))
    fig.add_hline(y=25, line_dash="dot", line_color="#00E676")
    fig.add_hline(y=-25, line_dash="dot", line_color="#FF5252")
    fig.update_layout(template="plotly_dark", title="Screener Map: RSI vs Sector Score (Bubble Size = Vol Ratio)", xaxis_title="RSI (14)", yaxis_title="Sector Score (-100 to +100)", height=520, margin=dict(l=40, r=40, t=55, b=40))
    return fig


# ==============================================================================
# SECTION 8: STREAMLIT APP RUNTIME & UI TABS
# ==============================================================================


def _render_live_quote_banner(symbol: str, company_label: str) -> None:
    quote = fetch_live_quote(symbol)
    now_myt = datetime.now(MYT)
    st.caption(f"**Exchange Status:** `{get_bursa_market_status(now_myt)}` | **Last updated:** `{quote.get('fetched_at', 'Now')}` | ⚠️ **Delayed data**")
    if not quote.get("ok"):
        st.warning(quote.get("error", "Live quote unavailable."))
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{symbol} ({company_label[:20]})", f"RM {quote['last_price']:.3f}", f"{quote['change']:+.3f} ({quote['pct_change']:+.2f}%)")
    c2.metric("Day Range", f"{quote['day_low']:.3f} – {quote['day_high']:.3f}")
    c3.metric("Volume", f"{quote['volume']:,.0f}")
    c4.metric(f"1 Board Lot ({BOARD_LOT_SIZE} sh)", f"RM {quote['lot_cost']:,.2f}")
    c5.metric("Prev Close", f"RM {quote['prev_close']:.3f}")


if hasattr(st, "fragment"):
    render_live_quote_section = st.fragment(run_every=60)(_render_live_quote_banner)
else:
    render_live_quote_section = _render_live_quote_banner


def main() -> None:
    st.set_page_config(page_title="Bursa Pro Analyzer | Watchlist & Workbench", page_icon="📈", layout="wide")

    # Initialize Watchlist in Session State
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["1155.KL", "1295.KL", "5347.KL", "5398.KL", "0166.KL"]

    st.title("📈 Bursa Pro Analyzer — Watchlist, Position Sizer & KLSE Workbench")
    st.markdown("Quantitative equity, sector intelligence, **Watchlist Dashboard**, and Bursa board-lot calculator.")

    with st.sidebar:
        st.header("⚙️ Analyzer Controls")

        preset_options = [f"{code} — {name}" for code, name in KLCI_COMPONENTS.items()]
        selected_preset = st.selectbox("Select Bursa Component / Index", options=preset_options, index=0)
        custom_ticker = st.text_input("Or Enter 4-Digit Code / Ticker", value="", placeholder="e.g. 0166, 5225, or 7113.KL")

        if custom_ticker.strip():
            active_symbol = normalize_bursa_ticker(custom_ticker)
            company_name = KLCI_COMPONENTS.get(active_symbol, f"Bursa Equity ({active_symbol})")
        else:
            active_symbol = selected_preset.split(" — ")[0].strip()
            company_name = KLCI_COMPONENTS.get(active_symbol, active_symbol)

        # Watchlist toggle button
        is_in_watchlist = active_symbol in st.session_state.watchlist
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if is_in_watchlist:
                if st.button("⭐ In Watchlist", use_container_width=True):
                    st.session_state.watchlist.remove(active_symbol)
                    st.rerun()
            else:
                if st.button("➕ Add to Watchlist", use_container_width=True):
                    if active_symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(active_symbol)
                    st.rerun()
        with col_w2:
            if st.button("🗑️ Clear List", use_container_width=True):
                st.session_state.watchlist = []
                st.rerun()

        st.subheader("🏛️ Sector Profile")
        sector_choices = ["Auto-Detect"] + list(BURSA_SECTOR_PROFILES.keys())
        selected_sector_mode = st.selectbox("Sector Weighting", options=sector_choices, index=0)
        sector_profile = get_sector_profile(active_symbol, selected_sector_mode)

        macro_choices = list(sector_profile.macro_proxies.keys())
        selected_macro_symbol = st.selectbox("Macro Proxy", options=macro_choices, format_func=lambda k: f"{k} — {sector_profile.macro_proxies[k]}")
        selected_macro_label = sector_profile.macro_proxies[selected_macro_symbol]

        col_p, col_i = st.columns(2)
        with col_p:
            period_label = st.selectbox("History Period", options=list(PERIOD_MAP.keys()), index=3)
        with col_i:
            interval = st.selectbox("Interval", options=["1d", "1wk", "1h", "5m"], index=0)

        st.subheader("📊 Chart & Forecast")
        show_sma, show_ema, show_bb, show_sr = st.checkbox("SMA", True), st.checkbox("EMA", True), st.checkbox("Bollinger", True), st.checkbox("Support/Resistance", True)
        panel4_mode = st.selectbox("Panel 4", ["All (Stoch + ATR + OBV)", "Stochastic Only", "ATR Only", "OBV Only"])
        forecast_model = st.selectbox("Model", ["ARIMA(1,1,1)", "Holt-Winters"])
        forecast_horizon = st.slider("Forecast Horizon", 5, 60, 14)
        uploaded_flow_csv = st.file_uploader("Foreign Flow CSV (Optional)", type=["csv"])

    yf_period, period_warning = validate_period_interval(period_label, interval)
    if period_warning: st.info(period_warning)

    render_live_quote_section(active_symbol, company_name)

    # Load data for active ticker
    with st.spinner(f"Loading data for {active_symbol}..."):
        raw_df, err_msg = fetch_historical_data(active_symbol, yf_period, interval)
        klci_df, _ = fetch_historical_data("^KLSE", "6mo", "1d")
        macro_df, macro_err = fetch_historical_data(selected_macro_symbol, yf_period, "1d")

    if err_msg or raw_df.empty:
        st.error(f"❌ Data Error: {err_msg or 'Empty dataset.'}")
        st.stop()

    df = compute_technical_indicators(raw_df)
    support_lvl, resistance_lvl, _, _ = compute_support_resistance(df)
    signal_df, composite_score, _ = evaluate_signals(df, sector_profile)
    mf_info = interpret_money_flow(df)

    last_row = df.iloc[-1]
    last_close = float(last_row["Close"])
    atr_now = max(float(last_row["ATR_14"]), last_close * 0.005)

    if composite_score >= 25.0:
        verdict_label = "BULLISH BIAS"
        default_stop_loss = round(max(0.01, last_close - 2.0 * atr_now), 3)
        default_take_profit = round(max(resistance_lvl, last_close + 3.0 * atr_now), 3)
    elif composite_score <= -25.0:
        verdict_label = "BEARISH / DEFENSIVE"
        default_stop_loss = round(max(0.01, support_lvl - 1.5 * atr_now), 3)
        default_take_profit = round(last_close + 1.5 * atr_now, 3)
    else:
        verdict_label = "NEUTRAL"
        default_stop_loss = round(max(0.01, last_close - 1.5 * atr_now), 3)
        default_take_profit = round(last_close + 2.25 * atr_now, 3)

    # Tabs definition
    tab_fav_dash, tab_deep_dive, tab_pos_sizer, tab_screener = st.tabs([
        "⭐ 0. Favorites Watchlist Dashboard",
        f"📈 1. Single-Stock Deep Dive ({active_symbol})",
        f"🧮 2. Position Sizer & Fee Calculator",
        "🔍 3. Multi-Stock Sector Screener",
    ])

    # ==========================================================================
    # TAB 0: FAVORITES WATCHLIST DASHBOARD
    # ==========================================================================
    with tab_fav_dash:
        st.subheader("⭐ Starred Watchlist Executive Dashboard")
        st.markdown("Side-by-side quantitative summary of all your bookmarked Bursa Malaysia stocks.")

        if not st.session_state.watchlist:
            st.info("Your watchlist is currently empty. Click **➕ Add to Watchlist** in the sidebar while viewing any stock.")
        else:
            fav_rows = []
            for fav_sym in st.session_state.watchlist:
                f_raw, f_err = fetch_historical_data(fav_sym, "3mo", "1d")
                if f_err or f_raw.empty: continue
                f_ind = compute_technical_indicators(f_raw)
                f_prof = get_sector_profile(fav_sym, "Auto-Detect")
                _, f_score, _ = evaluate_signals(f_ind, f_prof)
                f_last = f_ind.iloc[-1]
                f_close = float(f_last["Close"])
                f_prev = float(f_ind.iloc[-2]["Close"]) if len(f_ind) >= 2 else f_close
                f_chg = ((f_close - f_prev) / f_prev) * 100
                f_verdict = "Bullish" if f_score >= 25 else "Bearish" if f_score <= -25 else "Neutral"
                fav_rows.append({
                    "Ticker": fav_sym, "Company": KLCI_COMPONENTS.get(fav_sym, fav_sym),
                    "Sector": f_prof.sector_name, "Close (RM)": round(f_close, 3), "1D Chg (%)": round(f_chg, 2),
                    "Sector Score": round(f_score, 1), "Verdict": f_verdict, "RSI (14)": round(float(f_last["RSI_14"]), 1),
                })

            if fav_rows:
                fav_df = pd.DataFrame(fav_rows)
                st.dataframe(fav_df, use_container_width=True, hide_index=True)

                st.markdown("#### ⚡ Quick Comparison Cards")
                f_cols = st.columns(min(len(fav_rows), 4))
                for idx, row in enumerate(fav_rows[:4]):
                    with f_cols[idx]:
                        st.metric(
                            label=f"{row['Ticker']} ({row['Sector'][:12]})",
                            value=f"RM {row['Close (RM)']:.3f}",
                            delta=f"{row['1D Chg (%)']:+.2f}% | Score: {row['Sector Score']:+.0f}",
                        )

    # ==========================================================================
    # TAB 1: SINGLE-STOCK DEEP DIVE
    # ==========================================================================
    with tab_deep_dive:
        with st.container(border=True):
            sc1, sc2, sc3 = st.columns([1.3, 1.7, 2.0])
            sc1.markdown(f"**Sector Profile:** `{sector_profile.sector_name}`\n\n**Regime:** {sector_profile.regime_style}")
            sc2.markdown(f"**Key Indicators:**\n`{', '.join(sector_profile.primary_indicators)}`")
            sc3.markdown(f"**Regulatory Portals:**\n" + "\n".join(f"- {p}" for p in sector_profile.reference_portals))

        news_items, news_notices = fetch_bursa_news(company_name, active_symbol, sector_profile.rss_search_terms[0], tuple(sector_profile.catalyst_keywords))
        fc_result = generate_forecast_and_backtest(df, horizon=forecast_horizon, model_type=forecast_model)

        st.subheader("🧭 7-Step Quantitative Analysis Pipeline")
        step1 = f"Bullish" if not klci_df.empty and float(klci_df["Close"].iloc[-1]) > float(klci_df["Close"].rolling(20).mean().iloc[-1]) else "Neutral"
        step2 = "Uptrend" if last_close > float(last_row["SMA_50"]) else "Downtrend"
        step3 = f"RSI {last_row['RSI_14']:.1f}"
        step4 = mf_info["state"]
        step5 = f"{sum(1 for n in news_items if n['sentiment']=='Positive')} Pos / {sum(1 for n in news_items if n['sentiment']=='Negative')} Neg"
        step6 = f"{fc_result.get('expected_return_pct', 0.0):+.2f}%" if fc_result.get("ok") else "N/A"

        p_cols = st.columns(7)
        for col, (t_txt, b_txt) in zip(p_cols, [
            ("1. Market", step1), ("2. Trend", step2), ("3. Momentum", step3),
            ("4. Money Flow", step4), ("5. News", step5), ("6. Forecast", step6),
            ("7. Verdict", f"{verdict_label} ({composite_score:+.0f})")
        ]):
            col.markdown(f"**{t_txt}**\n\n`{b_txt}`")

        st.subheader("📉 Multi-Panel Technical Chart")
        main_fig = build_main_technical_chart(df, active_symbol, show_sma, show_ema, show_bb, show_sr, panel4_mode, support_lvl, resistance_lvl)
        st.plotly_chart(main_fig, use_container_width=True)

        st.subheader(f"🌐 Sector Macro Driver: {active_symbol} vs {selected_macro_label}")
        if not macro_err and not macro_df.empty:
            m_fig, l_corr = build_macro_proxy_chart(df, macro_df, active_symbol, selected_macro_symbol, selected_macro_label)
            if m_fig:
                st.caption(f"**20-Bar Return Correlation:** `{l_corr:+.2f}`")
                st.plotly_chart(m_fig, use_container_width=True)

        st.subheader("💸 Money Flow & Participation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OBV", f"{mf_info['obv']:,.0f}")
        m2.metric("CMF (20)", f"{mf_info['cmf']:+.3f}")
        m3.metric("MFI (14)", f"{mf_info['mfi']:.1f}")
        m4.metric("Vol / 20d Avg", f"{mf_info['vol_ratio']:.2f}x")
        st.info(mf_info["summary"])

        if uploaded_flow_csv is not None:
            try:
                uf = pd.read_csv(uploaded_flow_csv)
                st.plotly_chart(build_foreign_flow_chart(df, pd.DataFrame({"Date": pd.to_datetime(uf.iloc[:, 0]), "Net_Flow": pd.to_numeric(uf.iloc[:, 1])}).dropna(), active_symbol), use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing flow CSV: {e}")

        st.subheader("🧮 Sector-Weighted Signal Summary Table")
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

        st.subheader("🔮 Statistical Forecast & Walk-Backtest")
        if fc_result.get("ok"):
            f_col, b_col = st.columns([3, 1])
            with f_col: st.plotly_chart(build_forecast_chart(df, fc_result, active_symbol), use_container_width=True)
            with b_col:
                st.metric("Target Price", f"RM {fc_result['target_price']:.3f}", f"{fc_result['expected_return_pct']:+.2f}%")
                st.metric("Walk-Forward MAPE", f"{fc_result['mape_pct']:.2f}%")
                st.metric("Directional Acc.", f"{fc_result['directional_accuracy_pct']:.1f}%")

        st.subheader("📰 News & RSS Feed")
        if news_items:
            st.dataframe(pd.DataFrame(news_items), column_config={"link": st.column_config.LinkColumn("URL")}, use_container_width=True, hide_index=True)
        else:
            st.info("No RSS articles matched.")

    # ==========================================================================
    # TAB 2: POSITION SIZER & FEE CALCULATOR
    # ==========================================================================
    with tab_pos_sizer:
        st.subheader(f"🧮 Bursa Board Lot Position Sizer & Fee Calculator — {active_symbol}")
        ps1, ps2, ps3 = st.columns(3)
        with ps1:
            acct_cap = st.number_input("Capital (RM)", 1000.0, 10_000_000.0, 25000.0, 1000.0)
            max_risk = st.slider("Max Risk (%)", 0.25, 5.0, 1.5, 0.25)
            man_lots = st.number_input("Manual Lots (0=Auto)", 0, 50000, 0, 1)
        with ps2:
            ps_entry = st.number_input("Entry Price (RM)", 0.005, 500.0, round(last_close, 3), 0.01, "%.3f")
            ps_stop = st.number_input("Stop-Loss (RM)", 0.005, 500.0, float(default_stop_loss), 0.01, "%.3f")
            ps_tp = st.number_input("Take-Profit (RM)", 0.005, 500.0, float(default_take_profit), 0.01, "%.3f")
        with ps3:
            brk_pct = st.number_input("Brokerage (%)", 0.01, 1.0, 0.10, 0.01)
            min_brk = st.number_input("Min Brokerage (RM)", 0.0, 50.0, 8.0, 1.0)
            apply_sst = st.checkbox("Apply 8% SST", True)

        psr = compute_bursa_position_size(acct_cap, max_risk, ps_entry, ps_stop, ps_tp, brk_pct, min_brk, apply_sst, manual_lots_override=man_lots if man_lots > 0 else None)
        st.divider()
        if not psr.get("ok"):
            st.error(psr.get("error"))
        else:
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Board Lots", f"{psr['active_lots']:,} Lots", f"{psr['shares']:,} sh")
            k2.metric("Total Outlay", f"RM {psr['total_buy_outlay']:,.2f}", f"{psr['capital_utilization_pct']:.1f}% cap")
            k3.metric("Breakeven", f"RM {psr['breakeven_price']:.4f}")
            k4.metric("Net Risk", f"-RM {psr['net_loss_stop']:,.2f}", f"{psr['net_risk_pct_of_capital']:.2f}% risk")
            k5.metric("Net Profit", f"+RM {psr['net_profit_tp']:,.2f}")
            k6.metric("Net R/R", f"1 : {psr['net_rr']:.2f}")

    # ==========================================================================
    # TAB 3: MULTI-STOCK SECTOR SCREENER
    # ==========================================================================
    with tab_screener:
        st.subheader("🔍 Multi-Stock Bursa Sector Screener")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            sec_filter = st.selectbox("Universe Sector", ["All Preloaded Equities (35 Stocks)"] + list(BURSA_SECTOR_PROFILES.keys()))
        with s_col2:
            min_v = st.slider("Min Vol Ratio", 0.0, 3.0, 0.0, 0.1)

        target_syms = tuple(k for k in KLCI_COMPONENTS.keys() if k != "^KLSE") if sec_filter.startswith("All") else tuple(s for s, sec in TICKER_TO_SECTOR.items() if sec == sec_filter and s in KLCI_COMPONENTS and s != "^KLSE")

        with st.spinner("Scanning equities..."):
            scr_df = run_bursa_sector_screener(target_syms)

        if not scr_df.empty:
            f_scr = scr_df[scr_df["Vol / 20d Avg"] >= min_v].copy()
            st.plotly_chart(build_screener_scatter_chart(f_scr), use_container_width=True)
            st.dataframe(f_scr, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Screener CSV", f_scr.to_csv(index=False).encode("utf-8"), "bursa_screener.csv", "text/csv")
        else:
            st.warning("No screener data available.")

    st.divider()
    st.caption("**Disclaimer:** Educational tool only. Not financial advice.")


if __name__ == "__main__":
    main()