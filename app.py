"""
Bursa Pro Analyzer (Ultimate Edition)
Quantitative Equity, Sector Intelligence, Supabase Watchlist, Position Sizing & Sector Screener.
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
from supabase import create_client, Client
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: SUPABASE CLIENT & CONFIGURATION
# ==============================================================================

MYT = ZoneInfo("Asia/Kuala_Lumpur")
BOARD_LOT_SIZE = 100
CURRENCY_SYMBOL = "RM"


@st.cache_resource
def init_supabase() -> Optional[Client]:
    """Initialize Supabase client using Streamlit secrets."""
    try:
        if hasattr(st, "secrets") and "supabase" in st.secrets:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            return create_client(url, key)
    except Exception as e:
        print(f"Supabase client initialization error: {e}")
    return None


supabase_client = init_supabase()

KLCI_COMPONENTS: Dict[str, str] = {
    "1155.KL": "Malayan Banking (Maybank)", "1295.KL": "Public Bank", "1023.KL": "CIMB Group",
    "5347.KL": "Tenaga Nasional", "5225.KL": "IHH Healthcare", "5183.KL": "PETRONAS Chemicals",
    "6947.KL": "CelcomDigi", "8869.KL": "Press Metal", "5819.KL": "Hong Leong Bank",
    "1066.KL": "RHB Bank", "5681.KL": "PETRONAS Dagangan", "6033.KL": "PETRONAS Gas",
    "6012.KL": "Maxis", "4863.KL": "Telekom Malaysia", "3816.KL": "MISC",
    "2445.KL": "Kuala Lumpur Kepong", "5285.KL": "SD Guthrie", "4197.KL": "Sime Darby",
    "1961.KL": "IOI Corporation", "4707.KL": "Nestle (Malaysia)", "7084.KL": "QL Resources",
    "6742.KL": "YTL Power", "4677.KL": "YTL Corporation", "5296.KL": "MR D.I.Y. Group",
    "1082.KL": "Hong Leong Financial", "3182.KL": "Genting", "4715.KL": "Genting Malaysia",
    "4065.KL": "PPB Group", "5398.KL": "Gamuda", "5211.KL": "Sunway",
    "0166.KL": "Inari Amertron (Tech)", "3867.KL": "MPI (Tech)", "7277.KL": "Dialog Group (Energy)",
    "3336.KL": "IJM Corp (Construction)", "7113.KL": "Top Glove (Healthcare)", "^KLSE": "FBM KLCI Index",
}

PERIOD_MAP: Dict[str, str] = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "max",
}

POSITIVE_KEYWORDS = {"surge", "jump", "gain", "profit", "growth", "dividend", "upgrade", "buy", "outperform", "bullish", "record", "higher", "strong", "beat", "expansion", "contract", "award", "rally", "positive"}
NEGATIVE_KEYWORDS = {"drop", "fall", "loss", "decline", "downgrade", "sell", "underperform", "bearish", "lower", "weak", "miss", "slump", "plunge", "cut", "lawsuit", "fine", "penalty", "negative", "concern", "risk"}


@dataclass(frozen=True)
class SectorProfile:
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
        sector_name="Financial Services", regime_style="Mean-Reverting & Accumulation",
        primary_indicators=["Bollinger Bands (20,2)", "RSI (14)", "SMA (200)", "CMF (20)"],
        macro_proxies={"MYR=X": "USD/MYR Spot", "^KLSE": "FBM KLCI Index", "^TNX": "US 10Y Yield"},
        indicator_weights={"SMA (20)": 8.0, "SMA (50)": 12.0, "SMA (200)": 16.0, "EMA (12/26)": 6.0, "Bollinger Bands (20,2)": 15.0, "RSI (14)": 15.0, "MACD (12,26,9)": 8.0, "Stochastic (14,3,3)": 8.0, "OBV": 3.0, "Chaikin Money Flow (20)": 6.0, "Money Flow Index (14)": 3.0},
        rss_search_terms=["Bank Negara Malaysia OPR banking", "Bursa Malaysia banking NIM loan growth"],
        catalyst_keywords=["opr", "nim", "dividend", "loan", "bnm", "casa", "provision", "capital"],
        reference_portals=["Bank Negara Malaysia (bnm.gov.my)", "The Edge Malaysia — Banking"],
    ),
    "Plantation": SectorProfile(
        sector_name="Plantation", regime_style="Commodity-Cyclical (FCPO Driven)",
        primary_indicators=["MACD (12,26,9)", "EMA (12/26)", "OBV", "RSI (14)"],
        macro_proxies={"ZL=F": "Soybean Oil Futures (FCPO Proxy)", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 10.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 14.0, "Bollinger Bands (20,2)": 6.0, "RSI (14)": 12.0, "MACD (12,26,9)": 16.0, "Stochastic (14,3,3)": 6.0, "OBV": 6.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0},
        rss_search_terms=["MPOB palm oil stockpile export Malaysia", "crude palm oil FCPO Bursa Malaysia"],
        catalyst_keywords=["fcpo", "mpob", "cpo", "stockpile", "el nino", "la nina", "export", "biodiesel", "ffb"],
        reference_portals=["Malaysian Palm Oil Board (mpob.gov.my)", "Intertek & Amspec Export Surveys"],
    ),
    "Energy & Petrochemicals": SectorProfile(
        sector_name="Energy & Petrochemicals", regime_style="High-Beta Trend (Crude Correlated)",
        primary_indicators=["EMA (12/26)", "MACD (12,26,9)", "Stochastic (14,3,3)", "OBV"],
        macro_proxies={"BZ=F": "Brent Crude Oil", "CL=F": "WTI Crude", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 10.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 15.0, "Bollinger Bands (20,2)": 5.0, "RSI (14)": 10.0, "MACD (12,26,9)": 16.0, "Stochastic (14,3,3)": 8.0, "OBV": 6.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0},
        rss_search_terms=["PETRONAS contract oil gas Bursa Malaysia", "Brent crude oil OPEC Malaysia energy"],
        catalyst_keywords=["petronas", "fpso", "brent", "opec", "charter", "capex", "upstream", "rig"],
        reference_portals=["PETRONAS Activity Outlook (PAO)", "Bursa Malaysia Energy Announcements"],
    ),
    "Technology & EMS": SectorProfile(
        sector_name="Technology & EMS", regime_style="High-Momentum Breakout (SOX Correlated)",
        primary_indicators=["EMA (12/26)", "MACD (12,26,9)", "RSI (14)", "Money Flow Index (14)"],
        macro_proxies={"^SOX": "PHLX Semiconductor Index", "^IXIC": "Nasdaq Composite", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 12.0, "SMA (50)": 10.0, "SMA (200)": 8.0, "EMA (12/26)": 16.0, "Bollinger Bands (20,2)": 6.0, "RSI (14)": 14.0, "MACD (12,26,9)": 16.0, "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0},
        rss_search_terms=["Malaysia semiconductor OSAT Penang technology Bursa", "MIDA NIMP semiconductor AI server Malaysia"],
        catalyst_keywords=["osat", "wafer", "semiconductor", "apple", "broadcom", "utilization", "penang", "kulim"],
        reference_portals=["Semiconductor Industry Association (semiconductors.org)", "MIDA NIMP Updates"],
    ),
    "Construction, Property & Infrastructure": SectorProfile(
        sector_name="Construction, Property & Infrastructure", regime_style="Orderbook Catalyst & Volume Breakout",
        primary_indicators=["OBV", "Money Flow Index (14)", "SMA (20/50)", "MACD (12,26,9)"],
        macro_proxies={"^KLSE": "FBM KLCI Index", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 12.0, "SMA (50)": 12.0, "SMA (200)": 10.0, "EMA (12/26)": 10.0, "Bollinger Bands (20,2)": 6.0, "RSI (14)": 10.0, "MACD (12,26,9)": 12.0, "Stochastic (14,3,3)": 6.0, "OBV": 10.0, "Chaikin Money Flow (20)": 6.0, "Money Flow Index (14)": 6.0},
        rss_search_terms=["Bursa Malaysia construction contract award data centre", "MRT3 ECRL Johor Singapore SEZ property Malaysia"],
        catalyst_keywords=["letter of award", "order book", "tender", "mrt3", "ecrl", "data centre", "sez", "gdv"],
        reference_portals=["Bursa Malaysia Contracts Awarded", "Ministry of Works (KKR) Briefings"],
    ),
    "Utilities & Telecommunications": SectorProfile(
        sector_name="Utilities & Telecommunications", regime_style="Defensive Yield + Data Centre Demand",
        primary_indicators=["SMA (50/200)", "Bollinger Bands (20,2)", "CMF (20)", "RSI (14)"],
        macro_proxies={"^TNX": "US 10Y Yield", "^KLSE": "FBM KLCI Index", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 10.0, "SMA (50)": 14.0, "SMA (200)": 16.0, "EMA (12/26)": 8.0, "Bollinger Bands (20,2)": 12.0, "RSI (14)": 12.0, "MACD (12,26,9)": 10.0, "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 5.0, "Money Flow Index (14)": 3.0},
        rss_search_terms=["Tenaga YTL Power data centre grid tariff Malaysia", "MCMC 5G spectrum telecom Bursa Malaysia"],
        catalyst_keywords=["icpt", "tariff", "grid", "renewable", "solar", "lss", "5g", "spectrum", "hyperscale"],
        reference_portals=["Energy Commission (st.gov.my)", "MCMC Policy Releases"],
    ),
    "Industrial & Materials": SectorProfile(
        sector_name="Industrial & Materials", regime_style="Global Manufacturing & LME Cyclical",
        primary_indicators=["MACD (12,26,9)", "SMA (50)", "EMA (12/26)", "RSI (14)"],
        macro_proxies={"ALI=F": "Aluminium Futures (LME Proxy)", "HG=F": "Copper Futures", "MYR=X": "USD/MYR Spot"},
        indicator_weights={"SMA (20)": 10.0, "SMA (50)": 14.0, "SMA (200)": 12.0, "EMA (12/26)": 12.0, "Bollinger Bands (20,2)": 8.0, "RSI (14)": 12.0, "MACD (12,26,9)": 14.0, "Stochastic (14,3,3)": 6.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0},
        rss_search_terms=["Press Metal aluminium LME industrial Bursa Malaysia", "Malaysia manufacturing PMI export industrial"],
        catalyst_keywords=["lme", "aluminium", "smelter", "alumina", "pmi", "freight", "tariff"],
        reference_portals=["London Metal Exchange (lme.com)", "S&P Global Malaysia PMI"],
    ),
    "Consumer & Healthcare": SectorProfile(
        sector_name="Consumer & Healthcare", regime_style="Domestic Consumption & Defensive",
        primary_indicators=["SMA (20/50)", "RSI (14)", "Stochastic (14,3,3)", "MFI (14)"],
        macro_proxies={"MYR=X": "USD/MYR (Import Cost Proxy)", "^KLSE": "FBM KLCI Index"},
        indicator_weights={"SMA (20)": 12.0, "SMA (50)": 12.0, "SMA (200)": 12.0, "EMA (12/26)": 8.0, "Bollinger Bands (20,2)": 10.0, "RSI (14)": 14.0, "MACD (12,26,9)": 10.0, "Stochastic (14,3,3)": 10.0, "OBV": 4.0, "Chaikin Money Flow (20)": 4.0, "Money Flow Index (14)": 4.0},
        rss_search_terms=["Malaysia retail consumer F&B healthcare Bursa", "IHH Healthcare Nestle MR DIY QL Resources Malaysia"],
        catalyst_keywords=["subsidy", "sara", "str", "epf", "wage", "footfall", "asp", "hospital", "glove"],
        reference_portals=["Department of Statistics Malaysia (dosm.gov.my)", "Ministry of Health Briefings"],
    ),
}

TICKER_TO_SECTOR: Dict[str, str] = {
    "1155.KL": "Financial Services", "1295.KL": "Financial Services", "1023.KL": "Financial Services",
    "5819.KL": "Financial Services", "1066.KL": "Financial Services", "1082.KL": "Financial Services",
    "5285.KL": "Plantation", "1961.KL": "Plantation", "2445.KL": "Plantation",
    "5183.KL": "Energy & Petrochemicals", "5681.KL": "Energy & Petrochemicals", "3816.KL": "Energy & Petrochemicals",
    "5347.KL": "Utilities & Telecommunications", "6742.KL": "Utilities & Telecommunications", "6947.KL": "Utilities & Telecommunications",
    "5398.KL": "Construction, Property & Infrastructure", "5211.KL": "Construction, Property & Infrastructure",
    "8869.KL": "Industrial & Materials", "0166.KL": "Technology & EMS", "3867.KL": "Technology & EMS",
    "5225.KL": "Consumer & Healthcare", "4707.KL": "Consumer & Healthcare", "7084.KL": "Consumer & Healthcare", "^KLSE": "Financial Services",
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
# SECTION 2: SUPABASE WATCHLIST SYNC & POSITION SIZING
# ==============================================================================

def fetch_user_watchlist(user_id: str) -> List[str]:
    if not supabase_client:
        return st.session_state.get("fallback_watchlist", ["1155.KL", "1295.KL", "5347.KL", "5398.KL"])
    try:
        res = supabase_client.table("user_watchlists").select("ticker").eq("user_id", user_id).execute()
        if res.data:
            return [row["ticker"] for row in res.data]
    except Exception as e:
        print(f"Watchlist fetch error: {e}")
    return ["1155.KL", "1295.KL", "5347.KL", "5398.KL"]


def add_ticker_to_supabase(user_id: str, ticker: str) -> None:
    if not supabase_client:
        return
    try:
        supabase_client.table("user_watchlists").upsert({"user_id": user_id, "ticker": ticker}).execute()
    except Exception as e:
        print(f"Watchlist insert error: {e}")


def remove_ticker_from_supabase(user_id: str, ticker: str) -> None:
    if not supabase_client:
        return
    try:
        supabase_client.table("user_watchlists").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
    except Exception as e:
        print(f"Watchlist delete error: {e}")


def get_bursa_market_status(now_myt: Optional[datetime] = None) -> str:
    if now_myt is None:
        now_myt = datetime.now(MYT)
    if now_myt.weekday() >= 5:
        return "CLOSED (Weekend)"
    t = now_myt.time()
    if time(9, 0) <= t < time(12,