"""
Bursa Pro Analyzer (Supabase Auth & Persistent Watchlist Edition)
Quantitative Equity, Sector Intelligence, Cloud Watchlist, Position Sizing & KLSE Workbench
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
    """Initialize Supabase client using Streamlit secrets if available."""
    try:
        if "supabase" in st.st.secrets if hasattr(st, "st") else "supabase" in st.secrets:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            return create_client(url, key)
    except Exception:
        pass
    return None

supabase_client = init_supabase()

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
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "max",
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
        macro_proxies={"ALI=F": "Aluminium Futures", "HG=F": "Copper Futures", "MYR=X": "USD/MYR Spot"},
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
    """Fetch persistent watchlist from Supabase database."""
    if not supabase_client:
        return st.session_state.get("fallback_watchlist", ["1155.KL", "1295.KL", "5347.KL", "5398.KL"])
    try:
        res = supabase_client.table("user_watchlists").select("ticker").eq("user_id", user_id).execute()
        if res.data:
            return [row["ticker"] for row in res.data]
    except Exception:
        pass
    return ["1155.KL", "1295.KL", "5347.KL", "5398.KL"]


def add_ticker_to_supabase(user_id: str, ticker: str) -> None:
    if not supabase_client:
        return
    try:
        supabase_client.table("user_watchlists").upsert({"user_id": user_id, "ticker": ticker}).execute()
    except Exception:
        pass


def remove_ticker_from_supabase(user_id: str, ticker: str) -> None:
    if not supabase_client:
        return
    try:
        supabase_client.table("user_watchlists").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
    except Exception:
        pass


def get_bursa_market_status(now_myt: Optional[datetime] = None) -> str:
    if now_myt is None: now_myt = datetime.now(MYT)
    if now_myt.weekday() >= 5: return "CLOSED (Weekend)"
    t = now_myt.time()
    if time(9, 0) <= t < time(12, 30): return "OPEN (Morning)"
    if time(12, 30) <= t < time(14, 30): return "MIDDAY BREAK"
    if time(14, 30) <= t < time(17, 0): return "OPEN (Afternoon)"
    return "CLOSED"


def normalize_bursa_ticker(raw_symbol: str) -> str:
    cleaned = raw_symbol.strip().upper()
    if not cleaned: return "1155.KL"
    if cleaned == "^KLSE" or cleaned.endswith(".KL"): return cleaned
    return f"{cleaned}.KL"


def validate_period_interval(period_label: str, interval: str) -> Tuple[str, Optional[str]]:
    yf_period = PERIOD_MAP.get(period_label, "1y")
    if interval == "5m" and yf_period != "1mo": return "1mo", "5m limited to 1M."
    if interval == "1h" and yf_period in ("2y", "5y", "max"): return "1y", "1h limited to 1Y."
    return yf_period, None


def calculate_bursa_leg_fees(contract_value: float, brokerage_pct: float = 0.10, min_brk: float = 8.00, apply_sst: bool = True) -> Dict[str, float]:
    if contract_value <= 0: return {"total_fee": 0.0}
    brk = max(min_brk, contract_value * (brokerage_pct / 100.0))
    sst = brk * 0.08 if apply_sst else 0.0
    clearing = min(1000.0, contract_value * 0.0003)
    stamp = min(1000.0, math.ceil(contract_value / 1000.0) * 1.0)
    return {"total_fee": round(brk + sst + clearing + stamp, 2)}


def compute_bursa_position_size(account_capital: float, max_risk_pct: float, entry_price: float, stop_loss: float, take_profit: float) -> Dict[str, Any]:
    risk_budget = account_capital * (max_risk_pct / 100.0)
    dist = entry_price - stop_loss
    if entry_price <= 0 or dist <= 0: return {"ok": False, "error": "Invalid stop loss."}
    max_lots = int(account_capital // (entry_price * BOARD_LOT_SIZE))
    risk_lots = int(risk_budget // (dist * BOARD_LOT_SIZE))
    optimal = max(0, min(max_lots, risk_lots))
    shares = optimal * BOARD_LOT_SIZE
    if shares <= 0: return {"ok": False, "error": "Risk budget too small for 1 lot."}
    buy_val = shares * entry_price
    stop_val = shares * stop_loss
    tp_val = shares * take_profit
    f_buy = calculate_bursa_leg_fees(buy_val)["total_fee"]
    f_stop = calculate_bursa_leg_fees(stop_val)["total_fee"]
    f_tp = calculate_bursa_leg_fees(tp_val)["total_fee"]
    net_loss = (buy_val - stop_val) + f_buy + f_stop
    net_profit = (tp_val - buy_val) - f_buy - f_tp
    return {
        "ok": True, "active_lots": optimal, "shares": shares, "total_buy_outlay": buy_val + f_buy,
        "net_loss_stop": net_loss, "net_profit_tp": net_profit, "net_rr": (net_profit / net_loss) if net_loss > 0 else 0,
    }


# ==============================================================================
# SECTION 3: DATA FETCHING & TECHNICALS
# ==============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(symbol: str, yf_period: str, interval: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        df = yf.Ticker(symbol).history(period=yf_period, interval=interval, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame(), f"No data for {symbol}."
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        req = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in req): return pd.DataFrame(), f"Missing columns."
        df = df[req].dropna().copy()
        df["Volume"] = df["Volume"].fillna(0).astype(float)
        df.index = df.index.tz_convert(MYT) if df.index.tz else pd.to_datetime(df.index)
        return df.sort_index(), None
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(symbol: str) -> Dict[str, Any]:
    try:
        rec = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        if rec is None or rec.empty: return {"ok": False}
        if isinstance(rec.columns, pd.MultiIndex): rec.columns = [c[0] for c in rec.columns]
        last = rec.dropna(subset=["Close"]).iloc[-1]
        prev = float(rec.iloc[-2]["Close"]) if len(rec) >= 2 else float(last["Open"])
        close = float(last["Close"])
        return {"ok": True, "last_price": close, "change": close - prev, "pct_change": ((close - prev) / prev) * 100, "fetched_at": datetime.now(MYT).strftime("%H:%M:%S MYT")}
    except Exception:
        return {"ok": False}


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low, vol = out["Close"].astype(float), out["High"].astype(float), out["Low"].astype(float), out["Volume"].astype(float)
    out["SMA_20"] = close.rolling(20, min_periods=1).mean()
    out["SMA_50"] = close.rolling(50, min_periods=1).mean()
    out["SMA_200"] = close.rolling(200, min_periods=1).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    bb_std = close.rolling(20, min_periods=2).std(ddof=0)
    out["BB_Mid"], out["BB_Upper"], out["BB_Lower"] = out["SMA_20"], out["SMA_20"] + 2 * bb_std, out["SMA_20"] - 2 * bb_std
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean().replace(0, np.nan)
    out["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["ATR_14"] = tr.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    out["OBV"] = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    out["OBV_SMA_20"] = out["OBV"].rolling(20, min_periods=1).mean()
    hl = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / hl).fillna(0)
    out["CMF_20"] = (mfm * vol).rolling(20, min_periods=1).sum() / vol.rolling(20, min_periods=1).sum().replace(0, np.nan)
    out["CMF_20"] = out["CMF_20"].fillna(0)
    tp = (high + low + close) / 3
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, tp * vol, 0), index=out.index).rolling(14, min_periods=1).sum()
    neg_mf = pd.Series(np.where(tp_diff < 0, tp * vol, 0), index=out.index).rolling(14, min_periods=1).sum().replace(0, np.nan)
    out["MFI_14"] = (100 - (100 / (1 + (pos_mf / neg_mf)))).fillna(50)
    out["Vol_SMA_20"] = vol.rolling(20, min_periods=1).mean()
    return out


def compute_support_resistance(df: pd.DataFrame, window: int = 5) -> Tuple[float, float]:
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    sh, sl = [], []
    if n >= (2 * window + 1):
        for i in range(window, n - window):
            if highs[i] == max(highs[i - window : i + window + 1]): sh.append(float(highs[i]))
            if lows[i] == min(lows[i - window : i + window + 1]): sl.append(float(lows[i]))
    close = float(df["Close"].iloc[-1])
    supp = max([l for l in sl if l < close], default=float(df["Low"].min()))
    res = min([h for h in sh if h > close], default=float(df["High"].max()))
    return supp, res


def evaluate_signals(df: pd.DataFrame, sec: SectorProfile) -> Tuple[pd.DataFrame, float]:
    last = df.iloc[-1]
    close = float(last["Close"])
    w = sec.indicator_weights
    signals = []

    def add_sig(cat, ind, val, cond_b, cond_bear, rationale):
        sig = "Bullish" if cond_b else "Bearish" if cond_bear else "Neutral"
        score = 1 if sig == "Bullish" else -1 if sig == "Bearish" else 0
        signals.append(IndicatorSignal(cat, ind, val, sig, score, w.get(ind, 5.0), rationale))

    add_sig("Trend", "SMA (20)", f"{close:.3f}", close > last["SMA_20"], close < last["SMA_20"], "Price vs SMA20")
    add_sig("Trend", "SMA (50)", f"{close:.3f}", close > last["SMA_50"], close < last["SMA_50"], "Price vs SMA50")
    add_sig("Trend", "SMA (200)", f"{close:.3f}", close > last["SMA_200"], close < last["SMA_200"], "Price vs SMA200")
    add_sig("Trend", "EMA (12/26)", f"{last['EMA_12']:.3f}", last["EMA_12"] > last["EMA_26"], last["EMA_12"] < last["EMA_26"], "EMA crossover")
    add_sig("Volatility", "Bollinger Bands (20,2)", f"{close:.3f}", close <= last["BB_Lower"], close >= last["BB_Upper"], "Bollinger band limits")
    rsi = float(last["RSI_14"])
    add_sig("Momentum", "RSI (14)", f"{rsi:.1f}", rsi < 30 or (rsi >= 55 and rsi <= 70), rsi > 70 or rsi < 45, "RSI zone")
    add_sig("Momentum", "MACD (12,26,9)", f"{last['MACD_Hist']:.4f}", last["MACD"] > last["MACD_Signal"], last["MACD"] < last["MACD_Signal"], "MACD histogram")
    add_sig("Money Flow", "OBV", f"{last['OBV']:,.0f}", last["OBV"] > last["OBV_SMA_20"], last["OBV"] < last["OBV_SMA_20"], "OBV trend")
    cmf = float(last["CMF_20"])
    add_sig("Money Flow", "Chaikin Money Flow (20)", f"{cmf:.3f}", cmf > 0.05, cmf < -0.05, "CMF pressure")
    mfi = float(last["MFI_14"])
    add_sig("Money Flow", "Money Flow Index (14)", f"{mfi:.1f}", mfi < 20 or mfi > 55, mfi > 80 or mfi < 45, "MFI zone")

    comp = max(-100.0, min(100.0, float(sum(s.raw_score * s.weight for s in signals))))
    rows = [{"Category": s.category, "Indicator": s.indicator, "Reading": s.reading, "Signal": s.signal, "Weight": s.weight, "Score": s.raw_score * s.weight, "Note": s.rationale} for s in signals]
    return pd.DataFrame(rows), comp


def interpret_money_flow(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    vol, avg = float(last["Volume"]), float(last["Vol_SMA_20"])
    ratio = (vol / avg) if avg > 0 else 0
    cmf, mfi = float(last["CMF_20"]), float(last["MFI_14"])
    state = "Accumulation" if cmf > 0.05 and mfi > 55 else "Distribution" if cmf < -0.05 and mfi < 45 else "Neutral"
    return {"state": state, "vol_ratio": ratio, "summary": f"{state} (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={ratio:.2f}x)"}


# ==============================================================================
# SECTION 4: STREAMLIT APP & AUTHENTICATION UI
# ==============================================================================

def main() -> None:
    st.set_page_config(page_title="Bursa Pro Analyzer | Cloud Edition", page_icon="📈", layout="wide")

    # Session State Setup
    if "user" not in st.session_state:
        st.session_state.user = None
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["1155.KL", "1295.KL", "5347.KL", "5398.KL"]

    st.title("📈 Bursa Pro Analyzer — Cloud Workbench & Supabase Watchlist")

    # Sidebar Authentication & Controls
    with st.sidebar:
        st.header("🔐 User Authentication")
        if not st.session_state.user:
            auth_mode = st.radio("Mode", ["Sign In", "Sign Up"], horizontal=True)
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button(auth_mode, use_container_width=True):
                if supabase_client:
                    try:
                        if auth_mode == "Sign In":
                            res = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
                        else:
                            res = supabase_client.auth.sign_up({"email": email, "password": password})
                        if res.user:
                            st.session_state.user = res.user
                            st.session_state.watchlist = fetch_user_watchlist(res.user.id)
                            st.success("Successfully logged in!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Auth error: {e}")
                else:
                    # Mock auth fallback if Supabase keys aren't configured yet
                    if email:
                        st.session_state.user = type("User", (), {"id": "mock_user", "email": email})()
                        st.success("Logged in with mock user!")
                        st.rerun()
            st.caption("💡 Configure Supabase URL & Key in secrets to enable cloud sync.")
        else:
            st.success(f"Logged in as: {st.session_state.user.email}")
            if st.button("Log Out", use_container_width=True):
                st.session_state.user = None
                st.rerun()

        st.divider()
        st.header("⚙️ Ticker Selection")
        preset = [f"{c} — {n}" for c, n in KLCI_COMPONENTS.items()]
        sel_preset = st.selectbox("KLCI Constituent", options=preset)
        custom = st.text_input("Or Custom Ticker Code", placeholder="e.g. 0166")
        
        active_symbol = normalize_bursa_ticker(custom) if custom.strip() else sel_preset.split(" — ")[0].strip()
        comp_name = KLCI_COMPONENTS.get(active_symbol, active_symbol)

        # Watchlist toggle
        if st.session_state.user:
            user_id = getattr(st.session_state.user, "id", "mock_user")
            is_fav = active_symbol in st.session_state.watchlist
            if is_fav:
                if st.button("⭐ In Watchlist", use_container_width=True):
                    st.session_state.watchlist.remove(active_symbol)
                    remove_ticker_from_supabase(user_id, active_symbol)
                    st.rerun()
            else:
                if st.button("➕ Add to Watchlist", use_container_width=True):
                    if active_symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(active_symbol)
                    add_ticker_to_supabase(user_id, active_symbol)
                    st.rerun()

        sector_prof = get_sector_profile(active_symbol)
        period_label = st.selectbox("Period", list(PERIOD_MAP.keys()), index=3)
        interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)

    yf_per, _ = validate_period_interval(period_label, interval)

    # Load Data
    with st.spinner(f"Loading data for {active_symbol}..."):
        raw_df, err = fetch_historical_data(active_symbol, yf_per, interval)

    if err or raw_df.empty:
        st.error(f"❌ {err or 'Empty dataset.'}")
        st.stop()

    df = compute_technical_indicators(raw_df)
    supp, res = compute_support_resistance(df)
    signal_df, comp_score = evaluate_signals(df, sector_prof)
    mf = interpret_money_flow(df)

    last_close = float(df["Close"].iloc[-1])
    atr = float(df["ATR_14"].iloc[-1])

    # Main Tabs Layout
    tab_fav, tab_chart, tab_sizer = st.tabs(["⭐ Watchlist Dashboard", "📈 Technical Analysis", "🧮 Position Sizer"])

    with tab_fav:
        st.subheader("⭐ Personal Cloud Watchlist")
        if not st.session_state.watchlist:
            st.info("Your watchlist is empty.")
        else:
            rows = []
            for sym in st.session_state.watchlist:
                r_df, r_err = fetch_historical_data(sym, "3mo", "1d")
                if r_err or r_df.empty: continue
                r_ind = compute_technical_indicators(r_df)
                r_sec = get_sector_profile(sym)
                _, r_score = evaluate_signals(r_ind, r_sec)
                c_now = float(r_ind["Close"].iloc[-1])
                rows.append({"Ticker": sym, "Company": KLCI_COMPONENTS.get(sym, sym), "Close (RM)": round(c_now, 3), "Score": round(r_score, 1), "RSI": round(float(r_ind["RSI_14"].iloc[-1]), 1)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_chart:
        st.subheader(f"Analysis for {active_symbol} ({comp_score:+.0f}/100)")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

    with tab_sizer:
        st.subheader("Bursa Board Lot Position Sizer")
        cap = st.number_input("Capital (RM)", value=25000.0)
        risk = st.slider("Risk (%)", 0.5, 3.0, 1.5)
        stop = st.number_input("Stop Loss", value=round(last_close - 2 * atr, 3))
        tp = st.number_input("Take Profit", value=round(last_close + 3 * atr, 3))
        
        res_calc = compute_bursa_position_size(cap, risk, last_close, stop, tp)
        if res_calc.get("ok"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal Lots", f"{res_calc['active_lots']} Lots")
            c2.metric("Total Outlay", f"RM {res_calc['total_buy_outlay']:,.2f}")
            c3.metric("Net R/R", f"1 : {res_calc['net_rr']:.2f}")
        else:
            st.warning(res_calc.get("error"))

    st.divider()
    st.caption("**Disclaimer:** Educational tool only. Not financial advice.")

if __name__ == "__main__":
    main()