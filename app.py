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
    if time(9, 0) <= t < time(12, 30):
        return "OPEN (Morning)"
    if time(12, 30) <= t < time(14, 30):
        return "MIDDAY BREAK"
    if time(14, 30) <= t < time(17, 0):
        return "OPEN (Afternoon)"
    return "CLOSED"


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
    if interval == "5m" and yf_period != "1mo":
        return "1mo", "5m limited to 1M."
    if interval == "1h" and yf_period in ("2y", "5y", "max"):
        return "1y", "1h limited to 1Y."
    return yf_period, None


def calculate_bursa_leg_fees(contract_value: float, brokerage_pct: float = 0.10, min_brk: float = 8.00, apply_sst: bool = True) -> Dict[str, float]:
    if contract_value <= 0:
        return {"total_fee": 0.0}
    brk = max(min_brk, contract_value * (brokerage_pct / 100.0))
    sst = brk * 0.08 if apply_sst else 0.0
    clearing = min(1000.0, contract_value * 0.0003)
    stamp = min(1000.0, math.ceil(contract_value / 1000.0) * 1.0)
    return {"total_fee": round(brk + sst + clearing + stamp, 2)}


def compute_bursa_position_size(account_capital: float, max_risk_pct: float, entry_price: float, stop_loss: float, take_profit: float) -> Dict[str, Any]:
    risk_budget = account_capital * (max_risk_pct / 100.0)
    dist = entry_price - stop_loss
    if entry_price <= 0 or dist <= 0:
        return {"ok": False, "error": "Invalid stop loss. Entry must exceed Stop-Loss."}
    max_lots = int(account_capital // (entry_price * BOARD_LOT_SIZE))
    risk_lots = int(risk_budget // (dist * BOARD_LOT_SIZE))
    optimal = max(0, min(max_lots, risk_lots))
    shares = optimal * BOARD_LOT_SIZE
    if shares <= 0:
        return {"ok": False, "error": "Risk budget too small for 1 board lot."}
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
        "breakeven_price": (buy_val + f_buy + f_stop) / shares
    }

# ==============================================================================
# SECTION 3: DATA FETCHING & TECHNICALS
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(symbol: str, yf_period: str, interval: str) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        df = yf.Ticker(symbol).history(period=yf_period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(), f"No data for {symbol}."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        req = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in req):
            return pd.DataFrame(), "Missing OHLCV columns."
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
        if rec is None or rec.empty:
            return {"ok": False}
        if isinstance(rec.columns, pd.MultiIndex):
            rec.columns = [c[0] for c in rec.columns]
        last = rec.dropna(subset=["Close"]).iloc[-1]
        prev = float(rec.iloc[-2]["Close"]) if len(rec) >= 2 else float(last["Open"])
        close = float(last["Close"])
        return {
            "ok": True, "last_price": close, "prev_close": prev, "change": close - prev,
            "pct_change": ((close - prev) / prev) * 100, "day_high": float(last["High"]),
            "day_low": float(last["Low"]), "volume": float(last.get("Volume", 0.0)),
            "lot_cost": close * BOARD_LOT_SIZE, "fetched_at": datetime.now(MYT).strftime("%H:%M:%S MYT")
        }
    except Exception:
        return {"ok": False}


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float)

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
    
    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    out["Stoch_K"] = (100 * (close - low_14) / denom).rolling(3, min_periods=1).mean().fillna(50)
    out["Stoch_D"] = out["Stoch_K"].rolling(3, min_periods=1).mean().fillna(50)
    return out


def compute_support_resistance(df: pd.DataFrame, window: int = 5) -> Tuple[float, float]:
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    sh, sl = [], []
    if n >= (2 * window + 1):
        for i in range(window, n - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                sh.append(float(highs[i]))
            if lows[i] == min(lows[i - window : i + window + 1]):
                sl.append(float(lows[i]))
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
    st_k, st_d = float(last["Stoch_K"]), float(last["Stoch_D"])
    add_sig("Momentum", "Stochastic (14,3,3)", f"%K:{st_k:.1f}", (st_k < 20 and st_k >= st_d) or (st_k > st_d and st_k < 80), (st_k > 80 and st_k <= st_d) or (st_k < st_d and st_k > 20), "Stoch crossover")
    
    has_vol = float(df["Volume"].tail(20).sum()) > 0
    add_sig("Money Flow", "OBV", f"{last['OBV']:,.0f}", has_vol and last["OBV"] > last["OBV_SMA_20"], has_vol and last["OBV"] < last["OBV_SMA_20"], "OBV trend")
    cmf = float(last["CMF_20"])
    add_sig("Money Flow", "Chaikin Money Flow (20)", f"{cmf:.3f}", has_vol and cmf > 0.05, has_vol and cmf < -0.05, "CMF pressure")
    mfi = float(last["MFI_14"])
    add_sig("Money Flow", "Money Flow Index (14)", f"{mfi:.1f}", has_vol and (mfi < 20 or mfi >= 55), has_vol and (mfi > 80 or mfi <= 45), "MFI zone")

    comp = max(-100.0, min(100.0, float(sum(s.raw_score * s.weight for s in signals))))
    rows = [{"Category": s.category, "Indicator": s.indicator, "Reading": s.reading, "Signal": s.signal, "Weight": s.weight, "Score": s.raw_score * s.weight, "Note": s.rationale} for s in signals]
    return pd.DataFrame(rows), comp


def interpret_money_flow(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    vol, avg = float(last["Volume"]), float(last["Vol_SMA_20"])
    ratio = (vol / avg) if avg > 0 else 0
    cmf, mfi, obv, obv_sma = float(last["CMF_20"]), float(last["MFI_14"]), float(last["OBV"]), float(last["OBV_SMA_20"])

    if avg == 0:
        return {"obv": obv, "cmf": cmf, "mfi": mfi, "vol_ratio": 0.0, "state": "Neutral", "summary": "No volume reported."}
    score = 0
    if obv > obv_sma:
        score += 1
    elif obv < obv_sma:
        score -= 1
    if cmf > 0.05:
        score += 1
    elif cmf < -0.05:
        score -= 1
    if mfi >= 55:
        score += 1
    elif mfi <= 45:
        score -= 1

    if score >= 2:
        state, summary = "Accumulation", f"Accumulation — Institutional buying (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={ratio:.2f}x)."
    elif score <= -2:
        state, summary = "Distribution", f"Distribution — Net selling dominates (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={ratio:.2f}x)."
    else:
        state, summary = "Neutral", f"Neutral — Mixed flow signals (CMF={cmf:+.2f}, MFI={mfi:.1f}, Vol={ratio:.2f}x)."

    return {"obv": obv, "cmf": cmf, "mfi": mfi, "vol_ratio": ratio, "state": state, "summary": summary}


@st.cache_data(ttl=900, show_spinner=False)
def run_bursa_sector_screener(symbols: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        if sym == "^KLSE":
            continue
        raw_df, err = fetch_historical_data(sym, "6mo", "1d")
        if err or raw_df.empty or len(raw_df) < 20:
            continue
        ind_df = compute_technical_indicators(raw_df)
        sec_prof = get_sector_profile(sym, "Auto-Detect")
        _, comp_score = evaluate_signals(ind_df, sec_prof)
        mf = interpret_money_flow(ind_df)
        sup_lvl, res_lvl = compute_support_resistance(ind_df)
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
    if net > 0:
        return "Positive", net, matched
    if net < 0:
        return "Negative", net, matched
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
            if not title or not link:
                continue
            source_title = entry.get("source", {}).get("title", source_fallback) if isinstance(entry.get("source"), dict) else source_fallback
            pub_str = str(entry.get("published", entry.get("updated", "")))
            pub_myt = "Unknown time"
            if pub_str:
                try:
                    dt = parsedate_to_datetime(pub_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    pub_myt = dt.astimezone(MYT).strftime("%Y-%m-%d %H:%M")
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
        return {"ok": False, "error": "At least 35 bars required for forecasting."}
    
    values = close_series.values
    dates = close_series.index
    median_delta = pd.Series(dates).diff().median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        median_delta = pd.Timedelta(days=1)
        
    future_dates, curr_dt = [], dates[-1]
    while len(future_dates) < horizon:
        curr_dt += median_delta
        if median_delta >= pd.Timedelta(days=1) and curr_dt.weekday() >= 5:
            continue
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
            "ok": True,
            "model_type": model_type,
            "horizon": horizon,
            "forecast_df": fc_df,
            "target_price": float(mean_fc[-1]),
            "expected_return_pct": float((mean_fc[-1] - values[-1]) / values[-1] * 100.0),
            "mape_pct": mape,
            "directional_accuracy_pct": dir_acc,
            "bt_steps": bt_steps,
            "bt_dates": dates[train_start:],
            "bt_preds": preds,
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
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV_SMA_20"], name="OBV SMA 20", line=dict(color="#FFEE58", width=1.3, dash="dot")), row=4, col=1, secondary_y=False)

    fig.update_yaxes(title_text=f"Price ({CURRENCY_SYMBOL})", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_layout(template="plotly_dark", height=920, margin=dict(l=45, r=45, t=45, b=35), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.01, x=0), hovermode="x unified")
    return fig


def build_macro_proxy_chart(stock_df: pd.DataFrame, macro_df: pd.DataFrame, symbol: str, macro_symbol: str, macro_label: str) -> Tuple[Optional[go.Figure], Optional[float]]:
    s_close, m_close = stock_df["Close"].copy(), macro_df["Close"].copy()
    s_close.index, m_close.index = pd.to_datetime(s_close.index).tz_localize(None).normalize(), pd.to_datetime(m_close.index).tz_localize(None).normalize()
    aligned = pd.DataFrame({"Stock": s_close.groupby(s_close.index).last(), "Macro": m_close.groupby(m_close.index).last()}).dropna()
    if len(aligned) < 15:
        return None, None
    aligned["Rolling_Corr_20"] = aligned["Stock"].pct_change().rolling(window=20, min_periods=10).corr(aligned["Macro"].pct_change()).fillna(0.0)
    latest_corr = float(aligned["Rolling_Corr_20"].iloc[-1])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.68, 0.32], specs=[[{"secondary_y": True}], [{"secondary_y": False}]], subplot_titles=(f"{symbol} vs {macro_label} ({macro_symbol})", "20-Bar Rolling Return Correlation"))
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Stock"], name=f"{symbol}", line=dict(color="#00E676", width=2)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Macro"], name=f"{macro_label}", line=dict(color="#FFD54F", width=1.8, dash="dot")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["Rolling_Corr_20"], name="Rolling Corr", line=dict(color="#4FC3F7", width=1.6), fill="tozeroy", fillcolor="rgba(79,195,247,0.14)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=40, t=45, b=35), hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    return fig, latest_corr


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
    c4.metric("1 Board Lot", f"RM {quote['lot_cost']:,.2f}")
    c5.metric("Prev Close", f"RM {quote['prev_close']:.3f}")


if hasattr(st, "fragment"):
    render_live_quote_section = st.fragment(run_every=60)(_render_live_quote_banner)
else:
    render_live_quote_section = _render_live_quote_banner


def main() -> None:
    st.set_page_config(page_title="Bursa Pro Analyzer | Ultimate", page_icon="📈", layout="wide")

    if "user" not in st.session_state:
        st.session_state.user = None
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["1155.KL", "1295.KL", "5347.KL", "5398.KL"]

    st.title("📈 Bursa Pro Analyzer — Ultimate Edition")

    with st.sidebar:
        st.header("🔐 User Authentication")

        # Visual Diagnostic Indicator
        if supabase_client is not None:
            st.caption("🟢 **Supabase Connected**")
        else:
            st.caption("🔴 **Supabase Disconnected (Mock Mode)**")

        if not st.session_state.user:
            auth_mode = st.radio("Mode", ["Sign In", "Sign Up"], horizontal=True)
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button(auth_mode, use_container_width=True):
                if supabase_client is not None:
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
                        else:
                            st.warning("Signup initiated. If required, please confirm your email.")
                    except Exception as e:
                        st.error(f"Authentication Error: {e}")
                else:
                    if email:
                        st.session_state.user = type("User", (), {"id": "mock_user", "email": email})()
                        st.success("Logged in (Mock Local Mode)")
                        st.rerun()
        else:
            st.success(f"User: {st.session_state.user.email}")
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

        sector_prof = get_sector_profile(active_symbol, st.selectbox("Sector Weighting", ["Auto-Detect"] + list(BURSA_SECTOR_PROFILES.keys()), index=0))
        selected_macro_symbol = st.selectbox("Macro Proxy", list(sector_prof.macro_proxies.keys()), format_func=lambda k: f"{k} — {sector_prof.macro_proxies[k]}")
        selected_macro_label = sector_prof.macro_proxies[selected_macro_symbol]
        
        c_p, c_i = st.columns(2)
        with c_p:
            period_label = st.selectbox("Period", list(PERIOD_MAP.keys()), index=3)
        with c_i:
            interval = st.selectbox("Interval", ["1d", "1wk", "1h"], index=0)

        st.subheader("📊 Chart Options")
        show_sma, show_ema, show_bb, show_sr = st.checkbox("SMA", True), st.checkbox("EMA", True), st.checkbox("Bollinger", True), st.checkbox("Support/Resistance", True)
        panel4_mode = st.selectbox("Panel 4", ["All (Stoch + ATR + OBV)", "Stochastic Only", "ATR Only", "OBV Only"])
        forecast_model = st.selectbox("Model", ["ARIMA(1,1,1)", "Holt-Winters"])
        forecast_horizon = st.slider("Forecast Horizon", 5, 60, 14)

    yf_per, warn = validate_period_interval(period_label, interval)
    if warn:
        st.info(warn)

    render_live_quote_section(active_symbol, comp_name)

    with st.spinner(f"Loading data for {active_symbol}..."):
        raw_df, err = fetch_historical_data(active_symbol, yf_per, interval)
        klci_df, _ = fetch_historical_data("^KLSE", "6mo", "1d")
        macro_df, macro_err = fetch_historical_data(selected_macro_symbol, yf_per, "1d")

    if err or raw_df.empty:
        st.error(f"❌ {err or 'Empty dataset.'}")
        st.stop()

    df = compute_technical_indicators(raw_df)
    supp, res = compute_support_resistance(df)
    signal_df, comp_score = evaluate_signals(df, sector_prof)
    mf = interpret_money_flow(df)
    
    last_row = df.iloc[-1]
    last_close = float(last_row["Close"])
    atr_now = max(float(last_row["ATR_14"]), last_close * 0.005)

    if comp_score >= 25.0:
        verdict_label = "BULLISH BIAS"
        e_low, e_high = max(supp, last_close - 0.5 * atr_now), last_close
        d_stop, d_tp = round(max(0.01, last_close - 2.0 * atr_now), 3), round(max(res, last_close + 3.0 * atr_now), 3)
    elif comp_score <= -25.0:
        verdict_label = "BEARISH / DEFENSIVE"
        e_low, e_high = supp, min(last_close, supp + 0.5 * atr_now)
        d_stop, d_tp = round(max(0.01, supp - 1.5 * atr_now), 3), round(last_close + 1.5 * atr_now, 3)
    else:
        verdict_label = "NEUTRAL"
        e_low, e_high = last_close - 0.5 * atr_now, last_close + 0.5 * atr_now
        d_stop, d_tp = round(max(0.01, last_close - 1.5 * atr_now), 3), round(last_close + 2.25 * atr_now, 3)

    tab_fav, tab_deep, tab_sizer, tab_screen = st.tabs(["⭐ Watchlist", f"📈 Deep Dive ({active_symbol})", "🧮 Position Sizer", "🔍 Sector Screener"])

    # ------------------------- TAB 0: WATCHLIST -------------------------
    with tab_fav:
        st.subheader("⭐ Cloud Watchlist Dashboard")
        if not st.session_state.watchlist:
            st.info("Watchlist empty. Click **➕ Add to Watchlist** in the sidebar.")
        else:
            rows = []
            for sym in st.session_state.watchlist:
                r_df, r_err = fetch_historical_data(sym, "3mo", "1d")
                if r_err or r_df.empty:
                    continue
                r_ind = compute_technical_indicators(r_df)
                r_sec = get_sector_profile(sym)
                _, r_score = evaluate_signals(r_ind, r_sec)
                rows.append({"Ticker": sym, "Company": KLCI_COMPONENTS.get(sym, sym), "Close (RM)": round(float(r_ind["Close"].iloc[-1]), 3), "Score": round(r_score, 1), "RSI": round(float(r_ind["RSI_14"].iloc[-1]), 1)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ------------------------- TAB 1: DEEP DIVE -------------------------
    with tab_deep:
        st.subheader("🧭 7-Step Quantitative Analysis Pipeline")
        step1 = "Bullish" if not klci_df.empty and float(klci_df["Close"].iloc[-1]) > float(klci_df["Close"].rolling(20).mean().iloc[-1]) else "Neutral"
        step2 = "Uptrend" if last_close > float(last_row["SMA_50"]) else "Downtrend"
        step3 = f"RSI {last_row['RSI_14']:.1f}"
        step4 = mf["state"]
        news_items, news_notices = fetch_bursa_news(comp_name, active_symbol, sector_prof.rss_search_terms[0], tuple(sector_prof.catalyst_keywords))
        step5 = f"{sum(1 for n in news_items if n['sentiment']=='Positive')} Pos / {sum(1 for n in news_items if n['sentiment']=='Negative')} Neg"
        fc_result = generate_forecast_and_backtest(df, forecast_horizon, forecast_model)
        step6 = f"{fc_result.get('expected_return_pct', 0.0):+.2f}%" if fc_result.get("ok") else "N/A"

        cols = st.columns(7)
        for col, (t_txt, b_txt) in zip(cols, [("1. Market", step1), ("2. Trend", step2), ("3. Momentum", step3), ("4. Money Flow", step4), ("5. News", step5), ("6. Forecast", step6), ("7. Verdict", f"{verdict_label} ({comp_score:+.0f})")]):
            col.markdown(f"**{t_txt}**\n\n`{b_txt}`")

        with st.container(border=True):
            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric("Verdict", verdict_label, f"Score: {comp_score:+.0f}/100")
            v2.metric("Entry Zone", f"RM {e_low:.3f} – {e_high:.3f}")
            v3.metric("ATR Stop-Loss", f"RM {d_stop:.3f}", f"-RM{abs(last_close - d_stop):.3f}")
            v4.metric("Take-Profit", f"RM {d_tp:.3f}", f"+RM{abs(d_tp - last_close):.3f}")
            v5.metric("Gross R/R", f"1 : {(abs(d_tp - last_close) / abs(last_close - d_stop)) if abs(last_close - d_stop) > 0 else 0:.2f}")

        st.subheader("📉 Technical Chart")
        st.plotly_chart(build_main_technical_chart(df, active_symbol, show_sma, show_ema, show_bb, show_sr, panel4_mode, supp, res), use_container_width=True)

        st.subheader(f"🌐 Sector Macro Driver: {active_symbol} vs {selected_macro_label}")
        if not macro_err and not macro_df.empty:
            m_fig, l_corr = build_macro_proxy_chart(df, macro_df, active_symbol, selected_macro_symbol, selected_macro_label)
            if m_fig:
                st.plotly_chart(m_fig, use_container_width=True)

        st.subheader("💸 Money Flow & Signals")
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

        if fc_result.get("ok"):
            st.subheader("🔮 Forecast & Walk-Forward Backtest")
            f_col, b_col = st.columns([3, 1])
            with f_col:
                st.plotly_chart(build_forecast_chart(df, fc_result, active_symbol), use_container_width=True)
            with b_col:
                st.metric("Target", f"RM {fc_result['target_price']:.3f}", f"{fc_result['expected_return_pct']:+.2f}%")
                st.metric("Walk-Forward MAPE", f"{fc_result['mape_pct']:.2f}%")
                st.metric("Directional Acc.", f"{fc_result['directional_accuracy_pct']:.1f}%")
                
                fc_last = fc_result["forecast_df"].iloc[-1]
                st.markdown(
                    f"**{forecast_horizon}-Step Confidence Intervals:**\n"
                    f"- **80% CI:** `RM {fc_last['Lower_80']:.3f} - {fc_last['Upper_80']:.3f}`\n"
                    f"- **95% CI:** `RM {fc_last['Lower_95']:.3f} - {fc_last['Upper_95']:.3f}`"
                )

        if news_items:
            st.subheader("📰 News")
            st.dataframe(pd.DataFrame(news_items), column_config={"link": st.column_config.LinkColumn("URL")}, use_container_width=True, hide_index=True)

        st.subheader("📥 Export")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("⬇️ Download Historical Data CSV", df.to_csv().encode("utf-8"), f"{active_symbol}_data.csv", "text/csv", use_container_width=True)
        with e2:
            st.download_button("⬇️ Download Signals CSV", signal_df.to_csv(index=False).encode("utf-8"), f"{active_symbol}_signals.csv", "text/csv", use_container_width=True)

    # ------------------------- TAB 2: POSITION SIZER -------------------------
    with tab_sizer:
        st.subheader(f"🧮 Bursa Board Lot Sizer & Fee Calculator — {active_symbol}")
        ps1, ps2, ps3 = st.columns(3)
        with ps1:
            acct_cap = st.number_input("Capital (RM)", value=25000.0)
            max_risk = st.slider("Risk (%)", 0.5, 5.0, 1.5)
            man_lots = st.number_input("Manual Lots (0=Auto)", 0, 50000, 0)
        with ps2:
            ps_entry = st.number_input("Entry (RM)", value=round(last_close, 3), step=0.01)
            ps_stop = st.number_input("Stop Loss (RM)", value=d_stop, step=0.01)
            ps_tp = st.number_input("Take Profit (RM)", value=d_tp, step=0.01)
        with ps3:
            brk_pct = st.number_input("Brokerage (%)", 0.01, 1.0, 0.10, 0.01)
            min_brk = st.number_input("Min Brokerage (RM)", value=8.0)
            apply_sst = st.checkbox("Apply 8% SST", True)

        psr = compute_bursa_position_size(acct_cap, max_risk, ps_entry, ps_stop, ps_tp)
        if psr.get("ok"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lots", f"{psr['active_lots']:,}")
            c2.metric("Outlay", f"RM {psr['total_buy_outlay']:,.2f}")
            c3.metric("Net Risk", f"-RM {psr['net_loss_stop']:,.2f}")
            c4.metric("Net R/R", f"1 : {psr['net_rr']:.2f}")
        else:
            st.warning(psr.get("error"))

    # ------------------------- TAB 3: SCREENER -------------------------
    with tab_screen:
        st.subheader("🔍 Multi-Stock Sector Screener")
        s1, s2 = st.columns(2)
        with s1:
            sec_filter = st.selectbox("Universe Sector", ["All Preloaded Equities (35 Stocks)"] + list(BURSA_SECTOR_PROFILES.keys()))
        with s2:
            min_v = st.slider("Min Vol Ratio", 0.0, 3.0, 0.0)

        target_syms = tuple(k for k in KLCI_COMPONENTS.keys() if k != "^KLSE") if sec_filter.startswith("All") else tuple(s for s, sec in TICKER_TO_SECTOR.items() if sec == sec_filter and s in KLCI_COMPONENTS and s != "^KLSE")
        with st.spinner("Scanning equities..."):
            scr_df = run_bursa_sector_screener(target_syms)

        if not scr_df.empty:
            f_scr = scr_df[scr_df["Vol / 20d Avg"] >= min_v].copy()
            st.plotly_chart(build_screener_scatter_chart(f_scr), use_container_width=True)
            st.dataframe(f_scr, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()