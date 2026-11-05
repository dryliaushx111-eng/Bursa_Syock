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
            "ok": True, "model_type": model_type, "horizon": horizon, "forecast_df": fc_df,
            "target_price": float(mean_fc[-1]), "expected_return_pct": float((