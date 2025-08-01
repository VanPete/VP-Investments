import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logging
from config.labels import COLUMN_FORMAT_HINTS
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv
from config.config import FEATURE_TOGGLES

# === Setup ===
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
FMP_API_KEY = os.getenv("FMP_API_KEY", "MISSING")

# === Utility Functions ===
def human_format(num) -> str:
    if num is None or pd.isna(num):
        return ""
    num = float(num)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:.1f}{unit}" if unit else f"{num:.0f}"
        num /= 1000.0
    return f"{num:.1f}P"

def load_company_data() -> pd.DataFrame:
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        COMPANY_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "company_names.csv")
        df = pd.read_csv(COMPANY_DATA_PATH)
        df["Ticker"] = df["Ticker"].apply(lambda x: f"${str(x).upper()}")
        df["Company"] = df["Company"].astype(str)
        return df.drop_duplicates(subset="Ticker")
    except FileNotFoundError:
        raise FileNotFoundError(f"{COMPANY_DATA_PATH} not found. Ensure it exists.")

@lru_cache(maxsize=256)
def get_yf_ticker(ticker: str):
    return yf.Ticker(ticker.replace("$", ""))

# === Feature Extractors ===
def fetch_options_sentiment(ticker: str) -> Tuple[float, float, float, float, float]:
    try:
        yt = get_yf_ticker(ticker)
        opt_dates = yt.options
        if not opt_dates:
            return (np.nan,) * 5
        chain = yt.option_chain(opt_dates[0])
        calls = chain.calls
        puts = chain.puts
        call_oi_ratio = puts["openInterest"].sum() / calls["openInterest"].sum() if calls["openInterest"].sum() else np.nan
        call_vol_ratio = puts["volume"].sum() / calls["volume"].sum() if calls["volume"].sum() else np.nan
        skew = calls["impliedVolatility"].mean() - puts["impliedVolatility"].mean()
        call_vol_spike = np.nan
        if len(calls["volume"]) >= 5:
            rolling_mean = calls["volume"].rolling(5).mean().iloc[-1]
            call_vol_spike = calls["volume"].iloc[-1] / rolling_mean if rolling_mean else np.nan
        iv_spike = (
            calls["impliedVolatility"].pct_change().iloc[-1] * 100
            if len(calls["impliedVolatility"]) > 1 else np.nan
        )
        return call_oi_ratio, call_vol_ratio, skew, call_vol_spike, iv_spike
    except Exception as e:
        logger.warning(f"{ticker} options fetch failed: {e}")
        return (np.nan,) * 5

def fetch_insider_buys(ticker: str) -> Tuple[int, int, str, str]:
    try:
        symbol = ticker.replace("$", "")
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={symbol}&apikey={FMP_API_KEY}"
        )
        data = resp.json()
        df = pd.DataFrame(data if isinstance(data, list) else [data])
        if df.empty or 'transactionDate' not in df.columns:
            return 0, 0, "", "No"
        df['transactionDate'] = pd.to_datetime(df['transactionDate'], errors='coerce')
        recent = df[(df['transactionType'] == "Buy") & (df['transactionDate'] >= datetime.utcnow() - timedelta(days=30))]
        count = len(recent)
        volume = int(recent['securitiesTransacted'].sum()) if 'securitiesTransacted' in recent else 0
        last_date = recent['transactionDate'].max().strftime("%Y-%m-%d") if count else ""
        signal = "Yes" if count >= 1 else "No"
        return count, volume, last_date, signal
    except Exception as e:
        logger.warning(f"{ticker} insider fetch failed: {e}")
        return 0, 0, "", "No"

def fetch_sector_flows(ticker: str) -> Tuple[float, float, str]:
    try:
        resp = requests.get(
            f"https://financialmodelingprep.com/api/v4/etf-sector-weightings?symbol=SPY&apikey={FMP_API_KEY}"
        )
        data = resp.json()
        if not data or not isinstance(data, list):
            return 0.0, 0.0, "No"
        current_weights = {item["sector"]: float(item["weightPercentage"]) for item in data}
        sector = max(current_weights, key=current_weights.get)
        inflow = round(current_weights[sector] - current_weights[sector] * 0.97, 2)
        spike_ratio = round(current_weights[sector] / (current_weights[sector] * 0.95), 2)
        signal = "Yes" if inflow > 0.5 or spike_ratio > 1.5 else "No"
        return inflow, spike_ratio, signal
    except Exception as e:
        logger.warning(f"{ticker} ETF flow fetch failed: {e}")
        return 0.0, 0.0, "No"

# === Feature Builders ===
def calculate_technicals(closes: pd.Series) -> Dict[str, float]:
    current = closes.iloc[-1]
    volatility = closes.rolling(14).std().iloc[-1]
    ma_50 = closes.rolling(50).mean().iloc[-1]
    ma_200 = closes.rolling(200).mean().iloc[-1]
    price_vs_50 = ((current - ma_50) / ma_50 * 100) if ma_50 else np.nan
    price_vs_200 = ((current - ma_200) / ma_200 * 100) if ma_200 else np.nan

    delta = closes.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = -delta.clip(upper=0).rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).dropna().iloc[-1] if not rs.dropna().empty else np.nan

    exp1 = closes.ewm(span=12, adjust=False).mean()
    exp2 = closes.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = (macd - signal).iloc[-1] if len(macd) > 0 else np.nan

    bb_mean = closes.rolling(20).mean()
    bb_std = closes.rolling(20).std()
    upper = bb_mean + 2 * bb_std
    lower = bb_mean - 2 * bb_std
    bollinger_width = ((upper - lower) / closes).iloc[-1] if len(closes) >= 20 else np.nan

    return {
        "Volatility": volatility,
        "Above 50-Day MA %": price_vs_50,
        "Above 200-Day MA %": price_vs_200,
        "RSI": rsi,
        "MACD Histogram": macd_hist,
        "Bollinger Width": bollinger_width
    }

# === Main Pull Function ===
def fetch_single_ticker(ticker: str) -> Tuple[str, List]:
    try:
        yt = get_yf_ticker(ticker)
        info = yt.info
        hist = yt.history(period="1y")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise ValueError("No historical close data")

        current = closes.iloc[-1]
        change_1d = ((current - closes.iloc[-2]) / closes.iloc[-2] * 100) if len(closes) >= 2 else np.nan
        change_7d = ((current - closes.iloc[-7]) / closes.iloc[-7] * 100) if len(closes) >= 7 else np.nan
        volume_spike = info.get("volume", 0) / hist["Volume"].rolling(7).mean().iloc[-1] if len(hist) >= 7 else np.nan

        tech = calculate_technicals(closes)

        fundamentals = [
            info.get("trailingPE", np.nan), info.get("earningsQuarterlyGrowth", np.nan),
            info.get("returnOnEquity", np.nan), info.get("debtToEquity", np.nan),
            (info.get("freeCashflow", 0) / info.get("totalRevenue", 1)) if info.get("totalRevenue") else np.nan
        ]

        short_interest = [
            info.get("sharesShort", np.nan), info.get("shortRatio", np.nan),
            info.get("shortPercentOfFloat", np.nan), info.get("shortPercentOfSharesOutstanding", np.nan)
        ]

        options = fetch_options_sentiment(ticker)
        insiders = fetch_insider_buys(ticker)
        etf = fetch_sector_flows(ticker)

        momentum_30d = ((current - closes.iloc[-30]) / closes.iloc[-30] * 100) if len(closes) >= 30 else np.nan
        beta = info.get("beta", np.nan)

        try:
            earnings_date = yt.calendar.loc["Earnings Date"][0].strftime("%Y-%m-%d")
        except Exception:
            earnings_date = ""

        try:
            earn = yt.earnings_dates.index[-1]
            pre = closes[closes.index < earn].iloc[-1]
            post = closes[closes.index >= earn].iloc[0]
            earnings_gap = ((post - pre) / pre * 100)
        except Exception:
            earnings_gap = np.nan

        avg_val = (hist["Close"] * hist["Volume"]).rolling(30).mean().iloc[-1] if len(hist) >= 30 else np.nan

        return ticker, [
            ticker, info.get("shortName", ""), current, change_1d, change_7d, info.get("marketCap"),
            info.get("volume"), info.get("sector", ""), volume_spike, change_7d,
            tech["Above 50-Day MA %"], tech["Above 200-Day MA %"], tech["RSI"], tech["MACD Histogram"],
            tech["Bollinger Width"], tech["Volatility"], *fundamentals, *short_interest, *options,
            *insiders, *etf, momentum_30d, beta, earnings_date, earnings_gap,
            info.get("heldPercentInsiders", np.nan), info.get("heldPercentInstitutions", np.nan), avg_val
        ]
    except Exception as e:
        logger.warning(f"{ticker} failed: {e}")
        return ticker, [ticker] + [np.nan] * 43

def fetch_yf_data(ticker_list: List[str], max_workers: int = 8) -> pd.DataFrame:
    results = {}
    delisted = []
    tickers = [str(t) for t in ticker_list]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="📊 Fetching yfinance data", unit="ticker"):
            try:
                ticker, data = future.result()
                results[ticker] = data
                if len(data) != 44 or all(pd.isna(val) for val in data[1:]):
                    delisted.append(ticker)
            except Exception:
                t = futures[future]
                results[t] = [t] + [np.nan] * 43
                delisted.append(t)

    col_labels = [
        "Ticker", "Company", "Current Price", "Price 1D %", "Price 7D %", "Market Cap", "Volume", "Sector",
        "Volume Spike Ratio", "Relative Strength", "Above 50-Day MA %", "Above 200-Day MA %", "RSI", "MACD Histogram",
        "Bollinger Width", "Volatility", "P/E Ratio", "EPS Growth", "ROE", "Debt/Equity", "FCF Margin",
        "Shares Short", "Short Ratio", "Short Percent Float", "Short Percent Outstanding",
        "Put/Call OI Ratio", "Put/Call Volume Ratio", "Options Skew", "Call Volume Spike Ratio", "IV Spike %",
        "Insider Buys 30D", "Insider Buy Volume", "Last Insider Buy Date", "Insider Signal",
        "Sector Inflows", "ETF Flow Spike Ratio", "ETF Flow Signal",
        "Momentum 30D %", "Beta vs SPY", "Next Earnings Date", "Earnings Gap %",
        "Retail Holding %", "Float % Held by Institutions", "Avg Daily Value Traded"
    ]

    df = pd.DataFrame([results[t] for t in tickers], columns=col_labels)

    if "Volatility" in df.columns and not df["Volatility"].isnull().all():
        df["Volatility Rank"] = df["Volatility"].rank(pct=True)

    return df
