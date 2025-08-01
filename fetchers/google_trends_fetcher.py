import os
import time
import datetime
import logging

import pandas as pd
from tqdm import tqdm
from pytrends.request import TrendReq

from config.config import (
    FEATURE_TOGGLES,
    GOOGLE_TRENDS_BATCH_SIZE,
    GOOGLE_TRENDS_SLEEP_SEC
)

# === Setup ===
pytrends = TrendReq(hl='en-US', tz=360)
CACHE_DIR = "data/google_trends"
os.makedirs(CACHE_DIR, exist_ok=True)

# === Load Company Data ===
def load_company_data() -> pd.DataFrame:
    df = pd.read_csv("data/company_names.csv")
    df = df.dropna(subset=["Ticker", "Company", "Sector", "MarketCap"])
    return df

# === Google Trends Fetch Core ===
def fetch_trends_for_term(term: str, timeframe: str = "now 7-d") -> pd.DataFrame:
    try:
        pytrends.build_payload([term], cat=0, timeframe=timeframe, geo="", gprop="")
        data = pytrends.interest_over_time()
        return data[[term]] if not data.empty and term in data.columns else pd.DataFrame()
    except Exception as e:
        logging.info(f"⚠️ Trend fetch failed for '{term}': {e}")
        return pd.DataFrame()

def compute_signals(trend_data: pd.DataFrame, term: str) -> dict:
    if trend_data.empty or term not in trend_data.columns:
        return {"Google Interest": 0.0, "Trend Spike": 0.0}

    series = trend_data[term]
    avg = series.mean()
    spike = 0.0
    if len(series) >= 4:
        last = series.iloc[-1]
        prior_avg = series.iloc[-4:-1].mean()
        spike = round((last - prior_avg) / (prior_avg + 1e-5), 3)

    return {
        "Google Interest": round(avg, 2),
        "Trend Spike": spike
    }

def generate_search_terms(company_name: str, ticker: str) -> list:
    name_short = company_name.split(',')[0].split(' Inc')[0].split(' Corporation')[0].split(' Corp')[0].strip()
    return [ticker, name_short, name_short + " stock"]

# === Master Fetch Function ===
def fetch_google_trends(ticker_list, use_cache=True) -> pd.DataFrame:
    if not FEATURE_TOGGLES.get("Google Trends", True):
        logging.info("🛑 Google Trends disabled via toggle")
        return pd.DataFrame()

    today_str = datetime.date.today().isoformat()
    cache_path = os.path.join(CACHE_DIR, f"{today_str}.csv")

    if use_cache and os.path.exists(cache_path):
        try:
            df_cached = pd.read_csv(cache_path)
            if "Ticker" in df_cached.columns:
                return df_cached
            logging.warning(f"⚠️ Cache found but invalid format: {cache_path}")
        except pd.errors.EmptyDataError:
            logging.warning(f"⚠️ Cache file is empty: {cache_path}")

    df_all = load_company_data()
    base_set = {ticker.lstrip("$") for ticker in ticker_list}
    logging.info(f"🎯 Tickers requested for Google Trends: {sorted(base_set)}")

    df_core = df_all[df_all["Ticker"].isin(base_set)].copy()
    if df_core.empty:
        logging.warning("🚫 No valid tickers for Google Trends processing")
        return pd.DataFrame()

    df_core["Source"] = "Core"
    logging.info(f"📈 Fetching Google Trends for {len(df_core)} terms")

    results = []
    for i, row in enumerate(tqdm(df_core.itertuples(index=False, name='CompanyRow'), total=len(df_core), desc="📊 Google Trends"), 1):
        ticker = row.Ticker
        company_name = row.Company
        source = row.Source

        trend_data = pd.DataFrame()
        terms = generate_search_terms(company_name, ticker)
        used_term = None

        for term in terms:
            trend_data = fetch_trends_for_term(term)
            if not trend_data.empty:
                used_term = term
                break

        if trend_data.empty:
            logging.warning(f"⚠️ No Google Trends data for ticker {ticker} using terms {terms}")
            scores = {"Google Interest": 0.0, "Trend Spike": 0.0}
        else:
            scores = compute_signals(trend_data, used_term)

        scores.update({"Ticker": ticker, "Source": source})
        results.append(scores)

        logging.info(f"🔍 [{i}] {ticker} — Interest: {scores['Google Interest']}, Spike: {scores['Trend Spike']} [{used_term}]")

        if i % GOOGLE_TRENDS_BATCH_SIZE == 0:
            logging.info(f"⏳ Sleeping {GOOGLE_TRENDS_SLEEP_SEC}s after batch {i}")
            time.sleep(GOOGLE_TRENDS_SLEEP_SEC)

    df_out = pd.DataFrame(results, columns=["Ticker", "Google Interest", "Trend Spike", "Source"])
    df_out["Ticker"] = df_out["Ticker"].apply(lambda x: f"${x}" if not str(x).startswith("$") else str(x))
    df_out.to_csv(cache_path, index=False)
    logging.info(f"💾 Cached Google Trends to {cache_path}")

    return df_out
