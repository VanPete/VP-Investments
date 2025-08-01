"""
news_fetcher.py
Fetches and scores news sentiment for a list of tickers using NewsAPI and VADER.
Includes fuzzy matching, caching, and article summarization.
"""

import os
import json
import time
import re
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# === Config ===
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
assert NEWS_API_KEY, "❌ NEWS_API_KEY is not set in .env"

CACHE_DIR = os.getenv("NEWS_CACHE_DIR", "cache/news")
os.makedirs(CACHE_DIR, exist_ok=True)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
BATCH_SIZE = 5
CACHE_TTL = timedelta(hours=1)
DAILY_LIMIT = 90
API_CALL_COUNT = 0


# === Company Name Matching ===
def load_company_names() -> Dict[str, str]:
    """Load ticker → company name mapping from /data/company_names.csv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "data", "company_names.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path} — expected in /data.")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "ticker" not in df.columns or "company" not in df.columns:
        raise ValueError(f"Expected columns 'ticker' and 'company' in {path}")

    return dict(zip(df["ticker"], df["company"]))


def clean_company_name(name: str) -> str:
    """Remove common suffixes for cleaner fuzzy matching."""
    return re.sub(r"\b(inc|inc\.|corp|corporation|llc|ltd|plc|group|co|company|s\.a\.?)\b", "", name, flags=re.I).strip()


# === Caching Logic ===
def is_cache_fresh(symbol: str) -> bool:
    """Check if cached file is recent enough to reuse."""
    path = os.path.join(CACHE_DIR, f"{symbol}.json")
    if not os.path.exists(path):
        return False
    mod = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mod) < CACHE_TTL


def load_cached(symbol: str) -> List[dict]:
    """Load cached news articles for a ticker."""
    with open(os.path.join(CACHE_DIR, f"{symbol}.json"), 'r') as f:
        return json.load(f)


def save_cache(symbol: str, articles: List[dict]):
    """Cache news articles locally for a ticker."""
    with open(os.path.join(CACHE_DIR, f"{symbol}.json"), 'w') as f:
        json.dump(articles, f)


# === News Fetching ===
def fuzzy_match_articles(name: str, articles: List[dict], threshold: int = 80) -> List[dict]:
    """Return list of articles whose text matches the company name fuzzily."""
    cleaned = clean_company_name(name).lower()
    matched = []
    for a in articles:
        text = f"{a.get('title') or ''} {a.get('description') or ''}".lower()
        if fuzz.partial_ratio(cleaned, text) >= threshold:
            matched.append(a)
    return matched


def fetch_news_batch(names: List[str]) -> Dict[str, List[dict]]:
    """Batch query NewsAPI and apply fuzzy article matching for each company name."""
    global API_CALL_COUNT
    results = {}
    query = " OR ".join(names)
    params = {
        "q": query,
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100,
    }

    logging.info(f"📡 NewsAPI query: {params['q']}")
    resp = requests.get(NEWSAPI_URL, params=params)
    API_CALL_COUNT += 1
    logging.info(f"🔄 API calls used: {API_CALL_COUNT}")

    if resp.status_code != 200:
        logging.warning(f"❌ NewsAPI error {resp.status_code}: {resp.text}")
        return {name: [] for name in names}

    all_articles = resp.json().get("articles", [])
    for name in names:
        matched = fuzzy_match_articles(name, all_articles)
        results[name] = matched
        logging.info(f"📰 {name} matched {len(matched)} articles")

    return results


def fetch_news_for_tickers(tickers: List[str]) -> Dict[str, List[dict]]:
    """Main method to fetch and cache news articles for a list of tickers."""
    logging.info(f"🧠 News fetch starting for {len(tickers)} tickers")
    company_map = load_company_names()
    fetched = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        names = [company_map.get(t, t) for t in batch]
        uncached = [t for t in batch if not is_cache_fresh(t)]

        if not uncached:
            for t in batch:
                fetched[t] = load_cached(t)
            continue

        if API_CALL_COUNT >= DAILY_LIMIT:
            for t in batch:
                path = os.path.join(CACHE_DIR, f"{t}.json")
                fetched[t] = load_cached(t) if os.path.exists(path) else []
            logging.warning("🚫 Reached NewsAPI daily limit — using cache only")
            continue

        batch_res = fetch_news_batch(names)
        for sym, arts in zip(batch, batch_res.values()):
            fetched[sym] = arts
            save_cache(sym, arts)
        time.sleep(1)  # Rate limiting

    return fetched


# === Sentiment Scoring ===
vader = SentimentIntensityAnalyzer()


def score_articles_sentiment(articles: List[dict]) -> float:
    """Return average compound VADER sentiment score for a list of articles."""
    if not articles:
        return 0.0
    scores = []
    for art in articles:
        text = f"{art.get('title') or ''}. {art.get('description') or ''}".strip()
        if text:
            score = vader.polarity_scores(text)["compound"]
            scores.append(score)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def summarize_news_sentiment(news_dict: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """Summarize sentiment for each ticker into count and average sentiment score."""
    summary = {}
    for ticker, articles in news_dict.items():
        sentiment = score_articles_sentiment(articles)
        summary[ticker] = {
            "News Mentions": len(articles),
            "News Sentiment": sentiment
        }
    return summary
