"""
news_fetcher.py
Fetches and scores news sentiment using Google News RSS (no API key) and VADER,
then optionally summarizes with OpenAI for better results.
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
from tqdm import tqdm

import pandas as pd
from rapidfuzz import fuzz
from dotenv import load_dotenv
from config.config import (
    AI_FEATURES, NEWS_CACHE_DIR, NEWS_CACHE_TTL_HOURS,
    NEWS_RSS_TIMEOUT_SEC, NEWS_FETCH_PACING_SEC, NEWS_FUZZY_THRESHOLD, NEWS_MAX_ITEMS,
    TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST, BREAKER_FAIL_THRESHOLD, BREAKER_RESET_AFTER_SEC
)
from utils.ai_cache import get as ai_get, set as ai_set
from processors.chatgpt_integrator import ChatGPTIntegrator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from utils.http_client import build_session, get_token_bucket, CircuitBreaker
from utils.observability import emit_metric

# === Config ===
load_dotenv()

CACHE_DIR = NEWS_CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_TTL = timedelta(hours=NEWS_CACHE_TTL_HOURS)

# Shared HTTP session with caching to reduce repeated RSS hits
_rss_sess = build_session(
    cache_name=os.path.join(CACHE_DIR, "rss_cache"),
    cache_expire_seconds=int(CACHE_TTL.total_seconds()),
    timeout=NEWS_RSS_TIMEOUT_SEC,
)
_rss_bucket = get_token_bucket(TOKEN_BUCKET_RATE, TOKEN_BUCKET_BURST)
_rss_breaker = CircuitBreaker(BREAKER_FAIL_THRESHOLD, BREAKER_RESET_AFTER_SEC)
 


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
    """Remove common suffixes and stray punctuation for cleaner fuzzy matching."""
    # Remove common suffixes with optional trailing dot
    s = re.sub(r"\b(inc|corp|corporation|llc|ltd|plc|group|co|company|s\.a\.?)\.?\b", "", name, flags=re.I)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    # Strip common trailing punctuation and whitespace
    s = s.strip().strip(" .,-")
    return s


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
def fuzzy_match_articles(name: str, articles: List[dict], threshold: int = NEWS_FUZZY_THRESHOLD) -> List[dict]:
    """Return list of articles whose text matches the company name fuzzily."""
    cleaned = clean_company_name(name).lower()
    matched = []
    for a in articles:
        text = f"{a.get('title') or ''} {a.get('description') or ''}".lower()
        if fuzz.partial_ratio(cleaned, text) >= threshold:
            matched.append(a)
    return matched


def fetch_google_news_rss(name: str, max_items: int = NEWS_MAX_ITEMS) -> List[dict]:
    """Fetch recent Google News RSS items for a company name, last ~7 days."""
    q = requests.utils.quote(f'"{name}" when:7d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        # Circuit breaker: skip if upstream is degrading
        if not _rss_breaker.allow():
            logging.info("[NEWS] RSS circuit open; skipping fetch for %s", name)
            try:
                emit_metric("news_rss", {"ok": 0, "breaker_open": True, "name": name})
            except Exception:
                pass
            return []
        # Token bucket pacing
        waited = _rss_bucket.take(1.0)
        if waited > 0.2:
            logging.debug("[NEWS] Rate-limiter waited %.2fs before RSS call", waited)
        _start = time.time()
        resp = _rss_sess.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")
        articles = []
        for it in items[:max_items]:
            title = (it.title.text if it.title else "").strip()
            desc = (it.description.text if it.description else "").strip()
            pub = it.pubDate.text if it.pubDate else ""
            articles.append({
                "title": title,
                "description": desc,
                "publishedAt": pub
            })
        _rss_breaker.record(True)
        try:
            emit_metric("news_rss", {"ok": 1, "latency_ms": int((time.time()-_start)*1000), "items": len(articles)})
        except Exception:
            pass
        return articles
    except Exception as e:
        logging.warning(f"RSS fetch failed for {name}: {e}")
        _rss_breaker.record(False)
        try:
            emit_metric("news_rss", {"ok": 0, "error": str(e)[:200]})
        except Exception:
            pass
        return []


def fetch_news_for_tickers(tickers: List[str]) -> Dict[str, List[dict]]:
    """Fetch news via Google News RSS per ticker and cache results."""
    total = len(tickers)
    logging.info(f"🧠 News fetch (RSS) starting for {total} tickers")
    company_map = load_company_names()
    fetched: Dict[str, List[dict]] = {}

    for i, t in enumerate(tqdm(tickers, desc="📰 Fetching News", unit="ticker"), start=1):
        try:
            if is_cache_fresh(t):
                fetched[t] = load_cached(t)
                logging.debug(f"[NEWS {i}/{total}] {t}: cache hit ({len(fetched[t])} items)")
                continue
            company = company_map.get(t, t)
            articles = fetch_google_news_rss(company)
            # Fuzzy filter with company name
            articles = fuzzy_match_articles(company, articles)
            fetched[t] = articles
            save_cache(t, articles)
            try:
                emit_metric("news_ticker", {"ok": 1, "ticker": t, "items": len(articles)})
            except Exception:
                pass
            if i % 10 == 0 or i == total:
                logging.info(f"[NEWS] Progress: {i}/{total} tickers processed")
            # Gentle pacing in addition to token bucket to avoid bursty patterns
            if NEWS_FETCH_PACING_SEC:
                time.sleep(NEWS_FETCH_PACING_SEC)
        except Exception as e:
            logging.warning(f"News fetch failed for {t}: {e}")
            fetched[t] = []
            try:
                emit_metric("news_ticker", {"ok": 0, "ticker": t})
            except Exception:
                pass

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
    """Summarize sentiment per ticker and optionally add AI news summary text."""
    summary = {}
    integrator = ChatGPTIntegrator() if AI_FEATURES.get("ENABLED") and AI_FEATURES.get("NEWS_SUMMARY") else None
    total = len(news_dict)
    for idx, (ticker, articles) in enumerate(tqdm(news_dict.items(), desc="🧪 Scoring News", unit="ticker"), start=1):
        sentiment = score_articles_sentiment(articles)
        row = {
            "News Mentions": len(articles),
            "News Sentiment": sentiment
        }
        if integrator:
            cache_key = f"news_summary::{ticker}::{len(articles)}::{round(sentiment,3)}"
            cached = ai_get(cache_key)
            if cached:
                row["AI News Summary"] = cached
            else:
                try:
                    text = integrator.generate_news_summary(ticker, articles, sentiment)
                    row["AI News Summary"] = text
                    ai_set(cache_key, text)
                except Exception:
                    row["AI News Summary"] = ""
        if idx % 20 == 0 or idx == total:
            logging.info(f"[NEWS] Scoring progress: {idx}/{total}")
        summary[ticker] = row
    return summary
