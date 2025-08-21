"""
reddit_scraper.py
Scrapes Reddit posts from configured subreddits, filters by quality, extracts sentiment, and identifies signals.
"""

import os
import re
import logging
import datetime
import time
from math import log10
from typing import List, Dict

import pandas as pd
import praw
import prawcore
from praw.models import Submission
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import warnings
from config.config import AI_FEATURES
from utils.ai_cache import get as ai_get, set as ai_set
from processors.chatgpt_integrator import ChatGPTIntegrator
from utils.observability import emit_metric
from utils.http_client import get_token_bucket, CircuitBreaker
from utils.reddit_seen_cache import SeenCache

from config.config import (
    REDDIT_SUBREDDITS, REDDIT_LIMITS, ENABLE_REDDIT_SCRAPE, DEBUG_REDDIT_SCRAPE,
    REDDIT_TICKER_PATTERN, SUBREDDIT_WEIGHTS, KEYWORD_BOOSTS, FLAIR_BOOSTS,
    FLAIR_ALIASES, MIN_AUTHOR_KARMA, TITLE_COMMENT_BLEND, COMMENT_WEIGHT_SCALING, EXCLUDED_TICKERS, FEATURE_TOGGLES,
    MEME_TICKER_TERMS, REDDIT_PACING_SEC, REDDIT_RATE_PER_SEC, REDDIT_BURST,
    REDDIT_BREAKER_FAILS, REDDIT_BREAKER_RESET_SEC, REDDIT_SEEN_CACHE_TTL_MIN,
    FAST_MODE, FAST_REDDIT_CATEGORIES, REDDIT_MAX_POSTS, REDDIT_MAX_PER_SUB,
    REDDIT_LISTING_CACHE_ENABLED, REDDIT_LISTING_CACHE_TTL_MIN
)
from pathlib import Path
from utils.logger import setup_logging

# === Init ===
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*comment_sort.*already been fetched.*")

_RID = os.getenv("REDDIT_CLIENT_ID", "").strip()
_RSEC = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
_RUA = os.getenv("REDDIT_USER_AGENT", "").strip()
reddit = None
if _RID and _RSEC and _RUA:
    try:
        reddit = praw.Reddit(client_id=_RID, client_secret=_RSEC, user_agent=_RUA)
        # Touch a lightweight endpoint to surface auth errors early
        try:
            _ = reddit.auth.limits  # property access shouldn't call network
        except Exception:
            pass
    except Exception as e:
        logger.warning("Reddit client init failed (check REDDIT_CLIENT_* env vars): %s", e)
        reddit = None
else:
    logger.info("Reddit credentials not set; Reddit scraping will be skipped.")


try:
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error("VADER sentiment analyzer failed to initialize: %s", e)
    raise e

# === Pacing & Resilience ===
_bucket = get_token_bucket(REDDIT_RATE_PER_SEC, REDDIT_BURST, name="reddit")
_breaker = CircuitBreaker(REDDIT_BREAKER_FAILS, REDDIT_BREAKER_RESET_SEC)

# Simple in-memory seen-post cache to reduce duplicates across runs
_SEEN_TTL = float(REDDIT_SEEN_CACHE_TTL_MIN) * 60.0
_seen_cache = SeenCache(ttl_seconds=_SEEN_TTL)

# Optional lightweight listing cache (per subreddit/category)
_LISTING_CACHE_DIR = Path("cache/reddit_listings")
try:
    _LISTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def _cache_key(sub: str, category: str) -> Path:
    return _LISTING_CACHE_DIR / f"{sub}__{category}.json"

def _load_listing_cache(sub: str, category: str):
    try:
        if not REDDIT_LISTING_CACHE_ENABLED:
            return None
        p = _cache_key(sub, category)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > max(60, REDDIT_LISTING_CACHE_TTL_MIN * 60):
            return None
        import json
        with open(p, "r", encoding="utf-8") as f:
            ids = json.load(f)
        return ids if isinstance(ids, list) else None
    except Exception:
        return None

def _save_listing_cache(sub: str, category: str, ids: list[str]) -> None:
    try:
        if not REDDIT_LISTING_CACHE_ENABLED:
            return
        import json
        p = _cache_key(sub, category)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(ids[:200], f)
    except Exception:
        pass


# === Text Processing ===
def extract_tickers(text: str) -> List[str]:
    raw = re.findall(REDDIT_TICKER_PATTERN, text)
    cleaned = {f"${clean_ticker(t)}" for t in raw if len(clean_ticker(t)) > 1}
    matched = list(cleaned)
    if DEBUG_REDDIT_SCRAPE:
        logger.debug(f"Raw ticker candidates: {raw} → Matched: {matched}")
    return matched

def clean_ticker(ticker: str) -> str:
    return ticker.upper().replace("$", "").strip()

def normalize_flair(raw_flair: str) -> str:
    if not raw_flair:
        return ""
    cleaned = re.sub(r'[^\w\s]', '', raw_flair.lower().strip())
    return FLAIR_ALIASES.get(cleaned, cleaned.title())

def get_sentiment_score(text: str) -> float:
    return vader.polarity_scores(text)["compound"]

# === Sentiment Weighting ===
def get_top_comment_sentiment(post: Submission, limit: int = 3) -> float:
    try:
        # Set sort before any comments access to avoid UserWarning from PRAW
        try:
            # Locally silence PRAW's UserWarning about changing sort after fetch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                post.comment_sort = "top"
        except Exception:
            pass
        post.comments.replace_more(limit=0)
        comments = [c.body for c in post.comments[:limit] if not c.stickied and c.body != "[deleted]"]
        if not comments:
            return 0.0
        scores = [get_sentiment_score(c) for c in comments]
        return sum(scores) / len(scores)
    except Exception as e:
        logger.warning(f"Top comments failed for {post.id}: {e}")
        return 0.0

def get_weighted_sentiment(text: str, flair: str = "", comment_sent: float = 0.0,
                           subreddit: str = "", comment_count: int = 0) -> float:
    base = get_sentiment_score(text)
    blended = TITLE_COMMENT_BLEND[0] * base + TITLE_COMMENT_BLEND[1] * comment_sent
    boost = 1.0

    for word, mult in KEYWORD_BOOSTS.items():
        if word in text.lower():
            boost *= mult
    if flair in FLAIR_BOOSTS:
        boost *= FLAIR_BOOSTS[flair]
    if subreddit in SUBREDDIT_WEIGHTS:
        boost *= SUBREDDIT_WEIGHTS[subreddit]

    boost *= min(2.0, 1 + log10(comment_count + 1) * COMMENT_WEIGHT_SCALING)
    boost = min(boost, 3.0)

    return round(blended * boost, 4)


# === Summary Extraction ===
def score_summary_candidate(text: str, base_sentiment: float = 0.0) -> float:
    sentiment = get_sentiment_score(text)
    keyword_score = sum(word in text.lower() for word in KEYWORD_BOOSTS)
    length_penalty = 1.0 - min(len(text) / 500, 0.5)
    return round((0.5 * sentiment + 0.3 * base_sentiment + 0.2 * keyword_score) * length_penalty, 4)

def generate_reddit_summary(post: Submission, max_comments: int = 6) -> str:
    try:
        # Ensure we set sort before fetching comments to prevent warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                post.comment_sort = "top"
        except Exception:
            pass
        post.comments.replace_more(limit=0)
        base_sent = get_sentiment_score(post.title)
        candidates = [post.title.strip()]
        count = 0

        for comment in post.comments:
            if comment.stickied or comment.body in ("[deleted]", "[removed]"):
                continue
            candidates.append(comment.body.strip())
            count += 1
            if count >= max_comments:
                break

        scored = sorted(
            [(text, score_summary_candidate(text, base_sent)) for text in candidates],
            key=lambda x: x[1], reverse=True
        )
        return " | ".join([text for text, _ in scored[:3]])
    except Exception as e:
        logger.warning(f"Summary failed for post {post.id}: {e}")
        return post.title.strip()

def process_submission(post: Submission, sub: str, now: datetime.datetime) -> List[Dict]:
    try:
        created_time = datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc)
        if (now - created_time).days > 7:
            return []

        author = post.author
        karma = getattr(author, "link_karma", 0) + getattr(author, "comment_karma", 0) if author else 0
        if not author or karma < MIN_AUTHOR_KARMA:
            return []

        tickers = extract_tickers(post.title + " " + post.selftext)
        tickers = [t for t in tickers if t not in EXCLUDED_TICKERS]
        # Drop obvious meme/symbolic non-tickers seen in top lists
        tickers = [t for t in tickers if all(term not in t for term in MEME_TICKER_TERMS)]
        if not tickers:
            return []

        results = []
        integrator = ChatGPTIntegrator() if AI_FEATURES.get("ENABLED") and AI_FEATURES.get("REDDIT_SUMMARY") else None
        for ticker in tickers:
            comment_sent = get_top_comment_sentiment(post)
            flair = normalize_flair(post.link_flair_text or "")
            sentiment = get_weighted_sentiment(
                post.title + " " + post.selftext,
                flair=flair,
                comment_sent=comment_sent,
                subreddit=sub,
                comment_count=post.num_comments
            )
            summary = generate_reddit_summary(post)
            ai_summary = ""
            if integrator:
                cache_key = f"reddit_ai_summary::{ticker}::{post.id}::{round(sentiment,3)}"
                cached = ai_get(cache_key)
                if cached:
                    ai_summary = cached
                else:
                    try:
                        ai_summary = integrator.enhance_reddit_summary(ticker, summary, sentiment)
                        ai_set(cache_key, ai_summary)
                    except Exception:
                        ai_summary = ""
            results.append({
                "Ticker": ticker,
                "Subreddit": sub,
                "Title": post.title,
                "Upvotes": post.score,
                "created": created_time,
                "Reddit Sentiment": sentiment,
                "Reddit Summary": summary,
                "AI Reddit Summary": ai_summary,
                "Author": str(author),
                "Post URL": f"https://reddit.com{post.permalink}"
            })
        try:
            emit_metric("reddit_post", {"ok": 1, "subreddit": sub, "tickers": len(tickers)})
        except Exception:
            pass
        return results
    except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden) as e:
        # Post deleted/removed or forbidden; treat as expected and skip
        logger.info(f"Post skipped ({getattr(post, 'id', 'unknown')}): {e}")
        try:
            emit_metric("reddit_post", {"ok": 0, "subreddit": sub, "reason": "deleted/forbidden"})
        except Exception:
            pass
        return []
    except prawcore.exceptions.ResponseException as e:
        status = getattr(getattr(e, 'response', None), 'status_code', None)
        if status == 404:
            logger.info(f"Post skipped ({getattr(post, 'id', 'unknown')}): 404 Not Found")
            return []
        logger.warning(f"Post process failed: {getattr(post, 'id', 'unknown')} → HTTP {status}")
        try:
            emit_metric("reddit_post", {"ok": 0, "subreddit": sub, "status": status})
        except Exception:
            pass
        return []
    except Exception as e:
        logger.warning(f"Post process failed: {getattr(post, 'id', 'unknown')} → {e}")
        try:
            emit_metric("reddit_post", {"ok": 0, "subreddit": sub})
        except Exception:
            pass
        return []

def is_emerging(df: pd.DataFrame) -> pd.Series:
    df["date"] = df["created"].dt.date
    ticker_day_counts = df.groupby(["Ticker", "date"]).size().unstack(fill_value=0)

    flags = {}
    for ticker in ticker_day_counts.index:
        daily = ticker_day_counts.loc[ticker].sort_index()
        if len(daily) < 7:
            flags[ticker] = False
            continue
        today = daily.iloc[-1]
        prior_3 = daily.iloc[-4:-1].mean()
        baseline = daily.iloc[-10:-7].mean() if len(daily) >= 10 else 0

        flags[ticker] = today >= 3 and prior_3 < 2 and today > 2.5 * prior_3 and today > 2 * baseline

    return df["Ticker"].map(flags).fillna(False)

def tag_threads(df: pd.DataFrame) -> pd.Series:
    df["author_str"] = df["Title"].str.extract(r'u/([A-Za-z0-9_-]+)', expand=False).fillna("unknown")
    thread_flags = df.groupby(["Ticker", "author_str"]).size().reset_index(name="post_count")
    threads = set(zip(thread_flags[thread_flags["post_count"] >= 2]["Ticker"], thread_flags["author_str"]))
    tags = df.apply(lambda row: "Series" if (row["Ticker"], row["author_str"]) in threads else "Single", axis=1)
    return tags

def fetch_reddit_data(enable_scrape: bool = True) -> pd.DataFrame:
    if not ENABLE_REDDIT_SCRAPE or not enable_scrape:
        logger.info("🔕 Reddit scraping disabled.")
        return pd.DataFrame()
    if reddit is None:
        logger.info("🔕 Reddit API is not configured. Skipping scrape.")
        return pd.DataFrame()

    if not FEATURE_TOGGLES.get("Reddit Sentiment", True):
        logger.info("🔕 Reddit sentiment processing disabled via FEATURE_TOGGLES.")
        return pd.DataFrame()

    # Optional: ignore seen-cache for immediate re-runs (set REDDIT_IGNORE_SEEN=1)
    try:
        if os.getenv("REDDIT_IGNORE_SEEN", "0") == "1":
            _seen_cache.ttl = 0.0
            _seen_cache.data.clear()
            logger.info("[Reddit] Ignoring seen-cache for this run.")
    except Exception:
        pass

    posts = []
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    remaining_global = REDDIT_MAX_POSTS if REDDIT_MAX_POSTS and REDDIT_MAX_POSTS > 0 else None
    fast_categories = None
    if FAST_MODE:
        try:
            fast_categories = [c.strip() for c in FAST_REDDIT_CATEGORIES.split(',') if c.strip()]
        except Exception:
            fast_categories = ["hot", "top"]

    for sub in tqdm(REDDIT_SUBREDDITS, desc="🔎 Scraping Reddit", unit="sub"):
        try:
            subreddit = reddit.subreddit(sub)
            if not _breaker.allow():
                logger.info(f"Circuit open; skipping r/{sub}")
                try:
                    emit_metric("reddit_sub", {"ok": 0, "subreddit": sub, "breaker_open": True})
                except Exception:
                    pass
                continue
            per_sub_count = 0
            for category, limit in REDDIT_LIMITS.items():
                if fast_categories is not None and category not in fast_categories:
                    continue
                # Rate-limit token per listing fetch to avoid bursty access
                _bucket.take(1.0)
                # Try cached IDs first to avoid re-hitting PRAW during dev
                cached_ids = _load_listing_cache(sub, category)
                if cached_ids:
                    posts_iter = (reddit.submission(id=i) for i in cached_ids)
                else:
                    posts_iter = getattr(subreddit, category)(limit=limit)
                ids_seen = []
                for post in posts_iter:
                    try:
                        pid = getattr(post, "id", None)
                        if pid:
                            ids_seen.append(pid)
                    except Exception:
                        pass
                    if remaining_global is not None and remaining_global <= 0:
                        break
                    if REDDIT_MAX_PER_SUB and per_sub_count >= REDDIT_MAX_PER_SUB:
                        break
                    # De-dup within TTL
                    if _seen_cache.has(post.id):
                        continue
                    posts.extend(process_submission(post, sub, now))
                    _seen_cache.add(post.id)
                    per_sub_count += 1
                    if remaining_global is not None:
                        remaining_global -= 1
                    # Gentle pacing to avoid rate-limit bursts
                    if REDDIT_PACING_SEC:
                        time.sleep(REDDIT_PACING_SEC)
                # Save the IDs we just iterated for optional reuse
                try:
                    if ids_seen:
                        _save_listing_cache(sub, category, ids_seen)
                except Exception:
                    pass
                    if remaining_global is not None and remaining_global <= 0:
                        break
                    if REDDIT_MAX_PER_SUB and per_sub_count >= REDDIT_MAX_PER_SUB:
                        break
                    # De-dup within TTL
                    if _seen_cache.has(post.id):
                        continue
                    posts.extend(process_submission(post, sub, now))
                    _seen_cache.add(post.id)
                    per_sub_count += 1
                    if remaining_global is not None:
                        remaining_global -= 1
                    # Gentle pacing to avoid rate-limit bursts
                    if REDDIT_PACING_SEC:
                        time.sleep(REDDIT_PACING_SEC)
                if remaining_global is not None and remaining_global <= 0:
                    break
            logger.info(f"r/{sub} scrape completed.")
            try:
                emit_metric("reddit_sub", {"ok": 1, "subreddit": sub})
            except Exception:
                pass
        except prawcore.exceptions.OAuthException as e:
            logger.error(f"Reddit OAuth failed for r/{sub}: {e}")
            try:
                emit_metric("reddit_sub", {"ok": 0, "subreddit": sub, "error": "oauth"})
            except Exception:
                pass
        except prawcore.exceptions.ResponseException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 429:
                logger.warning(f"Rate limited while scraping r/{sub} (HTTP 429). Consider increasing REDDIT_PACING_SEC.")
                _breaker.record(False)
                try:
                    emit_metric("reddit_sub", {"ok": 0, "subreddit": sub, "status": 429})
                except Exception:
                    pass
            else:
                logger.warning(f"HTTP error while scraping r/{sub}: {status}")
                if status and int(status) >= 500:
                    _breaker.record(False)
                try:
                    emit_metric("reddit_sub", {"ok": 0, "subreddit": sub, "status": status})
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"r/{sub} failed: {e}")
            _breaker.record(False)
            try:
                emit_metric("reddit_sub", {"ok": 0, "subreddit": sub})
            except Exception:
                pass

    # Persist seen cache to disk at the end
    try:
        _seen_cache.flush()
    except Exception:
        pass

    df = pd.DataFrame(posts)
    if df.empty:
        logger.warning("🕳 No Reddit posts collected.")
        return df

    df.drop_duplicates(subset=["Title", "Subreddit", "created"], inplace=True)

    if FEATURE_TOGGLES.get("Post Recency", True):
        df["Emerging"] = is_emerging(df)
    else:
        df["Emerging"] = False

    if FEATURE_TOGGLES.get("Thread Detection", True):
        df["Thread Tag"] = tag_threads(df)
    else:
        df["Thread Tag"] = "Single"

    df.drop(columns=["author_str"], errors="ignore", inplace=True)

    if DEBUG_REDDIT_SCRAPE:
        logger.info("Top tickers:\n" + str(df["Ticker"].value_counts().head(10)))

    return df