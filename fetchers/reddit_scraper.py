"""
reddit_scraper.py
Scrapes Reddit posts from configured subreddits, filters by quality, extracts sentiment, and identifies signals.
"""

import os
import re
import logging
import datetime
from math import log10
from typing import List, Dict

import pandas as pd
import praw
from praw.models import Submission
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import warnings

from config.config import (
    REDDIT_SUBREDDITS, REDDIT_LIMITS, ENABLE_REDDIT_SCRAPE, DEBUG_REDDIT_SCRAPE,
    REDDIT_TICKER_PATTERN, SUBREDDIT_WEIGHTS, KEYWORD_BOOSTS, FLAIR_BOOSTS,
    FLAIR_ALIASES, MIN_AUTHOR_KARMA, TITLE_COMMENT_BLEND, COMMENT_WEIGHT_SCALING, EXCLUDED_TICKERS, FEATURE_TOGGLES
)
from utils.logger import setup_logging

# === Init ===
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*comment_sort.*already been fetched.*")

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

try:
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error("VADER sentiment analyzer failed to initialize.")
    raise e


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
        if not post.comments:
            post.comment_sort = "top"
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
        if not post.comments:
            post.comment_sort = "top"
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
        if not tickers:
            return []

        results = []
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
            results.append({
                "Ticker": ticker,
                "Subreddit": sub,
                "Title": post.title,
                "Upvotes": post.score,
                "created": created_time,
                "Reddit Sentiment": sentiment,
                "Reddit Summary": summary,
                "Author": str(author),
                "Post URL": f"https://reddit.com{post.permalink}"
            })
        return results
    except Exception as e:
        logger.warning(f"Post process failed: {getattr(post, 'id', 'unknown')} → {e}")
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

    if not FEATURE_TOGGLES.get("Reddit Sentiment", True):
        logger.info("🔕 Reddit sentiment processing disabled via FEATURE_TOGGLES.")
        return pd.DataFrame()

    posts = []
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    for sub in tqdm(REDDIT_SUBREDDITS, desc="🔎 Scraping Reddit", unit="sub"):
        try:
            subreddit = reddit.subreddit(sub)
            for category, limit in REDDIT_LIMITS.items():
                for post in getattr(subreddit, category)(limit=limit):
                    posts.extend(process_submission(post, sub, now))
            logger.info(f"r/{sub} scrape completed.")
        except Exception as e:
            logger.warning(f"r/{sub} failed: {e}")

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