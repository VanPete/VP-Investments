"""
Reddit Integration (3.0 Architecture - Phase 1+3)
==================================================
Fetch Reddit mentions and calculate social sentiment scores

Phase 1: Fetch Reddit data (scrape subreddits, detect tickers, extract mentions)
Phase 3: Calculate social sentiment score (mention velocity, sentiment, engagement)

Dependencies: PRAW (Reddit API), TextBlob (sentiment)
"""
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from backend.utils.logger import get_logger
from backend.utils.metrics import emit_metric

logger = get_logger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RedditMention:
    """Single Reddit mention of a ticker"""
    ticker: str
    post_id: str
    subreddit: str
    title: str
    body: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    sentiment: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'post_id': self.post_id,
            'subreddit': self.subreddit,
            'title': self.title,
            'body': self.body[:500],  # Truncate for storage
            'author': self.author,
            'score': self.score,
            'num_comments': self.num_comments,
            'created_utc': self.created_utc,
            'url': self.url,
            'sentiment': self.sentiment
        }


@dataclass
class RedditBundle:
    """Collection of Reddit mentions for a ticker"""
    ticker: str
    mentions: List[RedditMention] = field(default_factory=list)
    total_mentions: int = 0
    avg_sentiment: float = 0.0
    total_score: int = 0
    total_comments: int = 0
    subreddit_distribution: Dict[str, int] = field(default_factory=dict)
    fetch_timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_stats(self):
        """Calculate aggregate statistics"""
        if not self.mentions:
            return
        
        self.total_mentions = len(self.mentions)
        self.avg_sentiment = sum(m.sentiment for m in self.mentions) / self.total_mentions
        self.total_score = sum(m.score for m in self.mentions)
        self.total_comments = sum(m.num_comments for m in self.mentions)
        
        # Subreddit distribution
        self.subreddit_distribution = {}
        for mention in self.mentions:
            subreddit = mention.subreddit
            self.subreddit_distribution[subreddit] = self.subreddit_distribution.get(subreddit, 0) + 1


# ============================================================================
# PHASE 1: REDDIT FETCHER
# ============================================================================

class RedditFetcher:
    """Phase 1: Fetch Reddit data from API"""
    
    def __init__(self):
        self.enabled = PRAW_AVAILABLE
        self.reddit = None
        self.ticker_cache = set()  # Known tickers for fast lookup
        
        # Subreddits to monitor
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'StockMarket',
            'options',
            'pennystocks'
        ]
        
        if not self.enabled:
            logger.warning("PRAW not available - Reddit fetching disabled")
            return
        
        try:
            self.reddit = self._initialize_reddit()
            logger.info(f"Reddit fetcher initialized: {len(self.subreddits)} subreddits")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit: {e}")
            self.enabled = False
    
    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize Reddit API connection"""
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'VP-Investments/1.0'),
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )
        
        # Test connection
        reddit.user.me()
        logger.info("Reddit API connection established")
        
        return reddit
    
    def load_ticker_cache(self, tickers: List[str]):
        """Load known tickers into cache for fast lookup"""
        self.ticker_cache = set(ticker.upper() for ticker in tickers)
        logger.info(f"Loaded {len(self.ticker_cache)} tickers into Reddit cache")
    
    def fetch_reddit_bundle(self, ticker: str, lookback_hours: int = 24) -> Optional[RedditBundle]:
        """
        Phase 1: Fetch Reddit mentions for a ticker
        
        Args:
            ticker: Stock ticker symbol
            lookback_hours: Hours to look back for mentions
            
        Returns:
            RedditBundle with all mentions, or None if unavailable
        """
        if not self.enabled or not self.reddit:
            logger.debug(f"Reddit fetching disabled - skipping {ticker}")
            return None
        
        try:
            logger.debug(f"Fetching Reddit data for {ticker}")
            emit_metric("reddit.fetch.start", 1, tags={'ticker': ticker})
            
            # Scrape all configured subreddits
            all_mentions = []
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            for subreddit_name in self.subreddits:
                try:
                    subreddit_mentions = self._scrape_subreddit(
                        subreddit_name, 
                        ticker, 
                        cutoff_time
                    )
                    all_mentions.extend(subreddit_mentions)
                except Exception as e:
                    logger.warning(f"Failed to scrape r/{subreddit_name}: {e}")
                    continue
            
            # Create bundle
            bundle = RedditBundle(ticker=ticker, mentions=all_mentions)
            bundle.calculate_stats()
            
            logger.info(f"Fetched {bundle.total_mentions} Reddit mentions for {ticker}")
            emit_metric("reddit.fetch.success", 1, 
                       tags={'ticker': ticker, 'mentions': bundle.total_mentions})
            
            return bundle
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data for {ticker}: {e}")
            emit_metric("reddit.fetch.error", 1, tags={'ticker': ticker})
            return None
    
    def _scrape_subreddit(self, subreddit_name: str, ticker: str, 
                         cutoff_time: datetime) -> List[RedditMention]:
        """Scrape a single subreddit for ticker mentions"""
        mentions = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Search for recent posts (limit to 100 per subreddit)
            for submission in subreddit.new(limit=100):
                # Check if post is within time window
                post_time = datetime.fromtimestamp(submission.created_utc)
                if post_time < cutoff_time:
                    continue
                
                # Check if ticker is mentioned
                text = f"{submission.title} {submission.selftext}"
                if not self._contains_ticker(text, ticker):
                    continue
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(text)
                
                # Create mention object
                mention = RedditMention(
                    ticker=ticker,
                    post_id=submission.id,
                    subreddit=subreddit_name,
                    title=submission.title,
                    body=submission.selftext[:1000],  # Limit body length
                    author=str(submission.author) if submission.author else '[deleted]',
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    url=f"https://reddit.com{submission.permalink}",
                    sentiment=sentiment
                )
                
                mentions.append(mention)
            
            logger.debug(f"Found {len(mentions)} mentions in r/{subreddit_name}")
            
        except Exception as e:
            logger.warning(f"Error scraping r/{subreddit_name}: {e}")
        
        return mentions
    
    def _contains_ticker(self, text: str, ticker: str) -> bool:
        """Check if text contains the ticker symbol"""
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(ticker.upper()) + r'\b'
        return bool(re.search(pattern, text.upper()))
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob (-1 to 1 scale)"""
        if not TEXTBLOB_AVAILABLE or not text:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return 0.0


# ============================================================================
# PHASE 3: SOCIAL SCORE CALCULATOR
# ============================================================================

class SocialScoreCalculator:
    """Phase 3: Calculate social sentiment score from Reddit data"""
    
    def calculate_social_score(self, bundle: Optional[RedditBundle]) -> float:
        """
        Phase 3: Calculate normalized social score (0-1 scale)
        
        Scoring factors:
        - Mention count (40%): Number of mentions
        - Sentiment (30%): Average sentiment polarity
        - Engagement (30%): Upvotes + comments
        
        Args:
            bundle: RedditBundle with fetched data
            
        Returns:
            Score from 0.0 to 1.0, or 0.5 if no data
        """
        if not bundle or not bundle.mentions:
            logger.debug("No Reddit data - returning neutral score")
            return 0.5
        
        try:
            # Component 1: Mention count (normalized)
            mention_score = self._normalize_mentions(bundle.total_mentions)
            
            # Component 2: Sentiment (convert -1 to 1 → 0 to 1)
            sentiment_score = (bundle.avg_sentiment + 1.0) / 2.0
            
            # Component 3: Engagement (normalized)
            engagement_score = self._normalize_engagement(bundle.total_score, bundle.total_comments)
            
            # Weighted combination
            final_score = (
                mention_score * 0.40 +
                sentiment_score * 0.30 +
                engagement_score * 0.30
            )
            
            # Clamp to [0, 1]
            final_score = max(0.0, min(1.0, final_score))
            
            logger.debug(f"Social score for {bundle.ticker}: {final_score:.3f} "
                        f"(mentions={mention_score:.2f}, sentiment={sentiment_score:.2f}, "
                        f"engagement={engagement_score:.2f})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating social score: {e}")
            return 0.5
    
    def _normalize_mentions(self, mention_count: int) -> float:
        """Normalize mention count to 0-1 scale"""
        # Logarithmic scaling (1-100 mentions → 0-1)
        if mention_count <= 0:
            return 0.0
        elif mention_count >= 100:
            return 1.0
        else:
            # Log scale: log(x+1) / log(101)
            import math
            return math.log(mention_count + 1) / math.log(101)
    
    def _normalize_engagement(self, total_score: int, total_comments: int) -> float:
        """Normalize engagement (upvotes + comments) to 0-1 scale"""
        total_engagement = total_score + total_comments
        
        # Logarithmic scaling (1-1000 engagement → 0-1)
        if total_engagement <= 0:
            return 0.0
        elif total_engagement >= 1000:
            return 1.0
        else:
            import math
            return math.log(total_engagement + 1) / math.log(1001)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

_reddit_fetcher_instance = None
_social_score_calculator_instance = None

def create_reddit_fetcher() -> RedditFetcher:
    """Factory: Create new RedditFetcher instance"""
    return RedditFetcher()

def get_reddit_fetcher() -> RedditFetcher:
    """Factory: Get singleton RedditFetcher instance"""
    global _reddit_fetcher_instance
    if _reddit_fetcher_instance is None:
        _reddit_fetcher_instance = create_reddit_fetcher()
    return _reddit_fetcher_instance

def create_social_score_calculator() -> SocialScoreCalculator:
    """Factory: Create new SocialScoreCalculator instance"""
    return SocialScoreCalculator()

def get_social_score_calculator() -> SocialScoreCalculator:
    """Factory: Get singleton SocialScoreCalculator instance"""
    global _social_score_calculator_instance
    if _social_score_calculator_instance is None:
        _social_score_calculator_instance = create_social_score_calculator()
    return _social_score_calculator_instance
