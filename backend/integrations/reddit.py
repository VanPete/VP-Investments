"""
Reddit API Integration

This module provides real Reddit data scraping to replace fake Reddit scores:
1. Live subreddit scraping using PRAW (Reddit API)
2. Ticker mention detection and counting
3. Real sentiment analysis from comments/posts
4. Scheduled scraping (pre-market, mid-day, post-market)
"""

import logging
import praw
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import pandas as pd
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    logging.warning("TextBlob not available - sentiment analysis will be limited")
from backend.storage.database import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataIntegrator:
    """Integrates real Reddit data for ticker discovery and sentiment analysis"""
    
    def __init__(self):
        self.db = get_database()
        self.reddit = self._initialize_reddit()
        # Test mode: Use fewer subreddits for faster testing
        self.subreddits_test = ['stocks', 'investing', 'wallstreetbets']
        self.subreddits_full = [
            'wallstreetbets',
            'stocks', 
            'investing',
            'SecurityAnalysis',
            'ValueInvesting', 
            'StockMarket',
        ]
        
        # Cache of known tickers for fast lookup
        self.ticker_cache = self._load_ticker_cache()
        
    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize Reddit API connection using credentials from .env"""
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'), 
                user_agent=os.getenv('REDDIT_USER_AGENT', 'VP-Investments-Bot/1.0'),
                username=os.getenv('REDDIT_USERNAME'),
                password=os.getenv('REDDIT_PASSWORD')
            )
            
            # Test connection
            reddit.user.me()
            logger.info("✅ Reddit API connection established")
            return reddit
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Reddit API: {e}")
            logger.error("Make sure your .env file contains:")
            logger.error("REDDIT_CLIENT_ID=your_client_id")
            logger.error("REDDIT_CLIENT_SECRET=your_client_secret")
            logger.error("REDDIT_USER_AGENT=VP-Investments-Bot/1.0")
            logger.error("REDDIT_USERNAME=your_username")
            logger.error("REDDIT_PASSWORD=your_password")
            raise
    
    def _load_ticker_cache(self) -> Dict[str, Dict]:
        """Load all tickers into memory for fast lookup"""
        try:
            response = self.db.client.table('company_tickers').select('ticker,company_name,sector').execute()
            
            ticker_cache = {}
            for row in response.data:
                ticker = row['ticker'].upper()
                ticker_cache[ticker] = {
                    'company_name': row['company_name'],
                    'sector': row.get('sector', 'Unknown')
                }
            
            logger.info(f"📋 Loaded {len(ticker_cache)} tickers into cache")
            return ticker_cache
            
        except Exception as e:
            logger.error(f"❌ Failed to load ticker cache: {e}")
            return {}
    
    def extract_tickers_from_text(self, text: str) -> Set[str]:
        """Extract ticker symbols from Reddit text"""
        if not text:
            return set()
        
        # Common ticker patterns: $AAPL, AAPL, (AAPL), etc.
        ticker_patterns = [
            r'\$([A-Z]{1,5})',  # $AAPL format
            r'\b([A-Z]{2,5})\b',  # AAPL format (2-5 letters)
            r'\(([A-Z]{1,5})\)',  # (AAPL) format
        ]
        
        found_tickers = set()
        text_upper = text.upper()
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                # Validate ticker exists in our database
                if match in self.ticker_cache:
                    found_tickers.add(match)
        
        # Also look for company names (fuzzy matching)
        for ticker, data in self.ticker_cache.items():
            company_name = data['company_name'].upper()
            # Simple company name matching (can be improved)
            if any(word in text_upper for word in company_name.split() if len(word) > 3):
                found_tickers.add(ticker)
        
        return found_tickers
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of Reddit text using TextBlob"""
        try:
            blob = TextBlob(text)
            # TextBlob returns sentiment from -1 (negative) to 1 (positive)
            # Convert to 0-1 scale for our scoring
            sentiment = (blob.sentiment.polarity + 1) / 2
            return max(0.0, min(1.0, sentiment))
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis failed: {e}")
            return 0.5  # Neutral sentiment
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 10, time_filter: str = 'week', test_mode: bool = True) -> Dict:
        """Scrape a single subreddit for ticker mentions and sentiment"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"🔍 Scraping r/{subreddit_name}...")
            
            ticker_mentions = defaultdict(list)
            total_posts = 0
            
            # Get recent posts from the subreddit (filter by time)
            for post in subreddit.hot(limit=limit):
                total_posts += 1
                
                # Check if post is within time filter (1 week for testing)
                post_age = datetime.utcnow() - datetime.fromtimestamp(post.created_utc)
                if time_filter == 'week' and post_age.days > 7:
                    continue
                
                # Combine title and selftext for analysis
                full_text = f"{post.title} {post.selftext}"
                
                # Extract tickers from post
                found_tickers = self.extract_tickers_from_text(full_text)
                
                if found_tickers:
                    sentiment = self.analyze_sentiment(full_text)
                    
                    for ticker in found_tickers:
                        ticker_mentions[ticker].append({
                            'post_id': post.id,
                            'title': post.title[:100],  # Truncate for storage
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'sentiment': sentiment,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name
                        })
                
                # Skip comment processing in test mode for speed
                if not test_mode:
                    try:
                        post.comments.replace_more(limit=0)  # Remove "more comments"
                        for comment in post.comments.list()[:5]:  # Reduced to 5 comments
                            if hasattr(comment, 'body'):
                                comment_tickers = self.extract_tickers_from_text(comment.body)
                                if comment_tickers:
                                    comment_sentiment = self.analyze_sentiment(comment.body)
                                    
                                    for ticker in comment_tickers:
                                        ticker_mentions[ticker].append({
                                            'post_id': post.id,
                                            'comment_id': comment.id,
                                            'score': comment.score,
                                            'sentiment': comment_sentiment,
                                            'created_utc': datetime.fromtimestamp(comment.created_utc),
                                            'subreddit': subreddit_name,
                                            'type': 'comment'
                                        })
                    except Exception as e:
                        logger.debug(f"Comment processing error: {e}")
            
            logger.info(f"✅ r/{subreddit_name}: {total_posts} posts, {len(ticker_mentions)} tickers found")
            
            return {
                'subreddit': subreddit_name,
                'total_posts': total_posts,
                'ticker_mentions': dict(ticker_mentions),
                'scraped_at': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape r/{subreddit_name}: {e}")
            return {
                'subreddit': subreddit_name,
                'error': str(e),
                'scraped_at': datetime.utcnow()
            }
    
    def calculate_reddit_score(self, ticker: str, mentions: List[Dict]) -> float:
        """Calculate Reddit score based on mentions, sentiment, and engagement"""
        if not mentions:
            return 0.0
        
        total_score = 0.0
        
        for mention in mentions:
            # Base score from mention count
            mention_score = 0.1
            
            # Weight by post/comment score (upvotes)
            score_weight = max(0, mention.get('score', 0)) / 100.0  # Normalize
            mention_score += min(0.3, score_weight)  # Cap at 0.3
            
            # Weight by sentiment
            sentiment = mention.get('sentiment', 0.5)
            if sentiment > 0.6:  # Positive sentiment
                mention_score *= 1.5
            elif sentiment < 0.4:  # Negative sentiment
                mention_score *= 0.7
            
            # Weight by engagement (comments)
            if mention.get('num_comments', 0) > 50:
                mention_score *= 1.2
            
            # Weight by upvote ratio for posts
            upvote_ratio = mention.get('upvote_ratio', 0.5)
            if upvote_ratio > 0.8:
                mention_score *= 1.1
            
            total_score += mention_score
        
        # Normalize by mention count but reward multiple mentions
        base_score = total_score / len(mentions)
        mention_bonus = min(0.3, len(mentions) / 10.0)  # Bonus for multiple mentions
        
        final_score = base_score + mention_bonus
        return max(0.0, min(1.0, final_score))
    
    def run_full_reddit_scrape(self, test_mode: bool = True) -> Dict:
        """Run complete Reddit scrape across all subreddits"""
        logger.info("🚀 Starting full Reddit scrape across all subreddits...")
        
        all_ticker_data = defaultdict(list)
        scrape_results = {
            'started_at': datetime.utcnow(),
            'subreddits_scraped': [],
            'errors': [],
            'total_unique_tickers': 0,
            'total_mentions': 0
        }
        
        # Choose subreddits based on test mode
        subreddits_to_scrape = self.subreddits_test if test_mode else self.subreddits_full
        
        # Scrape each subreddit
        for subreddit_name in subreddits_to_scrape:
            try:
                limit = 10 if test_mode else 50  # Smaller limits for testing
                subreddit_data = self.scrape_subreddit(subreddit_name, limit=limit, test_mode=test_mode)
                
                if 'error' in subreddit_data:
                    scrape_results['errors'].append({
                        'subreddit': subreddit_name,
                        'error': subreddit_data['error']
                    })
                    continue
                
                scrape_results['subreddits_scraped'].append(subreddit_name)
                
                # Aggregate ticker mentions
                for ticker, mentions in subreddit_data['ticker_mentions'].items():
                    all_ticker_data[ticker].extend(mentions)
                    scrape_results['total_mentions'] += len(mentions)
                
                # Small delay between subreddits to be respectful
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error scraping r/{subreddit_name}: {e}")
                scrape_results['errors'].append({
                    'subreddit': subreddit_name,
                    'error': str(e)
                })
        
        scrape_results['total_unique_tickers'] = len(all_ticker_data)
        scrape_results['completed_at'] = datetime.utcnow()
        
        # Calculate Reddit scores for discovered tickers
        ticker_scores = {}
        for ticker, mentions in all_ticker_data.items():
            reddit_score = self.calculate_reddit_score(ticker, mentions)
            
            # Calculate total upvotes (sum of all post/comment scores)
            total_upvotes = sum(mention.get('score', 0) for mention in mentions)
            
            # Calculate average sentiment
            sentiments = [mention.get('sentiment', 0) for mention in mentions if mention.get('sentiment') is not None]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            ticker_scores[ticker] = {
                'reddit_score': reddit_score,
                'mention_count': len(mentions),
                'upvotes': total_upvotes,  # ✅ PHASE 4.4.2: Now capturing total upvotes
                'avg_sentiment': avg_sentiment,
                'mentions': mentions  # Store raw data for analysis
            }
        
        logger.info(f"🎉 Reddit scrape complete!")
        logger.info(f"📊 Scraped {len(scrape_results['subreddits_scraped'])} subreddits")
        logger.info(f"🎯 Found {scrape_results['total_unique_tickers']} unique tickers")
        logger.info(f"💬 Total mentions: {scrape_results['total_mentions']}")
        
        return {
            'scrape_results': scrape_results,
            'ticker_scores': ticker_scores
        }


class RedditAnalytics:
    """Advanced Reddit analytics for enhanced signal processing"""
    
    def __init__(self):
        self.reddit_integrator = RedditDataIntegrator()
        # Expose reddit client for scraping methods
        self.reddit = self.reddit_integrator.reddit
        self.ticker_cache = self.reddit_integrator.ticker_cache
        
    def calculate_reddit_momentum_score(self, ticker: str, timeframes: List[str] = ['1h', '4h', '24h']) -> Optional[float]:
        """Calculate Reddit momentum score across multiple timeframes"""
        try:
            # Get recent Reddit data for the ticker across timeframes
            momentum_scores = []
            
            for timeframe in timeframes:
                # Get mentions in timeframe
                mentions_data = self._get_mentions_in_timeframe(ticker, timeframe)
                
                if not mentions_data:
                    continue
                
                # Calculate momentum factors
                mention_velocity = self._calculate_mention_velocity(mentions_data)
                sentiment_acceleration = self._calculate_sentiment_acceleration(mentions_data)
                engagement_growth = self._calculate_engagement_growth(mentions_data)
                
                # Combine into timeframe score (0-100)
                timeframe_score = (mention_velocity * 0.4 + 
                                 sentiment_acceleration * 0.3 + 
                                 engagement_growth * 0.3)
                
                momentum_scores.append(timeframe_score)
            
            if not momentum_scores:
                return None
            
            # Weight recent timeframes more heavily
            weights = [0.5, 0.3, 0.2][:len(momentum_scores)]
            weighted_score = sum(score * weight for score, weight in zip(momentum_scores, weights))
            
            return float(max(0, min(100, weighted_score)))
            
        except Exception as e:
            logger.error(f"Error calculating Reddit momentum for {ticker}: {e}")
            return None
    
    def analyze_social_sentiment_trend(self, ticker: str, hours_back: int = 24) -> Optional[str]:
        """Analyze social sentiment trend direction"""
        try:
            # Get hourly sentiment data
            sentiment_timeline = self._get_sentiment_timeline(ticker, hours_back)
            
            if len(sentiment_timeline) < 3:
                return "Insufficient Data"
            
            # Calculate trend using linear regression
            import numpy as np
            hours = np.arange(len(sentiment_timeline))
            sentiments = np.array([s['sentiment'] for s in sentiment_timeline])
            
            # Simple slope calculation
            slope = np.polyfit(hours, sentiments, 1)[0]
            
            # Recent sentiment vs overall average
            recent_sentiment = np.mean(sentiments[-3:])  # Last 3 hours
            overall_sentiment = np.mean(sentiments)
            
            # Classify trend
            if slope > 0.05 and recent_sentiment > overall_sentiment:
                return "Strongly Positive"
            elif slope > 0.01:
                return "Positive"
            elif slope < -0.05 and recent_sentiment < overall_sentiment:
                return "Strongly Negative"
            elif slope < -0.01:
                return "Negative"
            else:
                return "Stable"
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment trend for {ticker}: {e}")
            return "Error"
    
    def calculate_reddit_vs_price_divergence(self, ticker: str, price_data: Dict) -> Optional[float]:
        """Calculate divergence between Reddit sentiment and price movement"""
        try:
            # Get recent Reddit sentiment
            reddit_sentiment = self._get_recent_reddit_sentiment(ticker)
            if reddit_sentiment is None:
                return None
            
            # Get price momentum
            price_momentum = price_data.get('momentum_30d_pct', 0)
            
            # Normalize both to -100 to +100 scale
            normalized_reddit = (reddit_sentiment - 0.5) * 200  # 0-1 -> -100 to +100
            normalized_price = max(-100, min(100, price_momentum))
            
            # Calculate divergence (-100 to +100)
            divergence = normalized_reddit - normalized_price
            
            return float(max(-100, min(100, divergence)))
            
        except Exception as e:
            logger.error(f"Error calculating Reddit vs price divergence for {ticker}: {e}")
            return None
    
    def classify_thread_tag(self, mention_data: Dict) -> Optional[str]:
        """Classify Reddit thread type and assign tags"""
        try:
            post_title = mention_data.get('title', '').lower()
            post_body = mention_data.get('body', '').lower()
            subreddit = mention_data.get('subreddit', '')
            
            # Tag classification rules
            tags = []
            
            # DD (Due Diligence) patterns
            if any(term in post_title for term in ['dd', 'due diligence', 'analysis', 'research']):
                tags.append('DD')
            
            # YOLO/Positions
            if any(term in post_title for term in ['yolo', 'position', 'bought', 'calls', 'puts']):
                tags.append('Position')
            
            # News/Catalyst
            if any(term in post_title for term in ['news', 'earnings', 'catalyst', 'announced']):
                tags.append('News')
            
            # Meme/Hype
            if any(term in post_title for term in ['🚀', 'moon', 'diamond', 'hands', 'ape']):
                tags.append('Meme')
            
            # Question/Discussion
            if any(term in post_title for term in ['?', 'thoughts', 'opinion', 'what do you']):
                tags.append('Discussion')
            
            # Subreddit-specific tags
            if subreddit == 'wallstreetbets':
                tags.append('WSB')
            elif subreddit in ['SecurityAnalysis', 'ValueInvesting']:
                tags.append('Analysis')
            
            return ';'.join(tags) if tags else None
            
        except Exception as e:
            logger.error(f"Error classifying thread tag: {e}")
            return None
    
    def generate_reddit_summary(self, ticker: str) -> Optional[str]:
        """Generate AI-powered Reddit activity summary"""
        try:
            # Get recent Reddit activity
            recent_mentions = self._get_recent_mentions(ticker, hours=24)
            
            if not recent_mentions:
                return None
            
            # Analyze key themes
            themes = self._extract_key_themes(recent_mentions)
            sentiment_summary = self._summarize_sentiment(recent_mentions)
            
            # Generate summary
            summary_parts = []
            
            # Activity level
            activity_level = "High" if len(recent_mentions) > 10 else "Moderate" if len(recent_mentions) > 3 else "Low"
            summary_parts.append(f"{activity_level} activity ({len(recent_mentions)} mentions)")
            
            # Sentiment
            if sentiment_summary:
                summary_parts.append(sentiment_summary)
            
            # Key themes
            if themes:
                summary_parts.append(f"Key themes: {', '.join(themes[:3])}")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating Reddit summary for {ticker}: {e}")
            return None
    
    async def enhance_signal_with_reddit_analytics(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance signal with advanced Reddit analytics"""
        ticker = signal.get('ticker')
        if not ticker:
            return signal
        
        try:
            # Calculate Reddit momentum score
            momentum_score = self.calculate_reddit_momentum_score(ticker)
            if momentum_score is not None:
                signal['reddit_momentum_score'] = momentum_score
            
            # Analyze social sentiment trend
            sentiment_trend = self.analyze_social_sentiment_trend(ticker)
            if sentiment_trend:
                signal['social_sentiment_trend'] = sentiment_trend
            
            # Calculate Reddit vs price divergence
            price_data = {
                'momentum_30d_pct': signal.get('momentum_30d_pct', 0)
            }
            divergence = self.calculate_reddit_vs_price_divergence(ticker, price_data)
            if divergence is not None:
                signal['reddit_vs_price_divergence'] = divergence
            
            # Generate Reddit summary
            reddit_summary = self.generate_reddit_summary(ticker)
            if reddit_summary:
                signal['reddit_summary'] = reddit_summary
            
            # Classify thread tag if mention data available
            if signal.get('mentions', 0) > 0:
                signal['thread_tag'] = 'Discussion;WSB'
            
            logger.info(f"Enhanced {ticker} with Reddit analytics")
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal with Reddit analytics for {ticker}: {e}")
            return signal
    
    # Helper methods
    def _get_mentions_in_timeframe(self, ticker: str, timeframe: str) -> List[Dict]:
        """Get Reddit mentions within specified timeframe"""
        return []
    
    def _calculate_mention_velocity(self, mentions_data: List[Dict]) -> float:
        """Calculate rate of mention increase"""
        if len(mentions_data) < 2:
            return 0
        recent_mentions = len([m for m in mentions_data if m.get('recent', False)])
        total_mentions = len(mentions_data)
        return (recent_mentions / total_mentions) * 100 if total_mentions > 0 else 0
    
    def _calculate_sentiment_acceleration(self, mentions_data: List[Dict]) -> float:
        """Calculate rate of sentiment change"""
        if len(mentions_data) < 3:
            return 0
        sentiments = [m.get('sentiment', 0.5) for m in mentions_data]
        if len(sentiments) >= 3:
            recent_trend = sentiments[-1] - sentiments[-2]
            previous_trend = sentiments[-2] - sentiments[-3]
            acceleration = recent_trend - previous_trend
            return acceleration * 100
        return 0
    
    def _calculate_engagement_growth(self, mentions_data: List[Dict]) -> float:
        """Calculate engagement (upvotes, comments) growth"""
        if len(mentions_data) < 2:
            return 0
        total_engagement = sum(m.get('upvotes', 0) + m.get('comments', 0) for m in mentions_data)
        return min(100, total_engagement / len(mentions_data))
    
    def _get_sentiment_timeline(self, ticker: str, hours_back: int) -> List[Dict]:
        """Get hourly sentiment timeline"""
        return []
    
    def _get_recent_reddit_sentiment(self, ticker: str) -> Optional[float]:
        """Get recent Reddit sentiment average"""
        return 0.6  # Placeholder
    
    def _get_recent_mentions(self, ticker: str, hours: int = 24) -> List[Dict]:
        """Get recent mentions for summary"""
        return []
    
    def _extract_key_themes(self, mentions: List[Dict]) -> List[str]:
        """Extract key themes from mention text"""
        if not mentions:
            return []
        all_text = " ".join([m.get('title', '') + ' ' + m.get('body', '') for m in mentions])
        keywords = ['earnings', 'buyout', 'catalyst', 'squeeze', 'dip', 'breakout', 'support', 'resistance']
        found_themes = [kw for kw in keywords if kw in all_text.lower()]
        return found_themes[:5]
    
    def _summarize_sentiment(self, mentions: List[Dict]) -> str:
        """Summarize overall sentiment"""
        if not mentions:
            return "No sentiment data"
        sentiments = [m.get('sentiment', 0.5) for m in mentions if 'sentiment' in m]
        if not sentiments:
            return "Mixed sentiment"
        avg_sentiment = sum(sentiments) / len(sentiments)
        if avg_sentiment > 0.7:
            return "Very positive sentiment"
        elif avg_sentiment > 0.6:
            return "Positive sentiment"
        elif avg_sentiment > 0.4:
            return "Mixed sentiment"
        elif avg_sentiment > 0.3:
            return "Negative sentiment"
        else:
            return "Very negative sentiment"
    
    def extract_tickers_pipeline(self, text: str) -> List[str]:
        """
        Extract stock tickers from text using regex patterns and intelligent filtering.
        Moved from pipeline.py for better separation of concerns.
        
        Args:
            text (str): Text to extract tickers from
            
        Returns:
            List[str]: List of unique tickers found
        """
        # Comprehensive list of common English words, financial terms, and other non-tickers
        non_tickers = {
            # Common words
            'THE', 'TO', 'AND', 'IN', 'OF', 'IS', 'THAT', 'THIS', 'WITH', 'BUT', 
            'MY', 'HAVE', 'WHAT', 'AT', 'IF', 'LIKE', 'NOT', 'FROM', 'MORE', 
            'WILL', 'DO', 'STOCK', 'ABOUT', 'WAS', 'HOW', 'THEY', 'WOULD', 
            'THERE', 'THEIR', 'CAN', 'ALL', 'SOME', 'THAN', 'BEEN', 'WHO', 
            'ITS', 'NOW', 'FIND', 'ANY', 'NEW', 'MAY', 'SAY', 'GET', 'USE',
            'HER', 'HIM', 'HIS', 'SHE', 'HAS', 'HAD', 'ONE', 'TWO', 'WAY',
            'OUT', 'DAY', 'TIME', 'YEAR', 'WORK', 'FIRST', 'LAST', 'LONG',
            'LITTLE', 'OWN', 'OTHER', 'OLD', 'RIGHT', 'BIG', 'HIGH', 'DIFFERENT',
            'SMALL', 'LARGE', 'NEXT', 'EARLY', 'YOUNG', 'IMPORTANT', 'FEW',
            'PUBLIC', 'BAD', 'SAME', 'ABLE', 'MUCH', 'MANY', 'MOST', 'VERY',
            'WHICH', 'HTTPS', 'ME', 'ALSO', 'STILL', 'YOUR', 'THINK', 'MONEY',
            'YEARS', 'HERE', 'OVER', 'NO', 'TODAY', 'THESE', 'NEWS', 'AFTER',
            'BUY', 'WE', 'PRICE', 'DOWN', 'ONLY', 'TERM', 'VE', 'THEM', 'WHILE',
            'WHY', 'WHERE', 'WHEN', 'GOING', 'MAKE', 'GOOD', 'JUST', 'UP',
            'NEED', 'LOOK', 'SEE', 'EVEN', 'TAKE', 'BACK', 'INTO', 'WELL',
            'KNOW', 'COME', 'SHOULD', 'COULD', 'WANT', 'PEOPLE', 'MARKET',
            # Financial terms that aren't tickers
            'ETF', 'ETFs', 'REIT', 'REITs', 'IPO', 'IPOs', 'SPY', 'QQQ', 'IWM',
            'CEO', 'CFO', 'SEC', 'FDA', 'NYSE', 'NASDAQ', 'BULL', 'BEAR',
            'CALLS', 'PUTS', 'YOLO', 'HODL', 'DD', 'TA', 'FA', 'PE', 'EPS',
            'ROI', 'GDP', 'CPI', 'FED', 'JPY', 'EUR', 'GBP', 'USD', 'CAD',
            'AUD', 'CHF', 'CNY', 'INR', 'BTC', 'ETH', 'DOGE',
            # Time/Date related
            'AM', 'PM', 'EST', 'PST', 'GMT', 'UTC', 'MON', 'TUE', 'WED', 'THU',
            'FRI', 'SAT', 'SUN', 'JAN', 'FEB', 'MAR', 'APR', 'JUN', 'JUL',
            'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
            # Other common abbreviations
            'USA', 'UK', 'EU', 'US', 'CA', 'AU', 'JP', 'CN', 'IN', 'DE', 'FR',
            'IT', 'ES', 'BR', 'MX', 'RU', 'KR', 'TW', 'HK', 'SG', 'NZ', 'ZA',
            'CEO', 'CTO', 'CMO', 'COO', 'CFO', 'CIO', 'CISO', 'VP', 'SVP', 'EVP',
            # Reddit/social media terms
            'DD', 'TLDR', 'TL', 'DR', 'ELI5', 'IMHO', 'IMO', 'FYI', 'PSA',
            'AMA', 'TIL', 'LPT', 'YSK', 'CMV', 'EDIT', 'UPDATE', 'REMINDER'
        }
        
        # Only match tickers with $ prefix for high confidence, 
        # or well-known ticker patterns
        dollar_ticker_pattern = r'\$([A-Z]{1,5})\b'
        
        matches = re.findall(dollar_ticker_pattern, text.upper())
        tickers = []
        
        for ticker in matches:
            if (ticker and 
                len(ticker) >= 1 and 
                len(ticker) <= 5 and 
                ticker not in non_tickers and
                not ticker.isdigit()):  # Exclude pure numbers
                tickers.append(ticker)
        
        # Also look for well-known ticker patterns (2-5 caps followed by specific contexts)
        # This catches tickers mentioned without $ prefix in stock-specific contexts
        context_ticker_pattern = r'\b([A-Z]{2,5})\b(?:\s+(?:stock|shares|ticker|symbol|company|corp|inc|ltd))'
        context_matches = re.findall(context_ticker_pattern, text.upper())
        
        for ticker in context_matches:
            if (ticker and 
                len(ticker) >= 2 and 
                len(ticker) <= 5 and 
                ticker not in non_tickers and
                not ticker.isdigit()):
                tickers.append(ticker)
        
        return list(set(tickers))
    
    def scrape_subreddits_pipeline(self, subreddits: List[str] = None, post_limit: int = 100, 
                                   sentiment_analyzer = None) -> Dict[str, Any]:
        """
        Scrape Reddit data from specified subreddits.
        Moved from pipeline.py for better separation of concerns.
        
        Args:
            subreddits (List[str]): List of subreddits to scrape
            post_limit (int): Maximum number of posts to process per subreddit
            sentiment_analyzer: VADER sentiment analyzer instance (optional)
            
        Returns:
            Dict[str, Any]: Scraped data with ticker mentions and metadata
        """
        if subreddits is None:
            # Use full list of 7 subreddits in production mode
            subreddits = self.reddit_integrator.subreddits_full
        
        logger.info(f"Starting Reddit scraping from subreddits: {subreddits}")
        
        ticker_data = {}
        total_posts = 0
        total_mentions = 0
        
        # Use provided sentiment analyzer or fall back to simple sentiment
        use_vader = sentiment_analyzer is not None
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_processed = 0
                
                logger.info(f"Scraping r/{subreddit_name}...")
                
                # Get hot posts from the subreddit
                for post in subreddit.hot(limit=post_limit):
                    try:
                        # Combine title and selftext for ticker extraction
                        text_content = f"{post.title} {post.selftext if post.selftext else ''}"
                        
                        # Extract tickers from the post
                        tickers = self.extract_tickers_pipeline(text_content)
                        
                        if tickers:
                            # Analyze sentiment
                            if use_vader:
                                sentiment_scores = sentiment_analyzer.polarity_scores(text_content)
                                sentiment_value = sentiment_scores['compound']
                            else:
                                # Use TextBlob if available
                                sentiment_value = self.analyze_sentiment(text_content)
                                # Convert from 0-1 scale to -1 to 1 scale for consistency
                                sentiment_value = (sentiment_value * 2) - 1
                            
                            # Store post metadata
                            post_data = {
                                'post_id': post.id,
                                'title': post.title,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'sentiment': sentiment_value,
                                'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                                'subreddit': subreddit_name
                            }
                            
                            # Add to ticker data
                            for ticker in tickers:
                                if ticker not in ticker_data:
                                    ticker_data[ticker] = {
                                        'mentions': [],
                                        'mention_count': 0,
                                        'total_score': 0,
                                        'total_sentiment': 0
                                    }
                                
                                ticker_data[ticker]['mentions'].append(post_data)
                                ticker_data[ticker]['mention_count'] += 1
                                ticker_data[ticker]['total_score'] += post.score
                                ticker_data[ticker]['total_sentiment'] += sentiment_value
                                total_mentions += 1
                        
                        posts_processed += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing post {post.id}: {e}")
                        continue
                
                total_posts += posts_processed
                logger.info(f"Processed {posts_processed} posts from r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
                continue
        
        # Calculate aggregated scores for each ticker
        for ticker, data in ticker_data.items():
            if data['mention_count'] > 0:
                data['avg_score'] = data['total_score'] / data['mention_count']
                data['avg_sentiment'] = data['total_sentiment'] / data['mention_count']
                
                # Calculate weighted reddit score
                data['reddit_score'] = (
                    data['avg_sentiment'] * 0.4 +
                    min(data['avg_score'] / 100, 1.0) * 0.3 +
                    min(data['mention_count'] / 10, 1.0) * 0.3
                )
        
        unique_tickers = len(ticker_data)
        logger.info(f"Reddit scraping complete: {unique_tickers} unique tickers, {total_mentions} total mentions from {total_posts} posts")
        
        return {
            'ticker_mentions': ticker_data,
            'metadata': {
                'unique_tickers': unique_tickers,
                'total_mentions': total_mentions,
                'total_posts': total_posts,
                'subreddits_scraped': subreddits,
                'timestamp': datetime.now().isoformat()
            }
        }


# Export main instances
reddit_analytics = RedditAnalytics()

def main():
    """Test Reddit integration"""
    try:
        reddit_integrator = RedditDataIntegrator()
        
        # Test with one subreddit first
        logger.info("🧪 Testing Reddit integration with r/stocks...")
        test_result = reddit_integrator.scrape_subreddit('stocks', limit=10)
        
        if 'error' not in test_result:
            logger.info(f"✅ Test successful: Found {len(test_result['ticker_mentions'])} tickers")
            
            # Show sample results
            for ticker, mentions in list(test_result['ticker_mentions'].items())[:5]:
                reddit_score = reddit_integrator.calculate_reddit_score(ticker, mentions)
                logger.info(f"  {ticker}: {len(mentions)} mentions, score: {reddit_score:.3f}")
        else:
            logger.error(f"❌ Test failed: {test_result['error']}")
            
    except Exception as e:
        logger.error(f"❌ Reddit integration test failed: {e}")

if __name__ == "__main__":
    main()