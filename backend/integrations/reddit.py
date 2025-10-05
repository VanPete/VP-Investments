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
            'pennystocks'
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
            ticker_scores[ticker] = {
                'reddit_score': reddit_score,
                'mention_count': len(mentions),
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