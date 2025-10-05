"""
VP Investments Unified Pipeline
===============================

This is the consolidated pipeline that replaces all other testing pipelines and provides
a single, comprehensive data gathering and processing system with real data integration.

Features:
- Reddit scraping from multiple subreddits
- Yahoo Finance data retrieval with caching
- Sentiment analysis using VADER
- Supabase database persistence
- Configurable limits and error handling
- Comprehensive logging
"""

import os
import sys
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enhanced integrations flag
ENHANCED_INTEGRATIONS = True

# Setup VP Investments logging (logs to logs/ folder)
from backend.utils.logger import setup_logging, get_logger

# Configure logging to use logs folder
setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    structured_logging=False
)
logger = get_logger(__name__)

# Import integrators from backend.integrations
try:
    from backend.integrations import get_technical_calculator, get_financial_calculator, get_signal_classifier
    INTEGRATORS_AVAILABLE = True
    logger.info("Integrators loaded successfully")
except ImportError as e:
    INTEGRATORS_AVAILABLE = False
    logger.warning(f"Integrators not available: {e}")# Simple configuration class
class Config:
    """Simple configuration class for pipeline settings."""
    
    def __init__(self):
        self.reddit_post_limit = 100
        self.min_mentions = 1
        self.max_signals = 50
    
    def get(self, key, default=None):
        """Get configuration value with dot notation support"""
        # Support environment variables for scoring weights
        if key == 'scoring.weights':
            from dotenv import load_dotenv
            load_dotenv()
            return {
                'financial': float(os.getenv('SCORE_WEIGHT_FINANCIAL', '0.4')),
                'technical': float(os.getenv('SCORE_WEIGHT_TECHNICAL', '0.2')),
                'options': float(os.getenv('SCORE_WEIGHT_OPTIONS', '0.15')),
                'short_interest': float(os.getenv('SCORE_WEIGHT_SHORT', '0.15')),
                'reddit': float(os.getenv('SCORE_WEIGHT_REDDIT', '0.1')),
                'news': float(os.getenv('SCORE_WEIGHT_NEWS', '0.0'))  # Disabled
            }
        return default

class UnifiedPipeline:
    """
    Unified pipeline that consolidates Reddit scraping, financial data retrieval,
    and database storage into a single, production-ready system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the unified pipeline with configuration."""
        self.config = config or Config()
        self.logger = logger
        
        # Initialize components
        self._init_reddit()
        self._init_finance()
        self._init_database()
        self._init_sentiment()
        self._init_enhanced_integrations()
        
    def _init_reddit(self):
        """Initialize Reddit API connection."""
        try:
            import praw
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'VP_Investments_Bot/1.0')
            )
            
            # Test connection
            try:
                self.reddit.user.me()
                self.logger.info("Reddit API connection established (authenticated)")
            except:
                self.logger.info("Reddit API connection established (read-only)")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit API: {e}")
            raise
    
    def _init_finance(self):
        """Initialize Yahoo Finance data retrieval."""
        try:
            import yfinance as yf
            self.yf = yf
            self.logger.info("Yahoo Finance initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Yahoo Finance: {e}")
            raise
    
    def _init_database(self):
        """Initialize Supabase database connection."""
        try:
            from supabase import create_client, Client
            
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase credentials not found in environment variables")
            
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.logger.info("Supabase connection initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase: {e}")
            raise
    
    def _init_sentiment(self):
        """Initialize sentiment analysis."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    def _init_enhanced_integrations(self):
        """Initialize enhanced integrations (news, AI, advanced finance)."""
        self.enhanced_available = False
        self.news_integrator = None
        self.ai_integrator = None
        self.finance_fetcher = None
        
        # Try to import enhanced integrations directly
        try:
            from backend.integrations.news import NewsIntegrator
            self.news_integrator = NewsIntegrator()
            self.logger.info("News integrator loaded successfully")
        except ImportError as e:
            self.logger.warning(f"News integrator not available: {e}")
        except Exception as e:
            self.logger.warning(f"News integrator initialization failed: {e}")
            
        try:
            from backend.integrations.ai import AIIntegrator
            self.ai_integrator = AIIntegrator()
            self.logger.info("AI integrator loaded successfully")
        except ImportError as e:
            self.logger.warning(f"AI integrator not available: {e}")
        except Exception as e:
            self.logger.warning(f"AI integrator initialization failed: {e}")
            
        # Check if we have at least one enhanced integration
        if any([self.news_integrator, self.ai_integrator]):
            self.enhanced_available = True
            self.logger.info("Enhanced integrations available")
        else:
            self.logger.info("Enhanced integrations not available, using basic mode")
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text using regex patterns and intelligent filtering.
        
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
    
    def scrape_reddit_data(self, subreddits: List[str] = None, post_limit: int = 100) -> Dict[str, Any]:
        """
        Scrape Reddit data from specified subreddits.
        
        Args:
            subreddits (List[str]): List of subreddits to scrape
            post_limit (int): Maximum number of posts to process per subreddit
            
        Returns:
            Dict[str, Any]: Scraped data with ticker mentions and metadata
        """
        if subreddits is None:
            subreddits = ['stocks', 'investing', 'wallstreetbets']
        
        self.logger.info(f"Starting Reddit scraping from subreddits: {subreddits}")
        
        ticker_data = {}
        total_posts = 0
        total_mentions = 0
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_processed = 0
                
                self.logger.info(f"Scraping r/{subreddit_name}...")
                
                # Get hot posts from the subreddit
                for post in subreddit.hot(limit=post_limit):
                    try:
                        # Combine title and selftext for ticker extraction
                        text_content = f"{post.title} {post.selftext if post.selftext else ''}"
                        
                        # Extract tickers from the post
                        tickers = self.extract_tickers(text_content)
                        
                        if tickers:
                            # Analyze sentiment
                            sentiment_scores = self.sentiment_analyzer.polarity_scores(text_content)
                            
                            # Store post metadata
                            post_data = {
                                'post_id': post.id,
                                'title': post.title,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'sentiment': sentiment_scores['compound'],
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
                                ticker_data[ticker]['total_sentiment'] += sentiment_scores['compound']
                                total_mentions += 1
                        
                        posts_processed += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing post {post.id}: {e}")
                        continue
                
                total_posts += posts_processed
                self.logger.info(f"Processed {posts_processed} posts from r/{subreddit_name}")
                
            except Exception as e:
                self.logger.error(f"Error scraping r/{subreddit_name}: {e}")
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
        self.logger.info(f"Reddit scraping complete: {unique_tickers} unique tickers, {total_mentions} total mentions from {total_posts} posts")
        
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
    
    def get_financial_data(self, ticker: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive financial data for a ticker with all requested metrics.
        
        Args:
            ticker (str): Stock ticker symbol
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            Optional[Dict[str, Any]]: Comprehensive financial data or None if unavailable
        """
        try:
            # Initialize integrators if available
            if INTEGRATORS_AVAILABLE:
                technical_calc = get_technical_calculator()
                financial_calc = get_financial_calculator()
                
                # Get comprehensive financial data
                financial_data = financial_calc.get_comprehensive_financial_data(ticker)
                
                # Get technical indicators
                technical_data = technical_calc.calculate_all_indicators(ticker)
                
                # Merge technical data into financial data
                financial_data.update(technical_data)
                
                # Ensure current_price is not limited to 10 (remove artificial limits)
                if financial_data.get('current_price'):
                    financial_data['current_price'] = round(float(financial_data['current_price']), 2)
                
                return financial_data
            else:
                # Fallback to enhanced basic method
                return self._get_enhanced_financial_data(ticker)
                
        except Exception as e:
            self.logger.error(f"Comprehensive financial data failed for {ticker}: {e}")
            # Final fallback
            return self._get_basic_financial_data(ticker)
    
    def _get_basic_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Basic financial data fallback method."""
        try:
            stock = self.yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get recent price data
            hist = stock.history(period='5d')
            
            if hist.empty or not info:
                return None
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close and current_price else 0
            
            # Calculate additional metrics
            avg_volume = info.get('averageVolume', 1)
            current_volume = int(hist['Volume'].iloc[-1]) if not hist.empty else 1
            volume_spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Extract key financial metrics
            financial_data = {
                'ticker': ticker,
                'company': info.get('shortName', info.get('longName', ticker)),
                'current_price': round(float(current_price), 2) if current_price else None,
                'price_1d_pct': round(float(price_change), 2),
                'market_cap': info.get('marketCap'),
                'volume': current_volume,
                'volume_spike_ratio': round(volume_spike_ratio, 2),
                'pe_ratio': round(info.get('trailingPE'), 2) if info.get('trailingPE') else None,
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': round(info.get('beta'), 2) if info.get('beta') else None,
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': avg_volume,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                # Financial ratios
                'roe': round(info.get('returnOnEquity') * 100, 2) if info.get('returnOnEquity') else None,
                'debt_equity': round(info.get('debtToEquity'), 2) if info.get('debtToEquity') else None,
                'eps_growth': round(info.get('earningsGrowth') * 100, 2) if info.get('earningsGrowth') else None,
                'short_pct_float': round(info.get('shortPercentOfFloat') * 100, 2) if info.get('shortPercentOfFloat') else None,
                'shares_short': info.get('sharesShort'),
                'timestamp': datetime.now().isoformat()
            }
            
            return financial_data
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve basic financial data for {ticker}: {e}")
            return None
            return None
    
    def _get_enhanced_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get enhanced financial data using the advanced finance fetcher."""
        try:
            # This would be async in the real implementation, but for now we'll simulate
            # Since the finance fetcher is async, we'll need to adapt
            
            # For now, create a comprehensive data structure that mimics what the 
            # advanced fetcher would provide
            stock = self.yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1y')  # Get more data for technical analysis
            
            if hist.empty or not info:
                return None
            
            # Calculate enhanced metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            # Price changes
            price_1d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            price_7d_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8] * 100) if len(hist) > 7 else 0
            
            # Volume analysis
            volume = hist['Volume'].iloc[-1] if not hist.empty else 0
            avg_volume_10d = hist['Volume'].tail(10).mean() if len(hist) >= 10 else volume
            volume_spike_ratio = (volume / avg_volume_10d) if avg_volume_10d > 0 else 1.0
            
            # Technical indicators
            close_prices = hist['Close']
            
            # Moving averages
            ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else None
            ma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else None
            
            above_50_ma_pct = ((current_price - ma_50) / ma_50 * 100) if ma_50 else None
            above_200_ma_pct = ((current_price - ma_200) / ma_200 * 100) if ma_200 else None
            
            # RSI calculation (simplified)
            def calculate_rsi(prices, periods=14):
                if len(prices) < periods + 1:
                    return None
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs)).iloc[-1]
            
            rsi = calculate_rsi(close_prices)
            
            # Volatility
            volatility = close_prices.pct_change().rolling(20).std().iloc[-1] * 100 if len(close_prices) >= 20 else None
            
            # Comprehensive financial data
            enhanced_data = {
                'ticker': ticker,
                'company': info.get('longName'),
                'sector': info.get('sector'),
                'current_price': float(current_price) if current_price else None,
                'price_1d_pct': float(price_1d_pct),
                'price_7d_pct': float(price_7d_pct),
                'market_cap': info.get('marketCap'),
                'volume': int(volume) if volume else None,
                'volume_spike_ratio': float(volume_spike_ratio),
                'above_50d_ma_pct': float(above_50_ma_pct) if above_50_ma_pct else None,
                'above_200d_ma_pct': float(above_200_ma_pct) if above_200_ma_pct else None,
                'pe_ratio': info.get('trailingPE'),
                'eps_growth': info.get('earningsGrowth'),
                'roe': info.get('returnOnEquity'),
                'debt_equity': info.get('debtToEquity'),
                'rsi': float(rsi) if rsi else None,
                'volatility': float(volatility) if volatility else None,
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'short_pct_float': info.get('shortPercentOfFloat'),
                'shares_short': info.get('sharesShort'),
                'timestamp': datetime.now().isoformat()
            }
            
            return enhanced_data
            
        except Exception as e:
            self.logger.warning(f"Enhanced financial data failed for {ticker}: {e}")
            return None
    
    async def get_news_data(self, ticker: str) -> Dict[str, Any]:
        """Get news sentiment data for a ticker."""
        if self.enhanced_available and self.news_integrator:
            try:
                return await self.news_integrator.get_news_sentiment(ticker)
            except Exception as e:
                self.logger.warning(f"News data failed for {ticker}: {e}")
        
        return {
            'news_score': None,
            'news_sentiment': None,
            'news_mentions': 0,
            'ai_news_summary': None
        }
    
    async def get_ai_commentary(self, ticker: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-generated commentary for a signal."""
        if self.enhanced_available and self.ai_integrator:
            try:
                return await self.ai_integrator.generate_signal_commentary(ticker, signal_data)
            except Exception as e:
                self.logger.warning(f"AI commentary failed for {ticker}: {e}")
        
        return {
            'ai_commentary': None,
            'ai_trends_commentary': None,
            'score_explanation': None
        }
    
    def calculate_signal_score(self, ticker: str, reddit_data: Dict[str, Any], financial_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate a weighted signal score combining Reddit and financial metrics.
        
        Args:
            ticker (str): Stock ticker symbol
            reddit_data (Dict[str, Any]): Reddit mention data
            financial_data (Optional[Dict[str, Any]]): Financial metrics
            
        Returns:
            float: Weighted signal score between 0 and 1
        """
        try:
            # Base Reddit score (40% weight)
            reddit_score = reddit_data.get('reddit_score', 0) * 0.4
            
            # Financial momentum score (30% weight)
            financial_score = 0
            if financial_data:
                # Price momentum
                price_change = financial_data.get('price_change_pct', 0)
                momentum_score = min(abs(price_change) / 10, 1.0) if price_change else 0
                
                # Volume factor
                volume = financial_data.get('volume', 0)
                avg_volume = financial_data.get('avg_volume', 1)
                volume_factor = min(volume / avg_volume, 2.0) if avg_volume else 1.0
                
                financial_score = (momentum_score * volume_factor) * 0.3
            
            # Mention frequency score (20% weight)
            mention_count = reddit_data.get('mention_count', 0)
            frequency_score = min(mention_count / 5, 1.0) * 0.2
            
            # Sentiment consistency score (10% weight)
            sentiment_score = max(reddit_data.get('avg_sentiment', 0), 0) * 0.1
            
            # Combine all scores
            total_score = reddit_score + financial_score + frequency_score + sentiment_score
            
            return min(max(total_score, 0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.warning(f"Error calculating signal score for {ticker}: {e}")
            return 0.0
    
    async def save_signals_to_database(self, signals: List[Dict[str, Any]]) -> bool:
        """
        Save processed signals to the Supabase database with comprehensive data population.
        
        Args:
            signals (List[Dict[str, Any]]): List of signal data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not signals:
                self.logger.warning("No signals to save")
                return False
            
            # Create a run record in the runs table
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_run_id = timestamp
            
            try:
                run_record = {
                    'run_id': unique_run_id,
                    'run_type': 'unified_pipeline',
                    'started_at': datetime.now().isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'total_signals': len(signals),
                    'status': 'completed',
                    'error_message': None,
                    'metadata': {
                        'signals_count': len(signals),
                        'pipeline_version': '1.0',
                        'subreddits': ['stocks', 'investing', 'wallstreetbets']
                    }
                }
                run_result = self.supabase.table('runs').insert(run_record).execute()
                
                if run_result.data:
                    db_id = run_result.data[0]['id']  # The auto-generated database ID
                    run_id = unique_run_id  # Use the string run_id for foreign key
                    self.logger.info(f"Created run record with database ID: {db_id}, run_id: {unique_run_id}")
                else:
                    raise ValueError("No run data returned")
                    
            except Exception as run_error:
                self.logger.error(f"Failed to create run record: {run_error}")
                return False
            
            # Prepare data for both tables using new signal structure (simplified for testing)
            current_time = datetime.now()
            
            self.logger.info(f"Processing {len(signals)} signals for database storage...")
            
            # Prepare signals_norm records (primary scored signals table)
            norm_records = []
            signals_records = []
            enhanced_signals = []
            
            for rank, signal in enumerate(signals, 1):
                ticker = signal['ticker']
                
                # Get the component data from new signal structure
                reddit_data = signal.get('reddit_data', {})
                financial_data = signal.get('financial_data', {})
                news_data = signal.get('news_data', {})
                
                # Create comprehensive signals record
                current_time = datetime.now()
                
                # Prepare reddit summary from reddit_data
                mentions = reddit_data.get('mentions', [])
                reddit_summary = self._create_reddit_summary(mentions) if mentions else None
                
                # Get AI commentary if enabled
                ai_data = await self.get_ai_commentary(ticker, {
                    'reddit_data': reddit_data,
                    'financial_data': financial_data,
                    'signal': signal
                })
                
                # Calculate comprehensive risk assessment and signal classification
                if INTEGRATORS_AVAILABLE:
                    signal_classifier = get_signal_classifier()
                    
                    # Prepare technical data from financial_data for signal processing
                    technical_data = {k: v for k, v in financial_data.items() 
                                    if k in ['rsi', 'macd', 'bollinger', 'volatility', 'momentum_30d_pct', 'relative_strength']}
                    
                    # Assess risk with comprehensive factors
                    risk_level, risk_desc = signal_classifier.assess_risk(financial_data, technical_data, reddit_data)
                    
                    # Classify signal type
                    signal_type = signal_classifier.classify_signal_type(financial_data, technical_data, reddit_data)
                    
                    # Calculate post recency
                    post_recency = signal_classifier.calculate_post_recency_score(reddit_data)
                else:
                    risk_level, risk_desc = self._calculate_risk_metrics(signal, financial_data)
                    signal_type = "Multi-Factor"
                    post_recency = 0.5
                
                # Map to signals table schema - all actual database columns
                basic_record = {
                    # Core identification
                    'run_id': run_id,
                    'ticker': ticker,
                    'company': financial_data.get('company', financial_data.get('company_name', ticker)),
                    'sector': financial_data.get('sector'),
                    
                    # Scores (0-1 range as per schema)
                    'weighted_score': self._safe_round(signal['weighted_score'], 4),
                    'reddit_score': self._safe_round(signal.get('reddit_score', 0), 4),
                    'news_score': self._safe_round(signal.get('news_score', 0), 4),
                    'financial_score': self._safe_round(signal.get('financial_score', 0), 4),
                    
                    # Signal classification
                    'trade_type': signal_type,
                    'risk_level': risk_level,
                    'risk_tags': risk_desc if 'High' in risk_level else '',
                    'risk_assessment': risk_desc,
                    'rank': rank,
                    'normalized_rank': self._safe_round(rank / max(len(signals), 1), 4),
                    'signal_confidence': self._safe_round(signal['weighted_score'], 4),
                    'top_factors': 'Reddit mentions, price momentum',
                    'signal_type': 'Multi-Factor',
                    
                    # Price and market data
                    'current_price': self._safe_round(financial_data.get('current_price'), 2),
                    'market_cap': financial_data.get('market_cap'),
                    'avg_daily_value_traded': self._safe_round(financial_data.get('avg_daily_value_traded'), 0),
                    
                    # Reddit metrics
                    'reddit_sentiment': self._safe_round(signal.get('reddit_data', {}).get('avg_sentiment'), 4),
                    'news_sentiment': self._safe_round(signal.get('news_sentiment', 0), 4),
                    'mentions': signal.get('reddit_data', {}).get('mention_count', 0),
                    'news_mentions': signal.get('news_mentions', 0),
                    'upvotes': reddit_data.get('upvotes', 0),
                    'post_recency': self._safe_round(post_recency, 4),
                    
                    # Price movements
                    'price_1d_pct': self._safe_round(financial_data.get('price_1d_pct', financial_data.get('price_change_pct')), 2),
                    'price_7d_pct': self._safe_round(financial_data.get('price_7d_pct'), 2),
                    
                    # Volume
                    'volume': financial_data.get('volume'),
                    'volume_spike_ratio': self._safe_round(financial_data.get('volume_spike_ratio'), 2),
                    
                    # Technical indicators
                    'relative_strength': self._safe_round(financial_data.get('relative_strength'), 2),
                    'momentum_30d_pct': self._safe_round(financial_data.get('momentum_30d_pct'), 2),
                    'rsi': self._safe_round(financial_data.get('rsi'), 2),
                    'macd_histogram': self._safe_round(financial_data.get('macd_histogram'), 4),
                    'bollinger_width': self._safe_round(financial_data.get('bollinger_width'), 4),
                    'volatility': self._safe_round(financial_data.get('volatility'), 4),
                    'volatility_rank': self._safe_round(financial_data.get('volatility_rank'), 2),
                    'above_50d_ma_pct': self._safe_round(financial_data.get('above_50d_ma_pct'), 2),
                    'above_200d_ma_pct': self._safe_round(financial_data.get('above_200d_ma_pct'), 2),
                    
                    # Phase B: Enhanced Technical Indicators (9 new fields)
                    'avg_daily_volume': financial_data.get('avg_daily_volume'),
                    'avg_volume_30d': financial_data.get('avg_volume_30d'),
                    'volume_price_correlation': self._safe_round(financial_data.get('volume_price_correlation'), 4),
                    'sector_relative_strength': self._safe_round(financial_data.get('sector_relative_strength'), 2),
                    'exit_signal_strength': self._safe_round(financial_data.get('exit_signal_strength'), 2),
                    'signal_strength_percentile': self._safe_round(financial_data.get('signal_strength_percentile'), 2),
                    
                    # Fundamental metrics
                    'pe_ratio': self._safe_round(financial_data.get('pe_ratio'), 2),
                    'earnings_gap_pct': self._safe_round(financial_data.get('earnings_gap_pct'), 2),
                    'eps_growth': self._safe_round(financial_data.get('eps_growth'), 2),
                    'roe': self._safe_round(financial_data.get('roe'), 2),
                    'debt_equity': self._safe_round(financial_data.get('debt_equity'), 2),
                    'fcf_margin': self._safe_round(financial_data.get('fcf_margin'), 2),
                    
                    # Options data
                    'put_call_oi_ratio': self._safe_round(financial_data.get('put_call_oi_ratio'), 4),
                    'put_call_vol_ratio': self._safe_round(financial_data.get('put_call_vol_ratio'), 4),
                    'iv_spike_pct': self._safe_round(financial_data.get('iv_spike_pct'), 2),
                    
                    # Ownership metrics
                    'retail_holding_pct': self._safe_round(financial_data.get('retail_holding_pct'), 2),
                    'insider_buy_volume': self._safe_round(financial_data.get('insider_buy_volume'), 0),
                    'short_pct_float': self._safe_round(financial_data.get('short_pct_float'), 2),
                    'short_pct_outstanding': self._safe_round(financial_data.get('short_pct_outstanding'), 2),
                    'shares_short': financial_data.get('shares_short'),
                    
                    # Flags and metadata
                    'liquidity_warning': None,  # Can be populated by liquidity analysis
                    'emerging': False,  # Can be set based on market cap or other criteria
                    'thread_tag': None,  # Can be populated from Reddit analysis
                    
                    # AI summaries
                    'reddit_summary': reddit_data.get('summary', None),
                    'ai_news_summary': signal.get('ai_news_summary', None),
                    'ai_trends_commentary': signal.get('ai_trends_commentary', None),
                    'ai_commentary': signal.get('ai_commentary', None),
                    'score_explanation': signal.get('score_explanation', None),
                    
                    # Timestamps
                    'run_datetime': current_time.isoformat(),
                    'signal_datetime': current_time.isoformat(),
                    'created_at': current_time.isoformat(),
                    'updated_at': current_time.isoformat()
                }
                
                enhanced_signals.append(basic_record)
            
            # ENHANCEMENT: Apply signal enhancement calculations
            enhanced_signals = self._apply_signal_enhancements(enhanced_signals)
            
            # Debug: Check for potential overflow values
            for i, record in enumerate(enhanced_signals[:3]):  # Check first 3 records
                for key, value in record.items():
                    if isinstance(value, (int, float)) and value is not None:
                        if abs(value) >= 10:
                            self.logger.warning(f"Potential overflow in record {i}, field '{key}': {value}")
            
            # NEW 3-TABLE STRUCTURE: Split data into signals, signal_metrics, and signal_performance
            
            # Step 1: Prepare core signal data (signals table)
            core_signals = []
            metrics_data = []
            
            for record in enhanced_signals:
                # Core signal fields only
                core_signal = {
                    'run_id': record['run_id'],
                    'ticker': record['ticker'],
                    'company': record['company'],
                    'sector': record['sector'],
                    'weighted_score': record['weighted_score'],
                    'reddit_score': record['reddit_score'],
                    'news_score': record['news_score'],
                    'financial_score': record['financial_score'],
                    'trade_type': record['trade_type'],
                    'risk_level': record['risk_level'],
                    'risk_tags': record['risk_tags'],
                    'risk_assessment': record['risk_assessment'],
                    'rank': record['rank'],
                    'normalized_rank': record['normalized_rank'],
                    'signal_confidence': record['signal_confidence'],
                    'top_factors': record['top_factors'],
                    'signal_type': record['signal_type'],
                    'current_price': record['current_price'],
                    'market_cap': record['market_cap'],
                    'avg_daily_value_traded': record['avg_daily_value_traded'],
                    'reddit_sentiment': record['reddit_sentiment'],
                    'news_sentiment': record['news_sentiment'],
                    'mentions': record['mentions'],
                    'news_mentions': record['news_mentions'],
                    'upvotes': record['upvotes'],
                    'post_recency': record['post_recency'],
                    'price_1d_pct': record['price_1d_pct'],
                    'price_7d_pct': record['price_7d_pct'],
                    'volume': record['volume'],
                    'liquidity_warning': record['liquidity_warning'],
                    'emerging': record['emerging'],
                    'thread_tag': record['thread_tag'],
                    'reddit_summary': record['reddit_summary'],
                    'ai_news_summary': record['ai_news_summary'],
                    'ai_trends_commentary': record['ai_trends_commentary'],
                    'ai_commentary': record['ai_commentary'],
                    'score_explanation': record['score_explanation'],
                    'run_datetime': record['run_datetime'],
                    'signal_datetime': record['signal_datetime'],
                    'created_at': record['created_at'],
                    'updated_at': record['updated_at']
                }
                core_signals.append(core_signal)
            
            # Step 2: Insert core signals first
            result_signals = self.supabase.table('signals').insert(core_signals).execute()
            
            if not result_signals.data:
                self.logger.error("❌ Database insertion failed for signals table")
                return False
            
            # Step 3: Prepare metrics data with signal_id references
            for i, record in enumerate(enhanced_signals):
                signal_id = result_signals.data[i]['id']
                
                # Helper function to convert float to int for bigint columns
                def to_bigint(value):
                    """Convert numeric value to integer for bigint columns."""
                    if value is None:
                        return None
                    try:
                        return int(float(value))
                    except (ValueError, TypeError):
                        return None
                
                metrics_record = {
                    'signal_id': signal_id,
                    'ticker': record['ticker'],
                    'run_id': record['run_id'],
                    # Technical indicators - Momentum
                    'relative_strength': record.get('relative_strength'),
                    'momentum_30d_pct': record.get('momentum_30d_pct'),
                    'rsi': record.get('rsi'),
                    'macd_histogram': record.get('macd_histogram'),
                    'macd_line': record.get('macd_line'),
                    'macd_signal': record.get('macd_signal'),
                    'signal_strength_percentile': record.get('signal_strength_percentile'),
                    'sector_relative_strength': record.get('sector_relative_strength'),
                    # Volatility
                    'volatility': record.get('volatility'),
                    'volatility_rank': record.get('volatility_rank'),
                    'bollinger_width': record.get('bollinger_width'),
                    'bollinger_upper': record.get('bollinger_upper'),
                    'bollinger_lower': record.get('bollinger_lower'),
                    # Moving averages
                    'above_50d_ma_pct': record.get('above_50d_ma_pct'),
                    'above_200d_ma_pct': record.get('above_200d_ma_pct'),
                    'ma_cross_signal': record.get('ma_cross_signal'),
                    # Volume - CONVERT TO BIGINT
                    'volume_spike_ratio': record.get('volume_spike_ratio'),
                    'avg_daily_volume': to_bigint(record.get('avg_daily_volume')),
                    'avg_volume_30d': to_bigint(record.get('avg_volume_30d')),
                    'volume_price_correlation': record.get('volume_price_correlation'),
                    'float_turnover_ratio': record.get('float_turnover_ratio'),
                    # Fundamentals
                    'pe_ratio': record.get('pe_ratio'),
                    'earnings_gap_pct': record.get('earnings_gap_pct'),
                    'eps_growth': record.get('eps_growth'),
                    'roe': record.get('roe'),
                    'debt_equity': record.get('debt_equity'),
                    'fcf_margin': record.get('fcf_margin'),
                    'profit_margin': record.get('profit_margin'),
                    'revenue_growth': record.get('revenue_growth'),
                    'beta': record.get('beta'),
                    # Options
                    'put_call_oi_ratio': record.get('put_call_oi_ratio'),
                    'put_call_vol_ratio': record.get('put_call_vol_ratio'),
                    'iv_spike_pct': record.get('iv_spike_pct'),
                    'iv_percentile': record.get('iv_percentile'),
                    # Ownership - CONVERT BIGINT COLUMNS
                    'retail_holding_pct': record.get('retail_holding_pct'),
                    'insider_buy_volume': to_bigint(record.get('insider_buy_volume')),
                    'institutional_ownership_pct': record.get('institutional_ownership_pct'),
                    'short_pct_float': record.get('short_pct_float'),
                    'short_pct_outstanding': record.get('short_pct_outstanding'),
                    'shares_short': to_bigint(record.get('shares_short')),
                    # Metadata
                    'created_at': record['created_at'],
                    'updated_at': record['updated_at']
                }
                metrics_data.append(metrics_record)
            
            # Step 4: Insert metrics data
            result_metrics = self.supabase.table('signal_metrics').insert(metrics_data).execute()
            
            if result_metrics.data:
                self.logger.info(f"✅ Successfully saved {len(result_signals.data)} signals + {len(result_metrics.data)} metrics to database")
                
                # Refresh materialized view if it exists
                try:
                    self.supabase.rpc('refresh_signals_norm').execute()
                    self.logger.info(f"✅ Refreshed signals_norm materialized view")
                except Exception as refresh_error:
                    self.logger.warning(f"⚠️ signals_norm refresh skipped: {refresh_error}")
                
                return True
            else:
                self.logger.error("❌ Database insertion failed for signal_metrics table")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving signals to database: {e}")
            return False
    
    def _create_reddit_summary(self, mentions: List[Dict]) -> str:
        """Create a summary from Reddit mentions."""
        if not mentions:
            return None
        
        try:
            # Extract top titles and combine
            top_mentions = sorted(mentions, key=lambda x: x.get('score', 0), reverse=True)[:3]
            titles = [mention.get('title', '')[:100] for mention in top_mentions if mention.get('title')]
            return " | ".join(titles)[:500] if titles else None
        except:
            return None
    
    def _calculate_risk_metrics(self, signal: Dict, financial_data: Dict) -> tuple:
        """Calculate risk level and risk tags (fallback method)."""
        risk_factors = []
        score = signal.get('weighted_score', 0)
        
        # Risk based on score - use database schema values
        if score >= 0.8:
            risk_level = 'High'  # High reward, high risk
        elif score >= 0.5:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        # Add risk tags based on financial data (handle None values)
        volatility = financial_data.get('volatility')
        if volatility and volatility > 0.5:  # 50% annualized
            risk_factors.append('High Volatility')
        
        pe_ratio = financial_data.get('pe_ratio')
        if pe_ratio and pe_ratio > 50:
            risk_factors.append('High Valuation')
        
        debt_equity = financial_data.get('debt_equity')
        if debt_equity and debt_equity > 2:
            risk_factors.append('High Debt')
        
        market_cap = financial_data.get('market_cap')
        if market_cap and market_cap < 1000000000:  # < $1B
            risk_factors.append('Small Cap')
        
        # Create risk description
        risk_desc = ", ".join(risk_factors) if risk_factors else "Standard risk factors"
        
        return risk_level, risk_desc
    
    def _determine_trade_type(self, signal: Dict) -> str:
        """Determine trade type based on signal characteristics."""
        score = signal.get('weighted_score', 0)
        reddit_data = signal.get('reddit_data', {})
        sentiment = reddit_data.get('avg_sentiment', 0)
        
        if score >= 0.7 and sentiment > 0.3:
            return 'Growth'
        elif score >= 0.5 and sentiment >= 0:
            return 'Value' 
        elif sentiment > 0.5:
            return 'Momentum'
        else:
            return 'Speculative'
    
    def _get_top_factors(self, signal: Dict, financial_data: Dict) -> List[str]:
        """Identify top contributing factors to the signal."""
        factors = []
        
        if signal.get('reddit_score', 0) > 0.5:
            factors.append('reddit_buzz')
        
        reddit_data = signal.get('reddit_data', {})
        if reddit_data.get('mention_count', 0) >= 5:
            factors.append('high_mentions')
        
        volume_spike_ratio = financial_data.get('volume_spike_ratio')
        if volume_spike_ratio and volume_spike_ratio > 2:
            factors.append('volume_spike')
        
        price_1d_pct = financial_data.get('price_1d_pct')
        if price_1d_pct and abs(price_1d_pct) > 5:
            factors.append('price_momentum')
        
        if signal.get('avg_sentiment', 0) > 0.3:
            factors.append('positive_sentiment')
        
        return factors[:5]  # Top 5 factors
    
    async def _run_ai_strategy_generation(self) -> Dict[str, Any]:
        """Run AI strategy generation for top signals"""
        try:
            # Check if AI strategies are enabled
            ai_enabled = os.getenv('AI_STRATEGY_ENABLED', 'false').lower() == 'true'
            
            if not ai_enabled:
                self.logger.info("AI strategy generation disabled, skipping")
                return {'success': True, 'strategies_count': 0, 'message': 'AI strategies disabled'}
            
            # Import AI strategy generator
            from backend.integrations.ai import AIStrategyGenerator
            
            # Initialize and run AI strategy generator
            generator = AIStrategyGenerator()
            
            if not generator.ai_enabled:
                self.logger.warning("AI strategy generator not properly initialized")
                return {'success': False, 'strategies_count': 0, 'message': 'AI generator not initialized'}
            
            # Generate strategies for top signals
            self.logger.info(f"Generating AI strategies for top {generator.top_signals_limit} signals...")
            strategies = await generator.generate_strategies_for_top_signals()
            
            if strategies:
                total_strategies = sum(len(s) for s in strategies.values())
                self.logger.info(f"✅ Generated {total_strategies} AI strategies for {len(strategies)} tickers")
                
                # Log strategy summary
                strategy_summary = []
                for ticker, ticker_strategies in strategies.items():
                    strategy_types = [s.strategy_type for s in ticker_strategies]
                    strategy_summary.append(f"{ticker}: {len(ticker_strategies)} ({', '.join(strategy_types)})")
                    self.logger.info(f"   📊 {ticker}: {len(ticker_strategies)} strategies")
                
                return {
                    'success': True, 
                    'strategies_count': total_strategies,
                    'tickers_count': len(strategies),
                    'strategy_summary': strategy_summary,
                    'message': f'Generated {total_strategies} strategies for {len(strategies)} tickers'
                }
            else:
                self.logger.warning("No AI strategies were generated")
                return {'success': False, 'strategies_count': 0, 'message': 'No strategies generated'}
                
        except Exception as e:
            self.logger.error(f"AI strategy generation failed: {e}")
            return {'success': False, 'strategies_count': 0, 'message': f'Error: {str(e)}'}
    
    def generate_reddit_signals(self, ticker_mentions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate Reddit-based signals from ticker mentions.
        
        Args:
            ticker_mentions: Raw Reddit ticker mention data
            
        Returns:
            List of Reddit signals with scores and metadata
        """
        reddit_signals = []
        
        for ticker, data in ticker_mentions.items():
            try:
                # Calculate Reddit-specific scores
                mention_count = data['mention_count']
                avg_sentiment = data.get('avg_sentiment', 0)
                avg_score = data.get('avg_score', 0)
                
                # Reddit signal score (0-1 scale)
                reddit_score = self._calculate_reddit_score(mention_count, avg_sentiment, avg_score)
                
                # Create Reddit signal
                reddit_signal = {
                    'ticker': ticker,
                    'signal_type': 'reddit',
                    'score': reddit_score,
                    'confidence': min(mention_count / 10, 1.0),  # More mentions = higher confidence
                    'metadata': {
                        'mention_count': mention_count,
                        'avg_sentiment': avg_sentiment,
                        'avg_score': avg_score,
                        'mentions': data.get('mentions', [])
                    }
                }
                
                reddit_signals.append(reddit_signal)
                
            except Exception as e:
                self.logger.warning(f"Error generating Reddit signal for {ticker}: {e}")
                continue
        
        # Sort by score descending
        reddit_signals.sort(key=lambda x: x['score'], reverse=True)
        return reddit_signals
    
    def _calculate_reddit_score(self, mention_count: int, avg_sentiment: float, avg_score: float) -> float:
        """Calculate Reddit-specific signal score."""
        try:
            # Normalize mention count (1-5 mentions = 0.2-1.0 score)
            mention_score = min(mention_count / 5, 1.0)
            
            # Sentiment factor (positive sentiment boosts, negative reduces)
            sentiment_factor = max(0.1, (avg_sentiment + 1) / 2)  # Convert -1,1 to 0.1,1
            
            # Average post score factor (normalize by typical Reddit scores)
            score_factor = min(max(avg_score / 100, 0.1), 2.0)  # 0.1 to 2.0 multiplier
            
            # Combine factors
            reddit_score = mention_score * sentiment_factor * min(score_factor, 1.5)
            
            return min(max(reddit_score, 0), 1.0)  # Clamp 0-1
            
        except Exception:
            return 0.0
    
    def generate_financial_signals(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Generate financial-based signals from market data.
        
        Args:
            tickers: List of tickers to analyze
            
        Returns:
            List of financial signals with scores and metadata
        """
        financial_signals = []
        
        for ticker in tickers:
            try:
                # Get financial data
                financial_data = self.get_financial_data(ticker)
                
                if not financial_data:
                    continue
                
                # Calculate financial signal score
                financial_score = self._calculate_financial_score(financial_data)
                
                # Create financial signal
                financial_signal = {
                    'ticker': ticker,
                    'signal_type': 'financial',
                    'score': financial_score,
                    'confidence': 0.8,  # Financial data generally reliable
                    'metadata': financial_data
                }
                
                financial_signals.append(financial_signal)
                
            except Exception as e:
                self.logger.warning(f"Error generating financial signal for {ticker}: {e}")
                continue
        
        # Sort by score descending
        financial_signals.sort(key=lambda x: x['score'], reverse=True)
        return financial_signals
    
    def _calculate_financial_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive financial score using ALL technical indicators.
        
        Formula: Technical (40%) + Fundamentals (30%) + Options (15%) + Short Interest (15%)
        
        This method now uses all 29+ technical indicators to calculate a comprehensive score.
        """
        try:
            # ===== TECHNICAL INDICATORS SCORE (40%) =====
            technical_score = self._calculate_technical_score(financial_data)
            
            # ===== FUNDAMENTALS SCORE (30%) =====
            fundamentals_score = self._calculate_fundamentals_score(financial_data)
            
            # ===== OPTIONS SENTIMENT SCORE (15%) =====
            options_score = self._calculate_options_score(financial_data)
            
            # ===== SHORT INTEREST SCORE (15%) =====
            short_score = self._calculate_short_interest_score(financial_data)
            
            # Combine all components
            financial_score = (
                technical_score * 0.40 +
                fundamentals_score * 0.30 +
                options_score * 0.15 +
                short_score * 0.15
            )
            
            return min(max(financial_score, 0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating financial score: {e}")
            return 0.0
    
    def _calculate_technical_score(self, financial_data: Dict[str, Any]) -> float:
        """Calculate technical indicators score from all available indicators."""
        try:
            technical_components = []
            
            # 1. MOMENTUM INDICATORS (25%)
            # Price momentum (1d, 7d, 30d)
            price_1d = financial_data.get('price_1d_pct', 0)
            price_7d = financial_data.get('price_7d_pct', 0)
            momentum_30d = financial_data.get('momentum_30d_pct', 0)
            
            momentum_score = min(
                (abs(price_1d) / 10 + abs(price_7d) / 20 + abs(momentum_30d) / 30) / 3,
                1.0
            )
            technical_components.append(momentum_score * 0.25)
            
            # 2. RSI INDICATOR (15%)
            rsi = financial_data.get('rsi')
            if rsi and not np.isnan(rsi):
                # Extreme RSI indicates opportunity (oversold <35 or overbought >65)
                if rsi < 35:
                    rsi_score = 1.0  # Oversold - buy opportunity
                elif rsi > 65:
                    rsi_score = 0.8  # Overbought - potential reversal
                else:
                    rsi_score = 0.5  # Neutral
                technical_components.append(rsi_score * 0.15)
            
            # 3. MOVING AVERAGE POSITION (15%)
            ma_50_pct = financial_data.get('above_50d_ma_pct')
            ma_200_pct = financial_data.get('above_200d_ma_pct')
            
            ma_score = 0.0
            if ma_50_pct is not None and not np.isnan(ma_50_pct):
                ma_score += 0.5 if ma_50_pct > 0 else 0.2
            if ma_200_pct is not None and not np.isnan(ma_200_pct):
                ma_score += 0.5 if ma_200_pct > 0 else 0.2
            
            technical_components.append((ma_score / 1.0) * 0.15)
            
            # 4. MACD INDICATOR (10%)
            macd = financial_data.get('macd')
            if macd and not np.isnan(macd):
                macd_score = 1.0 if macd > 0 else 0.3  # Positive MACD is bullish
                technical_components.append(macd_score * 0.10)
            
            # 5. VOLUME ANALYSIS (15%)
            volume_spike = financial_data.get('volume_spike_ratio', 1)
            avg_volume = financial_data.get('avg_volume_30d', 0)
            vol_price_corr = financial_data.get('volume_price_correlation', 0)
            
            volume_score = min(max(volume_spike - 1, 0) / 2, 1.0)  # Spike above normal
            if not np.isnan(vol_price_corr) and vol_price_corr > 0.3:
                volume_score = min(volume_score * 1.2, 1.0)  # Boost if volume confirms price
            
            technical_components.append(volume_score * 0.15)
            
            # 6. VOLATILITY & BOLLINGER BANDS (10%)
            volatility = financial_data.get('volatility', 0)
            volatility_rank = financial_data.get('volatility_rank', 0)
            bollinger = financial_data.get('bollinger', 0)
            
            # Moderate volatility preferred (10-40%), avoid extreme volatility
            if not np.isnan(volatility) and volatility > 0:
                if 10 < volatility < 40:
                    vol_score = 1.0
                elif volatility < 10:
                    vol_score = 0.6  # Too calm
                else:
                    vol_score = 0.4  # Too volatile
            else:
                vol_score = 0.5
            
            technical_components.append(vol_score * 0.10)
            
            # 7. RELATIVE STRENGTH (10%)
            relative_strength = financial_data.get('relative_strength', 0)
            sector_rs = financial_data.get('sector_relative_strength', 0)
            
            rs_score = 0.0
            if not np.isnan(relative_strength):
                rs_score += 0.5 if relative_strength > 0 else 0.2
            if not np.isnan(sector_rs):
                rs_score += 0.5 if sector_rs > 0 else 0.2
            
            technical_components.append((rs_score / 1.0) * 0.10)
            
            # Calculate total technical score
            return sum(technical_components)
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical score: {e}")
            return 0.0
    
    def _calculate_fundamentals_score(self, financial_data: Dict[str, Any]) -> float:
        """Calculate fundamentals score from financial metrics."""
        try:
            fundamental_components = []
            
            # Market cap (prefer mid-cap to large-cap)
            market_cap = financial_data.get('market_cap_numeric', 0)
            if market_cap:
                if market_cap > 10_000_000_000:  # >$10B
                    cap_score = 0.8
                elif market_cap > 2_000_000_000:  # $2B-$10B
                    cap_score = 1.0
                else:
                    cap_score = 0.6
                fundamental_components.append(cap_score * 0.2)
            
            # P/E ratio (reasonable valuation)
            pe_ratio = financial_data.get('pe_ratio')
            if pe_ratio and not np.isnan(pe_ratio) and pe_ratio > 0:
                if 10 < pe_ratio < 30:
                    pe_score = 1.0  # Reasonable valuation
                elif pe_ratio < 10:
                    pe_score = 0.7  # Potentially undervalued or issues
                else:
                    pe_score = 0.5  # Expensive
                fundamental_components.append(pe_score * 0.2)
            
            # Profitability metrics
            profit_margin = financial_data.get('profit_margin')
            roe = financial_data.get('roe')
            
            if profit_margin and not np.isnan(profit_margin):
                profit_score = min(profit_margin * 5, 1.0)  # Scale 0-1
                fundamental_components.append(profit_score * 0.15)
            
            if roe and not np.isnan(roe):
                roe_score = min(roe * 5, 1.0)  # Scale 0-1
                fundamental_components.append(roe_score * 0.15)
            
            # Growth metrics
            revenue_growth = financial_data.get('revenue_growth')
            earnings_growth = financial_data.get('earnings_growth')
            
            if revenue_growth and not np.isnan(revenue_growth):
                growth_score = min(max(revenue_growth * 2, 0), 1.0)
                fundamental_components.append(growth_score * 0.15)
            
            # Debt levels
            debt_to_equity = financial_data.get('debt_to_equity')
            if debt_to_equity and not np.isnan(debt_to_equity):
                debt_score = 1.0 if debt_to_equity < 0.5 else max(1.0 - (debt_to_equity - 0.5), 0.3)
                fundamental_components.append(debt_score * 0.15)
            
            return sum(fundamental_components)
            
        except Exception as e:
            self.logger.warning(f"Error calculating fundamentals score: {e}")
            return 0.0
    
    def _calculate_options_score(self, financial_data: Dict[str, Any]) -> float:
        """Calculate options sentiment score."""
        try:
            # Put/call ratio (lower is bullish)
            put_call_ratio = financial_data.get('put_call_ratio')
            if put_call_ratio and not np.isnan(put_call_ratio):
                if put_call_ratio < 0.7:
                    return 1.0  # Very bullish
                elif put_call_ratio < 1.0:
                    return 0.7  # Moderately bullish
                else:
                    return 0.4  # Bearish
            
            return 0.5  # Neutral if no data
            
        except Exception:
            return 0.5
    
    def _calculate_short_interest_score(self, financial_data: Dict[str, Any]) -> float:
        """Calculate short squeeze potential score."""
        try:
            short_pct = financial_data.get('short_pct_float', 0)
            short_ratio = financial_data.get('short_ratio', 0)
            
            if short_pct and not np.isnan(short_pct):
                if short_pct > 20:
                    return 1.0  # High short squeeze potential
                elif short_pct > 10:
                    return 0.7  # Moderate potential
                elif short_pct > 5:
                    return 0.5  # Some potential
                else:
                    return 0.3  # Low potential
            
            return 0.3  # Default low potential
            
        except Exception:
            return 0.3
    
    async def generate_news_signals(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Generate news-based signals from sentiment analysis.
        
        Args:
            tickers: List of tickers to analyze
            
        Returns:
            List of news signals with scores and metadata
        """
        news_signals = []
        
        if not self.enhanced_available or not self.news_integrator:
            self.logger.info("News integration not available, skipping news signals")
            return news_signals
        
        for ticker in tickers:
            try:
                # Get news data
                news_data = await self.get_news_data(ticker)
                
                if not news_data or news_data.get('news_mentions', 0) == 0:
                    continue
                
                # Calculate news signal score
                news_score = self._calculate_news_score(news_data)
                
                # Create news signal
                news_signal = {
                    'ticker': ticker,
                    'signal_type': 'news',
                    'score': news_score,
                    'confidence': min(news_data.get('news_mentions', 0) / 5, 1.0),
                    'metadata': news_data
                }
                
                news_signals.append(news_signal)
                
            except Exception as e:
                self.logger.warning(f"Error generating news signal for {ticker}: {e}")
                continue
        
        # Sort by score descending
        news_signals.sort(key=lambda x: x['score'], reverse=True)
        return news_signals
    
    def _calculate_news_score(self, news_data: Dict[str, Any]) -> float:
        """Calculate news-specific signal score."""
        try:
            # Base news sentiment score
            base_score = news_data.get('news_score', 0)
            
            # Normalize from -1,1 to 0,1 scale
            normalized_score = (base_score + 1) / 2
            
            # Boost based on number of mentions
            mention_count = news_data.get('news_mentions', 0)
            mention_multiplier = min(1 + (mention_count / 10), 2.0)
            
            news_score = normalized_score * mention_multiplier
            
            return min(max(news_score, 0), 1.0)
            
        except Exception:
            return 0.0
    
    def combine_signals_to_scored_signals(self, 
                                        reddit_signals: List[Dict], 
                                        financial_signals: List[Dict], 
                                        news_signals: List[Dict]) -> List[Dict[str, Any]]:
        """
        Combine all individual signals into final scored signals for signals_norm table.
        
        Args:
            reddit_signals: Reddit-based signals
            financial_signals: Financial-based signals  
            news_signals: News-based signals
            
        Returns:
            List of combined scored signals
        """
        # Create ticker-based signal mapping
        ticker_signals = {}
        
        # Index all signals by ticker
        for signal in reddit_signals:
            ticker = signal['ticker']
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {'reddit': None, 'financial': None, 'news': None}
            ticker_signals[ticker]['reddit'] = signal
        
        for signal in financial_signals:
            ticker = signal['ticker']
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {'reddit': None, 'financial': None, 'news': None}
            ticker_signals[ticker]['financial'] = signal
        
        for signal in news_signals:
            ticker = signal['ticker']
            if ticker not in ticker_signals:
                ticker_signals[ticker] = {'reddit': None, 'financial': None, 'news': None}
            ticker_signals[ticker]['news'] = signal
        
        # Get configurable scoring weights from config
        scoring_weights = self.config.get('scoring.weights', {
            'reddit': 0.5,
            'financial': 0.5,
            'news': 0.0
        })
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(scoring_weights.values())
        if total_weight > 0:
            scoring_weights = {k: v / total_weight for k, v in scoring_weights.items()}
        else:
            # Fallback if all weights are 0
            scoring_weights = {'reddit': 0.5, 'financial': 0.5, 'news': 0.0}
        
        self.logger.info(f"📊 Using scoring weights: Reddit={scoring_weights['reddit']:.1%}, "
                        f"Financial={scoring_weights['financial']:.1%}, "
                        f"News={scoring_weights['news']:.1%}")
        
        # Combine signals for each ticker
        combined_signals = []
        
        for ticker, signals in ticker_signals.items():
            try:
                # Extract individual scores (default 0 if signal missing)
                reddit_score = signals['reddit']['score'] if signals['reddit'] else 0.0
                financial_score = signals['financial']['score'] if signals['financial'] else 0.0
                news_score = signals['news']['score'] if signals['news'] else 0.0
                
                # Calculate weighted combined score using configurable weights
                weighted_score = (
                    reddit_score * scoring_weights['reddit'] + 
                    financial_score * scoring_weights['financial'] + 
                    news_score * scoring_weights['news']
                )
                
                # Calculate confidence based on available signals (excluding news if weight is 0)
                active_signal_count = sum(1 for k, s in [
                    ('reddit', signals['reddit']), 
                    ('financial', signals['financial']), 
                    ('news', signals['news'])
                ] if s is not None and scoring_weights.get(k, 0) > 0)
                
                expected_signal_count = sum(1 for w in scoring_weights.values() if w > 0)
                confidence = active_signal_count / expected_signal_count if expected_signal_count > 0 else 0.0
                
                # Create combined signal
                combined_signal = {
                    'ticker': ticker,
                    'weighted_score': weighted_score,
                    'reddit_score': reddit_score,
                    'financial_score': financial_score,
                    'news_score': news_score,
                    'confidence': confidence,
                    'signal_count': active_signal_count,
                    'scoring_weights': scoring_weights,  # Track which weights were used
                    'reddit_data': signals['reddit']['metadata'] if signals['reddit'] else {},
                    'financial_data': signals['financial']['metadata'] if signals['financial'] else {},
                    'news_data': signals['news']['metadata'] if signals['news'] else {}
                }
                
                combined_signals.append(combined_signal)
                
            except Exception as e:
                self.logger.warning(f"Error combining signals for {ticker}: {e}")
                continue
        
        # Sort by weighted score descending
        combined_signals.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return combined_signals
    
    def _clamp_decimal(self, value: Optional[float], min_val: float, max_val: float) -> Optional[float]:
        """Clamp a decimal value to database field limits and round to 2 decimal places."""
        if value is None:
            return None
        clamped = max(min_val, min(max_val, value))
        return round(clamped, 2)
    
    def _safe_round(self, value: Optional[float], decimals: int = 2) -> Optional[float]:
        """Safely round a value, handling NaN, infinity, and None."""
        if value is None:
            return None
        
        import math
        import numpy as np
        
        # Check for NaN or infinity
        if math.isnan(value) if not isinstance(value, (list, np.ndarray)) else np.isnan(value).any():
            return None
        if math.isinf(value) if not isinstance(value, (list, np.ndarray)) else np.isinf(value).any():
            return None
            
        try:
            return round(float(value), decimals)
        except (ValueError, TypeError, OverflowError):
            return None
            
    def _apply_signal_enhancements(self, signals: list) -> list:
        """Apply signal enhancements including calculated fields."""
        try:
            # Import the consolidated enhancer
            from backend.integrations.signal_processing import enhance_signals_batch
            self.logger.info(f"Applying signal enhancements to {len(signals)} records...")
            enhanced = enhance_signals_batch(signals)
            self.logger.info("Signal enhancement complete")
            return enhanced
        except ImportError:
            self.logger.warning("Signal enhancer module not available, applying basic enhancements...")
            return self._apply_basic_enhancements(signals)
        except Exception as e:
            self.logger.warning(f"Signal enhancement failed: {e}, applying basic enhancements...")
            return self._apply_basic_enhancements(signals)
    
    def _apply_basic_enhancements(self, signals: list) -> list:
        """Apply basic signal enhancements if the full enhancer is unavailable."""
        enhanced_signals = []
        
        for signal in signals:
            enhanced = signal.copy()
            
            # Basic market cap categorization
            market_cap = signal.get('market_cap')
            if market_cap:
                if market_cap < 300_000_000:
                    enhanced['market_cap_category'] = 'Nano'
                elif market_cap < 2_000_000_000:
                    enhanced['market_cap_category'] = 'Micro'
                elif market_cap < 10_000_000_000:
                    enhanced['market_cap_category'] = 'Small'
                elif market_cap < 200_000_000_000:
                    enhanced['market_cap_category'] = 'Mid'
                elif market_cap < 1_000_000_000_000:
                    enhanced['market_cap_category'] = 'Large'
                else:
                    enhanced['market_cap_category'] = 'Mega'
            else:
                enhanced['market_cap_category'] = 'Unknown'
            
            # Basic risk score calculation
            volatility = signal.get('volatility', 0.15)
            debt_equity = signal.get('debt_equity', 25)
            
            risk_score = min(100, max(0, 
                volatility * 30 +  # Volatility component
                (25 if debt_equity > 100 else 10 if debt_equity > 50 else 5) +  # Debt component
                (15 if market_cap and market_cap < 1_000_000_000 else 5)  # Size component
            ))
            enhanced['risk_score'] = self._safe_round(risk_score, 2)
            
            # Risk category
            if risk_score <= 25:
                enhanced['risk_category'] = 'Conservative'
            elif risk_score <= 45:
                enhanced['risk_category'] = 'Moderate'
            elif risk_score <= 65:
                enhanced['risk_category'] = 'Aggressive'
            else:
                enhanced['risk_category'] = 'Speculative'
            
            # Max position size (inverse of risk)
            enhanced['max_position_size'] = self._safe_round(max(0.01, 0.10 - (risk_score * 0.001)), 3)
            
            # Basic liquidity score
            avg_daily_value = signal.get('avg_daily_value_traded')
            if avg_daily_value and market_cap:
                turnover = avg_daily_value / market_cap
                enhanced['liquidity_score'] = self._safe_round(min(1.0, turnover * 100), 2)
            else:
                enhanced['liquidity_score'] = 0.5
            
            # Risk-adjusted score
            weighted_score = signal.get('weighted_score', 0)
            enhanced['risk_adjusted_score'] = self._safe_round(
                weighted_score * (100 - risk_score) / 100, 4
            )
            
            enhanced_signals.append(enhanced)
        
        self.logger.info(f"Applied basic enhancements to {len(enhanced_signals)} signals")
        return enhanced_signals
    
    async def _comprehensive_signal_enhancement(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Comprehensive enhancement that eliminates duplicate yfinance API calls
        
        Consolidates Steps 4.5-4.8 into single efficient process:
        - Single API call per ticker with caching
        - All technical indicators (MACD, Bollinger, RSI, Beta)
        - All performance metrics (1d, 3d, 7d returns)
        - Basic enhancements and AI data preparation
        """
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        import ta
        from scipy.stats import linregress
        
        # Group signals by ticker to minimize API calls
        ticker_groups = {}
        for signal in signals:
            ticker = signal.get('ticker', '').upper()
            if ticker:
                if ticker not in ticker_groups:
                    ticker_groups[ticker] = []
                ticker_groups[ticker].append(signal)
        
        self.logger.info(f"Grouped {len(signals)} signals into {len(ticker_groups)} unique tickers")
        
        # Cache for ticker data to avoid duplicate API calls
        ticker_cache = {}
        enhanced_signals = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            for ticker, ticker_signals in ticker_groups.items():
                try:
                    # Single comprehensive API call per ticker
                    if ticker not in ticker_cache:
                        ticker_cache[ticker] = await self._get_comprehensive_ticker_data(ticker, executor)
                    
                    ticker_data = ticker_cache[ticker]
                    
                    # Apply all enhancements to ticker signals
                    for signal in ticker_signals:
                        enhanced_signal = self._apply_all_enhancements_to_signal(signal, ticker_data)
                        enhanced_signals.append(enhanced_signal)
                        
                except Exception as e:
                    self.logger.warning(f"Enhancement failed for {ticker}: {e}")
                    # Add original signals without enhancement
                    enhanced_signals.extend(ticker_signals)
        
        self.logger.info(f"✅ Comprehensive enhancement complete: {len(enhanced_signals)} signals")
        return enhanced_signals
    
    async def _get_comprehensive_ticker_data(self, ticker: str, executor: ThreadPoolExecutor) -> Dict[str, Any]:
        """Single API call to get ALL data needed for enhancements"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._fetch_ticker_data_sync, ticker)
    
    def _fetch_ticker_data_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous data fetching for ThreadPoolExecutor"""
        import yfinance as yf
        import pandas as pd
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get all time periods needed in single session
            history_1y = stock.history(period="1y", interval="1d")
            history_3m = stock.history(period="3mo", interval="1d") 
            history_1m = stock.history(period="1mo", interval="1d")
            info = stock.info
            
            return {
                'ticker': ticker,
                'stock': stock,
                'info': info,
                'history_1y': history_1y,
                'history_3m': history_3m,
                'history_1m': history_1m
            }
            
        except Exception as e:
            self.logger.debug(f"Data fetch failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'stock': None,
                'info': {},
                'history_1y': pd.DataFrame(),
                'history_3m': pd.DataFrame(),
                'history_1m': pd.DataFrame()
            }
    
    def _apply_all_enhancements_to_signal(self, signal: Dict[str, Any], ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ALL enhancements to signal using cached ticker data"""
        enhanced_signal = signal.copy()
        
        # Basic enhancements (replaces Step 4.5)
        enhanced_signal = self._apply_basic_enhancements_cached(enhanced_signal, ticker_data)
        
        # Performance metrics (replaces Step 4.6)  
        enhanced_signal = self._apply_performance_metrics_cached(enhanced_signal, ticker_data)
        
        # Technical indicators (replaces Step 4.8)
        enhanced_signal = self._apply_technical_indicators_cached(enhanced_signal, ticker_data)
        
        # AI commentary data preparation (replaces Step 4.7 prep)
        enhanced_signal = self._prepare_ai_commentary_data_cached(enhanced_signal, ticker_data)
        
        # Score components and explanation (NEW)
        enhanced_signal = self._calculate_score_components(enhanced_signal)
        
        return enhanced_signal
    
    def _apply_basic_enhancements_cached(self, signal: Dict[str, Any], ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic signal enhancements using cached data"""
        try:
            info = ticker_data.get('info', {})
            
            # Market cap and basic metrics
            signal['market_cap'] = info.get('marketCap')
            signal['sector'] = info.get('sector')
            signal['industry'] = info.get('industry')
            signal['pe_ratio'] = info.get('trailingPE')
            signal['forward_pe'] = info.get('forwardPE')
            signal['price_to_book'] = info.get('priceToBook')
            signal['dividend_yield'] = info.get('dividendYield')
            
            # Current price data
            history_1m = ticker_data.get('history_1m', pd.DataFrame())
            if not history_1m.empty:
                current_price = history_1m['Close'].iloc[-1]
                signal['current_price'] = float(current_price)
                signal['volume'] = int(history_1m['Volume'].iloc[-1])
            
        except Exception as e:
            self.logger.debug(f"Basic enhancement failed for {signal.get('ticker')}: {e}")
            
        return signal
    
    def _apply_performance_metrics_cached(self, signal: Dict[str, Any], ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply performance tracking using cached data (replaces Step 4.6)"""
        try:
            history_1m = ticker_data.get('history_1m', pd.DataFrame())
            if history_1m.empty:
                return signal
                
            prices = history_1m['Close']
            
            # Calculate returns for different periods
            if len(prices) >= 2:
                signal['return_1d'] = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100)
            
            if len(prices) >= 3:
                signal['return_3d'] = float((prices.iloc[-1] / prices.iloc[-3] - 1) * 100)
                
            if len(prices) >= 7:
                signal['return_7d'] = float((prices.iloc[-1] / prices.iloc[-7] - 1) * 100)
                
            if len(prices) >= 14:
                signal['return_14d'] = float((prices.iloc[-1] / prices.iloc[-14] - 1) * 100)
            
            # Volatility metrics
            import numpy as np
            signal['volatility_30d'] = float(prices.pct_change().rolling(min_periods=5, window=min(30, len(prices))).std() * np.sqrt(252) * 100)
            
            # Volume metrics
            volumes = history_1m['Volume']
            if not volumes.empty:
                signal['avg_volume_10d'] = int(volumes.rolling(min_periods=1, window=min(10, len(volumes))).mean().iloc[-1])
                signal['volume_ratio'] = float(volumes.iloc[-1] / signal['avg_volume_10d']) if signal['avg_volume_10d'] > 0 else 1.0
                
        except Exception as e:
            self.logger.debug(f"Performance metrics failed for {signal.get('ticker')}: {e}")
            
        return signal
    
    def _apply_technical_indicators_cached(self, signal: Dict[str, Any], ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply technical indicators using cached data (replaces Step 4.8)"""
        try:
            import ta
            from scipy.stats import linregress
            import yfinance as yf
            
            df = ticker_data.get('history_3m', pd.DataFrame())
            if df.empty or len(df) < 26:  # Need minimum data for MACD
                return signal
            
            # MACD calculation
            macd_line = ta.trend.MACD(df['Close']).macd()
            macd_signal_line = ta.trend.MACD(df['Close']).macd_signal()
            macd_histogram = ta.trend.MACD(df['Close']).macd_diff()
            
            signal['macd_line'] = float(macd_line.iloc[-1]) if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else None
            signal['macd_signal'] = float(macd_signal_line.iloc[-1]) if not macd_signal_line.empty and not pd.isna(macd_signal_line.iloc[-1]) else None
            signal['macd_histogram'] = float(macd_histogram.iloc[-1]) if not macd_histogram.empty and not pd.isna(macd_histogram.iloc[-1]) else None
            
            # Bollinger Bands
            bb_upper = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            bb_middle = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()  
            bb_lower = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            
            signal['bb_upper'] = float(bb_upper.iloc[-1]) if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else None
            signal['bb_middle'] = float(bb_middle.iloc[-1]) if not bb_middle.empty and not pd.isna(bb_middle.iloc[-1]) else None
            signal['bb_lower'] = float(bb_lower.iloc[-1]) if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else None
            
            # RSI
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
            signal['rsi'] = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
            
            # Beta calculation using scipy linear regression (fixed)
            signal['beta'] = self._calculate_beta_cached(ticker_data)
            
        except Exception as e:
            self.logger.debug(f"Technical indicators failed for {signal.get('ticker')}: {e}")
            
        return signal
    
    def _calculate_beta_cached(self, ticker_data: Dict[str, Any]) -> Optional[float]:
        """Calculate Beta using scipy linear regression with cached data"""
        try:
            import yfinance as yf
            from scipy.stats import linregress
            import pandas as pd
            
            # Get SPY data for same period
            spy = yf.Ticker("SPY")
            spy_history = spy.history(period="1y", interval="1d")
            
            # Get stock data
            stock_df = ticker_data.get('history_1y', pd.DataFrame())
            if stock_df.empty or spy_history.empty:
                return None
                
            # Align dates and calculate returns
            common_dates = stock_df.index.intersection(spy_history.index)
            if len(common_dates) < 30:  # Need minimum data points
                return None
                
            stock_returns = stock_df.loc[common_dates]['Close'].pct_change().dropna()
            spy_returns = spy_history.loc[common_dates]['Close'].pct_change().dropna()
            
            # Ensure same length
            min_len = min(len(stock_returns), len(spy_returns))
            if min_len < 20:
                return None
                
            stock_returns = stock_returns.iloc[-min_len:]
            spy_returns = spy_returns.iloc[-min_len:]
            
            # Linear regression: stock_returns = alpha + beta * spy_returns
            slope, intercept, r_value, p_value, std_err = linregress(spy_returns, stock_returns)
            
            return float(slope)
            
        except Exception as e:
            self.logger.debug(f"Beta calculation failed: {e}")
            return None
    
    def _prepare_ai_commentary_data_cached(self, signal: Dict[str, Any], ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for AI commentary generation using cached data"""
        try:
            # Consolidate key metrics for AI analysis
            signal['ai_data_summary'] = {
                'price_momentum': {
                    'return_1d': signal.get('return_1d'),
                    'return_7d': signal.get('return_7d'),
                    'rsi': signal.get('rsi')
                },
                'technical_signals': {
                    'macd_signal': 'bullish' if signal.get('macd_line', 0) > signal.get('macd_signal', 0) else 'bearish',
                    'bb_position': 'upper' if signal.get('current_price', 0) > signal.get('bb_upper', 0) else 'lower' if signal.get('current_price', 0) < signal.get('bb_lower', 0) else 'middle',
                    'volume_signal': 'high' if signal.get('volume_ratio', 1) > 1.5 else 'normal'
                },
                'fundamental_context': {
                    'sector': signal.get('sector'),
                    'pe_ratio': signal.get('pe_ratio'),
                    'beta': signal.get('beta')
                }
            }
            
            # Mark as ready for AI commentary generation
            signal['ai_commentary_ready'] = True
            
        except Exception as e:
            self.logger.debug(f"AI data preparation failed for {signal.get('ticker')}: {e}")
            
        return signal
    
    def _calculate_score_components(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate and store detailed score components for transparency"""
        try:
            # Extract scoring components
            reddit_score = signal.get('reddit_score', 0)
            financial_score = signal.get('financial_score', 0)
            weighted_score = signal.get('weighted_score', 0)
            
            # Calculate component contributions
            reddit_weight = 0.4  # Default weights from scoring logic
            financial_weight = 0.6
            
            reddit_contribution = reddit_score * reddit_weight
            financial_contribution = financial_score * financial_weight
            
            # Technical factor contributions
            technical_factors = {
                'rsi_factor': self._calculate_rsi_factor(signal.get('rsi')),
                'macd_factor': self._calculate_macd_factor(signal),
                'volume_factor': self._calculate_volume_factor(signal.get('volume_ratio', 1)),
                'momentum_factor': self._calculate_momentum_factor(signal),
                'risk_penalty': self._calculate_risk_penalty(signal.get('risk_score', 50))
            }
            
            # Store detailed score breakdown
            signal['score_components'] = {
                'weighted_score': weighted_score,
                'reddit_contribution': round(reddit_contribution, 4),
                'financial_contribution': round(financial_contribution, 4),
                'reddit_weight': reddit_weight,
                'financial_weight': financial_weight,
                'technical_factors': technical_factors,
                'score_calculation_method': 'comprehensive_v1.0'
            }
            
            # Generate score explanation
            signal['score_explanation'] = self._generate_score_explanation(signal, technical_factors)
            
            # Set scoring metadata
            signal['scoring_version'] = '1.0'
            signal['prediction_confidence'] = self._calculate_prediction_confidence(signal)
            
        except Exception as e:
            logger.debug(f"Score components calculation failed for {signal.get('ticker')}: {e}")
            # Fallback minimal components
            signal['score_components'] = {'weighted_score': signal.get('weighted_score', 0)}
            signal['score_explanation'] = f"Score {signal.get('weighted_score', 0):.3f} based on combined reddit and financial metrics"
            
        return signal
    
    def _calculate_rsi_factor(self, rsi: Optional[float]) -> float:
        """Calculate RSI contribution to score"""
        if not rsi:
            return 0.0
        
        # RSI between 30-70 is neutral, outside adds momentum
        if rsi > 70:
            return min(0.02, (rsi - 70) * 0.001)  # Overbought boost (limited)
        elif rsi < 30:
            return min(0.02, (30 - rsi) * 0.001)  # Oversold boost (limited)
        else:
            return 0.0
    
    def _calculate_macd_factor(self, signal: Dict[str, Any]) -> float:
        """Calculate MACD contribution to score"""
        macd_line = signal.get('macd_line')
        macd_signal = signal.get('macd_signal')
        
        if macd_line and macd_signal:
            if macd_line > macd_signal:
                return 0.01  # Bullish MACD
            else:
                return -0.005  # Bearish MACD penalty
        return 0.0
    
    def _calculate_volume_factor(self, volume_ratio: float) -> float:
        """Calculate volume contribution to score"""
        if volume_ratio > 2.0:
            return 0.015  # High volume boost
        elif volume_ratio > 1.5:
            return 0.01   # Moderate volume boost
        elif volume_ratio < 0.5:
            return -0.01  # Low volume penalty
        else:
            return 0.0
    
    def _calculate_momentum_factor(self, signal: Dict[str, Any]) -> float:
        """Calculate momentum contribution to score"""
        return_1d = signal.get('return_1d', 0)
        return_7d = signal.get('return_7d', 0)
        
        momentum = 0.0
        
        # 1-day momentum
        if return_1d > 5:
            momentum += 0.01
        elif return_1d < -5:
            momentum -= 0.01
        
        # 7-day momentum
        if return_7d > 10:
            momentum += 0.015
        elif return_7d < -10:
            momentum -= 0.015
            
        return round(momentum, 4)
    
    def _calculate_risk_penalty(self, risk_score: float) -> float:
        """Calculate risk penalty for score"""
        if risk_score > 80:
            return -0.02  # High risk penalty
        elif risk_score > 60:
            return -0.01  # Moderate risk penalty  
        else:
            return 0.0
    
    def _generate_score_explanation(self, signal: Dict[str, Any], technical_factors: Dict[str, float]) -> str:
        """Generate human-readable score explanation"""
        
        ticker = signal.get('ticker', 'N/A')
        score = signal.get('weighted_score', 0)
        reddit_score = signal.get('reddit_score', 0)
        financial_score = signal.get('financial_score', 0)
        
        # Primary components
        explanation_parts = [
            f"{ticker} weighted score of {score:.3f} combines:",
            f"Reddit sentiment ({reddit_score:.2f} from {signal.get('mention_count', 0)} mentions)",
            f"Financial metrics ({financial_score:.2f} from market data)"
        ]
        
        # Technical adjustments
        technical_adjustments = []
        for factor, value in technical_factors.items():
            if abs(value) > 0.005:
                if factor == 'rsi_factor' and value > 0:
                    technical_adjustments.append("RSI momentum boost")
                elif factor == 'macd_factor' and value > 0:
                    technical_adjustments.append("MACD bullish signal")
                elif factor == 'volume_factor' and value > 0:
                    technical_adjustments.append("volume surge")
                elif factor == 'momentum_factor' and value > 0:
                    technical_adjustments.append("price momentum")
                elif factor == 'risk_penalty' and value < 0:
                    technical_adjustments.append("risk adjustment")
        
        if technical_adjustments:
            explanation_parts.append(f"Technical factors: {', '.join(technical_adjustments)}")
        
        return ". ".join(explanation_parts) + "."
    
    def _generate_unified_commentary(self, signal: Dict[str, Any]) -> str:
        """
        Generate unified commentary combining score_explanation + ai_commentary.
        This creates a single narrative field for frontend consumption.
        
        Priority #3: Commentary Consolidation
        """
        ticker = signal.get('ticker', 'N/A')
        score = signal.get('weighted_score', 0)
        trade_type = signal.get('trade_type', 'Signal')
        risk_level = signal.get('risk_level', 'Medium')
        
        score_explanation = signal.get('score_explanation', '').strip()
        ai_commentary = signal.get('ai_commentary', '').strip()
        
        # Build unified commentary
        commentary_parts = []
        
        # Section 1: Score Explanation (Factual)
        if score_explanation:
            commentary_parts.append(f"**Signal Analysis**\n{score_explanation}")
        else:
            # Fallback: Generate basic score explanation
            mentions = signal.get('mentions', 0) or signal.get('mention_count', 0)
            reddit_score = signal.get('reddit_score', 0)
            financial_score = signal.get('financial_score', 0)
            
            basic_explanation = (
                f"{ticker} {trade_type} signal with weighted score of {score:.3f}. "
                f"Reddit sentiment: {reddit_score:.2f} from {mentions} mentions. "
                f"Financial metrics: {financial_score:.2f}. Risk level: {risk_level}."
            )
            commentary_parts.append(f"**Signal Analysis**\n{basic_explanation}")
        
        # Section 2: AI Commentary (Insights)
        if ai_commentary:
            # Check if it's a basic commentary or full AI commentary
            if signal.get('ai_commentary_version') == 'basic':
                # Skip adding basic commentary to unified view (redundant with score explanation)
                pass
            else:
                # Add full AI commentary
                commentary_parts.append(f"**Market Insights**\n{ai_commentary}")
        
        # Section 3: Key Metrics Summary (if available)
        metrics_summary = []
        
        current_price = signal.get('current_price')
        if current_price:
            metrics_summary.append(f"Price: ${current_price:.2f}")
        
        market_cap = signal.get('market_cap')
        if market_cap:
            if market_cap >= 1e9:
                metrics_summary.append(f"Market Cap: ${market_cap/1e9:.2f}B")
            elif market_cap >= 1e6:
                metrics_summary.append(f"Market Cap: ${market_cap/1e6:.2f}M")
        
        rsi = signal.get('rsi')
        if rsi:
            metrics_summary.append(f"RSI: {rsi:.1f}")
        
        volume_spike = signal.get('volume_spike_ratio')
        if volume_spike and volume_spike > 1.2:
            metrics_summary.append(f"Volume Spike: {volume_spike:.1f}x")
        
        if metrics_summary:
            commentary_parts.append(f"**Key Metrics**\n{', '.join(metrics_summary)}")
        
        # Combine all parts with double newlines
        unified_commentary = "\n\n".join(commentary_parts)
        
        return unified_commentary
    
    def _calculate_prediction_confidence(self, signal: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on data quality"""
        
        confidence = 0.5  # Base confidence
        
        # Data completeness boosts confidence
        if signal.get('rsi'):
            confidence += 0.05
        if signal.get('macd_line') and signal.get('macd_signal'):
            confidence += 0.05
        if signal.get('beta'):
            confidence += 0.05
        if signal.get('pe_ratio'):
            confidence += 0.05
        if signal.get('mention_count', 0) >= 2:
            confidence += 0.1
        if signal.get('volume_ratio', 1) > 1.2:
            confidence += 0.05
        
        # High risk reduces confidence
        risk_score = signal.get('risk_score', 50)
        if risk_score > 70:
            confidence -= 0.1
        elif risk_score < 30:
            confidence += 0.05
            
        return round(min(0.95, max(0.1, confidence)), 4)
    
    async def _enhance_signals_with_ai_commentary_efficient(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Efficient AI commentary enhancement using pre-prepared data"""
        try:
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            client = AsyncOpenAI()
            
            enhanced_signals = []
            for signal in signals:
                try:
                    # Use pre-prepared AI data summary
                    ai_data = signal.get('ai_data_summary', {})
                    ticker = signal.get('ticker', 'Unknown')
                    
                    # Generate concise AI commentary
                    prompt = f"""
                    Analyze {ticker} stock signal:
                    
                    Price Action: {ai_data.get('price_momentum', {})}
                    Technical: {ai_data.get('technical_signals', {})} 
                    Fundamentals: {ai_data.get('fundamental_context', {})}
                    
                    Provide 2-sentence analysis: 1) Key signal strength 2) Risk/opportunity
                    """
                    
                    response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    signal['ai_commentary'] = response.choices[0].message.content.strip()
                    signal['ai_commentary_timestamp'] = datetime.now().isoformat()
                    
                except Exception as e:
                    self.logger.debug(f"AI commentary failed for {signal.get('ticker')}: {e}")
                    signal['ai_commentary'] = None
                    
                enhanced_signals.append(signal)
                
            commentary_count = len([s for s in enhanced_signals if s.get('ai_commentary')])
            self.logger.info(f"✅ AI commentary generated for {commentary_count} signals")
            return enhanced_signals
            
        except Exception as e:
            self.logger.warning(f"AI commentary enhancement failed: {e}")
            return signals
    
    async def run_pipeline(self, 
                    subreddits: List[str] = None,
                    post_limit: int = 100,
                    min_mentions: int = 1,
                    max_signals: int = 50) -> Dict[str, Any]:
        """
        Run the complete unified pipeline.
        
        Args:
            subreddits (List[str]): Subreddits to scrape
            post_limit (int): Posts per subreddit
            min_mentions (int): Minimum mentions required
            max_signals (int): Maximum signals to process
            
        Returns:
            Dict[str, Any]: Pipeline execution results
        """
        pipeline_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("STARTING VP INVESTMENTS UNIFIED PIPELINE")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Reddit Data Collection
            self.logger.info("Step 1: Collecting Reddit data...")
            reddit_result = self.scrape_reddit_data(subreddits, post_limit)
            
            if 'ticker_mentions' not in reddit_result:
                raise ValueError("Reddit scraping failed - no ticker mentions found")
            
            ticker_mentions = reddit_result['ticker_mentions']
            
            # Step 2: Filter and Process Tickers
            self.logger.info("Step 2: Processing and filtering tickers...")
            
            # Filter tickers by minimum mentions and sort by relevance
            filtered_tickers = {
                ticker: data for ticker, data in ticker_mentions.items()
                if data['mention_count'] >= min_mentions
            }
            
            # Sort by mention count and reddit score
            sorted_tickers = sorted(
                filtered_tickers.items(),
                key=lambda x: (x[1]['mention_count'], x[1].get('reddit_score', 0)),
                reverse=True
            )[:max_signals]
            
            self.logger.info(f"Processing top {len(sorted_tickers)} tickers...")
            
            # Step 3: Generate Individual Signals from Each Data Source
            self.logger.info("Step 3: Generating individual signals...")
            
            # Get list of all tickers to analyze
            all_tickers = list(filtered_tickers.keys())
            self.logger.info(f"Analyzing {len(all_tickers)} tickers for signal generation")
            
            # Generate Reddit signals
            self.logger.info("Generating Reddit signals...")
            reddit_signals = self.generate_reddit_signals(filtered_tickers)
            self.logger.info(f"Generated {len(reddit_signals)} Reddit signals")
            
            # Generate Financial signals  
            self.logger.info("Generating Financial signals...")
            financial_signals = self.generate_financial_signals(all_tickers)
            self.logger.info(f"Generated {len(financial_signals)} Financial signals")
            
            # Generate News signals (if enabled)
            self.logger.info("Generating News signals...")
            news_signals = await self.generate_news_signals(all_tickers)
            self.logger.info(f"Generated {len(news_signals)} News signals")
            
            # Step 4: Combine All Signals into Scored Signals
            self.logger.info("Step 4: Combining signals into final scores...")
            signals = self.combine_signals_to_scored_signals(reddit_signals, financial_signals, news_signals)
            
            # Step 4.5: Comprehensive Signal Enhancement (CONSOLIDATED - ELIMINATES DUPLICATE API CALLS)
            self.logger.info("Step 4.5: Applying comprehensive signal enhancement...")
            signals = await self._comprehensive_signal_enhancement(signals)
            self.logger.info("✅ Comprehensive enhancement complete (technical + performance + AI prep)")
            
            # Step 4.6: Comprehensive AI Commentary Generation (TOP 10 ONLY) + Unified Commentary
            try:
                self.logger.info("Step 4.6: Generating unified commentary for signals...")
                
                # Sort signals by weighted_score and separate top 10
                sorted_signals = sorted(signals, key=lambda x: x.get('weighted_score', 0), reverse=True)
                top_signals = sorted_signals[:10]
                other_signals = sorted_signals[10:]
                
                # Generate comprehensive AI commentary for top 10 only
                from backend.integrations.ai import ComprehensiveCommentaryGenerator
                generator = ComprehensiveCommentaryGenerator()
                enhanced_top = await generator.enhance_signals_batch(top_signals)
                
                # Add basic commentary for remaining signals (no AI call)
                for signal in other_signals:
                    score = signal.get('weighted_score', 0)
                    trade_type = signal.get('trade_type', 'Signal')
                    risk = signal.get('risk_level', 'Medium')
                    ticker = signal.get('ticker', 'N/A')
                    signal['ai_commentary'] = f"{ticker} {trade_type} signal with score {score:.3f} ({risk} risk)"
                    signal['ai_commentary_version'] = "basic"
                
                # Recombine signals in score order
                signals = enhanced_top + other_signals
                
                # Generate unified commentary for all signals (consolidate score_explanation + ai_commentary)
                for signal in signals:
                    signal['commentary'] = self._generate_unified_commentary(signal)
                    signal['commentary_metadata'] = {
                        'has_ai_commentary': bool(signal.get('ai_commentary')),
                        'has_score_explanation': bool(signal.get('score_explanation')),
                        'ai_commentary_version': signal.get('ai_commentary_version', '1.0'),
                        'generated_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                
                self.logger.info(f"✅ AI commentary generated (10 full, {len(other_signals)} basic)")
                self.logger.info(f"✅ Unified commentary generated for all {len(signals)} signals")
                
            except Exception as e:
                self.logger.warning(f"Comprehensive AI commentary failed: {e}")
                # Try fallback AI commentary for top 10
                try:
                    sorted_signals = sorted(signals, key=lambda x: x.get('weighted_score', 0), reverse=True)
                    top_10 = sorted_signals[:10]
                    others = sorted_signals[10:]
                    
                    enhanced = await self._enhance_signals_with_ai_commentary_efficient(top_10)
                    
                    # Basic commentary for others
                    for sig in others:
                        sig['ai_commentary'] = f"Signal {sig.get('weighted_score', 0):.3f} - {sig.get('trade_type', 'N/A')}"
                    
                    signals = enhanced + others
                    
                    # Generate unified commentary for fallback
                    for signal in signals:
                        signal['commentary'] = self._generate_unified_commentary(signal)
                    
                    self.logger.info("✅ AI commentary generated (fallback, top 10 only)")
                    self.logger.info(f"✅ Unified commentary generated for all {len(signals)} signals (fallback)")
                except Exception as e2:
                    self.logger.warning(f"AI commentary fallback also failed: {e2}")
            
            # Step 4.7: Smart Backtest Scheduling (INTERVAL-BASED, NON-BLOCKING)
            try:
                self.logger.info("Step 4.7: Scheduling smart backtest for eligible signals...")
                from backend.integrations.backtest import run_smart_historical_backtest
                
                # Only run backtest on signals with elapsed intervals (non-blocking)
                backtest_results = await run_smart_historical_backtest(limit=100)
                
                eligible_count = backtest_results.get('successful_backtests', 0)
                if eligible_count > 0:
                    self.logger.info(f"✅ Smart backtest complete: {eligible_count} signals updated")
                    self.logger.info(f"   Processed intervals: {list(backtest_results.get('interval_results', {}).keys())}")
                else:
                    self.logger.info("✅ Smart backtest: No eligible signals (intervals not elapsed)")
                
                # Mark new signals for future backtesting (no immediate backtest)
                new_signal_count = 0
                for signal in signals:
                    if not signal.get('backtest_phase'):
                        signal['backtest_phase'] = 'scheduled'
                        signal['backtest_timestamp'] = datetime.now().isoformat()
                        new_signal_count += 1
                
                if new_signal_count > 0:
                    self.logger.info(f"   Scheduled {new_signal_count} new signals for future backtesting")
                
            except Exception as e:
                self.logger.warning(f"Smart backtest scheduling failed: {e}")
                # Mark signals for backtest anyway
                for signal in signals:
                    if not signal.get('backtest_phase'):
                        signal['backtest_phase'] = 'scheduled'
                        signal['backtest_timestamp'] = datetime.now().isoformat()
            
            # Step 4.8: Enhancement Complete
            self.logger.info("✅ All signal enhancements complete (comprehensive + AI + backtest scheduling)")
            
            # Limit to max_signals
            signals = signals[:max_signals]
            
            self.logger.info(f"Combined into {len(signals)} final scored signals")
            
            # Step 4.9: Calculate Historical Success Rates (NEW - Phase A)
            self.logger.info("Step 4.9: Calculating historical success rates...")
            try:
                from backend.integrations.backtest import calculate_historical_success_rates_for_signals
                signals = await calculate_historical_success_rates_for_signals(signals)
                self.logger.info("✅ Historical success rates calculated")
            except Exception as e:
                self.logger.warning(f"⚠️ Historical success rate calculation failed: {e}")
            
            # Step 4.9.5: Backtest Previous Signals (NEW - Phase A)
            self.logger.info("Step 4.9.5: Backtesting previous signals with elapsed time...")
            try:
                from backend.integrations.backtest import backtest_eligible_signals
                backtest_results = await backtest_eligible_signals(limit=100)
                
                if backtest_results['success']:
                    backtested = backtest_results.get('backtested_count', 0)
                    total = backtest_results.get('total_eligible', 0)
                    
                    if backtested > 0:
                        self.logger.info(f"✅ Backtested {backtested}/{total} previous signals")
                    else:
                        self.logger.info("⭕ No eligible signals to backtest yet (need 1+ day elapsed)")
                else:
                    self.logger.warning(f"⚠️ Backtest had issues: {backtest_results.get('error')}")
            except Exception as e:
                self.logger.warning(f"⚠️ Backtest failed: {e}")
            
            # Step 5: Save to Database
            self.logger.info("Step 5: Saving signals to database...")
            
            if signals:
                # Sort signals by weighted score
                signals.sort(key=lambda x: x['weighted_score'], reverse=True)
                
                # Save to database
                save_success = await self.save_signals_to_database(signals)
                
                # Step 6: Generate AI Strategies (NEW)
                ai_strategies_success = False
                ai_strategies_count = 0
                
                if save_success:
                    self.logger.info("Step 6: Generating AI strategies for top signals...")
                    try:
                        ai_strategies_result = await self._run_ai_strategy_generation()
                        ai_strategies_success = ai_strategies_result['success']
                        ai_strategies_count = ai_strategies_result.get('strategies_count', 0)
                        
                        if ai_strategies_success:
                            self.logger.info(f"✅ Generated {ai_strategies_count} AI strategies")
                        else:
                            self.logger.warning("⚠️ AI strategy generation had issues")
                            
                    except Exception as e:
                        self.logger.error(f"AI strategy generation failed: {e}")
                
                # Pipeline Results
                pipeline_end = datetime.now()
                execution_time = (pipeline_end - pipeline_start).total_seconds()
                
                results = {
                    'success': save_success,
                    'execution_time_seconds': execution_time,
                    'signals_generated': len(signals),
                    'ai_strategies_generated': ai_strategies_count,
                    'ai_strategies_success': ai_strategies_success,
                    'reddit_data': reddit_result['metadata'],
                    'top_signals': signals[:10],  # Top 10 for summary
                    'pipeline_timestamp': pipeline_end.isoformat()
                }
                
                self.logger.info("=" * 60)
                self.logger.info("PIPELINE EXECUTION COMPLETE")
                self.logger.info(f"Signals generated: {len(signals)}")
                self.logger.info(f"Database save: {'SUCCESS' if save_success else 'FAILED'}")
                self.logger.info(f"Execution time: {execution_time:.2f} seconds")
                self.logger.info("=" * 60)
                
                return results
            
            else:
                raise ValueError("No valid signals generated")
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - pipeline_start).total_seconds(),
                'signals_generated': 0
            }


async def main():
    """Main execution function for the unified pipeline."""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize and run pipeline
        pipeline = UnifiedPipeline()
        
        # Run with default parameters
        results = await pipeline.run_pipeline(
            subreddits=['stocks', 'investing', 'wallstreetbets'],
            post_limit=100,
            min_mentions=1,
            max_signals=50
        )
        
        # Print summary
        if results['success']:
            print(f"\nPipeline completed successfully!")
            print(f"Generated {results['signals_generated']} signals")
            print(f"Execution time: {results['execution_time_seconds']:.2f}s")
            
            # AI Strategy Results
            if results.get('ai_strategies_generated', 0) > 0:
                print(f"🤖 Generated {results['ai_strategies_generated']} AI strategies")
                print(f"   AI Strategy Success: {'✅' if results.get('ai_strategies_success') else '❌'}")
            
            if 'top_signals' in results:
                print(f"\nTop 5 signals:")
                for i, signal in enumerate(results['top_signals'][:5], 1):
                    ticker = signal['ticker']
                    score = signal['weighted_score']
                    mentions = signal.get('mentions', signal.get('reddit_data', {}).get('mention_count', 0))
                    trade_type = signal.get('trade_type', 'Speculative')
                    risk_level = signal.get('risk_level', 'Medium')
                    print(f"  {i}. {ticker}: {score:.3f} ({mentions} mentions, {trade_type}, {risk_level} risk)")
        else:
            print(f"\nPipeline failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
    finally:
        # Clean up any pending tasks and connections
        try:
            # Get current event loop
            loop = asyncio.get_event_loop()
            
            # Cancel any pending tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending_tasks:
                logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                for task in pending_tasks:
                    task.cancel()
                
                # Wait for tasks to cancel with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not cancel within timeout")
            
            logger.info("Pipeline cleanup completed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    asyncio.run(main())