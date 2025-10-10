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
    logger.warning(f"Integrators not available: {e}")

# Import SignalScorer for scoring methods (Phase 6a consolidation)
from backend.core.signals import SignalScorer# Simple configuration class
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
        
        # Initialize SignalScorer for scoring methods (Phase 6a consolidation)
        self.signal_scorer = SignalScorer()
        
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
        """Delegate to RedditDataIntegrator for ticker extraction"""
        return self.reddit.extract_tickers_pipeline(text)
    
    def scrape_reddit_data(self, subreddits: List[str] = None, post_limit: int = 100) -> Dict[str, Any]:
        """Delegate to RedditDataIntegrator for Reddit scraping"""
        from backend.integrations.reddit import RedditDataIntegrator
        reddit_integrator = RedditDataIntegrator()
        return reddit_integrator.scrape_subreddits_pipeline(subreddits, post_limit, self.sentiment_analyzer)
    
    def get_financial_data(self, ticker: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Delegate to YahooFinanceIntegrator for financial data fetching"""
        from backend.integrations.yfinance import YahooFinanceIntegrator
        yf_integrator = YahooFinanceIntegrator()
        return yf_integrator.get_comprehensive_financial_data(ticker, use_cache)
    

    
    async def get_news_data(self, ticker: str) -> Dict[str, Any]:
        """Get news sentiment data for a ticker."""
        if self.enhanced_available and self.news_integrator:
            try:
                return await self.news_integrator.get_news_sentiment(ticker)
            except Exception as e:
                self.logger.warning(f"News data failed for {ticker}: {e}")
        
        return {
            'news_score': None,
            'news_sentiment_score': None,
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
                    'signal_confidence': self._safe_round(signal['weighted_score'], 4),
                    'top_factors': 'Reddit mentions, price momentum',
                    'signal_type': 'Multi-Factor',
                    
                    # Price and market data
                    'current_price': self._safe_round(financial_data.get('current_price'), 2),
                    'market_cap': financial_data.get('market_cap'),
                    'avg_daily_value_traded': self._safe_round(financial_data.get('avg_daily_value_traded'), 0),
                    
                    # Reddit metrics
                    'reddit_sentiment': self._safe_round(signal.get('reddit_data', {}).get('avg_sentiment'), 4),
                    'news_sentiment_score': self._safe_round(signal.get('news_sentiment_score', signal.get('news_sentiment', 0)), 4),
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
                    'momentum_consistency_score': self._safe_round(signal.get('momentum_consistency_score'), 2),  # Phase 1 ML metric
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
                    
                    # Phase 3: Analyst Data
                    'analyst_target_price': self._safe_round(financial_data.get('analyst_target_price'), 2),
                    'analyst_target_upside_pct': self._safe_round(financial_data.get('analyst_target_upside_pct'), 2),
                    'analyst_recommendation_mean': self._safe_round(financial_data.get('analyst_recommendation_mean'), 2),
                    'analyst_count': financial_data.get('analyst_count'),
                    
                    # Phase 3: Earnings Surprise Data
                    'last_earnings_surprise_pct': self._safe_round(financial_data.get('last_earnings_surprise_pct'), 2),
                    'avg_earnings_surprise_pct': self._safe_round(financial_data.get('avg_earnings_surprise_pct'), 2),
                    'earnings_surprise_trend': financial_data.get('earnings_surprise_trend'),
                    
                    # Phase 3: Institutional Activity
                    'institutional_change_qoq': self._safe_round(financial_data.get('institutional_change_qoq'), 2),
                    'top_10_institutional_holders_pct': self._safe_round(financial_data.get('top_10_institutional_holders_pct'), 2),
                    'num_institutional_holders': financial_data.get('num_institutional_holders'),
                    
                    # Phase 3: Insider Trading
                    'insider_activity_score': self._safe_round(financial_data.get('insider_activity_score'), 2),
                    'insider_buy_count': financial_data.get('insider_buy_count'),
                    'insider_sell_count': financial_data.get('insider_sell_count'),
                    'insider_net_shares': financial_data.get('insider_net_shares'),
                    
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
                    'created_at': current_time.isoformat(),
                    'updated_at': current_time.isoformat()
                }
                
                enhanced_signals.append(basic_record)
            
            # ENHANCEMENT: Apply signal enhancement calculations
            enhanced_signals = self._apply_signal_enhancements(enhanced_signals)
            
            # Debug: Check for potential overflow values
            for i, record in enumerate(enhanced_signals[:3]):  # Check first 3 records
                for key, value in record.items():
                    if isinstance(value, (int, float)) and value is not None and not isinstance(value, bool):
                        try:
                            if abs(value) >= 10:
                                self.logger.warning(f"Potential overflow in record {i}, field '{key}': {value}")
                        except (TypeError, ValueError):
                            # Skip comparison if value can't be used with abs()
                            pass
            
            # DENORMALIZED STRUCTURE: All signal data in signals table (including metrics)
            
            # Helper function to convert float to int for bigint columns
            def to_bigint(value):
                """Convert numeric value to integer for bigint columns."""
                if value is None:
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            # Step 1: Prepare COMPLETE signal data (signals table with ALL fields)
            core_signals = []
            metrics_data = []
            
            for record in enhanced_signals:
                # ALL signal fields including metrics (denormalized approach)
                core_signal = {
                    # Core identification
                    'run_id': record['run_id'],
                    'ticker': record['ticker'],
                    'company': record['company'],
                    'sector': record['sector'],
                    # Scoring
                    'weighted_score': record['weighted_score'],
                    'reddit_score': record['reddit_score'],
                    'news_score': record['news_score'],
                    'financial_score': record['financial_score'],
                    'trade_type': record['trade_type'],
                    'risk_level': record['risk_level'],
                    'risk_tags': record['risk_tags'],
                    'risk_assessment': record['risk_assessment'],
                    'rank': record['rank'],
                    'signal_confidence': record['signal_confidence'],
                    'top_factors': record['top_factors'],
                    'signal_type': record['signal_type'],
                    # Price & market cap
                    'current_price': record['current_price'],
                    'market_cap': record['market_cap'],
                    'avg_daily_value_traded': record['avg_daily_value_traded'],
                    # Social metrics
                    'reddit_sentiment': record['reddit_sentiment'],
                    'news_sentiment_score': record.get('news_sentiment_score', record.get('news_sentiment', 0)),
                    'mentions': record['mentions'],
                    'news_mentions': record['news_mentions'],
                    'upvotes': record['upvotes'],
                    'post_recency': record['post_recency'],
                    # Price action
                    'price_1d_pct': record['price_1d_pct'],
                    'price_7d_pct': record['price_7d_pct'],
                    'volume': record['volume'],
                    'liquidity_warning': record['liquidity_warning'],
                    'emerging': record['emerging'],
                    # Commentary
                    'ai_commentary': record['ai_commentary'],
                    'score_explanation': record['score_explanation'],
                    # Timestamps
                    'created_at': record['created_at'],
                    'updated_at': record['updated_at'],
                    
                    # METRICS FIELDS (v2.0 enhancements) - now included in signals table
                    # Technical indicators - Momentum
                    'relative_strength': record.get('relative_strength'),
                    'momentum_30d_pct': record.get('momentum_30d_pct'),
                    'rsi': record.get('rsi'),
                    'macd_histogram': record.get('macd_histogram'),
                    'macd_line': record.get('macd_line'),
                    'macd_signal': record.get('macd_signal'),
                    'signal_strength_percentile': record.get('signal_strength_percentile'),
                    'sector_relative_strength': record.get('sector_relative_strength'),
                    'momentum_consistency_score': record.get('momentum_consistency_score'),
                    # Volatility
                    'volatility': record.get('volatility'),
                    'volatility_rank': record.get('volatility_rank'),
                    'bollinger_width': record.get('bollinger_width'),
                    'bollinger_upper': record.get('bollinger_upper'),
                    'bollinger_lower': record.get('bollinger_lower'),
                    'bollinger_position': record.get('bollinger_position'),
                    'beta': record.get('beta'),
                    # Moving averages
                    'above_50d_ma_pct': record.get('above_50d_ma_pct'),
                    'above_200d_ma_pct': record.get('above_200d_ma_pct'),
                    # Volume
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
                    # Options
                    'put_call_oi_ratio': record.get('put_call_oi_ratio'),
                    'put_call_vol_ratio': record.get('put_call_vol_ratio'),
                    'iv_spike_pct': record.get('iv_spike_pct'),
                    'implied_volatility': record.get('implied_volatility'),
                    # v2.0 New fields - Ownership
                    'institutional_ownership_pct': record.get('institutional_ownership_pct'),
                    'retail_holding_pct': record.get('retail_holding_pct'),
                    'insider_buy_volume': to_bigint(record.get('insider_buy_volume')),
                    # v2.0 New fields - Short interest
                    'short_pct_float': record.get('short_pct_float'),
                    'short_pct_outstanding': record.get('short_pct_outstanding'),
                    'shares_short': to_bigint(record.get('shares_short')),
                    'short_ratio': record.get('short_ratio'),
                    # Phase 3: Analyst Data
                    'analyst_target_price': record.get('analyst_target_price'),
                    'analyst_target_upside_pct': record.get('analyst_target_upside_pct'),
                    'analyst_recommendation_mean': record.get('analyst_recommendation_mean'),
                    'analyst_count': record.get('analyst_count'),
                    # Phase 3: Earnings Surprise Data
                    'last_earnings_surprise_pct': record.get('last_earnings_surprise_pct'),
                    'avg_earnings_surprise_pct': record.get('avg_earnings_surprise_pct'),
                    'earnings_surprise_trend': record.get('earnings_surprise_trend'),
                    # Phase 3: Institutional Activity
                    'institutional_change_qoq': record.get('institutional_change_qoq'),
                    'top_10_institutional_holders_pct': record.get('top_10_institutional_holders_pct'),
                    'num_institutional_holders': record.get('num_institutional_holders'),
                    # Phase 3: Insider Trading
                    'insider_activity_score': record.get('insider_activity_score'),
                    'insider_buy_count': record.get('insider_buy_count'),
                    'insider_sell_count': record.get('insider_sell_count'),
                    'insider_net_shares': to_bigint(record.get('insider_net_shares')),
                    # Composite scores
                    'exit_signal_strength': record.get('exit_signal_strength'),
                    'risk_score': record.get('risk_score'),
                    'liquidity_score': record.get('liquidity_score'),
                    'risk_category': record.get('risk_category'),
                    'max_position_size': record.get('max_position_size'),
                    # Phase 1.2 composite metrics
                    'market_cap_category': record.get('market_cap_category'),
                    'expected_hold_duration': record.get('expected_hold_duration'),
                    # Phase 1.3 calendar events
                    'earnings_date': record.get('earnings_date'),
                    'dividend_ex_date': record.get('dividend_ex_date'),
                    'analyst_targets': record.get('analyst_targets'),
                }
                core_signals.append(core_signal)
            
            # Step 2: Insert core signals first
            result_signals = self.supabase.table('signals').insert(core_signals).execute()
            
            if not result_signals.data:
                self.logger.error("[ERROR] Database insertion failed for signals table")
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
                    # Phase 3: Analyst Data
                    'analyst_target_price': record.get('analyst_target_price'),
                    'analyst_target_upside_pct': record.get('analyst_target_upside_pct'),
                    'analyst_recommendation_mean': record.get('analyst_recommendation_mean'),
                    'analyst_count': record.get('analyst_count'),
                    # Phase 3: Earnings Surprise Data
                    'last_earnings_surprise_pct': record.get('last_earnings_surprise_pct'),
                    'avg_earnings_surprise_pct': record.get('avg_earnings_surprise_pct'),
                    'earnings_surprise_trend': record.get('earnings_surprise_trend'),
                    # Phase 3: Institutional Activity
                    'institutional_change_qoq': record.get('institutional_change_qoq'),
                    'top_10_institutional_holders_pct': record.get('top_10_institutional_holders_pct'),
                    'num_institutional_holders': record.get('num_institutional_holders'),
                    # Phase 3: Insider Trading
                    'insider_activity_score': record.get('insider_activity_score'),
                    'insider_buy_count': record.get('insider_buy_count'),
                    'insider_sell_count': record.get('insider_sell_count'),
                    'insider_net_shares': to_bigint(record.get('insider_net_shares')),
                    # Metadata
                    'created_at': record['created_at'],
                    'updated_at': record['updated_at']
                }
                metrics_data.append(metrics_record)
            
            # Step 4: All metrics data now stored in signals table (signal_metrics table dropped in Phase 4.1)
            self.logger.info(f"[SUCCESS] Successfully saved {len(result_signals.data)} signals to database")
            
            # Note: signals_norm materialized view refresh removed - we now use signals table directly
            # If you recreate signals_norm view in Supabase, uncomment this block:
            # try:
            #     self.supabase.rpc('refresh_signals_norm').execute()
            #     self.logger.info(f"[SUCCESS] Refreshed signals_norm materialized view")
            # except Exception as refresh_error:
            #     self.logger.warning(f"[WARNING] signals_norm refresh skipped: {refresh_error}")
            
            # Return success and run_id
            return {'success': True, 'run_id': run_id}
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error saving signals to database: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'run_id': None}
    
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
    
    async def _run_ai_strategy_generation(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Run AI strategy generation for top signals"""
        try:
            # Check if AI strategies are enabled
            ai_enabled = os.getenv('AI_STRATEGY_ENABLED', 'false').lower() == 'true'
            
            if not ai_enabled:
                self.logger.info("AI strategy generation disabled, skipping")
                return {'success': True, 'strategies_count': 0, 'message': 'AI strategies disabled'}
            
            # Import AI strategy generator
            from backend.integrations.ai import AIStrategyGenerator
            
            # Initialize and run AI strategy generator with run_id
            generator = AIStrategyGenerator(run_id=run_id)
            
            if not generator.ai_enabled:
                self.logger.warning("AI strategy generator not properly initialized")
                return {'success': False, 'strategies_count': 0, 'message': 'AI generator not initialized'}
            
            # Generate strategies for top signals
            self.logger.info(f"Generating AI strategies for top {generator.top_signals_limit} signals...")
            strategies = await generator.generate_strategies_for_top_signals()
            
            if strategies:
                total_strategies = sum(len(s) for s in strategies.values())
                self.logger.info(f"[SUCCESS] Generated {total_strategies} AI strategies for {len(strategies)} tickers")
                
                # Log strategy summary
                strategy_summary = []
                for ticker, ticker_strategies in strategies.items():
                    strategy_types = [s.strategy_type for s in ticker_strategies]
                    strategy_summary.append(f"{ticker}: {len(ticker_strategies)} ({', '.join(strategy_types)})")
                    self.logger.info(f"   [STATS] {ticker}: {len(ticker_strategies)} strategies")
                
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
                
                # Reddit signal score (0-1 scale) - delegated to SignalScorer
                reddit_score = self.signal_scorer._calculate_reddit_score(mention_count, avg_sentiment, avg_score)
                
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
                
                # Calculate financial signal score (delegated to SignalScorer - Phase 6c)
                financial_score = self.signal_scorer._calculate_financial_score(financial_data)
                
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
    
    def generate_financial_signals_cached(self, tickers: List[str], ticker_cache: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Generate financial-based signals using PRE-CACHED ticker data.
        NO API CALLS - all data already fetched!
        
        Args:
            tickers: List of tickers to analyze
            ticker_cache: Pre-fetched ticker data cache
            
        Returns:
            List of financial signals with scores and metadata
        """
        financial_signals = []
        
        for ticker in tickers:
            try:
                # Get cached ticker data (NO API CALL!)
                ticker_data = ticker_cache.get(ticker)
                
                if not ticker_data or ticker_data.get('stock') is None:
                    self.logger.debug(f"No cached data for {ticker}, skipping")
                    continue
                
                # Convert cached data to financial_data format
                financial_data = self._convert_cache_to_financial_data(ticker_data)
                
                if not financial_data:
                    continue
                
                # Calculate financial signal score (delegated to SignalScorer - Phase 6c)
                financial_score = self.signal_scorer._calculate_financial_score(financial_data)
                
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
        self.logger.info(f"✅ Generated {len(financial_signals)} financial signals using cached data (0 API calls)")
        return financial_signals
    
    def _convert_cache_to_financial_data(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert cached ticker data to financial_data format.
        This bridges the cache format with what _calculate_financial_score expects.
        """
        try:
            import pandas as pd
            import numpy as np
            import ta
            
            info = ticker_data.get('info', {})
            history_1y = ticker_data.get('history_1y', pd.DataFrame())
            history_3m = ticker_data.get('history_3m', pd.DataFrame())
            history_1m = ticker_data.get('history_1m', pd.DataFrame())
            
            if history_1m.empty:
                return None
            
            # Build financial_data dict matching expected format
            financial_data = {}
            
            # Basic info
            financial_data['ticker'] = ticker_data.get('ticker')
            financial_data['market_cap_numeric'] = info.get('marketCap')
            financial_data['pe_ratio'] = info.get('trailingPE')
            financial_data['forward_pe'] = info.get('forwardPE')
            financial_data['profit_margin'] = info.get('profitMargins')
            financial_data['roe'] = info.get('returnOnEquity')
            financial_data['revenue_growth'] = info.get('revenueGrowth')
            financial_data['earnings_growth'] = info.get('earningsGrowth')
            financial_data['debt_equity'] = info.get('debtToEquity')
            
            # Price and volume data
            if not history_1m.empty:
                prices = history_1m['Close']
                volumes = history_1m['Volume']
                
                financial_data['current_price'] = float(prices.iloc[-1])
                financial_data['volume'] = int(volumes.iloc[-1])
                financial_data['avg_volume_30d'] = int(volumes.mean())
                financial_data['volume_spike_ratio'] = float(volumes.iloc[-1] / volumes.mean()) if volumes.mean() > 0 else 1.0
                
                # Price momentum
                if len(prices) >= 2:
                    financial_data['price_1d_pct'] = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100)
                if len(prices) >= 7:
                    financial_data['price_7d_pct'] = float((prices.iloc[-1] / prices.iloc[-7] - 1) * 100)
                if len(prices) >= 30:
                    financial_data['momentum_30d_pct'] = float((prices.iloc[-1] / prices.iloc[-30] - 1) * 100)
                
                # Volatility
                financial_data['volatility'] = float(prices.pct_change().std() * np.sqrt(252) * 100)
                financial_data['volatility_rank'] = 50  # Placeholder
                
                # Volume-price correlation
                if len(prices) >= 30 and len(volumes) >= 30:
                    price_changes = prices.pct_change().dropna()
                    volume_changes = volumes.pct_change().dropna()
                    if len(price_changes) > 0 and len(volume_changes) > 0:
                        correlation = price_changes.corr(volume_changes)
                        financial_data['volume_price_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            
            # Technical indicators from 3-month data
            if not history_3m.empty and len(history_3m) >= 26:
                df = history_3m
                
                # RSI
                rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
                if not rsi.empty:
                    financial_data['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
                
                # MACD
                macd = ta.trend.MACD(df['Close']).macd()
                if not macd.empty:
                    financial_data['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None
                
                # Moving averages
                if len(df) >= 50:
                    ma_50 = df['Close'].rolling(50).mean().iloc[-1]
                    current_price = df['Close'].iloc[-1]
                    financial_data['above_50d_ma_pct'] = float((current_price / ma_50 - 1) * 100) if not np.isnan(ma_50) else None
                
                if len(df) >= 200 and not history_1y.empty and len(history_1y) >= 200:
                    ma_200 = history_1y['Close'].rolling(200).mean().iloc[-1]
                    current_price = history_1y['Close'].iloc[-1]
                    financial_data['above_200d_ma_pct'] = float((current_price / ma_200 - 1) * 100) if not np.isnan(ma_200) else None
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df['Close'])
                bb_upper = bb.bollinger_hband().iloc[-1] if not bb.bollinger_hband().empty else None
                bb_lower = bb.bollinger_lband().iloc[-1] if not bb.bollinger_lband().empty else None
                current_price = df['Close'].iloc[-1]
                
                if bb_upper and bb_lower and not np.isnan(bb_upper) and not np.isnan(bb_lower):
                    bb_range = bb_upper - bb_lower
                    if bb_range > 0:
                        financial_data['bollinger'] = float((current_price - bb_lower) / bb_range)
            
            # Options data
            financial_data['put_call_ratio'] = info.get('putCallRatio')
            financial_data['put_call_vol_ratio'] = info.get('putCallVolumeRatio')
            
            # Short interest
            financial_data['short_interest'] = info.get('shortPercentOfFloat')
            financial_data['short_ratio'] = info.get('shortRatio')
            
            # Sector/relative strength (placeholder - would need market data)
            financial_data['sector_relative_strength'] = 0.0
            financial_data['relative_strength'] = 0.0
            
            # Phase 3: Add Phase 3 fundamental data from cache
            # Map field names from yfinance methods to database schema
            phase3_data = ticker_data.get('phase3_data', {})
            if phase3_data:
                # Phase 3: Analyst Data (map field names)
                financial_data['analyst_target_price'] = phase3_data.get('target_price_mean')
                financial_data['analyst_target_upside_pct'] = phase3_data.get('target_upside_pct')
                financial_data['analyst_recommendation_mean'] = phase3_data.get('recommendation_mean')
                financial_data['analyst_count'] = phase3_data.get('num_analysts')
                
                # Phase 3: Earnings Surprise Data (field names match)
                financial_data['last_earnings_surprise_pct'] = phase3_data.get('last_earnings_surprise_pct')
                financial_data['avg_earnings_surprise_pct'] = phase3_data.get('avg_earnings_surprise_pct')
                financial_data['earnings_surprise_trend'] = phase3_data.get('earnings_surprise_trend')
                
                # Phase 3: Institutional Activity (map field names)
                financial_data['institutional_change_qoq'] = phase3_data.get('institutional_change_qoq')
                financial_data['top_10_institutional_holders_pct'] = phase3_data.get('top_10_holders_pct')
                financial_data['num_institutional_holders'] = phase3_data.get('num_institutions')
                
                # Phase 3: Insider Trading (map field names)
                financial_data['insider_activity_score'] = phase3_data.get('insider_activity_score')
                financial_data['insider_buy_count'] = phase3_data.get('insider_buy_transactions_3m')
                financial_data['insider_sell_count'] = phase3_data.get('insider_sell_transactions_3m')
                financial_data['insider_net_shares'] = phase3_data.get('insider_net_shares_3m')
            
            return financial_data
            
        except Exception as e:
            self.logger.debug(f"Error converting cache to financial_data for {ticker_data.get('ticker')}: {e}")
            return None
    
    # ===== REMOVED: _calculate_financial_score() - moved to SignalScorer (Phase 6c-Part3) - 77 lines =====
    # ===== REMOVED: _calculate_technical_score() - moved to SignalScorer (Phase 6c-Part2) - 235 lines =====
    # ===== REMOVED: _calculate_fundamentals_score() - moved to SignalScorer (Phase 6c-Part1) - 364 lines =====
    
    def _clamp_decimal(self, value: Optional[float], min_val: float, max_val: float) -> Optional[float]:
        """Clamp a decimal value to database field limits and round to 2 decimal places."""
        if value is None:
            return None
        clamped = max(min_val, min(max_val, value))
        return round(clamped, 2)
    
    def _safe_round(self, value: Optional[float], decimals: int = 2) -> Optional[float]:
        """Safely round a value to specified decimals, returning None if value is None."""
        return round(value, decimals) if value is not None else None
    
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
                
                # Calculate news signal score - delegated to SignalScorer
                news_score = self.signal_scorer._calculate_news_score(news_data=news_data)
                
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
                enhanced['market_cap_category'] = None  # NULL for missing data, not 'Unknown'
            
            # Basic risk score calculation
            volatility = signal.get('volatility') or 0.15
            debt_equity = signal.get('debt_equity') or 25  # Handle None explicitly
            
            risk_score = min(100, max(0, 
                volatility * 30 +  # Volatility component
                (25 if debt_equity > 100 else 10 if debt_equity > 50 else 5) +  # Debt component
                (15 if market_cap and market_cap > 0 and market_cap < 1_000_000_000 else 5)  # Size component
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
    
    async def _fetch_all_ticker_data_once(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetch comprehensive data for all tickers in parallel - ONCE!
        This eliminates duplicate API calls between generate_financial_signals and enhancement.
        
        Returns:
            Dict mapping ticker -> comprehensive_data
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        self.logger.info(f"📊 Fetching comprehensive data for {len(tickers)} tickers (SINGLE PASS)...")
        
        ticker_cache = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            
            # Fetch all tickers in parallel
            tasks = [
                loop.run_in_executor(executor, self._fetch_ticker_data_sync, ticker) 
                for ticker in tickers
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Build cache from results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.debug(f"Ticker fetch failed: {result}")
                    continue
                    
                if result and 'ticker' in result:
                    ticker_cache[result['ticker']] = result
        
        self.logger.info(f"✅ Successfully cached data for {len(ticker_cache)}/{len(tickers)} tickers")
        return ticker_cache
    
    async def _comprehensive_signal_enhancement(self, signals: List[Dict[str, Any]], 
                                               ticker_cache: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive enhancement using PRE-CACHED ticker data.
        NO MORE DUPLICATE API CALLS!
        
        Args:
            signals: List of signals to enhance
            ticker_cache: Pre-fetched ticker data cache (if None, will fetch - inefficient fallback)
        
        Consolidates Steps 4.5-4.8 into single efficient process:
        - Uses pre-cached ticker data (no API calls!)
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
        
        # If no cache provided, fetch data (fallback - shouldn't happen)
        if ticker_cache is None:
            self.logger.warning("⚠️  No ticker cache provided! Fetching data (inefficient fallback)...")
            unique_tickers = list(set(s.get('ticker', '').upper() for s in signals if s.get('ticker')))
            ticker_cache = await self._fetch_all_ticker_data_once(unique_tickers)
        
        # Group signals by ticker
        ticker_groups = {}
        for signal in signals:
            ticker = signal.get('ticker', '').upper()
            if ticker:
                if ticker not in ticker_groups:
                    ticker_groups[ticker] = []
                ticker_groups[ticker].append(signal)
        
        self.logger.info(f"Enhancing {len(signals)} signals grouped into {len(ticker_groups)} unique tickers")
        
        # Apply enhancements using cached data
        enhanced_signals = []
        for ticker, ticker_signals in ticker_groups.items():
            try:
                # Get cached data (NO API CALL!)
                ticker_data = ticker_cache.get(ticker)
                
                if not ticker_data:
                    self.logger.debug(f"No cached data for {ticker}, skipping enhancement")
                    enhanced_signals.extend(ticker_signals)
                    continue
                
                # Apply all enhancements to ticker signals
                for signal in ticker_signals:
                    enhanced_signal = self._apply_all_enhancements_to_signal(signal, ticker_data)
                    enhanced_signals.append(enhanced_signal)
                    
            except Exception as e:
                self.logger.warning(f"Enhancement failed for {ticker}: {e}")
                # Add original signals without enhancement
                enhanced_signals.extend(ticker_signals)
        
        self.logger.info(f"[SUCCESS] Comprehensive enhancement complete: {len(enhanced_signals)} signals")
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
            
            # Phase 3: Fetch Phase 3 fundamental data from yfinance integration
            phase3_data = {}
            try:
                from backend.integrations.yfinance import FinancialMetricsCalculator
                metrics_calc = FinancialMetricsCalculator()
                
                # Get current price for analyst data
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                if not history_1m.empty:
                    current_price = float(history_1m['Close'].iloc[-1])
                
                # Get analyst data (requires info and current_price)
                if current_price:
                    analyst_data = metrics_calc._get_analyst_data(stock, info, current_price)
                    phase3_data.update(analyst_data)
                
                # Get earnings surprise data (requires only stock)
                earnings_data = metrics_calc._get_earnings_surprise_data(stock)
                phase3_data.update(earnings_data)
                
                # Get institutional ownership data (requires stock and info)
                institutional_data = metrics_calc._get_institutional_ownership_data(stock, info)
                phase3_data.update(institutional_data)
                
                # Get insider trading data (requires only stock)
                insider_data = metrics_calc._get_insider_trading_data(stock)
                phase3_data.update(insider_data)
                
                self.logger.debug(f"Phase 3 data collected for {ticker}: {len(phase3_data)} fields")
                
            except Exception as e:
                self.logger.debug(f"Phase 3 data fetch failed for {ticker}: {e}")
                phase3_data = {}
            
            return {
                'ticker': ticker,
                'stock': stock,
                'info': info,
                'history_1y': history_1y,
                'history_3m': history_3m,
                'history_1m': history_1m,
                'phase3_data': phase3_data  # Phase 3 fields
            }
            
        except Exception as e:
            self.logger.debug(f"Data fetch failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'stock': None,
                'info': {},
                'history_1y': pd.DataFrame(),
                'history_3m': pd.DataFrame(),
                'history_1m': pd.DataFrame(),
                'phase3_data': {}
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
        
        # ML Analytics - Phase 1 metrics (momentum_consistency_score, pattern_match_score, etc.)
        try:
            from backend.integrations.signal_processing import SignalMLAnalyzer
            analyzer = SignalMLAnalyzer()
            before_keys = set(enhanced_signal.keys())
            enhanced_signal = analyzer.enhance_signal_with_ml_analytics(enhanced_signal)
            after_keys = set(enhanced_signal.keys())
            new_keys = after_keys - before_keys
            if 'momentum_consistency_score' in new_keys:
                self.logger.info(f"[ML] Added momentum_consistency_score={enhanced_signal.get('momentum_consistency_score')} to {enhanced_signal.get('ticker')}")
            else:
                self.logger.warning(f"[ML] momentum_consistency_score NOT added to {enhanced_signal.get('ticker')}")
        except Exception as e:
            import traceback
            self.logger.error(f"ML analytics enhancement failed for {enhanced_signal.get('ticker')}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Score components and explanation (NEW) - delegated to SignalScorer (Phase 6c)
        enhanced_signal = self.signal_scorer._calculate_score_components(enhanced_signal)
        
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
            
            # Beta calculation - delegate to YahooFinanceIntegrator
            from backend.integrations.yfinance import YahooFinanceIntegrator
            yf_integrator = YahooFinanceIntegrator()
            signal['beta'] = yf_integrator.calculate_beta(ticker_data)
            
        except Exception as e:
            self.logger.debug(f"Technical indicators failed for {signal.get('ticker')}: {e}")
            
        return signal
    

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
    
    # ===== REMOVED: _calculate_score_components() - moved to SignalScorer (Phase 6c-Part3) - 50 lines =====
    
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
            self.logger.info(f"[SUCCESS] AI commentary generated for {commentary_count} signals")
            return enhanced_signals
            
        except Exception as e:
            self.logger.warning(f"AI commentary enhancement failed: {e}")
            return signals
    
    async def generate_single_signal(self, ticker: str, include_reddit: bool = True) -> Dict[str, Any]:
        """
        Generate a complete signal for a single ticker (for on-demand user requests).
        
        This is the primary method for generating signals on-demand from the frontend.
        It handles the complete flow: data collection → scoring → enhancement → storage.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            include_reddit (bool): Whether to include Reddit sentiment data (default: True)
            
        Returns:
            Dict[str, Any]: Complete signal with all enhancements, or None if failed
            
        Example:
            >>> pipeline = UnifiedPipeline()
            >>> signal = await pipeline.generate_single_signal('AAPL')
            >>> print(f"Score: {signal['signal_score']}, Beta: {signal['beta']}")
        """
        try:
            self.logger.info(f"🎯 Generating signal for {ticker}...")
            start_time = datetime.now()
            
            # Validate ticker
            ticker = ticker.upper().strip()
            if not ticker or len(ticker) > 10:
                raise ValueError(f"Invalid ticker: {ticker}")
            
            # Step 1: Generate base financial signal
            self.logger.info(f"Step 1/4: Fetching financial data for {ticker}...")
            financial_signals = self.generate_financial_signals([ticker])
            
            if not financial_signals:
                self.logger.error(f"Failed to generate financial signal for {ticker}")
                return None
            
            signal = financial_signals[0]
            
            # Step 2: Add Reddit data if requested
            # Note: For now, Reddit data requires full pipeline run with scraping
            # Individual ticker Reddit lookup can be added in future enhancement
            self.logger.info(f"Step 2/4: Setting default Reddit values (full scraping not in single signal mode)")
            signal['upvotes'] = 0
            signal['reddit_score'] = 0
            signal['sentiment_score'] = 0
            signal['mention_count'] = 0
            
            # Step 3: Comprehensive enhancement (technical indicators, beta, etc.)
            self.logger.info(f"Step 3/4: Enhancing signal with technical data...")
            enhanced_signals = await self._comprehensive_signal_enhancement(
                [signal],
                ticker_cache=None  # Will fetch fresh data
            )
            
            if not enhanced_signals:
                self.logger.error(f"Enhancement failed for {ticker}")
                return None
            
            enhanced_signal = enhanced_signals[0]
            
            # Step 4: Save to database
            self.logger.info(f"Step 4/4: Saving signal to database...")
            
            # Add weighted_score default if missing (for database compatibility)
            if 'weighted_score' not in enhanced_signal:
                enhanced_signal['weighted_score'] = enhanced_signal.get('signal_score', 0)
            
            save_success = await self.save_signals_to_database([enhanced_signal])
            
            if save_success:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"✅ SUCCESS: Signal for {ticker} generated and saved in {elapsed:.2f}s")
                self.logger.info(f"   Score: {enhanced_signal.get('signal_score', 'N/A')}")
                self.logger.info(f"   Beta: {enhanced_signal.get('beta', 'N/A')}")
                self.logger.info(f"   MACD: {enhanced_signal.get('macd_line', 'N/A')}")
                self.logger.info(f"   Upvotes: {enhanced_signal.get('upvotes', 'N/A')}")
            else:
                self.logger.warning(f"⚠️  Signal generated but database save failed")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate signal for {ticker}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
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
            
            # Get list of all tickers to analyze
            all_tickers = list(filtered_tickers.keys())
            self.logger.info(f"Analyzing {len(all_tickers)} tickers for signal generation")
            
            # Step 2.5: Fetch ALL ticker data ONCE (eliminates duplicate API calls)
            self.logger.info("Step 2.5: Fetching comprehensive ticker data (SINGLE PASS - eliminates duplicates)...")
            ticker_data_cache = await self._fetch_all_ticker_data_once(all_tickers)
            self.logger.info(f"✅ Cached comprehensive data for {len(ticker_data_cache)} tickers")
            
            # Step 3: Generate Individual Signals from Each Data Source
            self.logger.info("Step 3: Generating individual signals...")
            
            # Generate Reddit signals
            self.logger.info("Generating Reddit signals...")
            reddit_signals = self.generate_reddit_signals(filtered_tickers)
            self.logger.info(f"Generated {len(reddit_signals)} Reddit signals")
            
            # Generate Financial signals using cached data (NO API CALLS)
            self.logger.info("Generating Financial signals (using cached data)...")
            financial_signals = self.generate_financial_signals_cached(all_tickers, ticker_data_cache)
            self.logger.info(f"Generated {len(financial_signals)} Financial signals")
            
            # Generate News signals (if enabled)
            self.logger.info("Generating News signals...")
            news_signals = await self.generate_news_signals(all_tickers)
            self.logger.info(f"Generated {len(news_signals)} News signals")
            
            # Step 4: Combine All Signals into Scored Signals
            self.logger.info("Step 4: Combining signals into final scores...")
            signals = self.combine_signals_to_scored_signals(reddit_signals, financial_signals, news_signals)
            
            # Step 4.5: Comprehensive Signal Enhancement (using cached data - NO MORE DUPLICATE API CALLS!)
            self.logger.info("Step 4.5: Applying comprehensive signal enhancement (using cached data)...")
            signals = await self._comprehensive_signal_enhancement(signals, ticker_data_cache)
        
            self.logger.info("[SUCCESS] Comprehensive enhancement complete (technical + performance + AI prep)")            # Step 4.6: Comprehensive AI Commentary Generation (TOP 10 ONLY) + Unified Commentary
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
                
                self.logger.info(f"[SUCCESS] AI commentary generated (10 full, {len(other_signals)} basic)")
                self.logger.info(f"[SUCCESS] Unified commentary generated for all {len(signals)} signals")
                
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
                    
                    self.logger.info("[SUCCESS] AI commentary generated (fallback, top 10 only)")
                    self.logger.info(f"[SUCCESS] Unified commentary generated for all {len(signals)} signals (fallback)")
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
                    self.logger.info(f"[SUCCESS] Smart backtest complete: {eligible_count} signals updated")
                    self.logger.info(f"   Processed intervals: {list(backtest_results.get('interval_results', {}).keys())}")
                else:
                    self.logger.info("[SUCCESS] Smart backtest: No eligible signals (intervals not elapsed)")
                
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
            
            self.logger.info("[SUCCESS] All signal enhancements complete (comprehensive + AI + backtest scheduling)")            # Limit to max_signals
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
                        self.logger.info(f"[SUCCESS] Backtested {backtested}/{total} previous signals")
                    else:
                        self.logger.info("[INFO] No eligible signals to backtest yet (need 1+ day elapsed)")
                else:
                    self.logger.warning(f"[WARNING] Backtest had issues: {backtest_results.get('error')}")
            except Exception as e:
                self.logger.warning(f"[WARNING] Backtest failed: {e}")
            
            # Step 5: Save to Database
            self.logger.info("Step 5: Saving signals to database...")
            
            if signals:
                # Sort signals by weighted score
                signals.sort(key=lambda x: x['weighted_score'], reverse=True)
                
                # Save to database
                save_result = await self.save_signals_to_database(signals)
                save_success = save_result.get('success', False) if isinstance(save_result, dict) else save_result
                run_id = save_result.get('run_id') if isinstance(save_result, dict) else None
                
                # Step 6: Generate AI Strategies (NEW)
                ai_strategies_success = False
                ai_strategies_count = 0
                
                if save_success:
                    self.logger.info("Step 6: Generating AI strategies for top signals...")
                    try:
                        ai_strategies_result = await self._run_ai_strategy_generation(run_id=run_id)
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
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)


async def cleanup_async_resources():
    """Properly cleanup async resources to prevent event loop errors."""
    try:
        # Give pending tasks a moment to complete naturally
        await asyncio.sleep(0.1)
        
        # Get current event loop
        loop = asyncio.get_running_loop()
        
        # Get all pending tasks except the current one
        current_task = asyncio.current_task()
        pending_tasks = [
            task for task in asyncio.all_tasks(loop) 
            if not task.done() and task is not current_task
        ]
        
        if pending_tasks:
            logger.info(f"[CLEANUP] Cancelling {len(pending_tasks)} pending tasks...")
            
            # Cancel all pending tasks
            for task in pending_tasks:
                task.cancel()
            
            # Wait for all tasks to finish cancellation with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning("[CLEANUP] Some tasks did not cancel within timeout")
            except Exception as e:
                logger.warning(f"[CLEANUP] Error during task cancellation: {e}")
        
        # Give the event loop time to clean up
        await asyncio.sleep(0.05)
        
        logger.info("[CLEANUP] Async resources cleaned up successfully")
        
    except Exception as cleanup_error:
        # Don't raise errors during cleanup - just log them
        logger.warning(f"[CLEANUP] Non-critical cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    try:
        # Run the main pipeline
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)
    finally:
        # Run cleanup in a separate event loop to avoid conflicts
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cleanup_async_resources())
            loop.close()
        except Exception:
            # Suppress any cleanup errors on Windows
            pass