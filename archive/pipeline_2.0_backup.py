"""
VP Investments Unified Pipeline
===============================

Refactored orchestration-only pipeline that delegates to phase modules.
This pipeline is the central coordinator for the 6-phase signal generation system.

Architecture:
- Phase 1: Data Fetching (Reddit, Yahoo Finance, News) - Phase1Fetcher
- Phase 2: Data Normalization (Standardize signals) - Phase2Normalizer  
- Phase 3: Signal Scoring (6 group scores via SignalScorer) - Built-in
- Phase 4: Score Assembly (Combine group scores) - Phase4Assembler
- Phase 5: Database Persistence (Save signals) - Phase5Persister
- Phase 6: Post-Operations (AI strategies, cleanup) - Built-in

3.0 Signal Groups (6 groups):
- technical (20%)
- fundamental (25%)
- news_macro (15%)
- social_alternative (10%)
- risk_stability (15%)
- institutional_smart_money (15%)

Features:
- Pure orchestration (delegates to phase modules)
- No duplicate API calls (single-pass data fetching)
- 3.0 compliant scoring system
- Comprehensive logging
"""

import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup VP Investments logging
from backend.utils.logger import setup_logging, get_logger

# Configure logging
setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    structured_logging=False
)
logger = get_logger(__name__)

# Import phase modules
from backend.phases.phase1_fetch import Phase1Fetcher
from archive.phase2_normalize import Phase2Normalizer
from archive.phase4_assemble import Phase4Assembler
from backend.phases.phase5_persist import Phase5Persister

# Import SignalScorer for Phase 3 scoring
from backend.core.signals import SignalScorer

# Optional integrations with graceful fallback
try:
    from backend.integrations.news import NewsIntegrator
except ImportError:
    NewsIntegrator = None
    logger.warning("NewsIntegrator not available")

try:
    from backend.integrations.ai import AIIntegrator, AIStrategyGenerator, ComprehensiveCommentaryGenerator
except ImportError:
    AIIntegrator = None
    AIStrategyGenerator = None
    ComprehensiveCommentaryGenerator = None
    logger.warning("AI integrations not available")# Simple configuration class
class Config:
    """Configuration class for pipeline settings with 3.0 signal group weights."""
    
    def __init__(self):
        self.reddit_post_limit = 100
        self.min_mentions = 1
        self.max_signals = 50
    
    def get(self, key, default=None):
        """Get configuration value with dot notation support"""
        # Support environment variables for 3.0 signal group weights
        if key == 'scoring.weights':
            from dotenv import load_dotenv
            load_dotenv()
            return {
                'technical': float(os.getenv('SCORE_WEIGHT_TECHNICAL', '0.20')),
                'fundamental': float(os.getenv('SCORE_WEIGHT_FUNDAMENTAL', '0.25')),
                'news_macro': float(os.getenv('SCORE_WEIGHT_NEWS_MACRO', '0.15')),
                'social_alternative': float(os.getenv('SCORE_WEIGHT_SOCIAL_ALTERNATIVE', '0.10')),
                'risk_stability': float(os.getenv('SCORE_WEIGHT_RISK_STABILITY', '0.15')),
                'institutional_smart_money': float(os.getenv('SCORE_WEIGHT_INSTITUTIONAL_SMART_MONEY', '0.15'))
            }
        return default


class UnifiedPipeline:
    """
    Refactored unified pipeline that orchestrates the 6-phase signal generation system.
    
    This pipeline delegates all work to phase modules and only handles orchestration.
    Phase modules are responsible for their own data processing and scoring logic.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the unified pipeline with phase module instances."""
        self.config = config or Config()
        self.logger = logger
        
        # Initialize phase modules
        self.phase1 = Phase1Fetcher()
        self.phase2 = Phase2Normalizer()
        self.phase4 = Phase4Assembler()
        self.phase5 = Phase5Persister()
        
        # Initialize SignalScorer for Phase 3 scoring
        self.signal_scorer = SignalScorer()
        
        self.logger.info("Pipeline initialized with phase modules (3.0 architecture)")
        self.logger.info(f"  Phase 1: Data Fetching ({self.phase1.__class__.__name__})")
        self.logger.info(f"  Phase 2: Normalization ({self.phase2.__class__.__name__})")
        self.logger.info(f"  Phase 3: Scoring (SignalScorer)")
        self.logger.info(f"  Phase 4: Assembly ({self.phase4.__class__.__name__})")
        self.logger.info(f"  Phase 5: Persistence ({self.phase5.__class__.__name__})")
        
        # Optional integrations (news, AI) - use gracefully imported globals
        self.news_integrator = NewsIntegrator() if NewsIntegrator else None
        self.ai_integrator = AIIntegrator() if AIIntegrator else None
        self.enhanced_available = bool(self.news_integrator or self.ai_integrator)
        
        if self.enhanced_available:
            self.logger.info("  Optional integrations: News and/or AI available")
        else:
            self.logger.info("  Optional integrations: Running in basic mode")
    
    def extract_tickers(self, text: str) -> List[str]:
        """Delegate to RedditDataIntegrator for ticker extraction"""
        return self.reddit.extract_tickers_pipeline(text)
    
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
        score = signal.get('signal_score', 0)  # Phase 7
        
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
    
    def _generate_risk_description(self, signal: Dict[str, Any], risk_category: str, risk_score: float) -> str:
        """
        Generate comprehensive risk assessment description using enhanced risk metrics.
        
        Args:
            signal: Enhanced signal with all metrics
            risk_category: Risk category from enhanced signal (Low/Moderate/High/Very High)
            risk_score: Numerical risk score 0-1 from enhanced signal
            
        Returns:
            Detailed risk description string
        """
        risk_factors = []
        
        # Start with category and score
        risk_factors.append(f"{risk_category} risk (score: {risk_score:.2f})")
        
        # Volatility assessment
        volatility = signal.get('volatility') or signal.get('historical_volatility', 0)
        if volatility:
            if volatility > 80:
                risk_factors.append("Very high volatility")
            elif volatility > 50:
                risk_factors.append("High volatility")
            elif volatility < 20:
                risk_factors.append("Low volatility")
        
        # Liquidity assessment
        liquidity_score = signal.get('liquidity_score', 0.5)
        if liquidity_score < 0.3:
            risk_factors.append("Low liquidity")
        elif liquidity_score > 0.8:
            risk_factors.append("High liquidity")
        
        # Beta assessment
        beta = signal.get('beta')
        if beta:
            if beta > 1.5:
                risk_factors.append(f"High beta ({beta:.2f})")
            elif beta < 0.5:
                risk_factors.append(f"Low beta ({beta:.2f})")
        
        # Market cap risk
        market_cap_category = signal.get('market_cap_category')
        if market_cap_category in ['Micro', 'Small']:
            risk_factors.append(f"{market_cap_category} cap stock")
        
        # Momentum risk
        momentum = signal.get('momentum_30d_pct', 0)
        if abs(momentum) > 50:
            risk_factors.append("Extreme momentum")
        elif abs(momentum) > 30:
            risk_factors.append("High momentum")
        
        # Technical risk
        rsi = signal.get('rsi')
        if rsi:
            if rsi > 80:
                risk_factors.append("Overbought (RSI)")
            elif rsi < 20:
                risk_factors.append("Oversold (RSI)")
        
        # Combine all factors
        if len(risk_factors) > 1:
            return " | ".join(risk_factors)
        else:
            return f"{risk_category} risk profile"
    
    def _determine_trade_type(self, signal: Dict) -> str:
        """Determine trade type based on signal characteristics."""
        score = signal.get('signal_score', 0)  # Phase 7
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
            
            # Use gracefully imported AIStrategyGenerator
            if not AIStrategyGenerator:
                self.logger.warning("AI strategy generator not available")
                return {'success': False, 'strategies_count': 0, 'message': 'AI generator not available'}
            
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
    
    async def generate_news_signals(self, tickers: List[str]) -> List[Dict[str, Any]]:
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
                    'financial_data': financial_data,  # Fixed: Changed from 'metadata' to 'financial_data'
                    'metadata': financial_data  # Keep for backward compatibility
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
                try:
                    financial_data = self._convert_cache_to_financial_data(ticker_data)
                except Exception as conv_error:
                    self.logger.error(f"[CACHED] Exception converting cache to financial_data for {ticker}: {conv_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
                
                if not financial_data:
                    self.logger.warning(f"[CACHED] Failed to convert cache to financial_data for {ticker}")
                    continue
                
                # Calculate financial signal score (delegated to SignalScorer - Phase 6c)
                financial_score = self.signal_scorer._calculate_financial_score(financial_data)
                
                # Create financial signal
                financial_signal = {
                    'ticker': ticker,
                    'signal_type': 'financial',
                    'score': financial_score,
                    'confidence': 0.8,  # Financial data generally reliable
                    'financial_data': financial_data,  # Fixed: Changed from 'metadata' to 'financial_data'
                    'metadata': financial_data  # Keep for backward compatibility
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
            ticker = ticker_data.get('ticker')
            
            self.logger.info(f"[CONVERT] START: ticker={ticker}, has_history_1m={not history_1m.empty}, has_info={len(info)}, has_stock={ticker_data.get('stock') is not None}")
            
            if history_1m.empty or not ticker:
                self.logger.warning(f"[CONVERT] EARLY EXIT: ticker={ticker}, history_1m empty={history_1m.empty}")
                return None
            
            stock = ticker_data.get('stock')
            if not stock:
                return None
            
            # Get base financial data using FinancialMetricsCalculator with cached data (NO API CALL!)
            from backend.integrations.yfinance import FinancialMetricsCalculator
            
            metrics_calc = FinancialMetricsCalculator()
            
            # Build financial_data using the calculator methods with cached data
            financial_data = {}
            financial_data.update(metrics_calc._get_basic_info(ticker, info))
            financial_data.update(metrics_calc._get_price_metrics(history_1y, info))
            financial_data.update(metrics_calc._get_fundamental_ratios(info, stock, history_1y))
            financial_data.update(metrics_calc._get_earnings_metrics(stock, info, history_1y))
            
            market_cap = financial_data.get('market_cap', info.get('marketCap'))
            financial_data.update(metrics_calc._get_balance_sheet_metrics(stock, info, market_cap))
            financial_data.update(metrics_calc._get_ownership_metrics(info))
            financial_data.update(metrics_calc._get_liquidity_metrics(history_1y, info, stock))
            
            # Phase 3 data
            current_price = financial_data.get('current_price', info.get('previousClose', 0))
            financial_data.update(metrics_calc._get_analyst_data(stock, info, current_price))
            financial_data.update(metrics_calc._get_earnings_surprise_data(stock))
            financial_data.update(metrics_calc._get_institutional_ownership_data(stock, info))
            financial_data.update(metrics_calc._get_insider_trading_data(stock))
            
            # NOTE: ImprovedFinancialCalculator removed (file deleted)
            # All financial calculations now use FinancialMetricsCalculator above
            # Additional metrics can be added to calculator.py if needed
            
            self.logger.info(f"[PIPELINE] _convert_cache_to_financial_data({ticker}) - roic={financial_data.get('roic')}, roe={financial_data.get('roe')}, pe={financial_data.get('pe_ratio')}")
            
            # financial_data already has all fundamental fields with proper formatting from FinancialMetricsCalculator
            # DON'T overwrite them with raw info values!
            # Only add fields that FinancialMetricsCalculator doesn't provide
            
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
            
            # NOTE: Most fundamental metrics now come from FinancialMetricsCalculator
            # Only add fields that are not provided by get_comprehensive_financial_data()
            
            # Sector/relative strength (not provided by FinancialMetricsCalculator)
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
            self.logger.error(f"[CONVERT] Error converting cache to financial_data for {ticker_data.get('ticker')}: {e}")
            import traceback
            self.logger.error(f"[CONVERT] Traceback: {traceback.format_exc()}")
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
                
                # Calculate signal score using configurable weights (Phase 7)
                signal_score = (
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
                
                # Create combined signal (Phase 7)
                combined_signal = {
                    'ticker': ticker,
                    'signal_score': signal_score,  # Phase 7
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
        
        # Sort by signal score descending (Phase 7)
        combined_signals.sort(key=lambda x: x.get('signal_score', 0), reverse=True)
        
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
        """Apply signal enhancements including calculated fields AND Phase 2-8 enhancements."""
        try:
            # Try to import the consolidated enhancer
            try:
                from backend.integrations.signal_processing import enhance_signals_batch
                self.logger.info(f"Applying signal enhancements to {len(signals)} records...")
                enhanced = enhance_signals_batch(signals)
            except ImportError:
                self.logger.warning("Signal enhancer module not available, using basic enhancements...")
                enhanced = signals
            
            # Apply Phase 2-8 enhancements
            self.logger.info(f"Applying Phase 2-8 enhancements to {len(enhanced)} signals...")
            enhanced = self._apply_phase2_8_enhancements(enhanced)
            
            self.logger.info("Signal enhancement complete (including Phase 2-8)")
            return enhanced
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
            
            # Risk-adjusted score (Phase 7)
            signal_score = signal.get('signal_score', 0)
            enhanced['risk_adjusted_score'] = self._safe_round(
                signal_score * (100 - risk_score) / 100, 4
            )
            
            enhanced_signals.append(enhanced)
        
        self.logger.info(f"Applied basic enhancements to {len(enhanced_signals)} signals")
        return enhanced_signals
    
    def _apply_phase2_8_enhancements(self, signals: list) -> list:
        """
        Apply Phase 2-8 enhancements to signals.
        
        Phases:
        - Phase 2: Z-Score normalization (4 columns)
        - Phase 3: Trade type confidence (1 column)
        - Phase 4: Detailed risk scoring (7 individual risk factors)
        - Phase 5: Enhanced data collection (11 columns - ATR, options, institutional)
        - Phase 6: Score adjustments (3 columns)
        - Phase 7: AI risk narratives (1 column)
        - Phase 8: Backtest parameters (6 columns)
        
        Total: 33 new columns populated
        """
        try:
            from backend.core.signals import ZScoreCalculator, TradeTypeClassifier, RiskScoreCalculator
            from backend.core.signals import TrendStrengthCalculator, ValuationCalculator
            import backend.integrations.yfinance as yf
            import numpy as np
            
            # Initialize calculators (if not already initialized)
            if not hasattr(self, 'z_calc'):
                self.z_calc = ZScoreCalculator(lookback_days=60, min_samples=30)
            if not hasattr(self, 'trend_calc'):
                self.trend_calc = TrendStrengthCalculator(self.z_calc)  # Requires z_calc
            if not hasattr(self, 'val_calc'):
                self.val_calc = ValuationCalculator(self.z_calc)  # Requires z_calc
            if not hasattr(self, 'trade_classifier'):
                self.trade_classifier = TradeTypeClassifier(self.z_calc, self.trend_calc, self.val_calc)
            if not hasattr(self, 'risk_calculator'):
                self.risk_calculator = RiskScoreCalculator()
            
            enhanced_signals = []
            phase_stats = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
            
            for signal in signals:
                enhanced = signal.copy()
                ticker = signal.get('ticker', '')
                
                try:
                    # PHASE 2: Z-Score Normalization
                    # Calculate z-scores for momentum, volume, volatility, valuation
                    momentum_30d = signal.get('momentum_30d_pct', 0.0)
                    volume = signal.get('volume', 0.0)
                    volatility = signal.get('volatility', 0.0)
                    pe_ratio = signal.get('pe_ratio', 0.0)
                    
                    enhanced['z_score_momentum'] = self._safe_round(
                        self.z_calc.calculate_z_score(momentum_30d, ticker, 'momentum_30d_pct'), 4
                    ) if momentum_30d else None
                    
                    enhanced['z_score_volume'] = self._safe_round(
                        self.z_calc.calculate_z_score(volume, ticker, 'volume'), 4
                    ) if volume else None
                    
                    enhanced['z_score_volatility'] = self._safe_round(
                        self.z_calc.calculate_z_score(volatility, ticker, 'volatility'), 4
                    ) if volatility else None
                    
                    enhanced['z_score_valuation'] = self._safe_round(
                        self.z_calc.calculate_z_score(pe_ratio, ticker, 'pe_ratio'), 4
                    ) if pe_ratio else None
                    
                    if any([enhanced.get('z_score_momentum'), enhanced.get('z_score_volume'), 
                            enhanced.get('z_score_volatility'), enhanced.get('z_score_valuation')]):
                        phase_stats[2] += 1
                    
                    # PHASE 3: Trade Type Confidence
                    # Extract confidence from trade type classification
                    component_scores = {
                        'technical_score': signal.get('technical_score', 0.0),
                        'fundamental_score': signal.get('fundamental_score', 0.0),
                        'news_score': signal.get('news_score', 0.0),
                        'social_score': signal.get('social_score', 0.0)
                    }
                    
                    # Calculate trade type confidence based on component score strengths
                    scores = [v for v in component_scores.values() if v > 0]
                    if scores:
                        # Confidence is based on how many strong signals we have
                        strong_signals = sum(1 for s in scores if s > 0.6)
                        confidence = min(1.0, (len(scores) / 4.0) * 0.5 + (strong_signals / 4.0) * 0.5)
                        enhanced['trade_type_confidence'] = self._safe_round(confidence, 4)
                        phase_stats[3] += 1
                    else:
                        enhanced['trade_type_confidence'] = None
                    
                    # PHASE 4: Detailed Risk Scoring
                    # Calculate individual risk factors using RiskScoreCalculator
                    risk_data = {
                        'atr_pct': signal.get('atr_percent', signal.get('volatility')),
                        'beta': signal.get('beta', 1.0),
                        'avg_volume': signal.get('avg_volume_30d', signal.get('volume')),
                        'float_pct': signal.get('float_turnover_ratio', 50.0),
                        'debt_to_equity': signal.get('debt_equity', signal.get('debt_to_equity')),
                        'interest_coverage': signal.get('interest_coverage', 5.0),
                        'short_interest': signal.get('short_pct_float', signal.get('short_interest')),
                        'market_cap': signal.get('market_cap'),
                        'price': signal.get('current_price')
                    }
                    
                    try:
                        _, _, risk_factors = self.risk_calculator.calculate_risk_score(
                            ticker, risk_data, theme=signal.get('theme')
                        )
                        
                        # Extract individual risk factors
                        enhanced['volatility_risk'] = self._safe_round(
                            risk_factors.get('volatility', {}).get('score', 0.0), 2
                        )
                        enhanced['liquidity_risk'] = self._safe_round(
                            risk_factors.get('liquidity', {}).get('score', 0.0), 2
                        )
                        enhanced['leverage_risk'] = self._safe_round(
                            risk_factors.get('leverage', {}).get('score', 0.0), 2
                        )
                        enhanced['concentration_risk'] = self._safe_round(
                            risk_factors.get('concentration', {}).get('score', 0.0), 2
                        )
                        enhanced['technical_risk'] = self._safe_round(
                            enhanced.get('volatility_risk', 0.0), 2
                        )  # Technical risk same as volatility risk
                        enhanced['fundamental_risk'] = self._safe_round(
                            enhanced.get('leverage_risk', 0.0), 2
                        )  # Fundamental risk same as leverage risk
                        enhanced['sentiment_risk'] = self._safe_round(
                            risk_factors.get('short_interest', {}).get('score', 0.0), 2
                        )
                        
                        if any([enhanced.get('volatility_risk'), enhanced.get('liquidity_risk')]):
                            phase_stats[4] += 1
                            
                    except Exception as e:
                        self.logger.debug(f"Phase 4 risk factors failed for {ticker}: {e}")
                        # Set all to None if calculation fails
                        for risk_col in ['volatility_risk', 'liquidity_risk', 'leverage_risk', 
                                        'concentration_risk', 'technical_risk', 'fundamental_risk', 'sentiment_risk']:
                            enhanced[risk_col] = None
                    
                    # PHASE 5: Enhanced Data Collection (ATR, Options, Institutional)
                    # Try to fetch enhanced data from yfinance
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info if hasattr(stock, 'info') else {}
                        hist = stock.history(period='1mo') if hasattr(stock, 'history') else None
                        
                        # ATR calculations
                        if hist is not None and not hist.empty and len(hist) >= 14:
                            high = hist['High']
                            low = hist['Low']
                            close = hist['Close']
                            
                            # True Range
                            tr1 = high - low
                            tr2 = abs(high - close.shift())
                            tr3 = abs(low - close.shift())
                            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            atr = tr.rolling(window=14).mean().iloc[-1]
                            
                            enhanced['atr'] = self._safe_round(atr, 4)
                            enhanced['atr_percent'] = self._safe_round(
                                (atr / close.iloc[-1]) * 100, 4
                            ) if close.iloc[-1] > 0 else None
                            
                            # Historical Volatility (20-day)
                            returns = close.pct_change().dropna()
                            if len(returns) >= 20:
                                hist_vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
                                enhanced['historical_volatility'] = self._safe_round(hist_vol * 100, 4)
                        
                        # Options data
                        if info:
                            enhanced['put_call_ratio'] = self._safe_round(
                                info.get('putCallRatio'), 4
                            ) if info.get('putCallRatio') else None
                            
                            enhanced['open_interest'] = info.get('openInterest')
                            
                            # Fundamental metrics
                            enhanced['operating_margin'] = self._safe_round(
                                info.get('operatingMargins', 0) * 100, 4
                            ) if info.get('operatingMargins') else None
                            
                            enhanced['debt_to_equity'] = self._safe_round(
                                info.get('debtToEquity'), 4
                            ) if info.get('debtToEquity') else signal.get('debt_equity')
                            
                            enhanced['current_ratio'] = self._safe_round(
                                info.get('currentRatio'), 4
                            ) if info.get('currentRatio') else None
                            
                            # Ownership metrics (FIXED: correct field names)
                            enhanced['institutional_ownership'] = self._safe_round(
                                info.get('heldPercentInstitutions', 0) * 100, 4
                            ) if info.get('heldPercentInstitutions') else None
                            
                            enhanced['insider_ownership'] = self._safe_round(
                                info.get('heldPercentInsiders', 0) * 100, 4
                            ) if info.get('heldPercentInsiders') else None
                            
                            enhanced['short_interest'] = self._safe_round(
                                info.get('shortPercentOfFloat', 0) * 100, 4
                            ) if info.get('shortPercentOfFloat') else None
                            
                        if any([enhanced.get('atr'), enhanced.get('put_call_ratio'), 
                                enhanced.get('institutional_ownership')]):
                            phase_stats[5] += 1
                            
                    except Exception as e:
                        self.logger.debug(f"Phase 5 enhanced data failed for {ticker}: {e}")
                        # Set Phase 5 columns to None if fetch fails
                        for col in ['atr', 'atr_percent', 'historical_volatility', 'put_call_ratio', 
                                   'open_interest', 'operating_margin', 'debt_to_equity', 'current_ratio',
                                   'institutional_ownership', 'insider_ownership', 'short_interest']:
                            if col not in enhanced or enhanced.get(col) is None:
                                enhanced[col] = signal.get(col)  # Use existing value if available
                    
                    # PHASE 5.5: Column Consolidation (use existing data if Phase 5 failed)
                    # This improves Phase 5 population rate by leveraging existing signal data
                    if enhanced.get('debt_to_equity') is None:
                        enhanced['debt_to_equity'] = signal.get('debt_equity')
                    
                    if enhanced.get('short_interest') is None:
                        enhanced['short_interest'] = signal.get('short_pct_float')
                    
                    if enhanced.get('institutional_ownership') is None:
                        enhanced['institutional_ownership'] = signal.get('institutional_ownership_pct')
                    
                    # PHASE 6: Score Adjustments
                    # Calculate adjusted signal score based on risk and trade type
                    signal_score = signal.get('signal_score', 0.0)
                    risk_score = signal.get('risk_score', 50.0)
                    trade_type = signal.get('trade_type', 'Multi-Factor')
                    
                    # Trade type multipliers
                    trade_multipliers = {
                        'Momentum': 1.1,
                        'Value': 1.05,
                        'Event-Driven': 1.15,
                        'Contrarian': 1.0,
                        'Speculative Growth': 0.9,
                        'Multi-Factor': 1.0
                    }
                    multiplier = trade_multipliers.get(trade_type, 1.0)
                    
                    # Risk adjustment (lower risk = higher adjustment)
                    risk_adjustment = 1.0 - (risk_score / 200.0)  # Range: 0.5 to 1.0
                    
                    adjusted_score = signal_score * multiplier * risk_adjustment
                    enhanced['adjusted_signal_score'] = self._safe_round(
                        min(1.0, max(0.0, adjusted_score)), 4
                    )
                    
                    # Position size recommendation (inverse of risk, trade type adjusted)
                    base_position = 0.10  # 10% base
                    risk_factor = (100 - risk_score) / 100.0  # Higher for lower risk
                    position_size = base_position * risk_factor * multiplier
                    enhanced['position_size_recommendation'] = self._safe_round(
                        min(0.25, max(0.01, position_size)), 4
                    )
                    
                    # Entry threshold (higher for riskier signals)
                    base_threshold = 0.60
                    risk_premium = (risk_score / 100.0) * 0.20  # 0-20% premium
                    enhanced['entry_threshold'] = self._safe_round(
                        min(0.90, base_threshold + risk_premium), 4
                    )
                    
                    if enhanced.get('adjusted_signal_score'):
                        phase_stats[6] += 1
                    
                    # PHASE 7: AI Risk Narrative
                    # Map ai_commentary to risk_narrative (if available)
                    ai_commentary = signal.get('ai_commentary', '')
                    if ai_commentary and len(ai_commentary) > 50:
                        # Extract risk-related content from AI commentary
                        enhanced['risk_narrative'] = ai_commentary[:1000]  # Cap at 1000 chars
                        phase_stats[7] += 1
                    else:
                        # Generate basic risk narrative
                        risk_level = signal.get('risk_level', 'Moderate')
                        narrative_parts = [
                            f"{ticker} is classified as {risk_level} risk",
                            f"with a {trade_type} trade setup."
                        ]
                        
                        if risk_score > 65:
                            narrative_parts.append("High volatility and concentration risk suggest smaller position sizing.")
                        elif risk_score < 35:
                            narrative_parts.append("Low risk profile supports larger position allocation.")
                        
                        if enhanced.get('volatility_risk', 0) > 60:
                            narrative_parts.append("Elevated price volatility warrants wider stops.")
                        
                        enhanced['risk_narrative'] = " ".join(narrative_parts)
                        phase_stats[7] += 1
                    
                    # PHASE 8: Backtest Parameters
                    # Calculate dynamic backtest parameters based on ATR and risk
                    current_price = signal.get('current_price', 0.0)
                    atr_value = enhanced.get('atr', current_price * 0.02)  # Default 2% if no ATR
                    
                    if current_price > 0 and atr_value:
                        # Entry threshold (lower for lower risk)
                        enhanced['backtest_entry_threshold'] = self._safe_round(
                            enhanced.get('entry_threshold', 0.65), 4
                        )
                        
                        # Hold period (shorter for momentum, longer for value)
                        if 'Momentum' in trade_type:
                            hold_period = 3
                        elif 'Value' in trade_type:
                            hold_period = 14
                        elif 'Event-Driven' in trade_type:
                            hold_period = 5
                        else:
                            hold_period = 7
                        enhanced['backtest_hold_period_days'] = hold_period
                        
                        # Position size (same as Phase 6 recommendation)
                        enhanced['backtest_position_size_pct'] = enhanced.get('position_size_recommendation', 0.05)
                        
                        # Stop loss (2x ATR below entry)
                        stop_loss = current_price - (2.0 * atr_value)
                        enhanced['backtest_stop_loss_price'] = self._safe_round(
                            max(0.0, stop_loss), 4
                        )
                        
                        # Take profit (3x ATR above entry for 1.5:1 risk/reward)
                        take_profit = current_price + (3.0 * atr_value)
                        enhanced['backtest_take_profit_price'] = self._safe_round(take_profit, 4)
                        
                        # Risk/reward ratio
                        risk_amount = 2.0 * atr_value
                        reward_amount = 3.0 * atr_value
                        enhanced['backtest_risk_reward_ratio'] = self._safe_round(
                            reward_amount / risk_amount if risk_amount > 0 else 1.5, 2
                        )
                        
                        if enhanced.get('backtest_entry_threshold'):
                            phase_stats[8] += 1
                    else:
                        # Set all backtest params to None if we can't calculate
                        for col in ['backtest_entry_threshold', 'backtest_hold_period_days', 
                                   'backtest_position_size_pct', 'backtest_stop_loss_price',
                                   'backtest_take_profit_price', 'backtest_risk_reward_ratio']:
                            enhanced[col] = None
                    
                except Exception as e:
                    self.logger.warning(f"Phase 2-8 enhancement failed for {ticker}: {e}")
                    # Continue with next signal even if one fails
                
                enhanced_signals.append(enhanced)
            
            # Log phase statistics
            total_signals = len(signals)
            self.logger.info(f"[Phase 2-8] Applied enhancements:")
            for phase, count in sorted(phase_stats.items()):
                pct = (count / total_signals * 100) if total_signals > 0 else 0
                self.logger.info(f"  Phase {phase}: {count}/{total_signals} signals ({pct:.1f}%)")
            
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Phase 2-8 enhancement failed: {e}")
            self.logger.exception(e)
            return signals  # Return original signals if Phase 2-8 fails
    
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
        import backend.integrations.yfinance as yf
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
        import backend.integrations.yfinance as yf
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
        
        # ML Analytics - Phase 1 metrics (liquidity_score, risk_score, plus momentum_consistency_score, pattern_match_score, etc.)
        try:
            try:
                from backend.integrations.signal_processing import SignalMLAnalyzer, SignalEnhancer
                
                # FIX Round 3: Add liquidity_score, risk_score via _enhance_single_signal
                enhancer = SignalEnhancer()
                enhanced_signal = enhancer._enhance_single_signal(enhanced_signal)
                
                # Then add ML analytics
                analyzer = SignalMLAnalyzer()
                before_keys = set(enhanced_signal.keys())
                enhanced_signal = analyzer.enhance_signal_with_ml_analytics(enhanced_signal)
                after_keys = set(enhanced_signal.keys())
                new_keys = after_keys - before_keys
                if 'momentum_consistency_score' in new_keys:
                    self.logger.info(f"[ML] Added momentum_consistency_score={enhanced_signal.get('momentum_consistency_score')} to {enhanced_signal.get('ticker')}")
                else:
                    self.logger.warning(f"[ML] momentum_consistency_score NOT added to {enhanced_signal.get('ticker')}")
            except ImportError:
                self.logger.debug(f"Signal processing modules not available for {enhanced_signal.get('ticker')}")
        except Exception as e:
            import traceback
            self.logger.error(f"ML analytics enhancement failed for {enhanced_signal.get('ticker')}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Calculate Phase 7: 6-group component scores using actual methods
        try:
            # Calculate each component score using Phase 7 methods in SignalScorer
            component_scores = {
                'technical': self.signal_scorer._calculate_technical_score(enhanced_signal),
                'fundamental': self.signal_scorer._calculate_fundamental_score(enhanced_signal),
                'news_macro': self.signal_scorer._calculate_news_macro_score(enhanced_signal),
                'social_alternative': self.signal_scorer._calculate_social_alternative_score(enhanced_signal),
                'risk_stability': self.signal_scorer._calculate_risk_stability_score(enhanced_signal),
                'institutional_smart_money': self.signal_scorer._calculate_institutional_smart_money_score(enhanced_signal)
            }
            
            # Save component scores to signal
            enhanced_signal['technical_score'] = component_scores['technical']
            enhanced_signal['fundamental_score'] = component_scores['fundamental']
            enhanced_signal['news_macro_score'] = component_scores['news_macro']
            enhanced_signal['social_alternative_score'] = component_scores['social_alternative']
            enhanced_signal['risk_stability_score'] = component_scores['risk_stability']
            enhanced_signal['institutional_smart_money_score'] = component_scores['institutional_smart_money']
            
            # Calculate final signal_score as weighted combination (Phase 7)
            enhanced_signal['signal_score'] = self.signal_scorer._calculate_signal_score_v2(
                enhanced_signal, component_scores
            )
            
            # Calculate Phase 7 confidence
            phase7_confidence = self.signal_scorer._calculate_confidence_v2(enhanced_signal, component_scores)
            enhanced_signal['phase7_confidence'] = phase7_confidence
            
            self.logger.info(
                f"[Phase 7] {enhanced_signal.get('ticker')}: "
                f"signal_score={enhanced_signal['signal_score']:.3f}, "
                f"confidence={phase7_confidence:.3f}, "
                f"components=(tech={component_scores['technical']:.2f}, "
                f"fund={component_scores['fundamental']:.2f}, "
                f"news={component_scores['news_macro']:.2f}, "
                f"social={component_scores['social_alternative']:.2f}, "
                f"risk={component_scores['risk_stability']:.2f}, "
                f"inst={component_scores['institutional_smart_money']:.2f})"
            )
        except Exception as e:
            import traceback
            self.logger.error(f"Phase 7 scoring failed for {enhanced_signal.get('ticker')}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Set default values to prevent pipeline failure
            enhanced_signal['signal_score'] = 0.5
            enhanced_signal['technical_score'] = 0.5
            enhanced_signal['fundamental_score'] = 0.5
            enhanced_signal['news_macro_score'] = 0.5
            enhanced_signal['social_alternative_score'] = 0.5
            enhanced_signal['risk_stability_score'] = 0.5
            enhanced_signal['institutional_smart_money_score'] = 0.5
            enhanced_signal['phase7_confidence'] = 0.5
        
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
            import backend.integrations.yfinance as yf
            
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
            
            # FIX: Calculate proper MACD indicator for scoring (macd_line is used, but set macd for consistency)
            # The scoring function looks for 'macd' which should be the MACD line value
            signal['macd'] = signal['macd_line']
            
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
            
            # FIX Round 2: Calculate 50-day MA percentage (requires 3-month history)
            if not df.empty and len(df) >= 50:
                ma_50 = df['Close'].rolling(50).mean().iloc[-1]
                current_price = df['Close'].iloc[-1]
                signal['above_50d_ma_pct'] = float((current_price / ma_50 - 1) * 100) if not pd.isna(ma_50) else None
            
            # FIX: Calculate 200-day MA percentage (requires 1-year history)
            history_1y = ticker_data.get('history_1y', pd.DataFrame())
            if not history_1y.empty and len(history_1y) >= 200:
                ma_200 = history_1y['Close'].rolling(200).mean().iloc[-1]
                current_price = history_1y['Close'].iloc[-1]
                signal['above_200d_ma_pct'] = float((current_price / ma_200 - 1) * 100) if not pd.isna(ma_200) else None
            
            # Beta calculation - delegate to YahooFinanceIntegrator
            from backend.integrations.yfinance import YahooFinanceIntegrator
            yf_integrator = YahooFinanceIntegrator()
            signal['beta'] = yf_integrator.calculate_beta(ticker_data)
            
            # FIX Round 2: Add momentum_30d_pct from history
            if not df.empty and len(df) >= 30:
                signal['momentum_30d_pct'] = float((df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1) * 100)
            
            # FIX Round 2: Add volume_price_correlation
            if not df.empty and len(df) >= 20:
                signal['volume_price_correlation'] = float(df['Close'].corr(df['Volume'])) if 'Volume' in df.columns else None
            
            # FIX Round 2: Add historical_volatility (annualized)
            if not df.empty and len(df) >= 20:
                returns = df['Close'].pct_change().dropna()
                signal['historical_volatility'] = float(returns.std() * (252 ** 0.5) * 100) if len(returns) > 0 else None
            
            # FIX Round 2: Add bollinger band fields (using ta library results)
            if signal.get('bb_upper') and signal.get('bb_lower'):
                current_price = float(df['Close'].iloc[-1])
                signal['bollinger_upper'] = signal['bb_upper']
                signal['bollinger_lower'] = signal['bb_lower']
                signal['bollinger_width'] = float((signal['bb_upper'] - signal['bb_lower']) / signal['bb_upper'] * 100)
                # Calculate position: 0 = at lower band, 0.5 = at middle, 1 = at upper band
                signal['bollinger_position'] = float((current_price - signal['bb_lower']) / (signal['bb_upper'] - signal['bb_lower'])) if (signal['bb_upper'] - signal['bb_lower']) > 0 else 0.5
            
            # FIX Round 2: Add ATR (Average True Range) 
            if not df.empty and len(df) >= 14 and 'High' in df.columns and 'Low' in df.columns:
                atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
                atr_value = atr_indicator.average_true_range()
                if not atr_value.empty and not pd.isna(atr_value.iloc[-1]):
                    signal['atr'] = float(atr_value.iloc[-1])
                    current_price = float(df['Close'].iloc[-1])
                    signal['atr_percent'] = float((signal['atr'] / current_price) * 100) if current_price > 0 else None
            
            # FIX Round 3: Add avg_daily_value_traded for liquidity_score calculation
            financial_data = ticker_data.get('financial_data', {})
            if financial_data.get('avg_daily_value_traded'):
                signal['avg_daily_value_traded'] = financial_data.get('avg_daily_value_traded')
            
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
        score = signal.get('signal_score', 0)  # Phase 7
        reddit_score = signal.get('reddit_score', 0)
        financial_score = signal.get('financial_score', 0)
        
        # Primary components
        explanation_parts = [
            f"{ticker} signal score of {score:.3f} combines:",  # Phase 7
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
        score = signal.get('signal_score', 0)  # Phase 7
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
            
            financial_signal = financial_signals[0]
            
            # Transform to combined signal format with proper score keys (Phase 7)
            signal = {
                'ticker': ticker,
                'financial_score': financial_signal['score'],
                'signal_score': financial_signal['score'],  # Phase 7
                'reddit_score': 0,
                'news_score': 0,
                'confidence': financial_signal['confidence'],
                'financial_data': financial_signal['metadata'],
                'reddit_data': {},
                'news_data': {}
            }
            
            # Step 2: Add Reddit data if requested
            # Note: For now, Reddit data requires full pipeline run with scraping
            # Individual ticker Reddit lookup can be added in future enhancement
            self.logger.info(f"Step 2/4: Setting default Reddit values (full scraping not in single signal mode)")
            signal['upvotes'] = 0
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
            save_success = await self.save_signals_to_database([enhanced_signal])
            
            if save_success:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"✅ SUCCESS: Signal for {ticker} generated and saved in {elapsed:.2f}s")
                self.logger.info(f"   Signal Score: {enhanced_signal.get('signal_score', 'N/A')}")  # Phase 7
                self.logger.info(f"   Financial Score: {enhanced_signal.get('financial_score', 'N/A')}")
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
                    max_signals: int = 50,
                    test_mode: bool = False) -> Dict[str, Any]:
        """
        Run the complete unified pipeline using phase module orchestration.
        
        This method orchestrates the 6-phase signal generation system:
        - Phase 1: Data Fetching (Reddit, Financial, News)
        - Phase 2: Signal Normalization (Reddit, Financial, News signals)
        - Phase 3: Signal Scoring (6 group scores: technical, fundamental, news_macro, 
                   social_alternative, risk_stability, institutional_smart_money)
        - Phase 4: Score Assembly (combine 6 group scores → final score)
        - Phase 5: Database Persistence (save signals with all metrics)
        - Phase 6: Post-Operations (AI strategies, backtests, cleanup)
        
        Args:
            subreddits (List[str]): Subreddits to scrape
            post_limit (int): Posts per subreddit
            min_mentions (int): Minimum mentions required
            max_signals (int): Maximum signals to process
            test_mode (bool): If True, use minimal settings for quick testing
            
        Returns:
            Dict[str, Any]: Pipeline execution results
            
        Test Mode Settings:
            - subreddits: ['wallstreetbets']
            - post_limit: 10
            - min_mentions: 1
            - max_signals: 5
        """
        # Override with test settings if test_mode enabled
        if test_mode:
            subreddits = ['wallstreetbets']
            post_limit = 10
            min_mentions = 1
            max_signals = 5
            self.logger.info("🧪 TEST MODE ENABLED - Using minimal settings")
            self.logger.info(f"   Subreddits: {subreddits}")
            self.logger.info(f"   Post limit: {post_limit}")
            self.logger.info(f"   Max signals: {max_signals}")
        
        pipeline_start = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("STARTING VP INVESTMENTS UNIFIED PIPELINE (6-Phase Architecture)")
        self.logger.info("=" * 60)
        
        try:
            # ===== PHASE 1: DATA FETCHING =====
            self.logger.info("📥 PHASE 1: Fetching all raw data...")
            phase1_data = await self.phase1.fetch_all_data(
                subreddits=subreddits or ['wallstreetbets', 'stocks', 'investing'],
                post_limit=post_limit,
                min_mentions=min_mentions
            )
            
            if not phase1_data or not phase1_data.get('ticker_mentions'):
                raise ValueError("Phase 1 failed - no ticker mentions found")
            
            ticker_mentions = phase1_data['ticker_mentions']
            ticker_data_cache = phase1_data.get('ticker_data', {})
            all_tickers = list(ticker_mentions.keys())
            
            self.logger.info(f"✅ Phase 1 complete: {len(all_tickers)} tickers with data")
            
            # ===== PHASE 2: SIGNAL NORMALIZATION =====
            self.logger.info("🔄 PHASE 2: Normalizing all signals...")
            phase2_signals = await self.phase2.normalize_all_signals(
                ticker_mentions=ticker_mentions,
                ticker_data_cache=ticker_data_cache,
                all_tickers=all_tickers
            )
            
            self.logger.info(f"✅ Phase 2 complete: {len(phase2_signals)} normalized signals")
            
            # ===== PHASE 3: SIGNAL SCORING (6 GROUP SCORES) =====
            self.logger.info("📊 PHASE 3: Scoring signals (6 3.0 groups)...")
            phase3_scored = []
            
            for signal in phase2_signals:
                ticker = signal['ticker']
                ticker_data = ticker_data_cache.get(ticker, {})
                
                # Calculate 6 group scores using SignalScorer
                group_scores = self.signal_scorer.score_ticker(ticker, ticker_data)
                
                # Add group scores to signal
                signal.update(group_scores)
                phase3_scored.append(signal)
            
            self.logger.info(f"✅ Phase 3 complete: {len(phase3_scored)} signals with 6 group scores")
            
            # ===== PHASE 4: SCORE ASSEMBLY =====
            self.logger.info("🎯 PHASE 4: Assembling final scores...")
            phase4_final = await self.phase4.assemble_final_scores(
                scored_signals=phase3_scored,
                ticker_data_cache=ticker_data_cache
            )
            
            # Limit to max_signals
            phase4_final = phase4_final[:max_signals]
            
            self.logger.info(f"✅ Phase 4 complete: {len(phase4_final)} signals with final scores")
            
            # ===== PHASE 5: DATABASE PERSISTENCE =====
            self.logger.info("💾 PHASE 5: Persisting signals to database...")
            phase5_result = await self.phase5.save_signals(
                signals=phase4_final,
                run_metadata={
                    'subreddits': subreddits,
                    'post_limit': post_limit,
                    'test_mode': test_mode
                }
            )
            
            if not phase5_result.get('success'):
                raise ValueError("Phase 5 failed - database save unsuccessful")
            
            run_id = phase5_result.get('run_id')
            self.logger.info(f"✅ Phase 5 complete: {len(phase4_final)} signals saved (run_id: {run_id})")
            
            # ===== PHASE 6: POST-OPERATIONS =====
            self.logger.info("🔧 PHASE 6: Running post-operations...")
            phase6_result = await self._run_post_operations(phase5_result)
            
            self.logger.info(f"✅ Phase 6 complete: {phase6_result.get('operations_completed', 0)} operations")
            
            # ===== PIPELINE COMPLETE =====
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()
            
            results = {
                'success': True,
                'execution_time_seconds': execution_time,
                'signals_generated': len(phase4_final),
                'run_id': run_id,
                'phase_results': {
                    'phase1': {'tickers': len(all_tickers)},
                    'phase2': {'signals': len(phase2_signals)},
                    'phase3': {'scored': len(phase3_scored)},
                    'phase4': {'final': len(phase4_final)},
                    'phase5': phase5_result,
                    'phase6': phase6_result
                },
                'top_signals': phase4_final[:10],  # Top 10 for summary
                'pipeline_timestamp': pipeline_end.isoformat()
            }
            
            self.logger.info("=" * 60)
            self.logger.info("✅ PIPELINE EXECUTION COMPLETE")
            self.logger.info(f"   Signals generated: {len(phase4_final)}")
            self.logger.info(f"   Execution time: {execution_time:.2f}s")
            self.logger.info(f"   Run ID: {run_id}")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - pipeline_start).total_seconds(),
                'signals_generated': 0
            }
    
    async def _run_post_operations(self, persist_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Phase 6 post-operations (AI strategies, backtests, cleanup).
        
        Args:
            persist_result: Results from Phase 5 persistence
            
        Returns:
            Dict with post-operation results
        """
        results = {'operations_completed': 0}
        run_id = persist_result.get('run_id')
        
        # 6a: AI Strategy Generation (if enabled)
        if AIStrategyGenerator:
            try:
                self.logger.info("  6a: Generating AI strategies...")
                ai_result = await self._run_ai_strategy_generation(run_id)
                results['ai_strategies'] = ai_result
                if ai_result.get('success'):
                    results['operations_completed'] += 1
                    self.logger.info(f"     ✅ Generated {ai_result.get('strategies_count', 0)} strategies")
            except Exception as e:
                self.logger.warning(f"  6a: AI strategy generation failed: {e}")
                results['ai_strategies'] = {'success': False, 'error': str(e)}
        else:
            self.logger.info("  6a: AI strategies disabled (module not available)")
            results['ai_strategies'] = {'success': True, 'strategies_count': 0, 'message': 'Disabled'}
        
        # 6b: Backtest Scheduling (if enabled)
        try:
            self.logger.info("  6b: Scheduling backtests...")
            from backend.integrations.backtest import backtest_eligible_signals
            backtest_result = await backtest_eligible_signals(limit=100)
            results['backtests'] = backtest_result
            if backtest_result.get('success'):
                results['operations_completed'] += 1
                backtested = backtest_result.get('backtested_count', 0)
                self.logger.info(f"     ✅ Backtested {backtested} eligible signals")
        except Exception as e:
            self.logger.warning(f"  6b: Backtest scheduling failed: {e}")
            results['backtests'] = {'success': False, 'error': str(e)}
        
        # 6c: Cleanup operations (optional)
        try:
            self.logger.info("  6c: Running cleanup operations...")
            # Add any cleanup tasks here (cache clearing, temp file removal, etc.)
            results['cleanup'] = {'success': True}
            results['operations_completed'] += 1
        except Exception as e:
            self.logger.warning(f"  6c: Cleanup failed: {e}")
            results['cleanup'] = {'success': False, 'error': str(e)}
        
        return results

    # ===== OLD PIPELINE METHODS - DEPRECATED =====
    # The following methods are kept temporarily for backward compatibility
    # They will be removed once phase module integration is fully validated


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
                    score = signal['signal_score']  # Phase 7
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