"""
VP Investments Core - Constants and Recommendations System

Consolidated core functionality including:
- Application constants and configuration values  
- Custom exception classes
- Enums for signal types, data sources, market regimes
- Advanced Trading Recommendation Engine with AI analysis
- Default tickers and rate limits
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict
from enum import Enum
from dataclasses import dataclass

# OpenAI functionality removed - use backend/core/ai.py for AI recommendations

# Removed imports to avoid circular dependencies
# These can be imported within functions when needed

logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION CONSTANTS
# ============================================================================

APP_NAME = "VP Investments"
APP_VERSION = "3.0.0"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Database Constants
DEFAULT_BATCH_SIZE = 1000
MAX_CONNECTION_POOL_SIZE = 10
CONNECTION_TIMEOUT = 30

# Analysis Constants
DEFAULT_ANALYSIS_WINDOW_DAYS = 30
MIN_DATA_POINTS = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Production Constants
PRODUCTION_TIMEOUT = 5
MAX_WORKERS = 16
MIN_WORKERS = 2
SCALING_THRESHOLD = 0.85

# Directory Constants
LOG_DIR = "logs"
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
CONFIG_DIR = "config"


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(Enum):
    """Signal type classifications"""
    REDDIT_SURGE = "reddit_surge"
    NEWS_MOMENTUM = "news_momentum" 
    EARNINGS_REACTION = "earnings_reaction"
    TECHNICAL_MOMENTUM = "technical_momentum"
    RETAIL_SPECULATIVE = "retail_speculative"
    MULTI_FACTOR = "multi_factor"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    VOLUME_BREAKOUT = "volume_breakout"
    PRICE_MOMENTUM = "price_momentum"


class TradeType(Enum):
    """Trade type classifications"""
    LONG = "long"
    SHORT = "short" 
    SWING = "swing"
    MOMENTUM = "momentum"
    GROWTH = "growth"
    VALUE = "value"
    SPECULATIVE = "speculative"
    BALANCED = "balanced"


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MODERATE = "moderate"  # Changed from MEDIUM to match signals.py usage
    HIGH = "high"
    EXTREME = "extreme"


class MarketCondition(Enum):
    """Market condition classifications"""
    BULL = "BULL"
    BEAR = "BEAR"
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    RECOVERY = "RECOVERY"


class DataSource(Enum):
    YAHOO_FINANCE = "yahoo_finance"
    NEWS_API = "news_api"
    REDDIT = "reddit"
    GOOGLE_TRENDS = "google_trends"


class FeatureType(Enum):
    """Types of features in analysis."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    VOLUME = "volume"
    PRICE = "price"
    NEWS = "news"
    SOCIAL = "social"


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

# Default Stock Universe
DEFAULT_TICKERS: List[str] = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
    
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'ICE',
    
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'MDT', 'BMY',
    
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
    
    # Industrial & Energy
    'GE', 'BA', 'CAT', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY'
]

# API Rate Limits
RATE_LIMITS = {
    "yahoo_finance": {"requests_per_second": 10, "requests_per_minute": 100},
    "news_api": {"requests_per_day": 1000},
    "reddit": {"requests_per_minute": 60}
}


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class VPInvestmentsError(Exception):
    """Base exception class for VP Investments platform."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(VPInvestmentsError):
    """Raised when there's a configuration-related error."""
    pass


class DataError(VPInvestmentsError):
    """Raised when there's a data-related error."""
    pass


class APIError(VPInvestmentsError):
    """Raised when there's an API-related error."""
    pass


# ============================================================================
# RECOMMENDATIONS DATA MODELS
# ============================================================================

class RecommendationType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY" 
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class DataSourceSignal:
    """Signal from a specific data source"""
    source: str
    signal_strength: float  # -1 to 1 (negative = bearish, positive = bullish)
    confidence: float  # 0 to 1
    key_points: List[str]
    raw_data: Dict[str, Any]


@dataclass
class TradingRecommendation:
    """Complete trading recommendation with AI analysis"""
    ticker: str
    company_name: str
    recommendation: RecommendationType
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str
    reasoning: str
    risk_factors: List[str]
    data_sources: List[DataSourceSignal]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'recommendation': self.recommendation.value,
            'confidence': self.confidence,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'time_horizon': self.time_horizon,
            'reasoning': self.reasoning,
            'risk_factors': self.risk_factors,
            'data_sources': [
                {
                    'source': ds.source,
                    'signal_strength': ds.signal_strength,
                    'confidence': ds.confidence,
                    'key_points': ds.key_points
                }
                for ds in self.data_sources
            ],
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# ADVANCED TRADING RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """
    Advanced Trading Recommendation Engine
    
    Integrates multiple data sources and uses AI to generate comprehensive
    trading recommendations with detailed explanations and risk analysis.
    """
    
    def __init__(self):
        self.db = None
        self.config = None
        self.logger = logging.getLogger('recommendation_engine')
    
    async def initialize(self):
        """Initialize recommendation engine"""
        try:
            # Import here to avoid circular imports
            from ..storage.database import get_database
            from ..core.config import get_config
            
            self.db = get_database()
            self.config = get_config()
            
            self.logger.info("Recommendation engine initialized")
            
        except Exception as e:
            self.logger.error(f"Recommendation engine initialization failed: {e}")
            raise
    
    async def generate_recommendation(self, ticker: str) -> Optional[TradingRecommendation]:
        """Generate comprehensive trading recommendation for a ticker"""
        try:
            # Gather data from all sources
            data_sources = await self._gather_data_sources(ticker)
            
            if not data_sources:
                self.logger.warning(f"No data sources available for {ticker}")
                return None
            
            # Generate rule-based recommendation (AI functionality moved to backend/core/ai.py)
            recommendation = await self._generate_rule_based_recommendation(ticker, data_sources)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed for {ticker}: {e}")
            return None
    
    async def _gather_data_sources(self, ticker: str) -> List[DataSourceSignal]:
        """Gather signals from all available data sources"""
        data_sources = []
        
        try:
            # Reddit sentiment data
            reddit_signal = await self._get_reddit_signal(ticker)
            if reddit_signal:
                data_sources.append(reddit_signal)
            
            # News sentiment data  
            news_signal = await self._get_news_signal(ticker)
            if news_signal:
                data_sources.append(news_signal)
            
            # Technical analysis data
            technical_signal = await self._get_technical_signal(ticker)
            if technical_signal:
                data_sources.append(technical_signal)
            
            # Financial data
            financial_signal = await self._get_financial_signal(ticker)
            if financial_signal:
                data_sources.append(financial_signal)
            
        except Exception as e:
            self.logger.error(f"Data gathering failed for {ticker}: {e}")
        
        return data_sources
    
    async def _get_reddit_signal(self, ticker: str) -> Optional[DataSourceSignal]:
        """Get Reddit sentiment signal"""
        try:
            query = """
            SELECT sentiment_score, confidence, post_count
            FROM reddit_sentiment 
            WHERE ticker = %s 
            AND created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC 
            LIMIT 1
            """
            
            results = await self.db.execute_query(query, {'ticker': ticker})
            result = results[0] if results else None
            
            if result:
                sentiment = result['sentiment_score']
                confidence = result['confidence']
                post_count = result['post_count']
                
                return DataSourceSignal(
                    source="Reddit",
                    signal_strength=sentiment,
                    confidence=confidence,
                    key_points=[
                        f"Reddit sentiment: {sentiment:.2f}",
                        f"Based on {post_count} posts in last 24h",
                        f"Confidence: {confidence:.2f}"
                    ],
                    raw_data=dict(result)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Reddit signal fetch failed for {ticker}: {e}")
            return None
    
    async def _get_news_signal(self, ticker: str) -> Optional[DataSourceSignal]:
        """Get news sentiment signal"""
        try:
            query = """
            SELECT sentiment_score, confidence, article_count
            FROM news_sentiment 
            WHERE ticker = %s 
            AND created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC 
            LIMIT 1
            """
            
            results = await self.db.execute_query(query, {'ticker': ticker})
            result = results[0] if results else None
            
            if result:
                sentiment = result['sentiment_score']
                confidence = result['confidence']
                article_count = result['article_count']
                
                return DataSourceSignal(
                    source="News",
                    signal_strength=sentiment,
                    confidence=confidence,
                    key_points=[
                        f"News sentiment: {sentiment:.2f}",
                        f"Based on {article_count} articles in last 24h",
                        f"Confidence: {confidence:.2f}"
                    ],
                    raw_data=dict(result)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"News signal fetch failed for {ticker}: {e}")
            return None
    
    async def _get_technical_signal(self, ticker: str) -> Optional[DataSourceSignal]:
        """Get technical analysis signal"""
        try:
            query = """
            SELECT close_price, volume, rsi, macd_signal
            FROM stock_prices 
            WHERE ticker = %s 
            ORDER BY date DESC 
            LIMIT 20
            """
            
            results = await self.db.execute_query(query, {'ticker': ticker})
            
            if len(results) >= 2:
                recent = results[0]
                previous = results[1]
                
                price_change = (recent['close_price'] - previous['close_price']) / previous['close_price']
                
                # Simple technical scoring
                technical_score = 0.0
                key_points = []
                
                if price_change > 0.02:
                    technical_score += 0.3
                    key_points.append(f"Price up {price_change*100:.1f}% today")
                elif price_change < -0.02:
                    technical_score -= 0.3
                    key_points.append(f"Price down {abs(price_change)*100:.1f}% today")
                
                # Add RSI analysis if available
                rsi = recent.get('rsi')
                if rsi:
                    if rsi < 30:
                        technical_score += 0.2
                        key_points.append("RSI indicates oversold conditions")
                    elif rsi > 70:
                        technical_score -= 0.2
                        key_points.append("RSI indicates overbought conditions")
                
                return DataSourceSignal(
                    source="Technical",
                    signal_strength=technical_score,
                    confidence=0.7,
                    key_points=key_points,
                    raw_data={
                        'price_change': price_change,
                        'rsi': rsi,
                        'volume': recent['volume']
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Technical signal fetch failed for {ticker}: {e}")
            return None
    
    async def _get_financial_signal(self, ticker: str) -> Optional[DataSourceSignal]:
        """Get fundamental financial signal"""
        try:
            # This would integrate with financial data APIs
            # For now, return a placeholder
            return DataSourceSignal(
                source="Financial",
                signal_strength=0.0,
                confidence=0.5,
                key_points=["Financial data analysis pending"],
                raw_data={}
            )
            
        except Exception as e:
            self.logger.error(f"Financial signal fetch failed for {ticker}: {e}")
            return None
    

    
    async def _generate_rule_based_recommendation(self, ticker: str, data_sources: List[DataSourceSignal]) -> TradingRecommendation:
        """Generate rule-based recommendation as fallback"""
        try:
            # Calculate overall signal strength
            total_strength = 0.0
            total_confidence = 0.0
            
            for source in data_sources:
                weight = self._get_source_weight(source.source)
                total_strength += source.signal_strength * weight * source.confidence
                total_confidence += source.confidence * weight
            
            avg_confidence = total_confidence / len(data_sources) if data_sources else 0.5
            
            # Determine recommendation type
            if total_strength > 0.3:
                if total_strength > 0.6:
                    recommendation_type = RecommendationType.STRONG_BUY
                else:
                    recommendation_type = RecommendationType.BUY
            elif total_strength < -0.3:
                if total_strength < -0.6:
                    recommendation_type = RecommendationType.STRONG_SELL
                else:
                    recommendation_type = RecommendationType.SELL
            else:
                recommendation_type = RecommendationType.HOLD
            
            # Generate reasoning
            reasoning = self._generate_rule_based_reasoning(ticker, data_sources, total_strength)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(data_sources, total_strength)
            
            return TradingRecommendation(
                ticker=ticker,
                company_name=ticker,  # Would be improved with actual company name lookup
                recommendation=recommendation_type,
                confidence=avg_confidence,
                target_price=None,  # Would be calculated based on analysis
                stop_loss=None,     # Would be calculated based on risk tolerance
                time_horizon="1-3 months",  # Default time horizon
                reasoning=reasoning,
                risk_factors=risk_factors,
                data_sources=data_sources,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based recommendation failed for {ticker}: {e}")
            # Return neutral recommendation as final fallback
            return TradingRecommendation(
                ticker=ticker,
                company_name=ticker,
                recommendation=RecommendationType.HOLD,
                confidence=0.5,
                target_price=None,
                stop_loss=None,
                time_horizon="Unknown",
                reasoning="Insufficient data for recommendation",
                risk_factors=["Data quality issues"],
                data_sources=data_sources,
                created_at=datetime.now()
            )
    

    
    def _generate_rule_based_reasoning(self, ticker: str, data_sources: List[DataSourceSignal], total_strength: float) -> str:
        """Generate reasoning for rule-based recommendation"""
        if total_strength > 0.3:
            return f"Positive signals from {len(data_sources)} data sources indicate bullish momentum for {ticker}. Consider buying for potential upside."
        elif total_strength < -0.3:
            return f"Negative signals from {len(data_sources)} data sources indicate bearish sentiment for {ticker}. Consider selling or avoiding."
        else:
            return f"Mixed signals from {len(data_sources)} data sources suggest neutral outlook for {ticker}. Hold current position."
    
    def _identify_risk_factors(self, data_sources: List[DataSourceSignal], total_strength: float) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []
        
        # Check for conflicting signals
        positive_sources = [s for s in data_sources if s.signal_strength > 0.1]
        negative_sources = [s for s in data_sources if s.signal_strength < -0.1]
        
        if positive_sources and negative_sources:
            risk_factors.append("Conflicting signals across data sources")
        
        # Check for low confidence
        low_confidence_sources = [s for s in data_sources if s.confidence < 0.6]
        if low_confidence_sources:
            risk_factors.append("Low confidence in some data sources")
        
        # Add general market risks
        risk_factors.append("General market volatility risk")
        risk_factors.append("Liquidity risk in volatile markets")
        
        return risk_factors
    
    def _get_source_weight(self, source: str) -> float:
        """Get weight for data source"""
        weights = {
            'Reddit': 0.2,
            'News': 0.3,
            'Technical': 0.3,
            'Financial': 0.2
        }
        return weights.get(source, 0.1)


# Export main classes
__all__ = [
    # Original exports
    'SignalType', 'TradeType', 'RiskLevel', 'DataSource', 'MarketCondition', 'FeatureType',
    'VPInvestmentsError', 'ConfigurationError', 'ValidationError', 'DataError', 'APIError',
    'DEFAULT_TICKERS',
    # Recommendation exports
    'RecommendationType', 'DataSourceSignal', 'TradingRecommendation', 'RecommendationEngine'
]


class DatabaseError(VPInvestmentsError):
    """Raised when there's a database-related error."""
    pass


class SignalError(VPInvestmentsError):
    """Raised when there's a signal processing error."""
    pass


class ValidationError(VPInvestmentsError):
    """Raised when data validation fails."""
    pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constants
    'APP_NAME', 'APP_VERSION', 'DEFAULT_TIMEOUT', 'MAX_RETRIES',
    'DEFAULT_BATCH_SIZE', 'MAX_CONNECTION_POOL_SIZE', 'CONNECTION_TIMEOUT',
    'DEFAULT_ANALYSIS_WINDOW_DAYS', 'MIN_DATA_POINTS', 'DEFAULT_CONFIDENCE_THRESHOLD',
    'PRODUCTION_TIMEOUT', 'MAX_WORKERS', 'MIN_WORKERS', 'SCALING_THRESHOLD',
    'LOG_DIR', 'CACHE_DIR', 'OUTPUT_DIR', 'CONFIG_DIR',
    'DEFAULT_TICKERS', 'RATE_LIMITS',
    
    # Enums
    'SignalType', 'DataSource', 'FeatureType',
    
    # Exceptions
    'VPInvestmentsError', 'ConfigurationError', 'DataError', 
    'APIError', 'DatabaseError', 'SignalError', 'ValidationError'
]