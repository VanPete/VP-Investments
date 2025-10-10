"""
VP Investments Core - Signal Processing & Scoring

Consolidated signal engine that combines:
- Multi-source signal aggregation (Reddit, News, Financial, Technical)
- Configurable scoring profiles and weights
- Real-time normalization and risk assessment
- ML-optimized feature engineering
- Trade type classification and risk profiling
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
import numpy as np
import pandas as pd

# Import enums and constants from core module
from .core import FeatureType, SignalType, TradeType, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Investment signal data model."""
    
    ticker: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: Optional[float] = None
    technical_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    news_score: Optional[float] = None
    volume_score: Optional[float] = None
    market_regime: str = "NORMAL"  # Using string instead of enum for simplicity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.price is not None and self.price < 0:
            raise ValueError("Price cannot be negative")


@dataclass
class SignalResult:
    """Result from individual ticker signal scoring"""
    ticker: str
    weighted_score: float
    trade_type: str
    risk_level: str
    reddit_score: float
    news_score: float
    financial_score: float
    top_factors: List[str]
    signal_type: str
    confidence: float


@dataclass  
class SignalBatchResult:
    """Result of batch signal analysis (formerly models.SignalResult)."""
    
    run_id: str
    signals: List[Signal]
    execution_time_seconds: float
    total_tickers_processed: int
    total_signals_generated: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    average_confidence: float


@dataclass
class DataSourceResult:
    """Result from a data source."""
    
    source_name: str
    data: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result from the data pipeline."""
    
    results: List[DataSourceResult]
    total_execution_time_ms: float
    success_count: int
    failure_count: int
    timestamp: datetime
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRequest:
    """Request for investment analysis."""
    
    tickers: List[str]
    run_id: Optional[str] = None
    lookback_days: int = 30
    use_production_optimizations: bool = False
    analysis_type: str = "full"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    
    request: AnalysisRequest
    signal_result: SignalBatchResult  # Updated to use renamed class
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        return {
            "run_id": self.signal_result.run_id,
            "tickers_processed": self.signal_result.total_tickers_processed,
            "signals_generated": self.signal_result.total_signals_generated,
            "average_confidence": self.signal_result.average_confidence,
            "execution_time": self.signal_result.execution_time_seconds,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SignalScore:
    """Comprehensive signal score with detailed metrics"""
    ticker: str = ""
    weighted_score: float = 0.0
    confidence: float = 0.0
    signal_type: SignalType = SignalType.MULTI_FACTOR
    trade_type: TradeType = TradeType.BALANCED
    risk_level: RiskLevel = RiskLevel.MODERATE
    reddit_score: float = 0.0
    news_score: float = 0.0
    financial_score: float = 0.0
    technical_score: float = 0.0
    top_features: List[str] = field(default_factory=list)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    highest_contributor: Optional[str] = None
    lowest_contributor: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)
    risk_tags: Optional[str] = None


@dataclass 
class ScoringProfile:
    """Configuration profile for signal scoring"""
    name: str
    reddit_financial_ratio: float = 0.4  # 40% reddit, 60% other
    sentiment_weights: Dict[str, float] = field(default_factory=dict)
    financial_weights: Dict[str, float] = field(default_factory=dict) 
    technical_weights: Dict[str, float] = field(default_factory=dict)
    momentum_weights: Dict[str, float] = field(default_factory=dict)
    soft_caps: Dict[str, float] = field(default_factory=dict)
    trade_type_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SignalScorer:
    """
    Multi-factor signal scoring engine with configurable profiles
    
    Features:
    - Trade type classification (Swing, Momentum, Growth, Value, Speculative)
    - Multi-factor weighted scoring 
    - Risk assessment and profiling
    - Feature normalization and threshold application
    """
    
    def __init__(self, profile: str = "ml_optimized"):
        self.profile = profile
        self.weights = self._load_signal_weights(profile)
        self.thresholds = self._load_thresholds()
        self.trade_type_profiles = self._load_trade_type_profiles()
        
        # Current profile and statistics
        self.current_profile: Optional[ScoringProfile] = None
        self.normalization_stats: Dict[str, Dict[str, float]] = {}
        self.feature_stats = defaultdict(list)
        self.batch_metrics = {}
        
    def _load_signal_weights(self, profile: str) -> Dict[str, float]:
        """Load signal weights for the specified profile"""
        profiles = {
            "ml_optimized": {
                # Reddit/Social factors (30% total weight)
                'Reddit Sentiment': 0.12,
                'Mentions': 0.10,
                'Upvotes': 0.05,
                'Post Recency': 0.03,
                
                # Technical factors (35% total weight)
                'Price 1D %': 0.08,
                'Price 7D %': 0.12,
                'Volume Spike Ratio': 0.08,
                'RSI': 0.03,
                'MACD Histogram': 0.04,
                
                # Financial factors (35% total weight)
                'Market Cap': 0.08,
                'P/E Ratio': 0.10,
                'Volume': 0.05,
                'Beta': 0.03,
                'Volatility': 0.04,
                'Revenue Growth': 0.05,
            },
            "conservative": {
                # Lower Reddit weight, higher financial weight
                'Reddit Sentiment': 0.08,
                'Mentions': 0.07,
                'P/E Ratio': 0.15,
                'Market Cap': 0.12,
                'Revenue Growth': 0.10,
                'Price 7D %': 0.08,
                'Volume Spike Ratio': 0.05,
            },
            "aggressive": {
                # Higher Reddit and momentum weights
                'Reddit Sentiment': 0.15,
                'Mentions': 0.12,
                'Price 1D %': 0.10,
                'Price 7D %': 0.15,
                'Volume Spike Ratio': 0.12,
                'RSI': 0.05,
            }
        }
        
        return profiles.get(profile, profiles["ml_optimized"])
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load threshold values for scoring"""
        return {
            'Reddit Sentiment': 0.6,
            'Mentions': 3,
            'Price 1D %': 2.0,
            'Price 7D %': 5.0,
            'Volume Spike Ratio': 1.5,
            'RSI_LOW': 30,
            'RSI_HIGH': 70,
            'MACD Histogram': 0.1,
            'PE_LOW': 5,
            'PE_HIGH': 30,
            'Market Cap': 1e9,  # $1B minimum
        }
    
    def _load_trade_type_profiles(self) -> Dict[str, Dict[str, float]]:
        """Load trade type classification profiles"""
        return {
            "momentum": {
                "technical": 0.5,
                "sentiment": 0.3,
                "financial": 0.2
            },
            "swing": {
                "technical": 0.4,
                "sentiment": 0.4,
                "financial": 0.2
            },
            "growth": {
                "financial": 0.5,
                "technical": 0.3,
                "sentiment": 0.2
            },
            "value": {
                "financial": 0.6,
                "technical": 0.2,
                "sentiment": 0.2
            },
            "speculative": {
                "sentiment": 0.5,
                "technical": 0.3,
                "financial": 0.2
            }
        }
    
    async def score_ticker(self, ticker_data: Dict) -> SignalResult:
        """Score a single ticker using multi-factor analysis"""
        
        try:
            # Calculate component scores
            component_scores = {
                'reddit': self._calculate_reddit_score(ticker_data),
                'news': self._calculate_news_score(ticker_data),
                'financial': self._calculate_financial_score(ticker_data),
                'technical': self._calculate_technical_score(ticker_data),
                'risk': self._calculate_risk_score(ticker_data)
            }
            
            # Calculate final weighted score
            weighted_score = self._calculate_weighted_score(ticker_data, component_scores)
            
            # Classifications
            trade_type = self._classify_trade_type(ticker_data, component_scores)
            risk_level = self._assess_risk_level(ticker_data)
            signal_type = self._determine_signal_type(ticker_data)
            
            # Analysis
            top_factors = self._identify_top_factors(ticker_data, component_scores)
            confidence = self._calculate_confidence(ticker_data, component_scores)
            
            return SignalResult(
                ticker=ticker_data.get('ticker', 'UNKNOWN'),
                weighted_score=round(weighted_score, 4),
                trade_type=trade_type,
                risk_level=risk_level,
                reddit_score=round(component_scores['reddit'], 3),
                news_score=round(component_scores['news'], 3),
                financial_score=round(component_scores['financial'], 3),
                top_factors=top_factors,
                signal_type=signal_type,
                confidence=round(confidence, 3)
            )
            
        except Exception as e:
            logger.error(f"Error scoring ticker {ticker_data.get('ticker', 'UNKNOWN')}: {e}")
            return SignalResult(
                ticker=ticker_data.get('ticker', 'UNKNOWN'),
                weighted_score=0.0,
                trade_type="Balanced",
                risk_level="Unknown",
                reddit_score=0.0,
                news_score=0.0,
                financial_score=0.0,
                top_factors=[],
                signal_type="Multi-Factor",
                confidence=0.0
            )
    
    def _calculate_reddit_score(self, mention_count: int = None, avg_sentiment: float = None, 
                               avg_score: float = None, data: Dict = None) -> float:
        """
        Calculate Reddit-specific signal score.
        Moved from pipeline.py for consolidation.
        
        Can be called with individual params OR with data dict for backward compatibility.
        """
        try:
            # Support both calling patterns
            if data is not None:
                mention_count = data.get('reddit_mentions', data.get('mentions', 0))
                avg_sentiment = data.get('reddit_sentiment', 0.5)
                avg_score = data.get('reddit_score', data.get('avg_score', 0))
            
            if mention_count is None or avg_sentiment is None or avg_score is None:
                return 0.0
            
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
    
    def _calculate_news_score(self, news_data: Dict[str, Any] = None, data: Dict = None) -> float:
        """
        Calculate news-specific signal score.
        Moved from pipeline.py for consolidation.
        
        Can be called with news_data param OR with data dict for backward compatibility.
        """
        try:
            # Support both calling patterns
            if data is not None:
                news_data = data
            
            if news_data is None:
                return 0.0
            
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
    
    def _calculate_financial_score(self, data: Dict) -> float:
        """Calculate financial fundamentals score"""
        score = 0.0
        
        # P/E Ratio (inverted - lower is better, but not too low)
        pe_ratio = data.get('pe_ratio')
        if pe_ratio and self.thresholds['PE_LOW'] <= pe_ratio <= self.thresholds['PE_HIGH']:
            pe_score = 1.0 - min((pe_ratio - 15) / 25, 1.0)  # Optimal around 15
            score += pe_score * self.weights.get('P/E Ratio', 0.0)
        
        # Market Cap (log-scaled)
        market_cap = data.get('market_cap')
        if market_cap and market_cap > 0:
            # Prefer mid to large cap (better liquidity and stability)
            cap_score = min(np.log10(market_cap / 1e9) / 3, 1.0)  # Scale to billions
            score += cap_score * self.weights.get('Market Cap', 0.0)
        
        # Revenue Growth
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth > 0:
            growth_score = min(revenue_growth / 0.2, 1.0)  # Cap at 20% growth
            score += growth_score * self.weights.get('Revenue Growth', 0.0)
        
        return max(0.0, score)
    
    def _calculate_technical_score(self, data: Dict) -> float:
        """Calculate technical analysis score"""
        score = 0.0
        
        # Price momentum
        price_1d = data.get('price_change_1d', data.get('price_1d_pct', 0))
        price_7d = data.get('price_change_7d', data.get('price_7d_pct', 0))
        
        if price_1d > self.thresholds['Price 1D %']:
            score += (price_1d / 10) * self.weights.get('Price 1D %', 0.0)  # Scale by 10%
        
        if price_7d > self.thresholds['Price 7D %']:
            score += (price_7d / 20) * self.weights.get('Price 7D %', 0.0)  # Scale by 20%
        
        # Volume spike
        volume_spike = data.get('volume_spike_ratio', 1.0)
        if volume_spike > self.thresholds['Volume Spike Ratio']:
            spike_score = min((volume_spike - 1.0) / 2.0, 1.0)  # Cap at 3x volume
            score += spike_score * self.weights.get('Volume Spike Ratio', 0.0)
        
        # RSI (prefer 30-70 range)
        rsi = data.get('rsi')
        if rsi and self.thresholds['RSI_LOW'] <= rsi <= self.thresholds['RSI_HIGH']:
            rsi_score = 1.0 - abs(rsi - 50) / 50  # Optimal at 50
            score += rsi_score * self.weights.get('RSI', 0.0)
        
        # MACD
        macd = data.get('macd', data.get('macd_histogram', 0))
        if macd > self.thresholds['MACD Histogram']:
            macd_score = min(macd / 5.0, 1.0)  # Cap at 5
            score += macd_score * self.weights.get('MACD Histogram', 0.0)
        
        return max(0.0, score)
    
    def _calculate_options_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate options sentiment score.
        Moved from pipeline.py for consolidation.
        """
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
    
    def _calculate_risk_penalty(self, risk_score: float) -> float:
        """
        Calculate risk penalty for score.
        Moved from pipeline.py for consolidation.
        """
        if risk_score > 80:
            return -0.02  # High risk penalty
        elif risk_score > 60:
            return -0.01  # Moderate risk penalty  
        else:
            return 0.0
    
    def _calculate_risk_score(self, data: Dict) -> float:
        """Calculate risk-adjusted score (negative factors)"""
        risk_penalty = 0.0
        
        # High volatility penalty
        volatility = data.get('volatility', 0)
        if volatility > 0.05:  # 5% volatility threshold
            risk_penalty += (volatility - 0.05) * 10  # Penalty for high vol
        
        # High beta penalty
        beta = data.get('beta', data.get('beta_vs_spy', 1.0))
        if beta > 1.5:
            risk_penalty += (beta - 1.5) * 0.2  # Penalty for high beta
        
        return -risk_penalty  # Negative because it's a penalty
    
    def _calculate_weighted_score(self, data: Dict, component_scores: Dict) -> float:
        """Calculate final weighted score"""
        total_score = 0.0
        
        # Weight components
        total_score += component_scores['reddit'] * 0.3      # 30% reddit
        total_score += component_scores['news'] * 0.1       # 10% news  
        total_score += component_scores['financial'] * 0.25  # 25% fundamentals
        total_score += component_scores['technical'] * 0.3   # 30% technical
        total_score += component_scores['risk'] * 0.05       # 5% risk adjustment
        
        # Apply emerging boost if applicable
        if self._is_emerging_signal(data):
            total_score *= 1.2  # 20% boost for emerging signals
        
        return max(0.0, total_score)
    
    def _classify_trade_type(self, data: Dict, component_scores: Dict) -> str:
        """Classify the trade type based on scoring profile"""
        
        # Simple heuristic classification
        reddit_strength = component_scores['reddit']
        technical_strength = component_scores['technical'] 
        financial_strength = component_scores['financial']
        
        # Determine dominant factor
        if reddit_strength > 0.5 and data.get('reddit_mentions', data.get('mentions', 0)) >= 5:
            return "Speculative" if data.get('volatility', 0) > 0.04 else "Swing"
        elif technical_strength > 0.4 and data.get('price_change_7d', data.get('price_7d_pct', 0)) > 5:
            return "Momentum"
        elif financial_strength > 0.3:
            return "Growth" if data.get('pe_ratio', 30) < 25 else "Value"
        else:
            return "Balanced"
    
    def _assess_risk_level(self, data: Dict) -> str:
        """Assess overall risk level"""
        risk_factors = 0
        
        # High volatility
        if data.get('volatility', 0) > 0.04:
            risk_factors += 1
        
        # High beta
        if data.get('beta', data.get('beta_vs_spy', 1.0)) > 1.3:
            risk_factors += 1
        
        # Low market cap (< $1B)
        if data.get('market_cap', 0) < 1e9:
            risk_factors += 1
        
        # High P/E ratio
        if data.get('pe_ratio', 0) > 30:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High"
        elif risk_factors >= 1:
            return "Moderate"
        else:
            return "Low"
    
    def _identify_top_factors(self, data: Dict, component_scores: Dict) -> List[str]:
        """Identify top contributing factors"""
        factors = []
        
        # Reddit factors
        if component_scores['reddit'] > 0.3:
            mentions = data.get('reddit_mentions', data.get('mentions', 0))
            sentiment = data.get('reddit_sentiment', 0.5)
            if mentions >= 5:
                factors.append("High Reddit Mentions")
            if sentiment > 0.6:
                factors.append("Positive Sentiment")
        
        # Technical factors
        if component_scores['technical'] > 0.3:
            price_7d = data.get('price_change_7d', data.get('price_7d_pct', 0))
            volume_spike = data.get('volume_spike_ratio', 1.0)
            if price_7d > 5:
                factors.append("Price Momentum")
            if volume_spike > 1.5:
                factors.append("Volume Surge")
        
        # Financial factors
        if component_scores['financial'] > 0.2:
            if data.get('market_cap', 0) > 1e10:  # $10B+
                factors.append("Large Cap Stability")
            if data.get('revenue_growth', 0) > 0.1:
                factors.append("Revenue Growth")
        
        return factors[:3]  # Return top 3 factors
    
    def _determine_signal_type(self, data: Dict) -> str:
        """Determine the type of signal"""
        mentions = data.get('reddit_mentions', data.get('mentions', 0))
        volume_spike = data.get('volume_spike_ratio', 1.0)
        price_7d = data.get('price_change_7d', data.get('price_7d_pct', 0))
        
        if mentions >= 10:
            return "Reddit Surge"
        elif volume_spike > 2.0:
            return "Volume Breakout"
        elif price_7d > 10:
            return "Price Momentum"
        else:
            return "Multi-Factor"
    
    def _calculate_confidence(self, data: Dict, component_scores: Dict) -> float:
        """Calculate confidence in the signal"""
        confidence_factors = []
        
        # Data quality factors
        mentions = data.get('reddit_mentions', data.get('mentions', 0))
        if mentions >= 3:
            confidence_factors.append(0.2)
        if data.get('volume', 0) > 1000000:
            confidence_factors.append(0.2)
        if data.get('market_cap', 0) > 1e9:
            confidence_factors.append(0.2)
        
        # Signal strength factors
        if component_scores['reddit'] > 0.3:
            confidence_factors.append(0.2)
        if component_scores['technical'] > 0.3:
            confidence_factors.append(0.2)
        
        return min(sum(confidence_factors), 1.0)
    
    def _is_emerging_signal(self, data: Dict) -> bool:
        """Check if this is an emerging signal (sudden appearance)"""
        mentions = data.get('reddit_mentions', data.get('mentions', 0))
        first_mention = data.get('first_mention')
        
        # Consider emerging if high mentions and recent first appearance
        if mentions >= 5 and first_mention:
            hours_since_first = (datetime.now() - first_mention).total_seconds() / 3600
            return hours_since_first <= 12  # First seen within 12 hours
        
        return False


# Global scorer instance
signal_scorer = SignalScorer()


async def score_ticker_data(ticker_data: Dict) -> SignalResult:
    """Score individual ticker data"""
    return await signal_scorer.score_ticker(ticker_data)


async def score_multiple_tickers(tickers_data: List[Dict]) -> List[SignalResult]:
    """Score multiple tickers and return sorted results"""
    results = []
    
    for ticker_data in tickers_data:
        try:
            result = await signal_scorer.score_ticker(ticker_data)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to score ticker: {e}")
            continue
    
    # Sort by weighted score
    results.sort(key=lambda x: x.weighted_score, reverse=True)
    
    return results


# Also include signal enhancement functionality from misc/signal_enhancer.py
class SignalEnhancer:
    """Enhanced signal processing with comprehensive risk and quality metrics."""
    
    def __init__(self):
        self.market_cap_thresholds = {
            'Nano': 50_000_000,      # $50M
            'Micro': 300_000_000,    # $300M
            'Small': 2_000_000_000,  # $2B
            'Mid': 10_000_000_000,   # $10B
            'Large': 200_000_000_000 # $200B
        }
    
    def enhance_signal_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a signal record with calculated fields."""
        enhanced = record.copy()
        
        # Calculate market cap category
        enhanced['market_cap_category'] = self._calculate_market_cap_category(
            record.get('market_cap')
        )
        
        # Calculate comprehensive risk score
        enhanced['risk_score'] = self._calculate_risk_score(record)
        
        # Calculate liquidity score
        enhanced['liquidity_score'] = self._calculate_liquidity_score(record)
        
        # Calculate entry quality score
        enhanced['entry_quality_score'] = self._calculate_entry_quality_score(record)
        
        # Calculate risk-adjusted score
        enhanced['risk_adjusted_score'] = self._calculate_risk_adjusted_score(
            record.get('weighted_score', 0), enhanced['risk_score']
        )
        
        # Calculate expected hold duration
        enhanced['expected_hold_duration'] = self._calculate_expected_hold_duration(record)
        
        # Calculate additional technical indicators
        enhanced.update(self._calculate_technical_indicators(record))
        
        # Calculate sentiment and flow indicators
        enhanced.update(self._calculate_sentiment_indicators(record))
        
        return enhanced
    
    def _calculate_market_cap_category(self, market_cap: Optional[float]) -> str:
        """Calculate market cap category."""
        if market_cap is None:
            return 'Unknown'
        
        if market_cap < self.market_cap_thresholds['Nano']:
            return 'Nano'
        elif market_cap < self.market_cap_thresholds['Micro']:
            return 'Micro'
        elif market_cap < self.market_cap_thresholds['Small']:
            return 'Small'
        elif market_cap < self.market_cap_thresholds['Mid']:
            return 'Mid'
        elif market_cap < self.market_cap_thresholds['Large']:
            return 'Large'
        else:
            return 'Mega'
    
    def _calculate_risk_score(self, record: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score (0-100)."""
        risk_components = []
        
        # Volatility risk (30% weight)
        volatility = record.get('volatility', 0.15)  # Default 15% annual volatility
        vol_risk = min(30, volatility * 30)
        risk_components.append(vol_risk)
        
        # Debt risk (20% weight)
        debt_equity = record.get('debt_equity', 25)  # Default moderate debt
        if debt_equity > 100:
            debt_risk = 25
        elif debt_equity > 50:
            debt_risk = 15
        elif debt_equity > 25:
            debt_risk = 10
        else:
            debt_risk = 5
        risk_components.append(debt_risk)
        
        # Market cap risk (15% weight)
        market_cap = record.get('market_cap', 5_000_000_000)  # Default $5B
        if market_cap < 1_000_000_000:  # <$1B = small cap
            cap_risk = 15
        elif market_cap < 10_000_000_000:  # <$10B = mid cap
            cap_risk = 10
        else:  # Large cap
            cap_risk = 5
        risk_components.append(cap_risk)
        
        # Momentum risk (15% weight) - high momentum = higher risk
        momentum = abs(record.get('momentum_30d_pct', 0))
        if momentum > 100:
            momentum_risk = 15
        elif momentum > 50:
            momentum_risk = 10
        else:
            momentum_risk = 5
        risk_components.append(momentum_risk)
        
        # Short squeeze risk (10% weight)
        short_pct = record.get('short_pct_float', 5)  # Default 5%
        if short_pct > 30:
            short_risk = 10
        elif short_pct > 20:
            short_risk = 7
        elif short_pct > 10:
            short_risk = 5
        else:
            short_risk = 2
        risk_components.append(short_risk)
        
        # Volume spike risk (10% weight)
        vol_spike = record.get('volume_spike_ratio', 1.0)
        if vol_spike > 5:
            vol_spike_risk = 10
        elif vol_spike > 2:
            vol_spike_risk = 7
        elif vol_spike > 1.5:
            vol_spike_risk = 5
        else:
            vol_spike_risk = 2
        risk_components.append(vol_spike_risk)
        
        return sum(risk_components)
    
    def _calculate_liquidity_score(self, record: Dict[str, Any]) -> float:
        """Calculate liquidity score based on volume and market cap."""
        volume_usd = record.get('volume_usd_1d', 0)
        market_cap = record.get('market_cap', 1)
        
        if not volume_usd or not market_cap:
            return 0.2  # Low liquidity default
        
        daily_turnover = volume_usd / market_cap
        
        if daily_turnover > 0.05:  # >5% of market cap daily
            return 1.0
        elif daily_turnover > 0.02:  # >2% of market cap daily
            return 0.8
        elif daily_turnover > 0.01:  # >1% of market cap daily
            return 0.6
        elif daily_turnover > 0.002:  # >0.2% of market cap daily
            return 0.4
        else:
            return 0.2
    
    def _calculate_entry_quality_score(self, record: Dict[str, Any]) -> float:
        """Calculate entry quality based on technical confluence."""
        score_components = []
        
        # RSI component (25% weight)
        rsi = record.get('rsi', 50)
        if 30 <= rsi <= 70:  # Neutral zone
            rsi_score = 0.25
        elif 20 <= rsi <= 80:  # Acceptable zone
            rsi_score = 0.15
        else:  # Extreme zones
            rsi_score = 0.05
        score_components.append(rsi_score)
        
        # Momentum component (25% weight)
        momentum = record.get('momentum_30d_pct', 0)
        if momentum > 20:
            momentum_score = 0.25
        elif momentum > 10:
            momentum_score = 0.20
        elif momentum > 0:
            momentum_score = 0.15
        else:
            momentum_score = 0.05
        score_components.append(momentum_score)
        
        # Volume component (25% weight)
        vol_spike = record.get('volume_spike_ratio', 1.0)
        if vol_spike > 2.0:
            vol_score = 0.25
        elif vol_spike > 1.5:
            vol_score = 0.20
        elif vol_spike > 1.2:
            vol_score = 0.15
        else:
            vol_score = 0.05
        score_components.append(vol_score)
        
        # Moving average position component (25% weight)
        above_200ma = record.get('above_200d_ma_pct', 0)
        above_50ma = record.get('above_50d_ma_pct', 0)
        
        if above_200ma > 10 and above_50ma > 5:
            ma_score = 0.25
        elif above_200ma > 0 or above_50ma > 0:
            ma_score = 0.15
        else:
            ma_score = 0.05
        score_components.append(ma_score)
        
        return sum(score_components)
    
    def _calculate_risk_adjusted_score(self, weighted_score: float, risk_score: float) -> float:
        """Calculate risk-adjusted weighted score."""
        if not weighted_score:
            return 0.0
        
        # Adjust score by risk (higher risk = lower adjusted score)
        risk_adjustment = (100 - risk_score) / 100
        return weighted_score * risk_adjustment
    
    def _calculate_expected_hold_duration(self, record: Dict[str, Any]) -> int:
        """Calculate expected holding period in days based on signal type and momentum."""
        trade_type = record.get('trade_type', 'Multi-Factor')
        momentum = record.get('momentum_30d_pct', 10)
        
        if 'Momentum' in trade_type:
            if momentum > 50:
                return 14  # High momentum: 2 weeks
            elif momentum > 20:
                return 21  # Medium momentum: 3 weeks
            else:
                return 30  # Low momentum: 1 month
        elif 'Value' in trade_type or trade_type == 'Mean Reversion':
            return 90  # Value plays: 3 months
        elif trade_type == 'Earnings Reaction':
            return 5   # Earnings plays: 1 week
        elif trade_type == 'Short Squeeze':
            return 10  # Squeeze plays: 2 weeks
        elif 'Growth' in trade_type:
            return 60  # Growth stories: 2 months
        else:
            return 30  # Default: 1 month
    
    def _calculate_technical_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional technical indicators."""
        indicators = {}
        
        # MACD components (if not already calculated)
        if not record.get('macd_histogram') and record.get('current_price'):
            # Simplified MACD calculation - in production, use proper price history
            indicators['macd_signal'] = None  # Would need price history
            indicators['macd_line'] = None
        
        # Bollinger Band position
        current_price = record.get('current_price', 0)
        if current_price and record.get('volatility'):
            # Simplified calculation - in production, use proper price history and SMA
            volatility = record.get('volatility', 0.2)
            # Estimate position within bands (0.5 = middle, 0 = lower, 1 = upper)
            indicators['bollinger_position'] = 0.5  # Placeholder - needs proper calculation
            indicators['bollinger_upper'] = None
            indicators['bollinger_lower'] = None
        
        # Volume-price correlation estimate
        vol_spike = record.get('volume_spike_ratio', 1.0)
        price_change = record.get('price_1d_pct', 0)
        
        if abs(price_change) > 0:
            # Rough estimate of volume-price correlation
            indicators['volume_price_correlation'] = min(1.0, vol_spike * abs(price_change) / 100)
        else:
            indicators['volume_price_correlation'] = 0.0
        
        return indicators
    
    def _calculate_sentiment_indicators(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sentiment and flow indicators."""
        indicators = {}
        
        # Reddit momentum score
        reddit_sentiment = record.get('reddit_sentiment', 0)
        mentions = record.get('mentions', 0)
        
        if mentions > 0:
            indicators['reddit_momentum_score'] = min(1.0, reddit_sentiment * mentions / 10)
        else:
            indicators['reddit_momentum_score'] = 0.0
        
        # Social sentiment trend (would need historical data in production)
        indicators['social_sentiment_trend'] = 'Stable'  # Placeholder
        
        # News sentiment score (enhanced from basic news_sentiment)
        news_sentiment = record.get('news_sentiment', 0)
        news_mentions = record.get('news_mentions', 0)
        
        if news_mentions > 0:
            indicators['news_sentiment_score'] = news_sentiment * min(1.0, news_mentions / 5)
        else:
            indicators['news_sentiment_score'] = 0.0
        
        # Options flow score (placeholder - would need actual options data)
        indicators['options_flow_score'] = 0.5  # Neutral placeholder
        
        # Institutional flow direction (placeholder - would need institutional data)
        indicators['institutional_flow_direction'] = 'Unknown'
        
        return indicators


def enhance_signals_batch(signals_data: list) -> list:
    """
    Enhance a batch of signals with calculated fields.
    
    Args:
        signals_data: List of signal dictionaries
        
    Returns:
        List of enhanced signal dictionaries
    """
    enhancer = SignalEnhancer()
    enhanced_signals = []
    
    for signal in signals_data:
        try:
            enhanced_signal = enhancer.enhance_signal_record(signal)
            enhanced_signals.append(enhanced_signal)
        except Exception as e:
            logger.warning(f"Failed to enhance signal for {signal.get('ticker', 'unknown')}: {e}")
            # Return original signal if enhancement fails
            enhanced_signals.append(signal)
    
    return enhanced_signals


# ============================================================================
# PORTFOLIO AND ANALYSIS MODELS (from models.py)
# ============================================================================

from decimal import Decimal

@dataclass
class Feature:
    """Analysis feature data model."""
    
    name: str
    feature_type: FeatureType
    value: float
    timestamp: datetime
    ticker: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRun:
    """Analysis run metadata."""
    
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    tickers_processed: int = 0
    features_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Risk analysis metrics."""
    
    ticker: str
    timestamp: datetime
    volatility: float
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    value_at_risk_95: Optional[float] = None
    value_at_risk_99: Optional[float] = None
    expected_shortfall: Optional[float] = None
    downside_deviation: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_level(self) -> str:
        """Categorize risk level."""
        if self.volatility < 0.15:
            return "LOW"
        elif self.volatility < 0.25:
            return "MEDIUM"
        else:
            return "HIGH"


@dataclass
class PerformanceMetrics:
    """Performance analysis metrics."""
    
    ticker: str
    timestamp: datetime
    return_1d: Optional[float] = None
    return_1w: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    return_1y: Optional[float] = None
    annualized_return: Optional[float] = None
    cumulative_return: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def performance_grade(self) -> str:
        """Grade performance based on returns."""
        if self.return_1m is None:
            return "N/A"
        
        if self.return_1m > 0.10:
            return "A+"
        elif self.return_1m > 0.05:
            return "A"
        elif self.return_1m > 0.02:
            return "B"
        elif self.return_1m > 0:
            return "C"
        else:
            return "D"


@dataclass
class PortfolioPosition:
    """Individual portfolio position."""
    
    ticker: str
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    entry_date: datetime
    weight: float
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return self.current_price * self.quantity
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.entry_price == 0:
            return 0.0
        return float(((self.current_price - self.entry_price) / self.entry_price) * 100)


@dataclass
class PortfolioAnalysis:
    """Portfolio-level analysis."""
    
    portfolio_id: str
    timestamp: datetime
    positions: List[PortfolioPosition]
    total_value: Decimal
    total_return: float
    risk_metrics: RiskMetrics
    performance_metrics: PerformanceMetrics
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    geographic_allocation: Dict[str, float] = field(default_factory=dict)
    rebalancing_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def position_count(self) -> int:
        """Get number of positions."""
        return len(self.positions)
    
    @property
    def top_performers(self) -> List[PortfolioPosition]:
        """Get top performing positions."""
        return sorted(self.positions, key=lambda p: p.total_return, reverse=True)[:5]
    
    @property
    def worst_performers(self) -> List[PortfolioPosition]:
        """Get worst performing positions."""
        return sorted(self.positions, key=lambda p: p.total_return)[:5]
    
    def get_allocation_by_ticker(self) -> Dict[str, float]:
        """Get portfolio allocation by ticker."""
        total_value = sum(pos.market_value for pos in self.positions)
        if total_value == 0:
            return {}
        
        return {
            pos.ticker: float(pos.market_value / total_value * 100)
            for pos in self.positions
        }


# Export main classes and functions
signal_enhancer = SignalEnhancer()

__all__ = [
    # Core signal classes
    'Signal',
    'SignalResult', 
    'SignalBatchResult',
    'SignalScore',
    'SignalScorer',
    'SignalEnhancer',
    
    # Analysis classes
    'AnalysisRequest',
    'AnalysisResult',
    'DataSourceResult', 
    'PipelineResult',
    
    # Portfolio and Analysis Models
    'Feature',
    'AnalysisRun', 
    'RiskMetrics',
    'PerformanceMetrics',
    'PortfolioPosition',
    'PortfolioAnalysis',
    
    # Enums
    'SignalType',
    'TradeType', 
    'RiskLevel',
    
    # Functions and instances
    'signal_scorer',
    'signal_enhancer',
    'score_ticker_data',
    'score_multiple_tickers',
    'enhance_signals_batch'
]