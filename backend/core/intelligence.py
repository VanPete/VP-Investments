"""
VP Investments Analysis Intelligence Hub

Consolidates comprehensive analysis capabilities including:
- Analysis Engine: Multi-source data processing and signal generation
- Signal Orchestrator: Pipeline coordination and comprehensive signal creation  
- Signal Optimizer: Advanced scoring improvements and market regime analysis

This unified intelligence system provides:
1. Feature engineering and technical analysis
2. Machine learning model integration
3. Signal orchestration and coordination
4. Dynamic optimization based on market conditions
5. Risk-adjusted confidence metrics
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum

from .signals import Signal, SignalBatchResult as SignalResult, Feature, AnalysisRun
from .core import SignalType, FeatureType
from ..core.config import get_config, ConfigManager
from ..storage.database import DatabaseInterface, get_supabase_database
from ..utils.logger import get_logger
from ..utils.observability import emit_metric

logger = logging.getLogger(__name__)


# ===== ANALYSIS ENGINE COMPONENTS =====

@dataclass
class AnalysisConfig:
    """Configuration for analysis engine"""
    min_confidence_threshold: float
    max_signals_per_run: int
    feature_lookback_days: int
    sentiment_weight: float
    technical_weight: float
    volume_weight: float
    news_weight: float


class FeatureEngineer:
    """Feature engineering for stock analysis"""
    
    def __init__(self, db: DatabaseInterface):
        self.db = db
        self.logger = logging.getLogger('feature_engineer')
    
    async def extract_features(self, ticker: str, lookback_days: int = 30) -> List[Feature]:
        """Extract comprehensive features for a ticker"""
        try:
            # Get price data
            price_features = await self._extract_price_features(ticker, lookback_days)
            
            # Get volume features
            volume_features = await self._extract_volume_features(ticker, lookback_days)
            
            # Get technical features
            technical_features = await self._extract_technical_features(ticker, lookback_days)
            
            # Get sentiment features
            sentiment_features = await self._extract_sentiment_features(ticker, lookback_days)
            
            all_features = price_features + volume_features + technical_features + sentiment_features
            
            self.logger.info(f"Extracted {len(all_features)} features for {ticker}")
            return all_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {ticker}: {e}")
            return []
    
    async def _extract_price_features(self, ticker: str, lookback_days: int) -> List[Feature]:
        """Extract price-based features"""
        features = []
        
        try:
            # Get recent price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            query = """
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM stock_prices 
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date DESC
            """
            
            price_data = await self.db.fetch_all(query, (ticker, start_date.date(), end_date.date()))
            
            if len(price_data) < 5:
                return features
            
            prices = [row['close_price'] for row in price_data]
            
            # Price momentum features
            features.append(Feature(
                ticker=ticker,
                feature_type=FeatureType.MOMENTUM,
                feature_name="price_momentum_5d",
                feature_value=self._calculate_momentum(prices[:5]),
                created_at=datetime.now()
            ))
            
            features.append(Feature(
                ticker=ticker,
                feature_type=FeatureType.MOMENTUM,
                feature_name="price_momentum_20d",
                feature_value=self._calculate_momentum(prices[:20]) if len(prices) >= 20 else 0.0,
                created_at=datetime.now()
            ))
            
            # Volatility features
            features.append(Feature(
                ticker=ticker,
                feature_type=FeatureType.VOLATILITY,
                feature_name="price_volatility",
                feature_value=np.std(prices[:20]) if len(prices) >= 20 else 0.0,
                created_at=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Price feature extraction failed for {ticker}: {e}")
            
        return features
    
    async def _extract_volume_features(self, ticker: str, lookback_days: int) -> List[Feature]:
        """Extract volume-based features"""
        features = []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            query = """
            SELECT volume, close_price
            FROM stock_prices 
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date DESC
            """
            
            volume_data = await self.db.fetch_all(query, (ticker, start_date.date(), end_date.date()))
            
            if len(volume_data) < 5:
                return features
            
            volumes = [row['volume'] for row in volume_data]
            avg_volume = np.mean(volumes)
            recent_volume = volumes[0] if volumes else 0
            
            features.append(Feature(
                ticker=ticker,
                feature_type=FeatureType.VOLUME,
                feature_name="volume_ratio",
                feature_value=recent_volume / avg_volume if avg_volume > 0 else 0.0,
                created_at=datetime.now()
            ))
            
        except Exception as e:
            self.logger.error(f"Volume feature extraction failed for {ticker}: {e}")
            
        return features
    
    async def _extract_technical_features(self, ticker: str, lookback_days: int) -> List[Feature]:
        """Extract technical analysis features"""
        features = []
        
        try:
            # RSI, MACD, Bollinger Bands, etc.
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 50)  # Extra data for indicators
            
            query = """
            SELECT date, close_price, high_price, low_price
            FROM stock_prices 
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date ASC
            """
            
            price_data = await self.db.fetch_all(query, (ticker, start_date.date(), end_date.date()))
            
            if len(price_data) < 20:
                return features
            
            closes = np.array([row['close_price'] for row in price_data])
            
            # Simple moving averages
            if len(closes) >= 20:
                sma_20 = np.mean(closes[-20:])
                current_price = closes[-1]
                
                features.append(Feature(
                    ticker=ticker,
                    feature_type=FeatureType.TECHNICAL,
                    feature_name="sma_20_ratio",
                    feature_value=current_price / sma_20 if sma_20 > 0 else 1.0,
                    created_at=datetime.now()
                ))
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed for {ticker}: {e}")
            
        return features
    
    async def _extract_sentiment_features(self, ticker: str, lookback_days: int) -> List[Feature]:
        """Extract sentiment-based features"""
        features = []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Reddit sentiment
            reddit_query = """
            SELECT sentiment_score, confidence
            FROM reddit_sentiment 
            WHERE ticker = %s AND created_at >= %s
            ORDER BY created_at DESC
            """
            
            reddit_data = await self.db.fetch_all(reddit_query, (ticker, start_date))
            
            if reddit_data:
                avg_sentiment = np.mean([row['sentiment_score'] for row in reddit_data])
                features.append(Feature(
                    ticker=ticker,
                    feature_type=FeatureType.SENTIMENT,
                    feature_name="reddit_sentiment",
                    feature_value=avg_sentiment,
                    created_at=datetime.now()
                ))
            
            # News sentiment
            news_query = """
            SELECT sentiment_score, confidence
            FROM news_sentiment 
            WHERE ticker = %s AND created_at >= %s
            ORDER BY created_at DESC
            """
            
            news_data = await self.db.fetch_all(news_query, (ticker, start_date))
            
            if news_data:
                avg_news_sentiment = np.mean([row['sentiment_score'] for row in news_data])
                features.append(Feature(
                    ticker=ticker,
                    feature_type=FeatureType.SENTIMENT,
                    feature_name="news_sentiment",
                    feature_value=avg_news_sentiment,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            self.logger.error(f"Sentiment feature extraction failed for {ticker}: {e}")
            
        return features
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        if len(prices) < 2:
            return 0.0
        
        return (prices[0] - prices[-1]) / prices[-1] if prices[-1] != 0 else 0.0


class MLSignalScorer:
    """Machine learning based signal scoring"""
    
    def __init__(self, db: DatabaseInterface):
        self.db = db
        self.model = None
        self.logger = logging.getLogger('ml_scorer')
    
    async def load_model(self, model_path: Optional[str] = None):
        """Load trained ML model"""
        try:
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Loaded ML model from {model_path}")
            else:
                self.logger.warning("No ML model available, using fallback scoring")
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
    
    async def score_signal(self, features: List[Feature]) -> float:
        """Score a signal based on features"""
        if not features:
            return 0.0
        
        try:
            if self.model:
                return await self._ml_score(features)
            else:
                return await self._heuristic_score(features)
        except Exception as e:
            self.logger.error(f"Signal scoring failed: {e}")
            return 0.0
    
    async def _ml_score(self, features: List[Feature]) -> float:
        """ML-based scoring"""
        # Convert features to model input format
        feature_vector = self._features_to_vector(features)
        
        if self.model and feature_vector is not None:
            score = self.model.predict([feature_vector])[0]
            return max(0.0, min(1.0, score))
        
        return 0.0
    
    async def _heuristic_score(self, features: List[Feature]) -> float:
        """Fallback heuristic scoring"""
        score = 0.0
        weights = {
            'momentum': 0.3,
            'sentiment': 0.25,
            'technical': 0.25,
            'volume': 0.2
        }
        
        feature_scores = {}
        
        for feature in features:
            category = feature.feature_name.split('_')[0]
            if category not in feature_scores:
                feature_scores[category] = []
            feature_scores[category].append(feature.feature_value)
        
        for category, values in feature_scores.items():
            if category in weights:
                avg_value = np.mean(values)
                normalized_value = max(0.0, min(1.0, (avg_value + 1) / 2))  # Normalize to 0-1
                score += weights[category] * normalized_value
        
        return score
    
    def _features_to_vector(self, features: List[Feature]) -> Optional[List[float]]:
        """Convert features to ML model input vector"""
        try:
            # Define expected feature order for model
            expected_features = [
                'price_momentum_5d', 'price_momentum_20d', 'price_volatility',
                'volume_ratio', 'sma_20_ratio', 'reddit_sentiment', 'news_sentiment'
            ]
            
            feature_dict = {f.feature_name: f.feature_value for f in features}
            
            vector = []
            for feature_name in expected_features:
                vector.append(feature_dict.get(feature_name, 0.0))
            
            return vector
            
        except Exception as e:
            self.logger.error(f"Feature vectorization failed: {e}")
            return None


class AnalysisEngine:
    """Main analysis engine for signal generation"""
    
    def __init__(self, db: DatabaseInterface, config: AnalysisConfig):
        self.db = db
        self.config = config
        self.feature_engineer = FeatureEngineer(db)
        self.ml_scorer = MLSignalScorer(db)
        self.logger = logging.getLogger('analysis_engine')
    
    async def initialize(self):
        """Initialize the analysis engine"""
        await self.ml_scorer.load_model()
        self.logger.info("Analysis engine initialized")
    
    async def generate_signals(self, tickers: List[str]) -> SignalResult:
        """Generate signals for a list of tickers"""
        start_time = datetime.now()
        signals = []
        
        try:
            for ticker in tickers[:self.config.max_signals_per_run]:
                signal = await self._analyze_ticker(ticker)
                if signal and signal.confidence >= self.config.min_confidence_threshold:
                    signals.append(signal)
            
            # Create analysis run record
            analysis_run = AnalysisRun(
                run_id=f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}",
                start_time=start_time,
                end_time=datetime.now(),
                tickers_analyzed=len(tickers),
                signals_generated=len(signals),
                status="completed"
            )
            
            result = SignalResult(
                signals=signals,
                metadata={
                    'analysis_run': analysis_run,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.logger.info(f"Generated {len(signals)} signals for {len(tickers)} tickers")
            return result
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return SignalResult(signals=[], metadata={'error': str(e)})
    
    async def _analyze_ticker(self, ticker: str) -> Optional[Signal]:
        """Analyze a single ticker"""
        try:
            # Extract features
            features = await self.feature_engineer.extract_features(
                ticker, self.config.feature_lookback_days
            )
            
            if not features:
                return None
            
            # Score the signal
            confidence = await self.ml_scorer.score_signal(features)
            
            if confidence < self.config.min_confidence_threshold:
                return None
            
            # Determine signal type based on features
            signal_type = self._determine_signal_type(features)
            
            signal = Signal(
                ticker=ticker,
                signal_type=signal_type,
                confidence=confidence,
                created_at=datetime.now(),
                features=features,
                metadata={
                    'feature_count': len(features),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Ticker analysis failed for {ticker}: {e}")
            return None
    
    def _determine_signal_type(self, features: List[Feature]) -> SignalType:
        """Determine signal type based on features"""
        momentum_score = 0.0
        sentiment_score = 0.0
        
        for feature in features:
            if 'momentum' in feature.feature_name:
                momentum_score += feature.feature_value
            elif 'sentiment' in feature.feature_name:
                sentiment_score += feature.feature_value
        
        if momentum_score > 0.1 and sentiment_score > 0.1:
            return SignalType.BUY
        elif momentum_score < -0.1 or sentiment_score < -0.1:
            return SignalType.SELL
        else:
            return SignalType.HOLD


# ===== SIGNAL ORCHESTRATOR COMPONENTS =====

@dataclass 
class SignalScore:
    """Placeholder for signal scoring data"""
    confidence: float
    signal_type: SignalType
    risk_level: str = "medium"

@dataclass
class ComprehensiveSignal:
    """Complete signal with all analysis components"""
    ticker: str
    
    # Core signal scoring
    signal_score: SignalScore
    
    # Component data
    reddit_data: Dict[str, Any]
    news_data: Dict[str, Any]
    financial_data: Dict[str, Any]
    
    # Analysis metadata
    created_at: datetime
    confidence: float
    risk_assessment: Dict[str, Any]
    
    # Performance tracking
    processing_time_ms: float = 0.0
    data_quality_score: float = 0.0


@dataclass
class OrchestrationConfig:
    """Configuration for signal orchestration"""
    max_concurrent_analyses: int = 10
    analysis_timeout_seconds: int = 30
    min_data_quality_threshold: float = 0.6
    enable_real_time_updates: bool = True
    cache_results_hours: int = 1
    
    # Component weights
    reddit_weight: float = 0.25
    news_weight: float = 0.30
    financial_weight: float = 0.35
    technical_weight: float = 0.10


class SignalOrchestrator:
    """Coordinates comprehensive signal generation pipeline"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.db = get_supabase_database()
        
        # Component analyzers - will be initialized lazily
        self._signal_engine = None
        self._reddit_analyzer = None
        self._news_analyzer = None
    
    async def initialize(self):
        """Initialize all analysis components"""
        try:
            # Initialize signal engine (with fallback)
            try:
                from vp_investments.analysis.signal_engine import get_consolidated_signal_engine
                self._signal_engine = await get_consolidated_signal_engine()
            except ImportError:
                self.logger.warning("Signal engine not available, using fallback")
                self._signal_engine = None
            
            # Initialize sentiment analyzers (with fallbacks)
            try:
                from vp_investments.analysis.reddit_sentiment import get_reddit_sentiment_analyzer
                self._reddit_analyzer = await get_reddit_sentiment_analyzer()
            except ImportError:
                self.logger.warning("Reddit analyzer not available")
                self._reddit_analyzer = None
                
            try:
                from vp_investments.analysis.news_sentiment import get_news_sentiment_analyzer
                self._news_analyzer = await get_news_sentiment_analyzer()
            except ImportError:
                self.logger.warning("News analyzer not available")
                self._news_analyzer = None
            
            self.logger.info("Signal orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def generate_comprehensive_signals(self, 
                                           tickers: List[str],
                                           force_refresh: bool = False) -> List[ComprehensiveSignal]:
        """Generate comprehensive signals for multiple tickers"""
        start_time = datetime.now()
        
        try:
            # Process tickers in batches to manage concurrency
            batch_size = self.config.max_concurrent_analyses
            all_signals = []
            
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self._generate_ticker_signal(ticker, force_refresh)
                    for ticker in batch_tickers
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter successful results
                for result in batch_results:
                    if isinstance(result, ComprehensiveSignal):
                        all_signals.append(result)
                    elif isinstance(result, Exception):
                        self.logger.warning(f"Batch processing error: {result}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Emit performance metrics
            emit_metric('orchestrator.signals_generated', len(all_signals))
            emit_metric('orchestrator.processing_time', processing_time)
            
            self.logger.info(f"Generated {len(all_signals)} comprehensive signals in {processing_time:.2f}s")
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Comprehensive signal generation failed: {e}")
            return []
    
    async def _generate_ticker_signal(self, ticker: str, force_refresh: bool = False) -> Optional[ComprehensiveSignal]:
        """Generate comprehensive signal for a single ticker"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if not force_refresh:
                cached_signal = await self._get_cached_signal(ticker)
                if cached_signal:
                    return cached_signal
            
            # Gather data from all sources concurrently
            tasks = {
                'core_signal': self._get_core_signal(ticker),
                'reddit_data': self._get_reddit_data(ticker),
                'news_data': self._get_news_data(ticker),
                'financial_data': self._get_financial_data(ticker)
            }
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=self.config.analysis_timeout_seconds
            )
            
            # Parse results
            core_signal, reddit_data, news_data, financial_data = results
            
            # Handle any exceptions
            if isinstance(core_signal, Exception):
                self.logger.warning(f"Core signal failed for {ticker}: {core_signal}")
                return None
            
            # Calculate data quality
            data_quality = self._assess_data_quality(reddit_data, news_data, financial_data)
            
            if data_quality < self.config.min_data_quality_threshold:
                self.logger.warning(f"Data quality too low for {ticker}: {data_quality:.2f}")
                return None
            
            # Generate comprehensive signal
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            comprehensive_signal = ComprehensiveSignal(
                ticker=ticker,
                signal_score=core_signal,
                reddit_data=reddit_data or {},
                news_data=news_data or {},
                financial_data=financial_data or {},
                created_at=datetime.now(timezone.utc),
                confidence=self._calculate_comprehensive_confidence(core_signal, reddit_data, news_data, financial_data),
                risk_assessment=self._assess_risk(core_signal, reddit_data, news_data, financial_data),
                processing_time_ms=processing_time,
                data_quality_score=data_quality
            )
            
            # Cache the result
            await self._cache_signal(comprehensive_signal)
            
            return comprehensive_signal
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Analysis timeout for {ticker}")
            return None
        except Exception as e:
            self.logger.error(f"Signal generation failed for {ticker}: {e}")
            return None
    
    async def _get_core_signal(self, ticker: str):
        """Get core signal from signal engine"""
        if not self._signal_engine:
            # Fallback signal generation
            return SignalScore(
                confidence=0.5,
                signal_type=SignalType.HOLD,
                risk_level="medium"
            )
        
        return await self._signal_engine.generate_signal(ticker)
    
    async def _get_reddit_data(self, ticker: str) -> Dict[str, Any]:
        """Get Reddit sentiment data"""
        try:
            if not self._reddit_analyzer:
                return {}
            
            sentiment_result = await self._reddit_analyzer.analyze_ticker(ticker)
            return {
                'sentiment_score': sentiment_result.get('sentiment', 0.0),
                'mention_count': sentiment_result.get('mentions', 0),
                'confidence': sentiment_result.get('confidence', 0.0),
                'top_posts': sentiment_result.get('top_posts', [])
            }
        except Exception as e:
            self.logger.warning(f"Reddit data fetch failed for {ticker}: {e}")
            return {}
    
    async def _get_news_data(self, ticker: str) -> Dict[str, Any]:
        """Get news sentiment data"""
        try:
            if not self._news_analyzer:
                return {}
            
            news_result = await self._news_analyzer.analyze_ticker(ticker)
            return {
                'sentiment_score': news_result.get('sentiment', 0.0),
                'article_count': news_result.get('article_count', 0),
                'confidence': news_result.get('confidence', 0.0),
                'key_headlines': news_result.get('headlines', [])
            }
        except Exception as e:
            self.logger.warning(f"News data fetch failed for {ticker}: {e}")
            return {}
    
    async def _get_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Get financial data"""
        try:
            # Get recent price and volume data
            query = """
            SELECT close_price, volume, market_cap
            FROM stock_prices 
            WHERE ticker = %s 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = await self.db.fetch_one(query, (ticker,))
            
            if result:
                return {
                    'current_price': result['close_price'],
                    'volume': result['volume'],
                    'market_cap': result.get('market_cap', 0)
                }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Financial data fetch failed for {ticker}: {e}")
            return {}
    
    def _assess_data_quality(self, reddit_data: Dict, news_data: Dict, financial_data: Dict) -> float:
        """Assess overall data quality score"""
        quality_score = 0.0
        
        # Reddit data quality
        if reddit_data and reddit_data.get('mention_count', 0) > 0:
            quality_score += 0.3
        
        # News data quality  
        if news_data and news_data.get('article_count', 0) > 0:
            quality_score += 0.4
        
        # Financial data quality
        if financial_data and financial_data.get('current_price', 0) > 0:
            quality_score += 0.3
        
        return quality_score
    
    def _calculate_comprehensive_confidence(self, core_signal, reddit_data: Dict, news_data: Dict, financial_data: Dict) -> float:
        """Calculate comprehensive confidence score"""
        base_confidence = getattr(core_signal, 'confidence', 0.5)
        
        # Adjust based on data availability and sentiment alignment
        reddit_boost = 0.0
        news_boost = 0.0
        
        if reddit_data and 'sentiment_score' in reddit_data:
            reddit_sentiment = reddit_data['sentiment_score']
            reddit_boost = abs(reddit_sentiment) * 0.1  # Max 0.1 boost
        
        if news_data and 'sentiment_score' in news_data:
            news_sentiment = news_data['sentiment_score']  
            news_boost = abs(news_sentiment) * 0.15  # Max 0.15 boost
        
        final_confidence = base_confidence + reddit_boost + news_boost
        return min(1.0, max(0.0, final_confidence))
    
    def _assess_risk(self, core_signal, reddit_data: Dict, news_data: Dict, financial_data: Dict) -> Dict[str, Any]:
        """Assess risk factors"""
        risk_factors = {
            'volatility_risk': 'medium',
            'sentiment_risk': 'low',
            'liquidity_risk': 'low',
            'overall_risk': 'medium'
        }
        
        # Assess sentiment consistency
        sentiments = []
        if reddit_data and 'sentiment_score' in reddit_data:
            sentiments.append(reddit_data['sentiment_score'])
        if news_data and 'sentiment_score' in news_data:
            sentiments.append(news_data['sentiment_score'])
        
        if len(sentiments) > 1:
            sentiment_variance = np.var(sentiments)
            if sentiment_variance > 0.5:
                risk_factors['sentiment_risk'] = 'high'
        
        return risk_factors
    
    async def _get_cached_signal(self, ticker: str) -> Optional[ComprehensiveSignal]:
        """Get cached signal if available"""
        try:
            # Check cache table for recent signals
            cutoff_time = datetime.now() - timedelta(hours=self.config.cache_results_hours)
            
            query = """
            SELECT signal_data
            FROM signal_cache 
            WHERE ticker = %s AND created_at > %s
            ORDER BY created_at DESC
            LIMIT 1
            """
            
            result = await self.db.fetch_one(query, (ticker, cutoff_time))
            
            if result:
                signal_data = json.loads(result['signal_data'])
                # Reconstruct ComprehensiveSignal from cached data
                # This would need proper deserialization logic
                return None  # Placeholder for now
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache lookup failed for {ticker}: {e}")
            return None
    
    async def _cache_signal(self, signal: ComprehensiveSignal):
        """Cache signal for future use"""
        try:
            # Serialize signal to JSON
            signal_data = {
                'ticker': signal.ticker,
                'confidence': signal.confidence,
                'processing_time_ms': signal.processing_time_ms,
                'data_quality_score': signal.data_quality_score,
                # Add other relevant fields
            }
            
            query = """
            INSERT INTO signal_cache (ticker, signal_data, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
            signal_data = %s, created_at = %s
            """
            
            signal_json = json.dumps(signal_data)
            now = datetime.now()
            
            await self.db.execute(query, (signal.ticker, signal_json, now, signal_json, now))
            
        except Exception as e:
            self.logger.warning(f"Signal caching failed for {signal.ticker}: {e}")


# ===== SIGNAL OPTIMIZER COMPONENTS =====

class MarketRegime(Enum):
    """Market regime classifications for dynamic weighting"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MOMENTUM_DRIVEN = "momentum_driven"
    MEAN_REVERTING = "mean_reverting"
    NEWS_DRIVEN = "news_driven"


@dataclass
class EnhancedSignalMetrics:
    """Enhanced signal quality metrics"""
    # Advanced scoring
    momentum_quality: float = 0.0
    sentiment_consistency: float = 0.0
    technical_convergence: float = 0.0
    fundamental_strength: float = 0.0
    
    # Risk metrics
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    correlation_risk: float = 0.0
    
    # Market context
    market_regime: MarketRegime = MarketRegime.LOW_VOLATILITY
    sector_performance: float = 0.0
    
    # Quality indicators
    data_completeness: float = 0.0
    confidence_stability: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for signal optimization"""
    lookback_period_days: int = 30
    min_improvement_threshold: float = 0.05
    max_optimization_iterations: int = 10
    
    # Feature weights by market regime
    regime_weights: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULL_TREND: {'momentum': 0.4, 'sentiment': 0.3, 'technical': 0.2, 'fundamental': 0.1},
        MarketRegime.BEAR_TREND: {'momentum': 0.2, 'sentiment': 0.2, 'technical': 0.3, 'fundamental': 0.3},
        MarketRegime.HIGH_VOLATILITY: {'momentum': 0.3, 'sentiment': 0.1, 'technical': 0.4, 'fundamental': 0.2},
        MarketRegime.LOW_VOLATILITY: {'momentum': 0.35, 'sentiment': 0.35, 'technical': 0.2, 'fundamental': 0.1}
    })


class MarketRegimeDetector:
    """Detects current market regime for dynamic optimization"""
    
    def __init__(self, db: DatabaseInterface):
        self.db = db
        self.logger = logging.getLogger('regime_detector')
    
    async def detect_current_regime(self) -> MarketRegime:
        """Detect current market regime based on recent data"""
        try:
            # Get market data for regime detection
            market_data = await self._get_market_indicators()
            
            if not market_data:
                return MarketRegime.LOW_VOLATILITY
            
            # Analyze volatility
            volatility = market_data.get('volatility', 0.15)
            if volatility > 0.25:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.10:
                return MarketRegime.LOW_VOLATILITY
            
            # Analyze trend
            momentum = market_data.get('momentum', 0.0)
            if momentum > 0.15:
                return MarketRegime.BULL_TREND
            elif momentum < -0.15:
                return MarketRegime.BEAR_TREND
            
            # Default to momentum driven for neutral conditions
            return MarketRegime.MOMENTUM_DRIVEN
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return MarketRegime.LOW_VOLATILITY
    
    async def _get_market_indicators(self) -> Dict[str, float]:
        """Get key market indicators for regime detection"""
        try:
            # Get SPY data as market proxy
            query = """
            SELECT close_price, volume, date
            FROM stock_prices 
            WHERE ticker = 'SPY'
            ORDER BY date DESC 
            LIMIT 30
            """
            
            spy_data = await self.db.fetch_all(query)
            
            if len(spy_data) < 20:
                return {}
            
            prices = [row['close_price'] for row in spy_data]
            
            # Calculate volatility (20-day)
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate momentum (20-day)
            momentum = (prices[0] - prices[19]) / prices[19]
            
            return {
                'volatility': volatility,
                'momentum': momentum
            }
            
        except Exception as e:
            self.logger.error(f"Market indicator calculation failed: {e}")
            return {}


class SignalOptimizer:
    """Advanced signal optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.db = get_supabase_database()
        self.regime_detector = MarketRegimeDetector(self.db)
        self.logger = logging.getLogger('signal_optimizer')
    
    async def optimize_signal_scoring(self, signals: List[Signal]) -> List[Signal]:
        """Optimize signal scoring based on current market conditions"""
        if not signals:
            return signals
        
        try:
            # Detect current market regime
            current_regime = await self.regime_detector.detect_current_regime()
            
            # Get regime-specific weights
            regime_weights = self.config.regime_weights.get(
                current_regime, 
                self.config.regime_weights[MarketRegime.LOW_VOLATILITY]
            )
            
            optimized_signals = []
            
            for signal in signals:
                # Calculate enhanced metrics
                enhanced_metrics = await self._calculate_enhanced_metrics(signal, current_regime)
                
                # Optimize confidence score
                optimized_confidence = await self._optimize_confidence_score(
                    signal, enhanced_metrics, regime_weights
                )
                
                # Create optimized signal
                optimized_signal = Signal(
                    ticker=signal.ticker,
                    signal_type=signal.signal_type,
                    confidence=optimized_confidence,
                    created_at=signal.created_at,
                    features=signal.features,
                    metadata={
                        **signal.metadata,
                        'original_confidence': signal.confidence,
                        'optimization_applied': True,
                        'market_regime': current_regime.value,
                        'enhanced_metrics': asdict(enhanced_metrics)
                    }
                )
                
                optimized_signals.append(optimized_signal)
            
            self.logger.info(f"Optimized {len(optimized_signals)} signals for {current_regime.value} regime")
            return optimized_signals
            
        except Exception as e:
            self.logger.error(f"Signal optimization failed: {e}")
            return signals
    
    async def _calculate_enhanced_metrics(self, signal: Signal, regime: MarketRegime) -> EnhancedSignalMetrics:
        """Calculate enhanced signal quality metrics"""
        metrics = EnhancedSignalMetrics(market_regime=regime)
        
        try:
            # Analyze features for enhanced metrics
            momentum_features = [f for f in signal.features if 'momentum' in f.feature_name]
            sentiment_features = [f for f in signal.features if 'sentiment' in f.feature_name]
            technical_features = [f for f in signal.features if 'technical' in f.feature_name or 'sma' in f.feature_name]
            
            # Calculate momentum quality
            if momentum_features:
                momentum_values = [f.feature_value for f in momentum_features]
                metrics.momentum_quality = np.mean(np.abs(momentum_values))
            
            # Calculate sentiment consistency
            if len(sentiment_features) >= 2:
                sentiment_values = [f.feature_value for f in sentiment_features]
                metrics.sentiment_consistency = 1.0 - np.std(sentiment_values)
            
            # Calculate technical convergence
            if technical_features:
                technical_values = [f.feature_value for f in technical_features]
                metrics.technical_convergence = np.mean(technical_values)
            
            # Calculate data completeness
            expected_feature_count = 7  # Based on our feature engineering
            actual_feature_count = len(signal.features)
            metrics.data_completeness = min(1.0, actual_feature_count / expected_feature_count)
            
            # Get volatility and liquidity scores
            volatility_score = await self._get_volatility_score(signal.ticker)
            liquidity_score = await self._get_liquidity_score(signal.ticker)
            
            metrics.volatility_score = volatility_score
            metrics.liquidity_score = liquidity_score
            
        except Exception as e:
            self.logger.error(f"Enhanced metrics calculation failed for {signal.ticker}: {e}")
        
        return metrics
    
    async def _optimize_confidence_score(self, 
                                       signal: Signal, 
                                       metrics: EnhancedSignalMetrics,
                                       regime_weights: Dict[str, float]) -> float:
        """Optimize confidence score based on enhanced metrics and market regime"""
        base_confidence = signal.confidence
        
        # Calculate component scores
        momentum_score = metrics.momentum_quality * regime_weights.get('momentum', 0.25)
        sentiment_score = metrics.sentiment_consistency * regime_weights.get('sentiment', 0.25) 
        technical_score = metrics.technical_convergence * regime_weights.get('technical', 0.25)
        fundamental_score = metrics.fundamental_strength * regime_weights.get('fundamental', 0.25)
        
        # Weighted component score
        component_score = momentum_score + sentiment_score + technical_score + fundamental_score
        
        # Apply quality adjustments
        data_quality_multiplier = 0.7 + (0.3 * metrics.data_completeness)
        volatility_adjustment = self._get_volatility_adjustment(metrics.volatility_score)
        liquidity_adjustment = self._get_liquidity_adjustment(metrics.liquidity_score)
        
        # Calculate optimized confidence
        optimized_confidence = (
            (base_confidence * 0.4 + component_score * 0.6) *
            data_quality_multiplier *
            volatility_adjustment *
            liquidity_adjustment
        )
        
        # Ensure bounds
        return max(0.0, min(1.0, optimized_confidence))
    
    async def _get_volatility_score(self, ticker: str) -> float:
        """Get volatility score for ticker"""
        try:
            query = """
            SELECT close_price
            FROM stock_prices 
            WHERE ticker = %s
            ORDER BY date DESC 
            LIMIT 20
            """
            
            price_data = await self.db.fetch_all(query, (ticker,))
            
            if len(price_data) < 10:
                return 0.5  # Default moderate volatility
            
            prices = [row['close_price'] for row in price_data]
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            volatility = np.std(returns)
            
            # Normalize to 0-1 scale (higher volatility = lower score)
            normalized_volatility = max(0.0, min(1.0, 1.0 - (volatility * 10)))
            return normalized_volatility
            
        except Exception as e:
            self.logger.error(f"Volatility score calculation failed for {ticker}: {e}")
            return 0.5
    
    async def _get_liquidity_score(self, ticker: str) -> float:
        """Get liquidity score for ticker"""
        try:
            query = """
            SELECT volume, close_price
            FROM stock_prices 
            WHERE ticker = %s
            ORDER BY date DESC 
            LIMIT 10
            """
            
            volume_data = await self.db.fetch_all(query, (ticker,))
            
            if not volume_data:
                return 0.5  # Default moderate liquidity
            
            avg_volume = np.mean([row['volume'] for row in volume_data])
            avg_price = np.mean([row['close_price'] for row in volume_data])
            
            # Calculate dollar volume as liquidity proxy
            dollar_volume = avg_volume * avg_price
            
            # Normalize (this would need market-specific calibration)
            if dollar_volume > 10000000:  # $10M+ daily volume
                return 1.0
            elif dollar_volume > 1000000:  # $1M+ daily volume  
                return 0.8
            elif dollar_volume > 100000:   # $100k+ daily volume
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Liquidity score calculation failed for {ticker}: {e}")
            return 0.5
    
    def _get_volatility_adjustment(self, volatility_score: float) -> float:
        """Get volatility-based confidence adjustment"""
        # Higher volatility = lower confidence adjustment
        return 0.8 + (0.2 * volatility_score)
    
    def _get_liquidity_adjustment(self, liquidity_score: float) -> float:
        """Get liquidity-based confidence adjustment"""
        # Higher liquidity = higher confidence adjustment
        return 0.9 + (0.1 * liquidity_score)


# ===== UNIFIED INTELLIGENCE SYSTEM =====

class IntelligenceHub:
    """Unified analysis intelligence system"""
    
    def __init__(self):
        self.db = get_supabase_database()
        self.config_manager = ConfigManager()
        self.logger = get_logger(__name__)
        
        # Components
        self.analysis_engine = None
        self.orchestrator = None
        self.optimizer = None
        
        # Configuration
        self.analysis_config = None
        self.orchestration_config = None
        self.optimization_config = None
    
    async def initialize(self):
        """Initialize all intelligence components"""
        try:
            # Load configurations
            config = await self.config_manager.get_config()
            
            self.analysis_config = AnalysisConfig(
                min_confidence_threshold=config.get('min_confidence_threshold', 0.6),
                max_signals_per_run=config.get('max_signals_per_run', 50),
                feature_lookback_days=config.get('feature_lookback_days', 30),
                sentiment_weight=config.get('sentiment_weight', 0.25),
                technical_weight=config.get('technical_weight', 0.25),
                volume_weight=config.get('volume_weight', 0.20),
                news_weight=config.get('news_weight', 0.30)
            )
            
            self.orchestration_config = OrchestrationConfig(
                max_concurrent_analyses=config.get('max_concurrent_analyses', 10),
                analysis_timeout_seconds=config.get('analysis_timeout_seconds', 30),
                min_data_quality_threshold=config.get('min_data_quality_threshold', 0.6)
            )
            
            self.optimization_config = OptimizationConfig(
                lookback_period_days=config.get('optimization_lookback_days', 30),
                min_improvement_threshold=config.get('min_improvement_threshold', 0.05)
            )
            
            # Initialize components
            self.analysis_engine = AnalysisEngine(self.db, self.analysis_config)
            await self.analysis_engine.initialize()
            
            self.orchestrator = SignalOrchestrator(self.orchestration_config)
            await self.orchestrator.initialize()
            
            self.optimizer = SignalOptimizer(self.optimization_config)
            
            self.logger.info("Intelligence Hub initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Intelligence Hub initialization failed: {e}")
            raise
    
    async def generate_intelligent_signals(self, 
                                         tickers: List[str],
                                         use_optimization: bool = True,
                                         force_refresh: bool = False) -> List[Signal]:
        """Generate comprehensive, optimized signals using all intelligence components"""
        try:
            # Step 1: Generate base signals with analysis engine
            signal_result = await self.analysis_engine.generate_signals(tickers)
            base_signals = signal_result.signals
            
            if not base_signals:
                self.logger.warning("No base signals generated")
                return []
            
            # Step 2: Generate comprehensive signals with orchestrator
            comprehensive_signals = await self.orchestrator.generate_comprehensive_signals(
                [s.ticker for s in base_signals],
                force_refresh=force_refresh
            )
            
            # Convert comprehensive signals back to base Signal format
            enhanced_signals = []
            comprehensive_map = {cs.ticker: cs for cs in comprehensive_signals}
            
            for base_signal in base_signals:
                comprehensive = comprehensive_map.get(base_signal.ticker)
                if comprehensive:
                    # Enhance base signal with comprehensive data
                    enhanced_signal = Signal(
                        ticker=base_signal.ticker,
                        signal_type=base_signal.signal_type,
                        confidence=comprehensive.confidence,
                        created_at=base_signal.created_at,
                        features=base_signal.features,
                        metadata={
                            **base_signal.metadata,
                            'comprehensive_data': {
                                'reddit_sentiment': comprehensive.reddit_data.get('sentiment_score', 0.0),
                                'news_sentiment': comprehensive.news_data.get('sentiment_score', 0.0),
                                'data_quality': comprehensive.data_quality_score,
                                'processing_time_ms': comprehensive.processing_time_ms
                            }
                        }
                    )
                    enhanced_signals.append(enhanced_signal)
            
            # Step 3: Apply optimization if requested
            if use_optimization and enhanced_signals:
                optimized_signals = await self.optimizer.optimize_signal_scoring(enhanced_signals)
                final_signals = optimized_signals
            else:
                final_signals = enhanced_signals
            
            # Emit metrics
            emit_metric('intelligence_hub.signals_processed', len(final_signals))
            
            self.logger.info(f"Generated {len(final_signals)} intelligent signals")
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Intelligent signal generation failed: {e}")
            return []
    
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis intelligence capabilities"""
        return {
            'components': {
                'analysis_engine': self.analysis_engine is not None,
                'orchestrator': self.orchestrator is not None,
                'optimizer': self.optimizer is not None
            },
            'configuration': {
                'analysis_config': asdict(self.analysis_config) if self.analysis_config else None,
                'orchestration_config': asdict(self.orchestration_config) if self.orchestration_config else None,
                'optimization_config': asdict(self.optimization_config) if self.optimization_config else None
            }
        }


# ===== SCORING VALIDATION SYSTEM =====

@dataclass
class ScoringValidationResult:
    """Results from scoring validation analysis"""
    overall_accuracy: float
    correlation_score_return: float
    prediction_accuracy_by_score_range: Dict[str, float]
    factor_importance_ranking: List[Tuple[str, float]]
    recommendations: List[str]
    sample_size: int
    validation_date: datetime
    model_performance_metrics: Dict[str, float]

class SignalScoringValidator:
    """
    Advanced scoring validation system that analyzes prediction accuracy
    and provides recommendations for scoring algorithm improvements.
    """
    
    def __init__(self, db: Optional[DatabaseInterface] = None):
        self.logger = get_logger(__name__)
        self.db = db or get_supabase_database()
        
        # Validation settings
        self.min_sample_size = 50
        self.confidence_threshold = 0.7
        self.lookback_days = 30
    
    async def validate_scoring_accuracy(self, 
                                      validation_days: int = 30,
                                      min_signals: int = 50) -> ScoringValidationResult:
        """
        Comprehensive scoring validation analysis.
        
        Args:
            validation_days: Days back to analyze
            min_signals: Minimum signals required for validation
            
        Returns:
            Validation results with accuracy metrics and recommendations
        """
        try:
            self.logger.info(f"Starting scoring validation for last {validation_days} days")
            
            # Get signals with performance data
            signals_data = await self._get_signals_with_performance(validation_days)
            
            if len(signals_data) < min_signals:
                self.logger.warning(f"Insufficient signals ({len(signals_data)}) for validation")
                return self._create_insufficient_data_result(len(signals_data))
            
            # Perform validation analysis
            overall_accuracy = self._calculate_overall_accuracy(signals_data)
            correlation_analysis = self._analyze_score_return_correlation(signals_data)
            score_range_analysis = self._analyze_accuracy_by_score_range(signals_data)
            factor_importance = await self._analyze_factor_importance(signals_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_accuracy, correlation_analysis, score_range_analysis, factor_importance
            )
            
            # Calculate model performance metrics
            model_metrics = self._calculate_model_metrics(signals_data)
            
            result = ScoringValidationResult(
                overall_accuracy=overall_accuracy,
                correlation_score_return=correlation_analysis['correlation'],
                prediction_accuracy_by_score_range=score_range_analysis,
                factor_importance_ranking=factor_importance,
                recommendations=recommendations,
                sample_size=len(signals_data),
                validation_date=datetime.now(timezone.utc),
                model_performance_metrics=model_metrics
            )
            
            # Log validation to database
            await self._log_validation_result(result)
            
            self.logger.info(f"Validation complete: {overall_accuracy:.3f} accuracy, {len(recommendations)} recommendations")
            return result
            
        except Exception as e:
            self.logger.error(f"Scoring validation failed: {e}")
            raise
    
    async def _get_signals_with_performance(self, days_back: int) -> List[Dict[str, Any]]:
        """Get signals with actual performance data for validation."""
        try:
            query = """
            SELECT 
                s.id, s.ticker, s.weighted_score, s.reddit_score, s.news_score, 
                s.financial_score, s.trade_type, s.risk_level, s.created_at,
                s.score_components, s.prediction_confidence,
                sph.return_7d, sph.return_30d, sph.max_return, sph.min_return,
                sph.volatility, sph.sharpe_ratio, sph.hit_rate,
                sp.prediction_accuracy, sp.current_return, sp.total_return
            FROM signals s
            LEFT JOIN signal_performance_history sph ON s.id = sph.signal_id
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            WHERE s.created_at >= NOW() - INTERVAL '%s days'
            AND (sph.return_7d IS NOT NULL OR sp.current_return IS NOT NULL)
            ORDER BY s.created_at DESC
            """
            
            result = await self.db.fetch_all(query, days_back)
            return [dict(row) for row in result] if result else []
            
        except Exception as e:
            self.logger.error(f"Failed to fetch signals with performance: {e}")
            return []
    
    def _calculate_overall_accuracy(self, signals_data: List[Dict[str, Any]]) -> float:
        """Calculate overall prediction accuracy."""
        if not signals_data:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for signal in signals_data:
            # Use available return data (7d preferred, fall back to current/total)
            actual_return = (signal.get('return_7d') or 
                           signal.get('current_return') or 
                           signal.get('total_return'))
            
            if actual_return is not None:
                weighted_score = signal.get('weighted_score', 0)
                
                # Consider prediction correct if:
                # High score (>0.7) and positive return, or low score (<0.3) and negative return
                if ((weighted_score > 0.7 and actual_return > 0) or 
                    (weighted_score < 0.3 and actual_return <= 0) or
                    (0.3 <= weighted_score <= 0.7 and abs(actual_return) < 0.05)):  # Neutral range tolerance
                    correct_predictions += 1
                
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _analyze_score_return_correlation(self, signals_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlation between signal scores and actual returns."""
        scores = []
        returns = []
        
        for signal in signals_data:
            weighted_score = signal.get('weighted_score')
            actual_return = (signal.get('return_7d') or 
                           signal.get('current_return') or 
                           signal.get('total_return'))
            
            if weighted_score is not None and actual_return is not None:
                scores.append(weighted_score)
                returns.append(actual_return)
        
        if len(scores) < 10:  # Need minimum samples for correlation
            return {'correlation': 0.0, 'sample_size': len(scores)}
        
        correlation = np.corrcoef(scores, returns)[0, 1] if len(scores) > 1 else 0.0
        
        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'sample_size': len(scores)
        }
    
    def _analyze_accuracy_by_score_range(self, signals_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze prediction accuracy by signal score ranges."""
        ranges = {
            'high_confidence': {'min': 0.8, 'max': 1.0, 'correct': 0, 'total': 0},
            'medium_high': {'min': 0.6, 'max': 0.8, 'correct': 0, 'total': 0},
            'medium': {'min': 0.4, 'max': 0.6, 'correct': 0, 'total': 0},
            'low': {'min': 0.0, 'max': 0.4, 'correct': 0, 'total': 0}
        }
        
        for signal in signals_data:
            weighted_score = signal.get('weighted_score', 0)
            actual_return = (signal.get('return_7d') or 
                           signal.get('current_return') or 
                           signal.get('total_return'))
            
            if actual_return is not None:
                # Determine which range this signal falls into
                for range_name, range_data in ranges.items():
                    if range_data['min'] <= weighted_score < range_data['max']:
                        range_data['total'] += 1
                        
                        # Check if prediction was correct
                        if ((weighted_score > 0.5 and actual_return > 0) or 
                            (weighted_score <= 0.5 and actual_return <= 0)):
                            range_data['correct'] += 1
                        break
        
        # Calculate accuracy for each range
        accuracy_by_range = {}
        for range_name, range_data in ranges.items():
            if range_data['total'] > 0:
                accuracy_by_range[range_name] = range_data['correct'] / range_data['total']
            else:
                accuracy_by_range[range_name] = 0.0
        
        return accuracy_by_range
    
    async def _analyze_factor_importance(self, signals_data: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Analyze importance of different scoring factors."""
        factor_performance = {
            'reddit_score': {'correct': 0, 'total': 0, 'sum_impact': 0},
            'news_score': {'correct': 0, 'total': 0, 'sum_impact': 0},
            'financial_score': {'correct': 0, 'total': 0, 'sum_impact': 0}
        }
        
        for signal in signals_data:
            actual_return = (signal.get('return_7d') or 
                           signal.get('current_return') or 
                           signal.get('total_return'))
            
            if actual_return is not None:
                prediction_correct = actual_return > 0 if signal.get('weighted_score', 0) > 0.5 else actual_return <= 0
                
                for factor in factor_performance.keys():
                    factor_score = signal.get(factor, 0)
                    if factor_score > 0:
                        factor_performance[factor]['total'] += 1
                        factor_performance[factor]['sum_impact'] += abs(actual_return) * factor_score
                        if prediction_correct:
                            factor_performance[factor]['correct'] += 1
        
        # Calculate importance scores
        importance_ranking = []
        for factor, data in factor_performance.items():
            if data['total'] > 0:
                accuracy = data['correct'] / data['total']
                avg_impact = data['sum_impact'] / data['total']
                importance_score = accuracy * avg_impact  # Combined metric
                importance_ranking.append((factor, importance_score))
        
        # Sort by importance score
        importance_ranking.sort(key=lambda x: x[1], reverse=True)
        return importance_ranking
    
    def _generate_recommendations(self, 
                                overall_accuracy: float,
                                correlation_analysis: Dict[str, float],
                                score_range_analysis: Dict[str, float],
                                factor_importance: List[Tuple[str, float]]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Overall accuracy recommendations
        if overall_accuracy < 0.6:
            recommendations.append("CRITICAL: Overall accuracy below 60%. Consider comprehensive model retraining.")
        elif overall_accuracy < 0.7:
            recommendations.append("WARNING: Overall accuracy below 70%. Review scoring algorithm and factor weights.")
        
        # Correlation recommendations
        correlation = correlation_analysis.get('correlation', 0)
        if correlation < 0.3:
            recommendations.append("LOW CORRELATION: Weighted score has weak correlation with returns. Review scoring methodology.")
        elif correlation < 0.5:
            recommendations.append("MODERATE CORRELATION: Consider fine-tuning factor weights to improve predictive power.")
        
        # Score range recommendations
        high_conf_accuracy = score_range_analysis.get('high_confidence', 0)
        if high_conf_accuracy < 0.8:
            recommendations.append("HIGH CONFIDENCE SIGNALS underperforming. Tighten criteria for top-tier signals.")
        
        # Factor importance recommendations
        if factor_importance:
            top_factor = factor_importance[0][0]
            worst_factor = factor_importance[-1][0] if len(factor_importance) > 1 else None
            
            recommendations.append(f"OPTIMIZE: {top_factor.replace('_', ' ').title()} shows highest importance. Consider increasing weight.")
            
            if worst_factor and factor_importance[-1][1] < 0.1:
                recommendations.append(f"REVIEW: {worst_factor.replace('_', ' ').title()} shows low importance. Consider reducing weight or improving data quality.")
        
        # Data quality recommendations
        sample_size = correlation_analysis.get('sample_size', 0)
        if sample_size < 100:
            recommendations.append("INSUFFICIENT DATA: Increase signal volume or extend validation period for more robust analysis.")
        
        return recommendations
    
    def _calculate_model_metrics(self, signals_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate additional model performance metrics."""
        if not signals_data:
            return {}
        
        returns = []
        scores = []
        
        for signal in signals_data:
            actual_return = (signal.get('return_7d') or 
                           signal.get('current_return') or 
                           signal.get('total_return'))
            weighted_score = signal.get('weighted_score')
            
            if actual_return is not None and weighted_score is not None:
                returns.append(actual_return)
                scores.append(weighted_score)
        
        if len(returns) < 10:
            return {'sample_size': len(returns)}
        
        returns_array = np.array(returns)
        scores_array = np.array(scores)
        
        # Calculate metrics
        mean_return = float(np.mean(returns_array))
        volatility = float(np.std(returns_array))
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        # Calculate hit rate (positive returns)
        hit_rate = float(np.mean(returns_array > 0))
        
        # Calculate score-weighted metrics
        high_score_mask = scores_array > 0.7
        if np.any(high_score_mask):
            high_score_return = float(np.mean(returns_array[high_score_mask]))
            high_score_hit_rate = float(np.mean(returns_array[high_score_mask] > 0))
        else:
            high_score_return = 0.0
            high_score_hit_rate = 0.0
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'high_score_return': high_score_return,
            'high_score_hit_rate': high_score_hit_rate,
            'sample_size': len(returns)
        }
    
    def _create_insufficient_data_result(self, sample_size: int) -> ScoringValidationResult:
        """Create validation result for insufficient data case."""
        return ScoringValidationResult(
            overall_accuracy=0.0,
            correlation_score_return=0.0,
            prediction_accuracy_by_score_range={},
            factor_importance_ranking=[],
            recommendations=[f"INSUFFICIENT DATA: Only {sample_size} signals available. Need at least {self.min_sample_size} for validation."],
            sample_size=sample_size,
            validation_date=datetime.now(timezone.utc),
            model_performance_metrics={}
        )
    
    async def _log_validation_result(self, result: ScoringValidationResult):
        """Log validation results to database."""
        try:
            insert_data = {
                'calibration_date': result.validation_date,
                'factors_tested': {
                    'overall_accuracy': result.overall_accuracy,
                    'correlation': result.correlation_score_return,
                    'score_range_accuracy': result.prediction_accuracy_by_score_range,
                    'factor_importance': dict(result.factor_importance_ranking),
                    'model_metrics': result.model_performance_metrics
                },
                'accuracy_improvement': result.overall_accuracy,  # Can be enhanced to track improvement over time
                'validation_sample_size': result.sample_size,
                'notes': '; '.join(result.recommendations)
            }
            
            await self.db.insert('scoring_calibration_log', insert_data)
            self.logger.info("Validation results logged to database")
            
        except Exception as e:
            self.logger.warning(f"Failed to log validation result: {e}")

# Export main classes
__all__ = [
    'AnalysisEngine', 'AnalysisConfig', 'FeatureEngineer', 'MLSignalScorer',
    'SignalOrchestrator', 'OrchestrationConfig', 'ComprehensiveSignal',
    'SignalOptimizer', 'OptimizationConfig', 'MarketRegimeDetector', 'MarketRegime', 'EnhancedSignalMetrics',
    'IntelligenceHub', 'SignalScoringValidator', 'ScoringValidationResult'
]