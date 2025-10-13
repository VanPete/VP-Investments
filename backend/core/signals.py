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
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Import enums and constants from core module
from .core import FeatureType, SignalType, TradeType, RiskLevel

logger = logging.getLogger(__name__)


# ============================================================================
# NORMALIZATION HELPERS (PHASE 2: FUNDAMENTAL REDESIGN)
# ============================================================================

def normalize_direct(value: Optional[float], low: float, high: float) -> Optional[float]:
    """
    Normalize where higher values are better (ROE, Margins, Growth).
    
    Args:
        value: Value to normalize
        low: Minimum threshold (0.0 score)
        high: Maximum threshold (1.0 score)
        
    Returns:
        Score from 0.0 to 1.0 (1.0 = best), or None if value is None
        
    Example:
        normalize_direct(15.0, 0.0, 25.0) = 0.60 (15% ROE normalized)
    """
    if value is None:
        return None
    # Clamp to range
    value = max(min(value, high), low)
    # Normalize to [0, 1]
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def normalize_inverted(value: Optional[float], good_low: float, bad_high: float) -> Optional[float]:
    """
    Normalize where lower values are better (PE, Debt/Equity).
    
    Args:
        value: Value to normalize
        good_low: Best possible value (lowest) = 1.0 score
        bad_high: Worst acceptable value (highest) = 0.0 score
        
    Returns:
        Score from 0.0 to 1.0 (1.0 = best), or None if value is None
        
    Example:
        normalize_inverted(25, 5, 50) = 0.56 (PE of 25 normalized)
        normalize_inverted(0.5, 0, 2) = 0.75 (Debt/Equity of 0.5 normalized)
    """
    if value is None:
        return None
    # Clamp to range
    value = max(min(value, bad_high), good_low)
    # Invert: lower is better
    if bad_high == good_low:
        return 0.5
    return (bad_high - value) / (bad_high - good_low)


def normalize_growth(value: Optional[float], negative_threshold: float = -0.10, 
                     positive_threshold: float = 0.30) -> Optional[float]:
    """
    Normalize growth metrics that can be negative.
    
    Args:
        value: Growth value as decimal (e.g., -0.10 to 0.30 for -10% to +30%)
        negative_threshold: Growth below this = 0.0 score (default -10%)
        positive_threshold: Growth above this = 1.0 score (default +30%)
        
    Returns:
        Score from 0.0 to 1.0, or None if value is None
        
    Example:
        normalize_growth(0.15, -0.10, 0.30) = 0.625 (15% growth normalized)
        normalize_growth(-0.05, -0.10, 0.30) = 0.125 (-5% growth normalized)
    """
    if value is None:
        return None
    # Clamp to range
    if value < negative_threshold:
        return 0.0
    if value > positive_threshold:
        return 1.0
    # Normalize to [0, 1]
    return (value - negative_threshold) / (positive_threshold - negative_threshold)


# ============================================================================
# PHASE 2: Z-SCORE AND ENHANCEMENT INFRASTRUCTURE
# ============================================================================


class ZScoreCalculator:
    """
    Rolling window z-score standardization for regime-aware signal classification.
    
    Uses 60-day rolling window (min 30 days). Falls back to universe statistics
    if ticker history insufficient (<20 samples).
    
    Formula: z = (x - μ) / σ
    where μ and σ are computed from rolling window.
    """
    
    def __init__(self, lookback_days: int = 60, min_samples: int = 30):
        """
        Args:
            lookback_days: Rolling window size (default 60 trading days)
            min_samples: Minimum samples required (default 30)
        """
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.universe_stats: Dict[str, Dict[str, float]] = {}  # Fallback statistics
        
        logger.info(f"ZScoreCalculator initialized: lookback={lookback_days}d, min={min_samples}")
    
    def calculate_z_score(
        self,
        value: float,
        ticker: str,
        feature: str,
        historical_data: Optional[List[Dict]] = None,
        db_manager=None
    ) -> float:
        """
        Calculate z-score using rolling window or universe fallback.
        
        Args:
            value: Current value to standardize
            ticker: Stock ticker
            feature: Feature name (e.g., 'technical_score')
            historical_data: Optional pre-fetched historical data
            db_manager: Database manager for fetching history
        
        Returns:
            Z-score (standardized value)
        """
        try:
            # Get historical data
            if historical_data is None and db_manager:
                historical_data = self._fetch_historical_data(ticker, feature, db_manager)
            
            if not historical_data or len(historical_data) < 2:
                # Not enough history - use universe stats
                return self._calculate_universe_z_score(value, feature)
            
            # Extract values from historical data
            values = [d.get(feature, 0.0) for d in historical_data if d.get(feature) is not None]
            
            if len(values) < self.min_samples:
                # Insufficient samples - use universe fallback
                logger.debug(f"{ticker}/{feature}: Only {len(values)} samples, using universe stats")
                return self._calculate_universe_z_score(value, feature)
            
            # Calculate rolling window stats
            values_array = np.array(values[-self.lookback_days:])  # Take most recent
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            # Prevent division by zero
            if std == 0 or np.isnan(std):
                logger.debug(f"{ticker}/{feature}: Zero std, using universe stats")
                return self._calculate_universe_z_score(value, feature)
            
            # Calculate z-score
            z_score = (value - mean) / std
            
            # Cap extreme values
            z_score = np.clip(z_score, -5.0, 5.0)
            
            return float(z_score)
            
        except Exception as e:
            logger.warning(f"Z-score calculation failed for {ticker}/{feature}: {e}")
            return 0.0
    
    def _fetch_historical_data(
        self,
        ticker: str,
        feature: str,
        db_manager
    ) -> List[Dict]:
        """Fetch historical data from database."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days + 10)
            
            result = db_manager.supabase.table('signals') \
                .select(f'created_at, {feature}') \
                .eq('ticker', ticker) \
                .gte('created_at', cutoff_date.isoformat()) \
                .order('created_at', desc=False) \
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.warning(f"Failed to fetch historical data for {ticker}/{feature}: {e}")
            return []
    
    def _calculate_universe_z_score(self, value: float, feature: str) -> float:
        """
        Calculate z-score using universe statistics (fallback).
        
        Universe stats are pre-computed from recent signals across all tickers.
        """
        if feature not in self.universe_stats:
            # No universe stats available - return 0
            logger.debug(f"No universe stats for {feature}, returning 0")
            return 0.0
        
        stats = self.universe_stats[feature]
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 1.0)
        
        if std == 0:
            return 0.0
        
        z_score = (value - mean) / std
        z_score = np.clip(z_score, -5.0, 5.0)
        
        return float(z_score)
    
    def update_universe_stats(self, all_signals: List[Dict]):
        """
        Update universe statistics from current batch of signals.
        
        Call this at the start of each pipeline run with all recent signals.
        """
        try:
            features = [
                'technical_score', 'fundamental_score', 'news_macro_score',
                'social_alternative_score', 'risk_stability_score',
                'institutional_smart_money_score'
            ]
            
            for feature in features:
                values = [s.get(feature, 0.0) for s in all_signals if s.get(feature) is not None]
                
                if len(values) >= 10:  # Need at least 10 samples
                    self.universe_stats[feature] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
                    
            logger.info(f"Updated universe stats for {len(self.universe_stats)} features")
            
        except Exception as e:
            logger.error(f"Failed to update universe stats: {e}")


class TrendStrengthCalculator:
    """
    Calculate composite trend strength from MA slopes and volume trends.
    
    Formula: TrendStrength = 0.5*z(slope_50) + 0.3*z(slope_200) + 0.2*z(volume_trend)
    
    MA Slope: Annualized OLS slope of log(price) over lookback period
              slope_L = 252 * slope(OLS(log(P_t) ~ t, last L days))
    
    Volume Trend: Z-score of 20-day average volume vs 60-day history
    """
    
    def __init__(self, z_calc: ZScoreCalculator):
        """
        Args:
            z_calc: ZScoreCalculator instance for standardization
        """
        self.z_calc = z_calc
        logger.info("TrendStrengthCalculator initialized")
    
    def calculate_trend_strength(
        self,
        ticker: str,
        price_history: List[float],
        volume_history: List[float],
        db_manager=None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite trend strength.
        
        Args:
            ticker: Stock ticker
            price_history: Historical prices (oldest first)
            volume_history: Historical volumes (oldest first)
            db_manager: Database manager for z-score history
        
        Returns:
            Tuple of (trend_strength, components_dict)
            components_dict contains: slope_50, slope_200, volume_trend, and their z-scores
        """
        try:
            components = {}
            
            # Calculate MA slopes
            if len(price_history) >= 50:
                slope_50 = self._calculate_ma_slope(price_history, lookback=50)
                components['ma_slope_50'] = slope_50
                components['ma_slope_50_z'] = self.z_calc.calculate_z_score(
                    slope_50, ticker, 'ma_slope_50', db_manager=db_manager
                )
            else:
                components['ma_slope_50'] = 0.0
                components['ma_slope_50_z'] = 0.0
            
            if len(price_history) >= 200:
                slope_200 = self._calculate_ma_slope(price_history, lookback=200)
                components['ma_slope_200'] = slope_200
                components['ma_slope_200_z'] = self.z_calc.calculate_z_score(
                    slope_200, ticker, 'ma_slope_200', db_manager=db_manager
                )
            else:
                components['ma_slope_200'] = 0.0
                components['ma_slope_200_z'] = 0.0
            
            # Calculate volume trend
            if len(volume_history) >= 60:
                volume_trend = self._calculate_volume_trend(volume_history)
                components['volume_trend'] = volume_trend
                components['volume_trend_z'] = self.z_calc.calculate_z_score(
                    volume_trend, ticker, 'volume_trend', db_manager=db_manager
                )
            else:
                components['volume_trend'] = 0.0
                components['volume_trend_z'] = 0.0
            
            # Composite trend strength
            trend_strength = (
                0.5 * components['ma_slope_50_z'] +
                0.3 * components['ma_slope_200_z'] +
                0.2 * components['volume_trend_z']
            )
            
            return float(trend_strength), components
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed for {ticker}: {e}")
            return 0.0, {}
    
    def _calculate_ma_slope(self, prices: List[float], lookback: int) -> float:
        """
        Calculate annualized MA slope using OLS regression on log(price).
        
        Formula: slope_L = 252 * slope(OLS(log(P_t) ~ t, last L days))
        
        Args:
            prices: Price history (oldest first)
            lookback: Lookback period in days
        
        Returns:
            Annualized slope
        """
        try:
            if len(prices) < lookback:
                return 0.0
            
            # Take last 'lookback' days
            recent_prices = prices[-lookback:]
            
            # Remove zeros and NaNs
            recent_prices = [p for p in recent_prices if p > 0 and not np.isnan(p)]
            
            if len(recent_prices) < 10:  # Need minimum data
                return 0.0
            
            # Log prices
            log_prices = np.log(recent_prices)
            
            # Time index (0, 1, 2, ...)
            time_index = np.arange(len(log_prices))
            
            # OLS regression
            slope, intercept = np.polyfit(time_index, log_prices, 1)
            
            # Annualize (252 trading days per year)
            annualized_slope = slope * 252
            
            return float(annualized_slope)
            
        except Exception as e:
            logger.warning(f"MA slope calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, volumes: List[int]) -> float:
        """
        Calculate volume trend: 20-day average vs 60-day history.
        
        Args:
            volumes: Volume history (oldest first)
        
        Returns:
            Volume trend value (20-day avg / 60-day avg)
        """
        try:
            if len(volumes) < 60:
                return 1.0  # Neutral
            
            # Last 60 days
            recent_volumes = volumes[-60:]
            
            # 20-day average (most recent)
            avg_20 = np.mean(recent_volumes[-20:])
            
            # 60-day average
            avg_60 = np.mean(recent_volumes)
            
            if avg_60 == 0:
                return 1.0
            
            # Ratio
            volume_trend = avg_20 / avg_60
            
            return float(volume_trend)
            
        except Exception as e:
            logger.warning(f"Volume trend calculation failed: {e}")
            return 1.0


class ValuationCalculator:
    """
    Calculate valuation composite z-score for Value trade type classification.
    
    Formula: valuation_z = mean(z(P/E), z(P/B), z(FCF_yield) * -1)
    
    Note: Higher FCF yield = cheaper, so multiply by -1 before averaging
    """
    
    def __init__(self, z_calc: ZScoreCalculator):
        """
        Args:
            z_calc: ZScoreCalculator instance
        """
        self.z_calc = z_calc
        logger.info("ValuationCalculator initialized")
    
    def calculate_valuation_z(
        self,
        ticker: str,
        pe_ratio: Optional[float],
        pb_ratio: Optional[float],
        fcf_yield: Optional[float],
        db_manager=None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite valuation z-score.
        
        Args:
            ticker: Stock ticker
            pe_ratio: Price-to-Earnings ratio
            pb_ratio: Price-to-Book ratio
            fcf_yield: Free Cash Flow yield (FCF / Market Cap)
            db_manager: Database manager for z-score history
        
        Returns:
            Tuple of (valuation_z, components_dict)
        """
        try:
            components = {}
            z_scores = []
            
            # P/E ratio (lower is better)
            if pe_ratio and pe_ratio > 0:
                pe_z = self.z_calc.calculate_z_score(
                    pe_ratio, ticker, 'pe_ratio', db_manager=db_manager
                )
                components['pe_ratio'] = pe_ratio
                components['pe_z'] = pe_z
                z_scores.append(pe_z)
            
            # P/B ratio (lower is better)
            if pb_ratio and pb_ratio > 0:
                pb_z = self.z_calc.calculate_z_score(
                    pb_ratio, ticker, 'price_to_book', db_manager=db_manager
                )
                components['pb_ratio'] = pb_ratio
                components['pb_z'] = pb_z
                z_scores.append(pb_z)
            
            # FCF yield (higher is better, so multiply by -1)
            if fcf_yield and fcf_yield != 0:
                fcf_z = self.z_calc.calculate_z_score(
                    fcf_yield, ticker, 'fcf_yield', db_manager=db_manager
                )
                components['fcf_yield'] = fcf_yield
                components['fcf_yield_z'] = fcf_z
                z_scores.append(fcf_z * -1)  # Invert: higher yield = lower z-score (cheaper)
            
            # Average z-scores
            if z_scores:
                valuation_z = float(np.mean(z_scores))
            else:
                valuation_z = 0.0
            
            components['valuation_z'] = valuation_z
            
            return valuation_z, components
            
        except Exception as e:
            logger.error(f"Valuation calculation failed for {ticker}: {e}")
            return 0.0, {}


class TradeTypeClassifier:
    """
    Trade type classification engine using z-score based thresholds.
    
    Classifies signals into one of 6 trade types (max 2 assigned):
    - Momentum: Strong technical + trend strength
    - Value: Undervalued fundamentals
    - Speculative Growth: High growth, negative cash flow
    - Event-Driven: Near earnings or significant news
    - Contrarian: Oversold with improving fundamentals
    - Multi-Factor: Strong across 3+ components
    
    Returns primary + optional secondary type (max 2 total).
    """
    
    # Event detection keywords
    EVENT_KEYWORDS = {
        'ma': ['merger', 'acquisition', 'takeover', 'go-private', 'buyout', 'acquiring', 'acquired'],
        'contract': ['contract', 'awarded', 'wins deal', 'option exercised', 'IDIQ', 'government contract'],
        'product': ['launches', 'unveils', 'announces', 'FDA approval', 'FDA clearance', 'product release', 'new product']
    }
    
    # Theme ticker mappings (will be expanded from config)
    THEME_TICKERS = {
        'AI': ['NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 'TSLA', 'PLTR', 'C3AI', 'AI', 'BBAI'],
        'Biotech': ['MRNA', 'BNTX', 'GILD', 'REGN', 'VRTX', 'BIIB', 'AMGN', 'ILMN'],
        'Defense': ['LMT', 'RTX', 'BA', 'NOC', 'GD', 'TXT', 'HII', 'LDOS'],
        'Green Energy': ['TSLA', 'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'BE', 'CHPT'],
        'Crypto': ['COIN', 'MSTR', 'MARA', 'RIOT', 'SI', 'HUT', 'BITF']
    }
    
    def __init__(self, z_calc: ZScoreCalculator, trend_calc: TrendStrengthCalculator, val_calc: ValuationCalculator):
        """
        Args:
            z_calc: ZScoreCalculator instance
            trend_calc: TrendStrengthCalculator instance
            val_calc: ValuationCalculator instance
        """
        self.z_calc = z_calc
        self.trend_calc = trend_calc
        self.val_calc = val_calc
        logger.info("TradeTypeClassifier initialized")
    
    def classify_trade_type(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        component_scores: Dict[str, float],
        db_manager=None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify trade type based on z-scores and thresholds.
        
        Args:
            ticker: Stock ticker
            signal_data: Raw signal data (financials, prices, etc.)
            component_scores: Dict with technical_score, fundamental_score, news_score, social_score
            db_manager: Database manager for historical data
        
        Returns:
            Tuple of (trade_tags list, classification_details dict)
            trade_tags: ['Primary Type', 'Secondary Type'] or ['Primary Type'] or ['Multi-Factor']
            classification_details: {
                'primary_type': str,
                'secondary_type': str or None,
                'multi_factor': bool,
                'scores': {...},
                'event_flags': {...},
                'theme': str or None
            }
        """
        try:
            # Calculate z-scores for component scores
            technical_z = self.z_calc.calculate_z_score(
                component_scores.get('technical_score', 0.0),
                ticker, 'technical_score', db_manager=db_manager
            )
            
            fundamental_z = self.z_calc.calculate_z_score(
                component_scores.get('fundamental_score', 0.0),
                ticker, 'fundamental_score', db_manager=db_manager
            )
            
            news_z = self.z_calc.calculate_z_score(
                component_scores.get('news_score', 0.0),
                ticker, 'news_score', db_manager=db_manager
            )
            
            social_z = self.z_calc.calculate_z_score(
                component_scores.get('social_score', 0.0),
                ticker, 'social_score', db_manager=db_manager
            )
            
            # Calculate trend strength (if price history available)
            trend_strength = 0.0
            ma_slope_50 = None
            ma_slope_200 = None
            volume_trend_z = None
            
            if 'price_history' in signal_data and 'volume_history' in signal_data:
                trend_strength, trend_components = self.trend_calc.calculate_trend_strength(
                    ticker,
                    signal_data['price_history'],
                    signal_data['volume_history'],
                    db_manager=db_manager
                )
                ma_slope_50 = trend_components.get('ma_slope_50')
                ma_slope_200 = trend_components.get('ma_slope_200')
                volume_trend_z = trend_components.get('volume_trend_z')
            
            # Calculate valuation z-score (if financials available)
            valuation_z = 0.0
            if any(k in signal_data for k in ['pe_ratio', 'price_to_book', 'fcf_yield']):
                valuation_z, _ = self.val_calc.calculate_valuation_z(
                    ticker,
                    signal_data.get('pe_ratio'),
                    signal_data.get('price_to_book'),
                    signal_data.get('fcf_yield'),
                    db_manager=db_manager
                )
            
            # Check for oversold conditions
            rsi = signal_data.get('rsi')
            price_z_20day = signal_data.get('price_z_20day', 0.0)
            is_oversold = (rsi and rsi <= 30) or (price_z_20day <= -2.0)
            
            # Detect events
            event_flags = self._detect_events(ticker, signal_data)
            
            # Detect theme
            theme = self._detect_theme(ticker, signal_data)
            
            # Calculate fundamental quality (if available)
            fundamental_quality_z = 0.0
            if 'roe' in signal_data and 'profit_margins' in signal_data:
                roe = signal_data.get('roe', 0.0)
                margins = signal_data.get('profit_margins', 0.0)
                quality_score = (roe + margins) / 2.0
                fundamental_quality_z = self.z_calc.calculate_z_score(
                    quality_score, ticker, 'fundamental_quality', db_manager=db_manager
                )
            
            # Calculate revenue growth z-score
            revenue_growth_z = 0.0
            if 'revenue_growth' in signal_data:
                revenue_growth_z = self.z_calc.calculate_z_score(
                    signal_data['revenue_growth'],
                    ticker, 'revenue_growth', db_manager=db_manager
                )
            
            # Calculate fundamentals trend
            fundamentals_trend_z = 0.0
            if 'revenue_growth' in signal_data and 'earnings_growth' in signal_data:
                trend_value = (signal_data['revenue_growth'] + signal_data['earnings_growth']) / 2.0
                fundamentals_trend_z = self.z_calc.calculate_z_score(
                    trend_value, ticker, 'fundamentals_trend', db_manager=db_manager
                )
            
            # Store all calculated values
            scores = {
                'technical_z': technical_z,
                'fundamental_z': fundamental_z,
                'news_z': news_z,
                'social_z': social_z,
                'trend_strength': trend_strength,
                'valuation_z': valuation_z,
                'fundamental_quality_z': fundamental_quality_z,
                'revenue_growth_z': revenue_growth_z,
                'fundamentals_trend_z': fundamentals_trend_z,
                'ma_slope_50': ma_slope_50,
                'ma_slope_200': ma_slope_200,
                'volume_trend_z': volume_trend_z,
                'price_z_20day': price_z_20day
            }
            
            # Classification logic with priority
            candidates = []
            
            # 1. Event-Driven (highest priority if criteria met)
            if event_flags['has_earnings'] or (event_flags['has_ma'] or event_flags['has_contract'] or event_flags['has_product']):
                if news_z >= 0.7 or event_flags['has_earnings']:
                    candidates.append(('Event-Driven', 10.0))  # High priority score
            
            # 2. Momentum
            if technical_z >= 0.8 and trend_strength >= 0.6:
                momentum_score = technical_z + trend_strength
                candidates.append(('Momentum', momentum_score))
            
            # 3. Value
            if valuation_z <= -0.6 and fundamental_quality_z >= 0.3:
                value_score = abs(valuation_z) + fundamental_quality_z
                candidates.append(('Value', value_score))
            
            # 4. Speculative Growth
            has_negative_fcf = signal_data.get('free_cash_flow', 0) < 0
            if revenue_growth_z >= 0.8 and has_negative_fcf:
                growth_score = revenue_growth_z
                candidates.append(('Speculative Growth', growth_score))
            
            # 5. Contrarian
            if is_oversold and social_z <= -0.5 and fundamentals_trend_z >= 0.2:
                contrarian_score = abs(social_z) + fundamentals_trend_z
                candidates.append(('Contrarian', contrarian_score))
            
            # Sort candidates by score (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Assign primary and secondary types
            trade_tags = []
            primary_type = None
            secondary_type = None
            
            if candidates:
                primary_type = candidates[0][0]
                trade_tags.append(primary_type)
                
                # Add secondary type if significantly different and strong enough
                if len(candidates) > 1 and candidates[1][1] >= 0.5:
                    # Don't add same type twice or conflicting types
                    if candidates[1][0] != primary_type:
                        secondary_type = candidates[1][0]
                        trade_tags.append(secondary_type)
            
            # Check for Multi-Factor (3+ components with z >= 0.5)
            strong_components = sum(1 for z in [technical_z, fundamental_z, news_z, social_z] if z >= 0.5)
            is_multi_factor = strong_components >= 3
            
            if is_multi_factor and 'Multi-Factor' not in trade_tags:
                trade_tags.append('Multi-Factor')
            
            # Default to Balanced if no classification
            if not trade_tags:
                trade_tags = ['Balanced']
                primary_type = 'Balanced'
            
            # Build classification details
            classification_details = {
                'primary_type': primary_type,
                'secondary_type': secondary_type,
                'multi_factor': is_multi_factor,
                'scores': scores,
                'event_flags': event_flags,
                'theme': theme,
                'is_oversold': is_oversold,
                'candidates': [{'type': c[0], 'score': c[1]} for c in candidates]
            }
            
            logger.info(f"Trade classification for {ticker}: {trade_tags} (primary={primary_type}, multi_factor={is_multi_factor})")
            
            return trade_tags, classification_details
            
        except Exception as e:
            logger.error(f"Trade type classification failed for {ticker}: {e}")
            return ['Balanced'], {'error': str(e)}
    
    def _detect_events(self, ticker: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect event flags (earnings, M&A, contracts, products).
        
        Returns:
            {
                'has_earnings': bool,
                'earnings_days_away': int or None,
                'has_ma': bool,
                'has_contract': bool,
                'has_product': bool,
                'keywords': list of detected keywords
            }
        """
        event_flags = {
            'has_earnings': False,
            'earnings_days_away': None,
            'has_ma': False,
            'has_contract': False,
            'has_product': False,
            'keywords': []
        }
        
        # Check earnings date
        earnings_date = signal_data.get('earnings_date')
        if earnings_date:
            # Calculate days away (assuming earnings_date is datetime or days_away is provided)
            if isinstance(earnings_date, int):
                days_away = earnings_date
            elif isinstance(earnings_date, (datetime, str)):
                # Parse and calculate
                try:
                    if isinstance(earnings_date, str):
                        earnings_dt = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
                    else:
                        earnings_dt = earnings_date
                    
                    days_away = (earnings_dt - datetime.now(timezone.utc)).days
                except:
                    days_away = None
            else:
                days_away = None
            
            if days_away is not None and abs(days_away) <= 7:
                event_flags['has_earnings'] = True
                event_flags['earnings_days_away'] = days_away
        
        # Check for keywords in news/social content
        content = ""
        if 'news_content' in signal_data:
            content += " " + str(signal_data['news_content'])
        if 'social_content' in signal_data:
            content += " " + str(signal_data['social_content'])
        
        content_lower = content.lower()
        
        # M&A keywords
        for keyword in self.EVENT_KEYWORDS['ma']:
            if keyword.lower() in content_lower:
                event_flags['has_ma'] = True
                event_flags['keywords'].append(keyword)
        
        # Contract keywords
        for keyword in self.EVENT_KEYWORDS['contract']:
            if keyword.lower() in content_lower:
                event_flags['has_contract'] = True
                event_flags['keywords'].append(keyword)
        
        # Product keywords
        for keyword in self.EVENT_KEYWORDS['product']:
            if keyword.lower() in content_lower:
                event_flags['has_product'] = True
                event_flags['keywords'].append(keyword)
        
        return event_flags
    
    def _detect_theme(self, ticker: str, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect investment theme from ticker mapping and keyword analysis.
        
        Returns:
            Theme name (e.g., 'AI', 'Biotech') or None
        """
        # Check ticker mapping first
        for theme, tickers in self.THEME_TICKERS.items():
            if ticker in tickers:
                return theme
        
        # TODO: Add keyword-based theme detection from news/social content
        # This would require more sophisticated NLP or keyword lists
        
        return None


class RiskScoreCalculator:
    """
    Comprehensive risk scoring engine with 5 subscores and worst-factor guard.
    
    Risk Score (0-100):
    - Composite: weighted average of 5 subscores
    - Worst-factor guard: max(composite, 0.9 * max_subfactor)
    
    Subscores:
    1. Volatility (40%): ATR%, beta
    2. Liquidity (25%): Average daily volume, float%
    3. Leverage (15%): D/E ratio, interest coverage
    4. Short Interest (10%): % of float
    5. Concentration (10%): Market cap tier, theme risk
    
    Risk Levels:
    - Low: <25
    - Moderate: 25-45
    - Elevated: 45-65
    - High: 65-80
    - Extreme: >80
    """
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'Low': (0, 25),
        'Moderate': (25, 45),
        'Elevated': (45, 65),
        'High': (65, 80),
        'Extreme': (80, 100)
    }
    
    # Market cap tiers (in billions)
    MARKET_CAP_TIERS = {
        'Mega': 200,      # >$200B
        'Large': 10,      # $10B-$200B
        'Mid': 2,         # $2B-$10B
        'Small': 0.3,     # $300M-$2B
        'Micro': 0.05,    # $50M-$300M
        'Nano': 0         # <$50M
    }
    
    # Theme risk multipliers
    THEME_RISK = {
        'Crypto': 1.3,
        'Biotech': 1.2,
        'Speculative Growth': 1.15,
        'Green Energy': 1.1,
        'AI': 1.05,
        'Defense': 0.95,
        'Utilities': 0.9,
        None: 1.0  # No theme
    }
    
    def __init__(self):
        """Initialize risk score calculator."""
        logger.info("RiskScoreCalculator initialized")
    
    def calculate_risk_score(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        theme: Optional[str] = None
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Calculate comprehensive risk score with subscores.
        
        Args:
            ticker: Stock ticker
            signal_data: Dict with financial/technical data:
                - atr_pct: Average True Range as % of price (20-day)
                - beta: Beta vs market
                - avg_volume: Average daily volume
                - float_pct: Float as % of shares outstanding
                - debt_to_equity: D/E ratio
                - interest_coverage: EBIT / Interest Expense
                - short_interest: Short interest as % of float
                - market_cap: Market capitalization
                - price: Current price
                - earnings_date: Days to earnings (optional)
            theme: Investment theme (optional)
        
        Returns:
            Tuple of (risk_score, risk_level, risk_factors)
        """
        try:
            # Calculate subscores
            volatility_subscore, volatility_details = self._calculate_volatility_score(
                signal_data.get('atr_pct'),
                signal_data.get('beta')
            )
            
            liquidity_subscore, liquidity_details = self._calculate_liquidity_score(
                signal_data.get('avg_volume'),
                signal_data.get('float_pct'),
                signal_data.get('price')
            )
            
            leverage_subscore, leverage_details = self._calculate_leverage_score(
                signal_data.get('debt_to_equity'),
                signal_data.get('interest_coverage')
            )
            
            short_subscore, short_details = self._calculate_short_interest_score(
                signal_data.get('short_interest')
            )
            
            concentration_subscore, concentration_details = self._calculate_concentration_score(
                signal_data.get('market_cap'),
                theme
            )
            
            # Weighted composite
            weights = {
                'volatility': 0.40,
                'liquidity': 0.25,
                'leverage': 0.15,
                'short_interest': 0.10,
                'concentration': 0.10
            }
            
            composite = (
                volatility_subscore * weights['volatility'] +
                liquidity_subscore * weights['liquidity'] +
                leverage_subscore * weights['leverage'] +
                short_subscore * weights['short_interest'] +
                concentration_subscore * weights['concentration']
            )
            
            # Worst-factor guard: ensure risk isn't understated
            max_subscore = max(
                volatility_subscore,
                liquidity_subscore,
                leverage_subscore,
                short_subscore,
                concentration_subscore
            )
            
            risk_score = max(composite, 0.9 * max_subscore)
            risk_score = min(100.0, max(0.0, risk_score))  # Clamp to 0-100
            
            # Determine risk level
            risk_level = self._get_risk_level(risk_score)
            
            # Check for special flags
            has_inverse_beta = signal_data.get('beta', 0) < 0
            has_event_week = False
            earnings_date = signal_data.get('earnings_date')
            if earnings_date and isinstance(earnings_date, (int, float)):
                has_event_week = abs(earnings_date) <= 7
            
            # Build risk_factors JSON
            risk_factors = {
                'volatility': {
                    'score': round(volatility_subscore, 1),
                    'label': self._get_risk_level(volatility_subscore),
                    **volatility_details
                },
                'liquidity': {
                    'score': round(liquidity_subscore, 1),
                    'label': self._get_risk_level(liquidity_subscore),
                    **liquidity_details
                },
                'leverage': {
                    'score': round(leverage_subscore, 1),
                    'label': self._get_risk_level(leverage_subscore),
                    **leverage_details
                },
                'short_interest': {
                    'score': round(short_subscore, 1),
                    'label': self._get_risk_level(short_subscore),
                    **short_details
                },
                'concentration': {
                    'score': round(concentration_subscore, 1),
                    'label': self._get_risk_level(concentration_subscore),
                    **concentration_details
                },
                'composite': {
                    'score': round(composite, 1),
                    'max_subscore': round(max_subscore, 1),
                    'guard_applied': risk_score > composite
                },
                'flags': {
                    'inverse_beta': has_inverse_beta,
                    'event_week': has_event_week
                }
            }
            
            logger.info(f"Risk score for {ticker}: {risk_score:.1f} ({risk_level})")
            
            return risk_score, risk_level, risk_factors
            
        except Exception as e:
            logger.error(f"Risk score calculation failed for {ticker}: {e}")
            return 50.0, 'Moderate', {'error': str(e)}
    
    def _calculate_volatility_score(
        self,
        atr_pct: Optional[float],
        beta: Optional[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate volatility risk subscore (0-100).
        
        ATR% Thresholds:
        - <1.5%: Low (0-20)
        - 1.5-3%: Moderate (20-40)
        - 3-5%: Elevated (40-60)
        - 5-8%: High (60-80)
        - >8%: Extreme (80-100)
        
        Beta Thresholds (use absolute value):
        - <0.8: Low (0-20)
        - 0.8-1.2: Moderate (20-40)
        - 1.2-1.5: Elevated (40-60)
        - 1.5-2.0: High (60-80)
        - >2.0: Extreme (80-100)
        
        Final: Average of ATR% and |beta| scores
        """
        scores = []
        details = {}
        
        # ATR% score
        if atr_pct is not None:
            details['atr_pct'] = round(atr_pct, 2)
            if atr_pct < 1.5:
                atr_score = 10 + (atr_pct / 1.5) * 10
            elif atr_pct < 3.0:
                atr_score = 20 + ((atr_pct - 1.5) / 1.5) * 20
            elif atr_pct < 5.0:
                atr_score = 40 + ((atr_pct - 3.0) / 2.0) * 20
            elif atr_pct < 8.0:
                atr_score = 60 + ((atr_pct - 5.0) / 3.0) * 20
            else:
                atr_score = 80 + min((atr_pct - 8.0) / 4.0, 1.0) * 20
            scores.append(atr_score)
        
        # Beta score (use absolute value for negative beta)
        if beta is not None:
            beta_abs = abs(beta)
            details['beta'] = round(beta, 2)
            details['beta_abs'] = round(beta_abs, 2)
            
            if beta_abs < 0.8:
                beta_score = 10 + (beta_abs / 0.8) * 10
            elif beta_abs < 1.2:
                beta_score = 20 + ((beta_abs - 0.8) / 0.4) * 20
            elif beta_abs < 1.5:
                beta_score = 40 + ((beta_abs - 1.2) / 0.3) * 20
            elif beta_abs < 2.0:
                beta_score = 60 + ((beta_abs - 1.5) / 0.5) * 20
            else:
                beta_score = 80 + min((beta_abs - 2.0) / 1.0, 1.0) * 20
            scores.append(beta_score)
        
        # Average or default
        if scores:
            volatility_score = float(np.mean(scores))
        else:
            volatility_score = 50.0  # Default moderate if no data
        
        return volatility_score, details
    
    def _calculate_liquidity_score(
        self,
        avg_volume: Optional[float],
        float_pct: Optional[float],
        price: Optional[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate liquidity risk subscore (0-100).
        
        Average Daily Volume (ADV in dollars):
        - >$50M: Low (0-20)
        - $10M-$50M: Moderate (20-40)
        - $2M-$10M: Elevated (40-60)
        - $500K-$2M: High (60-80)
        - <$500K: Extreme (80-100)
        
        Float %:
        - >70%: Low (0-20)
        - 50-70%: Moderate (20-40)
        - 30-50%: Elevated (40-60)
        - 15-30%: High (60-80)
        - <15%: Extreme (80-100)
        
        Final: Average of ADV and float% scores
        """
        scores = []
        details = {}
        
        # ADV score (in dollars)
        if avg_volume is not None and price is not None:
            adv_dollars = avg_volume * price
            details['avg_volume'] = int(avg_volume)
            details['adv_dollars'] = round(adv_dollars, 0)
            
            if adv_dollars > 50_000_000:
                adv_score = 10
            elif adv_dollars > 10_000_000:
                adv_score = 20 + (1 - (adv_dollars - 10_000_000) / 40_000_000) * 20
            elif adv_dollars > 2_000_000:
                adv_score = 40 + (1 - (adv_dollars - 2_000_000) / 8_000_000) * 20
            elif adv_dollars > 500_000:
                adv_score = 60 + (1 - (adv_dollars - 500_000) / 1_500_000) * 20
            else:
                adv_score = 80 + (1 - adv_dollars / 500_000) * 20
            scores.append(adv_score)
        
        # Float % score
        if float_pct is not None:
            details['float_pct'] = round(float_pct, 1)
            
            if float_pct > 70:
                float_score = 10
            elif float_pct > 50:
                float_score = 20 + (1 - (float_pct - 50) / 20) * 20
            elif float_pct > 30:
                float_score = 40 + (1 - (float_pct - 30) / 20) * 20
            elif float_pct > 15:
                float_score = 60 + (1 - (float_pct - 15) / 15) * 20
            else:
                float_score = 80 + (1 - float_pct / 15) * 20
            scores.append(float_score)
        
        # Average or default
        if scores:
            liquidity_score = float(np.mean(scores))
        else:
            liquidity_score = 50.0  # Default moderate
        
        return liquidity_score, details
    
    def _calculate_leverage_score(
        self,
        debt_to_equity: Optional[float],
        interest_coverage: Optional[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate leverage risk subscore (0-100).
        
        Debt/Equity:
        - <0.3: Low (0-20)
        - 0.3-0.8: Moderate (20-40)
        - 0.8-1.5: Elevated (40-60)
        - 1.5-3.0: High (60-80)
        - >3.0: Extreme (80-100)
        
        Interest Coverage:
        - >4.0x: Low (0-20)
        - 2.0-4.0x: Moderate (20-40)
        - 1.5-2.0x: Elevated (40-60)
        - 1.0-1.5x: High (60-80)
        - <1.0x: Extreme (80-100)
        
        Final: Average of D/E and coverage scores (if both), else single score
        """
        scores = []
        details = {}
        
        # D/E score
        if debt_to_equity is not None:
            details['debt_to_equity'] = round(debt_to_equity, 2)
            
            if debt_to_equity < 0.3:
                de_score = 10
            elif debt_to_equity < 0.8:
                de_score = 20 + ((debt_to_equity - 0.3) / 0.5) * 20
            elif debt_to_equity < 1.5:
                de_score = 40 + ((debt_to_equity - 0.8) / 0.7) * 20
            elif debt_to_equity < 3.0:
                de_score = 60 + ((debt_to_equity - 1.5) / 1.5) * 20
            else:
                de_score = 80 + min((debt_to_equity - 3.0) / 2.0, 1.0) * 20
            scores.append(de_score)
        
        # Interest coverage score (inverted: lower is riskier)
        if interest_coverage is not None:
            details['interest_coverage'] = round(interest_coverage, 2)
            
            if interest_coverage > 4.0:
                cov_score = 10
            elif interest_coverage > 2.0:
                cov_score = 20 + (1 - (interest_coverage - 2.0) / 2.0) * 20
            elif interest_coverage > 1.5:
                cov_score = 40 + (1 - (interest_coverage - 1.5) / 0.5) * 20
            elif interest_coverage > 1.0:
                cov_score = 60 + (1 - (interest_coverage - 1.0) / 0.5) * 20
            else:
                cov_score = 80 + (1 - max(interest_coverage, 0) / 1.0) * 20
            scores.append(cov_score)
        
        # Average or default
        if scores:
            leverage_score = float(np.mean(scores))
        else:
            leverage_score = 50.0  # Default moderate
        
        return leverage_score, details
    
    def _calculate_short_interest_score(
        self,
        short_interest: Optional[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate short interest risk subscore (0-100).
        
        Short Interest % of Float:
        - <5%: Low (0-20)
        - 5-10%: Moderate (20-40)
        - 10-20%: Elevated (40-60)
        - 20-30%: High (60-80)
        - >30%: Extreme (80-100)
        """
        details = {}
        
        if short_interest is not None:
            details['short_pct_float'] = round(short_interest, 2)
            
            if short_interest < 5:
                short_score = 10 + (short_interest / 5) * 10
            elif short_interest < 10:
                short_score = 20 + ((short_interest - 5) / 5) * 20
            elif short_interest < 20:
                short_score = 40 + ((short_interest - 10) / 10) * 20
            elif short_interest < 30:
                short_score = 60 + ((short_interest - 20) / 10) * 20
            else:
                short_score = 80 + min((short_interest - 30) / 20, 1.0) * 20
        else:
            short_score = 30.0  # Default low-moderate if no data
        
        return short_score, details
    
    def _calculate_concentration_score(
        self,
        market_cap: Optional[float],
        theme: Optional[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate concentration risk subscore (0-100).
        
        Market Cap Tier:
        - Mega (>$200B): Low (10)
        - Large ($10B-$200B): Moderate (25)
        - Mid ($2B-$10B): Elevated (40)
        - Small ($300M-$2B): High (60)
        - Micro ($50M-$300M): Extreme (80)
        - Nano (<$50M): Extreme (95)
        
        Theme Risk Multiplier:
        - Crypto: 1.3x
        - Biotech: 1.2x
        - Speculative Growth: 1.15x
        - Green Energy: 1.1x
        - AI: 1.05x
        - Defense: 0.95x
        - Utilities: 0.9x
        - None: 1.0x
        
        Final: base_score * theme_multiplier
        """
        details = {}
        
        # Determine market cap tier
        if market_cap is not None:
            details['market_cap'] = round(market_cap, 0)
            market_cap_b = market_cap / 1_000_000_000  # Convert to billions
            
            if market_cap_b >= self.MARKET_CAP_TIERS['Mega']:
                tier = 'Mega'
                base_score = 10
            elif market_cap_b >= self.MARKET_CAP_TIERS['Large']:
                tier = 'Large'
                base_score = 25
            elif market_cap_b >= self.MARKET_CAP_TIERS['Mid']:
                tier = 'Mid'
                base_score = 40
            elif market_cap_b >= self.MARKET_CAP_TIERS['Small']:
                tier = 'Small'
                base_score = 60
            elif market_cap_b >= self.MARKET_CAP_TIERS['Micro']:
                tier = 'Micro'
                base_score = 80
            else:
                tier = 'Nano'
                base_score = 95
            
            details['market_cap_tier'] = tier
        else:
            base_score = 50.0  # Default moderate
            details['market_cap_tier'] = 'Unknown'
        
        # Apply theme multiplier
        theme_multiplier = self.THEME_RISK.get(theme, 1.0)
        details['theme'] = theme
        details['theme_multiplier'] = theme_multiplier
        
        concentration_score = base_score * theme_multiplier
        concentration_score = min(100.0, concentration_score)  # Cap at 100
        
        return concentration_score, details
    
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Get risk level label from risk score.
        
        Args:
            risk_score: Risk score (0-100)
        
        Returns:
            Risk level: 'Low', 'Moderate', 'Elevated', 'High', or 'Extreme'
        """
        for level, (min_score, max_score) in self.RISK_THRESHOLDS.items():
            if min_score <= risk_score < max_score:
                return level
        
        # Fallback for edge case (100)
        return 'Extreme'
    
    def generate_risk_narrative(
        self,
        risk_score: float,
        risk_level: str,
        risk_factors: Dict[str, Any],
        theme: Optional[str] = None
    ) -> str:
        """
        Phase 6: Generate human-readable risk assessment from structured risk factors.
        
        Args:
            risk_score: Overall risk score (0-100)
            risk_level: Risk level label (Low/Moderate/Elevated/High/Extreme)
            risk_factors: Dict with subscores and worst factor
            theme: Optional market theme for context
        
        Returns:
            Narrative risk assessment string
        
        Example Output:
            "MODERATE RISK (52.0/100): Primary concern is liquidity (78.5), 
            indicating potential exit challenges. Concentration risk is elevated (55.0). 
            Volatility is manageable (45.2). Suitable for medium-risk tolerance portfolios."
        """
        # Extract subscores
        volatility = risk_factors.get('volatility_subscore', 50.0)
        liquidity = risk_factors.get('liquidity_subscore', 50.0)
        leverage = risk_factors.get('leverage_subscore', 50.0)
        short_interest = risk_factors.get('short_interest_subscore', 50.0)
        concentration = risk_factors.get('concentration_subscore', 50.0)
        worst_factor = risk_factors.get('worst_factor', 'unknown')
        max_subscore = risk_factors.get('max_subscore', 50.0)
        
        # Start with risk level and score
        narrative = f"{risk_level.upper()} RISK ({risk_score:.1f}/100): "
        
        # Identify primary concern (worst factor)
        concern_descriptions = {
            'volatility': f"volatility ({volatility:.1f}), indicating high price fluctuation",
            'liquidity': f"liquidity ({liquidity:.1f}), indicating potential exit challenges",
            'leverage': f"leverage ({leverage:.1f}), indicating high debt burden",
            'short_interest': f"short interest ({short_interest:.1f}), indicating bearish sentiment",
            'concentration': f"concentration ({concentration:.1f}), suggesting sector/asset over-exposure"
        }
        
        if worst_factor in concern_descriptions:
            narrative += f"Primary concern is {concern_descriptions[worst_factor]}. "
        
        # Add secondary concerns (scores > 60)
        secondary_concerns = []
        if volatility > 60 and worst_factor != 'volatility':
            secondary_concerns.append(f"Volatility is elevated ({volatility:.1f})")
        if liquidity > 60 and worst_factor != 'liquidity':
            secondary_concerns.append(f"Liquidity risk is high ({liquidity:.1f})")
        if leverage > 60 and worst_factor != 'leverage':
            secondary_concerns.append(f"Leverage is concerning ({leverage:.1f})")
        if short_interest > 60 and worst_factor != 'short_interest':
            secondary_concerns.append(f"Short interest is elevated ({short_interest:.1f})")
        if concentration > 60 and worst_factor != 'concentration':
            secondary_concerns.append(f"Concentration risk is high ({concentration:.1f})")
        
        if secondary_concerns:
            narrative += ". ".join(secondary_concerns) + ". "
        
        # Add positive notes (scores < 40)
        positive_notes = []
        if volatility < 40:
            positive_notes.append(f"Volatility is manageable ({volatility:.1f})")
        if liquidity < 40:
            positive_notes.append(f"Liquidity is adequate ({liquidity:.1f})")
        if leverage < 40:
            positive_notes.append(f"Leverage is reasonable ({leverage:.1f})")
        if short_interest < 40:
            positive_notes.append(f"Short interest is low ({short_interest:.1f})")
        if concentration < 40:
            positive_notes.append(f"Concentration is well-diversified ({concentration:.1f})")
        
        if positive_notes:
            narrative += ". ".join(positive_notes) + ". "
        
        # Add theme context if available
        if theme and theme != "Unknown":
            narrative += f"Aligns with {theme} theme. "
        
        # Add suitability recommendation based on risk level
        suitability = {
            'Low': "Suitable for conservative portfolios",
            'Moderate': "Suitable for medium-risk tolerance portfolios",
            'Elevated': "Requires above-average risk tolerance",
            'High': "Suitable for aggressive portfolios only",
            'Extreme': "Extreme risk - only for high-risk speculators"
        }
        
        narrative += suitability.get(risk_level, "Risk tolerance assessment required")
        narrative += "."
        
        return narrative
    
    async def generate_risk_narrative_ai(
        self,
        risk_score: float,
        risk_level: str,
        risk_factors: Dict[str, Any],
        theme: Optional[str] = None,
        ticker: Optional[str] = None,
        use_ai: bool = True
    ) -> str:
        """
        Phase 7: AI-enhanced risk narrative generation using OpenAI.
        
        Falls back to template-based narrative if AI unavailable or disabled.
        
        Args:
            risk_score: Overall risk score (0-100)
            risk_level: Risk level label (Low/Moderate/Elevated/High/Extreme)
            risk_factors: Dict with subscores and worst factor
            theme: Optional market theme for context
            ticker: Optional ticker symbol for context
            use_ai: If True, attempt AI generation; if False, use template
        
        Returns:
            AI-generated or template-based risk assessment narrative
        """
        # Fall back to template if AI disabled
        if not use_ai:
            return self.generate_risk_narrative(risk_score, risk_level, risk_factors, theme)
        
        try:
            # Try to import OpenAI directly
            from openai import AsyncOpenAI
            import os
            
            # Check if OpenAI API key available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.debug("OpenAI API key not found, using template-based narrative")
                return self.generate_risk_narrative(risk_score, risk_level, risk_factors, theme)
            
            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=api_key)
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            
            # Generate AI narrative
            prompt = f"""You are a financial risk analyst. Generate a concise, professional risk assessment narrative (150-200 words) based on the following data:

RISK PROFILE:
- Overall Risk Score: {risk_score:.1f}/100
- Risk Level: {risk_level}
- Ticker: {ticker or 'N/A'}
- Market Theme: {theme or 'N/A'}

RISK FACTORS (0-100 scale):
- Volatility: {risk_factors.get('volatility_subscore', 50):.1f}
- Liquidity: {risk_factors.get('liquidity_subscore', 50):.1f}
- Leverage: {risk_factors.get('leverage_subscore', 50):.1f}
- Short Interest: {risk_factors.get('short_interest_subscore', 50):.1f}
- Concentration: {risk_factors.get('concentration_subscore', 50):.1f}
- Primary Concern: {risk_factors.get('worst_factor', 'unknown')}

REQUIREMENTS:
1. Start with risk level and score: "{risk_level.upper()} RISK ({risk_score:.1f}/100):"
2. Identify and explain the primary risk concern with specific numbers
3. Mention 2-3 secondary concerns if their subscores are > 60
4. Note any positive factors if subscores < 40
5. Include theme context if relevant
6. End with investor suitability recommendation based on risk level:
   - Low: "Suitable for conservative portfolios"
   - Moderate: "Suitable for medium-risk tolerance portfolios"
   - Elevated: "Requires above-average risk tolerance"
   - High: "Suitable for aggressive portfolios only"
   - Extreme: "Extreme risk - only for high-risk speculators"

Write in clear, professional language. Be specific with numbers. Keep sentences concise."""
            
            # Call OpenAI API
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial risk analyst providing concise, actionable risk assessments."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            ai_narrative = response.choices[0].message.content.strip()
            
            if ai_narrative and len(ai_narrative) > 50:
                logger.info(f"Generated AI risk narrative for {ticker or 'ticker'} ({len(ai_narrative)} chars)")
                return ai_narrative
            else:
                # AI returned invalid response, use template
                logger.warning(f"AI narrative invalid for {ticker or 'ticker'}, using template")
                return self.generate_risk_narrative(risk_score, risk_level, risk_factors, theme)
        
        except ImportError:
            logger.debug("AI module not available, using template-based narrative")
            return self.generate_risk_narrative(risk_score, risk_level, risk_factors, theme)
        except Exception as e:
            logger.warning(f"AI narrative generation failed: {e}, using template")
            return self.generate_risk_narrative(risk_score, risk_level, risk_factors, theme)
    
    def _build_risk_context(
        self,
        risk_score: float,
        risk_level: str,
        risk_factors: Dict[str, Any],
        theme: Optional[str],
        ticker: Optional[str]
    ) -> str:
        """Build structured context for AI narrative generation"""
        context_parts = [
            f"Risk Score: {risk_score:.1f}/100",
            f"Risk Level: {risk_level}",
        ]
        
        if ticker:
            context_parts.append(f"Ticker: {ticker}")
        
        if theme:
            context_parts.append(f"Theme: {theme}")
        
        # Add subscores
        subscores = {
            'Volatility': risk_factors.get('volatility_subscore', 50),
            'Liquidity': risk_factors.get('liquidity_subscore', 50),
            'Leverage': risk_factors.get('leverage_subscore', 50),
            'Short Interest': risk_factors.get('short_interest_subscore', 50),
            'Concentration': risk_factors.get('concentration_subscore', 50)
        }
        
        context_parts.append("Subscores:")
        for name, score in subscores.items():
            context_parts.append(f"  {name}: {score:.1f}")
        
        context_parts.append(f"Worst Factor: {risk_factors.get('worst_factor', 'unknown')}")
        
        return "\n".join(context_parts)


# ============================================================================
# END PHASE 2 INFRASTRUCTURE
# ============================================================================


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
    signal_score: float  # Phase 7: renamed from weighted_score
    trade_type: str
    risk_level: str
    reddit_score: float
    news_score: float
    financial_score: float
    top_factors: List[str]
    signal_type: str
    confidence: float
    
    # Phase 5: Enhanced trade/risk fields
    trade_tags: Optional[List[str]] = None
    risk_score: Optional[float] = None
    risk_factors: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[str] = None  # Phase 6: Human-readable risk narrative
    theme: Optional[str] = None
    event_flags: Optional[Dict[str, Any]] = None
    
    # Phase 5: Z-scores
    technical_z: Optional[float] = None
    fundamental_z: Optional[float] = None
    news_z: Optional[float] = None
    social_z: Optional[float] = None
    trend_strength_z: Optional[float] = None
    valuation_z: Optional[float] = None
    
    # Phase 5: Historical metrics
    ma_slope_50: Optional[float] = None
    ma_slope_200: Optional[float] = None
    volume_trend_z: Optional[float] = None
    price_z_20day: Optional[float] = None
    
    # Phase 5: Risk metrics
    atr_pct: Optional[float] = None
    float_pct: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalResult to dictionary for database storage"""
        import json
        
        result = {
            'ticker': self.ticker,
            'signal_score': self.signal_score,
            'trade_type': self.trade_type,
            'risk_level': self.risk_level,
            'reddit_score': self.reddit_score,
            'news_score': self.news_score,
            'financial_score': self.financial_score,
            'top_factors': self.top_factors,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
        }
        
        # Phase 5: Add enhanced fields if present
        if self.trade_tags is not None:
            result['trade_tags'] = self.trade_tags
        if self.risk_score is not None:
            result['risk_score'] = self.risk_score
        if self.risk_factors is not None:
            result['risk_factors'] = self.risk_factors
        if self.risk_assessment is not None:
            result['risk_assessment'] = self.risk_assessment
        if self.theme is not None:
            result['theme'] = self.theme
        if self.event_flags is not None:
            result['event_flags'] = self.event_flags
        if self.technical_z is not None:
            result['technical_z'] = self.technical_z
        if self.fundamental_z is not None:
            result['fundamental_z'] = self.fundamental_z
        if self.news_z is not None:
            result['news_z'] = self.news_z
        if self.social_z is not None:
            result['social_z'] = self.social_z
        if self.trend_strength_z is not None:
            result['trend_strength_z'] = self.trend_strength_z
        if self.valuation_z is not None:
            result['valuation_z'] = self.valuation_z
        if self.ma_slope_50 is not None:
            result['ma_slope_50'] = self.ma_slope_50
        if self.ma_slope_200 is not None:
            result['ma_slope_200'] = self.ma_slope_200
        if self.volume_trend_z is not None:
            result['volume_trend_z'] = self.volume_trend_z
        if self.price_z_20day is not None:
            result['price_z_20day'] = self.price_z_20day
        if self.atr_pct is not None:
            result['atr_pct'] = self.atr_pct
        if self.float_pct is not None:
            result['float_pct'] = self.float_pct
        if self.interest_coverage is not None:
            result['interest_coverage'] = self.interest_coverage
        
        return result


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
    """Comprehensive signal score with detailed metrics (Phase 7)"""
    ticker: str = ""
    signal_score: float = 0.0  # Phase 7: renamed from weighted_score
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
    
    def __init__(self, profile: str = "ml_optimized", db_manager=None):
        self.profile = profile
        self.weights = self._load_signal_weights(profile)
        self.thresholds = self._load_thresholds()
        self.trade_type_profiles = self._load_trade_type_profiles()
        
        # Current profile and statistics
        self.current_profile: Optional[ScoringProfile] = None
        self.normalization_stats: Dict[str, Dict[str, float]] = {}
        self.feature_stats = defaultdict(list)
        self.batch_metrics = {}
        
        # Phase 2-4: Initialize calculators for enhanced risk/trade scoring
        self.z_calc = ZScoreCalculator(lookback_days=60, min_samples=30)
        self.trend_calc = TrendStrengthCalculator(self.z_calc)
        self.val_calc = ValuationCalculator(self.z_calc)
        self.trade_classifier = TradeTypeClassifier(
            self.z_calc, self.trend_calc, self.val_calc
        )
        self.risk_calc = RiskScoreCalculator()
        
        # Phase 4: Data cache to prevent re-fetching same ticker
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Database manager for historical data
        self.db_manager = db_manager
        
    def _load_signal_weights(self, profile: str) -> Dict[str, float]:
        """
        Phase 7: Load signal weights for 6-group scoring system.
        
        Returns category-level weights (not individual signal weights).
        Individual signal weights are handled within each scoring function.
        """
        profiles = {
            "ml_optimized": {
                'technical': 0.25,
                'fundamental': 0.25,
                'news_macro': 0.20,
                'social_alternative': 0.15,
                'risk_stability': 0.10,
                'institutional_smart_money': 0.05
            },
            "conservative": {
                'technical': 0.15,
                'fundamental': 0.35,
                'news_macro': 0.20,
                'social_alternative': 0.05,
                'risk_stability': 0.20,
                'institutional_smart_money': 0.05
            },
            "aggressive": {
                'technical': 0.35,
                'fundamental': 0.15,
                'news_macro': 0.15,
                'social_alternative': 0.25,
                'risk_stability': 0.05,
                'institutional_smart_money': 0.05
            },
            "value": {
                'technical': 0.15,
                'fundamental': 0.40,
                'news_macro': 0.15,
                'social_alternative': 0.05,
                'risk_stability': 0.15,
                'institutional_smart_money': 0.10
            },
            "news_driven": {
                'technical': 0.20,
                'fundamental': 0.20,
                'news_macro': 0.35,
                'social_alternative': 0.10,
                'risk_stability': 0.10,
                'institutional_smart_money': 0.05
            },
            "smart_money": {
                'technical': 0.20,
                'fundamental': 0.25,
                'news_macro': 0.15,
                'social_alternative': 0.05,
                'risk_stability': 0.15,
                'institutional_smart_money': 0.20
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
    
    def clear_cache(self):
        """Clear the data cache (call at start of each batch)"""
        self.data_cache = {}
        logger.debug("Cleared SignalScorer data cache")
    
    def _get_enhanced_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get enhanced risk/trade data with caching.
        Returns cached data if available, otherwise fetches and caches.
        """
        from backend.integrations.yfinance import fetch_enhanced_risk_data
        
        # Check cache first
        if ticker in self.data_cache:
            logger.debug(f"Using cached enhanced data for {ticker}")
            return self.data_cache[ticker]
        
        # Fetch if not cached
        logger.debug(f"Fetching enhanced data for {ticker} (single fetch)")
        data = fetch_enhanced_risk_data(ticker)
        self.data_cache[ticker] = data
        return data
    
    async def score_ticker(self, ticker_data: Dict) -> SignalResult:
        """
        Score a single ticker using Phase 7 comprehensive 6-group scoring system.
        
        Phase 5 Enhancement:
        - Fetches enhanced risk/trade data (single fetch per ticker)
        - Uses TradeTypeClassifier for advanced classification
        - Uses RiskScoreCalculator for detailed risk scoring
        - Applies dynamic weight adjustments by trade type
        - Adds contrarian bonus for oversold + negative sentiment
        
        Groups:
        1. Technical (25%): Price momentum, volume, RSI, MACD, Bollinger
        2. Fundamental (25%): P/E, PEG, revenue growth, margins, analyst ratings
        3. News/Macro (20%): News sentiment, earnings events, market regime
        4. Social/Alternative (15%): Reddit sentiment/mentions/upvotes
        5. Risk/Stability (15%): Beta, volatility, Sharpe ratio, liquidity
        6. Institutional/Smart Money (5%): Institutional ownership, insider activity
        """
        
        try:
            ticker = ticker_data.get('ticker', 'UNKNOWN')
            
            # Phase 5: Fetch enhanced risk/trade data (with caching)
            enhanced_data = self._get_enhanced_data(ticker)
            
            # Handle fetch errors gracefully
            if 'error' in enhanced_data:
                logger.warning(f"Enhanced data fetch failed for {ticker}: {enhanced_data.get('error')}")
                # Fall back to basic scoring
                return self._get_default_score(ticker)
            
            # Calculate 6 component scores (Phase 7)
            component_scores = {
                'technical': self._calculate_technical_score(ticker_data),
                'fundamental': self._calculate_fundamental_score(ticker_data),
                'news_macro': self._calculate_news_macro_score(ticker_data),
                'social_alternative': self._calculate_social_alternative_score(ticker_data),
                'risk_stability': self._calculate_risk_stability_score(ticker_data),
                'institutional_smart_money': self._calculate_institutional_smart_money_score(ticker_data)
            }
            
            # Phase 5: Advanced trade classification
            trade_tags, classification_details = self.trade_classifier.classify_trade_type(
                ticker, enhanced_data, component_scores, self.db_manager
            )
            
            # Phase 5: Advanced risk scoring
            risk_score, risk_level, risk_factors = self.risk_calc.calculate_risk_score(
                ticker, enhanced_data, classification_details.get('theme')
            )
            
            # Phase 6/7: Generate AI-enhanced risk narrative from structured risk factors
            risk_assessment = await self.risk_calc.generate_risk_narrative_ai(
                risk_score, 
                risk_level, 
                risk_factors, 
                classification_details.get('theme'),
                ticker,
                use_ai=True  # Set to False to disable AI and use template-based
            )
            
            # Phase 5: Dynamic weight adjustment by trade type
            adjusted_weights = self._adjust_weights_by_trade_type(trade_tags)
            
            # Calculate final signal score with adjusted weights (Phase 7)
            signal_score = self._calculate_signal_score_v2_adjusted(
                ticker_data, component_scores, adjusted_weights
            )
            
            # Phase 5: Contrarian bonus
            contrarian_bonus = self._calculate_contrarian_bonus(
                trade_tags, classification_details
            )
            signal_score += contrarian_bonus
            
            # Clamp to [0, 1]
            signal_score = max(0.0, min(signal_score, 1.0))
            
            # Classifications (use Phase 5 results)
            trade_type = ', '.join(trade_tags) if trade_tags else "Balanced"
            signal_type = self._determine_signal_type(ticker_data)
            
            # Analysis
            top_factors = self._identify_top_factors_v2(ticker_data, component_scores)
            confidence = self._calculate_confidence_v2(ticker_data, component_scores)
            
            return SignalResult(
                ticker=ticker,
                signal_score=round(signal_score, 4),
                trade_type=trade_type,
                risk_level=risk_level,
                reddit_score=round(component_scores['social_alternative'], 3),
                news_score=round(component_scores['news_macro'], 3),
                financial_score=round(component_scores['fundamental'], 3),
                top_factors=top_factors,
                signal_type=signal_type,
                confidence=round(confidence, 3),
                # Phase 5: Enhanced fields
                trade_tags=trade_tags,
                risk_score=round(risk_score, 2) if risk_score else None,
                risk_factors=risk_factors,
                risk_assessment=risk_assessment,  # Phase 6: Human-readable narrative
                theme=classification_details.get('theme'),
                event_flags=classification_details.get('event_flags'),
                technical_z=enhanced_data.get('technical_z'),
                fundamental_z=enhanced_data.get('fundamental_z'),
                news_z=enhanced_data.get('news_z'),
                social_z=enhanced_data.get('social_z'),
                trend_strength_z=enhanced_data.get('trend_strength_z'),
                valuation_z=enhanced_data.get('valuation_z'),
                ma_slope_50=enhanced_data.get('ma_slope_50'),
                ma_slope_200=enhanced_data.get('ma_slope_200'),
                volume_trend_z=enhanced_data.get('volume_trend_z'),
                price_z_20day=enhanced_data.get('price_z_20day'),
                atr_pct=enhanced_data.get('atr_pct'),
                float_pct=enhanced_data.get('float_pct'),
                interest_coverage=enhanced_data.get('interest_coverage')
            )
            
        except Exception as e:
            logger.error(f"Error scoring ticker {ticker_data.get('ticker', 'UNKNOWN')}: {e}")
            return SignalResult(
                ticker=ticker_data.get('ticker', 'UNKNOWN'),
                signal_score=0.0,  # Phase 7
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
    
    def _calculate_financial_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate comprehensive financial score using ALL available indicators.
        Moved from pipeline.py for consolidation (Phase 6c-Part3).
        
        **PHASE 2 ENHANCED:** Now uses 30+ indicators with optimized weighting
        
        Formula: Technical (40%) + Fundamentals (30%) + Options (15%) + Short Interest (15%)
        
        Technical Score (11 components):
        - Momentum indicators (18%): 1d, 7d, 30d price changes
        - RSI (12%): Overbought/oversold signals
        - Moving averages (12%): 50d, 200d MA position
        - MACD (10%): Trend direction and strength
        - Volume analysis (12%): Spike ratio, correlation
        - Volatility (10%): Level, rank, Bollinger bands
        - Relative strength (10%): vs SPY and sector
        - Beta (8%): Market correlation
        - Momentum consistency (7%): Phase 1.4 metric
        - Liquidity (6%): Phase 1.4 metric
        - Exit signals (5%): Inverted exit strength
        
        Fundamentals Score (10 components):
        - Market cap (12%): Size category scoring
        - Valuation (18%): P/E (8%), PEG (5%), P/S (5%)
        - Profitability (20%): Profit margin (8%), Op margin (6%), ROE (6%)
        - Growth (15%): Revenue (8%), Earnings (7%)
        - Financial health (15%): Debt/equity (8%), Current ratio (4%), Quick ratio (3%)
        - Cash flow (10%): Free cash flow yield
        - Ownership (10%): Institutional (5%), Retail (5%)
        
        Options Score: Put/call ratio sentiment
        Short Interest Score: Short squeeze potential (3 metrics)
        
        Returns:
            float: Composite score [0.0-1.0] with normalization for missing data
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
            return 0.0
    
    def _calculate_fundamentals_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate fundamentals score from financial metrics.
        Moved from pipeline.py for consolidation (Phase 6c).
        
        ENHANCED Phase 2: Now uses ALL 16+ fundamental metrics with optimized weights.
        ENHANCED Phase 3: Added analyst data, earnings momentum, institutional activity, insider sentiment (20 metrics total).
        
        Scoring Components:
        1. Market cap (11%) - Company size category
        2. Valuation (16%) - P/E, PEG, P/S ratios  
        3. Profitability (18%) - Margins and ROE
        4. Growth (13%) - Revenue and earnings growth
        5. Financial health (14%) - Debt ratios and liquidity
        6. Cash flow (10%) - FCF yield
        7. Ownership (8%) - Institutional and retail holdings
        8. Analyst consensus (5%) - Target upside and recommendations
        9. Earnings momentum (4%) - Surprise trends
        10. Institutional activity (3%) - QoQ changes
        11. Insider sentiment (3%) - Insider transactions
        
        Returns:
            float: Normalized score [0.0-1.0] with dynamic weight adjustment
        """
        try:
            fundamental_components = []
            weights_used = []
            
            # 1. MARKET CAP (11%)
            market_cap = financial_data.get('market_cap_numeric', 0)
            if market_cap and market_cap > 0:
                if market_cap > 50_000_000_000:
                    cap_score = 0.7
                elif market_cap > 10_000_000_000:
                    cap_score = 0.9
                elif market_cap > 2_000_000_000:
                    cap_score = 1.0
                elif market_cap > 500_000_000:
                    cap_score = 0.8
                else:
                    cap_score = 0.5
                fundamental_components.append(cap_score * 0.11)
                weights_used.append(0.11)
            
            # 2. VALUATION METRICS (16%)
            pe_ratio = financial_data.get('pe_ratio')
            if pe_ratio and not np.isnan(pe_ratio) and pe_ratio > 0:
                if 10 < pe_ratio < 25:
                    pe_score = 1.0
                elif 5 < pe_ratio <= 10:
                    pe_score = 0.8
                elif 25 <= pe_ratio < 40:
                    pe_score = 0.7
                else:
                    pe_score = 0.5
                fundamental_components.append(pe_score * 0.07)
                weights_used.append(0.07)
            
            peg_ratio = financial_data.get('peg_ratio')
            if peg_ratio and not np.isnan(peg_ratio) and peg_ratio > 0:
                if peg_ratio < 1.0:
                    peg_score = 1.0
                elif peg_ratio < 1.5:
                    peg_score = 0.8
                elif peg_ratio < 2.0:
                    peg_score = 0.6
                else:
                    peg_score = 0.4
                fundamental_components.append(peg_score * 0.05)
                weights_used.append(0.05)
            
            price_to_sales = financial_data.get('price_to_sales')
            if price_to_sales and not np.isnan(price_to_sales) and price_to_sales > 0:
                if price_to_sales < 2:
                    ps_score = 1.0
                elif price_to_sales < 4:
                    ps_score = 0.7
                else:
                    ps_score = 0.5
                fundamental_components.append(ps_score * 0.04)
                weights_used.append(0.04)
            
            # 3. PROFITABILITY METRICS (18%)
            profit_margin = financial_data.get('profit_margin')
            if profit_margin and not np.isnan(profit_margin):
                if profit_margin > 0.20:
                    profit_score = 1.0
                elif profit_margin > 0.10:
                    profit_score = 0.8
                elif profit_margin > 0.05:
                    profit_score = 0.6
                elif profit_margin > 0:
                    profit_score = 0.4
                else:
                    profit_score = 0.2
                fundamental_components.append(profit_score * 0.07)
                weights_used.append(0.07)
            
            operating_margin = financial_data.get('operating_margin')
            if operating_margin and not np.isnan(operating_margin):
                if operating_margin > 0.20:
                    op_score = 1.0
                elif operating_margin > 0.10:
                    op_score = 0.7
                elif operating_margin > 0:
                    op_score = 0.5
                else:
                    op_score = 0.2
                fundamental_components.append(op_score * 0.05)
                weights_used.append(0.05)
            
            roe = financial_data.get('roe')
            if roe and not np.isnan(roe):
                if roe > 0.15:
                    roe_score = 1.0
                elif roe > 0.10:
                    roe_score = 0.7
                elif roe > 0:
                    roe_score = 0.5
                else:
                    roe_score = 0.2
                fundamental_components.append(roe_score * 0.05)
                weights_used.append(0.05)
            
            # 4. GROWTH METRICS (13%)
            revenue_growth = financial_data.get('revenue_growth')
            if revenue_growth and not np.isnan(revenue_growth):
                if revenue_growth > 0.20:
                    rev_score = 1.0
                elif revenue_growth > 0.10:
                    rev_score = 0.8
                elif revenue_growth > 0:
                    rev_score = 0.6
                else:
                    rev_score = 0.3
                fundamental_components.append(rev_score * 0.07)
                weights_used.append(0.07)
            
            earnings_growth = financial_data.get('earnings_growth')
            if earnings_growth and not np.isnan(earnings_growth):
                if earnings_growth > 0.20:
                    earn_score = 1.0
                elif earnings_growth > 0.10:
                    earn_score = 0.8
                elif earnings_growth > 0:
                    earn_score = 0.6
                else:
                    earn_score = 0.3
                fundamental_components.append(earn_score * 0.06)
                weights_used.append(0.06)
            
            # 5. FINANCIAL HEALTH (14%)
            debt_to_equity = financial_data.get('debt_to_equity')
            if debt_to_equity and not np.isnan(debt_to_equity):
                if debt_to_equity < 0.3:
                    debt_score = 1.0
                elif debt_to_equity < 0.6:
                    debt_score = 0.8
                elif debt_to_equity < 1.0:
                    debt_score = 0.6
                else:
                    debt_score = 0.3
                fundamental_components.append(debt_score * 0.07)
                weights_used.append(0.07)
            
            current_ratio = financial_data.get('current_ratio')
            if current_ratio and not np.isnan(current_ratio):
                if current_ratio >= 2.0:
                    curr_score = 1.0
                elif current_ratio >= 1.5:
                    curr_score = 0.8
                elif current_ratio >= 1.0:
                    curr_score = 0.6
                else:
                    curr_score = 0.3
                fundamental_components.append(curr_score * 0.03)
                weights_used.append(0.03)
            
            quick_ratio = financial_data.get('quick_ratio')
            if quick_ratio and not np.isnan(quick_ratio):
                if quick_ratio >= 1.5:
                    quick_score = 1.0
                elif quick_ratio >= 1.0:
                    quick_score = 0.7
                elif quick_ratio >= 0.5:
                    quick_score = 0.5
                else:
                    quick_score = 0.3
                fundamental_components.append(quick_score * 0.03)
                weights_used.append(0.03)
            
            # 6. CASH FLOW (10%)
            free_cash_flow = financial_data.get('free_cash_flow')
            if free_cash_flow and market_cap and not np.isnan(free_cash_flow) and market_cap > 0:
                fcf_yield = free_cash_flow / market_cap
                if fcf_yield > 0.08:
                    fcf_score = 1.0
                elif fcf_yield > 0.04:
                    fcf_score = 0.8
                elif fcf_yield > 0:
                    fcf_score = 0.6
                else:
                    fcf_score = 0.3
                fundamental_components.append(fcf_score * 0.10)
                weights_used.append(0.10)
            
            # 7. OWNERSHIP METRICS (8%)
            institutional_pct = financial_data.get('institutional_ownership_pct')
            if institutional_pct and not np.isnan(institutional_pct):
                if 40 <= institutional_pct <= 70:
                    inst_score = 1.0
                elif 30 <= institutional_pct < 40 or 70 < institutional_pct <= 85:
                    inst_score = 0.7
                else:
                    inst_score = 0.5
                fundamental_components.append(inst_score * 0.04)
                weights_used.append(0.04)
            
            retail_pct = financial_data.get('retail_holding_pct')
            if retail_pct and not np.isnan(retail_pct):
                if retail_pct > 20:
                    retail_score = 1.0
                elif retail_pct > 10:
                    retail_score = 0.7
                else:
                    retail_score = 0.5
                fundamental_components.append(retail_score * 0.04)
                weights_used.append(0.04)
            
            # 8. ANALYST CONSENSUS (5%)
            target_upside_pct = financial_data.get('target_upside_pct')
            recommendation_mean = financial_data.get('recommendation_mean')
            if target_upside_pct is not None and not np.isnan(target_upside_pct):
                if target_upside_pct > 20:
                    analyst_score = 1.0
                elif target_upside_pct > 10:
                    analyst_score = 0.7
                elif target_upside_pct > 5:
                    analyst_score = 0.5
                elif target_upside_pct > 0:
                    analyst_score = 0.3
                else:
                    analyst_score = 0.0
                if recommendation_mean is not None and not np.isnan(recommendation_mean):
                    if recommendation_mean <= 2.0:
                        analyst_score = min(analyst_score + 0.2, 1.0)
                    elif recommendation_mean >= 3.5:
                        analyst_score = max(analyst_score - 0.2, 0.0)
                fundamental_components.append(analyst_score * 0.05)
                weights_used.append(0.05)
            
            # 9. EARNINGS MOMENTUM (4%)
            avg_surprise = financial_data.get('avg_earnings_surprise_pct')
            surprise_trend = financial_data.get('earnings_surprise_trend')
            if avg_surprise is not None and not np.isnan(avg_surprise):
                if avg_surprise > 10:
                    earnings_score = 1.0
                elif avg_surprise > 5:
                    earnings_score = 0.7
                elif avg_surprise > 0:
                    earnings_score = 0.5
                elif avg_surprise > -5:
                    earnings_score = 0.3
                else:
                    earnings_score = 0.0
                if surprise_trend == 'Improving':
                    earnings_score = min(earnings_score + 0.2, 1.0)
                elif surprise_trend == 'Declining':
                    earnings_score = max(earnings_score - 0.2, 0.0)
                fundamental_components.append(earnings_score * 0.04)
                weights_used.append(0.04)
            
            # 10. INSTITUTIONAL ACTIVITY (3%)
            inst_change_qoq = financial_data.get('institutional_change_qoq')
            top_10_holders_pct = financial_data.get('top_10_holders_pct')
            if inst_change_qoq is not None and not np.isnan(inst_change_qoq):
                if inst_change_qoq > 5:
                    inst_activity_score = 1.0
                elif inst_change_qoq > 2:
                    inst_activity_score = 0.7
                elif inst_change_qoq > 0:
                    inst_activity_score = 0.5
                elif inst_change_qoq > -2:
                    inst_activity_score = 0.3
                else:
                    inst_activity_score = 0.0
                if top_10_holders_pct is not None and not np.isnan(top_10_holders_pct):
                    if top_10_holders_pct > 40:
                        inst_activity_score = min(inst_activity_score + 0.1, 1.0)
                fundamental_components.append(inst_activity_score * 0.03)
                weights_used.append(0.03)
            
            # 11. INSIDER SENTIMENT (3%)
            insider_score_value = financial_data.get('insider_activity_score', 50.0)
            if insider_score_value is not None and not np.isnan(insider_score_value):
                if insider_score_value >= 80:
                    insider_sentiment = 1.0
                elif insider_score_value >= 60:
                    insider_sentiment = 0.7
                elif insider_score_value >= 40:
                    insider_sentiment = 0.5
                elif insider_score_value >= 20:
                    insider_sentiment = 0.3
                else:
                    insider_sentiment = 0.0
                fundamental_components.append(insider_sentiment * 0.03)
                weights_used.append(0.03)
            
            # Normalize by actual weights used
            if fundamental_components and weights_used:
                total_weight = sum(weights_used)
                if total_weight > 0:
                    normalization_factor = 1.0 / total_weight
                    total_score = sum(fundamental_components) * normalization_factor
                    return min(total_score, 1.0)
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_technical_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate technical indicators score from all available indicators.
        Moved from pipeline.py for consolidation (Phase 6c-Part2).
        
        ENHANCED Phase 2: Now uses ALL 15+ technical indicators with optimized weights.
        Total weight distribution normalized to 100%.
        
        Scoring Components:
        1. Momentum indicators (23%) - 1d, 7d, 30d price changes [INCREASED from 18%]
        2. RSI (12%) - Overbought/oversold signals
        3. Moving averages (12%) - 50d, 200d MA position
        4. MACD (10%) - Trend direction and strength
        5. Volume analysis (12%) - Spike ratio, correlation
        6. Volatility (10%) - Level, rank, Bollinger bands
        7. Relative strength (10%) - vs SPY and sector
        8. Beta (8%) - Market correlation
        9. Momentum consistency (7%) - Phase 1.4 metric
        10. Liquidity (6%) - Phase 1.4 metric
        
        NOTE: exit_signal_strength removed (was 5%), weight redistributed to momentum
        
        Returns:
            float: Normalized score [0.0-1.0] with dynamic weight adjustment
        """
        try:
            technical_components = []
            weights_used = []
            
            # 1. MOMENTUM INDICATORS (23%) - INCREASED from 18% (exit_signal_strength removed)
            price_1d = financial_data.get('price_1d_pct', 0)
            price_7d = financial_data.get('price_7d_pct', 0)
            momentum_30d = financial_data.get('momentum_30d_pct', 0)
            
            if not all(np.isnan([price_1d, price_7d, momentum_30d])):
                momentum_score = min(
                    (abs(price_1d) / 10 + abs(price_7d) / 20 + abs(momentum_30d) / 30) / 3,
                    1.0
                )
                technical_components.append(momentum_score * 0.23)
                weights_used.append(0.23)
            
            # 2. RSI INDICATOR (12%)
            rsi = financial_data.get('rsi')
            if rsi and not np.isnan(rsi):
                if rsi < 35:
                    rsi_score = 1.0
                elif rsi > 65:
                    rsi_score = 0.8
                elif 45 < rsi < 55:
                    rsi_score = 0.5
                else:
                    rsi_score = 0.7
                technical_components.append(rsi_score * 0.12)
                weights_used.append(0.12)
            
            # 3. MOVING AVERAGE POSITION (12%)
            ma_50_pct = financial_data.get('above_50d_ma_pct')
            ma_200_pct = financial_data.get('above_200d_ma_pct')
            
            ma_score = 0.0
            ma_factors = 0
            if ma_50_pct is not None and not np.isnan(ma_50_pct):
                if ma_50_pct > 5:
                    ma_score += 1.0
                elif ma_50_pct > 0:
                    ma_score += 0.7
                else:
                    ma_score += 0.3
                ma_factors += 1
                
            if ma_200_pct is not None and not np.isnan(ma_200_pct):
                if ma_200_pct > 5:
                    ma_score += 1.0
                elif ma_200_pct > 0:
                    ma_score += 0.7
                else:
                    ma_score += 0.3
                ma_factors += 1
            
            if ma_factors > 0:
                technical_components.append((ma_score / ma_factors) * 0.12)
                weights_used.append(0.12)
            
            # 4. MACD INDICATOR (10%)
            macd = financial_data.get('macd')
            if macd and not np.isnan(macd):
                if macd > 0:
                    macd_score = min(0.7 + abs(macd) * 0.3, 1.0)
                else:
                    macd_score = 0.3
                technical_components.append(macd_score * 0.10)
                weights_used.append(0.10)
            
            # 5. VOLUME ANALYSIS (12%)
            volume_spike = financial_data.get('volume_spike_ratio', 1)
            vol_price_corr = financial_data.get('volume_price_correlation', 0)
            
            if not np.isnan(volume_spike):
                volume_score = min(max(volume_spike - 1, 0) / 2, 1.0)
                
                if not np.isnan(vol_price_corr):
                    if vol_price_corr > 0.5:
                        volume_score = min(volume_score * 1.3, 1.0)
                    elif vol_price_corr > 0.3:
                        volume_score = min(volume_score * 1.15, 1.0)
                
                technical_components.append(volume_score * 0.12)
                weights_used.append(0.12)
            
            # 6. VOLATILITY ANALYSIS (10%)
            volatility = financial_data.get('volatility', 0)
            volatility_rank = financial_data.get('volatility_rank', 0)
            
            vol_score = 0.0
            vol_factors = 0
            
            if not np.isnan(volatility) and volatility > 0:
                if 15 < volatility < 35:
                    vol_score += 1.0
                elif 10 < volatility <= 15 or 35 <= volatility < 50:
                    vol_score += 0.7
                elif volatility < 10:
                    vol_score += 0.5
                else:
                    vol_score += 0.3
                vol_factors += 1
            
            if not np.isnan(volatility_rank):
                if 0.4 < volatility_rank < 0.8:
                    vol_score += 1.0
                elif volatility_rank <= 0.4:
                    vol_score += 0.6
                else:
                    vol_score += 0.7
                vol_factors += 1
            
            if vol_factors > 0:
                technical_components.append((vol_score / vol_factors) * 0.10)
                weights_used.append(0.10)
            
            # 7. RELATIVE STRENGTH (10%)
            relative_strength = financial_data.get('relative_strength', 0)
            sector_rs = financial_data.get('sector_relative_strength', 0)
            
            rs_score = 0.0
            rs_factors = 0
            
            if not np.isnan(relative_strength):
                if relative_strength > 5:
                    rs_score += 1.0
                elif relative_strength > 0:
                    rs_score += 0.7
                else:
                    rs_score += 0.3
                rs_factors += 1
                
            if not np.isnan(sector_rs):
                if sector_rs > 5:
                    rs_score += 1.0
                elif sector_rs > 0:
                    rs_score += 0.7
                else:
                    rs_score += 0.3
                rs_factors += 1
            
            if rs_factors > 0:
                technical_components.append((rs_score / rs_factors) * 0.10)
                weights_used.append(0.10)
            
            # 8. BETA / RISK METRICS (8%)
            beta = financial_data.get('beta')
            if beta and not np.isnan(beta):
                if 0.8 <= beta <= 1.2:
                    beta_score = 1.0
                elif 0.5 <= beta < 0.8 or 1.2 < beta <= 1.5:
                    beta_score = 0.7
                elif beta < 0.5:
                    beta_score = 0.5
                else:
                    beta_score = 0.4
                
                technical_components.append(beta_score * 0.08)
                weights_used.append(0.08)
            
            # 9. MOMENTUM CONSISTENCY (7%)
            momentum_consistency = financial_data.get('momentum_consistency_score')
            if momentum_consistency and not np.isnan(momentum_consistency):
                consistency_score = min(max(momentum_consistency / 100, 0), 1.0)
                technical_components.append(consistency_score * 0.07)
                weights_used.append(0.07)
            
            # 10. LIQUIDITY SCORE (6%)
            liquidity = financial_data.get('liquidity_score')
            if liquidity and not np.isnan(liquidity):
                liquidity_score = min(max(liquidity, 0), 1.0)
                technical_components.append(liquidity_score * 0.06)
                weights_used.append(0.06)
            
            # EXIT SIGNAL STRENGTH - REMOVED (was never implemented, 5% weight moved to momentum)
            
            # Normalize by actual weights used
            if technical_components and weights_used:
                total_weight = sum(weights_used)
                if total_weight > 0:
                    normalization_factor = 1.0 / total_weight
                    total_score = sum(technical_components) * normalization_factor
                    return min(total_score, 1.0)
            return 0.0
            
        except Exception:
            return 0.0
    
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
    
    def _calculate_short_interest_score(self, financial_data: Dict[str, Any]) -> float:
        """
        Calculate short squeeze potential score - ENHANCED v2.0.
        Moved from pipeline.py for consolidation (Phase 6b).
        
        Analyzes three metrics:
        1. Short % of float (50% weight) - Primary squeeze indicator
        2. Short % of outstanding (30% weight) - Additional confirmation  
        3. Short ratio / days to cover (20% weight) - Squeeze timing
        
        Returns:
            float: Score [0.0-1.0] indicating short squeeze potential
        """
        try:
            short_components = []
            
            # Short % of float (primary metric)
            short_pct_float = financial_data.get('short_pct_float', 0)
            if short_pct_float and not np.isnan(short_pct_float):
                if short_pct_float > 20:
                    short_components.append(1.0 * 0.5)  # High short squeeze potential
                elif short_pct_float > 10:
                    short_components.append(0.7 * 0.5)  # Moderate potential
                elif short_pct_float > 5:
                    short_components.append(0.5 * 0.5)  # Some potential
                else:
                    short_components.append(0.3 * 0.5)  # Low potential
            
            # NEW v2.0: Short % of outstanding (additional confirmation)
            short_pct_outstanding = financial_data.get('short_pct_outstanding', 0)
            if short_pct_outstanding and not np.isnan(short_pct_outstanding):
                if short_pct_outstanding > 15:
                    short_components.append(1.0 * 0.3)
                elif short_pct_outstanding > 7:
                    short_components.append(0.7 * 0.3)
                else:
                    short_components.append(0.4 * 0.3)
            
            # Short ratio (days to cover)
            short_ratio = financial_data.get('short_ratio', 0)
            if short_ratio and not np.isnan(short_ratio):
                if short_ratio > 5:  # More than 5 days to cover = squeeze risk
                    short_components.append(1.0 * 0.2)
                elif short_ratio > 3:
                    short_components.append(0.7 * 0.2)
                else:
                    short_components.append(0.4 * 0.2)
            
            return sum(short_components) if short_components else 0.3  # Default low potential
            
        except Exception as e:
            # Note: Using generic exception since we don't have logger in SignalScorer
            return 0.3
    
    def _calculate_score_components(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and store detailed score components for transparency.
        Moved from pipeline.py for consolidation (Phase 6c-Part3).
        
        This method breaks down the weighted score into its constituent parts
        for better explainability and debugging.
        """
        try:
            # Extract scoring components (Phase 7)
            reddit_score = signal.get('reddit_score', 0)
            financial_score = signal.get('financial_score', 0)
            signal_score = signal.get('signal_score', 0)  # Phase 7
            
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
                'signal_score': signal_score,  # Phase 7
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
            
            # Use Phase 7 confidence if available, otherwise calculate legacy confidence
            # Phase 7 confidence is more sophisticated (considers score, balance, completeness)
            if 'phase7_confidence' in signal:
                signal['confidence'] = signal['phase7_confidence']  # Use Phase 7
                signal['prediction_confidence'] = signal['phase7_confidence']  # Backward compat
            else:
                signal['prediction_confidence'] = self._calculate_prediction_confidence(signal)
                # Also set confidence for consistency
                if 'confidence' not in signal:
                    signal['confidence'] = signal['prediction_confidence']
            
        except Exception as e:
            # Fallback minimal components
            signal['score_components'] = {'weighted_score': signal.get('weighted_score', 0)}
            signal['score_explanation'] = f"Score {signal.get('weighted_score', 0):.3f} based on combined reddit and financial metrics"
            
        return signal
    
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
    
    # ===== PHASE 7: NEW 6-GROUP SCORING METHODS =====
    
    def _calculate_fundamental_score(self, data: Dict[str, Any]) -> float:
        """Phase 7: Calculate fundamental score using standalone function"""
        return _calculate_fundamental_score_standalone(data)
    
    def _calculate_social_alternative_score(self, data: Dict[str, Any]) -> float:
        """Phase 7: Calculate social/alternative score using standalone function"""
        return _calculate_social_alternative_score_standalone(data)
    
    def _calculate_news_macro_score(self, data: Dict[str, Any]) -> float:
        """Phase 7: Calculate news/macro score using standalone function"""
        return _calculate_news_macro_score_standalone(data)
    
    def _calculate_risk_stability_score(self, data: Dict[str, Any]) -> float:
        """Phase 7: Calculate risk/stability score using standalone function"""
        return _calculate_risk_stability_score_standalone(data)
    
    def _calculate_institutional_smart_money_score(self, data: Dict[str, Any]) -> float:
        """Phase 7: Calculate institutional/smart money score using standalone function"""
        return _calculate_institutional_smart_money_score_standalone(data)
    
    def _calculate_signal_score_v2(self, data: Dict, component_scores: Dict) -> float:
        """
        Phase 7: Calculate final signal score using 6-group structure.
        Uses profile-based weights from self.weights.
        """
        total_score = 0.0
        
        # Get weights from loaded profile (defaults to ml_optimized if not found)
        weights = self.weights
        
        # Weight 6 components based on profile
        total_score += component_scores.get('technical', 0) * weights.get('technical', 0.25)
        total_score += component_scores.get('fundamental', 0) * weights.get('fundamental', 0.25)
        total_score += component_scores.get('news_macro', 0) * weights.get('news_macro', 0.20)
        total_score += component_scores.get('social_alternative', 0) * weights.get('social_alternative', 0.15)
        total_score += component_scores.get('risk_stability', 0) * weights.get('risk_stability', 0.15)
        total_score += component_scores.get('institutional_smart_money', 0) * weights.get('institutional_smart_money', 0.05)
        
        # Apply emerging boost if applicable (keep existing logic)
        if self._is_emerging_signal(data):
            total_score *= 1.15  # Slightly lower boost for Phase 7
        
        return max(0.0, min(total_score, 1.0))
    
    def _calculate_signal_score_v2_adjusted(self, data: Dict, component_scores: Dict, 
                                           adjusted_weights: Dict[str, float]) -> float:
        """
        Phase 5: Calculate signal score with dynamically adjusted weights.
        Uses adjusted weights from trade type classification.
        """
        total_score = 0.0
        
        # Weight 6 components based on adjusted weights
        total_score += component_scores.get('technical', 0) * adjusted_weights.get('technical', 0.25)
        total_score += component_scores.get('fundamental', 0) * adjusted_weights.get('fundamental', 0.25)
        total_score += component_scores.get('news_macro', 0) * adjusted_weights.get('news_macro', 0.20)
        total_score += component_scores.get('social_alternative', 0) * adjusted_weights.get('social_alternative', 0.15)
        total_score += component_scores.get('risk_stability', 0) * adjusted_weights.get('risk_stability', 0.15)
        total_score += component_scores.get('institutional_smart_money', 0) * adjusted_weights.get('institutional_smart_money', 0.05)
        
        # Apply emerging boost if applicable
        if self._is_emerging_signal(data):
            total_score *= 1.15
        
        return max(0.0, min(total_score, 1.0))
    
    def _adjust_weights_by_trade_type(self, trade_tags: List[str]) -> Dict[str, float]:
        """
        Phase 5: Adjust component weights based on trade type.
        
        Multipliers:
        - Momentum: technical * 1.15
        - Value: fundamental * 1.15
        - Event-Driven: news_macro * 1.25
        
        After applying multipliers, renormalize with 35% cap per component.
        """
        weights = self.weights.copy()
        
        # Apply multipliers based on primary trade type
        if 'Momentum' in trade_tags:
            weights['technical'] *= 1.15
            logger.debug("Applied Momentum boost to technical weight")
        elif 'Value' in trade_tags:
            weights['fundamental'] *= 1.15
            logger.debug("Applied Value boost to fundamental weight")
        elif 'Event-Driven' in trade_tags:
            weights['news_macro'] *= 1.25
            logger.debug("Applied Event-Driven boost to news_macro weight")
        
        # Renormalize with 35% cap
        return self._renormalize_weights(weights, max_weight=0.35)
    
    def _renormalize_weights(self, weights: Dict[str, float], max_weight: float = 0.35) -> Dict[str, float]:
        """
        Phase 5: Cap maximum weight and renormalize to sum to 1.0.
        
        Args:
            weights: Dictionary of component weights
            max_weight: Maximum allowed weight for any component (default 35%)
        
        Returns:
            Renormalized weights that sum to 1.0
        """
        # Cap each weight
        capped = {k: min(v, max_weight) for k, v in weights.items()}
        
        # Renormalize to sum to 1.0
        total = sum(capped.values())
        if total > 0:
            return {k: v / total for k, v in capped.items()}
        else:
            return weights  # Fallback to original if all zeros
    
    def _calculate_contrarian_bonus(self, trade_tags: List[str], 
                                   classification_details: Dict) -> float:
        """
        Phase 5: Calculate contrarian bonus for oversold + negative sentiment.
        
        Bonus = +4% * |social_z| when:
        - Trade type is Contrarian
        - Price is oversold (RSI < 30)
        - Social sentiment is negative (social_z < 0)
        
        Returns:
            Bonus score to add (0.0 to ~0.04)
        """
        if 'Contrarian' not in trade_tags:
            return 0.0
        
        is_oversold = classification_details.get('is_oversold', False)
        social_z = classification_details.get('scores', {}).get('social_z', 0)
        
        if is_oversold and social_z < 0:
            bonus = 0.04 * abs(social_z)
            logger.debug(f"Contrarian bonus: {bonus:.4f} (social_z={social_z:.2f})")
            return bonus
        
        return 0.0
    
    def _get_default_score(self, ticker: str) -> SignalResult:
        """
        Phase 5: Return default SignalResult when enhanced data fetch fails.
        """
        return SignalResult(
            ticker=ticker,
            signal_score=0.0,
            trade_type="Unknown",
            risk_level="Unknown",
            reddit_score=0.0,
            news_score=0.0,
            financial_score=0.0,
            top_factors=["Data fetch failed"],
            signal_type="Unknown",
            confidence=0.0
        )
    
    def _classify_trade_type_v2(self, data: Dict, component_scores: Dict) -> str:
        """Phase 7: Classify trade type based on 6 component scores"""
        
        # Extract scores
        technical = component_scores.get('technical', 0)
        fundamental = component_scores.get('fundamental', 0)
        news_macro = component_scores.get('news_macro', 0)
        social = component_scores.get('social_alternative', 0)
        risk = component_scores.get('risk_stability', 0)
        inst = component_scores.get('institutional_smart_money', 0)
        
        # Classification logic
        if technical > 0.6 and social > 0.5:
            return "Momentum"
        elif news_macro > 0.6:
            return "Event-Driven"
        elif fundamental > 0.6 and inst > 0.5:
            return "Value"
        elif social > 0.7:
            return "Speculative"
        elif risk > 0.6 and fundamental > 0.5:
            return "Growth"
        else:
            return "Balanced"
    
    def _identify_top_factors_v2(self, data: Dict, component_scores: Dict) -> List[str]:
        """Phase 7: Identify top contributing factors from 6 groups"""
        factors = []
        
        try:
            # Technical factors
            if component_scores.get('technical', 0) > 0.5:
                price_7d = data.get('price_7d_pct')
                if price_7d is not None and price_7d > 5:
                    factors.append("Price Momentum")
                vol_spike = data.get('volume_spike_ratio')
                if vol_spike is not None and vol_spike > 1.5:
                    factors.append("Volume Surge")
                rsi = data.get('rsi')
                if rsi is not None and rsi < 35:
                    factors.append("Oversold (RSI)")
            
            # Fundamental factors
            if component_scores.get('fundamental', 0) > 0.5:
                pe = data.get('pe_ratio')
                if pe is not None and pe < 15:
                    factors.append("Attractive Valuation")
                rev_growth = data.get('revenue_growth')
                if rev_growth is not None and rev_growth > 0.15:
                    factors.append("Revenue Growth")
            
            # Social factors
            if component_scores.get('social_alternative', 0) > 0.5:
                mentions = data.get('reddit_mentions')
                if mentions is not None and mentions >= 5:
                    factors.append("High Reddit Mentions")
                sentiment = data.get('reddit_sentiment')
                if sentiment is not None and sentiment > 0.6:
                    factors.append("Positive Sentiment")
            
            # News factors
            if component_scores.get('news_macro', 0) > 0.5:
                news = data.get('news_score')
                if news is not None and news > 0.6:
                    factors.append("Positive News")
            
            # Risk factors
            if component_scores.get('risk_stability', 0) > 0.6:
                factors.append("Low Risk Profile")
            
            # Institutional factors
            if component_scores.get('institutional_smart_money', 0) > 0.5:
                inst_change = data.get('institutional_change_qoq')
                if inst_change is not None and inst_change > 3:
                    factors.append("Institutional Buying")
                insider = data.get('insider_activity_score')
                if insider is not None and insider > 70:
                    factors.append("Insider Buying")
        except Exception:
            pass  # Return whatever factors we collected
        
        return factors[:5]  # Top 5 factors
    
    def _calculate_confidence_v2(self, data: Dict, component_scores: Dict) -> float:
        """
        Phase 7: Calculate confidence score based on 6-group balance.
        Higher confidence when:
        - Multiple groups score highly
        - Scores are not dominated by one group
        - Data completeness is high
        """
        scores = [
            component_scores.get('technical', 0),
            component_scores.get('fundamental', 0),
            component_scores.get('news_macro', 0),
            component_scores.get('social_alternative', 0),
            component_scores.get('risk_stability', 0),
            component_scores.get('institutional_smart_money', 0)
        ]
        
        # Base confidence: average of all scores
        avg_score = sum(scores) / len(scores)
        
        # Balance factor: Lower std dev = more balanced = higher confidence
        std_dev = np.std(scores)
        balance_factor = max(0, 1 - std_dev)  # Lower variance = better
        
        # Data completeness: Check how many signals are present
        data_fields = [
            'price_7d_pct', 'rsi', 'volume_spike_ratio',  # Technical
            'pe_ratio', 'revenue_growth', 'profit_margin',  # Fundamental
            'news_score', 'news_mention_count',  # News
            'reddit_mentions', 'reddit_sentiment',  # Social
            'beta', 'volatility',  # Risk
            'institutional_ownership_pct', 'insider_activity_score'  # Institutional
        ]
        present_fields = sum(1 for field in data_fields if data.get(field) is not None)
        completeness = present_fields / len(data_fields)
        
        # Weighted confidence
        confidence = (avg_score * 0.5) + (balance_factor * 0.3) + (completeness * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    # ===== END PHASE 7 METHODS =====
    
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
    
    # Sort by signal score (Phase 7)
    results.sort(key=lambda x: x.signal_score, reverse=True)
    
    return results


# ===== PHASE 7: 6-GROUP COMPREHENSIVE SCORING METHODS =====
# Added new methods for Phase 7 comprehensive scoring system
# These supplement the existing SignalScorer class with new scoring logic

def _calculate_fundamental_score_standalone(financial_data: Dict[str, Any]) -> float:
    """
    Phase 2: REDESIGNED Fundamental Scoring with 100% Weight Distribution.
    
    New 6-Category System (13 signals total):
    
    1. Growth (25%): revenue_growth (10%), eps_growth (10%), fcf_growth_3y_cagr (5%)
    2. Profitability (20%): roe (5%), roic (10%), fcf_margin (5%)
    3. Valuation (20%): pe_ratio (5%), price_to_sales (5%), price_to_book (5%), 
                        sector_relative_percentile (5% - deferred to Phase 8)
    4. Financial Health (15%): debt_to_equity (5%), current_ratio (5%), interest_coverage (5%)
    5. Earnings Quality (10%): last_earnings_surprise (5%), earnings_surprise_streak (5%)
    6. Income (10%): dividend_yield (5%), share_buyback_yield (5%)
    
    NULL Handling Strategy (Hybrid):
    - Redistribute missing weights proportionally to available signals
    - If data completeness < 70%, apply 20% penalty to final score
    - Missing signals don't zero out valid ones
    
    Returns: float [0.0-1.0] representing normalized fundamental score
    """
    try:
        # Define weights for all fundamental signals (sums to 1.0)
        weights = {
            # Growth (25%)
            "revenue_growth": 0.10,
            "eps_growth": 0.10,
            "fcf_growth_3y_cagr": 0.05,
            
            # Profitability (20%)
            "roe": 0.05,
            "roic": 0.10,
            "fcf_margin": 0.05,
            
            # Valuation (20%)
            "pe_ratio": 0.05,
            "price_to_sales": 0.05,
            "price_to_book": 0.05,
            "sector_relative_percentile": 0.05,  # Phase 8 - currently NULL
            
            # Financial Health (15%)
            "debt_to_equity": 0.05,
            "current_ratio": 0.05,
            "interest_coverage": 0.05,
            
            # Earnings Quality (10%)
            "last_earnings_surprise_pct": 0.05,
            "earnings_surprise_streak": 0.05,
            
            # Income (10%)
            "dividend_yield": 0.05,
            "share_buyback_yield": 0.05,
        }
        
        # Normalize all fields using appropriate functions
        # Note: Values are already in percentage format from FinancialMetricsCalculator
        normalized_values = {}
        
        # === GROWTH (25%) ===
        # Revenue growth: -10% to +30% range (stored as percentages: 12.9 = 12.9%)
        rev_growth = financial_data.get('revenue_growth')
        if rev_growth is not None:
            # Convert percentage to decimal for normalize_growth
            normalized_values['revenue_growth'] = normalize_growth(rev_growth / 100, -0.10, 0.30)
        
        # EPS growth: -10% to +30% range (alias for earnings_growth, stored as percentage)
        eps_growth = financial_data.get('eps_growth')
        if eps_growth is not None:
            normalized_values['eps_growth'] = normalize_growth(eps_growth / 100, -0.10, 0.30)
        
        # FCF growth 3Y CAGR: 0% to +20% range (stored as percentage)
        fcf_growth = financial_data.get('fcf_growth_3y_cagr')
        if fcf_growth is not None:
            normalized_values['fcf_growth_3y_cagr'] = normalize_growth(fcf_growth / 100, 0.0, 0.20)
        
        # === PROFITABILITY (20%) ===
        # ROE: 0% to 25% range (stored as percentage: 5.0, 25.0)
        roe = financial_data.get('roe')
        if roe is not None and not np.isnan(roe):
            normalized_values['roe'] = normalize_direct(roe, 0.0, 25.0)
        
        # ROIC: 0% to 25% range (stored as percentage)
        roic = financial_data.get('roic')
        if roic is not None and not np.isnan(roic):
            normalized_values['roic'] = normalize_direct(roic, 0.0, 25.0)
        
        # FCF Margin: 0% to 25% range (stored as percentage)
        fcf_margin = financial_data.get('fcf_margin')
        if fcf_margin is not None and not np.isnan(fcf_margin):
            normalized_values['fcf_margin'] = normalize_direct(fcf_margin, 0.0, 25.0)
        
        # === VALUATION (20%) ===
        # PE Ratio: 5 to 50 range (lower is better)
        pe_ratio = financial_data.get('pe_ratio')
        if pe_ratio is not None and not np.isnan(pe_ratio) and pe_ratio > 0:
            normalized_values['pe_ratio'] = normalize_inverted(pe_ratio, 5, 50)
        
        # Price to Sales: 0.5 to 10 range (lower is better)
        price_to_sales = financial_data.get('price_to_sales')
        if price_to_sales is not None and not np.isnan(price_to_sales):
            normalized_values['price_to_sales'] = normalize_inverted(price_to_sales, 0.5, 10)
        
        # Price to Book: 0.5 to 8 range (lower is better)
        price_to_book = financial_data.get('price_to_book')
        if price_to_book is not None and not np.isnan(price_to_book):
            normalized_values['price_to_book'] = normalize_inverted(price_to_book, 0.5, 8)
        
        # Sector Relative Percentile: 0 to 1 range (already normalized) - Phase 8
        sector_pct = financial_data.get('sector_relative_percentile')
        if sector_pct is not None:
            normalized_values['sector_relative_percentile'] = sector_pct
        
        # === FINANCIAL HEALTH (15%) ===
        # Debt to Equity: 0 to 200 range (lower is better)
        # Note: Modern large-cap companies often have D/E ratios of 50-150
        # UNH=75.58, AAPL=154.49 are typical for Fortune 500 companies
        debt_to_equity = financial_data.get('debt_to_equity')
        if debt_to_equity is not None and not np.isnan(debt_to_equity):
            normalized_values['debt_to_equity'] = normalize_inverted(debt_to_equity, 0, 200)
        
        # Current Ratio: 1.0 to 2.5 range (higher is better)
        current_ratio = financial_data.get('current_ratio')
        if current_ratio is not None and not np.isnan(current_ratio):
            normalized_values['current_ratio'] = normalize_direct(current_ratio, 1.0, 2.5)
        
        # Interest Coverage: 1.5 to 8.0 range (higher is better)
        interest_coverage = financial_data.get('interest_coverage')
        if interest_coverage is not None and not np.isnan(interest_coverage):
            normalized_values['interest_coverage'] = normalize_direct(interest_coverage, 1.5, 8.0)
        
        # === EARNINGS QUALITY (10%) ===
        # Last Earnings Surprise: -10% to +10% range (as percentage)
        last_surprise = financial_data.get('last_earnings_surprise_pct')
        if last_surprise is not None and not np.isnan(last_surprise):
            normalized_values['last_earnings_surprise_pct'] = normalize_direct(last_surprise, -10.0, 10.0)
        
        # Earnings Surprise Streak: 0 to 1 range (already normalized as fraction)
        surprise_streak = financial_data.get('earnings_surprise_streak')
        if surprise_streak is not None and not np.isnan(surprise_streak):
            normalized_values['earnings_surprise_streak'] = surprise_streak  # Already 0-1
        
        # === INCOME (10%) ===
        # Dividend Yield: 0% to 6% range (stored as percentage)
        div_yield = financial_data.get('dividend_yield')
        if div_yield is not None and not np.isnan(div_yield):
            normalized_values['dividend_yield'] = normalize_direct(div_yield, 0.0, 6.0)
        
        # Share Buyback Yield: 0% to 5% range (stored as percentage)
        buyback_yield = financial_data.get('share_buyback_yield')
        if buyback_yield is not None and not np.isnan(buyback_yield):
            normalized_values['share_buyback_yield'] = normalize_direct(buyback_yield, 0.0, 5.0)
        
        # === CALCULATE WEIGHTED SCORE ===
        scores = []
        total_weight_used = 0.0
        missing_fields = []
        
        for field, weight in weights.items():
            norm_value = normalized_values.get(field)
            if norm_value is not None:
                scores.append(norm_value * weight)
                total_weight_used += weight
            else:
                missing_fields.append(field)
        
        # No data at all - return None to indicate insufficient data
        if total_weight_used == 0:
            logger.debug(f"[FUNDAMENTAL] No fundamental data available")
            return None
        
        # Calculate raw score (redistributes weights to available signals)
        raw_score = sum(scores) / total_weight_used
        
        # === DATA COMPLETENESS PENALTY ===
        total_possible_weight = sum(weights.values())
        coverage_ratio = total_weight_used / total_possible_weight
        
        # Log data completeness for diagnostics
        logger.info(f"[FUNDAMENTAL] Data completeness: {coverage_ratio:.1%} "
                   f"({len(normalized_values)}/{len(weights)} fields) - "
                   f"Missing: {', '.join(missing_fields) if missing_fields else 'None'}")
        
        # Apply 20% penalty if coverage < 70%
        if coverage_ratio < 0.70:
            raw_score *= 0.80
            logger.debug(f"[FUNDAMENTAL] Applied 20% penalty for low coverage ({coverage_ratio:.1%})")
        
        # Clamp to [0, 1] range
        final_score = max(0.0, min(1.0, raw_score))
        
        logger.debug(f"[FUNDAMENTAL] Final score: {final_score:.4f} (raw: {raw_score:.4f})")
        return round(final_score, 4)
        
    except Exception as e:
        logger.error(f"[FUNDAMENTAL] Error calculating fundamental score: {e}", exc_info=True)
        return None


def _calculate_social_alternative_score_standalone(reddit_data: Dict[str, Any]) -> float:
    """
    Phase 7: Social/Alternative scoring (15% of total score).
    
    Expanded from _calculate_reddit_score() to accommodate future integrations.
    
    Components (5 signals):
    - Reddit mentions (available)
    - Reddit sentiment (available)
    - Reddit upvotes (available)
    - Twitter/X mentions (MISSING - return 0.5)
    - Google Trends score (MISSING - return 0.5)
    
    Returns: float [0.0-1.0] with aggressive normalization
    """
    try:
        social_components = []
        weights_used = []
        
        # 1. REDDIT MENTIONS (30%)
        mention_count = reddit_data.get('reddit_mentions', reddit_data.get('mentions', 0))
        if mention_count is not None:
            mention_score = min(mention_count / 5, 1.0)  # 5+ mentions = 1.0
            # Aggressive range: Scale 0-1 to 0.2-0.8
            mention_score = 0.2 + (mention_score * 0.6)
            social_components.append(mention_score * 0.30)
            weights_used.append(0.30)
        
        # 2. REDDIT SENTIMENT (30%)
        avg_sentiment = reddit_data.get('reddit_sentiment', reddit_data.get('sentiment', None))
        if avg_sentiment is not None:
            # Convert -1,1 to 0,1
            sentiment_score = (avg_sentiment + 1) / 2
            # Aggressive range: 0.2-0.8
            sentiment_score = 0.2 + (sentiment_score * 0.6)
            social_components.append(sentiment_score * 0.30)
            weights_used.append(0.30)
        
        # 3. REDDIT UPVOTES (20%)
        avg_score = reddit_data.get('reddit_score', reddit_data.get('avg_score', None))
        if avg_score is not None:
            upvote_score = min(max(avg_score / 100, 0), 1.0)
            upvote_score = 0.2 + (upvote_score * 0.6)
            social_components.append(upvote_score * 0.20)
            weights_used.append(0.20)
        
        # 4. TWITTER/X MENTIONS (10%) - MISSING, return 0.5
        social_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 5. GOOGLE TRENDS (10%) - MISSING, return 0.5
        social_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # Normalize
        if social_components and weights_used:
            total_weight = sum(weights_used)
            if total_weight > 0:
                normalization_factor = 1.0 / total_weight
                total_score = sum(social_components) * normalization_factor
                return min(total_score, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5


def _calculate_news_macro_score_standalone(news_data: Dict[str, Any]) -> float:
    """
    Phase 7: News/Macro scoring (20% of total score).
    
    Expanded from _calculate_news_score() with macro indicators.
    
    Components (7 signals):
    - News sentiment (available)
    - News mention count (available)
    - Earnings date proximity (MISSING - return 0.5)
    - Market regime indicator (MISSING - return 0.5)
    - Sector momentum (MISSING - return 0.5)
    - Correlation to SPY (MISSING - return 0.5)
    - News recency score (calculated from timestamps if available)
    
    Returns: float [0.0-1.0] with aggressive normalization
    """
    try:
        news_components = []
        weights_used = []
        
        # 1. NEWS SENTIMENT (35%)
        news_sentiment = news_data.get('news_score', news_data.get('news_sentiment', None))
        if news_sentiment is not None:
            # Normalize and apply aggressive range
            sentiment_score = max(0, min(news_sentiment, 1.0))
            sentiment_score = 0.2 + (sentiment_score * 0.6)
            news_components.append(sentiment_score * 0.35)
            weights_used.append(0.35)
        
        # 2. NEWS MENTION COUNT (20%)
        mention_count = news_data.get('news_mention_count', news_data.get('mention_count', 0))
        if mention_count is not None:
            # 5+ news mentions = high score
            mention_score = min(mention_count / 5, 1.0)
            mention_score = 0.2 + (mention_score * 0.6)
            news_components.append(mention_score * 0.20)
            weights_used.append(0.20)
        
        # 3. EARNINGS DATE PROXIMITY (10%) - MISSING
        news_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 4. MARKET REGIME (10%) - MISSING
        news_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 5. SECTOR MOMENTUM (10%) - MISSING
        news_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 6. CORRELATION TO SPY (10%) - MISSING
        news_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 7. NEWS RECENCY (5%)
        # Can calculate from timestamps if available
        news_components.append(0.5 * 0.05)
        weights_used.append(0.05)
        
        # Normalize
        if news_components and weights_used:
            total_weight = sum(weights_used)
            if total_weight > 0:
                normalization_factor = 1.0 / total_weight
                total_score = sum(news_components) * normalization_factor
                return min(total_score, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5


def _calculate_risk_stability_score_standalone(financial_data: Dict[str, Any]) -> float:
    """
    Phase 7: Risk/Stability scoring (15% of total score).
    
    Enhanced from _calculate_risk_score() with more risk metrics.
    
    Components (8 signals):
    - Beta (available)
    - Volatility (available)
    - Volatility rank (available)
    - Liquidity/Volume (available)
    - Sharpe ratio (MISSING - return 0.5)
    - Max drawdown (MISSING - return 0.5)
    - RSI (available)
    - Bollinger band position (available)
    
    Returns: float [0.0-1.0] with aggressive normalization
    NOTE: Lower risk = higher score (inverted scoring)
    """
    try:
        risk_components = []
        weights_used = []
        
        # 1. BETA (20%) - Lower is better
        beta = financial_data.get('beta', financial_data.get('beta_vs_spy', None))
        if beta is not None and not np.isnan(beta):
            if 0.8 <= beta <= 1.2:
                beta_score = 0.8  # Market beta
            elif beta < 0.8:
                beta_score = 0.9  # Lower volatility
            elif beta < 1.5:
                beta_score = 0.5
            else:
                beta_score = 0.2  # High risk
            risk_components.append(beta_score * 0.20)
            weights_used.append(0.20)
        
        # 2. VOLATILITY (20%) - Lower is better
        volatility = financial_data.get('volatility', None)
        if volatility is not None and not np.isnan(volatility):
            if volatility < 20:
                vol_score = 0.8
            elif volatility < 35:
                vol_score = 0.5
            else:
                vol_score = 0.2
            risk_components.append(vol_score * 0.20)
            weights_used.append(0.20)
        
        # 3. LIQUIDITY (15%) - Higher volume is better
        avg_volume = financial_data.get('avg_volume', None)
        if avg_volume is not None and avg_volume > 0:
            if avg_volume > 5_000_000:
                liquidity_score = 0.8
            elif avg_volume > 1_000_000:
                liquidity_score = 0.5
            else:
                liquidity_score = 0.3
            risk_components.append(liquidity_score * 0.15)
            weights_used.append(0.15)
        
        # 4. RSI (15%) - Extreme values = risk
        rsi = financial_data.get('rsi', None)
        if rsi is not None and not np.isnan(rsi):
            if 35 <= rsi <= 65:
                rsi_score = 0.8  # Stable
            elif 25 <= rsi < 35 or 65 < rsi <= 75:
                rsi_score = 0.5
            else:
                rsi_score = 0.2  # Overbought/oversold = risk
            risk_components.append(rsi_score * 0.15)
            weights_used.append(0.15)
        
        # 5. VOLATILITY RANK (10%)
        vol_rank = financial_data.get('volatility_rank', None)
        if vol_rank is not None and not np.isnan(vol_rank):
            if vol_rank < 0.5:
                vr_score = 0.8  # Low volatility rank
            elif vol_rank < 0.75:
                vr_score = 0.5
            else:
                vr_score = 0.3
            risk_components.append(vr_score * 0.10)
            weights_used.append(0.10)
        
        # 6. SHARPE RATIO (10%) - MISSING
        risk_components.append(0.5 * 0.10)
        weights_used.append(0.10)
        
        # 7. MAX DRAWDOWN (5%) - MISSING
        risk_components.append(0.5 * 0.05)
        weights_used.append(0.05)
        
        # 8. BOLLINGER POSITION (5%)
        bollinger_position = financial_data.get('bollinger_position', None)
        if bollinger_position is not None:
            if 0.3 <= bollinger_position <= 0.7:
                bb_score = 0.8
            elif 0.1 <= bollinger_position < 0.3 or 0.7 < bollinger_position <= 0.9:
                bb_score = 0.5
            else:
                bb_score = 0.2
            risk_components.append(bb_score * 0.05)
            weights_used.append(0.05)
        
        # Normalize
        if risk_components and weights_used:
            total_weight = sum(weights_used)
            if total_weight > 0:
                normalization_factor = 1.0 / total_weight
                total_score = sum(risk_components) * normalization_factor
                return min(total_score, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5


def _calculate_institutional_smart_money_score_standalone(financial_data: Dict[str, Any]) -> float:
    """
    Phase 7: Institutional/Smart Money scoring (5% of total score).
    
    NEW category for Phase 7.
    
    Components (5 signals):
    - Institutional ownership % (available)
    - Institutional change QoQ (available)
    - Insider buy count (available)
    - Insider sell count (available)
    - Insider net shares (available)
    
    Missing future signals (return 0.5):
    - ETF net flows
    - Unusual options activity
    - 13F filing changes
    
    Returns: float [0.0-1.0] with aggressive normalization
    """
    try:
        inst_components = []
        weights_used = []
        
        # 1. INSTITUTIONAL OWNERSHIP % (25%)
        inst_pct = financial_data.get('institutional_ownership_pct', None)
        if inst_pct is not None and not np.isnan(inst_pct):
            if 40 <= inst_pct <= 70:
                inst_score = 0.8  # Sweet spot
            elif 30 <= inst_pct < 40 or 70 < inst_pct <= 85:
                inst_score = 0.5
            else:
                inst_score = 0.3
            inst_components.append(inst_score * 0.25)
            weights_used.append(0.25)
        
        # 2. INSTITUTIONAL CHANGE QoQ (25%)
        inst_change = financial_data.get('institutional_change_qoq', None)
        if inst_change is not None and not np.isnan(inst_change):
            if inst_change > 5:
                change_score = 0.8  # Strong buying
            elif inst_change > 2:
                change_score = 0.6
            elif inst_change > 0:
                change_score = 0.5
            elif inst_change > -2:
                change_score = 0.4
            else:
                change_score = 0.2  # Selling
            inst_components.append(change_score * 0.25)
            weights_used.append(0.25)
        
        # 3. INSIDER ACTIVITY (20%)
        insider_score_value = financial_data.get('insider_activity_score', None)
        if insider_score_value is not None and not np.isnan(insider_score_value):
            # Insider score is 0-100, normalize to 0-1
            insider_norm = insider_score_value / 100
            # Aggressive range
            insider_norm = 0.2 + (insider_norm * 0.6)
            inst_components.append(insider_norm * 0.20)
            weights_used.append(0.20)
        
        # 4. INSIDER NET SHARES (15%)
        insider_net = financial_data.get('insider_net_shares', None)
        if insider_net is not None and not np.isnan(insider_net):
            if insider_net > 100000:
                net_score = 0.8
            elif insider_net > 0:
                net_score = 0.6
            elif insider_net > -100000:
                net_score = 0.4
            else:
                net_score = 0.2
            inst_components.append(net_score * 0.15)
            weights_used.append(0.15)
        
        # 5. TOP HOLDERS CONCENTRATION (15%)
        top_10_pct = financial_data.get('top_10_holders_pct', None)
        if top_10_pct is not None and not np.isnan(top_10_pct):
            if 30 <= top_10_pct <= 50:
                holder_score = 0.8
            elif 20 <= top_10_pct < 30 or 50 < top_10_pct <= 60:
                holder_score = 0.5
            else:
                holder_score = 0.3
            inst_components.append(holder_score * 0.15)
            weights_used.append(0.15)
        
        # Normalize
        if inst_components and weights_used:
            total_weight = sum(weights_used)
            if total_weight > 0:
                normalization_factor = 1.0 / total_weight
                total_score = sum(inst_components) * normalization_factor
                return min(total_score, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5


# ===== END PHASE 7 METHODS =====


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