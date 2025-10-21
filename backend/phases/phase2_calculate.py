"""
Phase 2: Calculate Metrics
===========================

Compute ALL factors from raw Phase 1 data and assign to 6 signal groups.

This module:
1. Takes RawYFinanceData from Phase 1
2. Computes 100+ factors (technical, fundamental, news, social, risk, institutional)
3. Assigns each factor to exactly ONE of 6 groups per config/factor_to_group.yaml
4. Returns GroupFactors dataclass for Phase 3 normalization

NO API calls - all calculations from cached Phase 1 data.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from functools import wraps

from backend.integrations.yfinance import RawYFinanceData
from backend.utils.logger import get_logger
from backend.utils.metrics import emit_metric
from backend.utils.factor_monitor import FactorMonitor

logger = get_logger(__name__)


# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================

def safe_calculation(factor_name: str):
    """
    Decorator to safely calculate a single factor with comprehensive error handling.
    
    Features:
    - Catches all exceptions
    - Logs specific error per factor
    - Returns None for failed calculations
    - Handles division by zero
    - Validates output is numeric
    
    Usage:
        @safe_calculation("pe_ratio")
        def _calc_pe(self, data):
            return price / earnings
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Validate result is numeric
                if result is not None:
                    if isinstance(result, (int, float)):
                        # Check for invalid values
                        if np.isinf(result):
                            logger.debug(f"[{factor_name}] Infinite value, returning None")
                            return None
                        if np.isnan(result):
                            return None
                        return float(result)
                    else:
                        logger.warning(f"[{factor_name}] Non-numeric result: {type(result)}, returning None")
                        return None
                return None
                
            except ZeroDivisionError:
                logger.debug(f"[{factor_name}] Division by zero")
                return None
            except (KeyError, AttributeError) as e:
                logger.debug(f"[{factor_name}] Missing data: {e}")
                return None
            except (TypeError, ValueError) as e:
                logger.debug(f"[{factor_name}] Invalid data type: {e}")
                return None
            except Exception as e:
                logger.warning(f"[{factor_name}] Unexpected error: {e}")
                return None
        
        return wrapper
    return decorator


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GroupFactors:
    """
    Computed factors organized by signal group.
    
    Each group dict contains:
        {factor_name: float_value}
    
    ALL factors must be numeric (float). None/NaN indicates missing data.
    """
    ticker: str
    
    # Six signal groups (matches config/factor_to_group.yaml)
    technical: Dict[str, float] = field(default_factory=dict)
    fundamental: Dict[str, float] = field(default_factory=dict)
    news_macro: Dict[str, float] = field(default_factory=dict)
    social_alternative: Dict[str, float] = field(default_factory=dict)
    risk_stability: Dict[str, float] = field(default_factory=dict)
    institutional_smart_money: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    calculated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    factor_count: int = 0
    missing_count: int = 0
    calculation_errors: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def get_all_factors(self) -> Dict[str, float]:
        """Get flattened dict of all factors across all groups"""
        all_factors = {}
        all_factors.update(self.technical)
        all_factors.update(self.fundamental)
        all_factors.update(self.news_macro)
        all_factors.update(self.social_alternative)
        all_factors.update(self.risk_stability)
        all_factors.update(self.institutional_smart_money)
        return all_factors
    
    def count_factors(self) -> Tuple[int, int]:
        """Count total and missing factors"""
        all_factors = self.get_all_factors()
        total = len(all_factors)
        missing = sum(1 for v in all_factors.values() if v is None or (isinstance(v, float) and np.isnan(v)))
        return total, missing
    
    def get_coverage_stats(self) -> Dict[str, float]:
        """Calculate coverage statistics for each group and overall"""
        stats = {}
        
        # Technical
        tech_total = len(self.technical)
        tech_populated = sum(1 for v in self.technical.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['technical_total'] = tech_total
        stats['technical_populated'] = tech_populated
        stats['technical_coverage'] = (tech_populated / tech_total * 100) if tech_total > 0 else 0
        
        # Fundamental
        fund_total = len(self.fundamental)
        fund_populated = sum(1 for v in self.fundamental.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['fundamental_total'] = fund_total
        stats['fundamental_populated'] = fund_populated
        stats['fundamental_coverage'] = (fund_populated / fund_total * 100) if fund_total > 0 else 0
        
        # News/Macro
        news_total = len(self.news_macro)
        news_populated = sum(1 for v in self.news_macro.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['news_macro_total'] = news_total
        stats['news_macro_populated'] = news_populated
        stats['news_macro_coverage'] = (news_populated / news_total * 100) if news_total > 0 else 0
        
        # Social
        social_total = len(self.social_alternative)
        social_populated = sum(1 for v in self.social_alternative.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['social_alternative_total'] = social_total
        stats['social_alternative_populated'] = social_populated
        stats['social_alternative_coverage'] = (social_populated / social_total * 100) if social_total > 0 else 0
        
        # Risk
        risk_total = len(self.risk_stability)
        risk_populated = sum(1 for v in self.risk_stability.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['risk_stability_total'] = risk_total
        stats['risk_stability_populated'] = risk_populated
        stats['risk_stability_coverage'] = (risk_populated / risk_total * 100) if risk_total > 0 else 0
        
        # Institutional
        inst_total = len(self.institutional_smart_money)
        inst_populated = sum(1 for v in self.institutional_smart_money.values() if not (v is None or (isinstance(v, float) and np.isnan(v))))
        stats['institutional_smart_money_total'] = inst_total
        stats['institutional_smart_money_populated'] = inst_populated
        stats['institutional_smart_money_coverage'] = (inst_populated / inst_total * 100) if inst_total > 0 else 0
        
        # Overall
        overall_total = tech_total + fund_total + news_total + social_total + risk_total + inst_total
        overall_populated = tech_populated + fund_populated + news_populated + social_populated + risk_populated + inst_populated
        stats['overall_total'] = overall_total
        stats['overall_populated'] = overall_populated
        stats['overall_coverage'] = (overall_populated / overall_total * 100) if overall_total > 0 else 0
        
        return stats


# ============================================================================
# PHASE 2 CALCULATOR
# ============================================================================

class Phase2Calculator:
    """
    Phase 2: Calculate all metrics from raw Phase 1 data.
    
    Design principles:
    - Pure computation (no API calls)
    - Config-driven factor-to-group assignment
    - Graceful degradation (missing data → NaN)
    - Comprehensive logging
    """
    
    def __init__(self):
        """Initialize calculator with factor-to-group mapping"""
        self.logger = logger
        self.monitor = FactorMonitor()
        self._load_factor_mapping()
    
    def _load_factor_mapping(self):
        """Load factor-to-group mapping from config"""
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'factor_to_group.yaml'
            
            if not config_path.exists():
                self.logger.error(f"Factor mapping config not found: {config_path}")
                self.factor_mapping = {}
                return
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Build reverse mapping: factor_name -> group_name
            self.factor_mapping = {}
            for group_name in ['technical', 'fundamental', 'news_macro', 
                              'social_alternative', 'risk_stability', 
                              'institutional_smart_money']:
                if group_name in config:
                    for factor_name in config[group_name]:
                        if factor_name in self.factor_mapping:
                            self.logger.warning(f"Duplicate factor {factor_name} - already in {self.factor_mapping[factor_name]}")
                        self.factor_mapping[factor_name] = group_name
            
            # Set monitor's group mapping
            self.monitor.set_group_mapping(self.factor_mapping)
            
            self.logger.info(f"[SUCCESS] Loaded factor mapping: {len(self.factor_mapping)} factors across 6 groups")
            
        except Exception as e:
            self.logger.error(f"Failed to load factor mapping: {e}")
            self.factor_mapping = {}
    
    def calculate_all_factors(self, 
                             ticker: str,
                             raw_data: RawYFinanceData,
                             reddit_data: Optional[Dict] = None,
                             news_data: Optional[Any] = None,
                             market_data: Optional[Any] = None) -> GroupFactors:
        """
        Calculate all factors for a ticker from raw Phase 1 data.
        
        Args:
            ticker: Ticker symbol
            raw_data: RawYFinanceData from Phase 1
            reddit_data: Optional reddit data from Phase 1
            news_data: Optional NewsBundle from Phase 1
            market_data: Optional market-wide data (SPY, VIX, Treasuries)
            
        Returns:
            GroupFactors with all computed factors organized by group
        """
        self.logger.info(f"[STATS] Calculating factors for {ticker}")
        start_time = datetime.now()
        
        # Initialize result
        result = GroupFactors(ticker=ticker)
        
        try:
            # Calculate each category
            technical_factors = self._calculate_technical(raw_data)
            fundamental_factors = self._calculate_fundamental(raw_data)
            news_macro_factors = self._calculate_news_macro(raw_data, news_data, market_data, reddit_data)
            social_factors = self._calculate_social(reddit_data)
            risk_factors = self._calculate_risk(raw_data, market_data)
            institutional_factors = self._calculate_institutional(raw_data)
            
            # Assign to groups
            result.technical = technical_factors
            result.fundamental = fundamental_factors
            result.news_macro = news_macro_factors
            result.social_alternative = social_factors
            result.risk_stability = risk_factors
            result.institutional_smart_money = institutional_factors
            
            # Update counts
            result.factor_count, result.missing_count = result.count_factors()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            coverage = ((result.factor_count - result.missing_count) / result.factor_count * 100 
                       if result.factor_count > 0 else 0)
            
            self.logger.info(f"[SUCCESS] {ticker}: Calculated {result.factor_count} factors "
                           f"({coverage:.1f}% coverage) in {elapsed:.2f}s")
            
            emit_metric("phase2.calculate.success", 1, 
                       tags={'ticker': ticker, 'coverage': coverage})
            
            return result
            
        except Exception as e:
            self.logger.error(f"[ERROR] {ticker}: Factor calculation failed: {e}")
            result.calculation_errors['fatal'] = str(e)
            emit_metric("phase2.calculate.error", 1, tags={'ticker': ticker})
            return result
    
    def calculate_batch(self, 
                       raw_cache_by_ticker: Dict[str, RawYFinanceData],
                       reddit_data: Optional[Dict] = None,
                       news_data_by_ticker: Optional[Dict] = None,
                       market_data: Optional[Any] = None) -> Dict[str, GroupFactors]:
        """
        Calculate factors for multiple tickers in batch.
        
        Args:
            raw_cache_by_ticker: Dict mapping ticker -> RawYFinanceData
            reddit_data: Optional reddit data with ticker_mentions
            news_data_by_ticker: Optional dict mapping ticker -> NewsBundle
            market_data: Optional market-wide data (SPY, VIX, Treasuries)
            
        Returns:
            Dict mapping ticker -> GroupFactors
        """
        self.logger.info(f"[STATS] Calculating factors for {len(raw_cache_by_ticker)} tickers")
        batch_start = datetime.now()
        
        # Reset monitor for new batch
        self.monitor = FactorMonitor()
        self.monitor.set_group_mapping(self.factor_mapping)
        
        # Store market data for use in factor calculations
        self.market_data = market_data
        
        results = {}
        
        for ticker, raw_data in raw_cache_by_ticker.items():
            # Get ticker-specific context data
            ticker_reddit = None
            if reddit_data and 'ticker_mentions' in reddit_data:
                ticker_reddit = reddit_data['ticker_mentions'].get(ticker)
            
            ticker_news = None
            if news_data_by_ticker:
                ticker_news = news_data_by_ticker.get(ticker)
            
            # Calculate
            group_factors = self.calculate_all_factors(
                ticker=ticker,
                raw_data=raw_data,
                reddit_data=ticker_reddit,
                news_data=ticker_news,
                market_data=market_data
            )
            
            # Track factor success/failure in monitor
            self._track_factors_in_monitor(group_factors)
            
            results[ticker] = group_factors
        
        elapsed = (datetime.now() - batch_start).total_seconds()
        self.logger.info(f"[SUCCESS] Batch calculation complete: {len(results)} tickers in {elapsed:.2f}s")
        
        # Generate monitoring report
        self.logger.info("\n")
        monitoring_report = self.monitor.report(min_success_rate=0.7)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor.save_report(f"logs/factor_monitoring_{timestamp}.json")
        
        # Show recommendations
        recommendations = self.monitor.get_recommendations()
        if recommendations:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("IMPROVEMENT RECOMMENDATIONS")
            self.logger.info("=" * 80)
            for factor, issue, recommendation in recommendations[:10]:  # Top 10
                self.logger.info(f"⚠️  {factor}")
                self.logger.info(f"   Issue: {issue}")
                self.logger.info(f"   Recommendation: {recommendation}")
                self.logger.info("")
        
        return results
    
    def _track_factors_in_monitor(self, group_factors: GroupFactors):
        """Track factor calculations in the monitor"""
        all_factors = {}
        
        # Collect all factors from all groups
        for group_name in ['technical', 'fundamental', 'news_macro',
                          'social_alternative', 'risk_stability', 
                          'institutional_smart_money']:
            group_dict = getattr(group_factors, group_name, {})
            all_factors.update(group_dict)
        
        # Record success/failure for each factor
        for factor_name, value in all_factors.items():
            if value is not None and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                self.monitor.record_success(factor_name)
            else:
                if value is None:
                    self.monitor.record_failure(factor_name, "returned_none")
                elif isinstance(value, float) and np.isnan(value):
                    self.monitor.record_failure(factor_name, "nan_value")
                elif isinstance(value, float) and np.isinf(value):
                    self.monitor.record_failure(factor_name, "inf_value")
                else:
                    self.monitor.record_failure(factor_name, "invalid_value")


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
    
    def _calculate_technical(self, raw_data: RawYFinanceData) -> Dict[str, float]:
        """Calculate technical indicators from price/volume data"""
        factors = {}
        
        try:
            hist = raw_data.history
            if hist.empty:
                self.logger.warning(f"{raw_data.ticker}: No history data for technical calculation")
                return self._empty_technical_factors()
            
            # Ensure we have required columns
            required_cols = ['Close', 'High', 'Low', 'Volume']
            if not all(col in hist.columns for col in required_cols):
                self.logger.warning(f"{raw_data.ticker}: Missing required columns for technical")
                return self._empty_technical_factors()
            
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
            
            # Price momentum (simple % changes)
            factors['price_1d_pct'] = self._safe_pct_change(close, 1)
            factors['price_7d_pct'] = self._safe_pct_change(close, 7)
            factors['price_30d_pct'] = self._safe_pct_change(close, 30)
            factors['momentum_30d_pct'] = self._safe_pct_change(close, 30)
            factors['momentum_60d_pct'] = self._safe_pct_change(close, 60)
            factors['momentum_90d_pct'] = self._safe_pct_change(close, 90)
            
            # RSI (14-day)
            rsi = self._calculate_rsi(close, 14)
            factors['rsi_14'] = rsi
            factors['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
            factors['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            
            # MACD
            macd_val, macd_signal, macd_hist = self._calculate_macd(close)
            factors['macd_value'] = macd_val
            factors['macd_signal'] = macd_signal
            factors['macd_hist'] = macd_hist
            factors['macd_bullish'] = 1.0 if macd_hist > 0 else 0.0
            
            # Moving averages
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            ema_12 = close.ewm(span=12).mean().iloc[-1] if len(close) >= 12 else np.nan
            ema_26 = close.ewm(span=26).mean().iloc[-1] if len(close) >= 26 else np.nan
            
            factors['sma_20'] = sma_20
            factors['sma_50'] = sma_50
            factors['sma_200'] = sma_200
            factors['ema_12'] = ema_12
            factors['ema_26'] = ema_26
            
            current_price = close.iloc[-1]
            factors['price_vs_sma_50_pct'] = ((current_price - sma_50) / sma_50 * 100) if not np.isnan(sma_50) else np.nan
            factors['price_vs_sma_200_pct'] = ((current_price - sma_200) / sma_200 * 100) if not np.isnan(sma_200) else np.nan
            factors['sma_50_above_200'] = 1.0 if (not np.isnan(sma_50) and not np.isnan(sma_200) and sma_50 > sma_200) else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width, bb_pos = self._calculate_bollinger_bands(close, 20, 2)
            factors['bb_upper'] = bb_upper
            factors['bb_middle'] = bb_middle
            factors['bb_lower'] = bb_lower
            factors['bb_width'] = bb_width
            factors['bb_position'] = bb_pos
            
            # Volume
            vol_20d_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else np.nan
            factors['volume_20d_avg'] = vol_20d_avg
            factors['volume_spike_ratio'] = volume.iloc[-1] / vol_20d_avg if not np.isnan(vol_20d_avg) and vol_20d_avg > 0 else np.nan
            factors['volume_price_correlation'] = close.tail(60).corr(volume.tail(60)) if len(close) >= 60 else np.nan
            factors['adv_20d_usd'] = (close * volume).rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
            
            # Price levels
            high_52w = high.tail(252).max() if len(high) >= 252 else high.max()
            low_52w = low.tail(252).min() if len(low) >= 252 else low.min()
            factors['off_high_52w_pct'] = ((current_price - high_52w) / high_52w * 100) if high_52w > 0 else np.nan
            factors['off_low_52w_pct'] = ((current_price - low_52w) / low_52w * 100) if low_52w > 0 else np.nan
            factors['intraday_range_pct'] = ((high.iloc[-1] - low.iloc[-1]) / low.iloc[-1] * 100) if low.iloc[-1] > 0 else np.nan
            
            # ATR (Average True Range)
            atr_14 = self._calculate_atr(hist, 14)
            factors['atr_14'] = atr_14
            factors['atr_14_norm'] = atr_14 / current_price if current_price > 0 else np.nan
            
            # === NEW TECHNICAL SIGNALS ===
            
            # 1. Momentum Consistency (3-month) - % of up-days in last 63 trading days
            if len(close) >= 63:
                returns = close.tail(63).pct_change()
                up_days = (returns > 0).sum()
                factors['momentum_consistency_3m'] = (up_days / 63) * 100
            else:
                factors['momentum_consistency_3m'] = np.nan
            
            # 2. Breakout Flag (20-day) - Close > 20d max AND volume > 1.5× avg
            if len(close) >= 20 and len(volume) >= 20:
                max_20d = close.tail(21).iloc[:-1].max()  # Exclude current day
                avg_volume_20d = volume.tail(21).iloc[:-1].mean()
                
                is_price_breakout = close.iloc[-1] > max_20d
                is_volume_surge = volume.iloc[-1] > (avg_volume_20d * 1.5)
                
                factors['breakout_flag_20d'] = 1.0 if (is_price_breakout and is_volume_surge) else 0.0
            else:
                factors['breakout_flag_20d'] = 0.0
            
            # 3. Volatility Contraction Rank - Compare recent ATR to long-term average
            if len(hist) >= 252:
                # Calculate ATR for entire history
                high = hist['High']
                low = hist['Low']
                close = hist['Close']
                prev_close = close.shift(1)
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.rolling(window=14).mean()
                
                # Compare recent ATR (last 20 days) to long-term rolling mean
                rolling_atr_252 = atr.rolling(window=252).mean()
                
                if not rolling_atr_252.dropna().empty and len(atr) >= 20:
                    recent_atr = atr.iloc[-20:].mean()
                    longterm_atr = rolling_atr_252.iloc[-1]
                    
                    if pd.notna(recent_atr) and pd.notna(longterm_atr) and longterm_atr > 0:
                        # Contraction = (long-term - recent) / long-term
                        # Positive = volatility is contracting (good for breakouts)
                        contraction = (longterm_atr - recent_atr) / longterm_atr
                        factors['volatility_contraction_rank'] = contraction
                    else:
                        factors['volatility_contraction_rank'] = np.nan
                else:
                    factors['volatility_contraction_rank'] = np.nan
            else:
                factors['volatility_contraction_rank'] = np.nan
            
            # 4. Intraday Reversal Strength = (Close − Low) / (High − Low)
            if high.iloc[-1] > low.iloc[-1]:
                factors['intraday_reversal_strength'] = ((close.iloc[-1] - low.iloc[-1]) / 
                                                         (high.iloc[-1] - low.iloc[-1]))
            else:
                factors['intraday_reversal_strength'] = 0.5  # Mid-point if no range
            
            # 5. Volume Dry-Up Ratio = 5d avg / 60d avg volume
            if len(volume) >= 60:
                vol_5d = volume.tail(5).mean()
                vol_60d = volume.tail(60).mean()
                factors['volume_dryup_ratio'] = (vol_5d / vol_60d) if vol_60d > 0 else np.nan
            else:
                factors['volume_dryup_ratio'] = np.nan
            
            # 6. Gap Strength (30-day) - Mean |open gaps| / Close
            if len(hist) >= 30 and 'Open' in hist.columns:
                open_prices = hist['Open'].tail(30)
                prev_close = close.shift(1).tail(30)
                
                # Calculate gaps
                gaps = abs((open_prices - prev_close) / prev_close).dropna()
                
                if len(gaps) > 0:
                    factors['gap_strength_30d'] = gaps.mean() * 100  # As percentage
                else:
                    factors['gap_strength_30d'] = np.nan
            else:
                factors['gap_strength_30d'] = np.nan
            
        except Exception as e:
            self.logger.error(f"{raw_data.ticker}: Technical calculation error: {e}")
            factors = self._empty_technical_factors()
        
        return factors
    
    def _empty_technical_factors(self) -> Dict[str, float]:
        """Return dict with all technical factors set to NaN"""
        return {
            'price_1d_pct': np.nan, 'price_7d_pct': np.nan, 'price_30d_pct': np.nan,
            'momentum_30d_pct': np.nan, 'momentum_60d_pct': np.nan, 'momentum_90d_pct': np.nan,
            'rsi_14': np.nan, 'rsi_overbought': np.nan, 'rsi_oversold': np.nan,
            'macd_value': np.nan, 'macd_signal': np.nan, 'macd_hist': np.nan, 'macd_bullish': np.nan,
            'sma_20': np.nan, 'sma_50': np.nan, 'sma_200': np.nan, 'ema_12': np.nan, 'ema_26': np.nan,
            'price_vs_sma_50_pct': np.nan, 'price_vs_sma_200_pct': np.nan, 'sma_50_above_200': np.nan,
            'bb_upper': np.nan, 'bb_middle': np.nan, 'bb_lower': np.nan, 'bb_width': np.nan, 'bb_position': np.nan,
            'volume_20d_avg': np.nan, 'volume_spike_ratio': np.nan, 'volume_price_correlation': np.nan, 'adv_20d_usd': np.nan,
            'off_high_52w_pct': np.nan, 'off_low_52w_pct': np.nan, 'intraday_range_pct': np.nan,
            'atr_14': np.nan, 'atr_14_norm': np.nan,
            # NEW TECHNICAL SIGNALS
            'momentum_consistency_3m': np.nan, 'breakout_flag_20d': 0.0, 'volatility_contraction_rank': np.nan,
            'intraday_reversal_strength': 0.5, 'volume_dryup_ratio': np.nan, 'gap_strength_30d': np.nan
        }


# ============================================================================
# TECHNICAL HELPERS
# ============================================================================
    
    def _safe_pct_change(self, series: pd.Series, periods: int) -> float:
        """Safely calculate percent change over N periods"""
        try:
            if len(series) <= periods:
                return np.nan
            current = series.iloc[-1]
            past = series.iloc[-periods-1]
            if past == 0:
                return np.nan
            return ((current - past) / past) * 100
        except:
            return np.nan
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if len(prices) < period + 1:
                return np.nan
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return np.nan
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow + signal:
                return np.nan, np.nan, np.nan
            
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        except:
            return np.nan, np.nan, np.nan
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2) -> Tuple[float, float, float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            current_price = prices.iloc[-1]
            bb_upper = upper.iloc[-1]
            bb_lower = lower.iloc[-1]
            bb_middle = sma.iloc[-1]
            bb_width = bb_upper - bb_lower
            
            # Position in band (0 = at lower, 1 = at upper)
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) if bb_width > 0 else 0.5
            
            return bb_upper, bb_middle, bb_lower, bb_width, bb_position
        except:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR (Average True Range)"""
        try:
            if len(hist) < period + 1:
                return np.nan
            
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1]
        except:
            return np.nan


# ============================================================================
# FUNDAMENTAL INDICATORS
# ============================================================================
    
    def _calculate_fundamental(self, raw_data: RawYFinanceData) -> Dict[str, float]:
        """Calculate fundamental metrics from financial statements"""
        factors = {}
        
        try:
            info = raw_data.info
            if not info:
                self.logger.warning(f"{raw_data.ticker}: No info data for fundamental calculation")
                return self._empty_fundamental_factors()
            
            # Valuation ratios (from info)
            factors['pe_ratio'] = float(info.get('trailingPE', np.nan))
            factors['forward_pe'] = float(info.get('forwardPE', np.nan))
            
            # PEG Ratio - IMPROVED with better fallback logic and EPS trend data
            # PEG = PE / Annual EPS Growth Rate (as percentage)
            # Example: PE of 20, growth of 15% → PEG = 20/15 = 1.33
            peg_from_info = info.get('pegRatio')
            if peg_from_info and not np.isnan(float(peg_from_info)) and float(peg_from_info) > 0:
                factors['peg_ratio'] = float(peg_from_info)
            else:
                # Calculate: PE / (Earnings Growth Rate as %)
                pe = factors['pe_ratio']
                earnings_growth_pct = None
                
                # Try multiple sources for earnings growth
                # 1. earningsGrowth from info
                earnings_growth = info.get('earningsGrowth')
                if earnings_growth and not np.isnan(float(earnings_growth)) and abs(float(earnings_growth)) > 0.01:
                    earnings_growth_pct = float(earnings_growth) * 100
                
                # 2. earningsQuarterlyGrowth from info
                if not earnings_growth_pct:
                    earnings_growth = info.get('earningsQuarterlyGrowth')
                    if earnings_growth and not np.isnan(float(earnings_growth)) and abs(float(earnings_growth)) > 0.01:
                        earnings_growth_pct = float(earnings_growth) * 100 * 4  # Annualize quarterly
                
                # 3. Calculate from EPS trend data
                if not earnings_growth_pct:
                    try:
                        eps_trend = raw_data.ticker_obj.eps_trend if hasattr(raw_data, 'ticker_obj') else None
                        if eps_trend is not None and not eps_trend.empty and '0y' in eps_trend.index:
                            current_eps = eps_trend.loc['0y', 'current']
                            past_eps = eps_trend.loc['0y', '90daysAgo']
                            if pd.notna(current_eps) and pd.notna(past_eps) and past_eps > 0:
                                # Annualize the 90-day change
                                quarterly_growth = (current_eps - past_eps) / past_eps
                                earnings_growth_pct = quarterly_growth * (365/90) * 100
                    except Exception:
                        pass
                
                # 4. Use revenueGrowth as last resort
                if not earnings_growth_pct:
                    earnings_growth = info.get('revenueGrowth')
                    if earnings_growth and not np.isnan(float(earnings_growth)) and abs(float(earnings_growth)) > 0.01:
                        earnings_growth_pct = float(earnings_growth) * 100
                
                # Calculate PEG if we have both PE and growth
                if not np.isnan(pe) and pe > 0 and earnings_growth_pct and abs(earnings_growth_pct) > 1.0:
                    factors['peg_ratio'] = pe / abs(earnings_growth_pct)
                else:
                    factors['peg_ratio'] = np.nan
            
            factors['pb_ratio'] = float(info.get('priceToBook', np.nan))
            factors['ps_ratio'] = float(info.get('priceToSalesTrailing12Months', np.nan))
            factors['ev_ebitda'] = float(info.get('enterpriseToEbitda', np.nan))
            factors['ev_sales'] = float(info.get('enterpriseToRevenue', np.nan))
            
            # Calculate EV/FCF and P/FCF if we have the data
            market_cap = float(info.get('marketCap', np.nan))
            free_cash_flow = float(info.get('freeCashflow', np.nan))
            enterprise_value = float(info.get('enterpriseValue', np.nan))
            
            if not np.isnan(enterprise_value) and not np.isnan(free_cash_flow) and free_cash_flow != 0:
                factors['ev_fcf'] = enterprise_value / free_cash_flow
            else:
                factors['ev_fcf'] = np.nan
            
            if not np.isnan(market_cap) and not np.isnan(free_cash_flow) and free_cash_flow != 0:
                factors['p_fcf'] = market_cap / free_cash_flow
            else:
                factors['p_fcf'] = np.nan
            
            # Earnings yield (inverse of PE)
            pe = factors['pe_ratio']
            factors['earnings_yield'] = (1 / pe * 100) if not np.isnan(pe) and pe > 0 else np.nan
            
            # Profitability margins
            factors['gross_margin'] = float(info.get('grossMargins', np.nan)) * 100 if info.get('grossMargins') else np.nan
            factors['operating_margin'] = float(info.get('operatingMargins', np.nan)) * 100 if info.get('operatingMargins') else np.nan
            factors['net_margin'] = float(info.get('profitMargins', np.nan)) * 100 if info.get('profitMargins') else np.nan
            factors['ebitda_margin'] = float(info.get('ebitdaMargins', np.nan)) * 100 if info.get('ebitdaMargins') else np.nan
            
            # Calculate FCF margin if possible
            revenue = float(info.get('totalRevenue', np.nan))
            if not np.isnan(free_cash_flow) and not np.isnan(revenue) and revenue > 0:
                factors['fcf_margin'] = (free_cash_flow / revenue) * 100
            else:
                factors['fcf_margin'] = np.nan
            
            # Returns
            factors['roe'] = float(info.get('returnOnEquity', np.nan)) * 100 if info.get('returnOnEquity') else np.nan
            factors['roa'] = float(info.get('returnOnAssets', np.nan)) * 100 if info.get('returnOnAssets') else np.nan
            
            # Calculate ROIC: NOPAT / (Debt + Equity)
            # ROIC = EBIT * (1 - Tax Rate) / Invested Capital
            # Get EBIT from income statement (not in info dict)
            ebit = np.nan
            income_stmt = raw_data.income_stmt
            if income_stmt is not None and not income_stmt.empty and 'EBIT' in income_stmt.index:
                ebit = float(income_stmt.loc['EBIT'].iloc[0])
            
            # Get total debt and equity
            total_debt = float(info.get('totalDebt', 0))
            
            # Get stockholder equity from balance sheet
            stockholder_equity = np.nan
            balance_sheet = raw_data.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                if 'Stockholders Equity' in balance_sheet.index:
                    stockholder_equity = float(balance_sheet.loc['Stockholders Equity'].iloc[0])
                elif 'Common Stock Equity' in balance_sheet.index:
                    stockholder_equity = float(balance_sheet.loc['Common Stock Equity'].iloc[0])
            
            if not np.isnan(ebit) and not np.isnan(stockholder_equity):
                total_capital = total_debt + stockholder_equity
                if total_capital > 0:
                    # Approximate ROIC (without exact tax rate)
                    tax_rate = 0.21  # Approximate US corporate tax rate
                    nopat = ebit * (1 - tax_rate)
                    factors['roic'] = (nopat / total_capital) * 100
                else:
                    factors['roic'] = np.nan
            else:
                factors['roic'] = np.nan
            
            # Calculate ROCE: EBIT / (Total Assets - Current Liabilities)
            # ROCE = EBIT / Capital Employed
            # Get total assets and current liabilities from balance sheet
            total_assets = np.nan
            total_current_liabilities = np.nan
            if balance_sheet is not None and not balance_sheet.empty:
                if 'Total Assets' in balance_sheet.index:
                    total_assets = float(balance_sheet.loc['Total Assets'].iloc[0])
                if 'Current Liabilities' in balance_sheet.index:
                    total_current_liabilities = float(balance_sheet.loc['Current Liabilities'].iloc[0])
            
            if not np.isnan(ebit) and not np.isnan(total_assets) and not np.isnan(total_current_liabilities):
                capital_employed = total_assets - total_current_liabilities
                if capital_employed > 0:
                    factors['roce'] = (ebit / capital_employed) * 100
                else:
                    factors['roce'] = np.nan
            else:
                factors['roce'] = np.nan
            
            # Growth rates
            factors['revenue_growth_yoy'] = float(info.get('revenueGrowth', np.nan)) * 100 if info.get('revenueGrowth') else np.nan
            
            # Revenue Growth QoQ - calculate from income statements if available
            quarterly_revenue_growth = info.get('quarterlyRevenueGrowth')
            if quarterly_revenue_growth:
                factors['revenue_growth_qoq'] = float(quarterly_revenue_growth) * 100
            else:
                # Try to calculate from income_stmt
                income_stmt = raw_data.income_stmt
                if income_stmt is not None and not income_stmt.empty and 'Total Revenue' in income_stmt.index:
                    revenues = income_stmt.loc['Total Revenue']
                    if len(revenues) >= 2:
                        # QoQ = (Q_latest - Q_previous) / Q_previous
                        latest = revenues.iloc[0]
                        previous = revenues.iloc[1]
                        if pd.notna(latest) and pd.notna(previous) and previous != 0:
                            factors['revenue_growth_qoq'] = ((latest - previous) / abs(previous)) * 100
                        else:
                            factors['revenue_growth_qoq'] = np.nan
                    else:
                        factors['revenue_growth_qoq'] = np.nan
                else:
                    factors['revenue_growth_qoq'] = np.nan
            
            factors['earnings_growth_yoy'] = float(info.get('earningsGrowth', np.nan)) * 100 if info.get('earningsGrowth') else np.nan
            factors['earnings_growth_qoq'] = float(info.get('earningsQuarterlyGrowth', np.nan)) * 100 if info.get('earningsQuarterlyGrowth') else np.nan
            
            # EPS growth (use earnings growth as proxy)
            factors['eps_growth_yoy'] = factors['earnings_growth_yoy']
            factors['eps_growth_qoq'] = factors['earnings_growth_qoq']
            
            # FCF Growth YoY - calculate from cashflow statements
            cashflow = raw_data.cashflow
            if cashflow is not None and not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                fcf_series = cashflow.loc['Free Cash Flow']
                if len(fcf_series) >= 2:
                    # YoY = (Latest - Year Ago) / Year Ago
                    latest_fcf = fcf_series.iloc[0]
                    year_ago_fcf = fcf_series.iloc[1] if len(fcf_series) > 1 else np.nan
                    if pd.notna(latest_fcf) and pd.notna(year_ago_fcf) and year_ago_fcf != 0:
                        factors['fcf_growth_yoy'] = ((latest_fcf - year_ago_fcf) / abs(year_ago_fcf)) * 100
                    else:
                        factors['fcf_growth_yoy'] = np.nan
                else:
                    factors['fcf_growth_yoy'] = np.nan
            else:
                factors['fcf_growth_yoy'] = np.nan
            
            # Financial health
            factors['current_ratio'] = float(info.get('currentRatio', np.nan))
            factors['quick_ratio'] = float(info.get('quickRatio', np.nan))
            factors['debt_to_equity'] = float(info.get('debtToEquity', np.nan))
            
            # Debt to Assets - calculate using balance sheet total assets
            if not np.isnan(total_debt) and not np.isnan(total_assets) and total_assets > 0:
                factors['debt_to_assets'] = (total_debt / total_assets) * 100
            else:
                factors['debt_to_assets'] = np.nan
            
            # Interest Coverage: EBIT / Interest Expense
            # Both EBIT and Interest Expense need to come from income statement
            interest_expense = np.nan
            if income_stmt is not None and not income_stmt.empty and 'Interest Expense' in income_stmt.index:
                interest_expense = float(income_stmt.loc['Interest Expense'].iloc[0])
            
            if not np.isnan(ebit) and not np.isnan(interest_expense) and interest_expense != 0:
                factors['interest_coverage'] = ebit / abs(interest_expense)
            else:
                factors['interest_coverage'] = np.nan
            
            # Cash to debt
            cash = float(info.get('totalCash', np.nan))
            if not np.isnan(cash) and not np.isnan(total_debt) and total_debt > 0:
                factors['cash_to_debt'] = cash / total_debt
            else:
                factors['cash_to_debt'] = np.nan
            
            # Efficiency ratios - calculate from financial statements
            # Asset Turnover = Revenue / Average Total Assets
            # Get revenue from income statement
            revenue = np.nan
            if income_stmt is not None and not income_stmt.empty and 'Total Revenue' in income_stmt.index:
                revenue = float(income_stmt.loc['Total Revenue'].iloc[0])
            elif not np.isnan(float(info.get('totalRevenue', np.nan))):
                revenue = float(info.get('totalRevenue', np.nan))
                
            if not np.isnan(revenue) and not np.isnan(total_assets) and total_assets > 0:
                factors['asset_turnover'] = revenue / total_assets
            else:
                factors['asset_turnover'] = np.nan
            
            # Inventory Turnover = COGS / Inventory - IMPROVED with better field matching
            if balance_sheet is not None and not balance_sheet.empty and income_stmt is not None and not income_stmt.empty:
                # Try multiple field names for inventory
                inventory = None
                for inv_field in ['Inventory', 'Inventories', 'Total Inventory', 'Net Inventory']:
                    if inv_field in balance_sheet.index:
                        inv_value = balance_sheet.loc[inv_field].iloc[0]
                        if pd.notna(inv_value) and float(inv_value) > 0:
                            inventory = float(inv_value)
                            break
                
                # Try multiple field names for COGS - EXPANDED
                cost_of_revenue = None
                for cogs_field in ['Cost Of Revenue', 'CostOfRevenue', 'Reconciled Cost Of Revenue',
                                   'Cost of Goods Sold', 'COGS', 'Cost Of Goods Sold']:
                    if cogs_field in income_stmt.index:
                        cogs_value = income_stmt.loc[cogs_field].iloc[0]
                        if pd.notna(cogs_value):
                            cost_of_revenue = abs(float(cogs_value))  # Ensure positive
                            break
                
                # Calculate if both available
                if inventory and inventory > 0 and cost_of_revenue and cost_of_revenue > 0:
                    factors['inventory_turnover'] = cost_of_revenue / inventory
                else:
                    # For companies without inventory (services), set to NaN gracefully
                    factors['inventory_turnover'] = np.nan
            else:
                factors['inventory_turnover'] = np.nan
            
            # Receivables Turnover = Revenue / Average Receivables
            if balance_sheet is not None and not balance_sheet.empty:
                receivables = balance_sheet.loc['Receivables'].iloc[0] if 'Receivables' in balance_sheet.index else np.nan
                
                if pd.notna(receivables) and not np.isnan(revenue) and float(receivables) > 0:
                    factors['receivables_turnover'] = revenue / float(receivables)
                else:
                    factors['receivables_turnover'] = np.nan
            else:
                factors['receivables_turnover'] = np.nan
            
            # Per-share metrics
            factors['book_value_per_share'] = float(info.get('bookValue', np.nan))
            
            # Tangible Book Value Per Share = (Tangible Book Value) / Shares Outstanding
            # Get tangible book value from balance sheet
            tangible_book_value = np.nan
            if balance_sheet is not None and not balance_sheet.empty and 'Tangible Book Value' in balance_sheet.index:
                tangible_book_value = float(balance_sheet.loc['Tangible Book Value'].iloc[0])
            
            shares_outstanding = float(info.get('sharesOutstanding', np.nan))
            if not np.isnan(tangible_book_value) and not np.isnan(shares_outstanding) and shares_outstanding > 0:
                factors['tangible_book_per_share'] = tangible_book_value / shares_outstanding
            else:
                factors['tangible_book_per_share'] = np.nan
            if not np.isnan(free_cash_flow) and not np.isnan(shares_outstanding) and shares_outstanding > 0:
                factors['fcf_per_share'] = free_cash_flow / shares_outstanding
            else:
                factors['fcf_per_share'] = np.nan
            
            # === NEW FUNDAMENTAL SIGNALS ===
            
            # 1. Gross Profit to Assets = Gross Profit / Total Assets
            if income_stmt is not None and not income_stmt.empty and not np.isnan(total_assets) and total_assets > 0:
                gross_profit = None
                for gp_field in ['Gross Profit', 'GrossProfit']:
                    if gp_field in income_stmt.index:
                        gp_value = income_stmt.loc[gp_field].iloc[0]
                        if pd.notna(gp_value):
                            gross_profit = float(gp_value)
                            break
                
                if gross_profit:
                    factors['gross_profit_to_assets'] = (gross_profit / total_assets) * 100
                else:
                    factors['gross_profit_to_assets'] = np.nan
            else:
                factors['gross_profit_to_assets'] = np.nan
            
            # 2. Accruals Ratio = (Net Income - Operating Cash Flow) / Total Assets
            if income_stmt is not None and not income_stmt.empty and raw_data.cashflow is not None:
                net_income = None
                if 'Net Income' in income_stmt.index:
                    ni_value = income_stmt.loc['Net Income'].iloc[0]
                    if pd.notna(ni_value):
                        net_income = float(ni_value)
                
                operating_cf = None
                if not raw_data.cashflow.empty and 'Operating Cash Flow' in raw_data.cashflow.index:
                    ocf_value = raw_data.cashflow.loc['Operating Cash Flow'].iloc[0]
                    if pd.notna(ocf_value):
                        operating_cf = float(ocf_value)
                
                if net_income is not None and operating_cf is not None and not np.isnan(total_assets) and total_assets > 0:
                    factors['accruals_ratio'] = ((net_income - operating_cf) / total_assets) * 100
                else:
                    factors['accruals_ratio'] = np.nan
            else:
                factors['accruals_ratio'] = np.nan
            
            # 3. Interest Burden = EBIT / Interest Expense
            if not np.isnan(ebit) and ebit != 0:
                interest_expense = None
                if income_stmt is not None and not income_stmt.empty:
                    for int_field in ['Interest Expense', 'InterestExpense']:
                        if int_field in income_stmt.index:
                            int_value = income_stmt.loc[int_field].iloc[0]
                            if pd.notna(int_value) and float(int_value) != 0:
                                interest_expense = abs(float(int_value))  # Usually negative
                                break
                
                if interest_expense and interest_expense > 0:
                    factors['interest_burden'] = ebit / interest_expense
                else:
                    factors['interest_burden'] = np.nan
            else:
                factors['interest_burden'] = np.nan
            
            # 4. Profitability Trend (3-year slope from quarterly margins)
            try:
                if raw_data.quarterly_income_stmt is not None and not raw_data.quarterly_income_stmt.empty:
                    qtr_income = raw_data.quarterly_income_stmt
                    
                    # Look for profitability metrics
                    if 'Gross Profit' in qtr_income.index and 'Total Revenue' in qtr_income.index:
                        revenue = qtr_income.loc['Total Revenue']
                        gross_profit = qtr_income.loc['Gross Profit']
                        
                        # Calculate gross margin for each quarter
                        margins = (gross_profit / revenue * 100).dropna()
                        
                        # yfinance only provides ~5 quarters, so lower threshold
                        # If we have at least 4 quarters (1 year), calculate slope
                        if len(margins) >= 4:
                            # Use all available quarters (typically 4-5 from yfinance)
                            x = np.arange(len(margins))
                            y = margins.values
                            
                            # Calculate linear regression slope
                            if len(x) > 1 and not np.any(np.isnan(y)):
                                slope = np.polyfit(x, y, 1)[0]
                                factors['profitability_trend_3y'] = slope
                            else:
                                factors['profitability_trend_3y'] = np.nan
                        else:
                            factors['profitability_trend_3y'] = np.nan
                    else:
                        factors['profitability_trend_3y'] = np.nan
                else:
                    factors['profitability_trend_3y'] = np.nan
            except Exception as e:
                factors['profitability_trend_3y'] = np.nan
            
            # 5. CapEx to CFO = Capital Expenditures / Operating Cash Flow
            if raw_data.cashflow is not None and not raw_data.cashflow.empty:
                capex = None
                for capex_field in ['Capital Expenditure', 'Capital Expenditures', 'CapEx']:
                    if capex_field in raw_data.cashflow.index:
                        capex_value = raw_data.cashflow.loc[capex_field].iloc[0]
                        if pd.notna(capex_value):
                            capex = abs(float(capex_value))  # Usually negative
                            break
                
                operating_cf = None
                if 'Operating Cash Flow' in raw_data.cashflow.index:
                    ocf_value = raw_data.cashflow.loc['Operating Cash Flow'].iloc[0]
                    if pd.notna(ocf_value) and float(ocf_value) > 0:
                        operating_cf = float(ocf_value)
                
                if capex and operating_cf:
                    factors['capex_to_cfo'] = (capex / operating_cf) * 100
                else:
                    factors['capex_to_cfo'] = np.nan
            else:
                factors['capex_to_cfo'] = np.nan
            
            # 6. Net Debt to EBITDA = (Total Debt - Cash) / EBITDA
            ebitda = float(info.get('ebitda', np.nan))
            total_cash = float(info.get('totalCash', 0))
            if not np.isnan(ebitda) and ebitda > 0:
                net_debt = total_debt - total_cash
                factors['net_debt_to_ebitda'] = net_debt / ebitda
            else:
                factors['net_debt_to_ebitda'] = np.nan
            
            # 7. Shares Change (1-year) - calculate from quarterly balance sheet
            try:
                if raw_data.quarterly_balance_sheet is not None and not raw_data.quarterly_balance_sheet.empty:
                    qtr_bs = raw_data.quarterly_balance_sheet
                    
                    # Look for shares outstanding field
                    shares_field = None
                    for field_name in ['Share Issued', 'Ordinary Shares Number', 'Common Stock Shares Outstanding']:
                        if field_name in qtr_bs.index:
                            shares_field = field_name
                            break
                    
                    if shares_field:
                        shares_series = qtr_bs.loc[shares_field].dropna()
                        
                        # Need at least 4 quarters (1 year)
                        if len(shares_series) >= 4:
                            most_recent = shares_series.iloc[0]
                            one_year_ago = shares_series.iloc[3]  # 4 quarters ago
                            
                            if most_recent > 0 and one_year_ago > 0:
                                # Calculate percentage change
                                pct_change = ((most_recent - one_year_ago) / one_year_ago) * 100
                                factors['shares_change_1y'] = pct_change
                            else:
                                factors['shares_change_1y'] = np.nan
                        else:
                            factors['shares_change_1y'] = np.nan
                    else:
                        factors['shares_change_1y'] = np.nan
                else:
                    factors['shares_change_1y'] = np.nan
            except Exception as e:
                factors['shares_change_1y'] = np.nan
            
        except Exception as e:
            self.logger.error(f"{raw_data.ticker}: Fundamental calculation error: {e}")
            factors = self._empty_fundamental_factors()
        
        return factors
    
    def _empty_fundamental_factors(self) -> Dict[str, float]:
        """Return dict with all fundamental factors set to NaN"""
        return {
            'pe_ratio': np.nan, 'forward_pe': np.nan, 'peg_ratio': np.nan, 'pb_ratio': np.nan,
            'ps_ratio': np.nan, 'ev_ebitda': np.nan, 'ev_sales': np.nan, 'ev_fcf': np.nan,
            'p_fcf': np.nan, 'earnings_yield': np.nan,
            'gross_margin': np.nan, 'operating_margin': np.nan, 'net_margin': np.nan,
            'ebitda_margin': np.nan, 'fcf_margin': np.nan,
            'roe': np.nan, 'roa': np.nan, 'roic': np.nan, 'roce': np.nan,
            'revenue_growth_yoy': np.nan, 'revenue_growth_qoq': np.nan,
            'earnings_growth_yoy': np.nan, 'earnings_growth_qoq': np.nan,
            'eps_growth_yoy': np.nan, 'eps_growth_qoq': np.nan, 'fcf_growth_yoy': np.nan,
            'current_ratio': np.nan, 'quick_ratio': np.nan, 'debt_to_equity': np.nan,
            'debt_to_assets': np.nan, 'interest_coverage': np.nan, 'cash_to_debt': np.nan,
            'asset_turnover': np.nan, 'inventory_turnover': np.nan, 'receivables_turnover': np.nan,
            'book_value_per_share': np.nan, 'tangible_book_per_share': np.nan, 'fcf_per_share': np.nan,
            # NEW FUNDAMENTAL SIGNALS
            'gross_profit_to_assets': np.nan, 'accruals_ratio': np.nan, 'interest_burden': np.nan,
            'profitability_trend_3y': np.nan, 'capex_to_cfo': np.nan, 'net_debt_to_ebitda': np.nan,
            'shares_change_1y': np.nan
        }


# ============================================================================
# NEWS/MACRO INDICATORS
# ============================================================================
    
    def _calculate_news_macro(self, raw_data: RawYFinanceData, news_data: Optional[Any], market_data: Optional[Any] = None, reddit_data: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate news sentiment and macro indicators"""
        factors = {}
        
        try:
            # News sentiment - with fallback to yfinance news if news_data not provided
            sentiments = []
            
            if news_data and hasattr(news_data, 'articles') and news_data.articles:
                # Use NewsBundle if available
                self.logger.debug(f"Processing {len(news_data.articles)} news articles from NewsBundle")
                for article in news_data.articles:
                    if hasattr(article, 'sentiment_score') and article.sentiment_score is not None:
                        sentiments.append(article.sentiment_score)
                        self.logger.debug(f"  Article sentiment: {article.sentiment_score:.3f}")
            elif raw_data.news and len(raw_data.news) > 0:
                # Fallback: Try to get news directly from yfinance raw_data
                try:
                    self.logger.debug(f"Processing {len(raw_data.news)} news articles from yfinance")
                    # yfinance news doesn't have sentiment, so we'll use TextBlob for basic sentiment
                    from textblob import TextBlob
                    for article in raw_data.news[:10]:  # Limit to 10 most recent
                        # News structure: article['content']['title']
                        content = article.get('content', {})
                        title = content.get('title', '') if isinstance(content, dict) else ''
                        if title:
                            blob = TextBlob(title)
                            sentiment = blob.sentiment.polarity  # -1 to +1
                            sentiments.append(sentiment)
                            self.logger.debug(f"  Article '{title[:50]}...' sentiment (TextBlob): {sentiment:.3f}")
                except ImportError:
                    self.logger.debug("  TextBlob not available for news sentiment analysis")
                except Exception as e:
                    self.logger.debug(f"  Could not process yfinance news: {e}")
            
            # Calculate final sentiment
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                factors['news_sentiment'] = float(avg_sentiment)
                # Convert -1 to +1 scale to 0-100% consensus
                factors['news_sentiment_consensus'] = ((avg_sentiment + 1) / 2) * 100
                self.logger.debug(f"  Average news sentiment: {avg_sentiment:.3f} ({len(sentiments)} articles)")
            else:
                self.logger.debug("  No valid sentiment scores found")
                factors['news_sentiment'] = np.nan
                factors['news_sentiment_consensus'] = np.nan
            
            # Earnings events
            info = raw_data.info
            calendar = raw_data.calendar
            
            # Calculate days to earnings if calendar available
            if calendar and not isinstance(calendar, pd.DataFrame):
                earnings_date = calendar.get('Earnings Date')
                if earnings_date:
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        earnings_date = earnings_date[0]
                    try:
                        if isinstance(earnings_date, str):
                            earnings_dt = pd.to_datetime(earnings_date)
                        else:
                            earnings_dt = pd.to_datetime(earnings_date)
                        
                        days_to = (earnings_dt - pd.Timestamp.now()).days
                        factors['days_to_earnings'] = float(days_to)
                        factors['pre_earnings_flag'] = 1.0 if 0 <= days_to <= 5 else 0.0
                        factors['post_earnings_flag'] = 1.0 if -5 <= days_to < 0 else 0.0
                    except:
                        factors['days_to_earnings'] = np.nan
                        factors['pre_earnings_flag'] = 0.0
                        factors['post_earnings_flag'] = 0.0
                else:
                    factors['days_to_earnings'] = np.nan
                    factors['pre_earnings_flag'] = 0.0
                    factors['post_earnings_flag'] = 0.0
            else:
                factors['days_to_earnings'] = np.nan
                factors['pre_earnings_flag'] = 0.0
                factors['post_earnings_flag'] = 0.0
            
            # Earnings surprise (from earnings history)
            # Parse earnings history if available
            earnings_history = raw_data.earnings_history
            if earnings_history is not None and not earnings_history.empty:
                try:
                    # Check if we have epsEstimate and epsActual columns (yfinance uses lowercase)
                    if 'epsEstimate' in earnings_history.columns and 'epsActual' in earnings_history.columns:
                        # Get most recent earnings (first row)
                        latest = earnings_history.iloc[0]
                        estimate = latest.get('epsEstimate')
                        actual = latest.get('epsActual')
                        
                        if pd.notna(estimate) and pd.notna(actual) and estimate != 0:
                            surprise = ((actual - estimate) / abs(estimate)) * 100
                            factors['earnings_surprise_last'] = float(surprise)
                        else:
                            factors['earnings_surprise_last'] = np.nan
                        
                        # Calculate beat streak (how many quarters in a row they beat estimates)
                        beat_streak = 0
                        for idx, row in earnings_history.iterrows():
                            est = row.get('epsEstimate')
                            act = row.get('epsActual')
                            if pd.notna(est) and pd.notna(act):
                                if act >= est:
                                    beat_streak += 1
                                else:
                                    break
                            else:
                                break
                        factors['earnings_beat_streak'] = float(beat_streak)
                    else:
                        factors['earnings_surprise_last'] = np.nan
                        factors['earnings_beat_streak'] = np.nan
                except Exception as e:
                    self.logger.debug(f"{raw_data.ticker}: Could not parse earnings history: {e}")
                    factors['earnings_surprise_last'] = np.nan
                    factors['earnings_beat_streak'] = np.nan
            else:
                factors['earnings_surprise_last'] = np.nan
                factors['earnings_beat_streak'] = np.nan
            
            # Sector momentum (calculate from sector ETF)
            sector = info.get('sector', 'Unknown')
            if sector and sector != 'Unknown':
                try:
                    from backend.utils.sector_etfs import get_sector_etf
                    import yfinance as yf
                    
                    sector_etf = get_sector_etf(sector)
                    
                    # Fetch sector ETF data (30 days)
                    sector_ticker = yf.Ticker(sector_etf)
                    sector_hist = sector_ticker.history(period='1mo')
                    
                    if len(sector_hist) >= 2:
                        # Calculate sector 30-day return
                        sector_return_30d = ((sector_hist['Close'].iloc[-1] - sector_hist['Close'].iloc[0]) 
                                            / sector_hist['Close'].iloc[0]) * 100
                        factors['sector_momentum_30d'] = float(sector_return_30d)
                        
                        # Calculate stock's relative strength vs sector
                        stock_hist = raw_data.history
                        if stock_hist is not None and len(stock_hist) >= 2:
                            stock_return_30d = ((stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[0])
                                              / stock_hist['Close'].iloc[0]) * 100
                            factors['sector_relative_strength'] = float(stock_return_30d - sector_return_30d)
                        else:
                            factors['sector_relative_strength'] = np.nan
                    else:
                        factors['sector_momentum_30d'] = np.nan
                        factors['sector_relative_strength'] = np.nan
                except Exception as e:
                    self.logger.debug(f"{raw_data.ticker}: Could not calculate sector momentum: {e}")
                    factors['sector_momentum_30d'] = np.nan
                    factors['sector_relative_strength'] = np.nan
            else:
                factors['sector_momentum_30d'] = np.nan
                factors['sector_relative_strength'] = np.nan
            
            # Market regime and correlation indicators (from market_data)
            if market_data and market_data.is_valid():
                try:
                    from backend.integrations.yfinance import calculate_market_regime, calculate_spy_correlation
                    
                    # Market regime (bull/bear/neutral based on 200-day MA)
                    regime = calculate_market_regime(market_data.spy_history)
                    factors['market_regime'] = float(regime) if regime is not None else np.nan
                    
                    # SPY correlation (60-day rolling correlation with S&P 500)
                    stock_hist = raw_data.history
                    if stock_hist is not None and len(stock_hist) >= 60:
                        spy_corr = calculate_spy_correlation(stock_hist, market_data.spy_history, window=60)
                        factors['spy_correlation_60d'] = float(spy_corr) if spy_corr is not None else np.nan
                    else:
                        factors['spy_correlation_60d'] = np.nan
                        
                except Exception as e:
                    self.logger.debug(f"{raw_data.ticker}: Could not calculate market regime/correlation: {e}")
                    factors['market_regime'] = np.nan
                    factors['spy_correlation_60d'] = np.nan
            else:
                factors['market_regime'] = np.nan
                factors['spy_correlation_60d'] = np.nan
            
            # Macro indicators (from market_data)
            if market_data and market_data.is_valid():
                try:
                    # VIX level (volatility index)
                    if market_data.vix_current is not None:
                        factors['vix_level'] = float(market_data.vix_current)
                    else:
                        factors['vix_level'] = np.nan
                    
                    # 10-year Treasury yield
                    if market_data.treasury_yield_10y is not None:
                        factors['treasury_yield_10y'] = float(market_data.treasury_yield_10y)
                    else:
                        factors['treasury_yield_10y'] = np.nan
                    
                    # Credit spread (corporate bond yield - Treasury yield)
                    if market_data.credit_spread is not None:
                        factors['credit_spread'] = float(market_data.credit_spread)
                    else:
                        factors['credit_spread'] = np.nan
                    
                    # REMOVED: unemployment_rate, gdp_growth_rate, inflation_rate
                    # These require external data sources (FRED API) not available in yfinance
                        
                except Exception as e:
                    self.logger.debug(f"{raw_data.ticker}: Could not extract macro indicators: {e}")
                    factors['vix_level'] = np.nan
                    factors['treasury_yield_10y'] = np.nan
                    factors['credit_spread'] = np.nan
            else:
                factors['vix_level'] = np.nan
                factors['treasury_yield_10y'] = np.nan
                factors['credit_spread'] = np.nan
            
            # === NEW NEWS/MACRO SIGNALS ===
            
            # 1. Earnings Revision (3-month) - Net upgrades vs downgrades using eps_revisions
            try:
                # FIXED: Use raw_data.eps_revisions directly (not ticker_obj)
                eps_revisions = raw_data.eps_revisions if raw_data.eps_revisions is not None else None
                if eps_revisions is not None and not eps_revisions.empty:
                    # Use current quarter (0q) data for 30-day revisions
                    current_q = eps_revisions.loc['0q'] if '0q' in eps_revisions.index else None
                    if current_q is not None:
                        up_30d = float(current_q.get('upLast30days', 0))
                        down_30d = float(current_q.get('downLast30days', 0))
                        factors['earnings_revision_3m'] = up_30d - down_30d
                        self.logger.debug(f"  Earnings revision (3m): up={up_30d}, down={down_30d}, net={up_30d - down_30d}")
                    else:
                        factors['earnings_revision_3m'] = np.nan
                else:
                    factors['earnings_revision_3m'] = np.nan
            except Exception as e:
                self.logger.debug(f"  Earnings revision calculation error: {e}")
                factors['earnings_revision_3m'] = np.nan
            
            # 2. EPS Surprise Std Dev (4-quarter) - Volatility of earnings surprises
            if earnings_history is not None and not earnings_history.empty and len(earnings_history) >= 4:
                try:
                    surprises = []
                    for idx, row in earnings_history.head(4).iterrows():
                        est = row.get('epsEstimate')
                        act = row.get('epsActual')
                        if pd.notna(est) and pd.notna(act) and est != 0:
                            surprise_pct = ((act - est) / abs(est)) * 100
                            surprises.append(surprise_pct)
                    
                    if len(surprises) >= 2:
                        factors['eps_surprise_std_4q'] = np.std(surprises)
                    else:
                        factors['eps_surprise_std_4q'] = np.nan
                except Exception as e:
                    self.logger.debug(f"  EPS surprise std calculation error: {e}")
                    factors['eps_surprise_std_4q'] = np.nan
            else:
                factors['eps_surprise_std_4q'] = np.nan
            
            # 3. Post-Earnings Drift (21-day return after last earnings)
            if earnings_history is not None and not earnings_history.empty:
                try:
                    # Most recent earnings date is in the index
                    last_earnings_date = earnings_history.index[-1]  # Most recent (last in chronological order)
                    stock_hist = raw_data.history
                    
                    if stock_hist is not None and not stock_hist.empty:
                        # FIXED: Convert BOTH to timezone-naive for comparison
                        # Remove timezone from earnings date if present
                        if hasattr(last_earnings_date, 'tz') and last_earnings_date.tz is not None:
                            last_earnings_date = last_earnings_date.tz_localize(None)
                        
                        # Remove timezone from history index if present
                        history_index = stock_hist.index
                        if hasattr(history_index, 'tz') and history_index.tz is not None:
                            history_index = history_index.tz_localize(None)
                            # Create a copy with naive datetimes
                            stock_hist_naive = stock_hist.copy()
                            stock_hist_naive.index = history_index
                        else:
                            stock_hist_naive = stock_hist
                        
                        post_earnings = stock_hist_naive[stock_hist_naive.index > last_earnings_date]
                        
                        if len(post_earnings) >= 21:
                            drift = ((post_earnings['Close'].iloc[20] - post_earnings['Close'].iloc[0]) 
                                    / post_earnings['Close'].iloc[0]) * 100
                            factors['post_earnings_drift_21d'] = float(drift)
                            self.logger.debug(f"  Post-earnings drift (21d): {drift:.2f}%")
                        else:
                            factors['post_earnings_drift_21d'] = np.nan
                    else:
                        factors['post_earnings_drift_21d'] = np.nan
                except Exception as e:
                    self.logger.debug(f"  Post-earnings drift calculation error: {e}")
                    factors['post_earnings_drift_21d'] = np.nan
            else:
                factors['post_earnings_drift_21d'] = np.nan
            
            # 4. Price Target Dispersion Ratio = (High - Low) / Mean
            if info:
                try:
                    target_high = float(info.get('targetHighPrice', np.nan))
                    target_low = float(info.get('targetLowPrice', np.nan))
                    target_mean = float(info.get('targetMeanPrice', np.nan))
                    
                    if not np.isnan(target_high) and not np.isnan(target_low) and not np.isnan(target_mean) and target_mean > 0:
                        factors['price_target_dispersion_ratio'] = ((target_high - target_low) / target_mean) * 100
                    else:
                        factors['price_target_dispersion_ratio'] = np.nan
                except Exception:
                    factors['price_target_dispersion_ratio'] = np.nan
            else:
                factors['price_target_dispersion_ratio'] = np.nan
            
        except Exception as e:
            self.logger.error(f"{raw_data.ticker}: News/Macro calculation error: {e}")
            factors = self._empty_news_macro_factors()
        
        return factors
    
    def _empty_news_macro_factors(self) -> Dict[str, float]:
        """Return dict with all news/macro factors set to NaN"""
        return {
            'news_sentiment': np.nan, 'news_sentiment_consensus': np.nan,
            'days_to_earnings': np.nan, 'pre_earnings_flag': 0.0, 'post_earnings_flag': 0.0,
            'earnings_surprise_last': np.nan, 'earnings_beat_streak': np.nan,
            'market_regime': np.nan, 'spy_correlation_60d': np.nan,
            'sector_momentum_30d': np.nan, 'sector_relative_strength': np.nan,
            'vix_level': np.nan, 'treasury_yield_10y': np.nan, 'credit_spread': np.nan,
            # NEW NEWS/MACRO SIGNALS
            'earnings_revision_3m': np.nan, 'eps_surprise_std_4q': np.nan,
            'post_earnings_drift_21d': np.nan, 'price_target_dispersion_ratio': np.nan
        }


# ============================================================================
# SOCIAL/ALTERNATIVE INDICATORS  
# ============================================================================
    
    def _calculate_social(self, reddit_data: Optional[Dict]) -> Dict[str, float]:
        """Calculate social media metrics (Reddit) - 10 factors"""
        factors = {}
        
        try:
            if not reddit_data:
                return self._empty_social_factors()
            
            # Primary Reddit metrics
            mentions = float(reddit_data.get('mentions', 0))
            factors['reddit_mentions_7d'] = mentions
            factors['reddit_sentiment_7d'] = float(reddit_data.get('sentiment', 0))
            factors['reddit_upvotes_avg'] = float(reddit_data.get('avg_post_score', 0))
            factors['reddit_comments_avg'] = float(reddit_data.get('avg_comments', 0))
            
            # Reddit sentiment (primary sentiment score)
            sentiment = reddit_data.get('sentiment')
            if sentiment is not None:
                factors['reddit_sentiment'] = float(sentiment)
                # Reddit sentiment consensus (convert -1/+1 to 0-100%)
                factors['reddit_sentiment_consensus'] = ((sentiment + 1) / 2) * 100
            else:
                factors['reddit_sentiment'] = np.nan
                factors['reddit_sentiment_consensus'] = np.nan
            
            # Buzz metrics
            factors['buzz_vs_baseline'] = mentions  # Simple mentions count
            factors['mention_velocity'] = mentions / 7.0 / 24.0 if mentions > 0 else 0.0  # per hour estimate
            
            # Sentiment consensus (legacy - maps to reddit_sentiment_consensus)
            if sentiment is not None and not np.isnan(sentiment):
                factors['sentiment_consensus'] = factors['reddit_sentiment_consensus']
            else:
                factors['sentiment_consensus'] = np.nan
            
            # Contrarian signal (high buzz + negative sentiment)
            if mentions > 10 and sentiment is not None and sentiment < -0.2:
                factors['contrarian_signal'] = 1.0
            else:
                factors['contrarian_signal'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Social calculation error: {e}")
            factors = self._empty_social_factors()
        
        return factors
    
    def _empty_social_factors(self) -> Dict[str, float]:
        """Return dict with all social factors set to NaN or 0 - 10 factors"""
        return {
            'reddit_mentions_7d': 0.0,
            'reddit_sentiment_7d': np.nan,
            'reddit_sentiment': np.nan,
            'reddit_sentiment_consensus': np.nan,
            'reddit_upvotes_avg': 0.0,
            'reddit_comments_avg': 0.0,
            'buzz_vs_baseline': 0.0,
            'mention_velocity': 0.0,
            'sentiment_consensus': np.nan,
            'contrarian_signal': 0.0
        }


# ============================================================================
# RISK/STABILITY INDICATORS
# ============================================================================
    
    def _calculate_risk(self, raw_data: RawYFinanceData, market_data: Optional['MarketData'] = None) -> Dict[str, float]:
        """Calculate risk and volatility metrics"""
        factors = {}
        
        try:
            hist = raw_data.history
            if hist.empty:
                return self._empty_risk_factors()
            
            returns = hist['Close'].pct_change().dropna()
            
            # Volatility (annualized standard deviation)
            if len(returns) >= 30:
                vol_30d = returns.tail(30).std() * np.sqrt(252) * 100
                factors['volatility_30d'] = vol_30d
            else:
                factors['volatility_30d'] = np.nan
            
            if len(returns) >= 60:
                vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
                factors['volatility_60d'] = vol_60d
            else:
                factors['volatility_60d'] = np.nan
            
            if len(returns) >= 90:
                vol_90d = returns.tail(90).std() * np.sqrt(252) * 100
                factors['volatility_90d'] = vol_90d
            else:
                factors['volatility_90d'] = np.nan
            
            # Volatility percentile (vs 1y history)
            if len(returns) >= 252:
                vol_current = returns.tail(60).std() if len(returns) >= 60 else returns.std()
                rolling_vol = returns.rolling(60).std()
                percentile = (rolling_vol < vol_current).sum() / len(rolling_vol) * 100
                factors['volatility_percentile'] = percentile
            else:
                factors['volatility_percentile'] = np.nan
            
            # Downside deviation (only negative returns)
            negative_returns = returns[returns < 0]
            if len(negative_returns) >= 60:
                factors['downside_deviation_60d'] = negative_returns.tail(60).std() * np.sqrt(252) * 100
            else:
                factors['downside_deviation_60d'] = np.nan
            
            # Beta (would need SPY data for real calculation)
            factors['beta_60d'] = float(raw_data.info.get('beta', np.nan)) if raw_data.info else np.nan
            factors['beta_252d'] = factors['beta_60d']  # Use same beta as proxy
            
            # Correlation with SPY (if market_data available)
            if market_data and market_data.spy_history is not None and not market_data.spy_history.empty:
                try:
                    market_returns = market_data.spy_history['Close'].pct_change().dropna()
                    if len(returns) >= 60 and len(market_returns) >= 60:
                        # Align dates
                        aligned = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()
                        if len(aligned) >= 60:
                            factors['correlation_spy_60d'] = aligned['stock'].tail(60).corr(aligned['market'].tail(60))
                        else:
                            factors['correlation_spy_60d'] = np.nan
                    else:
                        factors['correlation_spy_60d'] = np.nan
                except Exception as e:
                    self.logger.debug(f"  Correlation calculation error: {e}")
                    factors['correlation_spy_60d'] = np.nan
            else:
                factors['correlation_spy_60d'] = np.nan
            
            # Drawdowns
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            factors['max_drawdown_1y'] = drawdown.tail(252).min() if len(drawdown) >= 252 else drawdown.min()
            factors['max_drawdown_6m'] = drawdown.tail(126).min() if len(drawdown) >= 126 else drawdown.min()
            factors['drawdown_current'] = drawdown.iloc[-1]
            
            # Recovery time (days since max drawdown)
            if drawdown.iloc[-1] == 0:
                factors['recovery_time'] = 0.0
            else:
                # Find last time we were at all-time high
                at_high = drawdown == 0
                if at_high.any():
                    last_high_idx = at_high[::-1].idxmax()
                    factors['recovery_time'] = float(len(drawdown) - drawdown.index.get_loc(last_high_idx))
                else:
                    factors['recovery_time'] = float(len(drawdown))
            
            # Sharpe ratio (assume 0% risk-free rate for simplicity)
            if len(returns) >= 60:
                mean_return = returns.tail(60).mean() * 252
                std_return = returns.tail(60).std() * np.sqrt(252)
                factors['sharpe_ratio_60d'] = mean_return / std_return if std_return > 0 else np.nan
            else:
                factors['sharpe_ratio_60d'] = np.nan
            
            # Sortino ratio (downside deviation)
            if len(returns) >= 60:
                mean_return = returns.tail(60).mean() * 252
                downside_std = negative_returns.tail(60).std() * np.sqrt(252) if len(negative_returns) >= 60 else np.nan
                factors['sortino_ratio_60d'] = mean_return / downside_std if downside_std and downside_std > 0 else np.nan
            else:
                factors['sortino_ratio_60d'] = np.nan
            
            # Calmar ratio (return / max drawdown) - FIXED
            if len(returns) >= 252 and factors['max_drawdown_1y'] < 0:
                annual_return = returns.tail(252).mean() * 252
                max_dd = abs(factors['max_drawdown_1y'] / 100)  # Convert from percentage
                factors['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else np.nan
            else:
                factors['calmar_ratio'] = np.nan
            
            # Liquidity metrics
            # Bid-ask spread (using volume as proxy)
            if 'Volume' in hist.columns and len(hist) >= 60:
                avg_volume = hist['Volume'].tail(60).mean()
                if avg_volume > 0:
                    # Lower volume = higher spread (inverse relationship)
                    # Normalize to percentage estimate
                    volume_score = np.log1p(avg_volume)
                    factors['bid_ask_spread_pct'] = 1.0 / volume_score if volume_score > 0 else np.nan
                else:
                    factors['bid_ask_spread_pct'] = np.nan
            else:
                factors['bid_ask_spread_pct'] = np.nan
            
            # Liquidity score (composite: volume + volatility + market cap)
            if 'Volume' in hist.columns and len(hist) >= 60:
                avg_volume = hist['Volume'].tail(60).mean()
                avg_price = hist['Close'].tail(60).mean()
                dollar_volume = avg_volume * avg_price
                
                # Get market cap
                market_cap = raw_data.info.get('marketCap', 0) if raw_data.info else 0
                
                if dollar_volume > 0 and market_cap > 0:
                    # Higher = more liquid (volume × market_cap / volatility)
                    vol = factors.get('volatility_60d', 1.0)
                    if vol and vol > 0:
                        liquidity = np.log1p(dollar_volume) + np.log1p(market_cap) - np.log1p(vol)
                        factors['liquidity_score'] = liquidity
                    else:
                        factors['liquidity_score'] = np.nan
                else:
                    factors['liquidity_score'] = np.nan
            else:
                factors['liquidity_score'] = np.nan
            
            # Days to trade position (using average volume)
            if 'Volume' in hist.columns and len(hist) >= 60:
                avg_volume = hist['Volume'].tail(60).mean()
                # Assume standard position size = 0.1% of daily volume
                if avg_volume > 0:
                    position_size = avg_volume * 0.001  # 0.1% of avg volume
                    factors['days_to_trade_position'] = position_size / avg_volume if avg_volume > 0 else np.nan
                else:
                    factors['days_to_trade_position'] = np.nan
            else:
                factors['days_to_trade_position'] = np.nan
            
            # === NEW RISK SIGNALS ===
            
            # Volatility of volatility (30-day windows)
            if len(returns) >= 90:
                rolling_vol = returns.rolling(30).std()
                vol_of_vol = rolling_vol.std() * np.sqrt(252) * 100
                factors['volatility_of_volatility_30d'] = vol_of_vol
            else:
                factors['volatility_of_volatility_30d'] = np.nan
            
            # Skewness (90-day distribution)
            if len(returns) >= 90:
                factors['skewness_90d'] = returns.tail(90).skew()
            else:
                factors['skewness_90d'] = np.nan
            
            # Kurtosis (90-day distribution)
            if len(returns) >= 90:
                factors['kurtosis_90d'] = returns.tail(90).kurtosis()
            else:
                factors['kurtosis_90d'] = np.nan
            
            # Drawdown frequency (count of days with returns <= -3%)
            if len(returns) >= 90:
                large_drops = (returns.tail(90) <= -0.03).sum()
                factors['drawdown_frequency_90d'] = float(large_drops)
            else:
                factors['drawdown_frequency_90d'] = np.nan
            
            # Downside capture ratio (vs SPY, 1-year)
            if market_data and market_data.spy_history is not None and not market_data.spy_history.empty and len(returns) >= 252:
                try:
                    market_returns = market_data.spy_history['Close'].pct_change().dropna()
                    aligned = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()
                    
                    if len(aligned) >= 252:
                        # Only use days when market was down
                        down_days = aligned[aligned['market'] < 0].tail(252)
                        
                        if len(down_days) >= 20:  # Need sufficient down days
                            # Covariance / Variance of down days
                            cov = down_days['stock'].cov(down_days['market'])
                            var = down_days['market'].var()
                            factors['downside_capture_1y'] = (cov / var) if var > 0 else np.nan
                        else:
                            factors['downside_capture_1y'] = np.nan
                    else:
                        factors['downside_capture_1y'] = np.nan
                except Exception as e:
                    self.logger.debug(f"  Downside capture calculation error: {e}")
                    factors['downside_capture_1y'] = np.nan
            else:
                factors['downside_capture_1y'] = np.nan
            
        except Exception as e:
            self.logger.error(f"{raw_data.ticker}: Risk calculation error: {e}")
            factors = self._empty_risk_factors()
        
        return factors
    
    def _empty_risk_factors(self) -> Dict[str, float]:
        """Return dict with all risk factors set to NaN"""
        return {
            'volatility_30d': np.nan, 'volatility_60d': np.nan, 'volatility_90d': np.nan,
            'volatility_percentile': np.nan, 'downside_deviation_60d': np.nan,
            'beta_60d': np.nan, 'beta_252d': np.nan, 'correlation_spy_60d': np.nan,
            'max_drawdown_1y': np.nan, 'max_drawdown_6m': np.nan, 'drawdown_current': np.nan,
            'recovery_time': np.nan,
            'sharpe_ratio_60d': np.nan, 'sortino_ratio_60d': np.nan, 'calmar_ratio': np.nan,
            'bid_ask_spread_pct': np.nan, 'liquidity_score': np.nan, 'days_to_trade_position': np.nan,
            # NEW RISK SIGNALS
            'volatility_of_volatility_30d': np.nan, 'skewness_90d': np.nan, 'kurtosis_90d': np.nan,
            'drawdown_frequency_90d': np.nan, 'downside_capture_1y': np.nan
        }


# ============================================================================
# INSTITUTIONAL/SMART MONEY INDICATORS
# ============================================================================
    
    def _calculate_institutional(self, raw_data: RawYFinanceData) -> Dict[str, float]:
        """Calculate institutional ownership and smart money metrics"""
        factors = {}
        
        try:
            info = raw_data.info
            
            # Institutional ownership
            inst_holders = raw_data.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                # Calculate total institutional ownership %
                if 'Shares' in inst_holders.columns and 'sharesOutstanding' in info:
                    total_inst_shares = inst_holders['Shares'].sum()
                    shares_out = info.get('sharesOutstanding', 0)
                    if shares_out > 0:
                        factors['inst_ownership_pct'] = (total_inst_shares / shares_out) * 100
                    else:
                        factors['inst_ownership_pct'] = np.nan
                else:
                    factors['inst_ownership_pct'] = np.nan
                
                factors['inst_holder_count'] = len(inst_holders)
                
                # Top 10 concentration - sum of pctHeld for top 10 holders
                if 'pctHeld' in inst_holders.columns:
                    top10_pct = inst_holders.head(10)['pctHeld'].sum() * 100  # Convert to percentage
                    factors['inst_concentration_top10'] = top10_pct
                else:
                    factors['inst_concentration_top10'] = np.nan
            else:
                factors['inst_ownership_pct'] = np.nan
                factors['inst_holder_count'] = 0.0
                factors['inst_concentration_top10'] = np.nan
            
            # REMOVED: inst_ownership_delta_3m, inst_holder_count_delta_3m, institutional_turnover_qoq
            # These require historical snapshots and caching system to track changes over time
            
            # Insider activity
            insider_txns = raw_data.insider_transactions
            if insider_txns is not None and not insider_txns.empty:
                # Count transactions
                factors['insider_txn_count_6m'] = len(insider_txns)
                
                # Calculate net shares traded
                if 'Shares' in insider_txns.columns:
                    net_shares = insider_txns['Shares'].sum()
                    factors['insider_net_shares_6m'] = float(net_shares) if pd.notna(net_shares) else np.nan
                else:
                    factors['insider_net_shares_6m'] = np.nan
                
                # REMOVED: insider_buy_ratio, insider_sell_ratio, insider_buy_score
                # These require detailed SEC Form 4 filing data with transaction types
            else:
                factors['insider_txn_count_6m'] = 0.0
                factors['insider_net_shares_6m'] = np.nan
            
            # Analyst ratings
            recommendations = raw_data.recommendations
            if recommendations is not None and not recommendations.empty:
                # Get most recent rating (index 0 is current month 0m)
                if 'strongBuy' in recommendations.columns:
                    latest = recommendations.iloc[0]  # Current month
                else:
                    latest = None
                
                if latest is not None and not latest.isna().all():
                    # Count ratings (strongBuy=5, buy=4, hold=3, sell=2, strongSell=1)
                    strong_buy = latest.get('strongBuy', 0) or 0
                    buy = latest.get('buy', 0) or 0
                    hold = latest.get('hold', 0) or 0
                    sell = latest.get('sell', 0) or 0
                    strong_sell = latest.get('strongSell', 0) or 0
                    
                    total_ratings = strong_buy + buy + hold + sell + strong_sell
                    
                    if total_ratings > 0:
                        # Weighted average (5=strong buy, 1=strong sell)
                        weighted_sum = (strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1)
                        factors['analyst_rating_avg'] = weighted_sum / total_ratings
                        factors['analyst_count'] = float(total_ratings)
                        factors['analyst_strong_buy_count'] = float(strong_buy)
                        factors['analyst_strong_sell_count'] = float(strong_sell)
                        
                        # Consensus strength (how much agreement)
                        max_count = max(strong_buy, buy, hold, sell, strong_sell)
                        factors['analyst_consensus_strength'] = max_count / total_ratings * 100
                    else:
                        factors['analyst_rating_avg'] = np.nan
                        factors['analyst_count'] = 0.0
                        factors['analyst_strong_buy_count'] = 0.0
                        factors['analyst_strong_sell_count'] = 0.0
                        factors['analyst_consensus_strength'] = np.nan
                else:
                    factors['analyst_rating_avg'] = np.nan
                    factors['analyst_count'] = 0.0
                    factors['analyst_strong_buy_count'] = 0.0
                    factors['analyst_strong_sell_count'] = 0.0
                    factors['analyst_consensus_strength'] = np.nan
            else:
                factors['analyst_rating_avg'] = np.nan
                factors['analyst_count'] = 0.0
                factors['analyst_strong_buy_count'] = 0.0
                factors['analyst_strong_sell_count'] = 0.0
                factors['analyst_consensus_strength'] = np.nan
            
            # Upgrade/Downgrade counts and momentum from upgrades_downgrades history
            try:
                upgrades_downgrades = raw_data.upgrades_downgrades
                if upgrades_downgrades is not None and not upgrades_downgrades.empty and 'Action' in upgrades_downgrades.columns:
                    # Filter last 3 months
                    from datetime import datetime, timedelta
                    three_months_ago = datetime.now() - timedelta(days=90)
                    recent_actions = upgrades_downgrades[upgrades_downgrades.index >= three_months_ago]
                    
                    if not recent_actions.empty:
                        # Count upgrades and downgrades
                        upgrade_count = len(recent_actions[recent_actions['Action'].str.contains('up', case=False, na=False)])
                        downgrade_count = len(recent_actions[recent_actions['Action'].str.contains('down', case=False, na=False)])
                        
                        factors['analyst_upgrade_count_3m'] = float(upgrade_count)
                        factors['analyst_downgrade_count_3m'] = float(downgrade_count)
                        
                        # Analyst momentum = net upgrades/downgrades
                        factors['analyst_momentum'] = float(upgrade_count - downgrade_count)
                    else:
                        factors['analyst_upgrade_count_3m'] = 0.0
                        factors['analyst_downgrade_count_3m'] = 0.0
                        factors['analyst_momentum'] = 0.0
                else:
                    factors['analyst_upgrade_count_3m'] = np.nan
                    factors['analyst_downgrade_count_3m'] = np.nan
                    factors['analyst_momentum'] = np.nan
            except Exception as e:
                factors['analyst_upgrade_count_3m'] = np.nan
                factors['analyst_downgrade_count_3m'] = np.nan
                factors['analyst_momentum'] = np.nan
            
            # Price targets
            if info:
                target_mean = info.get('targetMeanPrice')
                target_high = info.get('targetHighPrice')
                target_low = info.get('targetLowPrice')
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                factors['price_target_mean'] = float(target_mean) if target_mean else np.nan
                factors['price_target_high'] = float(target_high) if target_high else np.nan
                factors['price_target_low'] = float(target_low) if target_low else np.nan
                
                if target_mean and current_price and current_price > 0:
                    factors['price_target_upside_pct'] = ((target_mean - current_price) / current_price) * 100
                else:
                    factors['price_target_upside_pct'] = np.nan
                
                if target_high and target_low:
                    factors['price_target_dispersion'] = ((target_high - target_low) / target_mean) * 100 if target_mean else np.nan
                else:
                    factors['price_target_dispersion'] = np.nan
            else:
                factors['price_target_mean'] = np.nan
                factors['price_target_high'] = np.nan
                factors['price_target_low'] = np.nan
                factors['price_target_upside_pct'] = np.nan
                factors['price_target_dispersion'] = np.nan
            
            # Smart money composite (aggregate signal)
            components = []
            
            # Institutional: positive if ownership > 50%
            if not np.isnan(factors['inst_ownership_pct']):
                if factors['inst_ownership_pct'] > 50:
                    components.append(1.0)
                elif factors['inst_ownership_pct'] < 30:
                    components.append(-1.0)
                else:
                    components.append(0.0)
            
            # Insider: REMOVED insider_buy_score (requires detailed SEC Form 4 data)
            
            # Analyst: positive if avg rating > 3.5 (between hold and buy)
            if not np.isnan(factors['analyst_rating_avg']):
                if factors['analyst_rating_avg'] > 3.5:
                    components.append(1.0)
                elif factors['analyst_rating_avg'] < 2.5:
                    components.append(-1.0)
                else:
                    components.append(0.0)
            
            # Price target: positive if upside > 10%
            if not np.isnan(factors['price_target_upside_pct']):
                if factors['price_target_upside_pct'] > 10:
                    components.append(1.0)
                elif factors['price_target_upside_pct'] < -10:
                    components.append(-1.0)
                else:
                    components.append(0.0)
            
            if components:
                factors['smart_money_composite'] = np.mean(components)
            else:
                factors['smart_money_composite'] = np.nan
            
            # === NEW INSTITUTIONAL SIGNALS ===
            
            # Institutional concentration (top 5 holders)
            if inst_holders is not None and not inst_holders.empty and 'Shares' in inst_holders.columns:
                if 'sharesOutstanding' in info and info.get('sharesOutstanding', 0) > 0:
                    shares_out = info['sharesOutstanding']
                    top5_shares = inst_holders['Shares'].nlargest(5).sum()
                    factors['institutional_concentration_top5'] = (top5_shares / shares_out) * 100
                else:
                    factors['institutional_concentration_top5'] = np.nan
            else:
                factors['institutional_concentration_top5'] = np.nan
            
            # REMOVED: institutional_turnover_qoq
            # This requires quarterly snapshot comparison of institutional holdings
            
            # Insider transaction intensity (log-scaled count)
            if not np.isnan(factors['insider_txn_count_6m']) and factors['insider_txn_count_6m'] > 0:
                factors['insider_transaction_intensity'] = np.log1p(factors['insider_txn_count_6m'])
            else:
                factors['insider_transaction_intensity'] = 0.0
            
        except Exception as e:
            self.logger.error(f"{raw_data.ticker}: Institutional calculation error: {e}")
            factors = self._empty_institutional_factors()
        
        return factors
    
    def _empty_institutional_factors(self) -> Dict[str, float]:
        """Return dict with all institutional factors set to NaN or 0"""
        return {
            'inst_ownership_pct': np.nan,
            'inst_holder_count': 0.0,
            'inst_concentration_top10': np.nan,
            'insider_txn_count_6m': 0.0,
            'insider_net_shares_6m': np.nan,
            'analyst_rating_avg': np.nan, 'analyst_momentum': np.nan,
            'analyst_upgrade_count_3m': np.nan, 'analyst_downgrade_count_3m': np.nan,
            'analyst_count': 0.0, 'analyst_strong_buy_count': 0.0, 'analyst_strong_sell_count': 0.0,
            'analyst_consensus_strength': np.nan,
            'price_target_mean': np.nan, 'price_target_high': np.nan, 'price_target_low': np.nan,
            'price_target_upside_pct': np.nan, 'price_target_dispersion': np.nan,
            'smart_money_composite': np.nan,
            # NEW INSTITUTIONAL SIGNALS
            'institutional_concentration_top5': np.nan,
            'insider_transaction_intensity': 0.0
        }
