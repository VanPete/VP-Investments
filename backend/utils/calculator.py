"""
Financial & Technical Calculators (3.0 Architecture)
=====================================================
Process raw market data bundles into calculated metrics

Used in Phase 2 (Parse & Normalize) and Phase 3 (Score by Group)
All calculations are pure functions - no API calls

Dependencies: pandas, numpy
"""
from typing import Dict, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TechnicalMetrics:
    """Calculated technical indicators"""
    ticker: str
    
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Momentum indicators
    rsi: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_width: Optional[float] = None
    atr: Optional[float] = None
    atr_percentage: Optional[float] = None
    
    # Volume indicators
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    obv: Optional[float] = None
    
    # Price metrics
    price_change_1d: Optional[float] = None
    price_change_5d: Optional[float] = None
    price_change_30d: Optional[float] = None
    volatility_30d: Optional[float] = None
    
    # Support/Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


@dataclass
class FundamentalMetrics:
    """Calculated fundamental scores and metrics"""
    ticker: str
    
    # Valuation scores
    valuation_score: float = 0.5
    pe_score: float = 0.5
    pb_score: float = 0.5
    ps_score: float = 0.5
    
    # Profitability scores
    profitability_score: float = 0.5
    margin_score: float = 0.5
    roe_score: float = 0.5
    
    # Growth scores
    growth_score: float = 0.5
    revenue_growth_score: float = 0.5
    earnings_growth_score: float = 0.5
    
    # Financial health scores
    health_score: float = 0.5
    debt_score: float = 0.5
    liquidity_score: float = 0.5
    
    # Overall fundamental score
    fundamental_score: float = 0.5


# ============================================================================
# TECHNICAL CALCULATORS
# ============================================================================

class TechnicalCalculator:
    """Calculate technical indicators from historical data"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> Optional[float]:
        """Simple Moving Average"""
        try:
            if len(data) < period:
                return None
            return float(data.tail(period).mean())
        except Exception as e:
            logger.debug(f"SMA calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> Optional[float]:
        """Exponential Moving Average"""
        try:
            if len(data) < period:
                return None
            return float(data.ewm(span=period, adjust=False).mean().iloc[-1])
        except Exception as e:
            logger.debug(f"EMA calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> Optional[float]:
        """Relative Strength Index"""
        try:
            if len(data) < period + 1:
                return None
            
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.debug(f"RSI calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        try:
            if len(data) < slow + signal:
                return (None, None, None)
            
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return (
                float(macd_line.iloc[-1]),
                float(signal_line.iloc[-1]),
                float(histogram.iloc[-1])
            )
        except Exception as e:
            logger.debug(f"MACD calculation failed: {e}")
            return (None, None, None)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                  std_dev: float = 2.0) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Bollinger Bands"""
        try:
            if len(data) < period:
                return (None, None, None)
            
            sma = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return (
                float(upper.iloc[-1]),
                float(sma.iloc[-1]),
                float(lower.iloc[-1])
            )
        except Exception as e:
            logger.debug(f"Bollinger Bands calculation failed: {e}")
            return (None, None, None)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> Optional[float]:
        """Average True Range"""
        try:
            if len(high) < period + 1:
                return None
            
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1])
        except Exception as e:
            logger.debug(f"ATR calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_volatility(data: pd.Series, period: int = 30) -> Optional[float]:
        """Historical volatility (standard deviation of returns)"""
        try:
            if len(data) < period + 1:
                return None
            
            returns = data.pct_change()
            volatility = returns.tail(period).std() * np.sqrt(252)  # Annualized
            
            return float(volatility)
        except Exception as e:
            logger.debug(f"Volatility calculation failed: {e}")
            return None
    
    @staticmethod
    def calculate_all_technicals(hist_df: pd.DataFrame, ticker: str) -> TechnicalMetrics:
        """Calculate all technical indicators from historical data"""
        metrics = TechnicalMetrics(ticker=ticker)
        
        if hist_df.empty:
            logger.debug(f"No historical data for {ticker}")
            return metrics
        
        try:
            closes = hist_df['Close']
            
            # Moving averages
            metrics.sma_20 = TechnicalCalculator.calculate_sma(closes, 20)
            metrics.sma_50 = TechnicalCalculator.calculate_sma(closes, 50)
            metrics.sma_200 = TechnicalCalculator.calculate_sma(closes, 200)
            metrics.ema_12 = TechnicalCalculator.calculate_ema(closes, 12)
            metrics.ema_26 = TechnicalCalculator.calculate_ema(closes, 26)
            
            # MACD
            macd, signal, hist = TechnicalCalculator.calculate_macd(closes)
            metrics.macd = macd
            metrics.macd_signal = signal
            metrics.macd_histogram = hist
            
            # RSI
            metrics.rsi = TechnicalCalculator.calculate_rsi(closes, 14)
            
            # Bollinger Bands
            bb_upper, bb_mid, bb_lower = TechnicalCalculator.calculate_bollinger_bands(closes)
            metrics.bollinger_upper = bb_upper
            metrics.bollinger_middle = bb_mid
            metrics.bollinger_lower = bb_lower
            if bb_upper and bb_lower:
                metrics.bollinger_width = bb_upper - bb_lower
            
            # ATR
            if all(col in hist_df.columns for col in ['High', 'Low', 'Close']):
                metrics.atr = TechnicalCalculator.calculate_atr(
                    hist_df['High'], hist_df['Low'], hist_df['Close']
                )
                if metrics.atr and closes.iloc[-1] > 0:
                    metrics.atr_percentage = (metrics.atr / closes.iloc[-1]) * 100
            
            # Volatility
            metrics.volatility_30d = TechnicalCalculator.calculate_volatility(closes, 30)
            
            # Price changes
            if len(closes) >= 1:
                metrics.price_change_1d = float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100) if len(closes) > 1 else None
            if len(closes) >= 5:
                metrics.price_change_5d = float((closes.iloc[-1] / closes.iloc[-5] - 1) * 100)
            if len(closes) >= 30:
                metrics.price_change_30d = float((closes.iloc[-1] / closes.iloc[-30] - 1) * 100)
            
            # Volume
            if 'Volume' in hist_df.columns:
                volumes = hist_df['Volume']
                metrics.volume_sma_20 = TechnicalCalculator.calculate_sma(volumes, 20)
                if metrics.volume_sma_20 and volumes.iloc[-1] > 0:
                    metrics.volume_ratio = float(volumes.iloc[-1] / metrics.volume_sma_20)
            
            # Support/Resistance (simple version - last 52 weeks)
            if len(closes) >= 252:
                metrics.support_level = float(closes.tail(252).min())
                metrics.resistance_level = float(closes.tail(252).max())
            
        except Exception as e:
            logger.error(f"Error calculating technicals for {ticker}: {e}")
        
        return metrics


# ============================================================================
# FUNDAMENTAL CALCULATORS
# ============================================================================

class FundamentalCalculator:
    """Calculate fundamental scores from market data"""
    
    @staticmethod
    def calculate_fundamental_score(market_data: Dict[str, Any]) -> FundamentalMetrics:
        """
        Calculate comprehensive fundamental score from market data bundle
        
        Returns FundamentalMetrics with scores from 0-1
        """
        ticker = market_data.get('ticker', 'UNKNOWN')
        metrics = FundamentalMetrics(ticker=ticker)
        
        try:
            # Valuation scores
            metrics.pe_score = FundamentalCalculator._score_pe_ratio(market_data.get('pe_ratio'))
            metrics.pb_score = FundamentalCalculator._score_pb_ratio(market_data.get('pb_ratio'))
            metrics.ps_score = FundamentalCalculator._score_ps_ratio(market_data.get('ps_ratio'))
            metrics.valuation_score = (metrics.pe_score + metrics.pb_score + metrics.ps_score) / 3
            
            # Profitability scores
            metrics.margin_score = FundamentalCalculator._score_margins(
                market_data.get('profit_margin'),
                market_data.get('operating_margin')
            )
            metrics.roe_score = FundamentalCalculator._score_roe(market_data.get('roe'))
            metrics.profitability_score = (metrics.margin_score + metrics.roe_score) / 2
            
            # Growth scores
            metrics.revenue_growth_score = FundamentalCalculator._score_growth(
                market_data.get('revenue_growth')
            )
            metrics.earnings_growth_score = FundamentalCalculator._score_growth(
                market_data.get('earnings_growth')
            )
            metrics.growth_score = (metrics.revenue_growth_score + metrics.earnings_growth_score) / 2
            
            # Financial health scores
            metrics.debt_score = FundamentalCalculator._score_debt(market_data.get('debt_to_equity'))
            metrics.liquidity_score = FundamentalCalculator._score_liquidity(
                market_data.get('current_ratio'),
                market_data.get('quick_ratio')
            )
            metrics.health_score = (metrics.debt_score + metrics.liquidity_score) / 2
            
            # Overall fundamental score (weighted average)
            metrics.fundamental_score = (
                metrics.valuation_score * 0.30 +
                metrics.profitability_score * 0.30 +
                metrics.growth_score * 0.25 +
                metrics.health_score * 0.15
            )
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score for {ticker}: {e}")
        
        return metrics
    
    @staticmethod
    def _score_pe_ratio(pe: Optional[float]) -> float:
        """Score P/E ratio (lower is better, up to a point)"""
        if pe is None or pe <= 0:
            return 0.5
        
        # Optimal range: 10-20
        if pe < 5:
            return 0.3  # Too low might indicate problems
        elif pe < 10:
            return 0.7
        elif pe < 15:
            return 0.9
        elif pe < 20:
            return 0.8
        elif pe < 30:
            return 0.6
        elif pe < 50:
            return 0.4
        else:
            return 0.2
    
    @staticmethod
    def _score_pb_ratio(pb: Optional[float]) -> float:
        """Score P/B ratio (lower is generally better)"""
        if pb is None or pb <= 0:
            return 0.5
        
        if pb < 1:
            return 0.9  # Trading below book value
        elif pb < 2:
            return 0.8
        elif pb < 3:
            return 0.6
        elif pb < 5:
            return 0.4
        else:
            return 0.2
    
    @staticmethod
    def _score_ps_ratio(ps: Optional[float]) -> float:
        """Score P/S ratio (lower is generally better)"""
        if ps is None or ps <= 0:
            return 0.5
        
        if ps < 1:
            return 0.9
        elif ps < 2:
            return 0.8
        elif ps < 3:
            return 0.6
        elif ps < 5:
            return 0.4
        else:
            return 0.2
    
    @staticmethod
    def _score_margins(profit_margin: Optional[float], 
                      operating_margin: Optional[float]) -> float:
        """Score profit margins (higher is better)"""
        scores = []
        
        if profit_margin is not None:
            if profit_margin < 0:
                scores.append(0.1)
            elif profit_margin < 0.05:
                scores.append(0.3)
            elif profit_margin < 0.10:
                scores.append(0.5)
            elif profit_margin < 0.15:
                scores.append(0.7)
            elif profit_margin < 0.20:
                scores.append(0.9)
            else:
                scores.append(1.0)
        
        if operating_margin is not None:
            if operating_margin < 0:
                scores.append(0.1)
            elif operating_margin < 0.10:
                scores.append(0.4)
            elif operating_margin < 0.15:
                scores.append(0.6)
            elif operating_margin < 0.20:
                scores.append(0.8)
            else:
                scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    @staticmethod
    def _score_roe(roe: Optional[float]) -> float:
        """Score Return on Equity (higher is better)"""
        if roe is None:
            return 0.5
        
        if roe < 0:
            return 0.1
        elif roe < 0.05:
            return 0.3
        elif roe < 0.10:
            return 0.5
        elif roe < 0.15:
            return 0.7
        elif roe < 0.20:
            return 0.9
        else:
            return 1.0
    
    @staticmethod
    def _score_growth(growth_rate: Optional[float]) -> float:
        """Score growth rate (higher is better, but extreme values are suspicious)"""
        if growth_rate is None:
            return 0.5
        
        if growth_rate < -0.20:
            return 0.1  # Significant decline
        elif growth_rate < -0.10:
            return 0.2
        elif growth_rate < 0:
            return 0.4
        elif growth_rate < 0.05:
            return 0.5
        elif growth_rate < 0.10:
            return 0.7
        elif growth_rate < 0.20:
            return 0.9
        elif growth_rate < 0.50:
            return 1.0
        else:
            return 0.7  # Very high growth might not be sustainable
    
    @staticmethod
    def _score_debt(debt_to_equity: Optional[float]) -> float:
        """Score debt to equity (lower is better)"""
        if debt_to_equity is None:
            return 0.5
        
        if debt_to_equity < 0:
            return 0.5
        elif debt_to_equity < 0.3:
            return 1.0
        elif debt_to_equity < 0.5:
            return 0.9
        elif debt_to_equity < 1.0:
            return 0.7
        elif debt_to_equity < 2.0:
            return 0.5
        elif debt_to_equity < 3.0:
            return 0.3
        else:
            return 0.1
    
    @staticmethod
    def _score_liquidity(current_ratio: Optional[float], 
                        quick_ratio: Optional[float]) -> float:
        """Score liquidity ratios (higher is better, up to a point)"""
        scores = []
        
        if current_ratio is not None:
            if current_ratio < 1.0:
                scores.append(0.2)
            elif current_ratio < 1.5:
                scores.append(0.5)
            elif current_ratio < 2.0:
                scores.append(0.8)
            elif current_ratio < 3.0:
                scores.append(1.0)
            else:
                scores.append(0.9)  # Too high might indicate inefficiency
        
        if quick_ratio is not None:
            if quick_ratio < 0.5:
                scores.append(0.2)
            elif quick_ratio < 1.0:
                scores.append(0.5)
            elif quick_ratio < 1.5:
                scores.append(0.8)
            else:
                scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 0.5


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_score(value: Optional[float], min_val: float, max_val: float, 
                   invert: bool = False) -> float:
    """
    Normalize a value to 0-1 scale
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        invert: If True, higher values get lower scores
        
    Returns:
        Normalized score from 0.0 to 1.0
    """
    if value is None:
        return 0.5
    
    # Clamp to range
    value = max(min_val, min(max_val, value))
    
    # Normalize
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # Invert if needed
    if invert:
        normalized = 1.0 - normalized
    
    return normalized
