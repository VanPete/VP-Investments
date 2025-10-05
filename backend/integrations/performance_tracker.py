"""
VP Investments - Performance Tracking Integration
Calculates returns, SPY benchmarks, and performance metrics for signals
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import yfinance as yf
import pandas as pd
import numpy as np

from backend.storage.database import SupabaseInterface
from backend.utils.logger import get_logger

class PerformanceTracker:
    """Enhanced performance tracking with return calculations and SPY benchmarks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.db = SupabaseInterface()
        
        # Performance tracking settings
        self.return_windows = [1, 3, 7, 10]  # Days to track returns
        self.benchmark_ticker = "SPY"
        self.cost_basis_points = 10  # Transaction costs in basis points
        
        self.logger.info("Performance tracker initialized")
    
    async def calculate_signal_performance(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate comprehensive performance metrics for signals.
        
        Args:
            signals: List of signal dictionaries to enhance with performance data
            
        Returns:
            Enhanced signals with performance metrics
        """
        try:
            self.logger.info(f"Calculating performance metrics for {len(signals)} signals")
            
            # Get SPY benchmark data once
            spy_data = await self._get_benchmark_data()
            
            # Process signals in batches for efficiency
            enhanced_signals = []
            batch_size = 20
            
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                enhanced_batch = await self._process_signal_batch(batch, spy_data)
                enhanced_signals.extend(enhanced_batch)
                
                # Log progress
                self.logger.info(f"Processed {min(i + batch_size, len(signals))}/{len(signals)} signals")
            
            self.logger.info("Performance calculation complete")
            return enhanced_signals
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return signals  # Return original signals if enhancement fails
    
    async def _process_signal_batch(self, signals: List[Dict], spy_data: pd.Series) -> List[Dict]:
        """Process a batch of signals for performance calculations."""
        enhanced_signals = []
        
        for signal in signals:
            try:
                enhanced_signal = signal.copy()
                ticker = signal['ticker']
                
                # Get historical performance if signal has a timestamp
                if 'run_datetime' in signal:
                    performance_data = await self._calculate_historical_returns(
                        ticker, signal['run_datetime'], spy_data
                    )
                    enhanced_signal.update(performance_data)
                
                # Add forward-looking performance calculations
                forward_data = await self._calculate_forward_metrics(ticker)
                enhanced_signal.update(forward_data)
                
                enhanced_signals.append(enhanced_signal)
                
            except Exception as e:
                self.logger.warning(f"Performance calculation failed for {signal.get('ticker', 'unknown')}: {e}")
                enhanced_signals.append(signal)  # Keep original if enhancement fails
        
        return enhanced_signals
    
    async def _get_benchmark_data(self) -> pd.Series:
        """Get SPY benchmark data for performance comparisons."""
        try:
            # Get 1 year of SPY data for comprehensive benchmarking
            spy = yf.Ticker(self.benchmark_ticker)
            hist = spy.history(period="1y")
            
            if hist.empty:
                self.logger.warning("Could not retrieve SPY benchmark data")
                return pd.Series(dtype=float)
            
            # Return adjusted close prices with timezone-aware index
            spy_prices = hist['Close']
            spy_prices.index = pd.to_datetime(spy_prices.index, utc=True)
            
            return spy_prices
            
        except Exception as e:
            self.logger.warning(f"Failed to get benchmark data: {e}")
            return pd.Series(dtype=float)
    
    async def _calculate_historical_returns(self, ticker: str, run_datetime: str, spy_data: pd.Series) -> Dict[str, Any]:
        """Calculate historical returns for a signal from its run date."""
        try:
            # Parse run datetime
            run_date = pd.to_datetime(run_datetime, utc=True)
            
            # Calculate end date (today or max available data)
            end_date = min(
                pd.Timestamp.now(tz='UTC'),
                run_date + timedelta(days=max(self.return_windows) + 5)
            )
            
            # Skip if signal is too recent (need at least 1 day)
            if (pd.Timestamp.now(tz='UTC') - run_date).days < 1:
                return self._get_null_performance_data()
            
            # Get ticker price data
            ticker_prices = await self._get_price_data(ticker, run_date, end_date)
            
            if ticker_prices is None or ticker_prices.empty:
                return self._get_null_performance_data()
            
            # Calculate returns for each window
            performance_data = {}
            base_price = ticker_prices.iloc[0]
            
            # Calculate realized returns
            max_return = -float('inf')
            max_drawdown = 0
            peak_price = base_price
            
            for i, price in enumerate(ticker_prices):
                # Track max return and drawdown
                return_pct = (price - base_price) / base_price * 100
                max_return = max(max_return, return_pct)
                
                peak_price = max(peak_price, price)
                drawdown = (price - peak_price) / peak_price * 100
                max_drawdown = min(max_drawdown, drawdown)
            
            # Calculate returns for specific windows
            for window in self.return_windows:
                if len(ticker_prices) > window:
                    end_price = ticker_prices.iloc[window]
                    
                    # Gross return
                    gross_return = (end_price - base_price) / base_price * 100
                    performance_data[f'{window}d_return'] = self._safe_round(gross_return, 2)
                    
                    # Net return (after transaction costs)
                    net_return = gross_return - (self.cost_basis_points / 100)
                    performance_data[f'{window}d_return_net'] = self._safe_round(net_return, 2)
                    
                    # SPY benchmark comparison
                    spy_return = self._calculate_spy_return(run_date, window, spy_data)
                    if spy_return is not None:
                        performance_data[f'spy_{window}d_return'] = self._safe_round(spy_return, 2)
                        performance_data[f'beat_spy_{window}d'] = gross_return > spy_return
                else:
                    # Not enough data for this window
                    performance_data[f'{window}d_return'] = None
                    performance_data[f'{window}d_return_net'] = None
                    performance_data[f'spy_{window}d_return'] = None
                    performance_data[f'beat_spy_{window}d'] = None
            
            # Additional performance metrics
            performance_data.update({
                'max_return_pct': self._safe_round(max_return, 2),
                'drawdown_pct': self._safe_round(max_drawdown, 2),
                'signal_duration': len(ticker_prices),
                'forward_volatility': self._calculate_volatility(ticker_prices),
                'forward_sharpe_ratio': self._calculate_sharpe_ratio(ticker_prices),
                'realized_returns': ','.join([str(w) for w in self.return_windows if f'{w}d_return' in performance_data and performance_data[f'{w}d_return'] is not None]),
                'backtest_phase': self._determine_backtest_phase(performance_data),
                'backtest_timestamp': datetime.now(timezone.utc).isoformat(),
                'backtest_notes': f"Performance calculated for {len(ticker_prices)} days"
            })
            
            return performance_data
            
        except Exception as e:
            self.logger.warning(f"Historical return calculation failed for {ticker}: {e}")
            return self._get_null_performance_data()
    
    async def _calculate_forward_metrics(self, ticker: str) -> Dict[str, Any]:
        """Calculate forward-looking performance metrics."""
        try:
            # Get recent price data for technical analysis
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="3mo")  # 3 months for volatility analysis
            
            if hist.empty:
                return {}
            
            prices = hist['Close']
            returns = prices.pct_change().dropna()
            
            # Calculate forward volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized percentage
            
            # Calculate forward Sharpe ratio estimate (using risk-free rate of 4%)
            risk_free_rate = 0.04
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else None
            
            return {
                'forward_volatility': self._safe_round(volatility, 2),
                'forward_sharpe_ratio': self._safe_round(sharpe_ratio, 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Forward metrics calculation failed for {ticker}: {e}")
            return {}
    
    async def _get_price_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.Series]:
        """Get price data for a ticker within a date range."""
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Convert timestamps to date strings for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            hist = ticker_obj.history(start=start_str, end=end_str)
            
            if hist.empty:
                return None
            
            # Return adjusted close prices
            prices = hist['Adj Close'] if 'Adj Close' in hist.columns else hist['Close']
            return prices
            
        except Exception as e:
            self.logger.warning(f"Price data retrieval failed for {ticker}: {e}")
            return None
    
    def _calculate_spy_return(self, run_date: pd.Timestamp, window: int, spy_data: pd.Series) -> Optional[float]:
        """Calculate SPY return for a specific window from run date."""
        try:
            if spy_data.empty:
                return None
            
            # Find the closest SPY price to run date
            start_idx = spy_data.index.searchsorted(run_date, side='left')
            end_idx = start_idx + window
            
            if start_idx >= len(spy_data) or end_idx >= len(spy_data):
                return None
            
            start_price = spy_data.iloc[start_idx]
            end_price = spy_data.iloc[end_idx]
            
            spy_return = (end_price - start_price) / start_price * 100
            return spy_return
            
        except Exception:
            return None
    
    def _calculate_volatility(self, prices: pd.Series) -> Optional[float]:
        """Calculate annualized volatility from price series."""
        try:
            if len(prices) < 2:
                return None
            
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return None
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252) * 100
            return self._safe_round(volatility, 2)
            
        except Exception:
            return None
    
    def _calculate_sharpe_ratio(self, prices: pd.Series) -> Optional[float]:
        """Calculate Sharpe ratio from price series."""
        try:
            if len(prices) < 2:
                return None
            
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return None
            
            # Annualized return and volatility
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            
            if annual_vol == 0:
                return None
            
            # Assuming 4% risk-free rate
            risk_free_rate = 0.04
            sharpe = (annual_return - risk_free_rate) / annual_vol
            
            return self._safe_round(sharpe, 2)
            
        except Exception:
            return None
    
    def _get_null_performance_data(self) -> Dict[str, Any]:
        """Return null performance data structure."""
        data = {}
        
        for window in self.return_windows:
            data[f'{window}d_return'] = None
            data[f'{window}d_return_net'] = None
            data[f'spy_{window}d_return'] = None
            data[f'beat_spy_{window}d'] = None
        
        data.update({
            'max_return_pct': None,
            'drawdown_pct': None,
            'signal_duration': None,
            'forward_volatility': None,
            'forward_sharpe_ratio': None,
            'realized_returns': None,
            'backtest_phase': 'Pending',
            'backtest_timestamp': None,
            'backtest_notes': None
        })
        
        return data
    
    def _determine_backtest_phase(self, performance_data: Dict[str, Any]) -> str:
        """Determine the current backtest phase based on available data."""
        realized_windows = [w for w in self.return_windows 
                          if performance_data.get(f'{w}d_return') is not None]
        
        if len(realized_windows) == len(self.return_windows):
            return 'Complete'
        elif len(realized_windows) > 0:
            return 'Partial'
        else:
            return 'Pending'
    
    def _safe_round(self, value: Optional[float], decimals: int = 2) -> Optional[float]:
        """Safely round a value, handling NaN, infinity, and None."""
        if value is None:
            return None
        
        try:
            if np.isnan(value) or np.isinf(value):
                return None
            return round(float(value), decimals)
        except (ValueError, TypeError, OverflowError):
            return None

# Integration function for use in pipeline
async def enhance_signals_with_performance(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance signals with performance tracking data.
    
    Args:
        signals: List of signals to enhance
        
    Returns:
        Enhanced signals with performance metrics
    """
    try:
        tracker = PerformanceTracker()
        return await tracker.calculate_signal_performance(signals)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Performance enhancement failed: {e}")
        return signals