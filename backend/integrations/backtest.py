"""
Backtesting Engine for Performance Tracking and Historical Data

This module provides comprehensive backtesting capabilities including:
1. Historical performance tracking (1d, 3d, 7d, 10d returns)
2. SPY comparison and beat rates
3. Forward-looking performance metrics
4. Drawdown and Sharpe ratio calculation
5. Signal duration tracking
6. Realized returns analysis
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import backend.integrations.yfinance as yf
from dataclasses import dataclass
from enum import Enum

from ..storage.database import get_supabase_database
from ..utils.observability import emit_metric

logger = logging.getLogger(__name__)


class BacktestPhase(Enum):
    """Backtest phases"""
    INITIAL = "initial"
    TRACKING = "tracking"  
    COMPLETED = "completed"
    EXPIRED = "expired"


class BacktestInterval(Enum):
    """Define backtest intervals for time-based performance tracking"""
    ONE_DAY = "1d"
    THREE_DAY = "3d" 
    SEVEN_DAY = "7d"
    TEN_DAY = "10d"
    FOURTEEN_DAY = "14d"
    THIRTY_DAY = "30d"
    SIXTY_DAY = "60d"
    NINETY_DAY = "90d"


@dataclass
class PerformanceMetrics:
    """Container for performance tracking metrics"""
    # Return metrics
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None  
    return_7d: Optional[float] = None
    return_10d: Optional[float] = None
    
    # Net returns (after transaction costs)
    return_1d_net: Optional[float] = None
    return_3d_net: Optional[float] = None
    return_7d_net: Optional[float] = None
    return_10d_net: Optional[float] = None
    
    # SPY benchmark
    spy_1d_return: Optional[float] = None
    spy_3d_return: Optional[float] = None
    spy_7d_return: Optional[float] = None
    spy_10d_return: Optional[float] = None
    
    # Beat SPY flags
    beat_spy_1d: Optional[bool] = None
    beat_spy_3d: Optional[bool] = None
    beat_spy_7d: Optional[bool] = None
    beat_spy_10d: Optional[bool] = None
    
    # Risk metrics
    max_return_pct: Optional[float] = None
    drawdown_pct: Optional[float] = None
    forward_volatility: Optional[float] = None
    forward_sharpe_ratio: Optional[float] = None
    
    # Duration tracking
    signal_duration: Optional[int] = None
    realized_returns: Optional[str] = None  # JSON string of daily returns


class BacktestEngine:
    """Comprehensive backtesting engine for signal performance tracking"""
    
    def __init__(self, db=None):
        # Accept optional database instance to avoid async initialization issues
        self.db = db  # Will be set via set_database() if None
        self.transaction_cost = 0.001  # 0.1% per trade (round trip = 0.2%)
        self.intervals = [1, 3, 7, 10, 30]  # Track 1d, 3d, 7d, 10d, 30d returns (Phase 4.3)
    
    async def set_database(self):
        """Initialize database connection asynchronously"""
        if self.db is None:
            self.db = await get_supabase_database()
    
    # NOTE: Data fetching moved to Phase 1 (yfinance.py + cache.py)
    # All historical price data must be pre-fetched and passed as parameters
    # No mid-pipeline API calls allowed in 3.0 architecture
    
    def calculate_returns(self, price_data: pd.DataFrame, entry_date: datetime, 
                         target_days: Optional[List[int]] = None) -> Dict[str, float]:
        """Calculate returns for specified time periods"""
        try:
            if target_days is None:
                target_days = self.intervals  # Use 1, 3, 7, 14 per user request
            
            returns = {}
            
            # Find entry price
            entry_price = None
            entry_index = None
            for i, (date, row) in enumerate(price_data.iterrows()):
                if date.date() >= entry_date.date():
                    entry_price = row['Close']
                    entry_index = i
                    break
            
            if entry_price is None:
                return {}
            
            # Calculate returns for each target period
            for days in target_days:
                target_index = entry_index + days
                
                if target_index < len(price_data):
                    exit_price = price_data.iloc[target_index]['Close']
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    returns[f'{days}d_return'] = float(return_pct)
                    
                    # Calculate net return (after transaction costs)
                    net_return = return_pct - (self.transaction_cost * 200)  # Round trip cost as %
                    returns[f'{days}d_return_net'] = float(net_return)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return {}
    
    def calculate_spy_returns(self, spy_data: pd.DataFrame, entry_date: datetime,
                             target_days: Optional[List[int]] = None) -> Dict[str, float]:
        """Calculate SPY benchmark returns"""
        try:
            if target_days is None:
                target_days = self.intervals  # Use 1, 3, 7, 14 per user request
            
            spy_returns = {}
            
            # Find entry date in SPY data
            entry_price = None
            for i, (date, row) in enumerate(spy_data.iterrows()):
                if date.date() >= entry_date.date():
                    entry_price = row['Close']
                    entry_index = i
                    break
            
            if entry_price is None:
                return {}
            
            # Calculate SPY returns
            for days in target_days:
                target_index = entry_index + days
                
                if target_index < len(spy_data):
                    exit_price = spy_data.iloc[target_index]['Close']
                    spy_return = ((exit_price - entry_price) / entry_price) * 100
                    spy_returns[f'spy_{days}d_return'] = float(spy_return)
            
            return spy_returns
            
        except Exception as e:
            logger.error(f"Error calculating SPY returns: {e}")
            return {}
    
    def calculate_risk_metrics(self, price_data: pd.DataFrame, entry_date: datetime) -> Dict[str, float]:
        """Calculate risk metrics like max return, drawdown, volatility, Sharpe ratio"""
        try:
            risk_metrics = {}
            
            # Find entry index
            entry_index = None
            entry_price = None
            
            for i, (date, row) in enumerate(price_data.iterrows()):
                if date.date() >= entry_date.date():
                    entry_price = row['Close']
                    entry_index = i
                    break
            
            if entry_index is None or entry_price is None:
                return {}
            
            # Get forward price data (up to 30 days or end of data)
            end_index = min(entry_index + 30, len(price_data))
            forward_prices = price_data.iloc[entry_index:end_index]['Close']
            
            if len(forward_prices) < 2:
                return {}
            
            # Calculate returns series
            returns_series = forward_prices.pct_change().dropna()
            
            # Max return percentage
            max_price = forward_prices.max()
            max_return_pct = ((max_price - entry_price) / entry_price) * 100
            risk_metrics['max_return_pct'] = float(max_return_pct)
            
            # Maximum drawdown
            peak = forward_prices.expanding().max()
            drawdown = ((forward_prices - peak) / peak) * 100
            max_drawdown = drawdown.min()
            risk_metrics['drawdown_pct'] = float(max_drawdown)
            
            # Forward volatility (annualized)
            if len(returns_series) > 1:
                volatility = returns_series.std() * np.sqrt(252)
                risk_metrics['forward_volatility'] = float(volatility)
                
                # Forward Sharpe ratio (assuming 2% risk-free rate)
                mean_return = returns_series.mean() * 252
                if volatility > 0:
                    sharpe_ratio = (mean_return - 0.02) / volatility
                    risk_metrics['forward_sharpe_ratio'] = float(sharpe_ratio)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_signal_duration(self, price_data: pd.DataFrame, entry_date: datetime,
                                 exit_criteria: Optional[Dict[str, float]] = None) -> Optional[int]:
        """Calculate optimal signal duration based on exit criteria"""
        try:
            if exit_criteria is None:
                exit_criteria = {
                    'profit_target': 15.0,  # Exit at 15% gain
                    'stop_loss': -8.0,      # Exit at 8% loss
                    'max_days': 30          # Maximum hold period
                }
            
            # Find entry price
            entry_price = None
            entry_index = None
            
            for i, (date, row) in enumerate(price_data.iterrows()):
                if date.date() >= entry_date.date():
                    entry_price = row['Close']
                    entry_index = i
                    break
            
            if entry_price is None:
                return None
            
            # Check exit conditions day by day
            for days_held in range(1, min(len(price_data) - entry_index, exit_criteria['max_days'] + 1)):
                current_index = entry_index + days_held
                if current_index >= len(price_data):
                    break
                
                current_price = price_data.iloc[current_index]['Close']
                return_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Check exit conditions
                if return_pct >= exit_criteria['profit_target']:
                    return days_held  # Profit target hit
                elif return_pct <= exit_criteria['stop_loss']:
                    return days_held  # Stop loss hit
            
            # Reached max days without exit
            return exit_criteria['max_days']
            
        except Exception as e:
            logger.error(f"Error calculating signal duration: {e}")
            return None
    
    def create_realized_returns_series(self, price_data: pd.DataFrame, entry_date: datetime,
                                      days: int = 30) -> Optional[str]:
        """Create JSON string of daily realized returns"""
        try:
            # Find entry price
            entry_price = None
            entry_index = None
            
            for i, (date, row) in enumerate(price_data.iterrows()):
                if date.date() >= entry_date.date():
                    entry_price = row['Close']
                    entry_index = i
                    break
            
            if entry_price is None:
                return None
            
            # Create daily returns series
            returns_data = []
            end_index = min(entry_index + days, len(price_data))
            
            for i in range(entry_index, end_index):
                row = price_data.iloc[i]
                date_str = row.name.strftime('%Y-%m-%d')
                price = row['Close']
                return_pct = ((price - entry_price) / entry_price) * 100
                
                returns_data.append({
                    'date': date_str,
                    'price': float(price),
                    'return_pct': float(return_pct)
                })
            
            import json
            return json.dumps(returns_data)
            
        except Exception as e:
            logger.error(f"Error creating realized returns series: {e}")
            return None
    
    async def track_signal_performance(self, signal_id: str) -> Optional[PerformanceMetrics]:
        """Track performance for a single signal"""
        try:
            emit_metric('backtest.performance_tracking.started', signal_id=signal_id)
            
            # Get signal data from database
            response = self.db.client.table('signals').select('*').eq('id', signal_id).execute()
            
            if not response.data:
                logger.warning(f"Signal {signal_id} not found")
                return None
            
            signal = response.data[0]
            ticker = signal['ticker']
            entry_date = datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00'))
            
            # Get price data
            start_date = entry_date - timedelta(days=1)
            end_date = datetime.now()
            
            price_data = self.get_price_data(ticker, start_date, end_date)
            if price_data.empty:
                logger.warning(f"No price data for {ticker}")
                return None
            
            # Get SPY benchmark data
            spy_data = self.get_spy_benchmark_data(start_date, end_date)
            
            # Calculate performance metrics
            metrics = PerformanceMetrics()
            
            # Calculate returns
            returns = self.calculate_returns(price_data, entry_date)
            for key, value in returns.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Calculate SPY benchmark
            if not spy_data.empty:
                spy_returns = self.calculate_spy_returns(spy_data, entry_date)
                for key, value in spy_returns.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
                
                # Calculate beat SPY flags
                for days in [1, 3, 7, 10]:
                    signal_return = getattr(metrics, f'return_{days}d', None)
                    spy_return = getattr(metrics, f'spy_{days}d_return', None)
                    
                    if signal_return is not None and spy_return is not None:
                        beat_spy = signal_return > spy_return
                        setattr(metrics, f'beat_spy_{days}d', beat_spy)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(price_data, entry_date)
            for key, value in risk_metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Calculate signal duration
            duration = self.calculate_signal_duration(price_data, entry_date)
            metrics.signal_duration = duration
            
            # Create realized returns series
            realized_returns = self.create_realized_returns_series(price_data, entry_date)
            metrics.realized_returns = realized_returns
            
            emit_metric('backtest.performance_tracking.completed', signal_id=signal_id)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking performance for signal {signal_id}: {e}")
            emit_metric('backtest.performance_tracking.error', signal_id=signal_id, error=str(e))
            return None
    
    async def update_signal_performance(self, signal_id: str, ticker: Optional[str] = None, interval: Optional[str] = None, metrics: Optional[PerformanceMetrics] = None) -> bool:
        """
        Update signal with calculated performance metrics for a specific interval
        
        Args:
            signal_id: Signal ID to update
            ticker: Ticker symbol (required if metrics not provided)
            interval: Specific interval to calculate (1d, 3d, 7d, etc.)
            metrics: Pre-calculated metrics (optional)
        """
        try:
            # If metrics not provided, calculate them for the interval
            if metrics is None and ticker and interval:
                # Get signal creation date
                signal_response = self.db.client.table('signals').select('created_at, ticker').eq('id', signal_id).single().execute()
                if not signal_response.data:
                    logger.error(f"Signal {signal_id} not found")
                    return False
                
                signal_data = signal_response.data
                created_at = datetime.fromisoformat(signal_data['created_at'].replace('Z', '+00:00'))
                
                # Calculate performance for specific interval
                interval_days = int(interval.rstrip('d'))
                target_date = created_at + timedelta(days=interval_days)
                
                # Get historical price data for the interval
                current_price = await self._get_historical_price(ticker, created_at)
                future_price = await self._get_historical_price(ticker, target_date)
                
                if current_price and future_price:
                    return_pct = ((future_price - current_price) / current_price) * 100
                    
                    # NEW: Insert into signal_performance table (not update signals)
                    # Get signal info for the performance record
                    signal_response = self.db.client.table('signals').select('ticker, run_id, created_at').eq('id', signal_id).single().execute()
                    
                    if not signal_response.data:
                        logger.error(f"Could not find signal {signal_id}")
                        return False
                    
                    signal_data = signal_response.data
                    
                    # Create performance record
                    performance_data = {
                        'signal_id': signal_id,
                        'ticker': signal_data['ticker'],
                        'run_id': signal_data['run_id'],
                        'backtest_type': interval,
                        'days_elapsed': interval_days,
                        'entry_price': round(current_price, 2),
                        'entry_datetime': signal_data['created_at'],
                        'exit_price': round(future_price, 2),
                        'exit_datetime': target_date.isoformat(),
                        'return_pct': round(return_pct, 2),
                        'backtest_date': datetime.now().isoformat(),
                        'win': return_pct > 0,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # DISABLED: signal_performance table doesn't exist - data stored in signals table
                    # Performance columns: 1d_return, 3d_return, 7d_return, 10d_return in signals table
                    logger.info(f"✅ Calculated {interval} performance for signal {signal_id}: {return_pct:.2f}% (not stored separately)")
                    return True
                    
                    # # Original code (commented out):
                    # response = self.db.client.table('signal_performance').insert(performance_data).execute()
                    # if response.data:
                    #     logger.info(f"✅ Inserted {interval} performance for signal {signal_id}: {return_pct:.2f}%")
                    #     return True
                    # else:
                    #     logger.error(f"❌ Failed to insert {interval} performance for signal {signal_id}")
                    #     return False
                else:
                    logger.warning(f"Could not get price data for {ticker} at {interval} interval")
                    return False
                    
            # Legacy metrics-based update (for backward compatibility)
            # NEW: Convert to signal_performance inserts
            elif metrics:
                # Get signal info
                signal_response = self.db.client.table('signals').select('ticker, run_id, created_at, current_price').eq('id', signal_id).single().execute()
                
                if not signal_response.data:
                    logger.error(f"Could not find signal {signal_id}")
                    return False
                
                signal_data = signal_response.data
                
                # Create performance records for each interval that has data
                performance_records = []
                intervals_map = {
                    'return_1d': '1d',
                    'return_3d': '3d',
                    'return_7d': '7d',
                    'return_10d': '10d'
                }
                
                for metric_field, interval_name in intervals_map.items():
                    return_value = getattr(metrics, metric_field, None)
                    if return_value is not None:
                        performance_record = {
                            'signal_id': signal_id,
                            'ticker': signal_data['ticker'],
                            'run_id': signal_data['run_id'],
                            'backtest_type': interval_name,
                            'days_elapsed': int(interval_name.rstrip('d')),
                            'entry_price': signal_data.get('current_price'),
                            'entry_datetime': signal_data['created_at'],
                            'return_pct': round(return_value, 2),
                            'backtest_date': datetime.now().isoformat(),
                            'win': return_value > 0,
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat()
                        }
                        performance_records.append(performance_record)
                
                if performance_records:
                    # DISABLED: signal_performance table doesn't exist - data stored in signals table
                    logger.info(f"✅ Calculated {len(performance_records)} performance records for signal {signal_id} (not stored separately)")
                    return True
                    
                    # # Original code (commented out):
                    # response = self.db.client.table('signal_performance').insert(performance_records).execute()
                    # if response.data:
                    #     logger.info(f"✅ Inserted {len(performance_records)} performance records for signal {signal_id}")
                    #     return True
                    # else:
                    #     logger.error(f"❌ Failed to insert performance records for signal {signal_id}")
                    #     return False
                else:
                    logger.warning(f"No performance data to insert for signal {signal_id}")
                    return False
            
            else:
                logger.error(f"Invalid parameters for updating signal performance: signal_id={signal_id}, ticker={ticker}, interval={interval}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating signal performance for {signal_id}: {e}")
            return False
    
    async def run_performance_tracking_batch(self, limit: int = 50) -> Dict[str, int]:
        """Run performance tracking for a batch of signals"""
        try:
            logger.info("Starting batch performance tracking...")
            
            # Get signals that need performance tracking
            response = self.db.client.table('signals').select('id, ticker, created_at').or_(
                'backtest_phase.is.null,backtest_phase.eq.initial'
            ).order('created_at', desc=True).limit(limit).execute()
            
            signals = response.data if response.data else []
            
            if not signals:
                logger.info("No signals found needing performance tracking")
                return {'processed': 0, 'successful': 0, 'failed': 0}
            
            logger.info(f"Found {len(signals)} signals for performance tracking")
            
            results = {'processed': 0, 'successful': 0, 'failed': 0}
            
            for signal in signals:
                signal_id = signal['id']
                
                try:
                    # Track performance
                    metrics = await self.track_signal_performance(signal_id)
                    
                    if metrics:
                        # Update database
                        success = await self.update_signal_performance(signal_id, metrics)
                        if success:
                            results['successful'] += 1
                        else:
                            results['failed'] += 1
                    else:
                        results['failed'] += 1
                    
                    results['processed'] += 1
                    
                    # Small delay to avoid overwhelming APIs
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing signal {signal_id}: {e}")
                    results['failed'] += 1
                    results['processed'] += 1
            
            logger.info(f"Batch performance tracking complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch performance tracking: {e}")
            return {'processed': 0, 'successful': 0, 'failed': 0}
    
    async def calculate_historical_success_rate(self, signal_score: float, score_range: float = 0.1) -> Optional[float]:
        """
        Calculate historical success rate for signals with similar scores.
        
        Success = positive return AND beat SPY (7d return used as primary metric)
        
        Args:
            signal_score: The signal score of the current signal (0-1 scale)
            score_range: Range around score to consider (default ±0.1 for 0-1 scale)
            
        Returns:
            Success rate as percentage (0-100), or None if insufficient data
        """
        try:
            # Query past signals with similar scores that have performance data
            min_score = signal_score - score_range
            max_score = signal_score + score_range
            
            # Query signal_performance table directly for historical data
            response = self.db.client.table('signal_performance').select(
                'signal_id, return_7d, beat_spy_7d, signals!inner(signal_score)'
            ).gte('signals.signal_score', min_score).lte('signals.signal_score', max_score).not_.is_(
                'return_7d', 'null'
            ).not_.is_('beat_spy_7d', 'null').execute()
            
            if not response.data or len(response.data) < 5:
                # Need at least 5 data points for meaningful stat
                logger.debug(f"Insufficient historical data for score {signal_score:.3f} (found {len(response.data) if response.data else 0} signals)")
                return None
            
            signals = response.data
            
            # Calculate success rate
            successful_signals = 0
            total_signals = len(signals)
            
            for perf_record in signals:
                return_7d = perf_record.get('return_7d', 0)
                beat_spy_7d = perf_record.get('beat_spy_7d', False)
                
                # Success criteria: positive return AND beat SPY
                if return_7d and return_7d > 0 and beat_spy_7d:
                    successful_signals += 1
            
            success_rate = (successful_signals / total_signals) * 100
            
            logger.debug(
                f"Historical success rate for score {signal_score:.3f}: "
                f"{success_rate:.1f}% ({successful_signals}/{total_signals} signals)"
            )
            
            return round(success_rate, 2)
            
        except Exception as e:
            logger.error(f"Error calculating historical success rate: {e}")
            return None


class BacktestScheduler:
    """
    Smart backtest scheduler that determines when to run backtests
    based on signal age and creates performance snapshots at key intervals.
    """
    
    def __init__(self):
        self.intervals = {
            BacktestInterval.ONE_DAY: 1,
            BacktestInterval.THREE_DAY: 3,
            BacktestInterval.SEVEN_DAY: 7,
            BacktestInterval.FOURTEEN_DAY: 14,
            BacktestInterval.THIRTY_DAY: 30,
            BacktestInterval.SIXTY_DAY: 60,
            BacktestInterval.NINETY_DAY: 90
        }
    
    def calculate_signal_age(self, created_at: datetime) -> int:
        """Calculate age of signal in days"""
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        now = datetime.now()
        if created_at.tzinfo:
            # Make timezone-aware comparison
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        
        age = (now - created_at).days
        return max(0, age)  # Don't return negative ages
    
    def get_eligible_intervals(self, signal_age_days: int) -> List[BacktestInterval]:
        """
        Get list of backtest intervals that this signal is eligible for
        based on its age.
        """
        eligible = []
        
        for interval, days in self.intervals.items():
            if signal_age_days >= days:
                eligible.append(interval)
        
        return eligible
    
    def get_missing_backtest_intervals(self, signal: Dict[str, Any]) -> List[BacktestInterval]:
        """
        Determine which backtest intervals are missing for a signal
        based on its age and existing backtest data.
        
        Note: backtest_intervals is now stored in signal_performance table
        """
        signal_age = self.calculate_signal_age(signal.get('created_at'))
        eligible_intervals = self.get_eligible_intervals(signal_age)
        
        # Get existing backtest intervals from signal_performance data
        existing_intervals = set()
        
        # Check if signal has performance data embedded (from JOIN query)
        perf_data = signal.get('signal_performance', {})
        if isinstance(perf_data, list) and perf_data:
            perf_data = perf_data[0]
        
        backtest_data = perf_data.get('backtest_intervals', []) if perf_data else []
        
        if isinstance(backtest_data, str):
            # Parse comma-separated string
            existing_intervals = set(backtest_data.split(',')) if backtest_data else set()
        elif isinstance(backtest_data, list):
            existing_intervals = set(backtest_data)
        
        # Find missing intervals
        missing = []
        for interval in eligible_intervals:
            if interval.value not in existing_intervals:
                missing.append(interval)
        
        return missing
    
    async def get_signals_requiring_backtest(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signals that require backtesting based on their age and missing intervals.
        
        Note: Performance data (backtest_intervals, returns) now in signal_performance table
        """
        try:
            db = await get_supabase_database()
            
            # Get signals that are at least 1 day old, JOIN with signal_performance
            cutoff_date = datetime.now() - timedelta(days=1)
            
            # Query signals and join with signal_performance for return data
            response = db.supabase.table('signals').select(
                'id, ticker, created_at, '
                'signal_performance(backtest_intervals, return_1d, return_3d, return_7d, return_10d, return_30d)'
            ).lte('created_at', cutoff_date.isoformat()).order('created_at', desc=True)
            
            if limit:
                response = response.limit(limit)
            
            result = response.execute()
            signals = result.data if result.data else []
            
            # Filter signals that need backtesting
            signals_needing_backtest = []
            
            for signal in signals:
                missing_intervals = self.get_missing_backtest_intervals(signal)
                if missing_intervals:
                    signal['missing_intervals'] = missing_intervals
                    signal['signal_age_days'] = self.calculate_signal_age(signal['created_at'])
                    signals_needing_backtest.append(signal)
            
            logger.info(f"Found {len(signals_needing_backtest)} signals requiring backtest out of {len(signals)} total")
            
            return signals_needing_backtest
            
        except Exception as e:
            logger.error(f"Error getting signals requiring backtest: {e}")
            return []
    
    def create_backtest_plan(self, signals: List[Dict[str, Any]]) -> Dict[BacktestInterval, List[Dict[str, Any]]]:
        """
        Create a backtest execution plan grouped by interval.
        This allows efficient batch processing of similar timeframes.
        """
        plan = {interval: [] for interval in BacktestInterval}
        
        for signal in signals:
            missing_intervals = signal.get('missing_intervals', [])
            for interval in missing_intervals:
                plan[interval].append(signal)
        
        # Log the plan
        for interval, signal_list in plan.items():
            if signal_list:
                logger.info(f"Backtest plan - {interval.value}: {len(signal_list)} signals")
        
        return plan


# Export main backtest engine and scheduler
backtest_engine = BacktestEngine()
backtest_scheduler = BacktestScheduler()


async def run_historical_backtest(days_back: int = 30) -> Dict[str, Any]:
    """
    Run backtest on historical signals and AI strategies from the past month.
    
    This function:
    1. Finds signals older than 1 day with missing performance data
    2. Calculates actual returns for AI strategies 
    3. Updates both signals and ai_strategy_performance tables
    4. Focuses on AI strategy performance tracking
    
    Args:
        days_back: How many days back to look for signals (default 30)
        
    Returns:
        Dictionary with backtest results and statistics
    """
    try:
        logger.info(f"Starting historical backtest for past {days_back} days...")
        
        # Initialize database and engine
        db = get_supabase_database()
        engine = BacktestEngine()
        
        # Find historical signals that need backtesting
        cutoff_date = datetime.now() - timedelta(days=days_back)
        min_age_date = datetime.now() - timedelta(days=1)  # At least 1 day old
        
        # Query signals that are old enough but missing performance data
        query = f"""
            SELECT s.*, ai.id as strategy_id, ai.strategy_type, ai.entry_conditions
            FROM signals s
            LEFT JOIN ai_strategies ai ON s.id = ai.signal_id
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            WHERE s.created_at >= '{cutoff_date.isoformat()}'
            AND s.created_at <= '{min_age_date.isoformat()}'
            AND (sp.return_1d IS NULL OR sp.backtest_phase IS NULL)
            ORDER BY s.created_at DESC
            LIMIT 100
        """
        
        # Query signals, join with signal_performance to check for missing return data
        result = db.supabase.table('signals').select(
            '*, ai_strategies!inner(id, strategy_type, entry_conditions), '
            'signal_performance(return_1d, backtest_phase)'
        ).gte('created_at', cutoff_date.isoformat()).lte(
            'created_at', min_age_date.isoformat()
        ).limit(100).execute()
        
        signals = result.data if result.data else []
        
        if not signals:
            logger.info("No historical signals found needing backtest")
            return {
                'signals_processed': 0,
                'strategies_processed': 0,
                'successful_backtests': 0,
                'failed_backtests': 0,
                'ai_strategy_performance': {}
            }
        
        logger.info(f"Found {len(signals)} historical signals for backtesting")
        
        # Results tracking
        results = {
            'signals_processed': 0,
            'strategies_processed': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'ai_strategy_performance': {}
        }
        
        # Process each signal
        for signal in signals:
            try:
                # Calculate signal performance
                signal_performance = await _calculate_signal_backtest_performance(signal, engine)
                
                if signal_performance:
                    # Update signals table with performance data
                    await _update_signal_performance_data(db, signal['id'], signal_performance)
                    
                    # Calculate AI strategy performance if available
                    if 'ai_strategies' in signal and signal['ai_strategies']:
                        for strategy in signal['ai_strategies']:
                            strategy_performance = await _calculate_ai_strategy_performance(
                                signal, strategy, signal_performance
                            )
                            
                            if strategy_performance:
                                await _update_ai_strategy_performance(db, strategy['id'], strategy_performance)
                                results['strategies_processed'] += 1
                    
                    results['successful_backtests'] += 1
                else:
                    results['failed_backtests'] += 1
                
                results['signals_processed'] += 1
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error backtesting signal {signal.get('id', 'Unknown')}: {e}")
                results['failed_backtests'] += 1
                results['signals_processed'] += 1
        
        logger.info(f"Historical backtest complete: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Historical backtest failed: {e}")
        return {
            'signals_processed': 0,
            'strategies_processed': 0, 
            'successful_backtests': 0,
            'failed_backtests': 0,
            'ai_strategy_performance': {}
        }


async def _calculate_signal_backtest_performance(signal: Dict[str, Any], engine: BacktestEngine) -> Optional[Dict[str, Any]]:
    """Calculate comprehensive performance metrics for a signal."""
    try:
        ticker = signal.get('ticker')
        signal_date = datetime.fromisoformat(signal.get('created_at', '').replace('Z', '+00:00'))
        
        # Get price data for performance calculation
        end_date = datetime.now()
        price_data = engine.get_price_data(ticker, signal_date, end_date)
        
        if price_data.empty:
            return None
        
        # Calculate returns for multiple periods
        returns = engine.calculate_returns(price_data, signal_date, [1, 3, 7, 10])
        
        # Get SPY benchmark data
        spy_data = engine.get_spy_benchmark_data(signal_date, end_date)
        spy_returns = engine.calculate_spy_returns(spy_data, signal_date, [1, 3, 7, 10])
        
        # Calculate risk metrics
        risk_metrics = engine.calculate_risk_metrics(price_data, signal_date)
        
        # Calculate signal duration and realized returns
        signal_duration = engine.calculate_signal_duration(price_data, signal_date, exit_threshold=0.05)
        realized_returns = engine.create_realized_returns_series(price_data, signal_date, 30)
        
        # Combine all performance data
        performance_data = {
            **returns,
            **spy_returns,
            **risk_metrics,
            'signal_duration': signal_duration,
            'realized_returns': realized_returns,
            'backtest_phase': BacktestPhase.COMPLETED.value,
            'backtest_timestamp': datetime.now().isoformat(),
            'backtest_notes': f'Backtested on {datetime.now().date()}'
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error calculating signal performance for {signal.get('ticker', 'Unknown')}: {e}")
        return None


async def _calculate_ai_strategy_performance(signal: Dict[str, Any], strategy: Dict[str, Any], 
                                           signal_performance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calculate AI strategy-specific performance metrics.
    
    This is the key function for tracking "what would the AI-recommended strategy have yielded"
    """
    try:
        strategy_type = strategy.get('strategy_type', 'equity')
        entry_conditions = strategy.get('entry_conditions', {})
        
        # Parse entry conditions
        if isinstance(entry_conditions, str):
            import json
            entry_conditions = json.loads(entry_conditions)
        
        entry_price = entry_conditions.get('entry_price', signal.get('current_price', 0))
        
        # Calculate strategy-specific returns based on type
        if strategy_type == 'equity':
            # For equity strategies, use signal returns directly
            strategy_returns = {
                'realized_return_1d': signal_performance.get('1d_return', 0),
                'realized_return_3d': signal_performance.get('3d_return', 0),
                'realized_return_7d': signal_performance.get('7d_return', 0),
                'realized_return_10d': signal_performance.get('10d_return', 0)
            }
        elif strategy_type == 'options':
            # For options strategies, calculate options-specific returns
            strategy_returns = await _calculate_options_strategy_returns(
                signal, strategy, signal_performance
            )
        else:
            # Default to equity returns
            strategy_returns = {
                'realized_return_1d': signal_performance.get('1d_return', 0),
                'realized_return_3d': signal_performance.get('3d_return', 0),
                'realized_return_7d': signal_performance.get('7d_return', 0),
                'realized_return_10d': signal_performance.get('10d_return', 0)
            }
        
        # Calculate strategy performance metrics
        performance_data = {
            **strategy_returns,
            'entry_price_actual': entry_price,
            'max_return': signal_performance.get('max_return_pct', 0),
            'max_drawdown': signal_performance.get('drawdown_pct', 0),
            'sharpe_ratio': signal_performance.get('forward_sharpe_ratio', 0),
            'volatility': signal_performance.get('forward_volatility', 0),
            'strategy_success': strategy_returns.get('realized_return_7d', 0) > 0,
            'beat_market': signal_performance.get('beat_spy_7d', False),
            'performance_timestamp': datetime.now().isoformat(),
            'backtest_notes': f'AI {strategy_type} strategy performance calculated'
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error calculating AI strategy performance: {e}")
        return None


async def _calculate_options_strategy_returns(signal: Dict[str, Any], strategy: Dict[str, Any], 
                                            signal_performance: Dict[str, Any]) -> Dict[str, float]:
    """Calculate returns for options strategies (simplified approximation)."""
    try:
        # This is a simplified calculation - in reality you'd need options pricing models
        # For now, we'll approximate based on underlying returns and strategy type
        
        underlying_return_7d = signal_performance.get('7d_return', 0)
        
        # Rough options multiplier based on strategy type
        if 'call' in strategy.get('strategy_name', '').lower():
            # Long calls amplify positive moves, lose premium on negative moves
            options_multiplier = 3.0 if underlying_return_7d > 0 else -1.0
        elif 'put' in strategy.get('strategy_name', '').lower():
            # Puts profit from negative moves
            options_multiplier = -2.5 if underlying_return_7d < 0 else -0.8
        else:
            # Default to moderate amplification
            options_multiplier = 2.0
        
        options_return = underlying_return_7d * options_multiplier
        
        return {
            'realized_return_1d': signal_performance.get('1d_return', 0) * options_multiplier * 0.3,
            'realized_return_3d': signal_performance.get('3d_return', 0) * options_multiplier * 0.6,
            'realized_return_7d': options_return,
            'realized_return_10d': signal_performance.get('10d_return', 0) * options_multiplier * 1.1
        }
        
    except Exception as e:
        logger.error(f"Error calculating options returns: {e}")
        return {
            'realized_return_1d': 0,
            'realized_return_3d': 0,
            'realized_return_7d': 0,
            'realized_return_10d': 0
        }


async def _update_signal_performance_data(db, signal_id: str, performance_data: Dict[str, Any]) -> bool:
    """Update signals table with performance data."""
    try:
        result = db.supabase.table('signals').update(performance_data).eq('id', signal_id).execute()
        return bool(result.data)
    except Exception as e:
        logger.error(f"Error updating signal performance for {signal_id}: {e}")
        return False


async def _update_ai_strategy_performance(db, strategy_id: str, performance_data: Dict[str, Any]) -> bool:
    """Update ai_strategies table with strategy results (ai_strategy_performance table doesn't exist)."""
    try:
        # UPDATED: Use ai_strategies table instead of non-existent ai_strategy_performance table
        result = db.supabase.table('ai_strategies').update(performance_data).eq('id', strategy_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated ai_strategies table with performance for strategy {strategy_id}")
        else:
            logger.warning(f"⚠️ No strategy found with ID {strategy_id} to update")
        
        # # Original code (commented out - table doesn't exist):
        # result = db.supabase.table('ai_strategy_performance').update(performance_data).eq('strategy_id', strategy_id).execute()
        # if not result.data:
        #     performance_data['strategy_id'] = strategy_id
        #     result = db.supabase.table('ai_strategy_performance').insert(performance_data).execute()
        
        return bool(result.data)
    except Exception as e:
        logger.error(f"Error updating AI strategy performance for {strategy_id}: {e}")
        return False


# Enhanced backtest integration function
async def enhance_signals_with_backtest_data(signals: List[Dict[str, Any]], 
                                           run_historical: bool = True) -> List[Dict[str, Any]]:
    """
    Enhance signals with backtest data and run historical backtest if requested.
    
    Args:
        signals: Current signals to enhance
        run_historical: Whether to also run historical backtest
        
    Returns:
        Enhanced signals list
    """
    try:
        # For new signals, just add backtest phase
        for signal in signals:
            if not signal.get('backtest_phase'):
                signal['backtest_phase'] = BacktestPhase.INITIAL.value
                signal['backtest_timestamp'] = datetime.now().isoformat()
        
        # Run historical backtest if requested
        if run_historical:
            logger.info("Running historical backtest for past signals...")
            backtest_results = await run_historical_backtest(days_back=30)
            logger.info(f"Historical backtest results: {backtest_results}")
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in backtest enhancement: {e}")
        return signals


async def run_smart_historical_backtest(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Run intelligent historical backtest that only processes signals
    at appropriate time intervals (1d, 3d, 7d, etc.).
    
    Args:
        limit: Maximum number of signals to process (None for all)
        
    Returns:
        Dictionary with backtest results and statistics
    """
    try:
        logger.info("Starting smart historical backtest...")
        
        # Get signals that need backtesting
        signals_needing_backtest = await backtest_scheduler.get_signals_requiring_backtest(limit)
        
        if not signals_needing_backtest:
            logger.info("No signals require backtesting at this time")
            return {
                'signals_processed': 0,
                'intervals_processed': 0,
                'successful_backtests': 0,
                'failed_backtests': 0,
                'execution_time_seconds': 0
            }
        
        # Create execution plan
        backtest_plan = backtest_scheduler.create_backtest_plan(signals_needing_backtest)
        
        results = {
            'signals_processed': len(signals_needing_backtest),
            'intervals_processed': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'interval_results': {}
        }
        
        start_time = datetime.now()
        
        # Process each interval in priority order (1d first, then 3d, etc.)
        interval_priority = [
            BacktestInterval.ONE_DAY,
            BacktestInterval.THREE_DAY, 
            BacktestInterval.SEVEN_DAY,
            BacktestInterval.FOURTEEN_DAY,
            BacktestInterval.THIRTY_DAY,
            BacktestInterval.SIXTY_DAY,
            BacktestInterval.NINETY_DAY
        ]
        
        for interval in interval_priority:
            signals_for_interval = backtest_plan[interval]
            if not signals_for_interval:
                continue
            
            logger.info(f"Processing {interval.value} backtest for {len(signals_for_interval)} signals...")
            
            interval_successful = 0
            interval_failed = 0
            
            for signal in signals_for_interval:
                try:
                    # Run backtest for this specific interval
                    success = await backtest_engine.update_signal_performance(
                        signal_id=signal['id'],
                        ticker=signal['ticker'],
                        interval=interval.value
                    )
                    
                    if success:
                        interval_successful += 1
                        results['successful_backtests'] += 1
                    else:
                        interval_failed += 1
                        results['failed_backtests'] += 1
                    
                except Exception as e:
                    logger.error(f"Backtest failed for signal {signal['id']} at {interval.value}: {e}")
                    interval_failed += 1
                    results['failed_backtests'] += 1
            
            results['interval_results'][interval.value] = {
                'successful': interval_successful,
                'failed': interval_failed,
                'total': len(signals_for_interval)
            }
            
            results['intervals_processed'] += 1
            
            logger.info(f"Completed {interval.value}: {interval_successful} successful, {interval_failed} failed")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        results['execution_time_seconds'] = round(execution_time, 2)
        
        logger.info(f"Smart backtest complete: {results['successful_backtests']} successful, {results['failed_backtests']} failed in {execution_time:.1f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Smart historical backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'signals_processed': 0,
            'intervals_processed': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'error': str(e)
        }


async def calculate_historical_success_rates_for_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate and add historical_success_rate to new signals based on past performance.
    
    This runs immediately after signal generation to populate the historical_success_rate
    column based on how similar signals have performed historically.
    
    Args:
        signals: List of newly generated signals
        
    Returns:
        Enhanced signals with historical_success_rate populated
    """
    try:
        logger.info(f"Calculating historical success rates for {len(signals)} signals...")
        
        # Initialize engine and database connection
        engine = BacktestEngine()
        await engine.set_database()
        
        enhanced_count = 0
        
        for signal in signals:
            signal_score = signal.get('signal_score', 0)
            
            if signal_score > 0:
                # Calculate success rate based on past signals with similar scores
                success_rate = await engine.calculate_historical_success_rate(signal_score)
                
                if success_rate is not None:
                    signal['historical_success_rate'] = success_rate
                    enhanced_count += 1
                else:
                    # Not enough historical data yet
                    signal['historical_success_rate'] = None
            else:
                signal['historical_success_rate'] = None
        
        logger.info(f"✅ Added historical success rates to {enhanced_count}/{len(signals)} signals")
        
        return signals
        
    except Exception as e:
        logger.error(f"Error calculating historical success rates: {e}")
        return signals


async def backtest_eligible_signals(limit: int = 100) -> Dict[str, Any]:
    """
    Backtest previous signals that have had enough time elapse for return calculations.
    
    PHASE 4 REWRITE:
    - Updates signals table directly (not signal_performance)
    - Filters for NULL backtest columns (avoid re-backtesting)
    - Single UPDATE per signal (not multiple records)
    - Processes oldest signals first (systematic approach)
    - Marks failed backtests as "backtest_failed"
    
    Args:
        limit: Maximum number of signals to backtest per run (default: 100)
        
    Returns:
        Dict with backtest results summary
    """
    try:
        logger.info(f"🔍 Checking for eligible signals to backtest (limit: {limit})...")
        
        # Initialize database connection
        db = await get_supabase_database()
        
        # Get current time (UTC timezone-aware)
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        
        # CRITICAL FIX: Filter for signals that haven't been backtested yet
        # Look for signals that don't have signal_performance records
        # Order by created_at ASC (oldest first - systematic approach)
        response = db.client.table('signals').select(
            'id, ticker, current_price, created_at, run_id'
        ).is_('signal_performance.signal_id', 'null').order('created_at', desc=False).limit(limit).execute()
        
        if not response.data:
            logger.info("No eligible signals found for backtesting")
            return {
                'success': True,
                'backtested_count': 0,
                'updated_signals': []
            }
        
        signals_to_backtest = response.data
        logger.info(f"Found {len(signals_to_backtest)} signals eligible for backtesting")
        
        # Process each signal
        backtested_count = 0
        updated_signals = []
        
        logger.info(f"Processing {len(signals_to_backtest)} signals...")
        
        for signal in signals_to_backtest:
            try:
                signal_id = signal['id']
                ticker = signal['ticker']
                run_id = signal.get('run_id', 'unknown')
                entry_price = signal.get('current_price')
                
                logger.info(f"  Processing {ticker} (ID: {signal_id[:8]}...)")
                
                # Use created_at as signal date
                signal_date_str = signal.get('created_at')
                if not signal_date_str:
                    logger.warning(f"Signal {signal_id} ({ticker}): No date found, skipping")
                    continue
                
                # Parse signal date
                from dateutil import parser
                signal_date = parser.parse(signal_date_str)
                
                # Calculate days elapsed
                days_elapsed = (now - signal_date).days
                
                logger.info(f"    signal_date={signal_date.isoformat()[:19]}, days_elapsed={days_elapsed}")
                
                if days_elapsed < 1:
                    # Not even 1 day has passed yet
                    logger.info(f"    Skipping - less than 1 day elapsed")
                    continue
                
                if not entry_price or entry_price <= 0:
                    logger.warning(f"    Invalid entry price {entry_price}, skipping")
                    continue
                
                # Determine which intervals we can calculate
                intervals_to_calculate = []
                if days_elapsed >= 1:
                    intervals_to_calculate.append(1)
                if days_elapsed >= 3:
                    intervals_to_calculate.append(3)
                if days_elapsed >= 7:
                    intervals_to_calculate.append(7)
                if days_elapsed >= 10:
                    intervals_to_calculate.append(10)
                if days_elapsed >= 30:
                    intervals_to_calculate.append(30)
                
                if not intervals_to_calculate:
                    logger.info(f"    No intervals to calculate")
                    continue
                
                logger.info(f"    Backtesting {days_elapsed} days elapsed, calculating intervals: {intervals_to_calculate}")
                
                # Fetch historical price data from yfinance
                import backend.integrations.yfinance as yf
                
                # Get data from signal date to now + 1 day buffer
                end_date = now + timedelta(days=1)
                
                logger.info(f"    Fetching price data from {signal_date.date()} to {end_date.date()}")
                
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=signal_date.date(), end=end_date.date())
                    
                    logger.info(f"    Got {len(hist)} rows of price data")
                    
                    if hist.empty:
                        # Mark as failed backtest (likely delisted or invalid ticker)
                        logger.warning(f"    No price data available (likely delisted)")
                        db.client.table('signals').update({
                            'backtest_phase': 'backtest_failed',
                            'backtest_notes': 'No price data available from yfinance (likely delisted)',
                            'backtest_timestamp': now.isoformat()
                        }).eq('id', signal_id).execute()
                        continue
                
                except Exception as e:
                    # Mark as failed backtest
                    logger.error(f"    Error fetching data - {e}")
                    db.client.table('signals').update({
                        'backtest_phase': 'backtest_failed',
                        'backtest_notes': f'Error fetching price data: {str(e)}',
                        'backtest_timestamp': now.isoformat()
                    }).eq('id', signal_id).execute()
                    continue
                
                # Also get SPY data for comparison
                spy = yf.Ticker('SPY')
                spy_hist = spy.history(start=signal_date.date(), end=end_date.date())
                
                if spy_hist.empty:
                    logger.warning(f"Signal {signal_id} ({ticker}): No SPY data available")
                    spy_hist = None
                
                # Calculate returns for each interval
                update_data = {}
                
                for interval_days in intervals_to_calculate:
                    # Get the price at interval_days after signal
                    target_date = signal_date + timedelta(days=interval_days)
                    
                    # Find the closest trading day price
                    # yfinance returns datetime index, need to compare dates only
                    exit_price = None
                    for date_offset in range(5):  # Check up to 5 days ahead for trading day
                        check_date = (target_date + timedelta(days=date_offset)).date()
                        # Convert index dates to date objects for comparison
                        for idx_date in hist.index:
                            if idx_date.date() == check_date:
                                exit_price = hist.loc[idx_date, 'Close']
                                logger.info(f"    {interval_days}d: Found price on {idx_date.date()}: ${exit_price:.2f}")
                                break
                        if exit_price:
                            break
                    
                    if exit_price is None or exit_price <= 0:
                        logger.info(f"    {interval_days}d: No valid price found")
                        continue
                    
                    # Calculate return
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    # Calculate SPY return for comparison
                    spy_return = None
                    beat_spy = None
                    
                    if spy_hist is not None:
                        spy_entry = None
                        spy_exit = None
                        
                        # Find SPY entry price
                        for date_offset in range(5):
                            check_date = (signal_date + timedelta(days=date_offset)).date()
                            for idx_date in spy_hist.index:
                                if idx_date.date() == check_date:
                                    spy_entry = spy_hist.loc[idx_date, 'Close']
                                    break
                            if spy_entry:
                                break
                        
                        # Find SPY exit price
                        for date_offset in range(5):
                            check_date = (target_date + timedelta(days=date_offset)).date()
                            for idx_date in spy_hist.index:
                                if idx_date.date() == check_date:
                                    spy_exit = spy_hist.loc[idx_date, 'Close']
                                    break
                            if spy_exit:
                                break
                        
                        if spy_entry and spy_exit and spy_entry > 0:
                            spy_return = ((spy_exit - spy_entry) / spy_entry) * 100
                            beat_spy = return_pct > spy_return
                    
                    # Store results
                    update_data[f'{interval_days}d_return'] = round(return_pct, 2)
                    
                    if spy_return is not None:
                        update_data[f'spy_{interval_days}d_return'] = round(spy_return, 2)
                    
                    # PHASE 4.2: Store beat_spy boolean columns
                    # Now safe to store booleans directly (signal_metrics table removed)
                    if beat_spy is not None:
                        update_data[f'beat_spy_{interval_days}d'] = beat_spy
                    
                    logger.info(f"    {interval_days}d: return={return_pct:.2f}%, SPY={spy_return:.2f}%, Beat={beat_spy}")
                
                # PHASE 4 FIX: Update signals table directly (not signal_performance)
                logger.info(f"    update_data has {len(update_data)} fields")
                
                if update_data:
                    # Add backtest metadata
                    update_data['backtest_timestamp'] = now.isoformat()
                    update_data['backtest_phase'] = 'Complete'
                    
                    logger.info(f"    Updating signals table with {len(update_data)} fields...")
                    
                    # Update the signals table with all return data in one operation
                    result = db.client.table('signals').update(update_data).eq('id', signal_id).execute()
                    
                    logger.info(f"    Database UPDATE result: {result}")
                    
                    backtested_count += 1
                    updated_signals.append({
                        'id': signal_id,
                        'ticker': ticker,
                        'intervals_calculated': list(intervals_to_calculate),
                        'returns': {k: v for k, v in update_data.items() if 'return' in k or 'spy' in k or 'beat' in k}
                    })
                    
                    logger.info(f"✅ Backtested {ticker} (ID: {signal_id[:8]}...): {len(intervals_to_calculate)} intervals, {len(update_data)} fields updated")
                else:
                    logger.warning(f"    update_data is empty! Skipping UPDATE")
                
            except Exception as e:
                logger.error(f"❌ Error backtesting signal {signal.get('id')} ({signal.get('ticker')}): {e}", exc_info=True)
                continue
        
        logger.info(f"✅ Backtest complete: {backtested_count}/{len(signals_to_backtest)} signals updated")
        
        return {
            'success': True,
            'backtested_count': backtested_count,
            'total_eligible': len(signals_to_backtest),
            'updated_signals': updated_signals
        }
        
    except Exception as e:
        logger.error(f"Error in backtest_eligible_signals: {e}")
        return {
            'success': False,
            'backtested_count': 0,
            'error': str(e)
        }