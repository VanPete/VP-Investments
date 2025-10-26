"""
Phase 6: Performance Tracking System
=====================================

Tracks signal performance over time intervals without trade simulation.

Key Features:
1. Data Collection - Fetch historical price data
2. Performance Tracking - Calculate returns at 1d, 3d, 7d, 10d, 14d, 30d, 90d
3. SPY Benchmark - Compare signal performance vs market
4. Auto-scheduling - Run when pipeline executes, backfill old signals

Design:
- Baseline: Next day open (avoids lookahead bias)
- Intervals: 1d, 3d, 7d, 10d, 14d, 30d, 90d
- Success Criteria: 7d positive return AND beats SPY
- Base Capital: $10,000
- Status: pending (no baseline) → in_progress (some intervals) → completed (all intervals)
- Tracking: NULL checks + date math determine what to calculate
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class PositionStatus(Enum):
    """Position lifecycle states."""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"  # Hit stop loss
    PROFIT_TAKEN = "profit_taken"  # Hit take profit


class BacktestInterval(Enum):
    """Standard backtest intervals for performance tracking."""
    ONE_DAY = "1d"
    THREE_DAY = "3d"
    SEVEN_DAY = "7d"
    TEN_DAY = "10d"
    FOURTEEN_DAY = "14d"
    THIRTY_DAY = "30d"
    SIXTY_DAY = "60d"
    NINETY_DAY = "90d"


class BacktestPhase(Enum):
    """Backtest execution phases."""
    PENDING = "pending"  # Not yet backtested
    IN_PROGRESS = "in_progress"  # Currently backtesting
    COMPLETED = "completed"  # Backtest finished successfully
    FAILED = "failed"  # Backtest encountered errors


@dataclass
class HistoricalPrice:
    """Historical price data for a single ticker."""
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    dividends: Optional[float] = 0.0
    splits: Optional[float] = 1.0


@dataclass
class BacktestPosition:
    """Represents a backtested trading position."""
    ticker: str
    signal_score: float
    signal_date: datetime
    
    # Entry
    entry_date: datetime
    entry_price: float
    position_size: float  # % of portfolio or fixed amount
    
    # Exit
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # stop_loss, take_profit, time_exit, signal_decay
    
    # Performance
    return_pct: Optional[float] = None
    return_dollars: Optional[float] = None
    hold_days: Optional[int] = None
    status: PositionStatus = PositionStatus.OPEN
    
    # Risk management
    stop_loss_pct: float = 0.15  # 15% stop loss
    take_profit_pct: float = 0.30  # 30% take profit
    max_hold_days: int = 90  # Exit after 90 days


@dataclass
class BacktestResult:
    """Aggregate backtest results for a signal or strategy."""
    signal_id: Optional[str] = None
    ticker: Optional[str] = None
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Returns
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    
    # Time metrics
    avg_hold_days: float = 0.0
    median_hold_days: float = 0.0
    
    # Position breakdown
    positions: List[BacktestPosition] = field(default_factory=list)
    
    # Metadata
    backtest_start_date: Optional[datetime] = None
    backtest_end_date: Optional[datetime] = None
    total_capital: float = 100000.0  # Starting capital


# ============================================================================
# Phase 6.1: Historical Data Collection
# ============================================================================

class HistoricalDataFetcher:
    """
    Fetch and cache historical price data for backtesting.
    
    Data sources:
    - yfinance for OHLCV data
    - Handles splits, dividends, and adjustments
    - Caches data to avoid repeated API calls
    """
    
    def __init__(self, lookback_years: int = 2, cache_dir: str = "data/backtest_cache"):
        """
        Initialize historical data fetcher.
        
        Args:
            lookback_years: Years of historical data to fetch (default: 2)
            cache_dir: Directory to cache historical data
        """
        self.lookback_years = lookback_years
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache for active backtesting session
        self._price_cache: Dict[str, pd.DataFrame] = {}
    
    async def fetch_historical_data(
        self, 
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data (default: lookback_years ago)
            end_date: End date for historical data (default: today)
        
        Returns:
            Dictionary mapping ticker -> DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * self.lookback_years)
        if end_date is None:
            end_date = datetime.now()
        
        self.logger.debug(
            f"[BACKTEST] Fetching historical data for {len(tickers)} tickers "
            f"({start_date.date()} to {end_date.date()})"
        )
        
        # Check cache first
        uncached_tickers = [t for t in tickers if t not in self._price_cache]
        
        if uncached_tickers:
            # Fetch data for uncached tickers
            new_data = await self._fetch_from_yfinance(
                uncached_tickers, start_date, end_date
            )
            self._price_cache.update(new_data)
        
        # Return data for all requested tickers
        return {ticker: self._price_cache.get(ticker) 
                for ticker in tickers 
                if ticker in self._price_cache}
    
    async def _fetch_from_yfinance(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from yfinance API.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        import yfinance as yf
        
        historical_data = {}
        
        # Fetch in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            
            try:
                # Download batch
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True,
                    group_by='ticker',
                    auto_adjust=True  # Suppress FutureWarning
                )
                
                # Parse results
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker]
                        
                        if not ticker_data.empty:
                            # Clean data
                            ticker_data = ticker_data.dropna(how='all')
                            historical_data[ticker] = ticker_data
                            self.logger.debug(
                                f"   [SUCCESS] {ticker}: {len(ticker_data)} days"
                            )
                        else:
                            self.logger.warning(f"   [SKIP] {ticker}: No data available")
                    except Exception as e:
                        self.logger.warning(f"   [ERROR] {ticker}: {e}")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"[ERROR] Batch fetch failed: {e}")
        
        self.logger.debug(
            f"   [SUCCESS] Fetched historical data for {len(historical_data)}/{len(tickers)} tickers"
        )
        
        return historical_data
    
    def get_price_at_date(
        self,
        ticker: str,
        date: datetime,
        price_type: str = 'close'
    ) -> Optional[float]:
        """
        Get price for a ticker at a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date to get price for
            price_type: 'open', 'high', 'low', 'close', 'adj_close'
        
        Returns:
            Price at date, or None if not available
        """
        if ticker not in self._price_cache:
            return None
        
        df = self._price_cache[ticker]
        
        # Find closest date (handles weekends/holidays)
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in df.index:
            return df.loc[date_str, price_type.capitalize()]
        
        # Find next available date (forward fill)
        available_dates = df.index[df.index >= date_str]
        if len(available_dates) > 0:
            return df.loc[available_dates[0], price_type.capitalize()]
        
        return None
    
    def get_price_range(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get price data for a ticker within a date range.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data, or None if not available
        """
        if ticker not in self._price_cache:
            return None
        
        df = self._price_cache[ticker]
        
        # Filter by date range
        mask = (df.index >= start_date.strftime('%Y-%m-%d')) & \
               (df.index <= end_date.strftime('%Y-%m-%d'))
        
        return df[mask]


# ============================================================================
# Phase 6.2: Signal Replay System
# ============================================================================

class SignalReplayEngine:
    """
    Replay historical signals and simulate trading.
    
    Simulates realistic trading with:
    - Entry logic based on signal score threshold
    - Exit logic (stop loss, take profit, time decay)
    - Position sizing strategies
    - Transaction costs simulation
    - No lookahead bias (only uses data available at signal time)
    """
    
    def __init__(
        self,
        data_fetcher: HistoricalDataFetcher,
        entry_threshold: float = 0.3,
        stop_loss_pct: float = 0.15,
        take_profit_pct: float = 0.30,
        max_hold_days: int = 90,
        position_size_pct: float = 0.10,
        transaction_cost_pct: float = 0.001,  # 0.1% per trade
    ):
        """
        Initialize signal replay engine.
        
        Args:
            data_fetcher: HistoricalDataFetcher instance for price lookups
            entry_threshold: Minimum signal score to enter position (default: 0.3)
            stop_loss_pct: Stop loss percentage (default: 15%)
            take_profit_pct: Take profit percentage (default: 30%)
            max_hold_days: Maximum days to hold position (default: 90)
            position_size_pct: Position size as % of portfolio (default: 10%)
            transaction_cost_pct: Transaction cost per trade (default: 0.1%)
        """
        self.data_fetcher = data_fetcher
        self.entry_threshold = entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days
        self.position_size_pct = position_size_pct
        self.transaction_cost_pct = transaction_cost_pct
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"[REPLAY] SignalReplayEngine initialized "
            f"(entry>{entry_threshold:.2f}, SL:{stop_loss_pct:.0%}, "
            f"TP:{take_profit_pct:.0%}, max:{max_hold_days}d)"
        )
    
    async def replay_signal(
        self,
        ticker: str,
        signal_score: float,
        signal_date: datetime,
        historical_data: pd.DataFrame,
        portfolio_value: float = 100000.0
    ) -> Optional[BacktestPosition]:
        """
        Replay a single signal and simulate trading.
        
        Args:
            ticker: Ticker symbol
            signal_score: Signal score from Phase 4 (0-1)
            signal_date: Date signal was generated
            historical_data: Historical price DataFrame for this ticker
            portfolio_value: Current portfolio value for position sizing
        
        Returns:
            BacktestPosition with entry/exit details, or None if no entry
        """
        # Check if signal meets entry threshold
        if signal_score < self.entry_threshold:
            self.logger.debug(
                f"[SKIP] {ticker} score {signal_score:.3f} < threshold {self.entry_threshold:.3f}"
            )
            return None
        
        # Get entry price (next day's open to avoid lookahead bias)
        entry_date = signal_date + timedelta(days=1)
        entry_price = self.data_fetcher.get_price_at_date(ticker, entry_date, 'open')
        
        if entry_price is None:
            self.logger.warning(f"[SKIP] {ticker}: No entry price available for {entry_date.date()}")
            return None
        
        # Calculate position size
        position_size = portfolio_value * self.position_size_pct
        shares = position_size / entry_price
        
        # Apply transaction costs
        entry_cost = position_size * self.transaction_cost_pct
        
        # Initialize position
        position = BacktestPosition(
            ticker=ticker,
            signal_score=signal_score,
            signal_date=signal_date,
            entry_date=entry_date,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_hold_days=self.max_hold_days,
            status=PositionStatus.OPEN
        )
        
        self.logger.debug(
            f"[ENTRY] {ticker} @ ${entry_price:.2f} on {entry_date.date()} "
            f"(score: {signal_score:.3f}, size: ${position_size:,.0f})"
        )
        
        # Monitor position day by day
        current_date = entry_date
        max_exit_date = entry_date + timedelta(days=self.max_hold_days)
        
        # Get price range for monitoring
        price_data = self.data_fetcher.get_price_range(
            ticker, entry_date, max_exit_date
        )
        
        if price_data is None or len(price_data) == 0:
            self.logger.warning(f"[SKIP] {ticker}: No price data for monitoring period")
            return None
        
        # Iterate through each trading day
        for date_str, row in price_data.iterrows():
            current_date = pd.to_datetime(date_str)
            
            # Skip entry date
            if current_date <= entry_date:
                continue
            
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Calculate returns
            return_from_high = (high_price - entry_price) / entry_price
            return_from_low = (low_price - entry_price) / entry_price
            return_from_close = (close_price - entry_price) / entry_price
            
            # Check stop loss (hit during day)
            if return_from_low <= -self.stop_loss_pct:
                exit_price = entry_price * (1 - self.stop_loss_pct)
                position = self._close_position(
                    position, current_date, exit_price, 
                    PositionStatus.STOPPED, "stop_loss"
                )
                self.logger.debug(
                    f"[STOP] {ticker} @ ${exit_price:.2f} on {current_date.date()} "
                    f"(return: {return_from_low:.1%})"
                )
                break
            
            # Check take profit (hit during day)
            if return_from_high >= self.take_profit_pct:
                exit_price = entry_price * (1 + self.take_profit_pct)
                position = self._close_position(
                    position, current_date, exit_price,
                    PositionStatus.PROFIT_TAKEN, "take_profit"
                )
                self.logger.debug(
                    f"[PROFIT] {ticker} @ ${exit_price:.2f} on {current_date.date()} "
                    f"(return: {return_from_high:.1%})"
                )
                break
            
            # Check max hold time
            hold_days = (current_date - entry_date).days
            if hold_days >= self.max_hold_days:
                exit_price = close_price
                position = self._close_position(
                    position, current_date, exit_price,
                    PositionStatus.CLOSED, "time_exit"
                )
                self.logger.debug(
                    f"[TIME] {ticker} @ ${exit_price:.2f} on {current_date.date()} "
                    f"(return: {return_from_close:.1%}, {hold_days}d)"
                )
                break
        
        # If still open, close at last available price
        if position.status == PositionStatus.OPEN:
            last_date = pd.to_datetime(price_data.index[-1])
            last_price = price_data.iloc[-1]['Close']
            position = self._close_position(
                position, last_date, last_price,
                PositionStatus.CLOSED, "data_end"
            )
            self.logger.debug(
                f"[DATA_END] {ticker} @ ${last_price:.2f} on {last_date.date()}"
            )
        
        # Apply exit transaction costs
        if position.return_dollars:
            exit_cost = position.position_size * self.transaction_cost_pct
            position.return_dollars -= (entry_cost + exit_cost)
            position.return_pct = position.return_dollars / position.position_size
        
        return position
    
    def _close_position(
        self,
        position: BacktestPosition,
        exit_date: datetime,
        exit_price: float,
        status: PositionStatus,
        exit_reason: str
    ) -> BacktestPosition:
        """
        Close a position and calculate final returns.
        
        Args:
            position: Position to close
            exit_date: Exit date
            exit_price: Exit price
            status: Position status (STOPPED, PROFIT_TAKEN, CLOSED)
            exit_reason: Reason for exit
        
        Returns:
            Updated position with exit details
        """
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.status = status
        position.exit_reason = exit_reason
        
        # Calculate returns
        position.return_pct = (exit_price - position.entry_price) / position.entry_price
        position.return_dollars = position.position_size * position.return_pct
        
        # Calculate hold days
        position.hold_days = (exit_date - position.entry_date).days
        
        return position
    
    async def replay_signals_batch(
        self,
        signals: List[Dict[str, Any]],
        historical_data: Dict[str, pd.DataFrame],
        portfolio_value: float = 100000.0
    ) -> List[BacktestPosition]:
        """
        Replay multiple signals in batch.
        
        Args:
            signals: List of signal dictionaries with 'ticker', 'overall_score', 'signal_date'
            historical_data: Historical price data for all tickers
            portfolio_value: Portfolio value for position sizing
        
        Returns:
            List of BacktestPositions (only positions that were entered)
        """
        self.logger.info(f"[REPLAY] Processing {len(signals)} signals...")
        
        positions = []
        
        for signal in signals:
            ticker = signal.get('ticker')
            score = signal.get('overall_score', 0)
            signal_date = signal.get('signal_date', datetime.now())
            
            # Ensure signal_date is datetime
            if isinstance(signal_date, str):
                signal_date = pd.to_datetime(signal_date)
            
            # Get historical data for ticker
            ticker_data = historical_data.get(ticker)
            if ticker_data is None:
                self.logger.warning(f"[SKIP] {ticker}: No historical data available")
                continue
            
            # Replay signal
            position = await self.replay_signal(
                ticker=ticker,
                signal_score=score,
                signal_date=signal_date,
                historical_data=ticker_data,
                portfolio_value=portfolio_value
            )
            
            if position:
                positions.append(position)
        
        self.logger.info(
            f"[SUCCESS] Replayed {len(positions)} positions (entry rate: "
            f"{len(positions)/len(signals)*100:.1f}%)"
        )
        
        return positions


# ============================================================================
# Phase 6.3: Performance Calculator (Placeholder)
# ============================================================================

class PerformanceCalculator:
    """
    Calculate comprehensive performance metrics for backtested positions.
    
    Phase 6.3 Implementation:
    - Returns calculation (total, average, annualized)
    - Risk-adjusted metrics (Sharpe, Sortino)
    - Drawdown analysis
    - Win rate and hold time statistics
    - Historical success rate prediction
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, positions: List['BacktestPosition']) -> Dict[str, float]:
        """
        Calculate return metrics from positions.
        
        Returns:
            Dict with total_return_pct, avg_return_pct, total_return_dollars
        """
        if not positions:
            return {'total_return_pct': 0.0, 'avg_return_pct': 0.0, 'total_return_dollars': 0.0}
        
        returns_pct = [p.return_pct for p in positions if p.return_pct is not None]
        returns_dollars = [p.return_dollars for p in positions if p.return_dollars is not None]
        
        return {
            'total_return_pct': sum(returns_pct) * 100 if returns_pct else 0.0,
            'avg_return_pct': (sum(returns_pct) / len(returns_pct) * 100) if returns_pct else 0.0,
            'total_return_dollars': sum(returns_dollars) if returns_dollars else 0.0
        }
    
    def calculate_sharpe_ratio(self, returns: List[float], periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Args:
            returns: List of returns (as decimals, not percentages)
            periods_per_year: Trading periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)
        
        sharpe = (annualized_return - self.risk_free_rate) / annualized_std
        return float(sharpe)
    
    def calculate_sortino_ratio(self, returns: List[float], periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Only considers negative returns in risk calculation.
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Only consider negative returns for downside deviation
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0
        
        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_downside_std = downside_std * np.sqrt(periods_per_year)
        
        sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std
        return float(sortino)
    
    def calculate_max_drawdown(self, positions: List['BacktestPosition']) -> Dict[str, float]:
        """
        Calculate maximum drawdown from position returns.
        
        Returns:
            Dict with max_drawdown_pct, max_drawdown_duration_days
        """
        if not positions:
            return {'max_drawdown_pct': 0.0, 'max_drawdown_duration_days': 0}
        
        # Sort positions by exit date
        sorted_positions = sorted([p for p in positions if p.exit_date], key=lambda x: x.exit_date)
        
        # Build cumulative equity curve
        equity = 1.0  # Start at 100%
        equity_curve = [equity]
        peak = equity
        max_dd = 0.0
        dd_start = None
        max_dd_duration = 0
        
        for pos in sorted_positions:
            if pos.return_pct is not None:
                equity *= (1 + pos.return_pct)
                equity_curve.append(equity)
                
                # Track peak
                if equity > peak:
                    peak = equity
                    dd_start = None
                else:
                    # In drawdown
                    drawdown = (equity - peak) / peak
                    if drawdown < max_dd:
                        max_dd = drawdown
                    
                    # Track duration
                    if dd_start is None:
                        dd_start = pos.exit_date
                    elif pos.exit_date and dd_start:
                        duration = (pos.exit_date - dd_start).days
                        max_dd_duration = max(max_dd_duration, duration)
        
        return {
            'max_drawdown_pct': abs(max_dd) * 100,
            'max_drawdown_duration_days': max_dd_duration
        }
    
    def calculate_win_rate(self, positions: List['BacktestPosition']) -> Dict[str, Any]:
        """
        Calculate win rate statistics.
        
        Returns:
            Dict with win_count, loss_count, win_rate_pct, profit_factor
        """
        if not positions:
            return {'win_count': 0, 'loss_count': 0, 'win_rate_pct': 0.0, 'profit_factor': 0.0}
        
        winners = [p for p in positions if p.return_pct and p.return_pct > 0]
        losers = [p for p in positions if p.return_pct and p.return_pct <= 0]
        
        win_rate = (len(winners) / len(positions) * 100) if positions else 0.0
        
        # Profit factor = total wins / abs(total losses)
        total_wins = sum(p.return_pct for p in winners if p.return_pct) if winners else 0
        total_losses = abs(sum(p.return_pct for p in losers if p.return_pct)) if losers else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        return {
            'win_count': len(winners),
            'loss_count': len(losers),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99
        }
    
    def calculate_avg_hold_time(self, positions: List['BacktestPosition']) -> float:
        """Calculate average holding period in days."""
        hold_times = [p.hold_days for p in positions if p.hold_days is not None]
        return sum(hold_times) / len(hold_times) if hold_times else 0.0
    
    async def calculate_historical_success_rate(
        self, 
        signal_score: float, 
        score_range: float = 0.1,
        db = None
    ) -> Optional[float]:
        """
        Calculate historical success rate for signals with similar scores.
        
        Migrated from integrations/backtest.py.
        Success = positive return AND beat SPY (if SPY data available).
        
        Args:
            signal_score: The signal score of the current signal (0-1 scale)
            score_range: Range around score to consider (default ±0.1)
            db: Database interface (optional, will initialize if None)
            
        Returns:
            Success rate as percentage (0-100), or None if insufficient data
        """
        try:
            # Import here to avoid circular dependency
            if db is None:
                from ..storage.database import get_supabase_database
                db = await get_supabase_database()
            
            min_score = signal_score - score_range
            max_score = signal_score + score_range
            
            # Query signals with similar scores that have backtest results
            result = db.client.table('signals').select(
                'signal_score, backtest_return_pct, backtest_win'
            ).gte('signal_score', min_score).lte('signal_score', max_score).eq(
                'backtest_status', 'completed'
            ).not_.is_('backtest_return_pct', 'null').execute()
            
            if not result.data or len(result.data) < 5:
                # Need at least 5 data points for meaningful statistics
                self.logger.debug(
                    f"Insufficient historical data for score {signal_score:.3f} "
                    f"(found {len(result.data) if result.data else 0} signals)"
                )
                return None
            
            signals = result.data
            
            # Calculate success rate
            successful_signals = sum(1 for s in signals if s.get('backtest_win'))
            total_signals = len(signals)
            
            success_rate = (successful_signals / total_signals) * 100
            
            self.logger.debug(
                f"Historical success rate for score {signal_score:.3f}: "
                f"{success_rate:.1f}% ({successful_signals}/{total_signals} signals)"
            )
            
            return round(success_rate, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical success rate: {e}")
            return None
    
    def generate_performance_summary(self, positions: List['BacktestPosition']) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Combines all metrics into a single report.
        """
        returns_metrics = self.calculate_returns(positions)
        win_stats = self.calculate_win_rate(positions)
        drawdown_stats = self.calculate_max_drawdown(positions)
        
        # Get returns for Sharpe calculation
        returns_list = [p.return_pct for p in positions if p.return_pct is not None]
        sharpe = self.calculate_sharpe_ratio(returns_list) if len(returns_list) >= 2 else 0.0
        sortino = self.calculate_sortino_ratio(returns_list) if len(returns_list) >= 2 else 0.0
        
        return {
            'total_positions': len(positions),
            'avg_hold_days': self.calculate_avg_hold_time(positions),
            **returns_metrics,
            **win_stats,
            **drawdown_stats,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino
        }


# ============================================================================
# Backtest Scheduling (Phase 6.4)
# ============================================================================

class BacktestScheduler:
    """
    Smart backtest scheduler for interval-based performance tracking.
    
    Migrated from integrations/backtest.py.
    Determines when to run backtests based on signal age and creates
    performance snapshots at key intervals (1d, 3d, 7d, 14d, 30d, etc.).
    
    Usage in Phase 6.4:
    - Batch process signals by age eligibility
    - Avoid re-backtesting already processed intervals
    - Optimize API calls by grouping similar timeframes
    """
    
    def __init__(self):
        """Initialize scheduler with standard intervals."""
        self.intervals = {
            BacktestInterval.ONE_DAY: 1,
            BacktestInterval.THREE_DAY: 3,
            BacktestInterval.SEVEN_DAY: 7,
            BacktestInterval.FOURTEEN_DAY: 14,
            BacktestInterval.THIRTY_DAY: 30,
            BacktestInterval.SIXTY_DAY: 60,
            BacktestInterval.NINETY_DAY: 90
        }
        self.logger = logging.getLogger(__name__)
    
    def calculate_signal_age(self, created_at: datetime) -> int:
        """
        Calculate age of signal in days.
        
        Args:
            created_at: Signal creation datetime
            
        Returns:
            Age in days (non-negative)
        """
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
        Get list of backtest intervals that this signal is eligible for.
        
        Args:
            signal_age_days: Age of signal in days
            
        Returns:
            List of eligible intervals (e.g., [1d, 3d, 7d] for 10-day old signal)
        """
        eligible = []
        
        for interval, days in self.intervals.items():
            if signal_age_days >= days:
                eligible.append(interval)
        
        return eligible
    
    def get_missing_backtest_intervals(
        self, 
        signal: Dict[str, Any], 
        completed_intervals: Optional[List[str]] = None
    ) -> List[BacktestInterval]:
        """
        Determine which backtest intervals are missing for a signal.
        
        Args:
            signal: Signal dict with created_at field
            completed_intervals: List of already completed interval names (e.g., ['1d', '3d'])
            
        Returns:
            List of missing intervals that should be backtested
        """
        signal_age = self.calculate_signal_age(signal.get('created_at'))
        eligible_intervals = self.get_eligible_intervals(signal_age)
        
        if not completed_intervals:
            # No intervals completed yet, return all eligible
            return eligible_intervals
        
        # Find missing intervals
        completed_set = set(completed_intervals)
        missing = [interval for interval in eligible_intervals if interval.value not in completed_set]
        
        return missing
    
    async def get_signals_requiring_backtest(
        self, 
        db, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get signals that require backtesting based on age and missing intervals.
        
        Args:
            db: Database interface
            limit: Maximum signals to return (None for all)
            
        Returns:
            List of signals with missing_intervals and signal_age_days fields
        """
        try:
            # Get signals that are at least 1 day old with pending backtest status
            cutoff_date = datetime.now() - timedelta(days=1)
            
            query = db.client.table('signals').select(
                'id, ticker, created_at, backtest_status, signal_score'
            ).lte('created_at', cutoff_date.isoformat()).or_(
                'backtest_status.eq.pending,backtest_status.is.null'
            ).order('created_at', desc=True)
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            signals = result.data if result.data else []
            
            # Filter signals that need backtesting
            signals_needing_backtest = []
            
            for signal in signals:
                missing_intervals = self.get_missing_backtest_intervals(signal, completed_intervals=None)
                if missing_intervals:
                    signal['missing_intervals'] = missing_intervals
                    signal['signal_age_days'] = self.calculate_signal_age(signal['created_at'])
                    signals_needing_backtest.append(signal)
            
            self.logger.info(
                f"Found {len(signals_needing_backtest)} signals requiring backtest "
                f"out of {len(signals)} total"
            )
            
            return signals_needing_backtest
            
        except Exception as e:
            self.logger.error(f"Error getting signals requiring backtest: {e}")
            return []
    
    def create_backtest_plan(
        self, 
        signals: List[Dict[str, Any]]
    ) -> Dict[BacktestInterval, List[Dict[str, Any]]]:
        """
        Create backtest execution plan grouped by interval.
        
        This allows efficient batch processing of similar timeframes.
        
        Args:
            signals: List of signals with missing_intervals field
            
        Returns:
            Dict mapping intervals to lists of signals needing that interval
        """
        plan = {interval: [] for interval in BacktestInterval}
        
        for signal in signals:
            missing_intervals = signal.get('missing_intervals', [])
            for interval in missing_intervals:
                plan[interval].append(signal)
        
        # Log the plan
        for interval, signal_list in plan.items():
            if signal_list:
                self.logger.info(f"Backtest plan - {interval.value}: {len(signal_list)} signals")
        
        return plan


# ============================================================================
# Performance Tracker (Simplified - No Trade Simulation)
# ============================================================================

class PerformanceTracker:
    """
    Track signal performance over time intervals without trade simulation.
    
    Simpler approach than backtesting - just tracks price movement from baseline.
    
    Features:
    - Baseline: Next day open price (avoids lookahead bias)
    - Intervals: 1d, 3d, 7d, 10d, 14d, 30d, 90d
    - SPY comparison for market benchmarking
    - Auto-scheduling based on signal age
    """
    
    def __init__(self, db=None):
        """Initialize performance tracker."""
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.intervals = [1, 3, 7, 10, 14, 30, 90]
        self.data_fetcher = HistoricalDataFetcher(lookback_years=1)
    
    async def set_database(self):
        """Initialize database connection if not provided."""
        if self.db is None:
            from ..storage.database import get_supabase_database
            self.db = await get_supabase_database()
    
    def _calculate_signal_age(self, created_at: datetime) -> int:
        """Calculate signal age in days."""
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        now = datetime.now()
        if created_at.tzinfo:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        
        age = (now - created_at).days
        return max(0, age)
    
    def _get_eligible_intervals(self, signal_age_days: int) -> List[int]:
        """Get intervals that signal is old enough to calculate."""
        return [interval for interval in self.intervals if signal_age_days >= interval]
    
    async def calculate_interval_returns(
        self,
        ticker: str,
        signal_date: datetime,
        baseline_price: float,
        baseline_date: datetime,
        intervals: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate returns for specified intervals from baseline.
        
        Args:
            ticker: Stock ticker
            signal_date: When signal was created
            baseline_price: Next day open price (baseline)
            baseline_date: Date of baseline
            intervals: List of days to calculate (e.g., [1, 3, 7])
            
        Returns:
            Dict with return_1d, return_3d, etc. and SPY comparisons
        """
        try:
            # Fetch historical data
            end_date = datetime.now() + timedelta(days=1)
            start_date = baseline_date - timedelta(days=5)  # Buffer for weekends
            
            ticker_data = await self.data_fetcher.fetch_historical_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date
            )
            
            if ticker not in ticker_data or ticker_data[ticker].empty:
                self.logger.warning(f"No price data for {ticker}")
                return {}
            
            # Get SPY data for comparison
            spy_data = await self.data_fetcher.fetch_historical_data(
                tickers=['SPY'],
                start_date=start_date,
                end_date=end_date
            )
            
            df = ticker_data[ticker]
            spy_df = spy_data.get('SPY') if spy_data else None
            
            # Calculate returns for each interval
            results = {}
            
            for days in intervals:
                target_date = baseline_date + timedelta(days=days)
                
                # Get ticker price at target date (with forward fill for weekends)
                target_price = self._get_price_at_date(df, target_date, 'close')
                
                if target_price and baseline_price > 0:
                    return_pct = ((target_price - baseline_price) / baseline_price) * 100
                    results[f'return_{days}d'] = round(return_pct, 4)
                    
                    # Calculate SPY return for comparison
                    # Use 'open' for baseline to match ticker baseline (next day open)
                    if spy_df is not None and not spy_df.empty:
                        spy_baseline = self._get_price_at_date(spy_df, baseline_date, 'open')
                        spy_target = self._get_price_at_date(spy_df, target_date, 'close')
                        
                        if spy_baseline and spy_target and spy_baseline > 0:
                            spy_return_pct = ((spy_target - spy_baseline) / spy_baseline) * 100
                            results[f'spy_return_{days}d'] = round(spy_return_pct, 4)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating interval returns for {ticker}: {e}")
            return {}
    
    def _get_price_at_date(self, df: pd.DataFrame, target_date: datetime, price_type: str = 'close') -> Optional[float]:
        """Get price at specific date with forward fill for weekends."""
        try:
            # Normalize target_date to start of day
            target_ts = pd.Timestamp(target_date.date())
            price_col = price_type.capitalize()
            
            # Handle multi-level columns (ticker, price_type)
            if isinstance(df.columns, pd.MultiIndex):
                # Get first ticker from columns
                ticker = df.columns.get_level_values(0)[0]
                price_col = (ticker, price_col)
            
            # Try exact date first
            if target_ts in df.index:
                price = df.loc[target_ts, price_col]
                return float(price) if price is not None else None
            
            # Forward fill - find next available date
            available_dates = [d for d in df.index if d >= target_ts]
            if available_dates:
                price = df.loc[available_dates[0], price_col]
                return float(price) if price is not None else None
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error getting price at {target_date}: {e}")
            return None
    
    async def set_baseline_for_signal(self, signal_id: str, ticker: str, signal_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Set baseline price (next day open) for a signal.
        
        Args:
            signal_id: Signal ID
            ticker: Stock ticker
            signal_date: When signal was created
            
        Returns:
            Dict with baseline_price and baseline_date, or None if failed
        """
        try:
            # Baseline is next day's open
            baseline_date = signal_date + timedelta(days=1)
            
            # Fetch data around baseline date
            start_date = signal_date
            end_date = baseline_date + timedelta(days=5)  # Buffer
            
            ticker_data = await self.data_fetcher.fetch_historical_data(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date
            )
            
            if ticker not in ticker_data or ticker_data[ticker].empty:
                self.logger.warning(f"Cannot set baseline for {ticker}: No data")
                return None
            
            df = ticker_data[ticker]
            
            # Get next day's open price (with forward fill for weekends)
            baseline_price = self._get_price_at_date(df, baseline_date, 'open')
            
            if not baseline_price:
                self.logger.warning(f"Cannot set baseline for {ticker}: No price at {baseline_date.date()}")
                return None
            
            # Update database
            update_data = {
                'backtest_baseline_price': round(baseline_price, 2),
                'backtest_baseline_date': baseline_date.isoformat(),
                'backtest_status': 'baseline_set',
                'backtest_last_update': datetime.now().isoformat()
            }
            
            result = self.db.client.table('signals').update(update_data).eq('id', signal_id).execute()
            
            if result.data:
                self.logger.debug(f"✅ Set baseline for {ticker} (ID: {signal_id[:8]}...): ${baseline_price:.2f} @ {baseline_date.date()}")
                return {
                    'baseline_price': baseline_price,
                    'baseline_date': baseline_date
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error setting baseline for signal {signal_id}: {e}")
            return None
    
    async def update_signal_returns(self, signal: Dict[str, Any]) -> bool:
        """
        Update returns for a signal based on its age.
        
        Args:
            signal: Signal dict with id, ticker, created_at, baseline_price, etc.
            
        Returns:
            True if updated, False otherwise
        """
        try:
            signal_id = signal['id']
            ticker = signal['ticker']
            created_at = signal['created_at']
            
            # Check if baseline is set
            baseline_price = signal.get('backtest_baseline_price')
            baseline_date = signal.get('backtest_baseline_date')
            
            if not baseline_price or not baseline_date:
                # Set baseline first
                self.logger.debug(f"Setting baseline for {ticker} (ID: {signal_id[:8]}...)")
                baseline_result = await self.set_baseline_for_signal(
                    signal_id=signal_id,
                    ticker=ticker,
                    signal_date=datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                )
                
                if not baseline_result:
                    return False
                
                baseline_price = baseline_result['baseline_price']
                baseline_date = baseline_result['baseline_date']
            else:
                # Parse baseline date
                baseline_date = datetime.fromisoformat(baseline_date.replace('Z', '+00:00'))
            
            # Calculate signal age from CREATION date (not baseline)
            # This is because baseline is next-day open, so we need one extra day
            signal_age = self._calculate_signal_age(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
            
            # Get eligible intervals based on signal age
            eligible_intervals = self._get_eligible_intervals(signal_age)
            
            if not eligible_intervals:
                self.logger.debug(f"Signal {ticker} not old enough yet ({signal_age} days)")
                return False
            
            # Find missing intervals (NULL values that are now eligible)
            missing_intervals = []
            for interval in eligible_intervals:
                if signal.get(f'return_{interval}d') is None:
                    missing_intervals.append(interval)
            
            if not missing_intervals:
                self.logger.debug(f"Signal {ticker} already up-to-date")
                return False
            
            self.logger.debug(f"Updating {ticker} (ID: {signal_id[:8]}...): intervals {missing_intervals}")
            
            # Calculate returns for missing intervals
            returns = await self.calculate_interval_returns(
                ticker=ticker,
                signal_date=datetime.fromisoformat(created_at.replace('Z', '+00:00')),
                baseline_price=baseline_price,
                baseline_date=baseline_date,
                intervals=missing_intervals
            )
            
            if not returns:
                return False
            
            # Determine status based on whether all intervals are populated
            all_intervals_done = all(
                signal.get(f'return_{i}d') is not None or i in missing_intervals
                for i in self.intervals
            )
            
            # Update database
            update_data = {
                **returns,
                'backtest_status': 'completed' if all_intervals_done else 'in_progress',
                'backtest_last_update': datetime.now().isoformat()
            }
            
            result = self.db.client.table('signals').update(update_data).eq('id', signal_id).execute()
            
            if result.data:
                self.logger.debug(f"✅ Updated {ticker}: {len(missing_intervals)} intervals")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating signal returns: {e}")
            return False
    
    async def backfill_signals(self, start_date: datetime, limit: int = 100) -> Dict[str, int]:
        """
        Backfill performance data for signals created after start_date.
        
        Args:
            start_date: Only process signals created on or after this date
            limit: Maximum signals to process
            
        Returns:
            Dict with stats (processed, updated, failed)
        """
        try:
            await self.set_database()
            
            # Get signals needing backfill
            result = self.db.client.table('signals').select('*').gte(
                'created_at', start_date.isoformat()
            ).or_(
                'backtest_status.eq.pending,backtest_status.is.null,backtest_status.eq.baseline_set,backtest_status.eq.in_progress'
            ).order('created_at', desc=False).limit(limit).execute()
            
            signals = result.data if result.data else []
            
            self.logger.debug(f"Backfilling {len(signals)} signals from {start_date.date()}...")
            
            stats = {'processed': 0, 'updated': 0, 'failed': 0}
            
            for signal in signals:
                try:
                    updated = await self.update_signal_returns(signal)
                    stats['processed'] += 1
                    
                    if updated:
                        stats['updated'] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal {signal.get('id', 'unknown')}: {e}")
                    stats['failed'] += 1
                    stats['processed'] += 1
            
            self.logger.debug(f"Backfill complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in backfill_signals: {e}")
            return {'processed': 0, 'updated': 0, 'failed': 0}
    
    async def update_pending_signals(self, limit: int = 50) -> Dict[str, int]:
        """
        Update all signals with pending/in-progress status.
        
        Called during pipeline execution to keep performance data current.
        
        Returns:
            Dict with stats (processed, updated, failed)
        """
        try:
            await self.set_database()
            
            # Get signals needing updates (not completed)
            result = self.db.client.table('signals').select('*').in_(
                'backtest_status', ['pending', 'baseline_set', 'in_progress']
            ).order('created_at', desc=False).limit(limit).execute()
            
            signals = result.data if result.data else []
            
            if not signals:
                self.logger.info("No signals need performance updates")
                return {'processed': 0, 'updated': 0, 'failed': 0}
            
            self.logger.info(f"Updating {len(signals)} signals with pending performance data...")
            
            stats = {'processed': 0, 'updated': 0, 'failed': 0}
            
            for signal in signals:
                try:
                    updated = await self.update_signal_returns(signal)
                    stats['processed'] += 1
                    
                    if updated:
                        stats['updated'] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal {signal.get('id', 'unknown')}: {e}")
                    stats['failed'] += 1
                    stats['processed'] += 1
            
            self.logger.info(f"Performance update complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in update_pending_signals: {e}")
            return {'processed': 0, 'updated': 0, 'failed': 0}


# ============================================================================
# Phase 6: Main Backtesting Class
# ============================================================================

class Phase6Backtester:
    """
    Main backtesting engine coordinating all sub-phases.
    
    Current implementation: Phase 6.1 (Data Collection)
    TODO: Add Phase 6.2-6.5 implementations
    """
    
    def __init__(
        self,
        lookback_years: int = 2,
        starting_capital: float = 10000.0,
        entry_threshold: float = 0.3,
        stop_loss_pct: float = 0.15,
        take_profit_pct: float = 0.30
    ):
        """
        Initialize backtester.
        
        Args:
            lookback_years: Years of historical data (default: 2)
            starting_capital: Starting portfolio value (default: $10k)
            entry_threshold: Minimum signal score to enter (default: 0.3)
            stop_loss_pct: Stop loss percentage (default: 15%)
            take_profit_pct: Take profit percentage (default: 30%)
        """
        self.lookback_years = lookback_years
        self.starting_capital = starting_capital
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-components
        self.data_fetcher = HistoricalDataFetcher(lookback_years=lookback_years)
        self.replay_engine = SignalReplayEngine(
            data_fetcher=self.data_fetcher,
            entry_threshold=entry_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        # TODO: Initialize PerformanceCalculator
        
        self.logger.info(
            f"[BACKTEST] Phase6Backtester initialized "
            f"(lookback: {lookback_years}y, capital: ${starting_capital:,.0f})"
        )
    
    async def prepare_historical_data(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Phase 6.1: Fetch and prepare historical data for backtesting.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (default: lookback_years ago)
            end_date: End date (default: today)
        
        Returns:
            Dictionary mapping ticker -> historical price DataFrame
        """
        self.logger.info(
            f"[PHASE 6.1] Preparing historical data for {len(tickers)} tickers"
        )
        
        historical_data = await self.data_fetcher.fetch_historical_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate data quality
        valid_tickers = []
        for ticker, df in historical_data.items():
            if df is not None and len(df) > 30:  # At least 30 days of data
                valid_tickers.append(ticker)
            else:
                self.logger.warning(
                    f"[SKIP] {ticker}: Insufficient data ({len(df) if df is not None else 0} days)"
                )
        
        self.logger.info(
            f"[SUCCESS] Phase 6.1 complete: {len(valid_tickers)}/{len(tickers)} "
            f"tickers with sufficient historical data"
        )
        
        return {t: historical_data[t] for t in valid_tickers}
    
    async def backtest_signals(
        self,
        signals: List[Dict[str, Any]],
        historical_data: Dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """
        Run full backtest on signals (Phases 6.2-6.4).
        
        TODO: Implement signal replay, performance calculation, strategy validation
        
        Args:
            signals: List of signal dictionaries from Phase 4
            historical_data: Historical price data from Phase 6.1
        
        Returns:
            Aggregate backtest results
        """
        self.logger.warning(
            "[TODO] Full backtesting (Phases 6.2-6.4) not yet implemented. "
            "Currently only Phase 6.1 (data collection) is complete."
        )
        
        # Placeholder result
        return BacktestResult(
            total_trades=0,
            backtest_start_date=datetime.now() - timedelta(days=365),
            backtest_end_date=datetime.now()
        )


# ============================================================================
# Quick Test Function
# ============================================================================

async def quick_test_phase6():
    """Quick test of Phase 6.1 data collection."""
    logger.info("=" * 80)
    logger.info("PHASE 6.1 QUICK TEST: Historical Data Collection")
    logger.info("=" * 80)
    
    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    # Initialize backtester
    backtester = Phase6Backtester(lookback_years=1)
    
    # Fetch historical data
    historical_data = await backtester.prepare_historical_data(test_tickers)
    
    # Display results
    logger.info("\nHistorical Data Summary:")
    logger.info("-" * 80)
    for ticker, df in historical_data.items():
        logger.info(
            f"{ticker:6} | {len(df):4} days | "
            f"{df.index[0]} to {df.index[-1]} | "
            f"Close: ${df['Close'].iloc[-1]:.2f}"
        )
    
    logger.info("=" * 80)
    logger.info(f"✅ Phase 6.1 test complete: {len(historical_data)} tickers ready")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s'
    )
    
    # Run quick test
    asyncio.run(quick_test_phase6())
