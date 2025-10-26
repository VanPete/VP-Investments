"""
Phase 6: Performance Tracking (Hybrid Approach)
================================================

Progressively calculates interval returns for signals using the performance table.

Architecture:
- Phase 5 creates performance records with baseline (signal creation price)
- Phase 6 updates intervals as signals age (1d, 3d, 7d, 10d, 14d, 30d, 90d)
- Tracks which intervals are completed to avoid redundant API calls
- Compares against SPY (market) and sector ETF (peer group) benchmarks

Key Features:
1. Baseline at signal creation (no lookahead bias)
2. Progressive interval filling (only fetch what's needed)
3. Dual benchmark comparison (SPY + sector ETF)
4. Auto-calculated alpha via GENERATED columns
5. JSONB tracking of completed intervals
6. Graceful error handling (doesn't fail pipeline)

Design Decisions:
- Baseline = current_price at signal creation (realistic, simple)
- Intervals tracked in JSONB array to avoid redundant calculations
- Status: pending → in_progress → completed
- Runs during pipeline execution (incremental updates)

Database Auto-Calculations:
- All alpha_Xd columns are GENERATED ALWAYS AS (return_Xd - spy_return_Xd) STORED
- All sector_alpha_Xd columns are GENERATED ALWAYS AS (return_Xd - sector_return_Xd) STORED
- Phase 6 only sets: return_Xd, spy_return_Xd, sector_return_Xd
- PostgreSQL automatically calculates all alpha metrics whenever returns change
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceUpdater:
    """
    Update performance intervals for signals based on age.
    
    This replaces the old Phase 6 backtest logic that wrote to signals table.
    Now writes to dedicated performance table for clean separation.
    """
    
    def __init__(self, db=None):
        """
        Initialize performance updater.
        
        Args:
            db: SupabaseInterface instance (optional)
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.intervals = [1, 3, 7, 10, 14, 30, 90]  # Days to track
        
    async def set_database(self):
        """Initialize database connection if not provided."""
        if self.db is None:
            from ..storage.database import get_supabase_database
            self.db = await get_supabase_database()
    
    async def update_pending_performance(self, limit: int = 200) -> Dict[str, int]:
        """
        Update performance intervals for signals with pending/in-progress status.
        
        Called during pipeline execution to keep performance data current.
        Only fetches price data for intervals that are now eligible but not yet calculated.
        
        Args:
            limit: Maximum number of performance records to update (default: 200)
                   Increased from 50 to handle larger batches of signals efficiently.
            
        Returns:
            Dict with stats (processed, updated, failed)
        """
        try:
            await self.set_database()
            
            # Get performance records needing updates
            result = self.db.client.table('performance').select(
                'id, signal_id, baseline_price, baseline_date, intervals_completed, sector, sector_etf, signals!inner(ticker, created_at)'
            ).in_(
                'status', ['pending', 'in_progress']
            ).order('created_at', desc=False).limit(limit).execute()
            
            performance_records = result.data if result.data else []
            
            if not performance_records:
                self.logger.info("✅ No performance records need updates")
                return {'processed': 0, 'updated': 0, 'failed': 0}
            
            self.logger.info(f"⏳ Updating {len(performance_records)} performance records...")
            
            stats = {'processed': 0, 'updated': 0, 'failed': 0}
            
            for perf in performance_records:
                stats['processed'] += 1
                
                try:
                    # Parse dates - handle both string and datetime objects
                    baseline_date_raw = perf['baseline_date']
                    if isinstance(baseline_date_raw, str):
                        # Clean up timezone format and handle microseconds
                        baseline_date_str = baseline_date_raw.replace('Z', '+00:00')
                        # Truncate microseconds to 6 digits if needed
                        if '+' in baseline_date_str:
                            dt_part, tz_part = baseline_date_str.rsplit('+', 1)
                            if '.' in dt_part:
                                dt_main, microsec = dt_part.rsplit('.', 1)
                                microsec = microsec[:6]  # Keep only 6 digits
                                baseline_date_str = f"{dt_main}.{microsec}+{tz_part}"
                        baseline_date = datetime.fromisoformat(baseline_date_str)
                    else:
                        # Already a datetime object
                        baseline_date = baseline_date_raw
                    
                    signal_age_days = (datetime.now(timezone.utc) - baseline_date).days
                    
                    # Get eligible intervals (signal is old enough)
                    eligible_intervals = [i for i in self.intervals if i <= signal_age_days]
                    
                    # Get already completed intervals
                    completed = perf.get('intervals_completed') or []
                    
                    # Find missing intervals (eligible but not completed)
                    missing_intervals = [i for i in eligible_intervals if i not in completed]
                    
                    if not missing_intervals:
                        # All eligible intervals are done
                        if len(completed) == len(self.intervals):
                            # Mark as completed if ALL intervals are done
                            await self._update_performance_status(perf['id'], 'completed')
                            stats['updated'] += 1
                        continue
                    
                    # Calculate missing intervals
                    ticker = perf['signals']['ticker']
                    baseline_price = float(perf['baseline_price'])
                    sector_etf = perf.get('sector_etf')  # v3.2: Get sector ETF if available
                    
                    self.logger.debug(
                        f"  {ticker}: Age {signal_age_days}d, "
                        f"calculating intervals {missing_intervals}"
                    )
                    
                    # Fetch price data and calculate returns
                    update_data = await self._calculate_interval_returns(
                        ticker=ticker,
                        baseline_price=baseline_price,
                        baseline_date=baseline_date,
                        intervals=missing_intervals,
                        sector_etf=sector_etf  # v3.2: Pass sector ETF
                    )
                    
                    if update_data:
                        # Update intervals_completed
                        new_completed = sorted(list(set(completed + missing_intervals)))
                        update_data['intervals_completed'] = new_completed
                        
                        # Update status
                        if len(new_completed) == len(self.intervals):
                            update_data['status'] = 'completed'
                        else:
                            update_data['status'] = 'in_progress'
                        
                        # Persist to database
                        await self._update_performance_record(perf['id'], update_data)
                        stats['updated'] += 1
                        
                        self.logger.debug(
                            f"  ✅ {ticker}: Updated {len(missing_intervals)} intervals "
                            f"({len(new_completed)}/{len(self.intervals)} total)"
                        )
                    
                except Exception as e:
                    self.logger.error(f"  ❌ Failed to update performance record {perf['id']}: {e}")
                    stats['failed'] += 1
                    
                    # Log error to performance table
                    await self._update_performance_error(perf['id'], str(e))
            
            self.logger.info(
                f"✅ Performance update complete: "
                f"{stats['updated']}/{stats['processed']} updated, "
                f"{stats['failed']} failed"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error updating performance intervals: {e}")
            return {'processed': 0, 'updated': 0, 'failed': 0}
    
    async def _calculate_interval_returns(
        self,
        ticker: str,
        baseline_price: float,
        baseline_date: datetime,
        intervals: List[int],
        sector_etf: Optional[str] = None  # v3.2: Sector ETF for comparison
    ) -> Dict[str, Any]:
        """
        Calculate returns for specific intervals.
        
        Args:
            ticker: Stock ticker
            baseline_price: Baseline price (from signal creation)
            baseline_date: Baseline date
            intervals: List of intervals to calculate (e.g., [1, 3, 7])
            sector_etf: Sector ETF ticker (e.g., 'XLK') for sector comparison
            
        Returns:
            Dict with return_Xd, spy_return_Xd, alpha_Xd, sector_return_Xd, sector_alpha_Xd for each interval
        """
        try:
            import yfinance as yf
            
            # Fetch price data (baseline to now)
            end_date = datetime.now()
            start_date = baseline_date - timedelta(days=2)  # Small buffer
            
            # Download ticker data
            ticker_df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if ticker_df.empty:
                self.logger.warning(f"No price data for {ticker}")
                return {}
            
            # Download SPY data for comparison
            spy_df = yf.download(
                'SPY',
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            # Get SPY baseline price
            spy_baseline = self._get_price_at_date(spy_df, baseline_date, 'Close')
            
            # Download sector ETF data if available (v3.2)
            sector_df = None
            sector_baseline = None
            if sector_etf and sector_etf != 'SPY':
                try:
                    sector_df = yf.download(
                        sector_etf,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    if not sector_df.empty:
                        sector_baseline = self._get_price_at_date(sector_df, baseline_date, 'Close')
                except Exception as e:
                    self.logger.debug(f"Failed to fetch sector ETF {sector_etf}: {e}")
            
            # Calculate returns for each interval
            update_data = {}
            
            for interval_days in intervals:
                target_date = baseline_date + timedelta(days=interval_days)
                
                # Get ticker price at target date
                target_price = self._get_price_at_date(ticker_df, target_date, 'Close')
                
                if target_price and baseline_price > 0:
                    # Calculate ticker return
                    return_pct = ((target_price - baseline_price) / baseline_price) * 100
                    update_data[f'return_{interval_days}d'] = round(return_pct, 4)
                    
                    # Calculate SPY return for comparison
                    if spy_baseline and not spy_df.empty:
                        spy_target = self._get_price_at_date(spy_df, target_date, 'Close')
                        
                        if spy_target and spy_baseline > 0:
                            spy_return_pct = ((spy_target - spy_baseline) / spy_baseline) * 100
                            update_data[f'spy_return_{interval_days}d'] = round(spy_return_pct, 4)
                            
                            # NOTE: All alpha columns are GENERATED (auto-calculated by database)
                            # alpha_Xd = return_Xd - spy_return_Xd (for all 7 intervals)
                            # No need to set them here - database handles it automatically
                    
                    # Calculate sector ETF return (v3.2)
                    if sector_baseline and sector_df is not None and not sector_df.empty:
                        sector_target = self._get_price_at_date(sector_df, target_date, 'Close')
                        
                        if sector_target and sector_baseline > 0:
                            sector_return_pct = ((sector_target - sector_baseline) / sector_baseline) * 100
                            update_data[f'sector_return_{interval_days}d'] = round(sector_return_pct, 4)
                            
                            # NOTE: All sector_alpha columns are GENERATED (auto-calculated by database)
                            # sector_alpha_Xd = return_Xd - sector_return_Xd (for all 7 intervals)
                            # No need to set them here - database handles it automatically
            
            return update_data
            
        except Exception as e:
            self.logger.error(f"Error calculating returns for {ticker}: {e}")
            return {}
    
    def _get_price_at_date(
        self, 
        df: pd.DataFrame, 
        target_date: datetime, 
        price_col: str = 'Close'
    ) -> Optional[float]:
        """
        Get price at specific date with forward fill for weekends/holidays.
        
        Args:
            df: Price DataFrame
            target_date: Target date
            price_col: Column name ('Close', 'Open', etc.)
            
        Returns:
            Price at date, or None if not available
        """
        try:
            # Normalize to date
            target_ts = pd.Timestamp(target_date.date())
            
            # Try exact date first
            if target_ts in df.index:
                return float(df.loc[target_ts, price_col])
            
            # Forward fill - find next available date
            available_dates = [d for d in df.index if d >= target_ts]
            if available_dates:
                return float(df.loc[available_dates[0], price_col])
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error getting price at {target_date}: {e}")
            return None
    
    async def _update_performance_record(
        self, 
        performance_id: str, 
        update_data: Dict[str, Any]
    ) -> bool:
        """
        Update performance record in database.
        
        Args:
            performance_id: Performance record UUID
            update_data: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add last_updated timestamp
            update_data['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            result = self.db.client.table('performance').update(
                update_data
            ).eq('id', performance_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Error updating performance record {performance_id}: {e}")
            return False
    
    async def _update_performance_status(
        self, 
        performance_id: str, 
        status: str
    ) -> bool:
        """Update only the status of a performance record."""
        try:
            result = self.db.client.table('performance').update({
                'status': status,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }).eq('id', performance_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Error updating status for {performance_id}: {e}")
            return False
    
    async def _update_performance_error(
        self, 
        performance_id: str, 
        error_message: str
    ) -> bool:
        """Log error message to performance record."""
        try:
            result = self.db.client.table('performance').update({
                'error_message': error_message,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }).eq('id', performance_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"Error logging error for {performance_id}: {e}")
            return False


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_performance_updater(db=None):
    """
    Factory function to create PerformanceUpdater instance.
    
    Args:
        db: SupabaseInterface instance (optional)
        
    Returns:
        PerformanceUpdater instance
    """
    return PerformanceUpdater(db=db)
