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
    
    async def update_pending_performance(
        self,
        limit: int = 200,
        benchmark_cache: Optional[Dict[str, pd.DataFrame]] = None  # v3.3: Cached benchmark data from Phase 1
    ) -> Dict[str, int]:
        """
        Update performance intervals for signals with pending/in-progress status.
        
        Called during pipeline execution to keep performance data current.
        Only fetches price data for intervals that are now eligible but not yet calculated.
        
        v3.3: Now accepts optional benchmark_cache from Phase 1 to avoid redundant yfinance calls.
        
        Args:
            limit: Maximum number of performance records to update (default: 200)
                   Increased from 50 to handle larger batches of signals efficiently.
            benchmark_cache: Optional dict of ETF ticker -> DataFrame from Phase 1.
                             If provided, reuses SPY, QQQ, sector ETF data instead of fetching.
            
        Returns:
            Dict with stats (processed, updated, failed)
        """
        try:
            await self.set_database()
            
            self.logger.info("=" * 80)
            self.logger.info("PHASE 6: UPDATE PENDING PERFORMANCE - START")
            self.logger.info("=" * 80)
            
            # Log benchmark cache usage
            if benchmark_cache:
                cache_keys = list(benchmark_cache.keys()) if benchmark_cache else []
                self.logger.info(f"[CACHE] Using {len(benchmark_cache)} cached benchmarks from Phase 1: {cache_keys}")
            else:
                self.logger.info("[FETCH] No benchmark cache - will fetch benchmarks as needed")
            
            # Get performance records needing updates
            self.logger.info(f"[QUERY] Fetching performance records with status in ['pending', 'in_progress'], limit={limit}")
            result = self.db.client.table('performance').select(
                'id, signal_id, baseline_price, baseline_date, intervals_completed, sector, sector_etf, signals!inner(ticker, created_at)'
            ).in_(
                'status', ['pending', 'in_progress']
            ).order('created_at', desc=False).limit(limit).execute()
            
            performance_records = result.data if result.data else []
            
            self.logger.info(f"[QUERY RESULT] Found {len(performance_records)} performance records needing updates")
            
            if not performance_records:
                self.logger.info("[OK] No performance records need updates")
                self.logger.info("=" * 80)
                return {'processed': 0, 'updated': 0, 'failed': 0}
            
            # Log sample of records
            if len(performance_records) > 0:
                sample = performance_records[0]
                self.logger.info(f"[SAMPLE] First record: ticker={sample['signals']['ticker']}, "
                               f"baseline_date={sample['baseline_date']}, "
                               f"intervals_completed={sample.get('intervals_completed', [])}")
            
            self.logger.info(f"[UPDATING] Processing {len(performance_records)} performance records...")
            
            stats = {'processed': 0, 'updated': 0, 'failed': 0}
            
            for perf in performance_records:
                stats['processed'] += 1
                ticker = perf['signals']['ticker']
                
                try:
                    # Parse dates - handle both string and datetime objects
                    baseline_date_raw = perf['baseline_date']
                    if isinstance(baseline_date_raw, str):
                        # Clean up timezone format and handle microseconds
                        baseline_date_str = baseline_date_raw.replace('Z', '+00:00')
                        # Normalize microseconds to exactly 6 digits
                        if '+' in baseline_date_str:
                            dt_part, tz_part = baseline_date_str.rsplit('+', 1)
                            if '.' in dt_part:
                                dt_main, microsec = dt_part.rsplit('.', 1)
                                # Python's fromisoformat requires exactly 0, 1, 2, 3, 4, or 6 digits
                                # 5 digits is invalid, so pad to 6
                                if len(microsec) == 5:
                                    microsec = microsec + '0'
                                elif len(microsec) > 6:
                                    microsec = microsec[:6]
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
                    
                    self.logger.info(
                        f"[{ticker}] Age: {signal_age_days}d | "
                        f"Eligible: {eligible_intervals} | "
                        f"Completed: {completed} | "
                        f"Missing: {missing_intervals}"
                    )
                    
                    if not missing_intervals:
                        # All eligible intervals are done
                        if len(completed) == len(self.intervals):
                            # Mark as completed if ALL intervals are done
                            self.logger.info(f"[{ticker}] All intervals complete ({len(self.intervals)}/{len(self.intervals)}) - marking as completed")
                            await self._update_performance_status(perf['id'], 'completed')
                            stats['updated'] += 1
                        else:
                            self.logger.debug(f"[{ticker}] No new intervals to calculate (waiting for signal to age)")
                        continue
                    
                    # Calculate missing intervals
                    baseline_price = float(perf['baseline_price'])
                    sector_etf = perf.get('sector_etf')  # v3.2: Get sector ETF if available
                    
                    self.logger.info(
                        f"[{ticker}] CALCULATING {len(missing_intervals)} intervals: {missing_intervals} "
                        f"(baseline_price={baseline_price:.2f}, sector_etf={sector_etf})"
                    )
                    
                    # Fetch price data and calculate returns
                    update_data = await self._calculate_interval_returns(
                        ticker=ticker,
                        baseline_price=baseline_price,
                        baseline_date=baseline_date,
                        intervals=missing_intervals,
                        sector_etf=sector_etf,  # v3.2: Pass sector ETF
                        benchmark_cache=benchmark_cache  # v3.3: Pass cached benchmarks
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
            
            self.logger.info("=" * 80)
            self.logger.info(
                f"✅ PHASE 6 COMPLETE: "
                f"{stats['updated']}/{stats['processed']} updated, "
                f"{stats['failed']} failed"
            )
            self.logger.info("=" * 80)
            
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
        sector_etf: Optional[str] = None,  # v3.2: Sector ETF for comparison
        benchmark_cache: Optional[Dict[str, pd.DataFrame]] = None  # v3.3: Cached benchmark data from Phase 1
    ) -> Dict[str, Any]:
        """
        Calculate returns for specific intervals.
        
        v3.3: Now accepts optional benchmark_cache from Phase 1 to avoid redundant yfinance calls.
        If benchmark_cache is None, falls back to fetching directly (for standalone execution).
        
        Args:
            ticker: Stock ticker
            baseline_price: Baseline price (from signal creation)
            baseline_date: Baseline date
            intervals: List of intervals to calculate (e.g., [1, 3, 7])
            sector_etf: Sector ETF ticker (e.g., 'XLK') for sector comparison
            benchmark_cache: Optional dict of ETF ticker -> DataFrame from Phase 1
            
        Returns:
            Dict with return_Xd, spy_return_Xd, qqq_return_Xd, sector_return_Xd for each interval
        """
        try:
            import yfinance as yf
            
            # Fetch price data (baseline to now)
            end_date = datetime.now()
            start_date = baseline_date - timedelta(days=2)  # Small buffer
            
            # DEBUG: Log benchmark cache status
            self.logger.info(f"[BENCHMARK DEBUG] Ticker: {ticker}, Sector ETF: {sector_etf}")
            self.logger.info(f"[BENCHMARK DEBUG] Cache present: {benchmark_cache is not None}, Cache keys: {list(benchmark_cache.keys()) if benchmark_cache else 'None'}")
            
            # Download ticker data (ticker is NOT in benchmark cache)
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
            
            # Get benchmark data - use cache if available, otherwise fetch
            # SPY (S&P 500 benchmark)
            self.logger.info(f"[BENCHMARK DEBUG] Checking SPY: cache={'SPY' in benchmark_cache if benchmark_cache else False}")
            if benchmark_cache and 'SPY' in benchmark_cache:
                spy_df = benchmark_cache['SPY']
                self.logger.info(f"[BENCHMARK DEBUG] Using cached SPY data ({len(spy_df)} rows)")
            else:
                self.logger.info(f"[BENCHMARK DEBUG] Fetching SPY from yfinance (start={start_date}, end={end_date})")
                try:
                    spy_df = yf.download(
                        'SPY',
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    self.logger.info(f"[BENCHMARK DEBUG] Fetched SPY: {len(spy_df)} rows, empty={spy_df.empty}")
                except Exception as e:
                    self.logger.error(f"[BENCHMARK DEBUG] SPY fetch failed: {type(e).__name__}: {e}")
                    spy_df = None
            
            # Get SPY baseline price (use backward fill - last available price before/on baseline date)
            if spy_df is not None and not spy_df.empty:
                spy_baseline = self._get_price_at_date(spy_df, baseline_date, 'Close', fill_direction='backward')
                self.logger.info(f"[BENCHMARK DEBUG] SPY baseline price: {spy_baseline}")
            else:
                spy_baseline = None
                self.logger.warning(f"[BENCHMARK DEBUG] SPY data unavailable, baseline=None")
            
            # QQQ (Nasdaq benchmark)
            self.logger.info(f"[BENCHMARK DEBUG] Checking QQQ: cache={'QQQ' in benchmark_cache if benchmark_cache else False}")
            if benchmark_cache and 'QQQ' in benchmark_cache:
                qqq_df = benchmark_cache['QQQ']
                self.logger.info(f"[BENCHMARK DEBUG] Using cached QQQ data ({len(qqq_df)} rows)")
            else:
                self.logger.info(f"[BENCHMARK DEBUG] Fetching QQQ from yfinance (start={start_date}, end={end_date})")
                try:
                    qqq_df = yf.download(
                        'QQQ',
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    self.logger.info(f"[BENCHMARK DEBUG] Fetched QQQ: {len(qqq_df)} rows, empty={qqq_df.empty}")
                except Exception as e:
                    self.logger.error(f"[BENCHMARK DEBUG] QQQ fetch failed: {type(e).__name__}: {e}")
                    qqq_df = None
            
            # Get QQQ baseline price
            if qqq_df is not None and not qqq_df.empty:
                qqq_baseline = self._get_price_at_date(qqq_df, baseline_date, 'Close', fill_direction='backward')
                self.logger.info(f"[BENCHMARK DEBUG] QQQ baseline price: {qqq_baseline}")
            else:
                qqq_baseline = None
                self.logger.warning(f"[BENCHMARK DEBUG] QQQ data unavailable, baseline=None")
            
            # Sector ETF (if available)
            sector_df = None
            sector_baseline = None
            if sector_etf and sector_etf != 'SPY':
                self.logger.info(f"[BENCHMARK DEBUG] Checking Sector ETF {sector_etf}: cache={sector_etf in benchmark_cache if benchmark_cache else False}")
                try:
                    # Use cache if available
                    if benchmark_cache and sector_etf in benchmark_cache:
                        sector_df = benchmark_cache[sector_etf]
                        self.logger.info(f"[BENCHMARK DEBUG] Using cached {sector_etf} data ({len(sector_df)} rows)")
                    else:
                        self.logger.info(f"[BENCHMARK DEBUG] Fetching {sector_etf} from yfinance (start={start_date}, end={end_date})")
                        sector_df = yf.download(
                            sector_etf,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            auto_adjust=True
                        )
                        self.logger.info(f"[BENCHMARK DEBUG] Fetched {sector_etf}: {len(sector_df)} rows, empty={sector_df.empty}")
                    
                    if sector_df is not None and not sector_df.empty:
                        sector_baseline = self._get_price_at_date(sector_df, baseline_date, 'Close', fill_direction='backward')
                        self.logger.info(f"[BENCHMARK DEBUG] {sector_etf} baseline price: {sector_baseline}")
                    else:
                        self.logger.warning(f"[BENCHMARK DEBUG] {sector_etf} data empty")
                except Exception as e:
                    self.logger.error(f"[BENCHMARK DEBUG] Sector ETF {sector_etf} failed: {type(e).__name__}: {e}")
            
            # Calculate returns for each interval
            update_data = {}
            
            for interval_days in intervals:
                target_date = baseline_date + timedelta(days=interval_days)
                
                # Get ticker price at target date (use forward fill - next available price on/after target)
                target_price = self._get_price_at_date(ticker_df, target_date, 'Close', fill_direction='forward')
                
                if target_price and baseline_price > 0:
                    # Calculate ticker return
                    return_pct = ((target_price - baseline_price) / baseline_price) * 100
                    update_data[f'return_{interval_days}d'] = round(return_pct, 4)
                    
                    # Calculate SPY return for comparison
                    if spy_baseline and spy_df is not None and not spy_df.empty:
                        spy_target = self._get_price_at_date(spy_df, target_date, 'Close', fill_direction='forward')
                        
                        if spy_target and spy_baseline > 0:
                            spy_return_pct = ((spy_target - spy_baseline) / spy_baseline) * 100
                            update_data[f'spy_return_{interval_days}d'] = round(spy_return_pct, 4)
                            self.logger.info(f"[BENCHMARK DEBUG] {interval_days}d SPY return: {spy_return_pct:.2f}%")
                            
                            # NOTE: All alpha columns are GENERATED (auto-calculated by database)
                            # alpha_Xd = return_Xd - spy_return_Xd (for all 7 intervals)
                            # No need to set them here - database handles it automatically
                        else:
                            self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d SPY target unavailable (target={spy_target}, baseline={spy_baseline})")
                    else:
                        self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d SPY skipped (baseline={spy_baseline}, df_empty={spy_df is None or spy_df.empty})")
                    
                    # Calculate QQQ return for Nasdaq comparison (v3.3)
                    if qqq_baseline and qqq_df is not None and not qqq_df.empty:
                        qqq_target = self._get_price_at_date(qqq_df, target_date, 'Close', fill_direction='forward')
                        
                        if qqq_target and qqq_baseline > 0:
                            qqq_return_pct = ((qqq_target - qqq_baseline) / qqq_baseline) * 100
                            update_data[f'qqq_return_{interval_days}d'] = round(qqq_return_pct, 4)
                            self.logger.info(f"[BENCHMARK DEBUG] {interval_days}d QQQ return: {qqq_return_pct:.2f}%")
                            
                            # NOTE: All qqq_alpha columns are GENERATED (auto-calculated by database)
                            # qqq_alpha_Xd = return_Xd - qqq_return_Xd (for all 7 intervals)
                            # No need to set them here - database handles it automatically
                        else:
                            self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d QQQ target unavailable (target={qqq_target}, baseline={qqq_baseline})")
                    else:
                        self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d QQQ skipped (baseline={qqq_baseline}, df_empty={qqq_df is None or qqq_df.empty})")
                    
                    # Calculate sector ETF return (v3.2)
                    if sector_baseline and sector_df is not None and not sector_df.empty:
                        sector_target = self._get_price_at_date(sector_df, target_date, 'Close', fill_direction='forward')
                        
                        if sector_target and sector_baseline > 0:
                            sector_return_pct = ((sector_target - sector_baseline) / sector_baseline) * 100
                            update_data[f'sector_return_{interval_days}d'] = round(sector_return_pct, 4)
                            self.logger.info(f"[BENCHMARK DEBUG] {interval_days}d {sector_etf} return: {sector_return_pct:.2f}%")
                            
                            # NOTE: All sector_alpha columns are GENERATED (auto-calculated by database)
                            # sector_alpha_Xd = return_Xd - sector_return_Xd (for all 7 intervals)
                            # No need to set them here - database handles it automatically
                        else:
                            self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d {sector_etf} target unavailable (target={sector_target}, baseline={sector_baseline})")
                    else:
                        self.logger.warning(f"[BENCHMARK DEBUG] {interval_days}d Sector skipped (baseline={sector_baseline}, df_empty={sector_df is None or sector_df.empty})")
            
            return update_data
            
        except Exception as e:
            self.logger.error(f"Error calculating returns for {ticker}: {e}")
            return {}
    
    def _get_price_at_date(
        self, 
        df: pd.DataFrame, 
        target_date: datetime, 
        price_col: str = 'Close',
        fill_direction: str = 'forward'
    ) -> Optional[float]:
        """
        Get price at specific date with forward/backward fill for weekends/holidays.
        
        Args:
            df: Price DataFrame
            target_date: Target date
            price_col: Column name ('Close', 'Open', etc.)
            fill_direction: 'forward' (use next available) or 'backward' (use last available)
            
        Returns:
            Price at date, or None if not available
        """
        try:
            # Normalize to date
            target_ts = pd.Timestamp(target_date.date())
            
            # Try exact date first
            if target_ts in df.index:
                return float(df.loc[target_ts, price_col].iloc[0] if hasattr(df.loc[target_ts, price_col], 'iloc') else df.loc[target_ts, price_col])
            
            if fill_direction == 'forward':
                # Forward fill - find next available date (for future target dates)
                available_dates = [d for d in df.index if d >= target_ts]
                if available_dates:
                    return float(df.loc[available_dates[0], price_col].iloc[0] if hasattr(df.loc[available_dates[0], price_col], 'iloc') else df.loc[available_dates[0], price_col])
            else:
                # Backward fill - find last available date (for baseline dates)
                available_dates = [d for d in df.index if d <= target_ts]
                if available_dates:
                    return float(df.loc[available_dates[-1], price_col].iloc[0] if hasattr(df.loc[available_dates[-1], price_col], 'iloc') else df.loc[available_dates[-1], price_col])
            
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
