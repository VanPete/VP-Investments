"""
Migration Script: Recalculate SPY/QQQ Returns with Fixed Logic
================================================================

Purpose:
- Fix existing SPY returns that were calculated with forward-fill bug (showing 0.0%)
- Add QQQ benchmark returns for all existing performance records
- Use correct baseline logic: backward fill for baseline, forward fill for target

Usage:
    python scripts/fix_benchmark_returns.py [--dry-run] [--limit N]

Author: System
Date: 2025-10-28
"""

import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.storage.database import SupabaseInterface


class BenchmarkReturnFixer:
    """Recalculate benchmark returns with corrected logic."""
    
    def __init__(self, dry_run: bool = False):
        self.db = SupabaseInterface()
        self.dry_run = dry_run
        self.intervals = [1, 3, 7, 10, 14, 30, 90]
        
    def _get_price_at_date(
        self, 
        df: pd.DataFrame, 
        target_date: datetime, 
        price_col: str = 'Close',
        fill_direction: str = 'forward'
    ) -> Optional[float]:
        """
        Get price at specific date with forward/backward fill.
        
        Args:
            df: Price DataFrame
            target_date: Target date
            price_col: Column name
            fill_direction: 'forward' or 'backward'
            
        Returns:
            Price at date, or None
        """
        try:
            target_ts = pd.Timestamp(target_date.date())
            
            # Try exact date first
            if target_ts in df.index:
                return float(df.loc[target_ts, price_col].iloc[0] 
                           if hasattr(df.loc[target_ts, price_col], 'iloc') 
                           else df.loc[target_ts, price_col])
            
            if fill_direction == 'forward':
                # Find next available date
                available_dates = [d for d in df.index if d >= target_ts]
                if available_dates:
                    return float(df.loc[available_dates[0], price_col].iloc[0] 
                               if hasattr(df.loc[available_dates[0], price_col], 'iloc') 
                               else df.loc[available_dates[0], price_col])
            else:
                # Find last available date
                available_dates = [d for d in df.index if d <= target_ts]
                if available_dates:
                    return float(df.loc[available_dates[-1], price_col].iloc[0] 
                               if hasattr(df.loc[available_dates[-1], price_col], 'iloc') 
                               else df.loc[available_dates[-1], price_col])
            
            return None
            
        except Exception as e:
            print(f"    Error getting price: {e}")
            return None
    
    async def recalculate_performance_record(self, perf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recalculate SPY and QQQ returns for a single performance record.
        
        Args:
            perf: Performance record from database
            
        Returns:
            Update dict or None if calculation failed
        """
        try:
            # Parse baseline date
            baseline_date_str = perf['baseline_date']
            if isinstance(baseline_date_str, str):
                baseline_date_str = baseline_date_str.replace('Z', '+00:00')
                baseline_date = datetime.fromisoformat(baseline_date_str)
            else:
                baseline_date = baseline_date_str
            
            # Get intervals to calculate
            completed = perf.get('intervals_completed') or []
            if not completed:
                return None  # No intervals completed yet
            
            # Determine date range for fetching
            max_interval = max(completed)
            end_date = baseline_date + timedelta(days=max_interval + 2)
            start_date = baseline_date - timedelta(days=2)
            
            # Fetch SPY data
            spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
            if spy_df.empty:
                print(f"    ⚠️  No SPY data available")
                return None
            
            # Fetch QQQ data
            qqq_df = yf.download('QQQ', start=start_date, end=end_date, progress=False, auto_adjust=True)
            if qqq_df.empty:
                print(f"    ⚠️  No QQQ data available")
                return None
            
            # Get baseline prices (backward fill - last available before/on baseline)
            spy_baseline = self._get_price_at_date(spy_df, baseline_date, 'Close', fill_direction='backward')
            qqq_baseline = self._get_price_at_date(qqq_df, baseline_date, 'Close', fill_direction='backward')
            
            if not spy_baseline or not qqq_baseline:
                print(f"    ⚠️  Missing baseline prices")
                return None
            
            # Calculate returns for each completed interval
            update_data = {}
            
            for interval_days in completed:
                target_date = baseline_date + timedelta(days=interval_days)
                
                # Get target prices (forward fill - next available on/after target)
                spy_target = self._get_price_at_date(spy_df, target_date, 'Close', fill_direction='forward')
                qqq_target = self._get_price_at_date(qqq_df, target_date, 'Close', fill_direction='forward')
                
                # Calculate SPY return
                if spy_target and spy_baseline > 0:
                    spy_return_pct = ((spy_target - spy_baseline) / spy_baseline) * 100
                    update_data[f'spy_return_{interval_days}d'] = round(spy_return_pct, 4)
                
                # Calculate QQQ return
                if qqq_target and qqq_baseline > 0:
                    qqq_return_pct = ((qqq_target - qqq_baseline) / qqq_baseline) * 100
                    update_data[f'qqq_return_{interval_days}d'] = round(qqq_return_pct, 4)
            
            if update_data:
                update_data['last_updated'] = datetime.now(timezone.utc).isoformat()
                return update_data
            
            return None
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None
    
    async def run(self, limit: Optional[int] = None):
        """
        Run the migration to fix all benchmark returns.
        
        Args:
            limit: Maximum number of records to process (None = all)
        """
        print("\n" + "="*80)
        print("BENCHMARK RETURN RECALCULATION")
        print("="*80)
        print(f"Mode: {'DRY RUN (no changes)' if self.dry_run else 'LIVE (will update database)'}")
        print(f"Limit: {limit if limit else 'No limit (all records)'}")
        print()
        
        # Get performance records with completed intervals
        query = self.db.client.table('performance').select(
            'id, baseline_date, intervals_completed, spy_return_1d, qqq_return_1d, signals!inner(ticker)'
        ).neq('intervals_completed', [])
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        records = result.data if result.data else []
        
        print(f"Found {len(records)} performance records with completed intervals\n")
        
        if not records:
            print("✅ No records to process")
            return
        
        stats = {
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for i, perf in enumerate(records, 1):
            ticker = perf['signals']['ticker']
            intervals = perf.get('intervals_completed', [])
            old_spy = perf.get('spy_return_1d', 'NULL')
            old_qqq = perf.get('qqq_return_1d', 'NULL')
            
            print(f"[{i}/{len(records)}] {ticker} (intervals: {intervals})")
            print(f"  Current: SPY={old_spy}, QQQ={old_qqq}")
            
            stats['processed'] += 1
            
            # Recalculate
            update_data = await self.recalculate_performance_record(perf)
            
            if update_data:
                new_spy = update_data.get('spy_return_1d', old_spy)
                new_qqq = update_data.get('qqq_return_1d', 'NEW')
                
                print(f"  Updated: SPY={new_spy}, QQQ={new_qqq}")
                
                if not self.dry_run:
                    # Apply update
                    try:
                        self.db.client.table('performance').update(
                            update_data
                        ).eq('id', perf['id']).execute()
                        stats['updated'] += 1
                        print(f"  ✅ Saved to database")
                    except Exception as e:
                        print(f"  ❌ Database error: {e}")
                        stats['failed'] += 1
                else:
                    stats['updated'] += 1
                    print(f"  ℹ️  Would update (dry run)")
            else:
                print(f"  ⏭️  Skipped (no data)")
                stats['skipped'] += 1
            
            print()
        
        # Summary
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Processed: {stats['processed']}")
        print(f"Updated:   {stats['updated']}")
        print(f"Skipped:   {stats['skipped']}")
        print(f"Failed:    {stats['failed']}")
        print()
        
        if self.dry_run:
            print("⚠️  DRY RUN - No changes were made to the database")
            print("   Run without --dry-run to apply changes")
        else:
            print("✅ Migration complete!")


async def main():
    parser = argparse.ArgumentParser(
        description='Recalculate SPY/QQQ benchmark returns with fixed logic'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without updating database'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of records to process'
    )
    
    args = parser.parse_args()
    
    fixer = BenchmarkReturnFixer(dry_run=args.dry_run)
    await fixer.run(limit=args.limit)


if __name__ == '__main__':
    asyncio.run(main())
