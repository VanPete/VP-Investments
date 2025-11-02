"""
Backfill Benchmark Data for Existing Performance Records
========================================================

This script recalculates benchmark returns (SPY, QQQ, sector) for performance
records that have ticker returns but NULL benchmark data.

Target: Oct 28 records with intervals [1, 3] that have NULL benchmarks.

Strategy:
1. Find all performance records with NULL benchmarks but valid ticker returns
2. For each record, fetch benchmark data for the date range
3. Calculate benchmark returns for completed intervals
4. Update performance table with benchmark data

This fixes the historical issue where benchmarks weren't populated during
initial Phase 6 runs.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.storage.database import get_supabase_database
from backend.phases.phase6_performance import PerformanceUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def backfill_benchmarks(limit: int = 100, dry_run: bool = False):
    """
    Backfill benchmark data for existing performance records.
    
    Args:
        limit: Maximum number of records to process
        dry_run: If True, only show what would be updated without making changes
    """
    
    logger.info("=" * 100)
    logger.info("BENCHMARK BACKFILL SCRIPT")
    logger.info("=" * 100)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE UPDATE'}")
    logger.info(f"Limit: {limit} records")
    logger.info("=" * 100)
    
    try:
        # Initialize database
        db = await get_supabase_database()
        logger.info("✓ Database connected")
        
        # Find records needing backfill
        # Criteria: Has ticker return but NULL benchmark returns
        logger.info("\n🔍 Searching for records needing benchmark backfill...")
        
        result = db.client.table('performance').select(
            'id, signal_id, baseline_price, baseline_date, intervals_completed, '
            'sector, sector_etf, '
            'return_1d, spy_return_1d, qqq_return_1d, sector_return_1d, '
            'return_3d, spy_return_3d, qqq_return_3d, sector_return_3d, '
            'return_7d, spy_return_7d, qqq_return_7d, sector_return_7d, '
            'signals!inner(ticker)'
        ).not_.is_('return_1d', 'null').is_('spy_return_1d', 'null').limit(limit).execute()
        
        records_to_backfill = result.data if result.data else []
        
        if not records_to_backfill:
            logger.info("✓ No records need benchmark backfill")
            return
        
        logger.info(f"\n📊 Found {len(records_to_backfill)} records needing benchmark backfill")
        
        # Show sample
        if len(records_to_backfill) > 0:
            sample = records_to_backfill[0]
            logger.info(f"\n📝 Sample record:")
            logger.info(f"  Ticker: {sample['signals']['ticker']}")
            logger.info(f"  Baseline date: {sample['baseline_date']}")
            logger.info(f"  Intervals completed: {sample.get('intervals_completed', [])}")
            logger.info(f"  Return 1d: {sample.get('return_1d')} (has data)")
            logger.info(f"  SPY 1d: {sample.get('spy_return_1d')} (NULL - needs backfill)")
            logger.info(f"  QQQ 1d: {sample.get('qqq_return_1d')} (NULL - needs backfill)")
        
        if dry_run:
            logger.info("\n⚠️  DRY RUN MODE - No changes will be made")
            logger.info(f"Would backfill {len(records_to_backfill)} records")
            return
        
        # Initialize Phase 6 updater
        updater = PerformanceUpdater(db=db)
        
        # Process each record
        logger.info("\n🔄 Starting benchmark backfill...")
        stats = {'processed': 0, 'updated': 0, 'failed': 0}
        
        for record in records_to_backfill:
            stats['processed'] += 1
            ticker = record['signals']['ticker']
            
            try:
                # Parse baseline date
                baseline_date_str = record['baseline_date']
                if isinstance(baseline_date_str, str):
                    baseline_date_str = baseline_date_str.replace('Z', '+00:00')
                    if '+' in baseline_date_str:
                        dt_part, tz_part = baseline_date_str.rsplit('+', 1)
                        if '.' in dt_part:
                            dt_main, microsec = dt_part.rsplit('.', 1)
                            if len(microsec) == 5:
                                microsec = microsec + '0'
                            elif len(microsec) > 6:
                                microsec = microsec[:6]
                            baseline_date_str = f"{dt_main}.{microsec}+{tz_part}"
                    baseline_date = datetime.fromisoformat(baseline_date_str)
                else:
                    baseline_date = baseline_date_str
                
                # Get intervals that need benchmark data
                intervals_completed = record.get('intervals_completed', [])
                
                if not intervals_completed:
                    logger.warning(f"  [{ticker}] No completed intervals - skipping")
                    continue
                
                logger.info(f"  [{ticker}] Backfilling benchmarks for intervals {intervals_completed}")
                
                # Calculate benchmark returns for completed intervals
                baseline_price = float(record['baseline_price'])
                sector_etf = record.get('sector_etf')
                
                benchmark_data = await updater._calculate_interval_returns(
                    ticker=ticker,
                    baseline_price=baseline_price,
                    baseline_date=baseline_date,
                    intervals=intervals_completed,
                    sector_etf=sector_etf,
                    benchmark_cache=None  # Force direct fetch
                )
                
                if benchmark_data:
                    # Only update benchmark columns (preserve existing ticker returns)
                    update_payload = {}
                    
                    for interval in intervals_completed:
                        # Extract only benchmark returns (not ticker returns)
                        spy_key = f'spy_return_{interval}d'
                        qqq_key = f'qqq_return_{interval}d'
                        sector_key = f'sector_return_{interval}d'
                        
                        if spy_key in benchmark_data:
                            update_payload[spy_key] = benchmark_data[spy_key]
                        if qqq_key in benchmark_data:
                            update_payload[qqq_key] = benchmark_data[qqq_key]
                        if sector_key in benchmark_data:
                            update_payload[sector_key] = benchmark_data[sector_key]
                    
                    if update_payload:
                        # Update database
                        await updater._update_performance_record(record['id'], update_payload)
                        stats['updated'] += 1
                        
                        logger.info(f"  ✅ [{ticker}] Updated {len(update_payload)} benchmark fields")
                    else:
                        logger.warning(f"  ⚠️  [{ticker}] No benchmark data returned")
                else:
                    logger.warning(f"  ⚠️  [{ticker}] Failed to calculate benchmarks")
                    stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"  ❌ [{ticker}] Error: {e}")
                stats['failed'] += 1
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 100)
        logger.info(f"  Processed: {stats['processed']}")
        logger.info(f"  Updated:   {stats['updated']}")
        logger.info(f"  Failed:    {stats['failed']}")
        logger.info(f"  Success:   {stats['updated'] / stats['processed'] * 100:.1f}%" if stats['processed'] > 0 else "  Success:   N/A")
        logger.info("=" * 100)
        
        # Verify results
        logger.info("\n🔍 Verifying backfill results...")
        
        result = db.client.table('performance').select(
            'baseline_date, intervals_completed, return_1d, spy_return_1d, qqq_return_1d, '
            'signals!inner(ticker)'
        ).eq('baseline_date', '2025-10-28T15:43:38.172366+00:00').limit(3).execute()
        
        if result.data:
            logger.info(f"\n📈 Sample of Oct 28 records after backfill:")
            for rec in result.data[:3]:
                ticker = rec['signals']['ticker']
                intervals = rec.get('intervals_completed', [])
                
                logger.info(f"\n  {ticker}:")
                logger.info(f"    Intervals: {intervals}")
                
                if 1 in intervals:
                    spy = rec.get('spy_return_1d')
                    qqq = rec.get('qqq_return_1d')
                    status = "✓ FIXED" if spy is not None else "✗ STILL NULL"
                    logger.info(f"    1d: return={rec.get('return_1d')}, spy={spy}, qqq={qqq} {status}")
        
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}", exc_info=True)
        return 1
    
    return 0


async def main():
    """Run backfill with options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill benchmark data for performance records')
    parser.add_argument('--limit', type=int, default=100, help='Max records to process (default: 100)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--all', action='store_true', help='Process all records (no limit)')
    
    args = parser.parse_args()
    
    limit = 9999 if args.all else args.limit
    
    exit_code = await backfill_benchmarks(limit=limit, dry_run=args.dry_run)
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
