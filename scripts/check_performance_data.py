"""
Check performance data to diagnose factor-return correlation issues.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database


async def main():
    """Check performance data for factor-return correlations."""
    
    print('='*80)
    print('Checking Performance Data for Factor-Return Correlations')
    print('='*80)
    
    # Get database
    db = await get_supabase_database()
    
    try:
        # Fetch performance records
        print('\nFetching performance records...')
        result = db.client.table('performance').select(
            'baseline_date, interval, return_pct, signal_id'
        ).limit(1000).execute()
        
        if not result.data:
            print('❌ No performance data found!')
            return
        
        print(f'✓ Found {len(result.data)} performance records')
        
        # Count by interval
        interval_counts = Counter(r['interval'] for r in result.data)
        print(f'\n📊 Records by interval:')
        for interval in ['1d', '3d', '7d', '10d', '14d', '30d', '90d', 'all_time']:
            count = interval_counts.get(interval, 0)
            print(f'  {interval:10s}: {count:4d} records')
        
        # Count records with actual returns (not None)
        print(f'\n📈 Records with actual return_pct (not None):')
        returns_by_interval = {}
        for interval in ['1d', '3d', '7d', '10d', '14d', '30d', '90d', 'all_time']:
            records_with_returns = [
                r for r in result.data 
                if r['interval'] == interval and r.get('return_pct') is not None
            ]
            returns_by_interval[interval] = len(records_with_returns)
            has_enough = '✓' if len(records_with_returns) >= 50 else '✗'
            print(f'  {interval:10s}: {len(records_with_returns):4d} records {has_enough}')
        
        # Check baseline_date range
        baseline_dates = [
            datetime.fromisoformat(r['baseline_date'].replace('Z', '+00:00')) 
            for r in result.data if r.get('baseline_date')
        ]
        
        if baseline_dates:
            oldest = min(baseline_dates)
            newest = max(baseline_dates)
            today = datetime.now(timezone.utc)
            days_ago = (today - oldest).days
            
            print(f'\n📅 Baseline Date Range:')
            print(f'  Oldest: {oldest.date()} ({days_ago} days ago)')
            print(f'  Newest: {newest.date()}')
            print(f'  Today:  {today.date()}')
            
            print(f'\n⏱️  Expected intervals with returns (based on oldest record):')
            expected = []
            if days_ago >= 1: expected.append('1d')
            if days_ago >= 3: expected.append('3d')
            if days_ago >= 7: expected.append('7d')
            if days_ago >= 10: expected.append('10d')
            if days_ago >= 14: expected.append('14d')
            if days_ago >= 30: expected.append('30d')
            if days_ago >= 90: expected.append('90d')
            
            if expected:
                print(f'  {", ".join(expected)}')
            else:
                print(f'  None (oldest record is only {days_ago} days old)')
        
        # Check signal factor data
        print(f'\n🔍 Checking for signal factor data...')
        signal_ids = [r['signal_id'] for r in result.data[:10] if r.get('signal_id')]
        
        if signal_ids:
            tables = [
                'signals_technical',
                'signals_fundamental',
                'signals_news_macro',
                'signals_social_alternative',
                'signals_risk_stability',
                'signals_institutional_smart_money'
            ]
            
            for table in tables:
                try:
                    table_result = db.client.table(table).select('signal_id').in_(
                        'signal_id', signal_ids
                    ).limit(10).execute()
                    status = '✓' if table_result.data else '✗'
                    print(f'  {table:35s}: {len(table_result.data):2d} records {status}')
                except Exception as e:
                    print(f'  {table:35s}: ERROR - {e}')
        
        # Diagnosis
        print('\n' + '='*80)
        print('💡 DIAGNOSIS')
        print('='*80)
        
        print('\nFor factor-return correlations to work, we need:')
        print('  1. ✓ Performance records with baseline_date in the past')
        print('  2. ? Actual return_pct values (not None)')
        print('  3. ? Corresponding signal factor data in signals_* tables')
        print('  4. ? At least 50 samples per interval (min_samples=50)')
        
        print('\n📋 Current Status:')
        
        # Check if any interval has enough data
        intervals_ready = [
            interval for interval, count in returns_by_interval.items()
            if count >= 50
        ]
        
        if intervals_ready:
            print(f'  ✓ Intervals with enough data (≥50 samples): {", ".join(intervals_ready)}')
            print(f'  ✓ Factor-return correlations SHOULD be calculated for these intervals')
        else:
            max_count = max(returns_by_interval.values()) if returns_by_interval else 0
            print(f'  ✗ No interval has enough data yet')
            print(f'  ✗ Maximum samples found: {max_count} (need 50)')
            print(f'  ℹ️  Need to wait for more historical data to accumulate')
        
        # Check if Phase 7 is calculating correlations
        print('\n🔧 Next Steps:')
        if intervals_ready:
            print('  1. Run Phase 7 analytics - it should calculate correlations')
            print('  2. Check logs for "Calculating factor-return correlations"')
            print('  3. Verify analytics table has correlation data')
        else:
            print('  1. Wait for pipeline runs to accumulate historical data')
            print('  2. Once data accumulates, correlations will be calculated automatically')
            print(f'  3. Current oldest record: {oldest.date()} ({days_ago} days ago)')
            print(f'  4. Need at least 50 records with returns for each interval')
    
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
