"""
Quick Phase 6 Status Check - All 1,235 Records
==============================================

Checks the full performance table to see if ANY records have been updated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database
from datetime import datetime, timezone


async def main():
    db = await get_supabase_database()
    
    try:
        print('='*80)
        print('CHECKING ALL 1,235 PERFORMANCE RECORDS')
        print('='*80)
        
        # Get ALL records with status breakdown
        result = db.client.table('performance').select(
            'status, intervals_completed, return_1d, baseline_date'
        ).execute()
        
        total = len(result.data)
        print(f'\n✓ Found {total} total records\n')
        
        # Status counts
        from collections import Counter
        statuses = Counter(r['status'] for r in result.data)
        print('📊 Status Distribution:')
        for status, count in statuses.most_common():
            pct = (count/total)*100
            print(f'  {status:15s}: {count:4d} ({pct:5.1f}%)')
        
        # How many have ANY intervals completed?
        with_intervals = sum(1 for r in result.data if r.get('intervals_completed'))
        print(f'\n✓ Records with intervals_completed: {with_intervals}/{total} ({with_intervals/total*100:.1f}%)')
        
        # How many have ANY return data?
        with_returns = sum(1 for r in result.data if r.get('return_1d') is not None)
        print(f'✓ Records with return_1d populated: {with_returns}/{total} ({with_returns/total*100:.1f}%)')
        
        # Check age distribution
        now = datetime.now(timezone.utc)
        ages = []
        for r in result.data:
            if r.get('baseline_date'):
                try:
                    dt_str = r['baseline_date'].replace('Z', '+00:00')
                    # Fix microsecond precision (5 digits -> 6 digits)
                    if '+' in dt_str:
                        dt_part, tz_part = dt_str.rsplit('+', 1)
                        if '.' in dt_part:
                            dt_main, microsec = dt_part.rsplit('.', 1)
                            if len(microsec) == 5:
                                microsec = microsec + '0'
                            elif len(microsec) > 6:
                                microsec = microsec[:6]
                            dt_str = f"{dt_main}.{microsec}+{tz_part}"
                    dt = datetime.fromisoformat(dt_str)
                    age_days = (now - dt).days
                    ages.append(age_days)
                except Exception as e:
                    continue
        
        if ages:
            print(f'\n📅 Age Distribution:')
            print(f'  Oldest: {max(ages)} days')
            print(f'  Newest: {min(ages)} days')
            print(f'  Average: {sum(ages)/len(ages):.1f} days')
            
            # How many eligible for each interval?
            eligible_1d = sum(1 for a in ages if a >= 1)
            eligible_3d = sum(1 for a in ages if a >= 3)
            eligible_7d = sum(1 for a in ages if a >= 7)
            eligible_30d = sum(1 for a in ages if a >= 30)
            
            print(f'\n⏱️  Eligible Signals:')
            print(f'  1d interval:  {eligible_1d:4d}/{total}')
            print(f'  3d interval:  {eligible_3d:4d}/{total}')
            print(f'  7d interval:  {eligible_7d:4d}/{total}')
            print(f'  30d interval: {eligible_30d:4d}/{total}')
        
        # VERDICT
        print('\n' + '='*80)
        if with_returns == 0:
            print('❌ CRITICAL ISSUE: Phase 6 has NEVER updated any records')
            print('   - 0 records have return data populated')
            print('   - All records stuck in pending status')
            print('   - Phase 6 is not running or failing silently')
        elif with_returns < eligible_1d * 0.5:
            print('⚠️  WARNING: Phase 6 is updating but at low rate')
            print(f'   - Only {with_returns}/{eligible_1d} eligible signals have data')
        else:
            print('✓ Phase 6 appears to be working')
        print('='*80)
        
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
