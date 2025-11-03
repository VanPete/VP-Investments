"""
Deep dive: Why are specific old signals not updating?
====================================================

Checks if there's something wrong with the old EV signals.
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
        print('INVESTIGATING OLD EV SIGNALS')
        print('='*80)
        
        # Get all EV performance records
        result = db.client.table('performance').select(
            'id, baseline_date, baseline_price, status, intervals_completed, '
            'return_1d, spy_return_1d, last_updated, signals!inner(ticker)'
        ).eq('signals.ticker', 'EV').order('baseline_date', desc=False).execute()
        
        print(f'\n✓ Found {len(result.data)} EV performance records\n')
        
        for i, rec in enumerate(result.data, 1):
            baseline = rec['baseline_date']
            
            # Calculate age
            try:
                dt_str = baseline.replace('Z', '+00:00')
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
                age = (datetime.now(timezone.utc) - dt).days
            except:
                age = '?'
            
            intervals = rec.get('intervals_completed', [])
            has_data = 'YES' if rec.get('return_1d') is not None else 'NO'
            
            print(f'{i:2d}. Age: {age}d | Baseline: {baseline[:10]} | '
                  f'Status: {rec["status"]:12s} | Intervals: {intervals} | '
                  f'Has data: {has_data}')
        
        # Check if there's a pattern
        print('\n' + '='*80)
        print('ANALYSIS')
        print('='*80)
        
        pending_count = sum(1 for r in result.data if r['status'] == 'pending')
        in_progress_count = sum(1 for r in result.data if r['status'] == 'in_progress')
        with_data_count = sum(1 for r in result.data if r.get('return_1d') is not None)
        
        print(f'\nEV Performance Records:')
        print(f'  Total: {len(result.data)}')
        print(f'  Pending: {pending_count}')
        print(f'  In Progress: {in_progress_count}')
        print(f'  With data: {with_data_count}')
        
        # Check Phase 6 batch limit
        print(f'\n💡 Phase 6 processes 500 records per run')
        print(f'   With 629 pending signals total, old EV signals may be at position 500+')
        print(f'   Need one more pipeline run to catch remaining {629 - 500} = 129 signals')
        
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
