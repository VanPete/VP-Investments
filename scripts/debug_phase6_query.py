"""
Debug: Why are old pending signals not being updated?
=====================================================

Checks if Phase 6 is actually being called and processing old signals.
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
        print('CHECKING OLD PENDING SIGNALS')
        print('='*80)
        
        # Get old pending signals (should be updated by now)
        result = db.client.table('performance').select(
            'baseline_date, status, intervals_completed, last_updated, signals!inner(ticker)'
        ).eq('status', 'pending').order('baseline_date', desc=False).limit(20).execute()
        
        if not result.data:
            print('\n✓ No pending signals found!')
            return
        
        print(f'\n❌ Found {len(result.data)} OLD signals still pending\n')
        print('Expected behavior: These should have been updated in previous pipeline runs\n')
        
        now = datetime.now(timezone.utc)
        
        for i, rec in enumerate(result.data[:10], 1):
            ticker = rec['signals']['ticker']
            baseline = rec['baseline_date']
            
            # Parse date
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
                age = (now - dt).days
            except:
                age = '?'
            
            last_updated = rec.get('last_updated', 'Never')
            intervals = rec.get('intervals_completed', [])
            
            print(f'{i:2d}. {ticker:6s} | Age: {age:2}d | Intervals: {intervals} | Last updated: {last_updated}')
        
        print('\n' + '='*80)
        print('🔍 ANALYSIS')
        print('='*80)
        
        # Check if Phase 6 ran today
        recent_updates = db.client.table('performance').select(
            'last_updated'
        ).order('last_updated', desc=True).limit(1).execute()
        
        if recent_updates.data:
            last_update = recent_updates.data[0]['last_updated']
            print(f'\n✓ Most recent Phase 6 update: {last_update}')
            
            # Parse update time
            try:
                dt_str = last_update.replace('Z', '+00:00')
                if '+' in dt_str:
                    dt_part, tz_part = dt_str.rsplit('+', 1)
                    if '.' in dt_part:
                        dt_main, microsec = dt_part.rsplit('.', 1)
                        if len(microsec) == 5:
                            microsec = microsec + '0'
                        dt_str = f"{dt_main}.{microsec}+{tz_part}"
                update_dt = datetime.fromisoformat(dt_str)
                hours_ago = (now - update_dt).total_seconds() / 3600
                
                print(f'  Time since last update: {hours_ago:.1f} hours ago')
                
                if hours_ago > 24:
                    print(f'\n  ⚠️  Phase 6 has not run in {hours_ago/24:.1f} days!')
                    print(f'  💡 This explains why old signals are not being updated')
                elif hours_ago > 1:
                    print(f'\n  ⚠️  Phase 6 last ran {hours_ago:.1f} hours ago')
                    print(f'  💡 Old signals should have been processed then')
                else:
                    print(f'\n  ✓ Phase 6 ran recently')
            except:
                pass
        
        # Check if Phase 6 query is excluding these signals
        print('\n\n🔍 Query Diagnosis:')
        print('Phase 6 queries: status IN (pending, in_progress) ORDER BY created_at ASC LIMIT 200')
        
        # Check created_at vs baseline_date
        check_result = db.client.table('performance').select(
            'signals!inner(created_at), baseline_date, status'
        ).eq('status', 'pending').order('signals(created_at)', desc=False).limit(5).execute()
        
        if check_result.data:
            print('\nFirst 5 signals by created_at (what Phase 6 sees):')
            for i, rec in enumerate(check_result.data[:5], 1):
                created = rec['signals']['created_at']
                baseline = rec['baseline_date']
                print(f'  {i}. Created: {created} | Baseline: {baseline}')
        
        print('\n' + '='*80)
        print('💡 LIKELY ISSUES:')
        print('='*80)
        print('\n1. Phase 6 orders by signals.created_at (signal creation time)')
        print('   - If signals table has newer records, old performance records get skipped')
        print('   - 200 limit means it processes newest 200 signals, not oldest performance records')
        print('\n2. Old performance records may have newer signal_id foreign keys')
        print('   - Performance.baseline_date is old, but signals.created_at is new')
        print('   - Phase 6 sorts by signals.created_at, so "old" performance records appear "new"')
        print('\n3. Solution: Change Phase 6 query order to baseline_date instead of created_at')
        
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
