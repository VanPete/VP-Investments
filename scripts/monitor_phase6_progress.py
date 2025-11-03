"""
Monitor Phase 6 Progress in Real-Time
=====================================

Shows before/after comparison to verify fixes worked.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database
from collections import Counter


async def main():
    db = await get_supabase_database()
    
    try:
        print('='*80)
        print('PHASE 6 PROGRESS CHECK')
        print('='*80)
        
        # Get status counts
        result = db.client.table('performance').select('status').execute()
        statuses = Counter(r['status'] for r in result.data)
        total = len(result.data)
        
        print(f'\n📊 Current Status ({total} total records):')
        for status in ['pending', 'in_progress', 'completed']:
            count = statuses.get(status, 0)
            pct = (count/total*100) if total else 0
            print(f'  {status:15s}: {count:4d} ({pct:5.1f}%)')
        
        # Check data population
        with_data = sum(1 for r in result.data if r.get('status') in ['in_progress', 'completed'])
        print(f'\n✓ Records with data: {with_data}/{total} ({with_data/total*100:.1f}%)')
        
        print('\n' + '='*80)
        print('Expected after pipeline completes:')
        print('  • Pending should decrease by ~500')
        print('  • In_progress should increase by ~500')
        print('  • Oldest signals (5d old) should now have data')
        print('='*80)
        
    finally:
        await db.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
