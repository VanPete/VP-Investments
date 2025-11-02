"""
Get recent signals for Performance Tab testing
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database

async def get_recent_signals():
    db = await get_supabase_database()
    
    # Get 10 most recent signals
    result = db.client.table('signals').select(
        'id, ticker, created_at, overall_score'
    ).order('created_at', desc=True).limit(10).execute()
    
    print('\n' + '='*100)
    print('RECENT SIGNALS FOR TESTING')
    print('='*100)
    print(f'{"Ticker":<8} {"Signal ID":<40} {"Score":<8} {"Created At"}')
    print('-'*100)
    
    for s in result.data:
        print(f'{s["ticker"]:<8} {s["id"]:<40} {s["overall_score"]:>6.2f}   {s["created_at"]}')
    
    print('-'*100)
    print(f'Total: {len(result.data)} signals')
    print('='*100)
    
    # Return first signal ID for testing
    if result.data:
        first_signal = result.data[0]
        print(f'\n✅ Test Signal: {first_signal["ticker"]} ({first_signal["id"]})')
        return first_signal['id']
    
    return None

if __name__ == '__main__':
    signal_id = asyncio.run(get_recent_signals())
