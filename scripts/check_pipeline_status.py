"""Quick check of pipeline status"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface

async def check():
    db = SupabaseInterface()
    await db.connect()
    
    signals = await db.execute_query('SELECT COUNT(*) as count FROM signals')
    perf = await db.execute_query('SELECT COUNT(*) as count FROM performance')
    runs = await db.execute_query('SELECT * FROM signal_runs ORDER BY created_at DESC LIMIT 1')
    
    print(f'Signals: {signals[0]["count"]}')
    print(f'Performance: {perf[0]["count"]}')
    if runs:
        print(f'Last run: {runs[0]["created_at"]} - Status: {runs[0]["status"]}')
    
    await db.disconnect()

asyncio.run(check())
