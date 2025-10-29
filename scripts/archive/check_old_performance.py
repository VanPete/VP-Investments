"""Check performance records from Oct 26."""
import asyncio
from backend.storage.database import SupabaseInterface
from datetime import datetime, timezone, timedelta

async def main():
    db = SupabaseInterface()
    
    # Get records older than 1 day
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    
    result = db.client.table('performance').select(
        'id, return_1d, spy_return_1d, intervals_completed, status, baseline_date, signals!inner(ticker)'
    ).lt('baseline_date', cutoff).order('baseline_date', desc=True).limit(20).execute()
    
    print(f"\n=== Performance Records Older Than 1 Day (cutoff: {cutoff}) ===")
    print(f"Found {len(result.data)} records\n")
    
    if not result.data:
        print("No records found older than 1 day")
        return
    
    for record in result.data:
        ticker = record['signals']['ticker']
        baseline = datetime.fromisoformat(record['baseline_date'].replace('Z', '+00:00'))
        age_hours = (datetime.now(timezone.utc) - baseline).total_seconds() / 3600
        
        print(f"Ticker: {ticker}")
        print(f"  Age: {age_hours:.1f} hours ({age_hours/24:.1f} days)")
        print(f"  Status: {record.get('status')}")
        print(f"  Baseline Date: {record['baseline_date']}")
        print(f"  Return 1d: {record.get('return_1d', 'NULL')}")
        print(f"  SPY Return 1d: {record.get('spy_return_1d', 'NULL')}")
        print(f"  Intervals Completed: {record.get('intervals_completed', [])}")
        print()

if __name__ == '__main__':
    asyncio.run(main())
