"""Check Oct 26 performance records."""
import asyncio
from backend.storage.database import SupabaseInterface
from datetime import datetime, timezone

async def main():
    db = SupabaseInterface()
    
    # Get records from Oct 26 (should be 2+ days old)
    result = db.client.table('performance').select(
        'id, return_1d, spy_return_1d, intervals_completed, status, baseline_date, signals!inner(ticker)'
    ).gte('baseline_date', '2025-10-26T00:00:00+00:00').lt('baseline_date', '2025-10-27T00:00:00+00:00').limit(20).execute()
    
    print(f"\n=== Performance Records from Oct 26 ===")
    print(f"Found {len(result.data)} records\n")
    
    if not result.data:
        print("No records found from Oct 26")
        return
    
    for record in result.data:
        ticker = record['signals']['ticker']
        baseline = datetime.fromisoformat(record['baseline_date'].replace('Z', '+00:00'))
        age_hours = (datetime.now(timezone.utc) - baseline).total_seconds() / 3600
        
        print(f"Ticker: {ticker}")
        print(f"  Age: {age_hours:.1f} hours ({age_hours/24:.1f} days)")
        print(f"  Baseline: {record['baseline_date']}")
        print(f"  Return 1d: {record.get('return_1d', 'NULL')}")
        print(f"  SPY Return 1d: {record.get('spy_return_1d', 'NULL')}")
        print(f"  Intervals Completed: {record.get('intervals_completed', [])}")
        print()

if __name__ == '__main__':
    asyncio.run(main())
