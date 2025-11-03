"""Check the latest IC data from analytics table."""

import json
from backend.storage.database import get_database

def main():
    db = get_database()
    
    # Get most recent analytics records
    result = db.client.table('analytics').select(
        'created_at, period_start, period_end, ic_mean, ic_std, ic_series, hit_rate_top_decile'
    ).order('created_at', desc=True).limit(20).execute()
    
    if result.data and len(result.data) > 0:
        print("=" * 80)
        print(f"CHECKING {len(result.data)} ANALYTICS RECORDS FOR IC DATA")
        print("=" * 80)
        
        # Find records with IC series
        records_with_ic = [d for d in result.data if d.get('ic_series') is not None]
        print(f"\nRecords with IC series: {len(records_with_ic)} / {len(result.data)}")
        
        if not records_with_ic:
            print("\n❌ No records found with IC series data!")
            return
        
        data = records_with_ic[0]
        
        print("\n" + "=" * 80)
        print("LATEST RECORD WITH IC DATA")
        print("=" * 80)
        print(f"\nCreated: {data.get('created_at')}")
        print(f"Period: {data.get('period_start')} to {data.get('period_end')}")
        print(f"\nIC Mean: {data.get('ic_mean')}")
        print(f"IC Std: {data.get('ic_std')}")
        print(f"Hit Rate (Top 10%): {data.get('hit_rate_top_decile')}")
        
        ic_series = data.get('ic_series')
        if ic_series:
            print(f"\nIC Series: {len(ic_series)} entries")
            print(f"\nFirst 3 entries:")
            for entry in ic_series[:3]:
                print(f"  {entry}")
            print(f"\nLast 3 entries:")
            for entry in ic_series[-3:]:
                print(f"  {entry}")
        else:
            print("\n❌ IC Series: None")
        
        print("\n" + "=" * 80)
    else:
        print("No analytics data found")

if __name__ == '__main__':
    main()
