"""Check Phase 6 status and today's signals."""

from backend.storage.database import get_database
from datetime import datetime, timedelta

def main():
    db = get_database()
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    print("=" * 80)
    print("PHASE 6 PERFORMANCE STATUS CHECK")
    print("=" * 80)
    
    # Check performance records created recently
    result = db.client.table('performance').select(
        'id, baseline_date, status, intervals_completed'
    ).gte('baseline_date', yesterday.isoformat()).execute()
    
    print(f"\nPerformance records since {yesterday}:")
    print(f"  Total: {len(result.data)}")
    
    # Status breakdown
    statuses = {}
    for rec in result.data:
        status = rec['status']
        statuses[status] = statuses.get(status, 0) + 1
    
    print(f"\nStatus breakdown:")
    for status, count in statuses.items():
        print(f"  {status}: {count}")
    
    print(f"\nSample records (first 5):")
    for rec in result.data[:5]:
        print(f"  baseline_date={rec['baseline_date']}, status={rec['status']}, intervals={rec['intervals_completed']}")
    
    # Check what Phase 6 would query (signals >= 1 day old)
    cutoff_date = (datetime.now() - timedelta(days=1)).isoformat()
    
    query_result = db.client.table('performance').select(
        'id, baseline_date, status'
    ).in_(
        'status', ['pending', 'in_progress']
    ).lte(
        'baseline_date', cutoff_date
    ).limit(10).execute()
    
    print(f"\n" + "=" * 80)
    print("WHAT PHASE 6 SEES (signals >= 1 day old, pending/in_progress)")
    print("=" * 80)
    print(f"Records matching Phase 6 query: {len(query_result.data)}")
    
    for rec in query_result.data[:5]:
        print(f"  baseline_date={rec['baseline_date']}, status={rec['status']}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if len(query_result.data) == 0:
        print("❌ ISSUE: No signals are >= 1 day old yet!")
        print("   Today's signals were just created and need to age before Phase 6 processes them.")
        print("   Phase 6 will start updating them tomorrow.")
    else:
        print(f"✅ Found {len(query_result.data)} signals ready for Phase 6 processing")

if __name__ == '__main__':
    main()
