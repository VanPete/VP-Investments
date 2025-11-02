"""Check which performance records are eligible for Phase 6 updates"""
import os
import sys
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.storage.database import SupabaseInterface

def main():
    db = SupabaseInterface()
    
    print("\n" + "="*80)
    print("PHASE 6 ELIGIBILITY CHECK")
    print("="*80)
    
    # Get all performance records with ticker from signals table
    result = db.client.table('performance').select(
        'signal_id, baseline_date, status, intervals_completed, created_at, signals(ticker)'
    ).order('created_at', desc=True).limit(20).execute()
    
    if not result.data:
        print("\n❌ No performance records found")
        return
    
    print(f"\nFound {len(result.data)} recent performance records\n")
    
    now = datetime.now()
    eligible_count = 0
    
    for perf in result.data:
        signal_id = perf['signal_id']
        ticker = perf['signals']['ticker'] if perf.get('signals') else 'N/A'
        baseline_raw = perf['baseline_date']
        status = perf['status']
        completed = perf.get('intervals_completed') or []
        
        # Parse baseline date
        if isinstance(baseline_raw, str):
            baseline_date = datetime.fromisoformat(baseline_raw.replace('Z', '+00:00'))
        else:
            baseline_date = baseline_raw
        
        # Calculate age
        age_days = (now - baseline_date.replace(tzinfo=None)).days
        
        # Check eligibility (at least 1 day old)
        is_eligible = age_days >= 1
        
        # Intervals that could be filled (1d, 3d, 7d, 10d, 14d, 30d, 90d)
        intervals = [1, 3, 7, 10, 14, 30, 90]
        fillable = [i for i in intervals if age_days >= i and i not in completed]
        
        marker = "✅" if is_eligible and fillable else "❌"
        
        print(f"{marker} {ticker:6} | Age: {age_days:3}d | Status: {status:12} | Completed: {len(completed)}/7 | Fillable: {fillable}")
        
        if is_eligible and fillable:
            eligible_count += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {eligible_count} records eligible for Phase 6 updates")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
