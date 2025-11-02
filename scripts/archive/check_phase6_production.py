#!/usr/bin/env python
"""Check Phase 6 benchmark population from today's pipeline run."""
import sys
sys.path.insert(0, '.')

import asyncio
from backend.storage.database import get_supabase_database


async def main():
    db = await get_supabase_database()
    
    print("\n" + "="*80)
    print("PHASE 6 PRODUCTION VERIFICATION")
    print("="*80)
    
    # Get today's performance records (Nov 1, 2025)
    result = db.client.table('performance').select('''
        signals!inner(ticker, run_id),
        baseline_date,
        intervals_completed,
        return_1d,
        spy_return_1d,
        qqq_return_1d,
        sector_return_1d
    ''').gte('baseline_date', '2025-11-01').order('baseline_date', desc=True).limit(10).execute()
    
    records = result.data if result.data else []
    
    print(f"\n📊 Found {len(records)} performance records from today's pipeline run\n")
    
    if not records:
        print("⚠️ No records found - Pipeline may not have created performance records yet")
        return
    
    # Analyze benchmark population
    total = len(records)
    with_benchmarks = 0
    sample_shown = 0
    
    print("Sample Records:")
    print("-" * 80)
    
    for r in records:
        ticker = r['signals']['ticker']
        intervals = r.get('intervals_completed', [])
        return_1d = r.get('return_1d')
        spy_1d = r.get('spy_return_1d')
        qqq_1d = r.get('qqq_return_1d')
        sector_1d = r.get('sector_return_1d')
        
        # Show first 5 samples
        if sample_shown < 5:
            print(f"\n{ticker}:")
            print(f"  Baseline: {r['baseline_date'][:19]}")
            print(f"  Intervals completed: {intervals}")
            print(f"  Return 1d: {return_1d if return_1d else 'NULL'}")
            print(f"  SPY 1d: {spy_1d if spy_1d else 'NULL'} {'✓' if spy_1d else '✗'}")
            print(f"  QQQ 1d: {qqq_1d if qqq_1d else 'NULL'} {'✓' if qqq_1d else '✗'}")
            print(f"  Sector 1d: {sector_1d if sector_1d else 'NULL (OK if no sector)' if sector_1d is None else '✓'}")
            sample_shown += 1
        
        # Count records with benchmarks populated
        if spy_1d is not None and qqq_1d is not None:
            with_benchmarks += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total records: {total}")
    print(f"Records with benchmarks: {with_benchmarks}")
    print(f"Benchmark population rate: {with_benchmarks/total*100:.1f}%")
    
    if with_benchmarks == 0:
        print("\n⚠️ WARNING: No benchmarks populated!")
        print("This could mean:")
        print("  1. Records are too fresh (< 1 day old) - Phase 6 only populates after 1 day")
        print("  2. Phase 6 didn't run during this pipeline execution")
        print("  3. Phase 6 fallback logic failed")
        print("\nCheck logs/vp_investments.log for Phase 6 execution details")
    elif with_benchmarks < total:
        print(f"\n⚠️ PARTIAL: {total - with_benchmarks} records missing benchmarks")
        print("Expected if signals are less than 1 day old")
    else:
        print("\n✅ SUCCESS: All records have benchmarks populated!")
        print("Phase 6 fallback logic is working correctly in production")


if __name__ == "__main__":
    asyncio.run(main())
