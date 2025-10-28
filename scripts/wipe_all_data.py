"""
Complete Database Wipe - Fresh Start
=====================================
Removes all data from signals, performance, analytics, and signal_runs tables.
Use this to start completely fresh with all fixes in place.
"""
import asyncio
from backend.storage.database import SupabaseInterface

async def main():
    db = SupabaseInterface()
    
    print("\n" + "="*80)
    print("DATABASE WIPE - FRESH START")
    print("="*80)
    print("\n⚠️  WARNING: This will delete ALL data from:")
    print("   - performance table")
    print("   - signals table")
    print("   - analytics table")
    print("   - signal_runs table")
    print("\n" + "="*80)
    
    # Count current records
    print("\n[BEFORE] Current record counts:")
    signals = db.client.table('signals').select('id', count='exact').execute()
    performance = db.client.table('performance').select('id', count='exact').execute()
    analytics = db.client.table('analytics').select('id', count='exact').execute()
    signal_runs = db.client.table('signal_runs').select('id', count='exact').execute()
    
    print(f"  Signals: {signals.count}")
    print(f"  Performance: {performance.count}")
    print(f"  Analytics: {analytics.count}")
    print(f"  Signal Runs: {signal_runs.count}")
    
    print("\n" + "="*80)
    print("WIPING DATA...")
    print("="*80)
    
    try:
        # Delete in correct order (respecting foreign keys)
        # Performance depends on signals, so delete it first
        print("\n[1/4] Deleting performance records...")
        result = db.client.table('performance').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"  ✅ Deleted {len(result.data)} performance records")
        
        print("\n[2/4] Deleting analytics records...")
        result = db.client.table('analytics').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"  ✅ Deleted {len(result.data)} analytics records")
        
        print("\n[3/4] Deleting signals...")
        result = db.client.table('signals').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"  ✅ Deleted {len(result.data)} signals")
        
        print("\n[4/4] Deleting signal runs...")
        result = db.client.table('signal_runs').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        print(f"  ✅ Deleted {len(result.data)} signal runs")
        
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        # Verify all tables are empty
        signals = db.client.table('signals').select('id', count='exact').execute()
        performance = db.client.table('performance').select('id', count='exact').execute()
        analytics = db.client.table('analytics').select('id', count='exact').execute()
        signal_runs = db.client.table('signal_runs').select('id', count='exact').execute()
        
        print(f"\n[AFTER] Record counts:")
        print(f"  Signals: {signals.count}")
        print(f"  Performance: {performance.count}")
        print(f"  Analytics: {analytics.count}")
        print(f"  Signal Runs: {signal_runs.count}")
        
        if signals.count == 0 and performance.count == 0 and analytics.count == 0 and signal_runs.count == 0:
            print("\n✅ SUCCESS: All tables are clean!")
        else:
            print("\n⚠️  WARNING: Some records remain")
        
        print("\n" + "="*80)
        print("READY FOR FRESH PIPELINE RUN")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run: python run_pipeline_and_push.py")
        print("  2. All fixes will be applied:")
        print("     ✅ Sector extraction from Phase 1")
        print("     ✅ Current_price from Phase 1")
        print("     ✅ Performance baseline records")
        print("     ✅ Analytics with UPSERT (no duplicates)")
        print("     ✅ Frontend JOIN working")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
