#!/usr/bin/env python3
"""
Test Phase 7 logging directly to diagnose why logs aren't appearing.
This bypasses the pipeline to isolate the issue.
"""
import asyncio
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging BEFORE importing anything else
from backend.utils.logger import setup_logging

# Configure logging with DEBUG level
logger = setup_logging(
    log_level="DEBUG",
    log_dir="logs",
    console_output=True
)

print("=" * 80)
print("PHASE 7 LOGGING DIAGNOSTIC")
print("=" * 80)
print(f"Logger level: {logger.level}")
print(f"Logger handlers: {logger.handlers}")
print(f"Number of handlers: {len(logger.handlers)}")
for handler in logger.handlers:
    print(f"  - {handler.__class__.__name__}: level={handler.level}")
print("=" * 80)

from backend.storage.database import get_supabase_database
from backend.phases.phase7_analytics import get_analytics_engine


async def test_phase7():
    """Test Phase 7 directly with explicit logging."""
    print("\n>>> Starting Phase 7 test...")
    
    db = None
    try:
        # Connect to database
        print("\n1. Connecting to database...")
        db = await get_supabase_database()
        print("   [OK] Connected")
        
        # Get analytics engine
        print("\n2. Creating AnalyticsEngine...")
        analytics = get_analytics_engine(db)
        print("   [OK] Engine created")
        print(f"   Logger name: {analytics.logger.name}")
        print(f"   Logger level: {analytics.logger.level}")
        print(f"   Logger handlers: {len(analytics.logger.handlers)}")
        
        # Add explicit test log
        print("\n3. Testing logger directly...")
        analytics.logger.info("TEST LOG MESSAGE - If you see this in logs, logger works!")
        analytics.logger.info("[DEBUG] TEST - Debug marker test")
        print("   [OK] Test logs sent")
        
        # Get a recent run_id from signals table
        print("\n4. Fetching recent run_id...")
        response = db.client.from_('signals') \
            .select('run_id') \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
        
        if not response.data:
            print("   [WARN] No signals found - creating test run_id")
            run_id = "test_run_diagnostic"
        else:
            run_id = response.data[0]['run_id']
            print(f"   [OK] Found run_id: {run_id}")
        
        # Call Phase 7 (this should trigger all the debug logs)
        print(f"\n5. Calling calculate_and_persist_analytics(run_id='{run_id}')...")
        print("   Watch the logs folder for Phase 7 entries...")
        
        result = await analytics.calculate_and_persist_analytics(run_id=run_id)
        
        print("\n6. Phase 7 completed!")
        print(f"   Result keys: {list(result.keys())}")
        
        # Check if 1d and 3d have correlation data
        if '1d' in result:
            correlations_1d = result['1d'].get('factor_return_correlations', {})
            print(f"   1d correlations: {len(correlations_1d)} groups")
            
        if '3d' in result:
            correlations_3d = result['3d'].get('factor_return_correlations', {})
            print(f"   3d correlations: {len(correlations_3d)} groups")
        
        print("\n[OK] Test complete - check logs/vp_investments.log for Phase 7 entries")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if db:
            await db.disconnect()
            print("\n[DISCONNECTED] Database disconnected")


if __name__ == "__main__":
    asyncio.run(test_phase7())
