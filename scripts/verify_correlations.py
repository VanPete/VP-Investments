#!/usr/bin/env python3
"""
Quick verification script for factor-return correlations.
Checks database for correlation data without hanging.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.storage.database import get_supabase_database


async def verify_correlations():
    """Quick verification of factor-return correlations."""
    print("\n" + "="*80)
    print("FACTOR-RETURN CORRELATIONS VERIFICATION")
    print("="*80)
    
    db = None
    try:
        # Connect to database
        print("\n📊 Connecting to database...")
        db = await get_supabase_database()
        
        # Check if column exists
        print("\n1. Checking if factor_return_correlations column exists...")
        schema_check = db.client.from_('analytics').select('factor_return_correlations').limit(1).execute()
        print("   ✅ Column exists in analytics table")
        
        # Get recent analytics records
        print("\n2. Fetching recent analytics records...")
        response = db.client.from_('analytics') \
            .select('period_type, factor_return_correlations, created_at') \
            .order('created_at', desc=True) \
            .limit(10) \
            .execute()
        
        records = response.data
        print(f"   Found {len(records)} recent analytics records")
        
        if not records:
            print("\n   ⚠️  No analytics records found in database")
            return
        
        # Analyze correlation data
        print("\n3. Analyzing correlation data:")
        print(f"   {'Period':<12} {'Has Correlations':<20} {'Created At'}")
        print(f"   {'-'*12} {'-'*20} {'-'*25}")
        
        records_with_data = 0
        for record in records:
            period = record.get('period_type', 'unknown')
            corr_data = record.get('factor_return_correlations')
            created = record.get('created_at', 'unknown')[:19]  # Truncate timestamp
            
            if corr_data and isinstance(corr_data, dict) and len(corr_data) > 0:
                # Count total correlations across all groups
                total_corrs = sum(len(factors) for factors in corr_data.values())
                has_data = f"✅ Yes ({total_corrs} factors)"
                records_with_data += 1
            else:
                has_data = "❌ No (empty/null)"
            
            print(f"   {period:<12} {has_data:<20} {created}")
        
        # Summary
        print(f"\n4. Summary:")
        print(f"   Total records checked: {len(records)}")
        print(f"   Records with correlation data: {records_with_data}")
        print(f"   Records without data: {len(records) - records_with_data}")
        
        # Show detailed data for 1d and 3d if available
        print("\n5. Detailed data for 1d and 3d intervals:")
        for period in ['1d', '3d']:
            matching = [r for r in records if r.get('period_type') == period]
            if matching:
                record = matching[0]
                corr_data = record.get('factor_return_correlations')
                if corr_data and isinstance(corr_data, dict):
                    print(f"\n   {period} interval:")
                    for group, factors in corr_data.items():
                        if factors:
                            print(f"     • {group}: {len(factors)} factors")
                            # Show first 3 factors as examples
                            for i, factor in enumerate(factors[:3]):
                                corr = factor.get('correlation', 0)
                                p_val = factor.get('p_value', 1)
                                sig = factor.get('significance', 'none')
                                print(f"       - {factor.get('name')}: r={corr:.4f}, p={p_val:.4f}, sig={sig}")
                            if len(factors) > 3:
                                print(f"       ... and {len(factors) - 3} more")
                else:
                    print(f"\n   {period} interval: No correlation data")
            else:
                print(f"\n   {period} interval: No record found")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if db:
            await db.disconnect()
            print("\n✅ Database connection closed")


if __name__ == "__main__":
    asyncio.run(verify_correlations())
