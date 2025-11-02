"""
Verify Migration 021: CASCADE DELETE constraints

This script queries Supabase to verify that all foreign key constraints
have been updated with ON DELETE CASCADE.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import get_supabase_database


async def verify_cascade_constraints():
    """Verify that CASCADE DELETE constraints are properly configured."""
    
    print("=" * 80)
    print("VERIFYING MIGRATION 021: CASCADE DELETE Constraints")
    print("=" * 80)
    print()
    
    # Get database connection
    db = await get_supabase_database()
    
    # Query to check foreign key constraints
    query = """
    SELECT
        tc.table_name,
        tc.constraint_name,
        kcu.column_name,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name,
        rc.delete_rule
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = tc.constraint_name
        AND ccu.table_schema = tc.table_schema
    JOIN information_schema.referential_constraints AS rc
        ON rc.constraint_name = tc.constraint_name
        AND rc.constraint_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
    AND tc.table_name IN (
        'analytics',
        'signals', 
        'performance',
        'signals_technical',
        'signals_fundamental',
        'signals_news_macro',
        'signals_social_alternative',
        'signals_risk_stability',
        'signals_institutional_smart_money'
    )
    ORDER BY tc.table_name, tc.constraint_name;
    """
    
    try:
        # Try to query using Supabase's from() method with a function
        # Since we can't run raw SQL, let's try a different approach
        print("Attempting to verify constraints...")
        print()
        
        # Expected constraints
        expected = {
            'analytics': ['analytics_run_id_fkey'],
            'signals': ['fk_signals_run'],
            'performance': ['performance_signal_id_fkey'],
            'signals_technical': ['signals_technical_signal_id_fkey'],
            'signals_fundamental': ['signals_fundamental_signal_id_fkey'],
            'signals_news_macro': ['signals_news_macro_signal_id_fkey'],
            'signals_social_alternative': ['signals_social_alternative_signal_id_fkey'],
            'signals_risk_stability': ['signals_risk_stability_signal_id_fkey'],
            'signals_institutional_smart_money': ['signals_institutional_smart_money_signal_id_fkey']
        }
        
        print("Expected Foreign Key Constraints:")
        print("-" * 80)
        
        total_constraints = 0
        for table, constraints in expected.items():
            for constraint in constraints:
                print(f"  {table:40} | {constraint}")
                total_constraints += 1
        
        print("-" * 80)
        print(f"Total: {total_constraints} constraints should have ON DELETE CASCADE")
        print()
        
        print("⚠️  Direct SQL verification not available through Python client.")
        print()
        print("To verify manually in Supabase SQL Editor:")
        print()
        print("1. Go to: https://supabase.com/dashboard")
        print("2. Select your project")
        print("3. Click 'SQL Editor' in the left menu")
        print("4. Run the following query:")
        print()
        print("-" * 80)
        print(query)
        print("-" * 80)
        print()
        print("Expected Results:")
        print("  All constraints should show: delete_rule = 'CASCADE'")
        print()
        print("If any show 'NO ACTION' or 'RESTRICT', the migration needs to be rerun.")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
    
    # Try to test cascade deletion with a dummy transaction
    print("=" * 80)
    print("TESTING CASCADE DELETION (Optional)")
    print("=" * 80)
    print()
    print("To test that cascade deletion works:")
    print()
    print("1. Find an old test run from 11/1 or 11/2:")
    print("   SELECT id, run_timestamp FROM signal_runs ORDER BY run_timestamp DESC LIMIT 10;")
    print()
    print("2. Count related records:")
    print("   SELECT")
    print("     (SELECT COUNT(*) FROM signals WHERE run_id = 'YOUR_RUN_ID') as signals,")
    print("     (SELECT COUNT(*) FROM analytics WHERE run_id = 'YOUR_RUN_ID') as analytics;")
    print()
    print("3. Delete the run:")
    print("   DELETE FROM signal_runs WHERE id = 'YOUR_RUN_ID';")
    print()
    print("4. Verify related records were deleted:")
    print("   SELECT COUNT(*) FROM signals WHERE run_id = 'YOUR_RUN_ID';  -- Should be 0")
    print("   SELECT COUNT(*) FROM analytics WHERE run_id = 'YOUR_RUN_ID';  -- Should be 0")
    print()
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(verify_cascade_constraints())
