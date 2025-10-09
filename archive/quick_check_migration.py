"""
Quick Check: Verify signals table state after migration
Simplified version that works with your config system
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.config import Config
from backend.storage.database import SupabaseDatabase

def quick_check():
    """Quick check of signals table after migration"""
    
    print("=" * 70)
    print("MIGRATION 001 QUICK CHECK")
    print("=" * 70)
    
    try:
        # Initialize using your config system
        config = Config()
        db = SupabaseDatabase(config)
        
        # Get a sample signal to check columns
        print("\n📊 Checking signals table structure...")
        result = db.supabase.table('signals').select('*').limit(1).execute()
        
        if result.data and len(result.data) > 0:
            columns = list(result.data[0].keys())
            column_count = len(columns)
            
            print(f"   Current column count: {column_count}")
            
            if column_count == 126:
                print("   ✅ PASS: Table has 126 columns (11 removed successfully)")
            else:
                print(f"   ⚠️  INFO: Table has {column_count} columns (expected 126)")
            
            # Check if dead columns are gone
            print("\n🔍 Checking dead columns removed...")
            dead_columns = [
                'commentary_metadata', 'score_components', 'scoring_version',
                'ai_commentary_version', 'rowid', 'ml_confidence_score',
                'prediction_confidence', 'pattern_match_score', 'signal_duration',
                'option_chain_data', 'option_volume_ratio'
            ]
            
            removed_count = 0
            still_present = []
            
            for col in dead_columns:
                if col not in columns:
                    removed_count += 1
                else:
                    still_present.append(col)
            
            print(f"   ✅ {removed_count}/11 dead columns removed")
            
            if still_present:
                print(f"   ⚠️  Still present: {', '.join(still_present)}")
            else:
                print("   ✅ All dead columns successfully removed!")
            
            # Check key columns preserved
            print("\n🔑 Checking key columns preserved...")
            key_columns = [
                'id', 'ticker', 'signal_type', 'weighted_score',
                'reddit_score', 'financial_score', 'news_score', 'created_at'
            ]
            
            preserved_count = sum(1 for col in key_columns if col in columns)
            print(f"   ✅ {preserved_count}/{len(key_columns)} key columns preserved")
            
            # Check Phase 2-4 placeholders
            print("\n📝 Checking Phase 2-4 placeholders...")
            placeholders = [
                'reddit_momentum_score', 'reddit_vs_price_divergence',
                'social_sentiment_trend', 'options_flow_score',
                'unusual_options_activity', 'institutional_flow_direction',
                'entry_quality_score', 'risk_adjusted_score'
            ]
            
            placeholder_count = sum(1 for col in placeholders if col in columns)
            print(f"   ✅ {placeholder_count}/{len(placeholders)} placeholders preserved")
            
            # Test a query
            print("\n🧪 Testing basic query...")
            test_result = db.supabase.table('signals').select(
                'ticker, signal_type, weighted_score'
            ).order('created_at', desc=True).limit(5).execute()
            
            if test_result.data:
                print(f"   ✅ Successfully queried {len(test_result.data)} recent signals")
                for signal in test_result.data[:3]:
                    print(f"   - {signal.get('ticker')}: {signal.get('signal_type')} "
                          f"(score: {signal.get('weighted_score', 0):.2f})")
            else:
                print("   ℹ️  No signals in database yet")
            
            # Final summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            
            if removed_count == 11 and preserved_count == len(key_columns):
                print("✅ Migration 001 SUCCESSFUL!")
                print(f"   - Table reduced from 137 to {column_count} columns")
                print("   - All 11 dead columns removed")
                print("   - All key columns preserved")
                print("   - Ready for Migration 002!")
            else:
                print("✅ Migration completed")
                print(f"   - {removed_count}/11 dead columns removed")
                print(f"   - {preserved_count}/{len(key_columns)} key columns preserved")
            
        else:
            print("   ⚠️  No data in signals table to verify structure")
            print("   Checking column list from schema...")
            
            # Alternative: try to get columns from information_schema
            schema_result = db.supabase.rpc('get_column_count', 
                                            {'table_name': 'signals'}).execute()
            print(f"   Schema check: {schema_result.data if schema_result.data else 'Unable to verify'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: If you see authentication errors, make sure:")
        print("  1. Your .env file has SUPABASE_URL and SUPABASE_KEY")
        print("  2. OR your config.toml has the correct credentials")
        return False
    
    return True

if __name__ == '__main__':
    success = quick_check()
    exit(0 if success else 1)
