"""
Apply migration 012: Add advanced analytics fields

Since Supabase doesn't allow ALTER TABLE via RPC, you need to run this SQL
directly in the Supabase SQL Editor:

https://supabase.com/dashboard/project/YOUR_PROJECT/sql

Copy and paste the SQL statements below:
"""

SQL_MIGRATION = """
-- Migration 012: Add Advanced Analytics Fields
-- Run this in Supabase SQL Editor

ALTER TABLE analytics ADD COLUMN IF NOT EXISTS score_bucket_performance JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS factor_correlations JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS factor_contributions JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS group_performance JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS backtest_cumulative_returns JSONB;

-- Verify columns were added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'analytics' 
  AND column_name IN (
    'score_bucket_performance',
    'factor_correlations', 
    'factor_contributions',
    'group_performance',
    'backtest_cumulative_returns'
  )
ORDER BY column_name;
"""

if __name__ == "__main__":
    print("=" * 80)
    print("MIGRATION 012: Add Advanced Analytics Fields")
    print("=" * 80)
    print("\n⚠️  Please run the following SQL in Supabase SQL Editor:\n")
    print(SQL_MIGRATION)
    print("\n" + "=" * 80)
    print("After running the SQL, press Enter to continue...")
    input()
    print("✅ Migration marked as complete. Proceeding with backend implementation...")
