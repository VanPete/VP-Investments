-- Migration 020: Add Period Columns Back for Time Window Analytics
-- Date: 2025-11-01
-- Purpose: Support multiple analytics rows for different time windows (all-time, 90d, 30d)

BEGIN;

-- Add period columns back
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS period_type VARCHAR(20);
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS period_start TIMESTAMPTZ;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS period_end TIMESTAMPTZ;

-- Create index for period queries
CREATE INDEX IF NOT EXISTS idx_analytics_period_type ON analytics(period_type);

-- Update existing row to be 'all_time' type
UPDATE analytics SET period_type = 'all_time' WHERE period_type IS NULL;

-- Modify unique constraint to allow multiple analytics rows
-- We'll have one row per period type (all_time, 90d, 30d)

-- Drop the UNIQUE constraint on run_id column
ALTER TABLE analytics DROP CONSTRAINT IF EXISTS analytics_run_id_key;

-- Remove UNIQUE constraint from run_id column definition if it exists
-- Note: Supabase may have created this as a unique index instead
DROP INDEX IF EXISTS analytics_run_id_key;

-- Add new constraint: one row per period_type (allows multiple rows, each with different run_id)
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_period_type_unique 
ON analytics(period_type) WHERE period_type IS NOT NULL;

-- Keep run_id as a foreign key but allow duplicates (multiple periods can reference same run)
-- The FK constraint analytics_run_id_fkey should remain

COMMIT;

-- Summary:
-- Added period_type, period_start, period_end columns
-- Changed uniqueness from run_id to period_type
-- Now supports multiple analytics rows:
--   - period_type = 'all_time' (uses all historical data)
--   - period_type = '90d' (last 90 days)
--   - period_type = '30d' (last 30 days)
