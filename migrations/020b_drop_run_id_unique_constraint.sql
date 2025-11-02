-- Migration 020b: Fix - Drop run_id UNIQUE constraint that blocked multi-window analytics
-- The original migration 020 created the period_type unique index but failed to drop the run_id unique constraint
-- This constraint prevents multiple analytics rows from referencing the same run_id

-- Drop the UNIQUE constraint on run_id
ALTER TABLE analytics DROP CONSTRAINT IF EXISTS analytics_run_id_unique;

-- Drop the associated unique index (if it still exists as standalone index)
DROP INDEX IF EXISTS analytics_run_id_unique;

-- Verify: The table should now allow multiple analytics rows with the same run_id
-- Each row will be uniquely identified by period_type instead

-- Note: The foreign key analytics_run_id_fkey remains intact (good!)
-- Note: The unique index idx_analytics_period_type_unique already exists (from migration 020)
