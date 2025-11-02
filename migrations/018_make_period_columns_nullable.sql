-- ============================================================================
-- Migration 018: Make Period Columns Nullable for Run-Based Analytics
-- ============================================================================
-- Date: 2025-10-31
-- Purpose: Remove NOT NULL constraints on period_* columns to support v3.4 run-based analytics
-- Issue: Phase 7 v3.4 uses run_id (not period-based keys), but period_start/end/type have NOT NULL constraints
-- Error: "null value in column "period_start" of relation "analytics" violates not-null constraint"
-- 
-- Context:
--   - Migration 015 added run_id for run-based analytics (1 row per run)
--   - Old period-based approach used period_type/start/end (4 rows per analysis)
--   - Phase 7 v3.4 transitioned to use run_id only
--   - period_* columns remain for backwards compatibility but are no longer populated
--   
-- Solution:
--   - Drop NOT NULL constraints on period_start, period_end, period_type
--   - Keep columns (backwards compatible with old data)
--   - New records use run_id, leave period_* as NULL
--   - Result: Run-based analytics can persist successfully
-- ============================================================================

-- Remove NOT NULL constraints from period columns
ALTER TABLE public.analytics
  ALTER COLUMN period_start DROP NOT NULL,
  ALTER COLUMN period_end DROP NOT NULL,
  ALTER COLUMN period_type DROP NOT NULL;

-- Add comments explaining the migration
COMMENT ON COLUMN public.analytics.period_start IS 'DEPRECATED: Legacy period-based start date (nullable in v3.4+, use run_id instead)';
COMMENT ON COLUMN public.analytics.period_end IS 'DEPRECATED: Legacy period-based end date (nullable in v3.4+, use run_id instead)';
COMMENT ON COLUMN public.analytics.period_type IS 'DEPRECATED: Legacy period type (nullable in v3.4+, use run_id instead)';

-- Verification
DO $$ 
BEGIN
  -- Check that columns are now nullable
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND table_name = 'analytics' 
      AND column_name = 'period_start'
      AND is_nullable = 'NO'
  ) THEN
    RAISE EXCEPTION 'Migration 018 FAILED: period_start still has NOT NULL constraint';
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND table_name = 'analytics' 
      AND column_name = 'period_end'
      AND is_nullable = 'NO'
  ) THEN
    RAISE EXCEPTION 'Migration 018 FAILED: period_end still has NOT NULL constraint';
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
      AND table_name = 'analytics' 
      AND column_name = 'period_type'
      AND is_nullable = 'NO'
  ) THEN
    RAISE EXCEPTION 'Migration 018 FAILED: period_type still has NOT NULL constraint';
  END IF;
  
  RAISE NOTICE 'Migration 018 SUCCESS: period_* columns are now nullable';
END $$;

-- ============================================================================
-- ROLLBACK INSTRUCTIONS (if needed)
-- ============================================================================
-- To rollback (restore NOT NULL constraints), run:
/*
ALTER TABLE public.analytics
  ALTER COLUMN period_start SET NOT NULL,
  ALTER COLUMN period_end SET NOT NULL,
  ALTER COLUMN period_type SET NOT NULL;
*/
