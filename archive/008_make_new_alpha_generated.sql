-- Migration 008: Make New Alpha Columns Generated
-- Description: Convert alpha_3d, alpha_10d, alpha_14d to generated columns for consistency
-- Date: 2025-10-26
-- Related: Complete parity with existing alpha columns (auto-calculated from return - spy_return)

-- Drop the regular columns first (they were added in migration 007)
ALTER TABLE performance
DROP COLUMN IF EXISTS alpha_3d,
DROP COLUMN IF EXISTS alpha_10d,
DROP COLUMN IF EXISTS alpha_14d;

-- Re-add as GENERATED columns (auto-calculated by database)
ALTER TABLE performance
ADD COLUMN alpha_3d REAL GENERATED ALWAYS AS (return_3d - spy_return_3d) STORED,
ADD COLUMN alpha_10d REAL GENERATED ALWAYS AS (return_10d - spy_return_10d) STORED,
ADD COLUMN alpha_14d REAL GENERATED ALWAYS AS (return_14d - spy_return_14d) STORED;

-- Re-create indexes for new generated columns
CREATE INDEX IF NOT EXISTS idx_performance_alpha_3d ON performance(alpha_3d) WHERE alpha_3d IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_alpha_10d ON performance(alpha_10d) WHERE alpha_10d IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_alpha_14d ON performance(alpha_14d) WHERE alpha_14d IS NOT NULL;

-- Migration complete
-- Now ALL 7 alpha columns are auto-calculated:
--   alpha_1d = return_1d - spy_return_1d (existing)
--   alpha_3d = return_3d - spy_return_3d (NEW)
--   alpha_7d = return_7d - spy_return_7d (existing)
--   alpha_10d = return_10d - spy_return_10d (NEW)
--   alpha_14d = return_14d - spy_return_14d (NEW)
--   alpha_30d = return_30d - spy_return_30d (existing)
--   alpha_90d = return_90d - spy_return_90d (existing)
