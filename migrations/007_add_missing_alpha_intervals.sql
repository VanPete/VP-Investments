-- Migration 007: Add Missing Alpha Intervals
-- Description: Adds alpha_3d, alpha_10d, alpha_14d to match all sector alpha intervals
-- Date: 2025-10-26
-- Related: Complete parity between market and sector alpha tracking

-- Add missing market alpha columns to match sector alpha intervals
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS alpha_3d REAL,
ADD COLUMN IF NOT EXISTS alpha_10d REAL,
ADD COLUMN IF NOT EXISTS alpha_14d REAL;

-- Add indexes for new alpha columns (performance queries)
CREATE INDEX IF NOT EXISTS idx_performance_alpha_3d ON performance(alpha_3d) WHERE alpha_3d IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_alpha_10d ON performance(alpha_10d) WHERE alpha_10d IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_alpha_14d ON performance(alpha_14d) WHERE alpha_14d IS NOT NULL;

-- Migration complete
-- New columns: 3 (alpha_3d, alpha_10d, alpha_14d)
-- Total performance columns: 46 (was 43)
-- Now have complete parity: 7 intervals for both market and sector alpha
