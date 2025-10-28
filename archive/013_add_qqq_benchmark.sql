-- Migration 013: Add QQQ (Nasdaq) Benchmark Columns
-- Purpose: Track performance against both S&P 500 (SPY) and Nasdaq (QQQ)
-- Date: 2025-10-28
-- Related: Performance table enhancement for dual benchmark tracking

-- Add QQQ return columns (7 intervals: 1d, 3d, 7d, 10d, 14d, 30d, 90d)
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS qqq_return_1d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_3d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_7d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_10d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_14d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_30d REAL,
ADD COLUMN IF NOT EXISTS qqq_return_90d REAL;

-- Add QQQ alpha columns (auto-calculated as return - qqq_return)
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS qqq_alpha_1d REAL GENERATED ALWAYS AS (return_1d - qqq_return_1d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_3d REAL GENERATED ALWAYS AS (return_3d - qqq_return_3d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_7d REAL GENERATED ALWAYS AS (return_7d - qqq_return_7d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_10d REAL GENERATED ALWAYS AS (return_10d - qqq_return_10d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_14d REAL GENERATED ALWAYS AS (return_14d - qqq_return_14d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_30d REAL GENERATED ALWAYS AS (return_30d - qqq_return_30d) STORED,
ADD COLUMN IF NOT EXISTS qqq_alpha_90d REAL GENERATED ALWAYS AS (return_90d - qqq_return_90d) STORED;

-- Create index for analytics queries
CREATE INDEX IF NOT EXISTS idx_performance_qqq_returns 
ON performance (qqq_return_1d, qqq_return_7d, qqq_return_30d);

-- Add comment
COMMENT ON COLUMN performance.qqq_return_1d IS 'QQQ (Nasdaq) benchmark return over 1 day interval';
COMMENT ON COLUMN performance.qqq_alpha_1d IS 'Signal alpha vs QQQ (auto-calculated as return_1d - qqq_return_1d)';
