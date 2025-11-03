-- Migration: Add index to optimize Phase 6 performance queries
-- Date: 2025-11-03
-- Purpose: Speed up Phase 6 by indexing status and baseline_date

-- Add composite index on status and baseline_date
-- This allows PostgreSQL to efficiently find pending/in_progress records
-- ordered by baseline_date without scanning the entire table
CREATE INDEX IF NOT EXISTS idx_performance_status_baseline 
ON performance(status, baseline_date);

-- Add index on baseline_date alone for date range queries
CREATE INDEX IF NOT EXISTS idx_performance_baseline_date 
ON performance(baseline_date);

-- Optional: Add partial index for only pending/in_progress records
-- This is even more efficient since it only indexes rows we care about
CREATE INDEX IF NOT EXISTS idx_performance_active_signals 
ON performance(baseline_date) 
WHERE status IN ('pending', 'in_progress');

-- Verify indexes were created
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'performance'
ORDER BY indexname;
