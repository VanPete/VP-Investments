-- ============================================================================
-- Migration: Add Sector Column to Signals Table
-- ============================================================================
-- Purpose:
--   Move sector from performance table to signals table for better data normalization.
--   Sector is a fundamental property of the ticker, not performance-specific.
--   
-- Benefits:
--   1. Frontend can display sector without joining performance table
--   2. Faster queries (no join needed)
--   3. Better data normalization
--   4. Sector available immediately when signal created
--
-- Created: 2025-10-28
-- ============================================================================

-- ============================================================================
-- STEP 1: ADD SECTOR COLUMN TO SIGNALS TABLE
-- ============================================================================

ALTER TABLE signals
ADD COLUMN IF NOT EXISTS sector TEXT;

-- Add comment
COMMENT ON COLUMN signals.sector IS 'Industry sector from yfinance (e.g., Technology, Healthcare, Financial Services)';

-- ============================================================================
-- STEP 2: ADD INDEX FOR SECTOR QUERIES
-- ============================================================================

-- Index for filtering/grouping by sector
CREATE INDEX IF NOT EXISTS idx_signals_sector ON signals(sector) 
WHERE sector IS NOT NULL;

-- Composite index for sector + score queries
CREATE INDEX IF NOT EXISTS idx_signals_sector_score ON signals(sector, overall_score DESC)
WHERE sector IS NOT NULL;

-- ============================================================================
-- STEP 3: BACKFILL SECTOR DATA FROM PERFORMANCE TABLE
-- ============================================================================

-- Copy sector data from performance table to signals table
-- (for existing signals that have sector in performance table)
UPDATE signals s
SET sector = p.sector
FROM performance p
WHERE s.id = p.signal_id
  AND p.sector IS NOT NULL
  AND s.sector IS NULL;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check sector column exists
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'signals' AND column_name = 'sector';

-- Check sector populated
-- SELECT 
--   sector,
--   COUNT(*) as count
-- FROM signals
-- WHERE sector IS NOT NULL
-- GROUP BY sector
-- ORDER BY count DESC;

-- Verify sector matches between signals and performance
-- SELECT 
--   s.ticker,
--   s.sector as signal_sector,
--   p.sector as performance_sector,
--   CASE WHEN s.sector = p.sector THEN '✓' ELSE '✗' END as match
-- FROM signals s
-- LEFT JOIN performance p ON s.id = p.signal_id
-- WHERE s.sector IS NOT NULL OR p.sector IS NOT NULL
-- LIMIT 20;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- ✅ Sector column added to signals table
-- ✅ Indexes created for sector queries
-- ✅ Existing sector data backfilled from performance table
-- 
-- Next Steps:
-- - Update Phase 5 to save sector in signals table
-- - Update frontend to display sector column in dashboard
-- - (Optional) Remove sector from performance table in future migration
-- ============================================================================
