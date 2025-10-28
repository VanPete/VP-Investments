-- Migration 006: Add Sector Performance Tracking
-- Description: Adds sector-relative performance metrics to compare stocks vs their sector ETFs
-- Date: 2025-10-26
-- Related: Phase 6 Performance Tracking Enhancement

-- Add sector identification columns
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS sector TEXT,
ADD COLUMN IF NOT EXISTS sector_etf TEXT;

-- Add sector return columns (7 intervals: 1d, 3d, 7d, 10d, 14d, 30d, 90d)
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS sector_return_1d REAL,
ADD COLUMN IF NOT EXISTS sector_return_3d REAL,
ADD COLUMN IF NOT EXISTS sector_return_7d REAL,
ADD COLUMN IF NOT EXISTS sector_return_10d REAL,
ADD COLUMN IF NOT EXISTS sector_return_14d REAL,
ADD COLUMN IF NOT EXISTS sector_return_30d REAL,
ADD COLUMN IF NOT EXISTS sector_return_90d REAL;

-- Add sector alpha columns (ticker return - sector return)
ALTER TABLE performance
ADD COLUMN IF NOT EXISTS sector_alpha_1d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_3d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_7d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_10d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_14d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_30d REAL,
ADD COLUMN IF NOT EXISTS sector_alpha_90d REAL;

-- Add indexes for sector queries
CREATE INDEX IF NOT EXISTS idx_performance_sector ON performance(sector);
CREATE INDEX IF NOT EXISTS idx_performance_sector_etf ON performance(sector_etf);

-- Add comment explaining the new columns
COMMENT ON COLUMN performance.sector IS 'GICS sector name (Technology, Healthcare, etc.)';
COMMENT ON COLUMN performance.sector_etf IS 'Corresponding sector ETF ticker (XLK, XLV, etc.)';
COMMENT ON COLUMN performance.sector_alpha_1d IS 'Ticker return minus sector ETF return (1-day)';

-- Migration complete
-- New columns: 16 (sector, sector_etf, 7 sector_returns, 7 sector_alphas)
-- Total performance columns: 43 (was 27)
