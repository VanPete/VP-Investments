-- Migration 009: Make Sector Alpha Columns GENERATED
-- =====================================================
-- Makes sector_alpha_Xd columns auto-calculated by database
-- Formula: sector_alpha_Xd = return_Xd - sector_return_Xd
-- This ensures consistency with regular alpha columns

-- Drop existing sector alpha columns (they're regular columns)
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_1d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_3d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_7d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_10d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_14d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_30d;
ALTER TABLE performance DROP COLUMN IF EXISTS sector_alpha_90d;

-- Recreate as GENERATED columns
ALTER TABLE performance ADD COLUMN sector_alpha_1d NUMERIC 
    GENERATED ALWAYS AS (return_1d - sector_return_1d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_3d NUMERIC 
    GENERATED ALWAYS AS (return_3d - sector_return_3d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_7d NUMERIC 
    GENERATED ALWAYS AS (return_7d - sector_return_7d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_10d NUMERIC 
    GENERATED ALWAYS AS (return_10d - sector_return_10d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_14d NUMERIC 
    GENERATED ALWAYS AS (return_14d - sector_return_14d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_30d NUMERIC 
    GENERATED ALWAYS AS (return_30d - sector_return_30d) STORED;
    
ALTER TABLE performance ADD COLUMN sector_alpha_90d NUMERIC 
    GENERATED ALWAYS AS (return_90d - sector_return_90d) STORED;

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_1d ON performance(sector_alpha_1d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_3d ON performance(sector_alpha_3d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_7d ON performance(sector_alpha_7d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_10d ON performance(sector_alpha_10d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_14d ON performance(sector_alpha_14d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_30d ON performance(sector_alpha_30d);
CREATE INDEX IF NOT EXISTS idx_performance_sector_alpha_90d ON performance(sector_alpha_90d);
