-- ============================================================================
-- COMPREHENSIVE DATABASE FIXES
-- ============================================================================
-- Date: 2025-10-28
-- Issues Fixed:
--   1. Supabase foreign key relationship for signals->performance JOIN
--   2. Analytics table unique constraint to prevent duplicates
--   3. Cleanup old/orphaned data
-- ============================================================================

-- ============================================================================
-- FIX #1: Add Foreign Key Relationship for Supabase JOIN
-- ============================================================================
-- This enables Supabase to automatically JOIN signals and performance tables

-- First, verify the foreign key exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'performance_signal_id_fkey'
    ) THEN
        ALTER TABLE performance
        ADD CONSTRAINT performance_signal_id_fkey
        FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE;
        
        RAISE NOTICE 'Foreign key constraint added: performance_signal_id_fkey';
    ELSE
        RAISE NOTICE 'Foreign key constraint already exists: performance_signal_id_fkey';
    END IF;
END $$;

-- Create index on signal_id for faster JOINs
CREATE INDEX IF NOT EXISTS idx_performance_signal_id ON performance(signal_id);

COMMENT ON CONSTRAINT performance_signal_id_fkey ON performance IS 
'Foreign key relationship enabling Supabase automatic JOIN with signals table';

-- ============================================================================
-- FIX #2: Analytics Table - Prevent Duplicate Rows
-- ============================================================================
-- Add unique constraint on period_type + period_end to prevent duplicates
-- This will force Phase 7 to use UPSERT logic

-- First, remove any duplicate analytics records (keep most recent)
WITH duplicates AS (
    SELECT id, 
           ROW_NUMBER() OVER (
               PARTITION BY period_type, period_start, period_end 
               ORDER BY created_at DESC
           ) AS rn
    FROM analytics
)
DELETE FROM analytics
WHERE id IN (
    SELECT id FROM duplicates WHERE rn > 1
);

-- Add unique constraint
ALTER TABLE analytics
DROP CONSTRAINT IF EXISTS analytics_period_unique;

ALTER TABLE analytics
ADD CONSTRAINT analytics_period_unique 
UNIQUE (period_type, period_start, period_end);

COMMENT ON CONSTRAINT analytics_period_unique ON analytics IS 
'Prevents duplicate analytics for same time period - forces UPSERT behavior';

-- ============================================================================
-- FIX #3: Cleanup Orphaned Performance Records
-- ============================================================================
-- Remove performance records that don't have matching signals

DELETE FROM performance
WHERE signal_id NOT IN (SELECT id FROM signals);

-- ============================================================================
-- FIX #4: Cleanup Old Analytics Records (Optional)
-- ============================================================================
-- Remove analytics records that reference deleted signals

-- This is informational - shows how many analytics exist
DO $$
DECLARE
    analytics_count INT;
BEGIN
    SELECT COUNT(*) INTO analytics_count FROM analytics;
    RAISE NOTICE 'Current analytics records: %', analytics_count;
END $$;

-- Optional: Delete all analytics if you want to start fresh
-- Uncomment if you want to wipe analytics:
-- DELETE FROM analytics;
-- RAISE NOTICE 'All analytics records deleted - will be recalculated on next Phase 7 run';

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check foreign key relationship
SELECT 
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_name = 'performance';

-- Check unique constraints on analytics
SELECT constraint_name, constraint_type
FROM information_schema.table_constraints
WHERE table_name = 'analytics';

-- Count orphaned records
SELECT 
    (SELECT COUNT(*) FROM performance WHERE signal_id NOT IN (SELECT id FROM signals)) AS orphaned_performance,
    (SELECT COUNT(*) FROM signals) AS total_signals,
    (SELECT COUNT(*) FROM performance) AS total_performance,
    (SELECT COUNT(*) FROM analytics) AS total_analytics;

-- ============================================================================
-- POST-DEPLOYMENT NOTES
-- ============================================================================
-- 
-- After running this migration:
-- 
-- 1. Frontend JOIN will work:
--    - signals.performance() will return data correctly
--    - Error "Failed to fetch signals" should be resolved
-- 
-- 2. Analytics won't duplicate:
--    - Phase 7 will UPSERT instead of INSERT
--    - Old duplicate records cleaned up
-- 
-- 3. Data integrity:
--    - Performance records orphaned by signal deletion will cascade delete
--    - No more orphaned records
-- 
-- 4. Next steps:
--    - Update Phase 7 code to use UPSERT (ON CONFLICT UPDATE)
--    - Test frontend dashboard loads correctly
--    - Wipe all signals to start fresh (if desired)
-- 
-- ============================================================================
