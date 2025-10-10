-- ============================================================================
-- CLEAR ALL SIGNAL DATA - Fresh Start for Latest Run
-- ============================================================================
-- Purpose: Delete ALL signal data to ensure fresh calculations
-- Use Case: After code changes to data structure, calculations, or scoring
-- 
-- ⚠️ WARNING: This deletes ALL signals regardless of age!
-- Use this when you want completely fresh data with latest code/structure
-- ============================================================================

-- STEP 1: Show current data BEFORE deletion
SELECT 
    'BEFORE CLEARING' as status,
    COUNT(*) as total_signals,
    COUNT(DISTINCT run_id) as total_runs,
    COUNT(DISTINCT ticker) as unique_tickers,
    MIN(created_at) as oldest_signal,
    MAX(created_at) as newest_signal,
    ROUND(AVG(signal_score)::numeric, 3) as avg_signal_score,
    MAX(signal_score) as max_score
FROM signals;

-- Show breakdown by run
SELECT 
    'BY RUN' as category,
    run_id,
    COUNT(*) as signal_count,
    created_at as run_time
FROM signals
GROUP BY run_id, created_at
ORDER BY created_at DESC;

-- STEP 2: DELETE ALL SIGNALS
-- ⚠️ WARNING: This will permanently delete ALL signal data!
-- Uncomment the line below when ready to execute
-- DELETE FROM signals;

TRUNCATE TABLE signals CASCADE;

-- STEP 3: Verify deletion
SELECT 
    'AFTER CLEARING' as status,
    COUNT(*) as total_signals,
    'Table is now empty and ready for fresh run' as message
FROM signals;

-- STEP 4: Reset sequence if using serial run_id (optional)
-- This ensures next run_id starts clean
-- Uncomment if you want to reset run_id counter
-- ALTER SEQUENCE IF EXISTS signals_run_id_seq RESTART WITH 1;

-- ============================================================================
-- EXECUTION COMPLETE - DATABASE CLEARED
-- ============================================================================
-- Next Steps:
-- 1. Run fresh pipeline: python -m backend.pipeline
-- 2. All data will be calculated with latest code structure
-- 3. All scores will use current Phase 7 calculation methods
-- 
-- Benefits:
-- ✅ Latest scoring calculations applied
-- ✅ Latest data structure used
-- ✅ No old/stale data mixing with new
-- ✅ Clean slate for testing enhancements
-- ============================================================================
