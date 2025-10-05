-- ============================================================================
-- CLEAR ALL SIGNAL DATA - Fresh Database Start
-- ============================================================================
-- Purpose: Remove all signal data while preserving table structure
-- Use Case: Fresh start after migration, clean testing environment
-- CAUTION: This will DELETE all signals, metrics, and performance data!
-- ============================================================================

-- Display warning
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  WARNING: This will DELETE all signal data!';
    RAISE NOTICE '   - All signals will be removed';
    RAISE NOTICE '   - All signal_metrics will be removed';
    RAISE NOTICE '   - All signal_performance will be removed';
    RAISE NOTICE '';
    RAISE NOTICE '   Table structure will be preserved.';
    RAISE NOTICE '   This operation is IRREVERSIBLE!';
    RAISE NOTICE '';
    RAISE NOTICE 'Starting in 3 seconds...';
    RAISE NOTICE '';
END $$;

-- Wait (comment out if running in batch)
-- SELECT pg_sleep(3);

-- Get current counts before deletion
DO $$
DECLARE
    signals_count INT;
    metrics_count INT;
    performance_count INT;
BEGIN
    SELECT COUNT(*) INTO signals_count FROM signals;
    SELECT COUNT(*) INTO metrics_count FROM signal_metrics;
    SELECT COUNT(*) INTO performance_count FROM signal_performance;
    
    RAISE NOTICE '========================================';
    RAISE NOTICE 'CURRENT DATA COUNTS';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'signals: % rows', signals_count;
    RAISE NOTICE 'signal_metrics: % rows', metrics_count;
    RAISE NOTICE 'signal_performance: % rows', performance_count;
    RAISE NOTICE '========================================';
    RAISE NOTICE '';
END $$;

-- Delete data (cascades automatically due to foreign keys)
DELETE FROM signals;

-- Verify deletion
DO $$
DECLARE
    signals_count INT;
    metrics_count INT;
    performance_count INT;
BEGIN
    SELECT COUNT(*) INTO signals_count FROM signals;
    SELECT COUNT(*) INTO metrics_count FROM signal_metrics;
    SELECT COUNT(*) INTO performance_count FROM signal_performance;
    
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATABASE CLEARED';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'signals: % rows (expected: 0)', signals_count;
    RAISE NOTICE 'signal_metrics: % rows (expected: 0)', metrics_count;
    RAISE NOTICE 'signal_performance: % rows (expected: 0)', performance_count;
    RAISE NOTICE '';
    
    IF signals_count = 0 AND metrics_count = 0 AND performance_count = 0 THEN
        RAISE NOTICE '✅ All signal data successfully removed!';
        RAISE NOTICE '✅ Table structure preserved';
        RAISE NOTICE '✅ Ready for fresh pipeline run';
    ELSE
        RAISE NOTICE '❌ Warning: Some data may remain';
        RAISE NOTICE '   Please check foreign key constraints';
    END IF;
    
    RAISE NOTICE '========================================';
END $$;

-- Optional: Reset sequences (if you want IDs to start from 1 again)
-- Note: Not recommended if you're keeping any audit logs or external references
/*
ALTER SEQUENCE signals_id_seq RESTART WITH 1;
ALTER SEQUENCE signal_metrics_id_seq RESTART WITH 1;
ALTER SEQUENCE signal_performance_id_seq RESTART WITH 1;
*/

-- Vacuum to reclaim space
VACUUM FULL signals;
VACUUM FULL signal_metrics;
VACUUM FULL signal_performance;

-- Success message
SELECT '✅ Database cleared and ready for fresh start!' as status;
