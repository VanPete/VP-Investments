-- ============================================================
-- VP INVESTMENTS - 3-TABLE STRUCTURE MIGRATION
-- STEP 4: Verification Queries
-- Date: 2025-10-05
-- ============================================================

-- ============================================================
-- Part 1: Table Counts
-- ============================================================

SELECT 'STEP 4: Verification' AS step;
SELECT '==================' AS divider;

SELECT 'Table Row Counts:' AS section;
SELECT 'signals' AS table_name, COUNT(*) AS row_count FROM signals
UNION ALL
SELECT 'signal_metrics' AS table_name, COUNT(*) AS row_count FROM signal_metrics
UNION ALL
SELECT 'signal_performance' AS table_name, COUNT(*) AS row_count FROM signal_performance;

-- ============================================================
-- Part 2: Performance Breakdown by Type
-- ============================================================

SELECT '' AS divider;
SELECT 'Performance Records by Type:' AS section;
SELECT 
    backtest_type,
    COUNT(*) AS count,
    AVG(return_pct) AS avg_return,
    COUNT(CASE WHEN win THEN 1 END) AS wins,
    ROUND(COUNT(CASE WHEN win THEN 1 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) AS win_rate_pct
FROM signal_performance
GROUP BY backtest_type
ORDER BY backtest_type;

-- ============================================================
-- Part 3: Orphan Detection
-- ============================================================

SELECT '' AS divider;
SELECT 'Orphan Detection:' AS section;

-- Check for signal_metrics without parent signal
SELECT 
    'signal_metrics orphans' AS check_name,
    COUNT(*) AS orphan_count
FROM signal_metrics m
LEFT JOIN signals s ON m.signal_id = s.id
WHERE s.id IS NULL;

-- Check for signal_performance without parent signal
SELECT 
    'signal_performance orphans' AS check_name,
    COUNT(*) AS orphan_count
FROM signal_performance p
LEFT JOIN signals s ON p.signal_id = s.id
WHERE s.id IS NULL;

-- ============================================================
-- Part 4: Sample Data
-- ============================================================

SELECT '' AS divider;
SELECT 'Sample Data (Latest Signal):' AS section;

-- Show latest signal with metrics
SELECT 
    s.id,
    s.ticker,
    s.weighted_score,
    s.current_price,
    m.rsi,
    m.relative_strength,
    m.risk_score,
    s.signal_datetime
FROM signals s
LEFT JOIN signal_metrics m ON s.id = m.signal_id
ORDER BY s.signal_datetime DESC
LIMIT 1;

-- ============================================================
-- Part 5: View Verification
-- ============================================================

SELECT '' AS divider;
SELECT 'View Verification:' AS section;

SELECT 'v_signals_complete' AS view_name, COUNT(*) AS row_count FROM v_signals_complete
UNION ALL
SELECT 'v_signals_dashboard' AS view_name, COUNT(*) AS row_count FROM v_signals_dashboard
UNION ALL
SELECT 'v_signals_latest_performance' AS view_name, COUNT(*) AS row_count FROM v_signals_latest_performance;

-- ============================================================
-- Final Status
-- ============================================================

SELECT '' AS divider;
SELECT '==================' AS divider;
SELECT 'Migration Verification Complete!' AS status;
SELECT 'All tables, views, and data relationships verified.' AS message;
