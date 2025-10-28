-- ============================================================================
-- Migration: Add Coverage Columns & Remove Backtest Columns from Signals
-- ============================================================================
-- Purpose:
--   1. Add coverage columns back to signals table (fixed to show actual factor coverage)
--   2. Remove backtest/performance columns from signals table (moved to performance table)
-- Created: 2025-10-26
-- ============================================================================

-- ============================================================================
-- STEP 1: ADD COVERAGE COLUMNS
-- ============================================================================

ALTER TABLE signals
ADD COLUMN IF NOT EXISTS total_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS technical_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS fundamental_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS news_macro_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS social_alternative_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS risk_stability_coverage DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS institutional_smart_money_coverage DECIMAL(5, 4);

-- Add comments
COMMENT ON COLUMN signals.total_coverage IS 'Overall factor coverage (0.0-1.0): average of all group coverages';
COMMENT ON COLUMN signals.technical_coverage IS 'Technical factor coverage (0.0-1.0): populated technical factors / total expected';
COMMENT ON COLUMN signals.fundamental_coverage IS 'Fundamental factor coverage (0.0-1.0): populated fundamental factors / total expected';
COMMENT ON COLUMN signals.news_macro_coverage IS 'News/Macro factor coverage (0.0-1.0): populated news/macro factors / total expected';
COMMENT ON COLUMN signals.social_alternative_coverage IS 'Social/Alternative factor coverage (0.0-1.0): populated social factors / total expected';
COMMENT ON COLUMN signals.risk_stability_coverage IS 'Risk/Stability factor coverage (0.0-1.0): populated risk factors / total expected';
COMMENT ON COLUMN signals.institutional_smart_money_coverage IS 'Institutional/Smart Money factor coverage (0.0-1.0): populated institutional factors / total expected';

-- ============================================================================
-- STEP 2: REMOVE BACKTEST/PERFORMANCE COLUMNS
-- ============================================================================

ALTER TABLE signals
DROP COLUMN IF EXISTS backtest_baseline_price,
DROP COLUMN IF EXISTS backtest_baseline_date,
DROP COLUMN IF EXISTS return_1d,
DROP COLUMN IF EXISTS return_3d,
DROP COLUMN IF EXISTS return_7d,
DROP COLUMN IF EXISTS return_10d,
DROP COLUMN IF EXISTS return_14d,
DROP COLUMN IF EXISTS return_30d,
DROP COLUMN IF EXISTS return_90d,
DROP COLUMN IF EXISTS spy_return_1d,
DROP COLUMN IF EXISTS spy_return_3d,
DROP COLUMN IF EXISTS spy_return_7d,
DROP COLUMN IF EXISTS spy_return_10d,
DROP COLUMN IF EXISTS spy_return_14d,
DROP COLUMN IF EXISTS spy_return_30d,
DROP COLUMN IF EXISTS spy_return_90d,
DROP COLUMN IF EXISTS backtest_status,
DROP COLUMN IF EXISTS backtest_last_update,
DROP COLUMN IF EXISTS backtest_error;

-- ============================================================================
-- STEP 3: DROP OLD BACKTEST INDEXES (if they exist)
-- ============================================================================

DROP INDEX IF EXISTS idx_signals_backtest_status;
DROP INDEX IF EXISTS idx_signals_created_at_backtest;
DROP INDEX IF EXISTS idx_signals_return_7d;
DROP INDEX IF EXISTS idx_signals_performance;
DROP INDEX IF EXISTS idx_signals_backtest_date;

-- Note: idx_signals_age is kept (general purpose index on created_at, not backtest-specific)

-- ============================================================================
-- STEP 4: ADD COVERAGE INDEXES FOR ANALYTICS
-- ============================================================================

-- Index for filtering by coverage quality
CREATE INDEX IF NOT EXISTS idx_signals_total_coverage ON signals(total_coverage DESC) 
WHERE total_coverage IS NOT NULL;

-- Index for finding low-coverage signals
CREATE INDEX IF NOT EXISTS idx_signals_low_coverage ON signals(ticker, total_coverage) 
WHERE total_coverage < 0.8;

-- Composite index for coverage + score queries
CREATE INDEX IF NOT EXISTS idx_signals_coverage_score ON signals(total_coverage DESC, overall_score DESC);

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check coverage column types
-- SELECT column_name, data_type, numeric_precision, numeric_scale
-- FROM information_schema.columns
-- WHERE table_name = 'signals' AND column_name LIKE '%coverage%'
-- ORDER BY column_name;

-- Check backtest columns removed
-- SELECT column_name 
-- FROM information_schema.columns
-- WHERE table_name = 'signals' 
-- AND (column_name LIKE '%backtest%' OR column_name LIKE '%return_%' OR column_name LIKE '%spy_%')
-- ORDER BY column_name;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- ✅ Coverage columns added to signals table
-- ✅ Backtest columns removed from signals table  
-- ✅ Indexes optimized for coverage queries
-- ✅ Performance data now tracked in separate performance table
-- ============================================================================
