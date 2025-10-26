-- Migration 003: Add Performance Tracking + Remove Coverage Columns
-- Created: 2025-10-24
-- Purpose: 
--   1. Add performance tracking columns for backtesting (18 columns)
--   2. Remove all coverage columns (all constant 1.0, provide no information)

-- ============================================================================
-- CLEANUP: Remove Unused Coverage Columns
-- ============================================================================

-- Remove coverage columns (all constant 1.0 for all signals)
ALTER TABLE signals DROP COLUMN IF EXISTS total_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS technical_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS fundamental_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS news_macro_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS social_alternative_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS risk_stability_coverage;
ALTER TABLE signals DROP COLUMN IF EXISTS institutional_smart_money_coverage;

-- ============================================================================
-- PERFORMANCE TRACKING COLUMNS
-- ============================================================================

-- Baseline price and date (next day open to avoid lookahead bias)
ALTER TABLE signals ADD COLUMN IF NOT EXISTS backtest_baseline_price DECIMAL(10, 2);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS backtest_baseline_date TIMESTAMPTZ;

-- Interval returns (% change from baseline)
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_1d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_3d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_7d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_10d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_14d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_30d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS return_90d DECIMAL(10, 4);

-- SPY benchmark returns (for market comparison)
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_1d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_3d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_7d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_10d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_14d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_30d DECIMAL(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS spy_return_90d DECIMAL(10, 4);

-- Tracking metadata
ALTER TABLE signals ADD COLUMN IF NOT EXISTS backtest_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE signals ADD COLUMN IF NOT EXISTS backtest_last_update TIMESTAMPTZ;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS backtest_error TEXT;  -- Store any errors

-- ============================================================================
-- INDEXES FOR PERFORMANCE QUERIES
-- ============================================================================

-- Index for finding signals needing backtest
CREATE INDEX IF NOT EXISTS idx_signals_backtest_status ON signals(backtest_status);
CREATE INDEX IF NOT EXISTS idx_signals_created_at_backtest ON signals(created_at) WHERE backtest_status = 'pending';

-- Index for performance analysis queries
CREATE INDEX IF NOT EXISTS idx_signals_return_7d ON signals(return_7d) WHERE return_7d IS NOT NULL;

-- Index for finding signals by age
CREATE INDEX IF NOT EXISTS idx_signals_age ON signals(created_at DESC);

-- Composite index for filtering by performance
CREATE INDEX IF NOT EXISTS idx_signals_performance ON signals(backtest_status, return_7d, spy_return_7d);

-- ============================================================================
-- HELPER FUNCTION: Calculate Signal Age
-- ============================================================================

CREATE OR REPLACE FUNCTION get_signal_age_days(signal_created_at TIMESTAMPTZ)
RETURNS INTEGER AS $$
BEGIN
    RETURN EXTRACT(DAY FROM (NOW() - signal_created_at))::INTEGER;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN signals.backtest_baseline_price IS 'Next day open price used as baseline for return calculations';
COMMENT ON COLUMN signals.backtest_baseline_date IS 'Date when baseline price was set';
COMMENT ON COLUMN signals.return_1d IS '1-day return % from baseline';
COMMENT ON COLUMN signals.return_3d IS '3-day return % from baseline';
COMMENT ON COLUMN signals.return_7d IS '7-day return % from baseline';
COMMENT ON COLUMN signals.return_10d IS '10-day return % from baseline';
COMMENT ON COLUMN signals.return_14d IS '14-day return % from baseline';
COMMENT ON COLUMN signals.return_30d IS '30-day return % from baseline';
COMMENT ON COLUMN signals.return_90d IS '90-day return % from baseline';
COMMENT ON COLUMN signals.spy_return_1d IS 'SPY 1-day return for comparison';
COMMENT ON COLUMN signals.backtest_status IS 'pending, in_progress, completed, failed';
COMMENT ON COLUMN signals.backtest_last_update IS 'Timestamp of last backtest update';
COMMENT ON COLUMN signals.backtest_error IS 'Last error message if backtest failed';
