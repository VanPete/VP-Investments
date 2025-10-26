-- ============================================================================
-- VP Investments - Phase 6 Backtest Columns Migration
-- ============================================================================
-- Created: 2025-10-26
-- Purpose: Add backtest performance tracking columns to signals table
-- ============================================================================

-- Add backtest columns to signals table
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS backtest_baseline_price DECIMAL(12, 2),
ADD COLUMN IF NOT EXISTS backtest_baseline_date TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS backtest_status VARCHAR(20) DEFAULT 'pending' 
    CHECK (backtest_status IN ('pending', 'baseline_set', 'in_progress', 'completed', 'failed')),
ADD COLUMN IF NOT EXISTS backtest_last_update TIMESTAMPTZ,

-- Stock returns (as percentages, e.g., 5.23 = 5.23%)
ADD COLUMN IF NOT EXISTS return_1d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_3d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_7d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_10d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_14d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_30d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS return_90d DECIMAL(10, 4),

-- SPY (S&P 500) returns for comparison
ADD COLUMN IF NOT EXISTS spy_return_1d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_3d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_7d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_10d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_14d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_30d DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS spy_return_90d DECIMAL(10, 4);

-- Add index for backtest status queries
CREATE INDEX IF NOT EXISTS idx_signals_backtest_status ON signals(backtest_status) 
    WHERE backtest_status IS NOT NULL;

-- Add index for backtest date queries
CREATE INDEX IF NOT EXISTS idx_signals_backtest_date ON signals(backtest_baseline_date DESC) 
    WHERE backtest_baseline_date IS NOT NULL;

-- Add comments for documentation
COMMENT ON COLUMN signals.backtest_baseline_price IS 'Entry price (next day open after signal creation)';
COMMENT ON COLUMN signals.backtest_baseline_date IS 'Date when baseline price was recorded';
COMMENT ON COLUMN signals.backtest_status IS 'Backtest processing status: pending, baseline_set, in_progress, completed, failed';
COMMENT ON COLUMN signals.backtest_last_update IS 'Last time backtest data was updated';
COMMENT ON COLUMN signals.return_1d IS '1-day return percentage from baseline';
COMMENT ON COLUMN signals.return_7d IS '7-day return percentage from baseline';
COMMENT ON COLUMN signals.return_30d IS '30-day return percentage from baseline';
COMMENT ON COLUMN signals.return_90d IS '90-day return percentage from baseline';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Backtest columns added to signals table
-- Ready for Phase 6 backtest execution
-- ============================================================================
