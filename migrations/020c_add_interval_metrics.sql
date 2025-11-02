-- Migration 020c: Add Interval-Specific Metrics to Analytics Table
-- Date: 2025-11-02
-- Purpose: Add columns for interval-specific performance metrics (win_rate, sharpe, etc.)
-- Since each analytics row represents one interval (1d, 3d, 7d, etc.), we store the metrics directly

BEGIN;

-- Add interval-specific performance metrics
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS win_rate NUMERIC;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS sharpe_ratio NUMERIC;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS max_drawdown NUMERIC;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_return NUMERIC;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_alpha NUMERIC;

-- Add comments for documentation
COMMENT ON COLUMN analytics.win_rate IS 'Win rate percentage for this interval (e.g., 65.3 means 65.3%)';
COMMENT ON COLUMN analytics.sharpe_ratio IS 'Sharpe ratio (risk-adjusted return) for this interval';
COMMENT ON COLUMN analytics.max_drawdown IS 'Maximum drawdown percentage for this interval';
COMMENT ON COLUMN analytics.avg_return IS 'Average return percentage for this interval';
COMMENT ON COLUMN analytics.avg_alpha IS 'Average alpha vs SPY for this interval';

COMMIT;

-- Summary:
-- Each analytics row now stores:
--   - period_type: The holding period (1d, 3d, 7d, 10d, 14d, 30d, 90d)
--   - win_rate, sharpe_ratio, max_drawdown, avg_return, avg_alpha: Metrics for that specific period
