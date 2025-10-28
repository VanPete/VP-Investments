-- Migration 010: Add Analytics Table
-- Description: Adds pre-calculated analytics for portfolio performance metrics
-- Date: 2025-10-26
-- Related: Phase 7 Analytics Engine

CREATE TABLE IF NOT EXISTS analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Time period for analytics
  period_start TIMESTAMPTZ NOT NULL,
  period_end TIMESTAMPTZ NOT NULL,
  period_type VARCHAR NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly', 'all_time')),
  
  -- Portfolio-level metrics
  total_signals INTEGER NOT NULL DEFAULT 0,
  avg_overall_score NUMERIC,
  
  -- Win rate analysis (% of signals with positive returns)
  win_rate_1d NUMERIC,
  win_rate_3d NUMERIC,
  win_rate_7d NUMERIC,
  win_rate_10d NUMERIC,
  win_rate_14d NUMERIC,
  win_rate_30d NUMERIC,
  win_rate_90d NUMERIC,
  
  -- Sharpe ratio (risk-adjusted returns)
  sharpe_ratio_1d NUMERIC,
  sharpe_ratio_3d NUMERIC,
  sharpe_ratio_7d NUMERIC,
  sharpe_ratio_10d NUMERIC,
  sharpe_ratio_14d NUMERIC,
  sharpe_ratio_30d NUMERIC,
  sharpe_ratio_90d NUMERIC,
  
  -- Max drawdown (peak-to-trough decline)
  max_drawdown_1d NUMERIC,
  max_drawdown_3d NUMERIC,
  max_drawdown_7d NUMERIC,
  max_drawdown_10d NUMERIC,
  max_drawdown_14d NUMERIC,
  max_drawdown_30d NUMERIC,
  max_drawdown_90d NUMERIC,
  
  -- Average returns
  avg_return_1d NUMERIC,
  avg_return_3d NUMERIC,
  avg_return_7d NUMERIC,
  avg_return_10d NUMERIC,
  avg_return_14d NUMERIC,
  avg_return_30d NUMERIC,
  avg_return_90d NUMERIC,
  
  -- Alpha performance (vs SPY)
  avg_alpha_1d NUMERIC,
  avg_alpha_3d NUMERIC,
  avg_alpha_7d NUMERIC,
  avg_alpha_10d NUMERIC,
  avg_alpha_14d NUMERIC,
  avg_alpha_30d NUMERIC,
  avg_alpha_90d NUMERIC,
  
  -- Sector rotation analysis
  top_sector VARCHAR,
  top_sector_avg_return NUMERIC,
  top_sector_count INTEGER,
  worst_sector VARCHAR,
  worst_sector_avg_return NUMERIC,
  worst_sector_count INTEGER,
  sector_performance JSONB, -- {"Technology": {"avg_return": 0.15, "count": 25, "win_rate": 0.72}, ...}
  
  -- Signal quality metrics (avg scores by group)
  avg_technical_score NUMERIC,
  avg_fundamental_score NUMERIC,
  avg_news_macro_score NUMERIC,
  avg_social_alternative_score NUMERIC,
  avg_risk_stability_score NUMERIC,
  avg_institutional_score NUMERIC,
  
  -- Factor analysis - top contributors
  top_factors JSONB, -- {"technical": [{"name": "rsi", "avg_value": 0.85}, ...], ...}
  
  -- Metadata
  signals_analyzed INTEGER NOT NULL DEFAULT 0,
  performance_records_used INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_analytics_period ON analytics(period_type, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics(created_at DESC);

-- Comments
COMMENT ON TABLE analytics IS 'Pre-calculated analytics metrics for portfolio performance';
COMMENT ON COLUMN analytics.period_type IS 'Time period: daily, weekly, monthly, all_time';
COMMENT ON COLUMN analytics.sharpe_ratio_1d IS 'Risk-adjusted return metric (return / volatility)';
COMMENT ON COLUMN analytics.max_drawdown_1d IS 'Maximum peak-to-trough decline';
COMMENT ON COLUMN analytics.sector_performance IS 'JSON object with per-sector metrics';
COMMENT ON COLUMN analytics.top_factors IS 'JSON object with top contributing factors per group';

-- Migration complete
-- New table: analytics (71 columns total)
