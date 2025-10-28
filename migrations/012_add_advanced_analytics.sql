-- Migration 012: Add Advanced Analytics Fields
-- Description: Adds score bucket analysis, correlation heatmap, factor contributions, and backtest data
-- Date: 2025-10-27
-- Related: Phase 7 Analytics Expansion

-- Add new analytics columns
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS score_bucket_performance JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS factor_correlations JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS factor_contributions JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS group_performance JSONB;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS backtest_cumulative_returns JSONB;

-- Comments for new fields
COMMENT ON COLUMN analytics.score_bucket_performance IS 'Performance metrics by score bucket (Strong Buy, Buy, Hold, Sell, Strong Sell) across all intervals';
COMMENT ON COLUMN analytics.factor_correlations IS 'Correlation matrices: 6x6 group correlations and optionally 158x158 factor correlations';
COMMENT ON COLUMN analytics.factor_contributions IS 'Factor importance ranked by correlation with returns for each interval';
COMMENT ON COLUMN analytics.group_performance IS 'Factor group performance analysis - both per-signal and aggregated';
COMMENT ON COLUMN analytics.backtest_cumulative_returns IS 'Daily cumulative returns for VP strategy, SPY, and QQQ benchmarks';

-- Example JSON structures:

-- score_bucket_performance:
-- {
--   "strong_buy": {
--     "threshold": "> 0.75",
--     "count": 12,
--     "1d": {"avg_return": 0.025, "win_rate": 0.75, "sharpe": 1.8, "max": 0.12, "min": -0.03},
--     "3d": {"avg_return": 0.045, "win_rate": 0.80, "sharpe": 2.1, "max": 0.18, "min": -0.02},
--     ... (all intervals)
--   },
--   "buy": { ... },
--   "hold": { ... },
--   "sell": { ... },
--   "strong_sell": { ... }
-- }

-- factor_correlations:
-- {
--   "group_correlations": {
--     "matrix": [[1.0, 0.45, ...], [0.45, 1.0, ...], ...],
--     "labels": ["technical", "fundamental", "news_macro", "social_alternative", "risk_stability", "institutional_smart_money"]
--   },
--   "factor_correlations": {
--     "matrix": [[1.0, 0.02, ...], ...],  // 158x158 (optional, computed on demand)
--     "labels": ["rsi_14d", "macd_signal", ...]
--   },
--   "top_positive_pairs": [
--     {"factor1": "rsi_14d", "factor2": "momentum_90d", "correlation": 0.89},
--     ...
--   ],
--   "top_negative_pairs": [
--     {"factor1": "pe_ratio", "factor2": "price_momentum", "correlation": -0.72},
--     ...
--   ]
-- }

-- factor_contributions:
-- {
--   "1d": {
--     "top_20": [
--       {"factor": "rsi_14d", "group": "technical", "correlation": 0.45, "p_value": 0.001},
--       {"factor": "eps_growth", "group": "fundamental", "correlation": 0.38, "p_value": 0.002},
--       ...
--     ],
--     "bottom_20": [ ... ]  // Negative correlations
--   },
--   "3d": { ... },
--   ... (all intervals)
-- }

-- group_performance:
-- {
--   "per_signal_analysis": {
--     "dominant_group_distribution": {
--       "technical": 45,  // Number of signals where technical was strongest
--       "fundamental": 32,
--       ...
--     },
--     "avg_return_by_dominant_group": {
--       "technical": 0.023,
--       "fundamental": 0.031,
--       ...
--     }
--   },
--   "aggregated_analysis": {
--     "technical": {
--       "avg_score": 0.65,
--       "correlation_with_returns": {
--         "1d": 0.23,
--         "7d": 0.31,
--         "30d": 0.28
--       },
--       "signals_count": 64
--     },
--     ... (all groups)
--   }
-- }

-- backtest_cumulative_returns:
-- {
--   "start_date": "2024-01-01",
--   "end_date": "2025-10-27",
--   "daily_returns": [
--     {
--       "date": "2024-01-01",
--       "vp_strategy": 1.0,
--       "spy": 1.0,
--       "qqq": 1.0
--     },
--     {
--       "date": "2024-01-02",
--       "vp_strategy": 1.023,
--       "spy": 1.005,
--       "qqq": 1.008
--     },
--     ...
--   ],
--   "summary": {
--     "vp_total_return": 0.245,
--     "spy_total_return": 0.118,
--     "qqq_total_return": 0.156,
--     "vp_sharpe": 1.85,
--     "vp_max_drawdown": -0.12,
--     "vp_win_rate": 0.68
--   }
-- }

-- Migration complete
-- Added 5 JSONB columns to analytics table

