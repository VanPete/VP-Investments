-- Migration 020d: Add group_performance column to analytics table
-- This column stores quintile performance breakdowns for each score group (technical, fundamental, etc.)

-- Add group_performance as JSONB column
ALTER TABLE analytics
ADD COLUMN IF NOT EXISTS group_performance JSONB DEFAULT '{}'::jsonb;

-- Add comment explaining the structure
COMMENT ON COLUMN analytics.group_performance IS 
'Quintile performance breakdown for each score group. Structure: {group_name: {quintile_name: {count, avg_return, win_rate, sharpe, max_drawdown}}}. 
Groups: technical, fundamental, news_macro, social_alternative, risk_stability, institutional. 
Quintiles: top_20pct, q2, q3, q4, bottom_20pct';
