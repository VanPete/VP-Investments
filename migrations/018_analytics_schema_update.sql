-- Migration 018: Analytics Schema Update for Performance + Analytics Spec
-- Adds benchmark metrics, signal correlations, and removes unused columns
-- All additions are backwards compatible

-- 1. ADD new columns for benchmarking (SPY/QQQ)
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS alpha_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS alpha_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS rolling_sharpe_30d jsonb,
  ADD COLUMN IF NOT EXISTS benchmark_correlations jsonb;

-- 2. ADD new columns for signal-level correlations
ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS signal_correlations jsonb,
  ADD COLUMN IF NOT EXISTS top_positive_pairs jsonb,
  ADD COLUMN IF NOT EXISTS top_negative_pairs jsonb;

-- 3. DROP unused/duplicate columns (safe - these are NULL or unused)
ALTER TABLE public.analytics
  DROP COLUMN IF EXISTS top_factors,
  DROP COLUMN IF EXISTS group_performance;

-- 4. ADD helpful comments for JSON structure documentation
COMMENT ON COLUMN public.analytics.factor_contributions IS 
  'JSON structure: {group_key: {alpha_pct: 0.0-1.0, vol_pct: 0.0-1.0}}. Example: {"technical": {"alpha_pct": 0.32, "vol_pct": 0.18}}';

COMMENT ON COLUMN public.analytics.signal_correlations IS 
  'JSON structure: [{i: "signal_key", j: "signal_key", r: correlation, n: sample_size}]. Stores all signal-to-signal correlations.';

COMMENT ON COLUMN public.analytics.top_positive_pairs IS 
  'JSON structure: [{i: "signal_key", j: "signal_key", r: correlation}]. Top 50 most positively correlated signal pairs.';

COMMENT ON COLUMN public.analytics.top_negative_pairs IS 
  'JSON structure: [{i: "signal_key", j: "signal_key", r: correlation}]. Top 50 most negatively correlated signal pairs.';

COMMENT ON COLUMN public.analytics.benchmark_correlations IS 
  'JSON structure: {"SPY": 0.68, "QQQ": 0.57}. Correlation of portfolio returns vs benchmark returns.';

COMMENT ON COLUMN public.analytics.rolling_sharpe_30d IS 
  'JSON structure: [{"date": "YYYY-MM-DD", "sharpe": numeric}]. 30-day rolling Sharpe ratio series.';

COMMENT ON COLUMN public.analytics.ic_series IS 
  'JSON structure: [{"date": "YYYY-MM-DD", "ic": numeric}]. Information Coefficient (rank correlation) time series.';

COMMENT ON COLUMN public.analytics.score_bucket_performance IS 
  'JSON structure: {bucket_id: {avg_return: numeric, win_rate: numeric, count: integer}}. Performance by score decile/quintile.';

-- 5. Verify the structure
DO $$
BEGIN
  RAISE NOTICE 'Analytics table updated successfully';
  RAISE NOTICE 'New columns added: alpha_vs_spy, beta_vs_spy, alpha_vs_qqq, beta_vs_qqq, rolling_sharpe_30d, benchmark_correlations';
  RAISE NOTICE 'New columns added: signal_correlations, top_positive_pairs, top_negative_pairs';
  RAISE NOTICE 'Removed columns: top_factors, group_performance';
END $$;
