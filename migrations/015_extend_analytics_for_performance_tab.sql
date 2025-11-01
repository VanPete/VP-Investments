-- ============================================================================
-- Migration 015: Streamline Analytics Table for Performance + Analytics Tabs
-- ============================================================================
-- Date: 2025-10-31 (REVISED)
-- Purpose: Add spec-required columns, simplify from period-based to run-based
-- Strategy: Column-only additions, no drops (backwards compatible)
-- Impact: Enables per-run analytics tracking with efficient storage
-- 
-- Key Changes:
-- - Add run_id for run-based analytics (replaces period_type duplication)
-- - Add 19 new columns for Performance + Analytics features per spec
-- - Ignore existing bloat columns (49 unused horizon columns remain for compatibility)
-- - Total storage: ~102 KB per run (mostly signal_correlations matrix)
-- ============================================================================

-- ============================================================================
-- PART 1: Link Analytics to Signal Runs (Run-Based, Not Period-Based)
-- ============================================================================
-- STRATEGIC SHIFT: From period-based (4 rows per analysis) to run-based (1 row per pipeline run)
--
-- OLD APPROACH (current):
--   - Multiple rows per analysis with different period_type values
--   - UPSERT key: (period_type, period_start, period_end)
--   - Result: 4x data duplication for daily/weekly/monthly/all_time
--
-- NEW APPROACH (this migration):
--   - Single row per pipeline run
--   - UPSERT key: run_id (unique)
--   - Result: Efficient storage, no duplication
--
-- NOTE: We are NOT dropping period_* columns (backwards compatible)
--       Phase 7 code will transition to use run_id instead

ALTER TABLE public.analytics
  ADD COLUMN IF NOT EXISTS run_id uuid;

-- Add foreign key constraint to link analytics to pipeline runs
ALTER TABLE public.analytics
  ADD CONSTRAINT analytics_run_id_fkey 
    FOREIGN KEY (run_id) REFERENCES public.signal_runs(id) 
    ON DELETE CASCADE;

-- Make run_id unique to enable UPSERT pattern (one analytics row per run)
-- This prevents duplicate analytics for the same run
ALTER TABLE public.analytics
  ADD CONSTRAINT analytics_run_id_unique UNIQUE (run_id);

-- Add index for fast lookups by run_id
CREATE INDEX IF NOT EXISTS idx_analytics_run_id ON public.analytics(run_id);

COMMENT ON COLUMN public.analytics.run_id IS 'Links analytics to specific pipeline run (one row per run, replaces period_type approach)';

-- ============================================================================
-- PART 2: Predictive Strength Metrics
-- ============================================================================
-- Add columns for tracking predictive power and model performance

ALTER TABLE public.analytics
  -- Rolling RankIC (Information Coefficient) time series
  ADD COLUMN IF NOT EXISTS ic_series jsonb,
  
  -- RankIC statistics
  ADD COLUMN IF NOT EXISTS ic_mean numeric,
  ADD COLUMN IF NOT EXISTS ic_std numeric,
  
  -- Hit rate: percentage of top decile predictions that outperformed
  ADD COLUMN IF NOT EXISTS hit_rate_top_decile numeric,
  
  -- Profit factor: gross profits / gross losses
  ADD COLUMN IF NOT EXISTS profit_factor numeric,
  
  -- Win/loss ratio: average win size / average loss size
  ADD COLUMN IF NOT EXISTS win_loss_ratio numeric;

COMMENT ON COLUMN public.analytics.ic_series IS '[{"date":"YYYY-MM-DD", "ic":<numeric>}, ...] - Rolling RankIC time series';
COMMENT ON COLUMN public.analytics.ic_mean IS 'Mean of RankIC series (measures average predictive strength)';
COMMENT ON COLUMN public.analytics.ic_std IS 'Standard deviation of RankIC series (measures consistency)';
COMMENT ON COLUMN public.analytics.hit_rate_top_decile IS 'Fraction of top 10% signals that outperformed (0..1)';
COMMENT ON COLUMN public.analytics.profit_factor IS 'Ratio of gross profits to gross losses (>1 is profitable)';
COMMENT ON COLUMN public.analytics.win_loss_ratio IS 'Average winning trade size / average losing trade size';

-- ============================================================================
-- PART 3: Global Performance Summary (Benchmarked)
-- ============================================================================
-- Add risk-adjusted performance metrics vs SPY and QQQ benchmarks

ALTER TABLE public.analytics
  -- Already have: sharpe_ratio_* columns for each horizon
  -- Adding: additional risk-adjusted metrics
  
  ADD COLUMN IF NOT EXISTS cagr numeric,
  ADD COLUMN IF NOT EXISTS volatility numeric,
  ADD COLUMN IF NOT EXISTS sortino_ratio numeric,
  ADD COLUMN IF NOT EXISTS calmar_ratio numeric,
  
  -- SPY benchmark comparison
  ADD COLUMN IF NOT EXISTS alpha_vs_spy numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_spy numeric,
  
  -- QQQ benchmark comparison
  ADD COLUMN IF NOT EXISTS alpha_vs_qqq numeric,
  ADD COLUMN IF NOT EXISTS beta_vs_qqq numeric;

COMMENT ON COLUMN public.analytics.cagr IS 'Compound Annual Growth Rate (annualized return)';
COMMENT ON COLUMN public.analytics.volatility IS 'Annualized volatility (standard deviation of returns)';
COMMENT ON COLUMN public.analytics.sortino_ratio IS 'Sortino ratio (return / downside deviation)';
COMMENT ON COLUMN public.analytics.calmar_ratio IS 'Calmar ratio (CAGR / max drawdown)';
COMMENT ON COLUMN public.analytics.alpha_vs_spy IS 'Alpha vs SPY benchmark (excess return)';
COMMENT ON COLUMN public.analytics.beta_vs_spy IS 'Beta vs SPY benchmark (systematic risk)';
COMMENT ON COLUMN public.analytics.alpha_vs_qqq IS 'Alpha vs QQQ benchmark (excess return)';
COMMENT ON COLUMN public.analytics.beta_vs_qqq IS 'Beta vs QQQ benchmark (systematic risk)';

-- ============================================================================
-- PART 4: Backtest Enhancements
-- ============================================================================
-- Add time series data for charting and analysis

ALTER TABLE public.analytics
  -- Rolling 30-day Sharpe ratio time series
  ADD COLUMN IF NOT EXISTS rolling_sharpe_30d jsonb,
  
  -- Correlation with benchmark indices
  ADD COLUMN IF NOT EXISTS benchmark_correlations jsonb;

COMMENT ON COLUMN public.analytics.rolling_sharpe_30d IS '[{"date":"YYYY-MM-DD", "sharpe":<numeric>}, ...] - 30-day rolling Sharpe ratio';
COMMENT ON COLUMN public.analytics.benchmark_correlations IS '{"SPY": 0.68, "QQQ": 0.57} - Correlation coefficients with benchmarks';

-- ============================================================================
-- PART 5: Signal-Level Correlations (158×158 Matrix)
-- ============================================================================
-- Add storage for signal-to-signal correlation matrix and top pairs

ALTER TABLE public.analytics
  -- Full correlation matrix (upper triangle only, ~12,403 pairs)
  ADD COLUMN IF NOT EXISTS signal_correlations jsonb,
  
  -- Precomputed top 20 most positively correlated pairs
  ADD COLUMN IF NOT EXISTS top_positive_pairs jsonb,
  
  -- Precomputed top 20 most negatively correlated pairs
  ADD COLUMN IF NOT EXISTS top_negative_pairs jsonb;

COMMENT ON COLUMN public.analytics.signal_correlations IS '[{"i":"RSI_14", "j":"MACD", "r":0.42, "n":1284}, ...] - Pairwise signal correlations';
COMMENT ON COLUMN public.analytics.top_positive_pairs IS '[{"i":"...", "j":"...", "r":...}, ...] - Top 20 positively correlated signal pairs (r>0)';
COMMENT ON COLUMN public.analytics.top_negative_pairs IS '[{"i":"...", "j":"...", "r":...}, ...] - Top 20 negatively correlated signal pairs (r<0)';

-- ============================================================================
-- PART 6: Update Existing Column Comments for Clarity
-- ============================================================================
-- Add documentation for existing columns that will be used differently

COMMENT ON COLUMN public.analytics.factor_contributions IS '{"technical":{"alpha_pct":0.32,"vol_pct":0.18}, ...} - Normalized group contributions (fractions 0..1)';
COMMENT ON COLUMN public.analytics.score_bucket_performance IS '{"top10":{"avg_return":0.15,"win_rate":0.72,"count":10}, ...} - Performance by score bucket';
COMMENT ON COLUMN public.analytics.factor_correlations IS 'Group-level correlation matrix (6×6 for six signal groups)';
COMMENT ON COLUMN public.analytics.backtest_cumulative_returns IS '[{"date":"YYYY-MM-DD","vp":1.15,"spy":1.08,"qqq":1.12}, ...] - Cumulative return series';

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================
-- Run this to verify all new columns were added successfully

DO $$ 
DECLARE
  missing_columns text[];
  expected_columns text[] := ARRAY[
    'run_id',
    'ic_series',
    'ic_mean',
    'ic_std',
    'hit_rate_top_decile',
    'profit_factor',
    'win_loss_ratio',
    'cagr',
    'volatility',
    'sortino_ratio',
    'calmar_ratio',
    'alpha_vs_spy',
    'beta_vs_spy',
    'alpha_vs_qqq',
    'beta_vs_qqq',
    'rolling_sharpe_30d',
    'benchmark_correlations',
    'signal_correlations',
    'top_positive_pairs',
    'top_negative_pairs'
  ];
  col text;
BEGIN
  missing_columns := ARRAY[]::text[];
  
  FOREACH col IN ARRAY expected_columns
  LOOP
    IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns 
      WHERE table_schema = 'public' 
        AND table_name = 'analytics' 
        AND column_name = col
    ) THEN
      missing_columns := array_append(missing_columns, col);
    END IF;
  END LOOP;
  
  IF array_length(missing_columns, 1) > 0 THEN
    RAISE EXCEPTION 'Migration 015 FAILED: Missing columns: %', missing_columns;
  ELSE
    RAISE NOTICE 'Migration 015 SUCCESS: All 20 columns added to analytics table';
  END IF;
END $$;

-- ============================================================================
-- ROLLBACK INSTRUCTIONS (if needed)
-- ============================================================================
-- To rollback this migration, run:
/*
ALTER TABLE public.analytics
  DROP CONSTRAINT IF EXISTS analytics_run_id_unique,
  DROP CONSTRAINT IF EXISTS analytics_run_id_fkey,
  DROP COLUMN IF EXISTS run_id,
  DROP COLUMN IF EXISTS ic_series,
  DROP COLUMN IF EXISTS ic_mean,
  DROP COLUMN IF EXISTS ic_std,
  DROP COLUMN IF EXISTS hit_rate_top_decile,
  DROP COLUMN IF EXISTS profit_factor,
  DROP COLUMN IF EXISTS win_loss_ratio,
  DROP COLUMN IF EXISTS cagr,
  DROP COLUMN IF EXISTS volatility,
  DROP COLUMN IF EXISTS sortino_ratio,
  DROP COLUMN IF EXISTS calmar_ratio,
  DROP COLUMN IF EXISTS alpha_vs_spy,
  DROP COLUMN IF EXISTS beta_vs_spy,
  DROP COLUMN IF EXISTS alpha_vs_qqq,
  DROP COLUMN IF EXISTS beta_vs_qqq,
  DROP COLUMN IF EXISTS rolling_sharpe_30d,
  DROP COLUMN IF EXISTS benchmark_correlations,
  DROP COLUMN IF EXISTS signal_correlations,
  DROP COLUMN IF EXISTS top_positive_pairs,
  DROP COLUMN IF EXISTS top_negative_pairs;

DROP INDEX IF EXISTS idx_analytics_run_id;
*/
