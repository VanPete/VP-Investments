/**
 * useAnalytics Hook
 * 
 * Fetches global analytics data directly from Supabase analytics table.
 * Direct database access with TanStack Query caching.
 * 
 * Features:
 * - Fetch latest analytics or filter by run_id/period_type
 * - Automatic caching and refetch management
 * - TypeScript type safety
 */

import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/lib/supabase';

// Type definitions
export interface ScoreBucketMetrics {
  threshold: string;
  count: number;
  avg_return: number;     // Stored as percentage (2.91 = 2.91%)
  win_rate: number;       // Stored as decimal (0.5517 = 55.17%) - convert with * 100
  max: number;
  min: number;
  // Risk-adjusted metrics (raw ratios)
  sharpe_ratio: number;
  sortino_ratio: number;
  // Performance metrics
  volatility: number;     // Stored as percentage (95.35 = 95.35%)
  cagr: number;           // Stored as percentage (-95.98 = -95.98%)
  max_drawdown: number;   // Stored as percentage
  calmar_ratio: number;   // Raw ratio
  // Win/Loss metrics (raw ratios)
  win_loss_ratio: number;
  profit_factor: number;
}

export interface FactorGroupContribution {
  alpha_pct: number;
  vol_pct: number;
}

export interface AnalyticsData {
  // Metadata
  run_id: string;
  created_at: string;
  period_type: string;  // The interval: 1d, 3d, 7d, 10d, 14d, 30d, 90d, all_time
  period_start: string;
  period_end: string;
  total_signals: number;
  signals_analyzed: number;
  performance_records_used: number;
  
  // Basic Scores (raw decimal scores, typically -1 to 1)
  avg_overall_score: number | null;
  avg_technical_score: number | null;
  avg_fundamental_score: number | null;
  avg_news_macro_score: number | null;
  avg_social_alternative_score: number | null;
  avg_risk_stability_score: number | null;
  avg_institutional_score: number | null;
  
  // Global Performance Metrics
  // FORMAT: Percentage (32.65 = 32.65%)
  win_rate: number | null;
  cagr: number | null;
  volatility: number | null;
  alpha_vs_spy: number | null;      // Raw percentage points (-0.32 = -0.32% excess)
  alpha_vs_qqq: number | null;      // Raw percentage points
  max_drawdown: number | null;
  
  // FORMAT: Decimal fraction (0.05 = 5%)
  avg_return: number | null;        // Use formatDecimalAsPercent
  avg_alpha: number | null;         // Use formatDecimalAsPercent
  
  // FORMAT: Raw ratio values
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  calmar_ratio: number | null;
  beta_vs_spy: number | null;
  beta_vs_qqq: number | null;
  
  // Predictive Strength
  ic_series: Array<{ date: string; ic: number }> | null;
  ic_mean: number | null;
  ic_std: number | null;
  hit_rate_top_decile: number | null;  // Decimal fraction (0.65 = 65%)
  profit_factor: number | null;        // Raw ratio
  win_loss_ratio: number | null;       // Raw ratio
  
  // Benchmark Correlations
  benchmark_correlations: { SPY: number; QQQ: number } | null;
  
  // Score Buckets (interval-specific)
  score_bucket_performance: Record<string, ScoreBucketMetrics> | null;
  
  // Group Performance
  group_performance: Record<string, unknown> | null;
  
  // Correlations
  factor_correlations: Record<string, unknown> | null;
  
  // Contributions & Performance
  factor_contributions: Record<string, FactorGroupContribution> | null;
  factor_return_correlations: Record<string, unknown> | null;
  
  // Backtest Returns (interval-specific)
  backtest_cumulative_returns: {
    start_date: string;
    end_date: string;
    period_returns: Array<{ date: string; vp_strategy: number; spy: number; qqq: number }>;  // Note: key is period_returns, not daily_returns
    summary: {
      vp_total_return: number;
      spy_total_return: number;
      qqq_total_return: number;
      vp_sharpe: number;
      vp_max_drawdown: number;
      vp_win_rate: number;
    };
  } | null;
  
  // Sector Performance (interval-specific)
  top_sector: string | null;
  top_sector_avg_return: number | null;
  top_sector_count: number | null;
  worst_sector: string | null;
  worst_sector_avg_return: number | null;
  worst_sector_count: number | null;
  
  // Top Factors
  top_factors: Record<string, unknown> | null;
}

export interface UseAnalyticsOptions {
  runId?: string;  // Optional: fetch specific run, default to latest
  periodType?: string;  // Optional: fetch specific period_type (1d, 3d, 7d, etc.), default to latest
  enabled?: boolean;
}

/**
 * Fetch global analytics data from Supabase
 */
export function useAnalytics(options: UseAnalyticsOptions = {}) {
  const { runId, periodType, enabled = true } = options;
  
  return useQuery({
    queryKey: ['analytics', runId || 'latest', periodType || 'latest'],
    queryFn: async (): Promise<AnalyticsData | null> => {
      let query = supabase.from('analytics').select('*');
      
      // Filter by run_id if provided
      if (runId) {
        query = query.eq('run_id', runId);
      }
      
      // Filter by period_type if provided
      if (periodType) {
        query = query.eq('period_type', periodType);
      }
      
      // Order by created_at and get the latest
      const { data, error } = await query
        .order('created_at', { ascending: false })
        .limit(1)
        .single();
      
      if (error) {
        console.error('Error fetching analytics:', error);
        throw error;
      }
      
      return data as unknown as AnalyticsData;
    },
    enabled,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });
}

// Re-export formatting utilities for convenience
export {
  formatPercentage,
  formatDecimalAsPercent,
  formatRatio,
  formatCurrency,
  formatCount,
  formatNumber,
  getValueColor,
  getValueBgColor,
  formatSharpeWithRating,
  SCORE_BUCKET_COLORS,
  SCORE_BUCKET_LABELS,
  type ScoreBucket,
} from '@/lib/analytics-formatting';
