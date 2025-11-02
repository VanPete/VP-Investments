/**
 * Analytics Tab - Integrated Dashboard Analytics
 * 
 * Displays comprehensive analytics with:
 * - Score Interpretation filtering (Strong Buy/Buy/Hold/Sell/Strong Sell)
 * - Time interval selection (1d/3d/7d/10d/14d/30d/90d/all-time)
 * - Performance Summary Cards (2 rows of 7 metrics each)
 * - Predictive Strength (IC metrics)
 * - Score Bucket Performance (improved styling)
 * - Backtest vs Benchmarks
 * - Factor Correlations
 */

'use client';

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScoreBucketChart } from '@/components/analytics/ScoreBucketChart';
import { CorrelationHeatmap } from '@/components/analytics/CorrelationHeatmap';
import { BacktestChart } from '@/components/analytics/BacktestChart';
import { PredictiveStrength } from '@/components/analytics/PredictiveStrength';
import { useAnalytics } from '@/hooks/useAnalytics';
import { TrendingUp, Activity, BarChart3, Filter, Clock, AlertCircle } from 'lucide-react';

interface AnalyticsTabProps {
  loading?: boolean;
}

// Score Interpretation Categories
const SCORE_BUCKETS = [
  { value: 'all', label: 'All Signals', min: -Infinity, max: Infinity },
  { value: 'strong_buy', label: 'Strong Buy (> 0.75)', min: 0.75, max: Infinity },
  { value: 'buy', label: 'Buy (0.50 to 0.75)', min: 0.50, max: 0.75 },
  { value: 'hold', label: 'Hold (-0.25 to 0.50)', min: -0.25, max: 0.50 },
  { value: 'sell', label: 'Sell (-0.50 to -0.25)', min: -0.50, max: -0.25 },
  { value: 'strong_sell', label: 'Strong Sell (< -0.50)', min: -Infinity, max: -0.50 },
];

const TIME_INTERVALS = [
  { value: '1d', label: '1 Day' },
  { value: '3d', label: '3 Days' },
  { value: '7d', label: '1 Week' },
  { value: '10d', label: '10 Days' },
  { value: '14d', label: '2 Weeks' },
  { value: '30d', label: '1 Month' },
  { value: '90d', label: '3 Months' },
  { value: 'all_time', label: 'All Time' },
];

export function AnalyticsTab({ loading: parentLoading }: AnalyticsTabProps) {
  const [selectedBucket, setSelectedBucket] = useState('all');
  const [selectedInterval, setSelectedInterval] = useState('7d');
  
  // Fetch analytics for the selected interval
  const { data: analytics, isLoading, error } = useAnalytics({ 
    periodType: selectedInterval 
  });

  // Filter analytics based on score bucket
  const filteredAnalytics = useMemo(() => {
    if (!analytics || selectedBucket === 'all') return analytics;

    const bucket = SCORE_BUCKETS.find(b => b.value === selectedBucket);
    if (!bucket) return analytics;

    // Get bucket-specific performance data
    const bucketPerf = analytics.score_bucket_performance?.[selectedBucket];
    
    if (!bucketPerf) return analytics;

    // Return analytics with bucket-filtered metrics
    return {
      ...analytics,
      // Override with bucket-specific data where available
      total_signals: (bucketPerf as any).count || analytics.total_signals,
    };
  }, [analytics, selectedBucket]);

  // Loading state
  if (isLoading || parentLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6">
          <div className="flex items-center space-x-3">
            <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
            <p className="text-gray-600 dark:text-gray-400">Loading analytics...</p>
          </div>
        </Card>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6 border-red-200 dark:border-red-800">
          <div className="flex items-center gap-3 mb-3">
            <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
            <h3 className="text-lg font-semibold text-red-900 dark:text-red-100">
              Error Loading Analytics
            </h3>
          </div>
          <p className="text-sm text-red-800 dark:text-red-200 mb-4">
            {error instanceof Error ? error.message : 'Failed to fetch analytics data'}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors"
          >
            Retry
          </button>
        </Card>
      </div>
    );
  }

  // No data state
  if (!analytics) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800">
          <BarChart3 className="h-12 w-12 text-yellow-600 dark:text-yellow-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-100 mb-2 text-center">
            No Analytics Data
          </h3>
          <p className="text-sm text-yellow-800 dark:text-yellow-200 text-center">
            Run the pipeline to generate analytics data.
          </p>
        </Card>
      </div>
    );
  }

  // TypeScript guard: filteredAnalytics is guaranteed to be defined here
  if (!filteredAnalytics) {
    return null; // This should never happen but satisfies TypeScript
  }

  return (
    <div className="space-y-4">
      {/* Compact Global Controls */}
      <div className="flex items-center justify-between gap-4 px-4 py-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-4">
          {/* Score Filter */}
          <div className="flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-gray-500" />
            <select
              value={selectedBucket}
              onChange={(e) => setSelectedBucket(e.target.value)}
              className="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
              aria-label="Filter by score bucket"
            >
              {SCORE_BUCKETS.map((bucket) => (
                <option key={bucket.value} value={bucket.value}>
                  {bucket.label}
                </option>
              ))}
            </select>
          </div>

          {/* Period Filter */}
          <div className="flex items-center gap-2">
            <Clock className="h-3.5 w-3.5 text-gray-500" />
            <select
              value={selectedInterval}
              onChange={(e) => setSelectedInterval(e.target.value)}
              className="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
              aria-label="Select time interval"
            >
              {TIME_INTERVALS.map((interval) => (
                <option key={interval.value} value={interval.value}>
                  {interval.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Active Filter Badge */}
        {selectedBucket !== 'all' && (
          <div className="text-xs text-blue-600 dark:text-blue-400 font-medium">
            {SCORE_BUCKETS.find(b => b.value === selectedBucket)?.label} • {filteredAnalytics.total_signals || 0} signals
          </div>
        )}
      </div>

      {/* Performance Metrics - 2 Rows of 7 */}
      <div className="space-y-3">
        {/* Row 1: Win Rate, Sortino, CAGR, IC Mean, Profit Factor, β SPY, α SPY */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Win Rate</div>
            <div className={`text-lg font-bold ${filteredAnalytics.win_rate && filteredAnalytics.win_rate > 0.5 ? 'text-green-600 dark:text-green-400' : 'text-gray-900 dark:text-gray-100'}`}>
              {filteredAnalytics.win_rate != null ? `${(filteredAnalytics.win_rate * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Sortino</div>
            <div className={`text-lg font-bold ${filteredAnalytics.sortino_ratio && filteredAnalytics.sortino_ratio > 0 ? 'text-blue-600 dark:text-blue-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.sortino_ratio != null ? filteredAnalytics.sortino_ratio.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">CAGR</div>
            <div className={`text-lg font-bold ${filteredAnalytics.cagr && filteredAnalytics.cagr > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.cagr != null ? `${filteredAnalytics.cagr.toFixed(1)}%` : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">IC Mean</div>
            <div className={`text-lg font-bold ${filteredAnalytics.ic_mean && filteredAnalytics.ic_mean > 0 ? 'text-orange-600 dark:text-orange-400' : 'text-gray-700 dark:text-gray-300'}`}>
              {filteredAnalytics.ic_mean != null ? filteredAnalytics.ic_mean.toFixed(3) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Profit Factor</div>
            <div className={`text-lg font-bold ${filteredAnalytics.profit_factor && filteredAnalytics.profit_factor > 1 ? 'text-purple-600 dark:text-purple-400' : 'text-gray-700 dark:text-gray-300'}`}>
              {filteredAnalytics.profit_factor != null ? filteredAnalytics.profit_factor.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">β SPY</div>
            <div className="text-lg font-bold text-gray-700 dark:text-gray-300">
              {filteredAnalytics.beta_vs_spy != null ? filteredAnalytics.beta_vs_spy.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">α SPY</div>
            <div className={`text-lg font-bold ${filteredAnalytics.alpha_vs_spy && filteredAnalytics.alpha_vs_spy > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.alpha_vs_spy != null ? `${(filteredAnalytics.alpha_vs_spy * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </Card>
        </div>

        {/* Row 2: Sharpe, Calmar, Volatility, IC Std, Win/Loss, β QQQ, α QQQ */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Sharpe</div>
            <div className={`text-lg font-bold ${filteredAnalytics.sharpe_ratio && filteredAnalytics.sharpe_ratio > 0 ? 'text-blue-600 dark:text-blue-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.sharpe_ratio != null ? filteredAnalytics.sharpe_ratio.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Calmar</div>
            <div className={`text-lg font-bold ${filteredAnalytics.calmar_ratio && filteredAnalytics.calmar_ratio > 0 ? 'text-blue-600 dark:text-blue-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.calmar_ratio != null ? filteredAnalytics.calmar_ratio.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Volatility</div>
            <div className="text-lg font-bold text-gray-700 dark:text-gray-300">
              {filteredAnalytics.volatility != null ? `${filteredAnalytics.volatility.toFixed(1)}%` : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">IC Std</div>
            <div className="text-lg font-bold text-gray-700 dark:text-gray-300">
              {filteredAnalytics.ic_std != null ? filteredAnalytics.ic_std.toFixed(3) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">Win/Loss</div>
            <div className={`text-lg font-bold ${filteredAnalytics.win_loss_ratio && filteredAnalytics.win_loss_ratio > 1 ? 'text-indigo-600 dark:text-indigo-400' : 'text-gray-700 dark:text-gray-300'}`}>
              {filteredAnalytics.win_loss_ratio != null ? filteredAnalytics.win_loss_ratio.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">β QQQ</div>
            <div className="text-lg font-bold text-gray-700 dark:text-gray-300">
              {filteredAnalytics.beta_vs_qqq != null ? filteredAnalytics.beta_vs_qqq.toFixed(2) : 'N/A'}
            </div>
          </Card>

          <Card className="p-3 hover:shadow-md transition-shadow">
            <div className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider mb-1">α QQQ</div>
            <div className={`text-lg font-bold ${filteredAnalytics.alpha_vs_qqq && filteredAnalytics.alpha_vs_qqq > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {filteredAnalytics.alpha_vs_qqq != null ? `${(filteredAnalytics.alpha_vs_qqq * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </Card>
        </div>
      </div>

      {/* Predictive Strength */}
      {filteredAnalytics.ic_series && filteredAnalytics.ic_series.length > 0 && (
        <section>
          <PredictiveStrength analytics={filteredAnalytics} />
        </section>
      )}

      {/* Compact Tabbed Sections */}
      <Tabs defaultValue="score-buckets" className="w-full">
        <TabsList className="grid w-full grid-cols-3 h-auto">
          <TabsTrigger value="score-buckets" className="flex items-center gap-1.5 py-2 text-xs">
            <TrendingUp className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Score Buckets</span>
            <span className="sm:hidden">Scores</span>
          </TabsTrigger>
          <TabsTrigger value="correlations" className="flex items-center gap-1.5 py-2 text-xs">
            <Activity className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Correlations</span>
            <span className="sm:hidden">Corr</span>
          </TabsTrigger>
          <TabsTrigger value="backtest" className="flex items-center gap-1.5 py-2 text-xs">
            <BarChart3 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Backtest</span>
            <span className="sm:hidden">BT</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="score-buckets" className="mt-3">
          {filteredAnalytics.score_bucket_performance ? (
            <ScoreBucketChart data={filteredAnalytics.score_bucket_performance} />
          ) : (
            <Card className="p-6">
              <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
                No score bucket data available
              </div>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="correlations" className="mt-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Factor Group Correlations</CardTitle>
              <CardDescription className="text-xs">
                How factor groups relate to each other
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              {filteredAnalytics.factor_correlations ? (
                <CorrelationHeatmap data={filteredAnalytics.factor_correlations as any} />
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
                  No correlation data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="backtest" className="mt-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Strategy Backtest</CardTitle>
              <CardDescription className="text-xs">
                VP Strategy vs SPY & QQQ
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              {filteredAnalytics.backtest_cumulative_returns ? (
                <>
                  <BacktestChart data={filteredAnalytics.backtest_cumulative_returns as any} />
                  
                  {/* Backtest Summary Stats */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                    <div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 block mb-0.5">VP Total Return</span>
                      <span className="text-base font-bold text-green-600 dark:text-green-400">
                        {((filteredAnalytics.backtest_cumulative_returns as any).summary?.vp_total_return * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 block mb-0.5">SPY Total Return</span>
                      <span className="text-base font-bold text-gray-600 dark:text-gray-400">
                        {((filteredAnalytics.backtest_cumulative_returns as any).summary?.spy_total_return * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 block mb-0.5">Sharpe Ratio</span>
                      <span className="text-base font-bold text-blue-600 dark:text-blue-400">
                        {(filteredAnalytics.backtest_cumulative_returns as any).summary?.vp_sharpe.toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 block mb-0.5">Win Rate</span>
                      <span className="text-base font-bold text-purple-600 dark:text-purple-400">
                        {((filteredAnalytics.backtest_cumulative_returns as any).summary?.vp_win_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
                  No backtest data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Factor Contributions */}
      {filteredAnalytics.factor_contributions && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Factor Group Contributions</CardTitle>
            <CardDescription className="text-xs">
              Alpha and volatility by factor group
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {Object.entries(filteredAnalytics.factor_contributions).map(([group, contrib]) => {
                const contribution = contrib as { alpha_pct: number; vol_pct: number };
                return (
                  <div key={group} className="border-l-4 border-blue-500 dark:border-blue-400 pl-2 py-1.5 bg-gray-50 dark:bg-gray-800/50 rounded-r-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1 font-semibold">
                      {group.replace(/_/g, ' ')}
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 dark:text-gray-300">Alpha:</span>
                        <span className="text-xs font-bold text-green-600 dark:text-green-400">
                          {(contribution.alpha_pct * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 dark:text-gray-300">Vol:</span>
                        <span className="text-xs font-bold text-orange-600 dark:text-orange-400">
                          {(contribution.vol_pct * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
