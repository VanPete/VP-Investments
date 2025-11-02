/**
 * Analytics Tab - Full Analytics Dashboard in Tab
 * 
 * Features:
 * - Score Bucket filtering (Strong Buy/Buy/Hold/Sell/Strong Sell) with working metrics
 * - Time interval selection (1d/3d/7d/10d/14d/30d/90d/all-time)
 * - Performance Summary Cards (2 rows of 7 metrics)
 * - Predictive Strength (IC metrics)
 * - Backtest vs Benchmarks (above tabs)
 * - Tabbed sections: Score Buckets | Correlations
 * - Factor Contributions
 */

'use client';

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PerformanceSummaryCards } from '@/components/analytics/PerformanceSummaryCards';
import { ScoreBucketChart } from '@/components/analytics/ScoreBucketChart';
import { CorrelationHeatmap } from '@/components/analytics/CorrelationHeatmap';
import { PredictiveStrength } from '@/components/analytics/PredictiveStrength';
import { useAnalytics } from '@/hooks/useAnalytics';
import { TrendingUp, Activity, BarChart3, Filter, Clock, AlertCircle, Target } from 'lucide-react';

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

  // Filter analytics based on score bucket - extract bucket-specific metrics
  const filteredAnalytics = useMemo(() => {
    if (!analytics || selectedBucket === 'all') return analytics;

    const bucketPerf = analytics.score_bucket_performance?.[selectedBucket];
    
    if (!bucketPerf) return analytics;

    // Return analytics with ALL bucket-specific metrics replacing the global ones
    return {
      ...analytics,
      total_signals: bucketPerf.count || analytics.total_signals,
      avg_return: bucketPerf.avg_return || analytics.avg_return,
      win_rate: bucketPerf.win_rate || analytics.win_rate,
      sharpe_ratio: bucketPerf.sharpe_ratio || analytics.sharpe_ratio,
      sortino_ratio: bucketPerf.sortino_ratio || analytics.sortino_ratio,
      cagr: bucketPerf.cagr || analytics.cagr,
      volatility: bucketPerf.volatility || analytics.volatility,
      calmar_ratio: bucketPerf.calmar_ratio || analytics.calmar_ratio,
      max_drawdown: bucketPerf.max_drawdown || analytics.max_drawdown,
      profit_factor: bucketPerf.profit_factor || analytics.profit_factor,
      win_loss_ratio: bucketPerf.win_loss_ratio || analytics.win_loss_ratio,
    };
  }, [analytics, selectedBucket]);

  // Loading state
  if (isLoading || parentLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading analytics...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 max-w-md">
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
        </div>
      </div>
    );
  }

  // No data state
  if (!analytics) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 max-w-md text-center">
          <BarChart3 className="h-12 w-12 text-yellow-600 dark:text-yellow-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
            No Analytics Data
          </h3>
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            Run the pipeline to generate analytics data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Global Controls and Info */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              Analytics Dashboard
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Run: {analytics.run_id.substring(0, 8)} • {analytics.total_signals} signals • 
              Period: {analytics.period_type} • 
              Last updated: {new Date(analytics.created_at).toLocaleString()}
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Score Bucket Filter */}
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <select
                value={selectedBucket}
                onChange={(e) => setSelectedBucket(e.target.value)}
                className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Filter by score bucket"
              >
                {SCORE_BUCKETS.map((bucket) => (
                  <option key={bucket.value} value={bucket.value}>
                    {bucket.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Interval Selector */}
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-500" />
              <select
                value={selectedInterval}
                onChange={(e) => setSelectedInterval(e.target.value)}
                className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
        </div>
      </div>

      {/* Performance Summary Cards */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            Performance Metrics
            {selectedBucket !== 'all' && (
              <span className="text-sm font-normal text-blue-600 dark:text-blue-400">
                ({SCORE_BUCKETS.find(b => b.value === selectedBucket)?.label})
              </span>
            )}
          </h3>
          {selectedBucket !== 'all' && (
            <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 px-3 py-1 rounded-md border border-blue-200 dark:border-blue-800">
              Showing metrics for selected bucket only
            </span>
          )}
        </div>
        {filteredAnalytics && <PerformanceSummaryCards analytics={filteredAnalytics} />}
      </section>

      {/* Predictive Strength */}
      {filteredAnalytics?.ic_series && filteredAnalytics.ic_series.length > 0 && (
        <section>
          <PredictiveStrength analytics={filteredAnalytics} />
        </section>
      )}

      {/* Performance Summary - Key Metrics */}
      {analytics.backtest_cumulative_returns && (
        <section>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            Strategy Performance Summary
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
              <div className="text-center">
                <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2 font-semibold uppercase tracking-wide">VP Total Return</span>
                <span className="text-3xl font-bold text-green-600 dark:text-green-400">
                  {(analytics.backtest_cumulative_returns.summary.vp_total_return * 100).toFixed(2)}%
                </span>
              </div>
              <div className="text-center">
                <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2 font-semibold uppercase tracking-wide">SPY Total Return</span>
                <span className="text-3xl font-bold text-gray-600 dark:text-gray-400">
                  {(analytics.backtest_cumulative_returns.summary.spy_total_return * 100).toFixed(2)}%
                </span>
              </div>
              <div className="text-center">
                <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2 font-semibold uppercase tracking-wide">Sharpe Ratio</span>
                <span className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  {analytics.backtest_cumulative_returns.summary.vp_sharpe.toFixed(2)}
                </span>
              </div>
              <div className="text-center">
                <span className="text-xs text-gray-500 dark:text-gray-400 block mb-2 font-semibold uppercase tracking-wide">Win Rate</span>
                <span className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                  {(analytics.backtest_cumulative_returns.summary.vp_win_rate * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Tabbed Sections: Score Buckets | Correlations */}
      <Tabs defaultValue="score-buckets" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="score-buckets" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Score Buckets
          </TabsTrigger>
          <TabsTrigger value="correlations" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Correlations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="score-buckets" className="mt-4">
          <section>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
              <Target className="h-5 w-5 text-green-600" />
              Score Bucket Performance
            </h3>
            {analytics.score_bucket_performance ? (
              <ScoreBucketChart data={analytics.score_bucket_performance} />
            ) : (
              <Card className="p-6">
                <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
                  No score bucket data available
                </div>
              </Card>
            )}
          </section>
        </TabsContent>

        <TabsContent value="correlations" className="mt-4">
          <section>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
              <Activity className="h-5 w-5 text-orange-600" />
              Factor Correlations
            </h3>
            {analytics.factor_correlations && 'group_correlations' in analytics.factor_correlations ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Factor Correlation Matrix</CardTitle>
                  <CardDescription className="text-sm">
                    Correlation between 6 factor groups
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <CorrelationHeatmap data={analytics.factor_correlations as { 
                    group_correlations: { matrix: number[][], labels: string[] },
                    top_positive_pairs?: Array<{ factor1: string, factor2: string, correlation: number }>,
                    top_negative_pairs?: Array<{ factor1: string, factor2: string, correlation: number }>
                  }} />
                </CardContent>
              </Card>
            ) : (
              <Card className="p-6">
                <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-600">
                  No correlation data available
                </div>
              </Card>
            )}
          </section>
        </TabsContent>
      </Tabs>

      {/* Factor Contributions */}
      {analytics.factor_contributions && (
        <section>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Factor Group Contributions
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
              {Object.entries(analytics.factor_contributions).map(([group, contrib]) => (
                <div key={group} className="border-l-4 border-blue-500 pl-3 py-2">
                  <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1 font-semibold">
                    {group.replace(/_/g, ' ')}
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-600 dark:text-gray-300">Alpha:</span>
                      <span className="text-sm font-bold text-green-600 dark:text-green-400">
                        {(contrib.alpha_pct * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-600 dark:text-gray-300">Vol:</span>
                      <span className="text-sm font-bold text-orange-600 dark:text-orange-400">
                        {(contrib.vol_pct * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
