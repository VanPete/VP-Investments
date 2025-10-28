'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScoreBucketChart } from '@/components/analytics/ScoreBucketChart';
import { CorrelationHeatmap } from '@/components/analytics/CorrelationHeatmap';
import { BacktestChart } from '@/components/analytics/BacktestChart';
import { supabase } from '@/lib/supabase';
import { TrendingUp, Activity, BarChart3 } from 'lucide-react';

interface AnalyticsData {
  score_bucket_performance: Record<string, unknown> | null;
  factor_correlations: Record<string, unknown> | null;
  factor_contributions: Record<string, unknown> | null;
  group_performance: Record<string, unknown> | null;
  backtest_cumulative_returns: Record<string, unknown> | null;
}

interface AnalyticsTabProps {
  loading?: boolean;
}

export function AnalyticsTab({ loading: parentLoading }: AnalyticsTabProps) {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedInterval, setSelectedInterval] = useState<string>('7d');

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch latest analytics record
      const { data, error: fetchError } = await supabase
        .from('analytics')
        .select('score_bucket_performance, factor_correlations, factor_contributions, group_performance, backtest_cumulative_returns')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (fetchError) {
        throw fetchError;
      }

      if (!data) {
        throw new Error('No analytics data found');
      }

      setAnalyticsData(data);
    } catch (err) {
      console.error('Error fetching analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  };

  if (loading || parentLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6">
          <div className="flex items-center space-x-3">
            <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
            <p className="text-gray-600">Loading analytics data...</p>
          </div>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6">
          <p className="text-red-600">Error loading analytics: {error}</p>
          <button 
            onClick={fetchAnalytics}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </Card>
      </div>
    );
  }

  if (!analyticsData) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6">
          <p className="text-gray-600">No analytics data available. Run the pipeline to generate analytics.</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Interval Selector */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Analytics Dashboard</h3>
            <p className="text-sm text-gray-600">
              Comprehensive analysis of signal performance and factor relationships
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Time Interval:</span>
            <select
              value={selectedInterval}
              onChange={(e) => setSelectedInterval(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="1d">1 Day</option>
              <option value="3d">3 Days</option>
              <option value="7d">7 Days</option>
              <option value="10d">10 Days</option>
              <option value="14d">14 Days</option>
              <option value="30d">30 Days</option>
              <option value="90d">90 Days</option>
            </select>
          </div>
        </div>
      </Card>

      <Tabs defaultValue="score-buckets" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="score-buckets" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Score Analysis
          </TabsTrigger>
          <TabsTrigger value="correlations" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Factor Correlations
          </TabsTrigger>
          <TabsTrigger value="backtest" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Backtest Results
          </TabsTrigger>
        </TabsList>

        <TabsContent value="score-buckets" className="mt-6">
          <Card className="p-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold mb-2">Score Bucket Performance</h3>
              <p className="text-sm text-gray-600">
                Validates the hypothesis: <strong>Higher scores should produce higher returns</strong>
              </p>
            </div>
            <ScoreBucketChart 
              data={analyticsData.score_bucket_performance as never}
              interval={selectedInterval}
            />
          </Card>
        </TabsContent>

        <TabsContent value="correlations" className="mt-6">
          <Card className="p-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold mb-2">Factor Group Correlations</h3>
              <p className="text-sm text-gray-600">
                Understand how different factor groups relate to each other
              </p>
            </div>
            <CorrelationHeatmap data={analyticsData.factor_correlations as never} />
          </Card>
        </TabsContent>

        <TabsContent value="backtest" className="mt-6">
          <Card className="p-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold mb-2">Strategy Backtest</h3>
              <p className="text-sm text-gray-600">
                VP Strategy performance vs market benchmarks (SPY & QQQ)
              </p>
            </div>
            <BacktestChart data={analyticsData.backtest_cumulative_returns as never} />
          </Card>
        </TabsContent>
      </Tabs>

      {/* Additional Insights Section */}
      <Card className="p-6 bg-blue-50 dark:bg-blue-900/10">
        <h4 className="font-semibold mb-3 flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-600" />
          Key Insights
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <strong className="text-blue-700 dark:text-blue-400">Score Validation:</strong>
            <p className="text-gray-700 dark:text-gray-300 mt-1">
              The Score Bucket Analysis shows whether higher-scored signals actually outperform lower-scored ones across different time horizons.
            </p>
          </div>
          <div>
            <strong className="text-blue-700 dark:text-blue-400">Factor Relationships:</strong>
            <p className="text-gray-700 dark:text-gray-300 mt-1">
              The Correlation Heatmap reveals which factor groups move together (blue) or opposite (red), helping identify diversification opportunities.
            </p>
          </div>
          <div>
            <strong className="text-blue-700 dark:text-blue-400">Strategy Performance:</strong>
            <p className="text-gray-700 dark:text-gray-300 mt-1">
              The Backtest shows cumulative returns of the VP strategy compared to S&P 500 (SPY) and Nasdaq (QQQ) benchmarks.
            </p>
          </div>
          <div>
            <strong className="text-blue-700 dark:text-blue-400">Risk-Adjusted Returns:</strong>
            <p className="text-gray-700 dark:text-gray-300 mt-1">
              Sharpe ratio and max drawdown metrics help evaluate whether excess returns justify the additional risk taken.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
