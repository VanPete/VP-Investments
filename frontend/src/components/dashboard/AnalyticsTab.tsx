'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScoreBucketChart } from '@/components/analytics/ScoreBucketChart';
import { CorrelationHeatmap } from '@/components/analytics/CorrelationHeatmap';
import { BacktestChart } from '@/components/analytics/BacktestChart';
import { supabase } from '@/lib/supabase';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';

interface SectorData {
  avg_return: number;
  count: number;
  win_rate: number;
}

interface AnalyticsData {
  // JSONB columns from Phase 7
  score_bucket_performance: Record<string, unknown> | null;
  factor_correlations: Record<string, unknown> | null;
  factor_contributions: Record<string, unknown> | null;
  group_performance: Record<string, unknown> | null;
  backtest_cumulative_returns: Record<string, unknown> | null;
  
  // Aggregate metrics
  total_signals: number;
  avg_overall_score: number;
  
  // Win rates
  win_rate_1d: number | null;
  win_rate_3d: number | null;
  win_rate_7d: number | null;
  win_rate_10d: number | null;
  win_rate_14d: number | null;
  win_rate_30d: number | null;
  win_rate_90d: number | null;
  
  // Sharpe ratios
  sharpe_ratio_1d: number | null;
  sharpe_ratio_3d: number | null;
  sharpe_ratio_7d: number | null;
  sharpe_ratio_10d: number | null;
  sharpe_ratio_14d: number | null;
  sharpe_ratio_30d: number | null;
  sharpe_ratio_90d: number | null;
  
  // Max drawdowns
  max_drawdown_1d: number | null;
  max_drawdown_3d: number | null;
  max_drawdown_7d: number | null;
  max_drawdown_10d: number | null;
  max_drawdown_14d: number | null;
  max_drawdown_30d: number | null;
  max_drawdown_90d: number | null;
  
  // Returns & alpha
  avg_return_1d: number | null;
  avg_return_30d: number | null;
  avg_alpha_1d: number | null;
  avg_alpha_30d: number | null;
  
  // Sector analysis
  top_sector: string | null;
  top_sector_avg_return: number | null;
  top_sector_count: number | null;
  worst_sector: string | null;
  worst_sector_avg_return: number | null;
  worst_sector_count: number | null;
  sector_performance: Record<string, SectorData> | null;
  
  // Signal group scores
  avg_technical_score: number | null;
  avg_fundamental_score: number | null;
  avg_news_macro_score: number | null;
  avg_social_alternative_score: number | null;
  avg_risk_stability_score: number | null;
  avg_institutional_score: number | null;
}

interface AnalyticsTabProps {
  loading?: boolean;
}

export function AnalyticsTab({ loading: parentLoading }: AnalyticsTabProps) {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedInterval, setSelectedInterval] = useState<string>('7d');
  const [selectedScoreBucket, setSelectedScoreBucket] = useState<string>('all');

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch latest analytics record with all columns
      const { data, error: fetchError } = await supabase
        .from('analytics')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (fetchError) {
        throw fetchError;
      }

      if (!data) {
        throw new Error('No analytics data found');
      }

      setAnalyticsData(data as AnalyticsData);
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

  const intervals = [
    { key: '1d', label: '1 Day' },
    { key: '3d', label: '3 Days' },
    { key: '7d', label: '1 Week' },
    { key: '14d', label: '2 Weeks' },
    { key: '30d', label: '1 Month' },
    { key: '90d', label: '3 Months' },
  ];

  // Filter analytics data based on selected score bucket
  const getFilteredMetrics = (bucket: string) => {
    if (!analyticsData || bucket === 'all') {
      return analyticsData;
    }

    // Map bucket selection to score_bucket_performance keys
    const bucketMap: Record<string, string> = {
      'top_10': 'strong_buy',  // > 0.75 is roughly top 10%
      '10_25': 'buy',          // 0.50-0.75
      '25_50': 'hold',         // -0.25 to 0.50
      '50_75': 'sell',         // -0.50 to -0.25
      'bottom_25': 'strong_sell' // < -0.50
    };

    const bucketKey = bucketMap[bucket];
    const bucketData = analyticsData.score_bucket_performance?.[bucketKey];

    if (!bucketData) {
      return analyticsData; // Fallback to all data
    }

    // Return modified analytics with bucket-specific metrics
    // Note: This is a simplified filter - full implementation would require
    // backend support for percentile-based buckets
    return {
      ...analyticsData,
      // Override metrics with bucket-specific data if available
      total_signals: bucketData.count || 0,
      // TODO: Map other metrics from bucket data
    };
  };

  const displayData = getFilteredMetrics(selectedScoreBucket);

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Signals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsData.total_signals || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analyticsData.avg_overall_score?.toFixed(3) || 'N/A'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Top Sector
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analyticsData.top_sector || 'N/A'}</div>
            {analyticsData.top_sector_count && (
              <p className="text-xs text-muted-foreground mt-1">
                {analyticsData.top_sector_count} signals
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              30d Avg Return
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">
                {analyticsData.avg_return_30d !== null && analyticsData.avg_return_30d !== undefined
                  ? `${(analyticsData.avg_return_30d * 100).toFixed(2)}%`
                  : 'N/A'}
              </span>
              {analyticsData.avg_return_30d !== null && analyticsData.avg_return_30d !== undefined && (
                analyticsData.avg_return_30d > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                )
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Metrics Grids */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Win Rates */}
        <Card>
          <CardHeader>
            <CardTitle>Win Rates</CardTitle>
            <CardDescription>
              Percentage of signals with positive returns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {intervals.map(({ key, label }) => {
                const winRate = analyticsData[`win_rate_${key}` as keyof AnalyticsData] as number | null;
                return (
                  <div key={key} className="space-y-1">
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="text-xl font-bold">
                      {winRate !== null ? `${winRate.toFixed(1)}%` : 'N/A'}
                    </p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Sharpe Ratios */}
        <Card>
          <CardHeader>
            <CardTitle>Sharpe Ratios</CardTitle>
            <CardDescription>
              Risk-adjusted returns (higher is better)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {intervals.map(({ key, label }) => {
                const sharpe = analyticsData[`sharpe_ratio_${key}` as keyof AnalyticsData] as number | null;
                return (
                  <div key={key} className="space-y-1">
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="text-xl font-bold">
                      {sharpe !== null ? sharpe.toFixed(2) : 'N/A'}
                    </p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Max Drawdowns */}
      <Card>
        <CardHeader>
          <CardTitle>Maximum Drawdowns</CardTitle>
          <CardDescription>
            Largest peak-to-trough decline (lower is better)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {intervals.map(({ key, label }) => {
              const drawdown = analyticsData[`max_drawdown_${key}` as keyof AnalyticsData] as number | null;
              return (
                <div key={key} className="space-y-1">
                  <p className="text-sm text-muted-foreground">{label}</p>
                  <p className="text-xl font-bold text-red-500">
                    {drawdown !== null ? `${(drawdown * 100).toFixed(2)}%` : 'N/A'}
                  </p>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Signal Group Scores */}
      <Card>
        <CardHeader>
          <CardTitle>Average Signal Group Scores</CardTitle>
          <CardDescription>
            Average normalized scores across all signal groups
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {[
              { key: 'technical', label: 'Technical', icon: Activity },
              { key: 'fundamental', label: 'Fundamental', icon: BarChart3 },
              { key: 'news_macro', label: 'News & Macro', icon: TrendingUp },
              { key: 'social_alternative', label: 'Social', icon: Activity },
              { key: 'risk_stability', label: 'Risk', icon: Activity },
              { key: 'institutional', label: 'Institutional', icon: BarChart3 },
            ].map(({ key, label, icon: Icon }) => {
              const score = analyticsData[`avg_${key}_score` as keyof AnalyticsData] as number | null;
              return (
                <div key={key} className="flex items-center gap-3 p-3 rounded-lg border">
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">{label}</p>
                    <p className="text-lg font-bold">
                      {score !== null ? score.toFixed(3) : 'N/A'}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Interval and Score Bucket Selectors */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Comprehensive analysis of signal performance and factor relationships
            </p>
            {selectedScoreBucket !== 'all' && (
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                <span className="font-medium">Active Filter:</span> {
                  selectedScoreBucket === 'top_10' ? 'Top 10% (Strong Buy)' :
                  selectedScoreBucket === '10_25' ? '10-25% (Buy)' :
                  selectedScoreBucket === '25_50' ? '25-50% (Hold)' :
                  selectedScoreBucket === '50_75' ? '50-75% (Sell)' :
                  'Bottom 25% (Strong Sell)'
                }
              </p>
            )}
          </div>
          <div className="flex items-center gap-4">
            {/* Score Bucket Filter */}
            <div className="flex items-center gap-2">
              <label htmlFor="score-bucket-selector" className="text-sm text-gray-600 dark:text-gray-400">Score Bucket:</label>
              <select
                id="score-bucket-selector"
                value={selectedScoreBucket}
                onChange={(e) => setSelectedScoreBucket(e.target.value)}
                className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Signals</option>
                <option value="top_10">Top 10%</option>
                <option value="10_25">10-25%</option>
                <option value="25_50">25-50%</option>
                <option value="50_75">50-75%</option>
                <option value="bottom_25">Bottom 25%</option>
              </select>
            </div>

            {/* Time Interval Selector */}
            <div className="flex items-center gap-2">
              <label htmlFor="interval-selector" className="text-sm text-gray-600 dark:text-gray-400">Time Interval:</label>
              <select
                id="interval-selector"
                value={selectedInterval}
                onChange={(e) => setSelectedInterval(e.target.value)}
                className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
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

      {/* Sector Performance */}
      {analyticsData.sector_performance && Object.keys(analyticsData.sector_performance).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Sector Performance</CardTitle>
            <CardDescription>
              Performance breakdown by sector
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(analyticsData.sector_performance).map(([sector, data]: [string, unknown]) => {
                const sectorData = data as SectorData;
                return (
                  <div key={sector} className="flex items-center justify-between p-3 rounded-lg border">
                    <span className="font-medium">{sector}</span>
                    <div className="text-right">
                      <p className="font-bold">
                        {sectorData.avg_return !== undefined
                          ? `${(sectorData.avg_return * 100).toFixed(2)}%`
                          : 'N/A'}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {sectorData.count || 0} signals • {sectorData.win_rate !== undefined ? `${sectorData.win_rate.toFixed(1)}%` : 'N/A'} win rate
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

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
