'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { supabase } from '@/lib/supabase';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';

interface SectorData {
  avg_return: number;
  count: number;
  win_rate: number;
}

interface AnalyticsData {
  id: string;
  period_type: string;
  period_start: string;
  period_end: string;
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
  
  // Avg returns & alpha
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
  
  top_factors: Record<string, string[]> | null;
  
  created_at: string;
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAnalytics() {
      try {
        const { data, error } = await supabase
          .from('analytics')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(1)
          .single();

        if (error) throw error;
        setAnalytics(data);
      } catch (err) {
        console.error('Error fetching analytics:', err);
        setError(err instanceof Error ? err.message : 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    }

    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading analytics...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <p className="text-destructive mb-4">{error || 'No analytics data available'}</p>
            <p className="text-sm text-muted-foreground">
              Analytics will be generated after signals have performance data
            </p>
          </div>
        </div>
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

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Portfolio Analytics</h1>
        <p className="text-muted-foreground">
          Performance metrics and insights across all intervals
        </p>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Signals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analytics.total_signals}</div>
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
              {analytics.avg_overall_score?.toFixed(3) || 'N/A'}
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
            <div className="text-2xl font-bold">{analytics.top_sector || 'N/A'}</div>
            {analytics.top_sector_count && (
              <p className="text-xs text-muted-foreground mt-1">
                {analytics.top_sector_count} signals
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
                {analytics.avg_return_30d !== null
                  ? `${(analytics.avg_return_30d * 100).toFixed(2)}%`
                  : 'N/A'}
              </span>
              {analytics.avg_return_30d !== null && (
                analytics.avg_return_30d > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                )
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Win Rates */}
      <Card>
        <CardHeader>
          <CardTitle>Win Rates</CardTitle>
          <CardDescription>
            Percentage of signals with positive returns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {intervals.map(({ key, label }) => {
              const winRate = analytics[`win_rate_${key}` as keyof AnalyticsData] as number | null;
              return (
                <div key={key} className="space-y-1">
                  <p className="text-sm text-muted-foreground">{label}</p>
                  <p className="text-2xl font-bold">
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
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {intervals.map(({ key, label }) => {
              const sharpe = analytics[`sharpe_ratio_${key}` as keyof AnalyticsData] as number | null;
              return (
                <div key={key} className="space-y-1">
                  <p className="text-sm text-muted-foreground">{label}</p>
                  <p className="text-2xl font-bold">
                    {sharpe !== null ? sharpe.toFixed(2) : 'N/A'}
                  </p>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

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
              const drawdown = analytics[`max_drawdown_${key}` as keyof AnalyticsData] as number | null;
              return (
                <div key={key} className="space-y-1">
                  <p className="text-sm text-muted-foreground">{label}</p>
                  <p className="text-2xl font-bold text-red-500">
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
              const score = analytics[`avg_${key}_score` as keyof AnalyticsData] as number | null;
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

      {/* Sector Performance */}
      {analytics.sector_performance && (
        <Card>
          <CardHeader>
            <CardTitle>Sector Performance</CardTitle>
            <CardDescription>
              Performance breakdown by sector
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(analytics.sector_performance).map(([sector, data]: [string, SectorData]) => (
                <div key={sector} className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="font-medium">{sector}</span>
                  <div className="text-right">
                    <p className="font-bold">
                      {data.avg_return !== undefined
                        ? `${(data.avg_return * 100).toFixed(2)}%`
                        : 'N/A'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {data.count || 0} signals
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
