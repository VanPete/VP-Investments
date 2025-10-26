'use client';

import { useMemo } from 'react';
import type { SignalRanking } from '@/types/pipeline';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Activity, Target } from 'lucide-react';

interface PerformanceTabProps {
  signals: SignalRanking[];
  loading: boolean;
}

export function PerformanceTab({ signals, loading }: PerformanceTabProps) {
  // Calculate portfolio-level statistics
  const stats = useMemo(() => {
    if (!signals || signals.length === 0) {
      return null;
    }

    // Filter signals with backtest data for each interval
    const backtestedAny = signals.filter(s => 
      s.return_1d !== null && s.return_1d !== undefined
    );
    const backtested1d = signals.filter(s => s.return_1d !== null && s.return_1d !== undefined);
    const backtested7d = signals.filter(s => s.return_7d !== null && s.return_7d !== undefined);
    const backtested30d = signals.filter(s => s.return_30d !== null && s.return_30d !== undefined);
    const backtested90d = signals.filter(s => s.return_90d !== null && s.return_90d !== undefined);

    if (backtestedAny.length === 0) {
      return null;
    }

    // Calculate average returns (only for non-null values)
    const avgReturn1d = backtested1d.length > 0 
      ? backtested1d.reduce((sum, s) => sum + (s.return_1d || 0), 0) / backtested1d.length 
      : null;
    const avgReturn7d = backtested7d.length > 0
      ? backtested7d.reduce((sum, s) => sum + (s.return_7d || 0), 0) / backtested7d.length
      : null;
    const avgReturn30d = backtested30d.length > 0
      ? backtested30d.reduce((sum, s) => sum + (s.return_30d || 0), 0) / backtested30d.length
      : null;
    const avgReturn90d = backtested90d.length > 0
      ? backtested90d.reduce((sum, s) => sum + (s.return_90d || 0), 0) / backtested90d.length
      : null;

    // Calculate SPY average returns
    const avgSpy1d = backtested1d.length > 0
      ? backtested1d.reduce((sum, s) => sum + (s.spy_return_1d || 0), 0) / backtested1d.length
      : null;
    const avgSpy7d = backtested7d.length > 0
      ? backtested7d.reduce((sum, s) => sum + (s.spy_return_7d || 0), 0) / backtested7d.length
      : null;
    const avgSpy30d = backtested30d.length > 0
      ? backtested30d.reduce((sum, s) => sum + (s.spy_return_30d || 0), 0) / backtested30d.length
      : null;
    const avgSpy90d = backtested90d.length > 0
      ? backtested90d.reduce((sum, s) => sum + (s.spy_return_90d || 0), 0) / backtested90d.length
      : null;

    // Calculate win rates (beating SPY) - only for non-null values
    const winRate1d = backtested1d.length > 0
      ? backtested1d.filter(s => (s.return_1d || 0) > (s.spy_return_1d || 0)).length / backtested1d.length
      : null;
    const winRate7d = backtested7d.length > 0
      ? backtested7d.filter(s => (s.return_7d || 0) > (s.spy_return_7d || 0)).length / backtested7d.length
      : null;
    const winRate30d = backtested30d.length > 0
      ? backtested30d.filter(s => (s.return_30d || 0) > (s.spy_return_30d || 0)).length / backtested30d.length
      : null;
    const winRate90d = backtested90d.length > 0
      ? backtested90d.filter(s => (s.return_90d || 0) > (s.spy_return_90d || 0)).length / backtested90d.length
      : null;

    // Find top/worst performers - use 1D if 7D not available
    const performanceMetric = backtested7d.length > 0 ? 'return_7d' : 'return_1d';
    const sortableSignals = (backtested7d.length > 0 ? backtested7d : backtested1d);
    const sorted = [...sortableSignals].sort((a, b) => 
      (b[performanceMetric] || 0) - (a[performanceMetric] || 0)
    );
    const topPerformers = sorted.slice(0, 5);
    const worstPerformers = sorted.slice(-5).reverse();

    return {
      totalBacktested: backtestedAny.length,
      counts: {
        d1: backtested1d.length,
        d7: backtested7d.length,
        d30: backtested30d.length,
        d90: backtested90d.length,
      },
      avgReturns: { d1: avgReturn1d, d7: avgReturn7d, d30: avgReturn30d, d90: avgReturn90d },
      avgSpyReturns: { d1: avgSpy1d, d7: avgSpy7d, d30: avgSpy30d, d90: avgSpy90d },
      winRates: { d1: winRate1d, d7: winRate7d, d30: winRate30d, d90: winRate90d },
      topPerformers,
      worstPerformers,
      performanceMetric,
    };
  }, [signals]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex items-center space-x-3">
          <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
          <p className="text-gray-600">Loading performance data...</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className="p-6">
          <p className="text-gray-600">No backtest data available. Please run the pipeline first.</p>
        </Card>
      </div>
    );
  }

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatWinRate = (value: number) => `${(value * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* Portfolio Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1D Performance */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              1 Day Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold">
                  <span className={stats.avgReturns.d1 > 0 ? 'text-green-600' : 'text-red-600'}>
                    {formatPercent(stats.avgReturns.d1)}
                  </span>
                </span>
                {stats.avgReturns.d1 > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-600" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-600" />
                )}
              </div>
              <div className="text-xs text-gray-500">
                SPY: {formatPercent(stats.avgSpyReturns.d1)}
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Win Rate</span>
                  <Badge variant={stats.winRates.d1 > 0.5 ? 'default' : 'secondary'}>
                    {formatWinRate(stats.winRates.d1)}
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 7D Performance */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              7 Day Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold">
                  <span className={stats.avgReturns.d7 > 0 ? 'text-green-600' : 'text-red-600'}>
                    {formatPercent(stats.avgReturns.d7)}
                  </span>
                </span>
                {stats.avgReturns.d7 > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-600" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-600" />
                )}
              </div>
              <div className="text-xs text-gray-500">
                SPY: {formatPercent(stats.avgSpyReturns.d7)}
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Win Rate</span>
                  <Badge variant={stats.winRates.d7 > 0.5 ? 'default' : 'secondary'}>
                    {formatWinRate(stats.winRates.d7)}
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 30D Performance */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              30 Day Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold">
                  <span className={stats.avgReturns.d30 > 0 ? 'text-green-600' : 'text-red-600'}>
                    {formatPercent(stats.avgReturns.d30)}
                  </span>
                </span>
                {stats.avgReturns.d30 > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-600" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-600" />
                )}
              </div>
              <div className="text-xs text-gray-500">
                SPY: {formatPercent(stats.avgSpyReturns.d30)}
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Win Rate</span>
                  <Badge variant={stats.winRates.d30 > 0.5 ? 'default' : 'secondary'}>
                    {formatWinRate(stats.winRates.d30)}
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 90D Performance */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
              90 Day Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold">
                  <span className={stats.avgReturns.d90 > 0 ? 'text-green-600' : 'text-red-600'}>
                    {formatPercent(stats.avgReturns.d90)}
                  </span>
                </span>
                {stats.avgReturns.d90 > 0 ? (
                  <TrendingUp className="h-5 w-5 text-green-600" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-600" />
                )}
              </div>
              <div className="text-xs text-gray-500">
                SPY: {formatPercent(stats.avgSpyReturns.d90)}
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Win Rate</span>
                  <Badge variant={stats.winRates.d90 > 0.5 ? 'default' : 'secondary'}>
                    {formatWinRate(stats.winRates.d90)}
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top/Worst Performers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Performers */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-green-600" />
              Top Performers (7D)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {stats.topPerformers.map((signal, index) => (
                <div
                  key={signal.ticker}
                  className="flex items-center justify-between p-3 rounded-lg bg-green-50 dark:bg-green-900/20"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex items-center justify-center w-6 h-6 rounded-full bg-green-600 text-white text-xs font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-mono font-semibold">{signal.ticker}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        Score: {signal.overall_score.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-green-600">
                      {formatPercent(signal.return_7d || 0)}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      vs SPY: {formatPercent((signal.return_7d || 0) - (signal.spy_return_7d || 0))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Worst Performers */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-red-600" />
              Needs Attention (7D)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {stats.worstPerformers.map((signal, index) => (
                <div
                  key={signal.ticker}
                  className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-900/20"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex items-center justify-center w-6 h-6 rounded-full bg-red-600 text-white text-xs font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="font-mono font-semibold">{signal.ticker}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        Score: {signal.overall_score.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-red-600">
                      {formatPercent(signal.return_7d || 0)}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      vs SPY: {formatPercent((signal.return_7d || 0) - (signal.spy_return_7d || 0))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Portfolio Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Signals</div>
              <div className="text-2xl font-bold">{signals.length}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Backtested</div>
              <div className="text-2xl font-bold">{stats.totalBacktested}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Overall Win Rate (7D)</div>
              <div className="text-2xl font-bold">
                <span className={stats.winRates.d7 > 0.5 ? 'text-green-600' : 'text-red-600'}>
                  {formatWinRate(stats.winRates.d7)}
                </span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Beating SPY (7D)</div>
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(stats.winRates.d7 * stats.totalBacktested)}/{stats.totalBacktested}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
