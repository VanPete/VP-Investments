'use client';

import { useMemo } from 'react';
import type { SignalRanking } from '@/types/pipeline';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
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
      <div className="space-y-6">
        {/* Skeleton for Performance Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardHeader className="pb-3">
                <Skeleton className="h-4 w-32" />
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-baseline justify-between">
                  <Skeleton className="h-8 w-24" />
                  <Skeleton className="h-5 w-5 rounded-full" />
                </div>
                <Skeleton className="h-3 w-20" />
                <div className="pt-2 border-t">
                  <div className="flex items-center justify-between">
                    <Skeleton className="h-3 w-16" />
                    <Skeleton className="h-5 w-12" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Skeleton for Top/Worst Performers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(2)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-40" />
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-3">
                  {[...Array(5)].map((_, j) => (
                    <div key={j} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="space-y-2 flex-1">
                        <Skeleton className="h-4 w-16" />
                        <Skeleton className="h-3 w-24" />
                      </div>
                      <Skeleton className="h-6 w-20" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Skeleton for Portfolio Summary */}
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i}>
                  <Skeleton className="h-4 w-24 mb-2" />
                  <Skeleton className="h-8 w-16" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
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

  const formatPercent = (value: number | null) => value !== null ? `${(value * 100).toFixed(2)}%` : 'N/A';
  const formatWinRate = (value: number | null) => value !== null ? `${(value * 100).toFixed(1)}%` : 'N/A';

  const PerformanceCard = ({ 
    title, 
    avgReturn, 
    avgSpy, 
    winRate, 
    count 
  }: { 
    title: string; 
    avgReturn: number | null; 
    avgSpy: number | null; 
    winRate: number | null;
    count: number;
  }) => {
    const hasData = avgReturn !== null;
    const isPositive = hasData && avgReturn > 0;

    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!hasData ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-gray-400" />
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Pending ({count} signals)
                </span>
              </div>
              <p className="text-xs text-gray-400">
                {title.includes('7') ? 'Wait 7 days' : title.includes('30') ? 'Wait 30 days' : title.includes('90') ? 'Wait 90 days' : 'Calculating...'}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className={`text-2xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {formatPercent(avgReturn)}
                </span>
                {isPositive ? (
                  <TrendingUp className="h-5 w-5 text-green-600" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-600" />
                )}
              </div>
              <div className="text-xs text-gray-500">
                SPY: {formatPercent(avgSpy)}
              </div>
              <div className="pt-2 border-t">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">Win Rate</span>
                  <Badge variant={(winRate ?? 0) > 0.5 ? 'default' : 'secondary'}>
                    {formatWinRate(winRate)}
                  </Badge>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Portfolio Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <PerformanceCard 
          title="1 Day Performance"
          avgReturn={stats.avgReturns.d1}
          avgSpy={stats.avgSpyReturns.d1}
          winRate={stats.winRates.d1}
          count={stats.counts.d1}
        />
        <PerformanceCard 
          title="7 Day Performance"
          avgReturn={stats.avgReturns.d7}
          avgSpy={stats.avgSpyReturns.d7}
          winRate={stats.winRates.d7}
          count={stats.counts.d7}
        />
        <PerformanceCard 
          title="30 Day Performance"
          avgReturn={stats.avgReturns.d30}
          avgSpy={stats.avgSpyReturns.d30}
          winRate={stats.winRates.d30}
          count={stats.counts.d30}
        />
        <PerformanceCard 
          title="90 Day Performance"
          avgReturn={stats.avgReturns.d90}
          avgSpy={stats.avgSpyReturns.d90}
          winRate={stats.winRates.d90}
          count={stats.counts.d90}
        />
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
              <div className="text-sm text-gray-600 dark:text-gray-400">Overall Win Rate ({stats.performanceMetric === 'return_7d' ? '7D' : '1D'})</div>
              <div className="text-2xl font-bold">
                <span className={((stats.performanceMetric === 'return_7d' ? stats.winRates.d7 : stats.winRates.d1) ?? 0) > 0.5 ? 'text-green-600' : 'text-red-600'}>
                  {formatWinRate(stats.performanceMetric === 'return_7d' ? stats.winRates.d7 : stats.winRates.d1)}
                </span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Beating SPY ({stats.performanceMetric === 'return_7d' ? '7D' : '1D'})</div>
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(((stats.performanceMetric === 'return_7d' ? stats.winRates.d7 : stats.winRates.d1) ?? 0) * stats.totalBacktested)}/{stats.totalBacktested}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
