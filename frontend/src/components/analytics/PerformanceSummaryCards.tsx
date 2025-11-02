/**
 * Performance Summary Cards
 * 
 * Displays 14 key performance metrics in a 2-row grid (7 metrics per row).
 * All formatting handled by centralized analytics-formatting utility.
 */

import React from 'react';
import { type AnalyticsData } from '@/hooks/useAnalytics';
import { 
  formatPercentage, 
  formatRatio, 
  getValueColor 
} from '@/lib/analytics-formatting';
import { TrendingUp, Activity, Shield, Target, BarChart3 } from 'lucide-react';

interface PerformanceSummaryCardsProps {
  analytics: AnalyticsData;
}

export function PerformanceSummaryCards({ analytics }: PerformanceSummaryCardsProps) {
  // Row 1: Win Rate, Sortino, CAGR, IC Mean, Profit Factor, β SPY, α SPY
  const row1Metrics = [
    {
      label: 'Win Rate',
      value: formatPercentage(analytics.win_rate, { decimals: 1 }),
      rawValue: analytics.win_rate,
      icon: Target,
      description: 'Winning trades %',
    },
    {
      label: 'Sortino',
      value: formatRatio(analytics.sortino_ratio, { decimals: 2 }),
      rawValue: analytics.sortino_ratio,
      icon: Shield,
      description: 'Downside risk-adjusted',
    },
    {
      label: 'CAGR',
      value: formatPercentage(analytics.cagr, { decimals: 1 }),
      rawValue: analytics.cagr,
      icon: TrendingUp,
      description: 'Annual growth rate',
    },
    {
      label: 'IC Mean',
      value: formatRatio(analytics.ic_mean, { decimals: 3 }),
      rawValue: analytics.ic_mean,
      icon: BarChart3,
      description: 'Avg predictive power',
    },
    {
      label: 'Profit Factor',
      value: formatRatio(analytics.profit_factor, { decimals: 2 }),
      rawValue: analytics.profit_factor,
      icon: TrendingUp,
      description: 'Gross profit / loss',
    },
    {
      label: 'β SPY',
      value: formatRatio(analytics.beta_vs_spy, { decimals: 2 }),
      rawValue: analytics.beta_vs_spy,
      icon: Activity,
      description: 'Market sensitivity',
      neutral: true,
    },
    {
      label: 'α SPY',
      value: formatPercentage(analytics.alpha_vs_spy, { decimals: 2 }),
      rawValue: analytics.alpha_vs_spy,
      icon: TrendingUp,
      description: 'Excess return',
    },
  ];
  
  // Row 2: Sharpe, Calmar, Volatility, IC Std, Win/Loss, β QQQ, α QQQ
  const row2Metrics = [
    {
      label: 'Sharpe',
      value: formatRatio(analytics.sharpe_ratio, { decimals: 2 }),
      rawValue: analytics.sharpe_ratio,
      icon: Target,
      description: 'Risk-adjusted return',
    },
    {
      label: 'Calmar',
      value: formatRatio(analytics.calmar_ratio, { decimals: 2 }),
      rawValue: analytics.calmar_ratio,
      icon: Shield,
      description: 'Return vs drawdown',
    },
    {
      label: 'Volatility',
      value: formatPercentage(analytics.volatility, { decimals: 1 }),
      rawValue: analytics.volatility,
      icon: Activity,
      description: 'Price fluctuation',
      neutral: true,
    },
    {
      label: 'IC Std',
      value: formatRatio(analytics.ic_std, { decimals: 3 }),
      rawValue: analytics.ic_std,
      icon: BarChart3,
      description: 'IC consistency',
      neutral: true,
    },
    {
      label: 'Win/Loss',
      value: formatRatio(analytics.win_loss_ratio, { decimals: 2 }),
      rawValue: analytics.win_loss_ratio,
      icon: TrendingUp,
      description: 'Avg win / avg loss',
    },
    {
      label: 'β QQQ',
      value: formatRatio(analytics.beta_vs_qqq, { decimals: 2 }),
      rawValue: analytics.beta_vs_qqq,
      icon: Activity,
      description: 'Nasdaq sensitivity',
      neutral: true,
    },
    {
      label: 'α QQQ',
      value: formatPercentage(analytics.alpha_vs_qqq, { decimals: 2 }),
      rawValue: analytics.alpha_vs_qqq,
      icon: TrendingUp,
      description: 'Excess vs Nasdaq',
    },
  ];
  
  const MetricCard = ({ metric }: { metric: typeof row1Metrics[0] }) => {
    const Icon = metric.icon;
    const colorClass = metric.neutral 
      ? 'text-gray-700 dark:text-gray-300' 
      : getValueColor(metric.rawValue);
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-lg transition-all duration-200">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
            {metric.label}
          </span>
          <Icon className="h-4 w-4 text-gray-400 dark:text-gray-500" />
        </div>
        
        <div className={`text-2xl font-bold ${colorClass} mb-1`}>
          {metric.value}
        </div>
        
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {metric.description}
        </p>
      </div>
    );
  };
  
  return (
    <div className="space-y-4">
      {/* Row 1 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        {row1Metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
      
      {/* Row 2 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        {row2Metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
    </div>
  );
}
