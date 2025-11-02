/**
 * Predictive Strength Component
 * 
 * Displays Rank IC (Information Coefficient) metrics:
 * - IC time series line chart with 30-period moving average
 * - IC mean, std, hit rate (top decile), profit factor, win/loss ratio
 */

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import type { PlotParams } from 'react-plotly.js';
import { type AnalyticsData } from '@/hooks/useAnalytics';
import { formatRatio, formatDecimalAsPercent } from '@/lib/analytics-formatting';
import { TrendingUp, Target, DollarSign, TrendingDown } from 'lucide-react';

// Dynamic import for Plotly (client-side only)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PredictiveStrengthProps {
  analytics: AnalyticsData;
}

export function PredictiveStrength({ analytics }: PredictiveStrengthProps) {
  // Calculate 30-period moving average for IC
  const icData = useMemo(() => {
    if (!analytics.ic_series || analytics.ic_series.length === 0) {
      return { dates: [], ic: [], ma: [] };
    }
    
    const dates = analytics.ic_series.map(d => d.date);
    const ic = analytics.ic_series.map(d => d.ic);
    
    // Calculate 30-period MA
    const ma: (number | null)[] = [];
    for (let i = 0; i < ic.length; i++) {
      if (i < 29) {
        ma.push(null); // Not enough data for MA
      } else {
        const window = ic.slice(i - 29, i + 1);
        const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
        ma.push(avg);
      }
    }
    
    return { dates, ic, ma };
  }, [analytics.ic_series]);
  
  // Plotly chart data
  const plotData: PlotParams['data'] = [
    {
      x: icData.dates,
      y: icData.ic,
      type: 'scatter',
      mode: 'lines',
      name: 'Rank IC',
      line: { color: '#3b82f6', width: 1.5 },
      hovertemplate: '<b>%{x}</b><br>IC: %{y:.4f}<extra></extra>',
    },
    {
      x: icData.dates,
      y: icData.ma,
      type: 'scatter',
      mode: 'lines',
      name: '30-Period MA',
      line: { color: '#f59e0b', width: 2, dash: 'dash' },
      hovertemplate: '<b>%{x}</b><br>MA: %{y:.4f}<extra></extra>',
    },
  ];
  
  const plotLayout: Partial<PlotParams['layout']> = {
    autosize: true,
    height: 300,
    margin: { l: 50, r: 20, t: 20, b: 40 },
    xaxis: {
      title: { text: 'Date' },
      showgrid: true,
      gridcolor: '#374151',
    },
    yaxis: {
      title: { text: 'Rank IC' },
      showgrid: true,
      gridcolor: '#374151',
      zeroline: true,
      zerolinecolor: '#6b7280',
    },
    plot_bgcolor: '#1f2937',
    paper_bgcolor: '#111827',
    font: { color: '#9ca3af', size: 11 },
    hovermode: 'x unified',
    showlegend: true,
    legend: { x: 0, y: 1, orientation: 'h' },
  };
  
  // Stat cards
  const stats = [
    {
      label: 'IC Mean',
      value: formatRatio(analytics.ic_mean, { decimals: 4 }),
      icon: TrendingUp,
      description: 'Average correlation',
    },
    {
      label: 'IC Std Dev',
      value: formatRatio(analytics.ic_std, { decimals: 4 }),
      icon: TrendingDown,
      description: 'IC consistency',
    },
    {
      label: 'Hit Rate (Top 10%)',
      value: formatDecimalAsPercent(analytics.hit_rate_top_decile, { decimals: 1 }),
      icon: Target,
      description: 'Top decile win rate',
    },
    {
      label: 'Profit Factor',
      value: formatRatio(analytics.profit_factor, { decimals: 2 }),
      icon: DollarSign,
      description: 'Wins / |Losses|',
    },
    {
      label: 'Win/Loss Ratio',
      value: formatRatio(analytics.win_loss_ratio, { decimals: 2 }),
      icon: DollarSign,
      description: 'Avg win / avg loss',
    },
  ];
  
  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
          Predictive Strength
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Rank IC measures correlation between signal scores and forward returns
        </p>
      </div>
      
      {/* IC Chart */}
      {icData.dates.length > 0 ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <Plot
            data={plotData}
            layout={plotLayout}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      ) : (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            No IC data available. Run the pipeline to generate predictive metrics.
          </p>
        </div>
      )}
      
      {/* Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3"
            >
              <div className="flex items-center gap-2 mb-1">
                <Icon className="h-3.5 w-3.5 text-gray-400 dark:text-gray-500" />
                <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                  {stat.label}
                </span>
              </div>
              <div className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-0.5">
                {stat.value}
              </div>
              <p className="text-[10px] text-gray-500 dark:text-gray-400">
                {stat.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
