/**
 * FactorDetails component
 * Displays individual factors for a factor group with their values and scores
 */

import React from 'react';
import { useSignalFactors } from '@/hooks/useSignalFactors';
import { Loader2 } from 'lucide-react';

interface FactorDetailsProps {
  signalId: string;
  groupKey: string;
}

// Human-readable factor names (expand as needed)
const FACTOR_DISPLAY_NAMES: Record<string, string> = {
  // Technical
  price_1d_pct: 'Price 1D %',
  price_7d_pct: 'Price 7D %',
  price_30d_pct: 'Price 30D %',
  rsi_14: 'RSI 14',
  macd_value: 'MACD',
  sma_50: 'SMA 50',
  sma_200: 'SMA 200',
  volume_spike_ratio: 'Volume Spike',
  // Fundamental
  pe_ratio: 'P/E Ratio',
  forward_pe: 'Forward P/E',
  peg_ratio: 'PEG Ratio',
  pb_ratio: 'P/B Ratio',
  ev_ebitda: 'EV/EBITDA',
  roe: 'ROE',
  roa: 'ROA',
  gross_margin: 'Gross Margin',
  operating_margin: 'Operating Margin',
  net_margin: 'Net Margin',
  revenue_growth_yoy: 'Revenue Growth YoY',
  earnings_growth_yoy: 'Earnings Growth YoY',
  debt_to_equity: 'Debt/Equity',
  current_ratio: 'Current Ratio',
  // News/Macro
  news_sentiment: 'News Sentiment',
  days_to_earnings: 'Days to Earnings',
  earnings_surprise_last: 'Last Earnings Surprise',
  spy_correlation_60d: 'SPY Correlation',
  // Social
  reddit_sentiment: 'Reddit Sentiment',
  reddit_mentions_7d: 'Reddit Mentions 7D',
  buzz_vs_baseline: 'Buzz vs Baseline',
  // Risk
  volatility_30d: 'Volatility 30D',
  beta_60d: 'Beta 60D',
  sharpe_ratio_60d: 'Sharpe Ratio 60D',
  max_drawdown_1y: 'Max Drawdown 1Y',
  // Institutional
  inst_ownership_pct: 'Institutional Ownership %',
  analyst_rating_avg: 'Analyst Rating Avg',
  price_target_upside_pct: 'Price Target Upside %',
};

function formatFactorValue(value: number | null, factorKey: string): string {
  if (value === null || value === undefined) return 'N/A';
  
  // Convert to number if it's a string
  const numValue = typeof value === 'number' ? value : parseFloat(String(value));
  
  // Check if conversion resulted in a valid number
  if (isNaN(numValue)) return 'N/A';
  
  // Format based on factor type
  // NOTE: Many _pct fields are ALREADY in decimal form (0.50 = 50%), not raw percentages
  // Check if value is likely already a percentage (> 1.0 for most cases)
  if (factorKey.includes('_pct')) {
    // If value is > 2, it's likely already a percentage value, just append %
    if (Math.abs(numValue) > 2) {
      return `${numValue.toFixed(2)}%`;
    } else {
      // Small values are likely decimals (0.05 = 5%)
      return `${(numValue * 100).toFixed(2)}%`;
    }
  } else if (factorKey.includes('margin') || factorKey.includes('ownership')) {
    // Margins and ownership are typically stored as decimals (0.50 = 50%)
    return `${(numValue * 100).toFixed(2)}%`;
  } else if (factorKey.includes('ratio') && !factorKey.includes('spike')) {
    return numValue.toFixed(2);
  } else if (factorKey.includes('sentiment') || factorKey.includes('correlation')) {
    return numValue.toFixed(2);
  } else if (factorKey.includes('price') || factorKey.includes('sma') || factorKey.includes('ema')) {
    return `$${numValue.toFixed(2)}`;
  } else if (Math.abs(numValue) >= 1000000000) {
    return `$${(numValue / 1000000000).toFixed(2)}B`;
  } else if (Math.abs(numValue) >= 1000000) {
    return `$${(numValue / 1000000).toFixed(2)}M`;
  } else {
    return numValue.toFixed(2);
  }
}

export function FactorDetails({ signalId, groupKey }: FactorDetailsProps) {
  const { factors, loading, error } = useSignalFactors(signalId, groupKey);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-4 w-4 animate-spin text-gray-500" />
        <span className="ml-2 text-xs text-gray-500">Loading factors...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-xs text-red-500 py-2">
        Error loading factors: {error}
      </div>
    );
  }

  if (factors.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic py-2">
        No factor data available
      </div>
    );
  }

  // Show top 10 factors
  const topFactors = factors.slice(0, 10);

  return (
    <div className="space-y-1.5">
      {topFactors.map((factor, index) => {
        const displayName = FACTOR_DISPLAY_NAMES[factor.factor_key] || factor.factor_key;
        const formattedValue = formatFactorValue(factor.value, factor.factor_key);
        
        return (
          <div 
            key={factor.factor_key}
            className="flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-gray-50 dark:hover:bg-gray-700/50"
          >
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <span className="text-gray-400 dark:text-gray-500 font-mono text-[10px] w-4 flex-shrink-0">
                #{index + 1}
              </span>
              <span className="font-medium text-gray-700 dark:text-gray-300 truncate">
                {displayName}
              </span>
            </div>
            <span className="font-mono text-gray-600 dark:text-gray-400 text-right">
              {formattedValue}
            </span>
          </div>
        );
      })}
      {factors.length > 10 && (
        <div className="text-[10px] text-gray-400 dark:text-gray-500 text-center pt-1">
          +{factors.length - 10} more factors
        </div>
      )}
    </div>
  );
}
