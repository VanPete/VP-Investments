/**
 * Analytics Formatting Utilities
 * 
 * Centralized formatting for all analytics metrics with clear documentation
 * on how each metric is stored in the database.
 * 
 * STORAGE FORMATS (from backend):
 * 
 * 1. PERCENTAGES (stored as percentages, e.g., 32.65 = 32.65%):
 *    - win_rate
 *    - cagr
 *    - volatility
 *    - alpha_vs_spy (raw percentage points)
 *    - alpha_vs_qqq (raw percentage points)
 * 
 * 2. DECIMALS (stored as fractions, e.g., 0.3265 = 32.65%):
 *    - backtest_cumulative_returns.summary (all return fields)
 *    - top_sector_avg_return
 *    - worst_sector_avg_return
 *    - avg_return
 *    - hit_rate_top_decile
 * 
 * 3. RATIOS (raw numerical values):
 *    - sharpe_ratio
 *    - sortino_ratio
 *    - calmar_ratio
 *    - beta_vs_spy
 *    - beta_vs_qqq
 *    - profit_factor
 *    - win_loss_ratio
 *    - ic_mean
 *    - ic_std
 * 
 * 4. SCORE BUCKET METRICS (interval-specific):
 *    - avg_return: percentage (e.g., 2.91 = 2.91%)
 *    - win_rate: decimal (e.g., 0.5517 = 55.17%)
 *    - All ratios: raw values
 */

export type MetricType = 'percentage' | 'decimal' | 'ratio' | 'currency' | 'count';

export interface FormatOptions {
  decimals?: number;
  showSign?: boolean;
  prefix?: string;
  suffix?: string;
  nullText?: string;
}

/**
 * Format a percentage value that's already stored as percentage
 * Example: 32.65 → "32.65%"
 */
export function formatPercentage(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { decimals = 2, showSign = false, nullText = 'N/A' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Format a decimal fraction as percentage
 * Example: 0.3265 → "32.65%"
 */
export function formatDecimalAsPercent(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { decimals = 2, showSign = false, nullText = 'N/A' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  const percentValue = value * 100;
  const sign = showSign && percentValue > 0 ? '+' : '';
  return `${sign}${percentValue.toFixed(decimals)}%`;
}

/**
 * Format a ratio value
 * Example: 2.5432 → "2.54"
 */
export function formatRatio(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { decimals = 2, showSign = false, nullText = 'N/A' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}`;
}

/**
 * Format currency value
 * Example: 1234.56 → "$1,234.56"
 */
export function formatCurrency(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { decimals = 2, nullText = 'N/A', prefix = '$' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  return `${prefix}${value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

/**
 * Format count/integer value with thousand separators
 * Example: 1234 → "1,234"
 */
export function formatCount(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { nullText = 'N/A' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  return value.toLocaleString('en-US', {
    maximumFractionDigits: 0,
  });
}

/**
 * Format number with arbitrary precision
 */
export function formatNumber(
  value: number | null | undefined,
  options: FormatOptions = {}
): string {
  const { decimals = 2, showSign = false, nullText = 'N/A' } = options;
  
  if (value === null || value === undefined || isNaN(value)) return nullText;
  
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}`;
}

/**
 * Get color class based on value sign
 */
export function getValueColor(
  value: number | null | undefined,
  options: { inverse?: boolean; neutral?: boolean } = {}
): string {
  const { inverse = false, neutral = false } = options;
  
  if (value === null || value === undefined || isNaN(value) || neutral) {
    return 'text-gray-600 dark:text-gray-400';
  }
  
  if (value > 0) {
    return inverse 
      ? 'text-red-600 dark:text-red-400' 
      : 'text-green-600 dark:text-green-400';
  }
  
  if (value < 0) {
    return inverse 
      ? 'text-green-600 dark:text-green-400' 
      : 'text-red-600 dark:text-red-400';
  }
  
  return 'text-gray-600 dark:text-gray-400';
}

/**
 * Get background color class based on value
 */
export function getValueBgColor(
  value: number | null | undefined,
  options: { inverse?: boolean } = {}
): string {
  const { inverse = false } = options;
  
  if (value === null || value === undefined || isNaN(value)) {
    return 'bg-gray-100 dark:bg-gray-800';
  }
  
  if (value > 0) {
    return inverse 
      ? 'bg-red-100 dark:bg-red-900/30' 
      : 'bg-green-100 dark:bg-green-900/30';
  }
  
  if (value < 0) {
    return inverse 
      ? 'bg-green-100 dark:bg-green-900/30' 
      : 'bg-red-100 dark:bg-red-900/30';
  }
  
  return 'bg-gray-100 dark:bg-gray-800';
}

/**
 * Format Sharpe ratio with rating
 */
export function formatSharpeWithRating(sharpe: number | null | undefined): {
  value: string;
  rating: string;
  color: string;
} {
  if (sharpe === null || sharpe === undefined || isNaN(sharpe)) {
    return { value: 'N/A', rating: 'Unknown', color: 'text-gray-500' };
  }
  
  let rating: string;
  let color: string;
  
  if (sharpe > 3) {
    rating = 'Exceptional';
    color = 'text-green-700 dark:text-green-300';
  } else if (sharpe > 2) {
    rating = 'Excellent';
    color = 'text-green-600 dark:text-green-400';
  } else if (sharpe > 1) {
    rating = 'Good';
    color = 'text-green-500 dark:text-green-500';
  } else if (sharpe > 0) {
    rating = 'Fair';
    color = 'text-yellow-600 dark:text-yellow-400';
  } else {
    rating = 'Poor';
    color = 'text-red-600 dark:text-red-400';
  }
  
  return {
    value: sharpe.toFixed(2),
    rating,
    color,
  };
}

/**
 * Score bucket color utilities
 */
export const SCORE_BUCKET_COLORS = {
  strong_buy: {
    bg: 'bg-green-600',
    text: 'text-green-700 dark:text-green-400',
    border: 'border-green-300 dark:border-green-700',
    light: 'bg-green-100 dark:bg-green-900/30',
    chart: '#10b981',
  },
  buy: {
    bg: 'bg-green-500',
    text: 'text-green-600 dark:text-green-500',
    border: 'border-green-200 dark:border-green-800',
    light: 'bg-green-50 dark:bg-green-900/20',
    chart: '#22c55e',
  },
  hold: {
    bg: 'bg-yellow-500',
    text: 'text-yellow-600 dark:text-yellow-500',
    border: 'border-yellow-200 dark:border-yellow-800',
    light: 'bg-yellow-50 dark:bg-yellow-900/20',
    chart: '#fbbf24',
  },
  sell: {
    bg: 'bg-red-500',
    text: 'text-red-600 dark:text-red-500',
    border: 'border-red-200 dark:border-red-800',
    light: 'bg-red-50 dark:bg-red-900/20',
    chart: '#f87171',
  },
  strong_sell: {
    bg: 'bg-red-600',
    text: 'text-red-700 dark:text-red-400',
    border: 'border-red-300 dark:border-red-700',
    light: 'bg-red-100 dark:bg-red-900/30',
    chart: '#dc2626',
  },
} as const;

export const SCORE_BUCKET_LABELS = {
  strong_buy: 'Strong Buy (>0.75)',
  buy: 'Buy (0.50-0.75)',
  hold: 'Hold (-0.25-0.50)',
  sell: 'Sell (-0.50 to -0.25)',
  strong_sell: 'Strong Sell (<-0.50)',
} as const;

export type ScoreBucket = keyof typeof SCORE_BUCKET_LABELS;
