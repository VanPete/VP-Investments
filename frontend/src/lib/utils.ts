import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a score to 3 decimal places
 */
export function formatScore(score: number | undefined): string {
  if (score === undefined || score === null) return 'N/A';
  return score.toFixed(3);
}

/**
 * Format a percentage to 1 decimal place
 */
export function formatPercentage(value: number | undefined): string {
  if (value === undefined || value === null) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format a coverage value (0-1) to percentage
 */
export function formatCoverage(coverage: number | undefined): string {
  if (coverage === undefined || coverage === null) return 'N/A';
  return `${(coverage * 100).toFixed(1)}%`;
}

/**
 * Get color class based on score value
 */
export function getScoreColorClass(score: number | undefined): string {
  if (score === undefined || score === null) return 'text-gray-400 dark:text-gray-500';
  if (score >= 0.75) return 'text-green-600 dark:text-green-400';
  if (score >= 0.50) return 'text-green-500 dark:text-green-400';
  if (score >= 0) return 'text-gray-700 dark:text-gray-300';
  if (score >= -0.50) return 'text-orange-600 dark:text-orange-400';
  return 'text-red-600 dark:text-red-400';
}

/**
 * Get coverage quality indicator
 */
export function getCoverageQuality(coverage: number | undefined): {
  label: string;
  colorClass: string;
} {
  if (coverage === undefined || coverage === null) {
    return { label: 'Unknown', colorClass: 'text-gray-400' };
  }
  
  if (coverage >= 0.9) {
    return { label: 'Excellent', colorClass: 'text-green-600' };
  }
  if (coverage >= 0.8) {
    return { label: 'Good', colorClass: 'text-green-500' };
  }
  if (coverage >= 0.7) {
    return { label: 'Acceptable', colorClass: 'text-yellow-500' };
  }
  return { label: 'Caution', colorClass: 'text-orange-500' };
}

/**
 * Format a timestamp to readable date string
 */
export function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

/**
 * Truncate ticker to max length
 */
export function truncateTicker(ticker: string, maxLength: number = 10): string {
  return ticker.length > maxLength ? `${ticker.substring(0, maxLength)}...` : ticker;
}
