/**
 * SignalPerformanceSection Component
 * 
 * Compact performance tracking display for a single signal in the dashboard expandable row.
 * Shows all 7 time horizons in a grid format with ticker, SPY, and alpha rows.
 */

'use client';

import React from 'react';
import type { SignalRanking } from '@/types/pipeline';
import { TrendingUp } from 'lucide-react';

interface SignalPerformanceSectionProps {
  signal: SignalRanking;
}

export function SignalPerformanceSection({ signal }: SignalPerformanceSectionProps) {
  const intervals = ['1d', '3d', '7d', '10d', '14d', '30d', '90d'] as const;
  
  // Format return as percentage (data is stored as percentage, e.g., 7.4473 = 7.45%)
  const formatReturn = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return '---';
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };
  
  // Get color class for return value with better contrast
  const getReturnColorClass = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return 'text-gray-400 dark:text-gray-600';
    if (value > 0) return 'text-emerald-700 dark:text-emerald-400 font-semibold';
    if (value < 0) return 'text-rose-700 dark:text-rose-400 font-semibold';
    return 'text-gray-700 dark:text-gray-300';
  };
  
  // Get color class for alpha with even more contrast
  const getAlphaColorClass = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return 'text-gray-400 dark:text-gray-600';
    if (value > 0) return 'text-emerald-600 dark:text-emerald-300 font-bold';
    if (value < 0) return 'text-rose-600 dark:text-rose-300 font-bold';
    return 'text-gray-700 dark:text-gray-300';
  };

  // Check if signal has any performance data
  const hasPerformanceData = signal.backtest_baseline_price && signal.backtest_baseline_date;

  if (!hasPerformanceData) {
    return null;
  }

  // Format date as "Nov 1, 2025"
  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  // Calculate next update date and countdown
  const getNextUpdateInfo = () => {
    if (!signal.backtest_baseline_date) return null;
    
    const baselineDate = new Date(signal.backtest_baseline_date);
    const nextUpdate = new Date(baselineDate);
    nextUpdate.setDate(nextUpdate.getDate() + 3);
    
    const now = new Date();
    const diffMs = nextUpdate.getTime() - now.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    
    const formattedDate = nextUpdate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    
    if (diffMs < 0) return { date: formattedDate, countdown: 'Updating...' };
    
    let countdown = '';
    if (diffDays > 0) countdown = `${diffDays}d ${diffHours}h`;
    else countdown = `${diffHours}h`;
    
    return { date: formattedDate, countdown };
  };

  // Calculate date range for each interval
  const getDateRange = (interval: string) => {
    if (!signal.backtest_baseline_date) return '';
    
    const baseline = new Date(signal.backtest_baseline_date);
    const tradingDays = parseInt(interval.replace('d', ''));
    
    // Helper: Check if date is weekend
    const isWeekend = (date: Date) => {
      const day = date.getDay();
      return day === 0 || day === 6; // Sunday or Saturday
    };
    
    // Helper: Add trading days (skip weekends)
    const addTradingDays = (startDate: Date, days: number): Date => {
      const result = new Date(startDate);
      let addedDays = 0;
      
      while (addedDays < days) {
        result.setDate(result.getDate() + 1);
        if (!isWeekend(result)) {
          addedDays++;
        }
      }
      
      return result;
    };
    
    const endDate = addTradingDays(baseline, tradingDays);
    
    const formatShort = (date: Date) => {
      return date.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' });
    };
    
    // Calculate actual calendar days spanned
    const calendarDays = Math.ceil((endDate.getTime() - baseline.getTime()) / (1000 * 60 * 60 * 24));
    
    // If calendar days > trading days, show trading day count for clarity
    if (calendarDays > tradingDays) {
      return `${formatShort(baseline)} - ${formatShort(endDate)} (${tradingDays}td)`;
    }
    
    return `${formatShort(baseline)} - ${formatShort(endDate)}`;
  };

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-blue-600" />
          Performance Tracking
        </h4>
        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span>
            Baseline: <span className="font-semibold">${signal.backtest_baseline_price?.toFixed(2)}</span> on{' '}
            <span className="font-semibold">{formatDate(signal.backtest_baseline_date)}</span>
          </span>
          {(() => {
            const nextInfo = getNextUpdateInfo();
            if (!nextInfo) return null;
            return (
              <span>
                Next: <span className="font-semibold">{nextInfo.date}</span>
                {' '}(<span className="font-mono">{nextInfo.countdown}</span>)
              </span>
            );
          })()}
        </div>
      </div>
      
      {/* Performance Grid */}
      <div className="overflow-x-auto bg-white dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
              <th className="w-20 text-left px-3 py-2 font-semibold text-gray-700 dark:text-gray-300"></th>
              {intervals.map(interval => (
                <th 
                  key={interval}
                  className="px-3 py-2 text-center font-semibold text-gray-700 dark:text-gray-300 border-l border-gray-200 dark:border-gray-700"
                >
                  <div className="leading-tight">
                    <div>{interval.toUpperCase()}</div>
                    <div className="hidden md:block text-[10px] font-normal text-gray-500 dark:text-gray-400 mt-0.5">
                      {getDateRange(interval)}
                    </div>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Ticker Return Row */}
            <tr className="bg-blue-50 dark:bg-blue-950/20 border-l-4 border-l-blue-500 border-t-2 border-t-gray-300 dark:border-t-gray-600">
              <td className="px-3 py-2.5 text-xs font-bold text-blue-900 dark:text-blue-300">
                {signal.ticker}
              </td>
              {intervals.map(interval => {
                const field = `return_${interval}` as keyof SignalRanking;
                const value = signal[field] as number | undefined;
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2.5 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getReturnColorClass(value)}>
                      {formatReturn(value)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* SPY Return Row */}
            <tr className="bg-slate-100 dark:bg-slate-800/50 border-l-4 border-l-slate-400 border-t-2 border-t-gray-300 dark:border-t-gray-600">
              <td className="px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-400 align-middle">
                SPY
              </td>
              {intervals.map(interval => {
                const field = `spy_return_${interval}` as keyof SignalRanking;
                const value = signal[field] as number | undefined;
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getReturnColorClass(value)}>
                      {formatReturn(value)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* SPY Alpha Row */}
            <tr className="bg-white dark:bg-gray-900/50 border-l-4 border-l-violet-500">
              <td className="px-3 py-2 pl-6 text-xs font-medium italic text-violet-700 dark:text-violet-400 align-middle">
                α SPY
              </td>
              {intervals.map(interval => {
                const tickerField = `return_${interval}` as keyof SignalRanking;
                const spyField = `spy_return_${interval}` as keyof SignalRanking;
                const tickerValue = signal[tickerField] as number | undefined;
                const spyValue = signal[spyField] as number | undefined;
                
                let alpha: number | null = null;
                if (tickerValue !== null && tickerValue !== undefined && 
                    spyValue !== null && spyValue !== undefined) {
                  alpha = tickerValue - spyValue;
                }
                
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getAlphaColorClass(alpha)}>
                      {formatReturn(alpha)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* Spacer Row */}
            <tr className="h-2 bg-gray-100 dark:bg-gray-800">
              <td colSpan={8} className="border-b-2 border-gray-300 dark:border-gray-600"></td>
            </tr>
            
            {/* QQQ Return Row */}
            <tr className="bg-cyan-50 dark:bg-cyan-950/20 border-l-4 border-l-cyan-500">
              <td className="px-3 py-2 text-xs font-medium text-cyan-700 dark:text-cyan-400 align-middle">
                QQQ
              </td>
              {intervals.map(interval => {
                const field = `qqq_return_${interval}` as keyof SignalRanking;
                const value = signal[field] as number | undefined;
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getReturnColorClass(value)}>
                      {formatReturn(value)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* QQQ Alpha Row */}
            <tr className="bg-white dark:bg-gray-900/50 border-l-4 border-l-violet-500">
              <td className="px-3 py-2 pl-6 text-xs font-medium italic text-violet-700 dark:text-violet-400 align-middle">
                α QQQ
              </td>
              {intervals.map(interval => {
                const tickerField = `return_${interval}` as keyof SignalRanking;
                const qqqField = `qqq_return_${interval}` as keyof SignalRanking;
                const tickerValue = signal[tickerField] as number | undefined;
                const qqqValue = signal[qqqField] as number | undefined;
                
                let alpha: number | null = null;
                if (tickerValue !== null && tickerValue !== undefined && 
                    qqqValue !== null && qqqValue !== undefined) {
                  alpha = tickerValue - qqqValue;
                }
                
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getAlphaColorClass(alpha)}>
                      {formatReturn(alpha)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* Spacer Row */}
            <tr className="h-2 bg-gray-100 dark:bg-gray-800">
              <td colSpan={8} className="border-b-2 border-gray-300 dark:border-gray-600"></td>
            </tr>
            
            {/* Sector Return Row */}
            <tr className="bg-teal-50 dark:bg-teal-950/20 border-l-4 border-l-teal-500">
              <td className="px-3 py-2 text-xs font-medium text-teal-700 dark:text-teal-400 align-middle">
                Sector
              </td>
              {intervals.map(interval => {
                const field = `sector_return_${interval}` as keyof SignalRanking;
                const value = signal[field] as number | undefined;
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getReturnColorClass(value)}>
                      {formatReturn(value)}
                    </span>
                  </td>
                );
              })}
            </tr>
            
            {/* Sector Alpha Row */}
            <tr className="bg-white dark:bg-gray-900/50 border-l-4 border-l-violet-500">
              <td className="px-3 py-2 pl-6 text-xs font-medium italic text-violet-700 dark:text-violet-400 align-middle">
                α Sector
              </td>
              {intervals.map(interval => {
                const tickerField = `return_${interval}` as keyof SignalRanking;
                const sectorField = `sector_return_${interval}` as keyof SignalRanking;
                const tickerValue = signal[tickerField] as number | undefined;
                const sectorValue = signal[sectorField] as number | undefined;
                
                let alpha: number | null = null;
                if (tickerValue !== null && tickerValue !== undefined && 
                    sectorValue !== null && sectorValue !== undefined) {
                  alpha = tickerValue - sectorValue;
                }
                
                return (
                  <td 
                    key={interval}
                    className="px-3 py-2 text-center border-l border-gray-200 dark:border-gray-700"
                  >
                    <span className={getAlphaColorClass(alpha)}>
                      {formatReturn(alpha)}
                    </span>
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
