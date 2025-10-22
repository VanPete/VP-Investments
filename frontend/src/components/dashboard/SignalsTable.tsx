'use client';

import { useState, Fragment, useMemo, useEffect } from 'react';
import type { SignalRanking, WeightsConfig, FactorToGroup, GroupKey, SortDirection } from '@/types/pipeline';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronDown, ChevronRight, ChevronUp, ChevronsUpDown } from 'lucide-react';
import {
  formatScore,
  formatCoverage,
  getScoreColorClass,
} from '@/lib/utils';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';
import { CoverageBadge } from './CoverageBadge';
import { MetricTooltip, METRIC_TOOLTIPS } from './MetricTooltip';
import { ColumnVisibilityToggle } from './ColumnVisibilityToggle';
import type { ColumnVisibility } from '@/hooks/usePersistedState';

interface SignalsTableProps {
  rankings: SignalRanking[];
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
  columnVisibility?: ColumnVisibility;
  onColumnVisibilityChange?: (visibility: ColumnVisibility) => void;
}

export function SignalsTable({
  rankings,
  weightsConfig,
  columnVisibility = {
    rank: true,
    ticker: true,
    overallScore: true,
    coverage: true,
    technical: true,
    fundamental: true,
    newsMacro: true,
    social: true,
    risk: true,
    institutional: true,
  },
  onColumnVisibilityChange,
}: SignalsTableProps) {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<string>('overallScore');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const toggleRow = (ticker: string) => {
    setExpandedRow(expandedRow === ticker ? null : ticker);
  };

  // Handle column visibility changes - clear sort if hidden column is sorted
  useEffect(() => {
    if (sortColumn && columnVisibility) {
      const columnKey = sortColumn as keyof ColumnVisibility;
      if (columnVisibility[columnKey] === false) {
        setSortColumn('overallScore');
        setSortDirection('desc');
      }
    }
  }, [columnVisibility, sortColumn]);

  // Handle column header click for sorting
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      // Cycle through: asc -> desc -> null (original)
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortDirection(null);
        setSortColumn('overallScore');
      }
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  // Sort rankings based on current sort config
  const sortedRankings = useMemo(() => {
    if (!sortDirection || !sortColumn) {
      return rankings;
    }

    const sorted = [...rankings].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortColumn) {
        case 'rank':
          aValue = a.rank;
          bValue = b.rank;
          break;
        case 'ticker':
          aValue = a.ticker;
          bValue = b.ticker;
          break;
        case 'overallScore':
          aValue = a.overall_score;
          bValue = b.overall_score;
          break;
        case 'coverage':
          aValue = a.total_coverage;
          bValue = b.total_coverage;
          break;
        case 'technical':
          aValue = a.group_scores.technical || 0;
          bValue = b.group_scores.technical || 0;
          break;
        case 'fundamental':
          aValue = a.group_scores.fundamental || 0;
          bValue = b.group_scores.fundamental || 0;
          break;
        case 'newsMacro':
          aValue = a.group_scores.news_macro || 0;
          bValue = b.group_scores.news_macro || 0;
          break;
        case 'social':
          aValue = a.group_scores.social_alternative || 0;
          bValue = b.group_scores.social_alternative || 0;
          break;
        case 'risk':
          aValue = a.group_scores.risk_stability || 0;
          bValue = b.group_scores.risk_stability || 0;
          break;
        case 'institutional':
          aValue = a.group_scores.institutional_smart_money || 0;
          bValue = b.group_scores.institutional_smart_money || 0;
          break;
        default:
          return 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return sortDirection === 'asc' 
        ? (aValue as number) - (bValue as number)
        : (bValue as number) - (aValue as number);
    });

    return sorted;
  }, [rankings, sortColumn, sortDirection]);

  // Render sort icon
  const SortIcon = ({ column }: { column: string }) => {
    if (sortColumn !== column) {
      return <ChevronsUpDown className="ml-1 h-4 w-4 opacity-50" />;
    }
    if (sortDirection === 'asc') {
      return <ChevronUp className="ml-1 h-4 w-4" />;
    }
    if (sortDirection === 'desc') {
      return <ChevronDown className="ml-1 h-4 w-4" />;
    }
    return <ChevronsUpDown className="ml-1 h-4 w-4 opacity-50" />;
  };

  if (rankings.length === 0) {
    return (
      <Card className="shadow-lg rounded-2xl border-gray-200 dark:border-gray-800">
        <CardContent className="py-12 text-center text-gray-500 dark:text-gray-400">
          No signals match your filters. Try adjusting the filter criteria.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="shadow-lg rounded-2xl border-gray-200 dark:border-gray-800">
      {/* Column Visibility Toggle - Top Right */}
      <div className="flex justify-end p-4 border-b border-gray-200 dark:border-gray-800">
        {onColumnVisibilityChange && (
          <ColumnVisibilityToggle
            visibility={columnVisibility}
            onVisibilityChange={onColumnVisibilityChange}
          />
        )}
      </div>

      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-gray-50 dark:bg-gray-900/50">
              <TableHead className="w-12"></TableHead>
              {columnVisibility.rank && (
                <TableHead 
                  className="font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('rank')}
                >
                  <span className="inline-flex items-center">
                    Rank
                    <SortIcon column="rank" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.rank.title}
                      description={METRIC_TOOLTIPS.rank.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.ticker && (
                <TableHead 
                  className="font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('ticker')}
                >
                  <span className="inline-flex items-center">
                    Ticker
                    <SortIcon column="ticker" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.overallScore && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('overallScore')}
                >
                  <span className="inline-flex items-center justify-end">
                    Overall Score
                    <SortIcon column="overallScore" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.overallScore.title}
                      description={METRIC_TOOLTIPS.overallScore.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.coverage && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('coverage')}
                >
                  <span className="inline-flex items-center justify-end">
                    Coverage
                    <SortIcon column="coverage" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.coverage.title}
                      description={METRIC_TOOLTIPS.coverage.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.technical && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('technical')}
                >
                  <span className="inline-flex items-center justify-end">
                    Technical
                    <SortIcon column="technical" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.technical.title}
                      description={METRIC_TOOLTIPS.technical.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.fundamental && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('fundamental')}
                >
                  <span className="inline-flex items-center justify-end">
                    Fundamental
                    <SortIcon column="fundamental" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.fundamental.title}
                      description={METRIC_TOOLTIPS.fundamental.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.newsMacro && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('newsMacro')}
                >
                  <span className="inline-flex items-center justify-end">
                    News/Macro
                    <SortIcon column="newsMacro" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.newsMacro.title}
                      description={METRIC_TOOLTIPS.newsMacro.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.social && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('social')}
                >
                  <span className="inline-flex items-center justify-end">
                    Social
                    <SortIcon column="social" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.social.title}
                      description={METRIC_TOOLTIPS.social.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.risk && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('risk')}
                >
                  <span className="inline-flex items-center justify-end">
                    Risk
                    <SortIcon column="risk" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.risk.title}
                      description={METRIC_TOOLTIPS.risk.description}
                    />
                  </span>
                </TableHead>
              )}
              {columnVisibility.institutional && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('institutional')}
                >
                  <span className="inline-flex items-center justify-end">
                    Institutional
                    <SortIcon column="institutional" />
                    <MetricTooltip 
                      title={METRIC_TOOLTIPS.institutional.title}
                      description={METRIC_TOOLTIPS.institutional.description}
                    />
                  </span>
                </TableHead>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedRankings.map((ranking) => (
              <Fragment key={ranking.ticker}>
                {/* Main Row */}
                <TableRow
                  className="hover:bg-gradient-to-r hover:from-[#001F3F]/5 hover:to-[#00AEEF]/5 dark:hover:from-[#001F3F]/10 dark:hover:to-[#00AEEF]/10 cursor-pointer transition-colors duration-200"
                  onClick={() => toggleRow(ranking.ticker)}
                >
                  <TableCell>
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                      {expandedRow === ranking.ticker ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </Button>
                  </TableCell>
                  {columnVisibility.rank && (
                    <TableCell className="font-semibold text-gray-900 dark:text-gray-100">{ranking.rank}</TableCell>
                  )}
                  {columnVisibility.ticker && (
                    <TableCell>
                      <span className="font-mono font-medium text-gray-900 dark:text-gray-100 tracking-wide">
                        {ranking.ticker}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.overallScore && (
                    <TableCell className="text-right">
                      <span className={`text-lg font-bold ${getScoreColorClass(ranking.overall_score)}`}>
                        {formatScore(ranking.overall_score)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.coverage && (
                    <TableCell className="text-right">
                      <CoverageBadge coverage={ranking.total_coverage} size="md" />
                    </TableCell>
                  )}
                  {columnVisibility.technical && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.technical)}`}>
                        {formatScore(ranking.group_scores.technical)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.fundamental && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.fundamental)}`}>
                        {formatScore(ranking.group_scores.fundamental)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.newsMacro && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.news_macro)}`}>
                        {formatScore(ranking.group_scores.news_macro)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.social && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.social_alternative)}`}>
                        {formatScore(ranking.group_scores.social_alternative)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.risk && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.risk_stability)}`}>
                        {formatScore(ranking.group_scores.risk_stability)}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.institutional && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${getScoreColorClass(ranking.group_scores.institutional_smart_money)}`}>
                        {formatScore(ranking.group_scores.institutional_smart_money)}
                      </span>
                    </TableCell>
                  )}
                </TableRow>

                {/* Expanded Row */}
                {expandedRow === ranking.ticker && (
                  <TableRow>
                    <TableCell colSpan={11} className="bg-gray-50 dark:bg-gray-900/30 p-6">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                          Group Breakdown for {ranking.ticker}
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {(Object.keys(ranking.group_scores) as GroupKey[]).map((groupKey) => (
                            <Card key={groupKey} className="bg-white dark:bg-gray-800/50 shadow-md">
                              <CardContent className="p-4">
                                <div className="flex justify-between items-start mb-2">
                                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    {GROUP_DISPLAY_NAMES[groupKey]}
                                  </span>
                                  <Badge variant="outline" className="text-xs">
                                    {weightsConfig
                                      ? `${(weightsConfig.group_weights[groupKey] * 100).toFixed(0)}%`
                                      : 'N/A'}
                                  </Badge>
                                </div>
                                <div className="space-y-2">
                                  <div className="flex justify-between items-center">
                                    <span className="text-xs text-gray-500 dark:text-gray-400">Score:</span>
                                    <span className={`text-sm font-semibold ${getScoreColorClass(ranking.group_scores[groupKey])}`}>
                                      {formatScore(ranking.group_scores[groupKey])}
                                    </span>
                                  </div>
                                  {/* Score Progress Bar with VanPiQ Gradient */}
                                  <div className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-gradient-to-r from-[#001F3F] to-[#00AEEF] rounded-full transition-all duration-300"
                                      style={{ width: `${Math.min(100, Math.max(0, ((ranking.group_scores[groupKey] + 5) / 10) * 100))}%` }}
                                    />
                                  </div>
                                  <div className="flex justify-between items-center mt-1">
                                    <span className="text-xs text-gray-500 dark:text-gray-400">Coverage:</span>
                                    <span className="text-sm font-semibold dark:text-gray-200">
                                      {formatCoverage(ranking.group_coverages[groupKey])}
                                    </span>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
