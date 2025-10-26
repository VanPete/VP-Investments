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
import { Input } from '@/components/ui/input';
import { ChevronDown, ChevronRight, ChevronUp, ChevronsUpDown, TrendingUp, Search, X } from 'lucide-react';
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
  const [searchQuery, setSearchQuery] = useState<string>('');

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
        case 'baseline':
          aValue = a.backtest_baseline_price || 0;
          bValue = b.backtest_baseline_price || 0;
          break;
        case 'return1d':
          aValue = a.return_1d || 0;
          bValue = b.return_1d || 0;
          break;
        case 'return7d':
          aValue = a.return_7d || 0;
          bValue = b.return_7d || 0;
          break;
        case 'return30d':
          aValue = a.return_30d || 0;
          bValue = b.return_30d || 0;
          break;
        case 'return90d':
          aValue = a.return_90d || 0;
          bValue = b.return_90d || 0;
          break;
        case 'vsSpy':
          aValue = (a.return_7d || 0) - (a.spy_return_7d || 0);
          bValue = (b.return_7d || 0) - (b.spy_return_7d || 0);
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

  // Filter rankings based on search query
  const filteredRankings = useMemo(() => {
    if (!searchQuery.trim()) {
      return sortedRankings;
    }
    const query = searchQuery.toLowerCase();
    return sortedRankings.filter(r => r.ticker.toLowerCase().includes(query));
  }, [sortedRankings, searchQuery]);

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
      {/* Search and Column Visibility Toggle */}
      <div className="flex items-center justify-between gap-4 p-4 border-b border-gray-200 dark:border-gray-800">
        {/* Search Box */}
        <div className="flex-1 max-w-sm relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search by ticker..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 pr-9"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 w-7 p-0"
              onClick={() => setSearchQuery('')}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Results Counter */}
        {searchQuery && (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Showing {filteredRankings.length} of {rankings.length} signals
          </div>
        )}

        {/* Column Visibility Toggle */}
        <div className="flex-shrink-0">
          {onColumnVisibilityChange && (
            <ColumnVisibilityToggle
              visibility={columnVisibility}
              onVisibilityChange={onColumnVisibilityChange}
            />
          )}
        </div>
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
              {columnVisibility.baseline && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('baseline')}
                >
                  <span className="inline-flex items-center justify-end">
                    Baseline
                    <SortIcon column="baseline" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.return1d && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('return1d')}
                >
                  <span className="inline-flex items-center justify-end">
                    1D Return
                    <SortIcon column="return1d" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.return7d && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('return7d')}
                >
                  <span className="inline-flex items-center justify-end">
                    7D Return
                    <SortIcon column="return7d" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.return30d && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('return30d')}
                >
                  <span className="inline-flex items-center justify-end">
                    30D Return
                    <SortIcon column="return30d" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.return90d && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('return90d')}
                >
                  <span className="inline-flex items-center justify-end">
                    90D Return
                    <SortIcon column="return90d" />
                  </span>
                </TableHead>
              )}
              {columnVisibility.vsSpy && (
                <TableHead 
                  className="text-right font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSort('vsSpy')}
                >
                  <span className="inline-flex items-center justify-end">
                    vs SPY (7D)
                    <SortIcon column="vsSpy" />
                  </span>
                </TableHead>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredRankings.map((ranking) => (
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
                  {columnVisibility.baseline && (
                    <TableCell className="text-right">
                      {ranking.backtest_baseline_price && ranking.backtest_baseline_date ? (
                        <div className="text-sm">
                          <div className="font-semibold text-gray-900 dark:text-gray-100">
                            ${ranking.backtest_baseline_price.toFixed(2)}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            {new Date(ranking.backtest_baseline_date).toLocaleDateString()}
                          </div>
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </TableCell>
                  )}
                  {columnVisibility.return1d && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${ranking.return_1d && ranking.return_1d > 0 ? 'text-green-600 dark:text-green-400' : ranking.return_1d && ranking.return_1d < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'}`}>
                        {ranking.return_1d !== null && ranking.return_1d !== undefined ? `${(ranking.return_1d * 100).toFixed(2)}%` : '-'}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.return7d && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${ranking.return_7d && ranking.return_7d > 0 ? 'text-green-600 dark:text-green-400' : ranking.return_7d && ranking.return_7d < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'}`}>
                        {ranking.return_7d !== null && ranking.return_7d !== undefined ? `${(ranking.return_7d * 100).toFixed(2)}%` : '-'}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.return30d && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${ranking.return_30d && ranking.return_30d > 0 ? 'text-green-600 dark:text-green-400' : ranking.return_30d && ranking.return_30d < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'}`}>
                        {ranking.return_30d !== null && ranking.return_30d !== undefined ? `${(ranking.return_30d * 100).toFixed(2)}%` : '-'}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.return90d && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${ranking.return_90d && ranking.return_90d > 0 ? 'text-green-600 dark:text-green-400' : ranking.return_90d && ranking.return_90d < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'}`}>
                        {ranking.return_90d !== null && ranking.return_90d !== undefined ? `${(ranking.return_90d * 100).toFixed(2)}%` : '-'}
                      </span>
                    </TableCell>
                  )}
                  {columnVisibility.vsSpy && (
                    <TableCell className="text-right">
                      <span className={`font-semibold ${ranking.return_7d && ranking.spy_return_7d && (ranking.return_7d - ranking.spy_return_7d) > 0 ? 'text-green-600 dark:text-green-400' : ranking.return_7d && ranking.spy_return_7d && (ranking.return_7d - ranking.spy_return_7d) < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'}`}>
                        {ranking.return_7d !== null && ranking.spy_return_7d !== null && ranking.return_7d !== undefined && ranking.spy_return_7d !== undefined ? `${((ranking.return_7d - ranking.spy_return_7d) * 100).toFixed(2)}%` : '-'}
                      </span>
                    </TableCell>
                  )}
                </TableRow>

                {/* Expanded Row */}
                {expandedRow === ranking.ticker && (
                  <TableRow>
                    <TableCell colSpan={20} className="bg-gray-50 dark:bg-gray-900/30 p-6">
                      <div className="space-y-6">
                        {/* Backtest Performance Section */}
                        {ranking.backtest_baseline_price && (
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                              <TrendingUp className="h-5 w-5 text-blue-600" />
                              Backtest Performance
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                              <Card className="bg-white dark:bg-gray-800/50">
                                <CardContent className="p-4">
                                  <div className="text-sm text-gray-600 dark:text-gray-400">Baseline</div>
                                  <div className="text-2xl font-bold">
                                    ${ranking.backtest_baseline_price.toFixed(2)}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    {ranking.backtest_baseline_date && new Date(ranking.backtest_baseline_date).toLocaleDateString()}
                                  </div>
                                </CardContent>
                              </Card>
                              <Card className="bg-white dark:bg-gray-800/50">
                                <CardContent className="p-4">
                                  <div className="text-sm text-gray-600 dark:text-gray-400">Status</div>
                                  <Badge 
                                    variant={ranking.backtest_status === 'completed' ? 'default' : 'secondary'}
                                    className="mt-1"
                                  >
                                    {ranking.backtest_status || 'pending'}
                                  </Badge>
                                  {ranking.backtest_last_update && (
                                    <div className="text-xs text-gray-500 mt-1">
                                      Updated: {new Date(ranking.backtest_last_update).toLocaleString()}
                                    </div>
                                  )}
                                </CardContent>
                              </Card>
                            </div>
                            
                            {/* Returns Table */}
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm">
                                <thead>
                                  <tr className="border-b border-gray-200 dark:border-gray-700">
                                    <th className="text-left py-2 font-semibold">Interval</th>
                                    <th className="text-right py-2 font-semibold">Stock Return</th>
                                    <th className="text-right py-2 font-semibold">SPY Return</th>
                                    <th className="text-right py-2 font-semibold">vs SPY</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {[
                                    { label: '1 Day', stock: ranking.return_1d, spy: ranking.spy_return_1d },
                                    { label: '3 Days', stock: ranking.return_3d, spy: ranking.spy_return_3d },
                                    { label: '7 Days', stock: ranking.return_7d, spy: ranking.spy_return_7d },
                                    { label: '10 Days', stock: ranking.return_10d, spy: ranking.spy_return_10d },
                                    { label: '14 Days', stock: ranking.return_14d, spy: ranking.spy_return_14d },
                                    { label: '30 Days', stock: ranking.return_30d, spy: ranking.spy_return_30d },
                                    { label: '90 Days', stock: ranking.return_90d, spy: ranking.spy_return_90d },
                                  ].map((interval) => {
                                    const diff = (interval.stock || 0) - (interval.spy || 0);
                                    return (
                                      <tr key={interval.label} className="border-b border-gray-100 dark:border-gray-800">
                                        <td className="py-2">{interval.label}</td>
                                        <td className={`text-right py-2 font-semibold ${
                                          interval.stock && interval.stock > 0 ? 'text-green-600' : 
                                          interval.stock && interval.stock < 0 ? 'text-red-600' : 
                                          'text-gray-600'
                                        }`}>
                                          {interval.stock !== null && interval.stock !== undefined 
                                            ? `${(interval.stock * 100).toFixed(2)}%` 
                                            : '-'}
                                        </td>
                                        <td className="text-right py-2 text-gray-600">
                                          {interval.spy !== null && interval.spy !== undefined 
                                            ? `${(interval.spy * 100).toFixed(2)}%` 
                                            : '-'}
                                        </td>
                                        <td className={`text-right py-2 font-semibold ${
                                          diff > 0 ? 'text-green-600' : diff < 0 ? 'text-red-600' : 'text-gray-600'
                                        }`}>
                                          {interval.stock !== null && interval.spy !== null 
                                            ? `${(diff * 100).toFixed(2)}%` 
                                            : '-'}
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {/* Group Breakdown Section */}
                        <div>
                          <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                            Group Breakdown
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
