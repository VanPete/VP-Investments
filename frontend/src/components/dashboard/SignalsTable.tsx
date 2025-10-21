'use client';

import { useState, Fragment } from 'react';
import type { SignalRanking, WeightsConfig, FactorToGroup, GroupKey } from '@/types/pipeline';
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
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  formatScore,
  formatCoverage,
  getScoreColorClass,
  getCoverageQuality,
} from '@/lib/utils';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';

interface SignalsTableProps {
  rankings: SignalRanking[];
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
}

export function SignalsTable({
  rankings,
  weightsConfig,
  factorToGroup,
}: SignalsTableProps) {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const toggleRow = (ticker: string) => {
    setExpandedRow(expandedRow === ticker ? null : ticker);
  };

  if (rankings.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-gray-500">
          No signals match your filters. Try adjusting the filter criteria.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-gray-50">
              <TableHead className="w-[50px]"></TableHead>
              <TableHead className="w-[60px]">Rank</TableHead>
              <TableHead className="w-[100px]">Ticker</TableHead>
              <TableHead className="w-[120px] text-right">Overall Score</TableHead>
              <TableHead className="w-[100px] text-right">Coverage</TableHead>
              <TableHead className="text-right">Technical</TableHead>
              <TableHead className="text-right">Fundamental</TableHead>
              <TableHead className="text-right">News/Macro</TableHead>
              <TableHead className="text-right">Social</TableHead>
              <TableHead className="text-right">Risk</TableHead>
              <TableHead className="text-right">Institutional</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rankings.map((ranking) => (
              <Fragment key={ranking.ticker}>
                {/* Main Row */}
                <TableRow
                  className="hover:bg-gray-50 cursor-pointer"
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
                  <TableCell className="font-medium">{ranking.rank}</TableCell>
                  <TableCell>
                    <span className="font-mono font-semibold text-gray-900">
                      {ranking.ticker}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={`font-semibold ${getScoreColorClass(ranking.overall_score)}`}>
                      {formatScore(ranking.overall_score)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge
                      variant="outline"
                      className={getCoverageQuality(ranking.total_coverage).colorClass}
                    >
                      {formatCoverage(ranking.total_coverage)}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.technical)}>
                      {formatScore(ranking.group_scores.technical)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.fundamental)}>
                      {formatScore(ranking.group_scores.fundamental)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.news_macro)}>
                      {formatScore(ranking.group_scores.news_macro)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.social_alternative)}>
                      {formatScore(ranking.group_scores.social_alternative)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.risk_stability)}>
                      {formatScore(ranking.group_scores.risk_stability)}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={getScoreColorClass(ranking.group_scores.institutional_smart_money)}>
                      {formatScore(ranking.group_scores.institutional_smart_money)}
                    </span>
                  </TableCell>
                </TableRow>

                {/* Expanded Row */}
                {expandedRow === ranking.ticker && (
                  <TableRow>
                    <TableCell colSpan={11} className="bg-gray-50 p-6">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-gray-900 mb-3">
                          Group Breakdown for {ranking.ticker}
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {(Object.keys(ranking.group_scores) as GroupKey[]).map((groupKey) => (
                            <Card key={groupKey} className="bg-white">
                              <CardContent className="p-4">
                                <div className="flex justify-between items-start mb-2">
                                  <span className="text-sm font-medium text-gray-700">
                                    {GROUP_DISPLAY_NAMES[groupKey]}
                                  </span>
                                  <Badge variant="outline" className="text-xs">
                                    {weightsConfig
                                      ? `${(weightsConfig.group_weights[groupKey] * 100).toFixed(0)}%`
                                      : 'N/A'}
                                  </Badge>
                                </div>
                                <div className="space-y-1">
                                  <div className="flex justify-between">
                                    <span className="text-xs text-gray-500">Score:</span>
                                    <span className={`text-sm font-semibold ${getScoreColorClass(ranking.group_scores[groupKey])}`}>
                                      {formatScore(ranking.group_scores[groupKey])}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-xs text-gray-500">Coverage:</span>
                                    <span className="text-sm">
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
