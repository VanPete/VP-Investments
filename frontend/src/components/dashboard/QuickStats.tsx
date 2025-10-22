'use client';

import { Card, CardContent } from '@/components/ui/card';
import type { PipelineResults } from '@/types/pipeline';
import { TrendingUp, Target, CheckCircle2 } from 'lucide-react';

interface QuickStatsProps {
  results: PipelineResults;
}

export function QuickStats({ results }: QuickStatsProps) {
  const { rankings } = results;

  // Calculate stats
  const topPerformer = rankings[0];
  const avgScore = rankings.reduce((sum, r) => sum + r.overall_score, 0) / rankings.length;
  const highCoverageCount = rankings.filter(r => r.total_coverage > 0.9).length;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6 px-4">
      {/* Average Score */}
      <Card className="shadow-lg border-gray-200 dark:border-gray-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Average Score</span>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              {avgScore.toFixed(3)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Across {rankings.length} tickers
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top Performer */}
      <Card className="bg-gradient-to-br from-[#001F3F] to-[#00AEEF] text-white border-0 shadow-lg">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 mb-2">
            <Target className="h-5 w-5" />
            <span className="text-sm font-medium opacity-90">Top Performer</span>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold tracking-tight">
              {topPerformer.ticker}
            </div>
            <div className="text-lg font-semibold mt-1 opacity-90">
              {topPerformer.overall_score.toFixed(3)}
            </div>
            <div className="text-xs mt-2 opacity-75">
              {(topPerformer.total_coverage * 100).toFixed(1)}% coverage
            </div>
          </div>
        </CardContent>
      </Card>

      {/* High Coverage Count */}
      <Card className="shadow-lg border-gray-200 dark:border-gray-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">High Coverage</span>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              {highCoverageCount}/{rankings.length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Tickers {'>'} 90% coverage
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
