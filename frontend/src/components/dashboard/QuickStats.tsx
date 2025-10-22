'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { PipelineResults } from '@/types/pipeline';
import { TrendingUp, Target, CheckCircle2, Rss } from 'lucide-react';

interface QuickStatsProps {
  results: PipelineResults;
}

export function QuickStats({ results }: QuickStatsProps) {
  const { rankings, metadata } = results;

  // Calculate stats
  const topPerformer = rankings[0];
  const avgScore = rankings.reduce((sum, r) => sum + r.overall_score, 0) / rankings.length;
  const highCoverageCount = rankings.filter(r => r.total_coverage > 0.9).length;

  // Discovery breakdown
  const redditTickers = metadata.discovery?.reddit_tickers || 0;
  const newsTickers = metadata.discovery?.news_tickers || 0;
  const totalUniverse = metadata.discovery?.total_universe || metadata.total_tickers;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 px-4">
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

      {/* Discovery Breakdown */}
      <Card className="shadow-lg border-gray-200 dark:border-gray-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 mb-2">
            <Rss className="h-5 w-5 text-purple-600 dark:text-purple-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Discovery Sources</span>
          </div>
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:border-orange-800">
                Reddit
              </Badge>
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">{redditTickers}</span>
            </div>
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800">
                News
              </Badge>
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">{newsTickers}</span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              {totalUniverse} total discovered
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
