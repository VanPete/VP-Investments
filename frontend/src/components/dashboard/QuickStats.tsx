'use client';

import { Card, CardContent } from '@/components/ui/card';
import type { PipelineResults } from '@/types/pipeline';
import { Trophy, Medal, Award } from 'lucide-react';

interface QuickStatsProps {
  results: PipelineResults;
}

export function QuickStats({ results }: QuickStatsProps) {
  const { rankings } = results;

  // Get top 3 signals
  const topSignals = rankings.slice(0, 3);

  const icons = [Trophy, Medal, Award];
  const gradients = [
    'from-yellow-400 to-yellow-600',
    'from-gray-400 to-gray-600',
    'from-orange-400 to-orange-600',
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 px-4">
      {topSignals.map((signal, index) => {
        const Icon = icons[index];
        const gradient = gradients[index];
        
        return (
          <Card 
            key={signal.ticker} 
            className={`bg-gradient-to-br ${gradient} text-white border-0 shadow-lg`}
          >
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5" />
                  <span className="text-sm font-medium opacity-90">
                    #{index + 1} Signal
                  </span>
                </div>
                <div className="text-xs opacity-75">
                  Rank {signal.rank}
                </div>
              </div>
              <div className="mt-3">
                <div className="text-3xl font-bold tracking-tight">
                  {signal.ticker}
                </div>
                <div className="text-lg font-semibold mt-1 opacity-90">
                  Score: {signal.overall_score.toFixed(3)}
                </div>
                <div className="text-xs mt-2 opacity-75">
                  {(signal.total_coverage * 100).toFixed(1)}% coverage
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
