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
  
  // Softer, more elegant gradients
  const gradients = [
    'from-yellow-100 to-yellow-200 dark:from-yellow-900/40 dark:to-yellow-800/40',
    'from-slate-100 to-slate-200 dark:from-slate-800/40 dark:to-slate-700/40',
    'from-orange-100 to-orange-200 dark:from-orange-900/40 dark:to-orange-800/40',
  ];
  
  const textColors = [
    'text-yellow-800 dark:text-yellow-200',
    'text-slate-800 dark:text-slate-200',
    'text-orange-800 dark:text-orange-200',
  ];
  
  const iconColors = [
    'text-yellow-600 dark:text-yellow-400',
    'text-slate-600 dark:text-slate-400',
    'text-orange-600 dark:text-orange-400',
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 px-4">
      {topSignals.map((signal, index) => {
        const Icon = icons[index];
        const gradient = gradients[index];
        const textColor = textColors[index];
        const iconColor = iconColors[index];
        
        return (
          <Card 
            key={signal.ticker} 
            className={`bg-gradient-to-br ${gradient} border-0 shadow-lg hover:shadow-xl transition-shadow`}
          >
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon className={`h-5 w-5 ${iconColor}`} />
                  <span className={`text-sm font-medium ${textColor}`}>
                    #{index + 1} Signal
                  </span>
                </div>
              </div>
              <div className="mt-3">
                <div className={`text-3xl font-bold tracking-tight ${textColor}`}>
                  {signal.ticker}
                </div>
                {signal.company_name && (
                  <div className={`text-sm font-medium mt-1 opacity-75 ${textColor}`}>
                    {signal.company_name}
                  </div>
                )}
                <div className={`text-lg font-semibold mt-2 ${textColor}`}>
                  Score: {signal.overall_score.toFixed(3)}
                </div>
                <div className={`text-xs mt-2 opacity-70 ${textColor}`}>
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
