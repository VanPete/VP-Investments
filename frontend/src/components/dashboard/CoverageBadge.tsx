'use client';

import { Badge } from '@/components/ui/badge';
import { CheckCircle2, AlertCircle, XCircle } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface CoverageBadgeProps {
  coverage: number; // 0-1 range
  size?: 'sm' | 'md' | 'lg';
  showTooltip?: boolean;
}

export function CoverageBadge({ coverage, size = 'md', showTooltip = true }: CoverageBadgeProps) {
  const percentage = coverage * 100;
  
  // Determine quality tier
  let quality: 'excellent' | 'good' | 'poor';
  let qualityLabel: string;
  let qualityDescription: string;
  let bgColor: string;
  let textColor: string;
  let borderColor: string;
  let Icon: typeof CheckCircle2;

  if (percentage >= 90) {
    quality = 'excellent';
    qualityLabel = 'Excellent Coverage';
    qualityDescription = '90%+ data coverage indicates highly reliable signals with minimal missing data points.';
    bgColor = 'bg-green-50 dark:bg-green-950';
    textColor = 'text-green-700 dark:text-green-300';
    borderColor = 'border-green-200 dark:border-green-800';
    Icon = CheckCircle2;
  } else if (percentage >= 70) {
    quality = 'good';
    qualityLabel = 'Good Coverage';
    qualityDescription = '70-90% coverage is acceptable but some data points may be missing, slightly affecting reliability.';
    bgColor = 'bg-yellow-50 dark:bg-yellow-950';
    textColor = 'text-yellow-700 dark:text-yellow-300';
    borderColor = 'border-yellow-200 dark:border-yellow-800';
    Icon = AlertCircle;
  } else {
    quality = 'poor';
    qualityLabel = 'Limited Coverage';
    qualityDescription = 'Below 70% coverage means significant missing data. Scores may be less reliable - use with caution.';
    bgColor = 'bg-red-50 dark:bg-red-950';
    textColor = 'text-red-700 dark:text-red-300';
    borderColor = 'border-red-200 dark:border-red-800';
    Icon = XCircle;
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5',
  };

  const iconSizes = {
    sm: 'h-3 w-3',
    md: 'h-3.5 w-3.5',
    lg: 'h-4 w-4',
  };

  const badgeContent = (
    <Badge
      variant="outline"
      className={`${bgColor} ${textColor} ${borderColor} ${sizeClasses[size]} font-semibold inline-flex items-center gap-1.5`}
    >
      <Icon className={iconSizes[size]} />
      {percentage.toFixed(1)}%
      {quality === 'excellent' && size !== 'sm' && (
        <span className="ml-0.5 text-xs font-medium opacity-75">High</span>
      )}
    </Badge>
  );

  if (!showTooltip) {
    return badgeContent;
  }

  return (
    <TooltipProvider>
      <Tooltip delayDuration={200}>
        <TooltipTrigger asChild>
          {badgeContent}
        </TooltipTrigger>
        <TooltipContent 
          side="top" 
          className="max-w-xs bg-white dark:bg-gray-900 border-[#00AEEF]/30 shadow-xl"
        >
          <div className="space-y-1">
            <p className="font-semibold text-sm text-gray-900 dark:text-gray-100">
              {qualityLabel}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {qualityDescription}
            </p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
