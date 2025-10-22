'use client';

import { Badge } from '@/components/ui/badge';
import { CheckCircle2, AlertCircle, XCircle } from 'lucide-react';

interface CoverageBadgeProps {
  coverage: number; // 0-1 range
  size?: 'sm' | 'md' | 'lg';
}

export function CoverageBadge({ coverage, size = 'md' }: CoverageBadgeProps) {
  const percentage = coverage * 100;
  
  // Determine quality tier
  let quality: 'excellent' | 'good' | 'poor';
  let bgColor: string;
  let textColor: string;
  let borderColor: string;
  let Icon: typeof CheckCircle2;

  if (percentage >= 90) {
    quality = 'excellent';
    bgColor = 'bg-green-50 dark:bg-green-950';
    textColor = 'text-green-700 dark:text-green-300';
    borderColor = 'border-green-200 dark:border-green-800';
    Icon = CheckCircle2;
  } else if (percentage >= 70) {
    quality = 'good';
    bgColor = 'bg-yellow-50 dark:bg-yellow-950';
    textColor = 'text-yellow-700 dark:text-yellow-300';
    borderColor = 'border-yellow-200 dark:border-yellow-800';
    Icon = AlertCircle;
  } else {
    quality = 'poor';
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

  return (
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
}
