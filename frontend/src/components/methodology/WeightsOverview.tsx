'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { WeightsConfig, GroupKey } from '@/types/pipeline';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';
import { formatPercentage } from '@/lib/utils';

interface WeightsOverviewProps {
  weightsConfig: WeightsConfig;
  factorCounts: Record<GroupKey, number>;
}

export function WeightsOverview({ weightsConfig, factorCounts }: WeightsOverviewProps) {
  const groupEntries = (Object.keys(weightsConfig.group_weights) as GroupKey[]).map(
    (groupKey) => ({
      key: groupKey,
      name: GROUP_DISPLAY_NAMES[groupKey],
      weight: weightsConfig.group_weights[groupKey],
      factorCount: factorCounts[groupKey] || 0,
    })
  );

  // Sort by weight descending
  groupEntries.sort((a, b) => b.weight - a.weight);

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Signal Group Weights</CardTitle>
        <CardDescription>
          How different signal categories contribute to the overall score
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {groupEntries.map((group) => (
            <div key={group.key} className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <span className="font-medium text-gray-900 dark:text-gray-100">{group.name}</span>
                  <Badge variant="outline" className="text-xs">
                    {group.factorCount} factors
                  </Badge>
                </div>
                <span className="font-semibold text-blue-900 dark:text-blue-400">
                  {formatPercentage(group.weight)}
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-blue-900 dark:bg-blue-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${group.weight * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Total Signal Groups:</span>
            <span className="font-medium">{groupEntries.length}</span>
          </div>
          <div className="flex justify-between text-sm mt-2">
            <span className="text-gray-600 dark:text-gray-400">Total Factors:</span>
            <span className="font-medium">
              {Object.values(factorCounts).reduce((sum, count) => sum + count, 0)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
