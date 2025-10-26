'use client';

import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { FactorToGroup, WeightsConfig, GroupKey } from '@/types/pipeline';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';
import { Search } from 'lucide-react';

interface FactorLibraryProps {
  factorToGroup: FactorToGroup;
  weightsConfig: WeightsConfig;
}

export function FactorLibrary({ factorToGroup, weightsConfig }: FactorLibraryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGroup, setSelectedGroup] = useState<string>('all');

  // Flatten all factors with their group information
  const allFactors = useMemo(() => {
    const factors: Array<{
      factor: string;
      description: string;
      group: GroupKey;
      groupName: string;
      weight: number;
    }> = [];

    (Object.keys(factorToGroup) as GroupKey[]).forEach((groupKey) => {
      const factorDescriptions = factorToGroup[groupKey];
      const factorWeightsKey = `factor_weights_${groupKey}` as keyof WeightsConfig;
      const factorWeights = weightsConfig[factorWeightsKey] as Record<string, number> || {};

      Object.entries(factorDescriptions).forEach(([factorName, description]) => {
        factors.push({
          factor: factorName,
          description,
          group: groupKey,
          groupName: GROUP_DISPLAY_NAMES[groupKey],
          weight: factorWeights[factorName] || 0,
        });
      });
    });

    return factors;
  }, [factorToGroup, weightsConfig]);

  // Filter factors based on search and group selection
  const filteredFactors = useMemo(() => {
    let filtered = allFactors;

    // Filter by group
    if (selectedGroup !== 'all') {
      filtered = filtered.filter((f) => f.group === selectedGroup);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (f) =>
          f.factor.toLowerCase().includes(query) ||
          f.description.toLowerCase().includes(query) ||
          f.groupName.toLowerCase().includes(query)
      );
    }

    // Sort by group, then by weight descending
    return filtered.sort((a, b) => {
      if (a.group !== b.group) {
        return a.group.localeCompare(b.group);
      }
      return b.weight - a.weight;
    });
  }, [allFactors, searchQuery, selectedGroup]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Factor Library</CardTitle>
        <CardDescription>
          Complete catalog of all {allFactors.length} factors used in signal ranking
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Search and Filter Controls */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              type="text"
              placeholder="Search factors by name or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={selectedGroup} onValueChange={setSelectedGroup}>
            <SelectTrigger className="w-full sm:w-[240px]">
              <SelectValue placeholder="Filter by group" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Groups ({allFactors.length})</SelectItem>
              {(Object.keys(factorToGroup) as GroupKey[]).map((groupKey) => {
                const count = allFactors.filter((f) => f.group === groupKey).length;
                return (
                  <SelectItem key={groupKey} value={groupKey}>
                    {GROUP_DISPLAY_NAMES[groupKey]} ({count})
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>

        {/* Results Count */}
        <div className="mb-4 text-sm text-gray-600 dark:text-gray-400">
          Showing {filteredFactors.length} of {allFactors.length} factors
        </div>

        {/* Factor List */}
        <div className="space-y-3">
          {filteredFactors.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              No factors match your search criteria
            </div>
          ) : (
            filteredFactors.map((factor) => (
              <div
                key={`${factor.group}-${factor.factor}`}
                className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-gray-300 dark:hover:border-gray-600 transition-colors bg-white dark:bg-gray-800/50"
              >
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono font-medium text-gray-900 dark:text-gray-100">
                      {factor.factor}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {factor.groupName}
                    </Badge>
                  </div>
                  <Badge variant="secondary" className="text-xs w-fit">
                    Weight: {(factor.weight * 100).toFixed(2)}%
                  </Badge>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">{factor.description}</p>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
