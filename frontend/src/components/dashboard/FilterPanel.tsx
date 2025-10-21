'use client';

import { useMemo } from 'react';
import type { FilterState, WeightsConfig, FactorToGroup, GroupKey } from '@/types/pipeline';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';

interface FilterPanelProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
}

export function FilterPanel({
  filters,
  onFiltersChange,
  weightsConfig,
  factorToGroup,
}: FilterPanelProps) {
  // Get factors for selected group
  const availableFactors = useMemo(() => {
    if (!filters.selectedGroup || filters.selectedGroup === 'all' || !factorToGroup) {
      return [];
    }

    const groupKey = filters.selectedGroup as GroupKey;
    const factorsInGroup = factorToGroup[groupKey];
    
    if (!factorsInGroup) return [];

    return Object.keys(factorsInGroup).sort();
  }, [filters.selectedGroup, factorToGroup]);

  const handleReset = () => {
    onFiltersChange({
      selectedGroup: null,
      selectedFactor: null,
      minScore: -5,
      maxScore: 5,
      minCoverage: 0,
      searchQuery: '',
    });
  };

  const handleGroupChange = (value: string) => {
    onFiltersChange({
      ...filters,
      selectedGroup: value === 'all' ? null : value,
      selectedFactor: null, // Reset factor when group changes
    });
  };

  const handleFactorChange = (value: string) => {
    onFiltersChange({
      ...filters,
      selectedFactor: value === 'all' ? null : value,
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({
      ...filters,
      searchQuery: e.target.value,
    });
  };

  const handleCoverageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({
      ...filters,
      minCoverage: parseFloat(e.target.value) / 100,
    });
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Quick Filters</h3>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
          </div>

          {/* Filter Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Search */}
            <div className="space-y-2">
              <Label htmlFor="search">Search Ticker</Label>
              <Input
                id="search"
                type="text"
                placeholder="e.g., AAPL"
                value={filters.searchQuery}
                onChange={handleSearchChange}
              />
            </div>

            {/* Group Filter */}
            <div className="space-y-2">
              <Label htmlFor="group">Group</Label>
              <Select
                value={filters.selectedGroup || 'all'}
                onValueChange={handleGroupChange}
              >
                <SelectTrigger id="group">
                  <SelectValue placeholder="All Groups" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Groups</SelectItem>
                  {Object.entries(GROUP_DISPLAY_NAMES).map(([key, name]) => (
                    <SelectItem key={key} value={key}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Factor Filter */}
            <div className="space-y-2">
              <Label htmlFor="factor">Factor</Label>
              <Select
                value={filters.selectedFactor || 'all'}
                onValueChange={handleFactorChange}
                disabled={!filters.selectedGroup || filters.selectedGroup === 'all'}
              >
                <SelectTrigger id="factor">
                  <SelectValue placeholder="All Factors" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Factors</SelectItem>
                  {availableFactors.map((factor) => (
                    <SelectItem key={factor} value={factor}>
                      {factor.replace(/_/g, ' ')}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Min Coverage */}
            <div className="space-y-2">
              <Label htmlFor="coverage">
                Min Coverage: {(filters.minCoverage * 100).toFixed(0)}%
              </Label>
              <Input
                id="coverage"
                type="range"
                min="0"
                max="100"
                step="5"
                value={filters.minCoverage * 100}
                onChange={handleCoverageChange}
                className="cursor-pointer"
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
