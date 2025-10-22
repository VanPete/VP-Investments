'use client';

import type { FilterState } from '@/types/pipeline';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';

interface FilterPanelProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
}

export function FilterPanel({
  filters,
  onFiltersChange,
}: FilterPanelProps) {
  const handleReset = () => {
    onFiltersChange({
      minScore: -5,
      maxScore: 5,
      minCoverage: 0,
    });
  };

  const handleCoverageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({
      ...filters,
      minCoverage: parseFloat(e.target.value) / 100,
    });
  };

  return (
    <Card className="shadow-lg rounded-2xl border-gray-200 dark:border-gray-800">
      <CardContent className="pt-6">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Filters</h3>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
          </div>

          {/* Filter Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Min Coverage */}
            <div className="space-y-2">
              <Label htmlFor="coverage" className="dark:text-gray-200">
                Min Coverage: {(filters.minCoverage * 100).toFixed(0)}%
              </Label>
              <input
                id="coverage"
                type="range"
                min="0"
                max="100"
                step="5"
                value={filters.minCoverage * 100}
                onChange={handleCoverageChange}
                className="w-full cursor-pointer"
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
