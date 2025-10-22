'use client';

import { Badge } from '@/components/ui/badge';
import { X } from 'lucide-react';
import type { FilterState } from '@/types/pipeline';
import { GROUP_DISPLAY_NAMES } from '@/types/pipeline';

interface FilterChipsProps {
  filters: FilterState;
  onRemoveFilter: (filterKey: keyof FilterState) => void;
}

export function FilterChips({ filters, onRemoveFilter }: FilterChipsProps) {
  const activeFilters: Array<{ key: keyof FilterState; label: string }> = [];

  // Check which filters are active
  if (filters.selectedGroup) {
    const groupName = GROUP_DISPLAY_NAMES[filters.selectedGroup as keyof typeof GROUP_DISPLAY_NAMES] || filters.selectedGroup;
    activeFilters.push({ key: 'selectedGroup', label: `Group: ${groupName}` });
  }

  if (filters.selectedFactor) {
    const factorName = filters.selectedFactor.replace(/_/g, ' ');
    activeFilters.push({ key: 'selectedFactor', label: `Factor: ${factorName}` });
  }

  if (filters.minCoverage > 0) {
    activeFilters.push({ key: 'minCoverage', label: `Min Coverage: ${(filters.minCoverage * 100).toFixed(0)}%` });
  }

  if (filters.searchQuery && filters.searchQuery.trim() !== '') {
    activeFilters.push({ key: 'searchQuery', label: `Search: ${filters.searchQuery}` });
  }

  // Don't render if no active filters
  if (activeFilters.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-2 px-4 mb-4">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Active Filters:
      </span>
      {activeFilters.map(({ key, label }) => (
        <Badge
          key={key}
          variant="secondary"
          className="pl-3 pr-2 py-1.5 bg-gradient-to-r from-[#001F3F]/10 to-[#00AEEF]/10 border border-[#00AEEF]/30 text-gray-900 dark:text-gray-100 hover:border-[#00AEEF]/50 transition-colors"
        >
          <span className="mr-1.5 font-medium">{label}</span>
          <button
            onClick={() => onRemoveFilter(key)}
            className="ml-1 hover:bg-red-500/20 rounded-full p-0.5 transition-colors"
            aria-label={`Remove ${label} filter`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
    </div>
  );
}
