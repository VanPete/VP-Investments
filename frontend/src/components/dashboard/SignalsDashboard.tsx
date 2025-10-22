'use client';

import { useState, useMemo } from 'react';
import type {
  PipelineResults,
  WeightsConfig,
  FactorToGroup,
  FileOption,
} from '@/types/pipeline';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { SignalsTable } from './SignalsTable';
import { FilterPanel } from './FilterPanel';
import { DashboardHeader } from './DashboardHeader';
import { QuickStats } from './QuickStats';
import { FilterChips } from './FilterChips';
import { ColumnVisibilityToggle } from './ColumnVisibilityToggle';
import { usePersistedFilters, usePersistedColumnVisibility } from '@/hooks/usePersistedState';

interface SignalsDashboardProps {
  initialResults: PipelineResults | null;
  availableFiles: FileOption[];
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
}

export function SignalsDashboard({
  initialResults,
  availableFiles,
  weightsConfig,
  factorToGroup,
}: SignalsDashboardProps) {
  const [results, setResults] = useState<PipelineResults | null>(initialResults);
  const [selectedFile, setSelectedFile] = useState<string>(
    availableFiles[0]?.filename || ''
  );
  const [showAll, setShowAll] = useState(false);
  
  // Use persisted filters with localStorage
  const [filters, setFilters] = usePersistedFilters({
    selectedGroup: null,
    selectedFactor: null,
    minScore: -5,
    maxScore: 5,
    minCoverage: 0,
    searchQuery: '',
  });

  // Use persisted column visibility
  const [columnVisibility, setColumnVisibility] = usePersistedColumnVisibility({
    rank: true,
    ticker: true,
    overallScore: true,
    coverage: true,
    technical: true,
    fundamental: true,
    newsMacro: true,
    social: true,
    risk: true,
    institutional: true,
  });

  // Apply filters to rankings
  const filteredRankings = useMemo(() => {
    if (!results) return [];

    let filtered = [...results.rankings];

    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      filtered = filtered.filter((r) =>
        r.ticker.toLowerCase().includes(query)
      );
    }

    // Score filter
    filtered = filtered.filter(
      (r) => r.overall_score >= filters.minScore && r.overall_score <= filters.maxScore
    );

    // Coverage filter
    filtered = filtered.filter((r) => r.total_coverage >= filters.minCoverage);

    // Group filter (show only if group score is significant)
    if (filters.selectedGroup && filters.selectedGroup !== 'all') {
      filtered = filtered.filter((r) => {
        const groupScore = r.group_scores[filters.selectedGroup as keyof typeof r.group_scores];
        return groupScore !== undefined && groupScore !== 0;
      });
    }

    return filtered;
  }, [results, filters]);

  // Display only top 10 or all
  const displayedRankings = useMemo(() => {
    if (showAll) return filteredRankings;
    return filteredRankings.slice(0, 10);
  }, [filteredRankings, showAll]);

  const handleFileChange = async (filename: string) => {
    setSelectedFile(filename);
    
    try {
      // Fetch the JSON file from the results directory
      const response = await fetch(`/results/${filename}`);
      if (!response.ok) {
        throw new Error('Failed to load results file');
      }
      
      const newResults = await response.json();
      setResults(newResults);
      
      // Reset to top 10 view when switching files
      setShowAll(false);
    } catch (error) {
      console.error('Error loading results:', error);
      // Optionally show error toast
    }
  };

  const handleRefresh = () => {
    // Reload the page to get latest data
    window.location.reload();
  };

  if (!results) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="p-6">
          <p className="text-gray-600">No pipeline results found. Please run the pipeline first.</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <DashboardHeader
        metadata={results.metadata}
        availableFiles={availableFiles}
        selectedFile={selectedFile}
        onFileChange={handleFileChange}
        onRefresh={handleRefresh}
        totalCount={results.rankings.length}
        displayedCount={displayedRankings.length}
      />

      {/* Quick Stats Cards */}
      {results && <QuickStats results={results} />}

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6 space-y-6">
        {/* Filters */}
        <FilterPanel
          filters={filters}
          onFiltersChange={setFilters}
          weightsConfig={weightsConfig}
          factorToGroup={factorToGroup}
        />

        {/* Active Filter Chips and Column Visibility */}
        <div className="flex items-center justify-between gap-4">
          <FilterChips 
            filters={filters} 
            onRemoveFilter={(key) => {
              setFilters(prev => ({
                ...prev,
                [key]: key === 'minCoverage' ? 0 : key === 'searchQuery' ? '' : null
              }));
            }}
          />
          <ColumnVisibilityToggle
            visibility={columnVisibility}
            onVisibilityChange={setColumnVisibility}
          />
        </div>

        {/* Signals Table */}
        <SignalsTable
          rankings={displayedRankings}
          weightsConfig={weightsConfig}
          factorToGroup={factorToGroup}
          searchQuery={filters.searchQuery}
          columnVisibility={columnVisibility}
        />

        {/* Show All Button */}
        {!showAll && filteredRankings.length > 10 && (
          <div className="flex justify-center">
            <Button
              variant="outline"
              size="lg"
              onClick={() => setShowAll(true)}
              className="min-w-[200px]"
            >
              Show All {filteredRankings.length} Tickers
            </Button>
          </div>
        )}

        {showAll && filteredRankings.length > 10 && (
          <div className="flex justify-center">
            <Button
              variant="outline"
              size="lg"
              onClick={() => setShowAll(false)}
              className="min-w-[200px]"
            >
              Show Top 10 Only
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
