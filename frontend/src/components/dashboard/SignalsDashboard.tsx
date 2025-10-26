'use client';

import { useState, useMemo } from 'react';
import type {
  WeightsConfig,
  FactorToGroup,
} from '@/types/pipeline';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SignalsTable } from './SignalsTable';
import { DashboardHeader } from './DashboardHeader';
import { QuickStats } from './QuickStats';
import { PerformanceTab } from './PerformanceTab';
import { usePersistedColumnVisibility } from '@/hooks/usePersistedState';
import { useSupabaseSignals } from '@/hooks/useSupabaseSignals';
import { BarChart3, Table } from 'lucide-react';

interface SignalsDashboardProps {
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
}

export function SignalsDashboard({
  weightsConfig,
  factorToGroup,
}: SignalsDashboardProps) {
  // Fetch signals from Supabase with run selection
  const { signals, loading, error, refetch, runs, selectedRunId, setSelectedRunId } = useSupabaseSignals();
  
  const [showAll, setShowAll] = useState(false);
  const [activeTab, setActiveTab] = useState('signals');

  // Convert Supabase runs to FileOption format
  const allAvailableFiles = useMemo(() => {
    return runs.map((r: { run_timestamp: string; id: string }) => ({
      filename: r.id,
      timestamp: r.run_timestamp,
      label: `${new Date(r.run_timestamp).toLocaleString()}`,
    })).sort((a, b) => 
      b.timestamp.localeCompare(a.timestamp)
    );
  }, [runs]);

  // Use persisted column visibility - now with backtest columns
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
    // Backtest columns (Phase 6) - hidden by default
    baseline: false,
    return1d: false,
    return7d: false,
    return30d: false,
    return90d: false,
    vsSpy: false,
  });

  // Apply filters to rankings (simplified - no manual filters, just show all)
  const filteredRankings = useMemo(() => {
    if (!signals) return [];
    return [...signals];
  }, [signals]);

  // Display only top 10 or all
  const displayedRankings = useMemo(() => {
    if (showAll) return filteredRankings;
    return filteredRankings.slice(0, 10);
  }, [filteredRankings, showAll]);

  const handleRefresh = () => {
    refetch();
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="p-6">
          <div className="flex items-center space-x-3">
            <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
            <p className="text-gray-600">Loading signals from database...</p>
          </div>
        </Card>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="p-6">
          <p className="text-red-600">Error loading signals: {error}</p>
          <Button onClick={handleRefresh} className="mt-4">
            Retry
          </Button>
        </Card>
      </div>
    );
  }

  // No data state
  if (!signals || signals.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="p-6">
          <p className="text-gray-600">No signals found. Please run the pipeline first.</p>
        </Card>
      </div>
    );
  }

  // Create mock metadata for QuickStats and Header
  const selectedRun = runs.find((r: { id: string }) => r.id === selectedRunId);
  const mockResults = {
    metadata: {
      timestamp: selectedRun?.run_timestamp || new Date().toISOString(),
      total_tickers: selectedRun?.total_tickers || signals.length,
      source: 'supabase',
    },
    rankings: signals,
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <DashboardHeader
        metadata={mockResults.metadata}
        availableFiles={allAvailableFiles}
        selectedFile={selectedRunId || ''}
        onFileChange={(runId: string) => setSelectedRunId(runId)}
        onRefresh={handleRefresh}
        totalCount={signals.length}
        displayedCount={displayedRankings.length}
      />

      {/* Quick Stats Cards */}
      <QuickStats results={mockResults} />

      {/* Main Content with Tabs */}
      <div className="container mx-auto px-4 py-6">
        <Tabs defaultValue="signals" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="signals" className="flex items-center gap-2">
              <Table className="h-4 w-4" />
              Signals
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Performance
            </TabsTrigger>
          </TabsList>

          <TabsContent value="signals" className="space-y-6">
            {/* Signals Table */}
            <SignalsTable
              rankings={displayedRankings}
              weightsConfig={weightsConfig}
              factorToGroup={factorToGroup}
              columnVisibility={columnVisibility}
              onColumnVisibilityChange={setColumnVisibility}
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
          </TabsContent>

          <TabsContent value="performance">
            <PerformanceTab signals={signals} loading={loading} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
