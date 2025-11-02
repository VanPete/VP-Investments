'use client';

import { useState, useMemo } from 'react';
import type {
  WeightsConfig,
  FactorToGroup,
  MethodologyConfig,
  GroupKey,
} from '@/types/pipeline';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SignalsTable } from './SignalsTable';
import { DashboardHeader } from './DashboardHeader';
import { QuickStats } from './QuickStats';
import { AnalyticsTab } from './AnalyticsTab';
import { WeightsOverview } from '../methodology/WeightsOverview';
import { ScoringExplainer } from '../methodology/ScoringExplainer';
import { FactorLibrary } from '../methodology/FactorLibrary';
import { usePersistedColumnVisibility } from '@/hooks/usePersistedState';
import { useSupabaseSignals } from '@/hooks/useSupabaseSignals';
import { useSupabaseSignalsWithPerformance } from '@/hooks/useSupabaseSignalsWithPerformance';
import { Table, TrendingUp, BookOpen } from 'lucide-react';

interface SignalsDashboardProps {
  weightsConfig: WeightsConfig | null;
  factorToGroup: FactorToGroup | null;
  methodologyConfig: MethodologyConfig | null;
}

export function SignalsDashboard({
  weightsConfig,
  factorToGroup,
  methodologyConfig,
}: SignalsDashboardProps) {
  // Fetch signals from Supabase with run selection
  const { signals, loading, error, refetch, runs, selectedRunId, setSelectedRunId } = useSupabaseSignals();
  
  // Fetch signals with performance data for expandable rows
  const { signals: signalsWithPerf, loading: perfLoading } = useSupabaseSignalsWithPerformance(selectedRunId);
  
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

  // Use persisted column visibility - now with performance columns
  const [columnVisibility, setColumnVisibility] = usePersistedColumnVisibility({
    rank: true,  // Show rank column as first column
    ticker: true,
    companyName: true,  // Show company names by default
    sector: true,  // v3.3: Show sector column by default
    currentPrice: true,  // Show current price by default
    overallScore: true,
    coverage: true,
    technical: true,
    fundamental: true,
    newsMacro: true,
    social: true,
    risk: true,
    institutional: true,
  });

  // Apply filters to rankings (use performance-enriched data for Dashboard)
  const filteredRankings = useMemo(() => {
    // Use signalsWithPerf if available (has performance data), otherwise fall back to signals
    const dataSource = signalsWithPerf.length > 0 ? signalsWithPerf : signals;
    if (!dataSource) return [];
    return [...dataSource];
  }, [signals, signalsWithPerf]);

  // Display only top 10 or all
  const displayedRankings = useMemo(() => {
    if (showAll) return filteredRankings;
    return filteredRankings.slice(0, 10);
  }, [filteredRankings, showAll]);

  const handleRefresh = () => {
    refetch();
  };

  // Loading state (wait for both signals and performance data)
  if (loading || perfLoading) {
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

      {/* Quick Stats Cards - Reduced spacing */}
      <div className="mt-3">
        <QuickStats results={mockResults} />
      </div>

      {/* Main Content with Tabs */}
      <div className="px-4 py-3">
        <Tabs defaultValue="signals" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-3">
            <TabsTrigger value="signals" className="flex items-center gap-2">
              <Table className="h-4 w-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="methodology" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Methodology
            </TabsTrigger>
          </TabsList>

          <TabsContent value="signals" className="space-y-4">
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

          <TabsContent value="analytics">
            <AnalyticsTab loading={loading} />
          </TabsContent>

          <TabsContent value="methodology" className="space-y-4">
            {weightsConfig && factorToGroup && methodologyConfig ? (
              <>
                <WeightsOverview 
                  weightsConfig={weightsConfig}
                  factorCounts={
                    // Calculate factor counts per group
                    Object.keys(weightsConfig.group_weights).reduce((acc, group) => {
                      const groupKey = group as GroupKey;
                      acc[groupKey] = Object.values(factorToGroup).filter(g => g === groupKey).length;
                      return acc;
                    }, {} as Record<GroupKey, number>)
                  }
                />
                <ScoringExplainer methodologyConfig={methodologyConfig} />
                <FactorLibrary weightsConfig={weightsConfig} factorToGroup={factorToGroup} />
              </>
            ) : (
              <Card className="p-8">
                <div className="prose max-w-none dark:prose-invert">
                  <h2 className="text-2xl font-bold mb-4">VP Investments Methodology</h2>
                  
                  <section className="mb-8">
                    <h3 className="text-xl font-semibold mb-3">Overview</h3>
                    <p className="text-gray-700 dark:text-gray-300">
                      The VP Investments system analyzes stocks using 158 individual factors across 6 major categories,
                      combining quantitative metrics with alternative data to generate comprehensive investment signals.
                    </p>
                  </section>

                  <section className="mb-8">
                    <h3 className="text-xl font-semibold mb-3">Factor Groups</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">Technical Analysis</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Price momentum, trend indicators, volume analysis, and technical patterns
                        </p>
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">Fundamental Analysis</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Financial metrics, valuation ratios, profitability, and growth indicators
                        </p>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">News & Macro</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Market sentiment, economic indicators, and macroeconomic trends
                        </p>
                      </div>
                      <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-orange-700 dark:text-orange-400 mb-2">Social & Alternative</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Reddit mentions, social sentiment, and alternative data sources
                        </p>
                      </div>
                      <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">Risk & Stability</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Volatility measures, downside protection, and risk-adjusted metrics
                        </p>
                      </div>
                      <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
                        <h4 className="font-semibold text-indigo-700 dark:text-indigo-400 mb-2">Institutional & Smart Money</h4>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          Institutional ownership, insider activity, and smart money flows
                        </p>
                      </div>
                    </div>
                  </section>

                  <section className="mb-8">
                    <h3 className="text-xl font-semibold mb-3">Scoring Process</h3>
                    <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
                      <li>Individual factors are calculated from raw market data</li>
                      <li>Factors are normalized using robust z-scores (removes outliers)</li>
                      <li>Normalized factors are weighted within their groups</li>
                      <li>Group scores are combined using configurable weights</li>
                      <li>Final score ranges from -1.0 (strong sell) to +1.0 (strong buy)</li>
                    </ol>
                  </section>

                  <section className="mb-8">
                    <h3 className="text-xl font-semibold mb-3">Performance Tracking</h3>
                    <p className="text-gray-700 dark:text-gray-300 mb-3">
                      All signals are tracked across multiple time horizons to validate predictive power:
                    </p>
                    <div className="grid grid-cols-7 gap-2">
                      {['1d', '3d', '7d', '10d', '14d', '30d', '90d'].map((interval) => (
                        <div key={interval} className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-center">
                          <span className="font-mono font-semibold">{interval}</span>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                      Returns are compared against SPY (S&P 500) and sector-specific ETFs to calculate alpha.
                    </p>
                  </section>

                  <section>
                    <h3 className="text-xl font-semibold mb-3">Data Sources</h3>
                    <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                      <li>Yahoo Finance: Price data, fundamentals, analyst estimates</li>
                      <li>Reddit: Social sentiment from 12+ investment subreddits</li>
                      <li>Technical Indicators: Calculated from price/volume data</li>
                      <li>Market Data: SPY, VIX, Treasury yields for context</li>
                    </ul>
                  </section>
                </div>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
