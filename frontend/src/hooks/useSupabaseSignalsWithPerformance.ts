/**
 * Hook to fetch signals joined with performance data
 * Combines data from signals table and performance table for complete signal + backtest view
 */

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { SignalRanking } from '@/types/pipeline';

export interface SignalWithPerformance extends SignalRanking {
  // UUID from signals table (needed for performance lookup)
  id: string;
  // Performance fields from performance table
  baseline_price?: number;
  baseline_date?: string;
  sector?: string;
  market_cap?: number;
  // Alpha fields (ticker return - benchmark return)
  alpha_1d?: number;
  alpha_3d?: number;
  alpha_7d?: number;
  alpha_10d?: number;
  alpha_14d?: number;
  alpha_30d?: number;
  alpha_90d?: number;
  // QQQ benchmark returns
  qqq_return_1d?: number;
  qqq_return_3d?: number;
  qqq_return_7d?: number;
  qqq_return_10d?: number;
  qqq_return_14d?: number;
  qqq_return_30d?: number;
  qqq_return_90d?: number;
  // Sector benchmark returns
  sector_return_1d?: number;
  sector_return_3d?: number;
  sector_return_7d?: number;
  sector_return_10d?: number;
  sector_return_14d?: number;
  sector_return_30d?: number;
  sector_return_90d?: number;
  // Returns already in SignalRanking interface
}

interface UseSupabaseSignalsWithPerformanceResult {
  signals: SignalWithPerformance[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useSupabaseSignalsWithPerformance(
  runId: string | null
): UseSupabaseSignalsWithPerformanceResult {
  const [signals, setSignals] = useState<SignalWithPerformance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSignalsWithPerformance = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!runId) {
        setSignals([]);
        setLoading(false);
        return;
      }

      // Step 1: Fetch signals
      const { data: signalsData, error: signalsError } = await supabase
        .from('signals')
        .select(`
          id,
          ticker,
          company_name,
          current_price,
          market_cap,
          rank,
          overall_score,
          technical_score,
          fundamental_score,
          news_macro_score,
          social_alternative_score,
          risk_stability_score,
          institutional_smart_money_score,
          total_coverage,
          technical_coverage,
          fundamental_coverage,
          news_macro_coverage,
          social_alternative_coverage,
          risk_stability_coverage,
          institutional_smart_money_coverage,
          created_at
        `)
        .eq('run_id', runId)
        .order('overall_score', { ascending: false });

      if (signalsError) {
        console.error('Error fetching signals:', signalsError);
        throw signalsError;
      }

      console.log('Fetched signals:', signalsData?.length || 0);

      // Step 2: Fetch performance data for all signal IDs
      const signalIds = signalsData?.map(s => s.id) || [];
      
      if (signalIds.length === 0) {
        setSignals([]);
        setLoading(false);
        return;
      }

      const { data: performanceData, error: performanceError } = await supabase
        .from('performance')
        .select(`
          signal_id,
          baseline_price,
          baseline_date,
          sector,
          return_1d,
          return_3d,
          return_7d,
          return_10d,
          return_14d,
          return_30d,
          return_90d,
          alpha_1d,
          alpha_3d,
          alpha_7d,
          alpha_10d,
          alpha_14d,
          alpha_30d,
          alpha_90d,
          spy_return_1d,
          spy_return_3d,
          spy_return_7d,
          spy_return_10d,
          spy_return_14d,
          spy_return_30d,
          spy_return_90d,
          qqq_return_1d,
          qqq_return_3d,
          qqq_return_7d,
          qqq_return_10d,
          qqq_return_14d,
          qqq_return_30d,
          qqq_return_90d,
          sector_return_1d,
          sector_return_3d,
          sector_return_7d,
          sector_return_10d,
          sector_return_14d,
          sector_return_30d,
          sector_return_90d
        `)
        .in('signal_id', signalIds);

      if (performanceError) {
        console.error('Error fetching performance:', performanceError);
        console.error('Performance error message:', performanceError.message);
        console.error('Performance error code:', performanceError.code);
        console.error('Performance error details:', performanceError.details);
        // Don't throw - continue with signals data only
      }

      console.log('Fetched performance records:', performanceData?.length || 0);
      if (performanceData && performanceData.length > 0) {
        console.log('Sample performance record:', performanceData[0]);
      }

      // Step 3: Create a map for quick lookup
      const performanceMap = new Map(
        performanceData?.map(p => [p.signal_id, p]) || []
      );

      // Step 4: Join signals with performance data
      const joinedSignals: SignalWithPerformance[] = (signalsData || []).map((signal, index) => {
        const perf = performanceMap.get(signal.id);

        return {
          id: signal.id, // Add the UUID for performance lookups
          rank: signal.rank || index + 1,
          ticker: signal.ticker,
          company_name: signal.company_name || undefined,
          current_price: signal.current_price || undefined,
          overall_score: signal.overall_score || 0,
          total_coverage: signal.total_coverage || 0,
          group_scores: {
            technical: signal.technical_score || 0,
            fundamental: signal.fundamental_score || 0,
            news_macro: signal.news_macro_score || 0,
            social_alternative: signal.social_alternative_score || 0,
            risk_stability: signal.risk_stability_score || 0,
            institutional_smart_money: signal.institutional_smart_money_score || 0,
          },
          group_coverages: {
            technical: signal.technical_coverage || 0,
            fundamental: signal.fundamental_coverage || 0,
            news_macro: signal.news_macro_coverage || 0,
            social_alternative: signal.social_alternative_coverage || 0,
            risk_stability: signal.risk_stability_coverage || 0,
            institutional_smart_money: signal.institutional_smart_money_coverage || 0,
          },
          // Performance data from join (if available)
          baseline_price: perf?.baseline_price || undefined,
          baseline_date: perf?.baseline_date || undefined,
          sector: perf?.sector || undefined,
          market_cap: signal.market_cap || undefined,
          backtest_baseline_price: perf?.baseline_price || undefined,
          backtest_baseline_date: perf?.baseline_date || undefined,
          return_1d: perf?.return_1d ?? undefined,
          return_3d: perf?.return_3d ?? undefined,
          return_7d: perf?.return_7d ?? undefined,
          return_10d: perf?.return_10d ?? undefined,
          return_14d: perf?.return_14d ?? undefined,
          return_30d: perf?.return_30d ?? undefined,
          return_90d: perf?.return_90d ?? undefined,
          spy_return_1d: perf?.spy_return_1d ?? undefined,
          spy_return_3d: perf?.spy_return_3d ?? undefined,
          spy_return_7d: perf?.spy_return_7d ?? undefined,
          spy_return_10d: perf?.spy_return_10d ?? undefined,
          spy_return_14d: perf?.spy_return_14d ?? undefined,
          spy_return_30d: perf?.spy_return_30d ?? undefined,
          spy_return_90d: perf?.spy_return_90d ?? undefined,
          qqq_return_1d: perf?.qqq_return_1d ?? undefined,
          qqq_return_3d: perf?.qqq_return_3d ?? undefined,
          qqq_return_7d: perf?.qqq_return_7d ?? undefined,
          qqq_return_10d: perf?.qqq_return_10d ?? undefined,
          qqq_return_14d: perf?.qqq_return_14d ?? undefined,
          qqq_return_30d: perf?.qqq_return_30d ?? undefined,
          qqq_return_90d: perf?.qqq_return_90d ?? undefined,
          sector_return_1d: perf?.sector_return_1d ?? undefined,
          sector_return_3d: perf?.sector_return_3d ?? undefined,
          sector_return_7d: perf?.sector_return_7d ?? undefined,
          sector_return_10d: perf?.sector_return_10d ?? undefined,
          sector_return_14d: perf?.sector_return_14d ?? undefined,
          sector_return_30d: perf?.sector_return_30d ?? undefined,
          sector_return_90d: perf?.sector_return_90d ?? undefined,
          alpha_1d: perf?.alpha_1d ?? undefined,
          alpha_3d: perf?.alpha_3d ?? undefined,
          alpha_7d: perf?.alpha_7d ?? undefined,
          alpha_10d: perf?.alpha_10d ?? undefined,
          alpha_14d: perf?.alpha_14d ?? undefined,
          alpha_30d: perf?.alpha_30d ?? undefined,
          alpha_90d: perf?.alpha_90d ?? undefined,
          backtest_status: perf ? 'completed' : undefined,
          backtest_last_update: undefined,
        };
      });

      console.log('Joined signals count:', joinedSignals.length);
      if (joinedSignals.length > 0) {
        console.log('Sample joined signal:', {
          ticker: joinedSignals[0].ticker,
          return_1d: joinedSignals[0].return_1d,
          spy_return_1d: joinedSignals[0].spy_return_1d,
          alpha_1d: joinedSignals[0].alpha_1d,
          baseline_date: joinedSignals[0].baseline_date,
        });
      }

      setSignals(joinedSignals);
    } catch (err) {
      console.error('Error in useSupabaseSignalsWithPerformance:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch signals with performance');
    } finally {
      setLoading(false);
    }
  };

  // Fetch data when runId changes
  useEffect(() => {
    if (runId) {
      fetchSignalsWithPerformance();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  return {
    signals,
    loading,
    error,
    refetch: fetchSignalsWithPerformance,
  };
}
