import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { SignalRanking } from '@/types/pipeline';

interface SignalRun {
  id: string;
  run_timestamp: string;
  total_tickers: number;
  status: string;
}

interface UseSupabaseSignalsResult {
  signals: SignalRanking[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  runs: SignalRun[];
  selectedRunId: string | null;
  setSelectedRunId: (runId: string | null) => void;
}

export function useSupabaseSignals(): UseSupabaseSignalsResult {
  const [signals, setSignals] = useState<SignalRanking[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<SignalRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // Fetch available runs
  const fetchRuns = async () => {
    const { data, error } = await supabase
      .from('signal_runs')
      .select('id, run_timestamp, total_tickers, status')
      .order('run_timestamp', { ascending: false });  // Get ALL runs, no limit

    if (!error && data) {
      setRuns(data);
      // Auto-select most recent run if none selected
      if (!selectedRunId && data.length > 0) {
        setSelectedRunId(data[0].id);
      }
    }
  };

  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!selectedRunId) {
        setSignals([]);
        setLoading(false);
        return;
      }

      // Query signals for the selected run
      // v3.2: Performance data in separate table - join to get backtest results
      // v3.3: Include sector from signals table
      const { data, error: fetchError } = await supabase
        .from('signals')
        .select(`
          id,
          ticker,
          company_name,
          sector,
          current_price,
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
          created_at,
          performance (
            baseline_price,
            baseline_date,
            return_1d,
            return_3d,
            return_7d,
            return_10d,
            return_14d,
            return_30d,
            return_90d,
            spy_return_1d,
            spy_return_3d,
            spy_return_7d,
            spy_return_10d,
            spy_return_14d,
            spy_return_30d,
            spy_return_90d,
            status,
            last_update
          )
        `)
        .eq('run_id', selectedRunId)
        .order('overall_score', { ascending: false });

      if (fetchError) {
        console.error('Supabase error details:', fetchError);
        throw fetchError;
      }

      console.log('Successfully fetched signals:', data?.length || 0);

      // Transform Supabase data to SignalRanking format
      // v3.2: Coverage values and performance data now properly joined
      const rankings: SignalRanking[] = (data || []).map((signal, index) => {
        // Access performance data from joined table (array with single element or empty)
        const perf = Array.isArray(signal.performance) && signal.performance.length > 0 
          ? signal.performance[0] 
          : null;
        
        return {
          rank: signal.rank || index + 1,
          ticker: signal.ticker,
          company_name: signal.company_name || undefined,
          sector: signal.sector || undefined,  // v3.3: Include sector
          current_price: signal.current_price || undefined,
          overall_score: signal.overall_score || 0,
          total_coverage: signal.total_coverage || 0,  // Use database value, not calculated
          group_scores: {
            technical: signal.technical_score || 0,
            fundamental: signal.fundamental_score || 0,
            news_macro: signal.news_macro_score || 0,
            social_alternative: signal.social_alternative_score || 0,
            risk_stability: signal.risk_stability_score || 0,
            institutional_smart_money: signal.institutional_smart_money_score || 0,
          },
          group_coverages: {
            technical: signal.technical_coverage || 0,  // Use database values
            fundamental: signal.fundamental_coverage || 0,
            news_macro: signal.news_macro_coverage || 0,
            social_alternative: signal.social_alternative_coverage || 0,
            risk_stability: signal.risk_stability_coverage || 0,
            institutional_smart_money: signal.institutional_smart_money_coverage || 0,
          },
          // v3.2: Performance data from joined performance table
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
          backtest_status: perf?.status || undefined,
          backtest_last_update: perf?.last_update || undefined,
        };
      });

      setSignals(rankings);
    } catch (err) {
      console.error('Error fetching signals:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch signals');
    } finally {
      setLoading(false);
    }
  };

  // Fetch runs on mount
  useEffect(() => {
    fetchRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch signals when selectedRunId changes
  useEffect(() => {
    if (selectedRunId) {
      fetchSignals();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRunId]);

  return {
    signals,
    loading,
    error,
    refetch: fetchSignals,
    runs,
    selectedRunId,
    setSelectedRunId,
  };
}
