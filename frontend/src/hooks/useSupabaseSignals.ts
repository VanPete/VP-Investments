import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { SignalRanking } from '@/types/pipeline';

interface UseSupabaseSignalsResult {
  signals: SignalRanking[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useSupabaseSignals(): UseSupabaseSignalsResult {
  const [signals, setSignals] = useState<SignalRanking[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);

      const { data, error: fetchError } = await supabase
        .from('signals')
        .select(`
          ticker,
          signal_score,
          technical_score,
          fundamental_score,
          news_macro_score,
          social_score,
          risk_score,
          institutional_score,
          technical_coverage,
          fundamental_coverage,
          news_macro_coverage,
          social_coverage,
          risk_coverage,
          institutional_coverage,
          total_coverage,
          backtest_baseline_price,
          backtest_baseline_date,
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
          backtest_status,
          backtest_last_update
        `)
        .order('signal_score', { ascending: false });

      if (fetchError) {
        throw fetchError;
      }

      // Transform Supabase data to SignalRanking format
      const rankings: SignalRanking[] = (data || []).map((signal, index) => ({
        rank: index + 1,
        ticker: signal.ticker,
        overall_score: signal.signal_score || 0,
        total_coverage: signal.total_coverage || 0,
        group_scores: {
          technical: signal.technical_score || 0,
          fundamental: signal.fundamental_score || 0,
          news_macro: signal.news_macro_score || 0,
          social_alternative: signal.social_score || 0,
          risk_stability: signal.risk_score || 0,
          institutional_smart_money: signal.institutional_score || 0,
        },
        group_coverages: {
          technical: signal.technical_coverage || 0,
          fundamental: signal.fundamental_coverage || 0,
          news_macro: signal.news_macro_coverage || 0,
          social_alternative: signal.social_coverage || 0,
          risk_stability: signal.risk_coverage || 0,
          institutional_smart_money: signal.institutional_coverage || 0,
        },
        // Backtest data (Phase 6)
        backtest_baseline_price: signal.backtest_baseline_price,
        backtest_baseline_date: signal.backtest_baseline_date,
        return_1d: signal.return_1d,
        return_3d: signal.return_3d,
        return_7d: signal.return_7d,
        return_10d: signal.return_10d,
        return_14d: signal.return_14d,
        return_30d: signal.return_30d,
        return_90d: signal.return_90d,
        spy_return_1d: signal.spy_return_1d,
        spy_return_3d: signal.spy_return_3d,
        spy_return_7d: signal.spy_return_7d,
        spy_return_10d: signal.spy_return_10d,
        spy_return_14d: signal.spy_return_14d,
        spy_return_30d: signal.spy_return_30d,
        spy_return_90d: signal.spy_return_90d,
        backtest_status: signal.backtest_status,
        backtest_last_update: signal.backtest_last_update,
      }));

      setSignals(rankings);
    } catch (err) {
      console.error('Error fetching signals:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch signals');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
  }, []);

  return {
    signals,
    loading,
    error,
    refetch: fetchSignals,
  };
}
