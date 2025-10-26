import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { SignalRanking } from '@/types/pipeline';

interface SignalRun {
  id: string;
  run_timestamp: string;
  total_tickers: number;
  status: string;
}

interface CoverageData {
  technical: number;
  fundamental: number;
  news_macro: number;
  social_alternative: number;
  risk_stability: number;
  institutional_smart_money: number;
  total: number;
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

  // Helper function to fetch coverage from detail tables
  const fetchCoverageForSignals = async (signalIds: string[]): Promise<Record<string, CoverageData>> => {
    const coverageMap: Record<string, CoverageData> = {};

    if (signalIds.length === 0) return coverageMap;

    // Fetch all detail tables in parallel
    const [technical, fundamental, newsMacro, social, risk, institutional] = await Promise.all([
      supabase.from('signals_technical').select('signal_id, factors').in('signal_id', signalIds),
      supabase.from('signals_fundamental').select('signal_id, factors').in('signal_id', signalIds),
      supabase.from('signals_news_macro').select('signal_id, factors').in('signal_id', signalIds),
      supabase.from('signals_social_alternative').select('signal_id, factors').in('signal_id', signalIds),
      supabase.from('signals_risk_stability').select('signal_id, factors').in('signal_id', signalIds),
      supabase.from('signals_institutional_smart_money').select('signal_id, factors').in('signal_id', signalIds),
    ]);

    // Calculate coverage for each signal
    signalIds.forEach(signalId => {
      const techFactors = technical.data?.find(t => t.signal_id === signalId)?.factors || {};
      const fundFactors = fundamental.data?.find(f => f.signal_id === signalId)?.factors || {};
      const newsFactors = newsMacro.data?.find(n => n.signal_id === signalId)?.factors || {};
      const socialFactors = social.data?.find(s => s.signal_id === signalId)?.factors || {};
      const riskFactors = risk.data?.find(r => r.signal_id === signalId)?.factors || {};
      const instFactors = institutional.data?.find(i => i.signal_id === signalId)?.factors || {};

      // Count factors per group
      const techCov = Object.keys(techFactors).length;
      const fundCov = Object.keys(fundFactors).length;
      const newsCov = Object.keys(newsFactors).length;
      const socialCov = Object.keys(socialFactors).length;
      const riskCov = Object.keys(riskFactors).length;
      const instCov = Object.keys(instFactors).length;

      // Define max factors per group (from config/factor_to_group.yaml - actual counts)
      const MAX_FACTORS = {
        technical: 41,           // 41 technical indicators
        fundamental: 45,         // 45 fundamental metrics
        news_macro: 18,          // 18 news/macro factors
        social_alternative: 10,  // 10 social sentiment factors
        risk_stability: 23,      // 23 risk/stability factors
        institutional_smart_money: 21, // 21 institutional factors
      };

      // Convert counts to 0-1 scale (percentage coverage)
      coverageMap[signalId] = {
        technical: Math.min(techCov / MAX_FACTORS.technical, 1),
        fundamental: Math.min(fundCov / MAX_FACTORS.fundamental, 1),
        news_macro: Math.min(newsCov / MAX_FACTORS.news_macro, 1),
        social_alternative: Math.min(socialCov / MAX_FACTORS.social_alternative, 1),
        risk_stability: Math.min(riskCov / MAX_FACTORS.risk_stability, 1),
        institutional_smart_money: Math.min(instCov / MAX_FACTORS.institutional_smart_money, 1),
        total: Math.min(
          (techCov + fundCov + newsCov + socialCov + riskCov + instCov) / 
          (MAX_FACTORS.technical + MAX_FACTORS.fundamental + MAX_FACTORS.news_macro + 
           MAX_FACTORS.social_alternative + MAX_FACTORS.risk_stability + MAX_FACTORS.institutional_smart_money),
          1
        ),
      };
    });

    return coverageMap;
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
      // v3.2: Performance data moved to performance table
      const { data, error: fetchError } = await supabase
        .from('signals')
        .select(`
          id,
          ticker,
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
        .eq('run_id', selectedRunId)
        .order('overall_score', { ascending: false });

      if (fetchError) {
        console.error('Supabase error details:', fetchError);
        throw fetchError;
      }

      console.log('Successfully fetched signals:', data?.length || 0);

      // Fetch coverage data for all signals
      const signalIds = data?.map(s => s.id) || [];
      const coverageData = await fetchCoverageForSignals(signalIds);

      // Transform Supabase data to SignalRanking format
      const rankings: SignalRanking[] = (data || []).map((signal, index) => {
        const coverage = coverageData[signal.id] || {
          technical: 0,
          fundamental: 0,
          news_macro: 0,
          social_alternative: 0,
          risk_stability: 0,
          institutional_smart_money: 0,
          total: 0,
        };

        return {
          rank: signal.rank || index + 1,
          ticker: signal.ticker,
          overall_score: signal.overall_score || 0,
          total_coverage: coverage.total,
          group_scores: {
            technical: signal.technical_score || 0,
            fundamental: signal.fundamental_score || 0,
            news_macro: signal.news_macro_score || 0,
            social_alternative: signal.social_alternative_score || 0,
            risk_stability: signal.risk_stability_score || 0,
            institutional_smart_money: signal.institutional_smart_money_score || 0,
          },
          group_coverages: {
            technical: coverage.technical,
            fundamental: coverage.fundamental,
            news_macro: coverage.news_macro,
            social_alternative: coverage.social_alternative,
            risk_stability: coverage.risk_stability,
            institutional_smart_money: coverage.institutional_smart_money,
          },
          // v3.2: Performance data moved to separate performance table
          // These fields are optional and will be undefined until joined with performance data
          backtest_baseline_price: undefined,
          backtest_baseline_date: undefined,
          return_1d: undefined,
          return_3d: undefined,
          return_7d: undefined,
          return_10d: undefined,
          return_14d: undefined,
          return_30d: undefined,
          return_90d: undefined,
          spy_return_1d: undefined,
          spy_return_3d: undefined,
          spy_return_7d: undefined,
          spy_return_10d: undefined,
          spy_return_14d: undefined,
          spy_return_30d: undefined,
          spy_return_90d: undefined,
          backtest_status: undefined,
          backtest_last_update: undefined,
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
