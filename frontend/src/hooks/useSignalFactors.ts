/**
 * Hook to fetch detailed factor data for a specific signal
 * Fetches from signals_technical, signals_fundamental, etc. tables
 */

import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/lib/supabase';

// Map group keys to their table names
const GROUP_TABLE_MAP: Record<string, string> = {
  technical: 'signals_technical',
  fundamental: 'signals_fundamental',
  news_macro: 'signals_news_macro',
  social_alternative: 'signals_social_alternative',
  risk_stability: 'signals_risk_stability',
  institutional_smart_money: 'signals_institutional_smart_money',
};

export interface FactorDetail {
  factor_key: string;
  value: number | null;
  score: number | null;
  contribution?: number;
}

interface UseSignalFactorsResult {
  factors: FactorDetail[];
  loading: boolean;
  error: string | null;
}

export function useSignalFactors(
  signalId: string | undefined,
  groupKey: string
): UseSignalFactorsResult {
  const tableName = GROUP_TABLE_MAP[groupKey];

  const { data, isLoading, error } = useQuery({
    queryKey: ['signal-factors', signalId, groupKey],
    queryFn: async () => {
      if (!signalId || !tableName) {
        console.log(`[useSignalFactors] Missing signalId or tableName:`, { signalId, tableName, groupKey });
        return [];
      }

      console.log(`[useSignalFactors] Fetching from ${tableName} for signal_id:`, signalId);

      const { data, error } = await supabase
        .from(tableName)
        .select('factors')
        .eq('signal_id', signalId)
        .single();

      if (error) {
        console.error(`[useSignalFactors] Error fetching ${groupKey} factors:`, error);
        console.error(`[useSignalFactors] Error details:`, { 
          code: error.code, 
          message: error.message, 
          details: error.details,
          hint: error.hint 
        });
        // Return empty array instead of throwing to avoid breaking UI
        return [];
      }

      console.log(`[useSignalFactors] Fetched data for ${groupKey}:`, data);

      // Parse the factors JSONB and convert to array
      const factorsObj = data?.factors || {};
      const factorArray: FactorDetail[] = Object.entries(factorsObj).map(([key, val]) => {
        // Check if value is an object with raw/normalized properties
        if (typeof val === 'object' && val !== null) {
          const objVal = val as { raw?: number; normalized?: number; percentile?: number };
          return {
            factor_key: key,
            value: objVal.raw ?? null, // Use 'raw' for actual value
            score: objVal.normalized ?? null, // Use 'normalized' for z-score
            contribution: undefined,
          };
        } else {
          // Legacy format - just a number
          return {
            factor_key: key,
            value: val as number,
            score: null,
            contribution: undefined,
          };
        }
      });

      // Sort by contribution (if available) or score, then by absolute value
      return factorArray.sort((a, b) => {
        if (a.contribution !== undefined && b.contribution !== undefined) {
          return Math.abs(b.contribution) - Math.abs(a.contribution);
        }
        if (a.score !== null && b.score !== null) {
          return Math.abs(b.score) - Math.abs(a.score);
        }
        return Math.abs(b.value || 0) - Math.abs(a.value || 0);
      });
    },
    enabled: !!signalId && !!tableName,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  return {
    factors: data || [],
    loading: isLoading,
    error: error instanceof Error ? error.message : null,
  };
}
