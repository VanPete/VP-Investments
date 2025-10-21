// VP Investments - Trading Signals Hook
// React Query hook for fetching and managing trading signals

'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, API_ENDPOINTS, handleApiError } from '@/lib/api';
import { TradingSignal, DashboardFilters } from '@/types/api';

// Fetch all trading signals
export const useSignals = (filters?: DashboardFilters) => {
  return useQuery({
    queryKey: ['signals', filters],
    queryFn: async (): Promise<TradingSignal[]> => {
      const params = new URLSearchParams();
      
      if (filters?.signal_type && filters.signal_type !== 'ALL') {
        params.append('signal_type', filters.signal_type);
      }
      if (filters?.min_confidence) {
        params.append('min_confidence', filters.min_confidence.toString());
      }
      if (filters?.tickers?.length) {
        params.append('tickers', filters.tickers.join(','));
      }
      
      const response = await apiClient.get<TradingSignal[]>(
        `${API_ENDPOINTS.SIGNALS}?${params}`
      );
      return response.data || [];
    },
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Data considered fresh for 15 seconds
    retry: 2,
    retryDelay: 1000,
  });
};

// Fetch latest signals (most recent)
export const useLatestSignals = (limit: number = 10) => {
  return useQuery({
    queryKey: ['signals', 'latest', limit],
    queryFn: async (): Promise<TradingSignal[]> => {
      const response = await apiClient.get<TradingSignal[]>(
        `${API_ENDPOINTS.SIGNALS_LATEST}?limit=${limit}`
      );
      return response.data || [];
    },
    refetchInterval: 10000, // More frequent for latest signals
    staleTime: 5000,
    retry: 3,
  });
};

// Fetch single signal by ID (if your API supports it)
export const useSignal = (signalId: string) => {
  return useQuery({
    queryKey: ['signal', signalId],
    queryFn: async (): Promise<TradingSignal> => {
      const response = await apiClient.get<TradingSignal>(
        `${API_ENDPOINTS.SIGNALS}/${signalId}`
      );
      return response.data;
    },
    enabled: !!signalId,
    retry: 1,
  });
};

// Manual refresh mutation (for user-triggered updates)
export const useRefreshSignals = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (): Promise<void> => {
      // Trigger fresh analysis run
      await apiClient.post('/api/runs/start', {
        force_refresh: true
      });
    },
    onSuccess: () => {
      // Invalidate and refetch all related data
      queryClient.invalidateQueries({ queryKey: ['signals'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
      queryClient.invalidateQueries({ queryKey: ['tickers'] });
    },
    onError: (error) => {
      console.error('Failed to refresh signals:', handleApiError(error));
    },
  });
};

// Note: useSignalsStats has been moved to usePipelineData.ts to avoid duplication