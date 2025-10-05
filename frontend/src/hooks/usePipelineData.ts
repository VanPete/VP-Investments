// VP Investments - Pipeline Data Hooks
// React Query hooks for the new pipeline data endpoints (Phase 2)

'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, API_ENDPOINTS } from '@/lib/api';
import { 
  DashboardData, 
  TickerData, 
  DataSourcesResponse, 
  PipelineDataResponse,
  SignalsStatsResponse
} from '@/types/api';

// Fetch dashboard data (main endpoint for homepage)
export const useDashboardData = (tickers?: string) => {
  return useQuery({
    queryKey: ['dashboard', tickers],
    queryFn: async (): Promise<DashboardData> => {
      const params = new URLSearchParams();
      if (tickers) {
        params.append('tickers', tickers);
      }
      
      const response = await apiClient.get<DashboardData>(
        `${API_ENDPOINTS.DATA_DASHBOARD}?${params}`
      );
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 15000, // Data considered fresh for 15 seconds
    retry: 2,
    retryDelay: 1000,
  });
};

// Fetch ticker data (for individual ticker display)
export const useTickerData = (tickers?: string) => {
  return useQuery({
    queryKey: ['tickers', tickers],
    queryFn: async (): Promise<TickerData[]> => {
      const params = new URLSearchParams();
      if (tickers) {
        params.append('tickers', tickers);
      }
      
      const response = await apiClient.get<TickerData[]>(
        `${API_ENDPOINTS.DATA_TICKERS}?${params}`
      );
      return response.data;
    },
    refetchInterval: 20000, // Refresh every 20 seconds for ticker data
    staleTime: 10000,
    retry: 2,
    enabled: !!tickers, // Only run if tickers are provided
  });
};

// Fetch data sources status
export const useDataSources = () => {
  return useQuery({
    queryKey: ['data-sources'],
    queryFn: async (): Promise<DataSourcesResponse> => {
      const response = await apiClient.get<DataSourcesResponse>(
        API_ENDPOINTS.DATA_SOURCES
      );
      return response.data;
    },
    refetchInterval: 60000, // Refresh every minute (less frequent)
    staleTime: 30000,
    retry: 2,
  });
};

// Fetch fresh pipeline data (on-demand data collection)
export const useFetchPipelineData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (tickers: string[]): Promise<PipelineDataResponse> => {
      const response = await apiClient.post<PipelineDataResponse>(
        API_ENDPOINTS.DATA_FETCH,
        { tickers }
      );
      return response.data;
    },
    onSuccess: () => {
      // Invalidate and refetch dashboard data after successful pipeline fetch
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
      queryClient.invalidateQueries({ queryKey: ['tickers'] });
    },
  });
};

// Fetch signals stats (for the existing dashboard)
export const useSignalsStats = () => {
  return useQuery({
    queryKey: ['signals-stats'],
    queryFn: async (): Promise<SignalsStatsResponse> => {
      const response = await apiClient.get<SignalsStatsResponse>(
        API_ENDPOINTS.SIGNALS_STATS
      );
      return response.data;
    },
    refetchInterval: 30000,
    staleTime: 15000,
    retry: 2,
  });
};

// Health check
export const useHealthCheck = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<{ status: string; message?: string }> => {
      const response = await apiClient.get(API_ENDPOINTS.HEALTH);
      return response.data;
    },
    refetchInterval: 60000, // Check health every minute
    staleTime: 30000,
    retry: 1, // Don't retry health checks too much
  });
};