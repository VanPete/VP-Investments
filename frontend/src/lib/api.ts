// VP Investments API Client - Axios configuration for backend integration

import axios, { AxiosError } from 'axios';
import { ApiError } from '@/types/api';

// Create axios instance with base configuration
export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds for analysis operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling and data extraction
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response; // Return full response for proper typing
  },
  async (error: AxiosError) => {
    console.error(`API Error: ${error.response?.status} ${error.config?.url}`);
    
    // Handle different error types
    if (error.response?.status === 500) {
      // Server error - could retry
      console.log('Server error - consider retry logic');
    } else if (error.response?.status === 404) {
      // Not found - probably expected
      console.log('Resource not found');
    } else if (error.code === 'ECONNABORTED') {
      // Timeout
      console.log('Request timeout');
    }
    
    return Promise.reject({
      message: error.message,
      code: error.response?.status,
      details: error.response?.data
    } as ApiError);
  }
);

// API endpoint constants
export const API_ENDPOINTS = {
  // New Pipeline Data Endpoints (Phase 2)
  DATA_FETCH: '/api/data/fetch',
  DATA_SOURCES: '/api/data/sources',
  DATA_TICKERS: '/api/data/tickers',
  DATA_DASHBOARD: '/api/data/dashboard',
  
  // Trading signals (Legacy)
  SIGNALS: '/api/signals',
  SIGNALS_LATEST: '/api/signals/latest',
  SIGNALS_STATS: '/api/signals/stats',
  
  // Recommendations
  RECOMMENDATIONS: '/api/recommendations',
  
  // System health
  HEALTH: '/health',
  
  // Analysis runs
  ANALYSIS_RUNS: '/api/runs',
} as const;

// Helper function for error handling
export const handleApiError = (error: ApiError): string => {
  if (error.code === 503) {
    return 'Service temporarily unavailable. Please try again later.';
  } else if (error.code === 500) {
    return 'Server error occurred. Please contact support if this persists.';
  } else if (error.code === 404) {
    return 'Requested data not found.';
  } else if (error.message.includes('timeout')) {
    return 'Request timed out. Please check your connection and try again.';
  } else {
    return error.message || 'An unexpected error occurred.';
  }
};

export default apiClient;