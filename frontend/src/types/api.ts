// VP Investments API Types - TypeScript definitions for backend integration

export interface TradingSignal {
  ticker: string;
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  technical_score: number;
  sentiment_score: number;
  combined_score: number;
  created_at: string;
}

export interface AnalysisRun {
  id: string;
  start_time: string;
  end_time: string | null;
  status: 'running' | 'completed' | 'failed';
  signals_count: number;
}

export interface TradingRecommendation {
  ticker: string;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  price_target?: number;
  risk_assessment: string;
}

export interface PortfolioPosition {
  ticker: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  gain_loss: number;
  gain_loss_percent: number;
}

export interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
  timestamp?: string;
}

export interface ApiError {
  message: string;
  code?: number;
  details?: string;
}

// Chart data types
export interface PriceDataPoint {
  date: string;
  price: number;
  volume?: number;
}

export interface SignalHistoryPoint {
  date: string;
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'signal_update' | 'price_update' | 'system_status';
  data: Record<string, unknown>;
  timestamp: string;
}

// UI State types
export interface DashboardFilters {
  signal_type?: 'BUY' | 'SELL' | 'HOLD' | 'ALL';
  min_confidence?: number;
  tickers?: string[];
  date_range?: {
    start: string;
    end: string;
  };
}

export interface SortOptions {
  field: 'ticker' | 'confidence' | 'created_at' | 'signal_type';
  direction: 'asc' | 'desc';
}

// New Pipeline Data Types (Phase 2)
export interface DataSourceSummary {
  name: string;
  success: boolean;
  execution_time: number;
  data_count: number;
  error_message?: string;
}

export interface PipelineDataResponse {
  summary: {
    total_sources: number;
    successful_sources: number;
    failed_sources: number;
    success_rate: number;
    total_execution_time: number;
    timestamp: string;
    tickers_requested: number;
  };
  sources: DataSourceSummary[];
  data: Record<string, unknown>;
  timestamp: string;
  execution_time: number;
}

export interface TickerData {
  ticker: string;
  company_name?: string;
  current_price?: number;
  price_change?: number;
  price_change_percent?: number;
  volume?: number;
  recommendation?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
  last_updated: string;
}

export interface DashboardData {
  tickers: TickerData[];
  system_status: {
    api_online: boolean;
    pipeline_online: boolean;
    data_sources: Array<{
      name: string;
      enabled: boolean;
      status: string;
    }>;
    last_updated: string;
  };
  summary: {
    total_tickers: number;
    buy_recommendations: number;
    sell_recommendations: number;
    hold_recommendations: number;
  };
}

export interface DataSourcesResponse {
  sources: Array<{
    name: string;
    enabled: boolean;
    description: string;
  }>;
  total_sources: number;
  enabled_sources: number;
}

export interface SignalsStatsResponse {
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
  avg_confidence: number;
}