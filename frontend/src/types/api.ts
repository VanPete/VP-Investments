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

// Performance Tab Types (Phase 6 Integration)
export interface PerformanceHorizon {
  interval: string;
  days: number;
  status: 'complete' | 'in_progress' | 'pending';
  ticker_return: number | null;
  spy_return: number | null;
  qqq_return: number | null;
  sector_return: number | null;
  alpha_vs_spy: number | null;
  alpha_vs_qqq: number | null;
  alpha_vs_sector: number | null;
  eligible_at: string;
  hours_remaining: number | null;
}

export interface PerformanceData {
  signal_id: string;
  ticker: string;
  baseline_date: string;
  baseline_price: number;
  market_cap: number | null;
  beta: number | null;
  sector: string | null;
  overall_score: number;
  intervals_completed: number[];
  horizons: PerformanceHorizon[];
}

// Analytics Tab Types (Phase 7 Integration)
export interface ScoreBucketMetrics {
  avg_return: number;
  win_rate: number;
  sharpe: number;
  max: number;
  min: number;
  count: number;
}

export interface ScoreBucketPerformance {
  threshold: string;
  count: number;
  [key: string]: ScoreBucketMetrics | string | number; // Allows dynamic interval keys
}

export interface FactorCorrelation {
  factor1: string;
  factor2: string;
  correlation: number;
}

export interface FactorContribution {
  factor: string;
  group: string;
  correlation: number;
  abs_correlation: number;
}

export interface FactorContributions {
  [interval: string]: {
    top_contributors: FactorContribution[];
    all_correlations: FactorContribution[];
  };
}

export interface GroupPerformance {
  per_signal_analysis: {
    dominant_group_distribution: { [key: string]: number };
    avg_return_by_dominant_group: { [key: string]: number };
  };
  aggregated_analysis: {
    [group: string]: {
      avg_score: number;
      correlation_with_returns: { [interval: string]: number };
      signals_count: number;
    };
  };
}

export interface BacktestDataPoint {
  date: string;
  vp_strategy: number;
  spy: number;
  qqq: number;
}

export interface BacktestReturns {
  start_date: string;
  end_date: string;
  daily_returns: BacktestDataPoint[];
  summary: {
    vp_total_return: number;
    spy_total_return: number;
    qqq_total_return: number;
    vp_sharpe: number;
    vp_max_drawdown: number;
    vp_win_rate: number;
  };
}

export interface AnalyticsData {
  run_id: string;
  created_at: string;
  total_signals: number;
  signals_analyzed: number;
  avg_overall_score: number;
  avg_technical_score: number;
  avg_fundamental_score: number;
  avg_news_macro_score: number;
  avg_social_alternative_score: number;
  avg_risk_stability_score: number;
  avg_institutional_score: number;
  top_sector: string | null;
  top_sector_avg_return: number | null;
  sector_performance: { [sector: string]: { avg_return: number; count: number; win_rate: number } };
  score_bucket_performance: { [bucket: string]: ScoreBucketPerformance };
  factor_correlations: {
    group_correlations: {
      matrix: number[][];
      labels: string[];
    };
    top_positive_pairs: FactorCorrelation[];
    top_negative_pairs: FactorCorrelation[];
  };
  factor_contributions: FactorContributions;
  group_performance: GroupPerformance;
  backtest_cumulative_returns: BacktestReturns;
  top_factors: { [group: string]: any[] };
}