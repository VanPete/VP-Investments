export interface StockData {
  rank: number;
  ticker: string;
  company: string;
  trade_type: 'Swing' | 'Long-Term' | 'Balanced';
  sector?: string;
  run_datetime: string;
  source: string;
  subreddit?: string;
  raw_score: number;
  weighted_score: number;
  active_signals: string;
  mentions: number;
  upvotes: number;
  sentiment: number;
  post_recency?: number;
  post_score?: number;
  comment_sentiment?: number;
  title?: string;
  current_price: number;
  price_1d: number;
  price_7d: number;
  volume: number;
  market_cap?: number;
  volume_spike_ratio?: number;
  rel_strength?: number;
  ma_50?: number;
  ma_200?: number;
  rsi?: number;
  macd?: number;
  bollinger?: number;
  volatility?: number;
  pe_ratio?: number;
  eps_growth?: number;
  roe?: number;
  debt_equity?: number;
  fcf_margin?: number;
  momentum_tag?: string;
}

export interface DashboardStats {
  totalStocks: number;
  avgWeightedScore: number;
  topScore: number;
  avgSentiment: number;
  totalMentions: number;
  tradeTypeBreakdown: Record<string, number>;
}
