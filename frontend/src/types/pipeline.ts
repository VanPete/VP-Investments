// VP Investments Pipeline Types - TypeScript definitions for pipeline results

/**
 * Pipeline result metadata
 */
export interface PipelineMetadata {
  timestamp: string;
  total_tickers: number;
  source: string;
  discovery?: {
    reddit_tickers: number;
    news_tickers: number;
    total_universe: number;
  };
}

/**
 * Group scores and coverages for a single ticker
 */
export interface GroupScores {
  technical: number;
  fundamental: number;
  news_macro: number;
  social_alternative: number;
  risk_stability: number;
  institutional_smart_money: number;
}

export interface GroupCoverages {
  technical: number;
  fundamental: number;
  news_macro: number;
  social_alternative: number;
  risk_stability: number;
  institutional_smart_money: number;
}

/**
 * Single ticker ranking result
 */
export interface SignalRanking {
  rank: number;
  ticker: string;
  company_name?: string;
  current_price?: number;
  overall_score: number;
  total_coverage: number;
  group_scores: GroupScores;
  group_coverages: GroupCoverages;
  // Backtest performance tracking (Phase 6)
  backtest_baseline_price?: number;
  backtest_baseline_date?: string;
  return_1d?: number;
  return_3d?: number;
  return_7d?: number;
  return_10d?: number;
  return_14d?: number;
  return_30d?: number;
  return_90d?: number;
  spy_return_1d?: number;
  spy_return_3d?: number;
  spy_return_7d?: number;
  spy_return_10d?: number;
  spy_return_14d?: number;
  spy_return_30d?: number;
  spy_return_90d?: number;
  backtest_status?: string;
  backtest_last_update?: string;
}

/**
 * Complete pipeline results file structure
 */
export interface PipelineResults {
  metadata: PipelineMetadata;
  rankings: SignalRanking[];
}

/**
 * Group weight configuration
 */
export interface GroupWeights {
  technical: number;
  fundamental: number;
  news_macro: number;
  social_alternative: number;
  risk_stability: number;
  institutional_smart_money: number;
}

/**
 * Factor weights for a single group (key-value pairs)
 */
export type FactorWeights = Record<string, number>;

/**
 * Complete weights configuration
 */
export interface WeightsConfig {
  group_weights: GroupWeights;
  factor_weights_technical: FactorWeights;
  factor_weights_fundamental: FactorWeights;
  factor_weights_news_macro: FactorWeights;
  factor_weights_social_alternative: FactorWeights;
  factor_weights_risk_stability: FactorWeights;
  factor_weights_institutional_smart_money: FactorWeights;
}

/**
 * Factor description from factor_to_group.yaml
 */
export type FactorDescriptions = Record<string, string>;

/**
 * Factor to group mapping
 */
export interface FactorToGroup {
  technical: FactorDescriptions;
  fundamental: FactorDescriptions;
  news_macro: FactorDescriptions;
  social_alternative: FactorDescriptions;
  risk_stability: FactorDescriptions;
  institutional_smart_money: FactorDescriptions;
}

/**
 * Methodology phase information
 */
export interface MethodologyPhase {
  name: string;
  description: string;
  formula?: string;
  status: 'operational' | 'coming_soon';
  note?: string;
}

/**
 * Group methodology information
 */
export interface GroupMethodology {
  name: string;
  weight: number;
  description: string;
  key_factors: string[];
}

/**
 * Complete methodology configuration
 */
export interface MethodologyConfig {
  overview: {
    title: string;
    description: string;
    key_principles: string[];
  };
  phases: Record<string, MethodologyPhase>;
  scoring: {
    normalization: {
      method: string;
      description: string;
      advantages: string[];
    };
    factor_weighting: {
      description: string;
      example: string;
    };
    group_weighting: {
      description: string;
      current_weights: GroupWeights;
      rationale: string;
    };
  };
  groups: Record<string, GroupMethodology>;
  interpretation: {
    overall_score: {
      description: string;
      ranges: Record<string, string>;
      notes: string;
    };
    group_scores: {
      description: string;
      interpretation: string;
    };
    coverage: {
      description: string;
      guidance: Record<string, string>;
      notes: string;
    };
  };
  data_sources: Record<string, {
    source: string;
    description: string;
    update_frequency: string;
  }>;
  limitations: string[];
  version: {
    current: string;
    last_updated: string;
    changes: string[];
  };
}

/**
 * File selector option
 */
export interface FileOption {
  filename: string;
  timestamp: string;
  label: string;
}

/**
 * Filter state for dashboard
 */
export interface FilterState {
  minScore: number;
  maxScore: number;
  minCoverage: number;
}

/**
 * Sort configuration for table columns
 */
export type SortDirection = 'asc' | 'desc' | null;

export interface SortConfig {
  column: string;
  direction: SortDirection;
}

/**
 * Group names for type safety
 */
export type GroupKey = keyof GroupScores;

export const GROUP_KEYS: GroupKey[] = [
  'technical',
  'fundamental',
  'news_macro',
  'social_alternative',
  'risk_stability',
  'institutional_smart_money',
];

/**
 * Display names for groups
 */
export const GROUP_DISPLAY_NAMES: Record<GroupKey, string> = {
  technical: 'Technical',
  fundamental: 'Fundamental',
  news_macro: 'News & Macro',
  social_alternative: 'Social & Alternative',
  risk_stability: 'Risk & Stability',
  institutional_smart_money: 'Institutional & Smart Money',
};
