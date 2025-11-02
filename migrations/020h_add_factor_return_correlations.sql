-- Migration 020h: Add factor_return_correlations column
-- Date: 2025-11-02
-- Purpose: Store correlations between each of 158 factors and returns for ML phase

-- Add factor_return_correlations JSONB column to analytics table
ALTER TABLE analytics 
ADD COLUMN IF NOT EXISTS factor_return_correlations JSONB;

-- Add comment explaining structure
COMMENT ON COLUMN analytics.factor_return_correlations IS 
'Factor-return correlations grouped by factor group. Structure: 
{
  "technical": [
    {"factor": "rsi_14", "correlation": 0.34, "p_value": 0.001, "n": 200, "confidence": "high"},
    ...
  ],
  "fundamental": [...],
  ...
}
Used for ML feature importance analysis.';

-- Create index for JSONB queries (optional, for future queries)
CREATE INDEX IF NOT EXISTS idx_analytics_factor_correlations 
ON analytics USING gin(factor_return_correlations);
