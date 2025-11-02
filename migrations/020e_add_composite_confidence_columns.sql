-- Migration 020e: Add avg_composite_score and avg_confidence columns
-- Date: 2025-11-02
-- Purpose: Add composite score and confidence tracking to analytics table

BEGIN;

-- Add avg_composite_score and avg_confidence columns
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_composite_score NUMERIC;
ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_confidence NUMERIC;

-- Add comments for documentation
COMMENT ON COLUMN analytics.avg_composite_score IS 'Average composite score across all signals for this interval';
COMMENT ON COLUMN analytics.avg_confidence IS 'Average confidence level across all signals for this interval';

COMMIT;
