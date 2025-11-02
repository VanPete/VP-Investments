-- Migration 020f: Remove avg_composite_score and avg_confidence columns
-- These are redundant - avg_composite_score duplicates avg_overall_score
-- and avg_confidence isn't used anywhere

-- Remove avg_composite_score and avg_confidence columns
ALTER TABLE analytics DROP COLUMN IF EXISTS avg_composite_score;
ALTER TABLE analytics DROP COLUMN IF EXISTS avg_confidence;
