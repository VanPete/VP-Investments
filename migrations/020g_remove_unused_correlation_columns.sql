-- Migration 020g: Remove rolling_sharpe_30d, signal_correlations, top_positive_pairs, top_negative_pairs
-- These are either not needed yet or wrong approach for ML

-- Remove columns that aren't useful in current phase
ALTER TABLE analytics DROP COLUMN IF EXISTS rolling_sharpe_30d;
ALTER TABLE analytics DROP COLUMN IF EXISTS signal_correlations;
ALTER TABLE analytics DROP COLUMN IF EXISTS top_positive_pairs;
ALTER TABLE analytics DROP COLUMN IF EXISTS top_negative_pairs;
