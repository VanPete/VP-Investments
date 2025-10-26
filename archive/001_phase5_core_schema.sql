-- ============================================================================
-- VP Investments - Phase 5 Core Schema Migration
-- ============================================================================
-- Created: 2025-10-22
-- Purpose: Create core tables for JSON → Database migration
-- Tables: 2 core + 6 group detail = 8 tables total
--
-- Design Philosophy:
-- - JSONB for all ~150 factors (maximum flexibility)
-- - Store both raw and normalized values
-- - No premature optimization (can add columns later)
-- - Direct mapping from current JSON output structure
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE TABLES (2)
-- ============================================================================

-- Main signal records (replaces JSON output)
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL, -- Foreign key added after signal_runs created
    ticker VARCHAR(10) NOT NULL,
    
    -- Overall Scores
    overall_score DECIMAL(10, 6) NOT NULL,
    total_coverage DECIMAL(5, 4) NOT NULL CHECK (total_coverage >= 0 AND total_coverage <= 1),
    
    -- Group Scores (weighted aggregations)
    technical_score DECIMAL(10, 6),
    fundamental_score DECIMAL(10, 6),
    news_macro_score DECIMAL(10, 6),
    social_alternative_score DECIMAL(10, 6),
    risk_stability_score DECIMAL(10, 6),
    institutional_smart_money_score DECIMAL(10, 6),
    
    -- Group Coverages (0-1 scale)
    technical_coverage DECIMAL(5, 4) CHECK (technical_coverage >= 0 AND technical_coverage <= 1),
    fundamental_coverage DECIMAL(5, 4) CHECK (fundamental_coverage >= 0 AND fundamental_coverage <= 1),
    news_macro_coverage DECIMAL(5, 4) CHECK (news_macro_coverage >= 0 AND news_macro_coverage <= 1),
    social_alternative_coverage DECIMAL(5, 4) CHECK (social_alternative_coverage >= 0 AND social_alternative_coverage <= 1),
    risk_stability_coverage DECIMAL(5, 4) CHECK (risk_stability_coverage >= 0 AND risk_stability_coverage <= 1),
    institutional_smart_money_coverage DECIMAL(5, 4) CHECK (institutional_smart_money_coverage >= 0 AND institutional_smart_money_coverage <= 1),
    
    -- Metadata
    rank INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_run_ticker UNIQUE(run_id, ticker)
);

-- Pipeline run metadata
CREATE TABLE IF NOT EXISTS signal_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_timestamp TIMESTAMPTZ DEFAULT NOW(),
    pipeline_version VARCHAR(20),
    
    -- Ticker counts
    total_tickers INTEGER CHECK (total_tickers >= 0),
    successful_tickers INTEGER CHECK (successful_tickers >= 0),
    failed_tickers INTEGER CHECK (failed_tickers >= 0),
    
    -- Performance
    duration_seconds DECIMAL(10, 2) CHECK (duration_seconds >= 0),
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add foreign key constraint after both tables exist
ALTER TABLE signals 
ADD CONSTRAINT fk_signals_run 
FOREIGN KEY (run_id) REFERENCES signal_runs(id) ON DELETE CASCADE;

-- ============================================================================
-- GROUP DETAIL TABLES (6)
-- ============================================================================
-- Store all ~150 individual factor values (raw + normalized)
-- JSONB format: {"factor_name": {"raw": X, "normalized": Y, "percentile": Z}}
-- ============================================================================

-- Technical indicators detail (~60 factors)
-- Stores: RSI, MACD, moving averages, volume metrics, Bollinger bands, momentum, etc.
CREATE TABLE IF NOT EXISTS signals_technical (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All technical factors with raw + normalized values
    -- Example: {"rsi_14": {"raw": 65.3, "normalized": 0.82}, "macd_value": {...}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fundamental metrics detail (~45 factors)
-- Stores: PE ratio, margins, growth rates, financial health, efficiency ratios, etc.
CREATE TABLE IF NOT EXISTS signals_fundamental (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All fundamental factors with raw + normalized values
    -- Example: {"pe_ratio": {"raw": 15.2, "normalized": 0.45}, "roe": {...}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- News & Macro sentiment (~15 factors)
-- Stores: News sentiment, earnings events, market regime, macro indicators, etc.
CREATE TABLE IF NOT EXISTS signals_news_macro (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All news/macro factors with raw + normalized values
    -- Example: {"news_sentiment": {"raw": 0.75, "normalized": 0.82}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Social metrics (~10 factors)
-- Stores: Reddit mentions, sentiment, buzz metrics, social consensus, etc.
CREATE TABLE IF NOT EXISTS signals_social_alternative (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All social/alternative factors with raw + normalized values
    -- Example: {"reddit_mentions_7d": {"raw": 145, "normalized": 0.68}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk metrics (~25 factors)
-- Stores: Volatility, beta, drawdowns, Sharpe ratio, liquidity, etc.
CREATE TABLE IF NOT EXISTS signals_risk_stability (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All risk/stability factors with raw + normalized values
    -- Example: {"volatility_30d": {"raw": 0.35, "normalized": 0.55}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Institutional activity (~20 factors)
-- Stores: Institutional ownership, insider activity, analyst ratings, price targets, etc.
CREATE TABLE IF NOT EXISTS signals_institutional_smart_money (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    
    -- All institutional factors with raw + normalized values
    -- Example: {"inst_ownership_pct": {"raw": 0.78, "normalized": 0.62}, ...}
    factors JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Core signal indexes
CREATE INDEX IF NOT EXISTS idx_signals_run_ticker ON signals(run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_overall_score ON signals(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_rank ON signals(rank ASC) WHERE rank IS NOT NULL;

-- Signal run indexes
CREATE INDEX IF NOT EXISTS idx_signal_runs_timestamp ON signal_runs(run_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_runs_status ON signal_runs(status);

-- Group detail table indexes (for JOIN performance)
CREATE INDEX IF NOT EXISTS idx_technical_signal ON signals_technical(signal_id);
CREATE INDEX IF NOT EXISTS idx_fundamental_signal ON signals_fundamental(signal_id);
CREATE INDEX IF NOT EXISTS idx_news_macro_signal ON signals_news_macro(signal_id);
CREATE INDEX IF NOT EXISTS idx_social_signal ON signals_social_alternative(signal_id);
CREATE INDEX IF NOT EXISTS idx_risk_signal ON signals_risk_stability(signal_id);
CREATE INDEX IF NOT EXISTS idx_institutional_signal ON signals_institutional_smart_money(signal_id);

-- JSONB GIN indexes for fast factor lookups (optional, add if needed)
-- Example: CREATE INDEX idx_technical_factors_gin ON signals_technical USING gin (factors);

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE signals IS 'Main signal records with group scores and coverages (Phase 4 output)';
COMMENT ON TABLE signal_runs IS 'Pipeline execution metadata and run tracking';
COMMENT ON TABLE signals_technical IS 'Technical indicators (~60 factors): RSI, MACD, volume, momentum';
COMMENT ON TABLE signals_fundamental IS 'Fundamental metrics (~45 factors): PE ratio, margins, growth, health';
COMMENT ON TABLE signals_news_macro IS 'News/macro factors (~15 factors): sentiment, earnings, market regime';
COMMENT ON TABLE signals_social_alternative IS 'Social metrics (~10 factors): Reddit, buzz, consensus';
COMMENT ON TABLE signals_risk_stability IS 'Risk metrics (~25 factors): volatility, beta, drawdown, Sharpe';
COMMENT ON TABLE signals_institutional_smart_money IS 'Institutional factors (~20 factors): ownership, insiders, analysts';

-- Note: Each detail table has a 'factors' JSONB column with structure:
-- {"factor_name": {"raw": value, "normalized": score, "percentile": rank}}

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Total tables created: 8 (2 core + 6 group detail)
-- Ready for Phase 5 implementation
-- ============================================================================
