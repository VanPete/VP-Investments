-- ============================================================================
-- VP INVESTMENTS 3.0 NUCLEAR MIGRATION
-- ============================================================================
-- WARNING: This migration DELETES all tables except company_tickers and 
-- guardrails_config, then creates the new 3.0 schema from scratch.
--
-- BACKUP STATUS: ✅ Pre-3.0 code pushed to GitHub (commit: 1b5e517)
-- EXECUTION DATE: 2025-10-13
-- ARCHITECTURE: 3.0 Signal Grouping with Phase 1 Caching
-- ============================================================================

-- ============================================================================
-- PART 1: NUCLEAR CLEANUP (DELETE EVERYTHING EXCEPT REFERENCE TABLES)
-- ============================================================================

-- Drop all views first
DROP MATERIALIZED VIEW IF EXISTS signals_norm CASCADE;
DROP VIEW IF EXISTS signal_summary CASCADE;
DROP VIEW IF EXISTS performance_summary CASCADE;

-- Drop all tables EXCEPT company_tickers and guardrails_config
DROP TABLE IF EXISTS signal_performance CASCADE;
DROP TABLE IF EXISTS ai_strategies CASCADE;
DROP TABLE IF EXISTS signals CASCADE;
DROP TABLE IF EXISTS runs CASCADE;
DROP TABLE IF EXISTS prices CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS labels CASCADE;
DROP TABLE IF EXISTS metrics CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;

-- NOTE: company_tickers and guardrails_config are PRESERVED

-- ============================================================================
-- PART 2: CREATE NEW 3.0 SCHEMA
-- ============================================================================

-- ----------------------------------------------------------------------------
-- TABLE: runs (Fresh schema for 3.0)
-- ----------------------------------------------------------------------------
CREATE TABLE runs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    run_type TEXT NOT NULL CHECK (run_type IN ('discovery', 'targeted', 'backtest', 'scheduled')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_signals INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    phase_1_cache_hits INTEGER DEFAULT 0,
    phase_1_cache_misses INTEGER DEFAULT 0,
    phase_1_duration_ms INTEGER,
    phase_2_duration_ms INTEGER,
    phase_3_duration_ms INTEGER,
    phase_4_duration_ms INTEGER,
    phase_5_duration_ms INTEGER,
    phase_6_duration_ms INTEGER,
    total_duration_ms INTEGER,
    tickers_processed TEXT[],
    metadata JSONB DEFAULT '{}',
    error_log TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_runs_run_id ON runs(run_id);
CREATE INDEX idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX idx_runs_status ON runs(status);

COMMENT ON TABLE runs IS 'Pipeline execution tracking with 3.0 phase timings';

-- ----------------------------------------------------------------------------
-- TABLE: data_cache (Phase 1 Caching Layer)
-- ----------------------------------------------------------------------------
CREATE TABLE data_cache (
    id BIGSERIAL PRIMARY KEY,
    cache_key TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    data_group TEXT NOT NULL CHECK (data_group IN ('technical', 'fundamental', 'news_macro', 'social_alternative', 'risk_stability', 'institutional_smart_money')),
    provider TEXT NOT NULL CHECK (provider IN ('yfinance', 'reddit', 'openai', 'news_api', 'stocktwits', 'future_financial_api')),
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    version TEXT NOT NULL DEFAULT 'v1',
    endpoint TEXT,
    response_time_ms INTEGER,
    rate_limit_remaining INTEGER,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cache_key ON data_cache(cache_key);
CREATE INDEX idx_cache_ticker ON data_cache(ticker);
CREATE INDEX idx_cache_group_provider ON data_cache(data_group, provider);
CREATE INDEX idx_cache_expires ON data_cache(expires_at);
CREATE INDEX idx_cache_fetched ON data_cache(fetched_at DESC);

COMMENT ON TABLE data_cache IS 'Phase 1 cache for all external API data - prevents mid-pipeline API calls';
COMMENT ON COLUMN data_cache.cache_key IS 'Format: cache:{date}:{ticker}:{group}:{provider}:{version}';
COMMENT ON COLUMN data_cache.payload IS 'Raw JSON response from external API';
COMMENT ON COLUMN data_cache.metadata IS 'Provenance: endpoint, params, warnings, fetch duration';

-- ----------------------------------------------------------------------------
-- TABLE: signals (Core Aggregation Table - Minimal Columns)
-- ----------------------------------------------------------------------------
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    company TEXT,
    sector TEXT,
    industry TEXT,
    
    -- CORE AGGREGATED SCORES (0.0-1.0)
    signal_score NUMERIC CHECK (signal_score >= 0 AND signal_score <= 1),
    technical_score NUMERIC CHECK (technical_score >= 0 AND technical_score <= 1),
    fundamental_score NUMERIC CHECK (fundamental_score >= 0 AND fundamental_score <= 1),
    news_macro_score NUMERIC CHECK (news_macro_score >= 0 AND news_macro_score <= 1),
    social_alternative_score NUMERIC CHECK (social_alternative_score >= 0 AND social_alternative_score <= 1),
    risk_stability_score NUMERIC CHECK (risk_stability_score >= 0 AND risk_stability_score <= 1),
    institutional_smart_money_score NUMERIC CHECK (institutional_smart_money_score >= 0 AND institutional_smart_money_score <= 1),
    
    -- METADATA ONLY (no raw metrics - they live in group tables)
    signal_rank INTEGER,
    signal_confidence NUMERIC CHECK (signal_confidence >= 0 AND signal_confidence <= 1),
    trade_type TEXT,
    trade_type_confidence NUMERIC CHECK (trade_type_confidence >= 0 AND trade_type_confidence <= 1),
    risk_level TEXT CHECK (risk_level IN ('very_low', 'low', 'moderate', 'elevated', 'high')),
    signal_type TEXT,
    market_cap_category TEXT CHECK (market_cap_category IN ('mega', 'large', 'mid', 'small', 'micro', 'nano')),
    expected_hold_duration TEXT,
    
    -- MINIMAL PRICE/MARKET DATA (for quick reference)
    current_price NUMERIC,
    market_cap BIGINT,
    volume BIGINT,
    
    -- AI-GENERATED CONTENT (Top 10 only)
    risk_narrative TEXT,
    trade_strategy TEXT,
    ai_confidence NUMERIC CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
    ai_model_version TEXT,
    
    -- PROVENANCE
    scoring_version TEXT NOT NULL DEFAULT '3.0',
    data_sources TEXT[],
    cache_freshness JSONB,
    processing_metadata JSONB,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_ticker ON signals(ticker);
CREATE INDEX idx_signals_run_id ON signals(run_id);
CREATE INDEX idx_signals_signal_score ON signals(signal_score DESC);
CREATE INDEX idx_signals_signal_rank ON signals(signal_rank);
CREATE INDEX idx_signals_sector ON signals(sector);
CREATE INDEX idx_signals_trade_type ON signals(trade_type);
CREATE INDEX idx_signals_created_at ON signals(created_at DESC);

COMMENT ON TABLE signals IS '3.0 Core signals table - aggregated scores and metadata only';
COMMENT ON COLUMN signals.signal_score IS 'Final weighted score: tech(25%) + fund(25%) + news(20%) + social(15%) + risk(10%) + inst(5%)';

-- ----------------------------------------------------------------------------
-- TABLE: signals_technical (Group 1: Technical Indicators)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_technical (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- PRICE & MOMENTUM
    price_1d_pct NUMERIC,
    price_7d_pct NUMERIC,
    price_30d_pct NUMERIC,
    momentum_30d_pct NUMERIC,
    momentum_60d_pct NUMERIC,
    momentum_90d_pct NUMERIC,
    
    -- OSCILLATORS
    rsi NUMERIC CHECK (rsi >= 0 AND rsi <= 100),
    rsi_14 NUMERIC CHECK (rsi_14 >= 0 AND rsi_14 <= 100),
    relative_strength NUMERIC,
    
    -- MACD
    macd NUMERIC,
    macd_line NUMERIC,
    macd_signal NUMERIC,
    macd_histogram NUMERIC,
    macd_cross_signal TEXT,
    
    -- BOLLINGER BANDS
    bollinger_position NUMERIC CHECK (bollinger_position >= 0 AND bollinger_position <= 1),
    bollinger_upper NUMERIC,
    bollinger_lower NUMERIC,
    bollinger_width NUMERIC,
    bollinger_squeeze BOOLEAN,
    
    -- MOVING AVERAGES
    ma_50 NUMERIC,
    ma_200 NUMERIC,
    above_50d_ma_pct NUMERIC,
    above_200d_ma_pct NUMERIC,
    ma_cross_signal TEXT,
    ma_slope_50 NUMERIC,
    ma_slope_200 NUMERIC,
    
    -- VOLUME
    volume BIGINT,
    volume_spike_ratio NUMERIC,
    avg_volume_30d BIGINT,
    volume_trend_z NUMERIC,
    volume_price_correlation NUMERIC,
    
    -- VOLATILITY & ATR
    volatility NUMERIC,
    volatility_rank NUMERIC,
    atr NUMERIC,
    atr_percent NUMERIC,
    historical_volatility NUMERIC,
    
    -- SUPPORT/RESISTANCE
    day_high NUMERIC,
    day_low NUMERIC,
    support_level NUMERIC,
    resistance_level NUMERIC,
    
    -- Z-SCORES (Phase 2 normalization)
    z_score_momentum NUMERIC,
    z_score_volume NUMERIC,
    z_score_volatility NUMERIC,
    technical_z NUMERIC,
    price_z_20day NUMERIC,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_technical_signal_id ON signals_technical(signal_id);
CREATE INDEX idx_signals_technical_ticker ON signals_technical(ticker);

COMMENT ON TABLE signals_technical IS 'Group 1: Technical indicators from yfinance history data';

-- ----------------------------------------------------------------------------
-- TABLE: signals_fundamental (Group 2: Fundamental Metrics)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_fundamental (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- VALUATION
    pe_ratio NUMERIC,
    peg_ratio NUMERIC,
    price_to_book NUMERIC,
    price_to_sales NUMERIC,
    book_value NUMERIC,
    
    -- PROFITABILITY
    roe NUMERIC,
    roic NUMERIC,
    profit_margin NUMERIC,
    operating_margin NUMERIC,
    gross_margin NUMERIC,
    
    -- GROWTH
    eps_growth NUMERIC,
    revenue_growth NUMERIC,
    fcf_growth_3y_cagr NUMERIC,
    
    -- CASH FLOW
    fcf_margin NUMERIC,
    free_cash_flow BIGINT,
    operating_cash_flow BIGINT,
    
    -- FINANCIAL HEALTH
    debt_to_equity NUMERIC,
    debt_equity NUMERIC,
    current_ratio NUMERIC,
    interest_coverage NUMERIC,
    
    -- RETURNS
    dividend_yield NUMERIC,
    share_buyback_yield NUMERIC,
    
    -- EARNINGS
    earnings_gap_pct NUMERIC,
    last_earnings_surprise_pct NUMERIC,
    avg_earnings_surprise_pct NUMERIC,
    earnings_surprise_trend TEXT,
    earnings_surprise_streak NUMERIC,
    earnings_date TIMESTAMPTZ,
    
    -- DIVIDENDS
    dividend_ex_date TIMESTAMPTZ,
    
    -- Z-SCORES
    fundamental_z NUMERIC,
    valuation_z NUMERIC,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_fundamental_signal_id ON signals_fundamental(signal_id);
CREATE INDEX idx_signals_fundamental_ticker ON signals_fundamental(ticker);

COMMENT ON TABLE signals_fundamental IS 'Group 2: Fundamental metrics from yfinance financials/balance sheet/cashflow';

-- ----------------------------------------------------------------------------
-- TABLE: signals_news_macro (Group 3: News & Macro Data)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_news_macro (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- NEWS SENTIMENT
    news_sentiment_score NUMERIC CHECK (news_sentiment_score >= 0 AND news_sentiment_score <= 1),
    news_mentions INTEGER DEFAULT 0,
    news_sources TEXT[],
    top_headlines TEXT[],
    
    -- MACRO INDICATORS
    macro_sentiment_score NUMERIC CHECK (macro_sentiment_score >= 0 AND macro_sentiment_score <= 1),
    sector_news_sentiment NUMERIC CHECK (sector_news_sentiment >= 0 AND sector_news_sentiment <= 1),
    economic_indicator_score NUMERIC CHECK (economic_indicator_score >= 0 AND economic_indicator_score <= 1),
    
    -- SECTOR RELATIVE
    sector_relative_strength NUMERIC,
    sector_relative_percentile NUMERIC,
    
    -- EVENT FLAGS
    event_flags JSONB,
    catalyst_events TEXT[],
    
    -- Z-SCORES
    news_z NUMERIC,
    
    -- PLACEHOLDER FLAGS (for future integration)
    news_api_available BOOLEAN DEFAULT FALSE,
    macro_data_available BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_news_macro_signal_id ON signals_news_macro(signal_id);
CREATE INDEX idx_signals_news_macro_ticker ON signals_news_macro(ticker);

COMMENT ON TABLE signals_news_macro IS 'Group 3: News sentiment and macro indicators (PLACEHOLDER: awaiting News API integration)';

-- ----------------------------------------------------------------------------
-- TABLE: signals_social_alternative (Group 4: Social & Alternative Data)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_social_alternative (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- REDDIT DATA (PRAW - Active)
    reddit_sentiment NUMERIC CHECK (reddit_sentiment >= 0 AND reddit_sentiment <= 1),
    reddit_mentions INTEGER DEFAULT 0,
    reddit_upvotes INTEGER DEFAULT 0,
    reddit_comment_count INTEGER DEFAULT 0,
    reddit_post_recency NUMERIC,
    reddit_sentiment_trend TEXT,
    reddit_subreddits TEXT[],
    
    -- TWITTER DATA (PLACEHOLDER)
    twitter_sentiment NUMERIC CHECK (twitter_sentiment >= 0 AND twitter_sentiment <= 1),
    twitter_mentions INTEGER DEFAULT 0,
    twitter_api_available BOOLEAN DEFAULT FALSE,
    
    -- STOCKTWITS DATA (PLACEHOLDER)
    stocktwits_sentiment NUMERIC CHECK (stocktwits_sentiment >= 0 AND stocktwits_sentiment <= 1),
    stocktwits_mentions INTEGER DEFAULT 0,
    stocktwits_api_available BOOLEAN DEFAULT FALSE,
    
    -- AGGREGATED SOCIAL
    social_momentum_score NUMERIC,
    social_volume_spike BOOLEAN DEFAULT FALSE,
    
    -- Z-SCORES
    social_z NUMERIC,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_social_signal_id ON signals_social_alternative(signal_id);
CREATE INDEX idx_signals_social_ticker ON signals_social_alternative(ticker);

COMMENT ON TABLE signals_social_alternative IS 'Group 4: Social media sentiment (Reddit active, Twitter/StockTwits placeholders)';

-- ----------------------------------------------------------------------------
-- TABLE: signals_risk_stability (Group 5: Risk & Stability Metrics)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_risk_stability (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- MARKET RISK
    beta NUMERIC,
    market_correlation NUMERIC,
    
    -- VOLATILITY METRICS
    volatility NUMERIC,
    volatility_rank NUMERIC,
    max_drawdown_risk NUMERIC,
    sharpe_ratio NUMERIC,
    sortino_ratio NUMERIC,
    
    -- RISK SCORES (0-100 scale)
    risk_score NUMERIC CHECK (risk_score >= 0 AND risk_score <= 100),
    volatility_risk NUMERIC CHECK (volatility_risk >= 0 AND volatility_risk <= 100),
    liquidity_risk NUMERIC CHECK (liquidity_risk >= 0 AND liquidity_risk <= 100),
    leverage_risk NUMERIC CHECK (leverage_risk >= 0 AND leverage_risk <= 100),
    concentration_risk NUMERIC CHECK (concentration_risk >= 0 AND concentration_risk <= 100),
    technical_risk NUMERIC CHECK (technical_risk >= 0 AND technical_risk <= 100),
    fundamental_risk NUMERIC CHECK (fundamental_risk >= 0 AND fundamental_risk <= 100),
    sentiment_risk NUMERIC CHECK (sentiment_risk >= 0 AND sentiment_risk <= 100),
    
    -- LIQUIDITY
    liquidity_score NUMERIC,
    avg_daily_value_traded BIGINT,
    float_turnover_ratio NUMERIC,
    shares_float BIGINT,
    float_pct NUMERIC,
    
    -- SHORT INTEREST
    short_pct_float NUMERIC,
    short_pct_outstanding NUMERIC,
    short_ratio NUMERIC,
    shares_short BIGINT,
    short_interest NUMERIC,
    
    -- OPTIONS ACTIVITY
    put_call_vol_ratio NUMERIC,
    put_call_oi_ratio NUMERIC,
    put_call_ratio NUMERIC,
    unusual_options_activity BOOLEAN DEFAULT FALSE,
    options_volume BIGINT,
    open_interest BIGINT,
    implied_volatility NUMERIC,
    iv_spike_pct NUMERIC,
    
    -- POSITION SIZING
    max_position_size NUMERIC,
    position_size_recommendation NUMERIC,
    exit_signal_strength NUMERIC,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_risk_signal_id ON signals_risk_stability(signal_id);
CREATE INDEX idx_signals_risk_ticker ON signals_risk_stability(ticker);

COMMENT ON TABLE signals_risk_stability IS 'Group 5: Risk metrics, volatility, liquidity, and options data from yfinance';

-- ----------------------------------------------------------------------------
-- TABLE: signals_institutional_smart_money (Group 6: Institutional & Insider)
-- ----------------------------------------------------------------------------
CREATE TABLE signals_institutional_smart_money (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    
    -- INSTITUTIONAL OWNERSHIP
    institutional_ownership_pct NUMERIC,
    institutional_ownership NUMERIC,
    institutional_change_qoq NUMERIC,
    top_10_institutional_holders_pct NUMERIC,
    num_institutional_holders INTEGER,
    
    -- RETAIL OWNERSHIP
    retail_holding_pct NUMERIC,
    
    -- INSIDER ACTIVITY
    insider_ownership NUMERIC,
    insider_activity_score NUMERIC,
    insider_buy_volume BIGINT,
    insider_buy_count INTEGER,
    insider_sell_count INTEGER,
    insider_net_shares BIGINT,
    insider_sentiment TEXT,
    
    -- ANALYST RATINGS
    analyst_target_price NUMERIC,
    analyst_target_upside_pct NUMERIC,
    analyst_recommendation_mean NUMERIC,
    analyst_count INTEGER,
    analyst_rating_trend TEXT,
    
    -- HEDGE FUNDS
    hedge_fund_holdings_change NUMERIC,
    
    -- SEC FILINGS
    sec_filing_score NUMERIC CHECK (sec_filing_score >= 0 AND sec_filing_score <= 1),
    recent_filings TEXT[],
    
    -- SMART MONEY SIGNALS
    smart_money_flow NUMERIC,
    institutional_buying_pressure BOOLEAN DEFAULT FALSE,
    
    -- PLACEHOLDER (for future financial API)
    future_financial_api_available BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_institutional_signal_id ON signals_institutional_smart_money(signal_id);
CREATE INDEX idx_signals_institutional_ticker ON signals_institutional_smart_money(ticker);

COMMENT ON TABLE signals_institutional_smart_money IS 'Group 6: Institutional ownership, insider activity, analyst ratings from yfinance';

-- ============================================================================
-- PART 3: PERFORMANCE TRACKING (Future Implementation)
-- ============================================================================

-- NOTE: Backtest/performance tables will be added in future phase
-- For now, focusing on core 3.0 signal architecture

-- ============================================================================
-- PART 4: INDEXES & CONSTRAINTS
-- ============================================================================

-- Composite indexes for common queries
CREATE INDEX idx_signals_ticker_run ON signals(ticker, run_id);
CREATE INDEX idx_signals_score_rank ON signals(signal_score DESC, signal_rank);
CREATE INDEX idx_cache_ticker_group ON data_cache(ticker, data_group);

-- ============================================================================
-- PART 5: VALIDATION QUERIES
-- ============================================================================

-- Verify table creation
DO $$
BEGIN
    RAISE NOTICE '✅ 3.0 Schema Migration Complete!';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - runs (fresh schema)';
    RAISE NOTICE '  - data_cache (Phase 1 cache)';
    RAISE NOTICE '  - signals (core aggregation)';
    RAISE NOTICE '  - signals_technical (Group 1)';
    RAISE NOTICE '  - signals_fundamental (Group 2)';
    RAISE NOTICE '  - signals_news_macro (Group 3 - placeholders)';
    RAISE NOTICE '  - signals_social_alternative (Group 4 - Reddit active)';
    RAISE NOTICE '  - signals_risk_stability (Group 5)';
    RAISE NOTICE '  - signals_institutional_smart_money (Group 6)';
    RAISE NOTICE '';
    RAISE NOTICE 'Preserved tables:';
    RAISE NOTICE '  - company_tickers';
    RAISE NOTICE '  - guardrails_config';
END $$;

-- Row count check (should all be 0)
SELECT 
    'runs' as table_name, COUNT(*) as row_count FROM runs
UNION ALL
SELECT 'data_cache', COUNT(*) FROM data_cache
UNION ALL
SELECT 'signals', COUNT(*) FROM signals
UNION ALL
SELECT 'signals_technical', COUNT(*) FROM signals_technical
UNION ALL
SELECT 'signals_fundamental', COUNT(*) FROM signals_fundamental
UNION ALL
SELECT 'signals_news_macro', COUNT(*) FROM signals_news_macro
UNION ALL
SELECT 'signals_social_alternative', COUNT(*) FROM signals_social_alternative
UNION ALL
SELECT 'signals_risk_stability', COUNT(*) FROM signals_risk_stability
UNION ALL
SELECT 'signals_institutional_smart_money', COUNT(*) FROM signals_institutional_smart_money;
