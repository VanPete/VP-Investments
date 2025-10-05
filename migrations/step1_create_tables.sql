-- ============================================================
-- VP INVESTMENTS - 3-TABLE STRUCTURE MIGRATION
-- STEP 1: Create New Tables (signal_metrics, signal_performance)
-- Date: 2025-10-05
-- ============================================================

-- Drop existing tables if they exist (for clean migration)
DROP TABLE IF EXISTS signal_performance CASCADE;
DROP TABLE IF EXISTS signal_metrics CASCADE;

-- ============================================================
-- Table: signal_metrics (1-to-1 with signals)
-- Purpose: Technical & fundamental indicators
-- ============================================================

CREATE TABLE signal_metrics (
    id BIGSERIAL PRIMARY KEY,
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Technical Indicators (14 columns)
    price_7d_pct NUMERIC(10,2),
    price_30d_pct NUMERIC(10,2),
    relative_strength NUMERIC(10,2),
    momentum_30d_pct NUMERIC(10,2),
    rsi NUMERIC(10,2),
    macd_line NUMERIC(10,4),
    macd_signal NUMERIC(10,4),
    above_50d_ma_pct NUMERIC(10,2),
    above_200d_ma_pct NUMERIC(10,2),
    avg_daily_volume BIGINT,
    avg_volume_30d BIGINT,
    sector_relative_strength NUMERIC(10,2),
    exit_signal_strength NUMERIC(10,2),
    signal_strength_percentile NUMERIC(10,2),
    
    -- Fundamental Indicators (8 columns)
    pe_ratio NUMERIC(10,2),
    eps_growth NUMERIC(10,2),
    roe NUMERIC(10,2),
    debt_equity NUMERIC(10,2),
    fcf_margin NUMERIC(10,2),
    analyst_rating NUMERIC(3,2),
    ebitda_margin NUMERIC(10,2),
    quick_ratio NUMERIC(10,2),
    
    -- Options Data (5 columns)
    avg_iv NUMERIC(10,2),
    call_put_ratio NUMERIC(10,2),
    put_volume BIGINT,
    call_volume BIGINT,
    max_pain_price NUMERIC(12,2),
    
    -- Ownership & Short Interest (5 columns)
    institutional_pct NUMERIC(5,2),
    insider_pct NUMERIC(5,2),
    short_pct_float NUMERIC(5,2),
    shares_short BIGINT,
    days_to_cover NUMERIC(10,2),
    
    -- Technical Levels (3 columns)
    support_level NUMERIC(12,2),
    resistance_level NUMERIC(12,2),
    pivot_point NUMERIC(12,2),
    
    -- Bollinger Bands (2 columns)
    bollinger_upper NUMERIC(12,2),
    bollinger_lower NUMERIC(12,2),
    
    -- Risk Metrics (1 column)
    risk_score NUMERIC(5,2),
    
    -- Additional Metrics (7 columns)
    volume_surge_ratio NUMERIC(10,2),
    unusual_volume_activity BOOLEAN DEFAULT FALSE,
    price_momentum_signal TEXT,
    trend_strength NUMERIC(5,2),
    market_correlation NUMERIC(5,2),
    beta NUMERIC(10,4),
    sharpe_ratio NUMERIC(10,4),
    
    UNIQUE(signal_id)
);

-- ============================================================
-- Table: signal_performance (1-to-many with signals)
-- Purpose: Backtest results over time
-- ============================================================

CREATE TABLE signal_performance (
    id BIGSERIAL PRIMARY KEY,
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Backtest Configuration
    backtest_type TEXT NOT NULL,  -- '1d', '3d', '7d', '10d', '30d'
    backtest_date TIMESTAMPTZ NOT NULL,
    
    -- Entry/Exit Data
    entry_price NUMERIC(12,2),
    exit_price NUMERIC(12,2),
    return_pct NUMERIC(10,2),
    
    -- Performance Metrics
    win BOOLEAN,
    max_gain_pct NUMERIC(10,2),
    max_drawdown_pct NUMERIC(10,2),
    hold_period_days INTEGER,
    
    -- Market Comparison
    spy_return_pct NUMERIC(10,2),
    alpha NUMERIC(10,2),
    outperformed_spy BOOLEAN,
    
    -- Trade Statistics
    entry_volume BIGINT,
    exit_volume BIGINT,
    avg_volume_during_hold BIGINT,
    
    -- Risk Metrics
    volatility_during_hold NUMERIC(10,4),
    sharpe_ratio NUMERIC(10,4),
    sortino_ratio NUMERIC(10,4),
    max_adverse_excursion NUMERIC(10,2),
    
    -- Technical Context
    entry_rsi NUMERIC(10,2),
    exit_rsi NUMERIC(10,2),
    
    -- Additional Tracking
    backtest_notes TEXT,
    error_message TEXT
);

-- ============================================================
-- Indexes for Performance
-- ============================================================

-- signal_metrics indexes
CREATE INDEX idx_signal_metrics_signal_id ON signal_metrics(signal_id);
CREATE INDEX idx_signal_metrics_ticker ON signal_metrics(ticker);
CREATE INDEX idx_signal_metrics_created_at ON signal_metrics(created_at DESC);

-- signal_performance indexes
CREATE INDEX idx_signal_performance_signal_id ON signal_performance(signal_id);
CREATE INDEX idx_signal_performance_ticker ON signal_performance(ticker);
CREATE INDEX idx_signal_performance_run_id ON signal_performance(run_id);
CREATE INDEX idx_signal_performance_backtest_type ON signal_performance(backtest_type);
CREATE INDEX idx_signal_performance_backtest_date ON signal_performance(backtest_date DESC);
CREATE INDEX idx_signal_performance_ticker_backtest_type ON signal_performance(ticker, backtest_type);
CREATE INDEX idx_signal_performance_created_at ON signal_performance(created_at DESC);

-- ============================================================
-- Verification
-- ============================================================

SELECT 'Tables created successfully!' AS status;
SELECT COUNT(*) AS signal_metrics_count FROM signal_metrics;
SELECT COUNT(*) AS signal_performance_count FROM signal_performance;
