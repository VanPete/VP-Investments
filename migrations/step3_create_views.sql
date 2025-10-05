-- ============================================================
-- VP INVESTMENTS - 3-TABLE STRUCTURE MIGRATION
-- STEP 3: Create Helper Views
-- Date: 2025-10-05
-- ============================================================

-- Drop existing views if they exist
DROP VIEW IF EXISTS v_signals_dashboard CASCADE;
DROP VIEW IF EXISTS v_signals_latest_performance CASCADE;
DROP VIEW IF EXISTS v_signals_complete CASCADE;

-- ============================================================
-- View 1: v_signals_complete
-- Purpose: Full signal data with metrics for detailed analysis
-- ============================================================

CREATE OR REPLACE VIEW v_signals_complete AS
SELECT 
    -- Core signal fields (explicitly list to avoid duplicates)
    s.id,
    s.ticker,
    s.run_id,
    s.signal_datetime,
    s.source,
    s.reddit_score,
    s.financial_score,
    s.news_score,
    s.weighted_score,
    s.sentiment_score,
    s.current_price,
    s.market_cap,
    s.volume,
    s.avg_daily_value_traded,
    s.commentary,
    s.ai_commentary,
    s.headline,
    s.reasoning,
    s.subreddit_id,
    s.industry,
    s.sector,
    s.created_at,
    
    -- Selected metrics
    m.rsi,
    m.relative_strength,
    m.momentum_30d_pct,
    m.above_50d_ma_pct,
    m.above_200d_ma_pct,
    m.pe_ratio,
    m.debt_equity,
    m.roe,
    m.risk_score
FROM signals s
LEFT JOIN signal_metrics m ON s.id = m.signal_id;

-- ============================================================
-- View 2: v_signals_latest_performance
-- Purpose: Latest backtest result for each signal
-- ============================================================

CREATE OR REPLACE VIEW v_signals_latest_performance AS
SELECT DISTINCT ON (sp.signal_id, sp.backtest_type)
    sp.signal_id,
    sp.ticker,
    sp.backtest_type,
    sp.return_pct,
    sp.win,
    sp.spy_return_pct,
    sp.alpha,
    sp.outperformed_spy,
    sp.backtest_date,
    sp.created_at
FROM signal_performance sp
ORDER BY sp.signal_id, sp.backtest_type, sp.backtest_date DESC;

-- ============================================================
-- View 3: v_signals_dashboard
-- Purpose: Optimized for fast dashboard queries
-- ============================================================

CREATE OR REPLACE VIEW v_signals_dashboard AS
SELECT 
    s.id,
    s.ticker,
    s.run_id,
    s.signal_datetime,
    s.weighted_score,
    s.reddit_score,
    s.financial_score,
    s.news_score,
    s.sentiment_score,
    s.current_price,
    s.market_cap,
    s.volume,
    s.commentary,
    s.ai_commentary,
    s.sector,
    s.industry,
    m.rsi,
    m.relative_strength,
    m.risk_score,
    s.created_at
FROM signals s
LEFT JOIN signal_metrics m ON s.id = m.signal_id;

-- ============================================================
-- Verification
-- ============================================================

SELECT 'Views created successfully!' AS status;
SELECT COUNT(*) AS complete_view_count FROM v_signals_complete;
SELECT COUNT(*) AS dashboard_view_count FROM v_signals_dashboard;
SELECT COUNT(*) AS latest_performance_count FROM v_signals_latest_performance;
