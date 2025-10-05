-- ============================================================
-- VP INVESTMENTS - 3-TABLE STRUCTURE MIGRATION
-- STEP 2: Migrate Existing Data
-- Date: 2025-10-05
-- ============================================================

-- ============================================================
-- Part 1: Migrate to signal_metrics
-- ============================================================

INSERT INTO signal_metrics (
    signal_id,
    ticker,
    created_at,
    
    -- Technical Indicators
    price_7d_pct,
    price_30d_pct,
    relative_strength,
    momentum_30d_pct,
    rsi,
    macd_line,
    macd_signal,
    above_50d_ma_pct,
    above_200d_ma_pct,
    avg_daily_volume,
    avg_volume_30d,
    sector_relative_strength,
    exit_signal_strength,
    signal_strength_percentile,
    
    -- Fundamental Indicators
    pe_ratio,
    eps_growth,
    roe,
    debt_equity,
    fcf_margin,
    analyst_rating,
    ebitda_margin,
    quick_ratio,
    
    -- Options Data
    avg_iv,
    call_put_ratio,
    put_volume,
    call_volume,
    max_pain_price,
    
    -- Ownership & Short Interest
    institutional_pct,
    insider_pct,
    short_pct_float,
    shares_short,
    days_to_cover,
    
    -- Technical Levels
    support_level,
    resistance_level,
    pivot_point,
    
    -- Bollinger Bands
    bollinger_upper,
    bollinger_lower,
    
    -- Risk Metrics
    risk_score,
    
    -- Additional Metrics
    volume_surge_ratio,
    unusual_volume_activity,
    price_momentum_signal,
    trend_strength,
    market_correlation,
    beta,
    sharpe_ratio
)
SELECT 
    id AS signal_id,
    ticker,
    signal_datetime AS created_at,
    
    -- Technical Indicators
    price_7d_pct,
    price_30d_pct,
    relative_strength,
    momentum_30d_pct,
    rsi,
    macd_line,
    macd_signal,
    above_50d_ma_pct,
    above_200d_ma_pct,
    avg_daily_volume,
    avg_volume_30d,
    sector_relative_strength,
    exit_signal_strength,
    signal_strength_percentile,
    
    -- Fundamental Indicators
    pe_ratio,
    eps_growth,
    roe,
    debt_equity,
    fcf_margin,
    analyst_rating,
    ebitda_margin,
    quick_ratio,
    
    -- Options Data
    avg_iv,
    call_put_ratio,
    put_volume,
    call_volume,
    max_pain_price,
    
    -- Ownership & Short Interest
    institutional_pct,
    insider_pct,
    short_pct_float,
    shares_short,
    days_to_cover,
    
    -- Technical Levels
    support_level,
    resistance_level,
    pivot_point,
    
    -- Bollinger Bands
    bollinger_upper,
    bollinger_lower,
    
    -- Risk Metrics
    risk_score,
    
    -- Additional Metrics
    volume_surge_ratio,
    unusual_volume_activity,
    price_momentum_signal,
    trend_strength,
    market_correlation,
    beta,
    sharpe_ratio
FROM signals
WHERE id IS NOT NULL;

-- ============================================================
-- Part 2: Migrate to signal_performance (3-day returns)
-- ============================================================

INSERT INTO signal_performance (
    signal_id,
    ticker,
    run_id,
    created_at,
    backtest_type,
    backtest_date,
    return_pct,
    entry_price,
    exit_price,
    win,
    spy_return_pct,
    alpha,
    outperformed_spy
)
SELECT 
    id AS signal_id,
    ticker,
    run_id,
    signal_datetime AS created_at,
    '3d' AS backtest_type,
    signal_datetime + INTERVAL '3 days' AS backtest_date,
    "3d_return" AS return_pct,
    current_price AS entry_price,
    current_price * (1 + COALESCE("3d_return", 0) / 100) AS exit_price,
    ("3d_return" > 0) AS win,
    "3d_spy_return" AS spy_return_pct,
    COALESCE("3d_return", 0) - COALESCE("3d_spy_return", 0) AS alpha,
    (COALESCE("3d_return", 0) > COALESCE("3d_spy_return", 0)) AS outperformed_spy
FROM signals
WHERE "3d_return" IS NOT NULL;

-- ============================================================
-- Part 3: Migrate to signal_performance (7-day returns)
-- ============================================================

INSERT INTO signal_performance (
    signal_id,
    ticker,
    run_id,
    created_at,
    backtest_type,
    backtest_date,
    return_pct,
    entry_price,
    exit_price,
    win,
    spy_return_pct,
    alpha,
    outperformed_spy
)
SELECT 
    id AS signal_id,
    ticker,
    run_id,
    signal_datetime AS created_at,
    '7d' AS backtest_type,
    signal_datetime + INTERVAL '7 days' AS backtest_date,
    "7d_return" AS return_pct,
    current_price AS entry_price,
    current_price * (1 + COALESCE("7d_return", 0) / 100) AS exit_price,
    ("7d_return" > 0) AS win,
    "7d_spy_return" AS spy_return_pct,
    COALESCE("7d_return", 0) - COALESCE("7d_spy_return", 0) AS alpha,
    (COALESCE("7d_return", 0) > COALESCE("7d_spy_return", 0)) AS outperformed_spy
FROM signals
WHERE "7d_return" IS NOT NULL;

-- ============================================================
-- Part 4: Migrate to signal_performance (10-day returns)
-- ============================================================

INSERT INTO signal_performance (
    signal_id,
    ticker,
    run_id,
    created_at,
    backtest_type,
    backtest_date,
    return_pct,
    entry_price,
    exit_price,
    win,
    spy_return_pct,
    alpha,
    outperformed_spy
)
SELECT 
    id AS signal_id,
    ticker,
    run_id,
    signal_datetime AS created_at,
    '10d' AS backtest_type,
    signal_datetime + INTERVAL '10 days' AS backtest_date,
    "10d_return" AS return_pct,
    current_price AS entry_price,
    current_price * (1 + COALESCE("10d_return", 0) / 100) AS exit_price,
    ("10d_return" > 0) AS win,
    "10d_spy_return" AS spy_return_pct,
    COALESCE("10d_return", 0) - COALESCE("10d_spy_return", 0) AS alpha,
    (COALESCE("10d_return", 0) > COALESCE("10d_spy_return", 0)) AS outperformed_spy
FROM signals
WHERE "10d_return" IS NOT NULL;

-- ============================================================
-- Part 5: Migrate to signal_performance (30-day returns)
-- ============================================================

INSERT INTO signal_performance (
    signal_id,
    ticker,
    run_id,
    created_at,
    backtest_type,
    backtest_date,
    return_pct,
    entry_price,
    exit_price,
    win,
    spy_return_pct,
    alpha,
    outperformed_spy
)
SELECT 
    id AS signal_id,
    ticker,
    run_id,
    signal_datetime AS created_at,
    '30d' AS backtest_type,
    signal_datetime + INTERVAL '30 days' AS backtest_date,
    "30d_return" AS return_pct,
    current_price AS entry_price,
    current_price * (1 + COALESCE("30d_return", 0) / 100) AS exit_price,
    ("30d_return" > 0) AS win,
    "30d_spy_return" AS spy_return_pct,
    COALESCE("30d_return", 0) - COALESCE("30d_spy_return", 0) AS alpha,
    (COALESCE("30d_return", 0) > COALESCE("30d_spy_return", 0)) AS outperformed_spy
FROM signals
WHERE "30d_return" IS NOT NULL;

-- ============================================================
-- Verification
-- ============================================================

SELECT 'Data migration complete!' AS status;
SELECT COUNT(*) AS signal_count FROM signals;
SELECT COUNT(*) AS signal_metrics_count FROM signal_metrics;
SELECT COUNT(*) AS signal_performance_count FROM signal_performance;
SELECT COUNT(*) AS performance_3d FROM signal_performance WHERE backtest_type = '3d';
SELECT COUNT(*) AS performance_7d FROM signal_performance WHERE backtest_type = '7d';
SELECT COUNT(*) AS performance_10d FROM signal_performance WHERE backtest_type = '10d';
SELECT COUNT(*) AS performance_30d FROM signal_performance WHERE backtest_type = '30d';
