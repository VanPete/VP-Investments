-- Verification Query for Migration 017
-- Run this in Supabase SQL Editor after pipeline completes

-- Step 1: Get the most recent run_id
SELECT 
    id as run_id,
    run_timestamp,
    successful_tickers,
    status
FROM signal_runs 
ORDER BY run_timestamp DESC 
LIMIT 1;

-- Step 2: Check market_cap and beta population (replace <RUN_ID> with value from Step 1)
-- Copy the run_id from above and paste below
SELECT 
    ticker,
    company_name,
    sector,
    market_cap,
    beta,
    overall_score,
    CASE 
        WHEN market_cap IS NOT NULL THEN '✅' 
        ELSE '❌' 
    END as has_mktcap,
    CASE 
        WHEN beta IS NOT NULL THEN '✅' 
        ELSE '❌' 
    END as has_beta
FROM signals 
WHERE run_id = '<PASTE_RUN_ID_HERE>'
ORDER BY overall_score DESC
LIMIT 20;

-- Step 3: Get population statistics
SELECT 
    COUNT(*) as total_signals,
    COUNT(market_cap) as mktcap_populated,
    COUNT(beta) as beta_populated,
    ROUND(100.0 * COUNT(market_cap) / COUNT(*), 1) as mktcap_pct,
    ROUND(100.0 * COUNT(beta) / COUNT(*), 1) as beta_pct
FROM signals 
WHERE run_id = '<PASTE_RUN_ID_HERE>';

-- Expected Results:
-- - mktcap_pct >= 50% (at least half have market cap)
-- - beta_pct >= 50% (at least half have beta)
-- - Some NULL values are OK (not all tickers have this data)
