-- Migration 017: Add market_cap and beta to signals table
-- Purpose: Support VanPiQ Performance Tab header requirements (MktCap + β display)
-- Date: 2025-10-31

-- Add market_cap column (in USD)
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS market_cap BIGINT;

-- Add beta column (volatility vs SPY)
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS beta REAL;

-- Add column comments for documentation
COMMENT ON COLUMN signals.market_cap IS 'Market capitalization in USD (from YFinance info.marketCap)';
COMMENT ON COLUMN signals.beta IS 'Beta vs SPY - measures stock volatility relative to market (from YFinance info.beta)';

-- Verify columns added
SELECT 
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'signals' 
  AND column_name IN ('market_cap', 'beta')
ORDER BY column_name;
