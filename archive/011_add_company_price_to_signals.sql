-- Migration 011: Add company_name and current_price to signals table
-- Description: Adds company name and current price fields for better UX
-- Author: System
-- Date: 2025-10-26

-- Add company_name column
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS company_name TEXT;

-- Add current_price column (matches baseline_price from performance)
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS current_price NUMERIC(12, 4);

-- Create index for company name searches
CREATE INDEX IF NOT EXISTS idx_signals_company_name 
ON signals(company_name);

-- Add comment
COMMENT ON COLUMN signals.company_name IS 'Company full name fetched from yfinance';
COMMENT ON COLUMN signals.current_price IS 'Stock price at signal creation time (matches performance baseline_price)';
