-- Add coverage columns to signals table
-- These columns are calculated by the Python pipeline but were missing from the schema

ALTER TABLE signals
ADD COLUMN IF NOT EXISTS technical_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS fundamental_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS news_macro_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS social_alternative_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS risk_stability_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS institutional_smart_money_coverage NUMERIC DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_coverage NUMERIC DEFAULT 0;

-- Add check constraints to ensure coverage values are between 0 and 1
ALTER TABLE signals
ADD CONSTRAINT check_technical_coverage CHECK (technical_coverage >= 0 AND technical_coverage <= 1),
ADD CONSTRAINT check_fundamental_coverage CHECK (fundamental_coverage >= 0 AND fundamental_coverage <= 1),
ADD CONSTRAINT check_news_macro_coverage CHECK (news_macro_coverage >= 0 AND news_macro_coverage <= 1),
ADD CONSTRAINT check_social_alternative_coverage CHECK (social_alternative_coverage >= 0 AND social_alternative_coverage <= 1),
ADD CONSTRAINT check_risk_stability_coverage CHECK (risk_stability_coverage >= 0 AND risk_stability_coverage <= 1),
ADD CONSTRAINT check_institutional_smart_money_coverage CHECK (institutional_smart_money_coverage >= 0 AND institutional_smart_money_coverage <= 1),
ADD CONSTRAINT check_total_coverage CHECK (total_coverage >= 0 AND total_coverage <= 1);
