-- Migration 021: Add CASCADE DELETE to foreign key constraints
-- This allows deleting runs without manually deleting related records first

-- Drop and recreate analytics foreign key with CASCADE
ALTER TABLE public.analytics DROP CONSTRAINT IF EXISTS analytics_run_id_fkey;
ALTER TABLE public.analytics 
  ADD CONSTRAINT analytics_run_id_fkey 
  FOREIGN KEY (run_id) 
  REFERENCES public.signal_runs(id) 
  ON DELETE CASCADE;

-- Drop and recreate signals foreign key with CASCADE  
ALTER TABLE public.signals DROP CONSTRAINT IF EXISTS fk_signals_run;
ALTER TABLE public.signals 
  ADD CONSTRAINT fk_signals_run 
  FOREIGN KEY (run_id) 
  REFERENCES public.signal_runs(id) 
  ON DELETE CASCADE;

-- Note: performance table has signal_id FK to signals table
-- When signals are deleted via CASCADE, performance records will be orphaned
-- We should also add CASCADE for performance -> signals relationship

ALTER TABLE public.performance DROP CONSTRAINT IF EXISTS performance_signal_id_fkey;
ALTER TABLE public.performance 
  ADD CONSTRAINT performance_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

-- All signals_* tables reference signals(id) - add CASCADE to those too
ALTER TABLE public.signals_technical DROP CONSTRAINT IF EXISTS signals_technical_signal_id_fkey;
ALTER TABLE public.signals_technical 
  ADD CONSTRAINT signals_technical_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

ALTER TABLE public.signals_fundamental DROP CONSTRAINT IF EXISTS signals_fundamental_signal_id_fkey;
ALTER TABLE public.signals_fundamental 
  ADD CONSTRAINT signals_fundamental_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

ALTER TABLE public.signals_news_macro DROP CONSTRAINT IF EXISTS signals_news_macro_signal_id_fkey;
ALTER TABLE public.signals_news_macro 
  ADD CONSTRAINT signals_news_macro_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

ALTER TABLE public.signals_social_alternative DROP CONSTRAINT IF EXISTS signals_social_alternative_signal_id_fkey;
ALTER TABLE public.signals_social_alternative 
  ADD CONSTRAINT signals_social_alternative_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

ALTER TABLE public.signals_risk_stability DROP CONSTRAINT IF EXISTS signals_risk_stability_signal_id_fkey;
ALTER TABLE public.signals_risk_stability 
  ADD CONSTRAINT signals_risk_stability_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

ALTER TABLE public.signals_institutional_smart_money DROP CONSTRAINT IF EXISTS signals_institutional_smart_money_signal_id_fkey;
ALTER TABLE public.signals_institutional_smart_money 
  ADD CONSTRAINT signals_institutional_smart_money_signal_id_fkey 
  FOREIGN KEY (signal_id) 
  REFERENCES public.signals(id) 
  ON DELETE CASCADE;

-- Summary of CASCADE chain:
-- 1. Delete signal_runs(id) -> cascades to signals(run_id) AND analytics(run_id)
-- 2. Delete signals(id) -> cascades to performance(signal_id) AND all signals_*(signal_id) tables
-- 
-- Result: Deleting a run will automatically clean up:
--   - All signals from that run
--   - All performance records for those signals  
--   - All factor values in signals_* tables for those signals
--   - All analytics records for that run
