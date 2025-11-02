-- Verification script for migration 021: CASCADE DELETE
-- This script verifies that the CASCADE DELETE constraints are properly set up

-- Check analytics table constraint
SELECT 
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints AS rc
    ON rc.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_name IN (
    'analytics',
    'signals', 
    'performance',
    'signals_technical',
    'signals_fundamental',
    'signals_news_macro',
    'signals_social_alternative',
    'signals_risk_stability',
    'signals_institutional_smart_money'
)
ORDER BY tc.table_name, tc.constraint_name;

-- Expected results:
-- analytics_run_id_fkey: delete_rule = CASCADE
-- fk_signals_run: delete_rule = CASCADE  
-- performance_signal_id_fkey: delete_rule = CASCADE
-- signals_technical_signal_id_fkey: delete_rule = CASCADE
-- signals_fundamental_signal_id_fkey: delete_rule = CASCADE
-- signals_news_macro_signal_id_fkey: delete_rule = CASCADE
-- signals_social_alternative_signal_id_fkey: delete_rule = CASCADE
-- signals_risk_stability_signal_id_fkey: delete_rule = CASCADE
-- signals_institutional_smart_money_signal_id_fkey: delete_rule = CASCADE
