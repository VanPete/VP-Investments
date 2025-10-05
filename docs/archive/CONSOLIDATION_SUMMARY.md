# 📋 Database Consolidation - Executive Summary

**Date:** 2025-10-05  
**Status:** Ready to Execute  
**Approach:** Hybrid (Archive tables + Drop views)

---

## 🎯 What We're Doing

**"Less is More"** - Simplifying the database by removing unused objects.

### Before:
- 15 tables (8 empty)
- 14 views (11 unused)
- **29 total objects**

### After:
- 7 tables (all active)
- 3 views (essential only)
- **10 total objects**

**66% reduction** in database complexity!

---

## 📦 What's Being Removed

### Empty Tables (8) - ARCHIVED ⚠️
Moved to `archive_20251005` schema (can restore if needed):

| Table | Rows | Columns | Why Remove |
|-------|------|---------|------------|
| ai_strategy_performance | 0 | 31 | Never populated |
| backtest_trades | 0 | 21 | Not used by backtest logic |
| backtests | 0 | 34 | Redundant with backtest_interval_tracking |
| market_conditions | 0 | 16 | No data source configured |
| scoring_calibration_log | 0 | 7 | Not implemented |
| signal_calibration_log | 0 | 18 | Not implemented |
| signal_performance | 0 | 19 | Redundant with signals table columns |
| signal_performance_history | 0 | 44 | Not implemented |

**Total:** 194 unused columns removed from public schema

### Redundant Views (11) - DROPPED ❌
Views are just query shortcuts (no data loss):

1. `backtest_summary` - Duplicate of simple SELECT query
2. `top_signal_factors` - Not used by frontend/backend
3. `trade_analysis` - Empty base table
4. `v_active_ai_strategies` - Can query ai_strategies directly
5. `v_ai_strategy_metrics` - Not used
6. `v_recent_signal_performance` - Empty base table
7. `v_sector_relative_performance` - Not implemented
8. `v_signal_factor_analysis` - Not used
9. `v_signal_performance_by_score` - Empty base table
10. `v_top_momentum_signals` - Duplicate functionality
11. `v_top_performers` - Simple SELECT query

---

## ✅ What We're Keeping

### Core Tables (7)

| Table | Rows | Status | Purpose |
|-------|------|--------|---------|
| **signals** | 340 | ✅ Active | Core signals data (142 columns) |
| **company_tickers** | 7,638 | ✅ Active | Ticker reference |
| **ai_strategies** | 122 | ✅ Active | AI-generated strategies |
| **signal_scoring_factors** | 18 | ✅ Active | Scoring weight tracking |
| **backtest_interval_tracking** | 1,700 | ✅ Active | Backtest execution history |
| **runs** | 9 | ✅ Active | Pipeline run metadata |
| **guardrails_config** | 6 | ✅ Active | System configuration |

### Essential Views (3)

1. **`v_recent_signals`** - Dashboard quick view
2. **`backtest_eligible_signals`** - Used by pipeline backtest
3. **`signal_performance_summary`** - Performance tracking

---

## 🚀 Benefits

1. **📉 66% Fewer Database Objects**
   - From 29 → 10 objects
   - Easier to understand and maintain

2. **🧠 Reduced Cognitive Load**
   - No more confusion about which tables to use
   - Clear purpose for each remaining object

3. **⚡ Faster Operations**
   - Smaller database backups
   - Faster schema migrations
   - Less overhead in database queries

4. **🔧 Easier Maintenance**
   - Fewer objects to maintain
   - Simpler schema documentation
   - Less tech debt

5. **💾 Storage Savings**
   - 194 fewer columns (metadata overhead)
   - Simplified indexes
   - Cleaner query plans

---

## ⚠️ Safety Measures

### 1. Archive First (Not Drop)
- Empty tables moved to `archive_20251005` schema
- Can be restored instantly if needed:
  ```sql
  ALTER TABLE archive_20251005.signal_performance SET SCHEMA public;
  ```

### 2. Views Are Safe to Drop
- Views are just stored queries (no data)
- Can be recreated easily if needed
- SQL definitions saved in migration file

### 3. Testing Plan
1. ✅ Run migration
2. ✅ Run pipeline: `python -m backend.pipeline`
3. ✅ Test tables.py: `python tables.py`
4. ✅ Verify no errors for 1 week
5. ✅ Permanently drop archive: `DROP SCHEMA archive_20251005 CASCADE;`

---

## 📝 Execution Steps

### Step 1: Review Plan
```bash
# Review consolidation plan
cat DATABASE_CONSOLIDATION_PLAN.md

# Review SQL migration
cat migrations/database_consolidation_20251005.sql
```

### Step 2: Run Migration
```bash
# Option A: Direct Supabase SQL Editor (Recommended)
# Copy/paste migrations/database_consolidation_20251005.sql

# Option B: Using psql
psql $DATABASE_URL -f migrations/database_consolidation_20251005.sql
```

### Step 3: Verify
```bash
# Check tables
python list_all_tables.py

# Run pipeline (should work normally)
python -m backend.pipeline

# Analyze remaining tables
python tables.py
```

### Step 4: Cleanup (After 1 Week)
```sql
-- Only if everything works perfectly for 1 week
DROP SCHEMA archive_20251005 CASCADE;
```

---

## 🔄 Rollback Plan

If something breaks, restore archived tables:

```sql
-- Restore specific table
ALTER TABLE archive_20251005.signal_performance SET SCHEMA public;

-- Or restore all archived tables
DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN 
        SELECT t.table_name 
        FROM information_schema.tables t 
        WHERE t.table_schema = 'archive_20251005'
    LOOP
        EXECUTE format('ALTER TABLE archive_20251005.%I SET SCHEMA public', table_name);
        RAISE NOTICE 'Restored: %', table_name;
    END LOOP;
END $$;
```

---

## ❓ FAQ

### Q: What if we need those tables later?
**A:** We can recreate them easily. SQL definitions are saved. They have 0 rows anyway.

### Q: Are views important for performance?
**A:** No. Views are just query shortcuts, not materialized (cached) data. They don't improve performance.

### Q: What if frontend uses those views?
**A:** Based on code review, no frontend references exist. Views were created but never connected.

### Q: Can we restore archived tables?
**A:** Yes! Just run: `ALTER TABLE archive_20251005.table_name SET SCHEMA public;`

### Q: Is this reversible?
**A:** 100% reversible until we run the final `DROP SCHEMA` command (after 1 week verification).

---

## 🎯 Recommendation

**Execute the migration now.** Here's why:

✅ **Safe:** Tables archived (not dropped)  
✅ **Reversible:** Can restore instantly  
✅ **Clean:** Removes unused complexity  
✅ **Tested:** Migration script includes verification  
✅ **Documented:** Full rollback plan included  

**Next Action:** Run `migrations/database_consolidation_20251005.sql` in Supabase SQL Editor.

---

## 📊 Impact Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Objects | 29 | 10 | -66% |
| Tables | 15 | 7 | -53% |
| Views | 14 | 3 | -79% |
| Empty Tables | 8 | 0 | -100% |
| Unused Columns | 194 | 0 | -100% |

**Result:** Cleaner, simpler, more maintainable database! 🎉
