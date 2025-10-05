# 🗂️ Database Consolidation Plan - "Less is More"

**Date:** 2025-10-05  
**Goal:** Simplify database schema by removing redundant/unused tables and views

---

## 📊 Current State

### Tables (15):
- **Core:** signals (340 rows, 142 cols) ✅ KEEP
- **Reference:** company_tickers (7,638 rows) ✅ KEEP  
- **Metadata:** runs (9 rows), guardrails_config (6 rows) ✅ KEEP
- **AI:** ai_strategies (122 rows), signal_scoring_factors (18 rows) ✅ KEEP
- **Tracking:** backtest_interval_tracking (1,700 rows) ✅ KEEP

### Empty Tables (8 tables, 0 rows):
- ❌ ai_strategy_performance
- ❌ backtest_trades
- ❌ backtests
- ❌ market_conditions
- ❌ scoring_calibration_log
- ❌ signal_calibration_log
- ❌ signal_performance
- ❌ signal_performance_history

### Views (14):
Too many! Most are unused and provide no value.

---

## 🎯 Consolidation Strategy

### Phase 1: Drop Empty Tables (Immediate) 🔴

**Drop 8 empty tables that are NOT being used:**

```sql
-- These tables were created for "future features" but never implemented
DROP TABLE IF EXISTS ai_strategy_performance CASCADE;
DROP TABLE IF EXISTS backtest_trades CASCADE;
DROP TABLE IF EXISTS backtests CASCADE;
DROP TABLE IF EXISTS market_conditions CASCADE;
DROP TABLE IF EXISTS scoring_calibration_log CASCADE;
DROP TABLE IF EXISTS signal_calibration_log CASCADE;
DROP TABLE IF EXISTS signal_performance CASCADE;
DROP TABLE IF EXISTS signal_performance_history CASCADE;
```

**Rationale:**
- 0 rows in all 8 tables
- No backend code populating them
- Not referenced in pipeline.py
- Creating tech debt and confusion
- Can recreate later IF actually needed

**Impact:** 
- ✅ Removes 194 unused columns (31+21+34+16+7+18+19+44)
- ✅ Simplifies schema maintenance
- ✅ Faster database backups
- ✅ Less cognitive overhead

---

### Phase 2: Consolidate Views (High Priority) 🟡

**Current: 14 views** - Too many! Most duplicate functionality.

**Keep Only 3 Essential Views:**

1. **`v_recent_signals`** - Quick dashboard view of latest signals
2. **`backtest_eligible_signals`** - Used by pipeline backtest logic
3. **`signal_performance_summary`** - Performance tracking (if we implement it)

**Drop 11 Redundant Views:**

```sql
-- Most of these can be replaced with simple SELECT queries
DROP VIEW IF EXISTS backtest_summary CASCADE;
DROP VIEW IF EXISTS top_signal_factors CASCADE;
DROP VIEW IF EXISTS trade_analysis CASCADE;
DROP VIEW IF EXISTS v_active_ai_strategies CASCADE;
DROP VIEW IF EXISTS v_ai_strategy_metrics CASCADE;
DROP VIEW IF EXISTS v_recent_signal_performance CASCADE;
DROP VIEW IF EXISTS v_sector_relative_performance CASCADE;
DROP VIEW IF EXISTS v_signal_factor_analysis CASCADE;
DROP VIEW IF EXISTS v_signal_performance_by_score CASCADE;
DROP VIEW IF EXISTS v_top_momentum_signals CASCADE;
DROP VIEW IF EXISTS v_top_performers CASCADE;
```

**Rationale:**
- Not used in application code
- Easily replicated with ad-hoc queries
- Views don't improve performance (they're query shortcuts, not caching)
- Creating maintenance burden (schema changes break views)

---

### Phase 3: Keep Core Tables (7 tables) ✅

**Essential Tables to Maintain:**

| Table | Rows | Purpose | Status |
|-------|------|---------|--------|
| **signals** | 340 | Core signals data (142 columns) | ✅ ACTIVE |
| **company_tickers** | 7,638 | Ticker reference data | ✅ ACTIVE |
| **ai_strategies** | 122 | AI-generated trading strategies | ✅ ACTIVE |
| **signal_scoring_factors** | 18 | Scoring weight tracking | ✅ ACTIVE |
| **backtest_interval_tracking** | 1,700 | Backtest execution history | ✅ ACTIVE |
| **runs** | 9 | Pipeline run metadata | ✅ ACTIVE |
| **guardrails_config** | 6 | System configuration | ✅ ACTIVE |

**Total after consolidation: 7 tables + 3 views = 10 objects** (vs current 29)

---

## 📋 Implementation Steps

### Step 1: Backup First ⚠️
```sql
-- Create backup schema
CREATE SCHEMA IF NOT EXISTS archive_20251005;

-- Move empty tables to archive (instead of dropping immediately)
ALTER TABLE ai_strategy_performance SET SCHEMA archive_20251005;
ALTER TABLE backtest_trades SET SCHEMA archive_20251005;
ALTER TABLE backtests SET SCHEMA archive_20251005;
ALTER TABLE market_conditions SET SCHEMA archive_20251005;
ALTER TABLE scoring_calibration_log SET SCHEMA archive_20251005;
ALTER TABLE signal_calibration_log SET SCHEMA archive_20251005;
ALTER TABLE signal_performance SET SCHEMA archive_20251005;
ALTER TABLE signal_performance_history SET SCHEMA archive_20251005;
```

### Step 2: Drop Redundant Views
```sql
-- Drop all unused views (safe - they're just query shortcuts)
DROP VIEW IF EXISTS backtest_summary CASCADE;
DROP VIEW IF EXISTS top_signal_factors CASCADE;
DROP VIEW IF EXISTS trade_analysis CASCADE;
DROP VIEW IF EXISTS v_active_ai_strategies CASCADE;
DROP VIEW IF EXISTS v_ai_strategy_metrics CASCADE;
DROP VIEW IF EXISTS v_recent_signal_performance CASCADE;
DROP VIEW IF EXISTS v_sector_relative_performance CASCADE;
DROP VIEW IF EXISTS v_signal_factor_analysis CASCADE;
DROP VIEW IF EXISTS v_signal_performance_by_score CASCADE;
DROP VIEW IF EXISTS v_top_momentum_signals CASCADE;
DROP VIEW IF EXISTS v_top_performers CASCADE;
```

### Step 3: Verify & Test
```bash
# Run pipeline to ensure nothing breaks
python -m backend.pipeline

# Check tables script still works
python tables.py

# Verify dashboard queries (if any)
```

### Step 4: Drop Archive (After 1 Week)
```sql
-- After confirming nothing broke, permanently remove archive
DROP SCHEMA archive_20251005 CASCADE;
```

---

## 🎯 Final State

**Database Objects: 10 total**
- ✅ 7 Core Tables (actively used)
- ✅ 3 Essential Views (dashboard + backtest)
- ✅ 0 Empty Tables
- ✅ 0 Redundant Views

**Benefits:**
- 📉 66% reduction in database objects (29 → 10)
- 📉 194 fewer unused columns
- 🚀 Faster schema changes
- 🧠 Easier to understand
- 💰 Smaller database size
- ⚡ Faster backups

**Trade-offs:**
- ❌ Lose "nice to have" views (easily recreated)
- ❌ Lose "future feature" tables (recreate IF needed)
- ✅ Gain simplicity and maintainability

---

## ❓ Decision Points

**Do you want to:**

1. **Aggressive Cleanup (Recommended)** 🔥
   - Move 8 empty tables to archive schema
   - Drop 11 unused views immediately
   - Keep only 7 tables + 3 views
   - Result: Clean, minimal schema

2. **Conservative Cleanup** 🐌
   - Drop only views (safe)
   - Keep empty tables "just in case"
   - Result: Still have 15 tables

3. **Hybrid Approach** ⚖️
   - Drop views immediately
   - Archive empty tables (can restore if needed)
   - Drop archive after 1 week if no issues

**My Recommendation: Hybrid Approach (#3)**
- Views are just queries - safe to drop
- Empty tables archived (not lost) but out of the way
- Can restore if suddenly needed
- Permanent drop after verification period

---

## 🔄 Alternative: Keep Some "Future" Tables?

If you think you'll implement these soon, keep:
- ✅ `signal_performance` - If Phase A backtest will populate it
- ❌ All others - Not in any implementation plan

But honestly: **If we need them later, we can recreate them.** SQL files exist.

---

**Which approach do you prefer?** I recommend **Hybrid (#3)** for safety with aggressive cleanup.
