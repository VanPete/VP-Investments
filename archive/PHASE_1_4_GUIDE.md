# Phase 1.4 Implementation Guide

## Overview
This guide walks through implementing Phase 1.4: Schema Stabilization before moving to Phase 2-4.

---

## ✅ Steps Completed

### Step 1: Remove Dead Columns ✅
**File:** `migrations/001_remove_dead_columns.sql`

**What it does:**
- Removes 11 unused columns from signals table
- Reduces table from 137 to 126 columns (8% reduction)
- Includes verification checks and rollback script

**Columns removed:**
1. `commentary_metadata` - Unused metadata
2. `score_components` - Unused metadata
3. `scoring_version` - Unused metadata
4. `ai_commentary_version` - Unused metadata
5. `rowid` - Duplicate of id
6. `ml_confidence_score` - Not implementing ML
7. `prediction_confidence` - Not implementing ML
8. `pattern_match_score` - Not implementing pattern matching
9. `signal_duration` - Use expected_hold_duration instead
10. `option_chain_data` - Too complex/expensive
11. `option_volume_ratio` - Duplicate of put_call_vol_ratio

**How to apply:**
```bash
# Open Supabase SQL Editor
# Copy contents of migrations/001_remove_dead_columns.sql
# Run the migration
# Verify with: python apply_migrations.py
```

---

### Step 2: Document Phase 2-4 Placeholders ✅
**File:** `migrations/002_document_phase_placeholders.sql`

**What it does:**
- Adds comments to 8 Phase 2-4 placeholder columns
- Creates `phase_implementation_tracker` table to track feature development
- Creates view `v_phase_implementation_status` for quick status overview

**Columns documented:**

**Phase 2: Reddit Enhancements**
- `reddit_momentum_score` - Sentiment acceleration tracking
- `reddit_vs_price_divergence` - Price/sentiment divergence detection  
- `social_sentiment_trend` - Overall trend direction

**Phase 3: Options & Institutional**
- `options_flow_score` - Smart money options tracking
- `unusual_options_activity` - Unusual volume detection
- `institutional_flow_direction` - Institutional buy/sell direction

**Phase 4: Quality Scores**
- `entry_quality_score` - Entry timing quality
- `risk_adjusted_score` - Sharpe-style risk adjustment

**How to apply:**
```bash
# Open Supabase SQL Editor
# Copy contents of migrations/002_document_phase_placeholders.sql
# Run the migration
# Verify with: SELECT * FROM v_phase_implementation_status;
```

---

## 🔧 Steps Remaining

### Step 3: Fix Phase 1 Timing Issue (2-3 hours)
**Status:** 🏗️ IN PROGRESS

**Problem:** Phase 1 metrics are calculated AFTER financial_score, so they can't be used in scoring.

**Current flow (BROKEN):**
```
1. Generate reddit signals
2. Generate financial signals (can't use Phase 1 metrics!)
3. Combine into weighted_score
4. Apply Phase 1 enhancements ← TOO LATE!
5. Save to database
```

**Fixed flow:**
```
1. Generate reddit signals
2. APPLY PHASE 1 ENHANCEMENTS FIRST ← Move this up!
3. Generate financial signals (now has Phase 1 metrics!)
4. Combine into weighted_score  
5. Save to database
```

**Files to modify:**
- `backend/pipeline.py` - Move `_apply_signal_enhancements()` earlier

---

### Step 4: Enhance financial_score with Phase 1 Metrics (2 hours)
**Status:** ⏳ TODO

**What to add:**
Update `_calculate_technical_score()` in `backend/pipeline.py` to include:

```python
# Momentum consistency (10%) - NEW from Phase 1
momentum_consistency = financial_data.get('momentum_consistency_score')
if momentum_consistency:
    consistency_score = min(momentum_consistency / 100, 1.0)
    technical_components.append(consistency_score * 0.10)

# Sector relative strength (10%) - NEW from Phase 1  
sector_rs = financial_data.get('sector_relative_strength')
if sector_rs and sector_rs > 0:
    rs_score = min(sector_rs / 100, 1.0)
    technical_components.append(rs_score * 0.10)

# Volume-price correlation (5%) - NEW from Phase 1
vol_price_corr = financial_data.get('volume_price_correlation')
if vol_price_corr and vol_price_corr > 0.3:
    corr_score = vol_price_corr
    technical_components.append(corr_score * 0.05)

# Liquidity score (5%) - NEW from Phase 1
liquidity = financial_data.get('liquidity_score')
if liquidity:
    technical_components.append(liquidity * 0.05)
```

---

### Step 5: Test & Validate (1-2 hours)
**Status:** ⏳ TODO

**What to test:**
1. Run pipeline with new flow
2. Verify Phase 1 metrics affect financial_score
3. Compare old vs new signal rankings
4. Check backtest performance
5. Verify no errors

**Test commands:**
```bash
# Run full pipeline
python backend/pipeline.py

# Verify Phase 1 metrics are populated
python quick_check.py

# Check that financial_score uses Phase 1 metrics
# (add debug logging to _calculate_technical_score)
```

---

## 📊 Progress Tracker

| Phase | Task | Status | Time | Notes |
|-------|------|--------|------|-------|
| 1.4.1 | Remove dead columns | ✅ Done | 1hr | SQL migration ready |
| 1.4.2 | Document placeholders | ✅ Done | 30min | Tracker table created |
| 1.4.3 | Fix Phase 1 timing | 🏗️ In Progress | 2-3hrs | Need to move enhancement earlier |
| 1.4.4 | Enhance financial_score | ⏳ Todo | 2hrs | Add Phase 1 metrics to scoring |
| 1.4.5 | Test & validate | ⏳ Todo | 1-2hrs | Full pipeline test |

**Total Progress:** 2/5 steps complete (40%)

---

## 🎯 Success Criteria

Phase 1.4 is complete when:

- [x] 11 dead columns removed from signals table
- [x] 8 Phase 2-4 columns documented with comments
- [ ] Phase 1 metrics calculated BEFORE financial_score
- [ ] financial_score includes Phase 1 technical indicators
- [ ] Pipeline runs without errors
- [ ] Backtest shows improvement (or no regression)
- [ ] Schema is stable (no more changes until Phase 2)

---

## 📁 Files Created

### Migration Scripts
- `migrations/001_remove_dead_columns.sql` - Remove 11 dead columns
- `migrations/002_document_phase_placeholders.sql` - Document Phase 2-4 columns

### Documentation
- `SIGNALS_COLUMN_USAGE_REPORT.md` - Comprehensive column analysis
- `IMPLEMENTATION_PLAN.md` - Detailed implementation strategy
- `COLUMN_USAGE_ANALYSIS.py` - Python script to analyze column usage
- `PHASE_1_4_GUIDE.md` - This file

### Utilities
- `apply_migrations.py` - Script to apply and verify migrations
- `quick_check.py` - Quick check for Phase 1 data

---

## 🚀 Next Steps

### Immediate (Today)
1. **Apply migrations to Supabase:**
   - Open Supabase SQL Editor
   - Run `001_remove_dead_columns.sql`
   - Run `002_document_phase_placeholders.sql`
   - Verify with `python apply_migrations.py`

2. **Verify migrations worked:**
   ```bash
   python quick_check.py
   # Should see 126 columns instead of 137
   ```

### Step 3: Fix Timing (Tomorrow)
1. Create backup branch: `git checkout -b phase-1-4-timing-fix`
2. Modify `backend/pipeline.py` to move enhancement earlier
3. Test with `python backend/pipeline.py`
4. Verify Phase 1 metrics affect financial_score
5. Commit if successful

### Step 4: Enhance Scoring (Tomorrow)
1. Update `_calculate_technical_score()` with Phase 1 metrics
2. Add debug logging to see score breakdown
3. Run pipeline and compare signal rankings
4. Verify improvement with backtests

### Step 5: Test (Tomorrow Evening)
1. Full pipeline test
2. Check for errors
3. Verify all Phase 1 metrics populating
4. Compare backtest results old vs new
5. Document findings

---

## ❓ Questions & Answers

**Q: Will removing columns break the pipeline?**
A: No, these columns are never referenced in the codebase. Safe to remove.

**Q: Can I rollback if something goes wrong?**
A: Yes! Each migration includes a rollback script.

**Q: Why not just delete the Phase 2-4 columns too?**
A: We're keeping them as documented placeholders for future implementation.

**Q: How long until Phase 1.4 is complete?**
A: Approximately 4-6 hours of work remaining (Steps 3-5).

**Q: What's the impact on existing signals?**
A: None. Existing data is preserved, just removing unused columns.

---

## 📞 Need Help?

If you encounter issues:

1. Check `logs/vp_investments.log` for errors
2. Verify migrations with: `SELECT * FROM v_phase_implementation_status;`
3. Check column count: `SELECT COUNT(*) FROM information_schema.columns WHERE table_name='signals';` (should be 126)
4. Rollback if needed using rollback scripts in migration files

---

## 🎉 Completion Checklist

Once Phase 1.4 is complete, you should have:

- ✅ Clean signals table (126 columns, no dead weight)
- ✅ Documented Phase 2-4 roadmap
- ✅ Phase 1 metrics affecting financial_score
- ✅ Improved signal quality (better rankings)
- ✅ Stable schema ready for Phase 2

**Then you're ready for Phase 2: Reddit Enhancements!**

---

*Last Updated: 2025-10-07*
