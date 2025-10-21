# Phase 1-4 Cleanup Action Plan
**Priority**: Immediate cleanup to align with v3.1
**Timeline**: 1-2 days
**Goal**: Remove all old code and ensure 100% v3.1 architecture

---

## 🎯 Immediate Actions (Do Today)

### 1. Fix News Integration Bug ⚠️ CRITICAL
**File**: `backend/integrations/news.py`
**Problem**: Trying to import Ticker from backend.integrations.yfinance instead of yfinance package
**Impact**: 0/35 tickers getting news data

**Fix**:
```python
# Change from:
from backend.integrations import yfinance
ticker = yfinance.Ticker(symbol)

# To:
import yfinance as yf
ticker = yf.Ticker(symbol)
```

### 2. Delete Old Phase 1 Code
**File**: `backend/phases/phase1_fetch.py`
**Lines**: 725-950 (approximately)
**Action**: Delete all commented-out methods:
- `_fetch_phase3_fundamentals`
- `_compute_reddit_score`
- Any other commented methods

**Command**:
```bash
# Backup first
cp backend/phases/phase1_fetch.py backend/phases/phase1_fetch.py.backup

# Then manually delete lines 725+ that are commented
```

### 3. Update Pipeline Imports
**File**: `backend/pipeline.py`
**Lines**: 54-58

**Change from**:
```python
from archive.phase2_normalize import Phase2Normalizer
from archive.phase4_assemble import Phase4Assembler
from backend.phases.phase5_persist import Phase5Persister
from backend.phases.phase6_post_ops import Phase6PostOps
from backend.core.signals import SignalScorer
```

**To**:
```python
from backend.phases.phase2_calculate import Phase2Calculator
from backend.phases.phase3_normalize import Phase3Normalizer
from backend.phases.phase4_score_assemble import Phase4ScoreAssembler
from backend.phases.phase5_persist import Phase5Persister
from backend.phases.phase6_post_ops import Phase6PostOps
```

### 4. Mark SignalScorer as Deprecated
**File**: `backend/core/signals.py`
**Line**: 1743 (class SignalScorer)

**Add deprecation warning**:
```python
@deprecated("Use Phase2Calculator + Phase3Normalizer + Phase4ScoreAssembler instead")
class SignalScorer:
    """
    DEPRECATED: This class is being replaced by the v3.1 modular pipeline.
    - Phase 2: Calculate factors
    - Phase 3: Normalize factors  
    - Phase 4: Score & assemble
    
    Will be removed in v3.2
    """
```

---

## 📋 Quick Cleanup Checklist

### Files to Modify
- [ ] `backend/integrations/news.py` - Fix yfinance import
- [ ] `backend/phases/phase1_fetch.py` - Delete commented code (lines 725+)
- [ ] `backend/pipeline.py` - Update imports to v3.1 phases
- [ ] `backend/core/signals.py` - Add deprecation warning

### Files to Review (Don't Delete Yet)
- [ ] `archive/phase2_normalize.py` - Keep until Phase 5 is updated
- [ ] `archive/phase4_assemble.py` - Keep until Phase 5 is updated
- [ ] `backend/core/signals.py` - Keep SignalScorer until migration complete

### Tests to Update
- [ ] Update any tests importing old phase modules
- [ ] Run `test_integrated_v3_1.py` after changes
- [ ] Verify no imports from archive/ directory

---

## 🔍 Verification Steps

After completing cleanup:

1. **Check Imports**:
```bash
# Search for any remaining archive imports
grep -r "from archive" backend/
grep -r "import archive" backend/
```

2. **Run Tests**:
```bash
python test_integrated_v3_1.py
```

3. **Check News Integration**:
Look for: `News fetch complete: X/35 tickers with news` (X should be > 0)

4. **Verify Phase Execution**:
Ensure all 4 phases complete successfully:
- ✅ Phase 1: Fetch
- ✅ Phase 2: Calculate  
- ✅ Phase 3: Normalize
- ✅ Phase 4: Score

---

## 🚀 Next Steps (After Cleanup)

Once immediate cleanup is done:

1. **Add Input Validation** (2 hours)
   - Phase 1: Validate RawYFinanceData before return
   - Phase 2: Check for empty/invalid data
   - Phase 4: Validate score ranges

2. **Add Error Handling** (3 hours)
   - Phase 2: Per-factor try-catch
   - Phase 3: Zero-variance detection
   - Phase 4: Score bounds checking

3. **Update Phase 5** (4-6 hours)
   - Modify to accept Phase 4 output format
   - Remove dependency on old SignalScorer
   - Update database schema if needed

4. **Full Pipeline Integration** (2-3 hours)
   - Update `backend/pipeline.py` main execution flow
   - Wire phases 1→2→3→4→5→6
   - Test end-to-end

---

## 📊 Expected Results

**Before Cleanup**:
- Mixed v3.0/v3.1 architecture
- Old code cluttering files
- News fetch broken (0% success)
- Pipeline imports from archive/

**After Cleanup**:
- Pure v3.1 architecture
- Clean, focused phase files
- News fetch working (>50% success expected)
- Pipeline imports from backend/phases/
- Foundation ready for Phase 5 migration

---

## ⚠️ Rollback Plan

If something breaks:

1. **Restore backups**:
```bash
cp backend/phases/phase1_fetch.py.backup backend/phases/phase1_fetch.py
```

2. **Revert git changes**:
```bash
git checkout backend/pipeline.py
git checkout backend/integrations/news.py
```

3. **Run old test** to verify:
```bash
# Use old test if new one fails
```

---

## 💬 Questions to Answer

Before starting:

1. **Do we have any active code depending on SignalScorer?**
   - Check Phase 5, Phase 6, frontend API
   - If yes, create adapter/shim first

2. **Is Phase 5 ready to be updated?**
   - If no, keep SignalScorer and create compatibility layer
   - If yes, update Phase 5 simultaneously

3. **Do we need to maintain backward compatibility?**
   - If yes, keep old code but deprecate
   - If no, delete immediately

---

## 📝 Completion Criteria

✅ Cleanup is complete when:

1. No commented-out old methods in phase files
2. No imports from `archive/` in active code
3. `test_integrated_v3_1.py` passes
4. News fetch shows >0 successful fetches
5. All phase files have consistent v3.1 logging format
6. Pipeline.py uses only v3.1 phase imports

---

**Estimated Time**: 4-6 hours
**Difficulty**: Low-Medium
**Risk**: Low (all changes are deletions or import updates)
**Testing**: test_integrated_v3_1.py validates success
