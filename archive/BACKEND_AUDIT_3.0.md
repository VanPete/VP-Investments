# Backend 3.0 Architecture Audit
**Generated:** October 14, 2025  
**Purpose:** Identify files needing 3.0 refactor before pipeline.py work

---

## 📊 File Status Summary

### ✅ **COMPLETED - 3.0 Architecture** (5 files)
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `integrations/ai.py` | 841 | ✅ Complete | Phase 6 only, OpenAI integration, full implementation |
| `integrations/news.py` | 283 | ✅ Complete | Phase 1+3, Yahoo Finance + TextBlob |
| `integrations/reddit.py` | 378 | ✅ Complete | Phase 1+3, PRAW + sentiment scoring |
| `integrations/yfinance.py` | 332 | ✅ Complete | Phase 1 only, pure fetcher |
| `utils/calculator.py` | 579 | ✅ Complete | NEW - Phase 2+3 calculations |
| `integrations/cache.py` | 459 | ✅ Complete | Phase 1 cache layer with Supabase |

### ⚠️ **NEEDS REVIEW/UPDATE** (7 files)
| File | Lines | Issue | Action Needed |
|------|-------|-------|---------------|
| `integrations/signal_processing.py` | 2,491 | ❌ Pre-3.0 monolith | **DELETE** - Logic moved to signals.py & calculator.py |
| `integrations/backtest.py` | 1,613 | ❌ Old architecture | **REFACTOR** - Split to Phase 6 + utils |
| `integrations/performance_tracker.py` | 382 | ⚠️ Mixed phases | **REFACTOR** - Make Phase 6 only |
| `core/signals.py` | 4,568 | ⚠️ Needs audit | **REVIEW** - Check for Phase 1-5 compatibility |
| `core/backtest.py` | 1,055 | ⚠️ Duplicate? | **REVIEW** - vs integrations/backtest.py |
| `utils/observability.py` | 405 | ⚠️ Async issue | **FIX** - emit_metric is not async but decorated as such |
| `integrations/stocktwits.py` | 158 | ℹ️ Placeholder | **KEEP** - Valid placeholder, no action needed |

### ✅ **OK AS-IS** (5 files)
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `storage/database.py` | 1,390 | ✅ OK | Infrastructure - Phase-agnostic |
| `core/core.py` | 742 | ✅ OK | Constants & enums - Phase-agnostic |
| `core/config.py` | 375 | ✅ OK | Configuration - Phase-agnostic |
| `utils/logger.py` | 368 | ✅ OK | Logging utility - Phase-agnostic |
| `api/api.py` | 703 | ✅ OK | REST API - Phase-agnostic |
| `core/cli.py` | 479 | ✅ OK | CLI interface - Phase-agnostic |

### 🚧 **PENDING** (2 files)
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `pipeline.py` | 3,316 | 🚧 Next task | Needs 6-phase refactor (our next task) |
| `core/signals.py` | 4,568 | 🚧 Needs audit | May need updates for Phase 3+4 |

---

## 🔍 Detailed Analysis

### 1. **signal_processing.py** (2,491 lines) - ❌ DELETE

**Issues:**
- Pre-3.0 monolithic architecture
- Mixes Phase 1 (fetching), Phase 2 (parsing), Phase 3 (scoring)
- Duplicates functionality now in:
  - `calculator.py` - Technical/fundamental calculations
  - `signals.py` - Signal assembly and scoring
  - Integration files - Data fetching

**Evidence from file:**
```python
"""
Comprehensive Signal Processing Module
=====================================
Provides complete signal processing functionality including:
- Signal classification and risk assessment
- Signal enhancement with calculated fields and live market data
- Risk scoring and categorization  
...
"""
```

**Action:** 
```bash
# Move to archive
mv backend/integrations/signal_processing.py archive/signal_processing_old_monolith.py
```

**Migration completed in:**
- Technical calculations → `utils/calculator.py`
- Signal assembly → `core/signals.py` (Phase 4)
- Risk scoring → `core/signals.py` (Phase 3)

---

### 2. **integrations/backtest.py** (1,613 lines) - ❌ REFACTOR

**Issues:**
- Old architecture with mixed phases
- Performance tracking should be Phase 6 (Post-Ops)
- Contains both data fetching AND analysis

**Current structure:**
```python
class BacktestEngine:
    def __init__(self, db=None):
        self.intervals = [1, 3, 7, 10, 30]  # Track returns
    
    async def calculate_signal_performance(...)  # Mixed phase logic
    async def track_historical_returns(...)  # Data fetching
    async def _calculate_returns(...)  # Calculations
```

**3.0 Target:**
- **Phase 1:** Fetch historical price data (use yfinance fetcher)
- **Phase 6:** Calculate performance metrics after signal generation
- Move pure calculations to `utils/calculator.py`

**Action:**
1. Extract data fetching → Use `yfinance.fetch_historical_data()`
2. Extract calculations → Move to `calculator.py` or Phase 6 helper
3. Keep backtest orchestration as Phase 6 operation
4. Rename to `integrations/backtest_v3.py` or move logic to pipeline Phase 6

---

### 3. **integrations/performance_tracker.py** (382 lines) - ⚠️ REFACTOR

**Issues:**
- Mixes data fetching with calculations
- Should be Phase 6 only (Post-Ops)

**Current structure:**
```python
class PerformanceTracker:
    async def calculate_signal_performance(signals: List[Dict]) -> List[Dict]:
        # Gets SPY data (Phase 1)
        spy_data = await self._get_benchmark_data()
        # Calculates returns (Phase 6)
        performance_data = await self._calculate_historical_returns(...)
```

**3.0 Target:**
- Phase 1: Fetch SPY data separately via yfinance
- Phase 6: Calculate performance metrics using cached data
- No mid-pipeline API calls

**Action:**
1. Remove `_get_benchmark_data()` - use yfinance fetcher in Phase 1
2. Update to accept pre-fetched data bundles
3. Pure calculations only in Phase 6

---

### 4. **core/signals.py** (4,568 lines) - 🔍 NEEDS AUDIT

**Status:** Large file with signal assembly logic

**What to check:**
1. ✅ Phase 3 scoring logic present (normalize_direct, normalize_inverted)
2. ❓ Does it fetch data mid-processing? (should use bundles only)
3. ❓ Does Phase 4 (Assembly) work with new bundle structures?
4. ❓ Any database writes outside Phase 5?

**Questions for you:**
- Should we read this file together to identify issues?
- Is this file already 3.0 compatible or does it have old logic?

---

### 5. **core/backtest.py** (1,055 lines) - ❓ DUPLICATE CHECK

**Issue:** Two backtest files exist:
- `backend/integrations/backtest.py` (1,613 lines)
- `backend/core/backtest.py` (1,055 lines)

**Questions:**
- Are these duplicates?
- Which one is active?
- Should we consolidate or delete one?

**Action:** Need to compare both files to determine relationship

---

### 6. **utils/observability.py** (405 lines) - 🐛 BUG FIX

**Issue:** Type error in code:

```python
def emit_metric(metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None, **kwargs) -> None:
    """Emit a metric for monitoring and observability"""
    # Function is NOT async but...

@track_performance("signal_generation")
async def async_wrapper(*args, **kwargs):
    # ... it's being awaited here:
    await emit_metric(f"{operation_name}.duration", duration)  # ❌ ERROR
```

**Fix:** Either:
1. Make `emit_metric` async (recommended)
2. Remove `await` from calls (if sync is OK)

**Action:**
```python
# Option 1: Make async (recommended for consistency)
async def emit_metric(metric_name: str, value: float = 1.0, ...) -> None:
    # async implementation

# Option 2: Remove await (if sync is fine)
emit_metric(f"{operation_name}.duration", duration)  # No await
```

---

### 7. **integrations/stocktwits.py** (158 lines) - ✅ OK

**Status:** Placeholder for future API integration

**Structure:**
```python
class StockTwitsIntegrator:
    def __init__(self):
        self.enabled = False  # Placeholder mode

async def fetch_stocktwits_bundle(ticker: str) -> Dict:
    """Phase 1 compatible placeholder"""
    return {"available": False, ...}
```

**Action:** None needed - valid placeholder

---

## 📋 Action Plan

### **Immediate Actions (Before pipeline.py refactor):**

1. **DELETE signal_processing.py**
   ```bash
   mv backend/integrations/signal_processing.py archive/signal_processing_old_monolith.py
   ```

2. **FIX observability.py bug**
   - Make `emit_metric` async or remove awaits

3. **REVIEW core/signals.py**
   - Check for mid-pipeline API calls
   - Verify Phase 3+4 compatibility with new bundles
   - Ensure no Phase 1/2 logic present

4. **INVESTIGATE backtest duplication**
   - Compare `core/backtest.py` vs `integrations/backtest.py`
   - Delete or consolidate

5. **REFACTOR backtest.py & performance_tracker.py**
   - Make Phase 6 only
   - Remove data fetching (use Phase 1 bundles)
   - Pure calculations using cached data

### **Questions for You:**

1. **signal_processing.py:** OK to delete? All logic migrated?
2. **core/signals.py:** Should we audit this together now?
3. **backtest.py duplication:** Which file is the "source of truth"?
4. **Performance tracking:** Should it be Phase 6 in pipeline or separate post-processing?
5. **observability.py:** Prefer async emit_metric or sync?

---

## 📈 Progress Tracking

### Files Refactored to 3.0:
- ✅ integrations/ai.py (841 lines)
- ✅ integrations/news.py (283 lines)
- ✅ integrations/reddit.py (378 lines)
- ✅ integrations/yfinance.py (332 lines)
- ✅ integrations/cache.py (459 lines)
- ✅ utils/calculator.py (579 lines) - NEW

### Files Pending Cleanup:
- ❌ integrations/signal_processing.py (2,491 lines) → DELETE
- ⚠️ integrations/backtest.py (1,613 lines) → REFACTOR
- ⚠️ integrations/performance_tracker.py (382 lines) → REFACTOR
- 🔍 core/signals.py (4,568 lines) → AUDIT
- ❓ core/backtest.py (1,055 lines) → INVESTIGATE
- 🐛 utils/observability.py (405 lines) → FIX BUG

### Next Major Task:
- 🚧 pipeline.py (3,316 lines) → 6-PHASE REFACTOR

---

## 🎯 Recommendation

**Before proceeding to pipeline.py refactor, we should:**

1. ✅ Delete `signal_processing.py` (confirmed obsolete)
2. ✅ Fix `observability.py` async bug
3. 🔍 Audit `core/signals.py` together (it's critical for Phase 3+4)
4. ❓ Resolve backtest duplication question
5. ⏳ Defer `backtest.py` + `performance_tracker.py` refactor until after pipeline.py

**Rationale:** 
- `signals.py` is used BY pipeline, so must be 3.0 compatible first
- Backtest/performance are called FROM pipeline Phase 6, can be refactored after
- Signal processing is obsolete and safe to delete now

**Your input needed on:**
- signal_processing.py deletion approval
- signals.py audit priority
- backtest.py consolidation strategy
- observability.py async preference

---

