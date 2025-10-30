# Pipeline & Database Improvements

## Summary of Changes

### 1. ✅ FIXED: Sector Column Not Showing on Frontend
**Problem:** Sector data exists in database and column visibility toggle, but not displaying in table  
**Root Cause:** Users with old localStorage data don't have `sector` key, so it defaults to hidden  
**Solution:** Changed sector column from `required: true` to `required: false` in ColumnVisibilityToggle  
**User Action:** Click "Columns" button → Check "Sector" box OR click "Show All"  

**Files Modified:**
- `frontend/src/components/dashboard/ColumnVisibilityToggle.tsx` (line 48)

---

### 2. ⏳ TODO: Clean Up Duplicate Logging in Pipeline

**Current Issues:**
- Pipeline logs appear multiple times (backend logger + console)
- Verbose output makes it hard to track progress
- Some log levels are inconsistent

**Recommended Approach:**

#### Option A: Reduce Log Verbosity (Quick Win)
```python
# backend/utils/logger.py
# Change default level from DEBUG to INFO
logger.setLevel(logging.INFO)  # Less verbose
```

#### Option B: Consolidate Loggers (Better)
1. Remove duplicate loggers in pipeline phases
2. Use single configured logger instance
3. Add `--verbose` CLI flag for debug mode

**Implementation:**
```python
# run_pipeline_and_push.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Set log level based on flag
if args.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
```

#### Option C: Structured Logging with Progress Bar
- Use `rich` library for clean progress bars
- Show phase-level progress instead of every API call
- Separate detailed logs to file, minimal output to console

**Recommendation:** Start with Option A (5 min), then do Option B for cleaner architecture.

---

### 3. ⏳ TODO: Delete Signal Runs from Database

**Current State:** No UI or API endpoint to delete old signal runs

**Recommended Implementation:**

#### Backend: Add Delete Endpoint
```python
# backend/storage/database.py

async def delete_signal_run(self, run_id: str) -> bool:
    """
    Delete a signal run and all associated data.
    
    Cascading deletes:
    - signals table (run_id FK)
    - factor_technical (run_id FK)
    - factor_fundamental (run_id FK)
    - factor_news_macro (run_id FK)
    - factor_social_alternative (run_id FK)
    - factor_risk_stability (run_id FK)
    - factor_institutional_smart_money (run_id FK)
    - performance_tracking (signal_id FK → cascade)
    - analytics (if stored per run)
    
    Args:
        run_id: UUID of the signal run to delete
        
    Returns:
        bool: True if successful, False if run not found
    """
    try:
        # Delete signal_run (cascades to all related tables via FK constraints)
        result = self.supabase.table('signal_runs').delete().eq('id', run_id).execute()
        
        if result.data:
            self.logger.info(f"[SUCCESS] Deleted signal run {run_id} and all associated data")
            return True
        else:
            self.logger.warning(f"[WARNING] Signal run {run_id} not found")
            return False
            
    except Exception as e:
        self.logger.error(f"[ERROR] Failed to delete signal run {run_id}: {e}")
        return False
```

#### Frontend: Add Delete Button to Run Selector
```tsx
// frontend/src/components/dashboard/SignalsDashboard.tsx

const handleDeleteRun = async (runId: string) => {
  if (!confirm('Delete this signal run? This cannot be undone.')) return;
  
  try {
    const response = await fetch(`/api/signals/runs/${runId}`, {
      method: 'DELETE',
    });
    
    if (response.ok) {
      toast.success('Signal run deleted successfully');
      refetch();  // Refresh runs list
    } else {
      toast.error('Failed to delete signal run');
    }
  } catch (error) {
    toast.error('Error deleting signal run');
  }
};

// Add delete icon next to each run in dropdown
<DropdownMenuItem>
  <span>{run.label}</span>
  <Button
    size="sm"
    variant="ghost"
    onClick={(e) => {
      e.stopPropagation();
      handleDeleteRun(run.id);
    }}
  >
    <Trash2 className="h-4 w-4 text-red-500" />
  </Button>
</DropdownMenuItem>
```

#### Database: Ensure Cascade Deletes Are Set
```sql
-- Check foreign key constraints have ON DELETE CASCADE
ALTER TABLE signals
  DROP CONSTRAINT signals_run_id_fkey,
  ADD CONSTRAINT signals_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES signal_runs(id)
    ON DELETE CASCADE;

ALTER TABLE factor_technical
  DROP CONSTRAINT factor_technical_run_id_fkey,
  ADD CONSTRAINT factor_technical_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES signal_runs(id)
    ON DELETE CASCADE;

-- Repeat for all 6 factor tables...
```

**Where to Implement:**
1. **Backend API:** `backend/api/signals.py` (new DELETE endpoint)
2. **Frontend UI:** Add trash icon to run selector dropdown
3. **Database Migration:** Ensure CASCADE constraints exist
4. **Permissions:** Add RLS policy allowing authenticated users to delete their own runs

**Alternative: Archive Instead of Delete**
```python
# Safer approach - mark as archived instead of hard delete
async def archive_signal_run(self, run_id: str) -> bool:
    result = self.supabase.table('signal_runs')\
        .update({'archived': True})\
        .eq('id', run_id)\
        .execute()
    return bool(result.data)
```

---

## Testing Checklist

### Before Running Pipeline
- [ ] Commit Phase 2 `is_valid` fix (already done)
- [ ] Push sector column visibility fix
- [ ] Verify no other `.is_valid()` calls exist

### Pipeline Run
```bash
python run_pipeline_and_push.py
```

### Expected Results
- ✅ No "'bool' object is not callable" errors
- ✅ News/Macro success rate > 70% (was 11%)
- ✅ VIX, Treasury, SPY correlation factors have values
- ✅ news_macro_score column has non-zero values
- ✅ Database factors show actual numbers, not all NaN

### Frontend Verification
1. Open https://vanpiq.com
2. Click "Columns" dropdown
3. Check "Sector" checkbox
4. Verify sector column appears in table
5. Verify news/macro scores are non-zero
6. Verify tickers are clickable links to Yahoo Finance

---

## Next Steps

1. **Immediate:** Run pipeline to verify news/macro fix
2. **Short-term:** Reduce logging verbosity (Option A)
3. **Medium-term:** Implement delete signal runs feature
4. **Long-term:** Add structured logging with progress bars
