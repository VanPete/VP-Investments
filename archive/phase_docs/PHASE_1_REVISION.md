# Phase 1 Revision: Skip Dead Code Deletion
**Date:** October 9, 2025  
**Status:** Phase 1 Revised

---

## Analysis Update

### Original Phase 1 Plan:
Delete `_apply_basic_enhancements()` method (lines 2524-2701) as "dead code"

### Finding:
Method is NOT dead code - it's a **fallback handler**

**Code Flow:**
```python
def _apply_signal_enhancements(self, signals):
    try:
        # Try to use consolidated enhancer from signal_processing module
        from backend.integrations.signal_processing import enhance_signals_batch
        return enhance_signals_batch(signals)
    except ImportError:
        # FALLBACK: Use basic enhancement if module unavailable
        return self._apply_basic_enhancements(signals)
    except Exception:
        # FALLBACK: Use basic enhancement if module fails
        return self._apply_basic_enhancements(signals)
```

**Conclusion:** This is defensive coding - keeps system working even if signal_processing module fails. Should KEEP, not delete.

---

## Revised Plan: Skip to Phase 2

**New Approach:**
- ~~Phase 1: Delete dead code~~ → **SKIP** (no truly dead code found)
- **Phase 2: Fix core.py duplication** → **START HERE** (safer, valuable)
- Phase 3: Move Reddit logic
- Phase 4: Move financial fetching  
- Phase 5: Move beta calculation

---

## Proceeding to Phase 2

**Target:** Remove duplicate enums from signals.py
- SignalType (duplicated in core.py and signals.py)
- TradeType (duplicated in core.py and signals.py)

**Action:** Update signals.py to import from core.py instead

**Risk:** ⭐ Low
**Time:** 30 minutes
**Testing:** test_single_signal.py + import checks

---

**Next:** Execute Phase 2
