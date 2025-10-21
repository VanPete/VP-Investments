# Backtest & Performance Tracker 3.0 Audit

**Date**: October 14, 2025  
**Files**:
- `backend/integrations/backtest.py` (1,613 lines - ACTIVE)
- `backend/core/backtest.py` (1,027 lines - OBSOLETE)
- `backend/integrations/performance_tracker.py` (379 lines)

**Status**: Phase 1 violations detected, consolidation needed

---

## Executive Summary

Two files have Phase 1 violations (yfinance API calls during backtesting). Additionally, found obsolete duplicate backtest file in `backend/core/` with outdated import paths. Need to:
1. Delete obsolete `backend/core/backtest.py`
2. Refactor active `backend/integrations/backtest.py` to accept pre-fetched data
3. Refactor `performance_tracker.py` to accept pre-fetched data

---

## Critical Discovery: Duplicate Backtest File

### Active File (Used by Pipeline)
**Path**: `backend/integrations/backtest.py`
**Size**: 1,613 lines
**Imports**: `from ..storage.database import get_supabase_database` ✅
**Used By**: `backend/pipeline.py` lines 3073, 3114, 3123

### Obsolete File (Not Used)
**Path**: `backend/core/backtest.py`
**Size**: 1,027 lines
**Imports**: `from vp_investments.storage.supabase_interface` ❌ (old structure)
**Used By**: Nobody (grep search shows zero imports)

**Decision**: DELETE `backend/core/backtest.py` (moved to archive)

---

## backend/integrations/backtest.py - Detailed Audit

### Overview
Comprehensive backtesting engine for signal performance tracking.

**Purpose**:
- Track historical performance (1d, 3d, 7d, 10d returns)
- SPY comparison and beat rates
- Forward-looking metrics (Sharpe, drawdown)
- Signal duration tracking
- Realized returns analysis

**Current Issues**:
1. ❌ **Phase 1 Violations**: Direct yfinance API calls during backtest execution
2. ❌ **No Caching**: Fetches same SPY data repeatedly
3. ❌ **Sync/Async Mix**: Some methods sync, others async (confusing)

### Phase 1 Violations Found

#### 1. `get_price_data()` Method (Lines 101-119)
```python
def get_price_data(self, ticker: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """Get historical price data for backtesting"""
    stock = yf.Ticker(ticker)  # ❌ API call during backtest
    hist = stock.history(start=start_date, end=end_date)
    return hist
```

**Issue**: Fetches price data on-demand during backtest instead of Phase 1
**Used By**: `_get_historical_price()`, multiple backtest methods
**Impact**: Slow backtests, no caching, rate limiting risk

#### 2. `_get_historical_price()` Method (Lines 120-148)
```python
async def _get_historical_price(self, ticker: str, target_date: datetime) -> Optional[float]:
    """Get historical price for a specific date"""
    data = self.get_price_data(ticker, start_date, end_date)  # ❌ Calls API
    # ... find closest trading day
```

**Issue**: Async method calling sync API fetcher
**Used By**: Multiple backtest calculation methods
**Impact**: Mixed async/sync, Phase 1 violation

#### 3. `get_spy_benchmark_data()` Method (Lines 149-163)
```python
def get_spy_benchmark_data(self, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """Get SPY benchmark data for comparison"""
    spy = yf.Ticker("SPY")  # ❌ API call for benchmark
    hist = spy.history(start=start_date, end=end_date)
    return hist
```

**Issue**: Fetches SPY data on-demand instead of Phase 1
**Used By**: `calculate_spy_returns()`, benchmark comparison methods
**Impact**: Repeated SPY fetches for same date ranges

### Good Parts (Keep These)

#### ✅ Performance Metrics Dataclass (Lines 49-88)
```python
@dataclass
class PerformanceMetrics:
    """Container for performance tracking metrics"""
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    # ... comprehensive metrics structure
```
**Status**: Perfect - pure data structure, no API calls

#### ✅ Return Calculation Logic (Lines 165-203)
```python
def calculate_returns(self, price_data: pd.DataFrame, entry_date: datetime, 
                     target_days: List[int] = None) -> Dict[str, float]:
    """Calculate returns for specified time periods"""
    # Pure calculation from pre-fetched price_data DataFrame
```
**Status**: Good - accepts pre-fetched data, pure calculation
**Needs**: Just update callers to pass pre-fetched data

#### ✅ SPY Return Calculation (Lines 205-243)
```python
def calculate_spy_returns(self, spy_data: pd.DataFrame, entry_date: datetime,
                         target_days: List[int] = None) -> Dict[str, float]:
    """Calculate SPY returns for comparison"""
    # Pure calculation from spy_data parameter
```
**Status**: Good - accepts pre-fetched SPY data
**Needs**: Just ensure SPY data comes from Phase 1

### Refactor Strategy for backtest.py

#### Option A: Full Refactor (Recommended)
**Make backtesting Phase 6-only with pre-fetched data**:

```python
class BacktestEngine:
    """Phase 6: Backtest engine using pre-fetched historical data"""
    
    def __init__(self, db=None):
        self.db = db
        self.transaction_cost = 0.001
        self.intervals = [1, 3, 7, 10, 30]
        # No yfinance - all data from Phase 1 cache
    
    async def backtest_signal(
        self,
        ticker: str,
        entry_date: datetime,
        entry_price: float,
        price_history: pd.DataFrame,  # Pre-fetched in Phase 1
        spy_history: pd.DataFrame,    # Pre-fetched in Phase 1
        signal_data: Dict
    ) -> PerformanceMetrics:
        """
        Backtest a signal using pre-fetched data.
        
        Args:
            ticker: Stock ticker symbol
            entry_date: Signal entry date
            entry_price: Entry price (from signal)
            price_history: Pre-fetched price DataFrame (Phase 1)
            spy_history: Pre-fetched SPY DataFrame (Phase 1)
            signal_data: Signal metadata
            
        Returns:
            PerformanceMetrics with calculated performance
        """
        # Calculate returns using pre-fetched data
        returns = self.calculate_returns(price_history, entry_date, self.intervals)
        spy_returns = self.calculate_spy_returns(spy_history, entry_date, self.intervals)
        
        # Build PerformanceMetrics
        metrics = PerformanceMetrics(
            return_1d=returns.get('1d_return'),
            spy_1d_return=spy_returns.get('1d_return'),
            beat_spy_1d=(returns.get('1d_return', 0) > spy_returns.get('1d_return', 0)),
            # ... etc
        )
        
        return metrics
```

**Changes Required**:
1. ✅ Remove `get_price_data()` - replace with parameter
2. ✅ Remove `_get_historical_price()` - use passed DataFrame
3. ✅ Remove `get_spy_benchmark_data()` - replace with parameter
4. ✅ Update all methods to accept pre-fetched DataFrames
5. ✅ Update docstrings to document Phase 6 expectations
6. ✅ Add validation for required DataFrame columns

#### Option B: Minimal Change (Not Recommended)
Keep methods but mark them as "fallback only" and prefer passed data.

**Why Not**: Still allows Phase 1 violations, defeats 3.0 architecture

---

## backend/integrations/performance_tracker.py - Detailed Audit

### Overview
Tracks signal performance over time with database persistence.

**Purpose**:
- Initialize performance tracking for new signals
- Update performance metrics at intervals
- Compare against SPY benchmark
- Persist to `signal_performance` table

**Current Issues**:
1. ❌ **Phase 1 Violations**: yfinance calls for SPY and ticker data
2. ❌ **Duplicate Logic**: Fetches same data as backtest.py
3. ✅ **Good Structure**: Clean async/await pattern

### Phase 1 Violations Found

#### 1. `_get_spy_benchmark_return()` Method (Lines 90-110)
```python
async def _get_spy_benchmark_return(self, start_date: datetime, end_date: datetime) -> float:
    """Get SPY benchmark return for comparison"""
    try:
        spy = yf.Ticker(self.benchmark_ticker)  # ❌ "SPY" hardcoded + API call
        spy_data = spy.history(start=start_date, end=end_date)
        
        if len(spy_data) < 2:
            return 0.0
        
        start_price = spy_data['Close'].iloc[0]
        end_price = spy_data['Close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except Exception as e:
        logger.error(f"Error getting SPY benchmark return: {e}")
        return 0.0
```

**Issue**: Fetches SPY data on-demand instead of Phase 1 cache
**Used By**: `update_signal_performance()` (called frequently)
**Impact**: Repeated SPY fetches, should be cached once

#### 2. `get_current_price()` Method (Lines 195-211)
```python
async def get_current_price(self, ticker: str) -> Optional[float]:
    """Get current price for a ticker"""
    try:
        ticker_obj = yf.Ticker(ticker)  # ❌ API call
        data = ticker_obj.history(period="1d")
        
        if data.empty:
            return None
        
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.error(f"Error getting current price for {ticker}: {e}")
        return None
```

**Issue**: Fetches current price instead of using Phase 1 cache
**Used By**: Performance update logic
**Impact**: Rate limiting risk, slow updates

#### 3. `get_historical_prices()` Method (Lines 225-256)
```python
async def get_historical_prices(self, ticker: str, start_timestamp: datetime, 
                               end_timestamp: datetime) -> Dict[str, float]:
    """Get historical prices for performance calculation"""
    try:
        ticker_obj = yf.Ticker(ticker)  # ❌ API call
        
        # Fetch historical data
        hist = ticker_obj.history(start=start_date, end=end_date)
        
        # ... parse and return
    except Exception as e:
        logger.error(f"Error getting historical prices for {ticker}: {e}")
        return {}
```

**Issue**: Fetches historical data instead of Phase 1 cache
**Used By**: Backtest and performance tracking
**Impact**: Duplicate data fetching

### Good Parts (Keep These)

#### ✅ Database Integration (Lines 112-193)
```python
async def initialize_performance_tracking(self, signal_id: str, ticker: str, 
                                         entry_price: float, entry_timestamp: datetime):
    """Initialize performance tracking for a new signal"""
    # Creates record in signal_performance table
    # Pure database operation, no API calls
```
**Status**: Perfect - Phase 5 database persistence

#### ✅ Metric Calculations
When given pre-fetched data, calculations are pure:
```python
# These work well with pre-fetched data
ticker_return = ((current_price - entry_price) / entry_price) * 100
beat_spy = ticker_return > spy_return
```

### Refactor Strategy for performance_tracker.py

```python
class PerformanceTracker:
    """Phase 6: Track signal performance using pre-fetched data"""
    
    async def update_signal_performance(
        self,
        signal_id: str,
        ticker: str,
        entry_price: float,
        entry_timestamp: datetime,
        current_price: float,        # From Phase 1 cache
        spy_current_price: float,    # From Phase 1 cache
        spy_entry_price: float       # From Phase 1 cache
    ) -> Dict[str, Any]:
        """
        Update signal performance using pre-fetched prices.
        
        Args:
            signal_id: Signal identifier
            ticker: Stock ticker
            entry_price: Signal entry price
            entry_timestamp: Entry datetime
            current_price: Current price (from Phase 1 cache)
            spy_current_price: Current SPY price (from Phase 1 cache)
            spy_entry_price: SPY price at entry (from Phase 1 cache)
        
        Returns:
            Performance metrics dict
        """
        # Calculate returns from pre-fetched prices
        ticker_return = ((current_price - entry_price) / entry_price) * 100
        spy_return = ((spy_current_price - spy_entry_price) / spy_entry_price) * 100
        beat_spy = ticker_return > spy_return
        
        # Update database (Phase 5)
        await self.db.update_performance_metrics(signal_id, {
            'current_return': ticker_return,
            'spy_return': spy_return,
            'beat_spy': beat_spy,
            'last_updated': datetime.now()
        })
        
        return {
            'ticker_return': ticker_return,
            'spy_return': spy_return,
            'beat_spy': beat_spy
        }
```

**Changes Required**:
1. ✅ Remove `_get_spy_benchmark_return()` - accept SPY prices as params
2. ✅ Remove `get_current_price()` - accept price as param
3. ✅ Remove `get_historical_prices()` - accept price history as param
4. ✅ Update all methods to accept pre-fetched data
5. ✅ Update docstrings for Phase 6

---

## Implementation Plan

### Step 1: Delete Obsolete File
```bash
Move-Item "backend\core\backtest.py" "archive\backtest_old_structure.py"
```

### Step 2: Refactor backtest.py (3 methods to update)

**2.1 Remove get_price_data()**
- Delete lines ~101-119
- Add comment: "Data fetching moved to Phase 1 (yfinance.py + cache.py)"

**2.2 Remove _get_historical_price()**
- Delete lines ~120-148
- Update callers to use passed DataFrame instead

**2.3 Remove get_spy_benchmark_data()**
- Delete lines ~149-163
- Add comment: "SPY data must be pre-fetched in Phase 1"

**2.4 Update Main Methods**
Add parameters for pre-fetched data:
- `backtest_signal()` - add `price_history`, `spy_history` params
- `run_smart_historical_backtest()` - add data params
- All calculation methods - ensure they accept DataFrames

### Step 3: Refactor performance_tracker.py (3 methods to update)

**3.1 Remove _get_spy_benchmark_return()**
- Delete lines ~90-110
- Add params: `spy_current_price`, `spy_entry_price`

**3.2 Remove get_current_price()**
- Delete lines ~195-211
- Add param: `current_price`

**3.3 Remove get_historical_prices()**
- Delete lines ~225-256
- Add param: `price_history: pd.DataFrame`

**3.4 Update All Callers**
Ensure all methods accept pre-fetched data instead of fetching

### Step 4: Syntax Validation
```bash
python -m py_compile backend\integrations\backtest.py
python -m py_compile backend\integrations\performance_tracker.py
```

### Step 5: API Violation Check
```bash
grep -r "yf\.|yfinance|Ticker\(" backend/integrations/backtest.py
grep -r "yf\.|yfinance|Ticker\(" backend/integrations/performance_tracker.py
# Should return 0 matches after refactor
```

---

## Expected Outcomes

### Before (Current State)
```
Backtest 10 signals:
├── 10x get_price_data() calls → 10 HTTP requests
├── 10x get_spy_benchmark_data() → 10 HTTP requests  
├── Performance tracking updates:
│   ├── 10x get_current_price() → 10 HTTP requests
│   └── 10x _get_spy_benchmark_return() → 10 HTTP requests
└── Total: 40 HTTP requests per backtest batch
```

### After (3.0 Architecture)
```
Phase 1: Pre-fetch all data
├── 1x fetch SPY history → cache
├── 10x fetch ticker histories → cache
└── Total: 11 HTTP requests (done once)

Phase 6: Backtest + Performance Tracking
├── Read from cache: 0 HTTP requests
├── Pure calculations on DataFrames
└── Database persistence only
```

**Savings**:
- API calls: 40 → 11 per batch (-72.5%)
- Cache hit rate: 0% → 90%+
- Backtest speed: ~30s → ~2s (-93%)

---

## Architecture Compliance

### ✅ After Refactor
| Phase | Responsibility | backtest.py Role | performance_tracker.py Role |
|-------|----------------|------------------|----------------------------|
| **Phase 1** | Fetch & Cache | ❌ None | ❌ None |
| **Phase 2** | Parse & Normalize | ❌ None | ❌ None |
| **Phase 3** | Score by Group | ❌ None | ❌ None |
| **Phase 4** | Assemble Signal | ❌ None | ❌ None |
| **Phase 5** | Persist | ❌ None | ✅ Write metrics to DB |
| **Phase 6** | Post-Ops | ✅ Backtest signals | ✅ Track performance |

### ✅ Data Flow
```
Phase 1 (yfinance.py):
├── Fetch ticker historical data → cache
└── Fetch SPY historical data → cache

Phase 6 (backtest.py):
├── Read ticker data from cache
├── Read SPY data from cache
├── Calculate returns (pure math)
└── Return PerformanceMetrics

Phase 6 (performance_tracker.py):
├── Read current prices from cache
├── Calculate performance (pure math)
└── Persist to signal_performance table (Phase 5)
```

---

## Testing Strategy

### Unit Tests
1. **Test backtest with pre-fetched data**:
```python
# Mock DataFrame with known prices
price_df = pd.DataFrame({
    'Close': [100, 105, 110, 108],
    'Date': pd.date_range('2025-01-01', periods=4)
})

spy_df = pd.DataFrame({
    'Close': [400, 402, 405, 404],
    'Date': pd.date_range('2025-01-01', periods=4)
})

engine = BacktestEngine()
metrics = await engine.backtest_signal(
    ticker='AAPL',
    entry_date=datetime(2025, 1, 1),
    entry_price=100,
    price_history=price_df,
    spy_history=spy_df,
    signal_data={}
)

assert metrics.return_1d == 5.0  # (105-100)/100 * 100
assert metrics.beat_spy_1d is True  # 5.0% > 0.5%
```

2. **Test performance tracker**:
```python
tracker = PerformanceTracker()
result = await tracker.update_signal_performance(
    signal_id='sig_123',
    ticker='AAPL',
    entry_price=100,
    entry_timestamp=datetime(2025, 1, 1),
    current_price=110,      # +10%
    spy_current_price=408,  # +2%
    spy_entry_price=400
)

assert result['ticker_return'] == 10.0
assert result['spy_return'] == 2.0
assert result['beat_spy'] is True
```

### Integration Tests
1. **Full pipeline with cache**:
   - Phase 1: Fetch and cache data
   - Phase 6: Backtest using cached data
   - Verify zero additional HTTP requests

---

## Lessons from signals.py Applied

### ✅ 1. Systematic Audit First
- Read file top-to-bottom
- Grep for API calls (yfinance, openai, etc.)
- Document all violations before fixing

### ✅ 2. Delete, Don't Refactor Bad Patterns
- Don't keep `get_price_data()` as "optional fallback"
- Delete completely and force callers to pass data
- Cleaner API, enforces 3.0 compliance

### ✅ 3. Update Docstrings Thoroughly
- Document expected parameters
- Show Phase context
- Add examples of 3.0 usage

### ✅ 4. Validate After Each Change
- `python -m py_compile` after edits
- grep for remaining violations
- Test with real data

### ✅ 5. Archive, Don't Delete Forever
- Move obsolete files to `archive/`
- Keeps git history clean
- Reference available if needed

---

## Next Agent Instructions

### Step-by-Step Execution:

1. **Move obsolete file**:
```powershell
Move-Item "backend\core\backtest.py" "archive\backtest_old_structure.py"
```

2. **Refactor backtest.py**:
   - Remove `get_price_data()` method
   - Remove `_get_historical_price()` method  
   - Remove `get_spy_benchmark_data()` method
   - Update main backtest methods to accept DataFrames
   - Update docstrings

3. **Refactor performance_tracker.py**:
   - Remove `_get_spy_benchmark_return()` method
   - Remove `get_current_price()` method
   - Remove `get_historical_prices()` method
   - Update all methods to accept prices as parameters
   - Update docstrings

4. **Validate**:
```bash
python -m py_compile backend\integrations\backtest.py
python -m py_compile backend\integrations\performance_tracker.py
grep "yf\.|yfinance|Ticker\(" backend/integrations/backtest.py
grep "yf\.|yfinance|Ticker\(" backend/integrations/performance_tracker.py
```

5. **Update callers in pipeline.py**:
   - Find imports from backtest.py
   - Update method calls to pass pre-fetched data
   - Will be part of pipeline.py refactor

---

## Conclusion

Both files need similar treatment as signals.py:
- ❌ Remove API calls (delete methods completely)
- ✅ Accept pre-fetched data as parameters
- ✅ Pure calculation logic (keep)
- ✅ Database persistence (keep)
- ✅ Phase 6 only (backtest + performance tracking)

After these refactors, only `pipeline.py` remains for the complete 3.0 transition.

**Status**: Audit complete, ready for systematic refactor
