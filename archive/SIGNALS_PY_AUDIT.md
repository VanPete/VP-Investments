# signals.py 3.0 Audit Results
**File:** `backend/core/signals.py` (4,568 lines)  
**Date:** October 14, 2025

---

## 🔍 Analysis Summary

### ✅ **GOOD - 3.0 Compatible Logic** 

1. **Normalization Functions** (Lines 35-117)
   - `normalize_direct()`, `normalize_inverted()`, `normalize_growth()`
   - ✅ Pure functions, no external calls
   - ✅ Phase 3 compatible

2. **Calculator Classes** (Lines 119-878)
   - `ZScoreCalculator`, `TrendStrengthCalculator`, `ValuationCalculator`
   - `TradeTypeClassifier`, `RiskScoreCalculator`
   - ✅ Pure calculations, no API calls
   - ✅ Phase 3 compatible

3. **Data Structures** (Lines 1649-1872)
   - `Signal`, `SignalResult`, `SignalBatchResult`, etc.
   - ✅ Dataclasses for Phase 4 (Assembly)
   - ✅ No external dependencies

4. **Scoring Functions** (Lines 3448-4040)
   - `_calculate_fundamental_score_standalone()`
   - `_calculate_social_alternative_score_standalone()`
   - `_calculate_news_macro_score_standalone()`
   - `_calculate_risk_stability_score_standalone()`
   - `_calculate_institutional_smart_money_score_standalone()`
   - ✅ Pure functions using dict inputs
   - ✅ Phase 3 compatible

---

## ❌ **PROBLEMS - Needs Refactoring**

### 1. **OpenAI API Calls in RiskScoreCalculator** (Lines 1508-1620)

**Location:** `RiskScoreCalculator.generate_risk_narrative()`

**Issue:** Calls OpenAI API mid-processing (Phase 6 violation)

**Current Code:**
```python
async def generate_risk_narrative(self, signal_data: Dict[str, Any]) -> str:
    """Phase 7: AI-enhanced risk narrative generation using OpenAI."""
    try:
        # Try to import OpenAI directly
        from openai import AsyncOpenAI
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Call OpenAI API
        response = await client.chat.completions.create(...)
```

**3.0 Fix:**
- ❌ Remove this method from signals.py
- ✅ Use `integrations/ai.py` which already has Phase 6 OpenAI logic
- ✅ Call from pipeline Phase 6, not during scoring

---

### 2. **YFinance Data Fetching** (Lines 2030-2050)

**Location:** `SignalScorer._get_enhanced_data()`

**Issue:** Fetches data mid-pipeline (Phase 1 violation)

**Current Code:**
```python
def _get_enhanced_data(self, ticker: str) -> Dict[str, Any]:
    """Get enhanced risk/trade data with caching."""
    from backend.integrations.yfinance import fetch_enhanced_risk_data
    
    # Fetch if not cached
    data = fetch_enhanced_risk_data(ticker)  # ❌ API CALL
```

**Problems:**
1. ❌ `fetch_enhanced_risk_data()` doesn't exist in new yfinance.py
2. ❌ Fetching data during Phase 3/4 (should be Phase 1 only)
3. ❌ Breaks cache-first architecture

**3.0 Fix:**
- ❌ Remove `_get_enhanced_data()` method
- ✅ All data must be fetched in Phase 1
- ✅ Scoring functions should accept pre-fetched bundles only
- ✅ Update `score_ticker()` to use data from bundles, not fetch

---

### 3. **SignalEnhancer Class** (Lines 4040-4342)

**Location:** `SignalEnhancer` class

**Issue:** Unclear if it fetches data or just enhances

**Need to Check:**
- Does it call external APIs?
- Does it fetch from database?
- Is it Phase 5 (Persist) or Phase 4 (Assembly)?

**Audit Needed:** Read lines 4040-4342 to verify

---

## 📋 Required Changes

### **Change 1: Remove OpenAI Logic**

**Action:** Delete or comment out `generate_risk_narrative()` in `RiskScoreCalculator`

**Reason:** Logic moved to `integrations/ai.py` (Phase 6)

**Lines to Remove:** 1508-1620

```python
# DELETE THIS METHOD - AI commentary moved to integrations/ai.py
# async def generate_risk_narrative(self, signal_data: Dict[str, Any]) -> str:
#     ...
```

---

### **Change 2: Fix YFinance Data Fetching**

**Current Pattern (WRONG):**
```python
class SignalScorer:
    def _get_enhanced_data(self, ticker: str):
        data = fetch_enhanced_risk_data(ticker)  # ❌ API call
        return data
    
    async def score_ticker(self, ticker_data: Dict):
        enhanced = self._get_enhanced_data(ticker)  # ❌ Fetching mid-scoring
        # ... use enhanced data ...
```

**3.0 Pattern (CORRECT):**
```python
class SignalScorer:
    # Remove _get_enhanced_data() entirely
    
    async def score_ticker(self, ticker_data: Dict):
        # All data already in ticker_data from Phase 1
        # No fetching needed - just extract from dict
        ticker = ticker_data.get('ticker')
        market_data = ticker_data.get('market_data')  # From Phase 1
        historical_data = ticker_data.get('historical_data')  # From Phase 1
        
        # Calculate using pre-fetched data only
        tech_score = self._calculate_technical_score(historical_data)
        fund_score = self._calculate_fundamental_score(market_data)
        # ...
```

**Action:**
1. Delete `_get_enhanced_data()` method (lines ~2030-2050)
2. Update `score_ticker()` to expect all data in `ticker_data` dict
3. No API calls, only dict extraction

---

### **Change 3: Update SignalScorer.score_ticker()**

**Current Signature:**
```python
async def score_ticker(self, ticker_data: Dict) -> SignalResult:
```

**Expected Input Structure (3.0):**
```python
ticker_data = {
    'ticker': 'AAPL',
    
    # Phase 1 bundles (all pre-fetched)
    'market_data': MarketDataBundle(...),  # From yfinance
    'historical_data': HistoricalDataBundle(...),  # From yfinance
    'news_bundle': NewsBundle(...),  # From news.py
    'reddit_bundle': RedditBundle(...),  # From reddit.py
    
    # Phase 2 (parsed/normalized - if needed)
    # ...
    
    # Phase 3 (pre-calculated scores)
    'technical_score': 0.75,  # From calculator.py
    'fundamental_score': 0.65,  # From calculator.py
    'social_score': 0.80,  # From reddit.py Phase 3
    'news_score': 0.70,  # From news.py Phase 3
}
```

**Action:** Update method to extract from dict instead of fetching

---

## 🎯 Refactoring Strategy

### **Option A: Minimal Changes (Recommended)**

**Keep:**
- ✅ All calculator classes (ZScoreCalculator, etc.)
- ✅ All normalization functions
- ✅ All dataclasses (Signal, SignalResult, etc.)
- ✅ All standalone scoring functions

**Remove:**
- ❌ `generate_risk_narrative()` from RiskScoreCalculator
- ❌ `_get_enhanced_data()` from SignalScorer
- ❌ Any other methods that fetch external data

**Update:**
- 🔧 `SignalScorer.score_ticker()` to use bundles only
- 🔧 Input validation to expect Phase 1 bundles

---

### **Option B: Major Restructure (If Time Permits)**

**Move to calculator.py:**
- ZScoreCalculator → utils/calculator.py
- TrendStrengthCalculator → utils/calculator.py
- ValuationCalculator → utils/calculator.py

**Keep in signals.py:**
- SignalScorer (Phase 4 assembly)
- Signal dataclasses
- Trade type classification
- Risk assessment

**Benefits:** Better separation of concerns

**Cost:** More refactoring work

---

## 📊 Verdict

### **Current Status:** ⚠️ **MIXED - 80% compatible, 20% needs fixes**

### **Critical Issues:**
1. 🔴 Line 2038: `fetch_enhanced_risk_data()` doesn't exist
2. 🔴 Lines 1508-1620: OpenAI calls (use ai.py instead)
3. 🟡 SignalEnhancer needs audit

### **Recommendation:**
- ✅ Fix critical issues (remove fetching, remove OpenAI)
- ✅ Update score_ticker() to use bundles
- ⏳ Defer Option B restructure until after pipeline.py

### **Estimated Fix Time:**
- Delete fetching code: 5 minutes
- Update score_ticker(): 15 minutes  
- Testing: 10 minutes
- **Total: ~30 minutes**

---

## 🔧 Next Steps

1. **Delete OpenAI method** (line 1508-1620)
2. **Delete _get_enhanced_data()** (line 2030-2050)
3. **Update score_ticker()** to expect bundles
4. **Test with sample ticker_data dict**
5. **Move to next file**

Ready to proceed? 🚀

