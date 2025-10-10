# Database Optimization Action Plan

**Generated:** 2025-10-09  
**Based on:** database_analysis_20251009.md (tables.py analysis)  
**Status:** Ready for Implementation

---

## 📊 Analysis Summary

### Database Health
- **Total Tables:** 5 (down from 11+ originally)
- **Total Signals:** 1,091 rows
- **Columns in signals:** 140 columns
- **100% NULL columns:** 13 (need to drop)
- **>80% NULL columns:** 41 (data collection issues)
- **Constant value columns:** 15+ (bugs or expected defaults)

### Key Findings

**Critical Issues (Fix Immediately):**
1. ✅ **13 columns with 100% NULL** - Never populated, safe to drop
2. ❌ **Beta always 1.0** - Hardcoded default, needs real calculation
3. ❌ **Upvotes always 0** - Reddit scraper not capturing field
4. ❌ **Exchange always "NYSE"** - Should vary by ticker
5. ❌ **MACD/Bollinger 80% NULL** - yfinance data availability issue

**Medium Priority (Improve Data Quality):**
6. ⚠️ **Options data 89% NULL** - Limited yfinance coverage
7. ⚠️ **Institutional data 82% NULL** - Not available for all tickers
8. ⚠️ **Insider trading all 0** - Not implemented/not available
9. ⚠️ **Low variance in many fields** - Expected (Reddit bias) or data issue

---

## 🎯 Priority 1: Drop 100% NULL Columns (1 hour)

### Columns to Drop (13 Total)

**Never Implemented:**
```sql
ALTER TABLE signals DROP COLUMN IF EXISTS iv_spike_pct;
ALTER TABLE signals DROP COLUMN IF EXISTS options_flow_score;
ALTER TABLE signals DROP COLUMN IF EXISTS social_sentiment_trend;
ALTER TABLE signals DROP COLUMN IF EXISTS reddit_momentum_score;
ALTER TABLE signals DROP COLUMN IF EXISTS unusual_options_activity;
ALTER TABLE signals DROP COLUMN IF EXISTS implied_volatility;
```

**Backtest Intervals Not Yet Calculated:**
```sql
ALTER TABLE signals DROP COLUMN IF EXISTS 7d_return;
ALTER TABLE signals DROP COLUMN IF EXISTS 10d_return;
ALTER TABLE signals DROP COLUMN IF EXISTS 30d_return;
ALTER TABLE signals DROP COLUMN IF EXISTS spy_7d_return;
ALTER TABLE signals DROP COLUMN IF EXISTS spy_10d_return;
ALTER TABLE signals DROP COLUMN IF EXISTS beat_spy_7d;
ALTER TABLE signals DROP COLUMN IF EXISTS beat_spy_10d;
```

**Note:** We may want to keep 7d/10d/30d return columns if we plan to implement extended backtest intervals in Phase 8.

**Action:**
- [ ] Review if 7d/10d/30d returns will be implemented
- [ ] Run SQL in Supabase SQL editor
- [ ] Verify columns dropped successfully
- [ ] Update pipeline.py to remove references (if any)

**Before:** 140 columns  
**After:** 127 columns (13 dropped)

---

## 🔧 Priority 2: Fix Constant Value Bugs (3-4 hours)

### 2.1 Beta Calculation (Critical)

**Issue:** `beta = 1.0` for 100% of signals (80.5% NULL, rest are 1.0)

**Root Cause:** Hardcoded default value, not calculating from market data

**Fix Location:** `backend/integrations/signal_processing.py` or `backend/pipeline.py`

**Implementation:**
```python
# CURRENT (WRONG):
signal['beta'] = 1.0  # Hardcoded default

# FIXED (RIGHT):
# Option 1: Use yfinance info
ticker_info = yf.Ticker(ticker).info
signal['beta'] = ticker_info.get('beta', 1.0)  # Default 1.0 if unavailable

# Option 2: Calculate from price history (more reliable)
import numpy as np
stock_returns = calculate_returns(stock_prices)
spy_returns = calculate_returns(spy_prices)
covariance = np.cov(stock_returns, spy_returns)[0][1]
spy_variance = np.var(spy_returns)
beta = covariance / spy_variance if spy_variance != 0 else 1.0
signal['beta'] = beta
```

**Testing:**
```bash
# Test with known stocks
python -c "import yfinance as yf; print('AAPL beta:', yf.Ticker('AAPL').info.get('beta')); print('TSLA beta:', yf.Ticker('TSLA').info.get('beta'))"
# Expected: AAPL ~1.24, TSLA ~2.3
```

**Action:**
- [ ] Find where beta is set in codebase
- [ ] Implement yfinance beta retrieval
- [ ] Add fallback to 1.0 if unavailable
- [ ] Test with AAPL, TSLA, validate values
- [ ] Run pipeline, verify beta values vary by ticker

---

### 2.2 Upvotes Collection (Critical)

**Issue:** `upvotes = 0` for 100% of signals

**Root Cause:** Reddit scraper not extracting upvote count from posts

**Fix Location:** `backend/integrations/reddit.py` - Reddit scraping function

**Implementation:**
```python
# CURRENT (WRONG):
post_data = {
    'title': submission.title,
    'score': submission.score,  # This is upvotes!
    # But upvotes field not populated
}

# FIXED (RIGHT):
post_data = {
    'title': submission.title,
    'score': submission.score,
    'upvotes': submission.score,  # Add this field
    'upvote_ratio': submission.upvote_ratio,  # Also useful
}

# Then in signal creation:
signal['upvotes'] = sum(mention['upvotes'] for mention in mentions)
```

**Testing:**
```python
# Test Reddit API
import praw
reddit = praw.Reddit(...)
submission = reddit.submission(id='test_id')
print(f"Score (upvotes): {submission.score}")
print(f"Upvote ratio: {submission.upvote_ratio}")
```

**Action:**
- [ ] Locate Reddit scraping function
- [ ] Add upvotes field to extracted data
- [ ] Update aggregation to sum upvotes
- [ ] Test with live Reddit data
- [ ] Verify upvotes column populated

---

### 2.3 Exchange Field (Medium Priority)

**Issue:** `company_tickers.exchange = "NYSE"` for 100% of tickers

**Root Cause:** Static value or incorrect yfinance field extraction

**Fix Location:** Ticker data loading script

**Implementation:**
```python
# CURRENT (WRONG):
ticker_data = {
    'ticker': symbol,
    'exchange': 'NYSE'  # Hardcoded
}

# FIXED (RIGHT):
import yfinance as yf
ticker_info = yf.Ticker(symbol).info
ticker_data = {
    'ticker': symbol,
    'exchange': ticker_info.get('exchange', 'UNKNOWN')  # NASDAQ, NYSE, OTC, etc.
}
```

**Testing:**
```python
# Test with known tickers
import yfinance as yf
print('AAPL:', yf.Ticker('AAPL').info.get('exchange'))  # Expected: NASDAQ
print('IBM:', yf.Ticker('IBM').info.get('exchange'))    # Expected: NYSE
print('GME:', yf.Ticker('GME').info.get('exchange'))    # Expected: NYSE
```

**Action:**
- [ ] Find ticker data loading/updating code
- [ ] Implement dynamic exchange extraction
- [ ] Update existing records (batch update)
- [ ] Verify exchange values vary correctly

---

### 2.4 Other Constant Values (Review)

**These may be expected/acceptable:**

| Column | Value | Status | Action |
|--------|-------|--------|--------|
| `news_score` | 0.0 | Expected | News API disabled |
| `news_mentions` | 0 | Expected | News API disabled |
| `top_factors` | "Reddit mentions, price momentum" | Bug | Generate dynamic explanation |
| `signal_type` | "Multi-Factor" | Bug | Should vary (Reddit-only, Financial-only, etc.) |
| `emerging` | False | Low priority | Market cap based, can implement later |
| `spy_3d_return` | 0.22 | Expected | SPY benchmark for specific period |
| `backtest_phase` | "Complete" | Expected | All signals backtested |
| `backtest_eligible` | True | Expected | All signals are eligible |
| `expected_hold_duration` | 5 | Acceptable | Default swing trading timeframe |
| `insider_activity_score` | 50.0 | Bug | Default, not calculating real score |
| `insider_buy/sell_count` | 0 | Not implemented | Insider data not available/expensive |
| `institutional_change_qoq` | -95.2 | Bug | Broken calculation |

**Action:**
- [ ] Review each constant value
- [ ] Determine if expected or bug
- [ ] Prioritize fixes based on impact
- [ ] Document accepted constants

---

## 📈 Priority 3: Improve High NULL Columns (4-6 hours)

### 3.1 MACD Indicators (80.5% NULL)

**Columns:** macd_histogram, macd_signal, macd_line

**Issue:** yfinance may not provide MACD for all tickers or timeframes

**Diagnosis:**
```python
import yfinance as yf
import pandas as pd

ticker = yf.Ticker('AAPL')
hist = ticker.history(period='6mo')

# Check if we have enough data
print(f"Rows: {len(hist)}")
print(f"Columns: {hist.columns.tolist()}")

# MACD requires calculating from price history
# yfinance doesn't provide MACD directly!
```

**Fix:** Calculate MACD manually using TA-Lib or pandas

```python
import pandas as pd

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD manually"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    return {
        'macd_line': macd_line.iloc[-1],
        'macd_signal': macd_signal.iloc[-1],
        'macd_histogram': macd_histogram.iloc[-1]
    }

# Usage:
hist = yf.Ticker(ticker).history(period='6mo')
macd = calculate_macd(hist['Close'])
```

**Action:**
- [ ] Verify yfinance doesn't provide MACD directly
- [ ] Implement MACD calculation function
- [ ] Add to technical indicator pipeline
- [ ] Test with AAPL, TSLA
- [ ] Monitor NULL rate improvement (target: <20%)

---

### 3.2 Bollinger Bands (80.5% NULL)

**Columns:** bollinger_upper, bollinger_lower, bollinger_width, bollinger_position

**Issue:** Same as MACD - must calculate from price history

**Fix:** Calculate manually

```python
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = ((upper - lower) / sma) * 100
    position = ((prices - lower) / (upper - lower)) * 100
    
    return {
        'bollinger_upper': upper.iloc[-1],
        'bollinger_lower': lower.iloc[-1],
        'bollinger_width': width.iloc[-1],
        'bollinger_position': position.iloc[-1]
    }
```

**Action:**
- [ ] Implement Bollinger Bands calculation
- [ ] Integrate into technical pipeline
- [ ] Test and validate
- [ ] Monitor NULL rate improvement

---

### 3.3 Options Data (89% NULL)

**Columns:** put_call_oi_ratio, put_call_vol_ratio

**Issue:** Options data only available for liquid stocks

**Analysis:**
- yfinance provides options data but not for all tickers
- Small-cap and illiquid stocks don't have options
- 89% NULL may be expected

**Fix Options:**
1. **Accept current state** - 11% coverage is reasonable
2. **Improve error handling** - Log which tickers have options
3. **Premium API** - Upgrade to better options data source (future)

**Action:**
- [ ] Analyze which tickers have options (market cap correlation?)
- [ ] Improve error handling and logging
- [ ] Document expected NULL rate
- [ ] Defer premium API to future phase

---

### 3.4 Institutional Data (82% NULL)

**Columns:** retail_holding_pct, institutional_ownership_pct

**Issue:** Not all tickers report institutional holdings

**Analysis:**
- Institutional holdings data requires regulatory filings
- Smaller companies may not have this data
- 82% NULL may be partially expected

**Action:**
- [ ] Verify yfinance provides this data
- [ ] Check if extraction code is correct
- [ ] Accept high NULL rate for small-cap stocks
- [ ] Document coverage expectations

---

## 🧹 Priority 4: Schema Cleanup (2-3 hours)

### 4.1 Drop Redundant Tables

**Candidates for Deletion:**
```sql
-- These tables are empty or redundant
DROP TABLE IF EXISTS signal_scoring_factors;  -- Empty
DROP TABLE IF EXISTS signal_performance;      -- Empty (data in signals)
DROP TABLE IF EXISTS backtest_interval_tracking;  -- Can calculate from signals.created_at
```

**Action:**
- [ ] Verify tables are truly unused
- [ ] Backup before dropping (export schema)
- [ ] Run DROP statements
- [ ] Update any code references

---

### 4.2 Review signal_metrics Table

**Issue:** May be redundant with signals table

**Analysis:**
- signals: 1,091 rows, 140 columns
- signal_metrics: 1,017 rows, ~80 columns
- Overlap: Many columns exist in both tables

**Options:**
1. **Merge into signals** - Consolidate all data in one table
2. **Keep separate** - If genuinely different purposes
3. **Document relationship** - Clarify what goes where

**Action:**
- [ ] Compare column lists between tables
- [ ] Identify overlapping vs unique columns
- [ ] Decide on merge vs keep
- [ ] Execute chosen approach

---

### 4.3 Optimize Indexes

**Add indexes for common queries:**
```sql
-- Frequently queried columns
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_weighted_score ON signals(weighted_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_run_id ON signals(run_id);

-- Composite indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_signals_score_date ON signals(weighted_score DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(ticker, created_at DESC);
```

**Action:**
- [ ] Review existing indexes
- [ ] Add missing indexes
- [ ] Test query performance
- [ ] Monitor index usage

---

## 📊 Expected Improvements

### Before Optimization
- Tables: 5 active (+ old redundant tables)
- Columns in signals: 140
- 100% NULL columns: 13
- High NULL (>80%): 41
- Constant value bugs: 3 critical (beta, upvotes, exchange)

### After Optimization
- Tables: 5 (cleaned, optimized)
- Columns in signals: 127 (dropped 13 NULL columns)
- 100% NULL columns: 0 ✅
- High NULL (>80%): ~35 (improved MACD/Bollinger)
- Constant value bugs: 0 ✅ (beta, upvotes, exchange fixed)

### Data Quality Goals
- ✅ Beta: Accurate market correlation values
- ✅ Upvotes: Real Reddit engagement metrics
- ✅ Exchange: Correct NASDAQ/NYSE/OTC designation
- ✅ MACD/Bollinger: <50% NULL (from 80%)
- ✅ Schema: Clean, documented, optimized

---

## 🚀 Implementation Sequence

### Week 1: Critical Fixes
1. **Day 1:** Drop 100% NULL columns (1 hour)
2. **Day 1-2:** Fix beta calculation (2 hours)
3. **Day 2-3:** Fix upvotes collection (2 hours)
4. **Day 3:** Fix exchange field (1 hour)
5. **Day 4-5:** Implement MACD calculation (3 hours)
6. **Day 5:** Implement Bollinger Bands (2 hours)

### Week 2: Schema Optimization
7. **Day 1:** Review and drop redundant tables (2 hours)
8. **Day 2:** Optimize indexes (2 hours)
9. **Day 3:** Test and validate improvements (3 hours)
10. **Day 4-5:** Documentation and Phase 7 planning (4 hours)

---

## ✅ Success Criteria

### Data Quality
- [ ] Beta values vary by ticker (not all 1.0)
- [ ] Upvotes column populated with real values
- [ ] Exchange field shows NASDAQ, NYSE, OTC correctly
- [ ] MACD indicators <50% NULL (from 80%)
- [ ] Bollinger Bands <50% NULL (from 80%)

### Schema Cleanliness
- [ ] Zero 100% NULL columns
- [ ] Redundant tables dropped
- [ ] Indexes optimized for common queries
- [ ] Documentation updated

### Code Quality
- [ ] All references to dropped columns removed
- [ ] Technical indicator calculations tested
- [ ] Error handling improved
- [ ] Logging enhanced for data collection issues

---

## 📝 Next Steps

1. **Review this plan** - Validate priorities and approach
2. **Start with Priority 1** - Drop 100% NULL columns (quick win)
3. **Fix beta calculation** - Highest impact data quality issue
4. **Fix upvotes** - Second highest impact
5. **Proceed sequentially** - Complete each priority before moving to next

**Ready to begin!** 🚀
