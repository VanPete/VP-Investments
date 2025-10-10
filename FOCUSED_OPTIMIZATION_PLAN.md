# Focused Optimization Action Plan

**Generated:** 2025-10-09  
**Based on:** User feedback + database analysis  
**Focus:** Improve constant value columns and low variance issues

---

## 🎯 Key Insights

### What's Actually Working ✅
- **MACD indicators**: Recent signals have data (19.77, 6.30 for TSLA/AAPL)
- **Bollinger Bands**: Recent signals have data (461.11, 267.40)
- **Weighted/Financial Scores**: Working (0.41, 0.52)
- **Technical calculations**: All implemented and functional

### What Needs Fixing ❌
1. **Beta = 1.0** (constant) - Should vary by ticker
2. **Upvotes = 0** (constant) - Should capture Reddit engagement
3. **Low variance columns** - Many constant or near-constant values
4. **Constant text fields** - Generic descriptions instead of dynamic

### Database Observation
- **80% NULL columns** = Historical data before features were added
- **Recent signals** = Have the data (calculations working)
- **No need to drop columns** - They'll fill in over time with new signals

---

## 🔧 Priority 1: Fix Constant Value Bugs (3-4 hours)

### 1.1 Beta Calculation (CRITICAL - 2 hours)

**Current State:** Beta = 1.0 for ALL signals (even recent ones)

**Issue:** Hardcoded default, not using real market correlation

**Fix Location:** Check where beta is calculated in enhancement pipeline

**Implementation:**
```python
# Current location: backend/integrations/signal_processing.py
# Or: backend/pipeline.py in enhancement methods

# OPTION 1: Use yfinance beta (fastest)
import yfinance as yf
ticker_obj = yf.Ticker(ticker)
beta = ticker_obj.info.get('beta')
if beta and not pd.isna(beta):
    signal['beta'] = float(beta)
else:
    signal['beta'] = 1.0  # Fallback

# OPTION 2: Calculate from returns (more accurate)
import numpy as np
# Get 1 year of data
stock_hist = ticker_obj.history(period='1y')
spy_hist = yf.Ticker('SPY').history(period='1y')

# Align dates and calculate returns
stock_returns = stock_hist['Close'].pct_change().dropna()
spy_returns = spy_hist['Close'].pct_change().dropna()

# Calculate beta
covariance = np.cov(stock_returns, spy_returns)[0][1]
spy_variance = np.var(spy_returns)
beta = covariance / spy_variance if spy_variance > 0 else 1.0

signal['beta'] = round(beta, 4)
```

**Testing:**
```bash
# Expected results:
# AAPL: ~1.24
# TSLA: ~2.30
# KO (Coca-Cola): ~0.60
```

**Action Items:**
- [ ] Locate where beta is set in signal enhancement
- [ ] Implement yfinance .info['beta'] extraction
- [ ] Add validation (check if None or NaN)
- [ ] Test with known tickers (AAPL, TSLA, KO)
- [ ] Generate new signals and verify beta varies

---

### 1.2 Upvotes Collection (CRITICAL - 1.5 hours)

**Current State:** Upvotes = 0 for ALL signals

**Issue:** Reddit scraper not extracting upvote count from posts

**Fix Location:** `backend/integrations/reddit.py`

**Implementation:**
```python
# In Reddit scraping function (find where submissions are processed)

# CURRENT (WRONG):
mention_data = {
    'title': submission.title,
    'body': submission.selftext,
    'url': submission.url,
    # upvotes NOT extracted
}

# FIXED (RIGHT):
mention_data = {
    'title': submission.title,
    'body': submission.selftext,
    'url': submission.url,
    'score': submission.score,  # This is the upvote count!
    'upvote_ratio': submission.upvote_ratio,  # % upvoted
    'num_comments': submission.num_comments
}

# Then when aggregating mentions for a ticker:
total_upvotes = sum(mention.get('score', 0) for mention in ticker_mentions)
signal['upvotes'] = total_upvotes
```

**Testing:**
```python
# Test Reddit API directly
import praw
reddit = praw.Reddit(...)
submission = reddit.submission(url='https://reddit.com/r/wallstreetbets/...')
print(f"Score (upvotes): {submission.score}")
print(f"Upvote ratio: {submission.upvote_ratio}")
```

**Action Items:**
- [ ] Find Reddit scraping function in reddit.py
- [ ] Add 'score' field to mention extraction
- [ ] Update aggregation to sum upvotes
- [ ] Test with live Reddit scrape
- [ ] Verify upvotes column gets populated

---

### 1.3 Exchange Field (MEDIUM - 30 min)

**Current State:** Exchange = "NYSE" for all tickers

**Issue:** Static value in company_tickers table

**Fix:** Update ticker loading to extract from yfinance

```python
import yfinance as yf

ticker_obj = yf.Ticker(symbol)
info = ticker_obj.info

ticker_data = {
    'ticker': symbol,
    'company_name': info.get('longName', symbol),
    'sector': info.get('sector', 'Unknown'),
    'exchange': info.get('exchange', 'UNKNOWN'),  # FIX: Dynamic
    'market_cap': info.get('marketCap'),
}

# Expected values:
# AAPL -> NASDAQ
# IBM -> NYSE  
# GME -> NYSE
```

**Action Items:**
- [ ] Find ticker data loading script
- [ ] Update to extract exchange dynamically
- [ ] Run batch update for existing tickers (optional)
- [ ] Verify new tickers get correct exchange

---

## 📊 Priority 2: Improve Low Variance Columns (2-3 hours)

### 2.1 Dynamic Text Generation

**Constant Text Fields:**
- `top_factors` = "Reddit mentions, price momentum" (100%)
- `signal_type` = "Multi-Factor" (100%)
- `trade_type` = Only 4 unique values (0.4%)

**Goal:** Generate dynamic, signal-specific descriptions

**Implementation:**
```python
def generate_top_factors(signal):
    """Generate dynamic top factors based on actual signal data"""
    factors = []
    
    # Check which components are strongest
    if signal.get('reddit_score', 0) > 0.7:
        factors.append("Strong Reddit sentiment")
    elif signal.get('reddit_score', 0) > 0.4:
        factors.append("Reddit mentions")
    
    if signal.get('rsi', 50) < 30:
        factors.append("Oversold RSI")
    elif signal.get('rsi', 50) > 70:
        factors.append("Overbought RSI")
    
    if signal.get('volume_spike_ratio', 1) > 2.0:
        factors.append("High volume")
    
    if signal.get('relative_strength', 0) > 80:
        factors.append("Strong momentum")
    
    if signal.get('short_pct_float', 0) > 20:
        factors.append("High short interest")
    
    # Return top 3 factors
    return ", ".join(factors[:3]) if factors else "Multi-factor analysis"

def determine_signal_type(signal):
    """Determine signal type based on data sources"""
    has_reddit = signal.get('reddit_score', 0) > 0
    has_financial = signal.get('financial_score', 0) > 0
    has_news = signal.get('news_score', 0) > 0
    
    sources = sum([has_reddit, has_financial, has_news])
    
    if sources >= 2:
        return "Multi-Factor"
    elif has_reddit:
        return "Social Sentiment"
    elif has_financial:
        return "Technical/Fundamental"
    else:
        return "Single-Factor"

def determine_trade_type(signal):
    """Determine trade type based on signal characteristics"""
    weighted_score = signal.get('weighted_score', 0)
    rsi = signal.get('rsi', 50)
    momentum = signal.get('momentum_30d_pct', 0)
    
    if weighted_score > 0.7:
        if momentum > 20:
            return "Strong Momentum Buy"
        else:
            return "Quality Value Buy"
    elif weighted_score > 0.5:
        if rsi < 40:
            return "Contrarian Buy"
        else:
            return "Swing Trade"
    else:
        return "Speculative"
```

**Action Items:**
- [ ] Implement dynamic factor generation
- [ ] Implement signal type classification
- [ ] Implement trade type classification
- [ ] Test with various signal profiles
- [ ] Verify variance improves

---

### 2.2 Review Low Variance Acceptability

**Some low variance may be EXPECTED:**

| Column | Variance | Status | Action |
|--------|----------|--------|--------|
| `sector` | 1.0% unique | Expected | Reddit focuses on tech |
| `risk_level` | 0.3% unique | Review | Should vary more? |
| `post_recency` | 0.4% unique | Bug | Should calculate properly |
| `mentions` | 0.8% unique | Expected | Popular tickers get many mentions |
| `liquidity_score` | 1.1% unique | Review | Only large caps? |
| `market_cap_category` | 0.5% unique | Expected | Reddit focuses on mega-cap |

**Action Items:**
- [ ] Review each low variance column
- [ ] Determine if expected (Reddit bias) or bug
- [ ] Fix bugs (e.g., post_recency calculation)
- [ ] Document expected low variance columns

---

### 2.3 Insider Trading (FUTURE - Not Urgent)

**Constant Values:**
- `insider_activity_score` = 50.0 (default)
- `insider_buy_count` = 0
- `insider_sell_count` = 0
- `insider_net_shares` = 0

**Status:** Not implemented yet, yfinance may not have this data

**Options:**
1. Keep as-is (low priority)
2. Try yfinance insider transactions
3. Use premium API (e.g., Financial Modeling Prep)

**Recommendation:** Defer to future phase (not critical for core system)

---

## 🚀 Priority 3: Next Phase Planning (1 hour)

### Phase 7: Frontend Integration

**Now that data quality is good, focus on user interface:**

**Week 1: API Development**
- Build REST API endpoints
- On-demand signal generation (already working!)
- Signal listing with filters
- Signal detail views
- Statistics dashboard

**Week 2: React Dashboard**
- Signal list view (table with sorting/filtering)
- Signal detail page (full metrics + charts)
- On-demand generator form
- Performance statistics
- Real-time updates

**Benefits:**
- Users can view and interact with signals
- On-demand generation for any ticker
- Visual performance tracking
- Production-ready application

---

## 📈 Expected Improvements

### Before Fixes
- Beta: 1.0 constant (wrong)
- Upvotes: 0 constant (missing data)
- Exchange: "NYSE" constant (wrong)
- top_factors: Generic text (not descriptive)
- signal_type: "Multi-Factor" always (not accurate)
- trade_type: 4 values only (too generic)

### After Fixes
- Beta: Varies correctly (AAPL=1.24, TSLA=2.30, KO=0.60)
- Upvotes: Real engagement counts (varies by popularity)
- Exchange: Correct per ticker (NASDAQ, NYSE, OTC)
- top_factors: Dynamic, signal-specific (meaningful)
- signal_type: Accurate classification (Social, Technical, Multi-Factor)
- trade_type: Specific recommendations (8-10 categories)

### Data Quality Improvement
- ✅ Beta accuracy: Market correlation
- ✅ Social engagement: Real Reddit metrics
- ✅ Ticker metadata: Accurate exchange/sector
- ✅ Signal descriptions: Meaningful and specific
- ✅ Better classification: Helps users understand signals

---

## 🎯 Implementation Sequence

### This Week: Critical Fixes (Day 1-3)
1. **Day 1 Morning:** Fix beta calculation (2 hrs)
   - Locate beta setting code
   - Implement yfinance extraction
   - Test with AAPL, TSLA, KO
   
2. **Day 1 Afternoon:** Fix upvotes collection (1.5 hrs)
   - Update Reddit scraping
   - Test upvote extraction
   - Verify aggregation

3. **Day 2 Morning:** Fix exchange field (30 min)
   - Update ticker loading
   - Test with various exchanges
   
4. **Day 2 Afternoon:** Dynamic text generation (2 hrs)
   - Implement top_factors
   - Implement signal_type
   - Implement trade_type

5. **Day 3:** Testing & Validation (2 hrs)
   - Generate test signals
   - Verify all fixes working
   - Check variance improvements

### Next Week: Phase 7 Planning (Day 1-5)
6. **Design API endpoints** - REST API structure
7. **Plan React dashboard** - UI mockups and components
8. **Database queries** - Optimize for frontend needs
9. **Real-time strategy** - WebSocket or polling
10. **Begin implementation** - Start with API layer

---

## ✅ Success Criteria

### Data Quality
- [ ] Beta values vary by ticker (not constant 1.0)
- [ ] Upvotes reflect real Reddit engagement
- [ ] Exchange correct per ticker (NASDAQ/NYSE/OTC)
- [ ] top_factors dynamic and meaningful
- [ ] signal_type accurately classified
- [ ] trade_type specific and varied (>10 categories)

### System Health
- [ ] No dropped columns (keeping all data)
- [ ] Historical data preserved (80% NULL acceptable)
- [ ] Recent signals have complete data
- [ ] Code quality maintained
- [ ] Tests passing

### Ready for Phase 7
- [ ] Data quality sufficient for frontend
- [ ] API patterns established (single signal working)
- [ ] Documentation updated
- [ ] Team ready to build UI

---

## 📝 Key Takeaways

1. **Don't drop columns** - Historical data will fill in over time
2. **80% NULL is OK** - Older signals before features were added
3. **Focus on constants** - Beta, upvotes, exchange need fixes
4. **Improve descriptions** - Dynamic text > generic text
5. **Phase 7 next** - Frontend integration is the logical next step

**Ready to start with Priority 1 fixes! 🚀**
