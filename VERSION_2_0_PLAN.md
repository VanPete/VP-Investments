# VP Investments - Version 2.0 Implementation Plan

**Date:** October 5, 2025  
**Status:** Ready for Implementation

---

## 🎯 Goals

1. **Fix Critical Bugs**
   - AI Strategy Generation (NoneType errors)
   - Remove 60d_return/90d_return references (doesn't exist)
   - Remove unused signals.id column

2. **Populate Missing Data**
   - 95+ columns with 100% NULLs in signals table
   - Focus on data that affects signal scoring

3. **Verify Signal Scoring**
   - Ensure all populated columns are used in scoring
   - Update weights if needed

4. **GitHub Deployment**
   - Create v1.0 branch (preserve old code)
   - Make current code the new main
   - Tag as v2.0

---

## 🔍 Analysis Results

### Signals Table NULL Analysis (43 rows)

**100% NULL Columns (95 columns):**

**Technical Indicators (11 columns):**
- ✅ `volume_spike_ratio` - **NEEDED** for signal scoring
- ✅ `relative_strength` - **NEEDED** for signal scoring  
- ✅ `momentum_30d_pct` - **NEEDED** for AI strategies & scoring
- ✅ `rsi` - **NEEDED** for AI strategies & scoring
- ✅ `macd_histogram` - **NEEDED** for signal scoring
- ✅ `macd_signal` - **NEEDED** for signal scoring
- ✅ `macd_line` - **NEEDED** for signal scoring
- ✅ `bollinger_width` - **NEEDED** for signal scoring
- ✅ `bollinger_upper` - **NEEDED** for signal scoring
- ✅ `bollinger_lower` - **NEEDED** for signal scoring
- ✅ `bollinger_position` - **NEEDED** for signal scoring
- ✅ `volatility` - **NEEDED** for AI strategies & scoring
- ✅ `volatility_rank` - **NEEDED** for signal scoring
- ✅ `above_50d_ma_pct` - **NEEDED** for signal scoring
- ✅ `above_200d_ma_pct` - **NEEDED** for signal scoring
- `beta` - Nice to have

**Fundamental Data (6 columns):**
- ✅ `pe_ratio` - **NEEDED** for signal scoring
- ✅ `earnings_gap_pct` - **NEEDED** for signal scoring
- ✅ `eps_growth` - **NEEDED** for signal scoring
- ✅ `roe` - **NEEDED** for signal scoring
- ✅ `debt_equity` - **NEEDED** for signal scoring
- ✅ `fcf_margin` - **NEEDED** for signal scoring

**Options Data (4 columns):**
- ✅ `put_call_oi_ratio` - **NEEDED** for options strategies
- ✅ `put_call_vol_ratio` - **NEEDED** for options strategies
- ✅ `iv_spike_pct` - **NEEDED** for options strategies
- ✅ `implied_volatility` - **NEEDED** for options strategies
- `options_flow_score` - Nice to have
- `option_volume_ratio` - Nice to have
- `option_chain_data` - Nice to have (JSONB)
- `unusual_options_activity` - Nice to have (JSONB)

**Ownership Data (5 columns):**
- ✅ `retail_holding_pct` - **NEEDED** for signal scoring
- ✅ `insider_buy_volume` - **NEEDED** for signal scoring
- ✅ `short_pct_float` - **NEEDED** for signal scoring
- ✅ `short_pct_outstanding` - **NEEDED** for signal scoring
- ✅ `shares_short` - **NEEDED** for signal scoring
- `institutional_ownership` - Nice to have
- `short_interest_ratio` - Nice to have
- `short_ratio` - Nice to have

**Volume Data (2 columns):**
- ✅ `avg_volume_30d` - **NEEDED** (duplicate of signal_metrics)
- ✅ `avg_daily_volume` - **NEEDED** (duplicate of signal_metrics)

**Backtest Data (ALL NULL - Expected):**
- All return columns (1d, 3d, 7d, 10d, 30d, net, spy, beat_spy)
- `max_return_pct`, `drawdown_pct`, `signal_duration`
- `forward_volatility`, `forward_sharpe_ratio`
- `realized_returns`, `backtest_phase`, `backtest_timestamp`, `backtest_notes`

**Scoring & Risk Metrics (12 columns):**
- `entry_quality_score` - Nice to have
- `exit_signal_strength` - Nice to have
- `risk_adjusted_score` - Nice to have
- `sector_relative_strength` - Nice to have
- `liquidity_score` - Nice to have
- `market_cap_category` - Nice to have
- `float_turnover_ratio` - Nice to have
- `institutional_flow_direction` - Nice to have
- `risk_category` - Nice to have
- `risk_score` - Nice to have
- `max_position_size` - Nice to have
- `signal_strength_percentile` - Nice to have
- `momentum_consistency_score` - Nice to have
- `volume_price_correlation` - Nice to have
- `reddit_vs_price_divergence` - Nice to have
- `ml_confidence_score` - Nice to have
- `pattern_match_score` - Nice to have
- `historical_success_rate` - Nice to have

**Text/Commentary Fields:**
- `reddit_summary` - Currently NULL (AI not generating)
- `ai_news_summary` - Currently NULL (AI not generating)
- `ai_trends_commentary` - Currently NULL (AI not generating)
- `thread_tag` - Not used
- `liquidity_warning` - Conditional
- `social_sentiment_trend` - Nice to have
- `news_sentiment_score` - Nice to have
- `reddit_momentum_score` - Nice to have
- `commentary` - Nice to have

**Date Fields:**
- `earnings_date` - Nice to have
- `dividend_ex_date` - Nice to have
- `analyst_targets` - Nice to have (JSONB)
- `expected_hold_duration` - Nice to have

---

## 🔧 Implementation Tasks

### Phase 1: Quick Wins (30 minutes)

#### Task 1.1: Fix AI Strategy NoneType Errors ✅
**File:** `backend/integrations/ai.py`

**Problem Lines:**
- Line 431: `momentum = abs(analysis.get('momentum_30d', 0))`
- Line 495: `momentum = analysis.get('momentum_30d', 0)`
- Options eligibility checks comparing None values

**Fix:** Add proper None handling:
```python
def safe_abs(value, default=0):
    """Safely get absolute value, handling None"""
    return abs(value) if value is not None else default

def safe_get(d, key, default=0):
    """Safely get value from dict, handling None"""
    value = d.get(key, default)
    return value if value is not None else default
```

#### Task 1.2: Remove 60d_return and 90d_return References ✅
**File:** `backend/integrations/backtest.py` (line 788)

**Current:**
```python
'"1d_return", "3d_return", "7d_return", "10d_return", '
'"30d_return", "60d_return", "90d_return"'
```

**Fix:**
```python
'"1d_return", "3d_return", "7d_return", "10d_return", "30d_return"'
```

#### Task 1.3: Remove Unused signals.id Column ✅
**SQL:**
```sql
ALTER TABLE signals DROP COLUMN IF EXISTS id;
```

---

### Phase 2: Populate Critical Missing Data (2-3 hours)

#### Task 2.1: Update Yahoo Finance Integration
**File:** `backend/integrations/yfinance.py`

**Add to get_stock_data():**
- ✅ volume_spike_ratio (current volume / avg volume)
- ✅ relative_strength (price vs sector average)
- ✅ momentum_30d_pct (30-day momentum)
- ✅ RSI (14-period)
- ✅ MACD (histogram, signal, line)
- ✅ Bollinger Bands (width, upper, lower, position)
- ✅ Volatility & volatility_rank
- ✅ above_50d_ma_pct & above_200d_ma_pct
- ✅ PE ratio, EPS growth, ROE, debt/equity, FCF margin
- ✅ Put/call ratios, IV spike
- ✅ Insider/short data
- ✅ Beta

Most of this data is available from yfinance's `.info` and `.history()` methods.

#### Task 2.2: Update Pipeline to Calculate Derived Metrics
**File:** `backend/pipeline.py`

Calculate from existing data:
- volume_spike_ratio = current volume / avg_daily_volume
- momentum_30d_pct = (current_price / price_30d_ago - 1) * 100
- above_50d_ma_pct = (current_price / ma_50d - 1) * 100
- above_200d_ma_pct = (current_price / ma_200d - 1) * 100

#### Task 2.3: Verify signal_metrics Synchronization
**Files:** `backend/pipeline.py`

Ensure these columns are saved to BOTH signals and signal_metrics:
- Technical indicators
- Fundamental ratios
- Options data
- Ownership data

---

### Phase 3: Verify Signal Scoring (1 hour)

#### Task 3.1: Audit Signal Scoring Function
**File:** `backend/core/signals.py` (or wherever scoring happens)

**Check:**
- Which columns are used in `financial_score` calculation?
- Which columns are used in `reddit_score` calculation?
- Are all populated columns being utilized?
- Are there weights for technical, fundamental, options, ownership data?

#### Task 3.2: Update Scoring Weights (if needed)
Based on audit, update `backend/core/config.py` scoring weights to include:
- Technical indicators weight
- Fundamental analysis weight
- Options flow weight
- Ownership/sentiment weight

---

### Phase 4: GitHub Deployment (30 minutes)

#### Task 4.1: Create v1.0 Branch
```bash
git checkout main
git branch v1.0-archive
git push origin v1.0-archive
```

#### Task 4.2: Commit All Changes to Main
```bash
git add .
git commit -m "Version 2.0: 3-table structure, fixed AI strategies, comprehensive data population"
git push origin main
```

#### Task 4.3: Tag as v2.0
```bash
git tag -a v2.0 -m "VP Investments Version 2.0

Major Changes:
- Migrated to 3-table normalized structure (signals, signal_metrics, signal_performance)
- Fixed AI strategy generation NoneType errors
- Removed 60d_return/90d_return references
- Populated 95+ previously NULL columns
- Enhanced signal scoring with technical, fundamental, options, and ownership data
- Improved data quality to 100%
"
git push origin v2.0
```

---

## 📊 Success Criteria

### Before (Current State):
- ✅ 43 signals with 95+ NULL columns (100% NULL rate)
- ❌ 0 AI strategies generated (NoneType errors)
- ❌ Backtest errors (60d_return reference)
- ⚠️ Limited signal scoring (missing data)

### After (Target State):
- ✅ 43 signals with <10 NULL columns
- ✅ 10+ AI strategies generated successfully
- ✅ No backtest errors
- ✅ Comprehensive signal scoring using all available data
- ✅ Code deployed to GitHub as v2.0

---

## 🚀 Execution Order

1. **Quick Fixes** (30 min)
   - Fix AI strategy NoneType errors
   - Remove 60d_return references
   - Drop unused id column

2. **Test Pipeline** (10 min)
   - Run `python -m backend.pipeline`
   - Verify AI strategies generate
   - Verify no backtest errors

3. **Populate Missing Data** (2-3 hours)
   - Update yfinance integration
   - Update pipeline calculations
   - Run pipeline again
   - Verify columns populated

4. **Verify Scoring** (1 hour)
   - Audit scoring function
   - Update weights if needed
   - Run pipeline again
   - Compare signal rankings

5. **GitHub Deployment** (30 min)
   - Create v1.0 branch
   - Commit to main
   - Tag as v2.0
   - Push all

**Total Time: 4-5 hours**

---

## ❓ Questions for User

1. **Signal Scoring:** Which data sources should have highest weight?
   - Current: 50% reddit, 50% financial
   - Proposed: Add technical (20%), fundamental (20%), options (10%)?

2. **AI Strategies:** Should we generate for top 10 or top 20 signals?

3. **Data Sources:** Any paid data sources available (e.g., premium Yahoo Finance)?

4. **Backtest Intervals:** Current: 1d, 3d, 7d, 10d, 30d. Remove 10d? Add 14d?

5. **GitHub Branch:** Confirm naming - main (v2.0), v1.0-archive (old code)?

