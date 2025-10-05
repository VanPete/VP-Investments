# 📋 Implementation Plan - Based on User Decisions

**Date:** 2025-10-04  
**Status:** Ready for Implementation  
**Strategy:** Swing/Long Trading with AI-generated signals

---

## ✅ USER DECISIONS

### 🔴 HIGH PRIORITY

**Q1 - BACKTEST SYSTEM:**
- Q1.1: ✅ **YES** - Auto-run after each pipeline
- Q1.2: ✅ **1d, 3d, 7d, 14d** (recommended intervals for swing trading)
- Q1.3: ❌ **NO** - No backfill (testing phase, tables cleared frequently)
- Q1.4: ⚠️ **SKIP signal_duration** for now (not critical)
- Q1.5: ✅ **Score correlation to performance** (weighted_score → actual returns vs SPY)

**Q2 - TECHNICAL INDICATORS:**
- Q2.1: ✅ **ALL 9 indicators** (optimized for swing/long trading)
- Q2.2: ✅ **YES to TA-Lib** - Add comprehensive technical analysis
- Q2.3: ✅ **YES to sector comparison** - Relative strength analysis

**Q3 - FUNDAMENTAL DATA:**
- Q3.1: ✅ **All fundamentals** (analyst targets, earnings dates, institutional data)
- Q3.2: ✅ **Earnings momentum signals** - Trigger special signals pre-earnings
- Q3.3: ⚠️ **Insider data** - Yes, but API limited (use yfinance for now)

### 🟡 MEDIUM PRIORITY

**Q4 - OPTIONS DATA:**
- Q4.1: ✅ **yfinance only** (free API, upgrade later)
- Q4.2: ✅ **Unusual options activity** (but within yfinance limits)
- Q4.3: ✅ **Options flow → weighted_score modifier**

**Q5 - RISK & VOLATILITY:**
- Q5.1: ✅ **Risk warnings** (liquidity, volatility, drawdown)
- Q5.2: ✅ **Forward Sharpe ratio**
- Q5.3: ✅ **Risk affects ranking** (lower risk = higher rank)

**Q6 - SHORT INTEREST:**
- Q6.1: ✅ **Retail vs institutional** (API limited, best effort)
- Q6.2: ✅ **Short squeeze scoring** (high priority for strategy)

### 🟢 LOW PRIORITY

**Q7 - REDDIT:** ✅ Momentum detection, ⚠️ Maybe sentiment-price divergence
**Q8 - ML:** ✅ Future implementation, ❌ No chart patterns for now
**Q9 - NEWS:** ❌ Disabled (API costs)

---

## 🎯 IMPLEMENTATION PHASES

### **PHASE A: Backtest System** (Immediate - 4 hours)
**Goal:** Auto-calculate returns after each pipeline run

**Files to Modify:**
1. `backend/integrations/backtest.py`
   - Fix column references (use actual columns, not missing ones)
   - Add 1d, 3d, 7d, 14d interval tracking
   - Calculate beat_spy_* flags
   - Calculate historical_success_rate (score → return correlation)

2. `backend/pipeline.py`
   - Call BacktestScheduler after signal generation
   - Pass signal_id and ticker for tracking
   - Store returns in signal_performance table (NOT signals table)

3. `backend/storage/database.py`
   - Add methods for performance tracking
   - Query historical success rates

**New Columns to Populate:**
- Signal performance table: 1d/3d/7d/14d returns, beat_spy flags
- Signals table: historical_success_rate (calculated from past signals)

---

### **PHASE B: Technical Indicators** (Immediate - 3 hours)
**Goal:** Add all 9 missing indicators + TA-Lib enhancements

**Files to Modify:**
1. `backend/integrations/yfinance.py`
   - Add `above_200d_ma_pct` calculation
   - Add `avg_daily_volume` and `avg_volume_30d`
   - Add `volatility_rank` (percentile over 252 days)
   - Add `volume_price_correlation` (30-day correlation)
   - Add `relative_strength` (vs SPY)
   - Add `sector_relative_strength` (vs sector ETF)
   - Add `exit_signal_strength` (inverse momentum)
   - Add `signal_strength_percentile` (historical rank)

2. **NEW FILE:** `backend/integrations/technical_indicators.py`
   - TA-Lib integration module
   - Additional indicators for swing trading:
     - MACD histogram divergence
     - Stochastic RSI
     - Average True Range (ATR) bands
     - Parabolic SAR
     - Ichimoku Cloud components
     - Volume-weighted indicators

**Recommended TA-Lib Indicators:**
```python
import talib

# Momentum (swing trading focused)
- STOCHRSI (Stochastic RSI)
- MFI (Money Flow Index)
- WILLR (Williams %R)
- ULTOSC (Ultimate Oscillator)

# Trend
- ADX (Average Directional Index) - already have
- AROON (Aroon Oscillator)
- SAR (Parabolic SAR)

# Volatility
- NATR (Normalized ATR)
- TRANGE (True Range)

# Volume
- AD (Accumulation/Distribution)
- OBV (On Balance Volume) - already have
- ADOSC (Chaikin A/D Oscillator)
```

---

### **PHASE C: Fundamental Data** (Next - 3 hours)
**Goal:** Complete fundamental analysis for financial_score

**Files to Modify:**
1. `backend/integrations/yfinance.py`
   - Add `analyst_targets` (yfinance provides analyst recommendations)
   - Add `earnings_date` (next earnings calendar)
   - Add `dividend_ex_date` (for dividend stocks)
   - Add `earnings_gap_pct` (calculate from historical data)
   - Add `institutional_ownership` (yfinance institutional holders)
   - Add `insider_buy_volume` (yfinance insider transactions)

2. `backend/pipeline.py`
   - Add earnings momentum signal detection
   - Flag signals within 2 weeks of earnings
   - Boost score for positive analyst sentiment

**Earnings Momentum Logic:**
```python
if days_to_earnings <= 14:
    # Pre-earnings momentum signal
    if analyst_targets['avg'] > current_price * 1.1:
        earnings_momentum_boost = 10  # Add to financial_score
    
    signal_type = "earnings_momentum"
```

---

### **PHASE D: Risk Calculations** (Next - 2 hours)
**Goal:** Better risk assessment and warnings

**Files to Modify:**
1. `backend/pipeline.py` - Enhance `_calculate_risk_score()`
   - Add `drawdown_pct` (from 52-week high)
   - Add `forward_volatility` (GARCH model or historical vol)
   - Add `forward_sharpe_ratio` (expected return / forward vol)
   - Add `liquidity_warning` (if avg_volume < 100K)
   - Add `float_turnover_ratio` (volume / float)

2. **Risk-Adjusted Ranking:**
```python
# Modify weighted_score calculation
risk_penalty = (
    -5 if liquidity_warning else 0 +
    -10 if drawdown_pct > 30 else 0 +
    +5 if forward_sharpe_ratio > 1.5 else 0
)

final_score = weighted_score + risk_penalty
```

---

### **PHASE E: Options Data** (Next - 2 hours)
**Goal:** Basic options flow from yfinance

**Files to Modify:**
1. `backend/integrations/yfinance.py`
   - Get options chain from yfinance
   - Calculate `implied_volatility` (ATM options)
   - Calculate `iv_spike_pct` (current IV vs 30-day avg)
   - Calculate `option_volume_ratio` (options vol / stock vol)
   - Detect `unusual_options_activity` (volume > 2x avg)
   - Calculate `options_flow_score` (call/put sentiment)

**Options Flow Logic:**
```python
def calculate_options_flow_score(ticker):
    chain = yf.Ticker(ticker).option_chain()
    
    call_volume = chain.calls['volume'].sum()
    put_volume = chain.puts['volume'].sum()
    
    # Bullish if more call volume
    flow_score = (call_volume - put_volume) / (call_volume + put_volume)
    
    return flow_score * 10  # -10 (bearish) to +10 (bullish)

# Add to weighted_score
if options_flow_score > 5:
    weighted_score += 5  # Options bullish modifier
```

---

### **PHASE F: Short Squeeze Scoring** (Next - 1 hour)
**Goal:** Identify short squeeze candidates

**Files to Modify:**
1. `backend/pipeline.py`
   - Calculate `short_squeeze_score`
   - Use: short_pct_float, short_ratio, momentum, reddit_score

**Short Squeeze Logic:**
```python
def calculate_short_squeeze_potential(data):
    score = 0
    
    # High short interest
    if data['short_pct_float'] > 20:
        score += 30
    elif data['short_pct_float'] > 10:
        score += 15
    
    # High days to cover
    if data['short_ratio'] > 5:
        score += 20
    
    # Positive momentum (shorts trapped)
    if data['momentum_30d_pct'] > 10:
        score += 25
    
    # Reddit hype (retail buying pressure)
    if data['reddit_score'] > 70:
        score += 25
    
    return min(score, 100)
```

---

### **PHASE G: Reddit Enhancements** (Later - 1 hour)
**Goal:** Advanced Reddit analytics

**Files to Modify:**
1. `backend/integrations/reddit.py`
   - Add `reddit_momentum_score` (mentions change rate)
   - Add `reddit_vs_price_divergence` (sentiment vs price direction)
   - Add `social_sentiment_trend` (improving/declining)

---

### **PHASE H: Financial Score Enhancement** (Critical - 2 hours)
**Goal:** Make ALL technical indicators contribute to financial_score

**Current Problem:** Only basic indicators used in financial_score

**Solution:** Update `_calculate_financial_score()` in pipeline.py

**New Formula:**
```python
def _calculate_financial_score(self, data):
    score = 0
    
    # Technical Indicators (40 points)
    tech_score = (
        self._score_momentum(data) +      # 10 pts
        self._score_trend(data) +         # 10 pts
        self._score_volume(data) +        # 10 pts
        self._score_volatility(data)      # 10 pts
    )
    
    # Fundamentals (30 points)
    fund_score = (
        self._score_valuation(data) +     # 10 pts
        self._score_profitability(data) + # 10 pts
        self._score_growth(data)          # 10 pts
    )
    
    # Options Sentiment (15 points)
    options_score = self._score_options(data)
    
    # Short Interest (15 points)
    short_score = self._score_short_interest(data)
    
    return tech_score + fund_score + options_score + short_score
```

---

## 📦 FILES TO DELETE (Cleanup)

**Temporary Analysis Files:**
- ✅ DELETE: `phase1_column_analysis.py`
- ✅ DELETE: `check_actual_columns.py`
- ✅ DELETE: `check_empty_columns.py`
- ✅ DELETE: `analyze_empty_columns.py`
- ✅ DELETE: `run_phase1_1_migration.py`
- ✅ DELETE: `run_phase1_2_migration.py`
- ✅ DELETE: `run_phase2_migration.py`
- ✅ DELETE: `fix_numeric_columns.py`
- ✅ DELETE: `list_db_objects.py`
- ✅ DELETE: `analyze_signals_columns.py`

**Obsolete Migration Files:**
- ✅ DELETE: `migrations/phase1_1_document_columns.sql` (already executed)
- ✅ DELETE: `migrations/phase1_2_remove_unused_columns.sql` (NOT using - keeping columns)
- ✅ DELETE: `migrations/phase1_2_remove_unused_columns_verified.sql` (NOT using)
- ✅ KEEP: `migrations/phase2_add_performance_tracking.sql` (will use for signal_performance)

**Documentation Files:**
- ✅ DELETE: `PHASE_0_COMPLETE.md` (info in recommendations.md)
- ✅ DELETE: `PHASE_1_ANALYSIS_RESULTS.md` (info in recommendations.md)
- ✅ DELETE: `PHASE_1_COMPLETE.md` (info in recommendations.md)
- ✅ KEEP: `IMPLEMENTATION_QUESTIONS.md` (reference for decisions made)
- ✅ KEEP: `docs/recommendations.md` (single source of truth)

---

## 🚀 EXECUTION ORDER

**Week 1 - Core Performance:**
1. ✅ Phase A: Backtest System (4 hrs)
2. ✅ Phase B: Technical Indicators (3 hrs)
3. ✅ Phase C: Fundamental Data (3 hrs)

**Week 2 - Enhancements:**
4. ✅ Phase H: Financial Score Enhancement (2 hrs) - **CRITICAL**
5. ✅ Phase D: Risk Calculations (2 hrs)
6. ✅ Phase E: Options Data (2 hrs)
7. ✅ Phase F: Short Squeeze Scoring (1 hr)

**Week 3 - Polish:**
8. ✅ Phase G: Reddit Enhancements (1 hr)
9. ✅ Testing & Validation

---

## 🎯 SUCCESS METRICS

After implementation, you should have:
- ✅ Auto-backtesting with 1d/3d/7d/14d returns
- ✅ 29 technical indicators feeding into financial_score
- ✅ Complete fundamental analysis
- ✅ Risk-adjusted signal ranking
- ✅ Options flow sentiment
- ✅ Short squeeze detection
- ✅ 64 previously empty columns now populated
- ✅ Clean codebase with obsolete files removed

**Ready to start implementation!**
