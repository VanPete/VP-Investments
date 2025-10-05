# 📋 Data Collection Strategy - Questions for Implementation

**Date:** 2025-10-04  
**Purpose:** Determine what data to collect and how to implement empty columns  
**Total Empty Columns:** 64 (categorized into 11 groups)

---

## 🔴 HIGH PRIORITY QUESTIONS (Answer These First)

### 1. BACKTEST & PERFORMANCE TRACKING (20 columns)

**Returns Tracking:**
- ❓ **Q1.1:** Should the backtest system run automatically after each pipeline run?
  - Options: (a) Yes, automatic after signals generated, (b) Manual/scheduled separately, (c) Only on-demand
  
- ❓ **Q1.2:** What return intervals do you want to track?
  - Current empty: 1d, 3d, 7d, 10d
  - Options: Keep these? Add 14d, 30d, 60d?
  
- ❓ **Q1.3:** Should we backfill returns for existing 74 signals?
  - This would calculate historical performance for signals already in database
  
- ❓ **Q1.4:** Should `signal_duration` track actual hold time or expected hold time?
  - Actual = how long signal was/should be held
  - Expected = predicted based on signal type
  
- ❓ **Q1.5:** How do you want `historical_success_rate` calculated?
  - By signal_type? By ticker? Overall?
  - What defines "success"? (beat SPY? positive return? hit target?)

### 2. TECHNICAL INDICATORS (9 columns)

**Missing Indicators:**
- ❓ **Q2.1:** Which of these technical indicators are MOST important for your strategy?
  - [ ] `above_200d_ma_pct` - Price position vs 200-day MA (trend confirmation)
  - [ ] `avg_daily_volume` + `avg_volume_30d` - Volume averages
  - [ ] `volatility_rank` - Percentile ranking of volatility
  - [ ] `volume_price_correlation` - How volume relates to price movement
  - [ ] `relative_strength` - RS vs market
  - [ ] `sector_relative_strength` - RS vs sector
  - [ ] `exit_signal_strength` - When to exit positions
  - [ ] `signal_strength_percentile` - Historical percentile ranking
  
- ❓ **Q2.2:** Should we add MORE technical indicators using TA-Lib?
  - Examples: Ichimoku Cloud, Parabolic SAR, Aroon, etc.
  
- ❓ **Q2.3:** Is sector comparison important for your strategy?
  - Do you want to know if stock outperforms its sector?

### 3. FUNDAMENTAL DATA (7 columns)

**Missing Fundamentals:**
- ❓ **Q3.1:** Which fundamentals matter most?
  - [ ] `analyst_targets` - Analyst price targets (JSON: {high, low, avg, count})
  - [ ] `dividend_ex_date` - Next dividend ex-date
  - [ ] `earnings_date` - Next earnings announcement
  - [ ] `earnings_gap_pct` - Price gap after last earnings
  - [ ] `institutional_ownership` - % held by institutions
  - [ ] `institutional_flow_direction` - Are institutions buying or selling?
  - [ ] `insider_buy_volume` - Volume of insider buying
  
- ❓ **Q3.2:** Should earnings dates trigger special signals or alerts?
  - E.g., "earnings momentum" signal before announcements
  
- ❓ **Q3.3:** Is insider trading data valuable for your strategy?
  - Insider buying can be bullish signal

---

## 🟡 MEDIUM PRIORITY QUESTIONS (Answer if you want enhancements)

### 4. OPTIONS DATA (6 columns)

- ❓ **Q4.1:** Do you have access to an options data API?
  - Current: We can get basic IV from yfinance
  - Premium: Could use CBOE, TastyTrade, or other options APIs
  
- ❓ **Q4.2:** Is unusual options activity important for your strategy?
  - Large trades, high volume, directional bets
  
- ❓ **Q4.3:** Should options flow affect the weighted_score?
  - E.g., heavy call buying = bullish modifier

### 5. RISK & VOLATILITY (6 columns)

- ❓ **Q5.1:** Should we calculate and display risk warnings?
  - E.g., "Low liquidity", "High volatility", "Recent drawdown"
  
- ❓ **Q5.2:** Do you want forward-looking Sharpe ratio estimates?
  - Uses historical volatility to project risk-adjusted returns
  
- ❓ **Q5.3:** Should risk metrics affect signal ranking?
  - Lower risk = higher rank?

### 6. SHORT INTEREST (2 columns)

- ❓ **Q6.1:** Is retail vs institutional ownership important?
  - Retail-heavy stocks = different behavior
  
- ❓ **Q6.2:** Should we score "short squeeze potential"?
  - High short interest + positive momentum = squeeze candidate

---

## 🟢 LOW PRIORITY QUESTIONS (Nice to have)

### 7. REDDIT ENHANCEMENTS (5 columns)

- ❓ **Q7.1:** Do you want momentum detection for Reddit discussion?
  - E.g., "mentions increased 300% in last 24h"
  
- ❓ **Q7.2:** Should we detect sentiment-price divergence?
  - E.g., "bearish sentiment but price rising" = potential reversal

### 8. ML & PATTERN DETECTION (3 columns)

- ❓ **Q8.1:** Do you want ML-based predictions?
  - Would require training models on historical data
  
- ❓ **Q8.2:** Should we detect chart patterns?
  - Head & shoulders, cup & handle, triangles, etc.

### 9. NEWS ENHANCEMENTS (3 columns)

- ❓ **Q9.1:** Do you have a news API with higher limits?
  - Current: News API is disabled due to rate limits
  - Could use: NewsAPI.org (premium), Alpha Vantage, Benzinga

---

## 📊 CURRENT DATA SOURCES

**Working:**
- ✅ Reddit (PRAW) - mentions, sentiment, upvotes
- ✅ yfinance - basic price/volume/fundamentals
- ✅ FMP API - financial data
- ✅ OpenAI - AI commentary

**Partially Working:**
- ⚠️ yfinance technical indicators (some missing)
- ⚠️ Short interest (have some data, missing retail %)

**Not Implemented:**
- ❌ News API (disabled due to limits)
- ❌ Options data (beyond basic yfinance IV)
- ❌ Insider trading data
- ❌ Institutional flow data
- ❌ ML predictions

---

## 🎯 MY RECOMMENDATIONS (Based on Analysis)

### Implement Immediately:
1. **BACKTEST_SYSTEM** (Q1.1-Q1.5)
   - You have BacktestScheduler code, just needs activation
   - Critical for measuring strategy performance
   - **Recommendation:** Auto-run daily, track 1d/3d/7d/14d returns

2. **TECHNICAL_INDICATORS** (Q2.1-Q2.3)
   - Most are easy to calculate from existing price data
   - **Recommendation:** Add all 9 indicators, they're lightweight

3. **FUNDAMENTAL_DATA** (Q3.1-Q3.3)
   - Enhance financial_score with more fundamentals
   - **Recommendation:** Add earnings_date and analyst_targets at minimum

### Implement Next:
4. **RISK_CALCULATIONS** (Q5.1-Q5.3)
   - Better risk assessment = better position sizing
   - **Recommendation:** Add liquidity warnings and drawdown tracking

5. **OPTIONS_DATA** (Q4.1-Q4.3)
   - Only if you have options API access
   - **Recommendation:** Start with yfinance IV, add premium later

### Consider Later:
6. **REDDIT_ENHANCEMENTS** - Nice to have but not critical
7. **ML_FEATURES** - Requires training data and model development
8. **NEWS_DATA** - Only if you get premium news API

---

## ✅ DECISION TEMPLATE

Please answer the HIGH PRIORITY questions (Q1-Q3) with your preferences:

```
Q1.1: [a/b/c]
Q1.2: [Keep 1d/3d/7d/10d or Add more?]
Q1.3: [Yes/No]
Q1.4: [Actual/Expected/Both]
Q1.5: [By signal_type / Overall / Custom logic]

Q2.1: [Check all that apply]
Q2.2: [Yes/No - which indicators?]
Q2.3: [Yes/No]

Q3.1: [Check all that apply]
Q3.2: [Yes/No]
Q3.3: [Yes/No]
```

Once you answer these, I'll create implementation plans for each category!
