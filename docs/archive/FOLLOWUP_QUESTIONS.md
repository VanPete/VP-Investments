# Implementation Questions & Answers

**Date:** 2025-10-04  
**Status:** Ready to Implement

---

## 📋 Summary of User Decisions

Based on your answers, here's what we'll implement:

### ✅ High Priority (Start Immediately)

**1. Backtest System**
- ✅ Auto-run after pipeline
- ✅ Track: 1d, 3d, 7d, 14d returns
- ❌ NO backfilling (testing phase)
- ⚠️ Skip signal_duration for now (not critical)
- ✅ historical_success_rate = score correlation to performance (vs SPY/positive return)

**2. Technical Indicators**
- ✅ All 9 indicators for swing/long trading
- ✅ YES to TA-Lib integration
- ✅ YES to sector comparison

**3. Fundamental Data**
- ✅ All recommended fundamentals
- ✅ Earnings momentum signals
- ⚠️ Insider data: Yes but API limited (use yfinance basic for now)

### ✅ Medium Priority (Next Phase)

**4. Options Data**
- ✅ yfinance only (free API, upgrade later)
- ✅ Unusual options activity (within yfinance limits)
- ✅ Options flow affects weighted_score

**5. Risk & Volatility**
- ✅ Risk warnings
- ✅ Forward Sharpe ratio
- ✅ Risk affects signal ranking

**6. Short Interest**
- ✅ Retail vs institutional (API limited, best effort)
- ✅ Short squeeze scoring

### ✅ Low Priority (Future)

**7. Reddit Enhancements**
- ✅ Momentum detection
- ⚠️ Maybe sentiment-price divergence

**8. ML & Patterns**
- ✅ ML predictions (after everything else working)
- ❌ NO chart patterns right now

**9. News API**
- ❌ Disabled (don't want to pay for API)

---

## ❓ Follow-Up Questions

I have a few clarification questions before starting implementation:

### Q1: historical_success_rate Calculation

You said: "score correlation to performance taking into account SPY/positive return"

**I need to clarify exactly how you want this calculated:**

Option A: **Success Rate by Score Range**
```python
# For signals with weighted_score 80-90:
# - Count how many beat SPY
# - Store: "Signals in this score range beat SPY X% of the time"
historical_success_rate = (signals_that_beat_spy / total_signals_in_score_range) * 100
```

Option B: **Correlation Coefficient**
```python
# Statistical correlation between score and return
# -1 (inverse) to +1 (perfect correlation)
historical_success_rate = correlation(weighted_score, actual_return_vs_spy)
```

Option C: **Weighted Success Metric**
```python
# Composite score considering both positive return AND beat SPY
success_points = 0
if actual_return > 0:
    success_points += 50
if beat_spy:
    success_points += 50
historical_success_rate = success_points  # 0, 50, or 100
```

**Which calculation makes most sense for your strategy?**

---

### Q2: TA-Lib Indicators - Priorities

You approved TA-Lib integration. Here are the recommended indicators for swing/long trading:

**My Recommendations (pick top 5-10):**

**Momentum (for swing entries):**
- [ ] Stochastic RSI (STOCHRSI) - Overbought/oversold
- [ ] Money Flow Index (MFI) - Volume-weighted RSI
- [ ] Williams %R (WILLR) - Momentum oscillator
- [ ] Ultimate Oscillator (ULTOSC) - Multi-timeframe momentum
- [ ] Commodity Channel Index (CCI) - Trend strength

**Trend (for direction confirmation):**
- [ ] ADX (Average Directional Index) - Trend strength
- [ ] Aroon Oscillator (AROON) - Trend change detection
- [ ] Parabolic SAR (SAR) - Stop and reverse points
- [ ] Ichimoku Cloud components - Comprehensive trend system

**Volatility (for position sizing):**
- [ ] Normalized ATR (NATR) - Volatility-adjusted ATR
- [ ] Bollinger Bands %B - Price position in bands
- [ ] Keltner Channels - ATR-based channels

**Volume (for confirmation):**
- [ ] Accumulation/Distribution (AD) - Money flow
- [ ] Chaikin A/D Oscillator (ADOSC) - A/D momentum
- [ ] Volume-Weighted Average Price (VWAP) - Institutional price

**Which 5-10 are most important for your AI-generated signals?**  
Or should I implement all of them?

---

### Q3: Sector Comparison - Implementation

You said sector comparison is helpful. How deep should this go?

**Option A: Simple Sector ETF Comparison**
```python
sectors = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    # etc...
}

# Compare stock return vs sector ETF return
sector_relative_strength = stock_return - sector_etf_return
```

**Option B: Sector Rankings**
```python
# Rank stock within its sector
# Get all stocks in sector, calculate percentile
sector_strength_percentile = 85  # Top 15% in sector
```

**Option C: Multi-Sector Comparison**
```python
# Compare across all major sectors
# "This tech stock is outperforming 8 of 11 sectors"
sectors_outperformed = 8
```

**Which level of sector analysis is most useful?**

---

### Q4: Financial Score Formula Weights

For Phase H (Financial Score Enhancement), I need to confirm the weighting:

**Proposed Formula:**
```python
financial_score = (
    technical_score * 0.40 +      # 40 points
    fundamentals_score * 0.30 +   # 30 points
    options_score * 0.15 +        # 15 points
    short_interest_score * 0.15   # 15 points
)
```

**Within Technical Score (40 points):**
- Momentum indicators: 10 points
- Trend indicators: 10 points
- Volume indicators: 10 points
- Volatility indicators: 10 points

**Within Fundamentals Score (30 points):**
- Valuation (P/E, P/B, etc.): 10 points
- Profitability (margins, ROE): 10 points
- Growth (EPS, revenue): 10 points

**Does this weighting make sense for swing/long trading?**  
Or would you adjust the percentages?

---

### Q5: Risk-Adjusted Ranking - Implementation

You approved risk affecting signal ranking. How aggressive should the adjustment be?

**Option A: Penalty System (Conservative)**
```python
risk_penalty = 0
if liquidity_warning:
    risk_penalty -= 5  # Reduce score by 5
if drawdown_pct > 30:
    risk_penalty -= 10
if forward_sharpe_ratio > 1.5:
    risk_penalty += 5  # Bonus for good risk/reward

final_score = weighted_score + risk_penalty
```

**Option B: Risk Multiplier (Moderate)**
```python
risk_multiplier = 1.0
if forward_sharpe_ratio > 1.5:
    risk_multiplier = 1.1  # 10% boost
elif forward_sharpe_ratio < 0.5:
    risk_multiplier = 0.9  # 10% penalty

final_score = weighted_score * risk_multiplier
```

**Option C: Risk Tier System (Aggressive)**
```python
# Categorize signals into risk tiers
risk_score = calculate_composite_risk()

if risk_score < 30:  # High risk
    final_score = weighted_score * 0.7  # 30% penalty
elif risk_score > 70:  # Low risk
    final_score = weighted_score * 1.3  # 30% boost
```

**Which approach fits your risk tolerance?**

---

### Q6: Short Squeeze Detection - Thresholds

For short squeeze scoring, what thresholds matter to you?

**Current Proposed Logic:**
```python
short_squeeze_score = 0

# High short interest
if short_pct_float > 20:
    short_squeeze_score += 30
elif short_pct_float > 10:
    short_squeeze_score += 15

# High days to cover
if short_ratio > 5:
    short_squeeze_score += 20

# Positive momentum (shorts trapped)
if momentum_30d_pct > 10:
    short_squeeze_score += 25

# Reddit hype (retail pressure)
if reddit_score > 70:
    short_squeeze_score += 25

# Max 100
```

**Questions:**
- Is 20% short float your threshold for "high"?
- Is 5 days to cover your threshold for squeeze risk?
- Should Reddit sentiment be weighted heavily (25 points)?

---

### Q7: Options Flow Scoring - Sensitivity

How sensitive should options flow be to weighted_score?

**Conservative (±5 points):**
```python
if options_flow_score > 5:  # Very bullish
    weighted_score += 5
elif options_flow_score < -5:  # Very bearish
    weighted_score -= 5
```

**Moderate (±10 points):**
```python
# Scale from -10 to +10
weighted_score += options_flow_score
```

**Aggressive (±15 points with momentum):**
```python
# Options flow + volume confirmation
if options_flow_score > 7 and unusual_activity:
    weighted_score += 15
```

**How much should options flow move the score?**

---

## 📝 Next Steps

Once you answer these 7 questions, I can:

1. **Start Phase A** (Backtest System) with correct historical_success_rate calculation
2. **Start Phase B** (Technical Indicators) with prioritized TA-Lib indicators
3. **Start Phase C** (Fundamentals) with proper sector comparison
4. **Plan Phase H** (Financial Score) with correct weighting
5. **Implement risk adjustments** with your preferred approach
6. **Configure thresholds** for short squeeze and options scoring

**Most critical answers:**
- Q1 (historical_success_rate calculation)
- Q2 (which TA-Lib indicators)
- Q4 (financial score weights)

The others can use my recommendations if you prefer!

---

**Ready to implement once these are clarified!** 🚀
