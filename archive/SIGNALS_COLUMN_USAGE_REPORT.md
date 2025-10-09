# Signals Table Column Usage Analysis & Recommendations

**Date:** October 7, 2025  
**Analysis:** Comprehensive review of all 137 columns in signals table

---

## Executive Summary

**Critical Finding:** Only **4 columns** (3%) directly impact signal scoring, while **88 columns** (64%) are populated but unused in the scoring calculation. This represents a massive opportunity to improve signal quality by incorporating the rich data we're already collecting.

---

## Current State

### 📊 Column Breakdown

| Category | Count | % of Total | Status |
|----------|-------|------------|--------|
| **Total Columns** | 137 | 100% | - |
| **Used in Scoring** | 4 | 3% | ✅ Active |
| **Used in AI Strategy** | 15 | 11% | ✅ Active |
| **Pipeline Populated** | 88 | 64% | ⚠️ Collected but unused |
| **Backtest Populated** | 29 | 21% | ✅ Used for validation |
| **Orphaned/Unused** | 19 | 14% | ❌ Dead columns |

---

## Scoring Logic Analysis

### Current weighted_score Calculation

```python
weighted_score = (
    reddit_score * 0.10 +      # 10% weight
    financial_score * 0.40 +   # 40% weight  
    news_score * 0.00          # 0% weight (DISABLED)
)
```

**Only 3 inputs affect signal ranking:**
1. `reddit_score` (10% weight)
2. `financial_score` (40% weight)
3. `news_score` (0% weight - disabled)

**What this means:**
- 50% of the score comes from just 2 sources
- 50% is unused potential (news disabled)
- **All Phase 1 technical indicators are ignored**
- **All Phase 1 composite metrics are ignored**
- **All fundamental data is ignored**

---

## AI Strategy Generation (Better Utilization)

AI strategy generation uses **15 columns** for decision-making:

### Core Identification
- `id`, `ticker`

### Signal Metrics
- `weighted_score`, `signal_confidence`

### Market Characteristics
- `market_cap`, `market_cap_category`, `current_price`

### Technical Indicators
- `rsi`, `momentum_30d_pct`, `relative_strength`, `volatility`

### Risk Metrics
- `risk_score`, `risk_level`, `max_position_size`, `liquidity_score`

**Key Insight:** AI strategies leverage technical indicators that scoring completely ignores!

---

## Populated But Unused Columns (71 total)

These columns are **calculated and stored** but have **zero impact** on signal quality:

### Phase 1.1 Technical Indicators (Not Used in Scoring!)
- `rsi`, `macd_line`, `macd_signal`, `macd_histogram`
- `bollinger_upper`, `bollinger_lower`, `bollinger_position`, `bollinger_width`
- `volume_spike_ratio`, `volume_price_correlation`
- `volatility_rank`, `momentum_consistency_score`
- `sector_relative_strength`, `signal_strength_percentile`
- `above_50d_ma_pct`, `above_200d_ma_pct`

### Phase 1.2 Composite Metrics (Not Used!)
- `market_cap_category`, `expected_hold_duration`
- `float_turnover_ratio`, `liquidity_warning`
- `exit_signal_strength`

### Fundamentals (Not Used!)
- `pe_ratio`, `earnings_gap_pct`, `eps_growth`
- `roe`, `debt_equity`, `fcf_margin`

### Options Data (Not Used!)
- `put_call_oi_ratio`, `put_call_vol_ratio`
- `iv_spike_pct`, `implied_volatility`

### Ownership/Short Interest (Not Used!)
- `institutional_ownership_pct`, `retail_holding_pct`
- `short_pct_float`, `short_pct_outstanding`, `short_ratio`

---

## Orphaned/Dead Columns (19 total)

These columns exist but are **never populated or used:**

### ML/Prediction (Not Implemented)
- `ml_confidence_score`
- `prediction_confidence`
- `pattern_match_score`

### Options Flow (Not Implemented)
- `option_chain_data`
- `option_volume_ratio`
- `options_flow_score`
- `unusual_options_activity`

### Phase 2 Placeholders (Not Implemented)
- `reddit_momentum_score`
- `reddit_vs_price_divergence`
- `social_sentiment_trend`
- `institutional_flow_direction`

### Phase 4 Placeholders (Not Implemented)
- `entry_quality_score`
- `risk_adjusted_score`

### Metadata (Not Used)
- `commentary_metadata`
- `score_components`
- `scoring_version`
- `ai_commentary_version`
- `signal_duration`
- `rowid`

---

## Recommendations

### 🎯 Priority 1: Expand Weighted Score Calculation

**Problem:** Only using 3 inputs (reddit + financial) ignores all technical analysis.

**Solution:** Create a **multi-factor scoring system** that incorporates Phase 1 metrics:

```python
weighted_score = (
    # Current components (50%)
    reddit_score * 0.10 +
    financial_score * 0.40 +
    
    # NEW: Technical Score (30%)
    technical_score * 0.30 +
    
    # NEW: Risk-Adjusted Score (20%)
    risk_adjusted_score * 0.20
)

# Technical Score calculation
technical_score = average([
    rsi_score,                      # RSI momentum
    macd_strength,                  # MACD signal strength
    bollinger_position_score,       # Bollinger band position
    volume_score,                   # Volume confirmation
    sector_relative_strength_score  # Sector performance
])

# Risk-Adjusted Score calculation
risk_adjusted_score = (
    liquidity_score * 0.40 +        # Liquidity quality
    (1 - risk_score/100) * 0.30 +   # Inverse of risk
    momentum_consistency * 0.30      # Trend reliability
)
```

**Impact:**
- Signals ranked by quality, not just social buzz
- Phase 1 metrics finally used
- More stable, data-driven rankings

---

### 🔧 Priority 2: Clean Up Orphaned Columns

**Action Items:**

1. **Remove Dead Columns** (if not planning to implement):
   - `ml_confidence_score`, `prediction_confidence`, `pattern_match_score`
   - `option_chain_data`, `option_volume_ratio`, `options_flow_score`
   - `commentary_metadata`, `score_components`, `scoring_version`
   - `signal_duration`, `rowid`

2. **Mark Phase 2-4 Placeholders** (if planning to implement):
   - Add comments to schema indicating these are placeholders
   - Create implementation tickets for Phase 2-4 features

3. **Consolidate Metadata Fields**:
   - Combine `ai_commentary_version`, `scoring_version` into single `metadata` jsonb column

---

### 📈 Priority 3: Improve Financial Score Calculation

**Current Issue:** `financial_score` is opaque - we don't know what it includes.

**Recommendation:** Break down financial_score into sub-components:

```python
financial_score = (
    fundamental_score * 0.30 +   # PE, ROE, debt, FCF
    growth_score * 0.30 +        # EPS growth, revenue growth
    value_score * 0.20 +         # Price vs targets, earnings gap
    ownership_score * 0.20       # Institutional, insider activity
)
```

**Benefit:** Transparent, auditable, tunable scoring.

---

### 🎨 Priority 4: Implement Phase 2-4 Features (Optional)

If you want to use the orphaned columns:

#### Phase 2: Reddit Metrics
- Populate `reddit_momentum_score`, `reddit_vs_price_divergence`, `social_sentiment_trend`
- Add to weighted_score calculation

#### Phase 3: Options Flow
- Populate `unusual_options_activity`, `options_flow_score`
- Use for risk assessment

#### Phase 4: ML Predictions
- Implement `ml_confidence_score`, `prediction_confidence`, `pattern_match_score`
- Train models on historical backtest data

---

## Implementation Plan

### Phase A: Expand Scoring (2-3 days)
1. Create technical_score function
2. Create risk_adjusted_score function  
3. Update weighted_score calculation
4. Test on historical data
5. Deploy and monitor

### Phase B: Schema Cleanup (1 day)
1. Identify columns to drop
2. Create migration script
3. Update pipeline code
4. Test and deploy

### Phase C: Financial Score Breakdown (2-3 days)
1. Refactor financial signal generation
2. Create sub-component scores
3. Update weighted_score calculation
4. Test and validate

---

## Questions for Consideration

1. **Scoring Strategy:**
   - Should we prioritize technical quality or social momentum?
   - What weight should each category have?

2. **Schema Changes:**
   - Should we drop unused columns or keep as placeholders?
   - Do we plan to implement Phase 2-4 features?

3. **Backward Compatibility:**
   - How do we handle existing signals with old scoring?
   - Should we recalculate historical weighted_scores?

4. **Testing:**
   - How do we validate new scoring improves signal quality?
   - What metrics define "better" signals?

---

## Success Metrics

After implementing these changes, measure:

1. **Signal Quality:**
   - Average backtest return improvement
   - Win rate improvement
   - Sharpe ratio improvement

2. **Data Utilization:**
   - % of columns used in scoring (target: >50%)
   - Number of dead columns (target: <5)

3. **Transparency:**
   - Can we explain why each signal ranks where it does?
   - Are score components documented and logged?

---

## Conclusion

We have a **massive opportunity** to improve signal quality by using the rich data we're already collecting. The current scoring system ignores 90% of available data, relying almost entirely on reddit_score and financial_score.

**Recommended Next Steps:**
1. Implement expanded weighted_score calculation (Priority 1)
2. Validate improvement with backtests
3. Clean up orphaned columns (Priority 2)
4. Document and stabilize schema for production use

This will create a more robust, data-driven signal ranking system that leverages all the technical analysis and fundamental data we're collecting.
