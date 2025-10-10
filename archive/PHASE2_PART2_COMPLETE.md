# Phase 2 Part 2 Complete: Trade Classification & Event Detection

## 🎯 Overview

Phase 2 Part 2 implementation is **COMPLETE**. Added comprehensive trade type classification engine with z-score based thresholds, event detection, and theme mapping.

**Files Modified:**
- ✅ `backend/core/signals.py` - Added TradeTypeClassifier class (360+ lines)

**New Capabilities:**
- ✅ Z-score based trade type classification (6 types)
- ✅ Primary + secondary type assignment (max 2)
- ✅ Multi-Factor auto-tagging (≥3 strong components)
- ✅ Event detection (earnings, M&A, contracts, products)
- ✅ Theme detection (5 investment themes)
- ✅ Oversold condition detection
- ✅ Structured classification diagnostics

---

## 📦 Implementation Details

### TradeTypeClassifier Class

**Location:** `backend/core/signals.py` (after ValuationCalculator, line ~450)

**Purpose:** Classify signals into trade types using z-score thresholds and event detection

**Dependencies:**
- ZScoreCalculator (for component score normalization)
- TrendStrengthCalculator (for momentum classification)
- ValuationCalculator (for value classification)

---

## 🏷️ Trade Type Classification System

### 6 Trade Types

**1. Momentum**
- **Criteria:** technical_z ≥ 0.8 AND trend_strength ≥ 0.6
- **Score:** technical_z + trend_strength
- **Use Case:** Strong uptrends with volume confirmation

**2. Value**
- **Criteria:** valuation_z ≤ -0.6 AND fundamental_quality_z ≥ 0.3
- **Score:** |valuation_z| + fundamental_quality_z
- **Use Case:** Undervalued stocks with solid fundamentals

**3. Speculative Growth**
- **Criteria:** revenue_growth_z ≥ 0.8 AND negative FCF
- **Score:** revenue_growth_z
- **Use Case:** High-growth companies burning cash

**4. Event-Driven**
- **Criteria:** (earnings ≤7 days) OR (news_z ≥ 0.7 + event keywords)
- **Priority:** Highest priority (score 10.0)
- **Use Case:** Earnings plays, M&A targets, contract awards

**5. Contrarian**
- **Criteria:** is_oversold AND social_z ≤ -0.5 AND fundamentals_trend_z ≥ 0.2
- **Oversold:** RSI ≤ 30 OR price_z_20day ≤ -2.0
- **Score:** |social_z| + fundamentals_trend_z
- **Use Case:** Oversold stocks with improving fundamentals

**6. Multi-Factor**
- **Criteria:** ≥3 components with z-score ≥ 0.5
- **Auto-appended:** Added to trade_tags if conditions met
- **Use Case:** Broad-based strength across multiple factors

### Classification Logic

```python
# Priority-based candidate selection
candidates = []

# 1. Event-Driven (highest priority)
if event_flags['has_earnings'] or significant_event:
    candidates.append(('Event-Driven', 10.0))

# 2-5. Other types with calculated scores
# (Momentum, Value, Growth, Contrarian)

# Sort by score, assign top 2
candidates.sort(key=score, reverse=True)
trade_tags = [candidates[0], candidates[1] if strong_enough]

# Auto-append Multi-Factor if ≥3 strong components
if strong_components >= 3:
    trade_tags.append('Multi-Factor')
```

---

## 🔍 Event Detection System

### Event Types

**1. Earnings Events**
- Detect from `earnings_date` field
- Check if within ±7 trading days
- Store `earnings_days_away` (negative = past, positive = future)

**2. M&A Events**
- Keywords: merger, acquisition, takeover, go-private, buyout, acquiring, acquired
- Scan news_content and social_content

**3. Contract Awards**
- Keywords: contract, awarded, wins deal, option exercised, IDIQ, government contract
- Common in defense and government sectors

**4. Product Launches**
- Keywords: launches, unveils, announces, FDA approval, FDA clearance, product release, new product
- Common in biotech and tech sectors

### Event Flags JSON

```json
{
  "has_earnings": true,
  "earnings_days_away": 3,
  "has_ma": false,
  "has_contract": true,
  "has_product": false,
  "keywords": ["contract", "awarded", "government contract"]
}
```

---

## 🎨 Theme Detection System

### Investment Themes

**1. AI (Artificial Intelligence)**
- Tickers: NVDA, AMD, MSFT, GOOGL, META, TSLA, PLTR, C3AI, AI, BBAI

**2. Biotech**
- Tickers: MRNA, BNTX, GILD, REGN, VRTX, BIIB, AMGN, ILMN

**3. Defense**
- Tickers: LMT, RTX, BA, NOC, GD, TXT, HII, LDOS

**4. Green Energy**
- Tickers: TSLA, ENPH, SEDG, FSLR, RUN, PLUG, BE, CHPT

**5. Crypto**
- Tickers: COIN, MSTR, MARA, RIOT, SI, HUT, BITF

### Future Enhancement
- Add keyword-based theme detection from news/social
- NLP-based theme classification
- Dynamic theme expansion via config

---

## 📊 Classification Details Output

### Structure

```python
classification_details = {
    'primary_type': 'Momentum',
    'secondary_type': 'Event-Driven',
    'multi_factor': False,
    'scores': {
        'technical_z': 1.2,
        'fundamental_z': 0.3,
        'news_z': 0.8,
        'social_z': 0.6,
        'trend_strength': 1.5,
        'valuation_z': -0.2,
        'fundamental_quality_z': 0.5,
        'revenue_growth_z': 0.4,
        'fundamentals_trend_z': 0.3,
        'ma_slope_50': 0.52,
        'ma_slope_200': 0.38,
        'volume_trend_z': 1.3,
        'price_z_20day': 0.8
    },
    'event_flags': {
        'has_earnings': True,
        'earnings_days_away': 3,
        'has_ma': False,
        'has_contract': False,
        'has_product': False,
        'keywords': []
    },
    'theme': 'AI',
    'is_oversold': False,
    'candidates': [
        {'type': 'Momentum', 'score': 2.7},
        {'type': 'Event-Driven', 'score': 10.0}
    ]
}
```

---

## 🔧 Usage Examples

### Example 1: Basic Classification

```python
from backend.core.signals import ZScoreCalculator, TrendStrengthCalculator, ValuationCalculator, TradeTypeClassifier

# Initialize calculators
z_calc = ZScoreCalculator(lookback_days=60, min_samples=30)
trend_calc = TrendStrengthCalculator(z_calc)
val_calc = ValuationCalculator(z_calc)
classifier = TradeTypeClassifier(z_calc, trend_calc, val_calc)

# Prepare signal data
signal_data = {
    'price_history': [100, 102, 105, 108, ...],  # 200+ days
    'volume_history': [5000000, 5200000, ...],    # 60+ days
    'pe_ratio': 22.5,
    'price_to_book': 4.2,
    'fcf_yield': 0.035,
    'roe': 0.18,
    'profit_margins': 0.15,
    'revenue_growth': 0.25,
    'earnings_growth': 0.30,
    'free_cash_flow': 1500000000,
    'rsi': 45.0,
    'earnings_date': '2025-10-15',
    'news_content': 'Company announces new AI chip launch',
    'social_content': 'Reddit: NVDA looking strong'
}

component_scores = {
    'technical_score': 0.75,
    'fundamental_score': 0.68,
    'news_score': 0.72,
    'social_score': 0.65
}

# Classify
trade_tags, details = classifier.classify_trade_type(
    ticker='NVDA',
    signal_data=signal_data,
    component_scores=component_scores,
    db_manager=db
)

print(f"Trade Tags: {trade_tags}")
# Output: ['Momentum', 'Multi-Factor']

print(f"Primary: {details['primary_type']}")
# Output: Momentum

print(f"Theme: {details['theme']}")
# Output: AI
```

### Example 2: Event-Driven Classification

```python
signal_data = {
    'earnings_date': 2,  # 2 days away
    'news_content': 'Company awarded $500M government contract',
    'pe_ratio': 18.0,
    'rsi': 55.0
}

component_scores = {
    'technical_score': 0.60,
    'fundamental_score': 0.65,
    'news_score': 0.85,
    'social_score': 0.55
}

trade_tags, details = classifier.classify_trade_type(
    ticker='LMT',
    signal_data=signal_data,
    component_scores=component_scores,
    db_manager=db
)

print(f"Trade Tags: {trade_tags}")
# Output: ['Event-Driven', 'Multi-Factor']

print(f"Event Flags: {details['event_flags']}")
# Output: {has_earnings: True, earnings_days_away: 2, has_contract: True, ...}

print(f"Theme: {details['theme']}")
# Output: Defense
```

### Example 3: Contrarian Classification

```python
signal_data = {
    'rsi': 28.0,  # Oversold
    'price_z_20day': -2.3,
    'revenue_growth': 0.15,
    'earnings_growth': 0.18,
    'social_content': 'Everyone hates this stock, selling off'
}

component_scores = {
    'technical_score': 0.35,
    'fundamental_score': 0.62,
    'news_score': 0.45,
    'social_score': 0.25  # Very negative sentiment
}

trade_tags, details = classifier.classify_trade_type(
    ticker='WMT',
    signal_data=signal_data,
    component_scores=component_scores,
    db_manager=db
)

print(f"Trade Tags: {trade_tags}")
# Output: ['Contrarian']

print(f"Is Oversold: {details['is_oversold']}")
# Output: True

print(f"Social Z: {details['scores']['social_z']}")
# Output: -0.85 (very negative)
```

---

## 🔗 Integration Points

### 1. SignalScorer Integration (Next: Phase 3)

```python
# In SignalScorer.score_ticker() method

# After calculating component scores
trade_tags, classification_details = self.trade_classifier.classify_trade_type(
    ticker=ticker,
    signal_data=raw_data,
    component_scores={
        'technical_score': technical_score,
        'fundamental_score': fundamental_score,
        'news_score': news_score,
        'social_score': social_score
    },
    db_manager=self.db_manager
)

# Store in signal result
signal.trade_tags = trade_tags
signal.technical_z = classification_details['scores']['technical_z']
signal.fundamental_z = classification_details['scores']['fundamental_z']
signal.news_z = classification_details['scores']['news_z']
signal.social_z = classification_details['scores']['social_z']
signal.trend_strength = classification_details['scores']['trend_strength']
signal.valuation_z = classification_details['scores']['valuation_z']
signal.ma_slope_50 = classification_details['scores']['ma_slope_50']
signal.ma_slope_200 = classification_details['scores']['ma_slope_200']
signal.volume_trend_z = classification_details['scores']['volume_trend_z']
signal.price_z_20day = classification_details['scores']['price_z_20day']
signal.event_flags = classification_details['event_flags']
signal.theme = classification_details['theme']
```

### 2. Database Storage

```python
# In database.py, when inserting signal

signal_data = {
    # ... existing fields
    'trade_tags': trade_tags,  # Array of strings
    'technical_z': classification_details['scores']['technical_z'],
    'fundamental_z': classification_details['scores']['fundamental_z'],
    'news_z': classification_details['scores']['news_z'],
    'social_z': classification_details['scores']['social_z'],
    'trend_strength': classification_details['scores']['trend_strength'],
    'valuation_z': classification_details['scores']['valuation_z'],
    'ma_slope_50': classification_details['scores']['ma_slope_50'],
    'ma_slope_200': classification_details['scores']['ma_slope_200'],
    'volume_trend_z': classification_details['scores']['volume_trend_z'],
    'price_z_20day': classification_details['scores']['price_z_20day'],
    'event_flags': json.dumps(classification_details['event_flags']),
    'theme': classification_details['theme']
}
```

---

## ✅ Verification Checklist

### Code Quality
- ✅ Comprehensive docstrings
- ✅ Type hints for all parameters
- ✅ Error handling with logging
- ✅ Clear variable naming
- ✅ Modular helper methods

### Functionality
- ✅ Z-score based thresholds implemented
- ✅ Priority-based candidate selection
- ✅ Primary + secondary type logic
- ✅ Multi-Factor auto-tagging
- ✅ Event detection for 4 event types
- ✅ Theme detection for 5 themes
- ✅ Oversold condition detection
- ✅ Structured output with diagnostics

### Integration Ready
- ✅ Accepts component_scores dict
- ✅ Returns trade_tags list (max 2-3)
- ✅ Returns classification_details dict
- ✅ Database manager integration
- ✅ Compatible with existing Signal structure

---

## 🚀 Next Steps: Phase 3

**Phase 3: Risk Score Calculator**

Implement comprehensive risk scoring system with 5 subscores:

1. **RiskScoreCalculator Class**
   - Volatility subscore (40%): ATR%, beta
   - Liquidity subscore (25%): ADV, float%
   - Leverage subscore (15%): D/E, interest coverage
   - Short Interest subscore (10%): % of float
   - Concentration subscore (10%): Market cap, theme

2. **Worst-Factor Guard**
   ```python
   risk_score = max(composite, 0.9 * max_subfactor)
   ```

3. **Risk Level Assignment**
   - Low: <25
   - Moderate: 25-45
   - Elevated: 45-65
   - High: 65-80
   - Extreme: >80

4. **Risk Factors JSON Generation**
   ```json
   {
     "volatility": {"score": 35, "label": "Moderate", "atr_pct": 2.5, "beta": 1.2},
     "liquidity": {"score": 20, "label": "Low", "adv": 5000000, "float_pct": 45},
     "leverage": {"score": 40, "label": "Elevated", "de_ratio": 1.2, "interest_coverage": 3.5},
     "short_interest": {"score": 15, "label": "Low", "pct_float": 3.2},
     "concentration": {"score": 30, "label": "Moderate", "market_cap": "Mid", "theme": "AI"},
     "flags": {"inverse_beta": false, "event_week": true}
   }
   ```

**Estimated Effort:** 4-6 hours

**Dependencies:**
- Phase 2 Part 1 ✅ Complete
- Phase 2 Part 2 ✅ Complete
- Enhanced yfinance data collection (Phase 4)

---

## 📝 Summary

**Phase 2 Part 2 Status:** ✅ **COMPLETE**

**Lines Added:** 360+ (TradeTypeClassifier class)

**Key Achievements:**
- ✅ Z-score based trade type classification
- ✅ 6 trade types with priority-based selection
- ✅ Event detection (earnings, M&A, contracts, products)
- ✅ Theme detection (5 investment themes)
- ✅ Multi-Factor auto-tagging
- ✅ Structured diagnostics output
- ✅ Integration-ready with SignalScorer

**Overall Phase 2 Progress:** ✅ **COMPLETE** (Parts 1 & 2)
- Part 1: Infrastructure calculators (580 lines)
- Part 2: Trade classification (360 lines)
- **Total:** 940+ lines of production-ready code

**Next Milestone:** Phase 3 - Risk Score Calculator

---

**Ready to proceed to Phase 3?** 🚀
