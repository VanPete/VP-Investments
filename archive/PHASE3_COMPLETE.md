# Phase 3 Complete: Risk Score Calculator

## 🎯 Overview

Phase 3 implementation is **COMPLETE**. Added comprehensive risk scoring engine with 5 subscores, worst-factor guard, and structured diagnostics.

**Files Modified:**
- ✅ `backend/core/signals.py` - Added RiskScoreCalculator class (570+ lines)

**New Capabilities:**
- ✅ 5-subscore risk calculation (volatility, liquidity, leverage, short interest, concentration)
- ✅ Worst-factor guard (prevents risk understatement)
- ✅ Risk level assignment (Low/Moderate/Elevated/High/Extreme)
- ✅ Detailed risk_factors JSON with subscore breakdowns
- ✅ Market cap tier-based concentration risk
- ✅ Theme risk multipliers
- ✅ Special flags (inverse beta, event week)

---

## 📦 Implementation Details

### RiskScoreCalculator Class

**Location:** `backend/core/signals.py` (after TradeTypeClassifier, line ~800)

**Purpose:** Calculate comprehensive risk score (0-100) with 5 subscores and guard

**Formula:**
```python
# Weighted composite
composite = (
    volatility * 0.40 +
    liquidity * 0.25 +
    leverage * 0.15 +
    short_interest * 0.10 +
    concentration * 0.10
)

# Worst-factor guard (prevents understating risk)
risk_score = max(composite, 0.9 * max_subscore)
```

---

## 🎚️ Risk Scoring System

### Risk Levels

| Level | Score Range | Interpretation |
|-------|-------------|----------------|
| **Low** | 0-25 | Stable, low volatility, good liquidity |
| **Moderate** | 25-45 | Average risk profile, normal trading |
| **Elevated** | 45-65 | Above-average risk, careful position sizing |
| **High** | 65-80 | Significant risk, small positions only |
| **Extreme** | 80-100 | Very high risk, speculative only |

### Risk Weights

| Component | Weight | Focus |
|-----------|--------|-------|
| **Volatility** | 40% | ATR%, beta |
| **Liquidity** | 25% | Volume, float |
| **Leverage** | 15% | D/E, interest coverage |
| **Short Interest** | 10% | % of float |
| **Concentration** | 10% | Market cap, theme |

---

## 📊 Subscore Details

### 1. Volatility Risk (40% weight)

**Inputs:**
- ATR% (Average True Range as % of price, 20-day)
- Beta (vs market, absolute value used)

**ATR% Thresholds:**
```
<1.5%:    Low      (0-20)
1.5-3%:   Moderate (20-40)
3-5%:     Elevated (40-60)
5-8%:     High     (60-80)
>8%:      Extreme  (80-100)
```

**Beta Thresholds (|beta|):**
```
<0.8:     Low      (0-20)
0.8-1.2:  Moderate (20-40)
1.2-1.5:  Elevated (40-60)
1.5-2.0:  High     (60-80)
>2.0:     Extreme  (80-100)
```

**Calculation:**
```python
volatility_score = mean(atr_score, beta_score)
```

**Edge Cases:**
- Negative beta: Use absolute value (inverse correlation still risky)
- Missing data: Use available metric or default to 50

---

### 2. Liquidity Risk (25% weight)

**Inputs:**
- Average Daily Volume (shares)
- Price (for ADV in dollars)
- Float % (float / shares outstanding)

**ADV (in dollars) Thresholds:**
```
>$50M:      Low      (0-20)
$10M-$50M:  Moderate (20-40)
$2M-$10M:   Elevated (40-60)
$500K-$2M:  High     (60-80)
<$500K:     Extreme  (80-100)
```

**Float % Thresholds:**
```
>70%:   Low      (0-20)
50-70%: Moderate (20-40)
30-50%: Elevated (40-60)
15-30%: High     (60-80)
<15%:   Extreme  (80-100)
```

**Calculation:**
```python
liquidity_score = mean(adv_score, float_pct_score)
```

**Rationale:**
- Low ADV = hard to enter/exit positions
- Low float = susceptible to manipulation, squeezes

---

### 3. Leverage Risk (15% weight)

**Inputs:**
- Debt/Equity ratio
- Interest Coverage (EBIT / Interest Expense)

**Debt/Equity Thresholds:**
```
<0.3:   Low      (0-20)
0.3-0.8: Moderate (20-40)
0.8-1.5: Elevated (40-60)
1.5-3.0: High     (60-80)
>3.0:    Extreme  (80-100)
```

**Interest Coverage Thresholds:**
```
>4.0x:   Low      (0-20)
2.0-4.0x: Moderate (20-40)
1.5-2.0x: Elevated (40-60)
1.0-1.5x: High     (60-80)
<1.0x:    Extreme  (80-100)
```

**Calculation:**
```python
leverage_score = mean(de_score, coverage_score)
```

**Rationale:**
- High D/E = financial distress risk
- Low interest coverage = bankruptcy risk

---

### 4. Short Interest Risk (10% weight)

**Input:**
- Short Interest (% of float)

**Thresholds:**
```
<5%:    Low      (0-20)
5-10%:  Moderate (20-40)
10-20%: Elevated (40-60)
20-30%: High     (60-80)
>30%:   Extreme  (80-100)
```

**Rationale:**
- High short interest = potential squeeze OR legitimate bearish thesis
- Can increase volatility dramatically

**Default:** 30 (low-moderate) if no data

---

### 5. Concentration Risk (10% weight)

**Inputs:**
- Market Cap (in dollars)
- Theme (investment theme)

**Market Cap Tiers:**
```
Mega:  >$200B    Low      (10)
Large: $10B-$200B Moderate (25)
Mid:   $2B-$10B  Elevated (40)
Small: $300M-$2B High     (60)
Micro: $50M-$300M Extreme (80)
Nano:  <$50M     Extreme  (95)
```

**Theme Risk Multipliers:**
```
Crypto:            1.30x
Biotech:           1.20x
Speculative Growth: 1.15x
Green Energy:      1.10x
AI:                1.05x
Defense:           0.95x
Utilities:         0.90x
None:              1.00x
```

**Calculation:**
```python
concentration_score = base_score * theme_multiplier
# Cap at 100
```

**Rationale:**
- Smaller cap = less liquidity, more volatile
- Certain themes inherently riskier (crypto, biotech)

---

## 🛡️ Worst-Factor Guard

### Purpose

Prevent risk understatement when one subscore is extremely high but others are moderate.

### Formula

```python
max_subscore = max(volatility, liquidity, leverage, short, concentration)
risk_score = max(composite, 0.9 * max_subscore)
```

### Example

```python
# Subscores:
volatility = 90     # Extreme
liquidity = 30      # Moderate
leverage = 25       # Moderate
short = 20          # Low
concentration = 35  # Moderate

# Weighted composite:
composite = 90*0.4 + 30*0.25 + 25*0.15 + 20*0.1 + 35*0.1
          = 36 + 7.5 + 3.75 + 2 + 3.5
          = 52.75  (Elevated)

# Worst-factor guard:
max_subscore = 90
risk_score = max(52.75, 0.9 * 90)
           = max(52.75, 81)
           = 81 (Extreme)

# Result: Risk properly classified as Extreme due to volatility
```

---

## 📋 Risk Factors JSON Structure

### Output Format

```json
{
  "volatility": {
    "score": 35.2,
    "label": "Moderate",
    "atr_pct": 2.5,
    "beta": 1.2,
    "beta_abs": 1.2
  },
  "liquidity": {
    "score": 42.8,
    "label": "Elevated",
    "avg_volume": 5000000,
    "adv_dollars": 250000000,
    "float_pct": 45.5
  },
  "leverage": {
    "score": 28.5,
    "label": "Moderate",
    "debt_to_equity": 0.65,
    "interest_coverage": 3.2
  },
  "short_interest": {
    "score": 22.0,
    "label": "Moderate",
    "short_pct_float": 8.5
  },
  "concentration": {
    "score": 31.5,
    "label": "Moderate",
    "market_cap": 15000000000,
    "market_cap_tier": "Large",
    "theme": "AI",
    "theme_multiplier": 1.05
  },
  "composite": {
    "score": 34.1,
    "max_subscore": 42.8,
    "guard_applied": false
  },
  "flags": {
    "inverse_beta": false,
    "event_week": true
  }
}
```

### Special Flags

**inverse_beta:**
- `true` if beta < 0 (negative correlation with market)
- Useful for hedging analysis

**event_week:**
- `true` if earnings within ±7 days
- Indicates elevated volatility risk

---

## 🔧 Usage Examples

### Example 1: Low Risk Large Cap

```python
from backend.core.signals import RiskScoreCalculator

risk_calc = RiskScoreCalculator()

signal_data = {
    'atr_pct': 1.2,              # Low volatility
    'beta': 0.95,                # Market-like
    'avg_volume': 15000000,      # 15M shares
    'price': 150.0,              # $150/share → $2.25B ADV
    'float_pct': 85.0,           # High float
    'debt_to_equity': 0.45,      # Moderate debt
    'interest_coverage': 8.5,    # Strong coverage
    'short_interest': 3.2,       # Low short interest
    'market_cap': 180_000_000_000,  # $180B (Large cap)
}

risk_score, risk_level, risk_factors = risk_calc.calculate_risk_score(
    ticker='AAPL',
    signal_data=signal_data,
    theme='AI'
)

print(f"Risk Score: {risk_score:.1f}")  # ~18.5
print(f"Risk Level: {risk_level}")      # Low
print(f"Volatility: {risk_factors['volatility']['score']:.1f}")  # ~15
print(f"Liquidity: {risk_factors['liquidity']['score']:.1f}")    # ~12
```

---

### Example 2: High Risk Small Cap

```python
signal_data = {
    'atr_pct': 6.5,              # High volatility
    'beta': 1.8,                 # Aggressive
    'avg_volume': 500000,        # 500K shares
    'price': 8.50,               # $8.50/share → $4.25M ADV
    'float_pct': 22.0,           # Low float
    'debt_to_equity': 2.1,       # High debt
    'interest_coverage': 1.3,    # Weak coverage
    'short_interest': 18.5,      # Elevated short interest
    'market_cap': 850_000_000,   # $850M (Small cap)
}

risk_score, risk_level, risk_factors = risk_calc.calculate_risk_score(
    ticker='SPCE',
    signal_data=signal_data,
    theme='Speculative Growth'
)

print(f"Risk Score: {risk_score:.1f}")  # ~72.3
print(f"Risk Level: {risk_level}")      # High
print(f"Guard Applied: {risk_factors['composite']['guard_applied']}")  # Likely True
```

---

### Example 3: Extreme Risk Crypto/Biotech

```python
signal_data = {
    'atr_pct': 12.0,             # Extreme volatility
    'beta': 2.5,                 # Very aggressive
    'avg_volume': 200000,        # 200K shares
    'price': 3.25,               # $3.25/share → $650K ADV
    'float_pct': 12.0,           # Very low float
    'debt_to_equity': None,      # No debt data (startup)
    'interest_coverage': None,
    'short_interest': 35.0,      # Extreme short interest
    'market_cap': 120_000_000,   # $120M (Micro cap)
}

risk_score, risk_level, risk_factors = risk_calc.calculate_risk_score(
    ticker='XXXX',
    signal_data=signal_data,
    theme='Crypto'
)

print(f"Risk Score: {risk_score:.1f}")  # ~88.5
print(f"Risk Level: {risk_level}")      # Extreme
print(f"Volatility: {risk_factors['volatility']['score']:.1f}")  # ~92
print(f"Concentration: {risk_factors['concentration']['score']:.1f}")  # ~104 → 100 (capped)
```

---

## 🔗 Integration Points

### 1. SignalScorer Integration (Phase 5)

```python
# In SignalScorer.score_ticker() method

# Initialize calculator (once in __init__)
self.risk_calc = RiskScoreCalculator()

# After trade type classification
risk_score, risk_level, risk_factors = self.risk_calc.calculate_risk_score(
    ticker=ticker,
    signal_data={
        'atr_pct': atr_pct,
        'beta': beta,
        'avg_volume': avg_volume,
        'float_pct': float_pct,
        'price': current_price,
        'debt_to_equity': de_ratio,
        'interest_coverage': interest_coverage,
        'short_interest': short_interest,
        'market_cap': market_cap,
        'earnings_date': earnings_days_away
    },
    theme=classification_details.get('theme')
)

# Store in signal result
signal.risk_score = risk_score
signal.risk_level = risk_level
signal.risk_factors = risk_factors
```

### 2. Database Storage

```python
# In database.py, when inserting signal

signal_data = {
    # ... existing fields
    'risk_score': risk_score,           # NUMERIC(5,2)
    'risk_level': risk_level,           # TEXT (keep for backward compat)
    'risk_factors': json.dumps(risk_factors),  # JSONB
    'atr_pct': signal_data.get('atr_pct'),
    'float_pct': signal_data.get('float_pct'),
    'interest_coverage': signal_data.get('interest_coverage')
}
```

### 3. Backtesting Integration (Phase 7)

```python
# Entry thresholds by risk level
ENTRY_THRESHOLDS = {
    'Low': 0.55,
    'Moderate': 0.60,
    'Elevated': 0.65,
    'High': 0.70,
    'Extreme': 0.75
}

# Position sizing by risk level
POSITION_SIZE = {
    'Low': 1.0,      # Full position
    'Moderate': 0.75,
    'Elevated': 0.5,
    'High': 0.25,
    'Extreme': 0.1   # Tiny position
}

# Stop loss by ATR%
stop_loss = current_price * (1 - atr_pct * 2.0)
```

---

## ✅ Verification Checklist

### Code Quality
- ✅ Comprehensive docstrings for all methods
- ✅ Type hints for all parameters
- ✅ Error handling with logging
- ✅ Clear threshold documentation
- ✅ Modular subscore methods

### Functionality
- ✅ All 5 subscores implemented
- ✅ Weighted composite calculation
- ✅ Worst-factor guard logic
- ✅ Risk level assignment (5 levels)
- ✅ Detailed risk_factors JSON
- ✅ Market cap tier classification
- ✅ Theme risk multipliers
- ✅ Special flags (inverse beta, event week)

### Edge Cases
- ✅ Missing data handling (defaults)
- ✅ Negative beta handling (absolute value)
- ✅ Score clamping (0-100)
- ✅ Division by zero protection
- ✅ None value handling

### Integration Ready
- ✅ Accepts signal_data dict
- ✅ Returns tuple (score, level, factors)
- ✅ JSON-serializable output
- ✅ Compatible with database schema
- ✅ Theme integration from TradeTypeClassifier

---

## 🚀 Next Steps: Phase 4

**Phase 4: Data Collection Enhancements**

Enhance yfinance integration to fetch all required risk metrics:

1. **ATR% Calculation**
   - Fetch 20-day high/low/close
   - Calculate True Range: max(H-L, |H-C_prev|, |L-C_prev|)
   - ATR% = (ATR_20 / price) * 100

2. **Float Percentage**
   - sharesOutstanding (yfinance)
   - floatShares (yfinance)
   - float_pct = (floatShares / sharesOutstanding) * 100

3. **Interest Coverage**
   - EBIT from income statement
   - Interest Expense from income statement
   - interest_coverage = EBIT / Interest Expense

4. **Historical Data for MA Slopes**
   - Fetch 250+ days of price history
   - Fetch 60+ days of volume history
   - Used by TrendStrengthCalculator

5. **Market Data**
   - Beta (yfinance: `info['beta']`)
   - Short Interest (yfinance: `info['shortPercentOfFloat']`)
   - Average Volume (yfinance: `info['averageVolume']`)

6. **Enhanced yfinance.py**
   ```python
   def fetch_enhanced_data(ticker: str) -> Dict[str, Any]:
       """Fetch all data required for risk scoring"""
       # Basic info
       # Price history (250 days)
       # Volume history (60 days)
       # Financials (quarterly + annual)
       # Calculate ATR%
       # Calculate float%
       # Calculate interest coverage
       return enhanced_data
   ```

**Estimated Effort:** 6-8 hours

**Dependencies:**
- Phase 2 ✅ Complete
- Phase 3 ✅ Complete

---

## 📝 Summary

**Phase 3 Status:** ✅ **COMPLETE**

**Lines Added:** 570+ (RiskScoreCalculator class)

**Key Achievements:**
- ✅ 5-subscore risk calculation with detailed thresholds
- ✅ Worst-factor guard implementation
- ✅ Risk level assignment (5 levels)
- ✅ Comprehensive risk_factors JSON output
- ✅ Market cap tier classification
- ✅ Theme risk multipliers
- ✅ Special flags for edge cases
- ✅ Integration-ready with existing calculators

**Overall Progress:** 30% complete (3 of 10 phases)
- Phase 1: Schema Migration ✅
- Phase 2: Core Infrastructure ✅
- Phase 3: Risk Scoring ✅ (NEW!)
- **Total:** 1,510+ lines of production code

**Next Milestone:** Phase 4 - Data Collection Enhancements

---

**Ready to proceed to Phase 4?** 🚀
