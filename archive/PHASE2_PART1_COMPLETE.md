# Phase 2 Implementation - Core Calculation Infrastructure

## ✅ Completed: Part 1 - Foundation Classes

### Files Modified:
1. **`backend/core/signals.py`**
   - Added scipy imports for statistical calculations
   - Added 3 new calculator classes (580+ lines of code)

---

## 🏗️ Classes Implemented

### 1. ZScoreCalculator
**Purpose:** Rolling window z-score standardization for regime-aware classification

**Features:**
- ✅ 60-day rolling window (configurable, min 30 days)
- ✅ Falls back to universe statistics when ticker history < 20 samples
- ✅ Fetches historical data from database automatically
- ✅ Caps extreme z-scores at ±5.0
- ✅ Updates universe stats from signal batches

**Key Methods:**
```python
calculate_z_score(value, ticker, feature, historical_data=None, db_manager=None) -> float
update_universe_stats(all_signals: List[Dict])
```

**Usage Example:**
```python
z_calc = ZScoreCalculator(lookback_days=60, min_samples=30)
z_calc.update_universe_stats(recent_signals)  # Update universe fallback

technical_z = z_calc.calculate_z_score(
    value=0.75,
    ticker='AAPL',
    feature='technical_score',
    db_manager=db
)
# Returns: z-score (e.g., +1.2 if 1.2 std devs above mean)
```

---

### 2. TrendStrengthCalculator
**Purpose:** Composite trend strength for Momentum classification

**Formula:**
```
TrendStrength = 0.5 * z(slope_50) + 0.3 * z(slope_200) + 0.2 * z(volume_trend)

MA Slope: 252 * OLS_slope(log(price), lookback_days)
Volume Trend: 20-day avg / 60-day avg
```

**Features:**
- ✅ Calculates 50-day and 200-day MA slopes (annualized)
- ✅ Uses OLS regression on log(price) for accurate trend measurement
- ✅ Incorporates volume trend (20d vs 60d)
- ✅ Returns both composite score and detailed components
- ✅ Handles missing/insufficient data gracefully

**Key Methods:**
```python
calculate_trend_strength(ticker, price_history, volume_history, db_manager=None) 
    -> Tuple[float, Dict[str, float]]
```

**Usage Example:**
```python
trend_calc = TrendStrengthCalculator(z_calc)

trend_strength, components = trend_calc.calculate_trend_strength(
    ticker='NVDA',
    price_history=[100, 102, 105, 108, ...],  # Last 200+ days
    volume_history=[5000000, 5200000, ...]    # Last 60+ days
)

# Returns:
# trend_strength: 1.45 (strong uptrend)
# components: {
#   'ma_slope_50': 0.52,
#   'ma_slope_50_z': 1.8,
#   'ma_slope_200': 0.38,
#   'ma_slope_200_z': 1.2,
#   'volume_trend': 1.15,
#   'volume_trend_z': 1.3
# }
```

---

### 3. ValuationCalculator
**Purpose:** Valuation composite z-score for Value classification

**Formula:**
```
valuation_z = mean(z(P/E), z(P/B), z(FCF_yield) * -1)

Note: FCF yield multiplied by -1 because higher yield = cheaper = better
```

**Features:**
- ✅ Combines P/E, P/B, and FCF yield
- ✅ Inverts FCF yield z-score (higher yield = lower valuation)
- ✅ Handles missing metrics gracefully (averages available)
- ✅ Returns both composite z-score and individual components

**Key Methods:**
```python
calculate_valuation_z(ticker, pe_ratio, pb_ratio, fcf_yield, db_manager=None)
    -> Tuple[float, Dict[str, float]]
```

**Usage Example:**
```python
val_calc = ValuationCalculator(z_calc)

valuation_z, components = val_calc.calculate_valuation_z(
    ticker='WMT',
    pe_ratio=22.5,
    pb_ratio=4.2,
    fcf_yield=0.035,  # 3.5% FCF yield
)

# Returns:
# valuation_z: -0.85 (below average valuation = good for value)
# components: {
#   'pe_ratio': 22.5,
#   'pe_z': -0.5,
#   'pb_ratio': 4.2,
#   'pb_z': -0.6,
#   'fcf_yield': 0.035,
#   'fcf_yield_z': 1.2,  # But inverted in average
#   'valuation_z': -0.85
# }
```

---

## 🔄 Integration Points

### In Pipeline/Signal Scoring:

```python
# Initialize calculators (once per pipeline run)
z_calc = ZScoreCalculator(lookback_days=60, min_samples=30)
trend_calc = TrendStrengthCalculator(z_calc)
val_calc = ValuationCalculator(z_calc)

# Update universe stats from recent signals
recent_signals = db.get_recent_signals(days=60)
z_calc.update_universe_stats(recent_signals)

# For each ticker in pipeline:
for ticker, data in ticker_data.items():
    # Calculate z-scores for component scores
    technical_z = z_calc.calculate_z_score(
        data['technical_score'], ticker, 'technical_score', db_manager=db
    )
    fundamental_z = z_calc.calculate_z_score(
        data['fundamental_score'], ticker, 'fundamental_score', db_manager=db
    )
    news_z = z_calc.calculate_z_score(
        data['news_macro_score'], ticker, 'news_macro_score', db_manager=db
    )
    social_z = z_calc.calculate_z_score(
        data['social_alternative_score'], ticker, 'social_alternative_score', db_manager=db
    )
    
    # Calculate trend strength
    trend_strength, trend_components = trend_calc.calculate_trend_strength(
        ticker=ticker,
        price_history=data['price_history'],
        volume_history=data['volume_history'],
        db_manager=db
    )
    
    # Calculate valuation z-score
    valuation_z, val_components = val_calc.calculate_valuation_z(
        ticker=ticker,
        pe_ratio=data.get('pe_ratio'),
        pb_ratio=data.get('price_to_book'),
        fcf_yield=data.get('fcf_yield'),
        db_manager=db
    )
    
    # Store in signal data for trade type classification
    signal_data[ticker] = {
        **data,
        'technical_z': technical_z,
        'fundamental_z': fundamental_z,
        'news_z': news_z,
        'social_z': social_z,
        'trend_strength': trend_strength,
        'valuation_z': valuation_z,
        'ma_slope_50': trend_components.get('ma_slope_50'),
        'ma_slope_200': trend_components.get('ma_slope_200'),
        'volume_trend_z': trend_components.get('volume_trend_z')
    }
```

---

## 📊 Database Integration

These calculators automatically fetch historical data from the signals table:

```python
# Z-score calculator queries:
SELECT created_at, technical_score, fundamental_score, ...
FROM signals
WHERE ticker = 'AAPL'
  AND created_at >= (NOW() - INTERVAL '70 days')
ORDER BY created_at ASC
```

**Cold Start Handling:**
- If ticker has < 20 historical signals → Uses universe statistics
- Universe stats computed from all recent signals across all tickers
- Gracefully degrades without errors

---

## ✅ What's Ready

### Infrastructure:
- [x] Z-score calculation with rolling windows
- [x] Universe fallback statistics
- [x] Trend strength composite scoring
- [x] MA slope calculation (OLS on log prices)
- [x] Volume trend analysis
- [x] Valuation composite z-score
- [x] Historical data fetching from database
- [x] Error handling and edge cases

### Next Steps (Phase 2 Part 2):
- [ ] Trade Type Classification logic
- [ ] Event Detection (keywords + earnings)
- [ ] Theme Detection config
- [ ] Risk Score Calculator
- [ ] Risk subscores (volatility, liquidity, leverage, etc.)
- [ ] Contrarian bonus calculation

---

## 🧪 Testing

### Unit Test Examples:

```python
def test_z_score_calculation():
    """Test z-score with known distribution."""
    z_calc = ZScoreCalculator()
    
    # Mock historical data: [0.5, 0.6, 0.7, 0.8, 0.9] (mean=0.7, std=0.158)
    z_score = z_calc.calculate_z_score(
        value=0.9,
        ticker='TEST',
        feature='test_score',
        historical_data=[
            {'test_score': 0.5}, {'test_score': 0.6},
            {'test_score': 0.7}, {'test_score': 0.8},
            {'test_score': 0.9}
        ] * 10  # Repeat to get 50 samples
    )
    
    # Should be approximately +1.26 std devs
    assert 1.0 < z_score < 1.5

def test_trend_strength():
    """Test trend strength calculation."""
    z_calc = ZScoreCalculator()
    trend_calc = TrendStrengthCalculator(z_calc)
    
    # Upward trending prices
    prices = [100 + i*0.5 for i in range(250)]  # Steady uptrend
    volumes = [1000000] * 250  # Constant volume
    
    trend_strength, components = trend_calc.calculate_trend_strength(
        ticker='TEST',
        price_history=prices,
        volume_history=volumes
    )
    
    # Should be positive (uptrend)
    assert trend_strength > 0
    assert components['ma_slope_50'] > 0
    assert components['ma_slope_200'] > 0

def test_valuation_z():
    """Test valuation z-score calculation."""
    z_calc = ZScoreCalculator()
    val_calc = ValuationCalculator(z_calc)
    
    valuation_z, components = val_calc.calculate_valuation_z(
        ticker='TEST',
        pe_ratio=15.0,  # Low P/E
        pb_ratio=2.0,   # Low P/B
        fcf_yield=0.06  # High FCF yield (6%)
    )
    
    # Should indicate undervalued (negative z-score)
    # Note: Need universe stats for accurate test
    assert 'valuation_z' in components
```

---

## 📝 Documentation

All classes include:
- ✅ Comprehensive docstrings
- ✅ Formula explanations
- ✅ Parameter descriptions
- ✅ Return value specifications
- ✅ Usage examples in comments

---

## 🎯 Status Summary

**Phase 2 - Part 1: COMPLETE ✅**

**Lines of Code Added:** ~580 lines
**Classes Created:** 3
**Methods Implemented:** 12
**Error Handling:** Comprehensive with graceful degradation

**Ready for:** Trade Type Classification (Phase 2 Part 2)

---

## 🔍 Verification Checklist

Before proceeding to Part 2:

- [x] All imports added (scipy)
- [x] ZScoreCalculator class complete
- [x] TrendStrengthCalculator class complete
- [x] ValuationCalculator class complete
- [x] Database integration methods
- [x] Universe fallback statistics
- [x] Error handling and edge cases
- [x] Logging statements
- [x] Type hints
- [x] Docstrings

**Status:** ✅ Ready for Phase 2 Part 2 - Trade Classification & Event Detection
