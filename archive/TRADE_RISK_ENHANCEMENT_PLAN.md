# Trade Type & Risk Enhancement - Implementation Plan

## Overview
Upgrading trade_type and risk assessment from basic text labels to analytical, structured scoring system with z-score based classification and composite risk metrics.

---

## Phase 1: Database Schema Changes

### New Columns to Add

```sql
-- In SIGNAL METADATA section (after expected_hold_duration):
trade_tags TEXT[],  -- Primary + secondary trade types (max 2)

-- In GROUP 5: RISK/STABILITY section (after risk_category):
risk_score NUMERIC(5,2) CHECK (risk_score >= 0 AND risk_score <= 100),
risk_factors JSONB,  -- Structured risk diagnostics

-- Keep existing for backward compatibility:
-- trade_type TEXT (will be primary type)
-- risk_level TEXT (will be categorical label)
-- risk_assessment TEXT (will be generated narrative)
```

### Migration File
- Create: `migrations/003_trade_risk_enhancements.s### Phase 7: Narratives ✓
- [X] Implement risk_assessment generator
- [X] Add template-based narrative building
- [X] Handle NULL values gracefully
- [X] Integrate OpenAI for AI-enhanced narratives
- [X] Add graceful fallback to template when AI unavailable
- Add columns with proper indexes
- Update column comments

---

## Phase 2: Core Calculation Infrastructure

### 2.1 Z-Score Calculation Engine
**Location:** `backend/core/signals.py`

```python
class ZScoreCalculator:
    """
    Rolling window z-score standardization (60-day default, min 30).
    Cold start: fall back to universe statistics if ticker history < 20.
    """
    
    def __init__(self, lookback_days: int = 60, min_samples: int = 30):
        self.lookback_days = lookback_days
        self.min_samples = min_samples
    
    def calculate_z_score(
        self, 
        value: float, 
        ticker: str, 
        feature: str,
        historical_data: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate z-score using rolling window.
        Falls back to universe stats if insufficient history.
        """
        # Implementation here
```

**Features to standardize:**
- technical_score → technical_z
- fundamental_score → fundamental_z
- news_macro_score → news_z
- social_alternative_score → social_z
- risk_stability_score → risk_z
- institutional_smart_money_score → institutional_z

### 2.2 Trend Strength Calculator
**Location:** `backend/core/signals.py`

```python
def calculate_trend_strength(self, price_history: List[float], volume_history: List[float]) -> float:
    """
    TrendStrength = 0.5 * z(slope_50) + 0.3 * z(slope_200) + 0.2 * z(volume_trend)
    
    MA slope: OLS slope of log(price) over lookback L, annualized
    slope_L = 252 * slope(OLS(log(P_t) ~ t, last L))
    """
    # Calculate log-price slopes
    slope_50 = self._calculate_ma_slope(price_history, lookback=50)
    slope_200 = self._calculate_ma_slope(price_history, lookback=200)
    
    # Volume trend (20-day avg vs 60-day history)
    volume_trend = self._calculate_volume_trend(volume_history)
    
    # Z-scores and composite
    trend_strength = (
        0.5 * self.z_calc.calculate_z_score(slope_50, ticker, 'slope_50') +
        0.3 * self.z_calc.calculate_z_score(slope_200, ticker, 'slope_200') +
        0.2 * self.z_calc.calculate_z_score(volume_trend, ticker, 'volume_trend')
    )
    return trend_strength
```

### 2.3 Valuation Composite
**Location:** `backend/core/signals.py`

```python
def calculate_valuation_z(self, pe_ratio: float, pb_ratio: float, fcf_yield: float) -> float:
    """
    valuation_z = mean(z(P/E), z(P/B), z(FCF_yield) * (-1))
    Note: Higher FCF yield = cheaper, so multiply by -1
    """
    pe_z = self.z_calc.calculate_z_score(pe_ratio, ticker, 'pe_ratio')
    pb_z = self.z_calc.calculate_z_score(pb_ratio, ticker, 'pb_ratio')
    fcf_yield_z = self.z_calc.calculate_z_score(fcf_yield, ticker, 'fcf_yield') * -1
    
    return (pe_z + pb_z + fcf_yield_z) / 3
```

---

## Phase 3: Trade Type Classification

### 3.1 Trade Type Detector
**Location:** `backend/core/signals.py` (new class)

```python
class TradeTypeClassifier:
    """
    Assigns primary + secondary trade types based on z-score thresholds.
    Auto-appends Multi-Factor tag when applicable.
    """
    
    THRESHOLDS = {
        'Momentum': {
            'technical_z': 0.8,
            'trend_strength': 0.6
        },
        'Value': {
            'valuation_z': -0.6,  # Lower is better
            'fundamental_quality_z': 0.3
        },
        'Speculative Growth': {
            'revenue_growth_z': 0.8,
            'negative_fcf': True,
            'margin_tercile': 'bottom'
        },
        'Event-Driven': {
            'earnings_days': 7,
            'news_z': 0.7,
            'keyword_match': True
        },
        'Contrarian': {
            'rsi': 30,
            'price_sigma': -2.0,
            'social_news_z': -0.5,
            'fundamentals_trend_z': 0.2
        }
    }
    
    def classify(self, signal_data: Dict, z_scores: Dict) -> Tuple[str, Optional[str], List[str]]:
        """
        Returns: (primary_type, secondary_type, all_tags)
        
        Primary = highest z-score contribution to final signal
        Exception: Event-Driven wins if event flag active and News z ≥ 0.7
        
        Multi-Factor: auto-append if ≥3 components have z ≥ +0.5
        """
        candidate_scores = {}
        
        # Check each trade type
        if self._check_momentum(signal_data, z_scores):
            candidate_scores['Momentum'] = z_scores['technical_z']
        
        if self._check_value(signal_data, z_scores):
            candidate_scores['Value'] = z_scores['valuation_z']
        
        # ... other checks
        
        # Sort by contribution, take top 2
        sorted_types = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary = sorted_types[0][0] if sorted_types else 'Multi-Factor'
        secondary = sorted_types[1][0] if len(sorted_types) > 1 else None
        
        # Override for Event-Driven
        if self._has_event_flag(signal_data) and z_scores['news_z'] >= 0.7:
            if primary != 'Event-Driven':
                secondary = primary
                primary = 'Event-Driven'
        
        # Multi-Factor check
        tags = [primary]
        if secondary:
            tags.append(secondary)
        
        if self._is_multi_factor(z_scores):
            tags.append('Multi-Factor')
        
        return primary, secondary, tags
```

### 3.2 Event Detection
**Location:** `backend/integrations/news.py` (enhance existing)

```python
EVENT_KEYWORDS = {
    'M&A': ['merger', 'acquisition', 'takeover', 'go-private', 'buyout'],
    'Contract': ['contract', 'awarded', 'wins deal', 'option exercised', 'IDIQ'],
    'Product': ['launches', 'unveils', 'announces', 'FDA approval', 'FDA clearance']
}

def detect_event_flags(self, news_articles: List[Dict]) -> Dict[str, bool]:
    """
    Check for event keywords in last 5 days of news.
    Returns: {'has_ma': bool, 'has_contract': bool, 'has_product': bool}
    """
    # Case-insensitive keyword search with deduplication
```

---

## Phase 4: Risk Scoring System

### 4.1 Risk Score Calculator
**Location:** `backend/core/signals.py` (new class)

```python
class RiskScoreCalculator:
    """
    Composite risk score with worst-factor guard.
    Output: 0-100 scale mapped to categorical labels.
    """
    
    WEIGHTS = {
        'volatility': 0.40,      # ATR%, HV
        'liquidity': 0.25,       # ADV, float%
        'leverage': 0.15,        # D/E, interest coverage
        'short_interest': 0.10,
        'concentration': 0.10    # market cap, sector, theme
    }
    
    RISK_LEVELS = {
        'Low': (0, 25),
        'Moderate': (25, 45),
        'Elevated': (45, 65),
        'High': (65, 80),
        'Extreme': (80, 100)
    }
    
    def calculate_composite_risk(self, signal_data: Dict) -> Tuple[float, str, Dict]:
        """
        Returns: (risk_score, risk_level, risk_factors_json)
        
        Composite = weighted mean of subscores
        Guard: risk_score = max(composite, 0.9 * max_subfactor)
        """
        subscores = {
            'volatility': self._score_volatility(signal_data),
            'liquidity': self._score_liquidity(signal_data),
            'leverage': self._score_leverage(signal_data),
            'short_interest': self._score_short_interest(signal_data),
            'concentration': self._score_concentration(signal_data)
        }
        
        # Weighted composite
        composite = sum(subscores[k] * self.WEIGHTS[k] for k in subscores)
        
        # Worst-factor guard
        max_subfactor = max(subscores.values())
        risk_score = max(composite, 0.9 * max_subfactor)
        
        # Map to categorical
        risk_level = self._map_to_level(risk_score)
        
        # Generate JSON
        risk_factors = self._generate_risk_factors_json(subscores, signal_data)
        
        return risk_score, risk_level, risk_factors
```

### 4.2 Risk Subscores

**Volatility (40%):**
```python
def _score_volatility(self, data: Dict) -> float:
    """
    Uses ATR% (20-day default) and beta.
    Extreme if ATR% ≥ 10% or ≥ 90th percentile.
    Negative beta → use abs(), set inverse_beta flag.
    """
    atr_pct = data.get('atr_pct', 0)
    beta = data.get('beta', 1.0)
    
    # Normalize to 0-100
    atr_score = min(atr_pct * 10, 100)  # 10% ATR = 100 score
    beta_score = min(abs(beta) * 50, 100)  # Beta 2.0 = 100 score
    
    # Check percentile
    if atr_pct >= self._get_90th_percentile_atr():
        atr_score = max(atr_score, 80)
    
    return (atr_score * 0.6 + beta_score * 0.4)
```

**Liquidity (25%):**
```python
def _score_liquidity(self, data: Dict) -> float:
    """
    ADV (average daily volume) and float%.
    Low liquidity = high risk score.
    """
    adv = data.get('avg_daily_volume', 0)
    float_pct = data.get('float_pct', 50.0)
    
    # Inverse scoring (low volume = high risk)
    adv_score = 100 - min((adv / 1_000_000) * 10, 100)  # 10M vol = 0 risk
    float_score = 100 - (float_pct * 2)  # 50% float = 0 risk
    
    return (adv_score * 0.6 + float_score * 0.4)
```

**Leverage (15%):**
```python
def _score_leverage(self, data: Dict) -> float:
    """
    Debt/Equity and Interest Coverage.
    Thresholds:
      Interest Coverage < 1.5x → Extreme
      1.5-2.0x → High
      2.0-4.0x → Moderate
      > 4.0x → Low
    """
    de_ratio = data.get('debt_equity', 0)
    interest_coverage = data.get('interest_coverage', None)
    
    # D/E scoring
    if de_ratio > 3.0:
        de_score = 100
    elif de_ratio > 2.0:
        de_score = 75
    elif de_ratio > 1.0:
        de_score = 50
    else:
        de_score = 25
    
    # Interest coverage scoring
    if interest_coverage is None:
        ic_score = 50  # Default to moderate
    elif interest_coverage < 1.5:
        ic_score = 100
    elif interest_coverage < 2.0:
        ic_score = 80
    elif interest_coverage < 4.0:
        ic_score = 50
    else:
        ic_score = 20
    
    return (de_score * 0.5 + ic_score * 0.5)
```

**Short Interest (10%):**
```python
def _score_short_interest(self, data: Dict) -> float:
    """
    % of float shorted.
    High short interest = potential squeeze risk.
    """
    short_pct = data.get('short_pct_float', 0)
    
    if short_pct > 20:
        return 90
    elif short_pct > 15:
        return 70
    elif short_pct > 10:
        return 50
    elif short_pct > 5:
        return 30
    else:
        return 10
```

**Concentration (10%):**
```python
def _score_concentration(self, data: Dict) -> float:
    """
    Market cap tier + sector/theme concentration.
    Small cap + single theme = higher risk.
    """
    market_cap = data.get('market_cap', 0)
    theme = data.get('theme', None)
    
    # Market cap scoring
    if market_cap < 500_000_000:  # Micro cap
        cap_score = 90
    elif market_cap < 2_000_000_000:  # Small cap
        cap_score = 70
    elif market_cap < 10_000_000_000:  # Mid cap
        cap_score = 40
    else:  # Large cap
        cap_score = 10
    
    # Theme concentration (if AI/Biotech/single sector)
    theme_score = 50 if theme in ['AI', 'Biotech', 'Crypto'] else 20
    
    return (cap_score * 0.7 + theme_score * 0.3)
```

### 4.3 Risk Factors JSON Structure
```json
{
  "volatility": {
    "score": 82,
    "label": "High",
    "atr_pct": 11.2,
    "beta": 1.9,
    "inverse_beta": false
  },
  "liquidity": {
    "score": 70,
    "label": "Low",
    "adv": 1200000,
    "float_pct": 45.0
  },
  "leverage": {
    "score": 60,
    "label": "High",
    "debt_to_equity": 2.3,
    "interest_coverage": 1.1
  },
  "short_interest": {
    "score": 55,
    "label": "Elevated",
    "pct_float": 14.5
  },
  "concentration": {
    "score": 40,
    "label": "Moderate",
    "market_cap": "Small",
    "market_cap_value": 1200000000,
    "sector": "Technology",
    "theme": "AI"
  },
  "flags": {
    "inverse_beta": false,
    "event_week": true,
    "extreme_volatility": true
  }
}
```

---

## Phase 5: Data Collection Enhancements

### 5.1 Additional Metrics to Collect
**Location:** `backend/integrations/yfinance.py`

```python
def fetch_enhanced_metrics(self, ticker: str) -> Dict:
    """
    Collect additional data for risk calculations:
    - Interest coverage (EBIT / interest expense)
    - Float percentage (shares_float / shares_outstanding)
    - ATR% (20-day default)
    - Historical price data for MA slope calculations
    - Historical volume data for trend strength
    """
    stock = yf.Ticker(ticker)
    
    # Interest coverage from financials
    try:
        income_stmt = stock.income_stmt
        ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else None
        interest_expense = income_stmt.loc['Interest Expense'].iloc[0] if 'Interest Expense' in income_stmt.index else None
        
        if ebit and interest_expense:
            interest_coverage = ebit / abs(interest_expense)
        else:
            interest_coverage = None
    except:
        interest_coverage = None
    
    # Float percentage
    shares_float = stock.info.get('floatShares', 0)
    shares_outstanding = stock.info.get('sharesOutstanding', 1)
    float_pct = (shares_float / shares_outstanding) * 100 if shares_outstanding else None
    
    # ATR calculation (20-day)
    hist = stock.history(period='3mo')  # Need 60+ days for rolling calcs
    atr_pct = self._calculate_atr_pct(hist, period=20)
    
    return {
        'interest_coverage': interest_coverage,
        'float_pct': float_pct,
        'atr_pct': atr_pct,
        'price_history': hist['Close'].tolist(),
        'volume_history': hist['Volume'].tolist()
    }
```

### 5.2 Theme Detection Config
**Location:** `backend/core/config.py`

```python
THEME_MAPPINGS = {
    'AI': ['NVDA', 'PLTR', 'AI', 'IONQ', 'BBAI', 'SOUN', 'PATH'],
    'Biotech': ['MRNA', 'BNTX', 'GILD', 'BIIB', 'VRTX', 'REGN'],
    'Defense': ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'HII'],
    'Green Energy': ['ENPH', 'SEDG', 'NEE', 'FSLR', 'RUN'],
    'Crypto': ['COIN', 'RIOT', 'MARA', 'MSTR', 'SQ']
}

THEME_KEYWORDS = {
    'AI': ['artificial intelligence', 'machine learning', 'neural network', 'LLM', 'ChatGPT'],
    'Biotech': ['clinical trial', 'FDA approval', 'drug', 'therapy', 'biotech'],
    'Defense': ['defense contract', 'military', 'DoD', 'Navy', 'Army', 'Air Force'],
    'Green Energy': ['solar', 'renewable', 'clean energy', 'wind power', 'EV'],
    'Crypto': ['bitcoin', 'cryptocurrency', 'blockchain', 'crypto', 'digital asset']
}

def detect_theme(ticker: str, news_text: str = '', reddit_text: str = '') -> Optional[str]:
    """
    Detect theme from config mapping + keyword matching.
    """
    # Direct mapping
    for theme, tickers in THEME_MAPPINGS.items():
        if ticker in tickers:
            return theme
    
    # Keyword detection from news/social
    combined_text = (news_text + ' ' + reddit_text).lower()
    theme_scores = {}
    
    for theme, keywords in THEME_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in combined_text)
        if count > 0:
            theme_scores[theme] = count
    
    # Return highest scoring theme if confidence threshold met
    if theme_scores and max(theme_scores.values()) >= 2:
        return max(theme_scores.items(), key=lambda x: x[1])[0]
    
    return None
```

---

## Phase 6: Score Adjustment & Contrarian Feature

### 6.1 Dynamic Weight Adjustment
**Location:** `backend/core/signals.py` in `_calculate_signal_score_v2()`

```python
def _apply_trade_type_multipliers(self, base_weights: Dict, trade_tags: List[str]) -> Dict:
    """
    Apply multipliers based on trade type, cap at 35%, renormalize to sum 1.0.
    """
    weights = base_weights.copy()
    
    # Apply multipliers
    if 'Momentum' in trade_tags:
        weights['technical'] *= 1.15
    
    if 'Value' in trade_tags:
        weights['fundamental'] *= 1.15
    
    if 'Event-Driven' in trade_tags:
        weights['news_macro'] *= 1.25
    
    if 'Speculative Growth' in trade_tags:
        weights['social_alternative'] *= 1.10
        weights['news_macro'] *= 1.10
    
    # Cap any single component at 35%
    for key in weights:
        weights[key] = min(weights[key], 0.35)
    
    # Renormalize to sum to 1.0
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights
```

### 6.2 Contrarian Feature
**Location:** `backend/core/signals.py`

```python
def _calculate_contrarian_bonus(self, signal_data: Dict, z_scores: Dict) -> float:
    """
    contrarian_feature = max(0, -social_z) * 1.0 * indicator(oversold)
    
    Oversold: RSI ≤ 30 or price_z (20-day) ≤ -2.0
    Apply: +4% bonus to final signal_score
    """
    rsi = signal_data.get('rsi', 50)
    price_z = z_scores.get('price_z_20day', 0)
    social_z = z_scores.get('social_z', 0)
    
    # Check oversold condition
    oversold = (rsi <= 30) or (price_z <= -2.0)
    
    if not oversold:
        return 0.0
    
    # Contrarian feature (negative social sentiment is positive signal)
    if social_z < 0:
        contrarian_feature = abs(social_z)
    else:
        contrarian_feature = 0.0
    
    # Return as 4% bonus (applied to 0-1 scale)
    return contrarian_feature * 0.04
```

### 6.3 Updated Signal Score Calculation
```python
def _calculate_signal_score_v2(self, data: Dict, component_scores: Dict) -> float:
    """
    Enhanced Phase 7 calculation with trade type multipliers and contrarian bonus.
    """
    # Base weights (Phase 7)
    base_weights = {
        'technical': 0.25,
        'fundamental': 0.25,
        'news_macro': 0.20,
        'social_alternative': 0.15,
        'risk_stability': 0.10,
        'institutional': 0.05
    }
    
    # Calculate z-scores
    z_scores = self._calculate_z_scores(data, component_scores)
    
    # Classify trade type
    trade_classifier = TradeTypeClassifier()
    primary, secondary, trade_tags = trade_classifier.classify(data, z_scores)
    
    # Apply trade type multipliers
    adjusted_weights = self._apply_trade_type_multipliers(base_weights, trade_tags)
    
    # Calculate base signal score
    signal_score = sum(
        component_scores[key] * adjusted_weights[key.replace('_score', '')]
        for key in component_scores
    )
    
    # Add contrarian bonus
    contrarian_bonus = self._calculate_contrarian_bonus(data, z_scores)
    signal_score += contrarian_bonus
    
    # Clamp to [0, 1]
    signal_score = max(0.0, min(1.0, signal_score))
    
    return signal_score
```

---

## Phase 7: Narrative Generation

### 7.1 Risk Assessment Generator
**Location:** `backend/core/signals.py`

```python
def generate_risk_assessment_narrative(self, risk_factors: Dict) -> str:
    """
    Generate human-readable risk assessment from structured risk_factors JSON.
    Template-based initially, can enhance with AI later.
    """
    narrative_parts = []
    
    # Volatility
    vol = risk_factors['volatility']
    if vol['score'] >= 80:
        narrative_parts.append(
            f"⚠️ Extreme volatility: {vol['atr_pct']:.1f}% ATR, "
            f"Beta {vol['beta']:.2f} (above 90th percentile)"
        )
    elif vol['score'] >= 65:
        narrative_parts.append(
            f"High volatility: {vol['atr_pct']:.1f}% ATR, Beta {vol['beta']:.2f}"
        )
    
    if vol.get('inverse_beta'):
        narrative_parts.append("📊 Inverse beta: potential market hedge")
    
    # Liquidity
    liq = risk_factors['liquidity']
    if liq['score'] >= 70:
        narrative_parts.append(
            f"⚠️ Low liquidity: {liq['adv']:,.0f} avg daily volume, "
            f"{liq['float_pct']:.1f}% float"
        )
    
    # Leverage
    lev = risk_factors['leverage']
    if lev['score'] >= 80:
        cov = lev.get('interest_coverage', 'N/A')
        narrative_parts.append(
            f"⚠️ High leverage: D/E {lev['debt_to_equity']:.1f}, "
            f"Interest coverage {cov:.1f}x" if cov != 'N/A' else f"D/E {lev['debt_to_equity']:.1f}"
        )
    
    # Short Interest
    short = risk_factors['short_interest']
    if short['score'] >= 70:
        narrative_parts.append(
            f"📈 Elevated short interest: {short['pct_float']:.1f}% of float (squeeze potential)"
        )
    
    # Concentration
    conc = risk_factors['concentration']
    narrative_parts.append(
        f"Market cap: {conc['market_cap']} ({conc['market_cap_value'] / 1e9:.1f}B)"
    )
    
    if conc.get('theme'):
        narrative_parts.append(f"Theme: {conc['theme']}")
    
    # Flags
    flags = risk_factors.get('flags', {})
    if flags.get('event_week'):
        narrative_parts.append("📅 Event-driven: Catalyst within 7 days")
    
    return " | ".join(narrative_parts)
```

---

## Phase 8: Backtesting Integration

### 8.1 Enhanced Backtest Configuration
**Location:** `backend/core/backtest.py`

```python
BACKTEST_CONFIG = {
    'entry_thresholds': {
        'Low': 0.55,
        'Moderate': 0.60,
        'Elevated': 0.65,
        'High': 0.70,
        'Extreme': 0.75
    },
    'hold_periods': {
        'Momentum': (5, 20),
        'Event-Driven': (3, 15),
        'Value': (60, 180),
        'Contrarian': (60, 180),
        'Speculative Growth': (20, 90),
        'Multi-Factor': (30, 60)
    },
    'stop_loss_multipliers': {  # ATR multipliers
        'Low': 1.5,
        'Moderate': 1.8,
        'Elevated': 2.0,
        'High': 2.5,
        'Extreme': 3.0
    },
    'take_profit_multipliers': {  # ATR multipliers
        'Low': 2.5,
        'Moderate': 3.0,
        'Elevated': 3.0,
        'High': 3.5,
        'Extreme': 4.0
    }
}
```

### 8.2 Trade Type Performance Tracking
```python
class BacktestTracker:
    """Track performance by trade_type and risk_level."""
    
    def __init__(self):
        self.results_by_type = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_return': 0.0,
            'avg_hold_days': 0, 'trades': []
        })
        
        self.results_by_risk = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_return': 0.0,
            'avg_hold_days': 0, 'trades': []
        })
    
    def record_trade(self, trade_result: Dict):
        """Record trade outcome by type and risk level."""
        trade_type = trade_result['primary_trade_type']
        risk_level = trade_result['risk_level']
        
        # Update type stats
        self.results_by_type[trade_type]['trades'].append(trade_result)
        if trade_result['profit'] > 0:
            self.results_by_type[trade_type]['wins'] += 1
        else:
            self.results_by_type[trade_type]['losses'] += 1
        
        # Update risk stats
        self.results_by_risk[risk_level]['trades'].append(trade_result)
        if trade_result['profit'] > 0:
            self.results_by_risk[risk_level]['wins'] += 1
        else:
            self.results_by_risk[risk_level]['losses'] += 1
    
    def generate_report(self) -> Dict:
        """Generate performance report by trade type and risk level."""
        return {
            'by_trade_type': self._calculate_stats(self.results_by_type),
            'by_risk_level': self._calculate_stats(self.results_by_risk)
        }
```

---

## Phase 9: Testing & Validation

### 9.1 Unit Tests
```python
# tests/test_trade_classification.py
def test_momentum_classification():
    """Test momentum trade type detection."""
    signal_data = {
        'technical_score': 0.85,
        'trend_strength': 0.72,
        'rsi': 65,
        'momentum_30d_pct': 12.5
    }
    # Assert classified as Momentum

def test_multi_factor_tag():
    """Test Multi-Factor tag assignment."""
    z_scores = {
        'technical_z': 0.6,
        'fundamental_z': 0.7,
        'news_z': 0.5,
        'social_z': 0.8
    }
    # Assert Multi-Factor tag added

def test_risk_score_composite():
    """Test composite risk score calculation."""
    # Test with various risk profiles

def test_contrarian_bonus():
    """Test contrarian bonus calculation."""
    # Test oversold + negative sentiment
```

### 9.2 Integration Tests
```python
# tests/test_enhanced_pipeline.py
def test_full_pipeline_with_enhancements():
    """Test complete pipeline with trade type and risk enhancements."""
    pipeline = UnifiedPipeline()
    results = pipeline.run()
    
    # Verify all signals have:
    assert all('trade_tags' in r for r in results)
    assert all('risk_score' in r for r in results)
    assert all('risk_factors' in r for r in results)
    assert all(0 <= r['risk_score'] <= 100 for r in results)
```

---

## Implementation Checklist

### Phase 1: Schema ✓
- [X] Create migration file
- [X] Add trade_tags column
- [X] Add risk_score column
- [X] Add risk_factors JSONB column
- [X] Add indexes
- [X] Run migration on Supabase

### Phase 2: Core Infrastructure ✓
- [X] Implement ZScoreCalculator class
- [X] Implement trend_strength calculation
- [X] Implement valuation_z calculation
- [X] Add historical data fetching for rolling windows

### Phase 3: Trade Classification ✓
- [X] Implement TradeTypeClassifier
- [X] Add event detection (keywords + earnings date)
- [X] Add Multi-Factor auto-tagging
- [X] Add theme detection config

### Phase 4: Risk Scoring ✓
- [X] Implement RiskScoreCalculator
- [X] Implement volatility subscore
- [X] Implement liquidity subscore
- [X] Implement leverage subscore
- [X] Implement short interest subscore
- [X] Implement concentration subscore
- [X] Add worst-factor guard logic
- [X] Generate risk_factors JSON

### Phase 5: Data Collection ✓
- [X] Add interest coverage fetching (yfinance)
- [X] Add float percentage calculation
- [X] Add ATR% calculation (20-day)
- [X] Add historical price/volume for MA slopes
- [X] Add theme detection logic

### Phase 6: Score Adjustments ✓
- [X] Implement weight multipliers by trade type
- [X] Add renormalization logic
- [X] Add 35% cap enforcement
- [X] Implement contrarian bonus
- [X] Update _calculate_signal_score_v2()

### Phase 7: Narratives ✓
- [X] Implement risk_assessment generator
- [X] Add template-based narrative building
- [X] Handle NULL values gracefully

### Phase 8: Backtesting ✓
- [X] Add entry thresholds by risk level
- [X] Add hold period configs by trade type
- [X] Add stop/take profit configs
- [X] Implement BacktestTracker class
- [X] Add performance reporting by type/risk

### Phase 9: Testing ✓
- [X] Write unit tests for classification
- [X] Write unit tests for risk scoring
- [X] Write integration tests
- [X] Run full pipeline test
- [X] Validate data quality

### Phase 10: Documentation ✓
- [ ] Update operational_guidelines.md
- [ ] Update docs/recommendations.md
- [ ] Add inline code documentation
- [ ] Create example usage guide

---

## Expected Outcomes

### Database
- New columns populated with structured data
- Backward compatible (old fields still exist)
- Rich JSON diagnostics in risk_factors

### Signals
- Each signal has primary + optional secondary trade type
- Multi-Factor tag auto-applied when ≥3 components strong
- Risk score 0-100 with categorical label
- Detailed risk assessment narrative

### Scoring
- Dynamic weight adjustments based on trade type
- Contrarian bonus for oversold opportunities
- Z-score based classification (regime-aware)

### Backtesting
- Performance tracked by trade type
- Performance tracked by risk level
- Optimized entry/exit rules per category
- Separate hold periods and stops

---

## Timeline Estimate
- Phase 1 (Schema): 30 minutes
- Phase 2 (Infrastructure): 2 hours
- Phase 3 (Trade Classification): 2 hours
- Phase 4 (Risk Scoring): 2 hours
- Phase 5 (Data Collection): 1.5 hours
- Phase 6 (Score Adjustments): 1 hour
- Phase 7 (Narratives): 1 hour
- Phase 8 (Backtesting): 1.5 hours
- Phase 9 (Testing): 1.5 hours
- Phase 10 (Documentation): 1 hour

**Total: ~14 hours of development**

---

## Questions Resolved
All 10 clarification questions answered. Ready to proceed with implementation.

**Next Step:** Begin Phase 1 - Schema migration.
