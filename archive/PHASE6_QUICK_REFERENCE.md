# Phase 6 Quick Reference

## What Was Implemented

**Phase 6: Narrative Generation**

✅ Added `generate_risk_narrative()` method to RiskScoreCalculator  
✅ Transforms structured risk_factors (JSONB) → human-readable text  
✅ Integrated into score_ticker() method  
✅ Added risk_assessment field to SignalResult  
✅ All tests passing

---

## Key Method

### generate_risk_narrative()

**Location:** `backend/core/signals.py` line ~1317

**Purpose:** Convert numeric risk subscores into clear narrative

**Input:**
```python
risk_score = 52.0  # 0-100
risk_level = 'Moderate'  # Low/Moderate/Elevated/High/Extreme
risk_factors = {
    'volatility_subscore': 45.2,
    'liquidity_subscore': 78.5,
    'leverage_subscore': 12.3,
    'short_interest_subscore': 30.1,
    'concentration_subscore': 55.0,
    'worst_factor': 'liquidity',
    'max_subscore': 78.5
}
theme = 'Tech Rally'
```

**Output:**
```
MODERATE RISK (52.0/100): Primary concern is liquidity (78.5), 
indicating potential exit challenges. Leverage is reasonable (12.3). 
Short interest is low (30.1). Aligns with Tech Rally theme. 
Suitable for medium-risk tolerance portfolios.
```

---

## Narrative Components

### 1. Risk Header
- Format: `"{LEVEL} RISK ({score}/100): "`
- Always included

### 2. Primary Concern
- Highlights worst_factor with description
- Examples:
  - "volatility (85.0), indicating high price fluctuation"
  - "liquidity (78.5), indicating potential exit challenges"

### 3. Secondary Concerns (if subscore > 60)
- "Volatility is elevated (X)"
- "Liquidity risk is high (X)"
- "Leverage is concerning (X)"

### 4. Positive Notes (if subscore < 40)
- "Volatility is manageable (X)"
- "Liquidity is adequate (X)"
- "Leverage is reasonable (X)"

### 5. Theme Context (if available)
- "Aligns with {theme} theme."

### 6. Suitability Recommendation
- Low: "Suitable for conservative portfolios"
- Moderate: "Suitable for medium-risk tolerance portfolios"
- High: "Suitable for aggressive portfolios only"
- Extreme: "Extreme risk - only for high-risk speculators"

---

## Integration

### In score_ticker()

```python
# Phase 5: Calculate risk score
risk_score, risk_level, risk_factors = self.risk_calc.calculate_risk_score(...)

# Phase 6: Generate narrative
risk_assessment = self.risk_calc.generate_risk_narrative(
    risk_score, risk_level, risk_factors, theme
)

# Add to SignalResult
return SignalResult(
    ...
    risk_assessment=risk_assessment,  # ← New field
    ...
)
```

---

## Example Narratives

### Moderate Risk
```
MODERATE RISK (52.0/100): Primary concern is liquidity (78.5), 
indicating potential exit challenges. Leverage is reasonable (12.3). 
Suitable for medium-risk tolerance portfolios.
```

### High Risk
```
HIGH RISK (75.0/100): Primary concern is volatility (85.0), 
indicating high price fluctuation. Leverage is concerning (72.0). 
Short interest is elevated (65.0). Suitable for aggressive 
portfolios only.
```

### Low Risk
```
LOW RISK (20.0/100): Primary concern is leverage (25.0). 
Volatility is manageable (18.0). Liquidity is adequate (15.0). 
Suitable for conservative portfolios.
```

### Extreme Risk
```
EXTREME RISK (92.0/100): Primary concern is concentration (95.0), 
suggesting sector/asset over-exposure. Volatility is elevated (92.0). 
Liquidity risk is high (88.0). Extreme risk - only for high-risk 
speculators.
```

---

## Verification

```bash
# Run test
python test_phase6.py

# Result
✅ PHASE 6 VERIFICATION COMPLETE - ALL TESTS PASSED

# Checks
✅ Method exists and works
✅ Risk level header included
✅ Primary concern highlighted
✅ Secondary concerns (>60) mentioned
✅ Positive notes (<40) mentioned
✅ Theme context integrated
✅ Suitability recommendations provided
✅ Narratives are 50-500 characters
```

---

## Database Storage

**Field:** `risk_assessment` (TEXT)

**Example:**
```sql
SELECT ticker, risk_level, risk_assessment 
FROM signals 
WHERE risk_level = 'High';

-- Result:
-- AAPL | High | HIGH RISK (75.0/100): Primary concern is...
```

---

## Files Modified

- `backend/core/signals.py` - ~115 lines (generate_risk_narrative method)
- `backend/core/signals.py` - ~10 lines (integration in score_ticker)
- `backend/core/signals.py` - ~3 lines (risk_assessment field)
- `test_phase6.py` - 210 lines (verification tests)

**Total:** ~338 lines

---

## Progress

**Completed:** 6 of 10 phases (60%)  
**Code Added (Phases 2-6):** 2,658+ lines

**Next:** Phase 7 - Backtesting Integration  
(Entry thresholds, hold periods, position sizing)

Ready for Phase 7! 🚀
