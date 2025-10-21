# Phase 4: 3.0 Signal Groups Update

**Date**: January 2025  
**Status**: ✅ COMPLETE

## Summary

Updated Phase 4 (Assemble Scores) to align with the 3.0 signal group taxonomy. Removed old signal groups and replaced them with the standardized 6-group structure.

---

## Changes Made

### 1. Signal Groups Updated

**OLD Groups (Removed):**
```
❌ reddit
❌ sentiment  
❌ ai
✓ technical (kept)
✓ fundamental (kept)
```

**NEW 3.0 Groups (Implemented):**
```
✅ Technical
✅ Fundamental
✅ News/Macro
✅ Social/Alternative
✅ Risk/Stability
✅ Institutional/Smart Money
```

---

## Detailed Changes

### A. Module Documentation
```python
# OLD:
- Combining Reddit, Technical, Fundamental, Sentiment, AI scores

# NEW:
- Combining 3.0 signal groups: Technical, Fundamental, News/Macro, 
  Social/Alternative, Risk/Stability, Institutional/Smart Money
```

### B. Group Score Extraction
```python
# OLD (5 groups):
reddit_scores = phase3_scores.get('reddit_scores', [])
technical_scores = phase3_scores.get('technical_scores', [])
fundamental_scores = phase3_scores.get('fundamental_scores', [])
sentiment_scores = phase3_scores.get('sentiment_scores', [])
ai_scores = phase3_scores.get('ai_scores', [])

# NEW (6 groups):
technical_scores = phase3_scores.get('technical_scores', [])
fundamental_scores = phase3_scores.get('fundamental_scores', [])
news_macro_scores = phase3_scores.get('news_macro_scores', [])
social_alternative_scores = phase3_scores.get('social_alternative_scores', [])
risk_stability_scores = phase3_scores.get('risk_stability_scores', [])
institutional_smart_money_scores = phase3_scores.get('institutional_smart_money_scores', [])
```

### C. Scoring Weights
```python
# OLD Environment Variables:
SCORE_WEIGHT_REDDIT = 0.1
SCORE_WEIGHT_TECHNICAL = 0.2
SCORE_WEIGHT_FUNDAMENTAL = 0.4
SCORE_WEIGHT_SENTIMENT = 0.15
SCORE_WEIGHT_AI = 0.15

# NEW Environment Variables (3.0):
SCORE_WEIGHT_TECHNICAL = 0.20
SCORE_WEIGHT_FUNDAMENTAL = 0.25
SCORE_WEIGHT_NEWS_MACRO = 0.15
SCORE_WEIGHT_SOCIAL_ALTERNATIVE = 0.10
SCORE_WEIGHT_RISK_STABILITY = 0.15
SCORE_WEIGHT_INSTITUTIONAL_SMART_MONEY = 0.15
```

**Default Weight Distribution:**
- Technical: 20%
- Fundamental: 25%
- News/Macro: 15%
- Social/Alternative: 10%
- Risk/Stability: 15%
- Institutional/Smart Money: 15%
- **Total: 100%**

### D. Ticker Signal Indexing
```python
# OLD Structure (5 groups):
ticker_signals[ticker] = {
    'reddit': None,
    'technical': None,
    'fundamental': None,
    'sentiment': None,
    'ai': None
}

# NEW Structure (6 groups):
ticker_signals[ticker] = {
    'technical': None,
    'fundamental': None,
    'news_macro': None,
    'social_alternative': None,
    'risk_stability': None,
    'institutional_smart_money': None
}
```

### E. Score Assembly
```python
# OLD (5 scores):
reddit_score = group_scores['reddit'].get('score', 0.0)
technical_score = group_scores['technical'].get('score', 0.0)
fundamental_score = group_scores['fundamental'].get('score', 0.0)
sentiment_score = group_scores['sentiment'].get('score', 0.0)
ai_score = group_scores['ai'].get('score', 0.0)

signal_score = (
    reddit_score * weights['reddit'] +
    technical_score * weights['technical'] +
    fundamental_score * weights['fundamental'] +
    sentiment_score * weights['sentiment'] +
    ai_score * weights['ai']
)

# NEW (6 scores):
technical_score = group_scores['technical'].get('score', 0.0)
fundamental_score = group_scores['fundamental'].get('score', 0.0)
news_macro_score = group_scores['news_macro'].get('score', 0.0)
social_alternative_score = group_scores['social_alternative'].get('score', 0.0)
risk_stability_score = group_scores['risk_stability'].get('score', 0.0)
institutional_smart_money_score = group_scores['institutional_smart_money'].get('score', 0.0)

signal_score = (
    technical_score * weights['technical'] +
    fundamental_score * weights['fundamental'] +
    news_macro_score * weights['news_macro'] +
    social_alternative_score * weights['social_alternative'] +
    risk_stability_score * weights['risk_stability'] +
    institutional_smart_money_score * weights['institutional_smart_money']
)
```

### F. Confidence Calculation
```python
# OLD:
confidence = active_scores / 5.0  # 5 groups total

# NEW:
confidence = active_scores / 6.0  # 6 groups total in 3.0
```

### G. Final Signal Output
```python
# OLD Structure:
{
    'ticker': 'AAPL',
    'signal_score': 0.75,
    'reddit_score': 0.6,
    'technical_score': 0.8,
    'fundamental_score': 0.7,
    'sentiment_score': 0.65,
    'ai_score': 0.85,
    'reddit_data': {...},
    'technical_data': {...},
    'fundamental_data': {...},
    'sentiment_data': {...},
    'ai_data': {...},
}

# NEW Structure (3.0):
{
    'ticker': 'AAPL',
    'signal_score': 0.75,
    'technical_score': 0.8,
    'fundamental_score': 0.7,
    'news_macro_score': 0.65,
    'social_alternative_score': 0.6,
    'risk_stability_score': 0.75,
    'institutional_smart_money_score': 0.85,
    'technical_data': {...},
    'fundamental_data': {...},
    'news_macro_data': {...},
    'social_alternative_data': {...},
    'risk_stability_data': {...},
    'institutional_smart_money_data': {...},
}
```

---

## Validation

✅ **Syntax Check**: Passed `python -m py_compile`  
✅ **Group Count**: 6 groups (was 5)  
✅ **Weight Distribution**: Sums to 100%  
✅ **Confidence Calculation**: Updated to 6 groups  
✅ **Environment Variables**: New naming convention  
✅ **Data Structure**: All references updated

---

## Alignment with 3.0 Standards

This update ensures Phase 4 aligns with:
- **backend/core/signals.py** (SignalScorer uses same 6 groups)
- **3.0 Signal Group Taxonomy** (canonical group names)
- **Phase 3 Output** (expects these 6 group scores)
- **Database Schema** (stores these 6 group scores)

---

## Environment Variable Migration

**To Update Your .env File:**

```bash
# Remove OLD variables:
# SCORE_WEIGHT_REDDIT=0.1
# SCORE_WEIGHT_SENTIMENT=0.15
# SCORE_WEIGHT_AI=0.15

# Add NEW variables (3.0):
SCORE_WEIGHT_TECHNICAL=0.20
SCORE_WEIGHT_FUNDAMENTAL=0.25
SCORE_WEIGHT_NEWS_MACRO=0.15
SCORE_WEIGHT_SOCIAL_ALTERNATIVE=0.10
SCORE_WEIGHT_RISK_STABILITY=0.15
SCORE_WEIGHT_INSTITUTIONAL_SMART_MONEY=0.15
```

---

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Signal Groups** | 5 (reddit, technical, fundamental, sentiment, ai) | 6 (technical, fundamental, news_macro, social_alternative, risk_stability, institutional_smart_money) |
| **Code Lines Changed** | N/A | ~150 lines |
| **Environment Variables** | 5 weights | 6 weights |
| **Confidence Denominator** | 5.0 | 6.0 |
| **3.0 Compliance** | ❌ No | ✅ Yes |

---

## Next Steps

1. ✅ Phase 4 updated to 3.0 groups
2. ⏳ Proceed with pipeline.py refactoring
3. ⏳ Update Phase 1, 2, 5 if they reference old groups
4. ⏳ Update .env file with new weight variables
5. ⏳ Test full pipeline with 3.0 groups

---

## Questions Addressed

**Q: Are all scores using 3.0 signal groups?**  
A: Yes, Phase 4 now uses the 6 canonical 3.0 groups.

**Q: What if Phase 3 (SignalScorer) uses different names?**  
A: Phase 3 already uses these exact names (verified in signals.py lines 256-258).

**Q: Will old database records still work?**  
A: New records will use 3.0 groups. Old records remain unchanged (backward compatible).

---

**Status**: ✅ READY FOR PIPELINE.PY REFACTORING
