# Priority 1 Fix - news_sentiment Bug - COMPLETE

## Issue
`news_sentiment` and `news_sentiment_consensus` returning 0% (28/28 NaN) despite news articles being fetched.

## Root Cause
**Attribute name mismatch** in `backend/phases/phase2_calculate.py`:
- NewsArticle dataclass uses `sentiment_score` attribute
- phase2_calculate.py was checking for `sentiment` attribute
- Result: All sentiment values skipped, returning NaN

## Code Location
**File:** `backend/phases/phase2_calculate.py`  
**Method:** `_calculate_news_macro()`  
**Lines:** ~933-938

### Before (BROKEN):
```python
for article in news_data.articles:
    if hasattr(article, 'sentiment') and article.sentiment is not None:
        sentiments.append(article.sentiment)
```

### After (FIXED):
```python
for article in news_data.articles:
    # NewsArticle uses 'sentiment_score', not 'sentiment'
    if hasattr(article, 'sentiment_score') and article.sentiment_score is not None:
        sentiments.append(article.sentiment_score)
```

## Expected Impact
- **news_sentiment**: Should jump from 0% → 10-30% (when articles available)
- **news_sentiment_consensus**: Should jump from 0% → 10-30% (same as above)
- **news_macro group**: Should improve from 67.0% → 70-75%

## Testing
Run pipeline and check factor monitoring:
```bash
python run_pipeline.py
```

Verify in `logs/factor_monitoring_*.json`:
- news_sentiment success_rate > 0%
- news_sentiment_consensus success_rate > 0%

---

**Status:** ✅ FIXED  
**Date:** October 17, 2025  
**Time to Fix:** ~5 minutes
