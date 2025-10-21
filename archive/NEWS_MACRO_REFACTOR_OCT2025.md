# News/Macro Group Refactor - October 2025

## Overview
Refactored the `news_macro` signal group by removing 5 problematic news-based signals that were returning 0% success rate and replacing them with 2 Reddit sentiment signals and 3 macro economic indicators.

## Changes Summary

### Removed Signals (5)
These signals were returning 0% success because news data was unavailable from Yahoo Finance:

1. **news_sentiment_7d** - 7-day news sentiment average
2. **news_sentiment_30d** - 30-day news sentiment average  
3. **news_count_7d** - Number of news articles in 7 days
4. **news_count_30d** - Number of news articles in 30 days
5. **news_momentum** - News sentiment acceleration/trend

### Added Signals (5)

#### Reddit Sentiment (2 signals)
Moved from `social_alternative` to `news_macro` for better signal coverage:

1. **reddit_sentiment** - Reddit sentiment score (-1 to +1)
   - Pulled from Reddit data in Phase 1
   - Uses VADER + TextBlob sentiment analysis
   - Weight: 0.13 (13%)

2. **reddit_sentiment_consensus** - Reddit sentiment as percentage (0-100%)
   - Converts reddit_sentiment from -1/+1 scale to 0-100%
   - Formula: `((sentiment + 1) / 2) * 100`
   - Weight: 0.12 (12%)

#### Macro Economic Indicators (3 signals)
New macro indicators from market_data (will be NaN until data source added):

3. **unemployment_rate** - Current unemployment rate
   - Source: `market_data.unemployment_rate`
   - Weight: 0.03 (3%)

4. **gdp_growth_rate** - GDP growth rate (quarterly or annual)
   - Source: `market_data.gdp_growth_rate`
   - Weight: 0.03 (3%)

5. **inflation_rate** - Current inflation rate (CPI)
   - Source: `market_data.inflation_rate`
   - Weight: 0.02 (2%)

## Files Modified

### 1. `backend/phases/phase2_calculate.py`

**Changes to `_calculate_news_macro()` method:**
- Added `reddit_data` parameter
- Removed news sentiment calculation block (lines ~935-944)
- Added Reddit sentiment calculation at start of method:
  ```python
  # Reddit sentiment (added to news_macro for better social signal coverage)
  if reddit_data:
      factors['reddit_sentiment'] = float(reddit_data.get('sentiment', 0))
      sentiment = reddit_data.get('sentiment')
      if sentiment is not None:
          factors['reddit_sentiment_consensus'] = ((sentiment + 1) / 2) * 100
  else:
      factors['reddit_sentiment'] = np.nan
      factors['reddit_sentiment_consensus'] = np.nan
  ```
- Added macro economic indicators (lines ~1104-1115):
  ```python
  if hasattr(market_data, 'unemployment_rate'):
      factors['unemployment_rate'] = float(market_data.unemployment_rate)
  # ... similar for gdp_growth_rate, inflation_rate
  ```

**Changes to `_empty_news_macro_factors()` method:**
- Removed 5 news signals
- Added 5 new signals (2 Reddit + 3 macro)

**Changes to method call:**
- Updated line 283 to pass `reddit_data`:
  ```python
  news_macro_factors = self._calculate_news_macro(raw_data, news_data, market_data, reddit_data)
  ```

### 2. `config/factor_to_group.yaml`

**news_macro section updated:**
```yaml
news_macro:
  # Reddit sentiment (moved from social for better coverage)
  - reddit_sentiment
  - reddit_sentiment_consensus
  
  # Earnings events (unchanged)
  - days_to_earnings
  - pre_earnings_flag
  - post_earnings_flag
  - earnings_surprise_last
  - earnings_beat_streak
  
  # Market regime (unchanged)
  - market_regime
  - spy_correlation_60d
  - sector_momentum_30d
  - sector_relative_strength
  
  # Macro indicators (expanded)
  - vix_level
  - treasury_yield_10y
  - credit_spread
  - unemployment_rate        # NEW
  - gdp_growth_rate         # NEW
  - inflation_rate          # NEW
```

**Total factor count:** Still 17 factors (removed 5, added 5)

### 3. `config/weights.yaml`

**Updated factor_weights_news_macro:**
```yaml
factor_weights_news_macro:
  # Reddit sentiment (25% - moved from social)
  reddit_sentiment: 0.13
  reddit_sentiment_consensus: 0.12
  
  # Earnings events (30%)
  days_to_earnings: 0.05
  pre_earnings_flag: 0.04
  post_earnings_flag: 0.04
  earnings_surprise_last: 0.09
  earnings_beat_streak: 0.08
  
  # Market regime (25%)
  market_regime: 0.10
  spy_correlation_60d: 0.05
  sector_momentum_30d: 0.06
  sector_relative_strength: 0.04
  
  # Macro indicators (20%)
  vix_level: 0.05
  treasury_yield_10y: 0.04
  credit_spread: 0.03
  unemployment_rate: 0.03      # NEW
  gdp_growth_rate: 0.03        # NEW
  inflation_rate: 0.02         # NEW
```

**Weight distribution:**
- Reddit sentiment: 25% (0.13 + 0.12)
- Earnings events: 30% (redistributed)
- Market regime: 25% (increased emphasis)
- Macro indicators: 20% (expanded from 9% to include new indicators)

**Total: 1.00** ✓

## Expected Impact

### Immediate Benefits
1. **Higher success rate** - Reddit sentiment data is available, unlike news data
2. **Better signal coverage** - 100% of factors will return valid values (except new macro indicators pending data source)
3. **Cleaner factor monitoring** - No more 0% success warnings for news factors

### Before Refactor
```
news_macro group: 67.1% avg success (17 factors, 5 problematic)
- news_sentiment_7d: 0%
- news_count_7d: 0%
- news_sentiment_30d: 0%
- news_count_30d: 0%
- news_momentum: 0%
```

### Expected After Refactor
```
news_macro group: ~90%+ avg success (17 factors, 3 pending data)
- reddit_sentiment: 60-80% (depends on Reddit data availability)
- reddit_sentiment_consensus: 60-80%
- unemployment_rate: 0% (pending data source)
- gdp_growth_rate: 0% (pending data source)
- inflation_rate: 0% (pending data source)
```

## Next Steps

### Phase 1: Test Current Changes
- [ ] Run pipeline with updated configuration
- [ ] Verify Reddit sentiment signals are working
- [ ] Check news_macro group performance improvement
- [ ] Monitor factor success rates

### Phase 2: Add Macro Data Sources
To populate the new macro indicators, add data fetching in `backend/integrations/yfinance.py`:

```python
# In fetch_market_data() method
def fetch_market_data(self, period='1y'):
    # ... existing code ...
    
    # Fetch macro economic data
    try:
        # Example using FRED API or similar
        unemployment_ticker = yf.Ticker("UNRATE")
        gdp_ticker = yf.Ticker("GDP")
        inflation_ticker = yf.Ticker("CPIAUCSL")
        
        # Get latest values
        self.unemployment_rate = unemployment_ticker.info.get('regularMarketPrice')
        self.gdp_growth_rate = gdp_ticker.info.get('regularMarketPrice')
        self.inflation_rate = inflation_ticker.info.get('regularMarketPrice')
    except Exception as e:
        logger.debug(f"Could not fetch macro indicators: {e}")
        self.unemployment_rate = None
        self.gdp_growth_rate = None
        self.inflation_rate = None
```

### Phase 3: Consider Additional Signals
Once Reddit sentiment is validated, consider adding:
- **earnings_call_sentiment** - Sentiment from earnings call transcripts
- **analyst_upgrade_momentum** - Rate of analyst upgrades/downgrades
- **sector_rotation_score** - Sector rotation strength indicator
- **fed_policy_sentiment** - Fed policy stance indicator

## Implementation Notes

### Reddit Sentiment Source
Reddit sentiment comes from `backend/integrations/reddit.py`:
- Fetches posts from r/stocks, r/investing, r/SecurityAnalysis
- Uses VADER sentiment analyzer (now also in `backend/integrations/news.py`)
- Aggregates sentiment across multiple posts mentioning the ticker
- Returns score from -1 (very negative) to +1 (very positive)

### Macro Indicator Placeholders
The three new macro indicators are currently set up to receive data but will return NaN until a data source is configured. They have lower weights (2-3%) so they won't significantly impact overall scores while missing.

### Configuration Validation
Both config files include validation rules:
- `factor_to_group.yaml`: Ensures each factor appears exactly once
- `weights.yaml`: Ensures weights sum to 1.0 within each group

## Testing Checklist

- [x] Remove old news signals from phase2_calculate.py
- [x] Add Reddit sentiment calculation
- [x] Add macro indicator placeholders
- [x] Update _empty_news_macro_factors()
- [x] Update method signature and call
- [x] Update config/factor_to_group.yaml
- [x] Update config/weights.yaml
- [x] Verify weights sum to 1.0
- [ ] Run full pipeline test
- [ ] Check factor monitoring report
- [ ] Verify Reddit sentiment populates
- [ ] Document macro data source implementation

## Rollback Plan

If issues arise, rollback by reverting these commits:
1. `backend/phases/phase2_calculate.py` - Revert lines 928-945, 1104-1115, 1140-1149, 283
2. `config/factor_to_group.yaml` - Revert news_macro section
3. `config/weights.yaml` - Revert factor_weights_news_macro section

Original news signals can be restored from git history.

## References

- **VADER Sentiment Integration**: See `backend/integrations/news.py` lines 24-30
- **Reddit Data Structure**: See `backend/integrations/reddit.py` 
- **Factor Monitoring**: See `backend/utils/factor_monitor.py`
- **Market Data**: See `backend/integrations/yfinance.py` lines 1029-1111

---
**Author:** VP Investments Team  
**Date:** October 17, 2025  
**Version:** 1.0
