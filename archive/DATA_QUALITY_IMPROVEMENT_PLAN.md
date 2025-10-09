# 📊 DATA QUALITY IMPROVEMENT PLAN

## Executive Summary

You have **95 columns with 50-100% NULL values**. This comprehensive plan addresses each category of NULL data and provides actionable steps to improve data quality from ~30% to ~85%+ population rates.

---

## 🎯 Root Cause Analysis

### Category 1: Backtest Results (29 columns) - 100% NULL ⏰
**Why NULL**: These require signals to age 1-30 days before data can be calculated.

**Columns**:
- Returns: `1d_return`, `3d_return`, `7d_return`, `10d_return`, `30d_return`, `1d_return_net`, `3d_return_net`, `7d_return_net`, `10d_return_net`
- SPY Comparison: `spy_1d_return`, `spy_3d_return`, `spy_7d_return`, `spy_10d_return`, `beat_spy_1d`, `beat_spy_3d`, `beat_spy_7d`, `beat_spy_10d`
- Performance: `max_return_pct`, `drawdown_pct`, `signal_duration`, `forward_volatility`, `forward_sharpe_ratio`, `realized_returns`
- Metadata: `backtest_phase`, `backtest_timestamp`, `backtest_notes`

**Solution**: ✅ Already implemented - automated backtest runs every 6 hours
**Timeline**: Will populate naturally over 1-30 days
**Action Required**: NONE - wait for data accumulation

---

### Category 2: Placeholder/Future Features (18 columns) - 100% NULL 🚧
**Why NULL**: Columns exist in schema but logic not implemented yet.

**Columns**:
- Reddit: `thread_tag`, `reddit_summary`, `reddit_momentum_score`, `social_sentiment_trend`, `reddit_vs_price_divergence`
- News: `ai_news_summary`, `ai_trends_commentary`
- Options: `options_flow_score`, `option_chain_data`, `unusual_options_activity`, `option_volume_ratio`, `implied_volatility`, `iv_spike_pct`
- Quality: `entry_quality_score`, `risk_adjusted_score`
- Liquidity: `liquidity_warning`
- ML: `ml_confidence_score`, `pattern_match_score`

**Solution**: Implement calculation logic (see detailed implementations below)
**Timeline**: 2-3 days development
**Priority**: HIGH - these add significant value

---

### Category 3: Calculated Metrics (5 columns) - 100% NULL 📐
**Why NULL**: Can be calculated from existing data but logic not implemented.

**Columns**:
- `float_turnover_ratio` = `avg_daily_volume` / `shares_float`
- `momentum_consistency_score` = stddev of price momentum
- `volume_price_correlation` = correlation of volume and price changes
- `institutional_flow_direction` = trend in institutional ownership
- `market_cap_category` = bucketing of market_cap

**Solution**: Add calculation functions (see implementations below)
**Timeline**: 1 day
**Priority**: MEDIUM - useful but not critical

---

### Category 4: External Data Dependencies (10 columns) - 70-90% NULL 🔌
**Why NULL**: yfinance API doesn't provide consistent data for all tickers.

**Columns**:
- Technical: `rsi` (69% NULL), `macd_*` (75% NULL), `bollinger_*` (75% NULL), `volatility_rank` (72% NULL)
- Options: `put_call_vol_ratio` (91% NULL), `put_call_oi_ratio` (91% NULL)
- Fundamentals: `pe_ratio` (85% NULL), `eps_growth` (91% NULL), `beta` (75% NULL)
- Ownership: `institutional_ownership_pct` (77% NULL), `retail_holding_pct` (77% NULL)

**Solution**: 
1. Improve yfinance data extraction
2. Add fallback data sources
3. Calculate missing technical indicators from price history
**Timeline**: 2-3 days
**Priority**: HIGH - significant impact on scoring

---

### Category 5: Calendar Events (3 columns) - 100% NULL 📅
**Why NULL**: Event data not being extracted from yfinance.

**Columns**:
- `earnings_date`
- `dividend_ex_date`
- `analyst_targets`

**Solution**: Extract from yfinance Ticker object (see implementation below)
**Timeline**: 2 hours
**Priority**: LOW - nice to have but not critical for scoring

---

### Category 6: Composite Scores (6 columns) - 70-100% NULL 🎯
**Why NULL**: Formulas exist but not being applied consistently.

**Columns**:
- `historical_success_rate` (100% NULL) - needs backtest history
- `expected_hold_duration` (100% NULL) - can be calculated from signal type
- `signal_strength_percentile` (72% NULL) - needs percentile calculation
- `exit_signal_strength` (69% NULL) - needs exit logic
- `liquidity_score` (69% NULL) - needs liquidity formula
- `max_position_size` (69% NULL) - needs risk-based calculation

**Solution**: Implement composite score calculations
**Timeline**: 1 day
**Priority**: MEDIUM

---

## 🚀 Implementation Plan

### Phase 1: Quick Wins (1-2 days) - Target: 40% improvement

#### 1.1 Calculate Missing Technical Indicators
Many technical indicators are 70% NULL because we're relying on yfinance. We can calculate them ourselves from price history.

**File**: `backend/integrations/signal_processing.py`

```python
def calculate_advanced_technicals(ticker: str, hist: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical indicators that yfinance often misses"""
    try:
        if hist.empty or len(hist) < 50:
            return {}
        
        results = {}
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        # RSI - if missing
        if 'rsi' not in results or pd.isna(results.get('rsi')):
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        results['macd_line'] = macd_line.iloc[-1]
        results['macd_signal'] = signal_line.iloc[-1]
        results['macd_histogram'] = (macd_line - signal_line).iloc[-1]
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        results['bollinger_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
        results['bollinger_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
        results['bollinger_position'] = ((close.iloc[-1] - results['bollinger_lower']) / 
                                        (results['bollinger_upper'] - results['bollinger_lower'])) * 100
        results['bollinger_width'] = ((results['bollinger_upper'] - results['bollinger_lower']) / 
                                     sma_20.iloc[-1]) * 100
        
        # Volume indicators
        avg_volume = volume.rolling(window=30).mean().iloc[-1]
        results['volume_spike_ratio'] = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Price-Volume Correlation
        price_changes = close.pct_change().dropna()
        volume_changes = volume.pct_change().dropna()
        if len(price_changes) > 20 and len(volume_changes) > 20:
            results['volume_price_correlation'] = price_changes.corr(volume_changes)
        
        # Volatility Rank (relative to 252-day history)
        if len(hist) >= 252:
            returns = close.pct_change().dropna()
            current_vol = returns.iloc[-30:].std() * np.sqrt(252)  # 30-day annualized
            hist_vols = returns.rolling(window=30).std() * np.sqrt(252)
            results['volatility_rank'] = (hist_vols < current_vol).sum() / len(hist_vols) * 100
        
        # Momentum Consistency (lower is more consistent)
        if len(close) >= 90:
            returns_30d = close.pct_change(30)
            results['momentum_consistency_score'] = returns_30d.iloc[-60:].std() * 100
        
        # Sector Relative Strength (vs SPY)
        try:
            spy = yf.download('SPY', start=hist.index[0], end=hist.index[-1], progress=False)
            if not spy.empty and len(spy) > 20:
                ticker_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
                spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[0] - 1) * 100
                results['sector_relative_strength'] = ticker_return - spy_return
        except:
            pass
        
        return results
        
    except Exception as e:
        logger.warning(f"Error calculating advanced technicals for {ticker}: {e}")
        return {}
```

**Impact**: Reduces NULL rate from 70% to ~10% for:
- `rsi`, `macd_*`, `bollinger_*`, `volume_spike_ratio`, `volatility_rank`
- `momentum_consistency_score`, `volume_price_correlation`, `sector_relative_strength`

---

#### 1.2 Implement Calculated Metrics

**File**: `backend/integrations/signal_processing.py`

```python
def calculate_composite_metrics(signal: Dict, financial_data: Dict) -> Dict[str, float]:
    """Calculate metrics that can be derived from existing data"""
    results = {}
    
    # Float Turnover Ratio
    avg_daily_volume = financial_data.get('avg_daily_volume') or signal.get('avg_daily_volume')
    shares_float = financial_data.get('shares_float')
    if avg_daily_volume and shares_float and shares_float > 0:
        results['float_turnover_ratio'] = (avg_daily_volume / shares_float) * 100
    
    # Market Cap Category
    market_cap = signal.get('market_cap') or financial_data.get('market_cap')
    if market_cap:
        if market_cap < 300_000_000:
            results['market_cap_category'] = 'Micro'
        elif market_cap < 2_000_000_000:
            results['market_cap_category'] = 'Small'
        elif market_cap < 10_000_000_000:
            results['market_cap_category'] = 'Mid'
        elif market_cap < 200_000_000_000:
            results['market_cap_category'] = 'Large'
        else:
            results['market_cap_category'] = 'Mega'
    
    # Expected Hold Duration (based on signal type and momentum)
    signal_type = signal.get('signal_type', 'Multi-Factor')
    momentum = signal.get('momentum_30d_pct', 0)
    if 'Short' in signal_type:
        results['expected_hold_duration'] = '1-3 days'
    elif momentum and abs(momentum) > 20:
        results['expected_hold_duration'] = '3-7 days'
    else:
        results['expected_hold_duration'] = '7-14 days'
    
    # Liquidity Score (0-100)
    avg_daily_value = signal.get('avg_daily_value_traded', 0)
    if avg_daily_value:
        # Score based on daily trading value
        if avg_daily_value > 100_000_000:  # $100M+
            results['liquidity_score'] = 95
        elif avg_daily_value > 50_000_000:  # $50M+
            results['liquidity_score'] = 85
        elif avg_daily_value > 20_000_000:  # $20M+
            results['liquidity_score'] = 70
        elif avg_daily_value > 5_000_000:  # $5M+
            results['liquidity_score'] = 50
        else:
            results['liquidity_score'] = 25
    
    # Liquidity Warning
    if avg_daily_value and avg_daily_value < 5_000_000:
        results['liquidity_warning'] = '⚠️ Low liquidity - use limit orders'
    
    # Exit Signal Strength (inverse of entry)
    weighted_score = signal.get('weighted_score', 0)
    if weighted_score:
        # Strong entry = weak exit initially
        results['exit_signal_strength'] = max(0, 100 - (weighted_score * 300))
    
    # Max Position Size (% of portfolio based on risk)
    risk_level = signal.get('risk_level', 'Medium')
    liquidity_score = results.get('liquidity_score', 50)
    if risk_level == 'Low':
        base_size = 15
    elif risk_level == 'Medium':
        base_size = 10
    else:  # High/Speculative
        base_size = 5
    
    # Adjust for liquidity
    results['max_position_size'] = base_size * (liquidity_score / 100)
    
    return results
```

**Impact**: Populates 8 columns from 100% NULL to 100% populated:
- `float_turnover_ratio`, `market_cap_category`, `expected_hold_duration`
- `liquidity_score`, `liquidity_warning`, `exit_signal_strength`, `max_position_size`

---

#### 1.3 Add Calendar Events Extraction

**File**: `backend/integrations/yfinance.py` (add to `get_financial_data` function)

```python
# Around line 1400, add to the FinancialData object creation:

# Calendar events
earnings_date = None
dividend_ex_date = None
analyst_targets = None

try:
    # Earnings date
    if hasattr(ticker_obj, 'calendar') and ticker_obj.calendar is not None:
        earnings_dates = ticker_obj.calendar.get('Earnings Date', [])
        if earnings_dates:
            earnings_date = str(earnings_dates[0]) if isinstance(earnings_dates, list) else str(earnings_dates)
    
    # Dividend ex-date
    if hasattr(ticker_obj, 'dividends') and not ticker_obj.dividends.empty:
        dividend_ex_date = str(ticker_obj.dividends.index[-1].date())
    
    # Analyst targets
    if hasattr(ticker_obj, 'analyst_price_targets') and ticker_obj.analyst_price_targets:
        targets = ticker_obj.analyst_price_targets
        analyst_targets = {
            'mean': targets.get('mean'),
            'high': targets.get('high'),
            'low': targets.get('low'),
            'current': targets.get('current')
        }
except Exception as e:
    logger.debug(f"Could not extract calendar events for {ticker}: {e}")

# Add to FinancialData return object:
# earnings_date=earnings_date,
# dividend_ex_date=dividend_ex_date,
# analyst_targets=analyst_targets,
```

**Impact**: Populates 3 calendar columns from 100% NULL to ~60% populated

---

### Phase 2: Reddit & Social Enhancement (1 day) - Target: 25% improvement

#### 2.1 Reddit Metrics Enhancement

**File**: `backend/integrations/reddit.py`

```python
def calculate_reddit_momentum_score(ticker: str, posts: List[Dict]) -> float:
    """Calculate momentum score based on post frequency and recency"""
    if not posts:
        return 0.0
    
    now = datetime.now()
    scores = []
    
    for post in posts:
        created = datetime.fromtimestamp(post.get('created_utc', 0))
        hours_ago = (now - created).total_seconds() / 3600
        
        # Weight recent posts higher
        if hours_ago < 24:
            time_weight = 1.0
        elif hours_ago < 48:
            time_weight = 0.7
        elif hours_ago < 72:
            time_weight = 0.5
        else:
            time_weight = 0.3
        
        # Engagement score
        upvotes = post.get('upvotes', 0)
        comments = post.get('num_comments', 0)
        engagement = (upvotes * 1.0) + (comments * 2.0)  # Comments weighted higher
        
        scores.append(engagement * time_weight)
    
    return sum(scores) / len(posts) if scores else 0.0


def detect_reddit_vs_price_divergence(ticker: str, reddit_sentiment: float, 
                                       price_momentum: float) -> str:
    """Detect when Reddit sentiment diverges from price action"""
    if reddit_sentiment > 0.6 and price_momentum < -5:
        return "🔴 Positive sentiment, negative price - potential reversal"
    elif reddit_sentiment < 0.4 and price_momentum > 5:
        return "🟢 Negative sentiment, positive price - contrarian opportunity"
    elif reddit_sentiment > 0.7 and price_momentum > 10:
        return "⚠️ Euphoria - potential top"
    elif reddit_sentiment < 0.3 and price_momentum < -10:
        return "⚠️ Panic - potential bottom"
    else:
        return "Aligned"


def categorize_thread_tag(post: Dict) -> str:
    """Categorize Reddit posts by content type"""
    title = post.get('title', '').lower()
    body = post.get('selftext', '').lower()
    combined = f"{title} {body}"
    
    if any(word in combined for word in ['dd', 'due diligence', 'analysis']):
        return 'DD'
    elif any(word in combined for word in ['yolo', 'bet', 'calls', 'puts']):
        return 'YOLO'
    elif any(word in combined for word in ['earnings', 'er', 'report']):
        return 'Earnings'
    elif any(word in combined for word in ['news', 'announced', 'breaking']):
        return 'News'
    elif any(word in combined for word in ['?', 'should i', 'thoughts on']):
        return 'Discussion'
    else:
        return 'General'
```

**Implementation**: Add these calculations in `backend/pipeline.py` Step 4.5 enhancement

**Impact**: Populates 4 Reddit columns from 100% NULL to 100% populated:
- `reddit_momentum_score`, `reddit_vs_price_divergence`, `thread_tag`, `social_sentiment_trend`

---

### Phase 3: Options Data Improvement (2 days) - Target: 20% improvement

#### 3.1 Enhanced Options Data Extraction

**File**: `backend/integrations/yfinance.py`

```python
def get_enhanced_options_data(ticker: str) -> Dict[str, Any]:
    """Extract comprehensive options data with multiple fallbacks"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        results = {
            'put_call_vol_ratio': None,
            'put_call_oi_ratio': None,
            'option_volume_ratio': None,
            'implied_volatility': None,
            'iv_spike_pct': None,
            'unusual_options_activity': False,
            'options_flow_score': 0
        }
        
        # Get options chain
        try:
            options_dates = ticker_obj.options
            if not options_dates:
                return results
            
            # Use nearest expiration date
            nearest_exp = options_dates[0]
            opt_chain = ticker_obj.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if not calls.empty and not puts.empty:
                # Put/Call Volume Ratio
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                if call_volume > 0:
                    results['put_call_vol_ratio'] = put_volume / call_volume
                
                # Put/Call Open Interest Ratio
                call_oi = calls['openInterest'].sum()
                put_oi = puts['openInterest'].sum()
                if call_oi > 0:
                    results['put_call_oi_ratio'] = put_oi / call_oi
                
                # Option Volume vs Stock Volume
                total_opt_volume = call_volume + put_volume
                stock_volume = ticker_obj.info.get('volume', 0)
                if stock_volume > 0:
                    results['option_volume_ratio'] = total_opt_volume / stock_volume
                
                # Implied Volatility (ATM options)
                current_price = ticker_obj.info.get('currentPrice', 0)
                if current_price > 0:
                    # Find ATM calls
                    atm_calls = calls[abs(calls['strike'] - current_price) < (current_price * 0.05)]
                    if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                        results['implied_volatility'] = atm_calls['impliedVolatility'].mean()
                        
                        # IV Spike (compare to historical)
                        hist_vol = ticker_obj.info.get('fiftyTwoWeekVolatility', results['implied_volatility'])
                        if hist_vol and hist_vol > 0:
                            results['iv_spike_pct'] = ((results['implied_volatility'] - hist_vol) / hist_vol) * 100
                
                # Unusual Options Activity Detection
                avg_call_volume = calls['volume'].mean()
                avg_put_volume = puts['volume'].mean()
                max_call_volume = calls['volume'].max()
                max_put_volume = puts['volume'].max()
                
                # Unusual if max volume > 5x average
                if (max_call_volume > avg_call_volume * 5) or (max_put_volume > avg_put_volume * 5):
                    results['unusual_options_activity'] = True
                
                # Options Flow Score (0-100)
                # Bullish indicators: high call volume, low P/C ratio, unusual call activity
                score = 50  # Neutral baseline
                
                if results['put_call_vol_ratio']:
                    if results['put_call_vol_ratio'] < 0.7:  # Bullish
                        score += 20
                    elif results['put_call_vol_ratio'] > 1.3:  # Bearish
                        score -= 20
                
                if results['option_volume_ratio'] and results['option_volume_ratio'] > 0.5:
                    score += 10  # High options interest
                
                if results['unusual_options_activity']:
                    score += 15
                
                results['options_flow_score'] = max(0, min(100, score))
        
        except Exception as opt_error:
            logger.debug(f"Options chain error for {ticker}: {opt_error}")
        
        return results
        
    except Exception as e:
        logger.warning(f"Enhanced options data failed for {ticker}: {e}")
        return {}
```

**Impact**: Improves options data from 10% to ~40% population rate for:
- `put_call_vol_ratio`, `put_call_oi_ratio`, `option_volume_ratio`
- `implied_volatility`, `iv_spike_pct`, `unusual_options_activity`, `options_flow_score`

---

### Phase 4: AI-Generated Content (1 day) - Target: 10% improvement

#### 4.1 Add Reddit Summary Generation

**File**: `backend/integrations/ai.py`

```python
async def generate_reddit_summary(ticker: str, posts: List[Dict]) -> str:
    """Generate AI summary of Reddit sentiment and key themes"""
    if not posts or len(posts) < 3:
        return "Limited Reddit discussion"
    
    # Extract key points
    top_posts = sorted(posts, key=lambda x: x.get('upvotes', 0), reverse=True)[:5]
    
    prompt = f"""Summarize the Reddit sentiment for {ticker} in 2-3 sentences:

Top Posts:
{chr(10).join([f"- {p.get('title', 'N/A')} ({p.get('upvotes', 0)} upvotes)" for p in top_posts])}

Focus on: Overall sentiment, main catalysts mentioned, sentiment trend."""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except:
        return "Unable to generate summary"


async def generate_quality_scores(signal: Dict) -> Dict[str, float]:
    """Generate entry quality and risk-adjusted scores"""
    scores = {}
    
    # Entry Quality Score (0-100)
    quality = 50  # Baseline
    
    # Technical strength
    if signal.get('rsi') and 40 <= signal['rsi'] <= 70:
        quality += 10
    if signal.get('volume_spike_ratio') and signal['volume_spike_ratio'] > 1.5:
        quality += 10
    if signal.get('above_200d_ma_pct') and signal['above_200d_ma_pct'] > 0:
        quality += 10
    
    # Sentiment alignment
    reddit_score = signal.get('reddit_score', 0)
    financial_score = signal.get('financial_score', 0)
    if abs(reddit_score - financial_score) < 0.1:
        quality += 10  # Aligned signals
    
    # Liquidity
    if signal.get('liquidity_score', 0) > 70:
        quality += 10
    
    scores['entry_quality_score'] = max(0, min(100, quality))
    
    # Risk-Adjusted Score
    weighted_score = signal.get('weighted_score', 0)
    risk_score = signal.get('risk_score', 50)
    
    # Higher score, lower risk = better risk-adjusted score
    if risk_score > 0:
        scores['risk_adjusted_score'] = (weighted_score * 100) * (100 - risk_score) / 100
    else:
        scores['risk_adjusted_score'] = weighted_score * 100
    
    return scores
```

**Impact**: Populates 4 AI/quality columns from 100% NULL to 100% populated:
- `reddit_summary`, `entry_quality_score`, `risk_adjusted_score`, `ai_news_summary`

---

## 📈 Expected Results

### Data Quality Improvement Summary

| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| **Backtest Results** | 0% (100% NULL) | 80%+ | +80% (over 30 days) |
| **Technical Indicators** | 30% (70% NULL) | 90%+ | +60% |
| **Options Data** | 10% (90% NULL) | 40-50% | +30-40% |
| **Calculated Metrics** | 0% (100% NULL) | 100% | +100% |
| **Reddit Metrics** | 0% (100% NULL) | 100% | +100% |
| **Quality Scores** | 0-30% (70-100% NULL) | 100% | +70-100% |
| **Calendar Events** | 0% (100% NULL) | 60% | +60% |

### Overall Impact

**Current State**: 41 fully populated columns, 95 mostly NULL columns  
**After Implementation**: 100+ fully populated columns, 30 mostly NULL columns

**Data Quality Score**: Improves from ~32% to ~75%+ populated fields

---

## 🔧 Implementation Order

### Week 1: Foundation (5 days)
1. **Day 1**: Implement `calculate_advanced_technicals()` - Technical indicators
2. **Day 2**: Implement `calculate_composite_metrics()` - Calculated fields
3. **Day 3**: Add calendar events extraction + Reddit momentum/tagging
4. **Day 4**: Implement enhanced options data extraction
5. **Day 5**: Add AI quality scores + Reddit summaries

### Week 2: Integration & Testing (3 days)
6. **Day 6**: Integrate all new calculations into pipeline Step 4.5
7. **Day 7**: Test pipeline runs, verify data population
8. **Day 8**: Update scoring to use new fields, optimize weights

### Week 3: Monitoring (ongoing)
9. Monitor backtest data accumulation (automated, no action needed)
10. Track data quality improvements over multiple runs
11. Adjust scoring weights based on performance

---

## 💡 Next Steps

### Option 1: Full Implementation
Implement all phases over 1-2 weeks for maximum data quality improvement.

### Option 2: Phased Approach
Start with Phase 1 (Quick Wins) to see immediate 40% improvement, then evaluate.

### Option 3: Custom Priority
Tell me which categories are most important to you and I'll focus on those first.

**Which approach do you prefer?** I can start implementing immediately once you decide.

---

*Generated: October 6, 2025*  
*Target: Reduce NULL columns from 95 to <30*  
*Expected Data Quality: 32% → 75%+*
