# Phase 7: Scoring System Finalization

**Created:** 2025-10-09  
**Focus:** Comprehensive scoring calculation using all signal groups  
**Goal:** Production-ready weighted scoring with clear component breakdown

---

## 🎯 Current State Assessment

### ✅ What's Working
1. **Upvotes Collection** - Reddit scraper captures `submission.score` (Line 323 in reddit.py)
2. **Multi-Factor Scoring** - SignalScorer in `backend/core/signals.py` has framework
3. **Component Scores** - Reddit, News, Financial, Technical scores calculated
4. **Signal Classification** - Trade type, risk level, signal type determined
5. **Phase 6c Consolidation** - 841 lines consolidated, single signal working

### 📊 Current Scoring Architecture

**Location:** `backend/core/signals.py` (SignalScorer class)

**Component Scores** (0-1 scale):
```python
component_scores = {
    'reddit': 0.0-1.0,      # Social sentiment & engagement
    'news': 0.0-1.0,        # News sentiment (currently disabled)
    'financial': 0.0-1.0,   # Fundamentals (P/E, revenue, margins)
    'technical': 0.0-1.0,   # Price momentum, volume, indicators
    'risk': 0.0-1.0         # Volatility, beta, market cap
}
```

**Weighting Profiles:**
- `ml_optimized`: Balanced (30% Reddit, 35% Technical, 35% Financial)
- `conservative`: Financial-heavy (15% Reddit, 65% Financial, 20% Technical)
- `aggressive`: Momentum-heavy (27% Reddit, 52% Technical, 21% Financial)

---

## 🔧 Phase 7 Objectives

### 7.1: Organize All Signals into Scoring Groups (Week 1)

**Goal:** Create clear signal categories with normalized scoring

**Signal Groups:**

#### Group 1: Social Signals (Reddit) - Weight: 25%
```python
social_signals = {
    'reddit_sentiment': {
        'weight': 0.40,  # 40% of social score
        'range': [-1, 1],
        'normalize': lambda x: (x + 1) / 2,  # Convert to 0-1
        'importance': 'HIGH'
    },
    'mentions': {
        'weight': 0.25,
        'range': [0, 20],
        'normalize': lambda x: min(x / 10, 1.0),  # Cap at 10 mentions
        'importance': 'HIGH'
    },
    'upvotes': {
        'weight': 0.20,
        'range': [0, 1000],
        'normalize': lambda x: min(x / 500, 1.0),  # Cap at 500 upvotes
        'importance': 'MEDIUM'
    },
    'post_recency': {
        'weight': 0.10,
        'range': [0, 1],
        'normalize': lambda x: x,  # Already 0-1
        'importance': 'LOW'
    },
    'engagement_rate': {
        'weight': 0.05,
        'range': [0, 1],
        'normalize': lambda x: x,  # upvote_ratio from Reddit
        'importance': 'LOW'
    }
}
```

#### Group 2: Technical Signals - Weight: 30%
```python
technical_signals = {
    'price_momentum_1d': {
        'weight': 0.15,
        'range': [-20, 20],  # % change
        'normalize': lambda x: (x + 10) / 20,  # -10% to +10% maps to 0-1
        'importance': 'HIGH'
    },
    'price_momentum_7d': {
        'weight': 0.20,
        'range': [-30, 30],
        'normalize': lambda x: (x + 15) / 30,
        'importance': 'HIGH'
    },
    'volume_spike': {
        'weight': 0.15,
        'range': [0.5, 5.0],  # Ratio vs avg volume
        'normalize': lambda x: min((x - 0.5) / 2.5, 1.0),
        'importance': 'MEDIUM'
    },
    'rsi': {
        'weight': 0.10,
        'range': [0, 100],
        'normalize': lambda x: 1 - abs(x - 50) / 50,  # Optimal at 50
        'importance': 'MEDIUM'
    },
    'macd_histogram': {
        'weight': 0.15,
        'range': [-5, 5],
        'normalize': lambda x: (x + 2.5) / 5,
        'importance': 'MEDIUM'
    },
    'bollinger_position': {
        'weight': 0.10,
        'range': [0, 1],  # Where price is in BB range
        'normalize': lambda x: x,
        'importance': 'LOW'
    },
    'relative_strength': {
        'weight': 0.15,
        'range': [0, 100],
        'normalize': lambda x: x / 100,
        'importance': 'MEDIUM'
    }
}
```

#### Group 3: Fundamental Signals - Weight: 25%
```python
fundamental_signals = {
    'pe_ratio': {
        'weight': 0.25,
        'range': [5, 40],  # Optimal range
        'normalize': lambda x: 1 - abs(x - 20) / 20,  # Optimal at 20
        'importance': 'HIGH'
    },
    'revenue_growth': {
        'weight': 0.20,
        'range': [-20, 50],  # % YoY
        'normalize': lambda x: (x + 10) / 60,
        'importance': 'HIGH'
    },
    'profit_margin': {
        'weight': 0.15,
        'range': [0, 40],
        'normalize': lambda x: x / 40,
        'importance': 'MEDIUM'
    },
    'earnings_surprise': {
        'weight': 0.15,
        'range': [-30, 30],  # % vs estimate
        'normalize': lambda x: (x + 15) / 45,
        'importance': 'MEDIUM'
    },
    'analyst_rating': {
        'weight': 0.15,
        'range': [1, 5],  # 1=Strong Buy, 5=Sell
        'normalize': lambda x: (6 - x) / 4,  # Invert so 1=best
        'importance': 'MEDIUM'
    },
    'institutional_ownership': {
        'weight': 0.10,
        'range': [0, 100],
        'normalize': lambda x: x / 100,
        'importance': 'LOW'
    }
}
```

#### Group 4: Risk Signals - Weight: 15%
```python
risk_signals = {
    'beta': {
        'weight': 0.25,
        'range': [0.5, 3.0],
        'normalize': lambda x: 1 - abs(x - 1) / 2,  # Optimal at 1.0
        'importance': 'HIGH'
    },
    'volatility_30d': {
        'weight': 0.20,
        'range': [0, 100],  # % annualized
        'normalize': lambda x: 1 - min(x / 100, 1.0),  # Lower is better
        'importance': 'HIGH'
    },
    'liquidity_score': {
        'weight': 0.20,
        'range': [0, 1],
        'normalize': lambda x: x,  # Already 0-1
        'importance': 'MEDIUM'
    },
    'short_interest': {
        'weight': 0.15,
        'range': [0, 40],  # % of float
        'normalize': lambda x: min(x / 20, 1.0),  # High SI can be good/bad
        'importance': 'MEDIUM'
    },
    'market_cap_score': {
        'weight': 0.20,
        'range': [0, 1],
        'normalize': lambda x: x,  # Based on market cap category
        'importance': 'MEDIUM'
    }
}
```

#### Group 5: News Signals (Future) - Weight: 5%
```python
news_signals = {
    'news_sentiment': {
        'weight': 0.50,
        'range': [-1, 1],
        'normalize': lambda x: (x + 1) / 2,
        'importance': 'HIGH'
    },
    'news_mentions_24h': {
        'weight': 0.30,
        'range': [0, 10],
        'normalize': lambda x: min(x / 5, 1.0),
        'importance': 'MEDIUM'
    },
    'news_recency': {
        'weight': 0.20,
        'range': [0, 1],
        'normalize': lambda x: x,
        'importance': 'LOW'
    }
}
```

---

### 7.2: Implement Comprehensive Score Calculator (Week 1)

**New Class:** `ComprehensiveScorer` in `backend/core/signals.py`

```python
class ComprehensiveScorer:
    """
    Production-ready scoring engine that combines all signal groups
    with configurable weights and clear component breakdown.
    """
    
    def __init__(self, profile: str = "ml_optimized"):
        self.profile = profile
        self.group_weights = self._load_group_weights(profile)
        self.signal_definitions = self._load_signal_definitions()
        
    def _load_group_weights(self, profile: str) -> Dict[str, float]:
        """Load group-level weights based on profile"""
        profiles = {
            "ml_optimized": {
                'social': 0.25,      # Reddit/social media
                'technical': 0.30,   # Price/volume indicators
                'fundamental': 0.25, # Financial health
                'risk': 0.15,        # Risk metrics
                'news': 0.05         # News sentiment (future)
            },
            "conservative": {
                'social': 0.10,
                'technical': 0.20,
                'fundamental': 0.50,
                'risk': 0.15,
                'news': 0.05
            },
            "aggressive": {
                'social': 0.35,
                'technical': 0.40,
                'fundamental': 0.10,
                'risk': 0.10,
                'news': 0.05
            },
            "value": {
                'social': 0.05,
                'technical': 0.15,
                'fundamental': 0.60,
                'risk': 0.15,
                'news': 0.05
            }
        }
        return profiles.get(profile, profiles["ml_optimized"])
    
    def calculate_comprehensive_score(self, signal_data: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive weighted score with full breakdown.
        
        Returns:
            {
                'weighted_score': 0.0-1.0,
                'component_scores': {
                    'social': 0.0-1.0,
                    'technical': 0.0-1.0,
                    'fundamental': 0.0-1.0,
                    'risk': 0.0-1.0,
                    'news': 0.0-1.0
                },
                'signal_breakdown': {
                    'social': { 'reddit_sentiment': 0.75, 'mentions': 0.60, ... },
                    'technical': { 'price_momentum_1d': 0.80, ... },
                    ...
                },
                'top_signals': [('price_momentum_7d', 0.90), ...],
                'confidence': 0.0-1.0,
                'data_quality': 0.0-1.0  # % of signals with data
            }
        """
        
        # Calculate component scores for each group
        component_scores = {
            'social': self._calculate_social_score(signal_data),
            'technical': self._calculate_technical_score(signal_data),
            'fundamental': self._calculate_fundamental_score(signal_data),
            'risk': self._calculate_risk_score(signal_data),
            'news': self._calculate_news_score(signal_data)
        }
        
        # Calculate signal breakdown (individual signal contributions)
        signal_breakdown = {
            'social': self._breakdown_social(signal_data),
            'technical': self._breakdown_technical(signal_data),
            'fundamental': self._breakdown_fundamental(signal_data),
            'risk': self._breakdown_risk(signal_data),
            'news': self._breakdown_news(signal_data)
        }
        
        # Calculate final weighted score
        weighted_score = sum(
            component_scores[group] * self.group_weights[group]
            for group in self.group_weights.keys()
        )
        
        # Identify top contributing signals
        top_signals = self._identify_top_signals(signal_breakdown)
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(signal_data, signal_breakdown)
        
        # Calculate data quality score
        data_quality = self._calculate_data_quality(signal_breakdown)
        
        return {
            'weighted_score': round(weighted_score, 4),
            'component_scores': {k: round(v, 3) for k, v in component_scores.items()},
            'signal_breakdown': signal_breakdown,
            'top_signals': top_signals,
            'confidence': round(confidence, 3),
            'data_quality': round(data_quality, 3),
            'profile': self.profile
        }
    
    def _calculate_social_score(self, data: Dict) -> float:
        """Calculate social/Reddit component score"""
        signals = {
            'reddit_sentiment': self._normalize_signal(
                data.get('reddit_sentiment', 0),
                signal_type='reddit_sentiment'
            ),
            'mentions': self._normalize_signal(
                data.get('mentions', 0),
                signal_type='mentions'
            ),
            'upvotes': self._normalize_signal(
                data.get('upvotes', 0),
                signal_type='upvotes'
            ),
            'post_recency': data.get('post_recency', 0.5),
            'engagement_rate': data.get('upvote_ratio', 0.5)
        }
        
        # Weighted average of social signals
        weights = self.signal_definitions['social']
        weighted_sum = sum(
            signals[key] * weights[key]['weight']
            for key in signals.keys() if key in weights
        )
        
        return min(max(weighted_sum, 0), 1.0)
    
    def _normalize_signal(self, value: float, signal_type: str) -> float:
        """Normalize a raw signal value to 0-1 scale"""
        # Find signal definition
        for group in self.signal_definitions.values():
            if signal_type in group:
                signal_def = group[signal_type]
                normalize_func = signal_def['normalize']
                return min(max(normalize_func(value), 0), 1.0)
        
        return 0.5  # Default if signal not found
    
    def _identify_top_signals(self, breakdown: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
        """Identify the top N contributing signals"""
        all_signals = []
        
        for group, signals in breakdown.items():
            for signal_name, score in signals.items():
                all_signals.append((f"{group}_{signal_name}", score))
        
        # Sort by score descending
        all_signals.sort(key=lambda x: x[1], reverse=True)
        
        return all_signals[:top_n]
    
    def _calculate_confidence(self, data: Dict, breakdown: Dict) -> float:
        """
        Calculate confidence based on:
        1. Data availability (% of signals with data)
        2. Score consistency (variance across components)
        3. Data recency
        """
        
        # Data availability
        total_signals = sum(len(signals) for signals in breakdown.values())
        signals_with_data = sum(
            sum(1 for score in signals.values() if score > 0)
            for signals in breakdown.values()
        )
        data_availability = signals_with_data / total_signals if total_signals > 0 else 0
        
        # Score consistency (lower variance = higher confidence)
        component_scores = [
            sum(signals.values()) / len(signals) if signals else 0
            for signals in breakdown.values()
        ]
        score_variance = np.var(component_scores) if len(component_scores) > 1 else 0
        consistency_score = 1 - min(score_variance * 2, 1.0)  # Scale variance
        
        # Weighted confidence
        confidence = (data_availability * 0.6) + (consistency_score * 0.4)
        
        return min(max(confidence, 0), 1.0)
    
    def _calculate_data_quality(self, breakdown: Dict) -> float:
        """Calculate what % of expected signals have data"""
        total_expected = sum(len(signals) for signals in breakdown.values())
        total_with_data = sum(
            sum(1 for score in signals.values() if score != 0.5)  # 0.5 = default/no data
            for signals in breakdown.values()
        )
        
        return total_with_data / total_expected if total_expected > 0 else 0
```

**Implementation Steps:**
1. [ ] Create `ComprehensiveScorer` class
2. [ ] Define all signal groups with weights
3. [ ] Implement normalization functions
4. [ ] Add breakdown calculation methods
5. [ ] Add confidence/quality metrics
6. [ ] Test with real signals (AAPL, TSLA, GME)
7. [ ] Validate score distribution (0.2-0.8 range expected)

---

### 7.3: Integrate with Existing Pipeline (Week 2)

**Update:** `backend/pipeline.py` to use `ComprehensiveScorer`

```python
# In SignalProcessor.__init__()
from backend.core.signals import ComprehensiveScorer

self.comprehensive_scorer = ComprehensiveScorer(profile="ml_optimized")

# In generate_single_signal() or enhancement pipeline
def enhance_signal(self, signal: Dict) -> Dict:
    """Enhance signal with comprehensive scoring"""
    
    # Get comprehensive score breakdown
    score_result = self.comprehensive_scorer.calculate_comprehensive_score(signal)
    
    # Update signal with comprehensive data
    signal.update({
        'weighted_score': score_result['weighted_score'],
        'component_scores': score_result['component_scores'],
        'signal_breakdown': score_result['signal_breakdown'],
        'top_signals': score_result['top_signals'],
        'confidence': score_result['confidence'],
        'data_quality': score_result['data_quality'],
        'scoring_profile': score_result['profile']
    })
    
    # Generate dynamic descriptions based on top signals
    signal['top_factors'] = self._generate_top_factors(score_result['top_signals'])
    signal['signal_type'] = self._determine_signal_type(score_result['component_scores'])
    signal['trade_type'] = self._classify_trade_type(score_result['component_scores'])
    
    return signal
```

**Implementation Steps:**
1. [ ] Add ComprehensiveScorer to pipeline
2. [ ] Update signal enhancement flow
3. [ ] Ensure backward compatibility
4. [ ] Test with single signal generation
5. [ ] Test with batch generation
6. [ ] Verify all 140 columns populated correctly

---

### 7.4: Add Score Explainability (Week 2)

**Goal:** Users can see WHY a signal got its score

```python
class ScoreExplainer:
    """Generate human-readable explanations for signal scores"""
    
    def explain_score(self, score_result: Dict) -> Dict[str, str]:
        """
        Generate explanations for score components.
        
        Returns:
            {
                'overall': "Strong buy signal driven by momentum...",
                'social': "High Reddit engagement (15 mentions, 450 upvotes)...",
                'technical': "Strong price momentum (+12% 7d) with volume...",
                'fundamental': "Fair valuation (P/E 18.5) with growth...",
                'risk': "Moderate risk (Beta 1.24, 30d vol 45%)...",
                'top_factors': "1. Price momentum 7d (0.90)..."
            }
        """
        
        overall = self._explain_overall(score_result)
        component_explanations = {
            'social': self._explain_social(score_result),
            'technical': self._explain_technical(score_result),
            'fundamental': self._explain_fundamental(score_result),
            'risk': self._explain_risk(score_result),
        }
        
        top_factors_explanation = self._explain_top_factors(score_result['top_signals'])
        
        return {
            'overall': overall,
            **component_explanations,
            'top_factors': top_factors_explanation,
            'confidence': self._explain_confidence(score_result)
        }
    
    def _explain_overall(self, result: Dict) -> str:
        """Generate overall explanation"""
        score = result['weighted_score']
        components = result['component_scores']
        
        # Determine signal strength
        if score >= 0.7:
            strength = "Strong buy signal"
        elif score >= 0.5:
            strength = "Moderate buy signal"
        elif score >= 0.3:
            strength = "Weak signal"
        else:
            strength = "No clear signal"
        
        # Find dominant component
        dominant = max(components.items(), key=lambda x: x[1])
        
        return f"{strength} (score: {score:.2f}) driven primarily by {dominant[0]} factors ({dominant[1]:.2f})"
    
    def _explain_social(self, result: Dict) -> str:
        """Explain social component"""
        breakdown = result['signal_breakdown']['social']
        
        mentions = breakdown.get('mentions', 0)
        sentiment = breakdown.get('reddit_sentiment', 0.5)
        upvotes = breakdown.get('upvotes', 0)
        
        sentiment_desc = "positive" if sentiment > 0.6 else "negative" if sentiment < 0.4 else "neutral"
        
        return (
            f"{sentiment_desc.capitalize()} Reddit sentiment with {mentions:.0f} mentions "
            f"and {upvotes:.0f} total upvotes. "
            f"Social score: {result['component_scores']['social']:.2f}/1.00"
        )
```

**Implementation Steps:**
1. [ ] Create ScoreExplainer class
2. [ ] Implement explanation generators
3. [ ] Add to signal data (new column: `score_explanation`)
4. [ ] Test readability
5. [ ] Use in frontend tooltips

---

### 7.5: Validate and Backtest Scoring (Week 3)

**Goal:** Ensure new scoring performs better than old

**Validation Tests:**

1. **Score Distribution Test**
   - Check that scores span 0.2-0.8 (not clustered at 0.5)
   - Verify no constant scores
   - Ensure variance across tickers

2. **Component Balance Test**
   - No single component dominates all signals
   - All groups contribute meaningfully
   - Weights are respected

3. **Historical Performance Test**
   - Compare old vs new scoring on past signals
   - Check if high-scoring signals actually performed better
   - Measure prediction accuracy

4. **Edge Case Tests**
   - Ticker with only Reddit data
   - Ticker with only financial data
   - Ticker with mixed signals
   - Ticker with missing data

```python
class ScoringValidator:
    """Validate comprehensive scoring system"""
    
    async def validate_scoring(self) -> Dict[str, Any]:
        """Run full validation suite"""
        
        results = {
            'distribution': await self._test_score_distribution(),
            'component_balance': await self._test_component_balance(),
            'historical_performance': await self._test_historical_performance(),
            'edge_cases': await self._test_edge_cases()
        }
        
        # Overall pass/fail
        results['passed'] = all(test['passed'] for test in results.values())
        
        return results
    
    async def _test_score_distribution(self) -> Dict:
        """Test that scores are distributed properly"""
        
        # Get recent 100 signals
        signals = await self._fetch_recent_signals(100)
        scores = [s['weighted_score'] for s in signals]
        
        # Calculate distribution metrics
        mean = np.mean(scores)
        std = np.std(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Tests
        passed = (
            0.2 <= mean <= 0.8 and      # Mean in reasonable range
            std >= 0.1 and              # Sufficient variance
            max_score - min_score >= 0.3  # Sufficient spread
        )
        
        return {
            'passed': passed,
            'mean': mean,
            'std': std,
            'range': (min_score, max_score),
            'histogram': np.histogram(scores, bins=10)[0].tolist()
        }
```

**Implementation Steps:**
1. [ ] Create ScoringValidator class
2. [ ] Implement distribution tests
3. [ ] Implement component balance tests
4. [ ] Run backtest on historical signals
5. [ ] Compare old vs new scoring
6. [ ] Document performance improvements

---

## 📊 Expected Outcomes

### Before Phase 7
- Weighted score: Simple average, unclear components
- top_factors: Generic text ("Reddit mentions, price momentum")
- signal_type: Always "Multi-Factor"
- No score breakdown or explanation
- Confidence not data-driven

### After Phase 7
- Weighted score: Comprehensive calculation with 40+ signals
- Component breakdown: Clear contribution from each group
- Top signals: Specific drivers ("price_momentum_7d: 0.90")
- Signal explanations: Human-readable justifications
- Data quality: Measurable metric (% of signals with data)
- Confidence: Based on data availability and consistency
- Multiple profiles: ml_optimized, conservative, aggressive, value

### Performance Metrics
- **Score variance:** > 0.15 (vs current ~0.05)
- **Prediction accuracy:** Target 65%+ (vs current unknown)
- **Data quality:** Target 80%+ signals with data
- **Component balance:** No component >60% of total score
- **Explainability:** 100% of signals have explanations

---

## 🚀 Implementation Timeline

### Week 1: Core Implementation (20-25 hours)
- **Day 1-2:** Define signal groups and weights (6 hrs)
- **Day 3-4:** Implement ComprehensiveScorer (10 hrs)
- **Day 5:** Integration with pipeline (5 hrs)

### Week 2: Enhancement & Testing (15-20 hours)
- **Day 1-2:** Score explainability (6 hrs)
- **Day 3:** Testing with real data (4 hrs)
- **Day 4-5:** Refinement and tuning (6 hrs)

### Week 3: Validation & Documentation (10-15 hours)
- **Day 1-2:** Validation suite (6 hrs)
- **Day 3:** Backtesting (4 hrs)
- **Day 4-5:** Documentation and summary (5 hrs)

**Total Effort:** 45-60 hours (3 weeks part-time)

---

## 🎯 Success Criteria

- [ ] All 40+ signals organized into 5 groups
- [ ] ComprehensiveScorer calculates weighted score correctly
- [ ] Component breakdown shows contribution from each group
- [ ] Top signals identified automatically
- [ ] Score explanations generated for all signals
- [ ] Confidence and data quality metrics calculated
- [ ] Multiple scoring profiles available
- [ ] Score variance improved (>0.15)
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for Phase 8 (Frontend Integration)

---

## 📝 Next Phases Preview

### Phase 8: Frontend Integration (3-4 weeks)
- REST API with comprehensive score data
- React dashboard with score visualizations
- Signal detail page with breakdown charts
- On-demand signal generation
- Real-time score updates

### Phase 9: Performance Analytics (2-3 weeks)
- Track signal performance over time
- Measure prediction accuracy
- A/B test scoring profiles
- Optimize weights based on results
- ROI analysis and reporting

### Phase 10: Advanced Features (3-4 weeks)
- ML-based weight optimization
- Custom scoring profiles
- Alert system for high-scoring signals
- Portfolio recommendations
- Strategy backtesting

---

## 💡 Key Decisions

1. **Group Weights:** Using ml_optimized profile (25% social, 30% technical, 25% fundamental, 15% risk, 5% news)

2. **Signal Normalization:** Each signal normalized to 0-1 scale with specific ranges

3. **Confidence Calculation:** 60% data availability + 40% score consistency

4. **Top Signals:** Show top 5 contributing signals by score

5. **Data Quality:** Measure % of signals with real data (vs defaults)

6. **Backward Compatibility:** Old `weighted_score` column remains, new data in JSON columns

---

## 🔄 Migration Plan

**Database Changes:**
- Keep existing `weighted_score` column (single number)
- Add `component_scores` JSONB column (breakdown by group)
- Add `signal_breakdown` JSONB column (all 40+ signals)
- Add `score_explanation` TEXT column (human-readable)
- Add `data_quality` FLOAT column (0-1)
- Add `scoring_profile` VARCHAR column

**Code Changes:**
- Create new `ComprehensiveScorer` class (doesn't break existing)
- Update `SignalProcessor` to use new scorer (optional flag)
- Add migration script to recalculate old signals (optional)
- Keep old scoring code as fallback

**Testing:**
- Generate signals with old AND new scoring
- Compare results side-by-side
- Verify new scores are more predictive
- Rollback plan if issues found

---

**Ready to start Phase 7! 🚀**

**First Step:** Define signal groups and implement normalization functions
**Next Step:** Build ComprehensiveScorer class with component calculation
**Final Step:** Integrate and validate with real signals
