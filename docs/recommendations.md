# VP Investments - Implementation Guide# VP Investments - Implementation Guide



**Last Updated:** 2025-10-09  **Last Updated:** 2025-10-08  

**Status:** ✅ Phase 6 Complete - Production Ready System**Status:** ✅ Phase 4 In Progress - Schema Cleanup + Code Quality Improvements



------



## 🎯 Current Status## 🎯 Current Status



### Project Overview### Project Overview

- **Strategy:** Swing/Long Trading with AI-generated signals- **Strategy:** Swing/Long Trading with AI-generated signals

- **Architecture:** ✅ Clean separation - Pipeline orchestrates, SignalScorer scores  - **Database:** ✅ Clean 6-table schema (removed 5 unused tables)

- **Performance:** ⚡ 50% faster execution, 50% fewer API calls- **Performance:** ⚡ 50% faster execution, 50% fewer API calls

- **Code Quality:** 📦 841 lines consolidated into SignalScorer- **Pipeline:** Fully optimized with Phase 3 fundamentals integrated

- **Features:** 🎯 On-demand signal generation + Full pipeline mode- **Current Focus:** Phase 4 - Signal Score Validation & Backtesting

- **Database:** 💾 1,140+ signals, 6 active tables, production-ready

- **Next Focus:** 📊 Database optimization + Frontend integration### Recent Completions



---**✅ Phase 3: Fundamental Data Enhancement + Database Integration (COMPLETE - 2025-10-08)**



## ✅ Completed Phases (Summary)**New Data Categories (4 added):**

- 📈 **Analyst Consensus** - price targets, recommendations, upside % (90% coverage)

### Phase 6c: Complete Scoring Consolidation (2025-10-09)- 💹 **Earnings Momentum** - surprise history, trends (40% coverage)

- **841 lines consolidated** - All scoring moved to SignalScorer class- 🏢 **Institutional Activity** - ownership changes, concentration (67% coverage)

- **Clean architecture** - Pipeline = orchestration, SignalScorer = scoring logic- 👔 **Insider Sentiment** - trading activity scoring 0-100 (100% coverage)

- **Comprehensive scoring** - Technical (40%) + Fundamentals (30%) + Options (15%) + Short Interest (15%)

- **Integration tested** - AAPL (0.523 score) and TSLA (0.412 score) validated**Database Integration:**

- ✅ Added 14 Phase 3 columns to both `signals` and `signal_metrics` tables

### Single Signal Feature (2025-10-09)  - ✅ Fixed 4-layer data flow bug (cache → field mapping → database)

- **On-demand generation** - Generate signals for any ticker in ~5-6 seconds- ✅ Verified Phase 3 data persisting correctly

- **Frontend-ready** - API endpoint pattern ready for React dashboard- ✅ Field name mapping: yfinance outputs → database schema

- **Bug fixed** - Proper field mapping (weighted_score, financial_score)

- **Production tested** - Both with/without Reddit modes working**Fundamental Metrics Expansion:**

- Now tracking **20 fundamental metrics** (up from 16)

### Phase 3: Fundamental Data Enhancement (2025-10-08)- Data collection and scoring working perfectly

- **20 fundamental metrics** - Analyst targets, earnings surprises, institutional/insider activity- All Phase 3 fields integrated into financial_score calculation

- **90% coverage** - Analyst data for most tickers- Ready for frontend consumption

- **Database integration** - 14 new columns, all data persisting correctly

**Impact:**

### Phase 2: Financial Score Redesign (2025-10-07)- More comprehensive stock valuation context

- **30+ indicators** - 95% of collected data now utilized in scoring- Earnings quality indicators (surprise trends)

- **11 technical components** - Dynamic normalization, graduated scoring- Smart money tracking (institutions + insiders)

- **16 fundamental metrics** - FCF yield, liquidity ratios, growth indicators- Better differentiation between strong/weak fundamentals



### Phase 1.4: Performance Optimization (2025-10-07)**✅ Phase 2: Financial Score Redesign (COMPLETE - 2025-10-07)**

- **50% faster** - 150s → 75s through intelligent caching

- **50% fewer API calls** - 70 → 35 calls per run**Scoring System Overhaul:**

- **ML analytics** - Momentum consistency scoring integrated- 🎯 **30+ indicators utilized** (was ~60%, now ~95% of collected data)

- ⚖️ **Enhanced scoring with graduated metrics** (not binary good/bad)

### Phase 0, A, B: Foundation (2025-10-06)- 📊 **11 technical components** (was 9) with dynamic normalization

- **Configurable scoring** - Weights in .env configuration- 💰 **16 fundamental metrics** (was 8) including FCF yield, liquidity ratios

- **Backtest engine** - Track 1d/3d/7d/10d/30d returns vs SPY- ✅ **All tests passed** - validated on diverse stock types (AAPL, TSLA, KO, NVDA, F)

- **Technical indicators** - 9 Phase B indicators + baseline metrics

**Technical Scoring Enhancements:**

---- Added beta analysis (8% weight) - market correlation risk

- Added exit signal strength (5% weight) - inverted scoring

## 📋 Implementation Roadmap- Enhanced volume analysis with correlation boosting

- Graduated RSI scoring with neutral zones

### 🔥 ACTIVE: Database Optimization (This Week)- Moving average distance-based scoring (>5% above = max)

- Dynamic weight normalization handles missing data gracefully

**Priority 1: Schema Analysis** ⏱️ 1 hour

```bash**Fundamental Scoring Enhancements (Phase 2):**

# Run comprehensive analysis- Added PEG ratio (5%) - growth-adjusted valuation

python tables.py --report- Added Price/Sales ratio (4%) - revenue multiple

python tables.py --nulls signals- Added operating margin (5%) - efficiency metric

python tables.py --recommend- Added earnings growth (6%) - profitability momentum

python tables.py --export database_analysis_20251009.md- Added current ratio (3%) + quick ratio (3%) - liquidity health

```- Added free cash flow yield (10%) - cash generation metric

- 5-tier market cap classification (micro to mega)

**Key Questions:**

- Which columns have >80% NULL values?**✅ Phase 1.4: Performance Optimization & ML Metrics (COMPLETE - 2025-10-07)**

- Which columns have constant/broken values?

- Any empty or redundant tables?**Performance Improvements:**

- Data quality issues to fix?- ⚡ **50% faster pipeline** (150s → 75s) through intelligent caching

- 🚀 **50% fewer API calls** (70 → 35) - eliminated duplicate yfinance calls

**Priority 2: Data Quality Fixes** ⏱️ 3-4 hours- ✅ **100% reliable saves** - fixed 5 critical database bugs

- [x] **Fixed: signal_score mapping** - Use weighted_score/financial_score ✅- 🎯 **Single-pass data fetching** - all ticker data cached once

- [ ] **Fix: Beta calculation** - Use yfinance .info['beta'] instead of default 1.0

- [ ] **Fix: Upvotes collection** - Reddit scraper not capturing upvote counts  **ML Analytics Integration:**

- [ ] **Fix: MACD indicators** - 80% NULL rate, investigate yfinance data availability- ✅ **Momentum Consistency Score** - 7% weight in technical scoring

- [ ] **Fix: Bollinger Bands** - High NULL rate, may need manual calculation from prices  - Measures consistency across 1d, 7d, 30d timeframes

- [ ] **Fix: Exchange field** - All showing "NYSE", should vary (NASDAQ, OTC, etc.)  - Scale: 0-100 (higher = more consistent momentum)

  - Identifies sustainable trends vs short-term volatility

**Priority 3: Schema Cleanup** ⏱️ 2-3 hours

- [ ] **Drop 100% NULL columns** - Remove unused columns identified by tables.py- ✅ **Liquidity Score** - 6% weight in technical scoring (increased from 5%)

- [ ] **Drop redundant tables** - backtest_results, signal_performance (data in signals table)  - Based on daily value traded vs market cap

- [ ] **Optimize indexes** - Add indexes for common query patterns  - Scale: 0.0-1.0 (higher = more liquid)

- [ ] **Update pipeline code** - Remove references to dropped columns/tables  - Critical for risk assessment and position sizing



**Deliverables:****Technical Debt Resolved:**

- ✅ Database analysis report (markdown export)- ✅ Removed 11 unused columns (137 → 126 columns)

- 📋 Prioritized fix list based on tables.py output- ✅ Fixed market_cap_category constraint handling

- 🧹 Cleaner schema with <20% NULL rate for key columns- ✅ Fixed NoneType comparison errors (3 locations)

- 📈 Improved data accuracy (beta, upvotes, exchange)- ✅ Implemented comprehensive error handling

- ✅ Added ML analytics to enhancement pipeline

---

**✅ Phase 0: Configurable Scoring System (COMPLETE)**

### 📊 Phase 7: Frontend Integration (Next Week)- Moved scoring weights to `.env` configuration

- Financial, technical, options, short interest, Reddit scores configurable

**Goal:** Build React dashboard for signal visualization- News scoring disabled (API limits)



**Phase 7.1: API Endpoints** ⏱️ 4-6 hours**✅ Phase A: Historical Success Rate & Backtest (COMPLETE)**

- [ ] `POST /api/signals/generate` - On-demand signal generation- ✅ Backtest engine integrated into pipeline

- [ ] `GET /api/signals` - List with filtering (score, date, ticker)- ✅ Intervals: [1d, 3d, 7d, 10d, 30d] returns tracked

- [ ] `GET /api/signals/:id` - Detailed signal view- ✅ SPY comparison and beat_spy flags

- [ ] `GET /api/signals/stats` - Win rates, performance metrics- ✅ Performance records with full history

- ✅ NEW: 30d_return column added and supported

**Phase 7.2: Dashboard UI** ⏱️ 8-12 hours

- [ ] **Signal List** - Table with scores, filtering, sorting**✅ Phase B: Technical Indicators Enhancement (COMPLETE)**

- [ ] **Signal Detail** - Full metrics, charts, AI commentary- ✅ 9 Phase B indicators implemented and populated

- [ ] **Generator Form** - Input ticker → generate signal- ✅ All indicators integrated into financial_score calculation

- [ ] **Stats Dashboard** - Performance overview- ✅ Database schema updated and migrated



**Phase 7.3: Real-Time** ⏱️ 6-8 hours---

- [ ] WebSocket for live updates

- [ ] Price alerts when targets hit## 📋 Implementation Roadmap

- [ ] Real-time charts (TradingView integration)

### 🔥 ACTIVE: Phase 4 - Schema Cleanup + Foundation (Week 1)

---

**Goal:** Clean up database schema, fix critical bugs, prepare for advanced features

### 📈 Phase 8: Performance Analytics (Week 3)

**Phase 4.1: Schema Cleanup** ✅ READY TO RUN

**Goal:** Analyze and optimize signal performance- ✅ **Drop 36 NULL Columns** - Commentary, net returns, unused scoring (approved)

- ✅ **Drop signal_metrics Table** - Data should be in signals table (confirmed)

**Correlation Analysis** ⏱️ 4-6 hours- ✅ **Backtest Data** - 759/1065 signals backtested (71.3% coverage)

- [ ] Score vs Returns - Analyze weighted_score correlation with actual returns- ✅ **Migration Script** - phase4_schema_cleanup.sql ready

- [ ] Component Analysis - Which scores predict best (technical vs fundamental)- **Action:** Run SQL in Supabase, update pipeline code

- [ ] Win Rate by Score Bucket - <0.3, 0.3-0.5, 0.5-0.7, >0.7

- [ ] Threshold Optimization - Find optimal minimum score for trading**Phase 4.2: beat_spy Implementation** � NEXT

- [ ] **Add beat_spy Columns** - BOOLEAN columns for 1d, 3d, 7d, 10d, 30d

**Visualization** ⏱️ 4-6 hours- [ ] **Calculate Comparisons** - Compare signal returns vs SPY benchmark

- [ ] Matplotlib graphs - Score distributions, return correlations- [ ] **Update Backtest Function** - Populate beat_spy when backtesting

- [ ] Scatter plots - Score vs 1d/3d/7d returns- [ ] **Create SQL View** - v_signal_vs_spy for easy querying

- [ ] Heatmaps - Component contribution analysis- **Time:** 2-3 hours

- [ ] Performance dashboard - Win rate by score range- **Why:** Critical for determining if signals outperform market



**Weight Optimization** ⏱️ 6-8 hours**Phase 4.3: Extended Backtest Intervals** 📋 FUTURE

- [ ] Backtest current weights - Baseline performance- [ ] **Add 7d Return Columns** - 7d_return, spy_7d_return, beat_spy_7d

- [ ] Grid search - Test different weight combinations- [ ] **Add 10d Return Columns** - 10d_return, spy_10d_return, beat_spy_10d

- [ ] ML optimization - Use RandomForest for feature importance- [ ] **Add 30d Return Columns** - 30d_return, spy_30d_return, beat_spy_30d

- [ ] A/B testing - Compare old vs optimized weights- [ ] **Update Backtest Logic** - Calculate when signals are old enough

- [ ] **Batch Backtest Old Signals** - Fill in 7d/10d/30d for existing signals

---- **Time:** 3-4 hours

- **Why:** Longer time horizons show signal quality over time

### 🤖 Phase 9: Advanced ML (Week 4-5)

**Phase 4.4: Code Quality Fixes** � IN PROGRESS

**Feature Engineering** ⏱️ 6-8 hours- [ ] **Fix Beta Calculation** - Use yfinance .info['beta'] instead of hardcoded 1.0

- [ ] Historical features (past returns, volatility)- [ ] **Fix Reddit Upvotes** - Capture 'score' field from Reddit API

- [ ] Relative features (sector comparison, beta)- [ ] **Remove signal_metrics References** - Update pipeline.py to stop writing

- [ ] Sentiment trends (Reddit momentum, news flow)- [ ] **Combined Commentary Field** - Plan for future Reddit + News + Financial merge

- **Time:** 2-3 hours

**Predictive Model** ⏱️ 8-12 hours- **Why:** Improve data quality and remove technical debt

- [ ] Train model - All features → forward returns

- [ ] Cross-validation - Walk-forward to avoid overfitting**Clarifications:**

- [ ] Feature importance - What matters most- **SPY 3d return same value**: Expected (market benchmark for that period)

- [ ] Model comparison - RandomForest vs XGBoost vs LightGBM- **News scrape disabled**: Intentional, will re-enable with combined commentary

- **Score-return analysis**: Deferred to Phase 5 (will use matplotlib/graphs)

**ML Integration** ⏱️ 6-8 hours- **Kept 16 columns**: beat_spy (5), 7d/10d/30d returns (5), options (4), social (2)

- [ ] Add ml_score column to database

- [ ] Hybrid scoring - Combine rule-based + ML**Deliverables:**

- [ ] Model monitoring - Track accuracy over time- ✅ SQL migration script (phase4_schema_cleanup.sql)

- [ ] Auto-retraining - Weekly model updates- [ ] Updated pipeline code (remove signal_metrics writes)

- [ ] beat_spy implementation

---- [ ] Extended backtest intervals

- [ ] tables.py verification (~122 recommendations, down from 174)

### 🎯 Future Phases

### 📊 Phase 5 - Score-Return Analysis & Visualization (Week 2)

**Phase 10: Risk Management**

- Position sizing, portfolio correlation, stop-loss automation**Goal:** Analyze correlation between scores and returns using matplotlib/graphs



**Phase 11: Real-Time Trading****Phase 5 Tasks:**

- Live monitoring, intraday updates, unusual activity alerts- 📊 **Correlation Analysis** - weighted_score vs 1d/3d/7d returns

- 📈 **Component Breakdown** - Which score components predict returns best

**Phase 12: Advanced Analytics**- 🎯 **Threshold Optimization** - Find optimal score threshold for trading

- Sector rotation, market regime detection, custom screeners- 📉 **Visualization** - matplotlib graphs showing score distributions and returns

- 🔍 **Win Rate Analysis** - Calculate win rates by score bucket

---

**Deliverables:**

## 🗂️ Database Schema- analyze_phase5_correlations.py script

- Matplotlib graphs (score vs return scatter plots, histograms)

### Core Tables (6 Active)- Recommended trading thresholds (e.g., only trade signals >0.5)

- Performance report by score component

| Table | Rows | Purpose | Status |

|-------|------|---------|--------|### 📊 Phase 6 - Advanced Data Collection (Week 3-4)

| **signals** | 1,140+ | Main signals + backtest results | ✅ Active |

| **signal_metrics** | 1,017 | Detailed metrics (may be redundant) | 🔍 Review |**Goal:** Implement options flow and social sentiment features

| **ai_strategies** | 317 | AI trading strategies | ✅ Active |

| **runs** | 41+ | Pipeline execution tracking | ✅ Active |**Phase 6.1: Options Flow Integration**

| **company_tickers** | 7,638 | Ticker reference data | ✅ Active |- [ ] **options_flow_score** - Aggregate call/put sentiment

| **guardrails_config** | 6 | System configuration | ✅ Active |- [ ] **unusual_options_activity** - Detect large unusual trades

- [ ] **implied_volatility** - IV rank and percentile

### Score Columns (Verified Working)- [ ] **iv_spike_pct** - Detect IV expansion events

- [ ] **Data Source** - Research premium options API or yfinance alternatives

**Primary Scores:**- **Time:** 4-6 hours

- `weighted_score` - Combined score [0.0-1.0] ✅- **Why:** Options activity can predict price moves

- `financial_score` - Financial component ✅

- `reddit_score` - Reddit sentiment ✅**Phase 6.2: Social Sentiment Tracking**

- `news_score` - News (disabled, always 0)- [ ] **social_sentiment_trend** - Track sentiment changes over time

- [ ] **reddit_momentum_score** - Measure velocity of Reddit interest

**Component Scores:**- [ ] **Trending Detection** - Identify when tickers start trending

- `options_flow_score` - Options sentiment- [ ] **Time Decay** - Weight recent posts higher

- `momentum_consistency_score` - ML momentum- **Time:** 3-4 hours

- `insider_activity_score` - Insider trading- **Why:** Early detection of viral tickers

- `liquidity_score` - Liquidity analysis

- `risk_score` - Risk assessment### 🤖 Phase 7 - ML Scoring Optimization (Week 5-6)



**Known Issues (From Previous Analysis):****Goal:** Use machine learning to optimize scoring weights based on backtest results

- MACD/Bollinger indicators: 80% NULL (yfinance availability issue)

- Beta: Always 1.0 (hardcoded, needs fix)**Phase 7 Tasks:**

- Upvotes: Always 0 (scraper not capturing)- 📈 **Feature Importance** - Which components predict returns (RandomForest/XGBoost)

- Exchange: Always "NYSE" (needs per-ticker lookup)- ⚖️ **Dynamic Weights** - Adjust reddit/financial/technical weights based on historical performance

- 🎯 **Predictive Model** - Train model: all_metrics → forward_returns

---- 🔄 **Cross-Validation** - Walk-forward analysis to avoid overfitting

- 📊 **A/B Testing** - Compare old weights vs ML-optimized weights

## 🏗️ Architecture

**Deliverables:**

### System Design (Post-Phase 6)- ML model that predicts 7d returns from signal features

- Optimized scoring weights (may find reddit_score matters more/less than thought)

```- Feature importance report (e.g., "RSI contributes 8% to prediction accuracy")

UnifiedPipeline (2,547 lines)

  ├── Data Collection### 🤖 Phase 6 - AI Strategy Backtesting (Week 3-4)

  │   ├── Reddit (PRAW)

  │   ├── Yahoo Finance**Goal:** Validate if AI-generated trading strategies would have been profitable

  │   └── Supabase

  │**Phase 6 Tasks:**

  ├── SignalScorer (2,137 lines) ⭐- 📝 **Strategy Parsing** - Extract entry/exit rules from ai_strategies.entry_conditions

  │   ├── Reddit/News/Options scoring- 💼 **Position Simulation** - Simulate trades with position sizing

  │   ├── Short interest + risk penalty- 📉 **P&L Calculation** - Track profits/losses for each strategy

  │   ├── Fundamentals (10 categories, 364 lines)- 🎯 **Win Rate Analysis** - Which strategy types work best

  │   ├── Technical (11 categories, 235 lines)- 🔄 **Strategy Evolution** - Improve AI prompts based on backtest results

  │   ├── Financial orchestrator (77 lines)

  │   └── Score components (50 lines)**Deliverables:**

  │- Backtest results for all 317 existing AI strategies

  ├── Enhancement Pipeline- Analysis of which strategy types are profitable

  │   ├── ML Analytics- Updated AI prompts to generate better strategies

  │   ├── Technical Indicators

  │   └── Score Breakdowns### 🎯 Future Phases (Post-Core Validation)

  │

  └── Database Persistence**Phase 7: Risk Management & Portfolio Construction**

```- Position sizing based on signal strength + volatility

- Portfolio-level risk limits and correlation analysis

**Benefits:**- Stop-loss and take-profit automation

- ✅ Clear separation: Pipeline = workflow, SignalScorer = logic- Max drawdown constraints

- ✅ All scoring in one place (maintainable)

- ✅ Easy to test individual methods**Phase 8: Real-Time Enhancements**

- ✅ Simple to modify weights/formulas- Live price monitoring and alert system

- Intraday signal updates

---- Options unusual activity detection

- Short squeeze candidate monitoring

## 📊 Performance Metrics

**Phase 9: Advanced Analytics**

### System Performance- Sector rotation analysis

- Pipeline execution: ~75 seconds (50% improvement)- Market regime detection (bull/bear/sideways)

- API calls per run: ~35 (50% reduction)- Correlation matrix for diversification

- On-demand signal: ~5-6 seconds- Custom screeners and watchlists

- Database save rate: 100% reliability

---

### Code Quality

- Total codebase: 4,684 lines (pipeline + scorer)## 🗂️ Database Schema (Optimized 2025-10-08)

- Consolidation: 841 lines moved to SignalScorer

- Architecture: Clean separation achieved### Core Tables (6 Active) - Simplified Schema

- Test coverage: Integration tests passing

| Table | Rows | Columns | Purpose | Phase 4 Action |

### Data Quality (To Measure)|-------|------|---------|---------|----------------|

- Current: Unknown (need tables.py analysis)| **signals** | 1,065 | 140 | Main signals + backtest results | ✅ Keep - core table |

- Target: <20% NULL for key columns| **signal_metrics** | 1,017 | ~80 | Detailed technical/fundamental metrics | ✅ Keep - analytics |

- Target: Accurate beta, upvotes, exchange| **ai_strategies** | 317 | 45 | AI-generated trading strategies | ✅ Keep - Phase 6 |

- Target: All score fields populated| **runs** | 41 | 9 | Pipeline execution metadata | ✅ Keep - tracking |

| **company_tickers** | 7,638 | 11 | Ticker reference data | ✅ Keep - reference |

---| **guardrails_config** | 6 | 9 | System configuration | ✅ Keep - config |



## 🚀 Immediate Next Actions### Tables to Delete (Phase 4 Cleanup)



### 1. Run Database Analysis (30 min)| Table | Rows | Reason for Deletion |

```bash|-------|------|---------------------|

cd "C:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"| **signal_scoring_factors** | 0 | Empty - never used |

python tables.py --report| **signal_performance** | 0 | Empty - backtest data in `signals` table |

python tables.py --nulls signals  | **backtest_interval_tracking** | 5,325 | Can calculate eligibility from signal.created_at |

python tables.py --recommend| **reddit_posts** | ? | Not needed for core scoring |

python tables.py --export db_analysis_20251009.md| **ticker_mentions** | ? | Not needed for core scoring |

```

**Benefits of Cleanup:**

### 2. Review Results (30 min)- ✅ Simpler schema (6 tables instead of 11+)

- Identify high NULL columns (>80%)- ✅ Easier to understand and maintain

- Find constant/broken values- ✅ Faster backups and queries

- Check for empty tables- ✅ No "where does this data go?" confusion

- Prioritize fixes- ✅ All backtest data in one place (`signals` table)



### 3. Execute Fixes (3-4 hours)### Signals Table (142 columns)

- Fix beta calculation

- Fix upvote collection**Populated Columns (78):**

- Fix exchange field- Core metadata (5): id, ticker, signal_type, created_at, updated_at

- Test data improvements- Technical indicators (20): RSI, MACD, volume, moving averages, etc.

- Financial metrics (8): P/E ratio, EPS growth, debt ratio, margins

### 4. Plan Phase 7 (1 hour)- Options data (2): Put/call ratios (OI and volume)

- Design API contracts- Short interest (2): Short % float, short ratio

- Sketch UI mockups- Reddit/Social (9): Mentions, sentiment, upvotes, key posts

- Define database queries- Scores (23): Individual scores + weighted_score + explanations

- Create task breakdown- Backtest returns (4): 1d_return, 3d_return, 7d_return, 10d_return (Phase A)

- AI commentary (6): Various AI-generated text fields

---- Run tracking (3): run_id, pipeline timestamps



## 📚 Reference Documents**Empty Columns - Ready for Implementation (60):**

See [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for categorization and plans.

### Active Docs

- **[operational_guidelines.md](operational_guidelines.md)** - Coding standards---

- **[PHASE_6C_COMPLETE.md](../PHASE_6C_COMPLETE.md)** - Latest completion

- **[SINGLE_SIGNAL_FEATURE_COMPLETE.md](../SINGLE_SIGNAL_FEATURE_COMPLETE.md)** - Feature docs## 🏗️ Architecture Decisions



### Archived Docs### Data Sources (Free APIs Only - Current Phase)

- **[archive/phase_docs/](../archive/phase_docs/)** - Phase 0-6B completion docs

- **[archive/recommendations_old.md](../archive/recommendations_old.md)** - Previous version**Primary:**

- **yfinance** - Price data, technical indicators, basic options, fundamentals

---- **FMP API** - Financial metrics, company fundamentals (if needed)

- **Reddit (PRAW)** - Social sentiment and discussion data

## 📝 Notes- **OpenAI GPT-4** - AI commentary and strategy generation



### Recent Changes (2025-10-09)**Disabled:**

- ✅ Cleaned up root directory (moved 13 phase docs to archive)- ~~News API~~ - Rate limits, will enable with premium plan later

- ✅ Updated recommendations with Phase 6 completion- ~~Premium Options APIs~~ - Use yfinance for now, upgrade later

- ✅ Fixed signal_score bug (field mapping issue)- ~~Insider Data APIs~~ - Use yfinance basic, premium later

- ✅ Validated integration tests (AAPL, TSLA)

- 🔄 Ready for database optimization analysis### Database Design Philosophy



### Code Files in Root**"Less is More" Approach:**

- `PHASE_6C_COMPLETE.md` - Latest completion- Keep only active tables (7 core tables)

- `SINGLE_SIGNAL_FEATURE_COMPLETE.md` - Feature docs- Keep only essential views (3 views)

- `README.md` - Project overview- Archive unused tables for potential future use

- `tables.py` - Database analysis tool ⭐- Zero redundancy, maximum clarity

- `test_single_signal.py` - Integration test- Easy to understand and maintain

- `requirements.txt` - Dependencies

### Performance Optimization

---

**AI Commentary Strategy:**

**End of Recommendations** • Status: Phase 6 Complete • Next: Database Optimization- Full AI commentary for top 10 signals (by weighted_score)

- Basic commentary for remaining signals
- Result: 73% reduction in OpenAI API calls, 52.7% faster pipeline

---

## 📊 Data Quality Standards

### Validation Rules (tables.py)

**Numeric Columns:**
- Zero detection and percentage
- Negative value detection (context-aware)
- Min/max/average ranges
- Suspicious patterns (e.g., all zeros in price columns)

**Text Columns:**
- Empty string detection
- NULL vs empty differentiation
- Length statistics

**Quality Thresholds:**
- ⚠️ Warning: >20% zero values in non-zero columns
- ⚠️ Warning: Negative values in count/volume columns
- ⚠️ Warning: >10% empty strings in required text fields

---

## 🚀 Next Actions (Immediate)

### Ready to Implement (User Decisions Made)

1. **Start Phase A: Backtest System** (4 hours)
   - Activate BacktestScheduler
   - Add 1d/3d/7d/14d return tracking
   - Implement auto-run after pipeline
   - Calculate historical_success_rate

2. **Start Phase B: Technical Indicators** (3 hours)
   - Add 9 missing yfinance indicators
   - Create technical_indicators.py for TA-Lib
   - Implement sector relative strength

3. **Start Phase C: Fundamental Data** (3 hours)
   - Enhance yfinance fundamental scraping
   - Add earnings momentum signal detection
   - Implement analyst targets

4. **Critical: Phase H - Financial Score Redesign** (2 hours)
   - Refactor `_calculate_financial_score()`
   - Ensure ALL technical indicators contribute
   - Implement proper weighting system

### Files to Clean Up

**Delete temporary analysis files:**
- phase1_column_analysis.py
- check_actual_columns.py
- check_empty_columns.py
- analyze_empty_columns.py
- run_phase1_1_migration.py
- run_phase1_2_migration.py
- run_phase2_migration.py
- fix_numeric_columns.py
- list_db_objects.py
- analyze_signals_columns.py

**Delete obsolete migrations:**
- migrations/phase1_1_document_columns.sql (already executed)
- migrations/phase1_2_remove_unused_columns.sql (not using - keeping all columns)
- migrations/phase1_2_remove_unused_columns_verified.sql (not using)

**Delete old documentation:**
- PHASE_0_COMPLETE.md
- PHASE_1_ANALYSIS_RESULTS.md
- PHASE_1_COMPLETE.md

**Keep for reference:**
- IMPLEMENTATION_QUESTIONS.md (decisions made)
- IMPLEMENTATION_PLAN.md (detailed roadmap)
- migrations/phase2_add_performance_tracking.sql (will use)

---

## 📝 Implementation Notes

### User Preferences Summary

**Data Collection Strategy:**
- Free APIs only during testing phase
- Upgrade to premium APIs after core system proven
- No backfilling during testing (tables cleared frequently)
- Focus on swing/long trading indicators

**Technical Analysis:**
- All 9 missing indicators needed
- TA-Lib integration approved
- Sector comparison important

**Risk Management:**
- Risk-adjusted ranking desired
- Sharpe ratio and volatility tracking needed
- Liquidity warnings important

**Options & Short Interest:**
- Start with yfinance basic options data
- Short squeeze detection high priority
- Upgrade to premium options API later

**AI & ML:**
- ML predictions desired but low priority
- Chart patterns not needed now
- Focus on core data collection first

---

## 🔍 Monitoring & Validation

### Success Metrics

After implementation, system should have:
- ✅ Auto-backtesting with multi-interval returns
- ✅ 29 technical indicators in financial_score
- ✅ Complete fundamental analysis
- ✅ Risk-adjusted signal ranking
- ✅ Options flow sentiment
- ✅ Short squeeze detection
- ✅ 64 empty columns populated with real data

### Testing Strategy

**During Implementation:**
- Tables cleared frequently (testing phase)
- No backfilling historical data
- Focus on new signal generation
- Validate each phase before moving to next

**Quality Checks:**
- Run `python tables.py --detailed` after each phase
- Verify no NULL values in newly implemented columns
- Validate score correlations with returns
- Monitor API rate limits and costs

---

## � Data Collection Issues (From tables.py Analysis)

### High Priority (>80% NULL Rate)

**Technical Indicators (80.2% NULL):**
- `signals.macd_histogram`
- `signals.macd_signal`
- `signals.macd_line`
- `signals.bollinger_upper`
- `signals.bollinger_lower`
- `signals.bollinger_width`
- `signals.bollinger_position`
- **Issue**: yfinance may not return these for all tickers or timeframes
- **Fix**: Investigate yfinance data availability, consider calculating manually from price history

**Options Data (89.9% NULL):**
- `signals.put_call_oi_ratio`
- `signals.put_call_vol_ratio`
- **Issue**: Options data not available for all tickers through yfinance free tier
- **Fix**: Consider premium API or defer to future phase

**Institutional Holdings (82.4% NULL):**
- `signals.retail_holding_pct`
- `signals.institutional_ownership_pct`
- **Issue**: Not all tickers report institutional holdings
- **Fix**: May require premium data source

**Float Turnover (94.7% NULL):**
- `signal_metrics.float_turnover_ratio`
- **Issue**: Calculation missing or data unavailable
- **Fix**: Calculate from volume / shares_float if data available

### Medium Priority (Constant Values - Code Bugs)

**News Integration (Intentionally Disabled):**
- `news_score = 0.0`
- `news_mentions = 0`
- `news_sentiment_score = 0.0`
- **Status**: News API disabled due to rate limits
- **Future**: Re-enable with combined commentary field

**Beta Calculation (Needs Fix):**
- `beta = 1.0` (100% constant)
- **Issue**: Not calculating actual beta from price history
- **Fix**: Use yfinance `.info['beta']` or calculate from returns vs SPY

**Reddit Data (Missing Field):**
- `upvotes = 0` (100% constant)
- **Issue**: Scraper not capturing upvote counts from Reddit posts
- **Fix**: Add upvotes field to Reddit API extraction

**Insider Trading (Not Implemented):**
- `insider_buy_count = 0`
- `insider_sell_count = 0`
- `insider_net_shares = 0`
- `insider_activity_score = 50.0` (default)
- `institutional_change_qoq = -95.2` (broken calculation)
- **Issue**: yfinance may not provide insider transaction data
- **Fix**: Check yfinance support, else defer to premium API

**Exchange Field (Incorrect):**
- `company_tickers.exchange = "NYSE"` (100% constant)
- **Issue**: All tickers marked as NYSE, should vary (NYSE, NASDAQ, OTC, etc.)
- **Fix**: Extract from yfinance `.info['exchange']` field

### Low Priority (Low Variance - Expected)

**Acceptable Constants:**
- `signals.risk_level = "High"` (0.3% unique) - Most signals are high risk
- `signals.sector` (1.0% unique) - Reddit focuses on tech sector
- `signals.backtest_phase = "Complete"` (100%) - After Phase 4 execution
- `signals.expected_hold_duration = 5` - Reasonable default for swing trading
- `ai_strategies.ai_model_version = "gpt-4o-mini"` - Single model currently
- `runs.status = "completed"` - Error handling works (no errors captured)

**Investigate:**
- `signals.top_factors = "Reddit mentions, price momentum"` (100% same) - Too uniform?
- `signals.signal_type = "Multi-Factor"` (100%) - Should have single-factor signals?
- `signals.trade_type` (0.4% unique) - Very limited variety
- `signals.backtest_timestamp` (0.2% unique) - All backtested at once (expected after batch run)

---

## �📚 Reference Documents

- **[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)** - Detailed phase-by-phase implementation guide
- **[SCHEMA_IMPROVEMENTS.md](../SCHEMA_IMPROVEMENTS.md)** - Complete tables.py analysis and cleanup plan
- **[PHASE4_COMPLETE_SUMMARY.md](../PHASE4_COMPLETE_SUMMARY.md)** - Phase 4 backtest results
- **[IMPLEMENTATION_QUESTIONS.md](../IMPLEMENTATION_QUESTIONS.md)** - User decisions and rationale
- **[OPERATIONAL_GUIDELINES.md](../OPERATIONAL_GUIDELINES.md)** - Coding standards and best practices

---

**End of Recommendations** • Updated: 2025-10-04
