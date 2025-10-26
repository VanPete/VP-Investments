# VP Investments - Operational Guidelines
*Development Framework for 10-Phase Modular Architecture*

**Last Updated:** 2025-10-25  
**Status:** ✅ Phases 1-6 Complete (60%) - Production Ready with Enhancement Pipeline

---

## 🚀 10-Phase Architecture Overview

### **Current Status: 6/10 Phases Complete (60%)**

```
Phase 1: Fetch        ✅ COMPLETE - Data sources (YFinance, Reddit, News)
Phase 2: Calculate    ✅ COMPLETE - 143 factors across 6 groups  
Phase 3: Normalize    ✅ COMPLETE - MAD-based z-scores
Phase 4: Score        ✅ COMPLETE - Weighted multi-factor scoring
Phase 5: Persist      ✅ COMPLETE - Database storage (signals, metrics, performance)
Phase 6: Backtest     ✅ COMPLETE - Performance tracking (19 columns, 7 intervals)
Phase 7: ML           📋 PLANNED  - Machine learning (Q4 2025)
Phase 8: AI           📋 PLANNED  - AI enhancement (Q4 2025)
Phase 9: Reports      📋 PLANNED  - Reporting & alerts (Q1 2026)
Phase 10: Polish      📋 PLANNED  - Production ready (Q1 2026)
```

**See PHASE_SUMMARY.md for complete architecture documentation.**

---

## ✅ Completed Phases (1-6)

### **Phase 1: Data Fetch** ✅ COMPLETE
**Purpose:** Gather raw market data from multiple sources  
**Location:** `backend/phases/phase1_fetch.py`

**Features:**
- YFinance integration (price, volume, options, fundamentals)
- Reddit sentiment (wallstreetbets, stocks, investing)
- News API integration (company-specific news)
- Earnings calendar data
- Historical price data with validation

**Database Impact:**
- Raw data cached for 15 minutes
- No direct database writes (data passed to Phase 2)

---

### **Phase 2: Factor Calculation** ✅ COMPLETE
**Purpose:** Calculate 143 quantitative factors across 6 domains  
**Location:** `backend/phases/phase2_calculate.py`

**Factor Groups (143 total):**
1. **Technical (35 factors):** RSI, MACD, Bollinger Bands, momentum, volume analysis
2. **Fundamental (38 factors):** P/E, EPS growth, margins, ROE, debt ratios, liquidity
3. **News/Macro (17 factors):** Sentiment scores, article counts, earnings catalysts
4. **Social (13 factors):** Reddit mentions, sentiment, velocity, contrarian signals
5. **Risk (18 factors):** Volatility, beta, drawdown, Sharpe ratio, ATR, bid-ask spread
6. **Institutional (22 factors):** Analyst ratings, price targets, insider trading, ownership

**Features:**
- Per-factor error handling (graceful degradation)
- Zero-variance detection
- Missing data handling
- Input validation

**Database Impact:**
- Factors stored in `signal_metrics` table
- 143 columns for raw factor values
- Calculation metadata (timestamp, coverage %)

---

### **Phase 3: Normalization** ✅ COMPLETE
**Purpose:** Normalize factors to z-scores for consistent comparison  
**Location:** `backend/phases/phase3_normalize.py`

**Features:**
- MAD-based z-score calculation (robust to outliers)
- Extreme value clipping (±3 standard deviations)
- Zero-variance handling (avoid division by zero)
- Group-level normalization (within factor groups)

**Formula:**
```python
z_score = (value - median) / (MAD * 1.4826)
z_score_clipped = clip(z_score, -3, +3)
```

**Database Impact:**
- Z-scores stored in `signal_metrics` table
- 143 normalized factor columns
- Normalization metadata (median, MAD values)

---

### **Phase 4: Scoring & Assembly** ✅ COMPLETE
**Purpose:** Calculate weighted scores and assemble final signal  
**Location:** `backend/phases/phase4_score.py`

**Scoring Components (100% total):**
1. **Technical Score (25%)** - Price momentum, volume, volatility
2. **Fundamental Score (25%)** - Valuation, profitability, growth
3. **News/Macro Score (20%)** - Sentiment, catalysts, market conditions
4. **Social Score (15%)** - Reddit sentiment, mentions, velocity
5. **Risk Score (10%)** - Volatility, beta, drawdown
6. **Institutional Score (5%)** - Analyst ratings, insider trading

**Features:**
- 100% factor coverage (all 143 factors weighted)
- Group-level aggregation (median z-score per group)
- Weighted combination into final score
- Score validation (-3 to +3 range)

**Database Impact:**
- Component scores stored in `signals` table
- Final `signal_score` column
- Score breakdown for transparency

---

### **Phase 5: Database Persistence** ✅ COMPLETE
**Purpose:** Store signals, metrics, and performance data  
**Location:** `backend/phases/phase5_persist.py`

**Database Tables:**
1. **signals** - Main signal data (ticker, score, components, metadata)
2. **signal_metrics** - Detailed factor values (143 raw + 143 normalized)
3. **signal_performance** - Tracking table for backtesting

**Features:**
- Transaction-based writes (rollback on failure)
- Duplicate detection (same ticker + timeframe)
- Constraint validation (score ranges, data types)
- Error logging and recovery

**Database Columns:**
- `signals`: 25+ columns (ticker, score, components, timestamps)
- `signal_metrics`: 290+ columns (143 raw + 143 normalized factors)
- `signal_performance`: 50+ columns (returns, benchmarks, metadata)

---

### **Phase 6: Backtest Performance Tracking** ✅ COMPLETE
**Purpose:** Calculate historical performance for signal validation  
**Location:** `backend/phases/phase6_backtest.py`

**Performance Metrics:**
- **Baseline:** Next trading day's open price (signal creation + 1 day)
- **Returns:** 7 intervals (1d, 3d, 7d, 10d, 14d, 30d, 90d)
- **SPY Benchmark:** Same 7 intervals for comparison
- **Age-Based Calculation:** `eligible = signal_age >= interval + 1`

**Features:**
- Avoids lookahead bias (baseline = next day open)
- Incremental updates (calculates only eligible intervals)
- Error handling (delisted tickers, missing data)
- SPY benchmark tracking (relative performance)

**Database Impact (19 new columns):**
- `backtest_baseline_price`, `backtest_baseline_date`
- `return_1d`, `return_3d`, `return_7d`, `return_10d`, `return_14d`, `return_30d`, `return_90d`
- `spy_return_1d`, `spy_return_3d`, `spy_return_7d`, `spy_return_10d`, `spy_return_14d`, `spy_return_30d`, `spy_return_90d`
- `backtest_status`, `backtest_last_update`, `backtest_error`

**Migration:** `migrations/migration_003_add_backtest_columns.py`

---

## 📋 Planned Phases (7-10)

### **Phase 7: Machine Learning** 📋 PLANNED (Q4 2025)
**Purpose:** ML-based signal enhancement and prediction

**Planned Features:**
- Feature engineering from 143 factors
- Gradient boosting models (XGBoost, LightGBM)
- Prediction: Signal success probability
- Model versioning and retraining pipeline
- Validation: Out-of-sample backtesting

**Database Impact:**
- `ml_predictions` table (probabilities, confidence)
- `ml_models` table (version tracking, metadata)
- New columns in `signals`: `ml_score`, `ml_confidence`

---

### **Phase 8: AI Enhancement** 📋 PLANNED (Q4 2025)
**Purpose:** AI-generated narratives and insights

**Planned Features:**
- OpenAI GPT-4 integration
- Risk narratives (trade rationale, key factors)
- Natural language explanations
- Template-based fallback system
- Cost optimization (caching, batching)

**Database Impact:**
- New columns in `signals`: `risk_narrative` (TEXT), `ai_insights` (JSONB)
- `ai_generations` table (tracking, costs, quality metrics)

---

### **Phase 9: Reporting & Alerts** � PLANNED (Q1 2026)
**Purpose:** User-facing reports and notifications

**Planned Features:**
- Daily signal summaries (email, webhook)
- Performance dashboards (Streamlit)
- Custom alert rules (score thresholds, specific tickers)
- Historical performance reports
- PDF export for signals

**Database Impact:**
- `user_preferences` table (alert settings, watchlists)
- `report_history` table (tracking sent reports)

---

### **Phase 10: Production Polish** 📋 PLANNED (Q1 2026)
**Purpose:** Production-ready deployment and monitoring

**Planned Features:**
- CI/CD pipeline (GitHub Actions)
- Monitoring and alerting (Sentry, Datadog)
- Rate limiting and caching optimizations
- API versioning
- Comprehensive logging and observability

**Database Impact:**
- `system_health` table (uptime, performance metrics)
- `api_usage` table (rate limiting, quotas)

---

## 🎯 CRITICAL: Phase 4 Scoring System (MANDATORY)

### Signal Scoring Architecture
**ALL new features and enhancements MUST integrate into the Phase 4 six-component scoring system.**

#### The 6 Scoring Components (100% Total):

1. **Technical Score (25%)** - `technical_score`
   - Price momentum, RSI, MACD, Bollinger Bands, moving averages
   - Volume analysis, volatility metrics
   - Chart patterns, support/resistance levels
   - **New technical indicators → Add to technical_score calculation**

2. **Fundamental Score (25%)** - `fundamental_score`
   - P/E ratio, PEG ratio, price-to-book
   - EPS growth, revenue growth, margins (profit, FCF)
   - ROE, debt-to-equity, dividend yield
   - Earnings surprises and trends
   - **New fundamental metrics → Add to fundamental_score calculation**

3. **News/Macro Score (20%)** - `news_macro_score`
   - News sentiment analysis, article mentions
   - Macro economic indicators
   - Sector-specific news sentiment
   - **New news sources or macro data → Add to news_macro_score calculation**

4. **Social/Alternative Score (15%)** - `social_alternative_score`
   - Reddit sentiment and mentions
   - Twitter/StockTwits sentiment
   - Google Trends data
   - Retail investor sentiment
   - **New social data sources → Add to social_alternative_score calculation**

5. **Risk/Stability Score (10%)** - `risk_stability_score`
   - Beta, volatility, Sharpe ratio
   - Maximum drawdown, liquidity metrics
   - Options activity (put/call ratios, IV)
   - Short interest metrics
   - **New risk metrics → Add to risk_stability_score calculation**

6. **Institutional/Smart Money Score (5%)** - `institutional_smart_money_score`
   - Institutional ownership changes
   - Insider trading activity (buys/sells)
   - Analyst ratings and price targets
   - Hedge fund holdings
   - **New institutional data → Add to institutional_smart_money_score calculation**

#### Final Signal Score Calculation:
```python
signal_score = (
    technical_score * 0.25 +
    fundamental_score * 0.25 +
    news_macro_score * 0.20 +
    social_alternative_score * 0.15 +
    risk_stability_score * 0.10 +
    institutional_smart_money_score * 0.05
)
```

### Feature Integration Rules (MANDATORY):

**When adding ANY new feature or data source:**

1. **Identify Component Category**
   - Determine which of the 6 components the feature belongs to
   - If unclear, ask which component it should enhance

2. **Update Score Calculation**
   - Add feature to the appropriate component's calculation in `backend/core/signals.py`
   - Method: `_calculate_signal_score_v2()` and component-specific methods
   - Ensure proper normalization (0.0 to 1.0 scale)

3. **Add Database Column**
   - Add raw feature data column to `signals` table schema
   - Place in the appropriate GROUP comment section (1-6)
   - Include CHECK constraint if bounded
   - Create migration file in `migrations/`

4. **Update Documentation**
   - Add feature to the component's column list in schema comments
   - Document calculation methodology
   - Update `docs/recommendations.md`

5. **Test Impact**
   - Verify component score changes appropriately
   - Confirm overall `signal_score` reflects new feature
   - Check data quality and NULL handling

### Database Schema Organization:
```sql
-- signals table structure (signals_schema_v2.sql)
CREATE TABLE signals (
    -- Core identification...
    
    -- SCORING COLUMNS (Phase 7)
    signal_score NUMERIC (0.0-1.0),              -- Final weighted score
    technical_score NUMERIC (0.0-1.0),           -- 25%
    fundamental_score NUMERIC (0.0-1.0),         -- 25%
    news_macro_score NUMERIC (0.0-1.0),          -- 20%
    social_alternative_score NUMERIC (0.0-1.0),  -- 15%
    risk_stability_score NUMERIC (0.0-1.0),      -- 10%
    institutional_smart_money_score NUMERIC (0.0-1.0),  -- 5%
    
    -- GROUP 1: TECHNICAL (25%) - 22 columns
    rsi, macd, bollinger_position, volume_spike_ratio, ...
    
    -- GROUP 2: FUNDAMENTAL (25%) - 17 columns
    pe_ratio, eps_growth, roe, earnings_surprise_trend, ...
    
    -- GROUP 3: NEWS/MACRO (20%) - 3+ columns
    news_sentiment_score, news_mentions, top_factors, ...
    
    -- GROUP 4: SOCIAL/ALTERNATIVE (15%) - 5+ columns
    reddit_sentiment, mentions, upvotes, comment_count, ...
    
    -- GROUP 5: RISK/STABILITY (10%) - 16+ columns
    beta, volatility, max_drawdown_risk, short_pct_float, ...
    
    -- GROUP 6: INSTITUTIONAL (5%) - 14+ columns
    institutional_ownership_pct, insider_buy_count, analyst_target_price, ...
);
```

### Code Implementation Location:
- **Primary Calculation:** `backend/core/signals.py`
  - Method: `_calculate_signal_score_v2()`
  - Component methods: `_calculate_technical_score()`, `_calculate_fundamental_score()`, etc.
- **Data Collection:** `backend/integrations/*` (feature-specific integration)
- **Schema:** `migrations/signals_schema_v2.sql`

### Examples:

**Example 1: Adding New Technical Indicator (RSI 14-day)**
```python
# 1. Add to technical_score calculation in signals.py
def _calculate_technical_score(self, data: Dict) -> float:
    rsi_14 = data.get('rsi_14', 50)  # New feature
    rsi_score = self._normalize_rsi(rsi_14)  # Normalize to 0-1
    
    # Combine with other technical factors
    technical_score = (
        rsi_score * 0.15 +        # New RSI weight
        momentum_score * 0.20 +
        macd_score * 0.15 +
        # ... other technical factors
    )
    return technical_score

# 2. Add database column
ALTER TABLE signals ADD COLUMN rsi_14 NUMERIC;

# 3. Update schema comment
COMMENT ON COLUMN signals.rsi_14 IS 'RSI 14-day period (0-100)';
```

**Example 2: Adding Earnings Call Transcript Sentiment**
```python
# 1. Determine category: News/Macro (20%)
def _calculate_news_macro_score(self, data: Dict) -> float:
    earnings_call_sentiment = data.get('earnings_call_sentiment', 0.5)
    
    news_macro_score = (
        news_sentiment * 0.50 +
        earnings_call_sentiment * 0.30 +  # New feature
        macro_indicators * 0.20
    )
    return news_macro_score

# 2. Add database column in GROUP 3 section
ALTER TABLE signals ADD COLUMN earnings_call_sentiment NUMERIC 
    CHECK (earnings_call_sentiment >= 0 AND earnings_call_sentiment <= 1);
```

### Questions to Ask Before Implementation:
1. **Which of the 6 components does this feature belong to?**
2. **How should it be weighted within that component?**
3. **What's the data range? (Need normalization?)**
4. **What's the fallback value if data unavailable?**
5. **Will this improve signal quality or add noise?**

---

## 👥 Role Definition

### User Role: Project Orchestrator
- Provides ideas, features, and strategic direction
- Makes architectural decisions and approves changes
- Reviews results and guides project priorities
- Defines business logic and requirements

### AI Agent Role: Implementation Engineer
- Implements features based on user requirements
- Makes tactical coding decisions within guidelines
- Tests and validates changes thoroughly
- Uses `tables.py` to understand database schema before making changes
- Updates documentation after implementations

**Key Principle**: User provides the "what" and "why" - AI determines the "how"

---

## 🎯 Project Structure Overview

### Backend Architecture (Core Development Area)
```
backend/                    # Main consolidated codebase - MODIFY EXISTING FILES ONLY
├── api/                   # API endpoints and web services
│   ├── api.py            # Main API implementation
│   └── __init__.py
├── core/                 # Core business logic and configuration
│   ├── backtest.py       # Backtesting engine
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration management
│   ├── core.py           # Core enums and exceptions
│   ├── intelligence.py   # AI intelligence processing
│   ├── signals.py        # Signal processing logic
│   └── __init__.py
├── integrations/         # External service integrations
│   ├── ai.py             # AI service integration (OpenAI)
│   ├── ai_strategy_generator.py  # AI strategy generation
│   ├── backtest.py       # Integration-specific backtesting
│   ├── news.py           # News data integration
│   ├── production.py     # Production environment setup
│   ├── reddit.py         # Reddit scraping integration (PRAW)
│   ├── scheduler.py      # Task scheduling
│   ├── signal_processing.py  # Signal enhancement processing
│   ├── yfinance.py       # Yahoo Finance integration
│   └── __init__.py
├── storage/              # Supabase database interactions
│   ├── database.py       # Database operations and connections
│   └── __init__.py
├── utils/                # Utility functions and helpers
│   ├── logger.py         # Logging utilities
│   ├── observability.py  # Monitoring and observability
│   └── __init__.py
├── pipeline.py           # Main UnifiedPipeline - PRIMARY ENTRY POINT
├── py.typed              # Type checking marker
└── __init__.py
```

### Root Directory (Testing & Temporary Files)
```
root/                      # TEMPORARY FILES ONLY - Will be deleted when project complete
├── *.py                  # Temporary test scripts (allowed)
├── *.bat                 # Utility batch files (allowed)
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project configuration
├── migrations/           # Database schema migrations
└── [other temp files]    # Testing utilities, one-off scripts
```

### Documentation (docs/)
```
docs/                      # Project documentation - READ BEFORE DEVELOPMENT
├── recommendations.md    # ⭐ SINGLE SOURCE OF TRUTH - ALL recommendations go here
├── operational_guidelines.md  # This file - development framework
└── [additional docs]     # Technical documentation and guides
```

**CRITICAL DOCUMENTATION RULE:**
- ⭐ **ALL recommendations, priorities, and status updates MUST go in `docs/recommendations.md`**
- **NEVER create separate recommendation files** (e.g., OPTIMIZATION_RECOMMENDATIONS.md, PRIORITY_GUIDE.md)
- **NEVER put recommendations in root directory**
- `recommendations.md` is a **living document** - update it as project progresses
- This is the **ONLY place** for recommendations, priorities, and implementation guidance

---

## 🏗️ Architectural Principles

### 1. Single Source of Truth
**Principle:** One primary data source, all other representations derive from it

**Implementation:**
- `signals` table is the primary source for all signal data
- `signals_norm` is a materialized view derived from `signals`
- Pipeline writes **only to `signals`**, views refresh automatically
- No duplicate data storage

**Example:**
```python
# ❌ OLD WAY: Duplicate writes
await db.insert_signals(signals)
await db.insert_signals_norm(normalized_signals)  # Duplicate data

# ✅ NEW WAY: Single write, derived view
await db.insert_signals(signals)
await db.refresh_materialized_view('signals_norm')  # Derives from signals
```

### 2. Commentary Consolidation
**Principle:** Unified commentary field for frontend simplification

**Implementation:**
- Single `commentary` field in `signals` table
- Structured format: **Signal Analysis** → **Market Insights** → **Key Metrics**
- `commentary_metadata` JSONB field tracks generation details
- Backward compatible (old fields kept during migration)

**Example:**
```python
# Backend generates unified commentary
commentary = self._generate_unified_commentary(
    signal=signal_data,
    score_explanation=basic_explanation,
    ai_commentary=ai_analysis  # Optional, only for top signals
)

# Frontend uses single field
signal.commentary  # Complete, formatted commentary
```

### 3. Top-N AI Commentary Pattern
**Principle:** Full AI commentary for highest-value signals only

**Implementation:**
- Top 10 signals (sorted by `weighted_score`) get full AI commentary
- Remaining signals get basic commentary (no AI call)
- 73% reduction in API calls, 52.7% performance improvement

**Example:**
```python
# Sort signals by weighted_score
top_signals = sorted(signals, key=lambda s: s.weighted_score, reverse=True)[:10]

# Full AI commentary for top 10
for signal in top_signals:
    signal.ai_commentary = await ai_service.generate_commentary(signal)

# Basic commentary for remaining signals
for signal in remaining_signals:
    signal.commentary = generate_basic_commentary(signal)
```

### 4. Materialized View Pattern
**Principle:** Derived views for performance without data duplication

**Implementation:**
- Use materialized views for expensive queries
- Refresh after data changes (automated in pipeline)
- Indexes on materialized views for query performance

**Example:**
```sql
-- Create materialized view
CREATE MATERIALIZED VIEW signals_norm AS
SELECT id, ticker, signal_type, weighted_score, ...
FROM signals;

-- Create indexes for performance
CREATE INDEX idx_signals_norm_ticker ON signals_norm(ticker);

-- Refresh after pipeline runs
REFRESH MATERIALIZED VIEW signals_norm;
```

---

## 🚀 Development Rules & Constraints

### ✅ ALLOWED Operations

#### 1. Backend File Modifications
- **EXTEND existing files** in `backend/` when adding features
- **MODIFY existing functions/classes** to fix issues or enhance functionality
- **ADD new functions/methods** to existing backend files
- **UPDATE imports** within backend files to maintain consistency
- **FOLLOW architectural principles** (single source of truth, commentary consolidation)

#### 2. Root Directory Usage
- **CREATE temporary test scripts** for feature validation
- **WRITE utility scripts** for one-off operations (e.g., `refresh_signals_norm.py`)
- **ADD batch files** for automation during development
- **CREATE test data files** for validation purposes
- **ADD migration files** in `migrations/` directory

#### 3. Testing & Validation
- **ALWAYS test using existing pipeline**: `from backend.pipeline import UnifiedPipeline`
- **RUN integration tests** after modifications
- **VALIDATE database connections** through `backend.storage.database`
- **TEST API endpoints** via `backend.api.api`
- **USE utility scripts** for validation (e.g., `tables.py --detailed`)

#### 4. Database Operations
- **CREATE migrations** for schema changes in `migrations/` directory
- **USE CASCADE** when dropping dependent objects
- **ADD indexes** for performance-critical queries
- **REFRESH materialized views** after data changes
- **VALIDATE data quality** with quality check scripts

### ❌ RESTRICTED Operations

#### 1. Backend Structure Changes
- **NO new files** in `backend/` without explicit approval
- **NO directory restructuring** within backend
- **NO deletion** of existing backend files
- **NO breaking changes** to core interfaces
- **NO duplicate data storage** (follow single source of truth)

#### 2. Data Architecture Violations
- **NO writing to derived views/tables** (only to primary sources)
- **NO data duplication** across tables
- **NO bypassing materialized view refresh** after writes
- **NO direct manipulation of signals_norm** (it's a view)

#### 3. Dependency Management
- **ASK before adding** new Python packages
- **CONFIRM before modifying** existing integrations
- **VALIDATE compatibility** with Supabase storage layer
- **CONSIDER async alternatives** (e.g., Async PRAW instead of PRAW)

---

## 🛠 Development Workflow

### Feature Development Process
1. **Read Documentation**: Check `docs/recommendations.md` for current status and priorities
2. **Identify Target File**: Determine which existing backend file needs modification
3. **Follow Architectural Principles**: Apply single source of truth, commentary consolidation, etc.
4. **Extend Functionality**: Add new methods/functions to existing files
5. **Update Imports**: Ensure all imports use `backend.*` structure
6. **Create Migration**: If database changes needed, create migration in `migrations/`
7. **Test Integration**: Use `UnifiedPipeline` for end-to-end testing
8. **Validate Storage**: Confirm Supabase interactions work correctly
9. **⭐ UPDATE RECOMMENDATIONS.MD**: **ALWAYS** update `docs/recommendations.md` with changes, decisions, and status
   - Add new priorities to the Pending Priorities section
   - Update completed tasks in the appropriate sections
   - Document architectural decisions
   - Add questions/blockers that need resolution
   - **NEVER create separate recommendation files**

### Testing Protocol
```python
# Standard Testing Pattern (create in root as temporary file)
from backend.pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Test your modifications
await pipeline.run()  # Full pipeline test

# Or test specific step
signals = await pipeline._generate_signals()

# Validate results
print("✅ Backend modifications successful!")
```

### Database Migration Protocol
```sql
-- migrations/XXX_description.sql

-- Part 1: Schema Changes
ALTER TABLE table_name ADD COLUMN new_column TYPE;

-- Part 2: Data Backfill (if needed)
UPDATE table_name SET new_column = ...;

-- Part 3: Drop Dependencies with CASCADE
DROP VIEW IF EXISTS dependent_view CASCADE;

-- Part 4: Create Derived Objects
CREATE MATERIALIZED VIEW new_view AS ...;

-- Part 5: Performance Indexes
CREATE INDEX idx_name ON table_name(column_name);

-- Part 6: Validation
SELECT COUNT(*) FROM table_name WHERE new_column IS NULL;
```

### Code Modification Guidelines
- **File Selection**: Always extend existing backend files rather than creating new ones
- **Function Placement**: Add new functionality to the most appropriate existing module
- **Import Updates**: Maintain `backend.*` import structure throughout
- **Error Handling**: Follow existing error handling patterns in backend files
- **Commentary Generation**: Use `_generate_unified_commentary()` for all commentary
- **Database Writes**: Write to primary tables only, let views derive data
- **View Refresh**: Always refresh materialized views after primary table updates

---

## 📁 Primary Work Areas

### Core Business Logic
- `backend/core/` - Main business logic, configuration, CLI
- `backend/pipeline.py` - **PRIMARY ENTRY POINT** and workflow orchestration
  - Step 1: Reddit data scraping
  - Step 2: Signal preprocessing
  - Step 3: Signal validation
  - Step 4: Signal scoring and generation
  - Step 4.6: **Unified commentary generation** (top 10 only)
  - Step 5: AI strategy generation
  - Step 6: Database persistence and view refresh

### Data & Integrations  
- `backend/integrations/` - External service connections
  - `reddit.py` - Reddit scraping (PRAW) - **Consider migrating to Async PRAW**
  - `yfinance.py` - Yahoo Finance integration
  - `ai.py` - OpenAI integration (commentary, strategies)
  - `news.py` - News data integration
  - `backtest.py` - Backtesting integration
- `backend/storage/database.py` - **Supabase database operations**
  - All database writes go here
  - Materialized view refresh methods
  - Query optimization

### API & Services
- `backend/api/api.py` - Web endpoints and external interfaces
- `backend/utils/` - Logging, observability, and utility functions

---

## 🔄 Integration Points

### Database Operations (Supabase)
```python
from backend.storage.database import SupabaseStorage

# Initialize storage
storage = SupabaseStorage()

# Write to primary table (signals)
await storage.insert_signals(signals)

# Refresh derived view (signals_norm)
await storage.refresh_materialized_view('signals_norm')

# ❌ Don't write to derived views
# await storage.insert_signals_norm(...)  # WRONG!
```

### Pipeline Integration
```python
from backend.pipeline import UnifiedPipeline

# Initialize and run
pipeline = UnifiedPipeline()
await pipeline.run()

# Access specific steps
signals = await pipeline._generate_signals()
commentary = pipeline._generate_unified_commentary(signal)
```

### External Services
```python
from backend.integrations.reddit import RedditIntegration
from backend.integrations.yfinance import YFinanceIntegration
from backend.integrations.ai import AIIntegration

# Service integrations through dedicated modules
reddit = RedditIntegration()
yfinance = YFinanceIntegration()
ai = AIIntegration()
```

### Commentary Generation
```python
# In backend/pipeline.py
def _generate_unified_commentary(
    self,
    signal: Dict[str, Any],
    score_explanation: str,
    ai_commentary: Optional[str] = None
) -> str:
    """Generate unified commentary following architectural pattern"""
    
    # Structured format
    sections = []
    
    # Signal Analysis (always present)
    sections.append(f"📊 **Signal Analysis**\n{score_explanation}")
    
    # Market Insights (if AI commentary available)
    if ai_commentary:
        sections.append(f"\n\n🔍 **Market Insights**\n{ai_commentary}")
    
    # Key Metrics (always present)
    sections.append(f"\n\n📈 **Key Metrics**\n- Score: {signal['weighted_score']}")
    
    return "\n".join(sections)
```

---

## ⚡ Quick Reference Commands

### Test Pipeline Integration
```bash
cd "c:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"
python -c "from backend.pipeline import UnifiedPipeline; pipeline = UnifiedPipeline(); print('✅ Pipeline operational')"
```

### Validate Backend Structure
```bash
python -c "import backend; print('✅ Backend imports successful')"
```

### Run Full Pipeline Test
```bash
python -m backend.pipeline
```

### Validate Data Quality
```bash
python tables.py --detailed
```

### Refresh Materialized Views
```bash
python refresh_signals_norm.py
```

### Clear Test Data (Safe)
```bash
python safe_clear_data.py
```

---

## 📊 Current Implementation Status

### Completed Architectural Patterns
- ✅ **Single Source of Truth**: signals table primary, signals_norm derived
- ✅ **Commentary Consolidation**: Unified commentary field with metadata
- ✅ **Top-N AI Pattern**: Top 10 signals get full AI commentary
- ✅ **Materialized View Pattern**: signals_norm as materialized view

### Active Development Areas
- 🔄 **Backtest Integration**: Populating performance tables (Priority #4)
- 🔄 **Performance Optimization**: Caching, parallel processing, async PRAW
- 🔄 **Data Quality**: Enhanced validation and monitoring

### Pending Architectural Decisions
- ⏳ **Frontend Migration**: Transition to use `commentary` field
- ⏳ **View Freshness Monitoring**: Automated alerts for stale views
- ⏳ **Backtest Scheduling**: Automated performance tracking
- ⏳ **Market Conditions**: Contextual data population

---

## 🚨 Important Reminders

### 📝 Documentation (CRITICAL)
- **⭐ SINGLE RECOMMENDATIONS FILE**: ALL recommendations MUST go in `docs/recommendations.md`
- **NO SEPARATE FILES**: Never create OPTIMIZATION_RECOMMENDATIONS.md, PRIORITY_GUIDE.md, etc.
- **UPDATE AFTER EVERY CHANGE**: Update recommendations.md with status, decisions, blockers
- **LIVING DOCUMENT**: recommendations.md evolves with the project

### Database Operations
- **PRIMARY TABLE ONLY**: Write to `signals`, not `signals_norm`
- **REFRESH VIEWS**: Always refresh materialized views after writes
- **CASCADE DROPS**: Use CASCADE when dropping objects with dependencies
- **VALIDATE CHANGES**: Run validation queries after migrations

### Code Architecture
- **SINGLE SOURCE**: One primary data source, derived representations only
- **UNIFIED COMMENTARY**: Use `_generate_unified_commentary()` method
- **TOP-N PATTERN**: Full AI for top signals, basic for others
- **NO DUPLICATION**: Never duplicate data across tables

### Development Process
- **READ DOCS FIRST**: Check `docs/recommendations.md` before starting
- **TEST THOROUGHLY**: Use `UnifiedPipeline` for integration testing
- **⭐ UPDATE RECOMMENDATIONS.MD**: **MANDATORY** after every change
- **ASK BEFORE**: Consult before major architectural changes

### Performance Considerations
- **MATERIALIZED VIEWS**: Use for expensive queries
- **INDEXES**: Add indexes for frequently queried columns
- **API LIMITS**: Top-N pattern for expensive API calls
- **ASYNC PREFERRED**: Use async libraries when available

---

## 🎨 Frontend Development

### Current Status: ✅ Phase 1 & 2 Complete - Production Deployed

**Project:** VanPIQ Signals Dashboard  
**Tech Stack:** Next.js 15.5.4, React 19, TypeScript, Tailwind CSS 4, shadcn/ui  
**Repository:** https://github.com/VanPete/VP-Investments  
**Deployment:** Vercel (automatic on push to main)  
**Last Commit:** f4b69b5 (Frontend polish: QuickStats, filters, spacing)  
**Build Status:** ✅ Passing (2.5s, 198 KB bundle, 0 errors)

### Design System (VanPIQ Branding)
- **Gradient:** `#001F3F` → `#00AEEF` (dark blue to bright cyan)
- **Typography:** Inter font family
- **Spacing:** rounded-2xl cards, shadow-lg elevation
- **Components:** Gradient hover states, backdrop-blur-sm effects
- **Philosophy:** "VanPiQ: Precision Intelligence" - calm, confident, data-driven

### Architecture & Data Flow
- **Backend Pipeline:** Python → Reddit/News discovery → Factor scoring → JSON output
- **Data Format:** JSON files with rankings, group scores, factor breakdowns
- **Current Data:** 29 tickers, Reddit: 28, News: 9, Total Universe: 36
- **Frontend:** Static site generation, reads JSON, renders dashboard
- **State Management:** React hooks + localStorage persistence

### Component Architecture

#### Layout Components
1. **Navigation** (`frontend/src/components/Navigation.tsx`)
   - Purpose: Site-wide navigation bar
   - Features: Centered links (Dashboard, Methodology), gradient underline on active
   - Height: h-12
   - Status: ✅ Complete

2. **DashboardHeader** (`frontend/src/components/dashboard/DashboardHeader.tsx`)
   - Purpose: Dashboard header with branding and controls
   - Layout: VanPIQ logo (120x40px, hover glow) → Title + discovery badge → Stats → Controls
   - Discovery Stats: Hover tooltip on ticker count shows Reddit/News breakdown
   - Controls: File selector dropdown + refresh button (gradient)
   - Status: ✅ Complete

#### Dashboard Components
3. **SignalsDashboard** (`frontend/src/components/dashboard/SignalsDashboard.tsx`)
   - Purpose: Main orchestrator component, manages all state
   - State:
     * usePersistedFilters: search, selectedGroup, selectedFactor, minCoverage, scoreRange
     * usePersistedColumnVisibility: 10 boolean flags for column visibility
   - Filter Logic:
     * **Group Filter:** Checks `if (filters.selectedGroup)`, filters by non-zero group scores
     * **Factor Filter:** Finds parent group for selected factor, filters by that group's scores
     * **Search:** Case-insensitive ticker matching
     * **Coverage:** `>= filters.minCoverage`
     * **Score Range:** min/max boundaries
   - Component Composition:
     * DashboardHeader
     * QuickStats (if results exist)
     * FilterPanel
     * FilterChips + ColumnVisibilityToggle (side-by-side flex)
     * SignalsTable
   - Status: ✅ Complete, all filters working

4. **QuickStats** (`frontend/src/components/dashboard/QuickStats.tsx`)
   - Purpose: Display 3 key metrics above table
   - Layout: 3-card grid (lg:grid-cols-3)
     * Card 1: Average Score (TrendingUp icon, 0.171 across 29 tickers)
     * Card 2: Top Performer (Target icon, gradient bg, SHOP 0.956, 92.3% coverage) - CENTER
     * Card 3: High Coverage (CheckCircle2 icon, 23/29 tickers >90%)
   - Removed: Discovery Sources card (4th card), Badge/Rss imports, unused variables
   - Status: ✅ Complete, streamlined

5. **FilterPanel** (`frontend/src/components/dashboard/FilterPanel.tsx`)
   - Purpose: Search and filter controls
   - Features:
     * Search input (debounced)
     * Group dropdown (Technical, Fundamental, News/Macro, Social, Risk, Institutional)
     * Factor dropdown (grouped by parent category)
     * Coverage slider (0-100%)
     * Score range sliders (min/max 0.0-1.0)
   - Status: ✅ Complete

6. **SignalsTable** (`frontend/src/components/dashboard/SignalsTable.tsx`)
   - Purpose: Main data table with expandable rows
   - Features:
     * Column visibility: Conditional rendering based on columnVisibility prop
     * Search highlighting: HighlightedText on ticker symbols
     * Interactive tooltips: MetricTooltip on all headers
     * Coverage badges: CoverageBadge with quality-tier tooltips
     * Expandable rows: ChevronDown/Right, detailed factor scores
     * Gradient progress bars: VanPIQ gradient on score bars
   - Columns: Rank, Ticker*, Overall Score*, Coverage, Technical, Fundamental, News/Macro, Social, Risk, Institutional (* = required)
   - Spacing: Equal auto-sizing (removed fixed widths except expand button w-12)
   - Status: ✅ Complete, equal spacing

#### Reusable Components
7. **FilterChips** (`frontend/src/components/dashboard/FilterChips.tsx`)
   - Purpose: Show active filters as dismissible badges
   - Features: Detects 4 filter types, gradient styling, X button with hover effect
   - Status: ✅ Complete

8. **CoverageBadge** (`frontend/src/components/dashboard/CoverageBadge.tsx`)
   - Purpose: Color-coded coverage quality indicators
   - Quality Tiers:
     * Excellent (≥90%): Green, CheckCircle2, "High" label
     * Good (70-89%): Yellow, AlertCircle
     * Limited (<70%): Red, XCircle
   - Tooltips: Optional, explain each tier's reliability
   - Status: ✅ Complete

9. **MetricTooltip** (`frontend/src/components/dashboard/MetricTooltip.tsx`)
   - Purpose: Reusable tooltip for metric explanations
   - Predefined: Rank, Overall Score, Coverage, Technical, Fundamental, News/Macro, Social, Risk, Institutional
   - Styling: HelpCircle icon, border-[#00AEEF]/30, 200ms delay
   - Status: ✅ Complete

10. **HighlightedText** (`frontend/src/components/dashboard/HighlightedText.tsx`)
    - Purpose: Highlight search terms with VanPIQ gradient
    - Highlight: `<mark>` with gradient bg from-[#001F3F]/20 to-[#00AEEF]/20
    - Features: Case-insensitive, regex escaping
    - Status: ✅ Complete

11. **ColumnVisibilityToggle** (`frontend/src/components/dashboard/ColumnVisibilityToggle.tsx`)
    - Purpose: Dropdown to show/hide table columns
    - Features:
      * 10 columns (Ticker/Overall Score required, 8 optional)
      * Presets: "Show All", "Essential Only"
      * Counter: "Columns (X/10)"
      * Persistence via usePersistedColumnVisibility
    - Status: ✅ Complete

#### Hooks
12. **usePersistedState** (`frontend/src/hooks/usePersistedState.ts`)
    - Purpose: localStorage persistence
    - Exports:
      * useLocalStorage<T>: Generic localStorage hook
      * usePersistedFilters: FilterState persistence
      * usePersistedColumnVisibility: ColumnVisibility persistence
      * usePersistedSort: For future use
      * usePersistedExpandedRow: For future use
      * clearAllPersistedData: Reset all preferences
    - Storage Keys: vanpiq_filters, vanpiq_column_visibility, vanpiq_sort, vanpiq_expanded_row
    - Status: ✅ Complete

### Feature Implementation Status

#### ✅ Completed - Phase 1 (7/10 features)
- [x] Discovery Source Badges (hover tooltip on ticker count)
- [x] Enhanced Number Contrast (text-lg font-bold Overall Score)
- [x] Coverage Badge Color Coding (green/yellow/red quality tiers)
- [x] Quick Stats Cards (3 cards: Average, Top Performer, High Coverage)
- [x] Filter Chips (dismissible badges with remove handlers)
- [x] Gradient Accents on Score Bars (VanPIQ gradient in expanded rows)
- [x] Header Restructuring (logo above title, discovery stats as tooltip)

#### ⏳ Pending - Phase 1 (3/10 features)
- [ ] Sorting Indicators (chevron icons on column headers)
- [ ] Discovery Source Column (Reddit/News badge per ticker)
- [ ] Loading States & Export Functionality (skeleton loaders, CSV/Excel buttons)

#### ✅ Completed - Phase 2 (4/5 features)
- [x] Interactive Tooltips (MetricTooltip on all headers + coverage badges)
- [x] Persistent User Preferences (filters + column visibility via localStorage)
- [x] Search Highlighting (HighlightedText with VanPIQ gradient)
- [x] Column Visibility Toggle (ColumnVisibilityToggle with presets)
- [x] Keyboard Shortcuts (skipped per user request)

#### ⏳ Pending - Phase 3 (3/3 features)
- [ ] Historical Data Comparison
- [ ] Signal Strength Heatmap
- [ ] Alert System & Watchlist

### Recent Fixes (Oct 22, 2024)
1. **QuickStats Layout:** Removed 4th card (Discovery Sources), changed to 3-card grid, reordered with Top Performer centered
2. **Group Filter:** Fixed broken filter logic (was checking `!== 'all'` instead of `if (selectedGroup)`)
3. **Factor Filter:** Implemented from scratch (was completely missing), finds parent group and filters by group scores
4. **Table Spacing:** Removed fixed width classes (w-[60px], w-[100px], w-[120px]) for equal column distribution

### Git Commits
- `fe9dbaf`: Phase 1 UI Polish (QuickStats, FilterChips, CoverageBadge, branding)
- `3f1ad79`: Phase 2 UX Enhancements (tooltips, persistence, search highlighting, column toggle)
- `f4b69b5`: Final polish (3-card QuickStats, fixed filters, equal spacing)

### Development Guidelines
- **Component Pattern:** Functional components with TypeScript, shadcn/ui primitives
- **State Management:** React hooks (useState, useMemo) + localStorage for persistence
- **Styling:** Tailwind utility classes, VanPIQ gradient throughout
- **Responsive:** Mobile-first with lg: breakpoints for desktop
- **Type Safety:** Strict TypeScript, proper prop interfaces
- **Conditional Rendering:** Based on data availability and user preferences

### Data Types & Interfaces

```typescript
// Core data types
interface SignalRanking {
  ticker: string;
  overall_score: number;
  coverage: number;
  group_scores: {
    technical: number;
    fundamental: number;
    news_macro: number;
    social: number;
    risk: number;
    institutional: number;
  };
  factor_scores: Record<string, number>; // 50+ individual factors
}

interface FilterState {
  searchQuery: string;
  selectedGroup: string;
  selectedFactor: string;
  minCoverage: number;
  scoreRange: [number, number];
}

interface ColumnVisibility {
  rank: boolean;
  ticker: boolean;        // Required, always true
  overallScore: boolean;  // Required, always true
  coverage: boolean;
  technical: boolean;
  fundamental: boolean;
  newsMacro: boolean;
  social: boolean;
  risk: boolean;
  institutional: boolean;
}
```

### localStorage Schema
- **vanpiq_filters**: FilterState object (search, group, factor, coverage, score range)
- **vanpiq_column_visibility**: ColumnVisibility object (10 boolean flags)
- **vanpiq_sort**: (future) Sort state
- **vanpiq_expanded_row**: (future) Expanded row persistence

### Known Issues & Technical Debt
1. **Factor Scores Missing:** Data only includes group_scores, not individual factor_scores per ticker
   - Workaround: Factor filter finds parent group and filters by group score
   - Backend TODO: Add factor_scores to JSON output
2. **Sorting Not Implemented:** No sort state or chevron indicators yet
3. **Discovery Column Missing:** Individual ticker discovery source badges not added
4. **Loading States:** No skeleton loaders or loading indicators
5. **Export Feature:** No CSV/Excel export functionality

### Future Enhancements (Phase 3+)
- Historical data comparison charts
- Signal strength heatmap visualization
- Alert system for score changes
- Watchlist with custom notifications
- Advanced filtering (multi-select groups, factor combinations)
- Performance optimizations (virtualized table for large datasets)

### Testing & Validation
- **Build:** `cd frontend && npm run build` (2.5s, 198 KB bundle)
- **Dev Server:** `npm run dev` (Turbopack)
- **Type Check:** `npx tsc --noEmit`
- **Lint:** ESLint integrated in build
- **Manual Testing:** All filters, tooltips, persistence tested in browser

### Deployment
- **Platform:** Vercel
- **Trigger:** Automatic on push to main branch
- **Build Command:** `npm run build`
- **Output:** Static site generation
- **Environment:** Production environment variables set in Vercel dashboard

### Development Commands
```bash
# Frontend development
cd frontend
npm install              # Install dependencies
npm run dev              # Start dev server (Turbopack)
npm run build            # Build for production
npm run start            # Serve production build

# Git workflow
git status --short       # Check changes
git add .                # Stage changes
git commit -m "message"  # Commit with message
git push origin main     # Deploy to Vercel
```

### File Locations
- **Components:** `frontend/src/components/dashboard/*.tsx`
- **Hooks:** `frontend/src/hooks/*.ts`
- **Types:** `frontend/src/types/*.ts`
- **Styles:** `frontend/src/app/globals.css`
- **Config:** `frontend/tailwind.config.ts`, `frontend/tsconfig.json`

### Dependencies
- Next.js 15.5.4
- React 19 (with react-dom)
- TypeScript 5.x
- Tailwind CSS 4
- shadcn/ui components (Card, Button, Select, Dropdown, Tooltip, Badge, Table, Slider, etc.)
- Lucide React (icons)
- class-variance-authority + clsx (utility styling)

---

## 📚 Additional Resources

### Documentation
- **Recommendations**: `docs/recommendations.md` - Living document with priorities
- **This File**: `docs/operational_guidelines.md` - Development framework
- **Migrations**: `migrations/` - Database schema change history

### Utility Scripts

#### tables.py - Database Schema Inspector (PRIMARY TOOL)
**Purpose:** Comprehensive Supabase schema inspection and analysis

**Usage:**
```bash
# Interactive mode (recommended for exploration)
python tables.py

# List all tables with row counts
python tables.py --list

# Show table schema
python tables.py --schema signals

# Analyze NULL coverage and data quality
python tables.py --nulls signals

# Get optimization recommendations
python tables.py --recommend

# Generate full report
python tables.py --report

# Export report to file
python tables.py --export schema_report.md
```

**When to Use:**
- ✅ Before making any database changes (understand current state)
- ✅ When planning migrations (identify issues)
- ✅ During debugging (check data quality)
- ✅ For documentation (generate schema reports)
- ❌ Do NOT create new check scripts - use tables.py instead

**Importable Functions** (for use in other scripts):
```python
from tables import (
    check_table_exists,      # Check if table exists
    get_row_count,           # Get table row count
    get_column_names,        # List all columns
    check_column_exists,     # Check if column exists
    get_table_schema,        # Get full schema details
    analyze_column_nulls     # Get NULL coverage stats
)

# Example usage
if check_table_exists('signals'):
    row_count = get_row_count('signals')
    columns = get_column_names('signals')
    print(f"signals table has {row_count} rows and {len(columns)} columns")
```

**Features:**
- Lists all tables with row counts and status
- Shows detailed schema (columns, types, constraints)
- Analyzes NULL coverage and data quality
- Identifies redundant/useless columns
- Detects constant values and low variance
- Recommends schema optimizations
- Exports reports in Markdown format

**Recommendations Engine:**
- ❌ DROP EMPTY TABLE - 0 rows, no data
- ❌ DROP NULL COLUMN - 100% NULL, no useful data
- ⚠️ CONSTANT COLUMN - 100% same value, verify calculation
- ⚠️ LOW VARIANCE - <5% unique values, check data quality
- ⚠️ HIGH NULL RATE - >80% NULL, improve data collection
- 🔄 REDUNDANT COLUMNS - combine or remove duplicates
- 🔄 CALCULATED COLUMN - can derive from other columns
- ❌ REDUNDANT TABLE - duplicates another table's data

#### Other Utility Scripts
- `refresh_signals_norm.py` - Manual materialized view refresh
- `safe_clear_data.py` - Safe data deletion for testing

### Key Files
- `backend/pipeline.py` - Main entry point (line 1861: `_generate_unified_commentary`)
- `backend/storage/database.py` - Database operations
- `migrations/001_uuid_and_commentary_fixes.sql` - Recent schema changes

---

*Follow these guidelines to maintain the consolidated backend structure while enabling flexible development within the established architectural framework. Always prioritize single source of truth, unified commentary, and materialized view patterns.*

**Last Updated:** 2025-10-04 - Added architectural principles and current implementation status
