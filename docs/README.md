# VP Investments - Trading Signal Analysis Platform

A comprehensive quantitative trading platform using **143 multi-factor analysis** across 6 domains with robust z-score normalization and weighted scoring. Built on v3.1 modular architecture with production-grade validation and error handling.

## ✨ **Latest Updates - v3.1 Architecture Complete!**

**v3.1 (October 2025): 10-Phase Modular Pipeline - 6/10 Complete (60%)**
- 🎯 **143 factors, 100% coverage** - All factors weighted and utilized in scoring
- 🏗️ **10-phase architecture** - Fetch → Calculate → Normalize → Score → Persist → Backtest → ML → AI → Reports → Polish
- 📊 **6 phases complete** - Phases 1-6 production ready (Fetch through Backtest)
- 🔬 **4 phases planned** - ML, AI, Reports, Polish (Q4 2025 - Q1 2026)
- 📈 **Performance tracking** - 7 return intervals (1d-90d) with SPY benchmark
- 🛡️ **4-layer validation** - Input validation, calculation error handling, normalization checks, score validation
- ⚙️ **Robust z-score normalization** - MAD-based with extreme value clipping and zero-variance detection
- ✅ **32/32 tickers validated** - 89.7% avg coverage, -0.32 to +0.71 score range

**Key Improvements from v3.0:**
- ✅ **100% factor coverage** (was 42.7% - only 61/143 factors weighted)
- ✅ **10-phase modular architecture** (was monolithic SignalScorer, now 6 complete + 4 planned)
- ✅ **Performance tracking** (Phase 6: 19 backtest columns, 7 return intervals, SPY benchmark)
- ✅ **Production error handling** (graceful degradation, per-factor try-catch)
- ✅ **Comprehensive validation** (input, calculation, normalization, scoring)
- ✅ **Extreme value handling** (z-score clipping, zero-variance detection)

## 🎯 **What It Does**

### **Multi-Factor Quantitative Analysis**
Analyzes stocks through **143 factors** across 6 domains:
- **Technical (35 factors)**: Price momentum, RSI, MACD, moving averages, Bollinger Bands, volume analysis, ATR
- **Fundamental (38 factors)**: Valuation ratios, profitability metrics, growth rates, liquidity, leverage, efficiency
- **News/Macro (17 factors)**: News sentiment, earnings catalysts, market correlation, sector strength, VIX, yields
- **Social/Alternative (13 factors)**: Reddit mentions/sentiment, social velocity, contrarian signals
- **Risk/Stability (18 factors)**: Volatility, beta, drawdown, Sharpe/Calmar ratios, liquidity, bid-ask spreads
- **Institutional/Smart Money (22 factors)**: Analyst ratings, price targets, insider trading, institutional ownership

### **Weighted Scoring System**
- **Group Weights**: Technical (20%), Fundamental (25%), News/Macro (15%), Social (10%), Risk (15%), Institutional (15%)
- **Factor Weights**: Each factor within groups has optimized weight (sum=1.0 per group)
- **Overall Score**: Weighted sum of normalized z-scores across all 143 factors
- **Robust Normalization**: MAD-based z-scores with extreme value clipping (±5σ)

## 🏗️ **v3.1 Pipeline Architecture**

### **10-Phase Modular Design**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     VP INVESTMENTS - 10-PHASE PIPELINE               │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: FETCH              PHASE 2: CALCULATE         PHASE 3: NORMALIZE
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Data Sources     │──────▶│ 143 Factors      │──────▶│ Z-Score          │
│ • YFinance (40)  │       │ • Technical (35) │       │ • MAD-based      │
│ • Reddit (5)     │       │ • Fundamental(38)│       │ • Winsorization  │
│ • News API       │       │ • News/Macro(17) │       │ • Clipping       │
│                  │       │ • Social (13)    │       │                  │
│ ✅ COMPLETE      │       │ • Risk (18)      │       │ ✅ COMPLETE      │
└──────────────────┘       │ • Instit. (22)   │       └──────────────────┘
                           │                  │
                           │ ✅ COMPLETE      │
                           └──────────────────┘
                                    │
                                    ▼
                         PHASE 4: SCORE & ASSEMBLE
                         ┌──────────────────────┐
                         │ Weighted Scoring     │
                         │ • Group weights      │
                         │ • Factor weights     │
                         │ • Overall score      │
                         │ • Coverage metrics   │
                         │                      │
                         │ ✅ COMPLETE          │
                         └──────────────────────┘
                                    │
                                    ▼
                            PHASE 5: PERSIST
                         ┌──────────────────────┐
                         │ Database Storage     │
                         │ • Signals table      │
                         │ • Signal metrics     │
                         │ • Upsert logic       │
                         │                      │
                         │ ✅ COMPLETE          │
                         └──────────────────────┘
                                    │
                                    ▼
                           PHASE 6: BACKTEST
                         ┌──────────────────────┐
                         │ Performance Tracking │
                         │ • Baseline prices    │
                         │ • Return intervals   │
                         │ • SPY comparison     │
                         │                      │
                         │ ✅ COMPLETE          │
                         └──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
  PHASE 7: ML              PHASE 8: AI               PHASE 9: REPORTS
┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
│ Machine      │         │ AI           │         │ Reporting &      │
│ Learning     │         │ Enhancement  │         │ Alerts           │
│              │         │              │         │                  │
│ 📋 PLANNED   │         │ 📋 PLANNED   │         │ 📋 PLANNED       │
└──────────────┘         └──────────────┘         └──────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                           PHASE 10: POLISH
                         ┌──────────────────────┐
                         │ Production Ready     │
                         │ • Monitoring         │
                         │ • Documentation      │
                         │ • Deployment         │
                         │                      │
                         │ 📋 PLANNED           │
                         └──────────────────────┘
```

### **Phase Details**

**Phase 1: Fetch** (`backend/phases/phase1_fetch.py`) ✅ Complete
- Sources: Reddit (5 subreddits), News API, YFinance (40 endpoints)
- Validation: Critical field checks, minimum price history (5 rows)
- Caching: 24-hour TTL in `public.data_cache`
- Output: `RawYFinanceData` + Reddit/News data
- Time: ~9s per ticker (optimized with intelligent caching)

**Phase 2: Calculate** (`backend/phases/phase2_calculate.py`) ✅ Complete
- Calculates all 143 factors from raw data
- Error Handling: `@safe_calculation` decorator per factor
- Graceful Degradation: Missing factors return None, don't crash pipeline
- Output: `GroupFactors` (6 groups × factors per group)
- Coverage: ~60% avg (90/143 factors populated per ticker)

**Phase 3: Normalize** (`backend/phases/phase3_normalize.py`) ✅ Complete
- Method: Robust z-scores using MAD (Median Absolute Deviation)
- Extreme Handling: Clip z-scores >10σ to ±5σ
- Edge Cases: Zero-variance → 0.0, insufficient tickers → 0.0
- Winsorization: 1% outlier trimming
- Output: Normalized z-scores for cross-sectional comparison

**Phase 4: Score & Assemble** (`backend/phases/phase4_score_assemble.py`) ✅ Complete
- Weighted Scoring: Group weights × Factor weights
- Validation: NaN/Inf detection, extreme score warnings, coverage checks
- Output: Overall score + 6 group scores + coverage metrics
- Formula: `overall_score = Σ(group_weight × Σ(factor_weight × z_score))`

**Phase 5: Persist** (`backend/phases/phase5_persist.py`) ✅ Complete
- Database: Supabase (PostgreSQL)
- Tables: `signals`, `signal_metrics`, `signal_performance`
- Upsert Logic: Updates if exists, inserts if new

**Phase 6: Backtest** (`backend/phases/phase6_backtest.py`) ✅ Complete
- Performance Tracking: 1d, 3d, 7d, 10d, 14d, 30d, 90d returns
- Baseline Prices: Next trading day's open (realistic entry point)
- SPY Comparison: Calculate alpha vs benchmark
- Status: pending → baseline_set → in_progress → completed
- Migration: 003 applied (19 performance columns added)

**Phase 7: ML** 📋 Planned (Q4 2025)
- Pattern detection, predictive models, feature importance analysis
- Technologies: scikit-learn, XGBoost, TensorFlow

**Phase 8: AI** 📋 Planned (Q4 2025)
- GPT-4 narratives, advanced sentiment, trade insights
- Technologies: OpenAI GPT-4, LangChain

**Phase 9: Reports** 📋 Planned (Q1 2026)
- Daily summaries, email/SMS alerts, performance reports
- Technologies: SendGrid, Twilio, Chart.js

**Phase 10: Polish** 📋 Planned (Q1 2026)
- Production deployment, monitoring, documentation, CI/CD
- Technologies: Docker, GitHub Actions, Sentry

## 🚀 **Quick Start**

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd "VP Investments"
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Setup environment
copy .env.example .env
# Edit .env with your API keys
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 4. Start Development Servers
```bash
# Option A: Start both servers with one command
.\run_fullstack.bat

# Option B: Start individually
.\run_backend.bat    # Backend on http://127.0.0.1:8001
.\run_frontend.bat   # Frontend on http://localhost:3000
```

## 🔧 **Development Workflow**

### Backend Development
- **API Server**: `uvicorn vp_investments.api.server:app --host 127.0.0.1 --port 8001 --reload`
- **Health Check**: `python temp/backend_health_check.py`
- **Tests**: `python -m pytest tests/`

### Frontend Development
- **Dev Server**: `npm run dev` (in frontend/ directory)
- **Build**: `npm run build`
- **Type Check**: `npm run type-check`

### Full Stack Testing
- Backend API: http://127.0.0.1:8001/docs
- Frontend UI: http://localhost:3000
- Health Status: http://127.0.0.1:8001/api/health

## 📡 **API Integration**

### Backend Endpoints
- `GET /api/health` - System health check
- `GET /api/signals/latest` - Latest trading signals
- `GET /api/recommendations` - AI trading recommendations
- `POST /api/runs/start` - Trigger analysis run

### Frontend Configuration
```typescript
// Configured in frontend/src/lib/api.ts
const baseURL = 'http://localhost:8001'
```

## 🔑 **Environment Variables**

Required variables in `.env` (see `.env.example` for full details):
```bash
# Database (Required)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# External APIs (Required)
FMP_API_KEY=your_fmp_key
OPENAI_API_KEY=your_openai_key

# Reddit API (Required)
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# News API (Optional - not implemented)
NEWS_API_KEY=your_news_api_key

# Signal Scoring Weights (Optional - defaults shown)
SCORING_WEIGHT_REDDIT=0.5          # 50% weight for Reddit signals
SCORING_WEIGHT_FINANCIAL=0.5       # 50% weight for Financial signals
SCORING_WEIGHT_NEWS=0.0            # 0% weight (not implemented)
```

### 📊 **Configurable Scoring**

The `weighted_score` calculation is now fully configurable via environment variables:

**Default Configuration (News disabled):**
```
weighted_score = (reddit_score * 0.5) + (financial_score * 0.5) + (news_score * 0.0)
```

**To Adjust Weights:** Edit your `.env` file:
- **Reddit-heavy** (meme stocks): `SCORING_WEIGHT_REDDIT=0.7`, `SCORING_WEIGHT_FINANCIAL=0.3`
- **Financial-heavy** (value investing): `SCORING_WEIGHT_REDDIT=0.2`, `SCORING_WEIGHT_FINANCIAL=0.8`
- **Balanced with News** (when implemented): Set all three weights (they auto-normalize)

See `.env.example` for detailed configuration examples.

## 🧪 **Testing**

```bash
# Backend tests
python -m pytest tests/ -v

# Frontend tests
cd frontend
npm test

# Integration tests
python temp/backend_health_check.py
```

## 📦 **Deployment**

### Development
- Use the provided batch files for local development
- Both servers support hot reload

### Production
- Backend: Deploy with `uvicorn` or containerize with Docker
- Frontend: Deploy to Vercel, Netlify, or similar platforms
- Database: Supabase (already configured)

## 🗄️ **Database Architecture**

### 3-Table Normalized Structure (NEW!)

The database uses a modern 3-table design for optimal performance:

1. **`signals`** - Core signal data (426 rows typical)
   - Ticker, prices, scores, sentiment, metadata
   - Fast queries for dashboards (no heavy metrics)

2. **`signal_metrics`** - Technical & fundamental data (1-to-1)
   - RSI, MACD, volatility, volume indicators
   - P/E ratio, EPS growth, debt/equity, etc.
   - Options data, ownership metrics

3. **`signal_performance`** - Backtest history (1-to-many)
   - Multiple performance records per signal over time
   - 3d, 7d, 10d, 30d returns
   - SPY comparison, alpha calculation

### Helper Views

```sql
-- Fast dashboard queries
SELECT * FROM v_signals_dashboard LIMIT 20;

-- Full signal with metrics
SELECT * FROM v_signals_complete WHERE ticker = 'AAPL';

-- Performance history
SELECT * FROM signal_performance 
WHERE ticker = 'AAPL' 
ORDER BY backtest_date DESC;
```

### Database Management

**Clear All Data** (for fresh start):
```bash
python clear_data.py  # Clears signals, signal_metrics, signal_performance
```

**View Schema Info**:
```bash
python tables.py  # Shows table structure and row counts
```

**Migration Files**:
- See `migrations/` folder for step-by-step migration SQL
- See `docs/BACKEND_UPDATE_3TABLE.md` for implementation details

## � **Running the Pipeline**

### Generate New Signals

```bash
# Clear Python cache and run pipeline
Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
python -m backend.pipeline
```

**What It Does:**
1. Scrapes Reddit for trending tickers (r/wallstreetbets, r/stocks, etc.)
2. Retrieves financial data from Yahoo Finance
3. Calculates technical indicators (RSI, MACD, volatility, etc.)
4. Generates sentiment scores and AI commentary
5. Saves to database (signals + signal_metrics tables)

**Expected Output:**
- ✅ 20-50 signals generated
- ✅ Data saved to signals + signal_metrics tables
- ✅ Logs written to `logs/vp_investments.log`

### Backtest Performance (Coming Soon)

```bash
python -m backend.integrations.backtest
```

## 📊 **Key Features**

### Signal Generation
- **Multi-source**: Reddit + Financial Data + AI Analysis
- **Scoring System**: Configurable weights for different factors
- **Risk Assessment**: Automatic risk level classification
- **Trade Classification**: Momentum, Breakout, Value, etc.
- **ML Analytics**: Phase 1 metrics (momentum consistency, liquidity scoring)

### Performance Tracking
- **Historical Backtesting**: 3d, 7d, 10d, 30d return tracking
- **Benchmark Comparison**: Beat SPY calculation
- **Multiple Records**: Track signal performance over time

### Performance Optimizations
- **Intelligent Caching**: Single-pass data fetching eliminates duplicate API calls (50% reduction)
- **Parallel Processing**: Multi-threaded ticker data retrieval
- **Optimized Pipeline**: 50% faster execution through caching and code optimization
- **Error Handling**: Graceful degradation on API failures
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🎯 **Phase 1.4 Metrics**

### Momentum Consistency Score
- Measures momentum consistency across 1d, 7d, and 30d timeframes
- Scale: 0-100 (higher = more consistent momentum)
- Weight: 7% in technical score calculation
- Helps identify sustainable trends vs. short-term volatility

### Liquidity Score
- Measures stock liquidity based on average daily value traded vs market cap
- Scale: 0.0-1.0 (higher = more liquid)
- Weight: 5% in technical score calculation
- Critical for risk assessment and position sizing

## 🛠️ **Troubleshooting**

### Common Issues

**Pipeline Errors:**
```bash
# Clear cache and retry
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
python -m backend.pipeline
```

**Database Issues:**
```bash
# Clear all data and start fresh
python clear_data.py
python -m backend.pipeline
```

**Missing Columns:**
- Run migrations in `migrations/` folder sequentially
- See `migrations/step4_verify.sql` for verification queries

### Logs

Check `logs/vp_investments.log` for detailed error messages and debugging info.

## 📞 **Support**

- Review logs in `logs/` directory
- Check database with `python tables.py`
- See docs in `docs/` for detailed guides

---

**Status:** ✅ Production Ready - 3-table structure implemented and tested