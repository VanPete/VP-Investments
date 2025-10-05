# VP Investments - Trading Signal Analysis Platform

A comprehensive trading signal platform that combines Reddit sentiment analysis, technical analysis, and AI-powered insights with a modern database architecture.

## 🎯 **What It Does**

- **Signal Generation**: Scans Reddit for trading opportunities and validates with technical/fundamental analysis
- **Multi-Factor Scoring**: Combines Reddit sentiment, financial metrics, technical indicators, and options data
- **Performance Tracking**: Backtests signals with full history tracking (3d, 7d, 10d, 30d returns)
- **AI Commentary**: Generates AI-powered analysis and recommendations
- **Database Optimization**: 3-table normalized structure for fast queries and historical tracking

## 🏗️ **Project Structure**

```text
VP Investments/
├── backend/                 # 🐍 Python Backend Package
│   ├── api/                 #   FastAPI REST endpoints
│   ├── core/                #   Core logic (signals, backtest, config)
│   ├── integrations/        #   External integrations (Reddit, yfinance, AI)
│   │   ├── reddit.py        #   Reddit scraping & sentiment
│   │   ├── yfinance.py      #   Financial data retrieval
│   │   ├── backtest.py      #   Performance tracking
│   │   └── ai.py            #   AI commentary generation
│   ├── storage/             #   Database interfaces (Supabase)
│   ├── utils/               #   Logging & observability
│   └── pipeline.py          #   ⭐ Main signal generation pipeline
├── frontend/                # ⚛️ Next.js Frontend Application (In Development)
│   ├── src/app/             #   Next.js 14+ App Router pages
│   ├── src/components/      #   React components
│   └── package.json
├── docs/                    # 📚 Documentation
│   ├── BACKEND_UPDATE_3TABLE.md    #   3-table migration guide
│   ├── ADD_30D_RETURN.md           #   30-day backtest setup
│   ├── operational_guidelines.md   #   Development guidelines
│   └── archive/                    #   Historical docs
├── migrations/              # 🗄️ Database Migrations
│   ├── step1_create_tables.sql     #   Create 3-table structure
│   ├── step2_migrate_data.sql      #   Migrate existing data
│   ├── step3_create_views.sql      #   Helper views
│   └── step4_verify.sql            #   Verification queries
├── logs/                    # 📋 Application logs
├── clear_data.py            # 🧪 Testing utility (clear DB)
├── tables.py                # 🧪 Testing utility (schema info)
└── requirements.txt         # 📦 Python dependencies
```

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

### Performance Tracking
- **Historical Backtesting**: 3d, 7d, 10d, 30d return tracking
- **Benchmark Comparison**: Beat SPY calculation
- **Multiple Records**: Track signal performance over time

### Data Quality
- **Caching**: Yahoo Finance data cached to minimize API calls
- **Error Handling**: Graceful degradation on API failures
- **Logging**: Comprehensive logging for debugging

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