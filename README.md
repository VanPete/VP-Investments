# VP Investments - Alternative Signal Investment Research Platform

## 🎯 Project Overview

VP Investments is a comprehensive investment research platform that combines alternative data sources (Reddit sentiment, Google Trends, news sentiment) with traditional financial metrics to identify emerging stock opportunities. The system focuses on detecting early momentum signals and validating them with fundamental and technical analysis.

## 🏗️ System Architecture

### Core Philosophy
The platform operates on a multi-stage pipeline that:
1. **Discovers** signals from alternative data sources
2. **Validates** them with financial fundamentals
3. **Scores** and **ranks** opportunities using machine learning
4. **Reports** findings through Excel and web dashboards

### Data Flow
```
Reddit/News/Trends → Signal Detection → Financial Validation → ML Scoring → Reports/Dashboard
```

## Project Structure

### Main Entry Points
- **`VP Investments.py`** - Main orchestrator script that runs the complete pipeline
- **`run_all.py`** - Automated runner with database integration and scheduling
- **`web/vp-investments-web/`** - Next.js dashboard for visualization

### Core Modules

#### 🔍 Data Fetchers (`fetchers/`)
- **`reddit_scraper.py`** - Scrapes Reddit posts from investment subreddits
  - Uses PRAW to access Reddit API
  - Filters by quality metrics (upvotes, karma, recency)
  - Extracts ticker mentions using regex patterns
  - Applies VADER sentiment analysis with custom keyword boosting
  - Implements weighted scoring based on subreddit credibility

- **`finance_fetcher.py`** - Fetches financial data from Yahoo Finance and FMP
  - Retrieves real-time price data, market metrics, and technical indicators
  - Calculates momentum, volatility, and liquidity metrics
  - Extracts fundamental data (P/E, market cap, earnings growth)
  - Implements concurrent fetching for performance

- **`news_fetcher.py`** - Aggregates and analyzes news sentiment
  - Fetches news articles using financial APIs
  - Applies sentiment analysis using VADER
  - Matches company names with fuzzy string matching
  - Caches results to avoid API rate limits

- **`google_trends_fetcher.py`** - Captures search interest trends
  - Uses pytrends to get Google search popularity
  - Handles rate limiting and geographical restrictions
  - Normalizes search volume data

#### ⚙️ Data Processors (`processors/`)
- **`signal_scoring.py`** - Core ML scoring engine
  - Implements multi-factor scoring model
  - Normalizes features across different scales
  - Applies trade-type specific weighting (Swing, Momentum, Growth, Value, Speculative)
  - Calculates composite scores with risk assessment

- **`scoring_trainer.py`** - ML training pipeline (Ridge/Lasso)
  - Trains and evaluates models on historical features/labels
  - Pipeline includes imputation and scaling
  - Saves weights to `outputs/weights/ml_weights.json`
  - Select via profile in `config/config.py`

- **`reports.py`** - Excel report generation
  - Creates formatted Excel workbooks with multiple sheets
  - Implements conditional formatting and styling
  - Generates correlation analysis and feature importance
  - Includes legend and metadata sheets

- **`charts.py`** - HTML dashboard generation
  - Creates interactive charts using matplotlib and seaborn
  - Generates web-friendly visualizations
  - Implements Jinja2 templating for dynamic content

- **`backtest.py`** - Performance validation system
  - Tracks historical signal performance
  - Calculates forward returns and Sharpe ratios
  - Implements walk-forward analysis
  - Stores results in SQLite database
  - Benchmarks vs SPY across 1D/3D/7D/10D windows and computes Beat SPY flags
  - Batches price downloads, guards against future dates, and auto-adds missing DB columns
  - Emits diagnostics CSV for missing prices under `outputs/tables/`

#### 🔧 Configuration (`config/`)
- **`config.py`** - Central configuration hub
  - Feature toggles for enabling/disabling modules
  - API keys and connection settings
  - Scoring weights and thresholds
  - Subreddit weights and filtering criteria
  - `RETURN_WINDOWS` controls backtest windows; `PERCENT_NORMALIZE` normalizes percent-like outputs for Excel

- **`labels.py`** - Column definitions and formatting
  - Defines final output column order
  - Specifies formatting hints for Excel export
  - Maintains consistency across modules

 
#### 🛠️ Utilities (`utils/`)
 
- **`logger.py`** - Logging configuration
  - Implements rotating file handlers
  - Configures different log levels
  - Manages log retention and cleanup

 
#### 🧪 Maintenance & Diagnostics
 
- **`processors/db_audit.py`** - Audits database completeness; optional CLI to mark Unavailable
- **`processors/yf_diagnose.py`** - Quick yfinance symbol availability probe; exports status CSV
- **`tools/cleanup.py`** - Removes caches and build artifacts (safe)

 
### 🌐 Web Dashboard (`web/vp-investments-web/`)
 
- **Next.js 14** application with App Router
- **Tailwind CSS** for styling
- **shadcn/ui** components for consistent UI
- **Chart.js** for interactive visualizations
- **TypeScript** for type safety

## 🔄 How It Works

 
### 1. Data Collection Phase
 
- **Reddit Scraping**: Monitors 9 investment subreddits for ticker mentions
- **Financial Data**: Fetches real-time and historical data from Yahoo Finance
- **News Analysis**: Aggregates recent news articles and analyzes sentiment
- **Google Trends**: Captures search interest spikes

 
### 2. Signal Processing Phase
 
- **Quality Filtering**: Removes low-quality posts and spam
- **Sentiment Analysis**: Applies VADER with custom financial keyword boosting
- **Recency Weighting**: Prioritizes recent discussions (7-day window)
- **Emerging Detection**: Identifies new signals vs. historical baseline

 
### 3. Financial Validation Phase
 
- **Fundamental Screening**: Validates market cap, liquidity, and earnings
- **Technical Analysis**: Calculates momentum, volatility, and technical indicators
- **Risk Assessment**: Flags low liquidity and high volatility stocks

 
### 4. Scoring & Ranking Phase
 
- **Multi-Factor Model**: Combines sentiment, technical, and fundamental factors
- **Trade Type Classification**: Assigns optimal trading strategy (Swing, Momentum, etc.)
- **ML Weighting**: Uses trained weights for optimal feature combination
- **Risk-Adjusted Scoring**: Incorporates risk metrics into final scores

 
### 5. Output Generation Phase
 
- **Excel Reports**: Formatted workbooks with analysis and correlations
- **Web Dashboard**: Interactive visualizations and real-time updates
- **Database Storage**: Historical tracking for backtesting

## 🚀 Key Features

 
### ✅ Implemented Features
 
- **Multi-Source Data Integration** - Reddit, Yahoo Finance, Google Trends, News APIs
- **Real-Time Sentiment Analysis** - VADER with financial keyword boosting
- **Machine Learning Scoring** - Multi-factor model with trade-type optimization
- **Risk Management** - Liquidity flags, volatility warnings, position sizing
- **Performance Tracking** - Historical backtesting and forward returns
- **Professional Reporting** - Excel with conditional formatting and web dashboard
- **Emerging Signal Detection** - Identifies new opportunities vs. baseline
- **Thread Detection** - Recognizes coordinated discussion patterns

### 🔧 Technical Capabilities
- **Concurrent Processing** - Multi-threaded data fetching
- **Caching System** - Reduces API calls and improves performance
- **Database Integration** - SQLite for historical data and backtesting
- **Error Handling** - Robust error recovery and logging
- **Configuration Management** - Feature toggles and environment-specific settings

## 📊 Output Examples

### Excel Report Sheets
1. **Final Analysis** - Main rankings with scores and metrics
2. **Feature Correlations** - Statistical analysis of predictive factors
3. **Legend** - Definitions and methodology explanations
4. **Summary** - Run statistics and metadata

### Web Dashboard Sections
1. **Overview Cards** - Key metrics and statistics
2. **Score Distribution** - Histogram of signal strengths
3. **Trade Type Breakdown** - Strategy allocation pie chart
4. **Top Signals Table** - Interactive sortable rankings
5. **Performance Metrics** - Historical success rates

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ (for web dashboard)
- Reddit API credentials
- Financial data API keys (optional)

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with:
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_app_name
FMP_API_KEY=your_fmp_key (optional)
```

### Web Dashboard Setup
```bash
cd web/vp-investments-web
npm install
npm run dev
```

## 🎮 Usage

### Run Complete Analysis
```bash
python "VP Investments.py"
```

### Run with Automated Scheduling
```bash
set SCHED_ENABLED=1
set SCHED_EVERY_MINUTES=30
python run_all.py
```

### Run Once (immediate)

```bash
python run_all.py --once --stream --timeout 1800
```

### Start Web Dashboard

```bash
cd web/vp-investments-web
npm run dev
# Visit http://localhost:3000
```

## 🧹 Cleanup (optional)

Use the cleanup utility to remove caches, build artifacts, and obsolete files.

What it removes safely:
- Python caches: `__pycache__/`, `.pytest_cache/`
- Web build folders: `web/vp-investments-web/.next/`, `web/vp-investments-web/node_modules/`
- Rotated logs: `outputs/logs/` and `*.log.*` (keeps `vp_investments.log`)
- Obsolete files: `Roadmap.txt`, `bug_checker.py`

Run on Windows PowerShell from the repo root:

```powershell
python .\tools\cleanup.py
```

## 📈 Performance Metrics
## 🦆 DuckDB Quick Queries

Optional: query latest Final Analysis with DuckDB if installed.

```python
from utils.duckdb_utils import query_final_analysis
rows = query_final_analysis('outputs', 'SELECT Ticker, [Weighted Score] FROM final ORDER BY [Weighted Score] DESC', limit=20)
```


The system tracks multiple performance indicators:
- **Signal Success Rate** - Percentage of profitable signals
- **Average Returns** - Mean performance across time windows
- **Sharpe Ratio** - Risk-adjusted performance
- **Maximum Drawdown** - Worst-case scenario analysis
- **Beat Benchmark Rate** - Outperformance vs. SPY

## 🔮 Future Enhancements

See `Roadmap.txt` for detailed development plans including:
- **Advanced ML Models** - Deep learning and ensemble methods
- **Real-Time Streaming** - Live data feeds and instant alerts
- **Portfolio Optimization** - Automated position sizing and allocation
- **Mobile Application** - iOS/Android companion app
- **API Integration** - RESTful API for third-party access

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice and should not be used as the sole basis for investment decisions. Always conduct your own research and consult with financial professionals before making investment decisions.

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the development team.

---

**VP Investments** - Democratizing alternative data for smarter investment decisions.
  - ≥2 mentions, ≥3 upvotes
  - Author karma ≥ 100
- Boosts:
  - Flair boosts (e.g., DD, News)
  - Keyword boosts (e.g., earnings, buyback)
  - Subreddit weight boosts
- Reddit summary: extractive title + comment snapshot

### 💰 Financial Signal Analysis
- Pulls from Yahoo Finance:
  - Price % change (1D, 7D), Volume, Market Cap
  - RSI, MACD, MA, Bollinger Width, Volatility
  - EPS Growth, ROE, P/E Ratio, FCF Margin, Debt/Equity
- Human-readable formatting for volume, cap, percentages

### 🧮 Signal Scoring
- Multi-group weights:
  - Sentiment
  - Price
  - Technicals
  - Fundamentals
- Threshold-based gates
- Emerging ticker detection
- Trade type: Swing / Long-Term / Balanced
- Score normalization (0–100 scale)

### 📊 Excel Output (`Signal_Report.xlsx`)
- Sheets:
  - `Signals`: full dataset with formatting
  - `Dashboard`: top 10 signals (with conditional formatting)
- Human-friendly formats (`1.2M`, `%`, `$`)
- Highlights:
  - Emerging tickers (yellow)
  - Score heatmaps
- Export location: `outputs/(timestamped folder)/`

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `Filtered Reddit Signals.csv` | All Reddit data passing initial filters |
| `Final Analysis.csv` | Final dataset with scores, tags |
| `Signal_Report.xlsx` | Main Excel output |
| `historical_scores.csv` | Logs scores across runs |

---

## 🔧 Config Highlights

Edit in `config/config.py`:

- `SIGNAL_WEIGHT_PROFILES`: tweak per-factor weights
- `THRESHOLDS`: set minimums (e.g., RSI 30–70, PE 5–40)
- `REDDIT_SUBREDDITS`, `KEYWORD_BOOSTS`, `FLAIR_BOOSTS`

---

## 🛣️ Roadmap

### 🔜 Priority Features
- [ ] Rebalance Reddit vs. Financial influence
- [ ] Add per-feature toggle in config
- [ ] Tag DD series / threads

### 📅 Planned
- [ ] Web dashboard frontend
- [ ] Email alerts for emerging tickers
- [ ] Backtest signal effectiveness
- [ ] Database tracking mode

---

## 🚀 Setup Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.template .env

# Run the analysis
python VP_Investments.py
