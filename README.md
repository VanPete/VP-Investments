# VP Investments – Reddit & Financial Signal Screener 📈

## 🔍 Overview

**VP Investments** is a multi-factor screener that identifies emerging or strong trading signals by combining:

- **Reddit sentiment analysis** from top finance subreddits
- **Financial metrics** from Yahoo Finance via `yfinance`
- **Signal scoring** across sentiment, price action, technicals, fundamentals
- **Excel reports** with conditional formatting & human-readable outputs

---

## 🧠 Core Features

### 🧵 Reddit Signal Analysis
- Scrapes: `investing`, `stocks`, `valueinvesting`, `securityanalysis`, etc.
- Scores: VADER sentiment (title + comments)
- Filters:
  - Last 7 days
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
