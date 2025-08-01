import json
import os

# === Project Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "outputs", "backtest.db"))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "weights")
WEIGHTS_OUTPUT_PATH = os.path.join(WEIGHTS_DIR, "ml_weights.json")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# === Backtesting Constants ===
RETURN_WINDOWS = [1, 3, 7, 10]
BENCHMARK_WINDOWS = [3, 10]
FUTURE_PROOF_COLS = [
    "Realized Returns", "Backtest Phase", "Backtest Timestamp", "Beat SPY 3D", "Beat SPY 1D", "Backtest Notes",
    "Signal Duration", "Max Return %", "Drawdown %", "Normalized Rank", "Signal Confidence",
    "Forward Volatility", "Forward Sharpe Ratio", "Run Config Hash"
]

# === Reddit Scraper Settings ===
EXCLUDED_TICKERS = {
    "$OP", "$AI", "$ATH", "$EU", "$FAQ", "$ER", "$EPS", "$US",
    "$DD", "$IT", "$IN", "$FYI", "$API", "$PSA", "$ROI", "$PT", "$WSB",
    *{f"${l}" for l in list("BCDEGHIJKLMNOPQRSUVWXZ")}
}
REDDIT_SUBREDDITS = [
    "investing", "stocks", "securityanalysis", "valueinvesting",
    "dividends", "growthstocks", "financialindependence",
    "economics", "quant"
]
REDDIT_LIMITS = {
    "hot": 30,
    "new": 20,
    "top": 20,
    "rising": 20
}

REDDIT_TICKER_PATTERN = r'\b[A-Z]{1,5}\b'
MIN_MENTIONS = 2
MIN_UPVOTES = 2
MIN_AUTHOR_KARMA = 25
TITLE_COMMENT_BLEND = (0.7, 0.3)
COMMENT_WEIGHT_SCALING = 0.2
ENABLE_REDDIT_SCRAPE = True
DEBUG_REDDIT_SCRAPE = True

# === Sentiment Boosting ===
KEYWORD_BOOSTS = {
    "earnings": 1.25, "buyback": 1.25, "acquisition": 1.25, "merger": 1.25,
    "upgrade": 1.15, "bullish": 1.15, "undervalued": 1.20
}
FLAIR_BOOSTS = {
    "DD": 1.2, "Catalyst": 1.2, "News": 1.1, "Fundamentals": 1.15
}
FLAIR_ALIASES = {
    "deep dive": "DD", "research": "DD", "event": "Catalyst",
    "announcement": "News", "earnings": "News", "fundamentals": "Fundamentals"
}
SUBREDDIT_WEIGHTS = {
    "investing": 1.25, "stocks": 1.20, "securityanalysis": 1.15, "valueinvesting": 1.15,
    "dividends": 1.10, "growthstocks": 1.10, "financialindependence": 1.05,
    "economics": 1.05, "quant": 1.10
}

# === Feature Toggles ===
FEATURE_TOGGLES = {
    "RSI": True, "MACD Histogram": True, "Bollinger Width": True, "Volatility": True,
    "Volatility Rank": True, "Beta vs SPY": True,
    "P/E Ratio": True, "EPS Growth": True, "ROE": True, "Debt/Equity": True, "FCF Margin": True,
    "Above 50-Day MA %": True, "Above 200-Day MA %": True, "Price 1D %": True, "Price 7D %": True,
    "Relative Strength": True, "Momentum 30D %": True,
    "Reddit Sentiment": True, "News Sentiment": True, "News Mentions": True,
    "Post Recency": True, "Enable News Fetch": True, "Thread Detection": True,
    "Google Trends": False, "Twitter Mentions": True, "Sentiment Spike": True,
    "Put/Call OI Ratio": True, "Put/Call Volume Ratio": True, "Options Skew": True,
    "Call Volume Spike Ratio": True, "IV Spike %": True,
    "Insider Buys 30D": True, "Insider Buy Volume": True, "Last Insider Buy Date": True,
    "Insider Signal": True, "Retail Holding %": True, "Float % Held by Institutions": True,
    "Sector Inflows": True, "ETF Flow Spike Ratio": True, "ETF Flow Signal": True,
    "Avg Daily Value Traded": True,
    "Earnings Gap %": True, "Next Earnings Date": True,
}

# === Scoring Profiles ===
CURRENT_SIGNAL_PROFILE = "ml_optimized"
SIGNAL_WEIGHT_PROFILES = {"default": {
    "Reddit Sentiment": 1.1, "News Sentiment": 1.1, "News Mentions": 0.6, "Post Recency": 0.9,
    "Price 1D %": 0.6, "Price 7D %": 0.7, "Volume": 0.8, "Relative Strength": 1.0,
    "Above 50-Day MA %": 0.9, "Above 200-Day MA %": 0.8, "RSI": 0.6, "MACD Histogram": 0.5,
    "Bollinger Width": 0.4, "Volatility": 0.4, "P/E Ratio": 0.4, "EPS Growth": 1.3,
    "ROE": 1.2, "Debt/Equity": 0.4, "FCF Margin": 0.6, "Google Interest": 1.0,
    "Trend Spike": 1.2, "Squeeze Signal": 0.35, "Put/Call OI Ratio": 0.4,
    "Put/Call Volume Ratio": 0.4, "Options Skew": 0.3, "Call Volume Spike Ratio": 0.3,
    "IV Spike %": 0.3, "Insider Buys 30D": 0.25, "Insider Buy Volume": 0.25,
    "Insider Signal": 0.5, "Sector Inflows": 0.4, "ETF Flow Spike Ratio": 0.3,
    "ETF Flow Signal": 0.3, "Momentum 30D %": 0.5, "Volatility Rank": 0.3,
    "Beta vs SPY": 0.25, "Twitter Mentions": 0.0, "Sentiment Spike": 0.0,
    "Earnings Gap %": 0.7, "Retail Holding %": 0.3, "Float % Held by Institutions": 0.6,
    "Avg Daily Value Traded": 0.7
}}
try:
    with open("config/ml_weights.json") as f:
        SIGNAL_WEIGHT_PROFILES["ml_optimized"] = json.load(f)
except FileNotFoundError:
    pass  # Optional override

# === Pre-Score Thresholds ===
THRESHOLDS = {
    "Price 1D %": 0.0, "Price 7D %": 0.0, "Volume": 2000000, "Relative Strength": 0.0,
    "Above 50-Day MA %": 1.0, "Above 200-Day MA %": 0.0,
    "RSI_LOW": 25, "RSI_HIGH": 70, "MACD Histogram": 0.0, "Bollinger Width": 0.0, "Volatility": 0.0,
    "P/E Ratio": 0.0, "PE_LOW": 5, "PE_HIGH": 40,
    "EPS Growth": 0.05, "ROE": 0.03, "Debt/Equity": 0.15, "FCF Margin": 0.03,
    "News Sentiment": 0.0, "News Mentions": 2.0,
    "Put/Call OI Ratio": 0.5, "Put/Call Volume Ratio": 0.5, "Options Skew": 0.0,
    "Call Volume Spike Ratio": 1.2, "IV Spike %": 0.3,
    "Insider Buys 30D": 1, "Insider Buy Volume": 10000, "Insider Signal": 1.0,
    "Sector Inflows": 10000000, "ETF Flow Spike Ratio": 1.5, "ETF Flow Signal": 1.0,
    "Momentum 30D %": 0.05, "Volatility Rank": 0.3, "Beta vs SPY": 0.2,
    "Twitter Mentions": 0, "Sentiment Spike": 0, "Earnings Gap %": 3.0,
    "Retail Holding %": 0.5, "Float % Held by Institutions": 0.10, "Avg Daily Value Traded": 3000000
}

# === Misc Parameters ===
GOOGLE_TRENDS_BATCH_SIZE = 10
GOOGLE_TRENDS_SLEEP_SEC = 8
RECENCY_HALFLIFE_HOURS = 36
EMERGING_SCORE_BOOST = 1.2
REDDIT_FINANCIAL_WEIGHT_RATIO = 0.5
FEATURE_NORMALIZATION = True

# === Group Labels for Scoring Analysis ===
GROUP_LABELS = {
    "Reddit Sentiment": "Sentiment_Reddit", "News Sentiment": "Sentiment_News",
    "News Mentions": "Sentiment_News", "Post Recency": "Sentiment_Reddit",
    "Google Interest": "Sentiment_Search", "Trend Spike": "Sentiment_Search",
    "Price 1D %": "Price_Action", "Price 7D %": "Price_Action", "Volume": "Price_Action",
    "Relative Strength": "Momentum_Trend", "Momentum 30D %": "Momentum_Trend",
    "Earnings Gap %": "Price_Action",
    "Above 50-Day MA %": "Technical_Trend", "Above 200-Day MA %": "Technical_Trend",
    "RSI": "Technical_Oscillator", "MACD Histogram": "Technical_Oscillator",
    "Bollinger Width": "Technical_Oscillator", "Volatility": "Technical_Oscillator",
    "Volatility Rank": "Technical_Oscillator",
    "Put/Call OI Ratio": "Options_Sentiment", "Put/Call Volume Ratio": "Options_Sentiment",
    "Options Skew": "Options_Sentiment", "Call Volume Spike Ratio": "Options_Sentiment",
    "IV Spike %": "Options_Sentiment",
    "P/E Ratio": "Valuation", "EPS Growth": "Profitability", "ROE": "Profitability",
    "Debt/Equity": "Balance_Sheet", "FCF Margin": "Balance_Sheet",
    "Insider Buys 30D": "Insider_Activity", "Insider Buy Volume": "Insider_Activity",
    "Last Insider Buy Date": "Insider_Activity", "Insider Signal": "Insider_Activity",
    "Retail Holding %": "Ownership", "Float % Held by Institutions": "Ownership",
    "Beta vs SPY": "Market_Sensitivity",
    "Sector Inflows": "Macro_Flows", "ETF Flow Spike Ratio": "Macro_Flows", "ETF Flow Signal": "Macro_Flows",
    "Avg Daily Value Traded": "Liquidity"
}
