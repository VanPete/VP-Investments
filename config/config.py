import json
import os
from datetime import timedelta
from dotenv import load_dotenv, find_dotenv

# Ensure .env is loaded regardless of current working directory
try:
    load_dotenv(find_dotenv())
except Exception:
    pass

# === Project Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "outputs", "backtest.db"))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "weights")
WEIGHTS_OUTPUT_PATH = os.path.join(WEIGHTS_DIR, "ml_weights.json")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# === Env override switch ===
# Set USE_ENV_CONFIG=1 to allow environment variables to override values below.
USE_ENV_CONFIG = os.getenv("USE_ENV_CONFIG", "0") in {"1", "true", "True"}

def _env(name: str, default: str) -> str:
    return os.getenv(name, default) if USE_ENV_CONFIG else default

def _env_int(name: str, default: int) -> int:
    if not USE_ENV_CONFIG:
        return default
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    if not USE_ENV_CONFIG:
        return default
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    if not USE_ENV_CONFIG:
        return default
    val = os.getenv(name)
    if val is None:
        return default
    return val not in {"0", "false", "False"}

# === External API Keys / Providers ===
FMP_API_KEY = _env("FMP_API_KEY", "MISSING")

# === OpenAI Settings ===
# Always read secrets from environment; .env is loaded above
OPENAI = {
    "API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}

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
MEME_TICKER_TERMS = {
    "GPT", "CEO", "YOLO", "MOON", "STONKS", "HODL", "FIRE", "TO THE MOON", "MEME", "PUMP", "ROCKET"
}
MIN_ADV_FOR_REDDIT = _env_float("MIN_ADV_FOR_REDDIT", 2_000_000.0)
REDDIT_SUBREDDITS = [
    "investing", "stocks", "securityanalysis", "valueinvesting",
    "dividends", "growthstocks", "financialindependence",
    "economics", "quant"
]
REDDIT_LIMITS = {
    "hot": 5,
    "new": 5,
    "top": 5,
    "rising": 5
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
# Prefer ML weights saved to outputs/weights, fall back to legacy config path if present
_ML_WEIGHT_CANDIDATES = [
    WEIGHTS_OUTPUT_PATH,
    os.path.join(PROJECT_ROOT, "config", "ml_weights.json"),
]
for _path in _ML_WEIGHT_CANDIDATES:
    try:
        if os.path.exists(_path):
            with open(_path) as f:
                SIGNAL_WEIGHT_PROFILES["ml_optimized"] = json.load(f)
            break
    except Exception:
        # Ignore malformed files and continue to next candidate
        continue

# If ML weights weren't loaded, automatically fall back to default to avoid KeyError
if "ml_optimized" not in SIGNAL_WEIGHT_PROFILES:
    CURRENT_SIGNAL_PROFILE = "default"

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
PERCENT_NORMALIZE = _env_bool("PERCENT_NORMALIZE", True)

# === AI/ChatGPT Feature Flags ===
# These control optional OpenAI enrichments. Toggle via environment variables.
AI_FEATURES = {
    "ENABLED": bool(OPENAI.get("API_KEY")),
    "MAX_ROWS": _env_int("AI_MAX_ROWS", 20),
    "REDDIT_SUMMARY": _env_bool("AI_REDDIT_SUMMARY", True),
    "NEWS_SUMMARY": _env_bool("AI_NEWS_SUMMARY", True),
    "TRENDS_COMMENTARY": _env_bool("AI_TRENDS_COMMENTARY", False)
}

# === Caching and Paths ===
NEWS_CACHE_DIR = _env("NEWS_CACHE_DIR", os.path.join(PROJECT_ROOT, "cache", "news"))
NEWS_CACHE_TTL_HOURS = _env_int("NEWS_CACHE_TTL_HOURS", 1)

TRENDS_CACHE_DIR = _env("TRENDS_CACHE_DIR", os.path.join(PROJECT_ROOT, "data", "google_trends"))

# Outputs directory timestamp format, e.g. "(%d %B %y, %H_%M_%S)"
OUTPUTS_DIR_FORMAT = _env("OUTPUTS_DIR_FORMAT", "(%d %B %y, %H_%M_%S)")

# === Liquidity / Thresholds (UI-friendly flags) ===
LIQUIDITY_WARNING_ADV = _env_float("LIQUIDITY_WARNING_ADV", 1_000_000.0)  # $1M default

# === News Fetch Settings ===
NEWS_RSS_TIMEOUT_SEC = _env_int("NEWS_RSS_TIMEOUT_SEC", 15)
NEWS_FETCH_PACING_SEC = _env_float("NEWS_FETCH_PACING_SEC", 0.5)
NEWS_FUZZY_THRESHOLD = _env_int("NEWS_FUZZY_THRESHOLD", 80)
NEWS_MAX_ITEMS = _env_int("NEWS_MAX_ITEMS", 30)

# === Google Trends Settings ===
TRENDS_TIMEFRAME = _env("TRENDS_TIMEFRAME", "now 7-d")

# === Technical Windows ===
TECH_VOLATILITY_WINDOW = _env_int("TECH_VOLATILITY_WINDOW", 14)
TECH_RSI_PERIOD = _env_int("TECH_RSI_PERIOD", 14)
TECH_BB_PERIOD = _env_int("TECH_BB_PERIOD", 20)
TECH_MOMENTUM_DAYS = _env_int("TECH_MOMENTUM_DAYS", 30)
TECH_VOL_SPIKE_WINDOW = _env_int("TECH_VOL_SPIKE_WINDOW", 7)
BETA_WINDOW = _env_int("BETA_WINDOW", 90)

# === Data Provider Concurrency ===
YF_MAX_WORKERS = _env_int("YF_MAX_WORKERS", 8)

# === Web API Server Settings ===
WEB_API_DEBUG = _env_bool("WEB_API_DEBUG", True)
WEB_API_PORT = _env_int("WEB_API_PORT", 5001)
REDIS_URL = _env("REDIS_URL", "")

# === Scheduler Settings ===
# Enable simple interval scheduling when running run_all.py
SCHED_ENABLED = _env_bool("SCHED_ENABLED", False)
SCHED_EVERY_MINUTES = _env_int("SCHED_EVERY_MINUTES", 360)
SCHED_TIMEZONE = _env("SCHED_TIMEZONE", "US/Eastern")
# Twice-daily schedule: one hour before open and one hour after close
SCHED_TWICE_DAILY = _env_bool("SCHED_TWICE_DAILY", True)
MARKET_OPEN_TIME = _env("MARKET_OPEN_TIME", "09:30")   # HH:MM in ET
MARKET_CLOSE_TIME = _env("MARKET_CLOSE_TIME", "16:00")  # HH:MM in ET
PRE_OPEN_OFFSET_MINUTES = _env_int("PRE_OPEN_OFFSET_MINUTES", 60)
POST_CLOSE_OFFSET_MINUTES = _env_int("POST_CLOSE_OFFSET_MINUTES", 60)

# === Universe Expansion (Trending Tickers) ===
UNIVERSE_SOURCES = {
    "FMP_MOVERS": _env_bool("UNIVERSE_FMP_MOVERS", True),
    "STOCKTWITS_TRENDING": _env_bool("UNIVERSE_STOCKTWITS", True),
}
UNIVERSE_LIMIT = _env_int("UNIVERSE_LIMIT", 500)

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

# === Observability & Alerts ===
SENTRY_DSN = _env("SENTRY_DSN", "")
SLACK_WEBHOOK_URL = _env("SLACK_WEBHOOK_URL", "")
