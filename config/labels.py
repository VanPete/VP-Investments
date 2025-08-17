# === Final Output Column Order for Excel & Console Output ===
FINAL_COLUMN_ORDER = [
    # === Meta ===
    "Rank", "Ticker", "Company", "Sector", "Weighted Score", "Trade Type", "Secondary Flags", "Risk",
    "Highest Contributor", "Lowest Contributor", "Top Factors",
    "Emerging/Thread", "Post Recency", "Reddit Summary",

    # === Sentiment ===
    "Reddit Sentiment", "News Sentiment", "Mentions", "News Mentions", "Upvotes",
    "Sentiment Spike",

    # === Price & Volume ===
    "Current Price", "Price 1D %", "Price 7D %", "Volume", "Volume Spike Ratio",
    "Market Cap", "Relative Strength", "Above 50-Day MA %", "Above 200-Day MA %",

    # === Technicals ===
    "RSI", "MACD Histogram", "Bollinger Width", "Volatility", "Volatility Rank",
    "Momentum 30D %", "Momentum Rank", "Beta vs SPY",

    # === Fundamentals ===
    "P/E Ratio", "EPS Growth", "ROE", "Debt/Equity", "FCF Margin",
    "Retail Holding %", "Float % Held by Institutions", "Liquidity Rank",
    "Z-Score: Market Cap", "Z-Score: Avg Daily Value", "Z-Score: Reddit Activity", "Liquidity Flags",

    # === Short Interest ===
    "Shares Short", "Short Ratio", "Short Percent Float", "Short Percent Outstanding", "Squeeze Signal",

    # === Options ===
    "Put/Call OI Ratio", "Put/Call Volume Ratio", "Options Skew", "Call Volume Spike Ratio", "IV Spike %",

    # === Insider Activity ===
    "Insider Buys 30D", "Insider Buy Volume", "Last Insider Buy Date", "Insider Signal",

    # === ETF Flows ===
    "Sector Inflows", "ETF Flow Spike Ratio", "ETF Flow Signal",

    # === Earnings ===
    "Next Earnings Date", "Earnings Gap %",

    # === Trend ===
    "Trend Spike", "Google Interest", "AI News Summary", "AI Trends Commentary", "AI Commentary", "Score Explanation",

    # === Scoring Metadata ===
    "Run Datetime", "Signal Type", "Source",
    "Reddit Score", "Financial Score", "News Score"
]


# === Formatting hints for Excel output ===
COLUMN_FORMAT_HINTS = {
    # === Meta additions ===
    "Risk": "string",
    # === Sentiment ===
    "Reddit Sentiment": "float",
    "News Sentiment": "float",
    "Mentions": "float",
    "News Mentions": "float",
    "Upvotes": "float",
    "Sentiment Spike": "float",

    # === Price & Volume ===
    "Current Price": "currency",
    "Price 1D %": "percent",
    "Price 7D %": "percent",
    "Volume": "human",
    "Volume Spike Ratio": "float",
    "Market Cap": "human",
    "Relative Strength": "percent",
    "Above 50-Day MA %": "percent",
    "Above 200-Day MA %": "percent",

    # === Technicals ===
    "RSI": "float",
    "MACD Histogram": "float",
    "Bollinger Width": "float",
    "Volatility": "float",
    "Volatility Rank": "percent",
    "Momentum 30D %": "percent",
    "Momentum Rank": "percent",
    "Beta vs SPY": "float",

    # === Fundamentals ===
    "P/E Ratio": "float",
    "EPS Growth": "percent",
    "ROE": "percent",
    "Debt/Equity": "float",
    "FCF Margin": "percent",
    "Retail Holding %": "percent",
    "Float % Held by Institutions": "percent",
    "Liquidity Rank": "percent",
    "Z-Score: Market Cap": "float",
    "Z-Score: Avg Daily Value": "float",
    "Z-Score: Reddit Activity": "float",
    "Liquidity Flags": "string",

    # === Short Interest ===
    "Shares Short": "human",
    "Short Ratio": "float",
    "Short Percent Float": "percent",
    "Short Percent Outstanding": "percent",

    # === Options ===
    "Put/Call OI Ratio": "float",
    "Put/Call Volume Ratio": "float",
    "Options Skew": "float",
    "Call Volume Spike Ratio": "float",
    "IV Spike %": "percent",

    # === Insider Activity ===
    "Insider Buys 30D": "float",
    "Insider Buy Volume": "currency",

    # === ETF Flows ===
    "Sector Inflows": "currency",
    "ETF Flow Spike Ratio": "float",
    "ETF Flow Signal": "float",

    # === Earnings ===
    "Earnings Gap %": "percent",
    "Next Earnings Date": "date",

    # === Trend ===
    "Google Interest": "float",
    "AI News Summary": "string",
    "AI Trends Commentary": "string",
    "AI Commentary": "string",
    "Score Explanation": "string",

    # === Scoring Metadata ===
    "Weighted Score": "float",
    "Signal Type": "string",
    "Run Datetime": "string",
    "Source": "string",
    "Reddit Score": "float",
    "Financial Score": "float",
    "News Score": "float"
}
