# report_legend.py

LEGEND = {
    # === Meta ===
    "Rank": "Final position in sorted output based on weighted score.",
    "Ticker": "Stock symbol prefixed with $, linked to Yahoo Finance.",
    "Company": "Full company name as pulled from financial data.",
    "Sector": "Industry classification.",
    "Trade Type": "Type of trade (e.g., Long, Short, Option).",
    "Secondary Flags": "Additional flags indicating special conditions.",
    "Highest Contributor": "Top scoring factor driving the signal.",
    "Lowest Contributor": "Weakest factor lowering the score.",
    "Top Factors": "Concise explanation of the most influential scoring drivers.",
    "Emerging/Thread": "Emerging = sudden Reddit signal. Thread = multiple Reddit posts.",
    "Post Recency": "Recency weighting, scaled between 0–1. Newer = closer to 1.",
    "Reddit Summary": "Extracted headline and key Reddit comment excerpts.",

    # === Sentiment ===
    "Reddit Sentiment": "Blended VADER score (70% post, 30% top comments).",
    "News Sentiment": "Blended financial news sentiment score.",
    "Mentions": "Weighted Reddit post count.",
    "News Mentions": "Number of recent financial news articles.",
    "Upvotes": "Average Reddit post upvotes.",
    "Sentiment Spike": "Sudden surge in sentiment volume.",

    # === Price & Volume ===
    "Current Price": "Latest stock price from Yahoo Finance.",
    "Price 1D %": "Price return over the past 1 trading day.",
    "Price 7D %": "Price return over the past 7 trading days.",
    "Volume": "Latest trading volume.",
    "Volume Spike Ratio": "Ratio of current to average volume over past week.",
    "Market Cap": "Company valuation formatted in human-readable form (e.g. 32.5B).",
    "Relative Strength": "Ticker's performance relative to its sector or benchmark index.",
    "Above 50-Day MA %": "Percentage above (or below) the 50-day moving average.",
    "Above 200-Day MA %": "Percentage above (or below) the 200-day moving average.",

    # === Technicals ===
    "RSI": "Relative Strength Index (momentum measure).",
    "MACD Histogram": "MACD histogram value (momentum indicator).",
    "Bollinger Width": "Bollinger Band width (volatility measure).",
    "Volatility": "Normalized standard deviation of price movements.",
    "Volatility Rank": "Volatility percentile rank within sector or market.",
    "Momentum 30D %": "Price momentum over 30-day window.",
    "Momentum Rank": "Percentile rank of 30-day momentum versus all tickers.",
    "Beta vs SPY": "Beta coefficient relative to SPY.",

    # === Fundamentals ===
    "P/E Ratio": "Price-to-earnings ratio.",
    "EPS Growth": "Earnings per share growth rate (typically YoY).",
    "ROE": "Return on equity percentage.",
    "Debt/Equity": "Debt-to-equity ratio.",
    "FCF Margin": "Free cash flow as a percent of revenue.",
    "Retail Holding %": "Estimated percent of shares held by retail investors.",
    "Float % Held by Institutions": "Percent of float owned by institutional holders.",
    "Liquidity Rank": "Percentile rank of average daily traded dollar volume.",

    # === Short Interest ===
    "Shares Short": "Total number of shares shorted.",
    "Short Ratio": "Days to cover based on average volume.",
    "Short Percent Float": "Short interest as percent of float.",
    "Short Percent Outstanding": "Short interest as percent of total shares outstanding.",
    "Squeeze Signal": "Binary or graded signal indicating short squeeze potential.",

    # === Options ===
    "Put/Call OI Ratio": "Put-to-call ratio based on open interest.",
    "Put/Call Volume Ratio": "Put-to-call volume ratio based on latest trading day.",
    "Options Skew": "Implied volatility skew across strikes.",
    "Call Volume Spike Ratio": "Call volume relative to 30-day average.",
    "IV Spike %": "Percent change in implied volatility from recent average.",

    # === Insider Activity ===
    "Insider Buys 30D": "Number of Form 4 insider purchases in past 30 days.",
    "Insider Buy Volume": "Total shares purchased by insiders in recent filings.",
    "Last Insider Buy Date": "Most recent insider Form 4 buy date.",
    "Insider Signal": "Flag summarizing strength of insider activity.",

    # === ETF Flows ===
    "Sector Inflows": "Net capital inflow into the stock's sector ETFs (falls back to sector ETF price*volume proxy when primary data unavailable).",
    "ETF Flow Spike Ratio": "Sector ETF inflow relative to historical average (or PV spike ratio when proxied).",
    "ETF Flow Signal": "Flag indicating notable surge in ETF demand for sector (works with proxy mode too).",

    # === Earnings ===
    "Next Earnings Date": "Next scheduled earnings report date.",
    "Earnings Gap %": "Gap return from last earnings date.",

    # === Trend ===
    "Trend Spike": "Flag indicating a sharp increase in Google search interest.",
    "Google Interest": "Relative search popularity score from Google Trends (0–100).",

    # === Scoring Metadata ===
    "Weighted Score": "Raw weighted score before normalization.",
    "Run Datetime": "Timestamp of when signal was generated.",
    "Source": "Data source flag (e.g., Reddit, News, Trends)."
}
