import yfinance as yf

# Test news_sentiment
print("="*80)
print("Testing news_sentiment data availability")
print("="*80)

ticker = yf.Ticker('AAPL')

# Check if news exists
print("\n1. Checking raw_data.news:")
news = ticker.news
print(f"   News available: {len(news) if news else 0} articles")
if news and len(news) > 0:
    print(f"   First article keys: {news[0].keys()}")
    print(f"   Sample title: {news[0].get('title', 'NO TITLE')}")

# Test earnings_revision_3m
print("\n2. Testing earnings_revision_3m:")
try:
    eps_rev = ticker.eps_revisions
    print(f"   eps_revisions type: {type(eps_rev)}")
    if hasattr(eps_rev, 'index'):
        print(f"   Index values: {list(eps_rev.index)}")
        if '0q' in eps_rev.index:
            row = eps_rev.loc['0q']
            print(f"   0q row: {row}")
            if 'upLast30days' in row.index and 'downLast30days' in row.index:
                net_revisions = row['upLast30days'] - row['downLast30days']
                print(f"   Net revisions (up-down): {net_revisions}")
        else:
            print("   ERROR: No '0q' row found")
except Exception as e:
    print(f"   ERROR: {e}")

# Test post_earnings_drift_21d
print("\n3. Testing post_earnings_drift_21d:")
try:
    earnings_hist = ticker.earnings_history
    print(f"   earnings_history type: {type(earnings_hist)}")
    if hasattr(earnings_hist, 'index'):
        print(f"   Rows: {len(earnings_hist)}")
        print(f"   Columns: {list(earnings_hist.columns)}")
        if len(earnings_hist) > 0:
            print(f"   Last earnings date: {earnings_hist.index[-1]}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*80)
