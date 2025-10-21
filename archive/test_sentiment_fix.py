import yfinance as yf
from textblob import TextBlob

ticker = yf.Ticker('AAPL')
news = ticker.news

print(f"Found {len(news)} articles")
sentiments = []

for article in news[:10]:
    content = article.get('content', {})
    title = content.get('title', '') if isinstance(content, dict) else ''
    if title:
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)
        print(f"  '{title[:60]}...' -> {sentiment:.3f}")

if sentiments:
    avg = sum(sentiments) / len(sentiments)
    print(f"\nAverage sentiment: {avg:.3f}")
    print(f"Consensus (0-100): {((avg + 1) / 2) * 100:.1f}%")
