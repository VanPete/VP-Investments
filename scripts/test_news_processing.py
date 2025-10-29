"""Test news processing in Phase 2"""
import sys
sys.path.insert(0, '.')

from backend.integrations.yfinance import ComprehensiveYFinanceFetcher
from backend.phases.phase2_calculate import Phase2Calculator
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def test_news():
    """Test if news is being processed correctly"""
    print("=" * 60)
    print("TESTING NEWS PROCESSING")
    print("=" * 60)
    
    # Fetch raw data using yfinance fetcher directly
    yf_fetcher = ComprehensiveYFinanceFetcher()
    raw = yf_fetcher.fetch_ticker('NVDA')
    
    print(f"\n1. Raw Data Check:")
    print(f"   Has news attribute: {hasattr(raw, 'news')}")
    print(f"   News is not None: {raw.news is not None}")
    print(f"   News count: {len(raw.news) if raw.news else 0}")
    
    if raw.news and len(raw.news) > 0:
        print(f"\n2. First Article Structure:")
        article = raw.news[0]
        print(f"   Type: {type(article)}")
        print(f"   Keys: {list(article.keys()) if isinstance(article, dict) else 'Not a dict'}")
        if isinstance(article, dict):
            content = article.get('content', {})
            print(f"   Has 'content' key: {'content' in article}")
            print(f"   Content type: {type(content)}")
            if isinstance(content, dict):
                print(f"   Content keys: {list(content.keys())}")
                title = content.get('title', 'NO TITLE')
                print(f"   Title: {title[:100] if title else 'NO TITLE'}")
    
    # Calculate factors
    print(f"\n3. Running Phase 2 Calculator:")
    p2 = Phase2Calculator()
    group_factors = p2.calculate_all_factors('NVDA', raw, news_data=None)
    
    print(f"\n4. Results:")
    print(f"   news_sentiment: {group_factors.news_macro.get('news_sentiment', 'NOT SET')}")
    print(f"   news_sentiment_consensus: {group_factors.news_macro.get('news_sentiment_consensus', 'NOT SET')}")
    print(f"   All news/macro factors:")
    for key, val in group_factors.news_macro.items():
        if val is not None:
            print(f"      {key}: {val}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    test_news()
