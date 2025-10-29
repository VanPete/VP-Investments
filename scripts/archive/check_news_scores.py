"""Check news/macro scores in latest database run"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.database import SupabaseInterface
import pandas as pd

si = SupabaseInterface()

# Get all signals from latest run
df = si.get_all_signals()

# Get news/macro factors
news_cols = ['ticker', 'news_macro_score', 'news_macro_coverage']
print(df[news_cols].head(20))

print(f"\n\nStats:")
print(f"news_macro_score - Min: {df['news_macro_score'].min():.3f}, Max: {df['news_macro_score'].max():.3f}, Mean: {df['news_macro_score'].mean():.3f}")
print(f"news_macro_coverage - Mean: {df['news_macro_coverage'].mean():.3f}")
print(f"Unique scores: {df['news_macro_score'].nunique()}")
