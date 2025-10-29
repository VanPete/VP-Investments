"""Quick script to check news/macro scores in database"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.supabase_interface import SupabaseInterface
import pandas as pd

si = SupabaseInterface()
df = si.get_all_signals()

news_scores = df[['ticker', 'news_macro_score', 'news_macro_coverage']].sort_values('news_macro_score')
print(news_scores.head(20))

print(f'\n\nStats:')
print(f'Min: {df["news_macro_score"].min()}')
print(f'Max: {df["news_macro_score"].max()}')
print(f'Mean: {df["news_macro_score"].mean()}')
print(f'Std: {df["news_macro_score"].std()}')
print(f'Unique values: {df["news_macro_score"].nunique()}')
