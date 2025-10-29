"""Check news/macro scores from latest pipeline run."""
import json

# Load latest results
with open('frontend/public/results/pipeline_results_20251029_064326.json') as f:
    data = json.load(f)

rankings = data['rankings']

print(f"\nTotal signals: {len(rankings)}")
print("\nTop 10 News/Macro Scores:")
print("=" * 50)

for i, r in enumerate(rankings[:10], 1):
    ticker = r['ticker']
    nm_score = r.get('news_macro_score', 0)
    overall = r.get('overall_score', 0)
    print(f"{i:2}. {ticker:6} | news_macro={nm_score:7.4f} | overall={overall:7.4f}")

# Check if all are zero
nm_scores = [r.get('news_macro_score', 0) for r in rankings]
non_zero = [s for s in nm_scores if s != 0]

print(f"\n{'='*50}")
print(f"News/Macro Score Statistics:")
print(f"  Total signals: {len(nm_scores)}")
print(f"  Non-zero scores: {len(non_zero)}")
print(f"  Zero scores: {len(nm_scores) - len(non_zero)}")
if non_zero:
    print(f"  Min (non-zero): {min(non_zero):.4f}")
    print(f"  Max: {max(nm_scores):.4f}")
    print(f"  Avg (all): {sum(nm_scores)/len(nm_scores):.4f}")
