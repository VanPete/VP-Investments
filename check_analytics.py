from backend.storage.database import SupabaseInterface

db = SupabaseInterface()
result = db.client.table('analytics').select('id, period_type, created_at, total_signals').order('period_type').execute()

print('\nAnalytics rows in database:')
print('='*80)
for row in result.data:
    print(f"Period: {row['period_type']:10} | Signals: {row['total_signals']:3} | Created: {row['created_at'][:19]}")
print('='*80)
print(f'Total rows: {len(result.data)}')
