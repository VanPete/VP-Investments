"""Quick script to check improved risk fields in database"""
from backend.storage.database import SupabaseInterface

db = SupabaseInterface()
result = db.client.table('signals').select(
    'ticker, risk_level, risk_assessment, trade_type, created_at'
).order('created_at', desc=True).limit(2).execute()

print("\n" + "="*80)
print("IMPROVED RISK FIELDS CHECK")
print("="*80)

for signal in result.data:
    print(f"\n{signal['ticker']} ({signal['created_at'][:19]})")
    print(f"  Risk Level: {signal['risk_level']}")
    print(f"  Trade Type: {signal['trade_type']}")
    print(f"  Risk Assessment: {signal['risk_assessment']}")

print("\n" + "="*80)
