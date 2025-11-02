from backend.storage.database import SupabaseInterface

db = SupabaseInterface()
result = db.client.table('analytics').select('period_type, cagr, alpha_vs_spy, volatility, sharpe_ratio').eq('period_type', '1d').execute()

print("\nCAGR and Alpha value formats:")
print("="*60)
for row in result.data:
    print(f"Period: {row['period_type']}")
    print(f"  CAGR: {row['cagr']} (type: {type(row['cagr'])})")
    print(f"  Alpha vs SPY: {row['alpha_vs_spy']} (type: {type(row['alpha_vs_spy'])})")
    print(f"  Volatility: {row['volatility']} (type: {type(row['volatility'])})")
    print(f"  Sharpe: {row['sharpe_ratio']} (type: {type(row['sharpe_ratio'])})")
