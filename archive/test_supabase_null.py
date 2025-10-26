from backend.storage.database import SupabaseInterface

db = SupabaseInterface()
result = db.client.table('signals').select('id,ticker,return_1d,return_3d').limit(1).execute()

print("\nKeys in signal dict from Supabase:")
if result.data:
    print(list(result.data[0].keys()))
    print("\nActual data:")
    for key, val in result.data[0].items():
        print(f"  {key}: {val}")
