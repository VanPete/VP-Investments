"""
Apply migration 020d - Add group_performance column
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Connect using the database URL
db_url = os.getenv("SUPABASE_DATABASE_URL")

print("Connecting to database...")
conn = psycopg2.connect(db_url)
cursor = conn.cursor()

print("\nReading migration file...")
with open("migrations/020d_add_group_performance_column.sql", "r") as f:
    migration_sql = f.read()

print("Migration SQL:")
print(migration_sql)

print("\nExecuting migration...")
try:
    cursor.execute(migration_sql)
    conn.commit()
    print("✅ Migration applied successfully!")
    
    # Verify the column was added
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'analytics' AND column_name = 'group_performance'
    """)
    result = cursor.fetchone()
    if result:
        print(f"\n✅ Verified: Column '{result[0]}' exists with type '{result[1]}'")
    else:
        print("\n⚠️ Warning: Column not found after migration")
        
except Exception as e:
    conn.rollback()
    print(f"\n❌ Error applying migration: {e}")
finally:
    cursor.close()
    conn.close()
