"""Simple migration to remove unused correlation columns."""

import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("SUPABASE_DATABASE_URL")
if not db_url:
    print("❌ SUPABASE_DATABASE_URL not set")
    exit(1)

print("=" * 80)
print("Migration 020g: Remove unused correlation columns")
print("=" * 80)

result = urlparse(db_url)
conn = psycopg2.connect(
    database=result.path[1:],
    user=result.username,
    password=result.password,
    host=result.hostname,
    port=result.port
)

cursor = conn.cursor()

try:
    print("\n⚙️  Dropping rolling_sharpe_30d...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS rolling_sharpe_30d;")
    print("✅ Done")
    
    print("\n⚙️  Dropping signal_correlations...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS signal_correlations;")
    print("✅ Done")
    
    print("\n⚙️  Dropping top_positive_pairs...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS top_positive_pairs;")
    print("✅ Done")
    
    print("\n⚙️  Dropping top_negative_pairs...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS top_negative_pairs;")
    print("✅ Done")
    
    conn.commit()
    print("\n" + "=" * 80)
    print("✅ Migration 020g Complete!")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
    
finally:
    cursor.close()
    conn.close()
