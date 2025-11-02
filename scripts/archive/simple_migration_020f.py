"""Simple migration script to remove avg_composite_score and avg_confidence columns."""

import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Get DATABASE_URL
db_url = os.getenv("SUPABASE_DATABASE_URL")
if not db_url:
    print("❌ SUPABASE_DATABASE_URL not set")
    exit(1)

print("=" * 80)
print("Migration 020f: Remove avg_composite_score and avg_confidence")
print("=" * 80)

# Parse DATABASE_URL
result = urlparse(db_url)

# Connect to database
conn = psycopg2.connect(
    database=result.path[1:],
    user=result.username,
    password=result.password,
    host=result.hostname,
    port=result.port
)

cursor = conn.cursor()

try:
    print("\n⚙️  Dropping avg_composite_score column...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS avg_composite_score;")
    print("✅ Done")
    
    print("\n⚙️  Dropping avg_confidence column...")
    cursor.execute("ALTER TABLE analytics DROP COLUMN IF EXISTS avg_confidence;")
    print("✅ Done")
    
    conn.commit()
    print("\n" + "=" * 80)
    print("✅ Migration 020f Complete!")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
    
finally:
    cursor.close()
    conn.close()
