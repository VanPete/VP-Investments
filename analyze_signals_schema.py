"""
Analyze signals table schema and NULL values
"""
import psycopg2
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
conn = psycopg2.connect(os.getenv('SUPABASE_DATABASE_URL'))
cur = conn.cursor()

# Get schema
print("=" * 80)
print("SIGNALS TABLE SCHEMA")
print("=" * 80)
cur.execute("""
    SELECT column_name, data_type, is_nullable 
    FROM information_schema.columns 
    WHERE table_name = 'signals' 
    ORDER BY ordinal_position
""")
schema = cur.fetchall()
df_schema = pd.DataFrame(schema, columns=['column_name', 'data_type', 'is_nullable'])
print(df_schema.to_string(index=False))

# Get NULL counts
print("\n" + "=" * 80)
print("NULL VALUE ANALYSIS (Current Data)")
print("=" * 80)
cur.execute("SELECT COUNT(*) FROM signals")
total_rows = cur.fetchone()[0]
print(f"Total rows: {total_rows}\n")

if total_rows > 0:
    # Build dynamic query to count NULLs
    columns = [row[0] for row in schema]
    null_checks = [f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) as "{col}_nulls"' for col in columns]
    query = f"SELECT {', '.join(null_checks)} FROM signals"
    
    cur.execute(query)
    null_counts = cur.fetchone()
    
    # Create DataFrame with NULL analysis
    null_analysis = []
    for i, col in enumerate(columns):
        null_count = null_counts[i]
        null_pct = (null_count / total_rows) * 100
        if null_count > 0:
            null_analysis.append({
                'column': col,
                'null_count': null_count,
                'null_pct': f"{null_pct:.1f}%",
                'data_type': schema[i][1]
            })
    
    if null_analysis:
        df_nulls = pd.DataFrame(null_analysis)
        df_nulls = df_nulls.sort_values('null_count', ascending=False)
        print(df_nulls.to_string(index=False))
    else:
        print("✅ No NULL values found in any columns!")
else:
    print("⚠️ No data in signals table")

cur.close()
conn.close()
