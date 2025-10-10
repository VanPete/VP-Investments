"""
Comprehensive Technical Signal Analysis
Checks ALL 34 Technical columns for the latest signal
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_DATABASE_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Get the latest signal with ALL technical columns
cur.execute("""
    SELECT 
        ticker,
        technical_score,
        created_at,
        -- Momentum indicators (23% of technical score)
        price_1d_pct,
        price_7d_pct,
        momentum_30d_pct,
        relative_strength,
        -- RSI (12%)
        rsi,
        -- MACD (10%)
        macd,
        macd_line,
        macd_signal,
        macd_histogram,
        -- Moving Averages (12%)
        above_50d_ma_pct,
        above_200d_ma_pct,
        -- Volume (12%)
        volume,
        avg_volume_30d,
        volume_spike_ratio,
        volume_price_correlation,
        -- Volatility (10%)
        volatility,
        volatility_rank,
        historical_volatility,
        -- Bollinger Bands
        bollinger_upper,
        bollinger_lower,
        bollinger_position,
        bollinger_width,
        -- Relative Strength (10%)
        sector_relative_strength,
        -- Beta (8%)
        beta,
        -- Phase 1.4 ML metrics (13%)
        momentum_consistency_score,
        liquidity_score,
        -- ATR
        atr,
        atr_percent
    FROM signals 
    ORDER BY created_at DESC 
    LIMIT 1
""")

result = cur.fetchone()

if not result:
    print("❌ No signals found")
    exit(1)

columns = [
    'ticker', 'technical_score', 'created_at',
    'price_1d_pct', 'price_7d_pct', 'momentum_30d_pct', 'relative_strength',
    'rsi',
    'macd', 'macd_line', 'macd_signal', 'macd_histogram',
    'above_50d_ma_pct', 'above_200d_ma_pct',
    'volume', 'avg_volume_30d', 'volume_spike_ratio', 'volume_price_correlation',
    'volatility', 'volatility_rank', 'historical_volatility',
    'bollinger_upper', 'bollinger_lower', 'bollinger_position', 'bollinger_width',
    'sector_relative_strength',
    'beta',
    'momentum_consistency_score', 'liquidity_score',
    'atr', 'atr_percent'
]

print("\n" + "="*100)
print(f"COMPREHENSIVE TECHNICAL ANALYSIS - {result[0]}")
print("="*100)
print(f"Technical Score: {result[1]:.4f}")
print(f"Created: {result[2]}")
print("="*100)

# Group by scoring component
groups = {
    'Momentum (23%)': ['price_1d_pct', 'price_7d_pct', 'momentum_30d_pct', 'relative_strength'],
    'RSI (12%)': ['rsi'],
    'MACD (10%)': ['macd', 'macd_line', 'macd_signal', 'macd_histogram'],
    'Moving Averages (12%)': ['above_50d_ma_pct', 'above_200d_ma_pct'],
    'Volume (12%)': ['volume', 'avg_volume_30d', 'volume_spike_ratio', 'volume_price_correlation'],
    'Volatility (10%)': ['volatility', 'volatility_rank', 'historical_volatility'],
    'Bollinger Bands': ['bollinger_upper', 'bollinger_lower', 'bollinger_position', 'bollinger_width'],
    'Relative Strength (10%)': ['sector_relative_strength'],
    'Beta (8%)': ['beta'],
    'ML Metrics (13%)': ['momentum_consistency_score', 'liquidity_score'],
    'ATR Indicators': ['atr', 'atr_percent']
}

null_count = 0
populated_count = 0
total_scoring_weight = 0
missing_scoring_weight = 0

for group_name, field_names in groups.items():
    print(f"\n📊 {group_name}")
    print("-" * 100)
    
    # Extract weight from group name
    weight_pct = 0
    if '(' in group_name and '%' in group_name:
        try:
            weight_pct = int(group_name.split('(')[1].split('%')[0])
        except:
            pass
    
    group_has_null = False
    for field in field_names:
        idx = columns.index(field)
        value = result[idx]
        
        if value is None:
            print(f"  ❌ {field}: NULL")
            null_count += 1
            group_has_null = True
        else:
            print(f"  ✅ {field}: {value}")
            populated_count += 1
    
    # Track scoring impact
    if weight_pct > 0:
        total_scoring_weight += weight_pct
        if group_has_null:
            missing_scoring_weight += weight_pct

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"Total columns checked: {len(columns) - 3}")  # Exclude ticker, technical_score, created_at
print(f"✅ Populated: {populated_count}")
print(f"❌ NULL: {null_count}")
print(f"Populated rate: {populated_count / (populated_count + null_count) * 100:.1f}%")
print()
print(f"Total scoring weight covered: {total_scoring_weight}%")
print(f"Missing scoring weight: {missing_scoring_weight}%")
print(f"Scoring completeness: {(total_scoring_weight - missing_scoring_weight) / total_scoring_weight * 100:.1f}%")
print("="*100)

# Identify critical missing fields
critical_missing = []
for group_name, field_names in groups.items():
    if '(' in group_name and '%' in group_name:
        weight = int(group_name.split('(')[1].split('%')[0])
        for field in field_names:
            idx = columns.index(field)
            if result[idx] is None and weight >= 10:
                critical_missing.append((field, group_name, weight))

if critical_missing:
    print("\n⚠️  CRITICAL MISSING FIELDS (≥10% weight):")
    for field, group, weight in critical_missing:
        print(f"  - {field} ({group})")
else:
    print("\n✅ No critical missing fields!")

cur.close()
conn.close()
