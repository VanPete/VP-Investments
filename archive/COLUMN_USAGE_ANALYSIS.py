"""
Comprehensive analysis of signals table column usage

This script identifies:
1. Which columns are used in weighted_score calculation
2. Which columns are used in AI strategy generation
3. Which columns are unused/orphaned
"""

# ALL SIGNALS TABLE COLUMNS (137 total)
ALL_COLUMNS = [
    '10d_return', '10d_return_net', '1d_return', '1d_return_net', '30d_return',
    '3d_return', '3d_return_net', '7d_return', '7d_return_net', 'above_200d_ma_pct',
    'above_50d_ma_pct', 'ai_commentary', 'ai_commentary_version', 'ai_news_summary',
    'ai_trends_commentary', 'analyst_targets', 'avg_daily_value_traded', 'avg_daily_volume',
    'avg_volume_30d', 'backtest_eligible', 'backtest_intervals', 'backtest_notes',
    'backtest_phase', 'backtest_timestamp', 'beat_spy_10d', 'beat_spy_1d',
    'beat_spy_3d', 'beat_spy_7d', 'beta', 'bollinger_lower',
    'bollinger_position', 'bollinger_upper', 'bollinger_width', 'commentary_metadata',
    'company', 'created_at', 'current_price', 'debt_equity', 'dividend_ex_date',
    'drawdown_pct', 'earnings_date', 'earnings_gap_pct', 'emerging', 'entry_quality_score',
    'eps_growth', 'exit_signal_strength', 'expected_hold_duration', 'fcf_margin',
    'financial_score', 'float_turnover_ratio', 'forward_sharpe_ratio', 'forward_volatility',
    'historical_success_rate', 'id', 'implied_volatility', 'insider_buy_volume',
    'institutional_flow_direction', 'institutional_ownership_pct', 'iv_spike_pct',
    'last_backtest_intervals', 'liquidity_score', 'liquidity_warning', 'macd_histogram',
    'macd_line', 'macd_signal', 'market_cap', 'market_cap_category',
    'max_position_size', 'max_return_pct', 'mentions', 'ml_confidence_score',
    'momentum_30d_pct', 'momentum_consistency_score', 'news_mentions', 'news_score',
    'news_sentiment_score', 'option_chain_data', 'option_volume_ratio', 'options_flow_score',
    'pattern_match_score', 'pe_ratio', 'post_recency', 'prediction_confidence',
    'price_1d_pct', 'price_7d_pct', 'put_call_oi_ratio', 'put_call_vol_ratio',
    'rank', 'realized_returns', 'reddit_momentum_score', 'reddit_score',
    'reddit_sentiment', 'reddit_summary', 'reddit_vs_price_divergence', 'relative_strength',
    'retail_holding_pct', 'risk_adjusted_score', 'risk_assessment', 'risk_category',
    'risk_level', 'risk_score', 'risk_tags', 'roe',
    'rowid', 'rsi', 'run_id', 'score_components',
    'score_explanation', 'scoring_version', 'sector', 'sector_relative_strength',
    'shares_short', 'short_pct_float', 'short_pct_outstanding', 'short_ratio',
    'signal_confidence', 'signal_duration', 'signal_strength_percentile', 'signal_type',
    'social_sentiment_trend', 'spy_10d_return', 'spy_1d_return', 'spy_3d_return',
    'spy_7d_return', 'thread_tag', 'ticker', 'top_factors',
    'trade_type', 'unusual_options_activity', 'updated_at', 'upvotes',
    'volatility', 'volatility_rank', 'volume', 'volume_price_correlation',
    'volume_spike_ratio', 'weighted_score'
]

# COLUMNS USED IN WEIGHTED_SCORE CALCULATION
# Based on pipeline.py lines 1700-1770
WEIGHTED_SCORE_INPUTS = [
    # Core scoring components (the only 3 that matter!)
    'reddit_score',      # Weight: 0.10 (10%)
    'financial_score',   # Weight: 0.40 (40%)
    'news_score',        # Weight: 0.00 (0% - disabled)
    
    # Metadata for combining signals
    'ticker',            # Used to group signals by ticker
]

WEIGHTED_SCORE_OUTPUT = [
    'weighted_score',    # Final combined score
    'signal_confidence', # Set to weighted_score value
]

# COLUMNS USED IN AI STRATEGY GENERATION
# Based on ai.py _analyze_signal_characteristics() and strategy generation
AI_STRATEGY_INPUTS = [
    # Core identification
    'id',                      # signal_id
    'ticker',                  # Ticker symbol
    
    # Signal metrics
    'weighted_score',          # Overall signal strength
    'signal_confidence',       # Confidence level
    
    # Market characteristics
    'market_cap',              # Company size
    'market_cap_category',     # Size category (Micro/Small/Mid/Large/Mega)
    'current_price',           # Share price
    
    # Technical indicators
    'rsi',                     # Relative Strength Index
    'momentum_30d_pct',        # 30-day momentum
    'relative_strength',       # Relative strength vs market
    'volatility',              # Price volatility
    
    # Risk metrics
    'risk_score',              # Overall risk score
    'risk_level',              # Risk level (Low/Medium/High)
    'max_position_size',       # Maximum position size (%)
    'liquidity_score',         # Liquidity score
]

# COLUMNS POPULATED BY THE PIPELINE (not used in scoring, but filled)
PIPELINE_POPULATED = [
    # Core fields
    'run_id', 'ticker', 'company', 'sector', 'created_at', 'updated_at',
    'rank', 'signal_type', 'trade_type',
    
    # Scores
    'weighted_score', 'reddit_score', 'news_score', 'financial_score', 'signal_confidence',
    
    # Risk fields
    'risk_level', 'risk_tags', 'risk_assessment', 'risk_score', 'risk_category', 'max_position_size',
    
    # Commentary
    'top_factors', 'score_explanation', 'ai_commentary', 'reddit_summary', 
    'ai_news_summary', 'ai_trends_commentary', 'thread_tag',
    
    # Price & market data
    'current_price', 'market_cap', 'avg_daily_value_traded', 'market_cap_category',
    
    # Social metrics
    'reddit_sentiment', 'news_sentiment_score', 'mentions', 'news_mentions', 
    'upvotes', 'post_recency', 'emerging',
    
    # Price action
    'price_1d_pct', 'price_7d_pct', 'volume', 'liquidity_warning',
    
    # Technical indicators (Phase 1.1)
    'relative_strength', 'momentum_30d_pct', 'rsi', 'macd_histogram', 'macd_line', 
    'macd_signal', 'signal_strength_percentile', 'sector_relative_strength', 'momentum_consistency_score',
    'volatility', 'volatility_rank', 'bollinger_width', 'bollinger_upper', 'bollinger_lower', 
    'bollinger_position', 'beta', 'above_50d_ma_pct', 'above_200d_ma_pct',
    'volume_spike_ratio', 'volume_price_correlation', 'avg_daily_volume', 'avg_volume_30d',
    
    # Fundamentals
    'pe_ratio', 'earnings_gap_pct', 'eps_growth', 'roe', 'debt_equity', 'fcf_margin',
    
    # Options data
    'put_call_oi_ratio', 'put_call_vol_ratio', 'iv_spike_pct', 'implied_volatility',
    
    # Ownership data
    'institutional_ownership_pct', 'retail_holding_pct', 'insider_buy_volume',
    'short_pct_float', 'short_pct_outstanding', 'shares_short', 'short_ratio',
    
    # Composite scores (Phase 1.2)
    'exit_signal_strength', 'liquidity_score', 'float_turnover_ratio',
    'expected_hold_duration',
    
    # Calendar events (Phase 1.3)
    'earnings_date', 'dividend_ex_date', 'analyst_targets',
]

# COLUMNS POPULATED BY BACKTEST SYSTEM (after signal generation)
BACKTEST_POPULATED = [
    'backtest_eligible', 'backtest_intervals', 'backtest_notes', 'backtest_phase', 
    'backtest_timestamp', 'last_backtest_intervals', 'historical_success_rate',
    '1d_return', '1d_return_net', '3d_return', '3d_return_net', '7d_return', 
    '7d_return_net', '10d_return', '10d_return_net', '30d_return',
    'beat_spy_1d', 'beat_spy_3d', 'beat_spy_7d', 'beat_spy_10d',
    'spy_1d_return', 'spy_3d_return', 'spy_7d_return', 'spy_10d_return',
    'max_return_pct', 'drawdown_pct', 'forward_volatility', 'forward_sharpe_ratio',
    'realized_returns',
]

# UNUSED/ORPHANED COLUMNS (exist but not populated or used anywhere)
UNUSED_COLUMNS = [
    # ML/Prediction columns (not implemented)
    'ml_confidence_score',
    'prediction_confidence',
    'pattern_match_score',
    
    # Options flow columns (not implemented)
    'option_chain_data',
    'option_volume_ratio',
    'options_flow_score',
    'unusual_options_activity',
    
    # Institutional flow (not implemented)
    'institutional_flow_direction',
    
    # Social metrics (Phase 2 - not yet implemented)
    'reddit_momentum_score',
    'reddit_vs_price_divergence',
    'social_sentiment_trend',
    
    # Quality scores (Phase 4 - not yet implemented)
    'entry_quality_score',
    'risk_adjusted_score',
    
    # Metadata fields (not actively used)
    'commentary_metadata',
    'score_components',
    'scoring_version',
    'ai_commentary_version',
    'signal_duration',
    
    # Database metadata
    'rowid',
]

# SUMMARY
print("="*100)
print("SIGNALS TABLE COLUMN USAGE ANALYSIS")
print("="*100)
print()

print(f"📊 TOTAL COLUMNS: {len(ALL_COLUMNS)}")
print()

print(f"🎯 WEIGHTED_SCORE INPUTS: {len(WEIGHTED_SCORE_INPUTS)} columns")
print(f"   {', '.join(WEIGHTED_SCORE_INPUTS)}")
print()

print(f"🤖 AI STRATEGY INPUTS: {len(AI_STRATEGY_INPUTS)} columns")
for col in AI_STRATEGY_INPUTS:
    print(f"   • {col}")
print()

print(f"✅ PIPELINE POPULATED: {len(PIPELINE_POPULATED)} columns")
print(f"   (These are filled during signal generation)")
print()

print(f"⏱️  BACKTEST POPULATED: {len(BACKTEST_POPULATED)} columns")
print(f"   (These are filled after signal generation during backtesting)")
print()

print(f"❌ UNUSED/ORPHANED: {len(UNUSED_COLUMNS)} columns")
print(f"   (These exist but are not populated or used)")
for col in UNUSED_COLUMNS:
    print(f"   • {col}")
print()

print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()
print("1. WEIGHTED SCORE CALCULATION")
print("   • Only uses 3 inputs: reddit_score (10%), financial_score (40%), news_score (0%)")
print("   • News is currently DISABLED (0% weight)")
print("   • This means 90% of the table columns have NO impact on signal ranking!")
print()
print("2. AI STRATEGY GENERATION")
print(f"   • Uses {len(AI_STRATEGY_INPUTS)} columns for strategy decisions")
print("   • Considers: market cap, price, technicals (RSI, momentum), risk, liquidity")
print("   • Much more comprehensive than weighted_score calculation")
print()
print("3. UNUSED COLUMNS")
print(f"   • {len(UNUSED_COLUMNS)} columns ({len(UNUSED_COLUMNS)/len(ALL_COLUMNS)*100:.1f}%) are orphaned")
print("   • Mostly ML/prediction features and Phase 2-4 placeholders")
print("   • Can be cleaned up or implemented")
print()
print("4. RECOMMENDATION")
print("   • Consider expanding weighted_score to use more technical indicators")
print("   • Currently only reddit_score + financial_score, ignoring all Phase 1 metrics!")
print("   • Phase 1 metrics (RSI, momentum, liquidity, etc.) are calculated but not used in scoring")
print()
print("="*100)

# Calculate overlap
used_in_scoring = set(WEIGHTED_SCORE_INPUTS + WEIGHTED_SCORE_OUTPUT)
used_in_ai = set(AI_STRATEGY_INPUTS)
populated = set(PIPELINE_POPULATED)
backtested = set(BACKTEST_POPULATED)
unused = set(UNUSED_COLUMNS)

print()
print("OVERLAP ANALYSIS")
print("="*100)
print(f"Columns used in BOTH scoring and AI: {len(used_in_scoring & used_in_ai)}")
print(f"   {used_in_scoring & used_in_ai}")
print()
print(f"Columns populated but NOT used in scoring/AI: {len(populated - used_in_scoring - used_in_ai)}")
populated_not_used = populated - used_in_scoring - used_in_ai - backtested
print(f"   {len(populated_not_used)} columns are populated but never used:")
for col in sorted(populated_not_used):
    print(f"      • {col}")
print()
print("="*100)
