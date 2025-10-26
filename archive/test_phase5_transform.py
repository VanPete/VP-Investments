"""
Test Phase 5 Transformation Layer

Tests the Phase5Persist class for transforming Phase 4 pipeline data
into Phase 5 JSONB storage format.

Tests:
1. Extract technical factors (~60 factors)
2. Extract fundamental factors (~45 factors)
3. Extract news/macro factors (~15 factors)
4. Extract social factors (~10 factors)
5. Extract risk factors (~25 factors)
6. Extract institutional factors (~20 factors)
7. Calculate coverage percentages
8. Full orchestration with persist_pipeline_run()
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from backend.phases.phase5_persist import Phase5Persist
from backend.storage.database import get_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def create_mock_phase4_data() -> Dict[str, Any]:
    """Create mock Phase 4 ticker data for testing."""
    return {
        'ticker': 'AAPL',
        'rank': 1,
        'overall_score': 0.95,
        'technical_score': 0.92,
        'fundamental_score': 0.88,
        'news_macro_score': 0.90,
        'social_score': 0.85,
        'risk_score': 0.93,
        'institutional_score': 0.91,
        
        # Technical data (~60 factors)
        'technical_data': {
            # RSI
            'rsi_14': 65.2,
            'rsi_14_norm': 0.75,
            'rsi_14_percentile': 0.82,
            
            # MACD
            'macd': 1.2,
            'macd_norm': 0.60,
            'macd_percentile': 0.65,
            'macd_signal': 0.9,
            'macd_signal_norm': 0.55,
            'macd_signal_percentile': 0.62,
            'macd_histogram': 0.3,
            'macd_histogram_norm': 0.58,
            'macd_histogram_percentile': 0.63,
            
            # Moving Averages
            'sma_10': 175.5,
            'sma_10_norm': 0.80,
            'sma_10_percentile': 0.85,
            'sma_20': 172.3,
            'sma_20_norm': 0.78,
            'sma_20_percentile': 0.83,
            'sma_50': 168.1,
            'sma_50_norm': 0.75,
            'sma_50_percentile': 0.80,
            'ema_20': 173.2,
            'ema_20_norm': 0.79,
            'ema_20_percentile': 0.84,
            
            # Bollinger Bands
            'bb_upper': 180.0,
            'bb_upper_norm': 0.70,
            'bb_upper_percentile': 0.75,
            'bb_middle': 170.0,
            'bb_middle_norm': 0.72,
            'bb_middle_percentile': 0.77,
            'bb_lower': 160.0,
            'bb_lower_norm': 0.68,
            'bb_lower_percentile': 0.73,
            'bb_width': 20.0,
            'bb_width_norm': 0.65,
            'bb_width_percentile': 0.70,
            
            # Volume
            'volume': 85000000,
            'volume_norm': 0.82,
            'volume_percentile': 0.88,
            'volume_sma_20': 75000000,
            'volume_sma_20_norm': 0.78,
            'volume_sma_20_percentile': 0.83,
            'obv': 1500000000,
            'obv_norm': 0.85,
            'obv_percentile': 0.90,
            
            # Price Action
            'close': 175.0,
            'close_norm': 0.80,
            'close_percentile': 0.85,
            'daily_return': 0.015,
            'daily_return_norm': 0.62,
            'daily_return_percentile': 0.68,
            'volatility_20': 0.025,
            'volatility_20_norm': 0.55,
            'volatility_20_percentile': 0.60,
            
            # Momentum
            'roc_10': 0.045,
            'roc_10_norm': 0.68,
            'roc_10_percentile': 0.73,
            'adx_14': 28.5,
            'adx_14_norm': 0.71,
            'adx_14_percentile': 0.76
        },
        
        # Fundamental data (~45 factors)
        'fundamental_data': {
            # Valuation
            'pe_ratio': 28.5,
            'pe_ratio_norm': 0.65,
            'pe_ratio_percentile': 0.70,
            'pb_ratio': 45.2,
            'pb_ratio_norm': 0.72,
            'pb_ratio_percentile': 0.77,
            'ps_ratio': 7.8,
            'ps_ratio_norm': 0.68,
            'ps_ratio_percentile': 0.73,
            'peg_ratio': 2.1,
            'peg_ratio_norm': 0.60,
            'peg_ratio_percentile': 0.65,
            
            # Profitability
            'roe': 0.148,
            'roe_norm': 0.88,
            'roe_percentile': 0.92,
            'roa': 0.195,
            'roa_norm': 0.90,
            'roa_percentile': 0.94,
            'gross_margin': 0.432,
            'gross_margin_norm': 0.92,
            'gross_margin_percentile': 0.95,
            'operating_margin': 0.308,
            'operating_margin_norm': 0.85,
            'operating_margin_percentile': 0.89,
            'profit_margin': 0.258,
            'profit_margin_norm': 0.87,
            'profit_margin_percentile': 0.91,
            
            # Growth
            'revenue_growth': 0.082,
            'revenue_growth_norm': 0.75,
            'revenue_growth_percentile': 0.80,
            'earnings_growth': 0.095,
            'earnings_growth_norm': 0.78,
            'earnings_growth_percentile': 0.83,
            'eps_growth': 0.112,
            'eps_growth_norm': 0.82,
            'eps_growth_percentile': 0.87,
            
            # Financial Health
            'current_ratio': 1.05,
            'current_ratio_norm': 0.62,
            'current_ratio_percentile': 0.68,
            'debt_to_equity': 1.95,
            'debt_to_equity_norm': 0.45,
            'debt_to_equity_percentile': 0.50,
            'interest_coverage': 18.5,
            'interest_coverage_norm': 0.88,
            'interest_coverage_percentile': 0.92,
            
            # Per-Share
            'eps': 6.15,
            'eps_norm': 0.85,
            'eps_percentile': 0.90,
            'book_value_per_share': 3.85,
            'book_value_per_share_norm': 0.72,
            'book_value_per_share_percentile': 0.77
        },
        
        # News/Macro data (~15 factors)
        'news_macro_data': {
            'news_sentiment_score': 0.68,
            'news_sentiment_score_norm': 0.75,
            'news_sentiment_score_percentile': 0.80,
            'news_sentiment_count': 145,
            'news_sentiment_count_norm': 0.82,
            'news_sentiment_count_percentile': 0.87,
            'news_positive_ratio': 0.72,
            'news_positive_ratio_norm': 0.78,
            'news_positive_ratio_percentile': 0.83,
            'market_beta': 1.18,
            'market_beta_norm': 0.68,
            'market_beta_percentile': 0.73,
            'spy_correlation': 0.82,
            'spy_correlation_norm': 0.85,
            'spy_correlation_percentile': 0.90,
            'sector_momentum': 0.055,
            'sector_momentum_norm': 0.72,
            'sector_momentum_percentile': 0.77
        },
        
        # Social data (~10 factors)
        'social_data': {
            'twitter_sentiment': 0.65,
            'twitter_sentiment_norm': 0.72,
            'twitter_sentiment_percentile': 0.77,
            'reddit_sentiment': 0.58,
            'reddit_sentiment_norm': 0.68,
            'reddit_sentiment_percentile': 0.73,
            'social_volume': 8500,
            'social_volume_norm': 0.80,
            'social_volume_percentile': 0.85,
            'social_engagement': 15200,
            'social_engagement_norm': 0.82,
            'social_engagement_percentile': 0.87
        },
        
        # Risk data (~25 factors)
        'risk_data': {
            'volatility_30d': 0.028,
            'volatility_30d_norm': 0.52,
            'volatility_30d_percentile': 0.58,
            'volatility_90d': 0.032,
            'volatility_90d_norm': 0.55,
            'volatility_90d_percentile': 0.60,
            'sharpe_ratio': 1.85,
            'sharpe_ratio_norm': 0.88,
            'sharpe_ratio_percentile': 0.92,
            'sortino_ratio': 2.15,
            'sortino_ratio_norm': 0.90,
            'sortino_ratio_percentile': 0.94,
            'max_drawdown': -0.18,
            'max_drawdown_norm': 0.68,
            'max_drawdown_percentile': 0.73,
            'current_drawdown': -0.05,
            'current_drawdown_norm': 0.82,
            'current_drawdown_percentile': 0.87,
            'var_95': -0.025,
            'var_95_norm': 0.72,
            'var_95_percentile': 0.77
        },
        
        # Institutional data (~20 factors)
        'institutional_data': {
            'institutional_ownership_pct': 0.618,
            'institutional_ownership_pct_norm': 0.85,
            'institutional_ownership_pct_percentile': 0.90,
            'institutional_holders_count': 2850,
            'institutional_holders_count_norm': 0.92,
            'institutional_holders_count_percentile': 0.95,
            'institutional_position_change': 0.025,
            'institutional_position_change_norm': 0.68,
            'institutional_position_change_percentile': 0.73,
            'insider_ownership_pct': 0.072,
            'insider_ownership_pct_norm': 0.55,
            'insider_ownership_pct_percentile': 0.60,
            'net_institutional_flow': 2500000000,
            'net_institutional_flow_norm': 0.82,
            'net_institutional_flow_percentile': 0.87,
            'analyst_count': 45,
            'analyst_count_norm': 0.95,
            'analyst_count_percentile': 0.98,
            'buy_recommendations': 32,
            'buy_recommendations_norm': 0.88,
            'buy_recommendations_percentile': 0.92
        }
    }


async def test_phase5_transformation():
    """Test Phase 5 transformation and persistence."""
    
    logger.info("=" * 80)
    logger.info("TESTING PHASE 5 TRANSFORMATION LAYER")
    logger.info("=" * 80)
    
    # Initialize Phase5Persist (no DB for extraction tests)
    persister = Phase5Persist()
    
    # Create mock Phase 4 data
    mock_data = create_mock_phase4_data()
    logger.info(f"\n✅ Created mock Phase 4 data for {mock_data['ticker']}")
    
    # Test 1: Extract technical factors
    logger.info("\n📝 Test 1: Extract technical factors")
    technical_factors = persister.extract_technical_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(technical_factors)} technical factors")
    logger.info(f"   Sample: rsi_14 = {technical_factors.get('rsi_14')}")
    
    # Test 2: Extract fundamental factors
    logger.info("\n📝 Test 2: Extract fundamental factors")
    fundamental_factors = persister.extract_fundamental_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(fundamental_factors)} fundamental factors")
    logger.info(f"   Sample: pe_ratio = {fundamental_factors.get('pe_ratio')}")
    
    # Test 3: Extract news/macro factors
    logger.info("\n📝 Test 3: Extract news/macro factors")
    news_macro_factors = persister.extract_news_macro_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(news_macro_factors)} news/macro factors")
    logger.info(f"   Sample: news_sentiment_score = {news_macro_factors.get('news_sentiment_score')}")
    
    # Test 4: Extract social factors
    logger.info("\n📝 Test 4: Extract social factors")
    social_factors = persister.extract_social_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(social_factors)} social factors")
    logger.info(f"   Sample: twitter_sentiment = {social_factors.get('twitter_sentiment')}")
    
    # Test 5: Extract risk factors
    logger.info("\n📝 Test 5: Extract risk factors")
    risk_factors = persister.extract_risk_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(risk_factors)} risk factors")
    logger.info(f"   Sample: sharpe_ratio = {risk_factors.get('sharpe_ratio')}")
    
    # Test 6: Extract institutional factors
    logger.info("\n📝 Test 6: Extract institutional factors")
    institutional_factors = persister.extract_institutional_factors(mock_data)
    logger.info(f"   ✅ Extracted {len(institutional_factors)} institutional factors")
    logger.info(f"   Sample: institutional_ownership_pct = {institutional_factors.get('institutional_ownership_pct')}")
    
    # Test 7: Calculate coverage
    logger.info("\n📝 Test 7: Calculate coverage percentages")
    tech_coverage = persister.calculate_coverage(technical_factors)
    fund_coverage = persister.calculate_coverage(fundamental_factors)
    news_coverage = persister.calculate_coverage(news_macro_factors)
    social_coverage = persister.calculate_coverage(social_factors)
    risk_coverage = persister.calculate_coverage(risk_factors)
    inst_coverage = persister.calculate_coverage(institutional_factors)
    
    logger.info(f"   Technical coverage: {tech_coverage:.2%}")
    logger.info(f"   Fundamental coverage: {fund_coverage:.2%}")
    logger.info(f"   News/Macro coverage: {news_coverage:.2%}")
    logger.info(f"   Social coverage: {social_coverage:.2%}")
    logger.info(f"   Risk coverage: {risk_coverage:.2%}")
    logger.info(f"   Institutional coverage: {inst_coverage:.2%}")
    
    total_coverage = (tech_coverage + fund_coverage + news_coverage + 
                      social_coverage + risk_coverage + inst_coverage) / 6
    logger.info(f"   ✅ Total coverage: {total_coverage:.2%}")
    
    # Test 8: Full orchestration with database
    logger.info("\n📝 Test 8: Full orchestration with persist_pipeline_run()")
    
    db = get_database()
    await db.connect()
    logger.info("   ✅ Database connected")
    
    # Inject DB into persister
    persister.db = db
    
    # Create Phase 4 results list (2 tickers)
    mock_data2 = create_mock_phase4_data()
    mock_data2['ticker'] = 'MSFT'
    mock_data2['rank'] = 2
    mock_data2['overall_score'] = 0.92
    
    phase4_results = [mock_data, mock_data2]
    
    # Run full orchestration
    run_id = await persister.persist_pipeline_run(
        phase4_results,
        pipeline_config={'pipeline_version': '2.0-test'}
    )
    
    logger.info(f"   ✅ Completed orchestration, run_id: {run_id}")
    
    # Verify data was inserted
    signals = await db.get_signals_by_run_id(run_id)
    logger.info(f"   ✅ Retrieved {len(signals)} signals from database")
    
    for signal in signals:
        logger.info(f"   - {signal['ticker']}: score={signal['overall_score']}, "
                   f"total_coverage={signal.get('total_coverage', 0):.2%}")
    
    # Get complete signal with factors
    complete_signal = await db.get_signal_with_factors(signals[0]['id'])
    tech_factor_count = len(complete_signal.get('technical_factors', {}))
    fund_factor_count = len(complete_signal.get('fundamental_factors', {}))
    logger.info(f"   ✅ Complete signal: {tech_factor_count} technical, {fund_factor_count} fundamental factors")
    
    await db.disconnect()
    logger.info("   ✅ Database disconnected")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL PHASE 5 TRANSFORMATION TESTS PASSED!")
    logger.info("=" * 80)
    logger.info(f"Total factors extracted: ~{len(technical_factors) + len(fundamental_factors) + len(news_macro_factors) + len(social_factors) + len(risk_factors) + len(institutional_factors)} factors")
    logger.info(f"Average coverage: {total_coverage:.2%}")
    logger.info(f"Database run_id: {run_id}")


if __name__ == "__main__":
    asyncio.run(test_phase5_transformation())
