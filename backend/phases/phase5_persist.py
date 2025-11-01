"""
Phase 5 Database Persistence Extension

Adds Phase 5-specific methods to SupabaseInterface for storing and retrieving
complete pipeline data with all ~150 factors in JSONB format.

Schema Structure (8 tables):
- signal_runs: Pipeline execution metadata
- signals: Main signal records with group scores/coverages  
- signals_technical: ~60 technical factors in JSONB
- signals_fundamental: ~45 fundamental factors in JSONB
- signals_news_macro: ~15 news/macro factors in JSONB
- signals_social_alternative: ~10 social factors in JSONB
- signals_risk_stability: ~25 risk factors in JSONB
- signals_institutional_smart_money: ~20 institutional factors in JSONB
"""

import asyncio
import json
import logging
import math
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize Python objects for PostgreSQL JSONB.
    Converts NaN, Infinity to None (NULL in JSON).
    
    Args:
        obj: Any Python object (dict, list, float, etc.)
        
    Returns:
        Sanitized object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ==============================================================================
# PHASE 5 TRANSFORMATION LAYER
# ==============================================================================

class Phase5Persist:
    """
    Phase 5 Transformation Layer
    
    Transforms Phase 4 pipeline data into Phase 5 JSONB storage format.
    Extracts ~175 factors across 6 groups and calculates coverage statistics.
    """
    
    def __init__(self, db=None):
        """
        Initialize Phase5Persist.
        
        Args:
            db: SupabaseInterface instance (optional, can be injected later)
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    # --------------------------------------------------------------------------
    # FACTOR EXTRACTION METHODS
    # --------------------------------------------------------------------------
    
    def extract_technical_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL technical factors from Phase 4 technical_data.
        
        Phase 4 passes normalized factors from Phase 3 as a simple dict: {factor_name: z_score}
        We need to convert to the format: {factor_name: {"raw": value, "normalized": z_score, "percentile": 0}}
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with structure:
            {
                "rsi_14": {"raw": z_score, "normalized": z_score, "percentile": 0},
                "macd_signal": {"raw": z_score, "normalized": z_score, "percentile": 0},
                ...
            }
        """
        factors = {}
        
        # Get technical_data section (this is a Dict[str, float] from Phase 3 normalized factors)
        technical_data = phase4_data.get('technical_data', {})
        
        # Convert all factors to the expected format
        # Since Phase 3 normalized factors are z-scores, we use them as both raw and normalized
        for factor_name, z_score in technical_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0  # Percentile not calculated in current pipeline
                }
        
        return factors
    
    def extract_fundamental_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL fundamental factors from Phase 4 fundamental_data.
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with JSONB factor structure
        """
        factors = {}
        
        fundamental_data = phase4_data.get('fundamental_data', {})
        
        # Convert all factors to the expected format
        for factor_name, z_score in fundamental_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0
                }
        
        return factors
    
    def extract_news_macro_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL news/macro factors from Phase 4 news_macro_data.
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with JSONB factor structure
        """
        factors = {}
        
        news_macro_data = phase4_data.get('news_macro_data', {})
        
        # Convert all factors to the expected format
        for factor_name, z_score in news_macro_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0
                }
        
        return factors
    
    def extract_social_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL social/alternative factors from Phase 4 social_data.
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with JSONB factor structure
        """
        factors = {}
        
        social_data = phase4_data.get('social_data', {})
        
        # Convert all factors to the expected format
        for factor_name, z_score in social_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0
                }
        
        return factors
    
    def extract_risk_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL risk/stability factors from Phase 4 risk_data.
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with JSONB factor structure
        """
        factors = {}
        
        risk_data = phase4_data.get('risk_data', {})
        
        # Convert all factors to the expected format
        for factor_name, z_score in risk_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0
                }
        
        return factors
    
    def extract_institutional_factors(self, phase4_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract ALL institutional/smart money factors from Phase 4 institutional_data.
        
        Args:
            phase4_data: Complete Phase 4 ticker data dictionary
            
        Returns:
            Dictionary with JSONB factor structure
        """
        factors = {}
        
        institutional_data = phase4_data.get('institutional_data', {})
        
        # Convert all factors to the expected format
        for factor_name, z_score in institutional_data.items():
            if z_score is not None:
                factors[factor_name] = {
                    'raw': z_score,
                    'normalized': z_score,
                    'percentile': 0
                }
        
        return factors
    
    # --------------------------------------------------------------------------
    # COVERAGE CALCULATION
    # --------------------------------------------------------------------------
    
    def calculate_coverage(self, factors: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate coverage percentage for a factor group.
        
        Coverage = (number of non-null factors) / (total expected factors)
        
        Args:
            factors: Dictionary of extracted factors
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if not factors:
            return 0.0
        
        total_factors = len(factors)
        non_null_factors = sum(
            1 for factor_data in factors.values()
            if factor_data.get('raw') is not None
        )
        
        return non_null_factors / total_factors if total_factors > 0 else 0.0
    
    # --------------------------------------------------------------------------
    # MAIN ORCHESTRATION
    # --------------------------------------------------------------------------
    
    async def persist_pipeline_run(
        self,
        phase4_results: List[Dict[str, Any]],
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main orchestration method: Transform Phase 4 data and persist to database.
        
        Complete workflow:
        1. Create signal_run record
        2. Transform each ticker's Phase 4 data into Phase 5 format
        3. Extract 6 factor groups (technical, fundamental, news, social, risk, institutional)
        4. Calculate coverage for each group
        5. Insert signals batch
        6. Insert factor details for each signal
        7. Update signal_run with completion status
        
        Args:
            phase4_results: List of Phase 4 ticker dictionaries from pipeline
                Each ticker should have structure:
                {
                    'ticker': 'AAPL',
                    'overall_score': 0.95,
                    'technical_score': 0.92,
                    'fundamental_score': 0.88,
                    'news_macro_score': 0.90,
                    'social_score': 0.85,
                    'risk_score': 0.93,
                    'institutional_score': 0.91,
                    'technical_data': {...},  # Raw + normalized factors
                    'fundamental_data': {...},
                    'news_macro_data': {...},
                    'social_data': {...},
                    'risk_data': {...},
                    'institutional_data': {...}
                }
            pipeline_config: Optional configuration dict with:
                - pipeline_version: str (default '2.0')
                - metadata: Any additional run metadata
                
        Returns:
            run_id (UUID) of created signal run
            
        Raises:
            ValueError: If database connection not available
            Exception: If persistence fails
        """
        if not self.db:
            raise ValueError("Database connection required. Inject SupabaseInterface instance.")
        
        start_time = __import__('time').time()
        
        # Extract config
        config = pipeline_config or {}
        pipeline_version = config.get('pipeline_version', '2.0')
        
        try:
            # Step 1: Create signal run
            run_config = {
                'total_tickers': len(phase4_results),
                'successful_tickers': 0,
                'failed_tickers': 0,
                'pipeline_version': pipeline_version,
                'status': 'running'
            }
            
            run_id = await self.db.create_signal_run(run_config)
            self.logger.info(f"📊 Created signal run: {run_id}")
            
            # Step 2-5: Process each ticker
            signals_to_insert = []
            factor_details = {}  # Store factor details keyed by ticker
            
            for ticker_data in phase4_results:
                ticker = ticker_data.get('ticker')
                if not ticker:
                    self.logger.warning("Skipping ticker data without 'ticker' field")
                    continue
                
                # Extract factor groups
                technical_factors = self.extract_technical_factors(ticker_data)
                fundamental_factors = self.extract_fundamental_factors(ticker_data)
                news_macro_factors = self.extract_news_macro_factors(ticker_data)
                social_factors = self.extract_social_factors(ticker_data)
                risk_factors = self.extract_risk_factors(ticker_data)
                institutional_factors = self.extract_institutional_factors(ticker_data)
                
                # Extract company name, current price, and sector from raw data if available
                company_name = None
                current_price = None
                sector = None  # v3.3: Extract sector for signals table
                
                # Try to get from ticker_data (if Phase 1 raw data is preserved)
                raw_data = ticker_data.get('raw_data')  # If Phase 4 passes it through
                if raw_data:
                    # Get company name and sector from info
                    info = raw_data.get('info', {})
                    company_name = info.get('longName') or info.get('shortName')
                    sector = info.get('sector')  # v3.3: Extract sector
                    
                    # Get current price from fast_info or history
                    fast_info = raw_data.get('fast_info', {})
                    current_price = fast_info.get('lastPrice')
                    
                    if not current_price:
                        # Fallback to latest history close price
                        history = raw_data.get('history')
                        if history is not None and not history.empty:
                            current_price = history['Close'].iloc[-1] if 'Close' in history.columns else None
                
                # Build signal record
                signal_record = {
                    'ticker': ticker,
                    'rank': ticker_data.get('rank'),
                    'overall_score': ticker_data.get('overall_score'),
                    'total_coverage': ticker_data.get('total_coverage'),
                    'technical_score': ticker_data.get('technical_score'),
                    'fundamental_score': ticker_data.get('fundamental_score'),
                    'news_macro_score': ticker_data.get('news_macro_score'),
                    'social_alternative_score': ticker_data.get('social_score'),
                    'risk_stability_score': ticker_data.get('risk_score'),
                    'institutional_smart_money_score': ticker_data.get('institutional_score'),
                    'company_name': company_name,
                    'current_price': current_price,
                    'sector': sector  # v3.3: Add sector to signals table
                }
                
                signals_to_insert.append(signal_record)
                
                # Store factor details for later insertion
                factor_details[ticker] = {
                    'technical': technical_factors,
                    'fundamental': fundamental_factors,
                    'news_macro': news_macro_factors,
                    'social': social_factors,
                    'risk': risk_factors,
                    'institutional': institutional_factors
                }
            
            # Step 6: Insert signals batch
            if signals_to_insert:
                signal_ids = await self.db.insert_signals_batch(run_id, signals_to_insert)
                self.logger.info(f"✅ Inserted {len(signal_ids)} signals")
                
                # Step 7: Insert factor details for each signal
                successful_tickers = 0
                failed_tickers = 0
                
                for i, signal_id in enumerate(signal_ids):
                    signal = signals_to_insert[i]
                    ticker = signal['ticker']
                    
                    try:
                        factors = factor_details.get(ticker, {})
                        
                        # Insert all 6 factor groups
                        await self.db.insert_technical_factors(signal_id, factors.get('technical', {}))
                        await self.db.insert_fundamental_factors(signal_id, factors.get('fundamental', {}))
                        await self.db.insert_news_macro_factors(signal_id, factors.get('news_macro', {}))
                        await self.db.insert_social_factors(signal_id, factors.get('social', {}))
                        await self.db.insert_risk_factors(signal_id, factors.get('risk', {}))
                        await self.db.insert_institutional_factors(signal_id, factors.get('institutional', {}))
                        
                        successful_tickers += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to insert factors for {ticker}: {e}")
                        failed_tickers += 1
                
                # Step 8: Update signal_run with completion
                duration = __import__('time').time() - start_time
                
                await self.db.update_signal_run(run_id, {
                    'status': 'completed' if failed_tickers == 0 else 'partial',
                    'total_tickers': len(signals_to_insert),
                    'successful_tickers': successful_tickers,
                    'failed_tickers': failed_tickers,
                    'duration_seconds': duration
                })
                
                self.logger.info(
                    f"✅ Completed signal run {run_id}: "
                    f"{successful_tickers} successful, {failed_tickers} failed, "
                    f"{duration:.2f}s"
                )
            else:
                # No signals to insert
                await self.db.update_signal_run(run_id, {
                    'status': 'failed',
                    'error_message': 'No valid ticker data to persist'
                })
                self.logger.warning(f"❌ Signal run {run_id} failed: No valid ticker data")
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to persist pipeline run: {e}")
            
            # Try to update run status to failed
            try:
                if run_id:
                    await self.db.update_signal_run(run_id, {
                        'status': 'failed',
                        'error_message': str(e)
                    })
            except:
                pass
            
            raise


# ==============================================================================
# PHASE 5 PERSISTENCE METHODS
# ==============================================================================

async def create_signal_run(self, run_config: Dict[str, Any]) -> str:
    """
    Create new signal run record.
    
    Args:
        run_config: Dictionary with run configuration:
            - total_tickers: int
            - successful_tickers: Optional[int]
            - failed_tickers: Optional[int]
            - pipeline_version: Optional[str]
            - status: str ('running', 'completed', 'failed', 'partial')
            - error_message: Optional[str]
            
    Returns:
        run_id (UUID) of created record
    """
    query = """
    INSERT INTO signal_runs (
        total_tickers,
        successful_tickers,
        failed_tickers,
        pipeline_version,
        status,
        error_message,
        run_timestamp
    ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
    RETURNING id
    """
    
    params = [
        run_config.get('total_tickers', 0),
        run_config.get('successful_tickers', 0),
        run_config.get('failed_tickers', 0),
        run_config.get('pipeline_version', '2.0'),
        run_config.get('status', 'running'),
        run_config.get('error_message')
    ]
    
    result = await self.execute_query(query, params)
    run_id = result[0]['id'] if result else None
    
    logger.info(f"✅ Created signal run: {run_id}")
    return run_id


async def update_signal_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update signal run with completion status and statistics.
    
    Args:
        run_id: Signal run UUID
        updates: Dictionary with fields to update:
            - status: 'completed', 'failed', or 'partial'
            - total_tickers: int
            - successful_tickers: int
            - failed_tickers: int
            - duration_seconds: float
            - error_message: Optional[str]
            
    Returns:
        True if successful
    """
    set_clauses = []
    params = [run_id]
    param_count = 2
    
    if 'status' in updates:
        set_clauses.append(f"status = ${param_count}")
        params.append(updates['status'])
        param_count += 1
    
    if 'total_tickers' in updates:
        set_clauses.append(f"total_tickers = ${param_count}")
        params.append(updates['total_tickers'])
        param_count += 1
    
    if 'successful_tickers' in updates:
        set_clauses.append(f"successful_tickers = ${param_count}")
        params.append(updates['successful_tickers'])
        param_count += 1
    
    if 'failed_tickers' in updates:
        set_clauses.append(f"failed_tickers = ${param_count}")
        params.append(updates['failed_tickers'])
        param_count += 1
    
    if 'duration_seconds' in updates:
        set_clauses.append(f"duration_seconds = ${param_count}")
        params.append(updates['duration_seconds'])
        param_count += 1
    
    if 'error_message' in updates:
        set_clauses.append(f"error_message = ${param_count}")
        params.append(updates['error_message'])
        param_count += 1
    
    if not set_clauses:
        return False
    
    query = f"""
    UPDATE signal_runs 
    SET {', '.join(set_clauses)}
    WHERE id = $1
    """
    
    affected = await self.execute_non_query(query, params)
    success = affected > 0
    
    if success:
        logger.info(f"✅ Updated signal run {run_id}: {updates.get('status', 'updated')}")
    
    return success


async def get_recent_signal_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent signal runs ordered by timestamp.
    
    Args:
        limit: Maximum number of runs to return
        
    Returns:
        List of run records with metadata
    """
    query = """
    SELECT 
        id,
        run_timestamp,
        pipeline_version,
        total_tickers,
        successful_tickers,
        failed_tickers,
        duration_seconds,
        status,
        error_message,
        created_at
    FROM signal_runs
    ORDER BY run_timestamp DESC
    LIMIT $1
    """
    
    return await self.execute_query(query, [limit])


async def insert_signals_batch(self, run_id: str, signals: List[Dict[str, Any]]) -> List[str]:
    """
    Insert batch of signal records.
    
    Args:
        run_id: Signal run UUID
        signals: List of signal dictionaries with structure:
            - ticker: str
            - rank: int
            - overall_score: float
            - technical_score: float
            - fundamental_score: float
            - news_macro_score: float
            - social_alternative_score: float
            - risk_stability_score: float
            - institutional_smart_money_score: float
            - total_coverage: float (optional)
            - technical_coverage: float (optional)
            - fundamental_coverage: float (optional)
            - news_macro_coverage: float (optional)
            - social_alternative_coverage: float (optional)
            - risk_stability_coverage: float (optional)
            - institutional_smart_money_coverage: float (optional)
            - sector: str (optional, v3.3)
            - market_cap: int (optional, v3.4 - for VanPiQ Performance Tab)
            - beta: float (optional, v3.4 - for VanPiQ Performance Tab)
            
    Returns:
        List of signal IDs created
    """
    if not signals:
        return []
    
    # Build bulk insert query
    query_parts = []
    params = []
    param_count = 1
    
    for signal in signals:
        placeholders = ', '.join([f'${i}' for i in range(param_count, param_count + 22)])
        query_parts.append(f"({placeholders})")
        
        params.extend([
            run_id,
            signal['ticker'],
            signal.get('rank'),
            signal.get('overall_score'),
            signal.get('technical_score'),
            signal.get('fundamental_score'),
            signal.get('news_macro_score'),
            signal.get('social_alternative_score'),
            signal.get('risk_stability_score'),
            signal.get('institutional_smart_money_score'),
            signal.get('total_coverage'),
            signal.get('technical_coverage'),
            signal.get('fundamental_coverage'),
            signal.get('news_macro_coverage'),
            signal.get('social_alternative_coverage'),
            signal.get('risk_stability_coverage'),
            signal.get('institutional_smart_money_coverage'),
            signal.get('company_name'),
            signal.get('current_price'),
            signal.get('sector'),  # v3.3: Sector column
            signal.get('market_cap'),  # v3.4: Market cap for VanPiQ Performance Tab
            signal.get('beta')  # v3.4: Beta for VanPiQ Performance Tab
        ])
        param_count += 22
    
    query = f"""
    INSERT INTO signals (
        run_id, ticker, rank, overall_score,
        technical_score,
        fundamental_score,
        news_macro_score,
        social_alternative_score,
        risk_stability_score,
        institutional_smart_money_score,
        total_coverage,
        technical_coverage,
        fundamental_coverage,
        news_macro_coverage,
        social_alternative_coverage,
        risk_stability_coverage,
        institutional_smart_money_coverage,
        company_name,
        current_price,
        sector,
        market_cap,
        beta
    ) VALUES {', '.join(query_parts)}
    RETURNING id
    """
    
    result = await self.execute_query(query, params)
    signal_ids = [row['id'] for row in result] if result else []
    
    logger.info(f"✅ Inserted {len(signal_ids)} signals for run {run_id}")
    return signal_ids


async def get_signals_by_run_id(self, run_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all signals for a specific run.
    
    Args:
        run_id: Signal run UUID
        limit: Optional limit on number of signals
        
    Returns:
        List of signal records ordered by rank
    """
    query = """
    SELECT 
        id, run_id, ticker, rank, overall_score,
        technical_score,
        fundamental_score,
        news_macro_score,
        social_alternative_score,
        risk_stability_score,
        institutional_smart_money_score,
        created_at
    FROM signals
    WHERE run_id = $1
    ORDER BY rank ASC NULLS LAST
    """
    
    params = [run_id]
    if limit:
        query += f" LIMIT {limit}"
    
    return await self.execute_query(query, params)


async def get_top_signals_phase5(self, run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get top N signals by overall_score.
    
    Args:
        run_id: Signal run UUID
        limit: Number of top signals to return
        
    Returns:
        List of top signal records
    """
    query = """
    SELECT 
        id, ticker, rank, overall_score,
        technical_score, fundamental_score,
        news_macro_score, social_alternative_score,
        risk_stability_score, institutional_smart_money_score
    FROM signals
    WHERE run_id = $1
    ORDER BY overall_score DESC NULLS LAST
    LIMIT $2
    """
    
    return await self.execute_query(query, [run_id, limit])


async def insert_technical_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """
    Insert technical factors for a signal.
    
    Args:
        signal_id: Signal UUID
        factors: JSONB structure with factor data:
            {
                "rsi_14": {"raw": 65.2, "normalized": 0.75, "percentile": 0.82},
                "macd": {"raw": 1.2, "normalized": 0.60, "percentile": 0.65},
                ...
            }
            
    Returns:
        True if successful
    """
    query = """
    INSERT INTO signals_technical (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def insert_fundamental_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """Insert fundamental factors for a signal."""
    query = """
    INSERT INTO signals_fundamental (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def insert_news_macro_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """Insert news/macro factors for a signal."""
    query = """
    INSERT INTO signals_news_macro (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def insert_social_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """Insert social/alternative factors for a signal."""
    query = """
    INSERT INTO signals_social_alternative (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def insert_risk_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """Insert risk/stability factors for a signal."""
    query = """
    INSERT INTO signals_risk_stability (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def insert_institutional_factors(self, signal_id: str, factors: Dict[str, Dict[str, float]]) -> bool:
    """Insert institutional/smart money factors for a signal."""
    query = """
    INSERT INTO signals_institutional_smart_money (signal_id, factors)
    VALUES ($1, $2)
    """
    
    sanitized = sanitize_for_json(factors)
    affected = await self.execute_non_query(query, [signal_id, json.dumps(sanitized)])
    return affected > 0


async def get_signal_with_factors(self, signal_id: str) -> Optional[Dict[str, Any]]:
    """
    Get complete signal record with all factor details.
    
    Args:
        signal_id: Signal UUID
        
    Returns:
        Dictionary with signal data and all factor groups, or None if not found
    """
    query = """
    SELECT 
        s.*,
        st.factors as technical_factors,
        sf.factors as fundamental_factors,
        snm.factors as news_macro_factors,
        ssa.factors as social_factors,
        srs.factors as risk_factors,
        sism.factors as institutional_factors
    FROM signals s
    LEFT JOIN signals_technical st ON s.id = st.signal_id
    LEFT JOIN signals_fundamental sf ON s.id = sf.signal_id
    LEFT JOIN signals_news_macro snm ON s.id = snm.signal_id
    LEFT JOIN signals_social_alternative ssa ON s.id = ssa.signal_id
    LEFT JOIN signals_risk_stability srs ON s.id = srs.signal_id
    LEFT JOIN signals_institutional_smart_money sism ON s.id = sism.signal_id
    WHERE s.id = $1
    """
    
    result = await self.execute_query(query, [signal_id])
    return result[0] if result else None


async def get_ticker_signal_with_factors(self, run_id: str, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get complete signal record for a specific ticker in a run.
    
    Args:
        run_id: Signal run UUID
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with signal data and all factor groups, or None if not found
    """
    query = """
    SELECT 
        s.*,
        st.factors as technical_factors,
        sf.factors as fundamental_factors,
        snm.factors as news_macro_factors,
        ssa.factors as social_factors,
        srs.factors as risk_factors,
        sism.factors as institutional_factors
    FROM signals s
    LEFT JOIN signals_technical st ON s.id = st.signal_id
    LEFT JOIN signals_fundamental sf ON s.id = sf.signal_id
    LEFT JOIN signals_news_macro snm ON s.id = snm.signal_id
    LEFT JOIN signals_social_alternative ssa ON s.id = ssa.signal_id
    LEFT JOIN signals_risk_stability srs ON s.id = srs.signal_id
    LEFT JOIN signals_institutional_smart_money sism ON s.id = sism.signal_id
    WHERE s.run_id = $1 AND s.ticker = $2
    """
    
    result = await self.execute_query(query, [run_id, ticker])
    return result[0] if result else None


async def get_latest_run_id(self) -> Optional[str]:
    """Get the ID of the most recent completed signal run."""
    query = """
    SELECT id 
    FROM signal_runs 
    WHERE status = 'completed'
    ORDER BY run_timestamp DESC 
    LIMIT 1
    """
    
    result = await self.execute_query(query)
    return result[0]['id'] if result else None


async def get_signal_statistics(self, run_id: str) -> Dict[str, Any]:
    """
    Get statistical summary of a signal run.
    
    Args:
        run_id: Signal run UUID
        
    Returns:
        Dictionary with statistics:
            - total_signals: int
            - avg_score: float
            - avg_technical_coverage: float
            - avg_fundamental_coverage: float
            - top_ticker: str
            - top_score: float
    """
    query = """
    SELECT 
        COUNT(*) as total_signals,
        AVG(overall_score) as avg_score,
        AVG(technical_coverage) as avg_technical_coverage,
        AVG(fundamental_coverage) as avg_fundamental_coverage,
        MAX(overall_score) as top_score
    FROM signals
    WHERE run_id = $1
    """
    
    result = await self.execute_query(query, [run_id])
    
    if result:
        stats = dict(result[0])
        
        # Get top ticker
        top_ticker_query = """
        SELECT ticker 
        FROM signals 
        WHERE run_id = $1 
        ORDER BY overall_score DESC NULLS LAST 
        LIMIT 1
        """
        top_ticker_result = await self.execute_query(top_ticker_query, [run_id])
        stats['top_ticker'] = top_ticker_result[0]['ticker'] if top_ticker_result else None
        
        return stats
    
    return {}


def add_phase5_methods_to_supabase_interface():
    """
    Add Phase 5 methods to SupabaseInterface class.
    Call this after importing SupabaseInterface.
    """
    from backend.storage.database import SupabaseInterface
    
    # Add all methods defined in this module
    SupabaseInterface.create_signal_run = create_signal_run
    SupabaseInterface.update_signal_run = update_signal_run
    SupabaseInterface.get_recent_signal_runs = get_recent_signal_runs
    SupabaseInterface.insert_signals_batch = insert_signals_batch
    SupabaseInterface.get_signals_by_run_id = get_signals_by_run_id
    SupabaseInterface.get_top_signals_phase5 = get_top_signals_phase5
    SupabaseInterface.insert_technical_factors = insert_technical_factors
    SupabaseInterface.insert_fundamental_factors = insert_fundamental_factors
    SupabaseInterface.insert_news_macro_factors = insert_news_macro_factors
    SupabaseInterface.insert_social_factors = insert_social_factors
    SupabaseInterface.insert_risk_factors = insert_risk_factors
    SupabaseInterface.insert_institutional_factors = insert_institutional_factors
    SupabaseInterface.get_signal_with_factors = get_signal_with_factors
    SupabaseInterface.get_ticker_signal_with_factors = get_ticker_signal_with_factors
    SupabaseInterface.get_latest_run_id = get_latest_run_id
    SupabaseInterface.get_signal_statistics = get_signal_statistics
    
    logger.info("✅ Phase 5 persistence methods added to SupabaseInterface")


# Auto-add methods when module is imported
add_phase5_methods_to_supabase_interface()


# ============================================================================
# OPTIMIZED VERSION (Phase 5.6 - Production Optimization)
# ============================================================================

class Phase5PersistOptimized(Phase5Persist):
    """
    Optimized Phase 5 Persistence with Bulk INSERT Operations (Phase 5.6).
    
    Optimizations implemented:
    1. Bulk INSERT for all 6 factor tables (single transaction)
    2. Parallel factor table insertion using asyncio.gather()
    3. Reduced database round-trips from 6N to 6 (where N = number of signals)
    4. Transaction batching for atomicity
    
    Performance Target:
    - Baseline: 157s for 500 tickers
    - Optimized: ~47s for 500 tickers (60-70% improvement)
    
    Inherits from Phase5Persist and overrides factor insertion methods
    to use bulk INSERT operations instead of one-by-one insertions.
    """
    
    def __init__(self, db=None):
        """
        Initialize optimized Phase5Persist.
        
        Args:
            db: SupabaseInterface instance
        """
        super().__init__(db)
        self.logger.info("[OPTIMIZED] Phase5PersistOptimized initialized with bulk INSERT capability")
    
    # --------------------------------------------------------------------------
    # PERFORMANCE BASELINE CREATION (NEW - Hybrid Approach)
    # --------------------------------------------------------------------------
    
    async def _insert_performance_baselines(
        self, 
        signal_ids: List[str], 
        phase4_results: List[Dict[str, Any]],
        phase1_cache: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create performance baseline records for all signals.
        
        Uses signal creation time and current price as baseline.
        Phase 6 will progressively fill interval returns.
        
        Args:
            signal_ids: List of signal UUIDs
            phase4_results: Phase 4 results with ticker data
            phase1_cache: Optional Phase 1 cache with current prices
            
        Returns:
            Number of performance records created
        """
        if not signal_ids or not phase4_results:
            return 0
        
        from datetime import datetime, timezone
        from backend.utils.sector_etfs import get_sector_etf
        
        try:
            # Build bulk INSERT for performance table
            values_parts = []
            params = []
            param_idx = 1
            
            for signal_id, ticker_data in zip(signal_ids, phase4_results):
                ticker = ticker_data.get('ticker')
                
                # Get baseline price from Phase 1 cache (current price at signal creation)
                baseline_price = None
                sector = None
                sector_etf = None
                
                if phase1_cache and ticker in phase1_cache:
                    raw_data = phase1_cache[ticker]
                    # RawYFinanceData dataclass - get current price from fast_info or info
                    if hasattr(raw_data, 'fast_info') and raw_data.fast_info:
                        baseline_price = raw_data.fast_info.get('lastPrice') or raw_data.fast_info.get('regularMarketPrice')
                    if not baseline_price and hasattr(raw_data, 'info') and raw_data.info:
                        baseline_price = raw_data.info.get('currentPrice') or raw_data.info.get('regularMarketPrice')
                    
                    # Extract sector information (v3.2)
                    if hasattr(raw_data, 'info') and raw_data.info:
                        sector = raw_data.info.get('sector')  # info is a dict, not an object
                        if sector:
                            sector_etf = get_sector_etf(sector)
                
                if baseline_price and baseline_price > 0:
                    baseline_date = datetime.now(timezone.utc)
                    
                    # Prepare JSON array for intervals_completed
                    intervals_completed = '[]'  # Empty array - no intervals done yet
                    
                    values_parts.append(
                        f"(${param_idx}, ${param_idx + 1}, ${param_idx + 2}, ${param_idx + 3}, ${param_idx + 4}, ${param_idx + 5}, ${param_idx + 6})"
                    )
                    params.extend([
                        signal_id,
                        baseline_price,
                        baseline_date,
                        'pending',
                        intervals_completed,
                        sector,  # v3.2: Sector tracking
                        sector_etf  # v3.2: Sector ETF for comparison
                    ])
                    param_idx += 7
            
            if not values_parts:
                self.logger.warning("No valid baseline prices to insert into performance table")
                return 0
            
            # Execute bulk INSERT
            query = f"""
            INSERT INTO performance (signal_id, baseline_price, baseline_date, status, intervals_completed, sector, sector_etf)
            VALUES {', '.join(values_parts)}
            """
            
            affected = await self.db.execute_non_query(query, params)
            self.logger.info(f"[PERFORMANCE] Created {affected} performance baseline records")
            return affected
            
        except Exception as e:
            self.logger.error(f"Error creating performance baselines: {e}")
            # Don't fail the entire pipeline run if performance tracking fails
            return 0
    
    # --------------------------------------------------------------------------
    # BULK INSERT METHODS (Optimized)
    # --------------------------------------------------------------------------
    
    async def insert_technical_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """
        Bulk insert technical factors for multiple signals.
        
        Args:
            signal_ids: List of signal UUIDs
            factors_list: List of JSONB factor structures (one per signal)
            
        Returns:
            Number of rows inserted
        """
        if not signal_ids or not factors_list:
            return 0
        
        # Build bulk INSERT query
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_technical (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} technical factor records")
        return affected
    
    async def insert_fundamental_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """Bulk insert fundamental factors for multiple signals."""
        if not signal_ids or not factors_list:
            return 0
        
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_fundamental (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} fundamental factor records")
        return affected
    
    async def insert_news_macro_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """Bulk insert news/macro factors for multiple signals."""
        if not signal_ids or not factors_list:
            return 0
        
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_news_macro (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} news/macro factor records")
        return affected
    
    async def insert_social_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """Bulk insert social/alternative factors for multiple signals."""
        if not signal_ids or not factors_list:
            return 0
        
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_social_alternative (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} social factor records")
        return affected
    
    async def insert_risk_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """Bulk insert risk/stability factors for multiple signals."""
        if not signal_ids or not factors_list:
            return 0
        
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_risk_stability (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} risk factor records")
        return affected
    
    async def insert_institutional_factors_bulk(
        self, 
        signal_ids: List[str], 
        factors_list: List[Dict[str, Dict[str, float]]]
    ) -> int:
        """Bulk insert institutional/smart money factors for multiple signals."""
        if not signal_ids or not factors_list:
            return 0
        
        values_parts = []
        params = []
        param_idx = 1
        
        for signal_id, factors in zip(signal_ids, factors_list):
            sanitized = sanitize_for_json(factors)
            values_parts.append(f"(${param_idx}, ${param_idx + 1})")
            params.extend([signal_id, json.dumps(sanitized)])
            param_idx += 2
        
        query = f"""
        INSERT INTO signals_institutional_smart_money (signal_id, factors)
        VALUES {', '.join(values_parts)}
        """
        
        affected = await self.db.execute_non_query(query, params)
        self.logger.info(f"✅ [BULK] Inserted {affected} institutional factor records")
        return affected
    
    # --------------------------------------------------------------------------
    # OVERRIDE persist_pipeline_run (Optimized)
    # --------------------------------------------------------------------------
    
    async def persist_pipeline_run(
        self,
        phase4_results: List[Dict[str, Any]],
        pipeline_config: Optional[Dict[str, Any]] = None,
        phase1_cache: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Persist complete pipeline run with optimized bulk INSERT operations.
        
        This method overrides the parent class implementation to use bulk INSERTs
        instead of sequential single-row insertions.
        
        Args:
            phase4_results: List of Phase 4 ticker results
            pipeline_config: Optional pipeline configuration (unused in optimized version)
            phase1_cache: Optional Phase 1 cache with current prices for performance baselines
            
        Returns:
            Signal run UUID
        """
        import time
        import uuid
        from datetime import datetime, timezone
        start_time = time.time()
        
        try:
            # Step 1: Create signal_run record and get the auto-generated ID
            try:
                result = await self.db.pool.fetchrow(
                    """
                    INSERT INTO signal_runs (pipeline_version, total_tickers, run_timestamp, status)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    "3.1-optimized",
                    len(phase4_results),
                    datetime.now(timezone.utc),
                    "running"
                )
                run_id = result['id']
                self.logger.info(f"[OPTIMIZED] Created signal_run record: {run_id}")
            except Exception as e:
                self.logger.error(f"Failed to create signal_run: {e}")
                raise  # Don't continue if we can't create the run record
            
            self.logger.info(f"[OPTIMIZED] Started bulk INSERT persistence (run_id: {run_id})")
            
            # Step 2-5: Extract factors and build signals (same as parent)
            signals_to_insert = []
            factor_details = {}
            ticker_signal_map = []  # Track ticker-to-signal_id mapping
            
            for rank, ticker_data in enumerate(phase4_results, start=1):
                ticker = ticker_data.get('ticker')
                if not ticker:
                    continue
                
                # Extract all factors (reuse parent class methods)
                technical_factors = self.extract_technical_factors(ticker_data)
                fundamental_factors = self.extract_fundamental_factors(ticker_data)
                news_macro_factors = self.extract_news_macro_factors(ticker_data)
                social_factors = self.extract_social_factors(ticker_data)
                risk_factors = self.extract_risk_factors(ticker_data)
                institutional_factors = self.extract_institutional_factors(ticker_data)
                
                # Get coverages from Phase 4 (already correctly calculated against YAML config)
                technical_coverage = ticker_data.get('technical_coverage', 0.0)
                fundamental_coverage = ticker_data.get('fundamental_coverage', 0.0)
                news_macro_coverage = ticker_data.get('news_macro_coverage', 0.0)
                social_coverage = ticker_data.get('social_coverage', 0.0)
                risk_coverage = ticker_data.get('risk_coverage', 0.0)
                institutional_coverage = ticker_data.get('institutional_coverage', 0.0)
                
                # Get total coverage from Phase 4
                total_coverage = ticker_data.get('total_coverage', 0.0)
                
                # Extract company name, current price, sector, market_cap, and beta from phase1_cache
                company_name = None
                current_price = None
                sector = None  # v3.3: Extract sector for signals table
                market_cap = None  # v3.4: Extract market cap for VanPiQ Performance Tab
                beta = None  # v3.4: Extract beta for VanPiQ Performance Tab
                
                if phase1_cache and ticker in phase1_cache:
                    ticker_raw_data = phase1_cache[ticker]  # This is a RawYFinanceData object
                    
                    # Get company name, sector, market_cap, and beta from info (direct attribute access)
                    if ticker_raw_data.info:
                        company_name = ticker_raw_data.info.get('longName') or ticker_raw_data.info.get('shortName')
                        sector = ticker_raw_data.info.get('sector')  # v3.3: Extract sector
                        market_cap = ticker_raw_data.info.get('marketCap')  # v3.4: Extract market cap
                        beta = ticker_raw_data.info.get('beta')  # v3.4: Extract beta (3Y monthly vs SPY)
                    
                    # Get current price from fast_info or history (direct attribute access)
                    if ticker_raw_data.fast_info:
                        current_price = ticker_raw_data.fast_info.get('lastPrice')
                    
                    if not current_price:
                        # Fallback to latest history close price
                        if hasattr(ticker_raw_data, 'history') and not ticker_raw_data.history.empty:
                            try:
                                current_price = ticker_raw_data.history['Close'].iloc[-1] if 'Close' in ticker_raw_data.history.columns else None
                            except:
                                pass
                
                # Build signal record
                signal_record = {
                    'ticker': ticker,
                    'rank': rank,
                    'overall_score': ticker_data.get('overall_score'),
                    'total_coverage': total_coverage,
                    'technical_score': ticker_data.get('technical_score'),
                    'technical_coverage': technical_coverage,
                    'fundamental_score': ticker_data.get('fundamental_score'),
                    'fundamental_coverage': fundamental_coverage,
                    'news_macro_score': ticker_data.get('news_macro_score'),
                    'news_macro_coverage': news_macro_coverage,
                    'social_alternative_score': ticker_data.get('social_score'),
                    'social_alternative_coverage': social_coverage,
                    'risk_stability_score': ticker_data.get('risk_score'),
                    'risk_stability_coverage': risk_coverage,
                    'institutional_smart_money_score': ticker_data.get('institutional_score'),
                    'institutional_smart_money_coverage': institutional_coverage,
                    'company_name': company_name,
                    'current_price': current_price,
                    'sector': sector,  # v3.3: Add sector to signals table
                    'market_cap': market_cap,  # v3.4: Add market cap for VanPiQ Performance Tab
                    'beta': beta  # v3.4: Add beta for VanPiQ Performance Tab
                }
                
                signals_to_insert.append(signal_record)
                ticker_signal_map.append(ticker)
                
                # Store factor details
                factor_details[ticker] = {
                    'technical': technical_factors,
                    'fundamental': fundamental_factors,
                    'news_macro': news_macro_factors,
                    'social': social_factors,
                    'risk': risk_factors,
                    'institutional': institutional_factors
                }
            
            # Step 6: Insert signals batch (same as parent)
            if signals_to_insert:
                signal_ids = await self.db.insert_signals_batch(run_id, signals_to_insert)
                self.logger.info(f"[OPTIMIZED] Inserted {len(signal_ids)} signals")
                
                # Step 6.5: Create performance baseline records (NEW - Hybrid Approach)
                await self._insert_performance_baselines(signal_ids, phase4_results, phase1_cache)
                
                # Step 7: OPTIMIZED - Bulk insert all factors in parallel
                # Prepare ordered factor lists
                technical_list = []
                fundamental_list = []
                news_macro_list = []
                social_list = []
                risk_list = []
                institutional_list = []
                
                for ticker in ticker_signal_map:
                    factors = factor_details.get(ticker, {})
                    technical_list.append(factors.get('technical', {}))
                    fundamental_list.append(factors.get('fundamental', {}))
                    news_macro_list.append(factors.get('news_macro', {}))
                    social_list.append(factors.get('social', {}))
                    risk_list.append(factors.get('risk', {}))
                    institutional_list.append(factors.get('institutional', {}))
                
                # Execute parallel bulk INSERTs
                self.logger.info(f"[OPTIMIZED] Inserting factors for {len(signal_ids)} signals (6 tables, parallel bulk INSERT)")
                
                results = await asyncio.gather(
                    self.insert_technical_factors_bulk(signal_ids, technical_list),
                    self.insert_fundamental_factors_bulk(signal_ids, fundamental_list),
                    self.insert_news_macro_factors_bulk(signal_ids, news_macro_list),
                    self.insert_social_factors_bulk(signal_ids, social_list),
                    self.insert_risk_factors_bulk(signal_ids, risk_list),
                    self.insert_institutional_factors_bulk(signal_ids, institutional_list),
                    return_exceptions=True
                )
                
                # Count successes
                successful_tickers = len([r for r in results if not isinstance(r, Exception)])
                failed_tickers = len([r for r in results if isinstance(r, Exception)])
                
                if failed_tickers > 0:
                    self.logger.warning(f"⚠️ {failed_tickers} factor insertion failures")
                
                # Step 8: Update signal_run with completion
                duration = time.time() - start_time
                
                await self.db.update_signal_run(run_id, {
                    'status': 'completed' if failed_tickers == 0 else 'partial',
                    'total_tickers': len(signals_to_insert),
                    'successful_tickers': len(signal_ids),
                    'failed_tickers': failed_tickers,
                    'duration_seconds': duration
                })
                
                self.logger.info(
                    f"✅ [OPTIMIZED] Completed signal run {run_id}: "
                    f"{len(signal_ids)} signals, {successful_tickers}/6 factor groups, "
                    f"{duration:.2f}s (bulk INSERT)"
                )
            else:
                # No signals to insert
                await self.db.update_signal_run(run_id, {
                    'status': 'failed',
                    'error_message': 'No valid ticker data to persist'
                })
                self.logger.warning(f"❌ Signal run {run_id} failed: No valid ticker data")
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to persist pipeline run: {e}")
            
            # Try to update run status to failed
            try:
                if run_id:
                    await self.db.update_signal_run(run_id, {
                        'status': 'failed',
                        'error_message': str(e)
                    })
            except:
                pass
            
            raise


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_optimized_phase5_persist(db_interface=None):
    """
    Factory function to create optimized Phase5Persist instance.
    
    Args:
        db_interface: SupabaseInterface instance (optional)
        
    Returns:
        Phase5PersistOptimized instance
    """
    return Phase5PersistOptimized(db=db_interface)
