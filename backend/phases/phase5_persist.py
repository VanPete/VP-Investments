"""
Phase 5: Persist
================

Save signals to database - PURE PERSISTENCE ONLY!

This module is responsible for:
- Creating run records
- Saving signals to database
- Error handling and retry logic
- NO scoring
- NO calculations
- NO enhancements
- JUST database persistence

All logic (risk, commentary, classification) should happen in Phase 3.
Phase 5 just saves the data.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase5Persister:
    """
    Phase 5: Persist
    
    Saves signals to database. This is PURE persistence - no logic!
    """
    
    def __init__(self):
        """Initialize Phase 5 persister."""
        self.logger = logger
        self._init_database()
    
    def _init_database(self):
        """Initialize Supabase database connection."""
        try:
            import os
            from supabase import create_client, Client
            
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase credentials not found in environment")
            
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.logger.info("Supabase connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase: {e}")
            self.supabase = None
            raise
    
    async def save_signals(self, 
                          signals: List[Dict[str, Any]],
                          run_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for Phase 5.
        
        Saves signals to database with run record.
        
        Args:
            signals: List of signals from Phase 4 (with final signal_score)
            run_metadata: Optional metadata about the pipeline run
        
        Returns:
            Dict with success status and run_id
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5: PERSIST TO DATABASE")
        self.logger.info("=" * 60)
        
        phase5_start = datetime.now()
        
        if not signals:
            self.logger.warning("No signals to save")
            return {'success': False, 'error': 'No signals provided'}
        
        try:
            # Step 1: Create run record
            self.logger.info("Step 5.1: Creating run record...")
            db_id, run_id_string = await self._create_run_record(signals, run_metadata)
            
            # Step 2: Save signals to database
            self.logger.info(f"Step 5.2: Saving {len(signals)} signals to database...")
            saved_count = await self._save_signals_batch(signals, db_id)
            
            phase5_end = datetime.now()
            execution_time = (phase5_end - phase5_start).total_seconds()
            
            self.logger.info("=" * 60)
            self.logger.info(f"PHASE 5 COMPLETE - {execution_time:.2f}s")
            self.logger.info(f"  Run ID: {run_id_string}")
            self.logger.info(f"  Signals saved: {saved_count}/{len(signals)}")
            self.logger.info("=" * 60)
            
            return {
                'success': True,
                'run_id': run_id_string,
                'signals_saved': saved_count,
                'execution_time': execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Phase 5 persistence failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_run_record(self, 
                                 signals: List[Dict[str, Any]],
                                 run_metadata: Dict[str, Any] = None) -> tuple[int, str]:
        """
        Create a run record in the database.
        
        Args:
            signals: List of signals to be saved
            run_metadata: Optional metadata about the run
        
        Returns:
            tuple: (db_id: int, run_id: str) - db_id for FK reference, run_id for display
        """
        try:
            # Generate unique run_id
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_id = f"run_{timestamp}"
            
            # Prepare run record
            # Valid run_type values: 'discovery', 'targeted', 'backtest', 'scheduled'
            run_record = {
                'run_id': run_id,
                'run_type': 'discovery',  # Using 'discovery' for general pipeline runs
                'started_at': datetime.now().isoformat(),
                'completed_at': datetime.now().isoformat(),
                'total_signals': len(signals),
                'status': 'completed',
                'metadata': run_metadata or {
                    'signals_count': len(signals),
                    'pipeline_version': '3.0',
                    'architecture': '6-phase'
                }
            }
            
            # Insert run record
            result = self.supabase.table('runs').insert(run_record).execute()
            
            if result.data:
                db_id = result.data[0]['id']
                self.logger.info(f"✅ Run record created: {run_id} (DB ID: {db_id})")
                return db_id, run_id  # Return both: db_id for FK, run_id string for display
            else:
                raise ValueError("No run data returned from database")
                
        except Exception as e:
            self.logger.error(f"Failed to create run record: {e}")
            raise
    
    async def _save_signals_batch(self, 
                                  signals: List[Dict[str, Any]],
                                  run_id: int) -> int:
        """
        Save signals to database in batch.
        
        Args:
            signals: List of signals to save
            run_id: Run database ID (integer) for FK reference
        
        Returns:
            Number of signals successfully saved
        """
        saved_count = 0
        
        # Prepare signal records for database
        signal_records = []
        
        for rank, signal in enumerate(signals, 1):
            try:
                # Extract signal data (all should come from Phase 3/4)
                record = self._prepare_signal_record(signal, run_id, rank)
                signal_records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Failed to prepare signal record for {signal.get('ticker')}: {e}")
                continue
        
        # Batch insert signals
        try:
            if signal_records:
                result = self.supabase.table('signals').insert(signal_records).execute()
                
                if result.data:
                    saved_count = len(result.data)
                    self.logger.info(f"✅ Saved {saved_count} signals to database")
                else:
                    self.logger.warning("No data returned from signal insert")
                    
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            # Try individual inserts as fallback
            saved_count = await self._save_signals_individually(signal_records, run_id)
        
        return saved_count
    
    async def _save_signals_individually(self, 
                                        signal_records: List[Dict],
                                        run_id: int) -> int:
        """
        Fallback: Save signals one by one if batch insert fails.
        
        Args:
            signal_records: List of prepared signal records
            run_id: Run database ID (integer) - not used, already in records
        
        Returns:
            Number of signals successfully saved
        """
        saved_count = 0
        
        for record in signal_records:
            try:
                result = self.supabase.table('signals').insert(record).execute()
                if result.data:
                    saved_count += 1
            except Exception as e:
                ticker = record.get('ticker', 'UNKNOWN')
                self.logger.warning(f"Failed to save signal for {ticker}: {e}")
                continue
        
        self.logger.info(f"✅ Individually saved {saved_count}/{len(signal_records)} signals")
        return saved_count
    
    def _categorize_market_cap(self, market_cap: int) -> str:
        """
        Categorize market cap for database enum.
        
        Args:
            market_cap: Market capitalization in dollars
            
        Returns:
            Category: 'mega', 'large', 'mid', 'small', 'micro', 'nano'
        """
        if not market_cap or market_cap <= 0:
            return 'micro'
        
        if market_cap >= 200_000_000_000:  # $200B+
            return 'mega'
        elif market_cap >= 10_000_000_000:  # $10B+
            return 'large'
        elif market_cap >= 2_000_000_000:   # $2B+
            return 'mid'
        elif market_cap >= 300_000_000:     # $300M+
            return 'small'
        elif market_cap >= 50_000_000:      # $50M+
            return 'micro'
        else:
            return 'nano'
    
    def _prepare_signal_record(self, 
                               signal: Dict[str, Any],
                               run_id: int,
                               rank: int) -> Dict[str, Any]:
        """
        Prepare a signal record for database insertion.
        
        All data should already be calculated in Phase 3/4.
        This method just maps fields to database schema.
        
        Args:
            signal: Signal data from Phase 4
            run_id: Run database ID (integer) for FK reference to runs.id
            rank: Signal ranking
        
        Returns:
            Dict ready for database insertion
        """
        # Extract data (should all exist from Phase 3/4)
        ticker = signal.get('ticker', '').upper()
        signal_score = signal.get('signal_score', 0.0)
        
        # Extract group scores (3.0 signal groups)
        technical_score = signal.get('technical_score', 0.0)
        fundamental_score = signal.get('fundamental_score', 0.0)
        news_macro_score = signal.get('news_macro_score', 0.0)
        social_alternative_score = signal.get('social_alternative_score', 0.0)
        risk_stability_score = signal.get('risk_stability_score', 0.0)
        institutional_smart_money_score = signal.get('institutional_smart_money_score', 0.0)
        
        # Extract group data (3.0 signal groups)
        technical_data = signal.get('technical_data', {})
        fundamental_data = signal.get('fundamental_data', {})
        news_macro_data = signal.get('news_macro_data', {})
        social_alternative_data = signal.get('social_alternative_data', {})
        risk_stability_data = signal.get('risk_stability_data', {})
        institutional_smart_money_data = signal.get('institutional_smart_money_data', {})
        
        # Prepare database record - ONLY fields that exist in schema
        # Schema v3.0 has separate tables for detailed metrics
        record = {
            # Core identification
            'run_id': run_id,
            'ticker': ticker,
            'signal_rank': rank,
            
            # Scores (from Phase 4) - 3.0 signal groups
            'signal_score': signal_score,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'news_macro_score': news_macro_score,
            'social_alternative_score': social_alternative_score,
            'risk_stability_score': risk_stability_score,
            'institutional_smart_money_score': institutional_smart_money_score,
            'signal_confidence': signal.get('confidence', 0.0),
            
            # Company info (from Phase 3 fundamental data)
            'company': fundamental_data.get('company_name', ticker),
            'sector': fundamental_data.get('sector'),
            'industry': fundamental_data.get('industry'),
            'market_cap': fundamental_data.get('market_cap'),
            
            # Price data (from Phase 3 technical data) - only current_price & volume in main table
            'current_price': technical_data.get('current_price'),
            'volume': technical_data.get('volume'),
            
            # Risk & classification (from Phase 3)
            'risk_level': signal.get('risk_category', 'moderate').lower(),  # Schema uses lowercase
            'signal_type': signal.get('signal_type', 'Multi-Factor'),
            'trade_type': signal.get('trade_type', 'Signal'),
            'trade_type_confidence': signal.get('trade_type_confidence', 0.5),
            'market_cap_category': self._categorize_market_cap(fundamental_data.get('market_cap')),
            
            # AI-generated content (from Phase 6)
            'risk_narrative': signal.get('risk_description'),
            'trade_strategy': signal.get('ai_commentary'),
            'ai_confidence': signal.get('ai_confidence', 0.0),
            'ai_model_version': signal.get('ai_model_version', 'gpt-4'),
            
            # Metadata
            'scoring_version': '3.0',
            'data_sources': ['reddit', 'yfinance'],  # List of data sources used
            'processing_metadata': {
                'scoring_weights': signal.get('scoring_weights', {}),
                'phase': 'Phase 5: Persist',
                'created_by': 'unified_pipeline_v3'
            }
        }
        
        return record
