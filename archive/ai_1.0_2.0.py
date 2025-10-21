"""
VP Investments AI Integration
=============================

Complete AI integration for:
1. AI commentary, trends analysis, and explanations
2. AI strategy generation for Phase 2 enhancement
3. Trading strategy creation with risk management

Uses OpenAI for all AI-powered functionality.

This module consolidates all AI functionality previously spread across:
- ai_commentary.py (now deleted)
- ai_strategy_generator.py (now deleted)
"""

import os
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
from dataclasses import dataclass, asdict
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from openai import AsyncOpenAI
from ..storage.database import get_supabase_database
from ..utils.observability import emit_metric
from ..utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Public API
__all__ = [
    # Core AI integrator
    'AIIntegrator',
    'ai_analyzer',
    
    # Strategy generation (Phase 2)
    'AIStrategy',
    'AIStrategyGenerator',
    
    # Commentary generation
    'ComprehensiveCommentaryGenerator',
    'AICommentaryGenerator',  # Alias for compatibility
    'create_commentary_generator',
    
    # Utility functions
    'enhance_signals_with_ai_commentary',
]


@dataclass
class AIStrategy:
    """Container for AI-generated trading strategy"""
    signal_id: str
    ticker: str
    strategy_name: str
    strategy_type: str  # 'equity', 'options', 'combo'
    horizon: str  # 'intraday', 'short', 'medium', 'long'
    confidence_score: float
    risk_reward_ratio: float
    entry_type: str
    entry_conditions: str
    entry_sizing: float
    exit_conditions: str
    liquidity_score: float
    passes_guardrails: bool
    signal_provenance: Dict[str, Any]
    
    # Optional fields for options strategies
    option_strategy: Optional[str] = None
    strikes: Optional[List[float]] = None
    expiration_date: Optional[date] = None
    implied_volatility: Optional[float] = None
    max_loss: Optional[float] = None
    max_gain: Optional[float] = None


class AIStrategyGenerator:
    """Generates AI-powered trading strategies from enhanced signals"""
    
    def __init__(self, run_id: Optional[str] = None):
        self.db = None  # Will be initialized async
        self.run_id = run_id  # Store run_id for strategy tracking
        
        # Pipeline compatibility attributes
        self.ai_enabled = True
        self.top_signals_limit = 10  # Default to top 10 signals
        
        # Configure OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            self.ai_enabled = True
        else:
            logger.warning("OpenAI API key not found in environment variables")
            self.ai_enabled = False
            self.client = None
    
    # Helper methods for safe data access
    @staticmethod
    def _safe_float(value, default=0.0):
        """Safely convert value to float, handling None and invalid values"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_abs(value, default=0.0):
        """Safely get absolute value, handling None"""
        if value is None:
            return default
        try:
            return abs(float(value))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_get(d, key, default=0):
        """Safely get value from dict, handling None"""
        value = d.get(key, default)
        return value if value is not None else default
    
    async def _ensure_db_connection(self):
        """Ensure database connection is established"""
        if self.db is None:
            self.db = await get_supabase_database()
    
    async def generate_strategies_for_top_signals(self) -> Dict[str, List[AIStrategy]]:
        """Generate strategies for top signals (pipeline compatibility method)"""
        try:
            await self._ensure_db_connection()
            
            # Get top signals from database (Phase 7)
            result = self.db.supabase.table('signals').select('*').order('signal_score', desc=True).limit(self.top_signals_limit).execute()
            
            if not result.data:
                logger.warning("No signals found in database")
                return {}
            
            signals = result.data
            logger.info(f"Retrieved {len(signals)} top signals for AI strategy generation")
            
            # Generate strategies for all signals
            all_strategies = await self.generate_strategies_for_signals(signals)
            
            # Group strategies by ticker for pipeline compatibility
            strategies_by_ticker = {}
            for strategy in all_strategies:
                ticker = strategy.ticker
                if ticker not in strategies_by_ticker:
                    strategies_by_ticker[ticker] = []
                strategies_by_ticker[ticker].append(strategy)
            
            # Save strategies to database
            if all_strategies:
                await self.save_strategies_to_database(all_strategies)
            
            return strategies_by_ticker
            
        except Exception as e:
            logger.error(f"Error generating strategies for top signals: {e}")
            return {}
    
    async def generate_strategies_for_signals(self, signals: List[Dict[str, Any]], max_workers: int = 3) -> List[AIStrategy]:
        """Generate AI strategies for multiple signals concurrently"""
        try:
            if not self.ai_enabled:
                logger.warning("AI strategy generation disabled")
                return []
            
            logger.info(f"Starting AI strategy generation for {len(signals)} signals")
            emit_metric("ai_strategy_generation.start", len(signals))
            
            strategies = []
            
            # Process signals in batches with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks for each signal
                future_to_signal = {
                    executor.submit(self._generate_strategies_for_signal_sync, signal): signal
                    for signal in signals
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_signal):
                    signal = future_to_signal[future]
                    try:
                        signal_strategies = future.result()
                        strategies.extend(signal_strategies)
                        logger.info(f"Generated {len(signal_strategies)} strategies for {signal.get('ticker', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"Error generating strategies for signal {signal.get('id', 'Unknown')}: {e}")
            
            logger.info(f"Generated {len(strategies)} total AI strategies")
            emit_metric("ai_strategy_generation.complete", len(strategies))
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error in concurrent strategy generation: {e}")
            emit_metric("ai_strategy_generation.error", 1)
            return []
    
    def _generate_strategies_for_signal_sync(self, signal: Dict[str, Any]) -> List[AIStrategy]:
        """Synchronous wrapper for strategy generation (for ThreadPoolExecutor)"""
        return asyncio.run(self._generate_strategies_for_signal(signal))
    
    async def _generate_strategies_for_signal(self, signal: Dict[str, Any]) -> List[AIStrategy]:
        """Generate multiple strategies for a single signal"""
        try:
            ticker = signal.get('ticker', '')
            signal_id = signal.get('id', '')
            
            logger.debug(f"Generating AI strategies for {ticker}")
            
            # Analyze signal characteristics
            signal_analysis = self._analyze_signal_characteristics(signal)
            
            strategies = []
            
            # Generate equity strategy
            equity_strategy = await self._generate_equity_strategy(signal, signal_analysis)
            if equity_strategy:
                strategies.append(equity_strategy)
            
            # Generate options strategy if appropriate
            if self._should_generate_options_strategy(signal, signal_analysis):
                options_strategy = await self._generate_options_strategy(signal, signal_analysis)
                if options_strategy:
                    strategies.append(options_strategy)
            
            # Generate combo strategy for high-confidence signals
            if signal_analysis.get('confidence_level', 0) > 0.7:
                combo_strategy = await self._generate_combo_strategy(signal, signal_analysis)
                if combo_strategy:
                    strategies.append(combo_strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating strategies for signal {signal.get('id', 'Unknown')}: {e}")
            return []
    
    def _analyze_signal_characteristics(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze signal characteristics to inform strategy generation"""
        try:
            analysis = {}
            
            # Basic signal metrics (Phase 7)
            analysis['signal_score'] = signal.get('signal_score', 0)
            analysis['confidence_level'] = signal.get('signal_confidence', 0)
            analysis['volatility'] = signal.get('volatility', 0)
            analysis['liquidity_score'] = signal.get('liquidity_score', 0.5)
            
            # Market characteristics
            analysis['market_cap'] = signal.get('market_cap', 0)
            analysis['market_cap_category'] = signal.get('market_cap_category', 'Unknown')
            analysis['current_price'] = signal.get('current_price', 0)
            
            # Technical indicators
            analysis['rsi'] = signal.get('rsi', 50)
            analysis['momentum_30d'] = signal.get('momentum_30d_pct', 0)
            analysis['relative_strength'] = signal.get('relative_strength', 0)
            
            # Risk factors
            analysis['risk_score'] = signal.get('risk_score', 50)
            analysis['risk_level'] = signal.get('risk_level', 'Moderate')
            analysis['max_position_size'] = signal.get('max_position_size', 0.05)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing signal characteristics: {e}")
            return {}
    
    def _should_generate_options_strategy(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
        """Determine if options strategy should be generated"""
        try:
            # Check liquidity requirements
            liquidity_score = self._safe_float(analysis.get('liquidity_score'), 0)
            liquidity_ok = liquidity_score > 0.3
            
            # Check price range (avoid penny stocks)
            current_price = self._safe_float(analysis.get('current_price'), 0)
            price_ok = current_price > 5
            
            # Check market cap (prefer larger companies for options)
            market_cap = self._safe_float(analysis.get('market_cap'), 0)
            market_cap_ok = market_cap > 1e9  # > $1B
            
            return liquidity_ok and price_ok and market_cap_ok
            
        except Exception as e:
            logger.error(f"Error determining options strategy eligibility: {e}")
            return False
    
    async def _generate_equity_strategy(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[AIStrategy]:
        """Generate equity-focused trading strategy"""
        try:
            ticker = signal.get('ticker', '')
            signal_id = signal.get('id', '')
            
            # Determine strategy characteristics based on signal analysis
            horizon = self._determine_time_horizon(analysis)
            risk_reward = self._calculate_risk_reward_ratio(analysis)
            entry_sizing = min(self._safe_float(analysis.get('max_position_size'), 0.05), 0.10)  # Cap at 10%
            
            # Create strategy name
            strategy_name = f"{ticker} {self._get_strategy_descriptor(analysis)} Play"
            
            # Generate entry and exit conditions
            entry_conditions = self._generate_entry_conditions(signal, analysis)
            exit_conditions = self._generate_exit_conditions(analysis)
            
            # Create AI strategy object
            strategy = AIStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_type='equity',
                horizon=horizon,
                confidence_score=self._safe_float(analysis.get('confidence_level'), 0) * 100,
                risk_reward_ratio=risk_reward,
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=entry_sizing,
                exit_conditions=exit_conditions,
                liquidity_score=self._safe_float(analysis.get('liquidity_score'), 0.5),  # Already 0-1 scale
                passes_guardrails=self._passes_risk_guardrails(analysis),
                signal_provenance={
                    'signal_id': signal_id,
                    'generation_method': 'ai_strategy_generator',
                    'model': 'gpt-4o-mini',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating equity strategy: {e}")
            return None
    
    async def _generate_options_strategy(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[AIStrategy]:
        """Generate options-focused trading strategy"""
        try:
            ticker = signal.get('ticker', '')
            signal_id = signal.get('id', '')
            current_price = analysis.get('current_price', 0)
            
            if current_price == 0:
                return None
            
            # Determine options strategy type based on market conditions
            options_type = self._determine_options_strategy_type(analysis)
            
            # Generate strike prices and expiration
            strikes = self._generate_strike_prices(current_price, options_type)
            expiration = self._generate_expiration_date(analysis)
            
            # Create strategy name
            strategy_name = f"{ticker} {options_type.replace('_', ' ').title()}"
            
            # Generate entry conditions specific to options
            entry_conditions = self._generate_options_entry_conditions(signal, analysis, options_type)
            
            # Calculate risk metrics for options
            max_loss, max_gain = self._calculate_options_risk_reward(strikes, options_type)
            
            strategy = AIStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_type='options',
                horizon=self._determine_time_horizon(analysis),
                confidence_score=self._safe_float(analysis.get('confidence_level'), 0) * 100,
                risk_reward_ratio=max_gain / max_loss if max_loss > 0 else 3.0,
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=min(self._safe_float(analysis.get('max_position_size'), 0.05), 0.05),
                exit_conditions=self._generate_exit_conditions(analysis),
                liquidity_score=self._safe_float(analysis.get('liquidity_score'), 0.5),  # Already 0-1 scale
                passes_guardrails=self._passes_risk_guardrails(analysis),
                signal_provenance={
                    'signal_id': signal_id,
                    'generation_method': 'ai_strategy_generator',
                    'model': 'gpt-4o-mini',
                    'timestamp': datetime.now().isoformat()
                },
                option_strategy=options_type,
                strikes=strikes,
                expiration_date=expiration,
                max_loss=max_loss,
                max_gain=max_gain
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating options strategy: {e}")
            return None
    
    async def _generate_combo_strategy(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[AIStrategy]:
        """Generate combination equity + options strategy"""
        try:
            ticker = signal.get('ticker', '')
            signal_id = signal.get('id', '')
            
            # Combo strategies combine equity and options positions
            strategy_name = f"{ticker} Balanced Swing Play"
            
            # Generate combined entry conditions
            entry_conditions = {
                "equity_allocation": 0.6,  # 60% equity
                "options_allocation": 0.4,  # 40% options
                **self._generate_entry_conditions(signal, analysis)
            }
            
            strategy = AIStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_type='combo',
                horizon='medium',
                confidence_score=self._safe_float(analysis.get('confidence_level'), 0) * 100,
                risk_reward_ratio=2.5,  # Conservative for combo
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=min(self._safe_float(analysis.get('max_position_size'), 0.05), 0.05),
                exit_conditions=self._generate_exit_conditions(analysis),
                liquidity_score=self._safe_float(analysis.get('liquidity_score'), 0.5),  # Already 0-1 scale
                passes_guardrails=self._passes_risk_guardrails(analysis),
                signal_provenance={
                    'signal_id': signal_id,
                    'generation_method': 'ai_strategy_generator',
                    'model': 'gpt-4o-mini',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating combo strategy: {e}")
            return None
    
    # Helper methods for strategy generation
    def _determine_time_horizon(self, analysis: Dict[str, Any]) -> str:
        """Determine appropriate time horizon based on signal characteristics"""
        volatility = self._safe_float(analysis.get('volatility'), 0)
        momentum = self._safe_abs(analysis.get('momentum_30d'), 0)
        
        if volatility > 0.5 or momentum > 50:
            return 'short'
        elif volatility < 0.2 and momentum < 10:
            return 'long'
        else:
            return 'medium'
    
    def _calculate_risk_reward_ratio(self, analysis: Dict[str, Any]) -> float:
        """Calculate expected risk/reward ratio"""
        confidence = self._safe_float(analysis.get('confidence_level'), 0.5)
        risk_score_raw = self._safe_float(analysis.get('risk_score'), 50)
        risk_score = risk_score_raw / 100
        
        base_ratio = 2.0
        confidence_boost = confidence * 1.5
        risk_penalty = risk_score * 0.5
        
        return max(1.5, base_ratio + confidence_boost - risk_penalty)
    
    def _get_strategy_descriptor(self, analysis: Dict[str, Any]) -> str:
        """Get descriptive name for strategy based on characteristics"""
        momentum = self._safe_float(analysis.get('momentum_30d'), 0)
        rsi = self._safe_float(analysis.get('rsi'), 50)
        
        if momentum > 20:
            return "Momentum"
        elif momentum < -20:
            return "Contrarian"
        elif rsi > 70:
            return "Breakout"
        elif rsi < 30:
            return "Recovery"
        else:
            return "Long-Term Growth"
    
    def _generate_entry_conditions(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry conditions for the strategy (Phase 7)"""
        return {
            "entry_price": analysis.get('current_price', 0),
            "signal_strength": analysis.get('signal_score', 0),
            "momentum_score": analysis.get('momentum_30d', 0),
            "rsi_level": analysis.get('rsi', 50)
        }
    
    def _generate_exit_conditions(self, analysis: Dict[str, Any]) -> str:
        """Generate exit conditions string"""
        risk_level = analysis.get('risk_level', 'Moderate')
        
        if risk_level == 'High':
            return "Stop loss: -15%, Profit target: +25%, Time stop: 30 days"
        elif risk_level == 'Low':
            return "Stop loss: -8%, Profit target: +15%, Time stop: 60 days"
        else:
            return "Stop loss: -12%, Profit target: +20%, Time stop: 45 days"
    
    def _determine_options_strategy_type(self, analysis: Dict[str, Any]) -> str:
        """Determine the type of options strategy to use"""
        volatility = self._safe_float(analysis.get('volatility'), 0)
        momentum = self._safe_float(analysis.get('momentum_30d'), 0)
        
        if momentum > 15:
            return "long_call"
        elif momentum < -15:
            return "long_put"
        elif volatility > 0.4:
            return "short_call_spread"
        else:
            return "protective_put"
    
    def _generate_strike_prices(self, current_price: float, options_type: str) -> List[float]:
        """Generate appropriate strike prices for options strategy"""
        if not current_price or current_price <= 0:
            return []
        
        if options_type == "long_call":
            return [current_price * 1.05]
        elif options_type == "long_put":
            return [current_price * 0.95]
        elif options_type == "short_call_spread":
            return [current_price * 1.05, current_price * 1.15]
        else:
            return [current_price * 0.90]
    
    def _generate_expiration_date(self, analysis: Dict[str, Any]) -> date:
        """Generate appropriate options expiration date"""
        horizon = self._determine_time_horizon(analysis)
        
        if horizon == 'short':
            days_to_add = 14
        elif horizon == 'long':
            days_to_add = 90
        else:
            days_to_add = 45
        
        return (datetime.now() + timedelta(days=days_to_add)).date()
    
    def _generate_options_entry_conditions(self, signal: Dict[str, Any], analysis: Dict[str, Any], options_type: str) -> Dict[str, Any]:
        """Generate entry conditions specific to options"""
        current_price = analysis.get('current_price', 0)
        strikes = self._generate_strike_prices(current_price, options_type)
        
        conditions = {
            "underlying_price": current_price,
            "strategy_type": options_type,
            "implied_vol_threshold": 0.3,
        }
        
        if len(strikes) == 1:
            conditions["strike"] = strikes[0]
        elif len(strikes) == 2:
            conditions["buy_strike"] = strikes[0]
            conditions["sell_strike"] = strikes[1]
        
        return conditions
    
    def _calculate_options_risk_reward(self, strikes: List[float], options_type: str) -> Tuple[float, float]:
        """Calculate max loss and max gain for options strategy"""
        if not strikes:
            return 100.0, 300.0
        
        if options_type in ["long_call", "long_put"]:
            max_loss = 200.0
            max_gain = 600.0
        elif options_type == "short_call_spread":
            max_loss = 300.0
            max_gain = 150.0
        else:
            max_loss = 150.0
            max_gain = 450.0
        
        return max_loss, max_gain
    
    def _passes_risk_guardrails(self, analysis: Dict[str, Any]) -> bool:
        """Check if strategy passes basic risk guardrails"""
        try:
            position_size = self._safe_float(analysis.get('max_position_size'), 0.05)
            if position_size > 0.15:
                return False
            
            liquidity = self._safe_float(analysis.get('liquidity_score'), 0.5)
            if liquidity < 0.2:
                return False
            
            risk_score = self._safe_float(analysis.get('risk_score'), 50)
            if risk_score > 85:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk guardrails: {e}")
            return False
    
    async def save_strategies_to_database(self, strategies: List[AIStrategy]) -> bool:
        """Save generated strategies to the ai_strategies table"""
        try:
            if not strategies:
                logger.warning("No strategies to save")
                return True
            
            logger.info(f"Saving {len(strategies)} AI strategies to database")
            
            # Ensure database connection
            await self._ensure_db_connection()
            
            # Convert strategies to database format
            strategy_records = []
            for strategy in strategies:
                record = {
                    'signal_id': strategy.signal_id,
                    # Note: run_id removed - ai_strategies.run_id is UUID type, 
                    # but self.run_id is string format. Use signal_id for tracking instead.
                    'ticker': strategy.ticker,
                    'strategy_name': strategy.strategy_name,
                    'strategy_type': strategy.strategy_type,
                    'horizon': strategy.horizon,
                    'confidence_score': strategy.confidence_score,
                    'risk_reward_ratio': strategy.risk_reward_ratio,
                    'entry_type': strategy.entry_type,
                    'entry_conditions': strategy.entry_conditions,
                    'entry_sizing': strategy.entry_sizing,
                    'exit_conditions': strategy.exit_conditions,
                    'liquidity_score': strategy.liquidity_score,
                    'passes_guardrails': strategy.passes_guardrails,
                    'signal_provenance': strategy.signal_provenance,
                    'generation_timestamp': datetime.now().isoformat(),
                    'ai_model_version': 'gpt-4o-mini',
                    'generation_confidence': strategy.confidence_score,
                    'status': 'generated',
                    'monitoring_alerts': {'catalysts': [], 'data_signals': []},
                    'rebalancing_triggers': {},
                    'guardrail_checks': {},
                    'compliance_flags': {},
                    'blackout_periods': {},
                    'breakeven_points': []
                }
                
                # Add options-specific fields if present
                if strategy.option_strategy:
                    record['option_strategy'] = strategy.option_strategy
                if strategy.strikes:
                    record['strikes'] = strategy.strikes
                if strategy.expiration_date:
                    record['expiration_date'] = strategy.expiration_date.isoformat()
                if strategy.max_loss:
                    record['max_loss'] = strategy.max_loss
                if strategy.max_gain:
                    record['max_gain'] = strategy.max_gain
                
                strategy_records.append(record)
            
            # Insert into database
            result = self.db.supabase.table('ai_strategies').insert(strategy_records).execute()
            
            if result.data:
                logger.info(f"Successfully saved {len(result.data)} AI strategies")
                emit_metric("ai_strategies.saved", len(result.data))
                return True
            else:
                logger.error("Failed to save AI strategies to database")
                return False
                
        except Exception as e:
            logger.error(f"Error saving AI strategies to database: {e}")
            emit_metric("ai_strategies.save_error", 1)
            return False


class AIIntegrator:
    """
    AI integration for generating commentary and analysis using OpenAI.
    """
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("AI integration disabled")
            self.client = None
            return
        
        try:
            # Initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found, disabling AI integration")
                self.enabled = False
                self.client = None
                return
            
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            logger.info(f"AI integration initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI integration: {e}")
            self.enabled = False
            self.client = None
    
    async def generate_signal_commentary(self, 
                                       ticker: str, 
                                       signal_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Generate AI commentary for a trading signal.
        
        Returns:
            Dict with ai_commentary, ai_trends_commentary, score_explanation
        """
        if not self.enabled or not self.client:
            return {
                'ai_commentary': None,
                'ai_trends_commentary': None,
                'score_explanation': None
            }
        
        try:
            # Prepare context for AI analysis
            context = self._prepare_signal_context(ticker, signal_data)
            
            # Generate commentary
            commentary_tasks = [
                self._generate_commentary(context),
                self._generate_trends_commentary(context),
                self._generate_score_explanation(context, signal_data.get('signal_score', 0))
            ]
            
            results = await asyncio.gather(*commentary_tasks, return_exceptions=True)
            
            return {
                'ai_commentary': results[0] if not isinstance(results[0], Exception) else None,
                'ai_trends_commentary': results[1] if not isinstance(results[1], Exception) else None,
                'score_explanation': results[2] if not isinstance(results[2], Exception) else None
            }
            
        except Exception as e:
            logger.warning(f"Error generating AI commentary for {ticker}: {e}")
            return {
                'ai_commentary': None,
                'ai_trends_commentary': None, 
                'score_explanation': None
            }
    
    def _prepare_signal_context(self, ticker: str, signal_data: Dict[str, Any]) -> str:
        """Prepare context string for AI analysis"""
        context_parts = [f"Stock: {ticker}"]
        
        # Add available data points (Phase 7)
        if 'signal_score' in signal_data:
            context_parts.append(f"Signal Score: {signal_data['signal_score']:.3f}")
        
        if 'reddit_score' in signal_data:
            context_parts.append(f"Reddit Score: {signal_data['reddit_score']:.3f}")
        
        if 'financial_score' in signal_data:
            context_parts.append(f"Financial Score: {signal_data['financial_score']:.3f}")
        
        if 'mention_count' in signal_data:
            context_parts.append(f"Reddit Mentions: {signal_data['mention_count']}")
        
        if 'avg_sentiment' in signal_data:
            context_parts.append(f"Avg Sentiment: {signal_data['avg_sentiment']:.3f}")
        
        # Add financial metrics if available
        financial_data = signal_data.get('financial_data', {})
        if financial_data:
            if 'current_price' in financial_data:
                context_parts.append(f"Price: ${financial_data['current_price']:.2f}")
            if 'market_cap' in financial_data:
                context_parts.append(f"Market Cap: {financial_data['market_cap']}")
            if 'sector' in financial_data:
                context_parts.append(f"Sector: {financial_data['sector']}")
        
        return " | ".join(context_parts)
    
    async def _generate_commentary(self, context: str) -> Optional[str]:
        """Generate general AI commentary"""
        prompt = f"""
        Analyze this stock signal and provide a brief investment commentary (max 150 words):
        
        {context}
        
        Focus on key factors driving the signal and potential risks/opportunities.
        Be concise and actionable.
        """
        
        return await self._call_openai(prompt)
    
    async def _generate_trends_commentary(self, context: str) -> Optional[str]:
        """Generate trends-focused commentary"""
        prompt = f"""
        Analyze market trends and momentum for this stock signal (max 100 words):
        
        {context}
        
        Focus on trending factors, momentum indicators, and market context.
        Be specific about trend direction and strength.
        """
        
        return await self._call_openai(prompt)
    
    async def _generate_score_explanation(self, context: str, score: float) -> Optional[str]:
        """Generate explanation for the signal score"""
        prompt = f"""
        Explain why this stock received a signal score of {score:.3f} (max 80 words):
        
        {context}
        
        Break down the key factors contributing to this score in simple terms.
        """
        
        return await self._call_openai(prompt)
    
    async def _call_openai(self, prompt: str) -> Optional[str]:
        """Make OpenAI API call with error handling"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a financial analyst providing brief, actionable stock market insights."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return None


# ------------------------
# Unified commentary class (compatibility with former ai_commentary.py)
# ------------------------
class ComprehensiveCommentaryGenerator:
    """Generate unified AI commentary for trading signals.

    This class consolidates reddit/news/trends, technical/financial metrics,
    and score rationale into a single ai_commentary field, using OpenAI when
    available and a rule-based fallback otherwise.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.client = None
        self.ai_enabled = False

        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = AsyncOpenAI(api_key=api_key)
                self.ai_enabled = True
                self.logger.info("AI commentary: OpenAI client initialized")
            else:
                self.logger.warning("OPENAI_API_KEY missing; using fallback commentary")
        except Exception as e:
            self.logger.warning(f"OpenAI init failed: {e}")

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = 600
        self.temperature = 0.2

    async def generate_comprehensive_commentary(self, signal: Dict[str, Any]) -> str:
        try:
            if not self.ai_enabled or not self.client:
                return self._fallback_commentary(signal)

            data = self._extract(signal)
            prompt = self._build_prompt(data)

            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional financial analyst. Be concise, "
                            "objective, and specific. Provide risk and opportunity."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            self.logger.warning(
                f"AI commentary generation failed for {signal.get('ticker')}: {e}"
            )
            return self._fallback_commentary(signal)

    async def enhance_signals_batch(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not signals:
            return signals

        enhanced: List[Dict[str, Any]] = []
        for i, s in enumerate(signals):
            try:
                cmt = await self.generate_comprehensive_commentary(s)
            except Exception:
                cmt = self._fallback_commentary(s)

            enriched = {**s, "ai_commentary": cmt, "ai_commentary_version": "1.0"}
            enhanced.append(enriched)

            if self.ai_enabled and i < len(signals) - 1:
                await asyncio.sleep(0.3)

        self.logger.info(f"AI commentary generated for {len(enhanced)} signals")
        return enhanced

    # ---------- helpers ----------
    def _extract(self, s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ticker": s.get("ticker", "N/A"),
            "company": s.get("company", s.get("ticker", "N/A")),
            "sector": s.get("sector", "N/A"),
            "signal_score": float(s.get("signal_score", 0) or 0),  # Phase 7
            "trade_type": s.get("trade_type", "Signal"),
            "risk_level": s.get("risk_level", "Unknown"),
            # consolidated commentary
            "reddit_summary": s.get("reddit_summary", ""),
            "ai_news_summary": s.get("ai_news_summary", ""),
            "ai_trends_commentary": s.get("ai_trends_commentary", ""),
            "score_explanation": s.get("score_explanation", ""),
            # context
            "current_price": s.get("current_price"),
            "market_cap": s.get("market_cap"),
            "volume_spike_ratio": s.get("volume_spike_ratio"),
            "reddit_score": float(s.get("reddit_score", 0) or 0),
            "news_score": float(s.get("news_score", 0) or 0),
            "financial_score": float(s.get("financial_score", 0) or 0),
            "rsi": s.get("rsi"),
            "pe_ratio": s.get("pe_ratio"),
            "eps_growth": s.get("eps_growth"),
            # social
            "reddit_sentiment": s.get("reddit_sentiment"),
            "news_sentiment": s.get("news_sentiment"),
            "mentions": int(s.get("mentions", s.get("mention_count", 0)) or 0),
            "upvotes": int(s.get("upvotes", 0) or 0),
            # technical
            "price_1d_pct": s.get("price_1d_pct"),
            "price_7d_pct": s.get("price_7d_pct"),
            "momentum_30d_pct": s.get("momentum_30d_pct"),
            "relative_strength": s.get("relative_strength"),
        }

    def _build_prompt(self, d: Dict[str, Any]) -> str:
        sections: List[str] = []

        # Reddit
        if d.get("reddit_summary") or d.get("reddit_score", 0) > 0:
            seg = [f"Reddit (score {d['reddit_score']:.2f})"]
            if d.get("reddit_summary"):
                seg.append(f"summary: {d['reddit_summary']}")
            extra: List[str] = []
            if d.get("mentions"):
                extra.append(f"mentions {d['mentions']}")
            if d.get("upvotes"):
                extra.append(f"upvotes {d['upvotes']}")
            if d.get("reddit_sentiment") is not None:
                extra.append(f"sentiment {float(d['reddit_sentiment']):.2f}")
            if extra:
                seg.append("(" + ", ".join(extra) + ")")
            sections.append("; ".join(seg))

        # News
        if d.get("ai_news_summary") or d.get("news_score", 0) > 0:
            seg = [f"News (score {d['news_score']:.2f})"]
            if d.get("ai_news_summary"):
                seg.append(f"summary: {d['ai_news_summary']}")
            if d.get("news_sentiment") is not None:
                seg.append(f"(sentiment {float(d['news_sentiment']):.2f})")
            sections.append("; ".join(seg))

        # Trends
        if d.get("ai_trends_commentary"):
            sections.append(f"AI Trends: {d['ai_trends_commentary']}")

        # Financial
        fin = [f"Financial (score {d['financial_score']:.2f})"]
        if d.get("current_price") is not None:
            fin.append(f"price ${float(d['current_price']):.2f}")
        if d.get("pe_ratio") is not None:
            fin.append(f"P/E {float(d['pe_ratio']):.2f}")
        if d.get("eps_growth") is not None:
            fin.append(f"EPS growth {float(d['eps_growth']):.2f}%")
        if len(fin) > 1:
            sections.append("; ".join(fin))

        # Technical
        tech = ["Technical"]
        if d.get("rsi") is not None:
            tech.append(f"RSI {float(d['rsi']):.2f}")
        if d.get("price_1d_pct") is not None:
            tech.append(f"1D {float(d['price_1d_pct']):.2f}%")
        if d.get("price_7d_pct") is not None:
            tech.append(f"7D {float(d['price_7d_pct']):.2f}%")
        if d.get("momentum_30d_pct") is not None:
            tech.append(f"30D {float(d['momentum_30d_pct']):.2f}%")
        if d.get("relative_strength") is not None:
            tech.append(f"rel strength {float(d['relative_strength']):.2f}")
        if d.get("volume_spike_ratio") is not None:
            tech.append(f"volume {float(d['volume_spike_ratio']):.2f}x")
        if len(tech) > 1:
            sections.append("; ".join(tech))

        if d.get("score_explanation"):
            sections.append(f"Score rationale: {d['score_explanation']}")

        header = (
            f"Analyze {d['ticker']} ({d['company']}) in {d['sector']} sector.\n"
            f"Signal {d['signal_score']:.3f} - {d['trade_type']} ({d['risk_level']} risk)."
        )

        return (
            f"{header}\n\n"
            f"Data: {' | '.join(sections)}\n\n"
            "Provide a unified, professional commentary that synthesizes key themes, "
            "explains drivers, highlights risks/opportunities, and offers actionable insight. "
            "Keep it 4-7 sentences."
        ).strip()

    def _fallback_commentary(self, s: Dict[str, Any]) -> str:
        ticker = s.get("ticker", "UNKNOWN")
        company = s.get("company", ticker)
        score = float(s.get("signal_score", 0) or 0)  # Phase 7
        trade = s.get("trade_type", "Signal").lower()
        risk = str(s.get("risk_level", "Unknown")).lower()

        parts: List[str] = [
            f"{company} ({ticker}) shows a {trade} setup with a signal score of {score:.3f} and {risk} risk.",
        ]

        bullets: List[str] = []
        if s.get("reddit_summary"):
            bullets.append(f"Reddit: {s['reddit_summary']}")
        if s.get("ai_news_summary"):
            bullets.append(f"News: {s['ai_news_summary']}")
        if s.get("ai_trends_commentary"):
            bullets.append(f"Trends: {s['ai_trends_commentary']}")
        if s.get("score_explanation"):
            bullets.append(f"Score: {s['score_explanation']}")

        if bullets:
            parts.append("Key insights: " + "; ".join(bullets) + ".")
        else:
            drivers: List[str] = []
            rs = float(s.get("reddit_score", 0) or 0)
            ns = float(s.get("news_score", 0) or 0)
            fs = float(s.get("financial_score", 0) or 0)
            if rs > 0:
                drivers.append(f"social momentum {rs:.2f}")
            if ns > 0:
                drivers.append(f"news sentiment {ns:.2f}")
            if fs > 0:
                drivers.append(f"financial strength {fs:.2f}")
            if drivers:
                parts.append("Signal drivers include " + ", ".join(drivers) + ".")

        parts.append("Use risk management and await confirmation as needed.")
        return " ".join(parts)


# Backward-compatible alias and factory
AICommentaryGenerator = ComprehensiveCommentaryGenerator

def create_commentary_generator() -> ComprehensiveCommentaryGenerator:
    return ComprehensiveCommentaryGenerator()

# Singleton instance for reuse
_ai_integrator = None

def get_ai_integrator() -> AIIntegrator:
    """Get singleton AI integrator instance"""
    global _ai_integrator
    if _ai_integrator is None:
        # Read AI toggle from environment variable
        ai_enabled = os.getenv('DATA_SOURCES_AI_ENABLED', 'false').lower() == 'true'
        _ai_integrator = AIIntegrator(enabled=ai_enabled)
    return _ai_integrator


async def test_ai_integration():
    """Test AI integration functionality"""
    print("🤖 Testing Complete AI Integration...")
    
    # Test AI Commentary
    ai = get_ai_integrator()
    
    if ai.enabled:
        print("\n📝 Testing AI Commentary...")
        
        test_signal = {
            'ticker': 'AAPL',
            'signal_score': 0.85,  # Phase 7
            'reddit_score': 0.7,
            'financial_score': 0.9,
            'mention_count': 15,
            'avg_sentiment': 0.6,
            'financial_data': {
                'current_price': 175.50,
                'market_cap': '2.8T',
                'sector': 'Technology'
            }
        }
        
        result = await ai.generate_signal_commentary('AAPL', test_signal)
        
        print(f"  Commentary: {result['ai_commentary'][:100] if result['ai_commentary'] else 'None'}...")
        print(f"  Trends: {result['ai_trends_commentary'][:100] if result['ai_trends_commentary'] else 'None'}...")
        print(f"  Score Explanation: {result['score_explanation'][:100] if result['score_explanation'] else 'None'}...")
    else:
        print("❌ AI Commentary disabled")
    
    # Test AI Strategy Generation
    print("\n🎯 Testing AI Strategy Generation...")
    
    strategy_gen = AIStrategyGenerator()
    
    if strategy_gen.ai_enabled:
        # Create test signal for strategy generation
        test_signals = [{
            'id': str(uuid.uuid4()),
            'ticker': 'AAPL',
            'signal_score': 0.85,  # Phase 7
            'signal_confidence': 0.8,
            'current_price': 175.50,
            'market_cap': 2800000000000,
            'market_cap_category': 'Mega',
            'volatility': 0.25,
            'rsi': 65,
            'momentum_30d_pct': 15.5,
            'risk_score': 35,
            'risk_level': 'Moderate',
            'max_position_size': 0.05,
            'liquidity_score': 0.9
        }]
        
        strategies = await strategy_gen.generate_strategies_for_signals(test_signals)
        
        if strategies:
            print(f"  ✅ Generated {len(strategies)} strategies:")
            for strategy in strategies:
                print(f"    - {strategy.strategy_name} ({strategy.strategy_type})")
        else:
            print("  ❌ No strategies generated")
    else:
        print("  ❌ AI Strategy Generation disabled")


async def enhance_signals_with_ai_commentary(signals: List[Dict[str, Any]], 
                                           ai_integrator: Optional[AIIntegrator] = None,
                                           max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Enhance signals with AI-generated commentary and explanations.
    
    Args:
        signals: List of signal dictionaries to enhance
        ai_integrator: AIIntegrator instance (creates new if None)
        max_workers: Maximum concurrent AI requests
        
    Returns:
        Enhanced signals with AI commentary fields
    """
    if not signals:
        logger.info("No signals provided for AI commentary enhancement")
        return signals
    
    # Initialize AI integrator if not provided
    if ai_integrator is None:
        ai_integrator = AIIntegrator(enabled=True)
    
    if not ai_integrator.enabled:
        logger.info("AI integration disabled, skipping commentary enhancement")
        # Add empty AI fields to maintain schema consistency
        for signal in signals:
            signal.update({
                'ai_commentary': None,
                'ai_trends_commentary': None,
                'score_explanation': None
            })
        return signals
    
    logger.info(f"Enhancing {len(signals)} signals with AI commentary...")
    
    async def process_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual signal for AI commentary"""
        try:
            ticker = signal.get('ticker', 'Unknown')
            
            # Generate AI commentary
            commentary_data = await ai_integrator.generate_signal_commentary(ticker, signal)
            
            # Add AI commentary to signal
            signal.update(commentary_data)
            
            return signal
            
        except Exception as e:
            logger.warning(f"Failed to generate AI commentary for {signal.get('ticker', 'Unknown')}: {e}")
            # Add empty AI fields on error
            signal.update({
                'ai_commentary': None,
                'ai_trends_commentary': None,
                'score_explanation': None
            })
            return signal
    
    # Process signals concurrently with rate limiting
    try:
        semaphore = asyncio.Semaphore(max_workers)
        
        async def limited_process(signal):
            async with semaphore:
                return await process_signal(signal)
        
        # Process all signals concurrently with semaphore limiting
        enhanced_signals = await asyncio.gather(
            *[limited_process(signal) for signal in signals],
            return_exceptions=True
        )
        
        # Handle any exceptions and ensure we return valid signals
        valid_signals = []
        for i, result in enumerate(enhanced_signals):
            if isinstance(result, Exception):
                logger.warning(f"Exception processing signal {i}: {result}")
                # Add original signal with empty AI fields
                signal = signals[i].copy()
                signal.update({
                    'ai_commentary': None,
                    'ai_trends_commentary': None,
                    'score_explanation': None
                })
                valid_signals.append(signal)
            else:
                valid_signals.append(result)
        
        # Log summary
        commentary_count = sum(1 for s in valid_signals if s.get('ai_commentary'))
        trends_count = sum(1 for s in valid_signals if s.get('ai_trends_commentary'))
        explanation_count = sum(1 for s in valid_signals if s.get('score_explanation'))
        
        logger.info(f"AI commentary enhancement completed:")
        logger.info(f"  - {commentary_count}/{len(signals)} signals got general commentary")
        logger.info(f"  - {trends_count}/{len(signals)} signals got trends commentary")
        logger.info(f"  - {explanation_count}/{len(signals)} signals got score explanations")
        
        return valid_signals
        
    except Exception as e:
        logger.error(f"Critical error during AI commentary enhancement: {e}")
        # Return original signals with empty AI fields
        for signal in signals:
            signal.update({
                'ai_commentary': None,
                'ai_trends_commentary': None,
                'score_explanation': None
            })
        return signals


# Create AI analyzer instance for import
try:
    ai_analyzer = AIIntegrator()
except Exception as e:
    logger.warning(f"Failed to initialize ai_analyzer: {e}")
    ai_analyzer = None


if __name__ == "__main__":
    asyncio.run(test_ai_integration())