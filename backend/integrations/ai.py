"""
AI Integration (3.0 Architecture - Phase 6)
============================================
Generate AI commentary for top 10 signals only

Responsibilities:
- Risk narratives (market risks, opportunities, context)
- Trade strategies (equity, options, combo)
- OpenAI GPT-4o-mini integration

Pipeline calls this for Phase 6 only.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from openai import AsyncOpenAI

from backend.utils.logger import get_logger
from backend.utils.metrics import emit_metric

logger = get_logger(__name__)


@dataclass
class AIRiskNarrative:
    """Risk narrative for top signal"""
    signal_id: str
    ticker: str
    risk_commentary: str
    opportunity_commentary: str
    market_context: str
    confidence_level: float
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "gpt-4o-mini"


@dataclass
class AITradeStrategy:
    """Trade strategy for top signal"""
    signal_id: str
    ticker: str
    strategy_name: str
    strategy_type: str
    horizon: str
    entry_type: str
    entry_conditions: str
    entry_sizing: float
    exit_conditions: str
    confidence_score: float
    risk_reward_ratio: float
    max_loss: Optional[float] = None
    max_gain: Optional[float] = None
    option_strategy: Optional[str] = None
    strikes: Optional[List[float]] = None
    expiration_date: Optional[str] = None
    liquidity_score: float = 0.5
    passes_guardrails: bool = True
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "gpt-4o-mini"


class AICommentaryGenerator:
    """Phase 6: AI commentary for top 10 signals"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.client = None
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        if not self.enabled:
            logger.info("AI commentary disabled")
            return
        
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found - AI disabled")
                self.enabled = False
                return
            
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(f"AI commentary initialized: {self.model}")
            
        except Exception as e:
            logger.error(f"AI init failed: {e}")
            self.enabled = False
            self.client = None
    
    async def generate_commentary_for_top_signals(self, signals: List[Dict[str, Any]], 
                                                  limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Main entry point: Generate AI commentary for top N signals with rate limiting
        Processes 3 signals at a time with 1s delay between batches
        """
        if not self.enabled or not self.client:
            return {}
        
        try:
            # Sort by signal_score descending and take top N
            sorted_signals = sorted(
                signals, 
                key=lambda x: x.get('signal_score', 0), 
                reverse=True
            )[:limit]
            
            logger.info(f"Generating AI commentary for top {len(sorted_signals)} signals")
            emit_metric("ai.commentary.batch_start", 1, tags={'count': len(sorted_signals)})
            
            results = {}
            
            # Process in batches of 3 to avoid rate limits
            batch_size = 3
            for i in range(0, len(sorted_signals), batch_size):
                batch = sorted_signals[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num} ({len(batch)} signals)")
                
                # Process batch concurrently
                tasks = []
                for signal in batch:
                    tasks.append(self._process_single_signal(signal))
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for signal, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch task failed for {signal.get('ticker', 'UNKNOWN')}: {result}")
                        continue
                    
                    if result and isinstance(result, tuple) and len(result) == 2:
                        narrative, strategy = result
                        signal_id = signal.get('id', '')
                        if narrative or strategy:
                            results[signal_id] = {
                                'narrative': narrative,
                                'strategy': strategy
                            }
                
                # Rate limiting: sleep between batches
                if i + batch_size < len(sorted_signals):
                    logger.debug(f"Rate limiting: sleeping 1s before next batch")
                    await asyncio.sleep(1)
            
            logger.info(f"Generated commentary for {len(results)} signals")
            emit_metric("ai.commentary.batch_complete", 1, tags={'count': len(results)})
            
            return results
            
        except Exception as e:
            logger.error(f"AI commentary batch error: {e}")
            return {}
    
    async def _process_single_signal(self, signal: Dict[str, Any]) -> Optional[tuple]:
        """Process a single signal (generate narrative + strategy)"""
        ticker = signal.get('ticker', 'UNKNOWN')
        
        try:
            # Generate both in parallel
            narrative, strategy = await asyncio.gather(
                self._generate_risk_narrative(signal),
                self._generate_trade_strategy(signal),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(narrative, Exception):
                logger.error(f"Narrative generation failed for {ticker}: {narrative}")
                narrative = None
            if isinstance(strategy, Exception):
                logger.error(f"Strategy generation failed for {ticker}: {strategy}")
                strategy = None
            
            return (narrative, strategy)
            
        except Exception as e:
            logger.error(f"Error processing signal for {ticker}: {e}")
            return None
    
    async def _generate_risk_narrative(self, signal: Dict[str, Any]) -> Optional[AIRiskNarrative]:
        """Generate risk narrative for a signal"""
        if not self.enabled or not self.client:
            return None
        
        try:
            ticker = signal.get('ticker', 'UNKNOWN')
            signal_id = signal.get('id', '')
            
            logger.debug(f"Generating risk narrative for {ticker}")
            emit_metric("ai.risk_narrative.start", 1, tags={'ticker': ticker})
            
            # Build context from signal data
            context = self._build_risk_context(signal)
            
            # Generate three types of commentary in parallel
            risk_prompt = self._build_risk_prompt(ticker, context)
            opportunity_prompt = self._build_opportunity_prompt(ticker, context)
            market_prompt = self._build_market_context_prompt(ticker, context)
            
            # Call OpenAI for all three commentaries
            risk_commentary, opportunity_commentary, market_context = await asyncio.gather(
                self._call_openai(risk_prompt, max_tokens=250),
                self._call_openai(opportunity_prompt, max_tokens=200),
                self._call_openai(market_prompt, max_tokens=150),
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(risk_commentary, Exception):
                logger.warning(f"Risk commentary failed: {risk_commentary}")
                risk_commentary = None
            if isinstance(opportunity_commentary, Exception):
                logger.warning(f"Opportunity commentary failed: {opportunity_commentary}")
                opportunity_commentary = None
            if isinstance(market_context, Exception):
                logger.warning(f"Market context failed: {market_context}")
                market_context = None
            
            if not risk_commentary or not opportunity_commentary or not market_context:
                logger.warning(f"Incomplete AI narrative for {ticker}")
                emit_metric("ai.risk_narrative.incomplete", 1, tags={'ticker': ticker})
                return None
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_narrative_confidence(signal)
            
            narrative = AIRiskNarrative(
                signal_id=signal_id,
                ticker=ticker,
                risk_commentary=str(risk_commentary) if risk_commentary else "",
                opportunity_commentary=str(opportunity_commentary) if opportunity_commentary else "",
                market_context=str(market_context) if market_context else "",
                confidence_level=confidence
            )
            
            emit_metric("ai.risk_narrative.success", 1, tags={'ticker': ticker})
            return narrative
            
        except Exception as e:
            logger.error(f"Error generating risk narrative for {signal.get('ticker', 'UNKNOWN')}: {e}")
            emit_metric("ai.risk_narrative.error", 1)
            return None
    
    async def _generate_trade_strategy(self, signal: Dict[str, Any]) -> Optional[AITradeStrategy]:
        """Generate trade strategy for a signal"""
        if not self.enabled or not self.client:
            return None
        
        try:
            ticker = signal.get('ticker', 'UNKNOWN')
            signal_id = signal.get('id', '')
            
            logger.debug(f"Generating trade strategy for {ticker}")
            emit_metric("ai.trade_strategy.start", 1, tags={'ticker': ticker})
            
            # Analyze signal characteristics
            analysis = self._analyze_signal_for_strategy(signal)
            
            # Determine best strategy type
            strategy_type = self._determine_strategy_type(signal, analysis)
            
            # Generate strategy based on type
            if strategy_type == 'options':
                strategy = await self._generate_options_strategy(signal, analysis)
            elif strategy_type == 'combo':
                strategy = await self._generate_combo_strategy(signal, analysis)
            else:  # equity
                strategy = await self._generate_equity_strategy(signal, analysis)
            
            if strategy:
                emit_metric("ai.trade_strategy.success", 1, tags={'ticker': ticker, 'type': strategy_type})
            else:
                emit_metric("ai.trade_strategy.failed", 1, tags={'ticker': ticker})
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating trade strategy for {signal.get('ticker', 'UNKNOWN')}: {e}")
            emit_metric("ai.trade_strategy.error", 1)
            return None
    
    # ========================================================================
    # HELPERS: Context Building
    # ========================================================================
    
    def _build_risk_context(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Build context dictionary from signal data"""
        context = {
            'ticker': signal.get('ticker', 'UNKNOWN'),
            'score': signal.get('signal_score', 0),
            'confidence': signal.get('signal_confidence', 0),
        }
        
        # Technical metrics
        if 'technical_score' in signal:
            context['technical_score'] = signal['technical_score']
        if 'volatility' in signal:
            context['volatility'] = signal['volatility']
        if 'momentum_30d_pct' in signal:
            context['momentum'] = signal['momentum_30d_pct']
        if 'rsi' in signal:
            context['rsi'] = signal['rsi']
        
        # Fundamental metrics
        if 'fundamental_score' in signal:
            context['fundamental_score'] = signal['fundamental_score']
        if 'market_cap' in signal:
            context['market_cap'] = signal['market_cap']
        if 'sector' in signal:
            context['sector'] = signal['sector']
        if 'current_price' in signal:
            context['current_price'] = signal['current_price']
        
        # Social metrics
        if 'social_score' in signal:
            context['social_score'] = signal['social_score']
        if 'mention_count' in signal:
            context['mentions'] = signal['mention_count']
        if 'avg_sentiment' in signal:
            context['sentiment'] = signal['avg_sentiment']
        
        # Risk metrics
        if 'risk_score' in signal:
            context['risk_score'] = signal['risk_score']
        if 'risk_level' in signal:
            context['risk_level'] = signal['risk_level']
        
        return context
    
    def _build_risk_prompt(self, ticker: str, context: Dict[str, Any]) -> str:
        """Build OpenAI prompt for risk commentary"""
        context_str = self._format_context(context)
        
        prompt = f"""Analyze the key RISKS for {ticker} based on this data:

{context_str}

Provide a concise risk analysis (max 200 words) covering:
- Market volatility and price risks
- External factors (sector, macro, competition)
- Liquidity and execution risks
- Technical risk indicators

Be specific and actionable."""
        
        return prompt
    
    def _build_opportunity_prompt(self, ticker: str, context: Dict[str, Any]) -> str:
        """Build OpenAI prompt for opportunity commentary"""
        context_str = self._format_context(context)
        
        prompt = f"""Analyze the OPPORTUNITIES for {ticker} based on this data:

{context_str}

Provide a concise opportunity analysis (max 150 words) covering:
- Upside potential and catalysts
- Favorable trends and momentum
- Social sentiment drivers
- Strategic entry points

Be specific and actionable."""
        
        return prompt
    
    def _build_market_context_prompt(self, ticker: str, context: Dict[str, Any]) -> str:
        """Build OpenAI prompt for market context"""
        context_str = self._format_context(context)
        
        prompt = f"""Provide MARKET CONTEXT for {ticker} based on this data:

{context_str}

Give a brief market context summary (max 100 words) covering:
- Sector positioning
- Market cap category
- Current market conditions
- Relevant macro factors

Be concise and informative."""
        
        return prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary as readable string"""
        lines = []
        for key, value in context.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def _calculate_narrative_confidence(self, signal: Dict[str, Any]) -> float:
        """Calculate confidence level based on data completeness"""
        score = 0.0
        max_score = 6.0
        
        # Check data availability (1 point each)
        if signal.get('technical_score') is not None:
            score += 1.0
        if signal.get('fundamental_score') is not None:
            score += 1.0
        if signal.get('social_score') is not None:
            score += 1.0
        if signal.get('risk_score') is not None:
            score += 1.0
        if signal.get('signal_confidence', 0) > 0.5:
            score += 1.0
        if signal.get('mention_count', 0) > 10:  # Good social data
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    # ========================================================================
    # HELPERS: Strategy Analysis
    # ========================================================================
    
    def _analyze_signal_for_strategy(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze signal to determine strategy characteristics"""
        analysis = {
            'ticker': signal.get('ticker', 'UNKNOWN'),
            'signal_score': signal.get('signal_score', 0),
            'confidence': signal.get('signal_confidence', 0),
            'risk_score': signal.get('risk_score', 50),
            'risk_level': signal.get('risk_level', 'Moderate'),
            'volatility': signal.get('volatility', 0),
            'liquidity_score': signal.get('liquidity_score', 0.5),
            'market_cap': signal.get('market_cap', 0),
            'current_price': signal.get('current_price', 0),
            'momentum': signal.get('momentum_30d_pct', 0),
            'rsi': signal.get('rsi', 50),
            'max_position_size': signal.get('max_position_size', 0.05),
        }
        
        return analysis
    
    def _determine_strategy_type(self, signal: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Determine best strategy type (equity, options, combo)"""
        try:
            # High confidence + good liquidity -> consider options
            if analysis['confidence'] > 0.7 and analysis['liquidity_score'] > 0.5:
                # Check if stock is optionable (price > $5, market cap > $1B)
                if analysis['current_price'] > 5 and analysis['market_cap'] > 1e9:
                    # Very high confidence -> combo strategy
                    if analysis['confidence'] > 0.85:
                        return 'combo'
                    return 'options'
            
            # Default to equity
            return 'equity'
            
        except Exception as e:
            logger.error(f"Error determining strategy type: {e}")
            return 'equity'
    
    # ========================================================================
    # HELPERS: Strategy Generation
    # ========================================================================
    
    async def _generate_equity_strategy(self, signal: Dict[str, Any], 
                                       analysis: Dict[str, Any]) -> Optional[AITradeStrategy]:
        """Generate equity-focused strategy"""
        try:
            ticker = analysis['ticker']
            signal_id = signal.get('id', '')
            
            # Determine time horizon
            horizon = self._determine_time_horizon(analysis)
            
            # Calculate risk/reward
            risk_reward = self._calculate_risk_reward_ratio(analysis)
            
            # Position sizing (capped at 10% for equity)
            position_size = min(analysis['max_position_size'], 0.10)
            
            # Generate entry/exit conditions
            entry_conditions = self._generate_entry_conditions(signal, analysis)
            exit_conditions = self._generate_exit_conditions(signal, analysis)
            
            # Strategy name
            descriptor = self._get_strategy_descriptor(analysis)
            strategy_name = f"{ticker} {descriptor} Equity Play"
            
            strategy = AITradeStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_type='equity',
                horizon=horizon,
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=position_size,
                exit_conditions=json.dumps(exit_conditions),
                confidence_score=analysis['confidence'] * 100,
                risk_reward_ratio=risk_reward,
                liquidity_score=analysis['liquidity_score'],
                passes_guardrails=self._passes_guardrails(analysis)
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating equity strategy: {e}")
            return None
    
    async def _generate_options_strategy(self, signal: Dict[str, Any], 
                                        analysis: Dict[str, Any]) -> Optional[AITradeStrategy]:
        """Generate options-focused strategy"""
        try:
            ticker = analysis['ticker']
            signal_id = signal.get('id', '')
            current_price = analysis['current_price']
            
            if current_price <= 0:
                return None
            
            # Determine options type
            option_type = self._determine_options_type(analysis)
            
            # Generate strikes and expiration
            strikes = self._generate_strike_prices(current_price, option_type)
            expiration = self._generate_expiration_date(analysis)
            
            # Calculate max loss/gain
            max_loss, max_gain = self._calculate_options_payoff(strikes, option_type, current_price)
            
            # Entry/exit conditions
            entry_conditions = self._generate_options_entry_conditions(signal, analysis, option_type)
            exit_conditions = self._generate_exit_conditions(signal, analysis)
            
            # Strategy name
            strategy_name = f"{ticker} {option_type.replace('_', ' ').title()}"
            
            strategy = AITradeStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=strategy_name,
                strategy_type='options',
                horizon=self._determine_time_horizon(analysis),
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=min(analysis['max_position_size'], 0.05),  # Cap at 5% for options
                exit_conditions=json.dumps(exit_conditions),
                confidence_score=analysis['confidence'] * 100,
                risk_reward_ratio=max_gain / max_loss if max_loss > 0 else 3.0,
                option_strategy=option_type,
                strikes=strikes,
                expiration_date=expiration,
                max_loss=max_loss,
                max_gain=max_gain,
                liquidity_score=analysis['liquidity_score'],
                passes_guardrails=self._passes_guardrails(analysis)
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating options strategy: {e}")
            return None
    
    async def _generate_combo_strategy(self, signal: Dict[str, Any], 
                                      analysis: Dict[str, Any]) -> Optional[AITradeStrategy]:
        """Generate combo strategy (equity + options)"""
        try:
            ticker = analysis['ticker']
            signal_id = signal.get('id', '')
            
            # Combo: 50% equity, 50% options
            entry_conditions = {
                'equity_allocation': 0.5,
                'options_allocation': 0.5,
                'entry_trigger': 'market',
                'scaling': 'immediate'
            }
            
            exit_conditions = {
                'take_profit': analysis['signal_score'] * 0.15,  # 15% of signal score
                'stop_loss': -0.08,  # 8% stop
                'time_stop': '30_days',
                'partial_exits': True
            }
            
            strategy = AITradeStrategy(
                signal_id=signal_id,
                ticker=ticker,
                strategy_name=f"{ticker} High Conviction Combo",
                strategy_type='combo',
                horizon='swing',
                entry_type='market',
                entry_conditions=json.dumps(entry_conditions),
                entry_sizing=min(analysis['max_position_size'], 0.08),
                exit_conditions=json.dumps(exit_conditions),
                confidence_score=analysis['confidence'] * 100,
                risk_reward_ratio=self._calculate_risk_reward_ratio(analysis),
                liquidity_score=analysis['liquidity_score'],
                passes_guardrails=self._passes_guardrails(analysis)
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating combo strategy: {e}")
            return None
    
    # ========================================================================
    # HELPERS: Strategy Details
    # ========================================================================
    
    def _determine_time_horizon(self, analysis: Dict[str, Any]) -> str:
        """Determine time horizon based on signal characteristics"""
        volatility = analysis.get('volatility', 0)
        momentum = abs(analysis.get('momentum', 0))
        
        # High volatility + high momentum -> swing
        if volatility > 0.03 and momentum > 0.05:
            return 'swing'
        
        # Low volatility -> position
        if volatility < 0.015:
            return 'position'
        
        # Default
        return 'swing'
    
    def _calculate_risk_reward_ratio(self, analysis: Dict[str, Any]) -> float:
        """Calculate risk/reward ratio"""
        confidence = analysis.get('confidence', 0.5)
        risk_score = analysis.get('risk_score', 50)
        
        # Higher confidence + lower risk = better R/R
        base_rr = 2.0
        confidence_bonus = confidence * 2.0
        risk_penalty = (risk_score / 100) * 1.0
        
        return max(base_rr + confidence_bonus - risk_penalty, 1.0)
    
    def _get_strategy_descriptor(self, analysis: Dict[str, Any]) -> str:
        """Get descriptive name for strategy"""
        confidence = analysis.get('confidence', 0)
        momentum = analysis.get('momentum', 0)
        
        if confidence > 0.8:
            return "High Conviction"
        elif momentum > 0.05:
            return "Momentum"
        elif momentum < -0.05:
            return "Contrarian"
        else:
            return "Balanced"
    
    def _generate_entry_conditions(self, signal: Dict[str, Any], 
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate entry conditions"""
        return {
            'entry_type': 'market',
            'min_liquidity': 0.3,
            'max_spread': 0.02,
            'confirmation': 'signal_above_threshold'
        }
    
    def _generate_exit_conditions(self, signal: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate exit conditions"""
        risk_score = analysis.get('risk_score', 50)
        
        # More conservative exits for higher risk
        stop_loss = -0.05 if risk_score < 70 else -0.03
        take_profit = 0.12 if risk_score < 70 else 0.08
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time_stop': '30_days',
            'trailing_stop': True
        }
    
    def _determine_options_type(self, analysis: Dict[str, Any]) -> str:
        """Determine options strategy type"""
        confidence = analysis.get('confidence', 0)
        momentum = analysis.get('momentum', 0)
        volatility = analysis.get('volatility', 0)
        
        # Bullish signals
        if momentum > 0.03 and confidence > 0.7:
            if volatility > 0.03:
                return 'bull_call_spread'
            return 'long_call'
        
        # High volatility
        if volatility > 0.04:
            return 'iron_condor'
        
        # Default
        return 'long_call'
    
    def _generate_strike_prices(self, current_price: float, option_type: str) -> List[float]:
        """Generate strike prices for options strategy"""
        if option_type == 'long_call':
            # ATM or slightly OTM
            return [round(current_price * 1.02, 2)]
        
        elif option_type == 'bull_call_spread':
            # Long lower strike, short higher strike
            lower = round(current_price * 1.00, 2)
            upper = round(current_price * 1.05, 2)
            return [lower, upper]
        
        elif option_type == 'iron_condor':
            # Four strikes
            return [
                round(current_price * 0.95, 2),
                round(current_price * 0.97, 2),
                round(current_price * 1.03, 2),
                round(current_price * 1.05, 2)
            ]
        
        else:
            return [round(current_price, 2)]
    
    def _generate_expiration_date(self, analysis: Dict[str, Any]) -> str:
        """Generate expiration date for options"""
        horizon = self._determine_time_horizon(analysis)
        
        if horizon == 'swing':
            days = 30
        elif horizon == 'position':
            days = 60
        else:
            days = 14
        
        expiration = datetime.now() + timedelta(days=days)
        return expiration.strftime('%Y-%m-%d')
    
    def _calculate_options_payoff(self, strikes: List[float], option_type: str, 
                                 current_price: float) -> tuple[float, float]:
        """Calculate max loss and max gain for options strategy"""
        # Simplified calculation (actual option pricing would be more complex)
        if option_type == 'long_call':
            max_loss = current_price * 0.02  # ~2% premium
            max_gain = current_price * 0.20  # Assume 20% move
            
        elif option_type == 'bull_call_spread':
            spread = strikes[1] - strikes[0]
            max_loss = spread * 0.3  # ~30% of spread as premium
            max_gain = spread * 0.7
            
        elif option_type == 'iron_condor':
            max_loss = current_price * 0.03
            max_gain = current_price * 0.02
            
        else:
            max_loss = current_price * 0.02
            max_gain = current_price * 0.15
        
        return (max_loss, max_gain)
    
    def _generate_options_entry_conditions(self, signal: Dict[str, Any], 
                                          analysis: Dict[str, Any], 
                                          option_type: str) -> Dict[str, Any]:
        """Generate entry conditions specific to options"""
        return {
            'entry_type': 'limit',
            'max_premium': analysis['current_price'] * 0.03,
            'min_open_interest': 100,
            'max_bid_ask_spread': 0.05,
            'option_type': option_type
        }
    
    def _passes_guardrails(self, analysis: Dict[str, Any]) -> bool:
        """Check if strategy passes risk guardrails"""
        try:
            # Position size check
            if analysis['max_position_size'] > 0.15:
                return False
            
            # Liquidity check
            if analysis['liquidity_score'] < 0.2:
                return False
            
            # Risk score check
            if analysis['risk_score'] > 85:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking guardrails: {e}")
            return False
    
    # ========================================================================
    # HELPERS: OpenAI Integration
    # ========================================================================
    
    async def _call_openai(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call OpenAI API with error handling"""
        if not self.enabled or not self.client:
            return None
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst providing concise, actionable stock market insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            emit_metric("ai.openai.error", 1)
            return None


def create_ai_commentary_generator(enabled: bool = True) -> AICommentaryGenerator:
    """Factory: Create AI commentary generator"""
    return AICommentaryGenerator(enabled=enabled)


def get_ai_commentary_generator() -> AICommentaryGenerator:
    """Singleton: Get AI commentary generator"""
    if not hasattr(get_ai_commentary_generator, '_instance'):
        enabled = os.getenv('OPENAI_API_KEY') is not None
        get_ai_commentary_generator._instance = AICommentaryGenerator(enabled=enabled)
    
    return get_ai_commentary_generator._instance