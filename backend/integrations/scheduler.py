"""
Scheduled Pipeline Runner

Manages 3x daily discovery pipeline runs:
1. Pre-market: 8:30 AM EST (1 hour before market opens)
2. Mid-day: 1:00 PM EST (middle of trading day) 
3. Post-market: 5:00 PM EST (1 hour after market closes)

Plus daily ticker refresh at midnight EST from abbadata/stock-tickers
"""

import logging
import schedule
import time
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List
import json
import uuid
from backend.integrations.yfinance import YahooFinanceIntegrator
from backend.integrations.reddit import RedditDataIntegrator
from backend.storage.database import get_database
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScheduledPipelineRunner:
    """Manages scheduled execution of discovery pipeline runs"""
    
    def __init__(self):
        self.db = get_database()
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Initialize integrators
        self.yahoo_integrator = YahooFinanceIntegrator(batch_size=100, max_workers=8)
        try:
            self.reddit_integrator = RedditDataIntegrator()
            self.reddit_available = True
        except Exception as e:
            logger.error(f"⚠️ Reddit integration not available: {e}")
            self.reddit_available = False
    
    def create_run_record(self, run_type: str = 'discovery_pipeline') -> str:
        """Create a new run record in the database"""
        run_id = f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_data = {
            'run_id': run_id,
            'run_type': run_type,
            'started_at': datetime.utcnow().isoformat(),
            'status': 'running',
            'total_signals': 0,
            'metadata': {
                'scheduled_run': True,
                'est_time': datetime.now(self.est_tz).isoformat(),
                'subreddits': [
                    'wallstreetbets', 'stocks', 'investing', 
                    'SecurityAnalysis', 'ValueInvesting', 'StockMarket', 'pennystocks'
                ]
            }
        }
        
        self.db.client.table('runs').insert(run_data).execute()
        logger.info(f"✅ Created run record: {run_id}")
        return run_id
    
    def complete_run_record(self, run_id: str, total_signals: int, metadata: Dict = None):
        """Mark run as completed with results"""
        update_data = {
            'completed_at': datetime.utcnow().isoformat(),
            'status': 'completed',
            'total_signals': total_signals
        }
        
        if metadata:
            update_data['metadata'] = metadata
        
        self.db.client.table('runs').update(update_data).eq('run_id', run_id).execute()
        logger.info(f"✅ Completed run: {run_id} with {total_signals} signals")
    
    def store_signals(self, run_id: str, ticker_scores: Dict) -> int:
        """Store discovered signals in the database"""
        signals_data = []
        
        # Sort tickers by combined score
        sorted_tickers = sorted(
            ticker_scores.items(), 
            key=lambda x: x[1].get('combined_score', 0), 
            reverse=True
        )
        
        for rank, (ticker, scores) in enumerate(sorted_tickers, 1):
            signal_data = {
                'run_id': run_id,
                'ticker': ticker,
                'rank': rank,
                'weighted_score': round(scores.get('combined_score', 0), 3),
                'reddit_score': round(scores.get('reddit_score', 0), 3),
                'news_score': round(scores.get('news_score', 0.5), 3),  # Default neutral
                'financial_score': round(scores.get('financial_score', 0.5), 3),
                'sentiment_score': round(scores.get('sentiment_score', 0.5), 3),
                'volume_score': round(scores.get('volume_score', 0.5), 3),
                'trade_type': self._determine_trade_type(scores),
                'risk_level': self._determine_risk_level(scores),
                'metadata': {
                    'reddit_mentions': scores.get('mention_count', 0),
                    'discovery_method': 'scheduled_pipeline',
                    'subreddits_found': scores.get('subreddits_found', [])
                }
            }
            signals_data.append(signal_data)
        
        # Insert in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(signals_data), batch_size):
            batch = signals_data[i:i + batch_size]
            self.db.client.table('signals_norm').insert(batch).execute()
            total_inserted += len(batch)
            logger.info(f"✅ Inserted signals batch {i//batch_size + 1}")
        
        return total_inserted
    
    def _determine_trade_type(self, scores: Dict) -> str:
        """Determine trade type based on score characteristics"""
        financial_score = scores.get('financial_score', 0.5)
        reddit_score = scores.get('reddit_score', 0.5)
        
        if financial_score > 0.75 and reddit_score > 0.6:
            return 'Growth'
        elif financial_score > 0.8:
            return 'Value'  
        elif reddit_score > 0.8:
            return 'Momentum'
        elif reddit_score > 0.6:
            return 'Swing'
        else:
            return 'Speculative'
    
    def _determine_risk_level(self, scores: Dict) -> str:
        """Determine risk level based on score characteristics"""
        combined_score = scores.get('combined_score', 0.5)
        financial_score = scores.get('financial_score', 0.5)
        
        if combined_score > 0.7 and financial_score > 0.7:
            return 'Low'
        elif combined_score > 0.5 and financial_score > 0.5:
            return 'Medium'
        else:
            return 'High'
    
    def run_discovery_pipeline(self) -> Dict[str, Any]:
        """Execute complete discovery pipeline with real data"""
        est_time = datetime.now(self.est_tz)
        logger.info(f"🚀 Starting discovery pipeline run at {est_time.strftime('%Y-%m-%d %H:%M:%S EST')}")
        
        # Create run record
        run_id = self.create_run_record()
        
        try:
            # Step 1: Reddit Discovery (if available)
            ticker_scores = {}
            
            if self.reddit_available:
                logger.info("📋 Step 1: Reddit ticker discovery...")
                reddit_results = self.reddit_integrator.run_full_reddit_scrape()
                
                for ticker, data in reddit_results['ticker_scores'].items():
                    ticker_scores[ticker] = {
                        'reddit_score': data['reddit_score'],
                        'mention_count': data['mention_count'],
                        'sentiment_score': self._calculate_avg_sentiment(data['mentions']),
                        'subreddits_found': list(set([m.get('subreddit', '') for m in data['mentions']]))
                    }
                
                logger.info(f"✅ Reddit discovery: {len(ticker_scores)} tickers found")
            else:
                logger.warning("⚠️ Reddit integration unavailable, skipping Reddit discovery")
            
            # Step 2: Financial Scoring
            logger.info("💰 Step 2: Financial scoring...")
            for ticker in ticker_scores.keys():
                financial_score = self.yahoo_integrator.calculate_financial_score(ticker)
                ticker_scores[ticker]['financial_score'] = financial_score
            
            # Step 3: Combined Scoring
            logger.info("🎯 Step 3: Combined scoring...")
            for ticker, scores in ticker_scores.items():
                # Weighted combination of scores
                reddit_weight = 0.4
                financial_weight = 0.3
                sentiment_weight = 0.2
                volume_weight = 0.1  # Placeholder for future volume analysis
                
                combined_score = (
                    scores.get('reddit_score', 0) * reddit_weight +
                    scores.get('financial_score', 0.5) * financial_weight +
                    scores.get('sentiment_score', 0.5) * sentiment_weight +
                    0.5 * volume_weight  # Default volume score
                )
                
                ticker_scores[ticker]['combined_score'] = combined_score
                ticker_scores[ticker]['volume_score'] = 0.5  # Placeholder
                ticker_scores[ticker]['news_score'] = 0.5  # Placeholder
            
            # Step 4: Store Results
            logger.info("💾 Step 4: Storing results...")
            total_signals = self.store_signals(run_id, ticker_scores)
            
            # Step 5: Record Metrics
            self._record_metrics(run_id, ticker_scores, reddit_results if self.reddit_available else None)
            
            # Complete run
            self.complete_run_record(run_id, total_signals, {
                'reddit_available': self.reddit_available,
                'subreddits_scraped': reddit_results['scrape_results']['subreddits_scraped'] if self.reddit_available else [],
                'total_mentions': reddit_results['scrape_results']['total_mentions'] if self.reddit_available else 0,
                'top_ticker': max(ticker_scores.keys(), key=lambda t: ticker_scores[t]['combined_score']) if ticker_scores else None
            })
            
            logger.info(f"🎉 Discovery pipeline completed successfully!")
            logger.info(f"📊 Run ID: {run_id}")
            logger.info(f"🎯 Total signals: {total_signals}")
            
            return {
                'success': True,
                'run_id': run_id,
                'total_signals': total_signals,
                'ticker_scores': ticker_scores
            }
            
        except Exception as e:
            logger.error(f"❌ Discovery pipeline failed: {e}")
            
            # Mark run as failed
            self.db.client.table('runs').update({
                'completed_at': datetime.utcnow().isoformat(),
                'status': 'failed',
                'error_message': str(e)
            }).eq('run_id', run_id).execute()
            
            return {
                'success': False,
                'error': str(e),
                'run_id': run_id
            }
    
    def _calculate_avg_sentiment(self, mentions: List[Dict]) -> float:
        """Calculate average sentiment from Reddit mentions"""
        if not mentions:
            return 0.5
        
        sentiments = [m.get('sentiment', 0.5) for m in mentions]
        return sum(sentiments) / len(sentiments)
    
    def _record_metrics(self, run_id: str, ticker_scores: Dict, reddit_results: Dict = None):
        """Record pipeline performance metrics"""
        metrics_data = [
            {
                'run_id': run_id,
                'metric_name': 'total_tickers_discovered',
                'value': len(ticker_scores),
                'units': 'count'
            },
            {
                'run_id': run_id,
                'metric_name': 'avg_combined_score',
                'value': sum(s['combined_score'] for s in ticker_scores.values()) / len(ticker_scores) if ticker_scores else 0,
                'units': 'score'
            },
            {
                'run_id': run_id,
                'metric_name': 'high_confidence_signals',
                'value': len([s for s in ticker_scores.values() if s['combined_score'] > 0.7]),
                'units': 'count'
            }
        ]
        
        if reddit_results:
            metrics_data.extend([
                {
                    'run_id': run_id,
                    'metric_name': 'reddit_mentions_total',
                    'value': reddit_results['scrape_results']['total_mentions'],
                    'units': 'count'
                },
                {
                    'run_id': run_id,
                    'metric_name': 'subreddits_scraped',
                    'value': len(reddit_results['scrape_results']['subreddits_scraped']),
                    'units': 'count'
                }
            ])
        
        self.db.client.table('metrics').insert(metrics_data).execute()
    
    def refresh_company_tickers(self):
        """Daily refresh of company tickers from GitHub"""
        logger.info("🔄 Starting daily ticker refresh...")
        
        try:
            # Run the existing setup script to refresh from GitHub
            result = subprocess.run([
                sys.executable, 'setup_supabase_complete.py'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("✅ Company tickers refreshed successfully")
            else:
                logger.error(f"❌ Ticker refresh failed: {result.stderr}")
        except Exception as e:
            logger.error(f"❌ Ticker refresh error: {e}")
    
    def setup_schedule(self):
        """Setup the 3x daily schedule + daily ticker refresh"""
        logger.info("📅 Setting up discovery pipeline schedule...")
        
        # 3x daily discovery runs (EST times)
        schedule.every().day.at("08:30").do(self.run_discovery_pipeline).tag('discovery', 'premarket')
        schedule.every().day.at("13:00").do(self.run_discovery_pipeline).tag('discovery', 'midday') 
        schedule.every().day.at("17:00").do(self.run_discovery_pipeline).tag('discovery', 'postmarket')
        
        # Daily ticker refresh at midnight EST
        schedule.every().day.at("00:00").do(self.refresh_company_tickers).tag('refresh')
        
        logger.info("✅ Schedule configured:")
        logger.info("  • 08:30 EST: Pre-market discovery run")
        logger.info("  • 13:00 EST: Mid-day discovery run") 
        logger.info("  • 17:00 EST: Post-market discovery run")
        logger.info("  • 00:00 EST: Daily ticker refresh")
    
    def run_scheduler(self):
        """Run the scheduled pipeline (blocking)"""
        self.setup_schedule()
        
        logger.info("🚀 VP Investments Discovery Pipeline Scheduler started")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("⏹️ Scheduler stopped by user")

def main():
    """Run scheduled pipeline or test individual components"""
    import sys
    
    runner = ScheduledPipelineRunner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            # Test run
            logger.info("🧪 Running test discovery pipeline...")
            result = runner.run_discovery_pipeline()
            if result['success']:
                logger.info(f"✅ Test successful: {result['total_signals']} signals generated")
            else:
                logger.error(f"❌ Test failed: {result['error']}")
        
        elif command == 'refresh':
            # Test ticker refresh
            runner.refresh_company_tickers()
        
        elif command == 'schedule':
            # Start scheduler
            runner.run_scheduler()
        
        else:
            logger.error("Usage: python scheduled_pipeline_runner.py [test|refresh|schedule]")
    
    else:
        # Default: run single discovery pipeline
        logger.info("🎯 Running single discovery pipeline...")
        result = runner.run_discovery_pipeline()
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()