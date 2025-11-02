"""
VP Investments API Module

Comprehensive API functionality including:
1. FastAPI REST server for trading signals and analysis
2. Rate-limited HTTP client utilities for external APIs  
3. Admin configuration interface
4. Real-time data updates and authentication

Consolidates server.py + http_client.py into unified API module.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Import from VP Investments modules
from backend.enums import SignalType
from ..core.config import ConfigManager, get_config
from ..storage.database import get_database, DatabaseInterface

# Legacy imports - these modules are archived but API may still reference them
try:
    from ..core.models import AnalysisRun
except ImportError:
    AnalysisRun = None  # Archived with old pipeline

try:
    from archive.core.signals_legacy import Signal
except ImportError:
    Signal = None  # Archived with old pipeline

logger = logging.getLogger(__name__)


# ================================================================================
# HTTP CLIENT UTILITIES - Rate-limited client for external API calls
# ================================================================================

class RateLimitedClient:
    """
    Rate-limited HTTP client with automatic backoff and retry logic
    """
    
    def __init__(
        self,
        base_url: str = "",
        api_key: Optional[str] = None,
        requests_per_minute: int = 60,
        requests_per_second: Optional[int] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Rate limiting tracking
        self.request_times: List[float] = []
        self.last_request_time = 0.0
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"[SUCCESS] RateLimitedClient initialized: {requests_per_minute} req/min")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            
            headers = {
                'User-Agent': 'VP-Investments/2.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Add API key to headers if provided
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
                headers['X-API-Key'] = self.api_key
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
    
    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Remove old request times (older than 1 minute)
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Check per-minute rate limit
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                logger.debug(f"⏳ Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Check per-second rate limit
        if self.requests_per_second:
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.debug(f"⏳ Per-second limit, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request time
        current_time = time.time()
        self.request_times.append(current_time)
        self.last_request_time = current_time
    
    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retries and error handling"""
        
        await self._ensure_session()
        
        # Construct full URL
        if url.startswith('http'):
            full_url = url
        else:
            full_url = f"{self.base_url}{url}"
        
        # Merge headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self.wait_if_needed()
                
                # Make request
                async with self.session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data if data else None,
                    headers=request_headers
                ) as response:
                    
                    # Handle different response types
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"⚠️ Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}: {error_text}"
                        )
                    
                    # Try to parse JSON response
                    try:
                        return await response.json()
                    except Exception:
                        # Return text if not JSON
                        text = await response.text()
                        return {"content": text, "status": response.status}
                        
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"⚠️ Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Request failed after {self.max_retries + 1} attempts: {e}")
        
        # If we get here, all retries failed
        raise last_exception or Exception("Request failed after all retries")
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make GET request"""
        return await self._make_request('GET', url, params=params, headers=headers)
    
    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make POST request"""
        return await self._make_request('POST', url, data=data, headers=headers)
    
    async def put(self, url: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make PUT request"""
        return await self._make_request('PUT', url, data=data, headers=headers)
    
    async def delete(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make DELETE request"""
        return await self._make_request('DELETE', url, headers=headers)


class APIClientPool:
    """Pool of HTTP clients for different services"""
    
    def __init__(self):
        self._clients: Dict[str, RateLimitedClient] = {}
        self._lock = asyncio.Lock()
    
    async def get_client(self, service: str, **kwargs) -> RateLimitedClient:
        """Get or create a client for a service"""
        async with self._lock:
            if service not in self._clients:
                self._clients[service] = RateLimitedClient(**kwargs)
            return self._clients[service]
    
    async def close_all(self):
        """Close all clients"""
        async with self._lock:
            for client in self._clients.values():
                if client.session:
                    await client.session.close()
            self._clients.clear()


async def create_reddit_client(client_id: str, client_secret: str, user_agent: str) -> RateLimitedClient:
    """Create a Reddit API client with appropriate rate limiting"""
    
    # Reddit OAuth token endpoint for getting access token
    auth_client = RateLimitedClient(
        base_url="https://www.reddit.com/api/v1",
        requests_per_minute=60
    )
    
    # Get access token
    import base64
    
    auth_string = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    headers = {
        'Authorization': f'Basic {auth_string}',
        'User-Agent': user_agent
    }
    
    data = {
        'grant_type': 'client_credentials'
    }
    
    try:
        response = await auth_client.post('/access_token', data=data, headers=headers)
        access_token = response.get('access_token')
        
        if not access_token:
            raise ValueError("Failed to get Reddit access token")
        
        # Create authenticated Reddit client
        reddit_client = RateLimitedClient(
            base_url="https://oauth.reddit.com",
            api_key=access_token,
            requests_per_minute=60,  # Reddit's rate limit
            requests_per_second=1    # Conservative per-second limit
        )
        
        logger.info("[SUCCESS] Created authenticated Reddit client")
        return reddit_client
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create Reddit client: {e}")
        raise
    finally:
        if auth_client.session:
            await auth_client.session.close()


# Global client pool instance
_global_client_pool = APIClientPool()


def get_client_pool() -> APIClientPool:
    """Get the global client pool"""
    return _global_client_pool


# ================================================================================
# REST API SERVER - FastAPI application for serving trading signals
# ================================================================================

# Pydantic models for API
class SignalResponse(BaseModel):
    """API response model for signals (Phase 7)"""
    ticker: str
    signal_type: str
    confidence: float
    technical_score: float
    sentiment_score: float
    signal_score: float  # Phase 7: primary score field
    combined_score: float  # Backward compatibility (deprecated)
    created_at: datetime


class TradingRecommendationResponse(BaseModel):
    """API response model for trading recommendations"""
    ticker: str
    action: str  # buy, sell, hold
    confidence: float
    risk_level: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: List[str]
    created_at: datetime


class AnalysisRunResponse(BaseModel):
    """API response model for analysis runs"""
    id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    signals_processed: int
    recommendations_generated: int
    metrics: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates"""
    section: str
    updates: Dict[str, Any]


class PipelineRunRequest(BaseModel):
    """Request model for pipeline execution"""
    tickers: Optional[List[str]] = None
    force_refresh: bool = False
    analysis_type: str = "full"


# Global state for API
_app_state = {
    'database': None,
    'config_manager': None,
    'background_tasks': set()
}


# Authentication
security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    # Simple token verification - in production, use proper JWT validation
    config = get_config()
    valid_tokens = config.get('api', {}).get('valid_tokens', [])
    
    if credentials.credentials not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    logger.info("🚀 Starting VP Investments API...")
    
    try:
        # Initialize database
        _app_state['database'] = get_database()
        await _app_state['database'].connect()
        
        # Initialize configuration manager
        _app_state['config_manager'] = ConfigManager()
        
        logger.info("✅ API startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down VP Investments API...")
        
        # Close database
        if _app_state['database']:
            await _app_state['database'].disconnect()
        
        # Close HTTP client pool
        await get_client_pool().close_all()
        
        logger.info("✅ API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="VP Investments API",
    description="Trading signals and analysis API for VP Investments platform",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .admin import router as admin_router
from .auth import router as auth_router
from .monitoring import router as monitoring_router
app.include_router(admin_router)
app.include_router(auth_router, prefix="/api")
app.include_router(monitoring_router)


# ================================================================================
# API ENDPOINTS
# ================================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return """
    <html>
        <head>
            <title>VP Investments API</title>
        </head>
        <body>
            <h1>VP Investments API v2.0</h1>
            <p>Trading signals and analysis API</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = _app_state['database']
        if db:
            await db.execute_query("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now()
            }
        )


@app.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    limit: int = 10,
    signal_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    token: str = Depends(verify_token)
):
    """Get trading signals"""
    try:
        db = _app_state['database']
        
        # Build query
        query = "SELECT * FROM signals WHERE 1=1"
        params = {}
        
        if signal_type:
            query += " AND signal_type = :signal_type"
            params['signal_type'] = signal_type
        
        if min_confidence is not None:
            query += " AND confidence >= :min_confidence"
            params['min_confidence'] = min_confidence
        
        query += " ORDER BY created_at DESC LIMIT :limit"
        params['limit'] = limit
        
        results = await db.execute_query(query, params)
        
        # Convert to response models
        signals = []
        for result in results:
            signals.append(SignalResponse(
                ticker=result['ticker'],
                signal_type=result['signal_type'],
                confidence=result['confidence'],
                technical_score=result.get('technical_score', 0.0),
                sentiment_score=result.get('sentiment_score', 0.0),
                combined_score=result.get('combined_score', 0.0),
                created_at=result['created_at']
            ))
        
        return signals
        
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals: {e}")


@app.get("/signals/{ticker}", response_model=List[SignalResponse])
async def get_signals_for_ticker(
    ticker: str,
    limit: int = 10,
    token: str = Depends(verify_token)
):
    """Get signals for a specific ticker"""
    try:
        db = _app_state['database']
        
        query = """
        SELECT * FROM signals 
        WHERE ticker = :ticker 
        ORDER BY created_at DESC 
        LIMIT :limit
        """
        
        results = await db.execute_query(query, {'ticker': ticker.upper(), 'limit': limit})
        
        signals = []
        for result in results:
            signal_score_val = result.get('signal_score', result.get('combined_score', 0.0))  # Phase 7
            signals.append(SignalResponse(
                ticker=result['ticker'],
                signal_type=result['signal_type'],
                confidence=result['confidence'],
                technical_score=result.get('technical_score', 0.0),
                sentiment_score=result.get('sentiment_score', 0.0),
                signal_score=signal_score_val,  # Phase 7
                combined_score=signal_score_val,  # Backward compatibility
                created_at=result['created_at']
            ))
        
        return signals
        
    except Exception as e:
        logger.error(f"Error fetching signals for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals for {ticker}: {e}")


@app.post("/pipeline/run")
async def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Trigger analysis pipeline execution"""
    try:
        # Run pipeline in background
        task_id = f"pipeline_{int(time.time())}"
        
        async def pipeline_task():
            try:
                # Import here to avoid circular imports
                from ..data.pipeline import create_pipeline_from_config
                
                pipeline = create_pipeline_from_config()
                
                await pipeline.run_analysis(
                    tickers=request.tickers,
                    force_refresh=request.force_refresh
                )
                
                logger.info(f"Pipeline task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Pipeline task {task_id} failed: {e}")
            finally:
                _app_state['background_tasks'].discard(task_id)
        
        # Add to background tasks
        _app_state['background_tasks'].add(task_id)
        background_tasks.add_task(pipeline_task)
        
        return {
            "message": "Pipeline execution started",
            "task_id": task_id,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {e}")


@app.get("/analytics/global")
async def get_global_analytics(
    run_id: Optional[str] = None,
    bucket: Optional[str] = None,
    interval: Optional[str] = None
):
    """
    Get global analytics data with optional filtering.
    
    Args:
        run_id: Filter by specific pipeline run (optional, defaults to latest)
        bucket: Filter score bucket performance ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')
        interval: Filter by time interval ('1d', '3d', '7d', '10d', '14d', '30d', '90d')
        
    Returns:
        Complete analytics payload with all subsections
    """
    try:
        db = _app_state['database']
        
        # Get analytics record (latest or by run_id)
        if run_id:
            query = db.client.table('analytics').select('*').eq('run_id', run_id)
        else:
            query = db.client.table('analytics').select('*').order('created_at', desc=True).limit(1)
        
        result = query.execute()
        
        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="No analytics data found")
        
        analytics = result.data[0]
        
        # Build response with all subsections
        response = {
            "run_id": analytics['run_id'],
            "created_at": analytics['created_at'],
            "total_signals": analytics['total_signals'],
            "signals_analyzed": analytics.get('signals_analyzed'),
            
            # Basic metrics
            "avg_overall_score": analytics.get('avg_overall_score'),
            "avg_technical_score": analytics.get('avg_technical_score'),
            "avg_fundamental_score": analytics.get('avg_fundamental_score'),
            "avg_news_macro_score": analytics.get('avg_news_macro_score'),
            "avg_social_alternative_score": analytics.get('avg_social_alternative_score'),
            "avg_risk_stability_score": analytics.get('avg_risk_stability_score'),
            "avg_institutional_score": analytics.get('avg_institutional_score'),
            
            # Sector performance
            "top_sector": analytics.get('top_sector'),
            "top_sector_avg_return": analytics.get('top_sector_avg_return'),
            "top_sector_count": analytics.get('top_sector_count'),
            "worst_sector": analytics.get('worst_sector'),
            "worst_sector_avg_return": analytics.get('worst_sector_avg_return'),
            "worst_sector_count": analytics.get('worst_sector_count'),
            "sector_performance": analytics.get('sector_performance'),
            
            # Advanced analytics (JSONB columns)
            "score_bucket_performance": analytics.get('score_bucket_performance'),
            "factor_correlations": analytics.get('factor_correlations'),
            "factor_contributions": analytics.get('factor_contributions'),
            "group_performance": analytics.get('group_performance'),
            "backtest_cumulative_returns": analytics.get('backtest_cumulative_returns'),
            "top_factors": analytics.get('top_factors')
        }
        
        # Apply filters if provided
        if bucket and response.get('score_bucket_performance'):
            bucket_data = response['score_bucket_performance'].get(bucket)
            if bucket_data:
                response['score_bucket_performance'] = {bucket: bucket_data}
        
        if interval:
            # Filter interval-specific data
            for key in ['win_rate', 'sharpe_ratio', 'max_drawdown', 'avg_return', 'avg_alpha']:
                full_key = f'{key}_{interval}'
                if full_key in analytics:
                    response[full_key] = analytics[full_key]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


@app.get("/config")
async def get_config_info(token: str = Depends(verify_token)):
    """Get current configuration"""
    try:
        config = get_config()
        
        # Return non-sensitive config information
        safe_config = {
            "api": {
                "port": config.get('api', {}).get('port', 8000),
                "host": config.get('api', {}).get('host', '0.0.0.0')
            },
            "analysis": config.get('analysis', {}),
            "data_sources": {k: "configured" for k in config.get('data_sources', {}).keys()}
        }
        
        return safe_config
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {e}")


# ================================================================================
# SERVER STARTUP FUNCTIONS
# ================================================================================

def create_api_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    return app


async def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server (sync wrapper)"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


# ================================================================================
# EXPORTS
# ================================================================================

__all__ = [
    # HTTP Client utilities
    'RateLimitedClient',
    'APIClientPool',
    'get_client_pool',
    'create_reddit_client',
    
    # FastAPI server
    'app',
    'create_api_app',
    'start_api_server',
    'run_api_server',
    
    # API models
    'SignalResponse',
    'TradingRecommendationResponse',
    'AnalysisRunResponse',
    'ConfigUpdateRequest',
    'PipelineRunRequest'
]