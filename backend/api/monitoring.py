"""
Monitoring Dashboard API for VP Investments Platform.

Provides system health metrics, factor quality indicators, storage usage,
and pipeline performance analytics.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated
import logging
from datetime import datetime, timedelta
import psutil
import os

from .auth import get_current_admin_user, User

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

logger = logging.getLogger(__name__)


class SystemHealth(BaseModel):
    """System health metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    uptime_seconds: float
    timestamp: str


class FactorQuality(BaseModel):
    """Factor quality metrics."""
    total_factors: int
    success_rate: float
    avg_calculation_time_ms: float
    failed_factors: List[str]
    recent_runs: List[Dict]


class PipelineMetrics(BaseModel):
    """Pipeline performance metrics."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_tickers_per_run: float
    avg_signals_per_run: float
    avg_runtime_minutes: float
    last_run_time: Optional[str]
    runs_last_24h: int


class StorageMetrics(BaseModel):
    """Storage usage metrics."""
    total_pipeline_runs: int
    total_signals: int
    total_analytics: int
    database_size_mb: float
    table_sizes: Dict[str, float]


class DashboardOverview(BaseModel):
    """Complete dashboard overview."""
    system_health: SystemHealth
    factor_quality: FactorQuality
    pipeline_metrics: PipelineMetrics
    storage_metrics: StorageMetrics


@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get current system health metrics.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    """
    try:
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage (for the project directory)
        disk = psutil.disk_usage('/')
        
        # System uptime (approximation)
        uptime = 0
        try:
            boot_time = psutil.boot_time()
            uptime = datetime.now().timestamp() - boot_time
        except:
            pass
        
        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            uptime_seconds=uptime,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/factor-quality", response_model=FactorQuality)
async def get_factor_quality(
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get factor quality metrics from recent pipeline runs.
    
    Calculates factor coverage and success rates from actual signals in database.
    Counts factors from config/factor_to_group.yaml for accurate totals.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    """
    from backend.storage.database import get_supabase_database
    import yaml
    
    # Load factor count from config
    try:
        with open('config/factor_to_group.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Count total factors across all groups
        total_factors = 0
        for group_name in ['technical', 'fundamental', 'news_macro', 'social_alternative', 'risk_stability', 'institutional_smart_money']:
            if group_name in config:
                total_factors += len(config[group_name])
    except Exception as e:
        logger.warning(f"Could not load factor count from config: {e}")
        total_factors = 158  # Fallback estimate
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get recent runs (last 5) to calculate factor quality
        runs_result = db.client.table("signal_runs") \
            .select("id, run_timestamp, duration_seconds") \
            .order("run_timestamp", desc=True) \
            .limit(5) \
            .execute()
        
        if not runs_result.data:
            return FactorQuality(
                total_factors=total_factors,
                success_rate=0.0,
                avg_calculation_time_ms=0.0,
                failed_factors=[],
                recent_runs=[]
            )
        
        recent_runs = []
        total_success_rate = 0.0
        total_duration_ms = 0.0
        duration_count = 0
        
        for run in runs_result.data:
            run_id = run['id']
            
            # Get signals for this run
            signals_result = db.client.table("signals") \
                .select("technical_coverage, fundamental_coverage, news_macro_coverage, social_alternative_coverage, risk_stability_coverage, institutional_smart_money_coverage, total_coverage") \
                .eq("run_id", run_id) \
                .execute()
            
            if signals_result.data:
                # Calculate average coverage across all factor groups
                coverages = []
                for signal in signals_result.data:
                    coverages.extend([
                        signal.get('technical_coverage', 0) or 0,
                        signal.get('fundamental_coverage', 0) or 0,
                        signal.get('news_macro_coverage', 0) or 0,
                        signal.get('social_alternative_coverage', 0) or 0,
                        signal.get('risk_stability_coverage', 0) or 0,
                        signal.get('institutional_smart_money_coverage', 0) or 0
                    ])
                
                # Success rate = average coverage across all signals and factor groups
                run_success_rate = (sum(coverages) / len(coverages)) if coverages else 0.0
                total_success_rate += run_success_rate
                
                # Track duration for average calculation time
                if run.get('duration_seconds'):
                    # Convert to milliseconds and divide by number of signals for per-ticker average
                    num_signals = len(signals_result.data)
                    if num_signals > 0:
                        total_duration_ms += (run['duration_seconds'] * 1000) / num_signals
                        duration_count += 1
                
                recent_runs.append({
                    'timestamp': run['run_timestamp'],
                    'success_rate': run_success_rate,
                    'total_factors': total_factors
                })
        
        # Calculate overall averages
        avg_success_rate = total_success_rate / len(recent_runs) if recent_runs else 0.0
        avg_time_ms = total_duration_ms / duration_count if duration_count > 0 else 0.0
        
        return FactorQuality(
            total_factors=total_factors,
            success_rate=avg_success_rate,
            avg_calculation_time_ms=avg_time_ms,
            failed_factors=[],  # Would need detailed factor-level tracking
            recent_runs=recent_runs
        )
        
    except Exception as e:
        logger.error(f"Failed to get factor quality: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get factor quality: {str(e)}")
        
    finally:
        await db.disconnect()


@router.get("/pipeline-metrics", response_model=PipelineMetrics)
async def get_pipeline_metrics(
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get pipeline performance metrics.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get all pipeline runs
        result = db.client.table("signal_runs") \
            .select("*") \
            .execute()
        
        runs = result.data if result.data else []
        
        total_runs = len(runs)
        # Count successful runs based on status = 'completed'
        successful_runs = len([r for r in runs if r.get('status') == 'completed'])
        failed_runs = len([r for r in runs if r.get('status') == 'failed'])
        
        # Calculate averages
        avg_tickers = 0.0
        avg_signals = 0.0
        avg_runtime = 0.0
        last_run_time = None
        runs_last_24h = 0
        
        if runs:
            tickers = [r.get('total_tickers', 0) for r in runs]
            signals = [r.get('successful_tickers', 0) for r in runs]
            durations = [r.get('duration_seconds', 0) for r in runs if r.get('duration_seconds')]
            
            avg_tickers = sum(tickers) / len(tickers) if tickers else 0.0
            avg_signals = sum(signals) / len(signals) if signals else 0.0
            # Convert seconds to minutes
            avg_runtime = (sum(durations) / len(durations) / 60.0) if durations else 0.0
            
            # Get most recent run time
            sorted_runs = sorted(runs, key=lambda x: x.get('run_timestamp', ''), reverse=True)
            if sorted_runs:
                last_run_time = sorted_runs[0].get('run_timestamp')
                
                # Count runs in last 24h
                now = datetime.now()
                cutoff = now - timedelta(hours=24)
                
                for run in sorted_runs:
                    run_timestamp = run.get('run_timestamp', '')
                    try:
                        run_time = datetime.fromisoformat(run_timestamp.replace('Z', '+00:00'))
                        if run_time.replace(tzinfo=None) > cutoff:
                            runs_last_24h += 1
                    except:
                        continue
        
        return PipelineMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            avg_tickers_per_run=avg_tickers,
            avg_signals_per_run=avg_signals,
            avg_runtime_minutes=avg_runtime,
            last_run_time=last_run_time,
            runs_last_24h=runs_last_24h
        )
        
    except Exception as e:
        logger.error(f"Failed to get pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline metrics: {str(e)}")
        
    finally:
        await db.disconnect()


@router.get("/storage", response_model=StorageMetrics)
async def get_storage_metrics(
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get storage usage metrics.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get counts from main tables
        tables = ["signal_runs", "signals", "analytics", "performance"]
        table_counts = {}
        
        for table in tables:
            try:
                result = db.client.table(table) \
                    .select("*", count="exact") \
                    .execute()
                table_counts[table] = result.count if hasattr(result, 'count') else 0
            except:
                table_counts[table] = 0
        
        # Estimate database size (rough approximation)
        # Assume ~1KB per signal, ~500B per analytics, ~2KB per pipeline run
        estimated_size_mb = (
            table_counts.get('signals', 0) * 1.0 / 1024 +
            table_counts.get('analytics', 0) * 0.5 / 1024 +
            table_counts.get('signal_runs', 0) * 2.0 / 1024
        )
        
        return StorageMetrics(
            total_pipeline_runs=table_counts.get('signal_runs', 0),
            total_signals=table_counts.get('signals', 0),
            total_analytics=table_counts.get('analytics', 0),
            database_size_mb=estimated_size_mb,
            table_sizes={
                table: count * 1.0 / 1024  # Rough KB estimate
                for table, count in table_counts.items()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get storage metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage metrics: {str(e)}")
        
    finally:
        await db.disconnect()


@router.get("/dashboard", response_model=DashboardOverview)
async def get_dashboard_overview(
    # Temporarily disabled auth for development
    # current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get complete dashboard overview with all metrics.
    
    DEV MODE: Authentication temporarily disabled for development
    """
    try:
        # Create a mock user for the internal calls
        from .auth import User
        mock_user = User(username="dev", role="admin", full_name="Development User")
        
        system_health = await get_system_health(mock_user)
        factor_quality = await get_factor_quality(mock_user)
        pipeline_metrics = await get_pipeline_metrics(mock_user)
        storage_metrics = await get_storage_metrics(mock_user)
        
        return DashboardOverview(
            system_health=system_health,
            factor_quality=factor_quality,
            pipeline_metrics=pipeline_metrics,
            storage_metrics=storage_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard overview: {str(e)}")
