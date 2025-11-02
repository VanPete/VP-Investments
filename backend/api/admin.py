"""
Admin API endpoints for VP Investments Pipeline management.

Provides administrative functions including:
- Delete pipeline runs with cascade to all related tables
- Preview deletion impact before committing
- List and manage pipeline runs
- Bulk operations for multiple runs
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Annotated
import logging
from datetime import datetime

from .auth import get_current_admin_user, User

router = APIRouter(prefix="/api/admin", tags=["admin"])

logger = logging.getLogger(__name__)


class DeleteRunRequest(BaseModel):
    """Request to delete a pipeline run."""
    run_id: str
    confirm: bool = False  # Must be True to actually delete


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple pipeline runs."""
    run_ids: List[str]
    confirm: bool = False


class BulkDeleteResponse(BaseModel):
    """Response from bulk deletion."""
    success: bool
    results: Dict[str, Dict[str, int]]  # run_id -> deleted_counts
    total_deleted: int
    failed_runs: List[str]
    message: str


class DeleteRunResponse(BaseModel):
    """Response from run deletion."""
    success: bool
    run_id: str
    deleted_counts: Dict[str, int]
    message: str
    total_deleted: int


class PipelineRunInfo(BaseModel):
    """Information about a pipeline run."""
    run_id: str
    created_at: str
    tickers_processed: int
    signals_generated: int
    success_rate: float


class RunListResponse(BaseModel):
    """Response containing list of runs."""
    runs: List[PipelineRunInfo]
    total_count: int


# Tables to delete from (in order - child tables first, parent tables last)
DELETION_TABLES = [
    # Signal detail tables (reference signals.id via signal_id)
    "signals_technical",
    "signals_fundamental", 
    "signals_news_macro",
    "signals_social_alternative",
    "signals_risk_stability",
    "signals_institutional_smart_money",
    
    # Performance table (references signals.id via signal_id)
    "performance",
    
    # Main signals table (references signal_runs.id via run_id)
    "signals",
    
    # Parent table - delete last
    "signal_runs",
]


async def get_run_deletion_preview(run_id: str) -> Dict[str, int]:
    """
    Get counts of records that would be deleted for a given run_id.
    
    Args:
        run_id: The pipeline run ID to preview deletion for
        
    Returns:
        Dictionary mapping table names to record counts
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        counts = {}
        
        # First, get all signal IDs for this run
        signals_result = db.client.table("signals") \
            .select("id") \
            .eq("run_id", run_id) \
            .execute()
        
        signal_ids = [s['id'] for s in signals_result.data] if signals_result.data else []
        
        if not signal_ids:
            logger.warning(f"No signals found for run_id {run_id}")
            return counts
        
        # Count records in detail tables (they reference signal_id)
        detail_tables = [
            "signals_technical",
            "signals_fundamental",
            "signals_news_macro",
            "signals_social_alternative",
            "signals_risk_stability",
            "signals_institutional_smart_money",
            "performance"
        ]
        
        for table in detail_tables:
            try:
                result = db.client.table(table) \
                    .select("*", count="exact") \
                    .in_("signal_id", signal_ids) \
                    .execute()
                
                count = result.count if hasattr(result, 'count') else 0
                if count > 0:
                    counts[table] = count
                    
            except Exception as e:
                logger.debug(f"Could not count records in {table}: {e}")
                continue
        
        # Count signals (use run_id)
        if len(signal_ids) > 0:
            counts["signals"] = len(signal_ids)
        
        # Count signal_runs (use id, not run_id)
        try:
            runs_result = db.client.table("signal_runs") \
                .select("*", count="exact") \
                .eq("id", run_id) \
                .execute()
            
            if runs_result.count and runs_result.count > 0:
                counts["signal_runs"] = runs_result.count
        except Exception as e:
            logger.debug(f"Could not count signal_runs: {e}")
        
        return counts
        
    finally:
        await db.disconnect()


async def delete_run_cascade(run_id: str) -> Dict[str, int]:
    """
    Delete a pipeline run and ALL associated records across all tables.
    
    Deletes in proper order (child tables first) to avoid foreign key violations.
    
    Args:
        run_id: The pipeline run ID to delete
        
    Returns:
        Dictionary mapping table names to counts of deleted records
        
    Raises:
        Exception: If deletion fails
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        deleted_counts = {}
        
        # First, get all signal IDs for this run
        signals_result = db.client.table("signals") \
            .select("id") \
            .eq("run_id", run_id) \
            .execute()
        
        signal_ids = [s['id'] for s in signals_result.data] if signals_result.data else []
        
        if not signal_ids:
            logger.warning(f"No signals found for run_id {run_id}")
            # Still try to delete the run itself
        else:
            # Delete from detail tables (they reference signal_id)
            detail_tables = [
                "signals_technical",
                "signals_fundamental",
                "signals_news_macro",
                "signals_social_alternative",
                "signals_risk_stability",
                "signals_institutional_smart_money",
                "performance"
            ]
            
            for table in detail_tables:
                try:
                    result = db.client.table(table) \
                        .delete() \
                        .in_("signal_id", signal_ids) \
                        .execute()
                    
                    count = len(result.data) if result.data else 0
                    if count > 0:
                        deleted_counts[table] = count
                        logger.info(f"Deleted {count} records from {table}")
                        
                except Exception as e:
                    logger.warning(f"Could not delete from {table}: {e}")
                    continue
        
        # Delete from signals table (uses run_id)
        try:
            result = db.client.table("signals") \
                .delete() \
                .eq("run_id", run_id) \
                .execute()
            
            count = len(result.data) if result.data else 0
            if count > 0:
                deleted_counts["signals"] = count
                logger.info(f"Deleted {count} signals")
        except Exception as e:
            logger.warning(f"Could not delete signals: {e}")
        
        # Delete from signal_runs table (uses id, not run_id)
        try:
            result = db.client.table("signal_runs") \
                .delete() \
                .eq("id", run_id) \
                .execute()
            
            count = len(result.data) if result.data else 0
            if count > 0:
                deleted_counts["signal_runs"] = count
                logger.info(f"Deleted signal_run record")
        except Exception as e:
            logger.warning(f"Could not delete signal_run: {e}")
        
        return deleted_counts
        
    except Exception as e:
        logger.error(f"Failed to delete run {run_id}: {e}")
        raise
        
    finally:
        await db.disconnect()


@router.post("/runs/delete", response_model=DeleteRunResponse)
async def delete_pipeline_run(
    request: DeleteRunRequest,
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Delete a pipeline run and all associated data.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    
    Two modes:
    1. Preview (confirm=False): Returns counts of what would be deleted
    2. Delete (confirm=True): Actually performs the deletion
    
    Example usage:
    ```
    # Preview
    POST /api/admin/runs/delete
    { "run_id": "abc123", "confirm": false }
    
    # Actual deletion
    POST /api/admin/runs/delete
    { "run_id": "abc123", "confirm": true }
    ```
    """
    
    if not request.confirm:
        # Preview mode - just count records
        try:
            counts = await get_run_deletion_preview(request.run_id)
            total = sum(counts.values())
            
            return DeleteRunResponse(
                success=False,
                run_id=request.run_id,
                deleted_counts=counts,
                total_deleted=0,
                message=f"Preview: Would delete {total} records across {len(counts)} tables. Set confirm=True to proceed."
            )
            
        except Exception as e:
            logger.error(f"Preview failed for run {request.run_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
    
    # Confirm mode - perform actual deletion
    try:
        deleted_counts = await delete_run_cascade(request.run_id)
        total = sum(deleted_counts.values())
        
        # Log deletion for audit trail
        logger.warning(
            f"[ADMIN DELETE] Pipeline run deleted by {current_user.username}",
            extra={
                "run_id": request.run_id,
                "deleted_counts": deleted_counts,
                "total_deleted": total,
                "timestamp": datetime.now().isoformat(),
                "user": current_user.username,
                "role": current_user.role
            }
        )
        
        return DeleteRunResponse(
            success=True,
            run_id=request.run_id,
            deleted_counts=deleted_counts,
            total_deleted=total,
            message=f"Successfully deleted run {request.run_id[:8]}... and {total} associated records from {len(deleted_counts)} tables"
        )
        
    except Exception as e:
        logger.error(f"[ADMIN DELETE FAILED] Run {request.run_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Deletion failed: {str(e)}"
        )


@router.post("/runs/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_pipeline_runs(
    request: BulkDeleteRequest,
    # Temporarily disabled auth for development
    # current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Delete multiple pipeline runs at once.
    
    DEV MODE: Authentication temporarily disabled for development
    """
    if not request.confirm:
        # Preview mode
        all_counts = {}
        total = 0
        
        for run_id in request.run_ids:
            try:
                counts = await get_run_deletion_preview(run_id)
                all_counts[run_id] = counts
                total += sum(counts.values())
            except Exception as e:
                logger.error(f"Preview failed for run {run_id}: {e}")
                all_counts[run_id] = {}
        
        return BulkDeleteResponse(
            success=False,
            results=all_counts,
            total_deleted=0,
            failed_runs=[],
            message=f"Preview: Would delete {total} total records from {len(request.run_ids)} runs."
        )
    
    # Confirm mode - delete all runs
    results = {}
    failed_runs = []
    total = 0
    
    for run_id in request.run_ids:
        try:
            deleted_counts = await delete_run_cascade(run_id)
            results[run_id] = deleted_counts
            total += sum(deleted_counts.values())
        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {e}")
            failed_runs.append(run_id)
            results[run_id] = {}
    
    logger.warning(
        f"[ADMIN BULK DELETE] {len(results) - len(failed_runs)} runs deleted by dev",
        extra={"run_ids": request.run_ids, "total_deleted": total, "user": "dev"}
    )
    
    return BulkDeleteResponse(
        success=len(failed_runs) == 0,
        results=results,
        total_deleted=total,
        failed_runs=failed_runs,
        message=f"Deleted {len(results) - len(failed_runs)}/{len(request.run_ids)} runs. Total: {total} records."
    )


@router.get("/runs/list", response_model=RunListResponse)
async def list_pipeline_runs(
    # Temporarily disabled auth for development
    # current_user: Annotated[User, Depends(get_current_admin_user)],
    limit: int = 50,
    offset: int = 0
):
    """
    List recent pipeline runs with summary information.
    
    DEV MODE: Authentication temporarily disabled for development
    Returns runs ordered by creation date (newest first).
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get total count
        count_result = db.client.table("signal_runs") \
            .select("*", count="exact") \
            .execute()
        total_count = count_result.count if hasattr(count_result, 'count') else 0
        
        # Get paginated results
        result = db.client.table("signal_runs") \
            .select("id, run_timestamp, total_tickers, successful_tickers, status") \
            .order("run_timestamp", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        
        runs = []
        for run in result.data:
            # Calculate success rate based on successful vs total tickers
            total = run.get('total_tickers', 0)
            successful = run.get('successful_tickers', 0)
            success_rate = (successful / total) if total > 0 else 0.0
            
            runs.append(PipelineRunInfo(
                run_id=run['id'],
                created_at=run['run_timestamp'],
                tickers_processed=total,
                signals_generated=successful,
                success_rate=success_rate
            ))
        
        return RunListResponse(
            runs=runs,
            total_count=total_count
        )
        
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")
        
    finally:
        await db.disconnect()


@router.get("/runs/{run_id}")
async def get_run_details(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_admin_user)]
):
    """
    Get detailed information about a specific pipeline run.
    
    Requires: Authorization: Bearer <JWT token> with admin role
    Includes record counts across all tables.
    """
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        # Get run info
        result = db.client.table("signal_runs") \
            .select("*") \
            .eq("id", run_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        run_info = result.data[0]
        
        # Get record counts across all tables
        counts = await get_run_deletion_preview(run_id)
        
        return {
            "run": run_info,
            "record_counts": counts,
            "total_records": sum(counts.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run details for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get run details: {str(e)}")
        
    finally:
        await db.disconnect()
