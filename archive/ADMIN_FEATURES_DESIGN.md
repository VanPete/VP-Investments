# Admin Features Design - vanpiq.com/admin
**Date:** October 30, 2025

---

## Overview

Create a comprehensive admin dashboard at `vanpiq.com/admin` for monitoring and managing the VP Investments pipeline system.

---

## Feature 1: Delete Pipeline Runs

### Requirements
- Delete all data associated with a specific `run_id`
- Cascade deletion across all related tables
- Provide confirmation dialog before deletion
- Show what will be deleted (record counts per table)
- Log deletion actions for audit trail

### Database Tables Affected

**Primary Tables:**
1. `pipeline_runs` - Main run record
2. `pipeline_signals` - Signal data for each ticker
3. `pipeline_performance` - Performance tracking records
4. `pipeline_analytics` - Analytics calculations
5. `raw_yfinance_cache` (optional) - Raw data cache (may want to keep for analysis)

**Deletion Strategy:**

```sql
-- Option 1: CASCADE DELETE (if foreign keys configured)
DELETE FROM pipeline_runs WHERE run_id = ?;

-- Option 2: MANUAL CASCADE (safer, gives feedback)
BEGIN TRANSACTION;

-- Count records before deletion
SELECT 
  (SELECT COUNT(*) FROM pipeline_signals WHERE run_id = ?) as signals_count,
  (SELECT COUNT(*) FROM pipeline_performance WHERE run_id = ?) as performance_count,
  (SELECT COUNT(*) FROM pipeline_analytics WHERE run_id = ?) as analytics_count;

-- Delete in order (child tables first)
DELETE FROM pipeline_analytics WHERE run_id = ?;
DELETE FROM pipeline_performance WHERE run_id = ?;
DELETE FROM pipeline_signals WHERE run_id = ?;
DELETE FROM pipeline_runs WHERE run_id = ?;

COMMIT;
```

### API Endpoint Design

```python
# backend/api/admin.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional
import logging

router = APIRouter(prefix="/api/admin", tags=["admin"])

class DeleteRunRequest(BaseModel):
    run_id: str
    confirm: bool = False  # Must be True to actually delete

class DeleteRunResponse(BaseModel):
    success: bool
    run_id: str
    deleted_counts: Dict[str, int]
    message: str

@router.post("/runs/delete", response_model=DeleteRunResponse)
async def delete_pipeline_run(
    request: DeleteRunRequest,
    # TODO: Add authentication/authorization
    # current_user: User = Depends(get_current_admin_user)
):
    """
    Delete a pipeline run and all associated data.
    
    Requires confirm=True to actually perform deletion.
    If confirm=False, returns preview of what would be deleted.
    """
    
    if not request.confirm:
        # Preview mode - just count records
        counts = await get_run_deletion_preview(request.run_id)
        return DeleteRunResponse(
            success=False,
            run_id=request.run_id,
            deleted_counts=counts,
            message="Preview only - set confirm=True to delete"
        )
    
    # Perform actual deletion
    try:
        deleted_counts = await delete_run_cascade(request.run_id)
        
        # Log deletion for audit trail
        logging.warning(
            f"[ADMIN] Pipeline run deleted: {request.run_id}",
            extra={
                "run_id": request.run_id,
                "deleted_counts": deleted_counts,
                # "user": current_user.email  # When auth implemented
            }
        )
        
        return DeleteRunResponse(
            success=True,
            run_id=request.run_id,
            deleted_counts=deleted_counts,
            message=f"Successfully deleted run {request.run_id} and {sum(deleted_counts.values())} associated records"
        )
        
    except Exception as e:
        logging.error(f"[ADMIN] Failed to delete run {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_run_deletion_preview(run_id: str) -> Dict[str, int]:
    """Get counts of records that would be deleted."""
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        counts = {}
        
        # Count signals
        result = await db.client.table("pipeline_signals") \
            .select("*", count="exact") \
            .eq("run_id", run_id) \
            .execute()
        counts["signals"] = result.count
        
        # Count performance records
        result = await db.client.table("pipeline_performance") \
            .select("*", count="exact") \
            .eq("run_id", run_id) \
            .execute()
        counts["performance"] = result.count
        
        # Count analytics
        result = await db.client.table("pipeline_analytics") \
            .select("*", count="exact") \
            .eq("run_id", run_id) \
            .execute()
        counts["analytics"] = result.count
        
        return counts
        
    finally:
        await db.disconnect()

async def delete_run_cascade(run_id: str) -> Dict[str, int]:
    """Delete run and all associated records. Returns counts of deleted records."""
    from backend.storage.database import get_supabase_database
    
    db = await get_supabase_database()
    await db.connect()
    
    try:
        deleted_counts = {}
        
        # Delete analytics first (no foreign dependencies)
        result = await db.client.table("pipeline_analytics") \
            .delete() \
            .eq("run_id", run_id) \
            .execute()
        deleted_counts["analytics"] = len(result.data) if result.data else 0
        
        # Delete performance records
        result = await db.client.table("pipeline_performance") \
            .delete() \
            .eq("run_id", run_id) \
            .execute()
        deleted_counts["performance"] = len(result.data) if result.data else 0
        
        # Delete signals
        result = await db.client.table("pipeline_signals") \
            .delete() \
            .eq("run_id", run_id) \
            .execute()
        deleted_counts["signals"] = len(result.data) if result.data else 0
        
        # Finally delete the run itself
        result = await db.client.table("pipeline_runs") \
            .delete() \
            .eq("run_id", run_id) \
            .execute()
        deleted_counts["runs"] = len(result.data) if result.data else 0
        
        return deleted_counts
        
    finally:
        await db.disconnect()
```

---

## Feature 2: Admin Dashboard UI

### Page Structure: `/admin`

```
/admin
  ├── /runs          - Manage pipeline runs (list, view, delete)
  ├── /monitoring    - Real-time system health
  ├── /logs          - Browse error logs and factor monitoring
  ├── /cache         - Manage YFinance cache
  └── /settings      - System configuration
```

### Component: Run Management

```typescript
// frontend/src/app/admin/runs/page.tsx

'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Trash2, AlertTriangle } from 'lucide-react';

interface PipelineRun {
  run_id: string;
  created_at: string;
  tickers_processed: number;
  signals_generated: number;
  success_rate: number;
}

interface DeletionPreview {
  signals: number;
  performance: number;
  analytics: number;
}

export default function RunManagementPage() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [preview, setPreview] = useState<DeletionPreview | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    // TODO: Implement API call
    const response = await fetch('/api/admin/runs');
    const data = await response.json();
    setRuns(data.runs);
  };

  const handleDeleteClick = async (runId: string) => {
    setSelectedRun(runId);
    
    // Fetch preview
    const response = await fetch('/api/admin/runs/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId, confirm: false })
    });
    
    const data = await response.json();
    setPreview(data.deleted_counts);
    setShowConfirm(true);
  };

  const handleConfirmDelete = async () => {
    if (!selectedRun) return;
    
    setIsDeleting(true);
    
    try {
      const response = await fetch('/api/admin/runs/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: selectedRun, confirm: true })
      });
      
      if (response.ok) {
        // Remove from list
        setRuns(runs.filter(r => r.run_id !== selectedRun));
        // Show success toast
        console.log('Run deleted successfully');
      }
    } catch (error) {
      console.error('Delete failed:', error);
    } finally {
      setIsDeleting(false);
      setShowConfirm(false);
      setSelectedRun(null);
      setPreview(null);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Pipeline Run Management</h1>
      
      <div className="grid gap-4">
        {runs.map((run) => (
          <Card key={run.run_id}>
            <CardHeader>
              <CardTitle className="flex justify-between items-center">
                <span>Run {run.run_id.slice(0, 8)}</span>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDeleteClick(run.run_id)}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Created</p>
                  <p className="font-medium">
                    {new Date(run.created_at).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">Tickers</p>
                  <p className="font-medium">{run.tickers_processed}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Signals</p>
                  <p className="font-medium">{run.signals_generated}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Success Rate</p>
                  <p className="font-medium">{(run.success_rate * 100).toFixed(1)}%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Confirmation Dialog */}
      <AlertDialog open={showConfirm} onOpenChange={setShowConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-destructive" />
              Confirm Deletion
            </AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete run <code>{selectedRun?.slice(0, 8)}</code> and all associated data:
              
              {preview && (
                <div className="mt-4 p-4 bg-muted rounded-md">
                  <ul className="space-y-2">
                    <li>• {preview.signals} signals</li>
                    <li>• {preview.performance} performance records</li>
                    <li>• {preview.analytics} analytics records</li>
                  </ul>
                  <p className="mt-4 font-semibold text-destructive">
                    Total: {preview.signals + preview.performance + preview.analytics} records
                  </p>
                </div>
              )}
              
              <p className="mt-4 font-semibold">This action cannot be undone.</p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmDelete}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? 'Deleting...' : 'Delete Permanently'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
```

---

## Feature 3: System Monitoring

### Real-time Metrics Dashboard

**Metrics to Display:**
1. **Pipeline Health**
   - Last run timestamp
   - Success rate (last 10 runs)
   - Average execution time
   - Current status (running/idle)

2. **Database Stats**
   - Total runs
   - Total signals
   - Database size
   - Growth rate

3. **Factor Quality**
   - Display latest factor monitoring JSON
   - Highlight problematic factors
   - Show success rate trends

4. **Error Tracking**
   - Recent errors (from logs)
   - Error frequency
   - Most common error types

---

## Feature 4: CLI Tool (Alternative/Complement)

```python
# scripts/admin_cli.py

import click
import asyncio
from backend.storage.database import get_supabase_database
from backend.api.admin import delete_run_cascade, get_run_deletion_preview
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()

@click.group()
def cli():
    """VP Investments Admin CLI"""
    pass

@cli.command()
@click.argument('run_id')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
async def delete_run(run_id: str, yes: bool):
    """Delete a pipeline run and all associated data."""
    
    console.print(f"\n[bold yellow]Analyzing run {run_id}...[/]")
    
    # Get preview
    try:
        preview = await get_run_deletion_preview(run_id)
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        return
    
    # Show what will be deleted
    table = Table(title="Records to Delete")
    table.add_column("Table", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    table.add_row("Signals", str(preview['signals']))
    table.add_row("Performance", str(preview['performance']))
    table.add_row("Analytics", str(preview['analytics']))
    table.add_row("Total", str(sum(preview.values())), style="bold")
    
    console.print(table)
    
    # Confirm
    if not yes:
        if not Confirm.ask(f"\n[bold red]Permanently delete {sum(preview.values())} records?[/]"):
            console.print("[yellow]Cancelled[/]")
            return
    
    # Delete
    console.print("\n[bold]Deleting...[/]")
    try:
        deleted = await delete_run_cascade(run_id)
        console.print(f"[bold green]✓ Successfully deleted {sum(deleted.values())} records[/]")
    except Exception as e:
        console.print(f"[bold red]✗ Delete failed:[/] {e}")

@cli.command()
async def list_runs():
    """List all pipeline runs."""
    db = await get_supabase_database()
    await db.connect()
    
    try:
        result = await db.client.table("pipeline_runs") \
            .select("run_id, created_at, tickers_processed, success_rate") \
            .order("created_at", desc=True) \
            .limit(20) \
            .execute()
        
        table = Table(title="Recent Pipeline Runs")
        table.add_column("Run ID", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Tickers", justify="right")
        table.add_column("Success", justify="right")
        
        for run in result.data:
            table.add_row(
                run['run_id'][:8],
                run['created_at'][:19],
                str(run.get('tickers_processed', 0)),
                f"{run.get('success_rate', 0)*100:.1f}%"
            )
        
        console.print(table)
        
    finally:
        await db.disconnect()

if __name__ == '__main__':
    cli()
```

---

## Implementation Priority

**Phase 1 (Essential):**
1. ✅ Backend API endpoints (`/api/admin/runs/delete`)
2. ✅ Database deletion functions (cascade delete)
3. ✅ Basic admin UI page (`/admin/runs`)
4. ✅ Deletion preview + confirmation dialog

**Phase 2 (Enhanced):**
5. CLI tool for command-line management
6. System monitoring dashboard
7. Error log browser
8. Authentication/authorization

**Phase 3 (Advanced):**
9. Real-time pipeline status
10. Cache management tools
11. Bulk operations (delete multiple runs)
12. Audit log viewer

---

## Security Considerations

1. **Authentication Required**
   - Implement admin user authentication
   - Use JWT tokens or session-based auth
   - Role-based access control (RBAC)

2. **Audit Logging**
   - Log all deletion actions
   - Track who deleted what and when
   - Store in separate audit table

3. **Rate Limiting**
   - Prevent abuse of delete endpoints
   - Limit to X deletions per hour

4. **Confirmation Required**
   - Always require explicit confirmation
   - Show preview before deletion
   - Implement "are you sure" dialogs

---

## Next Steps

Would you like me to:
1. **Implement the backend API** (`backend/api/admin.py`) with delete functionality?
2. **Create the admin page** (`frontend/src/app/admin/runs/page.tsx`)?
3. **Build the CLI tool** for terminal-based management?
4. **All of the above** in sequence?

I recommend starting with #1 (backend API) and #2 (admin UI) as they're the most essential features. The CLI can be added later as a convenience tool.
