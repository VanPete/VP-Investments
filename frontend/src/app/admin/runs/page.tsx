'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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
import { Badge } from '@/components/ui/badge';
import { Trash2, AlertTriangle, Database, Loader2, RefreshCw, CheckSquare, Square } from 'lucide-react';
import { toast } from 'sonner';

interface PipelineRun {
  run_id: string;
  created_at: string;
  tickers_processed: number;
  signals_generated: number;
  success_rate: number;
}

interface DeletionPreview {
  [table: string]: number;
}

interface BulkDeleteResponse {
  success: boolean;
  results: { [run_id: string]: DeletionPreview };
  total_deleted: number;
  failed_runs: string[];
  message: string;
}

export default function AdminRunsPage() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<{ [run_id: string]: DeletionPreview } | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [totalCount, setTotalCount] = useState(0);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('admin_token');
      const response = await fetch('http://127.0.0.1:8000/api/admin/runs/list', {
        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
      });
      
      if (response.status === 401) {
        toast.error('Session expired. Please login again.');
        window.location.href = '/admin/login';
        return;
      }
      
      if (!response.ok) {
        throw new Error(`Failed to fetch runs: ${response.statusText}`);
      }
      
      const data = await response.json();
      setRuns(data.runs || []);
      setTotalCount(data.total_count || 0);
    } catch (error) {
      console.error('Error fetching runs:', error);
      toast.error('Failed to load pipeline runs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const toggleSelection = (runId: string) => {
    const newSelected = new Set(selectedRunIds);
    if (newSelected.has(runId)) {
      newSelected.delete(runId);
    } else {
      newSelected.add(runId);
    }
    setSelectedRunIds(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedRunIds.size === runs.length) {
      setSelectedRunIds(new Set());
    } else {
      setSelectedRunIds(new Set(runs.map(r => r.run_id)));
    }
  };

  const handleBulkDeleteClick = async () => {
    if (selectedRunIds.size === 0) return;
    
    try {
      const token = localStorage.getItem('admin_token');
      const response = await fetch('http://127.0.0.1:8000/api/admin/runs/bulk-delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify({
          run_ids: Array.from(selectedRunIds),
          confirm: false
        })
      });
      
      if (response.status === 401) {
        toast.error('Session expired. Please login again.');
        window.location.href = '/admin/login';
        return;
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch bulk deletion preview');
      }
      
      const data: BulkDeleteResponse = await response.json();
      setPreview(data.results);
      setShowConfirm(true);
    } catch (error) {
      console.error('Error fetching bulk preview:', error);
      toast.error('Failed to load deletion preview');
    }
  };

  const handleConfirmBulkDelete = async () => {
    if (selectedRunIds.size === 0) return;
    
    setIsDeleting(true);
    
    try {
      const token = localStorage.getItem('admin_token');
      const response = await fetch('http://127.0.0.1:8000/api/admin/runs/bulk-delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify({
          run_ids: Array.from(selectedRunIds),
          confirm: true
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Bulk deletion failed');
      }
      
      const data: BulkDeleteResponse = await response.json();
      
      // Remove deleted runs from list
      setRuns(runs.filter(r => !selectedRunIds.has(r.run_id)));
      setTotalCount(prev => prev - selectedRunIds.size);
      setSelectedRunIds(new Set());
      
      // Show success toast
      toast.success(`Deleted ${data.total_deleted} records from ${selectedRunIds.size} runs`);
      
      if (data.failed_runs.length > 0) {
        toast.error(`Failed to delete ${data.failed_runs.length} runs`);
      }
    } catch (error) {
      console.error('Bulk delete failed:', error);
      toast.error(error instanceof Error ? error.message : 'Bulk deletion failed');
    } finally {
      setIsDeleting(false);
      setShowConfirm(false);
      setPreview(null);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
    } catch {
      return dateString;
    }
  };

  const getSuccessRateColor = (rate: number) => {
    if (rate >= 0.9) return 'bg-green-500';
    if (rate >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getTotalPreviewCount = () => {
    if (!preview) return 0;
    return Object.values(preview).reduce((total, counts) => {
      return total + Object.values(counts).reduce((sum, count) => sum + count, 0);
    }, 0);
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Pipeline Run Management</h1>
            <p className="text-muted-foreground mt-2">
              Manage and delete pipeline runs with all associated data
            </p>
          </div>
          <div className="flex gap-2">
            {selectedRunIds.size > 0 && (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleBulkDeleteClick}
                disabled={isDeleting}
                className="hover:bg-destructive/90 transition-colors cursor-pointer"
              >
                {isDeleting ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4 mr-2" />
                )}
                Delete {selectedRunIds.size} Selected
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={fetchRuns}
              disabled={loading}
              className="hover:bg-accent transition-colors cursor-pointer"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-4 mt-6 text-sm">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-muted-foreground" />
            <span className="font-medium">{totalCount}</span>
            <span className="text-muted-foreground">total runs</span>
          </div>
          <div className="text-muted-foreground">•</div>
          <div className="text-muted-foreground">
            Showing {runs.length} most recent
          </div>
          {selectedRunIds.size > 0 && (
            <>
              <div className="text-muted-foreground">•</div>
              <div className="text-primary font-medium">
                {selectedRunIds.size} selected
              </div>
            </>
          )}
        </div>

        {runs.length > 0 && (
          <div className="mt-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleSelectAll}
              className="text-sm"
            >
              {selectedRunIds.size === runs.length ? (
                <>
                  <CheckSquare className="w-4 h-4 mr-2" />
                  Deselect All
                </>
              ) : (
                <>
                  <Square className="w-4 h-4 mr-2" />
                  Select All
                </>
              )}
            </Button>
          </div>
        )}
      </div>

      {runs.length === 0 ? (
        <Card>
          <CardContent className="pt-6 pb-6 text-center text-muted-foreground">
            No pipeline runs found
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {runs.map((run) => (
            <Card 
              key={run.run_id} 
              className={`hover:shadow-md transition-all cursor-pointer ${
                selectedRunIds.has(run.run_id) ? 'ring-2 ring-primary' : ''
              }`}
              onClick={(e) => {
                if ((e.target as HTMLElement).closest('button')) return;
                toggleSelection(run.run_id);
              }}
            >
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div className="flex items-start gap-3 flex-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleSelection(run.run_id);
                      }}
                      className="mt-1"
                    >
                      {selectedRunIds.has(run.run_id) ? (
                        <CheckSquare className="w-5 h-5 text-primary" />
                      ) : (
                        <Square className="w-5 h-5 text-muted-foreground" />
                      )}
                    </button>
                    <div className="flex-1">
                      <CardTitle className="flex items-center gap-3">
                        <code className="text-sm bg-muted px-2 py-1 rounded">
                          {run.run_id.slice(0, 8)}...
                        </code>
                        <Badge 
                          className={`${getSuccessRateColor(run.success_rate)} text-white`}
                          variant="outline"
                        >
                          {(run.success_rate * 100).toFixed(1)}% success
                        </Badge>
                      </CardTitle>
                      <CardDescription className="mt-2">
                        {formatDate(run.created_at)}
                      </CardDescription>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="space-y-1">
                    <p className="text-muted-foreground">Tickers</p>
                    <p className="text-2xl font-bold">{run.tickers_processed}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-muted-foreground">Signals</p>
                    <p className="text-2xl font-bold">{run.signals_generated}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-muted-foreground">Success Rate</p>
                    <p className="text-2xl font-bold">
                      {(run.success_rate * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-muted-foreground">Run ID</p>
                    <p className="text-xs font-mono text-muted-foreground break-all">
                      {run.run_id}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Bulk Deletion Confirmation Dialog */}
      <AlertDialog open={showConfirm} onOpenChange={setShowConfirm}>
        <AlertDialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="w-5 h-5" />
              Confirm Bulk Deletion
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-4">
                <p>
                  This will permanently delete <strong>{selectedRunIds.size} pipeline runs</strong> and ALL associated data:
                </p>
                
                {preview && (
                  <div className="border rounded-lg overflow-hidden">
                    <div className="bg-muted px-4 py-2 border-b">
                      <p className="text-sm font-medium">Total records to be deleted: {getTotalPreviewCount().toLocaleString()}</p>
                    </div>
                    <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
                      {Object.entries(preview).map(([runId, counts]) => (
                        <div key={runId} className="border rounded-lg p-3">
                          <p className="text-xs font-mono text-muted-foreground mb-2">
                            {runId.slice(0, 16)}...
                          </p>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            {Object.entries(counts)
                              .sort((a, b) => b[1] - a[1])
                              .map(([table, count]) => (
                                <div key={table} className="flex justify-between items-center">
                                  <span className="text-muted-foreground text-xs">{table}</span>
                                  <span className="font-mono text-xs">{count.toLocaleString()}</span>
                                </div>
                              ))}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="border-t p-4 bg-muted">
                      <div className="flex justify-between items-center font-semibold">
                        <span>Grand Total</span>
                        <span className="text-lg text-destructive">
                          {getTotalPreviewCount().toLocaleString()} records
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                  <p className="font-semibold text-destructive flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    This action cannot be undone!
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    All signals, performance data, and analytics for these {selectedRunIds.size} runs will be permanently removed from the database.
                  </p>
                </div>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmBulkDelete}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Deleting {selectedRunIds.size} Runs...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete {selectedRunIds.size} Runs Permanently
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
