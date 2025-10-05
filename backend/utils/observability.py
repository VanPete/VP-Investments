"""
VP Investments 2.0 - Observability Utilities

Provides performance monitoring, metrics collection, and observability features
for the VP Investments system.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Callable
from functools import wraps
import json

logger = logging.getLogger(__name__)

# Global metrics storage (in-memory for now, could be extended to external systems)
_metrics_store: Dict[str, Any] = {}
_performance_history: list = []


def emit_metric(metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None, **kwargs) -> None:
    """
    Emit a metric for monitoring and observability
    
    Args:
        metric_name: Name of the metric (e.g., "signal_engine.score_time")
        value: Numeric value of the metric (default 1.0)
        tags: Optional dictionary of tags for categorization
        **kwargs: Additional tags as keyword arguments
    """
    try:
        timestamp = datetime.now(timezone.utc)
        
        # Merge tags and kwargs
        all_tags = tags or {}
        all_tags.update(kwargs)
        
        metric_data = {
            'name': metric_name,
            'value': value,
            'tags': all_tags,
            'timestamp': timestamp.isoformat()
        }
        
        # Store in memory (could be enhanced to send to external systems)
        if metric_name not in _metrics_store:
            _metrics_store[metric_name] = []
        
        _metrics_store[metric_name].append(metric_data)
        
        # Keep only recent metrics to prevent memory bloat
        max_history = 1000
        if len(_metrics_store[metric_name]) > max_history:
            _metrics_store[metric_name] = _metrics_store[metric_name][-max_history:]
        
        logger.debug(f"[DATA] Metric emitted: {metric_name}={value} {tags or ''}")
        
    except Exception as e:
        logger.warning(f"[WARNING] Failed to emit metric {metric_name}: {e}")


def track_performance(operation_name: str):
    """
    Decorator to automatically track performance of functions/methods
    
    Usage:
        @track_performance("signal_generation")
        async def generate_signal(self, ticker):
            # function implementation
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Emit performance metrics
                await emit_metric(f"{operation_name}.duration", duration)
                await emit_metric(f"{operation_name}.success", 1 if success else 0)
                
                if error:
                    await emit_metric(f"{operation_name}.error", 1, {"error_type": type(error).__name__})
                
                # Store in performance history
                performance_record = {
                    'operation': operation_name,
                    'duration_ms': duration,
                    'success': success,
                    'error': error,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                _performance_history.append(performance_record)
                
                # Limit history size
                if len(_performance_history) > 1000:
                    _performance_history.pop(0)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # For sync functions, we can't await emit_metric, so store directly
                performance_record = {
                    'operation': operation_name,
                    'duration_ms': duration,
                    'success': success,
                    'error': error,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                _performance_history.append(performance_record)
                
                # Limit history size
                if len(_performance_history) > 1000:
                    _performance_history.pop(0)
            
            return result
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_metrics_summary(metric_name: Optional[str] = None, hours_back: int = 24) -> Dict[str, Any]:
    """
    Get summary statistics for metrics
    
    Args:
        metric_name: Specific metric name, or None for all metrics
        hours_back: Number of hours of history to include
        
    Returns:
        Dictionary containing metric statistics
    """
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        if metric_name and metric_name in _metrics_store:
            # Get specific metric
            metrics = _metrics_store[metric_name]
        else:
            # Get all metrics
            metrics = []
            for metric_list in _metrics_store.values():
                metrics.extend(metric_list)
        
        # Filter by time
        recent_metrics = [
            m for m in metrics 
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'count': 0, 'metrics': []}
        
        # Calculate summary statistics
        values = [m['value'] for m in recent_metrics]
        
        summary = {
            'count': len(recent_metrics),
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
            'avg': sum(values) / len(values) if values else 0,
            'total': sum(values) if values else 0,
            'latest': recent_metrics[-1] if recent_metrics else None
        }
        
        if metric_name:
            summary['metric_name'] = metric_name
        else:
            # Include breakdown by metric name
            by_name = {}
            for metric in recent_metrics:
                name = metric['name']
                if name not in by_name:
                    by_name[name] = []
                by_name[name].append(metric['value'])
            
            summary['by_name'] = {
                name: {
                    'count': len(values),
                    'avg': sum(values) / len(values) if values else 0,
                    'total': sum(values) if values else 0
                }
                for name, values in by_name.items()
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to get metrics summary: {e}")
        return {'error': str(e)}


def get_performance_summary(operation_name: Optional[str] = None, hours_back: int = 24) -> Dict[str, Any]:
    """
    Get performance summary for tracked operations
    
    Args:
        operation_name: Specific operation name, or None for all operations
        hours_back: Number of hours of history to include
        
    Returns:
        Dictionary containing performance statistics
    """
    try:
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        # Filter performance records by time and operation
        recent_records = []
        for record in _performance_history:
            if datetime.fromisoformat(record['timestamp']) >= cutoff_time:
                if operation_name is None or record['operation'] == operation_name:
                    recent_records.append(record)
        
        if not recent_records:
            return {'count': 0, 'operations': []}
        
        # Calculate statistics
        durations = [r['duration_ms'] for r in recent_records]
        successes = sum(1 for r in recent_records if r['success'])
        
        summary = {
            'count': len(recent_records),
            'success_rate': successes / len(recent_records) if recent_records else 0,
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'total_operations': len(recent_records),
            'successful_operations': successes,
            'failed_operations': len(recent_records) - successes
        }
        
        if operation_name:
            summary['operation_name'] = operation_name
        else:
            # Include breakdown by operation
            by_operation = {}
            for record in recent_records:
                op = record['operation']
                if op not in by_operation:
                    by_operation[op] = []
                by_operation[op].append(record)
            
            summary['by_operation'] = {}
            for op, records in by_operation.items():
                op_durations = [r['duration_ms'] for r in records]
                op_successes = sum(1 for r in records if r['success'])
                
                summary['by_operation'][op] = {
                    'count': len(records),
                    'success_rate': op_successes / len(records) if records else 0,
                    'avg_duration_ms': sum(op_durations) / len(op_durations) if op_durations else 0
                }
        
        return summary
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to get performance summary: {e}")
        return {'error': str(e)}


def export_metrics(format: str = 'json') -> str:
    """
    Export all metrics and performance data
    
    Args:
        format: Export format ('json' or 'csv')
        
    Returns:
        Formatted string containing all observability data
    """
    try:
        data = {
            'metrics': _metrics_store,
            'performance': _performance_history,
            'exported_at': datetime.now(timezone.utc).isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'csv':
            # Simple CSV export for performance data
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write performance data
            writer.writerow(['operation', 'duration_ms', 'success', 'error', 'timestamp'])
            for record in _performance_history:
                writer.writerow([
                    record['operation'],
                    record['duration_ms'],
                    record['success'],
                    record.get('error', ''),
                    record['timestamp']
                ])
            
            return output.getvalue()
        else:
            return json.dumps(data, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to export metrics: {e}")
        return f"Export failed: {e}"


def clear_metrics(older_than_hours: Optional[int] = None) -> int:
    """
    Clear metrics data, optionally keeping recent data
    
    Args:
        older_than_hours: If specified, only clear metrics older than this many hours
        
    Returns:
        Number of metric records cleared
    """
    try:
        cleared_count = 0
        
        if older_than_hours is None:
            # Clear everything
            cleared_count = sum(len(metrics) for metrics in _metrics_store.values())
            _metrics_store.clear()
            
            perf_count = len(_performance_history)
            _performance_history.clear()
            cleared_count += perf_count
            
        else:
            # Clear only old data
            from datetime import timedelta
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
            
            # Clear old metrics
            for metric_name, metrics in _metrics_store.items():
                original_count = len(metrics)
                _metrics_store[metric_name] = [
                    m for m in metrics 
                    if datetime.fromisoformat(m['timestamp']) >= cutoff_time
                ]
                cleared_count += original_count - len(_metrics_store[metric_name])
            
            # Clear old performance records
            original_perf_count = len(_performance_history)
            _performance_history[:] = [
                r for r in _performance_history
                if datetime.fromisoformat(r['timestamp']) >= cutoff_time
            ]
            cleared_count += original_perf_count - len(_performance_history)
        
        logger.info(f"[SUCCESS] Cleared {cleared_count} observability records")
        return cleared_count
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to clear metrics: {e}")
        return 0


# Export key functions
__all__ = [
    'emit_metric',
    'track_performance',
    'get_metrics_summary',
    'get_performance_summary',
    'export_metrics',
    'clear_metrics'
]