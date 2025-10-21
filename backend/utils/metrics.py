"""
Metrics emission utility for monitoring and observability.

This module provides a centralized way to emit metrics across the application.
Currently a stub implementation - can be extended with DataDog, Prometheus, etc.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def emit_metric(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
    metric_type: str = "gauge"
) -> None:
    """
    Emit a metric for monitoring/observability.
    
    This is a stub implementation that logs metrics. Can be extended to send
    metrics to DataDog, Prometheus, CloudWatch, etc.
    
    Args:
        metric_name: Name of the metric (e.g., 'api.request.duration')
        value: Numeric value of the metric
        tags: Optional dict of tags/labels for the metric
        metric_type: Type of metric ('gauge', 'counter', 'histogram')
        
    Example:
        emit_metric('api.request.duration', 0.142, 
                   tags={'endpoint': '/signals', 'status': 200},
                   metric_type='histogram')
    """
    tags = tags or {}
    timestamp = datetime.now().isoformat()
    
    # Format tags for logging
    tags_str = ", ".join([f"{k}={v}" for k, v in tags.items()])
    
    logger.debug(
        f"[METRIC] {metric_name}={value} type={metric_type} "
        f"tags=[{tags_str}] timestamp={timestamp}"
    )


def emit_counter(metric_name: str, value: int = 1, tags: Optional[Dict[str, Any]] = None) -> None:
    """Emit a counter metric (for counting events)."""
    emit_metric(metric_name, float(value), tags, metric_type="counter")


def emit_gauge(metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
    """Emit a gauge metric (for current value)."""
    emit_metric(metric_name, value, tags, metric_type="gauge")


def emit_histogram(metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
    """Emit a histogram metric (for distributions like latency)."""
    emit_metric(metric_name, value, tags, metric_type="histogram")
