"""
Error Buffer System for Pipeline Error Consolidation
====================================================

Captures ERROR/CRITICAL logs during pipeline execution and displays them
in a consolidated summary panel at the end instead of interrupting progress bars.
"""

import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ErrorRecord:
    """Single error record with full context."""
    timestamp: datetime
    phase: str
    ticker: Optional[str]
    message: str
    level: str  # 'ERROR' or 'CRITICAL'
    logger_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format error record for display."""
        parts = [f"[{self.level}]"]
        if self.ticker:
            parts.append(f"{self.ticker}:")
        parts.append(self.message)
        return " ".join(parts)


class ErrorBuffer:
    """
    Thread-safe buffer for capturing pipeline errors.
    
    Usage:
        buffer = ErrorBuffer()
        buffer.add_error(phase="phase1", message="Failed to fetch data", ticker="AAPL")
        
        # At end of pipeline
        if buffer.has_errors():
            for error in buffer.get_errors():
                print(error)
    """
    
    def __init__(self):
        self._errors: List[ErrorRecord] = []
        self._lock = threading.Lock()
    
    def add_error(
        self,
        message: str,
        phase: str = "unknown",
        ticker: Optional[str] = None,
        level: str = "ERROR",
        logger_name: str = "",
        **context
    ) -> None:
        """
        Add an error to the buffer.
        
        Args:
            message: Error message
            phase: Pipeline phase (phase1, phase2, etc.)
            ticker: Stock ticker if applicable
            level: Error severity ('ERROR' or 'CRITICAL')
            logger_name: Name of the logger that generated the error
            **context: Additional context (e.g., endpoint='price_history', coverage=0.3)
        """
        with self._lock:
            self._errors.append(ErrorRecord(
                timestamp=datetime.now(),
                phase=phase,
                ticker=ticker,
                message=message,
                level=level,
                logger_name=logger_name,
                context=context
            ))
    
    def get_errors(self, phase: Optional[str] = None) -> List[ErrorRecord]:
        """
        Get all errors, optionally filtered by phase.
        
        Args:
            phase: If provided, only return errors from this phase
            
        Returns:
            List of ErrorRecord objects
        """
        with self._lock:
            if phase:
                return [e for e in self._errors if e.phase == phase]
            return self._errors.copy()
    
    def count(self, phase: Optional[str] = None) -> int:
        """
        Count total errors, optionally filtered by phase.
        
        Args:
            phase: If provided, only count errors from this phase
            
        Returns:
            Number of errors
        """
        with self._lock:
            if phase:
                return sum(1 for e in self._errors if e.phase == phase)
            return len(self._errors)
    
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        with self._lock:
            return len(self._errors) > 0
    
    def clear(self) -> None:
        """Clear all errors from buffer."""
        with self._lock:
            self._errors.clear()
    
    def group_by_phase(self) -> Dict[str, List[ErrorRecord]]:
        """Group errors by pipeline phase."""
        with self._lock:
            grouped = {}
            for error in self._errors:
                if error.phase not in grouped:
                    grouped[error.phase] = []
                grouped[error.phase].append(error)
            return grouped
    
    def group_by_ticker(self) -> Dict[str, List[ErrorRecord]]:
        """Group errors by ticker (None for non-ticker errors)."""
        with self._lock:
            grouped = {}
            for error in self._errors:
                key = error.ticker or "general"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(error)
            return grouped
