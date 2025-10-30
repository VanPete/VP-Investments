"""
Dual Logging Configuration for VP Investments Pipeline

Provides:
- Console Handler: Clean output (WARNING+ by default, configurable)
- File Handler: Full detail (DEBUG always)
- Hierarchical logging support
- Integration with Rich progress display
- Error Buffer Handler: Captures errors for end-of-run summary
"""

import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(verbose_level: int = 0, log_dir: str = "logs", quiet: bool = False) -> logging.Logger:
    """
    Set up dual logging handlers for pipeline.
    
    Console verbosity levels:
        0 (default): ERROR only (for clean progress bars)
        1 (-v): INFO and above
        2 (-vv): DEBUG (everything)
        quiet: Suppress all console output except errors
        
    File logging: Always DEBUG (full detail preserved)
    
    Args:
        verbose_level: Console verbosity (0, 1, or 2)
        log_dir: Directory for log files
        quiet: If True, only show errors on console
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if needed
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Determine console log level
    # When showing progress bars (default), only show errors to keep display clean
    console_levels = {
        0: logging.ERROR,    # Default: Only errors (for clean progress bars)
        1: logging.INFO,     # -v: Include info
        2: logging.DEBUG     # -vv: Everything
    }
    console_level = console_levels.get(verbose_level, logging.ERROR)
    
    # If quiet mode, only show critical errors
    if quiet:
        console_level = logging.CRITICAL
    
    # Get root logger
    logger = logging.getLogger("vp_investments")
    logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # === Console Handler (Clean) ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Simple format for console (no timestamp, just message)
    console_format = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # === File Handler (Full Detail) ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pipeline_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture everything
    
    # Detailed format for file
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log configuration info
    logger.debug(f"Logging initialized - Console: {logging.getLevelName(console_level)}, File: DEBUG")
    logger.debug(f"Log file: {log_file}")
    
    return logger


def get_phase_logger(phase_name: str, verbose_level: int = 0) -> logging.Logger:
    """
    Get a logger for a specific pipeline phase.
    
    Args:
        phase_name: Name of the phase (e.g., "phase1_fetch")
        verbose_level: Console verbosity level
        
    Returns:
        Logger configured for this phase
    """
    logger = logging.getLogger(f"vp_investments.{phase_name}")
    
    # Inherit configuration from root logger
    # No need to add handlers (they're inherited)
    
    return logger


class QuietLogger:
    """
    Context manager to temporarily suppress console output.
    
    Useful for --quiet mode where we only want errors.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.original_level = None
        
    def __enter__(self):
        """Suppress console logging (keep file logging)."""
        # Store original console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.original_level = handler.level
                handler.setLevel(logging.ERROR)  # Only errors
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore console logging level."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and self.original_level:
                handler.setLevel(self.original_level)
        return False


class ErrorCaptureHandler(logging.Handler):
    """
    Custom logging handler that captures ERROR/CRITICAL logs and adds them to ErrorBuffer.
    
    This allows us to suppress error output during pipeline execution and show
    a consolidated error summary at the end.
    
    The handler extracts context from the log message:
    - Phase: Detected from logger name (e.g., backend.phases.phase1_fetch → phase1)
    - Ticker: Extracted from message if present (e.g., "AAPL: Missing price history")
    - Additional context: Parsed from message where possible
    """
    
    def __init__(self, error_buffer, level=logging.ERROR):
        """
        Initialize the error capture handler.
        
        Args:
            error_buffer: ErrorBuffer instance to store captured errors
            level: Minimum logging level to capture (default: ERROR)
        """
        super().__init__(level)
        self.error_buffer = error_buffer
        self.ticker_pattern = re.compile(r'^([A-Z]{1,5})[:|\s]')  # Match ticker at start of message
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Capture error log record and add to buffer.
        
        Args:
            record: LogRecord from Python logging system
        """
        try:
            # Extract phase from logger name (e.g., backend.phases.phase1_fetch → phase1)
            phase = "unknown"
            if "phase1" in record.name:
                phase = "phase1"
            elif "phase2" in record.name:
                phase = "phase2"
            elif "phase3" in record.name:
                phase = "phase3"
            elif "phase4" in record.name:
                phase = "phase4"
            elif "phase5" in record.name:
                phase = "phase5"
            elif "phase6" in record.name:
                phase = "phase6"
            elif "phase7" in record.name:
                phase = "phase7"
            
            # Try to extract ticker from message
            ticker = None
            message = self.format(record)
            ticker_match = self.ticker_pattern.match(message)
            if ticker_match:
                ticker = ticker_match.group(1)
                # Remove ticker prefix from message
                message = message[len(ticker) + 1:].strip()
            
            # Add to error buffer
            self.error_buffer.add_error(
                message=message,
                phase=phase,
                ticker=ticker,
                level=record.levelname,
                logger_name=record.name
            )
            
        except Exception:
            # Don't let errors in error handling break the pipeline
            self.handleError(record)


def configure_pipeline_logging(verbose: int = 0, quiet: bool = False, error_buffer=None) -> logging.Logger:
    """
    Convenience function to configure logging for pipeline execution.
    
    Args:
        verbose: Verbosity level (0, 1, or 2)
        quiet: If True, suppress all console output except errors
        error_buffer: Optional ErrorBuffer to capture errors for end-of-run summary
        
    Returns:
        Configured root logger
    """
    # Determine the console level based on verbosity
    console_levels = {
        0: logging.ERROR,    # Default: Only errors (for clean progress bars)
        1: logging.INFO,     # -v: Include info
        2: logging.DEBUG     # -vv: Everything
    }
    console_level = console_levels.get(verbose, logging.ERROR)
    
    # If quiet mode, only show critical errors
    if quiet:
        console_level = logging.CRITICAL
    
    # Reconfigure the ROOT logger (which was already initialized by backend.utils.logger)
    root_logger = logging.getLogger()
    
    # If error_buffer provided, add ErrorCaptureHandler and suppress console errors
    if error_buffer:
        # Add error capture handler
        error_handler = ErrorCaptureHandler(error_buffer, level=logging.ERROR)
        error_handler.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(error_handler)
        
        # Suppress ERROR logs from console (they'll be shown in summary)
        # But keep file logging at DEBUG level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                # Set console to CRITICAL so ERROR logs don't print (they go to buffer instead)
                handler.setLevel(logging.CRITICAL)
    else:
        # Normal mode: Update all existing StreamHandlers to use the new console level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(console_level)
    
    return root_logger
