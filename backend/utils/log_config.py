"""
Dual Logging Configuration for VP Investments Pipeline

Provides:
- Console Handler: Clean output (WARNING+ by default, configurable)
- File Handler: Full detail (DEBUG always)
- Hierarchical logging support
- Integration with Rich progress display
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(verbose_level: int = 0, log_dir: str = "logs") -> logging.Logger:
    """
    Set up dual logging handlers for pipeline.
    
    Console verbosity levels:
        0 (default): WARNING and above only
        1 (-v): INFO and above
        2 (-vv): DEBUG (everything)
        
    File logging: Always DEBUG (full detail preserved)
    
    Args:
        verbose_level: Console verbosity (0, 1, or 2)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if needed
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Determine console log level
    console_levels = {
        0: logging.WARNING,  # Default: Only warnings/errors
        1: logging.INFO,     # -v: Include info
        2: logging.DEBUG     # -vv: Everything
    }
    console_level = console_levels.get(verbose_level, logging.WARNING)
    
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


def configure_pipeline_logging(verbose: int = 0, quiet: bool = False) -> logging.Logger:
    """
    Convenience function to configure logging for pipeline execution.
    
    Args:
        verbose: Verbosity level (0, 1, or 2)
        quiet: If True, suppress all console output except errors
        
    Returns:
        Configured root logger
    """
    # Quiet mode overrides verbose
    if quiet:
        verbose = -1  # Will map to ERROR level
        
    logger = setup_logging(verbose_level=verbose)
    
    if quiet:
        # Suppress console output except errors
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)
                
    return logger
