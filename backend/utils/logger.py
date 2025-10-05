"""
VP Investments 2.0 - Enhanced Logging Utilities

Provides centralized logging configuration and utilities for the VP Investments system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import json


class VPInvestmentsFormatter(logging.Formatter):
    """Custom formatter for VP Investments with structured output"""
    
    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra
        
        # Color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green  
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m'       # Reset
        }
        
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Color for console
        color = self.colors.get(record.levelname, '')
        reset = self.colors['RESET']
        
        # Build base message with safe encoding for Windows
        try:
            message = record.getMessage()
            # On Windows, replace Unicode emojis with text equivalents to avoid encoding issues
            if sys.platform == 'win32':
                unicode_replacements = {
                    '✅': '[SUCCESS]',
                    '❌': '[ERROR]', 
                    '📋': '[DATA]',
                    '🎯': '[TARGET]',
                    '🚀': '[START]',
                    '⚠️': '[WARNING]',
                    '💰': '[FINANCE]',
                    '📈': '[GAIN]',
                    '📉': '[LOSS]',
                    '🔄': '[PROCESS]',
                    '🗄️': '[DATABASE]',
                    '🤖': '[AI]',
                    '📊': '[STATS]',
                    '🔌': '[CONNECT]',
                    '🧪': '[TEST]',
                    '🎉': '[COMPLETE]',
                    '⭐': '[SIGNAL]',
                    '💡': '[INSIGHT]',
                    '🔍': '[SEARCH]',
                    '📱': '[REDDIT]',
                    '🌟': '[HIGHLIGHT]'
                }
                for emoji, replacement in unicode_replacements.items():
                    message = message.replace(emoji, replacement)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback: Replace all non-ASCII characters with safe equivalents
            message = record.getMessage().encode('ascii', errors='replace').decode('ascii')
        
        base_msg = f"{timestamp} | {color}{record.levelname:8}{reset} | {record.name:20} | {message}"
        
        # Add extra information if available and enabled
        if self.include_extra and hasattr(record, '__dict__'):
            extra_fields = {}
            
            # Common extra fields to include
            for field in ['ticker', 'operation', 'duration_ms', 'status', 'error_type']:
                if hasattr(record, field):
                    extra_fields[field] = getattr(record, field)
            
            if extra_fields:
                extra_str = ' | '.join(f"{k}={v}" for k, v in extra_fields.items())
                base_msg += f" | {extra_str}"
        
        # Add exception info if present
        if record.exc_info and record.exc_text:
            base_msg += f"\n{record.exc_text}"
        
        return base_msg


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    structured_logging: bool = False,
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up centralized logging for VP Investments 2.0
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path (overrides log_dir)
        log_dir: Directory for log files (defaults to logs/)
        console_output: Whether to output logs to console
        structured_logging: Whether to use structured JSON logging
        max_log_size: Maximum size per log file in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    
    # Convert log level string to level constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Set up console logging
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if structured_logging:
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
        else:
            console_formatter = VPInvestmentsFormatter(include_extra=True)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Set up file logging
    if log_file or log_dir:
        # Determine log file path
        if log_file:
            log_file_path = Path(log_file)
        else:
            log_dir_path = Path(log_dir) if log_dir else Path("logs")
            log_dir_path.mkdir(exist_ok=True)
            log_file_path = log_dir_path / "vp_investments.log"
        
        # Create directory if it doesn't exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        # File formatter (always include timestamp and extra info)
        if structured_logging:
            file_formatter = StructuredJSONFormatter()
        else:
            file_formatter = VPInvestmentsFormatter(include_extra=True)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncpraw').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    return root_logger


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def get_logger(name: str, extra_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional extra context
    
    Args:
        name: Logger name (typically __name__)
        extra_context: Optional dictionary of context to include in all log messages
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if extra_context:
        # Create a custom LoggerAdapter to include extra context
        logger = VPInvestmentsLoggerAdapter(logger, extra_context)
    
    return logger


class VPInvestmentsLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes extra context in log messages
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add the extra context to the log record
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with context"""
        self.log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with context"""
        self.log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with context"""
        self.log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with context"""
        self.log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with context"""
        self.log(logging.CRITICAL, msg, *args, **kwargs)


def log_function_entry(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log function entry with parameters"""
    params = ', '.join(f"{k}={v}" for k, v in kwargs.items() if k != 'self')
    logger.debug(f"→ {func_name}({params})")


def log_function_exit(logger: logging.Logger, func_name: str, result: Any = None, duration_ms: Optional[float] = None) -> None:
    """Log function exit with result and duration"""
    msg = f"← {func_name}"
    
    if duration_ms is not None:
        msg += f" ({duration_ms:.2f}ms)"
    
    if result is not None:
        # Truncate long results for logging
        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:97] + "..."
        msg += f" → {result_str}"
    
    logger.debug(msg)


def log_performance_metric(logger: logging.Logger, operation: str, duration_ms: float, **context) -> None:
    """Log a performance metric"""
    logger.info(
        f"[TIME] {operation} completed in {duration_ms:.2f}ms",
        extra={'operation': operation, 'duration_ms': duration_ms, **context}
    )


def log_error_with_context(logger: logging.Logger, error: Exception, operation: str = None, **context) -> None:
    """Log an error with full context"""
    error_msg = f"[ERROR] {type(error).__name__}: {error}"
    
    if operation:
        error_msg = f"{operation} failed - {error_msg}"
    
    logger.error(
        error_msg,
        extra={
            'error_type': type(error).__name__,
            'operation': operation,
            **context
        },
        exc_info=True
    )


def configure_vp_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Configure VP Investments logging from configuration
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured root logger
    """
    if config is None:
        config = {}
    
    logging_config = config.get('logging', {})
    
    return setup_logging(
        log_level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('file'),
        log_dir=logging_config.get('directory', 'logs'),
        console_output=logging_config.get('console', True),
        structured_logging=logging_config.get('structured', False),
        max_log_size=logging_config.get('max_size', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5)
    )


# Initialize default logging if not already configured
if not logging.getLogger().handlers:
    setup_logging(console_output=True, log_dir="logs")


# Export key functions and classes
__all__ = [
    'setup_logging',
    'get_logger',
    'VPInvestmentsFormatter',
    'StructuredJSONFormatter',
    'VPInvestmentsLoggerAdapter',
    'log_function_entry',
    'log_function_exit',
    'log_performance_metric',
    'log_error_with_context',
    'configure_vp_logging'
]