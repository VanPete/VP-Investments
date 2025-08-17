import logging
import os
import sys
from typing import Optional
import glob
import time
from logging.handlers import RotatingFileHandler
try:
    from rich.logging import RichHandler  # type: ignore
    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


class SuppressNoiseFilter(logging.Filter):
    """Filter out common noisy messages from third-party libraries."""
    NOISY_SUBSTRINGS = (
        "possibly delisted; no earnings dates found",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for s in self.NOISY_SUBSTRINGS:
            if s in msg:
                return False
        return True

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "vp_investments.log"


def cleanup_old_logs(directory=DEFAULT_LOG_DIR, days_old=2):
    """
    Delete .log files older than the specified number of days in the target directory.
    """
    cutoff_time = time.time() - days_old * 86400
    for log_path in glob.glob(os.path.join(directory, "*.log")):
        if os.path.isfile(log_path) and os.path.getmtime(log_path) < cutoff_time:
            try:
                os.remove(log_path)
            except Exception as e:
                # Avoid emojis in console to prevent UnicodeEncodeError on Windows cp1252
                print(f"[WARN] Failed to delete old log {log_path}: {e}")


class SanitizeUnicodeFilter(logging.Filter):
    """Sanitize log records for non-UTF-8 consoles by replacing unencodable chars.

    This prevents UnicodeEncodeError when printing emojis or symbols on Windows
    terminals that use cp1252 or other legacy encodings.
    """
    def __init__(self, target_encoding: Optional[str]):
        super().__init__()
        self.encoding = (target_encoding or "cp1252").lower()

    def filter(self, record: logging.LogRecord) -> bool:  # always allow
        try:
            msg = record.getMessage()
            # Round-trip encode/decode with replacement for unencodable chars
            safe = msg.encode(self.encoding, errors="replace").decode(self.encoding, errors="replace")
            # Mutate the record's message safely
            record.msg = safe
            record.args = ()
        except Exception:
            pass
        return True


def setup_logging(log_dir=DEFAULT_LOG_DIR, log_file=DEFAULT_LOG_FILE):
    """
    Set up logging with rotation: stream + file output, cleanup on old logs.
    """
    os.makedirs(log_dir, exist_ok=True)
    cleanup_old_logs(directory=log_dir)

    full_log_path = os.path.join(log_dir, log_file)

    file_handler = RotatingFileHandler(
        full_log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8", delay=True
    )
    # Choose a console handler safe for the current terminal encoding
    enc = getattr(sys.stdout, "encoding", None) or os.environ.get("PYTHONIOENCODING", "")
    supports_utf = bool(enc) and ("utf" in enc.lower())
    if _HAS_RICH and supports_utf:
        stream_handler = RichHandler(rich_tracebacks=False, markup=True)
    else:
        stream_handler = logging.StreamHandler()

    noise_filter = SuppressNoiseFilter()
    file_handler.addFilter(noise_filter)
    stream_handler.addFilter(noise_filter)
    if not supports_utf:
        stream_handler.addFilter(SanitizeUnicodeFilter(enc))

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], format=fmt)

    # Tame noisy third-party loggers
    for name in ("yfinance", "yf", "urllib3", "praw", "prawcore", "openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
