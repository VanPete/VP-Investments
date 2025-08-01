import logging
import os
import glob
import time
from logging.handlers import RotatingFileHandler

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
                print(f"⚠️ Failed to delete old log {log_path}: {e}")


def setup_logging(log_dir=DEFAULT_LOG_DIR, log_file=DEFAULT_LOG_FILE):
    """
    Set up logging with rotation: stream + file output, cleanup on old logs.
    """
    os.makedirs(log_dir, exist_ok=True)
    cleanup_old_logs(directory=log_dir)

    full_log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            RotatingFileHandler(
                full_log_path, maxBytes=1_000_000, backupCount=3,
                encoding="utf-8", delay=True
            ),
            logging.StreamHandler()
        ]
    )
