import os
import sys
import subprocess
import sqlite3
import time
from datetime import datetime
import pytz

from config.config import DB_PATH

# === Constants ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

SCRIPTS = {
    "vp_investments": os.path.join(PROJECT_ROOT, "VP Investments.py"),
    "backtest": os.path.join(PROJECT_ROOT, "processors", "backtest.py"),
    "scoring_trainer": os.path.join(PROJECT_ROOT, "processors", "scoring_trainer.py")
}

# === Utilities ===
def get_et_timestamp():
    """Return Eastern Time timestamp (file-safe and date-only variants)."""
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(pytz.utc).astimezone(et)
    return now_et.strftime("%Y-%m-%d_%H-%M-%S"), now_et.strftime("%Y-%m-%d")

def run_script(label, path, log_dir):
    """Execute a Python script and log output to timestamped file."""
    timestamp, _ = get_et_timestamp()
    log_path = os.path.join(log_dir, f"{label}_{timestamp}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== {label.upper()} STARTED at {timestamp} ET ===\n")
            result = subprocess.run([sys.executable, path], stdout=log_file, stderr=subprocess.STDOUT)
            status = "SUCCESS" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
            log_file.write(f"\n=== {label.upper()} COMPLETED: {status} ===\n")
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nEXCEPTION: {str(e)}\n=== {label.upper()} COMPLETED: CRASHED ===\n")
        print(f"[CRASH] {label}: {e}")
        return 1
    print(f"{status} for {label}. Log saved to {log_path}")
    return result.returncode

def wait_for_signals_table(timeout_seconds=15):
    """Poll the database for existence of 'signals' table, up to a timeout."""
    print("⏳ Waiting for 'signals' table in DB...")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("SELECT 1 FROM signals LIMIT 1")
                print("✅ 'signals' table is ready.")
                return True
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print("[WAIT] Table not yet created...")
            else:
                print(f"[WARN] SQLite error: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected DB error: {e}")
        time.sleep(1)
    print("❌ Timeout: 'signals' table not ready.")
    return False

# === Pipeline Runner ===
def main():
    timestamp, date_str = get_et_timestamp()
    log_dir = os.path.join(BASE_LOG_DIR, date_str)
    os.makedirs(log_dir, exist_ok=True)

    print("▶️ Running VP Investments...")
    if run_script("vp_investments", SCRIPTS["vp_investments"], log_dir) != 0:
        sys.exit("❌ VP Investments script failed. Aborting pipeline.")

    if not wait_for_signals_table():
        sys.exit("❌ 'signals' table not ready. Aborting pipeline.")

    print("▶️ Running Backtest...")
    run_script("backtest", SCRIPTS["backtest"], log_dir)

    print("▶️ Running Scoring Trainer...")
    run_script("scoring_trainer", SCRIPTS["scoring_trainer"], log_dir)

if __name__ == "__main__":
    main()
