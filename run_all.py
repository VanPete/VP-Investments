import os
import sys
import subprocess
import sqlite3
import time
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv, find_dotenv
import argparse
import shlex

from config.config import (
    DB_PATH,
    SCHED_ENABLED,
    SCHED_EVERY_MINUTES,
    SCHED_TIMEZONE,
    SCHED_TWICE_DAILY,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    PRE_OPEN_OFFSET_MINUTES,
    POST_CLOSE_OFFSET_MINUTES,
)

# === Constants ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Ensure CWD is project root for all child processes and relative paths
os.chdir(PROJECT_ROOT)
# Load .env once for the whole pipeline
try:
    load_dotenv(find_dotenv())
except Exception:
    pass
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

def run_script(label, path, log_dir, args=None, timeout_sec: int | None = None, stream: bool = False):
    """Execute a Python script and log output to timestamped file.

    Args:
        label: Short name for logs.
        path: Script path.
        log_dir: Directory to write the timestamped log.
        args: Optional list of extra CLI args to pass to the script.
        timeout_sec: Optional timeout in seconds; kill process if exceeded.
        stream: If True, tee child output to console while writing to log.
    Returns exit code (int).
    """
    timestamp, _ = get_et_timestamp()
    log_path = os.path.join(log_dir, f"{label}_{timestamp}.log")
    cmd = [sys.executable, path]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))
    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== {label.upper()} STARTED at {timestamp} ET ===\n")
            if stream:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env, text=True, bufsize=1)
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log_file.write(line)
                        log_file.flush()
                        print(line, end="")
                        if timeout_sec and (time.time() - start) > timeout_sec:
                            proc.kill()
                            raise TimeoutError(f"Timeout after {timeout_sec}s")
                    proc.wait()
                    exit_code = proc.returncode
                finally:
                    try:
                        if proc and proc.poll() is None:
                            proc.kill()
                    except Exception:
                        pass
            else:
                result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env, timeout=timeout_sec)
                exit_code = result.returncode
            status = "SUCCESS" if exit_code == 0 else f"FAIL (exit {exit_code})"
            log_file.write(f"\n=== {label.upper()} COMPLETED: {status} in {time.time()-start:.1f}s ===\n")
    except TimeoutError as te:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nTIMEOUT: {str(te)}\n=== {label.upper()} COMPLETED: TIMEOUT ===\n")
        print(f"[TIMEOUT] {label}: {te}")
        return 124
    except KeyboardInterrupt:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\nINTERRUPTED by user.\n=== {label} ABORTED ===\n")
        print(f"[ABORTED] {label} by user.")
        return 130
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nEXCEPTION: {str(e)}\n=== {label.upper()} COMPLETED: CRASHED ===\n")
        print(f"[CRASH] {label}: {e}")
        return 1
    print(f"{status} for {label}. Log saved to {log_path}")
    return exit_code

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
    # CLI
    parser = argparse.ArgumentParser(description="Run VP Investments pipeline (scheduled or once).")
    parser.add_argument("--once", action="store_true", help="Run once immediately and exit.")
    parser.add_argument("--steps", type=str, default="vp,backtest,trainer", help="Comma-separated steps to run: vp,backtest,trainer")
    parser.add_argument("--timeout", type=int, default=0, help="Per-step timeout in seconds (0 = no timeout)")
    parser.add_argument("--stream", action="store_true", help="Stream child process output to console while logging.")
    parser.add_argument("--args-vp", type=str, default="", help="Extra args for VP Investments script (quoted string)")
    parser.add_argument("--args-backtest", type=str, default="", help="Extra args for backtest script (quoted string)")
    parser.add_argument("--args-trainer", type=str, default="", help="Extra args for scoring trainer (quoted string)")
    args_cli = parser.parse_args()

    run_immediately = args_cli.once
    selected_steps = [s.strip() for s in args_cli.steps.split(",") if s.strip()]
    timeout_sec = args_cli.timeout if args_cli.timeout and args_cli.timeout > 0 else None
    stream = args_cli.stream
    extra_args = {
        "vp": shlex.split(args_cli.args_vp) if args_cli.args_vp else [],
        "backtest": shlex.split(args_cli.args_backtest) if args_cli.args_backtest else [],
        "trainer": shlex.split(args_cli.args_trainer) if args_cli.args_trainer else [],
    }
    def run_once():
        timestamp, date_str = get_et_timestamp()
        log_dir = os.path.join(BASE_LOG_DIR, date_str)
        os.makedirs(log_dir, exist_ok=True)
        # Step order
        for step in ["vp", "backtest", "trainer"]:
            if step not in selected_steps:
                continue
            if step == "vp":
                print("▶️ Running VP Investments…")
                code = run_script("vp_investments", SCRIPTS["vp_investments"], log_dir, args=extra_args["vp"], timeout_sec=timeout_sec, stream=stream)
                if code != 0:
                    print("❌ VP Investments script failed. Skipping remaining steps.")
                    return
                if not wait_for_signals_table():
                    print("❌ 'signals' table not ready. Skipping remaining steps.")
                    return
            elif step == "backtest":
                print("▶️ Running Backtest…")
                run_script("backtest", SCRIPTS["backtest"], log_dir, args=extra_args["backtest"], timeout_sec=timeout_sec, stream=stream)
            elif step == "trainer":
                print("▶️ Running Scoring Trainer…")
                run_script("scoring_trainer", SCRIPTS["scoring_trainer"], log_dir, args=extra_args["trainer"], timeout_sec=timeout_sec, stream=stream)

    if run_immediately or not SCHED_ENABLED:
        run_once()
        return

    # Background schedule
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore
        import pytz as _pytz
    except Exception as e:
        print(f"[WARN] APScheduler not available ({e}). Running once.")
        run_once()
        return

    tz = _pytz.timezone(SCHED_TIMEZONE)
    sched = BlockingScheduler(timezone=tz)
    if SCHED_TWICE_DAILY:
        # Compute pre-open and post-close times
        def _parse_hhmm(s: str):
            hh, mm = s.split(":")
            return int(hh), int(mm)
        open_h, open_m = _parse_hhmm(MARKET_OPEN_TIME)
        close_h, close_m = _parse_hhmm(MARKET_CLOSE_TIME)
        pre_open_dt = (datetime.now(tz).replace(hour=open_h, minute=open_m, second=0, microsecond=0) - timedelta(minutes=PRE_OPEN_OFFSET_MINUTES))
        post_close_dt = (datetime.now(tz).replace(hour=close_h, minute=close_m, second=0, microsecond=0) + timedelta(minutes=POST_CLOSE_OFFSET_MINUTES))
        sched.add_job(run_once, 'cron', day_of_week='mon-fri', hour=pre_open_dt.hour, minute=pre_open_dt.minute, id='pre_open')
        sched.add_job(run_once, 'cron', day_of_week='mon-fri', hour=post_close_dt.hour, minute=post_close_dt.minute, id='post_close')
        print(f"⏱️ Scheduler started: Mon-Fri at {pre_open_dt.hour:02d}:{pre_open_dt.minute:02d} (pre-open) and {post_close_dt.hour:02d}:{post_close_dt.minute:02d} (post-close) in {SCHED_TIMEZONE}")
    else:
        sched.add_job(run_once, 'interval', minutes=max(1, int(SCHED_EVERY_MINUTES)))
        print(f"⏱️ Scheduler started: every {SCHED_EVERY_MINUTES} min in {SCHED_TIMEZONE} (Ctrl+C to stop)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

if __name__ == "__main__":
    main()
