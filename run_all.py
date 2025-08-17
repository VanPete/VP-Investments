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

# Ensure standard outputs structure exists for fresh starts
def _ensure_outputs_structure():
    base = os.path.join(PROJECT_ROOT, "outputs")
    subdirs = [
        base,
        os.path.join(base, "logs"),
        os.path.join(base, "plots"),
        os.path.join(base, "tables"),
        os.path.join(base, "top_signals"),
        os.path.join(base, "breakouts"),
        os.path.join(base, "dashboard"),
        os.path.join(base, "weights"),
        os.path.join(base, "References"),
    ]
    for d in subdirs:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

_ensure_outputs_structure()
os.makedirs(BASE_LOG_DIR, exist_ok=True)

def _write_root_index_to_latest_run():
    """Create outputs/index.html pointing to the latest per-run homepage when present.

    Falls back to dashboard/dashboard.html if no per-run index is found. Best-effort only.
    """
    try:
        outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
        # Find candidate run dirs that contain an index.html (per-run homepage)
        candidates = []
        for name in os.listdir(outputs_dir):
            p = os.path.join(outputs_dir, name)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.html")):
                try:
                    mtime = os.path.getmtime(os.path.join(p, "index.html"))
                except Exception:
                    mtime = 0
                candidates.append((mtime, name))
        target_rel = None
        if candidates:
            candidates.sort(reverse=True)
            # name is relative to outputs/
            target_rel = f"{candidates[0][1]}/index.html"
        else:
            # Fallback to the static dashboard
            target_rel = "dashboard/dashboard.html"
        idx_path = os.path.join(outputs_dir, "index.html")
        # Write minimal redirect page
        html = f"<meta http-equiv=\"refresh\" content=\"0; url={target_rel}\">"
        with open(idx_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        # Non-fatal; Pages step has its own fallback
        pass

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
    # RUN_DIR will be injected later once we know it; leave empty here
    env.setdefault("RUN_DIR", env.get("RUN_DIR", ""))
    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"=== {label.upper()} STARTED at {timestamp} ET ===\n")
            if stream:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
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

def aggregate_fetch_reliability(run_dir: str) -> None:
    """Aggregate component metrics JSONL files into References/Fetch_Reliability.csv."""
    try:
        import json, glob
        import pandas as _pd
        logs_dir = os.path.join(run_dir, "logs")
        refs_dir = os.path.join(run_dir, "References")
        os.makedirs(refs_dir, exist_ok=True)
        rows = []
        for path in glob.glob(os.path.join(logs_dir, "*_metrics.jsonl")):
            comp = os.path.basename(path).replace("_metrics.jsonl", "")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        obj["component"] = comp
                        rows.append(obj)
                    except Exception:
                        continue
        if not rows:
            return
        df = _pd.DataFrame(rows)
        # Normalize fields
        df["ok"] = _pd.to_numeric(df.get("ok"), errors='coerce')
        df["latency_ms"] = _pd.to_numeric(df.get("latency_ms"), errors='coerce')
        df["breaker_open"] = df.get("breaker_open").astype(bool) if "breaker_open" in df.columns else False
        agg = df.groupby("component").agg(
            calls=("component", "size"),
            success=("ok", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            breaker_opens=("breaker_open", "sum")
        ).reset_index()
        # Derive failures
        agg["failures"] = agg["calls"] - agg["success"].fillna(0).astype(int)
        # Save
        out_path = os.path.join(refs_dir, "Fetch_Reliability.csv")
        agg.to_csv(out_path, index=False)
    except Exception:
        pass

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

        # Precompute the run directory name to share via env
        from datetime import datetime as _dt
        run_dir = os.path.join(PROJECT_ROOT, "outputs", _dt.now().strftime("(%d %B %y, %H_%M_%S)"))
        os.makedirs(run_dir, exist_ok=True)
        print(f"📂 Run outputs directory: {run_dir}")

        # Step order
        for step in ["vp", "backtest", "trainer"]:
            if step not in selected_steps:
                continue
            if step == "vp":
                print("▶️ Running VP Investments…")
                # Inject RUN_DIR so children can emit metrics
                os.environ["RUN_DIR"] = run_dir
                code = run_script("vp_investments", SCRIPTS["vp_investments"], log_dir, args=extra_args["vp"], timeout_sec=timeout_sec, stream=stream)
                if code != 0:
                    print("❌ VP Investments script failed. Skipping remaining steps.")
                    return
                if not wait_for_signals_table():
                    print("❌ 'signals' table not ready. Skipping remaining steps.")
                    return
                # Update root index to point at the latest run homepage
                _write_root_index_to_latest_run()
            elif step == "backtest":
                print("▶️ Running Backtest…")
                run_script("backtest", SCRIPTS["backtest"], log_dir, args=extra_args["backtest"], timeout_sec=timeout_sec, stream=stream)
            elif step == "trainer":
                print("▶️ Running Scoring Trainer…")
                run_script("scoring_trainer", SCRIPTS["scoring_trainer"], log_dir, args=extra_args["trainer"], timeout_sec=timeout_sec, stream=stream)

        # After run ends, aggregate reliability metrics into this run dir
        try:
            aggregate_fetch_reliability(run_dir)
        except Exception:
            pass

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
