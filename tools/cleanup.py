import os
import shutil
import argparse
import stat
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import ctypes  # For Windows delete-on-reboot fallback
except Exception:
    ctypes = None

ROOT = Path(__file__).resolve().parents[1]

# Directories considered safe to purge
PURGE_DIRS = [
    ROOT / "__pycache__",
    ROOT / ".pytest_cache",
    ROOT / "web" / "vp-investments-web" / ".next",
    ROOT / "web" / "vp-investments-web" / "node_modules",
    ROOT / "outputs" / "logs",
]

# File patterns considered safe to remove
PATTERNS = [
    "*.pyc", "*.pyo", "*.tmp", "*.log.1", "*.log.2", "*.log.*",
]

# Individual files known to be obsolete
OBSOLETE_FILES = [
    ROOT / "Roadmap.txt",  # superseded by README sections
    ROOT / "bug_checker.py",  # temporary diagnostic script no longer used
]


def remove_path(p: Path):
    """Best-effort removal with retry and Windows read-only fix."""
    def _on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        try:
            func(path)
        except Exception as e2:
            print(f"[CLEAN][WARN] onerror could not remove {path}: {e2}")

    if p.is_dir():
        try:
            shutil.rmtree(p, ignore_errors=False, onerror=_on_rm_error)
            print(f"[CLEAN] Removed directory: {p}")
        except Exception as e:
            print(f"[CLEAN][WARN] Failed to remove directory {p}: {e}")
    elif p.exists():
        try:
            p.unlink()
            print(f"[CLEAN] Removed file: {p}")
        except Exception as e:
            print(f"[CLEAN][WARN] Failed to remove file {p}: {e}")


def try_remove_locked_file(file_path: Path, description: str = "file") -> None:
    """Attempt to remove a possibly-locked file on Windows.

    Strategy:
    - unlink()
    - if fails: try rename to a temp name in same folder, then unlink
    - if still fails on Windows: schedule delete on reboot via MoveFileExW
    - as a last resort, try truncating to zero bytes
    """
    if not file_path.exists():
        return
    # 1) direct unlink
    try:
        file_path.unlink()
        print(f"[CLEAN] Removed {description}: {file_path}")
        return
    except Exception as e:
        err = e
    # 2) rename then unlink
    try:
        tmp = file_path.with_suffix(file_path.suffix + f".to_delete_{int(time.time())}")
        file_path.rename(tmp)
        try:
            tmp.unlink()
            print(f"[CLEAN] Removed {description} (via rename): {file_path}")
            return
        except Exception as e2:
            err = e2
            file_path = tmp
    except Exception as e3:
        err = e3
    # 3) schedule delete on reboot (Windows only)
    if sys.platform.startswith("win") and ctypes is not None:
        try:
            MoveFileExW = ctypes.windll.kernel32.MoveFileExW
            MoveFileExW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
            MoveFileExW.restype = ctypes.c_int
            MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
            res = MoveFileExW(str(file_path), None, MOVEFILE_DELAY_UNTIL_REBOOT)
            if res:
                print(f"[CLEAN][INFO] Scheduled {description} for deletion on reboot: {file_path}")
                return
        except Exception as e4:
            err = e4
    # 4) last resort: truncate
    try:
        with open(file_path, "wb"):
            pass
        print(f"[CLEAN][INFO] Truncated {description} to zero bytes (couldn't delete now): {file_path}")
    except Exception as e5:
        print(f"[CLEAN][ERR] Could not remove or truncate {description} {file_path}: {err} / {e5}")


def wipe_db_data_file(db: Path) -> None:
    """Clear all user tables inside an SQLite DB without deleting the file."""
    try:
        import sqlite3
    except Exception as e:
        print(f"[CLEAN][ERR] sqlite3 not available to wipe DB: {e}")
        return
    if not db.exists():
        print(f"[CLEAN][INFO] DB file not found at {db}, nothing to wipe.")
        return
    try:
        with sqlite3.connect(str(db), timeout=10) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=OFF;")
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )]
            for t in tables:
                try:
                    cur = conn.execute(f'SELECT COUNT(*) FROM "{t}"')
                    cnt = cur.fetchone()[0]
                except Exception:
                    cnt = "?"
                try:
                    conn.execute(f'DELETE FROM "{t}"')
                    print(f"[CLEAN] Cleared table {t} (rows before: {cnt})")
                except Exception as e:
                    print(f"[CLEAN][WARN] Could not clear {t}: {e}")
            try:
                conn.execute('DELETE FROM sqlite_sequence')
            except Exception:
                pass
            try:
                conn.execute('VACUUM')
                print("[CLEAN] VACUUM completed.")
            except Exception as e:
                print(f"[CLEAN][WARN] VACUUM failed: {e}")
    except Exception as e:
        print(f"[CLEAN][ERR] Could not open DB for wiping (locked?): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-reset", action="store_true", help="Wipe outputs (runs, plots, tables, dashboard), caches, and backtest DB.")
    parser.add_argument("--keep-weights", action="store_true", help="Preserve outputs/weights even during hard reset.")
    parser.add_argument("--wipe-db-data", action="store_true", help="Clear all data from backtest.db (keep file/schema).")
    args = parser.parse_args()

    # Purge directories
    for d in PURGE_DIRS:
        if d.exists():
            remove_path(d)

    # Remove pattern-matched files recursively
    for pattern in PATTERNS:
        for p in ROOT.rglob(pattern):
            # keep main logs/vp_investments.log but remove rotated ones
            if p.name == "vp_investments.log":
                continue
            remove_path(p)

    # Remove obsolete individual files if present
    for f in OBSOLETE_FILES:
        if f.exists():
            remove_path(f)

    if args.hard_reset:
        # Wipe outputs except keep outputs/config_logs (and optionally weights)
        outputs = ROOT / "outputs"
        if outputs.exists():
            keep_dirs = {"config_logs"}
            if args.keep_weights:
                keep_dirs.add("weights")
            for child in list(outputs.iterdir()):
                if child.is_dir():
                    if child.name not in keep_dirs:
                        remove_path(child)
                elif child.is_file():
                    if child.name == "backtest.db":
                        # We'll handle backtest.db explicitly below
                        continue
                    if child.name != "README.md":
                        remove_path(child)
        # Also clear web app outputs mirror if present
        web_outputs = ROOT / "web" / "vp-investments-web" / "outputs"
        if web_outputs.exists():
            for child in list(web_outputs.iterdir()):
                if child.is_dir():
                    remove_path(child)
                elif child.is_file():
                    remove_path(child)
        # Remove caches
        for p in [ROOT / "cache" / "news", ROOT / "cache" / "ai", ROOT / "cache" / "finance_cache.sqlite", ROOT / "cache" / "universe_cache.sqlite"]:
            remove_path(p)
    # Wipe backtest DB data (keep file)
    db = ROOT / "outputs" / "backtest.db"
    wipe_db_data_file(db)

    # Wipe data inside SQLite DB without deleting the file
    if args.wipe_db_data:
        db = ROOT / "outputs" / "backtest.db"
        wipe_db_data_file(db)

    print("[CLEAN] Done.")


if __name__ == "__main__":
    main()
