"""
Sync the latest GitHub Actions run outputs from the repo's `data` branch
into the local `outputs/` folder so they appear exactly like a local run.

Requirements:
- git installed and available on PATH
- network access to GitHub

Behavior:
- Clones or updates a shallow workdir under .sync/vp-data
- Checks out the `data` branch which accumulates all runs under runs/<timestamp>
- Finds the latest timestamp folder and copies it to ./outputs/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/VanPete/VP-Investments.git"
DATA_BRANCH = "data"
WORKDIR = Path(".sync") / "vp-data"
RUNS_DIR = WORKDIR / "runs"
DEST_DIR = Path("outputs")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[sync] Command failed: {' '.join(cmd)}\n{e}\n{e.stdout}\n{e.stderr}")
        sys.exit(2)


def ensure_repo() -> bool:
    """Clone or update the repo and ensure the data branch exists.
    Returns True if the data branch exists remotely, False otherwise.
    """
    WORKDIR.parent.mkdir(parents=True, exist_ok=True)
    if not WORKDIR.exists():
        print(f"[sync] Cloning {REPO_URL} into {WORKDIR} ...")
        run(["git", "clone", "--no-checkout", REPO_URL, str(WORKDIR)])
    # Check if data branch exists on remote
    res = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", "origin", DATA_BRANCH], cwd=str(WORKDIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[sync] Remote branch '{DATA_BRANCH}' not found yet. Trigger the GitHub Action (vp-pipeline) to persist outputs.")
        return False
    print(f"[sync] Fetching branch {DATA_BRANCH} ...")
    run(["git", "fetch", "origin", DATA_BRANCH], cwd=WORKDIR)
    # Checkout or switch to data branch
    run(["git", "checkout", "-f", DATA_BRANCH], cwd=WORKDIR)
    # Pull latest
    run(["git", "pull", "--ff-only", "origin", DATA_BRANCH], cwd=WORKDIR)
    return True


def latest_run_dir() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Timestamps are ISO-like and sort lexicographically
    latest = sorted(candidates, key=lambda p: p.name, reverse=True)[0]
    return latest


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    # Copy contents of src into dst (merge/overwrite)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            d = target_root / f
            shutil.copy2(s, d)


def main() -> int:
    print("[sync] Starting sync from GitHub data branch ...")
    if not ensure_repo():
        return 0
    latest = latest_run_dir()
    if latest is None:
        print("[sync] No runs found under data branch (runs/...). Nothing to sync.")
        return 0
    print(f"[sync] Latest run: {latest.name}")
    print(f"[sync] Copying into: {DEST_DIR}")
    copy_tree(latest, DEST_DIR)
    print("[sync] Done. Open outputs/index.html or dashboard/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
