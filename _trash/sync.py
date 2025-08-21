"""
Sync all runs from the GitHub data branch into local outputs/runs.
- Clones/updates a shallow copy of the repo's data branch to a temp dir.
- Copies runs/<timestamp>/ into outputs/runs/<timestamp>/ (no deletion of local runs).
- Windows-friendly; no external deps.

Usage (PowerShell):
  python tools/sync.py

Optional env:
  VP_REPO_URL       # override origin remote
  VP_SYNC_DEST      # override destination (default: outputs/runs)
"""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs" / "runs"
TMP = ROOT / ".tmp-data-repo"


def sh(cmd: list[str], cwd: Path | None = None) -> str:
    print("$", " ".join(cmd))
    return subprocess.check_output(cmd, cwd=str(cwd or ROOT), text=True).strip()


def main() -> None:
    repo_url = os.environ.get("VP_REPO_URL")
    dest_env = os.environ.get("VP_SYNC_DEST")
    dest = Path(dest_env) if dest_env else DEST

    if not repo_url:
        try:
            repo_url = sh(["git", "remote", "get-url", "origin"])  # type: ignore[assignment]
        except Exception:
            raise SystemExit("Set VP_REPO_URL or run inside a git repo with 'origin' remote")

    # Fresh clone/update of data branch
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    try:
        sh(["git", "clone", "--depth", "1", "--branch", "data", repo_url, str(TMP)])
    except subprocess.CalledProcessError:
        raise SystemExit("No 'data' branch found. Trigger a pipeline run to create it.")

    runs_dir = TMP / "runs"
    if not runs_dir.exists():
        raise SystemExit("No runs/ found in data branch yet")

    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        target = dest / run_dir.name
        if target.exists():
            # incremental update: replace contents
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(run_dir, target)
        copied += 1
    print(f"Synced {copied} run(s) to {dest}")


if __name__ == "__main__":
    main()
