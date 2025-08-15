import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "cloud-data"

def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

def main():
    repo_url = os.environ.get("VP_REPO_URL")
    if not repo_url:
        # default to origin
        try:
            out = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=str(ROOT), text=True).strip()
            repo_url = out
        except Exception:
            raise SystemExit("Set VP_REPO_URL or run inside a git repo with 'origin' remote")

    tmp = ROOT / ".tmp-data-repo"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    run(["git", "clone", "--depth", "1", "--branch", "data", repo_url, str(tmp)])

    runs_dir = tmp / "runs"
    if not runs_dir.exists():
        raise SystemExit("No runs/ found in data branch yet")

    stamps = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    if not stamps:
        raise SystemExit("No runs in data branch")
    latest = runs_dir / stamps[-1]
    TARGET.mkdir(parents=True, exist_ok=True)
    # Copy latest
    for child in latest.iterdir():
        dst = TARGET / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, TARGET)
    print(f"Synced latest run {latest.name} to {TARGET}")

if __name__ == "__main__":
    main()
