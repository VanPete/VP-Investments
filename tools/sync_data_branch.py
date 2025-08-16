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
    # Allow overriding destination
    dest = os.environ.get("VP_SYNC_DEST")
    target = Path(dest) if dest else TARGET
    clean = os.environ.get("VP_SYNC_CLEAN", "0") in {"1", "true", "True"}
    # resolve origin when not provided
    if not repo_url:
        try:
            out = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=str(ROOT), text=True).strip()
            repo_url = out
        except Exception:
            raise SystemExit("Set VP_REPO_URL or run inside a git repo with 'origin' remote")

    tmp = ROOT / ".tmp-data-repo"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        run(["git", "clone", "--depth", "1", "--branch", "data", repo_url, str(tmp)])
    except subprocess.CalledProcessError as e:
        msg = str(e)
        print("[ERROR] Could not clone 'data' branch. It likely hasn't been created yet.")
        print("Reason:", msg)
        print("Next steps: trigger a GitHub Actions run and wait for the 'Persist outputs to data branch' step to complete, then re-run this sync.")
        print("Alternatively, create the branch once with: 'git checkout --orphan data && git commit --allow-empty -m \"init data\" && git push origin data'.")
        raise SystemExit(2)

    runs_dir = tmp / "runs"
    if not runs_dir.exists():
        raise SystemExit("No runs/ found in data branch yet")

    stamps = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    if not stamps:
        raise SystemExit("No runs in data branch")
    latest = runs_dir / stamps[-1]
    # Optionally clean out existing target to avoid stale files across runs
    if target.exists() and clean:
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    for child in latest.iterdir():
        dst = target / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, target)
    print(f"Synced latest run {latest.name} to {target}")

if __name__ == "__main__":
    main()
