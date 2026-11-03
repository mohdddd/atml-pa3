"""Push small result files from a Kaggle session back to GitHub.

Auth comes from ~/.git-credentials, written by the Kaggle bootstrap cell from
the GH_TOKEN secret, so the token never appears in remote URLs or cell output.

Two notebooks (or Kaggle + the code author) may push at the same time, so every
push does `pull --rebase` first and retries. Keep each run's outputs in its own
results/<task>/<run>/ folder so rebases never conflict.

    from pa3.common.gitsync import push
    push(["results/task1/full_ft"], "task1: full_ft final eval")
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable

from pa3.common.paths import REPO

MAX_FILE_MB = 95  # GitHub hard limit is 100 MB


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], check=check, capture_output=True, text=True)


def _too_large(paths: Iterable[Path]) -> list[str]:
    bad = []
    for p in paths:
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        bad += [str(f) for f in files if f.stat().st_size > MAX_FILE_MB * 2**20]
    return bad


def push(paths: Iterable[str | Path], message: str, retries: int = 5) -> bool:
    """Commit and push; serialised across processes with a file lock (two GPU workers share one clone)."""
    import fcntl
    import os

    if os.environ.get("PA3_NO_PUSH") == "1":  # local testing
        print(f"[gitsync] PA3_NO_PUSH=1, skipping: {message}")
        return False
    with open(REPO / ".git" / "pa3_push.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _push(paths, message, retries)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _push(paths, message: str, retries: int) -> bool:
    paths = [Path(p) if Path(p).is_absolute() else REPO / p for p in paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("[gitsync] nothing to push")
        return False
    big = _too_large(paths)
    if big:
        raise ValueError(f"[gitsync] refusing to commit files > {MAX_FILE_MB} MB: {big}")

    _git("add", "--", *map(str, paths))
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("[gitsync] no changes")
        return False
    _git("commit", "-m", message)

    for attempt in range(1, retries + 1):
        pull = _git("pull", "--rebase", "--autostash", check=False)
        if pull.returncode != 0:
            _git("rebase", "--abort", check=False)
            print(f"[gitsync] pull --rebase failed (attempt {attempt}): {pull.stderr.strip()[:300]}")
        else:
            res = _git("push", check=False)
            if res.returncode == 0:
                print(f"[gitsync] pushed: {message}")
                return True
            print(f"[gitsync] push failed (attempt {attempt}): {res.stderr.strip()[:300]}")
        time.sleep(5 * attempt)
    print("[gitsync] giving up; commit kept locally — re-run push() later")
    return False


def pull() -> None:
    res = _git("pull", "--rebase", "--autostash", check=False)
    print(res.stdout.strip() or res.stderr.strip())


def head_commit() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()
