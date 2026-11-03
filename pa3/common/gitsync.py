"""Push small result files from a Kaggle session back to GitHub.

Auth comes from ~/.git-credentials, written by the Kaggle bootstrap cell from
the GH_TOKEN secret, so the token never appears in remote URLs or cell output.

Several processes (two GPU workers, or Kaggle + the code author) may push at the same time. Pushes never
touch the main working tree: the files are copied into a dedicated push worktree (<repo>_push, sharing the
clone's object store), committed on top of the freshly fetched remote branch, and pushed, with retries.
(The earlier `pull --rebase --autostash` in the main tree briefly stashed files that other processes were
still appending to; in the Task 2 run one train_log.jsonl row written in that window was lost.)
Keep each run's outputs in its own results/<task>/<run>/ folder.

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


BRANCH = "main"


def _wt(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(_push_dir()), *args], check=check, capture_output=True, text=True)


def _push_dir() -> Path:
    return REPO.parent / f"{REPO.name}_push"


def _ensure_worktree() -> None:
    d = _push_dir()
    if (d / ".git").exists():
        return
    _git("worktree", "prune", check=False)
    _git("fetch", "origin", BRANCH)
    _git("worktree", "add", "--detach", "--force", str(d), f"origin/{BRANCH}")


def _push(paths, message: str, retries: int) -> bool:
    import shutil

    paths = [Path(p) if Path(p).is_absolute() else REPO / p for p in paths]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("[gitsync] nothing to push")
        return False
    big = _too_large(paths)
    if big:
        raise ValueError(f"[gitsync] refusing to commit files > {MAX_FILE_MB} MB: {big}")
    rel = [p.resolve().relative_to(REPO.resolve()) for p in paths]
    _ensure_worktree()
    d = _push_dir()
    for attempt in range(1, retries + 1):
        f = _git("fetch", "origin", BRANCH, check=False)
        if f.returncode != 0:
            print(f"[gitsync] fetch failed (attempt {attempt}): {f.stderr.strip()[:300]}")
            time.sleep(5 * attempt)
            continue
        _wt("reset", "--hard", f"origin/{BRANCH}")
        for r in rel:  # snapshot the current files into the push worktree
            src, dst = REPO / r, d / r
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        _wt("add", "--", *map(str, rel))
        if _wt("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[gitsync] no changes")
            return False
        _wt("commit", "-m", message)
        res = _wt("push", "origin", f"HEAD:{BRANCH}", check=False)
        if res.returncode == 0:
            print(f"[gitsync] pushed: {message}")
            return True
        print(f"[gitsync] push failed (attempt {attempt}): {res.stderr.strip()[:300]}")
        time.sleep(5 * attempt)
    print("[gitsync] giving up; files remain in the working tree - re-run push() later")
    return False


def pull() -> None:
    res = _git("pull", "--rebase", "--autostash", check=False)
    print(res.stdout.strip() or res.stderr.strip())


def head_commit() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()
