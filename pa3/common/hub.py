"""Back up checkpoints to private Hugging Face Hub repos.

Kaggle sessions end after 12 h and /kaggle/working is capped at 20 GB, so
  * weights needed later (evaluation models, adapters)  -> <user>/atml-pa3-ckpts
  * resumable recovery checkpoints                       -> <user>/atml-pa3-rec-<run>
Recovery repos keep only the newest checkpoint: older folders are deleted and the history squashed,
so multi-GB optimizer states do not accumulate in Hub storage.
Requires the HF_TOKEN secret (write scope). Every function is a no-op with a warning without a token.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

MAIN_REPO = "atml-pa3-ckpts"


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token) if token else None


_user_cache: dict = {}


def repo_id(name: str = MAIN_REPO) -> str | None:
    api = _api()
    if api is None:
        return None
    if "user" not in _user_cache:
        _user_cache["user"] = api.whoami()["name"]
    return f"{_user_cache['user']}/{name}"


def ensure_repo(name: str = MAIN_REPO) -> str | None:
    api = _api()
    if api is None:
        print("[hub] HF_TOKEN not set; checkpoint backup disabled")
        return None
    rid = repo_id(name)
    api.create_repo(rid, private=True, repo_type="model", exist_ok=True)
    return rid


def upload(local: str | Path, path_in_repo: str, name: str = MAIN_REPO, message: str | None = None) -> None:
    api = _api()
    rid = ensure_repo(name)
    if rid is None:
        return
    local = Path(local)
    msg = message or f"upload {path_in_repo}"
    for attempt in range(3):
        try:
            if local.is_dir():
                api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo, repo_id=rid, commit_message=msg)
            else:
                api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=rid, commit_message=msg)
            print(f"[hub] uploaded {local} -> {rid}/{path_in_repo}", flush=True)
            return
        except Exception as e:  # network hiccups: retry, never kill training
            print(f"[hub] upload failed (attempt {attempt + 1}): {e}", flush=True)
    print(f"[hub] giving up on {path_in_repo}", flush=True)


def upload_async(*args, **kwargs) -> threading.Thread:
    t = threading.Thread(target=upload, args=args, kwargs=kwargs, daemon=False)
    t.start()
    return t


def list_files(name: str = MAIN_REPO) -> list[str]:
    api = _api()
    rid = repo_id(name)
    if api is None or rid is None or not api.repo_exists(rid):
        return []
    return api.list_repo_files(rid)


def exists(path_in_repo: str, name: str = MAIN_REPO) -> bool:
    p = path_in_repo.rstrip("/")
    return any(f == p or f.startswith(p + "/") for f in list_files(name))


def download(path_in_repo: str, local_dir: str | Path, name: str = MAIN_REPO) -> Path | None:
    """Download a file or folder prefix; returns the local path or None if absent."""
    from huggingface_hub import snapshot_download

    rid = repo_id(name)
    if rid is None or not exists(path_in_repo, name):
        return None
    p = path_in_repo.rstrip("/")
    snapshot_download(rid, allow_patterns=[p, f"{p}/*"], local_dir=str(local_dir), token=os.environ.get("HF_TOKEN"))
    out = Path(local_dir) / p
    return out if out.exists() else None


def prune_recovery(name: str, keep_prefix: str) -> None:
    """Delete every top-level folder except keep_prefix, then squash history to free storage."""
    api = _api()
    rid = repo_id(name)
    if api is None or rid is None or not api.repo_exists(rid):
        return
    old = sorted({f.split("/")[0] for f in api.list_repo_files(rid) if "/" in f} - {keep_prefix})
    try:
        for d in old:
            api.delete_folder(path_in_repo=d, repo_id=rid, commit_message=f"prune {d}")
        if old:
            api.super_squash_history(repo_id=rid)
    except Exception as e:
        print(f"[hub] prune failed (harmless): {e}", flush=True)
