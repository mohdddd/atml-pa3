"""Back up checkpoints to a private Hugging Face Hub repo.

Kaggle sessions die after 12 h (or earlier when idle) and /kaggle/working is
capped at 20 GB, so resumable checkpoints and final model/adapter weights are
uploaded to a private HF model repo (default: <hf_user>/atml-pa3-ckpts).
Requires the HF_TOKEN Kaggle secret (write scope). Everything here is a no-op
with a warning if no token is configured.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_NAME = "atml-pa3-ckpts"


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        return None, None
    api = HfApi(token=token)
    return api, token


def ckpt_repo_id() -> str | None:
    api, _ = _api()
    if api is None:
        return None
    rid = os.environ.get("PA3_HF_REPO")
    if rid:
        return rid
    return f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"


def ensure_repo() -> str | None:
    api, _ = _api()
    if api is None:
        print("[hub] HF_TOKEN not set; checkpoint backup disabled")
        return None
    rid = ckpt_repo_id()
    api.create_repo(rid, private=True, repo_type="model", exist_ok=True)
    return rid


def upload(local: str | Path, path_in_repo: str, message: str | None = None) -> None:
    """Upload a file or folder to <ckpt repo>/<path_in_repo>."""
    api, _ = _api()
    rid = ensure_repo()
    if rid is None:
        return
    local = Path(local)
    msg = message or f"upload {path_in_repo}"
    if local.is_dir():
        api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo, repo_id=rid, commit_message=msg)
    else:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=rid, commit_message=msg)
    print(f"[hub] uploaded {local} -> {rid}/{path_in_repo}")


def download(path_in_repo: str, local_dir: str | Path) -> Path | None:
    """Download a folder (prefix) from the ckpt repo; returns local path or None if absent."""
    from huggingface_hub import snapshot_download

    rid = ckpt_repo_id()
    if rid is None:
        return None
    out = snapshot_download(rid, allow_patterns=[f"{path_in_repo}/*", path_in_repo], local_dir=str(local_dir),
                            token=os.environ.get("HF_TOKEN"))
    p = Path(out) / path_in_repo
    return p if p.exists() else None


def exists(path_in_repo: str) -> bool:
    api, _ = _api()
    rid = ckpt_repo_id()
    if api is None or rid is None or not api.repo_exists(rid):
        return False
    files = api.list_repo_files(rid)
    return any(f == path_in_repo or f.startswith(path_in_repo.rstrip("/") + "/") for f in files)
