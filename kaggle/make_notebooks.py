"""Generates the Kaggle driver notebooks in kaggle/ from plain-Python cell sources.

Every notebook starts with the same BOOTSTRAP cell (secrets -> git credentials ->
clone repo into /tmp -> pip install). Edit here, then run:  python kaggle/make_notebooks.py
"""
import json
from pathlib import Path

GH_OWNER = "mohdddd"
GH_REPO = "atml-pa3"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"  # ATML-PA3-Data

BOOTSTRAP = f'''# ==== PA3 bootstrap (identical in every notebook) ====
# Needs: Internet ON, accelerator "GPU T4 x2", secrets GH_TOKEN (+ optional HF_TOKEN) attached.
import os, subprocess, sys
from kaggle_secrets import UserSecretsClient

GH_OWNER, GH_REPO = "{GH_OWNER}", "{GH_REPO}"
REPO_DIR = "/tmp/" + GH_REPO          # outside /kaggle/working -> never saved to notebook output
BRANCH = "main"

_s = UserSecretsClient()
_gh = _s.get_secret("GH_TOKEN")
try:
    os.environ["HF_TOKEN"] = _s.get_secret("HF_TOKEN")
except Exception:
    print("HF_TOKEN secret not attached (checkpoint backup to HF Hub disabled)")

cred = os.path.expanduser("~/.git-credentials")
with open(cred, "w") as f:
    f.write(f"https://{{GH_OWNER}}:{{_gh}}@github.com\\n")
os.chmod(cred, 0o600)
del _gh
for k, v in [("credential.helper", "store"), ("user.name", GH_OWNER),
             ("user.email", "203875678+mohdddd@users.noreply.github.com"), ("pull.rebase", "true")]:
    subprocess.run(["git", "config", "--global", k, v], check=True)
os.environ["GIT_TERMINAL_PROMPT"] = "0"

if os.path.isdir(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase", "--autostash"], check=True)
else:
    subprocess.run(["git", "clone", "-b", BRANCH, f"https://github.com/{{GH_OWNER}}/{{GH_REPO}}.git", REPO_DIR], check=True)
os.environ["PA3_REPO"] = REPO_DIR
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
!nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || echo "no GPU (CPU session)"
'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "00_setup_smoke_test.ipynb": [
        ("markdown", """# PA3 · 00 · Setup smoke test
Settings (right sidebar): **Accelerator = GPU T4 x2**, **Internet = On**,
**Add-ons → Secrets**: attach `GH_TOKEN` and `HF_TOKEN`.
Run all cells (interactive is fine, ~10 min). Results are pushed to `results/env/`."""),
        ("code", BOOTSTRAP),
        ("code", "# working dir is the repo (set by bootstrap)\n!python scripts/smoke_test.py"),
        ("code", '''from pa3.common.gitsync import push
push(["results/env"], "env: Kaggle setup smoke test")'''),
    ],
    "01_fetch_task3_data.ipynb": [
        ("markdown", """# PA3 · 01 · Fetch supplied Task 3 rollouts (ATML-PA3-Data)
**CPU only** (Accelerator = None — uses no GPU quota), **Internet = On**, secrets `GH_TOKEN` (+ `HF_TOKEN`).
Run with **Save Version → Save & Run All (Commit)** so `/kaggle/working/ATML-PA3-Data` becomes this
notebook's output; later notebooks attach it via *Add Input → Your Work → Notebooks*."""),
        ("code", BOOTSTRAP),
        ("code", f'''DRIVE_FOLDER = "{DRIVE_FOLDER}"
DEST = "/kaggle/working/ATML-PA3-Data"
!pip install -q -U gdown
!gdown --folder "$DRIVE_FOLDER" -O "$DEST" --remaining-ok
!du -sh "$DEST"; find "$DEST" -maxdepth 3 | sort'''),
        ("code", '!python scripts/inspect_task3_data.py --root "$DEST"'),
        ("code", '''from pa3.common.gitsync import push
push(["splits/task3/data_manifest.json", "configs/task3"], "task3: manifest of supplied rollout data")'''),
    ],
}

if __name__ == "__main__":
    here = Path(__file__).parent
    for name, cells in NOTEBOOKS.items():
        (here / name).write_text(json.dumps(nb(cells), indent=1))
        print("wrote", here / name)
