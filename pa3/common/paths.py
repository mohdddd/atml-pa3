"""Central path resolution so the same code runs on Kaggle and locally.

Layout
------
REPO      git clone of this repository (code, configs, splits, small results).
          On Kaggle it lives in /tmp so the token-bearing clone is never saved
          into the notebook's public output.
WORK      persistent notebook output (/kaggle/working on Kaggle, 20 GB cap).
          Only things we deliberately want to keep as Kaggle output.
SCRATCH   large, throwaway files: recovery checkpoints, feature caches,
          HF cache. Override with $PA3_SCRATCH.

Results that go into the report are written under REPO/results/ and pushed
to GitHub with pa3.common.gitsync.
"""
from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists() and "KAGGLE_KERNEL_RUN_TYPE" in os.environ

REPO = Path(os.environ.get("PA3_REPO", Path(__file__).resolve().parents[2]))
WORK = Path(os.environ.get("PA3_WORK", "/kaggle/working" if ON_KAGGLE else REPO / "_work"))
SCRATCH = Path(os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch" if ON_KAGGLE else REPO / "_scratch"))

CONFIGS = REPO / "configs"
SPLITS = REPO / "splits"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
INPUT = Path("/kaggle/input") if ON_KAGGLE else REPO / "_input"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def results_dir(task: str, *parts: str) -> Path:
    """results/<task>/<parts...>, created on demand. task in {'task1','task2','task3','env'}."""
    p = RESULTS.joinpath(task, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scratch_dir(*parts: str) -> Path:
    p = SCRATCH.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
