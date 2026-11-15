"""Task 3 configuration and paths."""
from __future__ import annotations

import os
from pathlib import Path

from pa3.common.paths import CONFIGS, REPO, SPLITS
from pa3.common.runlog import load_config

CFG_PATH = CONFIGS / "task3" / "task3.yaml"
DATASETS = ["math", "mathvista"]
NO_ANSWER = "NO_ANSWER"


def is_dry() -> bool:
    return os.environ.get("PA3_DRY_RUN", "0") == "1"


def results_task() -> str:
    return "task3_dryrun" if is_dry() else "task3"


def config() -> dict:
    cfg = load_config(CFG_PATH)
    cfg["is_dry_run"] = is_dry()
    if is_dry():
        cfg["live"]["caps"] = cfg["dry_run"]["live"]["caps"]
        cfg["live"]["n_questions"] = cfg["dry_run"]["live"]["n_questions"]
    else:
        cfg["live"]["n_questions"] = cfg["splits"]["n_live"]
    if os.environ.get("PA3_T3_LIVE_MODEL"):      # local CPU smoke test only (never set on Kaggle)
        cfg["live"]["model"], cfg["live"]["revision"] = os.environ["PA3_T3_LIVE_MODEL"], None
    return cfg


def data_root(cfg: dict | None = None) -> Path:
    cfg = cfg or config()
    p = Path(os.environ.get("PA3_T3_DATA", cfg["data"]["root"]))
    return p if p.is_absolute() else REPO / p


def dataset_dir(ds: str, cfg: dict | None = None) -> Path:
    cfg = cfg or config()
    return data_root(cfg) / cfg["data"]["datasets"][ds]["folder"]


SPLIT_DIR = SPLITS / "task3"
