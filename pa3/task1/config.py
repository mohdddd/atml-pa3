"""Load and resolve the Task 1 config (common + per-run overrides, optional dry-run overrides)."""
from __future__ import annotations

import copy
import os

from pa3.common.paths import CONFIGS
from pa3.common.runlog import load_config

CFG_PATH = CONFIGS / "task1" / "task1.yaml"
RUNS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
CONDITIONS = ["baseline", "full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]


def is_dry() -> bool:
    return os.environ.get("PA3_DRY_RUN", "0") == "1"


def results_task() -> str:
    """Results folder name: dry runs never touch results/task1."""
    return "task1_dryrun" if is_dry() else "task1"


def hub_prefix() -> str:
    return "dryrun/task1" if is_dry() else "task1"


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def raw() -> dict:
    return load_config(CFG_PATH)


def run_config(run: str) -> dict:
    c = raw()
    cfg = deep_update(c["common"], c["runs"][run])
    cfg["run"] = run
    if is_dry():
        dr = c["dry_run"]
        cfg = deep_update(cfg, {"train": dr["train"], "eval": dr["eval"]})
        if run == "full_ft":
            cfg["train"]["eval_updates"] = [2, 3]
    cfg["dry_run"] = is_dry()
    if os.environ.get("PA3_T1_MODEL"):  # testing only (tiny model on CPU); never set on Kaggle
        cfg["model"] = os.environ["PA3_T1_MODEL"]
    return cfg


def common_config() -> dict:
    c = raw()
    cfg = copy.deepcopy(c["common"])
    if is_dry():
        cfg = deep_update(cfg, {"train": c["dry_run"]["train"], "eval": c["dry_run"]["eval"]})
    cfg["dry_run"] = is_dry()
    if os.environ.get("PA3_T1_MODEL"):
        cfg["model"] = os.environ["PA3_T1_MODEL"]
    return cfg


def condition(name: str) -> dict:
    c = raw()["conditions"][name]
    c = dict(c, name=name)
    if is_dry() and c["run"] is not None:
        c["update"] = 2 if name == "full_ft_mid" else 3
    return c
