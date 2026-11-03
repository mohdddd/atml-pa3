"""Task 2 configuration: common settings + variant overrides (+ dry-run reductions)."""
from __future__ import annotations

import copy
import math
import os

from pa3.common.paths import CONFIGS
from pa3.common.runlog import load_config

CFG_PATH = CONFIGS / "task2" / "task2.yaml"
VARIANTS = ["t196", "t49"]
SUITES = ["grounding", "vqa"]          # final-test / dev question suites


def is_dry() -> bool:
    return os.environ.get("PA3_DRY_RUN", "0") == "1"


def results_task() -> str:
    return "task2_dryrun" if is_dry() else "task2"


def hub_prefix() -> str:
    return "dryrun/task2" if is_dry() else "task2"


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def raw() -> dict:
    return load_config(CFG_PATH)


def config(variant: str = "t196") -> dict:
    c = raw()
    cfg = deep_update(c["common"], c["variants"][variant])
    cfg["variant"] = variant
    cfg["dry_run"] = is_dry()
    if is_dry():
        dr = c["dry_run"]
        cfg["train"] = deep_update(cfg["train"], dr["train"])
        for st, u in zip(cfg["stages"], dr["stage_updates"]):
            st["updates"] = u
        cfg["subset"] = dr["subset"]
    # local CPU smoke tests only (never set on Kaggle)
    if os.environ.get("PA3_T2_STAGE_UPDATES"):
        for st, u in zip(cfg["stages"], map(int, os.environ["PA3_T2_STAGE_UPDATES"].split(","))):
            st["updates"] = u
    if os.environ.get("PA3_T2_EVAL_LIMIT"):
        cfg["eval"]["limit_pairs"] = int(os.environ["PA3_T2_EVAL_LIMIT"])
    return cfg


def stage_bounds(cfg: dict) -> list[int]:
    """Cumulative update count at the end of each stage, e.g. [2250, 3750, 4500]."""
    out, s = [], 0
    for st in cfg["stages"]:
        s += st["updates"]
        out.append(s)
    return out


def total_updates(cfg: dict) -> int:
    return stage_bounds(cfg)[-1]


def stage_of(cfg: dict, update: int) -> tuple[int, int]:
    """(stage index, local index) of the 0-based global update `update`."""
    start = 0
    for i, st in enumerate(cfg["stages"]):
        if update < start + st["updates"]:
            return i, update - start
        start += st["updates"]
    raise ValueError(update)


def stage_start(cfg: dict, i: int) -> int:
    return ([0] + stage_bounds(cfg))[i]


def warmup(updates: int, frac: float) -> int:
    return max(1, math.ceil(frac * updates))


def dev_points(cfg: dict) -> list[int]:
    """Updates after which dev metrics are computed: 0 (before training), every dev_every, stage ends."""
    T = total_updates(cfg)
    pts = {0, *stage_bounds(cfg), *range(cfg["train"]["dev_every"], T + 1, cfg["train"]["dev_every"])}
    return sorted(p for p in pts if p <= T)


def ckpt_points(cfg: dict) -> list[int]:
    T = total_updates(cfg)
    return sorted({*range(cfg["train"]["ckpt_every"], T + 1, cfg["train"]["ckpt_every"]), *stage_bounds(cfg)})
