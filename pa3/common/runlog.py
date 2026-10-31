"""Small logging helpers shared by all tasks: JSONL metric logs, timers, GPU memory, config dumps."""
from __future__ import annotations

import gzip
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


def _default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    import dataclasses

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if hasattr(o, "items"):  # mapping-like (e.g. transformers SizeDict)
        return dict(o.items())
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows, compress: bool | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compress = path.suffix == ".gz" if compress is None else compress
    opener = gzip.open if compress else open
    with opener(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")


def read_jsonl(path: str | Path) -> list:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_config(path: str | Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


class JsonlLogger:
    """Append-only metrics log (one JSON object per line). Safe to resume: just keeps appending."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **row) -> None:
        row.setdefault("time", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=_default) + "\n")


@contextmanager
def timer(store: dict, key: str):
    """with timer(res, 'train_s'): ...  -> res['train_s'] = elapsed seconds (accumulates)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def gpu_mem_report(device=None) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def dir_size_mb(path: str | Path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 2**20
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20
