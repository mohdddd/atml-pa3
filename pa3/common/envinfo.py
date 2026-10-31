"""Collect a machine-readable environment report (stored with every run's results)."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys

PACKAGES = [
    "torch", "torchvision", "transformers", "peft", "accelerate", "datasets", "huggingface_hub",
    "lm_eval", "math_verify", "pycocotools", "numpy", "pandas", "pyarrow", "scipy", "vllm", "flash_attn",
]


def _version(pkg: str) -> str | None:
    for name in (pkg, pkg.replace("_", "-")):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return None


def env_report(disk_paths=("/kaggle/working", "/tmp", "/kaggle/input")) -> dict:
    rep: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {p: _version(p) for p in PACKAGES},
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }
    try:
        import psutil

        rep["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    rep["disk_free_gb"] = {}
    for p in disk_paths:
        if os.path.exists(p):
            rep["disk_free_gb"][p] = round(shutil.disk_usage(p).free / 2**30, 1)
    try:
        import torch

        rep["cuda"] = torch.version.cuda
        rep["cudnn"] = torch.backends.cudnn.version()
        rep["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        rep["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        pass
    try:
        rep["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        from pa3.common.gitsync import head_commit

        rep["git_commit"] = head_commit()
    except Exception:
        pass
    return rep
