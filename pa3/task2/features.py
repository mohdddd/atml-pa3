"""Frozen SigLIP feature cache.

google/siglip-base-patch16-224, deterministic preprocessing (the model's own image processor: resize to
224x224 bicubic, rescale 1/255, normalise mean = std = 0.5, RGB). The cached feature of an image is
`last_hidden_state` of SiglipVisionModel: the 196 final patch embeddings in row-major 14 x 14 order with
SigLIP's final `post_layernorm` already applied - exactly once (verified in checks: it equals
post_layernorm(hidden_states[-1]); the pooling head is not used). Stored in FP16 as one [N, 196, 768]
memmap plus a key index. Caching runs in its own process, so the vision encoder is gone from GPU memory
before any Qwen training starts; its time / memory are reported separately (feature_cache.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from pa3.common.paths import scratch_dir
from pa3.common.runlog import write_json

N_PATCH, D_VIS = 196, 768


def cache_dir(mode: str) -> Path:
    return scratch_dir("task2", "features", mode)


class _ImageDS(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


def load_vision(name: str, device: str):
    from transformers import AutoImageProcessor, SiglipVisionModel

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SiglipVisionModel.from_pretrained(name, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, AutoImageProcessor.from_pretrained(name)


@torch.no_grad()
def build_cache(keys: list[str], image_path, mode: str, vision: str, batch: int = 128, workers: int = 4) -> dict:
    """Encode every key once; reuses an existing cache that already contains all keys."""
    d = cache_dir(mode)
    idx_path, arr_path = d / "index.json", d / "siglip_patch196_fp16.npy"
    keys = sorted(set(keys))
    if idx_path.exists() and arr_path.exists():
        have = json.loads(idx_path.read_text())
        if set(keys) <= set(have):
            print(f"[features] cache complete ({len(have)} images)", flush=True)
            return json.loads((d / "feature_cache.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model, proc = load_vision(vision, device)
    t_load = time.perf_counter() - t_load
    tmp = d / "tmp_features.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(keys), N_PATCH, D_VIS))
    dl = torch.utils.data.DataLoader(_ImageDS([str(image_path(k)) for k in keys], proc), batch_size=batch,
                                     num_workers=workers, shuffle=False, pin_memory=device == "cuda")
    t0 = time.perf_counter()
    t_gpu = 0.0
    i = 0
    for pix in dl:
        tg = time.perf_counter()
        out = model(pixel_values=pix.to(device, dtype=next(model.parameters()).dtype)).last_hidden_state
        assert out.shape[1:] == (N_PATCH, D_VIS), out.shape
        f = out.to(torch.float16).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        t_gpu += time.perf_counter() - tg
        arr[i:i + len(f)] = f
        i += len(f)
        if (i // batch) % 20 == 0:
            print(f"[features] {i}/{len(keys)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    tmp.rename(arr_path)
    idx_path.write_text(json.dumps(keys))
    total = time.perf_counter() - t0
    rep = {"images": len(keys), "seconds_total": total, "seconds_model_forward": t_gpu,
           "model_load_seconds": t_load, "images_per_second": len(keys) / total, "batch_size": batch,
           "dataloader_workers": workers, "device": device,
           "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "peak_gpu_alloc_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0,
           "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else 0.0,
           "cache_size_gb": arr_path.stat().st_size / 2**30, "dtype": "float16",
           "shape_per_image": [N_PATCH, D_VIS], "model_dtype": str(next(model.parameters()).dtype)}
    write_json(d / "feature_cache.json", rep)
    print(f"[features] done: {rep}", flush=True)
    return rep


@torch.no_grad()
def layernorm_check(vision: str, image_paths: list[str]) -> dict:
    """last_hidden_state == post_layernorm(hidden_states[-1]) -> SigLIP's final LN is applied exactly once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_vision(vision, device)
    model.float()
    from PIL import Image

    pix = torch.stack([proc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"][0]
                       for p in image_paths]).to(device)
    out = model(pixel_values=pix, output_hidden_states=True)
    last, pre = out.last_hidden_state, out.hidden_states[-1]
    post_ln = model.vision_model.post_layernorm if hasattr(model, "vision_model") else model.post_layernorm
    ln = post_ln(pre)
    rep = {"feature_shape": list(last.shape[1:]),
           "max_abs_diff_last_vs_postLN(hidden[-1])": float((last - ln).abs().max()),
           "max_abs_diff_last_vs_hidden[-1]": float((last - pre).abs().max()),
           "max_abs_diff_last_vs_postLN(postLN(hidden[-1]))":
               float((last - post_ln(ln)).abs().max())}
    rep["layernorm_applied_once"] = rep["max_abs_diff_last_vs_postLN(hidden[-1])"] < 1e-4 < \
        rep["max_abs_diff_last_vs_hidden[-1]"]
    del model
    return rep


class FeatureStore:
    """Read-only access to the cache: get(keys) -> FP16 tensor [B, 196, 768]."""

    def __init__(self, mode: str):
        d = cache_dir(mode)
        self.keys = json.loads((d / "index.json").read_text())
        self.row = {k: i for i, k in enumerate(self.keys)}
        self.arr = np.load(d / "siglip_patch196_fp16.npy", mmap_mode="r")

    def __contains__(self, k):
        return k in self.row

    def get(self, keys: list[str]) -> torch.Tensor:
        rows = [self.row[k] for k in keys]
        return torch.from_numpy(np.stack([self.arr[r] for r in rows]))
