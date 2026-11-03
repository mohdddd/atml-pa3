"""Unattended Task 2 pipeline for one Kaggle session (2 x T4).

    python -m pa3.task2.pipeline prepare   # selection (if splits/task2 is missing), image download, label audit
    python -m pa3.task2.pipeline cache     # SigLIP feature cache (+ final-LN check), then the encoder is released
    python -m pa3.task2.pipeline run       # prepare + cache + GPU0: t196 train->eval, GPU1: t49 train->eval + aggregate

t196 is the required run; t49 (optional 49-token compression) trains concurrently on the second GPU (with
one GPU it runs after t196). Finished jobs are recorded in results/task2/_status/ and skipped on rerun;
interrupted training resumes from its newest checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pa3.common.paths import REPO, WORK, results_dir
from pa3.common.runlog import write_json
from pa3.task2 import config as C


def status_dir() -> Path:
    return results_dir(C.results_task(), "_status")


def is_done(job: str) -> bool:
    return (status_dir() / f"{job.replace(':', '_')}.done").exists()


def mark_done(job: str, seconds: float) -> None:
    (status_dir() / f"{job.replace(':', '_')}.done").write_text(
        json.dumps({"job": job, "seconds": seconds, "finished": time.ctime()}))
    _push([status_dir()], f"task2: {job} done")


def _push(paths, msg):
    try:
        from pa3.common.gitsync import push

        push(paths, msg)
    except Exception as e:
        print(f"[git] {e}", flush=True)


def time_left_min() -> float:
    return (float(os.environ.get("PA3_DEADLINE", "inf")) - time.time()) / 60


def needed_images() -> list[str]:
    from pa3.task2 import data as D

    s = D.load_splits()
    cfg = C.config("t196")
    keys = set(D.train_images(s, cfg.get("subset")))
    for v in D.eval_sets(s).values():
        keys |= set(v)
    return sorted(keys)


def prepare() -> None:
    from huggingface_hub import snapshot_download

    from pa3.task2 import data as D

    cfg = C.config("t196")
    for m in (cfg["lm"], cfg["vision"]):
        snapshot_download(m)
    if not (D.SPLIT_DIR / "vqa.json").exists():
        D.select(cfg)
        _push([D.SPLIT_DIR], "task2: data selection (IDs, texts, answers)")
    iso = D.isolation_report()
    assert iso["ok"], iso
    rep = D.download_images(needed_images())
    write_json(results_dir(C.results_task()) / "images.json", rep)
    from pa3.task2.aggregate import label_audit

    label_audit()
    _push([results_dir(C.results_task()) / "images.json", Path("figures") / C.results_task()], "task2: images ready")


def cache() -> None:
    from pa3.task2 import data as D
    from pa3.task2.features import build_cache, layernorm_check

    cfg = C.config("t196")
    keys = needed_images()
    rep = build_cache(keys, D.image_path, "dry" if C.is_dry() else "full", cfg["vision"])
    out = results_dir(C.results_task())
    write_json(out / "feature_cache.json", rep)
    ev = D.eval_sets()["vqa_test"][:4]
    write_json(out / "feature_checks.json", layernorm_check(cfg["vision"], [str(D.image_path(k)) for k in ev]))
    _push([out / "feature_cache.json", out / "feature_checks.json"], "task2: SigLIP feature cache")


def run_module(args: list[str], gpu: int, log: str) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    d = WORK / "task2_logs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{log}.log", "a") as f:
        f.write(f"\n===== {time.ctime()} start {' '.join(args)} on GPU {gpu} =====\n")
        f.flush()
        return subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def variant_worker(variants: list[str], gpu: int) -> None:
    for v in variants:
        for kind, args in (("train", ["pa3.task2.train", "--variant", v]), ("eval", ["pa3.task2.evaluate", "--variant", v])):
            job = f"{kind}:{v}"
            if is_done(job):
                continue
            if time_left_min() < (5 if C.is_dry() else 20):
                print(f"[worker{gpu}] not enough time left for {job}", flush=True)
                return
            t0 = time.time()
            print(f"[worker{gpu}] {time.strftime('%H:%M')} start {job} ({time_left_min():.0f} min left)", flush=True)
            rc = run_module(args, gpu, job.replace(":", "_"))
            if rc == 0:
                mark_done(job, time.time() - t0)
                print(f"[worker{gpu}] {job} done in {(time.time() - t0) / 60:.1f} min", flush=True)
            else:
                print(f"[worker{gpu}] {job} {'stopped for the deadline' if rc == 3 else f'FAILED (exit {rc})'}; "
                      f"see {WORK / 'task2_logs'}", flush=True)
                return


def run_all() -> None:
    import threading

    import torch

    if not is_done("prepare"):
        t0 = time.time()
        prepare()
        mark_done("prepare", time.time() - t0)
    if not is_done("cache"):
        t0 = time.time()
        rc = subprocess.run([sys.executable, "-m", "pa3.task2.pipeline", "cache"], cwd=REPO,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")).returncode
        if rc != 0:
            raise SystemExit(f"feature caching failed ({rc})")
        mark_done("cache", time.time() - t0)
    n = torch.cuda.device_count()
    plans = [["t196"], ["t49"]] if n >= 2 else [["t196", "t49"]]
    ths = [threading.Thread(target=variant_worker, args=(p, g)) for g, p in enumerate(plans)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    from pa3.task2 import aggregate

    aggregate.main()
    jobs = [f"{k}:{v}" for v in C.VARIANTS for k in ("train", "eval")]
    rem = [j for j in jobs if not is_done(j)]
    print("\nREMAINING JOBS:", rem if rem else "none - Task 2 complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "cache", "run"])
    a = ap.parse_args()
    {"prepare": prepare, "cache": cache, "run": run_all}[a.cmd]()


if __name__ == "__main__":
    main()
