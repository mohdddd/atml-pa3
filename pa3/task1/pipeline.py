"""Unattended Task 1 pipeline for one Kaggle session with 2 GPUs.

    python -m pa3.task1.pipeline prepare          # data selection, probe set, downloads (CPU)
    python -m pa3.task1.pipeline worker --gpu 0   # pulls jobs until nothing is left / deadline
    python -m pa3.task1.pipeline run              # prepare + two workers + aggregate (what the notebook calls)

Jobs: 5 training runs and 7 evaluation conditions. A worker takes the first job whose dependencies
are done and which no other worker has claimed (training first, longest first). Finished jobs are
recorded in results/task1/_status/ (pushed to GitHub), so a rerun in a new session skips them, and
an interrupted training run resumes from its newest recovery checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pa3.common.paths import REPO, WORK, results_dir, scratch_dir
from pa3.task1 import config as C

JOBS = (["train:full_ft", "train:l2sp", "train:lora16", "train:oplora16", "train:lora4"]
        + [f"eval:{c}" for c in ["full_ft_mid", "full_ft", "l2sp", "oplora16", "lora16", "lora4", "baseline"]])
EVAL_MINUTES = 8 if C.is_dry() else 50   # do not start an evaluation with less time than this left
TRAIN_MINUTES = 5 if C.is_dry() else 40


def deps(job: str) -> list[str]:
    kind, name = job.split(":")
    if kind == "eval":
        run = C.condition(name)["run"]
        return [f"train:{run}"] if run else []
    return []


def status_dir() -> Path:
    return results_dir(C.results_task(), "_status")


def claims_dir() -> Path:
    return scratch_dir(C.results_task(), "_claims")


def is_done(job: str) -> bool:
    return (status_dir() / f"{job.replace(':', '_')}.done").exists()


def is_failed(job: str) -> bool:
    return (claims_dir() / f"{job.replace(':', '_')}.failed").exists()


def is_claimed(job: str) -> bool:
    return (claims_dir() / f"{job.replace(':', '_')}.claim").exists()


def claim(job: str, gpu: int) -> bool:
    try:
        fd = os.open(claims_dir() / f"{job.replace(':', '_')}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(gpu).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(job: str) -> None:
    (claims_dir() / f"{job.replace(':', '_')}.claim").unlink(missing_ok=True)


def blocked(job: str) -> bool:
    return any(is_failed(d) or blocked(d) for d in deps(job))


def time_left_min() -> float:
    return (float(os.environ.get("PA3_DEADLINE", "inf")) - time.time()) / 60


def log_path(job: str, gpu: int) -> Path:
    d = WORK / "task1_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job.replace(':', '_')}.log"


def run_job(job: str, gpu: int) -> int:
    kind, name = job.split(":")
    mod, arg = ("pa3.task1.train", "--run") if kind == "train" else ("pa3.task1.evaluate", "--cond")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(log_path(job, gpu), "a") as f:
        f.write(f"\n===== {time.ctime()} start {job} on GPU {gpu} =====\n")
        f.flush()
        p = subprocess.run([sys.executable, "-m", mod, arg, name], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def mark_done(job: str, gpu: int, seconds: float) -> None:
    from pa3.common.gitsync import push

    p = status_dir() / f"{job.replace(':', '_')}.done"
    p.write_text(json.dumps({"job": job, "gpu": gpu, "seconds": seconds, "finished": time.ctime()}))
    try:
        push([status_dir()], f"task1: {job} done")
    except Exception as e:
        print(f"[git] {e}")


def worker(gpu: int) -> None:
    print(f"[worker{gpu}] start; {time_left_min():.0f} min until deadline", flush=True)
    while True:
        pending = [j for j in JOBS if not is_done(j) and not is_failed(j) and not blocked(j)]
        if not pending:
            print(f"[worker{gpu}] nothing left", flush=True)
            return
        ready = [j for j in pending if not is_claimed(j) and all(is_done(d) for d in deps(j))]
        job = None
        for j in ready:
            need = TRAIN_MINUTES if j.startswith("train") else EVAL_MINUTES
            if time_left_min() < need:
                continue
            if claim(j, gpu):
                job = j
                break
        if job is None:
            running = [j for j in pending if is_claimed(j)]
            if not running:
                print(f"[worker{gpu}] no runnable job (deadline or dependencies); stopping", flush=True)
                return
            time.sleep(60)
            continue
        t0 = time.time()
        print(f"[worker{gpu}] {time.strftime('%H:%M')} start {job} ({time_left_min():.0f} min left)", flush=True)
        rc = run_job(job, gpu)
        dt = time.time() - t0
        if rc == 0:
            mark_done(job, gpu, dt)
            print(f"[worker{gpu}] {job} done in {dt / 60:.1f} min", flush=True)
            release(job)
        elif rc == 3:
            print(f"[worker{gpu}] {job} stopped for the deadline (resumable); worker exits", flush=True)
            release(job)
            return
        else:
            (claims_dir() / f"{job.replace(':', '_')}.failed").write_text(str(rc))
            release(job)
            print(f"[worker{gpu}] {job} FAILED (exit {rc}); see {log_path(job, gpu)}", flush=True)


def prepare() -> None:
    """CPU-side preparation shared by both workers (run once before they start)."""
    from huggingface_hub import snapshot_download

    from pa3.task1.data import build_probe_set, load_tokenized
    from pa3.task1.modeling import load_tokenizer

    cfg = C.common_config()
    if not Path(cfg["model"]).exists():
        snapshot_download(cfg["model"])
    tok = load_tokenizer(cfg["model"])
    load_tokenized(cfg, tok)          # selection + tokenised cache (scratch), IDs -> splits/task1/
    build_probe_set(cfg)              # probe IDs -> splits/task1/probe_set.json
    from lm_eval.tasks import TaskManager, get_task_dict

    get_task_dict(cfg["eval"]["tasks_gen"] + cfg["eval"]["tasks_ll"], TaskManager())  # dataset downloads
    from pa3.common.gitsync import push

    push(["splits/task1"], "task1: data selection and probe set IDs")


def run_all(n_gpus: int = 2) -> None:
    import torch

    prepare()
    n = min(n_gpus, torch.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, "-m", "pa3.task1.pipeline", "worker", "--gpu", str(g)], cwd=REPO,
                              env=dict(os.environ, PYTHONUNBUFFERED="1")) for g in range(n)]
    for p in procs:
        p.wait()
    from pa3.task1 import aggregate

    aggregate.main()
    remaining = [j for j in JOBS if not is_done(j)]
    print("\nREMAINING JOBS:", remaining if remaining else "none - Task 1 complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "worker", "run"])
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "worker":
        worker(a.gpu)
    else:
        run_all()


if __name__ == "__main__":
    main()
