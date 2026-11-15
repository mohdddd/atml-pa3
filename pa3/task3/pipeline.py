"""Task 3 pipeline.

    python -m pa3.task3.pipeline offline   # CPU: prepare -> offline selection -> adaptive -> signals -> aggregate
                                           # (needs $PA3_T3_DATA = supplied ATML-PA3-Data folder)
    python -m pa3.task3.pipeline live      # GPU (Kaggle): Step 4 live generation (resumable) -> analysis -> push
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pa3.common.paths import results_dir
from pa3.task3.config import results_task


def status_dir() -> Path:
    return results_dir(results_task(), "_status")


def mark_done(job: str, seconds: float, extra: dict | None = None) -> None:
    (status_dir() / f"{job}.done").write_text(json.dumps({"job": job, "seconds": seconds, "finished": time.ctime(),
                                                          **(extra or {})}))


def _push(paths, msg):
    try:
        from pa3.common.gitsync import push

        push(paths, msg)
    except Exception as e:
        print(f"[git] {e}", flush=True)


def offline() -> None:
    from pa3.task3 import adaptive, aggregate, offline as off, prepare, signals

    for name, fn in [("prepare", prepare.main), ("offline", off.main), ("adaptive", adaptive.main),
                     ("signals", signals.main), ("aggregate", aggregate.main)]:
        t0 = time.time()
        fn()
        mark_done(name, time.time() - t0)
        print(f"[task3] {name} done in {time.time() - t0:.0f}s", flush=True)


def live() -> None:
    from pa3.task3 import live as L

    t0 = time.time()
    L.run()
    s = L.analyze()
    n = s["n_questions"]
    expect = L.config()["live"]["n_questions"]
    if n == expect:
        mark_done("live", time.time() - t0, {"n_questions": n})
    _push([L.out_dir(), status_dir()], f"task3: live generation results ({n}/{expect} questions)")
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["offline", "live"])
    a = ap.parse_args()
    {"offline": offline, "live": live}[a.cmd]()


if __name__ == "__main__":
    main()
