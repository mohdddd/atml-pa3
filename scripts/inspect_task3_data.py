"""Record the layout and schemas of the supplied ATML-PA3-Data folder (Task 3 rollouts).

Writes  splits/task3/data_manifest.json  (files, sizes, parquet row counts / schemas / small samples)
and copies the supplied README into configs/task3/ATML-PA3-Data_README.md so the field
definitions are versioned with the code.

Usage: python scripts/inspect_task3_data.py --root /kaggle/working/ATML-PA3-Data
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from pa3.common.paths import CONFIGS, SPLITS
from pa3.common.runlog import write_json


def sha256(path: Path, limit_mb: int = 2048) -> str | None:
    if path.stat().st_size > limit_mb * 2**20:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**22), b""):
            h.update(chunk)
    return h.hexdigest()


def short(v, n=160):
    if isinstance(v, (list, tuple)):
        return {"len": len(v), "head": [short(x, 40) for x in v[:5]]}
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"... [{len(v)} chars]"
    return v


def describe_parquet(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    md = pf.metadata
    schema = {f.name: str(f.type) for f in pf.schema_arrow}
    first = pf.read_row_group(0).slice(0, 2).to_pylist()
    return {
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "columns": schema,
        "sample_rows": [{k: short(v) for k, v in r.items()} for r in first],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    man = {"root": str(root), "files": []}
    for p in files:
        e = {"path": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 2**20, 2), "sha256": sha256(p)}
        if p.suffix == ".parquet":
            try:
                e["parquet"] = describe_parquet(p)
            except Exception as ex:
                e["parquet_error"] = repr(ex)
        man["files"].append(e)
        print(e["path"], e["size_mb"], "MB", e.get("parquet", {}).get("num_rows", ""))
    write_json(SPLITS / "task3" / "data_manifest.json", man)
    readmes = [p for p in files if p.name.lower() == "readme.md"]
    if readmes:
        (CONFIGS / "task3").mkdir(parents=True, exist_ok=True)
        for r in readmes:
            rel = "_".join(r.relative_to(root).parts[:-1])
            dst = CONFIGS / "task3" / (f"ATML-PA3-Data_{rel + '_' if rel else ''}README.md")
            shutil.copy(r, dst)
            print("copied", r, "->", dst)


if __name__ == "__main__":
    main()
