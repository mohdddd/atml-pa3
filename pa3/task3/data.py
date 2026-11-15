"""Loading the supplied rollouts (join + alignment checks), MathVista metadata, splits and rollout orders.

Rows are kept in a fixed layout: questions sorted lexicographically by `sample_id`, rollouts 0..31 within each
question, so row = question_index * 32 + rollout_id. Per-token lists are stored as flat float32 arrays plus
row offsets (`tok_start[row]` .. `tok_start[row] + n_tokens[row]`).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pa3.task3.config import DATASETS, SPLIT_DIR, config, data_root, dataset_dir

KEYS = ["dataset", "sample_id", "rollout_id"]
N_ROLLOUTS = 32


@dataclass
class Rollouts:
    ds: str
    qids: list                      # sorted question IDs (length Q)
    gen: pd.DataFrame               # one row per rollout, fixed layout
    tok_start: np.ndarray           # (Q*32,) int64 offsets into the flat token arrays
    n_tokens: np.ndarray            # (Q*32,) int64 retained token counts
    tok: dict = field(default_factory=dict)   # name -> flat float32 array (logp, kl, img, prompt, gen_attn)
    checks: dict = field(default_factory=dict)

    @property
    def n_questions(self) -> int:
        return len(self.qids)

    def row(self, qi: int, rid: int) -> int:
        return qi * N_ROLLOUTS + rid

    def tokens(self, name: str, row: int) -> np.ndarray:
        s = self.tok_start[row]
        return self.tok[name][s:s + self.n_tokens[row]]


def _sorted_table(path: Path, columns=None) -> pa.Table:
    t = pq.read_table(path, columns=columns)
    return t.sort_by([("sample_id", "ascending"), ("rollout_id", "ascending")])


def _flat(t: pa.Table, col: str) -> tuple[np.ndarray, np.ndarray]:
    arr = t.column(col).combine_chunks()
    lens = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    vals = arr.flatten().to_numpy(zero_copy_only=False)
    return vals, lens


def load(ds: str, with_tokens: bool = True, cfg: dict | None = None) -> Rollouts:
    """Join generations / probability_metrics (/ attention_metrics) on (dataset, sample_id, rollout_id)
    and verify that every per-token list aligns with generated_token_ids."""
    cfg = cfg or config()
    d = dataset_dir(ds, cfg)
    g = _sorted_table(d / "generations.parquet")
    keys = g.select(KEYS).to_pandas()
    checks = {"rows": len(keys)}
    qids = sorted(keys.sample_id.unique().tolist())
    assert keys.sample_id.tolist() == [q for q in qids for _ in range(N_ROLLOUTS)], "unexpected question layout"
    assert keys.rollout_id.tolist() == list(range(N_ROLLOUTS)) * len(qids), "rollout ids must be 0..31"
    assert keys[KEYS].duplicated().sum() == 0
    tid_vals, tid_lens = _flat(g, "generated_token_ids")
    n_tok = g.column("n_generated_text_tokens").to_numpy().astype(np.int64)
    checks["token_ids_len_eq_n_generated"] = bool((tid_lens == n_tok).all())
    assert checks["token_ids_len_eq_n_generated"]
    gen = g.drop_columns(["generated_token_ids"]).to_pandas()
    gen["extracted_answer"] = gen["extracted_answer"].astype(object).where(gen["extracted_answer"].notna(), None)
    tok_start = np.concatenate([[0], np.cumsum(n_tok)[:-1]]).astype(np.int64)
    r = Rollouts(ds=ds, qids=qids, gen=gen, tok_start=tok_start, n_tokens=n_tok, checks=checks)
    if not with_tokens:
        return r

    p = _sorted_table(d / "probability_metrics.parquet")
    assert p.select(KEYS).to_pandas().equals(keys), "probability_metrics keys differ from generations"
    for col, name in [("logp", "logp"), ("kl_u_p", "kl")]:
        v, lens = _flat(p, col)
        checks[f"{col}_aligned"] = bool((lens == n_tok).all())
        assert checks[f"{col}_aligned"], col
        r.tok[name] = v.astype(np.float32)
    del p
    if (d / "attention_metrics.parquet").exists():
        a = _sorted_table(d / "attention_metrics.parquet")
        assert a.select(KEYS).to_pandas().equals(keys), "attention_metrics keys differ from generations"
        for col, name in [("img_attn_total", "img"), ("prompt_text_attn_total", "prompt"),
                          ("generated_text_attn_total", "gen_attn")]:
            v, lens = _flat(a, col)
            checks[f"{col}_aligned"] = bool((lens == n_tok).all())
            assert checks[f"{col}_aligned"], col
            r.tok[name] = v.astype(np.float32)
        pos, lens = _flat(a, "n_generated_text_tokens")   # per-token preceding counts 0..T-1
        expect = np.arange(len(pos)) - np.repeat(tok_start, n_tok)
        checks["attn_preceding_counts_are_0_to_T-1"] = bool((lens == n_tok).all() and (pos == expect).all())
        checks["first_token_generated_attention_zero"] = bool(
            np.all(r.tok["gen_attn"][tok_start[n_tok > 0]] == 0))
        den = r.tok["img"] + r.tok["prompt"] + r.tok["gen_attn"]
        checks["attn_denominator_min"] = float(den.min())
        checks["attn_group_sum_max"] = float(den.max())
        r.gen["n_image_tokens"] = a.column("n_image_tokens").to_numpy()
        r.gen["n_prompt_text_tokens"] = a.column("n_prompt_text_tokens").to_numpy()
        del a
    checks["zero_length_rollouts"] = int((n_tok == 0).sum())
    checks["retained_tokens"] = int(n_tok.sum())
    checks["capped_rollouts"] = int(r.gen.hit_max_new_tokens.sum())
    checks["null_extracted"] = int(r.gen.extracted_answer.isna().sum())
    return r


# ----------------------------------------------------------------------------------------------- metadata
def mathvista_meta(cfg: dict | None = None) -> pd.DataFrame:
    """Released MathVista testmini metadata (pinned revision), without images. Cached next to the data."""
    cfg = cfg or config()
    mc = cfg["data"]["mathvista_meta"]
    cache = data_root(cfg).parent / "mathvista_testmini_meta.parquet"
    if not cache.exists():
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(mc["repo"], mc["file"], repo_type="dataset", revision=mc["revision"])
        pq.write_table(pq.read_table(f, columns=mc["columns"]), cache)
    m = pq.read_table(cache).to_pandas().set_index("pid")
    m["choices"] = m["choices"].map(lambda c: None if c is None else [str(x) for x in c])
    return m


# ------------------------------------------------------------------------------------------------- splits
def make_splits(cfg: dict | None = None) -> dict:
    """Sort the 1,000 question IDs lexicographically, permute with default_rng(seed); first n_dev = dev."""
    cfg = cfg or config()
    out = {"seed": cfg["seed"], "rule": "sorted(ids) permuted by np.random.default_rng(seed).permutation; "
                                        "first n_dev = development, rest = held-out evaluation"}
    for ds in DATASETS:
        ids = sorted(pq.read_table(dataset_dir(ds, cfg) / "generations.parquet", columns=["sample_id"])
                     .column("sample_id").unique().to_pylist())
        assert len(ids) == 1000
        perm = np.random.default_rng(cfg["seed"]).permutation(len(ids))
        order = [ids[i] for i in perm]
        out[ds] = {"permuted_ids": order, "dev": order[:cfg["splits"]["n_dev"]],
                   "eval": order[cfg["splits"]["n_dev"]:]}
    out["live_math"] = out["math"]["eval"][:cfg["splits"]["n_live"]]
    return out


def make_orders(qids: list, ds_index: int, cfg: dict | None = None) -> np.ndarray:
    """(Q, n_orders, 32) rollout-ID permutations; rng = default_rng([seed, ds_index]), questions in sorted order."""
    cfg = cfg or config()
    rng = np.random.default_rng([cfg["seed"], ds_index])
    n = cfg["offline"]["n_orders"]
    return np.stack([np.stack([rng.permutation(N_ROLLOUTS) for _ in range(n)]) for _ in qids]).astype(np.int16)


def save_orders(orders: dict, path: Path = SPLIT_DIR / "orders.json.gz") -> None:
    with gzip.open(path, "wt") as f:
        json.dump({ds: {"qids": o["qids"], "orders": o["orders"].tolist()} for ds, o in orders.items()}, f)


def load_orders(path: Path = SPLIT_DIR / "orders.json.gz") -> dict:
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return {ds: {"qids": v["qids"], "orders": np.asarray(v["orders"], dtype=np.int16)} for ds, v in d.items()}
