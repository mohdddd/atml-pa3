"""Task 1 data: MetaMathQA selection + tokenisation, training batches, and the activation-probe set.

Formatting: one user turn (MetaMathQA `query`) and one assistant turn (`response`) rendered with
Qwen3's chat template in non-thinking mode. The prompt part is exactly the generation prompt
(`...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`), so training matches inference. Supervised
tokens = response tokens + the end-of-turn token <|im_end|>; prompt and padding are masked (-100).
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata

import numpy as np
import torch

from pa3 import SEED
from pa3.common.paths import SPLITS, scratch_dir
from pa3.common.runlog import read_json, write_json

IGNORE = -100


def normalize_question(text: str) -> str:
    """Normalisation used for GSM8K-test decontamination: NFKC, lower-case,
    non-alphanumerics -> space, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def prompt_text(tok, query: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": query}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_example(tok, query: str, response: str, im_end_id: int) -> tuple[list[int], int]:
    p = tok(prompt_text(tok, query), add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"]
    return p + r + [im_end_id], len(p)


def selection_path():
    return SPLITS / "task1" / "metamath_selection.json"


def tokenized_cache_path():
    return scratch_dir("task1") / "metamath_tokenized.pt"


def build_selection(cfg: dict, tok) -> dict:
    """Scan MetaMathQA in source order and keep the first n eligible examples."""
    from datasets import load_dataset

    dcfg = cfg["data"]
    t0 = time.time()
    gsm = load_dataset(*dcfg["gsm8k_test"][:2], split=dcfg["gsm8k_test"][2])
    gsm_norm = {normalize_question(q) for q in gsm["question"]}
    ds = load_dataset(dcfg["dataset"], split=dcfg["split"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    keep, ids_list, nprompt_list = [], [], []
    excl = {"empty": 0, "too_long": 0, "gsm8k_test_match": 0}
    n_scanned = 0
    for i, ex in enumerate(ds):
        if len(keep) >= dcfg["n_examples"]:
            break
        n_scanned += 1
        q, r = ex.get("query") or "", ex.get("response") or ""
        if not q.strip() or not r.strip():
            excl["empty"] += 1
            continue
        if normalize_question(q) in gsm_norm or normalize_question(ex.get("original_question") or "") in gsm_norm:
            excl["gsm8k_test_match"] += 1
            continue
        ids, npr = encode_example(tok, q, r, im_end)
        if len(ids) > dcfg["max_len"]:
            excl["too_long"] += 1
            continue
        keep.append(i)
        ids_list.append(ids)
        nprompt_list.append(npr)
    if len(keep) < dcfg["n_examples"]:
        raise RuntimeError(f"only {len(keep)} eligible examples")

    lengths = np.array([len(x) for x in ids_list])
    resp_tokens = lengths - np.array(nprompt_list)
    digest = hashlib.sha256(np.array(keep, dtype=np.int64).tobytes()).hexdigest()
    sel = {
        "dataset": dcfg["dataset"], "split": dcfg["split"], "n_selected": len(keep), "n_scanned": n_scanned,
        "excluded": excl, "normalization": "NFKC, lowercase, [^0-9a-z]+ -> ' ', collapse whitespace",
        "tokenizer": cfg["model"], "format": "chat template, enable_thinking=False; loss on response + <|im_end|>",
        "length_stats": {"total_tokens": int(lengths.sum()), "response_tokens": int(resp_tokens.sum()),
                         "mean": float(lengths.mean()), "p50": float(np.median(lengths)),
                         "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max())},
        "indices_sha256": digest, "seconds": round(time.time() - t0, 1),
        "source_indices": keep,
    }
    write_json(selection_path(), sel)
    torch.save({"input_ids": ids_list, "n_prompt": nprompt_list, "indices_sha256": digest}, tokenized_cache_path())
    return sel


def load_tokenized(cfg: dict, tok) -> dict:
    p = tokenized_cache_path()
    if p.exists():
        d = torch.load(p)
        if selection_path().exists() and read_json(selection_path())["indices_sha256"] == d["indices_sha256"]:
            return d
    build_selection(cfg, tok)
    return torch.load(p)


def collate(examples: list[tuple[list[int], int]], pad_id: int) -> dict:
    L = max(len(ids) for ids, _ in examples)
    B = len(examples)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for b, (ids, npr) in enumerate(examples):
        t = torch.tensor(ids)
        input_ids[b, : len(ids)] = t
        labels[b, npr: len(ids)] = t[npr:]
        attn[b, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def effective_batch(data: dict, update: int, cfg: dict, pad_id: int) -> list[dict]:
    """Microbatches of optimizer update `update` (0-based): examples [32u, 32u+32) in source order."""
    tc = cfg["train"]
    eb = tc["micro_batch"] * tc["grad_accum"]
    start = update * eb
    ex = [(data["input_ids"][i], data["n_prompt"][i]) for i in range(start, start + eb)]
    return [collate(ex[j: j + tc["micro_batch"]], pad_id) for j in range(0, eb, tc["micro_batch"])]


# ---------------------------------------------------------------- activation-probe set
def probe_path():
    return SPLITS / "task1" / "probe_set.json"


def build_probe_set(cfg: dict) -> dict:
    """32 GSM8K test questions + 32 HellaSwag validation contexts, sampled with seed 6304 before any
    model results are inspected. IDs are lm-eval doc_ids (= dataset row indices)."""
    if probe_path().exists():
        return read_json(probe_path())
    rng = np.random.default_rng(SEED)
    gsm_ids = sorted(rng.choice(1319, size=cfg["probe"]["n_math"], replace=False).tolist())
    hs_ids = sorted(rng.choice(10042, size=cfg["probe"]["n_nonmath"], replace=False).tolist())
    probe = {"seed": SEED, "gsm8k_test_doc_ids": gsm_ids, "hellaswag_validation_doc_ids": hs_ids,
             "rule": "np.random.default_rng(6304): choice(1319, 32) then choice(10042, 32), no replacement"}
    write_json(probe_path(), probe)
    return probe


def probe_prompts(tok) -> list[dict]:
    """Probe prompts in the benchmark formats: GSM8K as the lm-eval gsm8k_cot 0-shot chat prompt,
    HellaSwag as the stock lm-eval completion context (no chat template)."""
    from lm_eval.tasks import TaskManager, get_task_dict

    probe = read_json(probe_path())
    td = get_task_dict(["gsm8k_cot", "hellaswag"], TaskManager())
    gsm_docs = list(td["gsm8k_cot"].test_docs())
    hs_docs = list(td["hellaswag"].validation_docs())
    out = []
    for i in probe["gsm8k_test_doc_ids"]:
        text = td["gsm8k_cot"].doc_to_text(gsm_docs[i])
        out.append({"kind": "math", "doc_id": i, "prompt": prompt_text(tok, text)})
    for i in probe["hellaswag_validation_doc_ids"]:
        out.append({"kind": "nonmath", "doc_id": i, "prompt": td["hellaswag"].doc_to_text(hs_docs[i])})
    return out
