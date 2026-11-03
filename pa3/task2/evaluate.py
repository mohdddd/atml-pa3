"""Task 2 final evaluation of the completed Stage-3 model on both final-test suites.

Conditions (question and original references fixed): correct image; complementary swap (the partner
image of the pair, whose answer differs); mismatched (fixed mapping to an image of another pair of the same
suite, splits/task2/mismatched.json); no image (visual tokens omitted, masks/positions rebuilt).
Greedy decoding, thinking disabled, 16 new tokens, batch size 8.

Scoring: grounding - normalised exact match against the target (presence / counting reported
separately); natural VQAv2 - local soft consensus accuracy against all ten human answers (yes/no,
number, other separately). Swapped / mismatched / no-image predictions are always scored against the
ORIGINAL references. Empty outputs stay in every denominator.
Per suite and condition: accuracy, pair accuracy (both predictions exactly match their original canonical
answers), identical-answer rate (both predictions of a pair equal after normalisation), empty-answer rate,
mean reference-answer NLL (teacher-forced, termination excluded), generation time per answer, and for
the complementary swap whether the prediction follows the partner image's answer.
95% CIs: 4,000 bootstrap resamples of complete pairs; differences between conditions use the same
resampled pair IDs for both conditions.

Usage: python -m pa3.task2.evaluate --variant t196
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from pa3 import SEED
from pa3.common import hub
from pa3.common.paths import results_dir, scratch_dir
from pa3.common.runlog import write_json, write_jsonl
from pa3.task2 import config as C
from pa3.task2 import data as D
from pa3.task2.features import FeatureStore
from pa3.task2.model import DEVICE, VLM, autocast
from pa3.task2.train import feature_mode, pair_examples, score_item

CONDS = ["correct", "complementary", "mismatched", "no_image"]


def load_final(vlm: VLM, cfg: dict, variant: str, stage: int = 3) -> str:
    from safetensors.torch import load_file

    d = scratch_dir(C.results_task(), variant) / f"stage{stage}"
    src = "local"
    if not (d / "connector.safetensors").exists():
        got = hub.download(f"{C.hub_prefix()}/{variant}/stage{stage}", scratch_dir(C.results_task(), variant, "hub"))
        if got is None:
            raise FileNotFoundError(f"stage{stage} weights of {variant} neither local nor on the Hub")
        d, src = got, "hub"
    sd = load_file(str(d / "connector.safetensors"))
    if (d / "adapter.safetensors").exists():
        vlm.add_lora(cfg["seed"] + 1000)
        sd.update(load_file(str(d / "adapter.safetensors")))
    vlm.load_trainable_state(sd)
    vlm.to(DEVICE).eval()
    return f"{src}:{d}"


@torch.no_grad()
def run_suite(vlm: VLM, cfg: dict, store: FeatureStore, suite: str, pairs: list[dict], mism: dict) -> list[dict]:
    bs = cfg["eval"]["batch_size"]
    items = {it["qid"]: it for p in pairs for it in p["items"]}
    partner = {it["qid"]: p["items"][1 - j] for p in pairs for j, it in enumerate(p["items"])}
    recs = []
    for cond in CONDS:
        exs = [e for p in pairs for e in pair_examples(cfg, p, cond, mism)]
        for i in range(0, len(exs), bs):
            chunk = exs[i:i + bs]
            keys = [e.image for e in chunk if e.image is not None]
            feats = store.get(keys) if keys else None
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gens = vlm.generate(chunk, feats, cfg["eval"]["max_new_tokens"])
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / len(chunk)
            with autocast():
                nll = vlm.example_losses(vlm.build(chunk, True), feats, include_eot=False).float().tolist()
            for e, g, n in zip(chunk, gens, nll):
                it = items[e.meta["qid"]]
                sc = score_item(suite, g["text"], it)
                pn = sc["pred_norm"]
                recs.append({"suite": suite, "condition": cond, "pair_id": e.meta["pair_id"], "qid": e.meta["qid"],
                             "type": e.meta["type"], "pos": e.meta["pos"], "image": e.image,
                             "question": it.get("question", next(p for p in pairs if p["pair_id"] == e.meta["pair_id"])["question"]),
                             "reference": it["answer"], "canonical": it["canonical"], "refs": it["refs"],
                             "partner_canonical": partner[e.meta["qid"]]["canonical"],
                             "pred": g["text"], **sc, "nll": n, "gen_tokens": g["n_tokens"],
                             "truncated": g["truncated"], "gen_seconds": dt,
                             "follows_partner": pn == partner[e.meta["qid"]]["canonical"]})
        print(f"[eval] {suite}/{cond}: acc={np.mean([r['score'] for r in recs if r['condition'] == cond]):.3f}",
              flush=True)
    return recs


# ----------------------------------------------------------------------------- metrics
def pair_table(recs: list[dict]) -> dict:
    """cond -> pair_id -> {'score','both','identical','empty','nll','follow','type'}"""
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["condition"]][r["pair_id"]].append(r)
    out = {}
    for cond, d in by.items():
        out[cond] = {}
        for pid, rs in d.items():
            rs = sorted(rs, key=lambda r: r["pos"])
            out[cond][pid] = {"type": rs[0]["type"], "score": np.mean([r["score"] for r in rs]),
                              "both": float(all(r["exact"] for r in rs)),
                              "identical": float(rs[0]["pred_norm"] == rs[1]["pred_norm"]),
                              "empty": np.mean([r["empty"] for r in rs]), "nll": np.mean([r["nll"] for r in rs]),
                              "follow": np.mean([r["follows_partner"] for r in rs]),
                              "truncated": np.mean([r["truncated"] for r in rs]),
                              "gen_seconds": np.mean([r["gen_seconds"] for r in rs])}
    return out


METRICS = ["score", "both", "identical", "empty", "nll", "follow", "truncated", "gen_seconds"]
NAMES = {"score": "accuracy", "both": "pair_accuracy", "identical": "identical_answer_rate",
         "empty": "empty_answer_rate", "nll": "mean_reference_nll", "follow": "follows_partner_rate",
         "truncated": "truncation_rate", "gen_seconds": "generation_seconds_per_answer"}
DIFFS = [("correct", "complementary"), ("correct", "mismatched"), ("correct", "no_image")]


def summarize(recs: list[dict], n_boot: int, seed: int = SEED) -> dict:
    pt = pair_table(recs)
    pids = sorted(pt["correct"])
    types = sorted({pt["correct"][p]["type"] for p in pids})
    rng = np.random.default_rng(seed)
    out = {}
    for t in [None] + types:
        sel = [p for p in pids if t is None or pt["correct"][p]["type"] == t]
        arr = {c: {m: np.array([pt[c][p][m] for p in sel], dtype=float) for m in METRICS} for c in CONDS}
        idx = rng.integers(0, len(sel), size=(n_boot, len(sel)))
        res = {"n_pairs": len(sel), "n_questions": 2 * len(sel)}
        for c in CONDS:
            res[c] = {}
            for m in METRICS:
                v = arr[c][m]
                bs = v[idx].mean(1)
                res[c][NAMES[m]] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                       float(np.percentile(bs, 97.5))]}
        for a, b in DIFFS:
            key = f"{a}_minus_{b}"
            res[key] = {}
            for m, sign in (("score", 1), ("both", 1), ("nll", -1)):
                v = sign * (arr[a][m] - arr[b][m])   # NLL: b - a (positive = correct image makes reference likelier)
                bs = v[idx].mean(1)
                name = "nll_advantage(b_minus_a)" if m == "nll" else NAMES[m] + "_advantage"
                res[key][name] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                    float(np.percentile(bs, 97.5))]}
        out["all" if t is None else t] = res
    # complementary swap: does the prediction follow the partner image's answer?
    comp = [r for r in recs if r["condition"] == "complementary"]
    corr = {r["qid"]: r for r in recs if r["condition"] == "correct"}
    partner_q = {}
    for r in corr.values():
        partner_q.setdefault(r["pair_id"], []).append(r["qid"])
    other = {q: next(x for x in partner_q[corr[q]["pair_id"]] if x != q) for q in corr}
    out["swap_outcomes"] = {
        "follows_partner": float(np.mean([r["follows_partner"] for r in comp])),
        "keeps_original_answer": float(np.mean([r["pred_norm"] == r["canonical"] for r in comp])),
        "other_answer": float(np.mean([r["pred_norm"] not in (r["canonical"], r["partner_canonical"]) for r in comp])),
        # image-insensitivity: the answer did not change when the image was swapped
        "unchanged_from_correct_image_prediction": float(np.mean([r["pred_norm"] == corr[r["qid"]]["pred_norm"]
                                                                  for r in comp])),
        # sanity check: identical wording -> swap input of A == correct input of B (expect ~1.0)
        "equals_partner_correct_image_prediction": float(np.mean([r["pred_norm"] == corr[other[r["qid"]]]["pred_norm"]
                                                                  for r in comp]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    a = ap.parse_args()
    cfg = C.config(a.variant)
    out = results_dir(C.results_task(), a.variant, "eval")
    s = D.load_splits()
    store = FeatureStore(feature_mode())
    torch.manual_seed(cfg["seed"])
    vlm = VLM(cfg)
    src = load_final(vlm, cfg, a.variant)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    summary = {"variant": a.variant, "weights": src, "decoding": {"greedy": True, "max_new_tokens": cfg["eval"]["max_new_tokens"],
                                                                 "batch_size": cfg["eval"]["batch_size"],
                                                                 "enable_thinking": cfg["enable_thinking"]},
               "bootstrap_resamples": cfg["eval"]["bootstrap"]}
    all_recs = []
    for suite in C.SUITES:
        pairs = s[suite]["test"]
        if cfg["eval"].get("limit_pairs"):
            pairs = pairs[:cfg["eval"]["limit_pairs"]]
        recs = run_suite(vlm, cfg, store, suite, pairs, s["mismatched"][suite])
        all_recs += recs
        summary[suite] = summarize(recs, cfg["eval"]["bootstrap"])
    summary["eval_seconds"] = time.perf_counter() - t0
    summary["generation_seconds_total"] = float(sum(r["gen_seconds"] for r in all_recs))
    summary["n_generated_answers"] = len(all_recs)
    summary["peak_gpu_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    summary["peak_gpu_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30 if DEVICE == "cuda" else 0.0
    summary["gpu"] = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"
    write_jsonl(out / "test_examples.jsonl.gz", all_recs)
    write_json(out / "metrics.json", summary)
    try:
        from pa3.common.gitsync import push

        push([out], f"task2: {a.variant} final evaluation")
    except Exception as e:
        print(f"[git] {e}")


if __name__ == "__main__":
    main()
