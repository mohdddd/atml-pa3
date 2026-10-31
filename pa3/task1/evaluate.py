"""Evaluate one Task 1 condition: benchmarks (lm-eval), qualitative generations, diagnostics.

Conditions: baseline, full_ft_mid (update 625), full_ft, l2sp, lora4, lora16, oplora16.
All models are evaluated in FP16 with dropout disabled. Adapter runs are merged into a fresh FP16
backbone using the effective (for OPLoRA: projected) factors.

Outputs -> results/task1/eval/<condition>/
  metrics.json                  aggregate scores (primary + raw/normalised accuracies, GSM8K extraction/truncation)
  gsm8k_samples.jsonl.gz        per question: response, extracted answers (strict / flexible), correctness,
                                generated-token count, truncation flag
  <task>_samples.jsonl.gz       per example log-likelihoods and correctness (hellaswag, boolq, arc_easy)
  lm_eval_config.json           task configs / versions used
  qualitative.jsonl             24 greedy responses (512-token cap), grouped by question and format
  diagnostics.json              down_proj layers 0/13/27: rho_k, eta, activation ratio R(z) math vs non-math
  diagnostics_per_prompt.jsonl.gz

Usage: python -m pa3.task1.evaluate --cond lora16
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pa3.common import hub
from pa3.common.paths import results_dir, scratch_dir
from pa3.common.runlog import load_config, read_json, write_json, write_jsonl
from pa3.common.seed import set_seed
from pa3.task1 import config as C
from pa3.task1 import lora as L
from pa3.task1.data import build_probe_set, probe_prompts, prompt_text
from pa3.task1.modeling import DEVICE, gpu_peak_gb, load_base, load_full_ft_weights, load_tokenizer, reset_peak
from pa3.task1.train import weights_prefix

DEV = DEVICE


def _empty():
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- artefacts
def weights_dir(run: str, update: int):
    """Local dir with the saved weights of run@update (downloaded from the Hub if not on this machine)."""
    local = scratch_dir(C.results_task(), run) / f"u{update}"
    if local.exists() and any(local.iterdir()):
        return local
    root = scratch_dir(C.results_task(), "_hub")
    p = hub.download(weights_prefix(run, update), root)
    if p is None:
        raise FileNotFoundError(f"weights for {run}@u{update} not found locally or on the Hub")
    return p


def run_method(run: str) -> dict:
    return C.run_config(run)


def build_eval_model(cond: dict, cfg: dict):
    """FP16 eval model for a condition (+ FP32 delta-W accessor for diagnostics)."""
    if cond["run"] is None:
        return load_base(cfg["model"], torch.float16, DEV).eval(), None
    rc = run_method(cond["run"])
    d = weights_dir(cond["run"], cond["update"])
    if rc["method"] == "full":
        m = load_base(cfg["model"], torch.float32, DEV)
        load_full_ft_weights(m, d / "model.safetensors")
        return m.half().eval(), {"kind": "full", "path": d / "model.safetensors"}
    from safetensors.torch import load_file

    meta = read_json(d / "adapter_meta.json")
    eff = load_file(str(d / "adapter_effective.safetensors"))
    m = load_base(cfg["model"], torch.float16, DEV)
    L.merge_into(m, eff, meta["scale"])
    return m.eval(), {"kind": "adapter", "factors": eff, "scale": meta["scale"]}


# ---------------------------------------------------------------- benchmarks
def _merge_gsm8k_samples(samples: list, tok, max_gen: int) -> list[dict]:
    by = {}
    for s in samples:
        r = by.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"]})
        resp = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        r["response"] = resp
        key = "flexible" if "flexible" in s["filter"] else "strict"
        r[f"{key}_extracted"] = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        r[f"{key}_correct"] = float(s["exact_match"])
    for r in by.values():
        n = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        r["n_generated_tokens"] = n
        # lm-eval returns decoded text (stop sequence removed); re-tokenised length within 1 token of the cap
        # means generation hit max_gen_toks.
        r["truncated"] = n >= max_gen - 1
        r["flexible_unextractable"] = r.get("flexible_extracted") == "[invalid]"
        r["strict_unextractable"] = r.get("strict_extracted") == "[invalid]"
    return [by[k] for k in sorted(by)]


def _ll_rows(samples: list, task: str) -> list[dict]:
    rows = []
    for s in samples:
        r = {"doc_id": s["doc_id"], "target": s["target"],
             "loglikelihoods": [x[0] if isinstance(x, (list, tuple)) else x for x in s["filtered_resps"]],
             "acc": s.get("acc")}
        if "acc_norm" in s:
            r["acc_norm"] = s["acc_norm"]
        rows.append(r)
    return sorted(rows, key=lambda r: r["doc_id"])


def run_benchmarks(model, tok, cfg: dict, out) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    ec = cfg["eval"]
    limit = ec.get("limit")
    seeds = dict(random_seed=6304, numpy_random_seed=6304, torch_random_seed=6304, fewshot_random_seed=6304)
    metrics, timing, configs = {}, {}, {}

    t0 = time.perf_counter()
    bs_gen = ec["batch_size_gen"] if DEV == "cuda" else 8
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs_gen, enable_thinking=False)
    r = simple_evaluate(model=lm, tasks=ec["tasks_gen"], num_fewshot=0, apply_chat_template=True,
                        fewshot_as_multiturn=False, gen_kwargs=f"max_gen_toks={ec['gsm8k_max_gen_toks']}",
                        log_samples=True, limit=limit, **seeds)
    timing["gsm8k_seconds"] = time.perf_counter() - t0
    res = r["results"]["gsm8k_cot"]
    rows = _merge_gsm8k_samples(r["samples"]["gsm8k_cot"], tok, ec["gsm8k_max_gen_toks"])
    write_jsonl(out / "gsm8k_samples.jsonl.gz", rows)
    metrics["gsm8k"] = {
        "primary": "exact_match,flexible-extract",
        "exact_match_flexible": res["exact_match,flexible-extract"],
        "exact_match_flexible_stderr": res.get("exact_match_stderr,flexible-extract"),
        "exact_match_strict": res["exact_match,strict-match"],
        "n": len(rows),
        "truncation_rate": float(np.mean([x["truncated"] for x in rows])),
        "flexible_unextractable_rate": float(np.mean([x["flexible_unextractable"] for x in rows])),
        "strict_unextractable_rate": float(np.mean([x["strict_unextractable"] for x in rows])),
        "mean_generated_tokens": float(np.mean([x["n_generated_tokens"] for x in rows])),
        "accuracy_on_non_truncated": float(np.mean([x["flexible_correct"] for x in rows if not x["truncated"]] or [0])),
    }
    configs["gsm8k_cot"] = r["configs"]["gsm8k_cot"]
    del lm

    t0 = time.perf_counter()
    bs_ll = ec["batch_size_ll"] if DEV == "cuda" else 2  # CPU path: smoke tests only
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs_ll, max_batch_size=ec.get("max_batch_size_ll", 64))
    r = simple_evaluate(model=lm, tasks=ec["tasks_ll"], num_fewshot=0, apply_chat_template=False,
                        fewshot_as_multiturn=False, log_samples=True, limit=limit, **seeds)
    timing["loglikelihood_seconds"] = time.perf_counter() - t0
    for t in ec["tasks_ll"]:
        res = r["results"][t]
        m = {"acc": res["acc,none"], "acc_stderr": res.get("acc_stderr,none")}
        if "acc_norm,none" in res:
            m.update(acc_norm=res["acc_norm,none"], acc_norm_stderr=res.get("acc_norm_stderr,none"))
        m["primary"] = "acc" if t == "boolq" else "acc_norm"
        m["n"] = r["n-samples"][t]["effective"]
        metrics[t] = m
        configs[t] = r["configs"][t]
        write_jsonl(out / f"{t}_samples.jsonl.gz", _ll_rows(r["samples"][t], t))
    import lm_eval

    write_json(out / "lm_eval_config.json", {"lm_eval_version": lm_eval.__version__, "limit": limit,
                                             "task_configs": configs, "versions": r.get("versions")})
    return {"metrics": metrics, "timing": timing}


# ---------------------------------------------------------------- qualitative
def qual_prompts(cfg: dict) -> list[dict]:
    from pa3.common.paths import REPO

    qs = load_config(REPO / cfg["eval"]["qual_prompts"])
    out = []
    for q in qs:
        qid = q["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid.removesuffix("_json").removesuffix("_template")
        out.append({"id": qid, "question": base, "format": fmt, "system": q.get("system"),
                    "user": q["paraphrases"][0]})
    return out


@torch.no_grad()
def run_qualitative(model, tok, cfg: dict, out) -> dict:
    t0 = time.perf_counter()
    qs = qual_prompts(cfg)
    tok.padding_side = "left"
    texts = [prompt_text(tok, q["user"], q["system"]) for q in qs]
    b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    cap = cfg["eval"]["qual_max_new_tokens"]
    gen = model.generate(**b, max_new_tokens=cap, do_sample=False, use_cache=True,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for q, text, g in zip(qs, texts, gen[:, b["input_ids"].shape[1]:]):
        g = g.tolist()
        n = g.index(eos) + 1 if eos in g else len([t for t in g if t != tok.pad_token_id])
        rows.append(dict(q, prompt=text, response=tok.decode(g[:n], skip_special_tokens=True),
                         n_generated_tokens=n, truncated=eos not in g))
    order = {"plain": 0, "json": 1, "template": 2}
    qorder = list(dict.fromkeys(r["question"] for r in rows))
    rows.sort(key=lambda r: (qorder.index(r["question"]), order[r["format"]]))
    write_jsonl(out / "qualitative.jsonl", rows)
    return {"qualitative_seconds": time.perf_counter() - t0, "n": len(rows),
            "truncated": int(sum(r["truncated"] for r in rows))}


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def cache_probe_activations(cfg: dict, tok) -> dict:
    """Starting model (FP32) down_proj inputs at the last prompt token, for the probe set. Cached on disk."""
    path = scratch_dir(C.results_task(), "_cache") / "probe_z.pt"
    if path.exists():
        return torch.load(path)
    build_probe_set(cfg)
    prompts = probe_prompts(tok)
    layers = cfg["diagnostics"]["layers"]
    base = load_base(cfg["model"], torch.float32, DEV).eval()
    store: dict[int, list] = {l: [] for l in layers}
    hooks = []
    for l in layers:
        mod = base.model.layers[l].mlp.down_proj
        hooks.append(mod.register_forward_hook(lambda m, inp, o, l=l: store[l].append(inp[0][0, -1].float().cpu())))
    for p in prompts:
        ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).to(DEV)
        base.model(**ids)  # no padding: the last position is the last prompt token
    for h in hooks:
        h.remove()
    W0 = {l: base.model.layers[l].mlp.down_proj.weight.detach().float().cpu() for l in layers}
    z = {"z": {l: torch.stack(store[l]) for l in layers}, "W0": W0,
         "kinds": [p["kind"] for p in prompts], "doc_ids": [p["doc_id"] for p in prompts]}
    torch.save(z, path)
    del base
    _empty()
    return z


@torch.no_grad()
def delta_w(acc: dict, layer: int, W0: torch.Tensor) -> torch.Tensor:
    if acc["kind"] == "full":
        from safetensors import safe_open

        with safe_open(str(acc["path"]), framework="pt") as f:
            W = f.get_tensor(f"model.layers.{layer}.mlp.down_proj.weight").float()
        return W - W0
    name = f"model.layers.{layer}.mlp.down_proj"
    A, B = acc["factors"][f"{name}.A"].float(), acc["factors"][f"{name}.B"].float()
    return acc["scale"] * (B @ A)


@torch.no_grad()
def run_diagnostics(acc: dict, cfg: dict, tok, out) -> dict:
    t0 = time.perf_counter()
    cache = cache_probe_activations(cfg, tok)
    k = cfg["diagnostics"]["k"]
    kinds = np.array(cache["kinds"])
    res, per = {}, []
    for l in cfg["diagnostics"]["layers"]:
        W0 = cache["W0"][l].to(DEV)
        dW = delta_w(acc, l, cache["W0"][l]).to(DEV)
        U, S, Vh = L.svd_fp32(W0)
        Uk = U[:, :k]
        fro2 = float(dW.pow(2).sum())
        rho = float((Uk.T @ dW).pow(2).sum()) / fro2 if fro2 > 0 else float("nan")
        eta = float(dW.norm() / W0.norm())
        z = cache["z"][l].to(DEV)                       # [64, 3072]
        R = ((z @ dW.T).norm(dim=1) / ((z @ W0.T).norm(dim=1) + 1e-8)).cpu().numpy()
        res[str(l)] = {"rho_k": rho, "eta": eta, "dW_fro": fro2 ** 0.5, "W0_fro": float(W0.norm()), "k": k,
                       "R_math_mean": float(R[kinds == "math"].mean()), "R_math_median": float(np.median(R[kinds == "math"])),
                       "R_nonmath_mean": float(R[kinds == "nonmath"].mean()),
                       "R_nonmath_median": float(np.median(R[kinds == "nonmath"])),
                       "R_ratio_math_over_nonmath": float(R[kinds == "math"].mean() / R[kinds == "nonmath"].mean())}
        per += [{"layer": l, "kind": kd, "doc_id": d, "R": float(r)} for kd, d, r in zip(cache["kinds"], cache["doc_ids"], R)]
    write_json(out / "diagnostics.json", {"layers": res, "module": "mlp.down_proj", "precision": "fp32",
                                          "seconds": time.perf_counter() - t0})
    write_jsonl(out / "diagnostics_per_prompt.jsonl.gz", per)
    return res


# ---------------------------------------------------------------- main
def evaluate(cond_name: str) -> int:
    cfg = C.common_config()
    cond = C.condition(cond_name)
    set_seed(cfg["seed"])
    out = results_dir(C.results_task(), "eval", cond_name)
    tok = load_tokenizer(cfg["model"])
    reset_peak()
    t0 = time.perf_counter()
    model, acc = build_eval_model(cond, cfg)
    info = {"condition": cond_name, "run": cond["run"], "update": cond["update"],
            "load_seconds": time.perf_counter() - t0, "dtype": "float16", "dry_run": C.is_dry()}
    tok.padding_side = "left"
    b = run_benchmarks(model, tok, cfg, out)
    info["timing"] = b["timing"]
    write_json(out / "metrics.json", dict(info, **b["metrics"]))
    info["qualitative"] = run_qualitative(model, tok, cfg, out)
    del model
    _empty()
    if acc is not None:
        info["diagnostics"] = run_diagnostics(acc, cfg, tok, out)
    info["peak_gpu_alloc_gb"] = gpu_peak_gb()[0]
    info["total_seconds"] = time.perf_counter() - t0
    write_json(out / "metrics.json", dict(info, **b["metrics"]))
    from pa3.task1.train import _push

    _push(out, f"task1: eval {cond_name}")
    print(f"[eval:{cond_name}] done in {info['total_seconds'] / 60:.1f} min: "
          + ", ".join(f"{t}={v.get('exact_match_flexible', v.get('acc_norm', v.get('acc'))):.4f}"
                      for t, v in b["metrics"].items()), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=C.CONDITIONS)
    evaluate(ap.parse_args().cond)


if __name__ == "__main__":
    main()
