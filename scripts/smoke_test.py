"""Setup smoke test for Kaggle (T4 x2).

Verifies everything later tasks depend on and records it under results/env/:
  * environment / package versions / disks / GPUs           -> kaggle_env.json, pip_freeze.txt
  * Qwen3-0.6B loads in FP16, non-thinking chat template     -> smoke_test.json["qwen3"]
  * Full-FT memory/throughput benchmark (FP32 master + FP16
    autocast + GradScaler + grad checkpointing, microbatch 2) -> smoke_test.json["train_bench"]
  * SigLIP gives 196x768 patch features, final LN applied once -> smoke_test.json["siglip"]
  * lm-eval tasks + chat-template/thinking args available     -> smoke_test.json["lm_eval"]

Usage:  python scripts/smoke_test.py [--skip-train-bench]
"""
from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
import time
import traceback

import torch

from pa3.common.envinfo import env_report
from pa3.common.paths import results_dir
from pa3.common.runlog import write_json
from pa3.common.seed import set_seed

QWEN = "Qwen/Qwen3-0.6B"
SIGLIP = "google/siglip-base-patch16-224"


def section(name):
    def deco(fn):
        def wrapped(out, *a, **k):
            print(f"\n=== {name} ===", flush=True)
            t0 = time.perf_counter()
            try:
                out[name] = fn(*a, **k)
                out[name]["ok"] = True
            except Exception as e:  # keep going; record failure
                traceback.print_exc()
                out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            out[name]["seconds"] = round(time.perf_counter() - t0, 1)
            print({k: v for k, v in out[name].items() if k not in ("template_text",)}, flush=True)
        return wrapped
    return deco


@section("gpus")
def check_gpus():
    n = torch.cuda.device_count()
    res = {"count": n, "matmul_ok": []}
    for i in range(n):
        a = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.float16)
        torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize(i)
        tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
        res["matmul_ok"].append({"gpu": i, "fp16_matmul_tflops": round(tflops, 1)})
    return res


@section("secrets")
def check_secrets():
    res = {"git_credentials": os.path.exists(os.path.expanduser("~/.git-credentials"))}
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi

        res["hf_user"] = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
    else:
        res["hf_user"] = None
    return res


@section("qwen3")
def check_qwen3():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16).to("cuda:0").eval()
    msgs = [{"role": "user", "content": "What is 12 * 7? Answer briefly."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=False)
    reply = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "84"}], tokenize=False,
                                   enable_thinking=False)
    uniq = {p.data_ptr(): p.numel() for p in model.parameters()}
    res = {
        "template_text": text,
        "empty_think_block_in_prompt": "<think>\n\n</think>" in text,
        "train_format_example": full,
        "reply": reply,
        "eos_token": tok.eos_token,
        "im_end_id": tok.convert_tokens_to_ids("<|im_end|>"),
        "pad_token": tok.pad_token,
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "unique_params": sum(uniq.values()),
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }
    del model
    torch.cuda.empty_cache()
    return res


def _chunk_ce(head, h, y):
    logits = head(h).float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=-100, reduction="sum")


def chunked_lm_loss(model, x, labels, chunk=256):
    """Next-token CE without materialising [B, L, V] FP32 logits: lm_head + CE per sequence chunk,
    each chunk checkpointed (recomputed in backward)."""
    from torch.utils.checkpoint import checkpoint

    h = model.model(input_ids=x).last_hidden_state[:, :-1]
    y = labels[:, 1:]
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    for s in range(0, h.shape[1], chunk):
        total = total + checkpoint(_chunk_ce, model.lm_head, h[:, s:s + chunk], y[:, s:s + chunk],
                                   use_reentrant=False)
    return total / (y != -100).sum().clamp(min=1)


@section("train_bench")
def train_bench(cases=(("hf_default", 512), ("hf_default", 1024), ("chunked_ce", 1024), ("chunked_ce", 2048)),
                micro=2, steps=3):
    """Full FT step: FP32 master weights, FP16 autocast, dynamic loss scaling, grad checkpointing.
    hf_default = model(labels=...) (full FP32 logits); chunked_ce = chunked lm_head + CE."""
    from transformers import AutoModelForCausalLM

    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    res = {"device": dev, "micro_batch": micro, "results": []}
    for mode, L in cases:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            times = []
            for s in range(steps + 1):
                x = torch.randint(0, 150000, (micro, L), device=dev)
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=x, labels=x).loss if mode == "hf_default" else chunked_lm_loss(model, x, x)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(dev)
                if s > 0:  # first step is warm-up (optimizer state allocation)
                    times.append(time.perf_counter() - t0)
            st = sum(times) / len(times)
            res["results"].append({
                "loss_mode": mode, "seq_len": L, "sec_per_microstep": round(st, 3),
                "tokens_per_sec": round(micro * L / st),
                "peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2**30, 2),
            })
        except torch.OutOfMemoryError:
            res["results"].append({"loss_mode": mode, "seq_len": L, "oom": True,
                                   "peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)})
            loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(res["results"][-1], flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return res


@section("siglip")
def check_siglip():
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    vm = SiglipVisionModel.from_pretrained(SIGLIP, dtype=torch.float32).to("cuda:0").eval()
    img = Image.fromarray(np.random.default_rng(6304).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    px = proc(images=img, return_tensors="pt")["pixel_values"].to("cuda:0")
    with torch.no_grad():
        o = vm(pixel_values=px, output_hidden_states=True)
    last = o.last_hidden_state
    ln_name, post_ln = next((n, m) for n, m in vm.named_modules() if n.endswith("post_layernorm"))
    post_ln_of_last_hidden = post_ln(o.hidden_states[-1])
    res = {
        "post_layernorm_module": ln_name,
        "num_hidden_states": len(o.hidden_states),
        "pixel_values_shape": list(px.shape),
        "last_hidden_state_shape": list(last.shape),
        "last_hidden_equals_post_layernorm(hidden_states[-1])": bool(
            torch.allclose(last, post_ln_of_last_hidden, atol=1e-4)),
        "hidden_states[-1]_is_pre_ln": not torch.allclose(last, o.hidden_states[-1], atol=1e-4),
        "processor": {"size": getattr(proc, "size", None), "image_mean": proc.image_mean, "image_std": proc.image_std},
    }
    del vm
    torch.cuda.empty_cache()
    return res


@section("lm_eval")
def check_lm_eval():
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    wanted = ["gsm8k_cot", "gsm8k", "hellaswag", "boolq", "arc_easy"]
    sig = inspect.signature(HFLM.__init__).parameters
    return {
        "version": getattr(lm_eval, "__version__", None),
        "tasks_available": {t: t in tm.all_tasks for t in wanted},
        "hflm_has_enable_thinking": "enable_thinking" in sig,
        "hflm_init_params": [p for p in sig if p != "self"],
    }


@section("misc_imports")
def check_misc():
    import math_verify  # noqa: F401
    import peft
    import pycocotools  # noqa: F401

    vllm = None
    try:
        import vllm as _v

        vllm = _v.__version__
    except Exception:
        pass
    return {"peft": peft.__version__, "vllm": vllm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train-bench", action="store_true")
    args = ap.parse_args()
    set_seed()
    out_dir = results_dir("env")

    rep = env_report()
    write_json(out_dir / "kaggle_env.json", rep)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "pip_freeze.txt").write_text(freeze)
    print(rep)

    out: dict = {}
    check_gpus(out)
    check_secrets(out)
    check_misc(out)
    check_lm_eval(out)
    check_qwen3(out)
    check_siglip(out)
    if not args.skip_train_bench:
        train_bench(out)
    write_json(out_dir / "smoke_test.json", out)
    failed = [k for k, v in out.items() if not v.get("ok")]
    print("\nFAILED SECTIONS:", failed if failed else "none")


if __name__ == "__main__":
    main()
