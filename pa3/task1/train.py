"""Task 1 training: Full FT, Full FT + L2-SP, LoRA, OPLoRA - one shared loop.

Recipe (configs/task1/task1.yaml): one epoch over 40,000 MetaMathQA examples in source order,
microbatch 2 x accumulation 16 = 32, 1,250 AdamW updates, warmup 38 + cosine, clip 1.0,
FP16 autocast + dynamic GradScaler (unscale before clipping; skipped updates logged),
gradient checkpointing. The loss of an update is the sum of response-token CE over its 16
microbatches divided by the number of supervised tokens in the whole effective batch.

L2-SP: Omega = 1/2 sum_j (theta_j - theta0_j)^2 over all unique parameters (incl. norms, tied
embedding once); theta0 is kept as an FP32 copy in pinned CPU memory and streamed to the GPU each
update (it does not fit next to FP32 weights + Adam states on a 16 GB T4); lambda * (theta - theta0)
is added to the unscaled gradient before clipping. Transfer/compute overhead is timed.

Robustness: recovery checkpoints at the configured updates (and when the session deadline
PA3_DEADLINE approaches) contain weights/adapters, optimizer, scheduler, scaler, RNG and data position;
they are uploaded to a private HF repo, and a restarted job resumes from the newest one.

Usage: python -m pa3.task1.train --run full_ft     (exit code 3 = stopped for the deadline; resume later)
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from pa3.common import hub
from pa3.common.paths import results_dir, scratch_dir
from pa3.common.runlog import JsonlLogger, dir_size_mb, read_json, save_config, write_json
from pa3.common.seed import get_rng_state, set_rng_state, set_seed
from pa3.task1 import config as C
from pa3.task1 import lora as L
from pa3.task1.data import effective_batch, load_tokenized
from pa3.task1.modeling import (AMP_DTYPE, DEVICE, build_train_model, gpu_peak_gb, load_tokenizer, param_groups,
                                reset_peak, save_full_ft_weights, sft_loss_sum, sync)

PREEMPT_EXIT = 3


def lr_lambda(warmup: int, total: int):
    def f(u: int) -> float:  # u = 0-based index of the update about to be applied
        if u < warmup:
            return (u + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (u - warmup) / max(1, total - warmup)))
    return f


def rec_repo(run: str) -> str:
    return f"atml-pa3-rec-{'dry-' if C.is_dry() else ''}{run}"


def weights_prefix(run: str, update: int) -> str:
    return f"{C.hub_prefix()}/{run}/u{update}"


class Trainer:
    def __init__(self, run: str, device: str = DEVICE):
        self.run, self.device = run, device
        self.cfg = C.run_config(run)
        self.tc = self.cfg["train"]
        self.out = results_dir(C.results_task(), run)
        self.ck_root = scratch_dir(C.results_task(), run)
        self.deadline = float(os.environ.get("PA3_DEADLINE", "inf"))
        self.is_full = self.cfg["method"] == "full"
        self.lam = float(self.cfg.get("l2sp_lambda", 0.0))
        self.uploads: list = []

    # ------------------------------------------------------------------ setup
    def setup(self):
        set_seed(self.cfg["seed"])
        save_config(self.out / "config.yaml", self.cfg)
        self.tok = load_tokenizer(self.cfg["model"])
        self.pad_id = self.tok.pad_token_id
        self.data = load_tokenized(self.cfg, self.tok)
        reset_peak()
        t0 = time.perf_counter()
        self.model, self.info = build_train_model(self.cfg, self.device)
        self.info["model_build_seconds"] = time.perf_counter() - t0
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(param_groups(self.model, self.tc["weight_decay"]), lr=self.cfg["lr"],
                                     betas=tuple(self.tc["betas"]), eps=self.tc["eps"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(self.tc["warmup"], self.tc["updates"]))
        self.scaler = torch.amp.GradScaler(DEVICE)
        # starting point theta0 (FP32, CPU): L2-SP reference and displacement for both full-FT runs
        self.ref = None
        if self.is_full:
            t0 = time.perf_counter()
            pin = self.lam > 0 and DEVICE == "cuda"
            self.ref = [p.detach().float().cpu().pin_memory() if pin else p.detach().float().cpu().clone()
                        for p in self.model.parameters()]
            self.info["reference_copy_seconds"] = time.perf_counter() - t0
            self.info["reference_storage_bytes"] = sum(r.numel() * 4 for r in self.ref)
            self.info["reference_pinned"] = pin
        self.state = {"update": 0, "train_seconds": 0.0, "ckpt_seconds": 0.0, "l2sp_seconds": 0.0,
                      "l2sp_transfer_seconds": 0.0, "skipped_updates": [], "displacement": {},
                      "n_sessions": 0, "ckpt_sizes_mb": {}}

    # ------------------------------------------------------------------ L2-SP
    @torch.no_grad()
    def l2sp_grad(self) -> float:
        t0 = time.perf_counter()
        omega = torch.zeros((), device=self.device, dtype=torch.float64)
        ttrans = 0.0
        for p, r in zip(self.model.parameters(), self.ref):
            ta = time.perf_counter()
            r_gpu = r.to(self.device, non_blocking=True)
            sync()
            ttrans += time.perf_counter() - ta
            diff = p.detach() - r_gpu
            omega += diff.double().pow(2).sum()
            if p.grad is not None:
                p.grad.add_(diff, alpha=self.lam)
        om = 0.5 * float(omega)
        self.state["l2sp_seconds"] += time.perf_counter() - t0
        self.state["l2sp_transfer_seconds"] += ttrans
        return om

    @torch.no_grad()
    def displacement(self) -> float:
        tot = 0.0
        for p, r in zip(self.model.parameters(), self.ref):
            tot += float((p.detach().float() - r.to(self.device)).double().pow(2).sum())
        return math.sqrt(tot)

    # ------------------------------------------------------------------ checkpoints
    def _trainable_state(self) -> dict:
        if self.is_full:
            return {k: v.detach().float().cpu().contiguous() for k, v in self.model.state_dict().items()
                    if k != "lm_head.weight"}
        return {n: p.detach().float().cpu().contiguous() for n, p in self.model.named_parameters() if p.requires_grad}

    def save_recovery(self, update: int, tag: str, sync: bool) -> None:
        from safetensors.torch import save_file

        t0 = time.perf_counter()
        d = self.ck_root / f"recovery_u{update}{tag}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        save_file(self._trainable_state(), str(d / "trainable.safetensors"))
        torch.save(self.opt.state_dict(), d / "optimizer.pt")
        torch.save({"scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(), "rng": get_rng_state(),
                    "update": update, "examples_seen": update * self.tc["micro_batch"] * self.tc["grad_accum"],
                    "state": self.state}, d / "meta.pt")
        size = dir_size_mb(d)
        self.state["ckpt_sizes_mb"][f"recovery_u{update}{tag}"] = round(size, 1)
        self.state["ckpt_seconds"] += time.perf_counter() - t0
        name = rec_repo(self.run)

        def _up():
            hub.upload(d, d.name, name=name, message=f"{self.run} recovery u{update}")
            hub.prune_recovery(name, keep_prefix=d.name)

        if sync:
            _up()
        else:
            import threading

            t = threading.Thread(target=_up)
            t.start()
            self.uploads.append(t)
        print(f"[train:{self.run}] recovery checkpoint u{update}{tag}: {size:.0f} MB", flush=True)

    def find_resume(self) -> Path | None:
        def upd(p: str) -> int:
            return int(p.split("_u")[1].split("_")[0].replace("preempt", ""))

        local = sorted(self.ck_root.glob("recovery_u*"), key=lambda p: upd(p.name))
        best_local = local[-1] if local else None
        remote = sorted({f.split("/")[0] for f in hub.list_files(rec_repo(self.run)) if f.startswith("recovery_u")},
                        key=upd)
        if remote and (best_local is None or upd(remote[-1]) > upd(best_local.name)):
            print(f"[train:{self.run}] downloading {remote[-1]} from the Hub", flush=True)
            return hub.download(remote[-1], self.ck_root, name=rec_repo(self.run))
        return best_local

    def load_recovery(self, d: Path) -> None:
        from safetensors.torch import load_file

        sd = load_file(str(d / "trainable.safetensors"))
        if self.is_full:
            self.model.load_state_dict(sd, strict=False)
        else:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for k, v in sd.items():
                    params[k].copy_(v)
        self.opt.load_state_dict(torch.load(d / "optimizer.pt", map_location=self.device, weights_only=False))
        meta = torch.load(d / "meta.pt", weights_only=False)
        self.sched.load_state_dict(meta["scheduler"])
        self.scaler.load_state_dict(meta["scaler"])
        set_rng_state(meta["rng"])
        self.state = meta["state"]
        self.state["update"] = meta["update"]
        print(f"[train:{self.run}] resumed at update {meta['update']}", flush=True)

    def save_eval_weights(self, update: int) -> None:
        t0 = time.perf_counter()
        d = self.ck_root / f"u{update}"
        d.mkdir(parents=True, exist_ok=True)
        if self.is_full:
            save_full_ft_weights(self.model, d / "model.safetensors")
        else:
            from safetensors.torch import save_file

            raw = L.adapter_state(self.model, projected=False)
            save_file(raw, str(d / "adapter_raw.safetensors"))
            t1 = time.perf_counter()
            proj = L.adapter_state(self.model, projected=True)
            save_file(proj, str(d / "adapter_effective.safetensors"))
            if self.cfg["method"] == "oplora":
                self.state["oplora_export_seconds"] = time.perf_counter() - t1
            write_json(d / "adapter_meta.json", {"r": self.cfg["lora"]["r"], "alpha": self.cfg["lora"]["alpha"],
                                                 "scale": self.cfg["lora"]["alpha"] / self.cfg["lora"]["r"],
                                                 "method": self.cfg["method"]})
        self.state["ckpt_sizes_mb"][f"eval_weights_u{update}"] = round(dir_size_mb(d), 1)
        self.state["ckpt_seconds"] += time.perf_counter() - t0
        self.uploads.append(hub.upload_async(d, weights_prefix(self.run, update)))

    # ------------------------------------------------------------------ loop
    def train(self) -> int:
        self.setup()
        logger = JsonlLogger(self.out / "train_log.jsonl")
        d = self.find_resume()
        if d is not None:
            self.load_recovery(d)
            if (self.out / "train_log.jsonl").exists():  # drop rows logged after the checkpoint
                rows = [r for r in _read_rows(self.out / "train_log.jsonl") if r["update"] < self.state["update"]]
                _write_rows(self.out / "train_log.jsonl", rows)
        elif (self.out / "train_log.jsonl").exists():
            (self.out / "train_log.jsonl").unlink()
        self.state["n_sessions"] += 1

        U = self.tc["updates"]
        margin = (900 if self.is_full else 300) + 300  # seconds needed to checkpoint + upload before deadline
        eb = self.tc["micro_batch"] * self.tc["grad_accum"]
        print(f"[train:{self.run}] updates {self.state['update']}..{U}, trainable={self.info['trainable_params']:,}",
              flush=True)
        t_sess = time.perf_counter()
        for u in range(self.state["update"], U):
            if time.time() + margin > self.deadline:
                self.state["train_seconds"] += time.perf_counter() - t_sess
                self.save_recovery(u, "_preempt", sync=True)
                self.write_resources(final=False)
                print(f"[train:{self.run}] deadline reached at update {u}; checkpointed, exiting", flush=True)
                return PREEMPT_EXIT
            t0 = time.perf_counter()
            mbs = effective_batch(self.data, u, self.cfg, self.pad_id)
            n_sup = sum(int((mb["labels"][:, 1:] != -100).sum()) for mb in mbs)
            n_tok = sum(int(mb["attention_mask"].sum()) for mb in mbs)
            n_pad_tok = sum(mb["input_ids"].numel() for mb in mbs)
            lr = self.sched.get_last_lr()[0]
            loss_sum = 0.0
            for mb in mbs:
                mb = {k: v.to(self.device, non_blocking=True) for k, v in mb.items()}
                with torch.autocast(DEVICE, dtype=AMP_DTYPE):
                    ls, _ = sft_loss_sum(self.model, mb, self.tc["loss_chunk_tokens"])
                self.scaler.scale(ls / n_sup).backward()
                loss_sum += float(ls.detach())
            sft = loss_sum / n_sup
            self.scaler.unscale_(self.opt)
            omega = self.l2sp_grad() if self.lam > 0 else None
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.tc["clip_norm"]))
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            skipped = self.scaler.get_scale() < scale_before
            self.sched.step()
            self.opt.zero_grad(set_to_none=True)
            if skipped:
                self.state["skipped_updates"].append(u + 1)
            row = {"update": u + 1, "examples_seen": (u + 1) * eb, "lr": lr, "sft_loss": sft, "grad_norm": gnorm,
                   "loss_scale": scale_before, "skipped": skipped, "supervised_tokens": n_sup, "tokens": n_tok,
                   "padded_tokens": n_pad_tok, "step_seconds": time.perf_counter() - t0}
            if omega is not None:
                row.update(omega=omega, lambda_omega=self.lam * omega, total_loss=sft + self.lam * omega)
            logger.log(**row)
            if (u + 1) % 25 == 0 or u < 3:
                el = self.state["train_seconds"] + time.perf_counter() - t_sess
                print(f"[train:{self.run}] u{u + 1}/{U} loss={sft:.4f} lr={lr:.2e} gn={gnorm:.2f} "
                      f"scale={scale_before:.0f} {row['step_seconds']:.1f}s/upd elapsed={el / 3600:.2f}h", flush=True)

            done = u + 1
            if done in self.tc["eval_updates"] or done in self.tc["ckpt_updates"]:
                self.state["train_seconds"] += time.perf_counter() - t_sess
                if self.is_full and done in self.tc["eval_updates"]:
                    self.state["displacement"][str(done)] = self.displacement()
                if done in self.tc["eval_updates"]:
                    self.save_eval_weights(done)
                if done in self.tc["ckpt_updates"]:
                    self.state["update"] = done
                    self.save_recovery(done, "", sync=False)
                self.write_resources(final=False)
                _push(self.out, f"task1: {self.run} progress u{done}")
                t_sess = time.perf_counter()
            self.state["update"] = done
        self.state["train_seconds"] += time.perf_counter() - t_sess
        self.merge_check()
        for t in self.uploads:
            t.join()
        self.write_resources(final=True)
        _push(self.out, f"task1: {self.run} training complete")
        return 0

    # ------------------------------------------------------------------ adapters: merged vs unmerged check
    @torch.no_grad()
    def merge_check(self) -> None:
        if self.is_full:
            return
        from pa3.task1.data import prompt_text
        from pa3.task1.modeling import load_base

        self.model.eval()
        texts = [prompt_text(self.tok, "What is 15% of 80? Explain."),
                 "The capital of France is", prompt_text(self.tok, "Name three primary colors.")]
        self.tok.padding_side = "right"
        b = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        un = self.model(**b).logits.float()
        merged = load_base(self.cfg["model"], torch.float16, self.device).eval()
        proj = L.adapter_state(self.model, projected=True)
        L.merge_into(merged, proj, self.cfg["lora"]["alpha"] / self.cfg["lora"]["r"])
        me = merged(**b).logits.float()
        m = b["attention_mask"].bool()
        diff = (un - me).abs()[m]
        res = {"max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
               "max_abs_logit": float(un.abs()[m].max()),
               "argmax_agreement": float((un.argmax(-1) == me.argmax(-1))[m].float().mean())}
        if self.cfg["method"] == "oplora":  # numerical check of the projection constraints
            mods = L.lora_modules(self.model)
            worst_b = worst_a = 0.0
            for mod in mods.values():  # ||U_k^T B~||_F / ||B||_F and ||A~ V_k||_F / ||A||_F (raw factors)
                A, B = mod.factors()
                worst_b = max(worst_b, float((mod.U_k.T @ B).norm() / (mod.lora_B.float().norm() + 1e-12)))
                worst_a = max(worst_a, float((A @ mod.V_k).norm() / (mod.lora_A.float().norm() + 1e-12)))
            res["oplora_max_rel_UkT_Btilde"] = worst_b
            res["oplora_max_rel_Atilde_Vk"] = worst_a
        write_json(self.out / "merge_check.json", res)
        print(f"[train:{self.run}] merge check {res}", flush=True)
        del merged
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        self.model.train()

    def write_resources(self, final: bool) -> None:
        res = {
            "run": self.run, "method": self.cfg["method"], "final": final,
            "updates_completed": self.state["update"],
            "trainable_params": self.info["trainable_params"],
            "total_unique_params": self.info["total_unique_params"],
            "train_seconds": self.state["train_seconds"], "train_hours": self.state["train_seconds"] / 3600,
            "checkpoint_seconds": self.state["ckpt_seconds"], "n_sessions": self.state["n_sessions"],
            "peak_gpu_alloc_gb": gpu_peak_gb()[0], "peak_gpu_reserved_gb": gpu_peak_gb()[1],
            "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", "skipped_updates": self.state["skipped_updates"],
            "checkpoint_sizes_mb": self.state["ckpt_sizes_mb"],
            "displacement_l2": self.state["displacement"],
            "model_build_seconds": self.info.get("model_build_seconds"),
        }
        if self.is_full:
            res.update(reference_storage_bytes=self.info["reference_storage_bytes"],
                       reference_pinned=self.info["reference_pinned"],
                       reference_copy_seconds=self.info["reference_copy_seconds"])
        if self.lam > 0:
            res.update(l2sp_lambda=self.lam, l2sp_overhead_seconds=self.state["l2sp_seconds"],
                       l2sp_transfer_seconds=self.state["l2sp_transfer_seconds"])
        if self.cfg["method"] in ("lora", "oplora"):
            res.update(n_adapted_modules=self.info["n_adapted_modules"])
        if self.cfg["method"] == "oplora":
            res.update(svd_preprocess_seconds=self.info["svd_preprocess_seconds"],
                       basis_storage_bytes=self.info["basis_storage_bytes"],
                       export_seconds=self.state.get("oplora_export_seconds"))
        write_json(self.out / "resources.json", res)


def _read_rows(p):
    from pa3.common.runlog import read_jsonl

    return read_jsonl(p)


def _write_rows(p, rows):
    from pa3.common.runlog import write_jsonl

    write_jsonl(p, rows, compress=False)


def _push(path, msg):
    try:
        from pa3.common.gitsync import push

        push([path], msg)
    except Exception as e:
        print(f"[git] push failed (will retry later): {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=C.RUNS)
    args = ap.parse_args()
    sys.exit(Trainer(args.run).train())


if __name__ == "__main__":
    main()
