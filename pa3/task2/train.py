"""Task 2 curriculum trainer (one variant: t196 main run, or t49 optional compression run).

Stage 1 caption alignment  2,250 updates  connector only (LR 1e-3)       two passes over 18,000 captions
Stage 2 visual instruction 1,500 updates  connector 1e-4 + LoRA 2e-5    draws: presence .5 / counting .25 / caption .25
Stage 3 natural VQA          750 updates  connector 5e-5 + LoRA 1e-5    draws: VQA .7 / presence .1 / counting .1 / caption .1
Microbatch 8 x accumulation 2 = 16 examples per update. In stages 2-3 one draw contributes two examples
(a complete question pair, or two caption examples for replay): 8 draws per update, 4 per microbatch. The
component of every draw comes from a seeded categorical sampler (no batch / stage quotas); items of each
component are taken from its pool in seeded shuffled cycles (pools may be revisited). The whole draw plan
of a stage is a deterministic function of (seed, stage), so the sampler position is just the update index.

Per stage: fresh AdamW (wd 0.01, biases excluded), linear warmup over ceil(5%) of the stage's updates,
then cosine decay to 10% of the stage's peak LR; global-norm clipping at 1.0 after unscaling; FP16
autocast + dynamic GradScaler; gradient checkpointing. LoRA (r 16, alpha 32, dropout .05, q/k/v/o, no
biases, random A / zero B) is added at the start of stage 2.

Development metrics before training, every 500 updates and at stage ends (2,250 / 3,750 / 4,500).
Resumable checkpoints every 250 updates (trainable weights, optimizer, scheduler, scaler, RNG, sampler
position / stage progress, run state) go to a private HF repo; connector/adapter weights after each stage
to <user>/atml-pa3-ckpts/task2/<variant>/stage<k>.

Usage: python -m pa3.task2.train --variant t196      (exit code 3 = stopped for the deadline; resume later)
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pa3.common import hub
from pa3.common.paths import results_dir, scratch_dir
from pa3.common.runlog import JsonlLogger, dir_size_mb, read_jsonl, save_config, write_json, write_jsonl
from pa3.common.seed import get_rng_state, set_rng_state, set_seed
from pa3.task2 import config as C
from pa3.task2 import data as D
from pa3.task2.features import FeatureStore
from pa3.task2.model import DEVICE, Ex, VLM, autocast, param_groups
from pa3.task2.vqa_norm import normalize, soft_score

PREEMPT_EXIT = 3
COMPONENTS = ["caption", "presence", "counting", "vqa"]


def feature_mode() -> str:
    return "dry" if C.is_dry() else "full"


def rec_repo(variant: str) -> str:
    return f"atml-pa3-rec-{'dry-' if C.is_dry() else ''}t2-{variant}"


def lr_factor(u: int, updates: int, warm: int, min_frac: float) -> float:
    """u = 0-based index of the update about to be applied within the stage."""
    if u < warm:
        return (u + 1) / warm
    prog = (u - warm) / max(1, updates - warm)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))


# ----------------------------------------------------------------------------- examples
def q_text(cfg: dict, question: str) -> str:
    return f"{question}\n{cfg['prompts']['answer_suffix']}"


def pair_examples(cfg: dict, pair: dict, swap: str | None = None, mism: dict | None = None) -> list[Ex]:
    """The two questions of a pair. swap: None/'correct', 'complementary', 'mismatched', 'no_image'."""
    out = []
    items = pair["items"]
    for j, it in enumerate(items):
        q = it.get("question", pair["question"])
        img = it["image"]
        if swap == "complementary":
            img = items[1 - j]["image"]
        elif swap == "mismatched":
            img = mism[it["qid"]]
        elif swap == "no_image":
            img = None
        out.append(Ex(img, q_text(cfg, q), it["answer"],
                      {"pair_id": pair["pair_id"], "qid": it["qid"], "type": pair["type"], "pos": j}))
    return out


def caption_examples(cfg: dict, recs: list[dict]) -> list[Ex]:
    return [Ex(r["image"], cfg["prompts"]["caption"], c, {"image_id": r["image_id"], "caption_id": cid})
            for r in recs for c, cid in zip(r["captions"], r["caption_ids"])]


def score_item(suite: str, pred: str, item: dict) -> dict:
    pn = normalize(pred)
    if suite == "vqa":
        s = soft_score(pn, [normalize(a) for a in item["refs"]])
    else:
        s = float(pn == item["canonical"])
    return {"pred_norm": pn, "score": s, "exact": float(pn == item["canonical"]), "empty": pn == ""}


class Plan:
    """Deterministic draw plan of one stage: plan[u] = list of 8 (component, item) draws."""

    def __init__(self, cfg: dict, stage: int, sizes: dict[str, int]):
        st = cfg["stages"][stage]
        U = st["updates"]
        eb = cfg["train"]["micro_batch"] * cfg["train"]["grad_accum"]
        seed = cfg["seed"]
        self.updates = []
        if list(st["mix"]) == ["caption"]:  # stage 1: passes over the caption examples, 16 per update
            n = sizes["caption_examples"]
            seq = []
            p = 0
            while len(seq) < U * eb:
                seq += np.random.default_rng([seed, 1, p]).permutation(n).tolist()
                p += 1
            self.updates = [[("caption1", i) for i in seq[u * eb:(u + 1) * eb]] for u in range(U)]
            self.draws_per_update = eb
            return
        names = list(st["mix"])
        probs = np.array([st["mix"][k] for k in names], dtype=float)
        rng = np.random.default_rng([seed, stage + 1])
        comps = rng.choice(len(names), size=U * eb // 2, p=probs / probs.sum())
        cursors = {}

        def cycle(name: str, n: int):
            k = 0
            while True:
                for i in np.random.default_rng([seed, stage + 1, COMPONENTS.index(name) + 10, k]).permutation(n):
                    yield int(i)
                k += 1

        for name in names:
            n = sizes["caption_examples"] if name == "caption" else sizes[name]
            cursors[name] = cycle(name, n)
        draws = []
        for c in comps:
            name = names[int(c)]
            if name == "caption":
                draws.append(("caption", (next(cursors[name]), next(cursors[name]))))
            else:
                draws.append((name, next(cursors[name])))
        k = eb // 2
        self.updates = [draws[u * k:(u + 1) * k] for u in range(U)]
        self.draws_per_update = k


# ----------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, variant: str):
        self.variant = variant
        self.cfg = C.config(variant)
        self.tc = self.cfg["train"]
        self.out = results_dir(C.results_task(), variant)
        self.ck_root = scratch_dir(C.results_task(), variant)
        self.deadline = float(os.environ.get("PA3_DEADLINE", "inf"))
        self.uploads = []

    # ------------------------------------------------------------------ setup
    def setup(self):
        set_seed(self.cfg["seed"])
        save_config(self.out / "config.yaml", self.cfg)
        self.splits = D.load_splits()
        self.lists = D.training_lists(self.splits, self.cfg.get("subset"))
        self.store = FeatureStore(feature_mode())
        self.vlm = VLM(self.cfg)
        if self.tc["grad_checkpointing"]:
            self.vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.caps = caption_examples(self.cfg, self.lists["caption"])
        self.sizes = {"caption_examples": len(self.caps), **{k: len(self.lists[k]) for k in ("presence", "counting", "vqa")}}
        self.scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
        self.state = {"update": 0, "stage_seconds": {}, "dev_seconds": 0.0, "ckpt_seconds": 0.0, "n_sessions": 0,
                      "skipped_updates": [], "mixture_draws": {}, "mixture_examples": {}, "peak_gpu": {},
                      "trainable_params": {}, "ckpt_sizes_mb": {}, "stage_weights_mb": {}}
        self.checks: dict = {}

    def feats(self, exs: list[Ex]):
        keys = [e.image for e in exs if e.image is not None]
        return self.store.get(keys) if keys else None

    def stage_examples(self, draw) -> list[Ex]:
        name, item = draw
        if name == "caption1":
            return [self.caps[item]]
        if name == "caption":
            return [self.caps[item[0]], self.caps[item[1]]]
        return pair_examples(self.cfg, self.lists[name][item])

    # ------------------------------------------------------------------ optimizer per stage
    def build_opt(self, stage: int):
        st = self.cfg["stages"][stage]
        if st["lr_lora"] is not None and not self.vlm.has_lora:
            info = self.vlm.add_lora(self.cfg["seed"] + 1000)
            self.state["lora_modules"] = info["n_adapted_modules"]
        for p in self.vlm.connector.parameters():
            p.requires_grad_(True)
        for p in self.vlm.lora_params():
            p.requires_grad_(st["lr_lora"] is not None)
        self.trainable = [p for p in self.vlm.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(param_groups(self.vlm, st["lr_connector"], st["lr_lora"], self.tc["weight_decay"]),
                                     betas=tuple(self.tc["betas"]), eps=self.tc["eps"])
        U = st["updates"]
        w = C.warmup(U, self.tc["warmup_frac"])
        mf = self.tc["min_lr_frac"]
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda u: lr_factor(u, U, w, mf))
        self.plan = Plan(self.cfg, stage, self.sizes)
        n_tr = sum(p.numel() for p in self.trainable)
        self.state["trainable_params"][st["name"]] = {
            "total": n_tr, "connector": sum(p.numel() for p in self.vlm.connector.parameters()),
            "lora": sum(p.numel() for p in self.vlm.lora_params()) if st["lr_lora"] is not None else 0,
            "warmup_updates": w}
        self.check_frozen(stage)

    # ------------------------------------------------------------------ checks (Step 1)
    def check_frozen(self, stage: int):
        lm_tr = [n for n, p in self.vlm.lm.named_parameters() if p.requires_grad]
        non_lora = [n for n in lm_tr if not (n.endswith("lora_A") or n.endswith("lora_B"))]
        self.checks[f"stage{stage + 1}_frozen"] = {
            "lm_trainable_tensors": len(lm_tr), "lm_trainable_non_lora": non_lora[:5],
            "lm_frozen_params": sum(p.numel() for p in self.vlm.lm.parameters() if not p.requires_grad),
            "lm_backbone_dtype": str(next(self.vlm.lm.parameters()).dtype),
            "connector_trainable": all(p.requires_grad for p in self.vlm.connector.parameters()),
            "connector_dtype": str(next(self.vlm.connector.parameters()).dtype),
            "lora_dtype": str(self.vlm.lora_params()[0].dtype) if self.vlm.has_lora else None,
            "trainable_params": sum(p.numel() for p in self.trainable),
            "ok": not non_lora and all(p.requires_grad for p in self.vlm.connector.parameters())}

    def check_grads(self, stage: int, update: int):
        """Called after scaler.unscale_ (gradients are unscaled). FP16 overflow on the first updates of a stage is
        normal with a fresh loss scale (the scaler skips those updates); the check is recorded at the first update
        whose gradients are finite and counts the overflowed updates before it."""
        cg = [p.grad for p in self.vlm.connector.parameters()]
        if not all(g is not None and torch.isfinite(g).all().item() for g in cg):
            self._overflow_before_check = getattr(self, "_overflow_before_check", 0) + 1
            return
        sc = 1.0
        rep = {"connector_grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in cg if g is not None))) / sc,
               "connector_all_have_grad": all(g is not None for g in cg),
               "connector_grads_finite": True, "checked_at_update": update,
               "overflowed_updates_before_check": getattr(self, "_overflow_before_check", 0),
               "backbone_params_with_grad": sum(1 for n, p in self.vlm.lm.named_parameters()
                                                if p.grad is not None and "lora_" not in n)}
        if self.vlm.has_lora and self.vlm.lora_params()[0].requires_grad:
            lp = self.vlm.lora_params()
            rep["lora_A_grad_norm"] = float(sum(float(p.grad.float().norm()) for p in lp[0::2] if p.grad is not None)) / sc
            rep["lora_B_grad_norm"] = float(sum(float(p.grad.float().norm()) for p in lp[1::2] if p.grad is not None)) / sc
        rep["ok"] = rep["connector_all_have_grad"] and rep["connector_grad_norm"] > 0 and rep["backbone_params_with_grad"] == 0
        self.checks[f"stage{stage + 1}_first_backward"] = rep
        self._overflow_before_check = 0

    def check_batches(self):
        v = self.vlm
        pair = self.lists["presence"][0]
        ex = pair_examples(self.cfg, pair) + pair_examples(self.cfg, pair, "no_image")[:1] + [self.caps[0]]
        b = v.build(ex, with_targets=True)
        lab_pos = (b["labels"] != -100)
        tgt_lens = [len(v.target_ids(e.target)) + 1 for e in ex]
        f = self.store.get([ex[0].image])
        with torch.no_grad():
            z = v.connector(f.to(DEVICE))
        with_img = len(v.prompt_ids(ex[0].text, True)[0]) + len(v.prompt_ids(ex[0].text, True)[1]) + v.n_vis
        no_img = len(v.prompt_ids(ex[0].text, False)[0])
        self.checks["batch_construction"] = {
            "feature_shape": list(f.shape[1:]), "visual_tokens_after_connector": list(z.shape[1:]),
            "visual_positions_per_image": [int(r.sum()) for r in b["vis"]],
            "supervised_tokens_per_example": lab_pos.sum(1).tolist(), "expected_target_plus_eot": tgt_lens,
            "labels_only_on_targets": lab_pos.sum(1).tolist() == tgt_lens,
            "no_visual_labels": int((lab_pos & b["vis"]).sum()) == 0,
            "prompt_len_with_image": with_img, "prompt_len_no_image": no_img,
            "no_image_shorter_by": with_img - no_img,
            "position_ids_contiguous": all(
                (b["position_ids"][i, :int(b["attention_mask"][i].sum())] ==
                 torch.arange(int(b["attention_mask"][i].sum()))).all().item() for i in range(len(ex))),
            "example_text": ex[0].text, "example_prompt": v.tok.decode(v.prompt_ids(ex[0].text, True)[0])
            + f"<{v.n_vis} visual tokens>" + v.tok.decode(v.prompt_ids(ex[0].text, True)[1]),
        }
        self.checks["batch_construction"]["ok"] = (self.checks["batch_construction"]["labels_only_on_targets"]
                                                   and self.checks["batch_construction"]["no_visual_labels"]
                                                   and self.checks["batch_construction"]["position_ids_contiguous"])
        self.checks["split_isolation"] = D.isolation_report(self.splits)

    def write_checks(self):
        write_json(self.out / "checks.json", self.checks)

    # ------------------------------------------------------------------ calibration
    def calibrate(self):
        rng = np.random.default_rng([self.cfg["seed"], 77])
        recs = self.lists["caption"]
        idx = rng.choice(len(recs), size=min(self.tc["calib_images"], len(recs)), replace=False)
        imgs = [recs[int(i)]["image"] for i in idx]
        texts = []
        for i in idx:  # full rendered training sequences (prompt + target) of the calibration images
            r = recs[int(i)]
            texts += [self.vlm.tok.apply_chat_template([{"role": "user", "content": self.cfg["prompts"]["caption"]}],
                                                       tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False) + c for c in r["captions"]]
        for k in ("presence", "counting", "vqa"):
            for p in self.lists[k][:200]:
                for it in p["items"]:
                    texts.append(q_text(self.cfg, it.get("question", p["question"])) + " " + it["answer"])
        rep = self.vlm.calibrate(self.store.get(imgs), texts)
        rep["n_images"] = len(imgs)
        write_json(self.out / "calibration.json", rep)
        print(f"[train:{self.variant}] calibration scale {rep['scale']:.4f}", flush=True)

    # ------------------------------------------------------------------ dev evaluation
    @torch.no_grad()
    def dev_eval(self, update: int) -> dict:
        t0 = time.perf_counter()
        self.vlm.eval()
        bs = self.cfg["eval"]["dev_batch_size"]
        row = {"update": update}
        per_ex = []

        def losses(exs, include_eot):
            out = []
            for i in range(0, len(exs), bs):
                chunk = exs[i:i + bs]
                with autocast():
                    out += self.vlm.example_losses(self.vlm.build(chunk, True), self.feats(chunk), include_eot).tolist()
            return out

        cd = self.splits["captions"]["dev"]
        lim = self.cfg["eval"].get("limit_pairs")  # local CPU tests only
        if lim:
            cd = cd[:2 * lim]
        cex = [Ex(r["image"], self.cfg["prompts"]["caption"], r["caption"]) for r in cd]
        cno = [Ex(None, e.text, e.target) for e in cex]
        cl, cn, cl0 = losses(cex, True), losses(cex, False), losses(cno, True)
        row.update(caption_loss=float(np.mean(cl)), caption_nll=float(np.mean(cn)),
                   caption_loss_noimage=float(np.mean(cl0)))
        for suite in C.SUITES:
            pairs = self.splits[suite]["dev"]
            if lim:
                pairs = pairs[:lim]
            res = {}
            for cond in ("correct", "complementary", "no_image"):
                exs = [e for p in pairs for e in pair_examples(self.cfg, p, cond)]
                res[cond] = (exs, losses(exs, False))
            exs = res["correct"][0]
            gens = []
            for i in range(0, len(exs), bs):
                chunk = exs[i:i + bs]
                gens += self.vlm.generate(chunk, self.feats(chunk), self.cfg["eval"]["max_new_tokens"])
            items = {it["qid"]: it for p in pairs for it in p["items"]}
            types = sorted({p["type"] for p in pairs})
            recs = []
            for k, e in enumerate(exs):
                sc = score_item(suite, gens[k]["text"], items[e.meta["qid"]])
                r = {"update": update, "suite": suite, "qid": e.meta["qid"], "type": e.meta["type"],
                     "nll_correct": res["correct"][1][k], "nll_complementary": res["complementary"][1][k],
                     "nll_no_image": res["no_image"][1][k], "pred": gens[k]["text"], **sc}
                r["delta_nll"] = r["nll_complementary"] - r["nll_correct"]
                recs.append(r)
            per_ex += recs
            for t in [None] + types:
                sub = [r for r in recs if t is None or r["type"] == t]
                pre = f"{suite}_" if t is None else f"{suite}_{t}_"
                for m in ("nll_correct", "nll_complementary", "nll_no_image", "delta_nll"):
                    row[pre + m] = float(np.mean([r[m] for r in sub]))
                row[pre + "acc_correct"] = float(np.mean([r["score"] for r in sub]))
        self.vlm.train()
        dt = time.perf_counter() - t0
        self.state["dev_seconds"] += dt
        row["seconds"] = dt
        JsonlLogger(self.out / "dev_log.jsonl").log(**row)
        old = self.out / "dev_examples.jsonl.gz"
        prev = read_jsonl(old) if old.exists() else []
        write_jsonl(old, [r for r in prev if r["update"] != update] + per_ex)
        print(f"[train:{self.variant}] dev u{update}: cap={row['caption_loss']:.3f} "
              f"g_nll={row['grounding_nll_correct']:.3f} dNLL={row['grounding_delta_nll']:.3f} "
              f"g_acc={row['grounding_acc_correct']:.3f} v_nll={row['vqa_nll_correct']:.3f} "
              f"dNLL={row['vqa_delta_nll']:.3f} v_acc={row['vqa_acc_correct']:.3f} ({dt:.0f}s)", flush=True)
        return row

    # ------------------------------------------------------------------ checkpoints
    def save_recovery(self, update: int, tag: str, sync: bool):
        from safetensors.torch import save_file

        t0 = time.perf_counter()
        d = self.ck_root / f"recovery_u{update}{tag}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        save_file(self.vlm.trainable_state(), str(d / "trainable.safetensors"))
        torch.save(self.opt.state_dict(), d / "optimizer.pt")
        stage, local = C.stage_of(self.cfg, update) if update < C.total_updates(self.cfg) else (len(self.cfg["stages"]), 0)
        torch.save({"scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(), "rng": get_rng_state(),
                    "update": update, "stage": stage, "stage_local_update": local,
                    "sampler_position": {"stage": stage, "update_in_stage": local,
                                         "draws_consumed_in_stage": local * self.plan.draws_per_update},
                    "has_lora": self.vlm.has_lora, "state": self.state}, d / "meta.pt")
        self.state["ckpt_sizes_mb"][f"recovery_u{update}{tag}"] = round(dir_size_mb(d), 2)
        self.state["ckpt_seconds"] += time.perf_counter() - t0
        name = rec_repo(self.variant)

        def _up():
            hub.upload(d, d.name, name=name, message=f"t2 {self.variant} recovery u{update}")
            hub.prune_recovery(name, keep_prefix=d.name)

        if sync:
            _up()
        else:
            import threading

            t = threading.Thread(target=_up)
            t.start()
            self.uploads.append(t)

    def find_resume(self) -> Path | None:
        def upd(p: str) -> int:
            return int(p.split("_u")[1].split("_")[0])

        local = sorted(self.ck_root.glob("recovery_u*"), key=lambda p: upd(p.name))
        best = local[-1] if local else None
        remote = sorted({f.split("/")[0] for f in hub.list_files(rec_repo(self.variant)) if f.startswith("recovery_u")},
                        key=upd)
        if remote and (best is None or upd(remote[-1]) > upd(best.name)):
            print(f"[train:{self.variant}] downloading {remote[-1]} from the Hub", flush=True)
            return hub.download(remote[-1], self.ck_root, name=rec_repo(self.variant))
        return best

    def load_recovery(self, d: Path) -> None:
        from safetensors.torch import load_file

        meta = torch.load(d / "meta.pt", weights_only=False)
        self.state = meta["state"]
        u = meta["update"]
        stage = C.stage_of(self.cfg, u)[0] if u < C.total_updates(self.cfg) else len(self.cfg["stages"]) - 1
        if meta["has_lora"] and not self.vlm.has_lora:
            self.vlm.add_lora(self.cfg["seed"] + 1000)
        self.vlm.load_trainable_state(load_file(str(d / "trainable.safetensors")))
        self.vlm.to(DEVICE)
        # the optimizer / scheduler in the checkpoint belong to the stage that contains update u-1
        prev_stage = C.stage_of(self.cfg, u - 1)[0]
        self.build_opt(prev_stage)
        self.opt.load_state_dict(torch.load(d / "optimizer.pt", map_location=DEVICE, weights_only=False))
        self.sched.load_state_dict(meta["scheduler"])
        if prev_stage != stage:  # checkpoint at a stage boundary: next stage starts fresh
            self.build_opt(stage)
        self.scaler.load_state_dict(meta["scaler"])
        set_rng_state(meta["rng"])
        self.state["update"] = u
        print(f"[train:{self.variant}] resumed at update {u} (stage {stage + 1})", flush=True)

    def save_stage_weights(self, stage: int, update: int):
        from safetensors.torch import save_file

        d = self.ck_root / f"stage{stage + 1}"
        d.mkdir(parents=True, exist_ok=True)
        sd = self.vlm.trainable_state()
        save_file({k: v for k, v in sd.items() if k.startswith("connector.")}, str(d / "connector.safetensors"))
        if any(k.startswith("lora.") for k in sd):
            save_file({k: v for k, v in sd.items() if k.startswith("lora.")}, str(d / "adapter.safetensors"))
        write_json(d / "meta.json", {"variant": self.variant, "stage": stage + 1, "update": update,
                                     "lora": self.cfg["lora"] if any(k.startswith("lora.") for k in sd) else None,
                                     "n_visual_tokens": self.cfg["n_visual_tokens"]})
        self.state["stage_weights_mb"][f"stage{stage + 1}"] = round(dir_size_mb(d), 2)
        self.uploads.append(hub.upload_async(d, f"{C.hub_prefix()}/{self.variant}/stage{stage + 1}"))

    # ------------------------------------------------------------------ loop
    def train(self) -> int:
        self.setup()
        logger = JsonlLogger(self.out / "train_log.jsonl")
        d = self.find_resume()
        fresh = d is None
        if fresh:
            for f in ("train_log.jsonl", "dev_log.jsonl", "dev_examples.jsonl.gz"):
                (self.out / f).unlink(missing_ok=True)
            self.calibrate()
            self.build_opt(0)
            self.check_batches()
        else:
            self.load_recovery(d)
            u0 = self.state["update"]
            for f in ("train_log.jsonl", "dev_log.jsonl", "dev_examples.jsonl.gz"):  # drop rows after the checkpoint
                p = self.out / f
                if p.exists():
                    write_jsonl(p, [r for r in read_jsonl(p) if r["update"] <= u0])
            if (self.out / "checks.json").exists():
                from pa3.common.runlog import read_json

                self.checks = read_json(self.out / "checks.json")
        self.state["n_sessions"] += 1
        T = C.total_updates(self.cfg)
        devs, ckpts, bounds = C.dev_points(self.cfg), C.ckpt_points(self.cfg), C.stage_bounds(self.cfg)
        eb = self.tc["micro_batch"] * self.tc["grad_accum"]
        if self.state["update"] == 0 and 0 in devs and fresh:
            self.dev_eval(0)
        self.vlm.train()
        cur_stage = C.stage_of(self.cfg, self.state["update"])[0] if self.state["update"] < T else None
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_sess = time.perf_counter()
        print(f"[train:{self.variant}] updates {self.state['update']}..{T}", flush=True)
        for u in range(self.state["update"], T):
            stage, local = C.stage_of(self.cfg, u)
            sname = self.cfg["stages"][stage]["name"]
            if stage != cur_stage:  # entering a new stage
                self.build_opt(stage)
                cur_stage = stage
                if DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
            if time.time() + 300 > self.deadline:
                self._acc_time(sname, t_sess)
                self.save_recovery(u, "_preempt", sync=True)
                self.write_resources(final=False)
                _push(self.out, f"task2: {self.variant} preempted at u{u}")
                print(f"[train:{self.variant}] deadline reached at update {u}; checkpointed, exiting", flush=True)
                return PREEMPT_EXIT
            t0 = time.perf_counter()
            draws = self.plan.updates[local]
            k = len(draws) // self.tc["grad_accum"]
            lr = self.sched.get_last_lr()
            comp_loss: dict = {}
            loss_tot = 0.0
            for m in range(self.tc["grad_accum"]):
                mds = draws[m * k:(m + 1) * k]
                exs, comps = [], []
                for dr in mds:
                    e = self.stage_examples(dr)
                    exs += e
                    comps += [dr[0].rstrip("1")] * len(e)
                b = self.vlm.build(exs, with_targets=True)
                with autocast():
                    per = self.vlm.example_losses(b, self.feats(exs), include_eot=True)
                loss = per.sum() / eb
                self.scaler.scale(loss).backward()
                for c, l in zip(comps, per.detach().float().tolist()):
                    comp_loss.setdefault(c, []).append(l)
                loss_tot += float(per.detach().sum())
            self.scaler.unscale_(self.opt)
            if f"stage{stage + 1}_first_backward" not in self.checks:
                self.check_grads(stage, u + 1)   # recorded at the first update with finite (unscaled) gradients
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.tc["clip_norm"]))
            sc0 = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            skipped = DEVICE == "cuda" and self.scaler.get_scale() < sc0
            self.sched.step()
            self.opt.zero_grad(set_to_none=True)
            if skipped:
                self.state["skipped_updates"].append(u + 1)
            dc = Counter(dr[0].rstrip("1") for dr in draws)
            for c, n in dc.items():
                md = self.state["mixture_draws"].setdefault(sname, {})
                md[c] = md.get(c, 0) + n
            for c, ls in comp_loss.items():
                me = self.state["mixture_examples"].setdefault(sname, {})
                me[c] = me.get(c, 0) + len(ls)
            row = {"update": u + 1, "stage": stage + 1, "lr_connector": lr[0],
                   "lr_lora": lr[2] if len(lr) > 2 else None, "loss": loss_tot / eb, "grad_norm": gnorm,
                   "loss_scale": sc0, "skipped": skipped, "draws": dict(dc),
                   **{f"loss_{c}": float(np.mean(v)) for c, v in comp_loss.items()},
                   "step_seconds": time.perf_counter() - t0}
            logger.log(**row)
            if (u + 1) % 50 == 0 or u < 3:
                print(f"[train:{self.variant}] u{u + 1}/{T} s{stage + 1} loss={row['loss']:.4f} lr={lr[0]:.2e} "
                      f"gn={gnorm:.2f} {row['step_seconds']:.2f}s/upd", flush=True)
            done = u + 1
            if done in devs or done in ckpts or done in bounds:
                self._acc_time(sname, t_sess)
                if done in bounds:
                    self._record_peak(sname)
                    self.save_stage_weights(stage, done)
                if done in devs:
                    self.dev_eval(done)
                self.state["update"] = done
                if done in ckpts:
                    self.save_recovery(done, "", sync=False)
                self.write_resources(final=False)
                self.write_checks()
                _push(self.out, f"task2: {self.variant} progress u{done}")
                t_sess = time.perf_counter()
            self.state["update"] = done
        for t in self.uploads:
            t.join()
        self.write_resources(final=True)
        self.write_checks()
        _push(self.out, f"task2: {self.variant} training complete")
        return 0

    def _acc_time(self, sname: str, t_sess: float):
        ss = self.state["stage_seconds"]
        ss[sname] = ss.get(sname, 0.0) + time.perf_counter() - t_sess

    def _record_peak(self, sname: str):
        if DEVICE == "cuda":
            self.state["peak_gpu"][sname] = {"alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                                             "reserved_gb": torch.cuda.max_memory_reserved() / 2**30}

    def write_resources(self, final: bool):
        res = {"variant": self.variant, "final": final, "updates_completed": self.state["update"],
               "n_visual_tokens": self.cfg["n_visual_tokens"],
               "trainable_params_by_stage": self.state["trainable_params"],
               "train_seconds_by_stage": self.state["stage_seconds"],
               "train_hours_total": sum(self.state["stage_seconds"].values()) / 3600,
               "seconds_per_update_by_stage": {s: self.state["stage_seconds"].get(s, 0) / st["updates"]
                                               for s, st in ((st["name"], st) for st in self.cfg["stages"])
                                               if s in self.state["stage_seconds"]},
               "dev_eval_seconds": self.state["dev_seconds"], "checkpoint_seconds": self.state["ckpt_seconds"],
               "peak_gpu_by_stage": self.state["peak_gpu"], "skipped_updates": self.state["skipped_updates"],
               "mixture_draws_by_stage": self.state["mixture_draws"],
               "mixture_examples_by_stage": self.state["mixture_examples"],
               "checkpoint_sizes_mb": self.state["ckpt_sizes_mb"], "stage_weights_mb": self.state["stage_weights_mb"],
               "n_sessions": self.state["n_sessions"],
               "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"}
        write_json(self.out / "resources.json", res)


def _push(path, msg):
    try:
        from pa3.common.gitsync import push

        push([path], msg)
    except Exception as e:
        print(f"[git] push failed (will retry later): {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    a = ap.parse_args()
    sys.exit(Trainer(a.variant).train())


if __name__ == "__main__":
    main()
