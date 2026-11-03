"""SigLIP-Qwen VLM: connector, visual-token insertion, losses, teacher-forced NLL and greedy generation.

Sequence layout (Qwen3 chat template, enable_thinking=False):
    <|im_start|>user\\n [Z_1 .. Z_N] \\n <text><|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n <target><|im_end|>
The visual tokens Z = g_phi(F(I)) (N = 196, or 49 after 2x2 average pooling of the 14 x 14 grid) are
inserted in the user message before the text. They are placed by rendering the template with one
<|image_pad|> placeholder, expanding it to N positions and overwriting those positions' embeddings with
the connector output (the placeholder's own embedding is never used). Without an image, the template is
rendered from the text alone: the visual tokens and their "\\n" separator are omitted, so masks and
positions are rebuilt for the shorter prefix. Position ids are 0..L-1 over non-padding tokens (right
padding for teacher forcing, left padding for generation).

Supervised tokens: target tokens + <|im_end|>; visual tokens, prompt and padding are masked.
Training loss: per-example mean over supervised tokens, then mean over the effective batch.
Reference NLL (dev / final): per-example mean token NLL over the reference answer tokens EXCLUDING the
termination token, then mean over examples.

Precision: frozen FP16 Qwen backbone, FP32 connector and LoRA parameters, FP16 autocast on CUDA.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pa3.task1 import lora as L

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE = -100


def autocast():
    return torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else contextlib.nullcontext()


@dataclass
class Ex:
    """One image-text example. image=None -> no-image input."""
    image: str | None
    text: str
    target: str = ""
    meta: dict = field(default_factory=dict)


class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024 (optionally after 2x2 average pooling: 196 -> 49).

    `out_scale` is a fixed (non-trainable) scalar set once by the magnitude calibration; keeping it outside
    the weights leaves the MLP at its standard initialisation, so AdamW's per-step relative change of the
    weights is not inflated by a tiny weight scale."""

    def __init__(self, in_dim=768, hidden=1024, out_dim=1024, pool_2x2=False):
        super().__init__()
        self.pool_2x2 = pool_2x2
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.register_buffer("out_scale", torch.ones(()), persistent=True)

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        g = int(math.isqrt(N))
        x = x.view(B, g // 2, 2, g // 2, 2, D)          # (row block, row in block, col block, col in block)
        return x.mean(dim=(2, 4)).reshape(B, (g // 2) ** 2, D)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = f.float()
        if self.pool_2x2:
            f = self.pool(f)
        return self.fc2(self.act(self.fc1(f))) * self.out_scale


class VLM(nn.Module):
    def __init__(self, cfg: dict, device: str = DEVICE):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg, self.device = cfg, device
        self.tok = AutoTokenizer.from_pretrained(cfg["lm"])
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(cfg["lm"], dtype=dtype, attn_implementation="sdpa").to(device)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.config.use_cache = False
        c = cfg["connector"]
        self.connector = Connector(c["in_dim"], c["hidden"], c["out_dim"], cfg["pool_2x2"]).to(device)
        self.n_vis = cfg["n_visual_tokens"]
        self.img_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = {self.im_end, self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.pad_id = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.has_lora = False
        self._pcache: dict = {}
        self._tcache: dict = {}

    # ------------------------------------------------------------------ prompts
    def prompt_ids(self, text: str, with_image: bool) -> tuple[list[int], list[int]]:
        """(ids before the visual block, ids after it); without image the second list is empty."""
        key = (text, with_image)
        if key not in self._pcache:
            content = ("<|image_pad|>\n" + text) if with_image else text
            s = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=self.cfg["enable_thinking"])
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            if with_image:
                assert ids.count(self.img_id) == 1
                i = ids.index(self.img_id)
                self._pcache[key] = (ids[:i], ids[i + 1:])
            else:
                assert self.img_id not in ids
                self._pcache[key] = (ids, [])
        return self._pcache[key]

    def target_ids(self, target: str) -> list[int]:
        if target not in self._tcache:
            self._tcache[target] = self.tok(target, add_special_tokens=False)["input_ids"]
        return self._tcache[target]

    # ------------------------------------------------------------------ batching
    def build(self, exs: list[Ex], with_targets: bool, left_pad: bool = False) -> dict:
        seqs, labs, vis, kinds = [], [], [], []
        for e in exs:
            pre, post = self.prompt_ids(e.text, e.image is not None)
            v = [self.img_id] * self.n_vis if e.image is not None else []
            ids = pre + v + post
            lab = [IGNORE] * len(ids)
            kind = [0] * len(ids)                     # 1 = answer content token, 2 = termination token
            if with_targets:
                t = self.target_ids(e.target)
                ids = ids + t + [self.im_end]
                lab = lab + t + [self.im_end]
                kind = kind + [1] * len(t) + [2]
            seqs.append(ids)
            labs.append(lab)
            kinds.append(kind)
            vis.append([0] * len(pre) + [1] * len(v) + [0] * (len(ids) - len(pre) - len(v)))
        T = max(len(s) for s in seqs)
        B = len(seqs)
        out = {k: torch.full((B, T), f, dtype=torch.long) for k, f in
               (("input_ids", self.pad_id), ("labels", IGNORE), ("kind", 0), ("vis", 0), ("attention_mask", 0))}
        for i, (s, lab, vv, kd) in enumerate(zip(seqs, labs, vis, kinds)):
            n = len(s)
            sl = slice(T - n, T) if left_pad else slice(0, n)
            out["input_ids"][i, sl] = torch.tensor(s)
            out["labels"][i, sl] = torch.tensor(lab)
            out["kind"][i, sl] = torch.tensor(kd)
            out["vis"][i, sl] = torch.tensor(vv)
            out["attention_mask"][i, sl] = 1
        out["position_ids"] = (out["attention_mask"].cumsum(1) - 1).clamp(min=0)
        out["vis"] = out["vis"].bool()
        out["images"] = [e.image for e in exs if e.image is not None]
        return out

    def embed(self, b: dict, feats: torch.Tensor | None) -> torch.Tensor:
        emb = self.lm.get_input_embeddings()(b["input_ids"].to(self.device))
        if b["images"]:
            z = self.connector(feats.to(self.device))                     # [n_img, N, d]
            assert z.shape[1] == self.n_vis
            m = b["vis"].to(self.device)
            assert int(m.sum()) == z.shape[0] * z.shape[1]
            emb = emb.masked_scatter(m.unsqueeze(-1), z.reshape(-1, z.shape[-1]).to(emb.dtype))
        return emb

    def hidden(self, b: dict, feats) -> torch.Tensor:
        emb = self.embed(b, feats)
        return self.lm.model(inputs_embeds=emb, attention_mask=b["attention_mask"].to(self.device),
                             position_ids=b["position_ids"].to(self.device)).last_hidden_state

    # ------------------------------------------------------------------ losses
    def token_nll(self, b: dict, feats, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per supervised position: (nll, example index, kind). lm_head + CE only on supervised positions."""
        h = self.hidden(b, feats)[:, :-1]
        y = b["labels"][:, 1:].to(self.device)
        kind = b["kind"][:, 1:].to(self.device)
        mask = y != IGNORE
        ex = torch.arange(y.shape[0], device=self.device).unsqueeze(1).expand_as(y)[mask]
        hs, ys = h[mask], y[mask]
        parts = []
        for s in range(0, hs.shape[0], chunk):
            parts.append(checkpoint(_ce, self.lm.lm_head, hs[s:s + chunk], ys[s:s + chunk], use_reentrant=False)
                         if self.training else _ce(self.lm.lm_head, hs[s:s + chunk], ys[s:s + chunk]))
        return torch.cat(parts), ex, kind[mask]

    def example_losses(self, b: dict, feats, include_eot: bool = True) -> torch.Tensor:
        """Mean token NLL per example over target tokens (+ termination if include_eot)."""
        nll, ex, kind = self.token_nll(b, feats)
        keep = (kind == 1) | ((kind == 2) & include_eot)
        B = b["input_ids"].shape[0]
        s = torch.zeros(B, device=self.device).index_add(0, ex[keep], nll[keep])
        n = torch.zeros(B, device=self.device).index_add(0, ex[keep], torch.ones_like(nll[keep]))
        return s / n.clamp(min=1)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, exs: list[Ex], feats, max_new_tokens: int) -> list[dict]:
        from transformers import DynamicCache

        b = self.build(exs, with_targets=False, left_pad=True)
        am = b["attention_mask"].to(self.device)
        pos = b["position_ids"].to(self.device)
        with autocast():
            emb = self.embed(b, feats)
            cache = DynamicCache()
            out = self.lm(inputs_embeds=emb, attention_mask=am, position_ids=pos, past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
            B = len(exs)
            gen = [[] for _ in range(B)]
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            stopped = torch.zeros(B, dtype=torch.bool, device=self.device)
            nxt = out.logits[:, -1].argmax(-1)
            last = pos[:, -1]
            for step in range(max_new_tokens):
                for i in range(B):
                    if not done[i]:
                        gen[i].append(int(nxt[i]))
                is_stop = torch.tensor([int(t) in self.eos_ids for t in nxt.tolist()], device=self.device)
                stopped |= is_stop & ~done
                done |= is_stop
                if bool(done.all()) or step == max_new_tokens - 1:
                    break
                am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=self.device)], 1)
                last = last + 1
                out = self.lm(input_ids=nxt.unsqueeze(1), attention_mask=am, position_ids=last.unsqueeze(1),
                              past_key_values=cache, use_cache=True)
                nxt = out.logits[:, -1].argmax(-1)
        res = []
        for i in range(B):
            toks = [t for t in gen[i] if t not in self.eos_ids]
            res.append({"text": self.tok.decode(toks, skip_special_tokens=True).strip(), "n_tokens": len(gen[i]),
                        "stopped": bool(stopped[i]), "truncated": not bool(stopped[i])})
        return res

    # ------------------------------------------------------------------ LoRA / params / state
    def add_lora(self, seed: int) -> dict:
        lc = self.cfg["lora"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            info = L.inject(self.lm, lc["targets"], lc["r"], lc["alpha"], lc["dropout"])
        self.has_lora = True
        return info

    def lora_params(self) -> list[nn.Parameter]:
        return [p for m in L.lora_modules(self.lm).values() for p in (m.lora_A, m.lora_B)]

    def trainable_state(self) -> dict:
        sd = {f"connector.{k}": v.detach().float().cpu().contiguous() for k, v in self.connector.state_dict().items()}
        if self.has_lora:
            sd.update({f"lora.{k}": v for k, v in L.adapter_state(self.lm, projected=False).items()})
        return sd

    @torch.no_grad()
    def load_trainable_state(self, sd: dict) -> None:
        self.connector.load_state_dict({k[len("connector."):]: v for k, v in sd.items() if k.startswith("connector.")})
        lora = {k[len("lora."):]: v for k, v in sd.items() if k.startswith("lora.")}
        if lora:
            assert self.has_lora, "add_lora() before loading adapter weights"
            for n, m in L.lora_modules(self.lm).items():
                m.lora_A.copy_(lora[f"{n}.A"])
                m.lora_B.copy_(lora[f"{n}.B"])

    # ------------------------------------------------------------------ connector-scale calibration
    @torch.no_grad()
    def calibrate(self, feats: torch.Tensor, texts: list[str]) -> dict:
        """Rule: multiply the connector output by a fixed scalar s = mean ||e_text|| / mean ||z||,
        where e_text are Qwen input embeddings of the tokens of training prompts + targets and z are the
        initial connector outputs for training images, so visual tokens start with the same mean L2 norm
        as text tokens. Training data only."""
        E = self.lm.get_input_embeddings().weight
        ids = [t for s in texts for t in self.tok(s, add_special_tokens=False)["input_ids"]]
        text_norm = float(E[torch.tensor(ids, device=E.device)].float().norm(dim=-1).mean())
        z = torch.cat([self.connector(feats[i:i + 64].to(self.device)) for i in range(0, len(feats), 64)])
        vis_norm = float(z.float().norm(dim=-1).mean())
        s = text_norm / vis_norm
        self.connector.out_scale.fill_(s)
        z2 = torch.cat([self.connector(feats[i:i + 64].to(self.device)) for i in range(0, len(feats), 64)])
        return {"rule": "connector output multiplied by a fixed scalar s = mean_token ||text embedding||_2 / "
                        "mean_token ||initial connector output||_2 (training prompts+targets / training images)",
                "text_embedding_mean_norm": text_norm, "visual_token_mean_norm_before": vis_norm, "scale": s,
                "visual_token_mean_norm_after": float(z2.float().norm(dim=-1).mean()),
                "n_text_tokens": len(ids), "n_images": int(len(feats)),
                "embedding_norm_vocab_mean": float(E.float().norm(dim=-1).mean())}


def _ce(head: nn.Module, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(head(h).float(), y, reduction="none")


def param_groups(vlm: VLM, lr_conn: float, lr_lora: float | None, wd: float) -> list[dict]:
    """Biases / 1-D parameters are excluded from weight decay."""
    groups = []
    cw = [p for p in vlm.connector.parameters() if p.ndim >= 2]
    cb = [p for p in vlm.connector.parameters() if p.ndim < 2]
    groups += [{"params": cw, "lr": lr_conn, "weight_decay": wd, "name": "connector"},
               {"params": cb, "lr": lr_conn, "weight_decay": 0.0, "name": "connector_bias"}]
    if lr_lora is not None:
        groups.append({"params": vlm.lora_params(), "lr": lr_lora, "weight_decay": wd, "name": "lora"})
    return groups
