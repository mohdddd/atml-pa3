"""Model construction for Task 1 runs and evaluation conditions, plus the memory-efficient SFT loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pa3.common.seed import set_seed
from pa3.task1 import lora as L

# CUDA on Kaggle; the CPU path exists only so the pipeline can be smoke-tested without a GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEVICE == "cuda" else torch.bfloat16


def gpu_peak_gb() -> tuple[float, float]:
    if DEVICE != "cuda":
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


def reset_peak() -> None:
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load_base(name: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    return model


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def build_train_model(cfg: dict, device: str) -> tuple[nn.Module, dict]:
    """Full FT: FP32 master parameters (FP16 autocast). LoRA/OPLoRA: frozen FP16 backbone, FP32 adapters."""
    info: dict = {}
    if cfg["method"] == "full":
        model = load_base(cfg["model"], torch.float32, device)
    else:
        model = load_base(cfg["model"], torch.float16, device)
        for p in model.parameters():
            p.requires_grad_(False)
        lc = cfg["lora"]
        set_seed(cfg["seed"])  # identical adapter init for LoRA-16 and OPLoRA-16
        info.update(L.inject(model, lc["targets"], lc["r"], lc["alpha"], lc["dropout"],
                             protect_k=lc.get("protect_k", 0) if cfg["method"] == "oplora" else 0))
    if cfg["train"]["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    info["trainable_params"] = sum(p.numel() for p in trainable)
    info["total_unique_params"] = sum(p.numel() for p in model.parameters())
    return model, info


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for p in model.parameters():  # .parameters() yields tied tensors once
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _ce_sum(head: nn.Module, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = head(h).float()
    return F.cross_entropy(logits, y, reduction="sum")


def sft_loss_sum(model: nn.Module, batch: dict, chunk: int = 1024) -> tuple[torch.Tensor, int]:
    """Sum of next-token cross-entropy over supervised (response + <|im_end|>) positions.

    Equivalent to the standard causal-LM loss with reduction='sum'; the lm_head and CE are applied only
    to supervised positions, in checkpointed chunks, so the [B, L, vocab] FP32 logits are never
    materialised (needed to fit full FT at 2,048 tokens on a 16 GB T4)."""
    h = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    h, y = h[:, :-1], batch["labels"][:, 1:]
    mask = y != -100
    hs, ys = h[mask], y[mask]
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for s in range(0, hs.shape[0], chunk):
        total = total + checkpoint(_ce_sum, model.lm_head, hs[s:s + chunk], ys[s:s + chunk], use_reentrant=False)
    return total, int(mask.sum())


# ---------------------------------------------------------------- evaluation models
def load_full_ft_weights(model: nn.Module, path) -> None:
    from safetensors.torch import load_file

    sd = load_file(str(path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("lm_head.weight")]  # tied to embed_tokens
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def save_full_ft_weights(model: nn.Module, path) -> None:
    from safetensors.torch import save_file

    sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    save_file(sd, str(path))
