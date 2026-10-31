"""LoRA and OPLoRA adapters on a frozen backbone.

LoRA:    W = W0 + (alpha/r) B A,  A in R^{r x d_in} (Kaiming-uniform, as in the reference LoRA/PEFT
         init), B in R^{d_out x r} (zeros), dropout on the adapter input, no biases.
OPLoRA:  same trainable factors and initialisation; the correction uses projected factors
         B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T,  dW = (alpha/r) B~ A~
         with U_k, V_k the leading-k singular bases of W0 (computed once in FP32, frozen).
         Effective factors are used directly; no dense d x d projection matrices are formed.

One implementation serves both, so LoRA-16 and OPLoRA-16 start from identical unprojected factors
(same seed, same module order).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        dout, din = base.weight.shape
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, din, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(dout, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.protect_k = 0

    # ---- OPLoRA
    @torch.no_grad()
    def set_protection(self, k: int) -> None:
        W0 = self.base.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        self.register_buffer("U_k", U[:, :k].contiguous(), persistent=True)
        self.register_buffer("V_k", Vh[:k].T.contiguous(), persistent=True)
        self.protect_k = k

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Effective (projected for OPLoRA) factors in FP32."""
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A, B = self.lora_A.float(), self.lora_B.float()
            if self.protect_k:
                B = B - self.U_k @ (self.U_k.T @ B)
                A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self) -> torch.Tensor:
        A, B = self.factors()
        with torch.autocast(device_type=A.device.type, enabled=False):
            return self.scale * (B @ A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A, B = self.factors()
        d = self.dropout(x)
        return y + (d @ A.T.to(d.dtype)) @ B.T.to(d.dtype) * self.scale


def inject(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float,
           protect_k: int = 0) -> dict:
    """Replace target nn.Linear modules in every transformer layer. Returns overhead info."""
    names = []
    for name, mod in list(model.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                setattr(mod, t, LoRALinear(child, r, alpha, dropout))
                names.append(f"{name}.{t}")
    info = {"n_adapted_modules": len(names)}
    if protect_k:
        t0 = time.perf_counter()
        for m in lora_modules(model).values():
            m.set_protection(protect_k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # SVD timing
        info["svd_preprocess_seconds"] = time.perf_counter() - t0
        info["basis_storage_bytes"] = sum(m.U_k.numel() * 4 + m.V_k.numel() * 4 for m in lora_modules(model).values())
    return info


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state(model: nn.Module, projected: bool) -> dict[str, torch.Tensor]:
    """Raw trainable factors; with projected=True the effective (projected) factors for inference."""
    out = {}
    for n, m in lora_modules(model).items():
        if projected:
            A, B = m.factors()
        else:
            A, B = m.lora_A, m.lora_B
        out[f"{n}.A"] = A.detach().float().cpu().contiguous()
        out[f"{n}.B"] = B.detach().float().cpu().contiguous()
    return out


@torch.no_grad()
def merge_into(backbone: nn.Module, factors: dict[str, torch.Tensor], scale: float) -> int:
    """W <- W0 + scale * B A into a fresh backbone (FP32 arithmetic, cast to the backbone dtype)."""
    mods = dict(backbone.named_modules())
    n = 0
    for key in [k for k in factors if k.endswith(".A")]:
        name = key[:-2]
        lin = mods[name]
        A = factors[f"{name}.A"].to(lin.weight.device)
        B = factors[f"{name}.B"].to(lin.weight.device)
        lin.weight.copy_((lin.weight.float() + scale * (B @ A)).to(lin.weight.dtype))
        n += 1
    return n
