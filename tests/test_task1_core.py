"""CPU checks of Task 1 building blocks: LoRA/OPLoRA math, merging, and the chunked SFT loss.
Run: PA3_T1_MODEL=<small Qwen3 dir> python tests/test_task1_core.py
"""
import os

import torch

from pa3.task1 import lora as L
from pa3.task1.data import collate, encode_example
from pa3.task1.modeling import load_base, load_tokenizer, sft_loss_sum

MODEL = os.environ.get("PA3_T1_MODEL", "Qwen/Qwen3-0.6B")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make(protect_k=0, seed=6304):
    m = load_base(MODEL, torch.float32, "cpu")
    for p in m.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    info = L.inject(m, TARGETS, r=16, alpha=32, dropout=0.0, protect_k=protect_k)
    return m, info


def test_init_identical_and_projection():
    a, _ = make(0)
    b, info = make(128)
    ma, mb = L.lora_modules(a), L.lora_modules(b)
    assert list(ma) == list(mb) and len(ma) == 28 * 7
    for n in ma:
        assert torch.equal(ma[n].lora_A, mb[n].lora_A), n          # same unprojected init
    with torch.no_grad():
        for mod in mb.values():
            mod.lora_B.normal_(0, 0.02)                              # make B nonzero
            if min(mod.base.weight.shape) <= mod.protect_k:  # whole space protected (tiny test model only)
                continue
            A, B = mod.factors()
            assert float((mod.U_k.T @ B).abs().max()) < 1e-5 * float(B.abs().max()) + 1e-7
            assert float((A @ mod.V_k).abs().max()) < 1e-5 * float(A.abs().max()) + 1e-7
            dW = mod.delta_weight()
            # rho_k of the OPLoRA update is ~0 by construction
            assert float((mod.U_k.T @ dW).pow(2).sum() / dW.pow(2).sum()) < 1e-8
    print("init identical + projection constraints OK; basis MB", info["basis_storage_bytes"] / 2**20)
    return b


def test_merge(model):
    tok = load_tokenizer(MODEL)
    model.eval()
    x = tok(["hello world, what is 2+2?", "The capital of France"], return_tensors="pt", padding=True)
    with torch.no_grad():
        un = model(**x).logits
        fresh = load_base(MODEL, torch.float32, "cpu").eval()
        L.merge_into(fresh, L.adapter_state(model, projected=True), 32 / 16)
        me = fresh(**x).logits
    d = float((un - me).abs().max())
    assert d < 1e-3, d
    print("merged vs unmerged max |logit diff| =", d)


def test_loss():
    tok = load_tokenizer(MODEL)
    m = load_base(MODEL, torch.float32, "cpu")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ex = [encode_example(tok, "What is 3+4?", "3+4 = 7. The answer is 7.", im_end),
          encode_example(tok, "Name a colour.", "Blue.", im_end)]
    b = collate(ex, tok.pad_token_id)
    s, n = sft_loss_sum(m, b, chunk=5)
    ref = m(**b).loss * n  # HF mean over the same masked labels
    assert abs(float(s) - float(ref)) < 1e-3 * float(ref), (float(s), float(ref))
    # the supervised span is exactly response + <|im_end|>
    ids, npr = ex[1]
    sup = tok.decode(ids[npr:])
    assert sup == "Blue.<|im_end|>", sup
    print("chunked SFT loss == HF loss:", float(s), float(ref), "| supervised text:", repr(sup))


if __name__ == "__main__":
    test_loss()
    m = test_init_identical_and_projection()
    test_merge(m)
    print("ALL OK")
