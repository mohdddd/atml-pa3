"""Unit tests for Task 2 helpers that do not need models: python -m pytest tests/test_task2_core.py -q"""
import numpy as np
import pytest
import torch

from pa3.task2 import config as C
from pa3.task2.data import mismatched_mapping
from pa3.task2.model import Connector
from pa3.task2.train import Plan, lr_factor
from pa3.task2.vqa_norm import consensus, normalize, soft_score


def test_normalize_official_rules():
    assert normalize("Two") == "2"
    assert normalize("A dog.") == "dog"
    assert normalize("the Red, car!") == "red car"
    assert normalize("isnt") == "isn't"
    assert normalize("  yes\n") == "yes"
    assert normalize("1,000") == "1000"          # comma between digits removed
    assert normalize("3.5") == "3.5"             # decimal point kept
    assert normalize("") == ""


def test_soft_score_formula():
    refs = ["2"] * 3 + ["3"] * 7
    # each held-out subset of 9 answers: '2' appears 3 times (if a '3' is held out) or 2 times
    exp = (7 * 1.0 + 3 * (2 / 3)) / 10
    assert soft_score("2", refs) == pytest.approx(exp)
    assert soft_score("3", refs) == pytest.approx(1.0)
    assert soft_score("4", refs) == 0.0
    assert soft_score("", refs) == 0.0


def test_consensus_group_and_raw_string():
    cons, n, raw = consensus(["Yes", "yes", "yes.", "no", "yes", "YES", "yes", "yes", "yes", "nope"])
    assert cons == "yes" and n == 8 and raw == "yes"


def test_connector_pool_2x2_blocks():
    x = torch.arange(196, dtype=torch.float32).view(1, 196, 1).expand(1, 196, 3).clone()
    p = Connector.pool(x)
    assert p.shape == (1, 49, 3)
    # first pooled token = mean of grid cells (0,0),(0,1),(1,0),(1,1) = indices 0,1,14,15
    assert p[0, 0, 0].item() == pytest.approx((0 + 1 + 14 + 15) / 4)
    # second pooled token = cells (0,2),(0,3),(1,2),(1,3)
    assert p[0, 1, 0].item() == pytest.approx((2 + 3 + 16 + 17) / 4)
    # first token of the second pooled row = cells (2,0),(2,1),(3,0),(3,1)
    assert p[0, 7, 0].item() == pytest.approx((28 + 29 + 42 + 43) / 4)


def test_connector_architecture():
    c = Connector()
    assert [type(m).__name__ for m in c.children()] == ["Linear", "GELU", "Linear"]
    assert not any(n == "out_scale" for n, _ in c.named_parameters())
    assert c(torch.zeros(2, 196, 768)).shape == (2, 196, 1024)
    assert sum(p.numel() for p in c.parameters()) == 768 * 1024 + 1024 + 1024 * 1024 + 1024


def test_lr_schedule_warmup_cosine_floor():
    U, w = 1500, C.warmup(1500, 0.05)
    assert w == 75
    assert C.warmup(2250, 0.05) == 113 and C.warmup(750, 0.05) == 38
    assert lr_factor(0, U, w, 0.1) == pytest.approx(1 / 75)
    assert lr_factor(w - 1, U, w, 0.1) == pytest.approx(1.0)
    assert lr_factor(w, U, w, 0.1) == pytest.approx(1.0)
    assert lr_factor(U - 1, U, w, 0.1) == pytest.approx(0.1, abs=1e-4)
    assert min(lr_factor(u, U, w, 0.1) for u in range(w, U)) >= 0.1 - 1e-9


def test_schedule_points():
    cfg = C.raw()["common"]
    cfg = dict(cfg, train=dict(cfg["train"]))
    assert C.stage_bounds(cfg) == [2250, 3750, 4500]
    assert C.dev_points(cfg) == [0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500]
    assert C.ckpt_points(cfg) == list(range(250, 4501, 250))
    assert C.stage_of(cfg, 2249) == (0, 2249) and C.stage_of(cfg, 2250) == (1, 0) and C.stage_of(cfg, 4499) == (2, 749)


def test_plan_stage1_two_passes_and_mixture():
    cfg = C.raw()["common"]
    sizes = {"caption_examples": 18000, "presence": 3000, "counting": 2000, "vqa": 1536}
    p1 = Plan(cfg, 0, sizes)
    flat = [i for u in p1.updates for _, i in u]
    assert len(p1.updates) == 2250 and len(flat) == 36000
    assert sorted(flat[:18000]) == list(range(18000)) and sorted(flat[18000:]) == list(range(18000))
    p2 = Plan(cfg, 1, sizes)
    assert len(p2.updates) == 1500 and all(len(u) == 8 for u in p2.updates)
    comps = [c for u in p2.updates for c, _ in u]
    frac = {k: comps.count(k) / len(comps) for k in ("presence", "counting", "caption")}
    assert frac["presence"] == pytest.approx(0.5, abs=0.02) and frac["counting"] == pytest.approx(0.25, abs=0.02)
    assert all(isinstance(i, tuple) and len(i) == 2 for c, i in (d for u in p2.updates for d in u) if c == "caption")
    p3 = Plan(cfg, 2, sizes)
    comps = [c for u in p3.updates for c, _ in u]
    assert comps.count("vqa") / len(comps) == pytest.approx(0.7, abs=0.03)
    # deterministic (resume = same plan)
    assert Plan(cfg, 2, sizes).updates == p3.updates


def test_mismatched_mapping_excludes_own_pair():
    pairs = [{"pair_id": str(i), "items": [{"qid": f"{i}a", "image": f"img{i}a"}, {"qid": f"{i}b", "image": f"img{i}b"}]}
             for i in range(10)]
    mp = mismatched_mapping(pairs, np.random.default_rng(0))
    for p in pairs:
        own = {it["image"] for it in p["items"]}
        for it in p["items"]:
            assert mp[it["qid"]] not in own
    assert len(mp) == 20
