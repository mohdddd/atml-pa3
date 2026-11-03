"""Task 2 data: selection of COCO captions, annotation-derived grounding pairs and VQAv2 complementary
pairs; fixed mismatched-image mapping; image download; tokenised example construction.

Selection (python -m pa3.task2.data select) reads the raw COCO-2014 / VQAv2 annotation files once and
writes everything later stages need (IDs, texts, answers) to splits/task2/, so training sessions never
need the raw annotation files. Seed 6304; train2014 images for training, val2014 images for development
and final test; every development / final-test image is used at most once across all evaluation sets.

Grounding templates (generated programmatically from COCO instance annotations):
  presence  "Does this image contain <a/an category>?"  one image with the category annotated (largest
            non-crowd instance >= presence_min_area_frac of the image), one annotated image with no
            annotation of that category (crowd included) -> yes / no.
  counting  "How many <category plural> can you see?"   two images whose number of annotated instances
            differs, both in 1..4; every instance non-crowd and >= counting_min_area_frac of the image
            (small or crowd instances make counts ambiguous) -> "1".."4". Categories whose instances do not
            correspond to countable objects (broccoli, skis) are not used for counting.
The category is drawn uniformly over eligible categories, then the images; for counting the two counts are
drawn uniformly among the available distinct values. The order of the two images in a pair is random.

VQAv2: complementary pairs whose two questions have the same text and answer type, normalised
consensus answers that differ, and consensus agreement >= 8/10 on each image. The training target is the
most frequent original answer string in the consensus group; all ten answers are kept for scoring.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pa3 import SEED
from pa3.common.paths import INPUT, SPLITS, scratch_dir
from pa3.common.runlog import read_json, write_json
from pa3.task2.vqa_norm import consensus, normalize

SPLIT_DIR = SPLITS / "task2"
COCO_IMG_URL = "http://images.cocodataset.org/{split}/{file}"
RAW_URLS = {
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    **{f"{n}.zip": f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/{n}.zip" for n in [
        "v2_Questions_Train_mscoco", "v2_Questions_Val_mscoco", "v2_Annotations_Train_mscoco",
        "v2_Annotations_Val_mscoco", "v2_Complementary_Pairs_Train_mscoco", "v2_Complementary_Pairs_Val_mscoco"]},
}

# (phrase with article for presence, plural for counting); None plural = not used for counting
CATEGORY_PHRASES = {
    "person": ("a person", "people"), "bicycle": ("a bicycle", "bicycles"), "car": ("a car", "cars"),
    "motorcycle": ("a motorcycle", "motorcycles"), "airplane": ("an airplane", "airplanes"),
    "bus": ("a bus", "buses"), "train": ("a train", "trains"), "truck": ("a truck", "trucks"),
    "boat": ("a boat", "boats"), "traffic light": ("a traffic light", "traffic lights"),
    "fire hydrant": ("a fire hydrant", "fire hydrants"), "stop sign": ("a stop sign", "stop signs"),
    "parking meter": ("a parking meter", "parking meters"), "bench": ("a bench", "benches"),
    "bird": ("a bird", "birds"), "cat": ("a cat", "cats"), "dog": ("a dog", "dogs"), "horse": ("a horse", "horses"),
    "sheep": ("a sheep", "sheep"), "cow": ("a cow", "cows"), "elephant": ("an elephant", "elephants"),
    "bear": ("a bear", "bears"), "zebra": ("a zebra", "zebras"), "giraffe": ("a giraffe", "giraffes"),
    "backpack": ("a backpack", "backpacks"), "umbrella": ("an umbrella", "umbrellas"),
    "handbag": ("a handbag", "handbags"), "tie": ("a tie", "ties"), "suitcase": ("a suitcase", "suitcases"),
    "frisbee": ("a frisbee", "frisbees"), "skis": ("skis", None), "snowboard": ("a snowboard", "snowboards"),
    "sports ball": ("a ball", "balls"), "kite": ("a kite", "kites"), "baseball bat": ("a baseball bat", "baseball bats"),
    "baseball glove": ("a baseball glove", "baseball gloves"), "skateboard": ("a skateboard", "skateboards"),
    "surfboard": ("a surfboard", "surfboards"), "tennis racket": ("a tennis racket", "tennis rackets"),
    "bottle": ("a bottle", "bottles"), "wine glass": ("a wine glass", "wine glasses"), "cup": ("a cup", "cups"),
    "fork": ("a fork", "forks"), "knife": ("a knife", "knives"), "spoon": ("a spoon", "spoons"),
    "bowl": ("a bowl", "bowls"), "banana": ("a banana", "bananas"), "apple": ("an apple", "apples"),
    "sandwich": ("a sandwich", "sandwiches"), "orange": ("an orange", "oranges"), "broccoli": ("broccoli", None),
    "carrot": ("a carrot", "carrots"), "hot dog": ("a hot dog", "hot dogs"), "pizza": ("a pizza", "pizzas"),
    "donut": ("a donut", "donuts"), "cake": ("a cake", "cakes"), "chair": ("a chair", "chairs"),
    "couch": ("a couch", "couches"), "potted plant": ("a potted plant", "potted plants"), "bed": ("a bed", "beds"),
    "dining table": ("a dining table", "dining tables"), "toilet": ("a toilet", "toilets"), "tv": ("a TV", "TVs"),
    "laptop": ("a laptop", "laptops"), "mouse": ("a computer mouse", "computer mice"),
    "remote": ("a remote control", "remote controls"), "keyboard": ("a keyboard", "keyboards"),
    "cell phone": ("a cell phone", "cell phones"), "microwave": ("a microwave", "microwaves"),
    "oven": ("an oven", "ovens"), "toaster": ("a toaster", "toasters"), "sink": ("a sink", "sinks"),
    "refrigerator": ("a refrigerator", "refrigerators"), "book": ("a book", "books"), "clock": ("a clock", "clocks"),
    "vase": ("a vase", "vases"), "scissors": ("scissors", "pairs of scissors"),
    "teddy bear": ("a teddy bear", "teddy bears"), "hair drier": ("a hair drier", "hair driers"),
    "toothbrush": ("a toothbrush", "toothbrushes"),
}


def presence_question(cat: str) -> str:
    return f"Does this image contain {CATEGORY_PHRASES[cat][0]}?"


def counting_question(cat: str) -> str:
    return f"How many {CATEGORY_PHRASES[cat][1]} can you see?"


def image_key(split: str, image_id: int) -> str:
    return f"{split}/COCO_{split}_{image_id:012d}.jpg"


def _rng(name: str) -> np.random.Generator:
    """Independent, named, seeded generator per selection component."""
    return np.random.default_rng([SEED, sum(ord(c) * 131 ** i for i, c in enumerate(name)) % (2**31)])


# ----------------------------------------------------------------------------- raw files
def raw_dir() -> Path:
    return Path(os.environ.get("PA3_T2_RAW", scratch_dir("task2", "raw")))


def ensure_raw() -> Path:
    import urllib.request

    d = raw_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, url in RAW_URLS.items():
        marker = d / f".{name}.unzipped"
        if marker.exists():
            continue
        z = d / name
        if not z.exists():
            print(f"[data] downloading {url}", flush=True)
            urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        marker.touch()
    return d


def _find(d: Path, name: str) -> Path:
    hits = list(d.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {d}")
    return hits[0]


# ----------------------------------------------------------------------------- tokenizer caps
class Caps:
    def __init__(self, tok, caps: dict):
        self.tok, self.caps = tok, caps

    def n(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def ok(self, kind: str, text: str) -> bool:
        return self.n(text) <= self.caps[kind]


# ----------------------------------------------------------------------------- captions
def select_captions(d: Path, cfg: dict, caps: Caps, used_eval: set) -> tuple[dict, dict]:
    cc = cfg["data"]["captions"]
    out, stats = {}, {}
    for split in ("train2014", "val2014"):
        js = json.load(open(_find(d, f"captions_{split}.json")))
        by_img = defaultdict(list)
        for a in sorted(js["annotations"], key=lambda a: a["id"]):
            by_img[a["image_id"]].append(a)
        del js
        n_long = 0
        elig = {}
        for img, anns in by_img.items():
            good = []
            for a in anns:
                if caps.ok("caption", a["caption"].strip()):
                    good.append(a)
                else:
                    n_long += 1
            elig[img] = good
        stats[split] = {"images": len(by_img), "captions_over_cap": n_long}
        if split == "train2014":
            pool = sorted(i for i, g in elig.items() if len(g) >= cc["captions_per_train_image"])
            rng = _rng("captions_train")
            chosen = rng.choice(pool, size=cc["train_images"], replace=False)
            recs = []
            for img in sorted(int(i) for i in chosen):
                g = elig[img]
                idx = sorted(rng.choice(len(g), size=cc["captions_per_train_image"], replace=False).tolist())
                recs.append({"image": image_key(split, img), "image_id": img,
                             "caption_ids": [g[i]["id"] for i in idx],
                             "captions": [" ".join(g[i]["caption"].split()) for i in idx]})
            out["train"] = recs
        else:
            pool = sorted(i for i, g in elig.items() if g and image_key(split, i) not in used_eval)
            rng = _rng("captions_dev")
            chosen = sorted(int(i) for i in rng.choice(pool, size=cc["dev_images"], replace=False))
            recs = []
            for img in chosen:
                g = elig[img]
                a = g[int(rng.integers(len(g)))]
                recs.append({"image": image_key(split, img), "image_id": img, "caption_id": a["id"],
                             "caption": " ".join(a["caption"].split())})
                used_eval.add(image_key(split, img))
            out["dev"] = recs
    return out, stats


# ----------------------------------------------------------------------------- grounding
class InstanceIndex:
    """Per (image, category): non-crowd count, crowd count, largest / smallest non-crowd area fraction."""

    def __init__(self, path: Path, split: str):
        js = json.load(open(path))
        self.split = split
        self.cats = {c["id"]: c["name"] for c in js["categories"]}
        self.size = {im["id"]: im["width"] * im["height"] for im in js["images"]}
        st: dict = {}
        for a in js["annotations"]:
            k = (a["image_id"], a["category_id"])
            s = st.setdefault(k, [0, 0, 0.0, 1.0])
            if a.get("iscrowd", 0):
                s[1] += 1
            else:
                f = a["area"] / self.size[a["image_id"]]
                s[0] += 1
                s[2] = max(s[2], f)
                s[3] = min(s[3], f)
        del js
        self.stats = st
        self.annotated = sorted({i for i, _ in st})
        self.has = defaultdict(set)
        for (i, c) in st:
            self.has[i].add(c)

    def presence_pos(self, gc: dict) -> dict[int, list[int]]:
        out = defaultdict(list)
        for (i, c), (n, crowd, mx, mn) in self.stats.items():
            if n >= 1 and mx >= gc["presence_min_area_frac"]:
                out[c].append(i)
        return {c: sorted(v) for c, v in out.items()}

    def counting(self, gc: dict) -> dict[int, dict[int, list[int]]]:
        out = defaultdict(lambda: defaultdict(list))
        for (i, c), (n, crowd, mx, mn) in self.stats.items():
            if CATEGORY_PHRASES[self.cats[c]][1] is None:
                continue
            if crowd == 0 and n in gc["counts"] and mn >= gc["counting_min_area_frac"]:
                out[c][n].append(i)
        return {c: {k: sorted(v) for k, v in d.items()} for c, d in out.items() if len(d) >= 2}


def _presence_pairs(idx: InstanceIndex, gc: dict, n: int, rng, used: set | None, tag: str) -> list[dict]:
    pos = idx.presence_pos(gc)
    cats = sorted(c for c, v in pos.items() if len(v) >= 2)
    pairs, seen, tries = [], set(), 0
    while len(pairs) < n:
        tries += 1
        if tries > 200 * n:
            raise RuntimeError("could not sample enough presence pairs")
        c = int(rng.choice(cats))
        p = int(rng.choice(pos[c]))
        q = int(rng.choice(idx.annotated))
        if c in idx.has[q] or p == q or (p, q) in seen:
            continue
        kp, kq = image_key(idx.split, p), image_key(idx.split, q)
        if used is not None and (kp in used or kq in used):
            continue
        seen.add((p, q))
        if used is not None:
            used.update([kp, kq])
        name = idx.cats[c]
        items = [{"image": kp, "image_id": p, "answer": "yes"}, {"image": kq, "image_id": q, "answer": "no"}]
        if rng.random() < 0.5:
            items = items[::-1]
        pid = f"{tag}_presence_{len(pairs):05d}"
        for j, it in enumerate(items):
            it.update(qid=f"{pid}_{'ab'[j]}", canonical=normalize(it["answer"]), refs=[it["answer"]])
        pairs.append({"pair_id": pid, "type": "presence", "category": name, "question": presence_question(name),
                      "items": items})
    return pairs


def _counting_pairs(idx: InstanceIndex, gc: dict, n: int, rng, used: set | None, tag: str) -> list[dict]:
    cnt = idx.counting(gc)
    cats = sorted(cnt)
    pairs, seen, tries = [], set(), 0
    while len(pairs) < n:
        tries += 1
        if tries > 200 * n:
            raise RuntimeError("could not sample enough counting pairs")
        c = int(rng.choice(cats))
        avail = sorted(cnt[c])
        a, b = sorted(int(x) for x in rng.choice(avail, size=2, replace=False))
        p, q = int(rng.choice(cnt[c][a])), int(rng.choice(cnt[c][b]))
        if (p, q, c) in seen:
            continue
        kp, kq = image_key(idx.split, p), image_key(idx.split, q)
        if used is not None and (kp in used or kq in used):
            continue
        seen.add((p, q, c))
        if used is not None:
            used.update([kp, kq])
        name = idx.cats[c]
        items = [{"image": kp, "image_id": p, "answer": str(a)}, {"image": kq, "image_id": q, "answer": str(b)}]
        if rng.random() < 0.5:
            items = items[::-1]
        pid = f"{tag}_counting_{len(pairs):05d}"
        for j, it in enumerate(items):
            it.update(qid=f"{pid}_{'ab'[j]}", canonical=normalize(it["answer"]), refs=[it["answer"]])
        pairs.append({"pair_id": pid, "type": "counting", "category": name, "question": counting_question(name),
                      "items": items})
    return pairs


def select_grounding(d: Path, cfg: dict, used_eval: set) -> tuple[dict, dict]:
    gc = cfg["data"]["grounding"]
    out: dict = {}
    tr = InstanceIndex(_find(d, "instances_train2014.json"), "train2014")
    out["train_pool"] = {
        "presence": _presence_pairs(tr, gc, gc["train_pool"]["presence"], _rng("g_train_presence"), None, "gtrain"),
        "counting": _counting_pairs(tr, gc, gc["train_pool"]["counting"], _rng("g_train_counting"), None, "gtrain"),
    }
    stats = {"train2014": _grounding_stats(tr, gc)}
    del tr
    va = InstanceIndex(_find(d, "instances_val2014.json"), "val2014")
    for split in ("test", "dev"):  # test first: it is the larger set
        k = gc[f"{split}_pairs"]
        out[split] = (_presence_pairs(va, gc, k["presence"], _rng(f"g_{split}_presence"), used_eval, f"g{split}")
                      + _counting_pairs(va, gc, k["counting"], _rng(f"g_{split}_counting"), used_eval, f"g{split}"))
    stats["val2014"] = _grounding_stats(va, gc)
    return out, stats


def _grounding_stats(idx: InstanceIndex, gc: dict) -> dict:
    pos = idx.presence_pos(gc)
    cnt = idx.counting(gc)
    return {"annotated_images": len(idx.annotated), "presence_categories": len(pos),
            "presence_positive_images": int(sum(len(v) for v in pos.values())),
            "counting_categories": len(cnt),
            "counting_images_by_count": {str(k): int(sum(len(d.get(k, [])) for d in cnt.values()))
                                         for k in gc["counts"]}}


# ----------------------------------------------------------------------------- VQAv2
def _vqa_split(d: Path, split: str, cfg: dict, caps: Caps) -> tuple[list[dict], dict]:
    vc = cfg["data"]["vqa"]
    qs = json.load(open(_find(d, f"v2_OpenEnded_mscoco_{split}_questions.json")))["questions"]
    qmap = {q["question_id"]: q for q in qs}
    del qs
    anns = json.load(open(_find(d, f"v2_mscoco_{split}_annotations.json")))["annotations"]
    amap = {a["question_id"]: a for a in anns}
    del anns
    comp = json.load(open(_find(d, f"v2_mscoco_{split}_complementary_pairs.json")))
    reasons = Counter()
    cand = []
    for q1, q2 in comp:
        a1, a2 = amap[q1], amap[q2]
        t1, t2 = qmap[q1]["question"].strip(), qmap[q2]["question"].strip()
        if " ".join(t1.lower().split()) != " ".join(t2.lower().split()):
            reasons["question_text_differs"] += 1
            continue
        if a1["answer_type"] != a2["answer_type"]:
            reasons["answer_type_differs"] += 1
            continue
        c1 = consensus([x["answer"] for x in a1["answers"]])
        c2 = consensus([x["answer"] for x in a2["answers"]])
        if c1[1] < vc["min_agreement"] or c2[1] < vc["min_agreement"]:
            reasons["agreement_below_min"] += 1
            continue
        if c1[0] == c2[0]:
            reasons["same_consensus"] += 1
            continue
        if not (caps.ok("question", t1) and caps.ok("question", t2)):
            reasons["question_over_cap"] += 1
            continue
        if not (caps.ok("answer", c1[2]) and caps.ok("answer", c2[2])):
            reasons["answer_over_cap"] += 1
            continue
        if a1["image_id"] == a2["image_id"]:
            reasons["same_image"] += 1
            continue
        items = []
        for qid, a, c, t in ((q1, a1, c1, t1), (q2, a2, c2, t2)):
            items.append({"qid": str(qid), "question": t, "image": image_key(split, a["image_id"]),
                          "image_id": a["image_id"], "answer": c[2], "canonical": c[0], "agreement": c[1],
                          "refs": [x["answer"] for x in a["answers"]]})
        cand.append({"pair_id": f"{q1}_{q2}", "type": a1["answer_type"], "question": t1, "items": items})
    reasons["kept"] = len(cand)
    reasons["total_pairs"] = len(comp)
    return sorted(cand, key=lambda p: p["pair_id"]), dict(reasons)


def _sample_vqa(cand: list[dict], counts: dict, rng, used: set | None) -> list[dict]:
    out = []
    for t in sorted(counts):
        pool = [p for p in cand if p["type"] == t]
        order = rng.permutation(len(pool))
        got = 0
        for i in order:
            p = pool[int(i)]
            keys = [it["image"] for it in p["items"]]
            if used is not None and any(k in used for k in keys):
                continue
            if used is not None:
                used.update(keys)
            out.append(p)
            got += 1
            if got == counts[t]:
                break
        if got < counts[t]:
            raise RuntimeError(f"not enough VQA pairs of type {t}")
    return out


def select_vqa(d: Path, cfg: dict, caps: Caps, used_eval: set) -> tuple[dict, dict]:
    vc = cfg["data"]["vqa"]
    out, stats = {}, {}
    cand, stats["train2014"] = _vqa_split(d, "train2014", cfg, caps)
    out["train"] = _sample_vqa(cand, vc["train_pairs"], _rng("vqa_train"), None)
    cand, stats["val2014"] = _vqa_split(d, "val2014", cfg, caps)
    out["test"] = _sample_vqa(cand, vc["test_pairs"], _rng("vqa_test"), used_eval)
    out["dev"] = _sample_vqa(cand, vc["dev_pairs"], _rng("vqa_dev"), used_eval)
    return out, stats


# ----------------------------------------------------------------------------- mismatched mapping
def mismatched_mapping(pairs: list[dict], rng) -> dict[str, str]:
    """qid -> image of the same position in the next pair of a seeded cyclic order: always from another pair
    of the same suite, never either image of the question's own pair."""
    order = rng.permutation(len(pairs))
    mp = {}
    for i, pi in enumerate(order):
        p, nxt = pairs[int(pi)], pairs[int(order[(i + 1) % len(order)])]
        for j, it in enumerate(p["items"]):
            mp[it["qid"]] = nxt["items"][j]["image"]
    return mp


# ----------------------------------------------------------------------------- main selection
def select(cfg: dict | None = None) -> dict:
    from transformers import AutoTokenizer

    from pa3.task2 import config as C

    cfg = cfg or C.config("t196")
    t0 = time.time()
    d = ensure_raw()
    tok = AutoTokenizer.from_pretrained(cfg["lm"])
    caps = Caps(tok, cfg["caps"])
    used_eval: set = set()
    vqa, vqa_stats = select_vqa(d, cfg, caps, used_eval)            # scarcest first
    grounding, g_stats = select_grounding(d, cfg, used_eval)
    captions, c_stats = select_captions(d, cfg, caps, used_eval)
    mism = {"grounding": mismatched_mapping(grounding["test"], _rng("mismatch_grounding")),
            "vqa": mismatched_mapping(vqa["test"], _rng("mismatch_vqa"))}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SPLIT_DIR / "captions.json", captions)
    write_json(SPLIT_DIR / "grounding.json", grounding)
    write_json(SPLIT_DIR / "vqa.json", vqa)
    write_json(SPLIT_DIR / "mismatched.json", mism)
    summary = {"seed": SEED, "seconds": time.time() - t0, "config": {k: cfg[k] for k in ("data", "caps", "prompts")},
               "caption_stats": c_stats, "grounding_stats": g_stats, "vqa_filter": vqa_stats,
               "isolation": isolation_report(), "counts": counts_report()}
    write_json(SPLIT_DIR / "selection_summary.json", summary)
    print(json.dumps(summary["isolation"], indent=1), json.dumps(summary["counts"], indent=1))
    return summary


# ----------------------------------------------------------------------------- loading + checks
def load_splits() -> dict:
    return {k: read_json(SPLIT_DIR / f"{k}.json") for k in ("captions", "grounding", "vqa", "mismatched")}


def eval_sets(s: dict | None = None) -> dict[str, list]:
    s = s or load_splits()
    return {"caption_dev": [r["image"] for r in s["captions"]["dev"]],
            "grounding_dev": [it["image"] for p in s["grounding"]["dev"] for it in p["items"]],
            "grounding_test": [it["image"] for p in s["grounding"]["test"] for it in p["items"]],
            "vqa_dev": [it["image"] for p in s["vqa"]["dev"] for it in p["items"]],
            "vqa_test": [it["image"] for p in s["vqa"]["test"] for it in p["items"]]}


def train_images(s: dict | None = None, subset: dict | None = None) -> list[str]:
    s = s or load_splits()
    t = training_lists(s, subset)
    keys = {r["image"] for r in t["caption"]}
    for k in ("presence", "counting", "vqa"):
        keys |= {it["image"] for p in t[k] for it in p["items"]}
    return sorted(keys)


def training_lists(s: dict, subset: dict | None = None) -> dict[str, list]:
    t = {"caption": s["captions"]["train"], "presence": s["grounding"]["train_pool"]["presence"],
         "counting": s["grounding"]["train_pool"]["counting"], "vqa": s["vqa"]["train"]}
    if subset:  # dry runs only
        t = {"caption": t["caption"][:subset["caption_train_images"]], "presence": t["presence"][:subset["presence"]],
             "counting": t["counting"][:subset["counting"]], "vqa": t["vqa"][:subset["vqa"]]}
    return t


def isolation_report(s: dict | None = None) -> dict:
    s = s or load_splits()
    ev = eval_sets(s)
    tr = set(train_images(s))
    rep = {"train_images": len(tr), "train_all_from_train2014": all(k.startswith("train2014/") for k in tr)}
    allev = []
    for name, keys in ev.items():
        rep[f"{name}_images"] = len(keys)
        rep[f"{name}_all_from_val2014"] = all(k.startswith("val2014/") for k in keys)
        rep[f"{name}_overlap_train"] = len(set(keys) & tr)
        allev += keys
    rep["eval_images_total"] = len(allev)
    rep["eval_images_unique"] = len(set(allev))
    rep["eval_sets_pairwise_disjoint"] = len(allev) == len(set(allev))
    names = list(ev)
    rep["pairwise_overlaps"] = {f"{a}|{b}": len(set(ev[a]) & set(ev[b]))
                                for i, a in enumerate(names) for b in names[i + 1:]}
    ok_mm = True
    for suite in ("grounding", "vqa"):
        own = {it["qid"]: {x["image"] for x in p["items"]} for p in s[suite]["test"] for it in p["items"]}
        test_imgs = {it["image"] for p in s[suite]["test"] for it in p["items"]}
        for q, img in s["mismatched"][suite].items():
            ok_mm &= img not in own[q] and img in test_imgs
    rep["mismatched_mapping_valid"] = bool(ok_mm)
    rep["ok"] = bool(rep["train_all_from_train2014"] and rep["eval_sets_pairwise_disjoint"] and ok_mm
                     and all(rep[f"{n}_overlap_train"] == 0 and rep[f"{n}_all_from_val2014"] for n in ev))
    return rep


def counts_report(s: dict | None = None) -> dict:
    s = s or load_splits()
    out = {"caption_train_images": len(s["captions"]["train"]),
           "caption_train_examples": sum(len(r["captions"]) for r in s["captions"]["train"]),
           "caption_dev_images": len(s["captions"]["dev"])}
    for k in ("presence", "counting"):
        out[f"grounding_train_pool_{k}_pairs"] = len(s["grounding"]["train_pool"][k])
    for sp in ("dev", "test"):
        out[f"grounding_{sp}_pairs_by_type"] = dict(Counter(p["type"] for p in s["grounding"][sp]))
    for sp in ("train", "dev", "test"):
        out[f"vqa_{sp}_pairs_by_type"] = dict(Counter(p["type"] for p in s["vqa"][sp]))
    out["grounding_test_answers"] = dict(Counter(it["answer"] for p in s["grounding"]["test"] for it in p["items"]))
    out["grounding_pool_answers"] = dict(Counter(it["answer"] for k in ("presence", "counting")
                                                 for p in s["grounding"]["train_pool"][k] for it in p["items"]))
    return out


# ----------------------------------------------------------------------------- images
def image_root() -> Path:
    return scratch_dir("task2", "images")


def _kaggle_coco_dirs() -> dict[str, Path]:
    """Use an attached Kaggle COCO-2014 dataset if one is present (any layout with train2014/ val2014 dirs)."""
    out = {}
    if INPUT.exists():
        for split in ("train2014", "val2014"):
            for p in INPUT.glob(f"**/{split}"):
                if p.is_dir() and any(p.glob("COCO_*.jpg")):
                    out[split] = p
                    break
    return out


def image_path(key: str) -> Path:
    return image_root() / key


def download_images(keys: list[str], workers: int = 48) -> dict:
    import shutil
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    root = image_root()
    local = _kaggle_coco_dirs()
    todo = [k for k in keys if not (root / k).exists()]
    t0 = time.time()
    stats = Counter()

    def one(k: str) -> str:
        split, fname = k.split("/")
        dst = root / k
        dst.parent.mkdir(parents=True, exist_ok=True)
        if split in local and (local[split] / fname).exists():
            shutil.copyfile(local[split] / fname, dst)
            return "copied"
        url = COCO_IMG_URL.format(split=split, file=fname)
        for attempt in range(5):
            try:
                tmp = dst.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                tmp.rename(dst)
                return "downloaded"
            except Exception:
                time.sleep(2 * (attempt + 1))
        return "failed"

    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, todo)):
            stats[res] += 1
            if (i + 1) % 2000 == 0:
                print(f"[data] images {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    failed = [k for k in keys if not (root / k).exists()]
    if failed:
        raise RuntimeError(f"{len(failed)} images could not be fetched, e.g. {failed[:3]}")
    rep = {"requested": len(keys), "already_present": len(keys) - len(todo), **stats, "seconds": time.time() - t0,
           "kaggle_dataset_dirs": {k: str(v) for k, v in local.items()}}
    print(f"[data] images ready: {rep}", flush=True)
    return rep


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "check"])
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    else:
        print(json.dumps(isolation_report(), indent=1), json.dumps(counts_report(), indent=1))


if __name__ == "__main__":
    main()
