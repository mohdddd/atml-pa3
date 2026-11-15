"""Step 4: parallel sampling vs a single longer response vs sequential refinement (live generation, one GPU).

For each of the 50 fixed MATH evaluation questions (splits/task3/live_questions.json), with the same 3,072-token
output allowance:
  single     one response, max 3,072 new tokens
  parallel   three independent responses, max 1,024 each; self-consistency over their answers
  refine     the FIRST parallel response is reused as the initial response; then one critique-and-revision
             response (max 2,048) to [system, user(problem + instruction), assistant(initial), user(review)];
             the revised answer is returned (no gold / correctness feedback)
Qwen/Qwen2.5-3B-Instruct at the pinned revision, FP16, no quantisation, batch size 1, temperature 0.7,
top-p 0.8, top-k 20, repetition penalty 1.0, native chat template, system "You are a helpful assistant.".
Every call is seeded independently (seed = 6304 * 1000 + 10 * question_position + call_index; call_index
0 = single, 1-3 = parallel, 4 = revision). Calls run in the order single, parallel 1-3, revision, question by
question, all on the same GPU; one short untimed warm-up generation precedes the first timed call.
End-to-end time per call = tokenisation + generate + decoding (CUDA synchronised), excluding model loading.

    python -m pa3.task3.live run       # resumable: finished calls (results/task3/live/calls.jsonl) are skipped
    python -m pa3.task3.live analyze   # -> results/task3/live/{per_question.csv, summary.json, transitions.csv}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pa3.common.paths import results_dir
from pa3.common.runlog import read_json, read_jsonl, save_config, write_json
from pa3.task3.config import NO_ANSWER, SPLIT_DIR, config, results_task

CALLS = [("single", 0), ("parallel", 1), ("parallel", 2), ("parallel", 3), ("revision", 4)]


def extract_boxed(text: str):
    """Last literal '\\boxed{' with balanced-brace scanning (supplied README rule); absent/empty/unclosed -> None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth, k = 1, j
    while k < len(text):
        ch = text[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                ans = text[j:k].strip()
                return ans or None
        k += 1
    return None


def call_seed(seed: int, qpos: int, call_index: int) -> int:
    return seed * 1000 + 10 * qpos + call_index


def out_dir() -> Path:
    return results_dir(results_task(), "live")


def questions(cfg: dict) -> list[dict]:
    q = read_json(SPLIT_DIR / "live_questions.json")
    ids = read_json(SPLIT_DIR / "splits.json")["live_math"]
    assert [x["sample_id"] for x in q] == ids
    return q[:cfg["live"]["n_questions"]]


def messages(cfg: dict, problem: str, initial: str | None = None) -> list[dict]:
    lc = cfg["live"]
    m = [{"role": "system", "content": lc["system"]},
         {"role": "user", "content": problem + "\n\n" + lc["instruction"]}]
    if initial is not None:
        m += [{"role": "assistant", "content": initial}, {"role": "user", "content": lc["review_instruction"]}]
    return m


class Generator:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lc = cfg["live"]
        self.torch = torch
        self.cfg = cfg
        self.device = lc["device"] if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(lc["model"], revision=lc["revision"])
        dtype = getattr(torch, lc["dtype"]) if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(lc["model"], revision=lc["revision"], dtype=dtype).to(self.device)
        self.model.eval()
        self.load_seconds = time.time() - t0
        self.eos = self.model.generation_config.eos_token_id
        self.eos = set(self.eos if isinstance(self.eos, list) else [self.eos])
        self.info = {"model": lc["model"], "revision": lc["revision"], "device": self.device,
                     "dtype": str(next(self.model.parameters()).dtype), "eos_token_ids": sorted(self.eos),
                     "load_seconds": self.load_seconds,
                     "gpu": torch.cuda.get_device_name(self.device) if self.device != "cpu" else None}

    def sync(self):
        if self.device != "cpu":
            self.torch.cuda.synchronize(self.device)

    def generate(self, msgs: list[dict], max_new: int, seed: int) -> dict:
        torch = self.torch
        lc = self.cfg["live"]
        if self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sync()
        t0 = time.perf_counter()
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        if self.device != "cpu":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = self.model.generate(**enc, do_sample=True, max_new_tokens=max_new,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                                      **lc["sampling"])
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        text = self.tok.decode(gen, skip_special_tokens=True)
        self.sync()
        dt = time.perf_counter() - t0
        stopped = len(gen) > 0 and gen[-1] in self.eos
        n_out = len(gen) - (1 if stopped else 0)          # retained tokens (stop token excluded, as in the data)
        return {"prompt": prompt, "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": n_out,
                "output_tokens_incl_stop": len(gen), "truncated": (not stopped) and len(gen) >= max_new,
                "max_new_tokens": max_new, "response": text, "seconds": dt,
                "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated(self.device) / 2**30) if self.device != "cpu" else None}


def run(push_every: int = 5) -> None:
    cfg = config()
    lc = cfg["live"]
    od = out_dir()
    save_config(od / "config.yaml", cfg)
    path = od / "calls.jsonl"
    done = {r["call_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    qs = questions(cfg)
    todo = [(qp, q) for qp, q in enumerate(qs)
            if any(f"{q['sample_id']}|{s}|{k}" not in done for s, k in CALLS)]
    if not todo:
        print("[live] all calls done", flush=True)
        return
    g = Generator(cfg)
    write_json(od / "generator_info.json", g.info)
    g.generate(messages(cfg, "What is 1 + 1?"), 8, 0)       # untimed warm-up
    deadline = float(os.environ.get("PA3_DEADLINE", "inf"))
    caps = {"single": lc["caps"]["single"], "parallel": lc["caps"]["parallel"], "revision": lc["caps"]["revision"]}
    assert lc["caps"]["initial"] == lc["caps"]["parallel"]
    for n_done, (qp, q) in enumerate(todo, 1):
        if time.time() > deadline - 15 * 60:
            print("[live] close to the session deadline; stopping (rerun to resume)", flush=True)
            break
        for strat, k in CALLS:
            cid = f"{q['sample_id']}|{strat}|{k}"
            if cid in done:
                continue
            if strat == "revision":
                init = done[f"{q['sample_id']}|parallel|1"]["response"]
                msgs = messages(cfg, q["question_text"], init)
            else:
                msgs = messages(cfg, q["question_text"])
            seed = call_seed(cfg["seed"], qp, k)
            r = g.generate(msgs, caps[strat], seed)
            r.update({"call_id": cid, "sample_id": q["sample_id"], "qpos": qp, "strategy": strat, "call_index": k,
                      "seed": seed, "answer": extract_boxed(r["response"]), "finished_at": time.time()})
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            done[cid] = r
            print(f"[live] q{qp:02d} {strat}{k} out={r['output_tokens']} trunc={r['truncated']} "
                  f"{r['seconds']:.1f}s ans={r['answer']!r}", flush=True)
        if n_done % push_every == 0:
            _push([od], f"task3: live generation progress ({len({c.split('|')[0] for c in done})} questions)")
    _push([od], "task3: live generation calls")


def _push(paths, msg):
    try:
        from pa3.common.gitsync import push

        push(paths, msg)
    except Exception as e:
        print(f"[git] {e}", flush=True)


def analyze() -> dict:
    from pa3.task3.answers import MathScorer, group_answers
    from pa3.task3.selection import self_consistency
    from pa3.task3.stats import paired_bootstrap

    cfg = config()
    od = out_dir()
    calls = {r["call_id"]: r for r in read_jsonl(od / "calls.jsonl")}
    sc = MathScorer()
    rows = []
    for qp, q in enumerate(questions(cfg)):
        sid = q["sample_id"]
        c = {(s, k): calls.get(f"{sid}|{s}|{k}") for s, k in CALLS}
        if any(v is None for v in c.values()):
            continue
        gold = q["gold_answer"]
        grade = lambda a: a is not None and sc.equiv(gold, a)
        par = [c[("parallel", k)] for k in (1, 2, 3)]
        p_ans = [p["answer"] for p in par]
        gids, reps = group_answers(p_ans, sc.equiv)
        win, pos = self_consistency(gids)
        sc_ans = reps[win]
        init, rev, single = par[0], c[("revision", 4)], c[("single", 0)]
        rows.append({
            "sample_id": sid, "qpos": qp, "gold": gold,
            "single_answer": single["answer"], "single_correct": grade(single["answer"]),
            "single_in": single["input_tokens"], "single_out": single["output_tokens"],
            "single_seconds": single["seconds"], "single_truncated": single["truncated"],
            "single_extracted": single["answer"] is not None, "single_peak_gb": single["peak_gpu_mem_gb"],
            "parallel_answers": json.dumps(p_ans), "parallel_answer": None if sc_ans == NO_ANSWER else sc_ans,
            "parallel_correct": sc_ans != NO_ANSWER and grade(sc_ans),
            "parallel_n_correct_candidates": sum(grade(a) for a in p_ans),
            "parallel_in": sum(p["input_tokens"] for p in par), "parallel_out": sum(p["output_tokens"] for p in par),
            "parallel_seconds": sum(p["seconds"] for p in par),
            "parallel_truncated_calls": sum(p["truncated"] for p in par),
            "parallel_extracted_calls": sum(p["answer"] is not None for p in par),
            "parallel_selected_extracted": sc_ans != NO_ANSWER,
            "parallel_peak_gb": max(p["peak_gpu_mem_gb"] or 0 for p in par),
            "initial_answer": init["answer"], "initial_correct": grade(init["answer"]),
            "refine_answer": rev["answer"], "refine_correct": grade(rev["answer"]),
            "refine_answer_changed": (init["answer"] is None) != (rev["answer"] is None) or (
                init["answer"] is not None and rev["answer"] is not None and not sc.equiv(init["answer"], rev["answer"])),
            "refine_in": init["input_tokens"] + rev["input_tokens"], "revision_prompt_tokens": rev["input_tokens"],
            "refine_out": init["output_tokens"] + rev["output_tokens"], "revision_out": rev["output_tokens"],
            "refine_seconds": init["seconds"] + rev["seconds"], "revision_seconds": rev["seconds"],
            "initial_truncated": init["truncated"], "revision_truncated": rev["truncated"],
            "refine_extracted": rev["answer"] is not None,
            "refine_peak_gb": max(init["peak_gpu_mem_gb"] or 0, rev["peak_gpu_mem_gb"] or 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(od / "per_question.csv", index=False)
    n = len(df)
    summ = {"n_questions": n, "scorer": "math-verify (pa3.task3.answers.MathScorer)", "strategies": {}}
    for s, trunc in [("single", "single_truncated"), ("parallel", None), ("refine", None)]:
        d = {"accuracy": float(df[f"{s}_correct"].mean() * 100),
             "mean_input_tokens": float(df[f"{s}_in"].mean()), "mean_output_tokens": float(df[f"{s}_out"].mean()),
             "total_output_tokens": int(df[f"{s}_out"].sum()),
             "mean_seconds": float(df[f"{s}_seconds"].mean()), "total_seconds": float(df[f"{s}_seconds"].sum()),
             "peak_gpu_mem_gb": float(df[f"{s}_peak_gb"].max())}
        if s == "single":
            d.update(extraction_rate=float(df.single_extracted.mean()), truncation_rate=float(df.single_truncated.mean()))
        elif s == "parallel":
            d.update(extraction_rate_calls=float(df.parallel_extracted_calls.sum() / (3 * n)),
                     truncation_rate_calls=float(df.parallel_truncated_calls.sum() / (3 * n)),
                     selected_answer_extracted_rate=float(df.parallel_selected_extracted.mean()),
                     oracle_coverage=float((df.parallel_n_correct_candidates > 0).mean() * 100))
        else:
            d.update(extraction_rate=float(df.refine_extracted.mean()),
                     initial_truncation_rate=float(df.initial_truncated.mean()),
                     revision_truncation_rate=float(df.revision_truncated.mean()),
                     initial_accuracy=float(df.initial_correct.mean() * 100),
                     mean_revision_prompt_tokens=float(df.revision_prompt_tokens.mean()),
                     mean_revision_output_tokens=float(df.revision_out.mean()),
                     answer_changed_rate=float(df.refine_answer_changed.mean()))
        summ["strategies"][s] = d
    tr = pd.crosstab(df.initial_correct.map({True: "initial correct", False: "initial wrong"}),
                     df.refine_correct.map({True: "revised correct", False: "revised wrong"}))
    tr.to_csv(od / "transitions.csv")
    summ["transitions"] = {
        "wrong_to_correct": int((~df.initial_correct & df.refine_correct).sum()),
        "correct_to_wrong": int((df.initial_correct & ~df.refine_correct).sum()),
        "correct_to_correct": int((df.initial_correct & df.refine_correct).sum()),
        "wrong_to_wrong": int((~df.initial_correct & ~df.refine_correct).sum())}
    # transitions split by whether the shared initial response produced an answer / hit its cap
    init_ok = df.initial_answer.notna()
    summ["transitions_by_initial_state"] = {
        lab: {"n": int(m.sum()),
              "wrong_to_correct": int((m & ~df.initial_correct & df.refine_correct).sum()),
              "correct_to_wrong": int((m & df.initial_correct & ~df.refine_correct).sum()),
              "answer_changed": int((m & df.refine_answer_changed).sum())}
        for lab, m in [("initial_answer_extracted", init_ok), ("initial_no_answer", ~init_ok),
                       ("initial_truncated", df.initial_truncated.astype(bool))]}
    # parallel: questions where self-consistency returned NO_ANSWER although a candidate had an answer
    summ["parallel_no_answer_selected"] = int((~df.parallel_selected_extracted).sum())
    summ["parallel_selected_wrong_with_correct_candidate"] = int((~df.parallel_correct & (df.parallel_n_correct_candidates > 0)).sum())
    summ["paired_bootstrap_accuracy_pp"] = {
        f"{a}_minus_{b}": paired_bootstrap(df[f"{a}_correct"] * 100.0, df[f"{b}_correct"] * 100.0,
                                           cfg["offline"]["bootstrap"], cfg["seed"])
        for a, b in [("parallel", "single"), ("refine", "single"), ("parallel", "refine")]}
    summ["cost_note"] = ("refine charges the shared initial response (parallel call 1) in full, plus the revision "
                         "prompt (which contains the initial response) and all critique/revision output tokens; "
                         "parallel = three prompts and three outputs; times are sums of sequential batch-1 calls")
    gi = od / "generator_info.json"
    if gi.exists():
        summ["generator"] = read_json(gi)
    write_json(od / "summary.json", summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    a = ap.parse_args()
    if a.cmd == "run":
        run()
    else:
        print(json.dumps(analyze(), indent=1))


if __name__ == "__main__":
    main()
