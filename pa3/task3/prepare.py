"""Task 3 preparation (CPU): splits, rollout orders, live-question file and the frozen answer labels.

    python -m pa3.task3.prepare            # needs $PA3_T3_DATA -> ATML-PA3-Data folder

Outputs
  splits/task3/splits.json           dev / eval IDs per dataset (+ full permutation) and the 50 live MATH IDs
  splits/task3/orders.json.gz        10 rollout-ID permutations per question (nested prefixes = candidate sets)
  splits/task3/live_questions.json   the 50 live questions (ID, problem text, gold answer)
  results/task3/answers/<ds>_rollouts.parquet   per-rollout answer key, group, NO_ANSWER flag, correctness, length
  results/task3/answers/<ds>_groups.jsonl.gz    answer groups per question (representative, members, correctness)
  results/task3/answers/scorer.json  scorer version (fixed before evaluation) + label summary
  results/task3/data_checks.json     join / token-alignment checks
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from pa3.common.paths import results_dir
from pa3.common.runlog import save_config, write_json, write_jsonl
from pa3.task3 import answers as A
from pa3.task3.config import DATASETS, SPLIT_DIR, config, results_task
from pa3.task3.data import load, make_orders, make_splits, mathvista_meta, save_orders


def main():
    cfg = config()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = results_dir(results_task())
    save_config(out / "config.yaml", cfg)
    splits = make_splits(cfg)
    write_json(SPLIT_DIR / "splits.json", splits)

    checks, orders, summary, timeouts = {}, {}, {}, []
    for k, ds in enumerate(DATASETS):
        t0 = time.time()
        R = load(ds, with_tokens=True, cfg=cfg)          # full join + alignment checks
        checks[ds] = R.checks
        orders[ds] = {"qids": R.qids, "orders": make_orders(R.qids, k, cfg)}
        if ds == "math":
            scorer = A.MathScorer()
            rows, groups = A.build_math(R.gen, R.qids, scorer)
            timeouts = [{"reference": a, "candidate": b} for a, b in scorer.timeouts]
            live = []
            for qid in splits["live_math"]:
                qi = R.qids.index(qid)
                r = R.gen.iloc[qi * 32]
                live.append({"sample_id": qid, "question_text": r.question_text, "gold_answer": r.gold_answer,
                             "sha256": hashlib.sha256(r.question_text.encode()).hexdigest()})
            write_json(SPLIT_DIR / "live_questions.json", live)
        else:
            meta = mathvista_meta(cfg)
            assert set(R.qids) <= set(meta.index)
            rows, groups = A.build_mathvista(R.gen, R.qids, meta)
            gold = R.gen.groupby("sample_id").gold_answer.first()
            mc = meta.loc[R.qids]
            letter = [chr(65 + list(c).index(a)) if t == "multi_choice" else str(a)
                      for c, a, t in zip(mc.choices, mc.answer, mc.question_type)]
            checks[ds]["gold_matches_metadata"] = bool((gold.loc[R.qids].values == np.array(letter)).all())
            rows["question_type"] = np.repeat(mc.question_type.values, 32)
            rows["answer_type"] = np.repeat(mc.answer_type.values, 32)
        rows["n_tokens"] = R.n_tokens
        rows["hit_max_new_tokens"] = R.gen.hit_max_new_tokens.values
        rows["split"] = rows.sample_id.map({**{q: "dev" for q in splits[ds]["dev"]},
                                            **{q: "eval" for q in splits[ds]["eval"]}})
        d = results_dir(results_task(), "answers")
        rows.to_parquet(d / f"{ds}_rollouts.parquet", index=False)
        write_jsonl(d / f"{ds}_groups.jsonl.gz", groups.to_dict("records"))
        q = rows.groupby("sample_id")
        summary[ds] = {
            "rollouts": len(rows), "no_answer": int(rows.no_answer.sum()),
            "invalid_typed_non_null": int((~rows.no_answer & ~rows.valid).sum()),
            "rollout_accuracy": float(rows.correct.mean()),
            "groups_per_question_mean": float(groups.groupby("sample_id").size().mean()),
            "questions_with_any_correct": int(q.correct.any().sum()),
            "group_label_vs_own_string_disagreements": int((rows.correct != rows.correct_own_string).sum()),
            "seconds": round(time.time() - t0, 1),
        }
        print(ds, summary[ds], flush=True)
    save_orders(orders)
    write_json(results_dir(results_task()) / "data_checks.json", checks)
    write_json(results_dir(results_task(), "answers") / "scorer.json",
               {**A.scorer_info(), "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "labels": summary,
                "math_verify_timeouts_treated_as_not_equivalent": timeouts})


if __name__ == "__main__":
    main()
