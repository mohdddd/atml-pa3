"""Offline selection over the supplied pools (Steps 1, 2 and the per-prefix state used by Step 5).

For every question, each of the 10 saved rollout orders and every budget N, the candidate set is the first N
rollouts of the order. Every method sees only that set: scores, VisConf calibration (pooled token percentiles)
and RWC ranks are recomputed from the revealed candidates; gold answers are used only to grade the result.

    python -m pa3.task3.offline        ->  results/task3/offline/<ds>_selections.parquet   (one row per q/order/N/method)
                                            results/task3/offline/<ds>_prefix_state.parquet (coverage, #groups, cost, adaptive state)
                                            results/task3/offline/<ds>_rollout_features.parquet (L, K, V, length per rollout)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pa3.common.paths import results_dir
from pa3.common.runlog import write_json
from pa3.task3 import selection as S
from pa3.task3.config import DATASETS, NO_ANSWER, config, results_task
from pa3.task3.data import N_ROLLOUTS, load, load_orders

# method -> (score name or None, aggregation). "first" = FirstAns; "vote" = self-consistency.
METHODS_BOTH = {
    "first": (None, "first"),
    "self_consistency": (None, "vote"),
    "logp_max": ("L", "max"),
    "logp_rwc": ("L", "rwc"),                 # extra cell of the score x aggregation grid
    "selfcert_max": ("K", "max"),
    "selfcert_rwc": ("K", "rwc"),
    "shortest": ("negT", "max"),              # length-only control (Step 3)
}
METHODS_VLM = {
    "vision_max": ("V", "max"),
    "vision_rwc": ("V", "rwc"),
    "visconf_max": ("Z", "max"),
    "visconf_rwc": ("Z", "rwc"),
    "vctoken_max": ("Ztok", "max"),           # tokenwise fusion ablation (Step 3)
    "vctoken_rwc": ("Ztok", "rwc"),
}


def methods(ds: str) -> dict:
    return {**METHODS_BOTH, **(METHODS_VLM if ds == "mathvista" else {})}


def rollout_features(R, ans: pd.DataFrame) -> pd.DataFrame:
    """Per-rollout L (mean logp), K (mean kl_u_p = self-certainty), V (mean visual engagement), length."""
    n = R.n_tokens
    ends = R.tok_start + n
    f = pd.DataFrame({"sample_id": ans.sample_id, "rollout_id": ans.rollout_id, "n_tokens": n,
                      "group": ans.group, "correct": ans.correct, "no_answer": ans.no_answer,
                      "hit_max_new_tokens": ans.hit_max_new_tokens, "split": ans.split})

    def seg_mean(x):
        cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        return (cs[ends] - cs[R.tok_start]) / np.maximum(n, 1)

    f["L"] = seg_mean(R.tok["logp"])
    f["K"] = seg_mean(R.tok["kl"])
    if "img" in R.tok:
        R.tok["v"] = (R.tok["img"] / (R.tok["img"] + R.tok["prompt"] + R.tok["gen_attn"])).astype(np.float32)
        f["V"] = seg_mean(R.tok["v"])
    return f


def run_dataset(ds: str, cfg: dict) -> dict:
    t0 = time.time()
    out = results_dir(results_task(), "offline")
    R = load(ds, with_tokens=True, cfg=cfg)
    ans = pd.read_parquet(results_dir(results_task(), "answers") / f"{ds}_rollouts.parquet")
    assert ans.sample_id.tolist() == R.gen.sample_id.tolist() and ans.rollout_id.tolist() == R.gen.rollout_id.tolist()
    feats = rollout_features(R, ans)
    feats.to_parquet(out / f"{ds}_rollout_features.parquet", index=False)

    od = load_orders()[ds]
    assert od["qids"] == R.qids
    orders = od["orders"]
    budgets = cfg["offline"]["budgets"]
    qcfg = tuple(cfg["offline"]["calib_percentiles"])
    eps = cfg["offline"]["calib_eps"]
    M = methods(ds)
    vlm = ds == "mathvista"
    qscore = {"self_certainty": "K", "visconf": "Z"}[cfg["adaptive"]["quality_score"][ds]]
    adapt_n = set(cfg["adaptive"]["check_points"]) | {cfg["adaptive"]["max_n"]}

    G = feats.group.values.reshape(-1, N_ROLLOUTS)
    C = feats.correct.values.reshape(-1, N_ROLLOUTS)
    T = feats.n_tokens.values.reshape(-1, N_ROLLOUTS)
    NA = feats.no_answer.values.reshape(-1, N_ROLLOUTS)
    base = {"L": feats.L.values.reshape(-1, N_ROLLOUTS), "K": feats.K.values.reshape(-1, N_ROLLOUTS),
            "negT": -T.astype(np.float64)}
    if vlm:
        base["V"] = feats.V.values.reshape(-1, N_ROLLOUTS)

    sel_rows, state_rows, new_rows = [], [], []
    for qi in range(R.n_questions):
        if vlm:
            kl_r = [R.tokens("kl", R.row(qi, r)) for r in range(N_ROLLOUTS)]
            v_r = [R.tokens("v", R.row(qi, r)) for r in range(N_ROLLOUTS)]
        for o in range(orders.shape[1]):
            order = orders[qi, o].astype(int)
            g_all, c_all = G[qi, order], C[qi, order]
            # new answer group / first correct group per revealed position (Step 6)
            seen, first_corr = set(), False
            for k in range(N_ROLLOUTS):
                is_new = g_all[k] not in seen
                seen.add(g_all[k])
                new_corr = bool(c_all[k]) and not first_corr
                first_corr = first_corr or bool(c_all[k])
                new_rows.append((qi, o, k + 1, is_new, new_corr, bool(NA[qi, order[k]])))
            for N in budgets:
                idx = order[:N]
                g, c, t = G[qi, idx], C[qi, idx], T[qi, idx]
                sc = {k: v[qi, idx] for k, v in base.items()}
                calib = (np.nan, np.nan)
                if vlm:
                    if N > 1:
                        Ci, Vi, Zi, Zt, calib = S.visconf_scores([kl_r[i] for i in idx], [v_r[i] for i in idx],
                                                                 qcfg, eps)
                    else:
                        Zi = Zt = np.zeros(1)
                    sc["Z"], sc["Ztok"] = Zi, Zt
                cost_all = int(t.sum())
                for name, (score, agg) in M.items():
                    if agg == "first" or N == 1:
                        pos = 0
                    elif agg == "vote":
                        _, pos = S.self_consistency(g.tolist())
                    elif agg == "max":
                        pos = S.highest(sc[score])
                    else:
                        _, pos = S.rwc(g.tolist(), sc[score])
                    cost = int(t[0]) if agg == "first" else cost_all
                    sel_rows.append((qi, o, N, name, int(idx[pos]), int(g[pos]), bool(c[pos]), cost))
                st = {"qi": qi, "order": o, "N": N, "coverage": bool(c.any()), "n_correct": int(c.sum()),
                      "n_groups": len(set(g.tolist())), "cost_all": cost_all,
                      "n_no_answer": int(NA[qi, idx].sum()), "calib_q05": calib[0], "calib_q95": calib[1]}
                if N in adapt_n:
                    lead, cnt, unique = S.sc_leader(g.tolist())
                    rg, _ = S.rwc(g.tolist(), sc[qscore])
                    st.update(sc_group=int(lead), sc_correct=bool(C[qi, idx][g == lead][0]),
                              sc_count=cnt, sc_unique=unique,
                              sc_is_no_answer=bool(NA[qi, idx][g == lead][0]),
                              rwc_group=int(rg), rwc_margin=S.rwc_margin(g.tolist(), sc[qscore]))
                state_rows.append(st)
        if qi % 100 == 0:
            print(f"[{ds}] {qi}/{R.n_questions}  {time.time() - t0:.0f}s", flush=True)

    qids = np.array(R.qids)
    split = feats.groupby("sample_id", sort=False).split.first().reindex(R.qids).values
    sel = pd.DataFrame(sel_rows, columns=["qi", "order", "N", "method", "sel_rollout", "sel_group", "correct", "cost"])
    sel.insert(0, "sample_id", qids[sel.qi.values])
    sel.insert(1, "split", split[sel.qi.values])
    sel["method"] = sel.method.astype("category")
    sel.to_parquet(out / f"{ds}_selections.parquet", index=False)
    st = pd.DataFrame(state_rows)
    st.insert(0, "sample_id", qids[st.qi.values])
    st.insert(1, "split", split[st.qi.values])
    st.to_parquet(out / f"{ds}_prefix_state.parquet", index=False)
    nw = pd.DataFrame(new_rows, columns=["qi", "order", "k", "new_group", "first_correct", "no_answer"])
    nw.insert(1, "split", split[nw.qi.values])
    nw.to_parquet(out / f"{ds}_reveal_steps.parquet", index=False)
    info = {"dataset": ds, "methods": {k: list(v) for k, v in M.items()}, "budgets": budgets,
            "n_orders": int(orders.shape[1]), "adaptive_quality_score": qscore, "seconds": round(time.time() - t0, 1),
            "note": "candidate set of (qi, order, N) = splits/task3/orders.json.gz[ds].orders[qi][order][:N]"}
    write_json(out / f"{ds}_offline_info.json", info)
    print(f"[{ds}] done in {time.time() - t0:.0f}s", flush=True)
    return info


def main():
    cfg = config()
    for ds in DATASETS:
        run_dataset(ds, cfg)


if __name__ == "__main__":
    main()
