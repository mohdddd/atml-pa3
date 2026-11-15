"""Step 3: what the quality signals capture (N = 32, i.e. the full pool of each question).

* Within-question associations between rollout length, scores and correctness: Spearman(length, score) per
  question; Pearson(score, correct) per question with mixed correctness; top-quartile minus bottom-quartile
  correctness rate per score within a question (8 highest vs 8 lowest of 32; ties broken by rollout ID),
  averaged across questions (all questions, and questions with mixed correctness). Shortest-rollout selection is
  the length-only control (method `shortest` in offline.py).
* MathVista: calibrated confidence c_{i,t} (calibration pooled over the question's 32 rollouts) vs visual
  engagement v_{i,t}: Pearson correlation within each rollout, summarised across questions; tokenwise fusion
  Z_token = mean_t c v vs VisConf Z = V C and the identity Z_token - Z = Cov_t(c, v).
* Illustrative trajectories, fixed selection rule: held-out questions in split order; the first 3 whose pool
  has 8-24 correct rollouts; in each, the correct and the incorrect rollout with the highest quality score
  (VisConf Z for MathVista, self-certainty K for MATH).

    python -m pa3.task3.signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pa3.common.paths import results_dir
from pa3.common.runlog import read_json, write_json
from pa3.task3 import selection as S
from pa3.task3.config import DATASETS, SPLIT_DIR, config, results_task
from pa3.task3.data import N_ROLLOUTS, load
from pa3.task3.stats import bootstrap_mean

TRAJ_RULE = ("held-out questions in split order; first 3 with 8-24 correct rollouts of 32; in each, the correct "
             "and the incorrect rollout with the highest quality score (Z for MathVista, K for MATH)")


def _quartile_diff(score: np.ndarray, corr: np.ndarray, q: int = 8) -> float:
    order = np.lexsort((np.arange(len(score)), -np.nan_to_num(score, nan=-np.inf)))   # desc score, then id
    return float(corr[order[:q]].mean() - corr[order[-q:]].mean())


def run_dataset(ds: str, cfg: dict) -> dict:
    out = results_dir(results_task(), "signals")
    feats = pd.read_parquet(results_dir(results_task(), "offline") / f"{ds}_rollout_features.parquet")
    R = load(ds, with_tokens=True, cfg=cfg)
    vlm = ds == "mathvista"
    if vlm:
        R.tok["v"] = (R.tok["img"] / (R.tok["img"] + R.tok["prompt"] + R.tok["gen_attn"])).astype(np.float32)
        qcfg = tuple(cfg["offline"]["calib_percentiles"])
        C32, Z32, Zt32, corr_cv, corr_kv, cov_err = [], [], [], [], [], []
        for qi in range(R.n_questions):
            rows = [R.row(qi, r) for r in range(N_ROLLOUTS)]
            kl = [R.tokens("kl", r) for r in rows]
            v = [R.tokens("v", r) for r in rows]
            C, V, Z, Zt, (lo, hi) = S.visconf_scores(kl, v, qcfg, cfg["offline"]["calib_eps"])
            C32 += list(C)
            Z32 += list(Z)
            Zt32 += list(Zt)
            for k_i, v_i, Ci, Vi, Zi, Zti in zip(kl, v, C, V, Z, Zt):
                c_i = np.clip((k_i.astype(np.float64) - lo) / max(hi - lo, cfg["offline"]["calib_eps"]), 0, 1)
                v64 = v_i.astype(np.float64)
                cov_err.append(abs((Zti - Zi) - (np.mean(c_i * v64) - c_i.mean() * v64.mean())))
                ok = len(c_i) >= 3 and c_i.std() > 0 and v64.std() > 0
                corr_cv.append(pearsonr(c_i, v64)[0] if ok else np.nan)
                corr_kv.append(pearsonr(k_i, v64)[0] if len(k_i) >= 3 and k_i.std() > 0 and v64.std() > 0 else np.nan)
        feats["C32"], feats["Z32"], feats["Ztok32"] = C32, Z32, Zt32
        feats["corr_c_v"], feats["corr_k_v"] = corr_cv, corr_kv
        write_json(out / "fusion_identity.json", {
            "identity": "Z_token - Z = Cov_t(c, v) (population covariance over a rollout's tokens)",
            "max_abs_error": float(max(cov_err)), "n_rollouts": len(cov_err)})
    feats["negT"] = -feats.n_tokens
    feats.to_parquet(out / f"{ds}_rollout_signals.parquet", index=False)

    scores = ["n_tokens", "negT", "L", "K"] + (["V", "C32", "Z32", "Ztok32"] if vlm else [])
    recs = []
    for qid, g in feats.groupby("sample_id", sort=False):
        corr = g.correct.values.astype(float)
        mixed = 0 < corr.sum() < len(corr)
        T = g.n_tokens.values
        for s in scores:
            x = g[s].values.astype(float)
            rho_len = spearmanr(T, x)[0] if s not in ("n_tokens", "negT") and np.std(x) > 0 else np.nan
            r_corr = pearsonr(x, corr)[0] if mixed and np.std(x) > 0 else np.nan
            recs.append({"sample_id": qid, "split": g.split.iloc[0], "score": s, "mixed": mixed,
                         "spearman_with_length": rho_len, "pearson_with_correct": r_corr,
                         "top_minus_bottom_quartile": _quartile_diff(x, corr)})
    per_q = pd.DataFrame(recs)
    per_q.to_parquet(out / f"{ds}_associations_per_question.parquet", index=False)
    summ = []
    for (split, s), g in per_q.groupby(["split", "score"]):
        tb = g.top_minus_bottom_quartile.values * 100
        tbm = g[g.mixed].top_minus_bottom_quartile.values * 100
        lo, hi = bootstrap_mean(tb, cfg["offline"]["bootstrap"], cfg["seed"])
        lom, him = bootstrap_mean(tbm, cfg["offline"]["bootstrap"], cfg["seed"])
        summ.append({"split": split, "score": s, "n_questions": len(g), "n_mixed": int(g.mixed.sum()),
                     "mean_spearman_with_length": g.spearman_with_length.mean(),
                     "mean_pearson_with_correct_mixed": g.pearson_with_correct.mean(),
                     "top_minus_bottom_pp_all": tb.mean(), "ci_lo_all": lo, "ci_hi_all": hi,
                     "top_minus_bottom_pp_mixed": tbm.mean(), "ci_lo_mixed": lom, "ci_hi_mixed": him})
    pd.DataFrame(summ).to_csv(out / f"{ds}_associations.csv", index=False)

    # correct vs incorrect rollout means (within mixed questions, averaged across questions)
    m = feats[feats.groupby("sample_id").correct.transform(lambda c: 0 < c.sum() < len(c))]
    cmp = (m.groupby(["split", "sample_id", "correct"])[scores].mean().groupby(["split", "correct"]).mean())
    cmp.to_csv(out / f"{ds}_score_means_by_correctness.csv")

    if vlm:
        cv = []
        for (split, corr_flag), g in [((s, c), g) for (s, c), g in feats.groupby(["split", "correct"])] + \
                                     [((s, "all"), g) for s, g in feats.groupby("split")]:
            per_q_mean = g.groupby("sample_id").corr_c_v.mean().dropna()
            cv.append({"split": split, "rollouts": corr_flag, "n_rollouts": int(g.corr_c_v.notna().sum()),
                       "n_questions": len(per_q_mean),
                       "mean_of_question_means": per_q_mean.mean(), "median_of_question_means": per_q_mean.median(),
                       "q25": per_q_mean.quantile(.25), "q75": per_q_mean.quantile(.75),
                       "frac_rollouts_positive": float((g.corr_c_v.dropna() > 0).mean()),
                       "mean_corr_raw_kl_v": g.groupby("sample_id").corr_k_v.mean().mean()})
        pd.DataFrame(cv).to_csv(out / "mathvista_confidence_vision_corr.csv", index=False)

    # illustrative trajectories (fixed rule)
    split_ids = read_json(SPLIT_DIR / "splits.json")[ds]["eval"]
    nc = feats.groupby("sample_id").correct.sum()
    picked = [q for q in split_ids if 8 <= nc[q] <= 24][:3]
    qs = "Z32" if vlm else "K"
    traj = []
    for q in picked:
        g = feats[feats.sample_id == q]
        for flag in (True, False):
            r = g[g.correct == flag].sort_values([qs, "rollout_id"], ascending=[False, True]).iloc[0]
            row = R.row(R.qids.index(q), int(r.rollout_id))
            rec = {"sample_id": q, "rollout_id": int(r.rollout_id), "correct": bool(flag),
                   "n_tokens": int(r.n_tokens), "answer": str(R.gen.extracted_answer.iloc[row]),
                   "gold": str(R.gen.gold_answer.iloc[row]), "L": r.L, "K": r.K,
                   "kl": R.tokens("kl", row).tolist(), "logp": R.tokens("logp", row).tolist()}
            if vlm:
                g_all = [R.row(R.qids.index(q), k) for k in range(N_ROLLOUTS)]
                pooled = np.concatenate([R.tokens("kl", x) for x in g_all]).astype(np.float64)
                lo, hi = np.percentile(pooled, cfg["offline"]["calib_percentiles"])
                c = np.clip((R.tokens("kl", row) - lo) / max(hi - lo, 1e-8), 0, 1)
                rec.update(V=r.V, C=r.C32, Z=r.Z32, Ztok=r.Ztok32, corr_c_v=r.corr_c_v,
                           c=c.tolist(), v=R.tokens("v", row).tolist())
            traj.append(rec)
    write_json(out / f"{ds}_trajectories.json", {"rule": TRAJ_RULE, "questions": picked, "rollouts": traj})
    return {"dataset": ds, "trajectory_questions": picked}


def main():
    cfg = config()
    for ds in DATASETS:
        print(run_dataset(ds, cfg), flush=True)


if __name__ == "__main__":
    main()
