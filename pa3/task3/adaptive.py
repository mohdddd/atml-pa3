"""Step 5: adaptive sampling simulated on the supplied pools, in the saved rollout orders.

Rollouts are revealed in each saved order; stopping is checked after n in {4, 8, 16} candidates, otherwise the
controller stops at 32. Both controllers return the current self-consistency answer.
  agreement(tau):        unique, non-NO_ANSWER SC leader with vote fraction >= tau
  quality(tau, gamma):   agreement(tau) AND SC winner == RWC winner AND RWC margin >= gamma
                         (RWC score: self-certainty for MATH, VisConf for MathVista; scores, VisConf calibration
                         and ranks computed from the revealed candidates only - see offline.py)
Settings are selected on the development split only (lowest mean token cost with accuracy no more than 1 pp
below fixed-32 self-consistency; ties: higher accuracy, then smaller tau, then smaller gamma; the always-32
fallback is part of each family) and written to adaptive_selected.json BEFORE the held-out evaluation runs.

    python -m pa3.task3.adaptive
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from pa3.common.paths import results_dir
from pa3.common.runlog import write_json
from pa3.task3.config import DATASETS, config, results_task
from pa3.task3.stats import paired_bootstrap

FALLBACK = "always32"


def policies(cfg: dict) -> list[dict]:
    a = cfg["adaptive"]
    out = [{"family": "agreement", "name": f"agreement_tau{t}", "tau": t, "gamma": None} for t in a["tau"]]
    out += [{"family": "quality", "name": f"quality_tau{t}_gamma{g}", "tau": t, "gamma": g}
            for t, g in itertools.product(a["tau"], a["gamma"])]
    for fam in ("agreement", "quality"):
        out.append({"family": fam, "name": f"{fam}_{FALLBACK}", "tau": None, "gamma": None})
    return out


def simulate(state: pd.DataFrame, pol: dict, cfg: dict) -> pd.DataFrame:
    """One row per (question, order): stop n, returned SC answer correctness, token cost."""
    checks = cfg["adaptive"]["check_points"]
    nmax = cfg["adaptive"]["max_n"]
    s = state[state.N.isin(checks + [nmax])]
    for n in checks + [nmax]:
        assert (s[s.N == n].groupby(["sample_id", "order"]).size() == 1).all()
    wide = {n: s[s.N == n].set_index(["sample_id", "order"]).sort_index() for n in checks + [nmax]}
    base = wide[nmax]
    for n in checks:
        assert wide[n].index.equals(base.index)
    stop_n = np.full(len(base), nmax)
    decided = np.zeros(len(base), bool)
    if pol["tau"] is not None:
        for n in checks:
            w = wide[n]
            ok = (w.sc_unique.astype(bool) & ~w.sc_is_no_answer.astype(bool)
                  & (w.sc_count >= pol["tau"] * n - 1e-9))
            if pol["family"] == "quality":
                ok &= (w.sc_group == w.rwc_group) & (w.rwc_margin >= pol["gamma"] - 1e-12)
            ok = ok.values
            new = ok & ~decided
            stop_n[new] = n
            decided |= ok
    res = pd.DataFrame({"split": base.split, "stop_n": stop_n})
    res["fixed32_correct"] = base.sc_correct.astype(bool)
    res["n_correct_pool"] = base.n_correct
    res["leader_frac_at4"] = wide[checks[0]].sc_count.values / checks[0]
    corr = np.zeros(len(res), bool)
    cost = np.zeros(len(res), np.int64)
    for n in checks + [nmax]:
        m = (res.stop_n == n).values
        corr[m] = wide[n].sc_correct.astype(bool).values[m]
        cost[m] = wide[n].cost_all.values[m]
    res["correct"] = corr
    res["cost"] = cost
    return res.reset_index()


def summarize(res: pd.DataFrame, cfg: dict) -> dict:
    q = res.groupby("sample_id")[["correct", "cost", "stop_n"]].mean()
    out = {"accuracy": float(q.correct.mean() * 100), "mean_cost": float(q.cost.mean()),
           "mean_rollouts": float(q.stop_n.mean()), "n_questions": len(q)}
    for n in cfg["adaptive"]["check_points"] + [cfg["adaptive"]["max_n"]]:
        out[f"stop_frac_{n}"] = float((res.stop_n == n).mean())
    return out


def select(rows: list[dict], ref_acc: float, tol: float) -> dict:
    ok = [r for r in rows if r["accuracy"] >= ref_acc - tol - 1e-9]
    key = lambda r: (r["mean_cost"], -r["accuracy"], math.inf if r["tau"] is None else r["tau"],
                     -1 if r["gamma"] is None else r["gamma"])
    return min(ok, key=key)


def run_dataset(ds: str, cfg: dict) -> dict:
    off = results_dir(results_task(), "offline")
    out = results_dir(results_task(), "adaptive")
    state = pd.read_parquet(off / f"{ds}_prefix_state.parquet")
    sel = pd.read_parquet(off / f"{ds}_selections.parquet")
    sc = sel[sel.method == "self_consistency"]
    fixed = (sc.groupby(["split", "N"]).agg(accuracy=("correct", "mean"), mean_cost=("cost", "mean"))
             .reset_index())
    fixed["accuracy"] *= 100
    fixed.to_csv(out / f"{ds}_fixed_sc.csv", index=False)
    tol = cfg["adaptive"]["tolerance_pp"]

    # ---------------- development: grid + selection (no held-out data touched)
    dev_state = state[state.split == "dev"]
    ref = float(fixed[(fixed.split == "dev") & (fixed.N == 32)].accuracy.iloc[0])
    grid = []
    for pol in policies(cfg):
        r = simulate(dev_state, pol, cfg)
        grid.append({**pol, **summarize(r, cfg)})
    pd.DataFrame(grid).to_csv(out / f"{ds}_dev_grid.csv", index=False)
    chosen = {fam: select([g for g in grid if g["family"] == fam], ref, tol) for fam in ("agreement", "quality")}
    frozen = {"dataset": ds, "dev_fixed32_sc_accuracy": ref, "tolerance_pp": tol,
              "rule": "lowest dev mean token cost with dev accuracy >= fixed-32 SC - 1 pp; ties: higher accuracy, "
                      "smaller tau, smaller gamma; always-32 fallback included in each family",
              "selected": {fam: {k: c[k] for k in ("name", "tau", "gamma", "accuracy", "mean_cost", "mean_rollouts")}
                           for fam, c in chosen.items()}}
    write_json(out / f"{ds}_selected.json", frozen)            # frozen before held-out evaluation

    # ---------------- held-out evaluation of the frozen settings
    ev_state = state[state.split == "eval"]
    pols = {p["name"]: p for p in policies(cfg)}
    rows, per_q = [], {}
    f32 = sc[(sc.split == "eval") & (sc.N == 32)].groupby("sample_id").correct.mean()
    for fam, c in chosen.items():
        r = simulate(ev_state, pols[c["name"]], cfg)
        r.to_parquet(out / f"{ds}_eval_{fam}_runs.parquet", index=False)
        s = summarize(r, cfg)
        qa = r.groupby("sample_id").correct.mean().reindex(f32.index)
        bt = paired_bootstrap(qa.values * 100, f32.values * 100, cfg["offline"]["bootstrap"], cfg["seed"])
        early = r[r.stop_n < cfg["adaptive"]["max_n"]]
        wrong_early = early[~early.correct.astype(bool)]
        s.update({"family": fam, "policy": c["name"], "tau": c["tau"], "gamma": c["gamma"],
                  "acc_minus_fixed32_pp": bt["diff"], "ci_lo": bt["ci_lo"], "ci_hi": bt["ci_hi"],
                  "early_stop_rate": float(len(early) / len(r)),
                  "early_stop_accuracy": float(early.correct.mean() * 100) if len(early) else float("nan"),
                  "incorrect_early_stops": int(len(wrong_early)),
                  "incorrect_early_stops_frac": float(len(wrong_early) / len(r)),
                  "incorrect_early_recoverable_by_fixed32": int(wrong_early.fixed32_correct.sum()),
                  "correct_early_where_fixed32_wrong": int((early.correct.astype(bool) & ~early.fixed32_correct).sum()),
                  "questions_with_incorrect_early_stop": int(wrong_early.sample_id.nunique())})
        rows.append(s)
        per_q[fam] = r
    pd.DataFrame(rows).to_csv(out / f"{ds}_eval_selected.csv", index=False)

    # whole grid on held-out: retrospective accuracy-cost frontier only (never used for selection)
    full = [{**p, **summarize(simulate(ev_state, p, cfg), cfg)} for p in policies(cfg)]
    pd.DataFrame(full).to_csv(out / f"{ds}_eval_grid_retrospective.csv", index=False)

    # stopping behaviour vs agreement and vs fraction of correct candidates (retrospective)
    bins = [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0]
    labels = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"]
    rel = []
    for fam, r in per_q.items():
        r = r.copy()
        r["frac_correct_pool"] = r.n_correct_pool / 32
        r["pool_bin"] = pd.cut(r.frac_correct_pool, bins, labels=labels)
        for key, col in [("pool_correct_fraction", "pool_bin"), ("leader_fraction_at_4", "leader_frac_at4")]:
            g = r.groupby(col, observed=True).agg(n=("stop_n", "size"), mean_stop_n=("stop_n", "mean"),
                                                  early_rate=("stop_n", lambda x: (x < 32).mean()),
                                                  accuracy=("correct", "mean"), fixed32_accuracy=("fixed32_correct", "mean"))
            g = g.reset_index().rename(columns={col: "bin"})
            g.insert(0, "by", key)
            g.insert(0, "family", fam)
            rel.append(g)
    pd.concat(rel).to_csv(out / f"{ds}_stopping_relations.csv", index=False)

    # fixed budgets at similar cost
    fe = fixed[fixed.split == "eval"].sort_values("mean_cost")
    comp = []
    for s in rows:
        j = (fe.mean_cost - s["mean_cost"]).abs().idxmin()
        lo = fe[fe.mean_cost <= s["mean_cost"]].tail(1)
        hi = fe[fe.mean_cost >= s["mean_cost"]].head(1)
        interp = np.nan
        if len(lo) and len(hi) and hi.mean_cost.iloc[0] > lo.mean_cost.iloc[0]:
            w = (s["mean_cost"] - lo.mean_cost.iloc[0]) / (hi.mean_cost.iloc[0] - lo.mean_cost.iloc[0])
            interp = lo.accuracy.iloc[0] + w * (hi.accuracy.iloc[0] - lo.accuracy.iloc[0])
        comp.append({"family": s["family"], "policy": s["policy"], "adaptive_accuracy": s["accuracy"],
                     "adaptive_cost": s["mean_cost"], "nearest_fixed_N": int(fe.loc[j, "N"]),
                     "nearest_fixed_accuracy": float(fe.loc[j, "accuracy"]),
                     "nearest_fixed_cost": float(fe.loc[j, "mean_cost"]),
                     "fixed_sc_accuracy_interpolated_at_same_cost": float(interp)})
    pd.DataFrame(comp).to_csv(out / f"{ds}_vs_fixed_budget.csv", index=False)
    return frozen


def main():
    cfg = config()
    for ds in DATASETS:
        f = run_dataset(ds, cfg)
        print(ds, f["selected"])


if __name__ == "__main__":
    main()
