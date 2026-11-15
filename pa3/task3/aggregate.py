"""Task 3 summary tables, figures and qualitative examples (from the saved result files only).

    python -m pa3.task3.aggregate      ->  results/task3/summary/, figures/task3/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pa3.common.paths import FIGURES, results_dir
from pa3.common.runlog import read_json, read_jsonl, write_json
from pa3.task3 import selection as S
from pa3.task3.config import DATASETS, NO_ANSWER, SPLIT_DIR, config, results_task
from pa3.task3.data import load_orders
from pa3.task3.stats import paired_bootstrap

DS_LABEL = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
M_LABEL = {"first": "FirstAns", "self_consistency": "Self-consistency", "logp_max": "Mean log-prob (max)",
           "logp_rwc": "Mean log-prob + RWC", "selfcert_max": "Self-certainty (max)",
           "selfcert_rwc": "Self-certainty + RWC", "shortest": "Shortest (length control)",
           "vision_max": "Vision-only (max)", "vision_rwc": "Vision-only + RWC", "visconf_max": "VisConf (max)",
           "visconf_rwc": "VisConf + RWC", "vctoken_max": "Tokenwise fusion (max)", "vctoken_rwc": "Tokenwise fusion + RWC"}
# fixed categorical order (reference palette, light mode); colour follows the method everywhere
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHORT = {"first": "FirstAns", "self_consistency": "SC", "logp_max": "LogP max", "logp_rwc": "LogP+RWC",
         "selfcert_max": "SelfCert max", "selfcert_rwc": "SelfCert+RWC", "shortest": "Shortest",
         "vision_max": "Vision max", "vision_rwc": "Vision+RWC", "visconf_max": "VisConf max",
         "visconf_rwc": "VisConf+RWC", "vctoken_max": "Z_token max", "vctoken_rwc": "Z_token+RWC"}
MAIN = {"math": ["first", "self_consistency", "logp_max", "selfcert_max", "selfcert_rwc"],
        "mathvista": ["first", "self_consistency", "logp_max", "selfcert_max", "selfcert_rwc", "vision_rwc",
                      "visconf_max", "visconf_rwc"]}
COLOR = {m: PAL[i] for i, m in enumerate(MAIN["mathvista"])}
MARK = dict(zip(MAIN["mathvista"], "osD^v<>P"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, name):
    d = FIGURES / ("task3_dryrun" if results_task() != "task3" else "task3")
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- tables
def method_tables(cfg, out: Path) -> dict:
    off = results_dir(results_task(), "offline")
    res = {}
    allrows, boots, fails, per_q_all, divs = [], [], [], [], []
    for ds in DATASETS:
        sel = pd.read_parquet(off / f"{ds}_selections.parquet")
        st = pd.read_parquet(off / f"{ds}_prefix_state.parquet")
        # order-averaged per question (the unit of every reported mean and bootstrap)
        pq_ = sel.groupby(["sample_id", "split", "method", "N"], observed=True)[["correct", "cost"]].mean().reset_index()
        cov = st.groupby(["sample_id", "N"])[["coverage", "n_groups", "cost_all"]].mean().reset_index()
        pq_ = pq_.merge(cov, on=["sample_id", "N"])
        pq_.insert(0, "dataset", ds)
        per_q_all.append(pq_)
        for split in ("eval", "dev"):
            p = pq_[pq_.split == split]
            g = p.groupby(["method", "N"], observed=True).agg(
                accuracy=("correct", "mean"), mean_cost=("cost", "mean"), coverage=("coverage", "mean"),
                distinct_groups=("n_groups", "mean")).reset_index()
            g[["accuracy", "coverage"]] *= 100
            g["missed_given_covered"] = g.coverage - g.accuracy
            g["no_correct_candidate"] = 100 - g.coverage
            g.insert(0, "split", split)
            g.insert(0, "dataset", ds)
            allrows.append(g)
            fails.append(g[["dataset", "split", "method", "N", "accuracy", "missed_given_covered", "no_correct_candidate"]])
            # paired bootstrap of the main accuracy differences
            wide = p.pivot_table(index="sample_id", columns=["method", "N"], values="correct", observed=True) * 100
            meths = sorted(p.method.unique())
            pairs = [(m, "self_consistency") for m in meths if m != "self_consistency"]
            pairs += [(a, b) for a, b in [("logp_rwc", "logp_max"), ("selfcert_rwc", "selfcert_max"),
                                          ("vision_rwc", "vision_max"), ("visconf_rwc", "visconf_max"),
                                          ("visconf_rwc", "selfcert_rwc"), ("visconf_max", "selfcert_max"),
                                          ("visconf_rwc", "vision_rwc"),
                                          ("vctoken_max", "visconf_max"), ("vctoken_rwc", "visconf_rwc")]
                      if a in meths and b in meths]
            for N in cfg["offline"]["budgets"][1:]:
                for a, b in pairs:
                    bt = paired_bootstrap(wide[(a, N)].values, wide[(b, N)].values, cfg["offline"]["bootstrap"], cfg["seed"])
                    boots.append({"dataset": ds, "split": split, "N": N, "method_a": a, "method_b": b, **bt})
        # answer diversity: does an extra sample add a new answer group / the first correct one?
        rv = pd.read_parquet(off / f"{ds}_reveal_steps.parquet")
        rv = rv[rv.split == "eval"]
        b = rv.groupby("k")[["new_group", "first_correct", "no_answer"]].mean().reset_index()
        b.insert(0, "dataset", ds)
        divs.append(b)
    tab = pd.concat(allrows)
    tab.to_csv(out / "methods_by_N.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap_diffs.csv", index=False)
    pd.concat(fails).to_csv(out / "failure_decomposition.csv", index=False)
    pd.concat(per_q_all).to_parquet(out / "per_question_order_averaged.parquet", index=False)
    div = pd.concat(divs)
    div.to_csv(out / "reveal_step_novelty.csv", index=False)
    # novelty per budget interval (N/2, N]
    nov = []
    for ds, d in div.groupby("dataset"):
        for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]:
            x = d[(d.k > lo) & (d.k <= hi)]
            nov.append({"dataset": ds, "added_candidates": f"{lo + 1}-{hi}",
                        "frac_new_answer_group": x.new_group.mean(), "frac_repeating_existing": 1 - x.new_group.mean(),
                        "frac_first_correct": x.first_correct.mean(), "frac_no_answer": x.no_answer.mean()})
    pd.DataFrame(nov).to_csv(out / "novelty_by_budget.csv", index=False)
    res["tab"], res["boots"], res["div"], res["nov"] = tab, pd.DataFrame(boots), div, pd.DataFrame(nov)
    return res


# -------------------------------------------------------------------------------------------- figures
def fig_curves(tab):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for r, ds in enumerate(DATASETS):
        t = tab[(tab.dataset == ds) & (tab.split == "eval")]
        cov = t[t.method == "self_consistency"].sort_values("N")
        for c, xcol in enumerate(["N", "mean_cost"]):
            ax = axes[r, c]
            _style(ax)
            ax.plot(cov[xcol], cov.coverage, ls="--", color=MUTED, lw=1.6, label="Oracle coverage")
            for m in MAIN[ds]:
                x = t[t.method == m].sort_values("N")
                ax.plot(x[xcol], x.accuracy, color=COLOR[m], lw=1.8, marker=MARK[m], ms=4.5, label=M_LABEL[m])
            ax.set_xscale("log", base=2 if xcol == "N" else 10)
            if xcol == "N":
                ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
            ax.set_xlabel("Candidates N" if xcol == "N" else "Mean generated tokens per question (log)", fontsize=9, color=INK)
            ax.set_ylabel("Accuracy / coverage (%)", fontsize=9, color=INK)
            ax.set_title(f"{DS_LABEL[ds]} - held-out 800", fontsize=9.5, color=INK, loc="left")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "accuracy_coverage_curves.png")


def fig_failures(tab):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    cols = {"accuracy": ("Correct", PAL[0]), "missed_given_covered": ("Correct candidate available, missed", PAL[3]),
            "no_correct_candidate": ("No correct candidate generated", "#b9b8b0")}
    for r, ds in enumerate(DATASETS):
        t = tab[(tab.dataset == ds) & (tab.split == "eval")]
        for c, (sub, xs, xl) in enumerate([
                (t[t.method == "self_consistency"].sort_values("N"), None, "Self-consistency, candidates N"),
                (t[(t.N == 32) & t.method.isin(MAIN[ds] + ["shortest"])], None, "Method at N = 32")]):
            ax = axes[r, c]
            _style(ax)
            labels = [str(n) for n in sub.N] if c == 0 else [SHORT[m] for m in sub.method]
            x = np.arange(len(sub))
            bottom = np.zeros(len(sub))
            for k, (lab, col) in cols.items():
                ax.bar(x, sub[k].values, bottom=bottom, color=col, width=0.72, label=lab, edgecolor="white", lw=1)
                bottom += sub[k].values
            ax.set_xticks(x, labels, fontsize=7.5 if c else 8, rotation=30 if c else 0, ha="right" if c else "center")
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of questions (order-averaged)", fontsize=9, color=INK)
            ax.set_xlabel(xl, fontsize=9, color=INK)
            ax.set_title(f"{DS_LABEL[ds]} - held-out", fontsize=9.5, loc="left", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    _save(fig, "failure_decomposition.png")


def fig_diversity(tab, div):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, ds in enumerate(DATASETS):
        t = tab[(tab.dataset == ds) & (tab.split == "eval") & (tab.method == "self_consistency")].sort_values("N")
        _style(axes[0])
        axes[0].plot(t.N, t.distinct_groups, color=PAL[i], marker="os"[i], lw=1.8, ms=4.5, label=DS_LABEL[ds])
        d = div[div.dataset == ds]
        _style(axes[1])
        axes[1].plot(d.k, d.new_group * 100, color=PAL[i], lw=1.8, label=f"{DS_LABEL[ds]}: new answer group")
        axes[1].plot(d.k, d.first_correct * 100, color=PAL[i], lw=1.4, ls=":", label=f"{DS_LABEL[ds]}: first correct candidate")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axes[0].set_xlabel("Candidates N", fontsize=9)
    axes[0].set_ylabel("Distinct answer groups (incl. NO_ANSWER)", fontsize=9)
    axes[1].set_xlabel("Position k of the revealed rollout", fontsize=9)
    axes[1].set_ylabel("% of (question, order) pairs", fontsize=9)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=7, frameon=False)
    axes[0].set_title("Answer diversity (held-out)", fontsize=9.5, loc="left")
    axes[1].set_title("Does the k-th sample add a new answer?", fontsize=9.5, loc="left")
    fig.tight_layout()
    _save(fig, "answer_diversity.png")


def fig_signals():
    sd = results_dir(results_task(), "signals")
    lab = {"negT": "Shortness (-length)", "L": "Mean log-prob L", "K": "Self-certainty K", "V": "Vision V",
           "C32": "Calibrated conf. C", "Z32": "VisConf Z = VC", "Ztok32": "Tokenwise Z_token"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [0.7, 1.3, 1]})
    for i, ds in enumerate(DATASETS):
        a = pd.read_csv(sd / f"{ds}_associations.csv")
        a = a[(a.split == "eval") & a.score.isin(lab)]
        a = a.set_index("score").loc[[s for s in lab if s in set(a.score)]]
        ax = axes[i]
        _style(ax)
        y = np.arange(len(a))
        ax.barh(y, a.top_minus_bottom_pp_mixed, color=PAL[i], height=0.6)
        ax.errorbar(a.top_minus_bottom_pp_mixed, y, xerr=[a.top_minus_bottom_pp_mixed - a.ci_lo_mixed,
                                                         a.ci_hi_mixed - a.top_minus_bottom_pp_mixed],
                    fmt="none", ecolor=INK, lw=1)
        ax.set_yticks(y, [lab[s] for s in a.index], fontsize=8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Top - bottom quartile correctness (pp)", fontsize=8.5)
        ax.set_title(f"{DS_LABEL[ds]}\nmixed-correctness held-out questions", fontsize=9, loc="left")
    s = pd.read_parquet(sd / "mathvista_rollout_signals.parquet")
    s = s[s.split == "eval"]
    ax = axes[2]
    _style(ax)
    q = s.groupby("sample_id").corr_c_v.mean().dropna()
    ax.hist(q, bins=40, color=PAL[1], edgecolor="white", lw=0.5)
    ax.axvline(q.median(), color=INK, lw=1, ls="--")
    ax.text(q.median(), ax.get_ylim()[1] * 0.92, f" median {q.median():.2f}", fontsize=8)
    ax.set_xlabel("Within-rollout corr(c_t, v_t), question mean", fontsize=8.5)
    ax.set_ylabel("Questions", fontsize=8.5)
    ax.set_title("MathVista: confidence vs visual engagement", fontsize=9, loc="left")
    fig.tight_layout()
    _save(fig, "signal_analysis.png")


def _roll(x, w=32):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fig_trajectories():
    sd = results_dir(results_task(), "signals")
    for ds in DATASETS:
        t = read_json(sd / f"{ds}_trajectories.json")
        qs = t["questions"]
        fig, axes = plt.subplots(len(qs), 2, figsize=(10, 2.4 * len(qs)), squeeze=False)
        for r in t["rollouts"]:
            i = qs.index(r["sample_id"])
            for j, (key, name) in enumerate([("c", "calibrated confidence c_t"), ("v", "visual engagement v_t")]
                                            if ds == "mathvista" else [("kl", "self-certainty k_t"), ("logp", "log-prob")]):
                ax = axes[i, j]
                _style(ax)
                y = _roll(r[key])
                x = np.linspace(0, 1, len(y))
                ax.plot(x, y, color=PAL[0] if r["correct"] else PAL[1], lw=1.4,
                        label=("correct" if r["correct"] else "incorrect") + f" (T={r['n_tokens']}, ans {r['answer'][:18]})")
                ax.set_title(f"q {r['sample_id']} (gold {r['gold'][:18]}): {name}, 32-token moving mean", fontsize=8, loc="left")
                ax.legend(fontsize=7, frameon=False)
                ax.set_xlabel("relative position in rollout", fontsize=7.5)
        fig.tight_layout()
        _save(fig, f"trajectories_{ds}.png")


def adaptive_tables(cfg, out: Path) -> pd.DataFrame:
    ad = results_dir(results_task(), "adaptive")
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for r, ds in enumerate(DATASETS):
        sel = read_json(ad / f"{ds}_selected.json")
        ev = pd.read_csv(ad / f"{ds}_eval_selected.csv")
        fixed = pd.read_csv(ad / f"{ds}_fixed_sc.csv")
        fe = fixed[fixed.split == "eval"]
        for _, e in ev.iterrows():
            rows.append({"dataset": ds, "family": e.family, "policy": e.policy, "tau": e.tau, "gamma": e.gamma,
                         "dev_accuracy": sel["selected"][e.family]["accuracy"],
                         "dev_cost": sel["selected"][e.family]["mean_cost"],
                         "dev_fixed32_sc": sel["dev_fixed32_sc_accuracy"],
                         "eval_accuracy": e.accuracy, "eval_mean_cost": e.mean_cost,
                         "eval_mean_rollouts": e.mean_rollouts,
                         "eval_fixed32_sc": float(fe[fe.N == 32].accuracy.iloc[0]),
                         "eval_fixed32_cost": float(fe[fe.N == 32].mean_cost.iloc[0]),
                         "diff_vs_fixed32_pp": e.acc_minus_fixed32_pp, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi,
                         "stop@4": e.stop_frac_4, "stop@8": e.stop_frac_8, "stop@16": e.stop_frac_16,
                         "stop@32": e.stop_frac_32, "incorrect_early_stops_frac": e.incorrect_early_stops_frac,
                         "incorrect_early_recoverable_by_fixed32": e.incorrect_early_recoverable_by_fixed32,
                         "correct_early_where_fixed32_wrong": e.correct_early_where_fixed32_wrong})
        grid = pd.read_csv(ad / f"{ds}_eval_grid_retrospective.csv")
        ax = axes[r, 0]
        _style(ax)
        for fam, col, mk in [("agreement", PAL[2], "o"), ("quality", PAL[6], "s")]:
            g = grid[grid.family == fam]
            ax.scatter(g.mean_cost, g.accuracy, s=14, color=col, alpha=0.35, marker=mk,
                       label=f"{fam} grid (held-out, retrospective)")
            e = ev[ev.family == fam].iloc[0]
            ax.scatter([e.mean_cost], [e.accuracy], s=70, color=col, marker=mk, edgecolor=INK, lw=1.2, zorder=5,
                       label=f"{fam}: dev-selected {e.policy.split('_', 1)[1]}")
        ax.plot(fe.mean_cost, fe.accuracy, color=PAL[0], marker="D", ms=4, lw=1.6, label="fixed-N self-consistency")
        for _, f in fe.iterrows():
            ax.annotate(f"N={int(f.N)}", (f.mean_cost, f.accuracy), fontsize=6.5, color=MUTED,
                        xytext=(3, -9), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Mean generated tokens per question (log)", fontsize=8.5)
        ax.set_ylabel("Accuracy (%)", fontsize=8.5)
        ax.set_title(f"{DS_LABEL[ds]} - adaptive vs fixed budgets (held-out)", fontsize=9, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        ax = axes[r, 1]
        _style(ax)
        x = np.arange(4)
        w = 0.36
        for k, fam in enumerate(["agreement", "quality"]):
            e = ev[ev.family == fam].iloc[0]
            ax.bar(x + (k - 0.5) * w, [e.stop_frac_4 * 100, e.stop_frac_8 * 100, e.stop_frac_16 * 100, e.stop_frac_32 * 100],
                   width=w, color=[PAL[2], PAL[6]][k], label=f"{fam} ({e.policy.split('_', 1)[1]})", edgecolor="white")
        ax.set_xticks(x, ["4", "8", "16", "32"])
        ax.set_xlabel("Rollouts consumed at stop", fontsize=8.5)
        ax.set_ylabel("% of (question, order) runs", fontsize=8.5)
        ax.set_title("Stopping-count distribution (held-out)", fontsize=9, loc="left")
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, "adaptive_stopping.png")
    df = pd.DataFrame(rows)
    df.to_csv(out / "adaptive_summary.csv", index=False)

    # stopping behaviour vs pool correctness / agreement (retrospective)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for r, ds in enumerate(DATASETS):
        rel = pd.read_csv(ad / f"{ds}_stopping_relations.csv")
        rel = rel[(rel.family == "quality")]
        for j, by in enumerate(["pool_correct_fraction", "leader_fraction_at_4"]):
            x = rel[rel.by == by]
            ax = axes[j]
            _style(ax)
            ax.plot(range(len(x)), x.early_rate * 100, color=PAL[r], marker="os"[r], lw=1.6, label=f"{DS_LABEL[ds]}: early-stop rate")
            ax.plot(range(len(x)), x.accuracy * 100, color=PAL[r], lw=1.2, ls=":", marker="os"[r], ms=3,
                    label=f"{DS_LABEL[ds]}: accuracy")
            ax.set_xticks(range(len(x)), [str(b)[:6] for b in x.bin], fontsize=7.5)
    axes[0].set_xlabel("Fraction of correct rollouts in the full pool of 32 (retrospective)", fontsize=8.5)
    axes[1].set_xlabel("Leader vote fraction after 4 rollouts", fontsize=8.5)
    for ax in axes:
        ax.set_ylabel("%", fontsize=8.5)
        ax.legend(fontsize=6.5, frameon=False)
    axes[0].set_title("Quality-aware controller (dev-selected), held-out", fontsize=9, loc="left")
    fig.tight_layout()
    _save(fig, "stopping_vs_agreement.png")
    return df


# ------------------------------------------------------------------------------------- live (Step 4)
def live_tables(out: Path):
    ld = results_dir(results_task(), "live")
    if not (ld / "summary.json").exists():
        return None
    s = read_json(ld / "summary.json")
    st = s["strategies"]
    rows = []
    for k, name in [("single", "Single 3,072-token response"), ("parallel", "3 x 1,024 + self-consistency"),
                    ("refine", "1,024 initial + 2,048 critique/revision")]:
        d = st[k]
        rows.append({"strategy": name, "accuracy_%": d["accuracy"], "mean_input_tokens": d["mean_input_tokens"],
                     "mean_output_tokens": d["mean_output_tokens"], "mean_seconds": d["mean_seconds"],
                     "total_seconds": d["total_seconds"], "peak_gpu_mem_gb": d["peak_gpu_mem_gb"],
                     "extraction_rate_%": 100 * d.get("extraction_rate", d.get("selected_answer_extracted_rate", np.nan)),
                     "truncation_rate_%": 100 * d.get("truncation_rate", d.get("truncation_rate_calls",
                                                                                 d.get("revision_truncation_rate", np.nan)))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "live_table.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col, lab in zip(axes, ["accuracy_%", "mean_output_tokens", "mean_seconds"],
                            ["Accuracy (%)", "Mean output tokens / question", "Mean end-to-end seconds / question"]):
        _style(ax)
        ax.bar(range(3), df[col], color=PAL[:3], width=0.6)
        for i, v in enumerate(df[col]):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3), ["Single", "Parallel+SC", "Refine"], fontsize=8)
        ax.set_title(lab, fontsize=9, loc="left")
    fig.tight_layout()
    _save(fig, "live_strategies.png")
    return s, df


# ---------------------------------------------------------------------------------------- examples
def examples(cfg, out: Path) -> str:
    off = results_dir(results_task(), "offline")
    ad = results_dir(results_task(), "adaptive")
    sigd = results_dir(results_task(), "signals")
    splits = read_json(SPLIT_DIR / "splits.json")
    orders = load_orders()
    lines, js = ["# Task 3 - qualitative examples", ""], {}
    for ds in DATASETS:
        ans = pd.read_parquet(results_dir(results_task(), "answers") / f"{ds}_rollouts.parquet")
        sig = pd.read_parquet(sigd / f"{ds}_rollout_signals.parquet")
        qscore, qm = ("Z32", "visconf_rwc") if ds == "mathvista" else ("K", "selfcert_rwc")
        sel = pd.read_parquet(off / f"{ds}_selections.parquet")
        s32 = sel[(sel.N == 32) & (sel.order == 0) & sel.method.isin(["self_consistency", qm])]
        w = s32.pivot_table(index="sample_id", columns="method", values="correct", observed=True)
        gen_q = pd.read_parquet(Path(str(results_dir(results_task(), "answers"))) / f"{ds}_rollouts.parquet")
        picks = []
        for want in [(False, True), (True, False)]:
            for q in splits[ds]["eval"]:
                if (bool(w.loc[q, "self_consistency"]), bool(w.loc[q, qm])) == want:
                    picks.append(q)
                    break
        rule = (f"held-out questions in split order, rollout order 0, N = 32: the first question where {M_LABEL[qm]} is "
                f"right and self-consistency wrong, and the first where the reverse holds")
        lines += [f"## {DS_LABEL[ds]}: selection disagreement", f"Selection rule: {rule}.", ""]
        js[f"{ds}_disagreement"] = {"rule": rule, "cases": []}
        for q in picks:
            a = ans[ans.sample_id == q]
            sg = sig[sig.sample_id == q].set_index("rollout_id")
            qi = orders[ds]["qids"].index(q)
            order = orders[ds]["orders"][qi, 0].astype(int)
            g = a.set_index("rollout_id").loc[order]
            wts = S.rwc_weights(sg.loc[order, qscore].values)
            tbl = (pd.DataFrame({"answer": g.answer_key.values, "correct": g.correct.values, "w": wts})
                   .groupby(["answer", "correct"]).agg(votes=("w", "size"), rwc_weight=("w", "sum"))
                   .reset_index().sort_values("votes", ascending=False))
            sc_g, _ = S.self_consistency(g.group.tolist())
            rw_g, _ = S.rwc(g.group.tolist(), sg.loc[order, qscore].values)
            key = dict(zip(g.group, g.answer_key))
            case = {"sample_id": q, "self_consistency": key[sc_g], qm: key[rw_g], "groups": tbl.to_dict("records")}
            js[f"{ds}_disagreement"]["cases"].append(case)
            lines += [f"**Question {q}** - self-consistency -> `{key[sc_g]}`, {M_LABEL[qm]} -> `{key[rw_g]}`", "",
                      _md(tbl.head(6), "{:.2f}"), ""]
        # premature stopping
        r = pd.read_parquet(ad / f"{ds}_eval_quality_runs.parquet")
        bad = r[(r.stop_n < 32) & ~r.correct & r.fixed32_correct]
        order_pos = {q: i for i, q in enumerate(splits[ds]["eval"])}
        bad = bad.assign(pos=bad.sample_id.map(order_pos)).sort_values(["pos", "order"]).drop_duplicates("sample_id").head(2)
        rule = ("dev-selected quality-aware controller on held-out runs; questions in split order, first order where "
                "it stopped early with a wrong answer while fixed-32 self-consistency (same order) was right; first 2")
        lines += [f"## {DS_LABEL[ds]}: premature stopping", f"Selection rule: {rule}.", ""]
        js[f"{ds}_premature_stop"] = {"rule": rule, "cases": []}
        for _, b in bad.iterrows():
            qi = orders[ds]["qids"].index(b.sample_id)
            order = orders[ds]["orders"][qi, int(b.order)].astype(int)
            a = ans[ans.sample_id == b.sample_id].set_index("rollout_id")
            early = a.loc[order[:int(b.stop_n)]].answer_key.value_counts().to_dict()
            full = a.loc[order].groupby(["answer_key", "correct"]).size().sort_values(ascending=False)
            full = {f"{k[0]}{' (correct)' if k[1] else ''}": int(v) for k, v in full.items()}
            case = {"sample_id": b.sample_id, "order": int(b.order), "stop_n": int(b.stop_n),
                    "revealed_votes": early, "full_pool_votes": full}
            js[f"{ds}_premature_stop"]["cases"].append(case)
            lines += [f"**Question {b.sample_id}, order {int(b.order)}** - stopped at n = {int(b.stop_n)} with votes "
                      f"{early}; the full pool of 32: {full}", ""]
    lr = results_dir(results_task(), "live") / "per_question.csv"
    if lr.exists():
        d = pd.read_csv(lr)
        calls = {c["call_id"]: c for c in read_jsonl(results_dir(results_task(), "live") / "calls.jsonl")}
        rule = ("live questions in their fixed order: the first 2 wrong-to-correct and the first 2 correct-to-wrong "
                "revisions (relative to the shared initial answer)")
        lines += ["## Live MATH: sequential refinement", f"Selection rule: {rule}.", ""]
        js["live_revision"] = {"rule": rule, "cases": []}
        for lab, m in [("wrong -> correct", ~d.initial_correct & d.refine_correct),
                       ("correct -> wrong", d.initial_correct & ~d.refine_correct)]:
            for _, x in d[m].head(2).iterrows():
                rev = calls[f"{x.sample_id}|revision|4"]["response"]
                case = {"type": lab, "sample_id": x.sample_id, "gold": x.gold, "initial": x.initial_answer,
                        "revised": x.refine_answer, "revision_tail": rev[-500:]}
                js["live_revision"]["cases"].append(case)
                lines += [f"**{lab}: {x.sample_id}** - gold `{x.gold}`, initial `{x.initial_answer}`, revised "
                          f"`{x.refine_answer}`", "", "> " + rev[-400:].replace("\n", " "), ""]
    write_json(out / "examples.json", js)
    (out / "examples.md").write_text("\n".join(lines))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- summary.md
def main():
    cfg = config()
    out = results_dir(results_task(), "summary")
    r = method_tables(cfg, out)
    tab, boots = r["tab"], r["boots"]
    fig_curves(tab)
    fig_failures(tab)
    fig_diversity(tab, r["div"])
    fig_signals()
    fig_trajectories()
    ad = adaptive_tables(cfg, out)
    live = live_tables(out)
    examples(cfg, out)
    scorer = read_json(results_dir(results_task(), "answers") / "scorer.json")

    md = ["# Task 3 - summary (all numbers from results/task3/; held-out = 800 evaluation questions per dataset)", "",
          f"Scorer: `{scorer['scorer_version']}`, math-verify {scorer['math_verify']}, "
          f"latex2sympy2_extended {scorer['latex2sympy2_extended']} (fixed {scorer['fixed_at']}).", ""]
    for ds in DATASETS:
        t = tab[(tab.dataset == ds) & (tab.split == "eval")]
        piv = t.pivot_table(index="method", columns="N", values="accuracy", observed=True)
        cov = t[t.method == "self_consistency"].set_index("N")
        piv.loc["ORACLE coverage"] = cov.coverage
        piv.loc["distinct groups"] = cov.distinct_groups
        piv.loc["mean tokens (all N)"] = cov.mean_cost
        piv.index = [M_LABEL.get(i, i) for i in piv.index]
        md += [f"## {DS_LABEL[ds]}: accuracy (%) by candidate budget N", "", _md(piv.reset_index().rename(columns={"index": "method"}), "{:.1f}"), ""]
        b = boots[(boots.dataset == ds) & (boots.split == "eval") & (boots.N == 32)]
        b = b.assign(comparison=b.method_a.map(M_LABEL) + " - " + b.method_b.map(M_LABEL))[
            ["comparison", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi"]]
        md += [f"### {DS_LABEL[ds]}: paired bootstrap at N = 32 (pp, 95% CI, 1,000 question resamples)", "", _md(b, "{:.2f}"), ""]
        f = t[t.N == 32][["method", "accuracy", "missed_given_covered", "no_correct_candidate"]].copy()
        f["method"] = f.method.map(M_LABEL)
        md += [f"### {DS_LABEL[ds]}: failure decomposition at N = 32 (%)", "", _md(f, "{:.1f}"), ""]
    md += ["## Answer novelty of added candidates (held-out)", "", _md(r["nov"], "{:.3f}"), ""]
    sd = results_dir(results_task(), "signals")
    for ds in DATASETS:
        a = pd.read_csv(sd / f"{ds}_associations.csv")
        a = a[a.split == "eval"][["score", "mean_spearman_with_length", "mean_pearson_with_correct_mixed",
                                  "top_minus_bottom_pp_all", "top_minus_bottom_pp_mixed", "ci_lo_mixed", "ci_hi_mixed"]]
        md += [f"## {DS_LABEL[ds]}: signal associations at N = 32 (held-out)", "", _md(a, "{:.3f}"), ""]
    cv = pd.read_csv(sd / "mathvista_confidence_vision_corr.csv")
    md += ["## MathVista: within-rollout corr(c_t, v_t)", "", _md(cv[cv.split == "eval"], "{:.3f}"), "",
           f"Fusion identity check: {read_json(sd / 'fusion_identity.json')}", ""]
    md += ["## Adaptive sampling (thresholds selected on dev, evaluated on held-out)", "",
           _md(ad, "{:.3f}"), ""]
    if live:
        s, df = live
        md += ["## Live MATH (50 questions, one T4, FP16)", "", _md(df, "{:.2f}"), "",
               f"Transitions vs the shared initial answer: {s['transitions']}", "",
               "Paired bootstrap (pp): " + json.dumps({k: [round(v['diff'], 1), round(v['ci_lo'], 1), round(v['ci_hi'], 1)]
                                                      for k, v in s["paired_bootstrap_accuracy_pp"].items()}), ""]
    else:
        md += ["## Live MATH", "", "_Pending: run kaggle/31_task3_full.ipynb, then `python -m pa3.task3.aggregate`._", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    main()
