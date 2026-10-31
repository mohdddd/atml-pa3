"""Collect Task 1 results into report tables, figures and the qualitative annotation sheet.

Reads results/task1/{<run>/, eval/<condition>/} and writes
  results/task1/summary/   benchmarks.csv/.md, gsm8k_extraction.csv, resources.csv, regularization.csv,
                           diagnostics.csv, merge_checks.csv, qualitative_all.md, annotation_sheet.csv
  figures/task1/           sft_loss.png, l2sp_penalty.png, benchmark_deltas.png, diagnostics.png
Missing runs/conditions are skipped, so this can be run at any point (also locally from the repo).

Usage: python -m pa3.task1.aggregate
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pa3.common.paths import FIGURES, RESULTS
from pa3.common.runlog import read_json, read_jsonl
from pa3.task1 import config as C

BENCH = [("gsm8k", "exact_match_flexible", "GSM8K EM (flex)"), ("hellaswag", "acc_norm", "HellaSwag acc_norm"),
         ("boolq", "acc", "BoolQ acc"), ("arc_easy", "acc_norm", "ARC-Easy acc_norm")]
RAW = [("gsm8k", "exact_match_strict", "GSM8K EM (strict)"), ("hellaswag", "acc", "HellaSwag acc"),
       ("arc_easy", "acc", "ARC-Easy acc")]
LABEL = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid, u625)", "full_ft": "Full FT", "l2sp": "Full FT + L2-SP",
         "lora4": "LoRA r=4", "lora16": "LoRA r=16", "oplora16": "OPLoRA r=16"}
RUN_LABEL = {"full_ft": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA r=4", "lora16": "LoRA r=16",
             "oplora16": "OPLoRA r=16"}


def _root() -> Path:
    return RESULTS / C.results_task()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0])
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    return "\n".join(out)


def load_metrics() -> dict:
    out = {}
    for c in C.CONDITIONS:
        p = _root() / "eval" / c / "metrics.json"
        if p.exists():
            out[c] = read_json(p)
    return out


def benchmark_tables(M: dict, summ: Path) -> list[dict]:
    rows = []
    base = M.get("baseline")
    for c in C.CONDITIONS:
        if c not in M:
            continue
        r = {"condition": LABEL[c]}
        for t, k, lab in BENCH:
            v = 100 * M[c][t][k]
            r[lab] = v
            if base is not None:
                r[f"Δ {lab} (pp)"] = v - 100 * base[t][k]
        for t, k, lab in RAW:
            r[lab] = 100 * M[c][t][k]
        rows.append(r)
    _write_csv(summ / "benchmarks.csv", rows)
    (summ / "benchmarks.md").write_text(_md_table(rows) + "\n")
    ext = [{"condition": LABEL[c], "EM flex (%)": 100 * M[c]["gsm8k"]["exact_match_flexible"],
            "EM strict (%)": 100 * M[c]["gsm8k"]["exact_match_strict"],
            "truncated (%)": 100 * M[c]["gsm8k"]["truncation_rate"],
            "unextractable flex (%)": 100 * M[c]["gsm8k"]["flexible_unextractable_rate"],
            "unextractable strict (%)": 100 * M[c]["gsm8k"]["strict_unextractable_rate"],
            "mean gen tokens": M[c]["gsm8k"]["mean_generated_tokens"],
            "EM on non-truncated (%)": 100 * M[c]["gsm8k"]["accuracy_on_non_truncated"]}
           for c in C.CONDITIONS if c in M]
    _write_csv(summ / "gsm8k_extraction.csv", ext)
    return rows


def resources_table(summ: Path) -> list[dict]:
    rows, reg, merges = [], [], []
    for run in C.RUNS:
        p = _root() / run / "resources.json"
        if not p.exists():
            continue
        r = read_json(p)
        sizes = r.get("checkpoint_sizes_mb", {})
        rec = [v for k, v in sizes.items() if k.startswith("recovery") and "preempt" not in k]
        fin = [v for k, v in sizes.items() if k.startswith("eval_weights")]
        row = {"run": RUN_LABEL[run], "trainable params": r["trainable_params"],
               "trainable (%)": 100 * r["trainable_params"] / r["total_unique_params"],
               "train time (h)": r["train_hours"], "peak GPU alloc (GB)": r["peak_gpu_alloc_gb"],
               "peak GPU reserved (GB)": r["peak_gpu_reserved_gb"],
               "recovery ckpt (MB)": rec[-1] if rec else None, "final weights/adapter (MB)": fin[-1] if fin else None,
               "skipped updates": len(r.get("skipped_updates", [])), "sessions": r.get("n_sessions"),
               "complete": r.get("final")}
        if "l2sp_overhead_seconds" in r:
            row.update({"L2-SP overhead (min)": r["l2sp_overhead_seconds"] / 60,
                        "L2-SP transfer (min)": r["l2sp_transfer_seconds"] / 60,
                        "reference storage (MB)": r["reference_storage_bytes"] / 2**20})
        if "svd_preprocess_seconds" in r:
            row.update({"OPLoRA SVD (s)": r["svd_preprocess_seconds"],
                        "basis storage (MB)": r["basis_storage_bytes"] / 2**20,
                        "export (s)": r.get("export_seconds")})
        rows.append(row)
        if r["method"] == "full":
            log = read_jsonl(_root() / run / "train_log.jsonl")
            last = log[-1] if log else {}
            reg.append({"run": RUN_LABEL[run], "lambda": r.get("l2sp_lambda", 0.0),
                        "omega_final": last.get("omega"), "lambda_omega_final": last.get("lambda_omega"),
                        "sft_loss_final_50": float(np.mean([x["sft_loss"] for x in log[-50:]])) if log else None,
                        **{f"||θ-θ0||₂ @u{k}": v for k, v in r.get("displacement_l2", {}).items()}})
        mp = _root() / run / "merge_check.json"
        if mp.exists():
            merges.append({"run": RUN_LABEL[run], **read_json(mp)})
    _write_csv(summ / "resources.csv", rows)
    _write_csv(summ / "regularization.csv", reg)
    _write_csv(summ / "merge_checks.csv", merges)
    return rows


def diagnostics_table(M: dict, summ: Path) -> list[dict]:
    rows = []
    for c in C.CONDITIONS:
        p = _root() / "eval" / c / "diagnostics.json"
        if not p.exists():
            continue
        for l, d in read_json(p)["layers"].items():
            rows.append({"condition": LABEL[c], "layer": int(l), "rho_k": d["rho_k"], "eta": d["eta"],
                         "R_math_mean": d["R_math_mean"], "R_nonmath_mean": d["R_nonmath_mean"],
                         "R_math_median": d["R_math_median"], "R_nonmath_median": d["R_nonmath_median"],
                         "R_math/R_nonmath": d["R_ratio_math_over_nonmath"]})
    _write_csv(summ / "diagnostics.csv", rows)
    return rows


def qualitative(summ: Path) -> None:
    resp = {}
    for c in C.CONDITIONS:
        p = _root() / "eval" / c / "qualitative.jsonl"
        if p.exists():
            resp[c] = read_jsonl(p)
    if not resp:
        return
    first = next(iter(resp.values()))
    questions = list(dict.fromkeys(r["question"] for r in first))
    lines = ["# Task 1 qualitative responses (greedy, non-thinking, 512-token cap)\n"]
    for q in questions:
        for fmt in ["plain", "json", "template"]:
            lines.append(f"\n## {q} — {fmt}\n")
            for c, rows in resp.items():
                r = next((x for x in rows if x["question"] == q and x["format"] == fmt), None)
                if r is None:
                    continue
                flag = " **[TRUNCATED]**" if r["truncated"] else ""
                lines.append(f"\n### {LABEL[c]} ({r['n_generated_tokens']} tokens){flag}\n\n```\n{r['response']}\n```\n")
    (summ / "qualitative_all.md").write_text("".join(lines))
    # annotation subset: first and last underlying questions, all formats, all conditions (42)
    subset = [questions[0], questions[-1]]
    sheet = []
    for q in subset:
        for fmt in ["plain", "json", "template"]:
            for c, rows in resp.items():
                r = next((x for x in rows if x["question"] == q and x["format"] == fmt), None)
                if r is None:
                    continue
                sheet.append({"question": q, "format": fmt, "condition": LABEL[c], "truncated": r["truncated"],
                              "n_tokens": r["n_generated_tokens"], "format_compliance": "", "coherence_relevance": "",
                              "repetition": "", "unnecessary_math_framing": "", "inappropriate_content": "",
                              "excerpt": "", "notes": "", "response": r["response"]})
    _write_csv(summ / "annotation_sheet.csv", sheet)


def figures(M: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = FIGURES / C.results_task()
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"full_ft": "#1f77b4", "l2sp": "#ff7f0e", "lora4": "#2ca02c", "lora16": "#d62728", "oplora16": "#9467bd"}

    logs = {r: read_jsonl(_root() / r / "train_log.jsonl") for r in C.RUNS if (_root() / r / "train_log.jsonl").exists()}
    if logs:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for r, log in logs.items():
            u = np.array([x["update"] for x in log])
            y = np.array([x["sft_loss"] for x in log])
            w = min(25, len(y))
            ys = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(u, y, color=colors[r], alpha=0.15, lw=0.6)
            ax.plot(u[w - 1:], ys, color=colors[r], lw=1.6, label=RUN_LABEL[r])
        ax.set_xlabel("optimizer update")
        ax.set_ylabel("SFT loss (response tokens)")
        ax.set_title("Training loss (thin: per update; thick: 25-update moving average)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "sft_loss.png", dpi=200)
        plt.close(fig)
        if "l2sp" in logs:
            log = logs["l2sp"]
            fig, ax = plt.subplots(figsize=(6.5, 3.0))
            u = [x["update"] for x in log]
            ax.plot(u, [x["lambda_omega"] for x in log], color=colors["l2sp"], label="λΩ (penalty)")
            ax.set_xlabel("optimizer update")
            ax.set_ylabel("λΩ")
            ax2 = ax.twinx()
            ax2.plot(u, [x["sft_loss"] for x in log], color="gray", alpha=0.4, lw=0.7, label="SFT loss")
            ax2.set_ylabel("SFT loss")
            ax.set_title("Full FT + L2-SP: penalty vs SFT loss", fontsize=9)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_dir / "l2sp_penalty.png", dpi=200)
            plt.close(fig)

    if "baseline" in M and len(M) > 1:
        conds = [c for c in C.CONDITIONS if c in M and c != "baseline"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=False)
        for ax, (t, k, lab) in zip(axes, BENCH):
            vals = [100 * (M[c][t][k] - M["baseline"][t][k]) for c in conds]
            ax.barh(range(len(conds)), vals, color=["#1f77b4" if v >= 0 else "#d62728" for v in vals])
            ax.set_yticks(range(len(conds)))
            ax.set_yticklabels([LABEL[c] for c in conds] if ax is axes[0] else [])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{lab}\n(baseline {100 * M['baseline'][t][k]:.1f}%)", fontsize=9)
            ax.set_xlabel("Δ vs baseline (pp)")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "benchmark_deltas.png", dpi=200)
        plt.close(fig)

    diag = {c: read_json(_root() / "eval" / c / "diagnostics.json")["layers"] for c in C.CONDITIONS
            if (_root() / "eval" / c / "diagnostics.json").exists()}
    if diag:
        conds = list(diag)
        layers = sorted(int(l) for l in next(iter(diag.values())))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        wbar = 0.8 / len(layers)
        for i, l in enumerate(layers):
            x = np.arange(len(conds)) + (i - (len(layers) - 1) / 2) * wbar
            axes[0].bar(x, [diag[c][str(l)]["rho_k"] for c in conds], wbar, label=f"layer {l}")
            axes[1].bar(x, [diag[c][str(l)]["eta"] for c in conds], wbar, label=f"layer {l}")
            axes[2].bar(x, [diag[c][str(l)]["R_math_mean"] / max(diag[c][str(l)]["R_nonmath_mean"], 1e-12)
                            for c in conds], wbar, label=f"layer {l}")
        for ax, t in zip(axes, ["ρ_k (fraction of ||ΔW||² in top-128 output subspace)",
                                "η = ||ΔW||_F / ||W0||_F", "mean R(z) math / mean R(z) non-math"]):
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels([LABEL[c] for c in conds], rotation=35, ha="right", fontsize=8)
            ax.set_title(t, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
        axes[1].set_yscale("log")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "diagnostics.png", dpi=200)
        plt.close(fig)


def main() -> None:
    summ = _root() / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    M = load_metrics()
    benchmark_tables(M, summ)
    resources_table(summ)
    diagnostics_table(M, summ)
    qualitative(summ)
    figures(M)
    try:
        from pa3.common.gitsync import push

        push([summ, FIGURES / C.results_task()], "task1: summary tables and figures")
    except Exception as e:
        print(f"[git] {e}")
    print(f"[aggregate] {len(M)} conditions summarised -> {summ}")


if __name__ == "__main__":
    main()
